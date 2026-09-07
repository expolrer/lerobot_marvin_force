"""Small, typed, non-pickle ZMQ protocol shared by simulator and policy service."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import zmq

PROTOCOL_VERSION = "manifeel-lerobot-zmq-v1"
MAX_HEADER_BYTES = 64 * 1024


class ProtocolError(RuntimeError):
    """Raised when a peer violates the bridge contract."""


def _json_bytes(header: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(dict(header), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ProtocolError(f"Header exceeds {MAX_HEADER_BYTES} bytes")
    return encoded


def encode_message(
    *,
    command: str,
    arrays: Mapping[str, np.ndarray] | None = None,
    fields: Mapping[str, Any] | None = None,
) -> list[bytes | memoryview]:
    descriptors: list[dict[str, Any]] = []
    frames: list[bytes | memoryview] = []
    for name, raw in (arrays or {}).items():
        array = np.asarray(raw)
        if array.dtype != np.float32:
            raise ProtocolError(f"Array {name} must be float32, got {array.dtype}")
        array = np.ascontiguousarray(array)
        descriptors.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            }
        )
        frames.append(memoryview(array))
    header = {
        "protocol": PROTOCOL_VERSION,
        "command": command,
        "arrays": descriptors,
        **dict(fields or {}),
    }
    return [_json_bytes(header), *frames]


def decode_message(
    frames: list[bytes], *, max_message_bytes: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not frames:
        raise ProtocolError("Received an empty multipart message")
    total = sum(len(frame) for frame in frames)
    if total > max_message_bytes:
        raise ProtocolError(f"Message has {total} bytes, limit is {max_message_bytes}")
    if len(frames[0]) > MAX_HEADER_BYTES:
        raise ProtocolError("Header is too large")
    try:
        header = json.loads(frames[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Invalid JSON header: {error}") from error
    if not isinstance(header, dict):
        raise ProtocolError("Header must be a JSON object")
    if header.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Protocol mismatch: expected {PROTOCOL_VERSION}, got {header.get('protocol')!r}"
        )
    descriptors = header.get("arrays", [])
    if not isinstance(descriptors, list) or len(descriptors) != len(frames) - 1:
        raise ProtocolError("Array descriptors do not match multipart frames")
    arrays: dict[str, np.ndarray] = {}
    for descriptor, frame in zip(descriptors, frames[1:], strict=True):
        if not isinstance(descriptor, dict):
            raise ProtocolError("Array descriptor must be an object")
        name = descriptor.get("name")
        shape = descriptor.get("shape")
        if not isinstance(name, str) or not name or name in arrays:
            raise ProtocolError(f"Invalid or duplicate array name: {name!r}")
        if descriptor.get("dtype") != "float32":
            raise ProtocolError(f"Only float32 arrays are accepted, got {descriptor.get('dtype')!r}")
        if not isinstance(shape, list) or any(not isinstance(item, int) or item < 0 for item in shape):
            raise ProtocolError(f"Invalid shape for {name}: {shape!r}")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        if descriptor.get("nbytes") != expected_nbytes or len(frame) != expected_nbytes:
            raise ProtocolError(
                f"Byte count mismatch for {name}: expected {expected_nbytes}, got {len(frame)}"
            )
        arrays[name] = np.frombuffer(frame, dtype=np.float32).reshape(shape).copy()
    return header, arrays


def configure_socket(socket: zmq.Socket, *, timeout_ms: int, max_message_bytes: int) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.MAXMSGSIZE, max_message_bytes)


def request(
    socket: zmq.Socket,
    *,
    command: str,
    arrays: Mapping[str, np.ndarray] | None,
    fields: Mapping[str, Any] | None,
    max_message_bytes: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    socket.send_multipart(encode_message(command=command, arrays=arrays, fields=fields), copy=False)
    header, response_arrays = decode_message(
        socket.recv_multipart(), max_message_bytes=max_message_bytes
    )
    if header.get("command") == "error":
        raise ProtocolError(str(header.get("message", "unknown policy-server error")))
    return header, response_arrays
