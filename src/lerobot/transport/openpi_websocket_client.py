#!/usr/bin/env python

"""Small OpenPI-compatible WebSocket client without the openpi-client package.

Only ``websockets`` and ``msgpack`` are required. This intentionally avoids the
official client's NumPy<2 dependency so it can coexist with LeRobot 0.6.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _dependencies():
    try:
        import msgpack
        import websockets.sync.client
    except ImportError as error:
        raise ImportError(
            "ForceVLA transport needs only `pip install websockets msgpack`; "
            "do not install openpi-client into the lerobot_marvin environment."
        ) from error
    return msgpack, websockets.sync.client


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in {"V", "O", "c"}:
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"]
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def packb(value: Any) -> bytes:
    msgpack, _ = _dependencies()
    return msgpack.packb(value, default=_pack_array)


def unpackb(value: bytes) -> Any:
    msgpack, _ = _dependencies()
    return msgpack.unpackb(value, object_hook=_unpack_array)


class OpenPIWebSocketClient:
    def __init__(
        self,
        host: str,
        port: int = 8000,
        api_key: str | None = None,
        connect_timeout_s: float = 10.0,
        retry_interval_s: float = 2.0,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.api_key = api_key
        self.connect_timeout_s = connect_timeout_s
        self.retry_interval_s = retry_interval_s
        self._connection = None
        self._metadata: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.connect()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def connect(self) -> None:
        _, websocket_client = _dependencies()
        deadline = time.monotonic() + self.connect_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
                self._connection = websocket_client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=min(5.0, self.connect_timeout_s),
                )
                metadata = self._connection.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"ForceVLA server returned text during handshake: {metadata}")
                self._metadata = unpackb(metadata)
                logger.info("Connected to ForceVLA server %s", self.uri)
                return
            except (OSError, TimeoutError) as error:
                last_error = error
                time.sleep(self.retry_interval_s)
        raise TimeoutError(f"Could not connect to {self.uri}: {last_error}")

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._connection is None:
                raise RuntimeError("ForceVLA client is not connected")
            self._connection.send(packb(observation))
            response = self._connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"ForceVLA inference server error: {response}")
        decoded = unpackb(response)
        if not isinstance(decoded, dict):
            raise TypeError(f"Expected ForceVLA response dict, got {type(decoded).__name__}")
        return decoded

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
