#!/usr/bin/env python3
"""Serve a trained LeRobot policy to ManiFeel's official runner over loopback ZMQ."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zmq

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from manifeel_adapter.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    configure_socket,
    decode_message,
    encode_message,
)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def feature_shape(policy: dict[str, Any], group: str, key: str) -> list[int] | None:
    item = policy.get(group, {}).get(key)
    if not isinstance(item, dict) or not isinstance(item.get("shape"), list):
        return None
    return list(item["shape"])


def validate_checkpoint(
    checkpoint: Path, config: dict[str, Any], model_name: str, model: dict[str, Any]
) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"Checkpoint is incomplete; missing {checkpoint / filename}")
    policy = read_json(checkpoint / "config.json")
    if policy.get("type") != model["policy_type"]:
        raise ValueError(
            f"Checkpoint type {policy.get('type')!r} does not match selected {model['policy_type']!r}"
        )
    expected_force = None if model_name == "vision" else config["tactile_target_key"]
    if policy.get("force_feature_key") != expected_force:
        raise ValueError(
            f"Checkpoint force_feature_key must be {expected_force!r}, got {policy.get('force_feature_key')!r}"
        )
    expected_inputs = {
        config["state_target_key"]: [int(config["state_dim"])],
        config["wrist_target_key"]: [
            int(config["image_channels"]),
            int(config["image_height"]),
            int(config["image_width"]),
        ],
    }
    if bool(model["use_force"]):
        expected_inputs[config["tactile_target_key"]] = [int(config["tactile_dim"])]
    for key, expected in expected_inputs.items():
        actual = feature_shape(policy, "input_features", key)
        if actual != expected:
            raise ValueError(f"Checkpoint input {key} expected {expected}, got {actual}")
    action = feature_shape(policy, "output_features", "action")
    if action != [int(config["action_dim"])]:
        raise ValueError(f"Checkpoint action must be [6] EEF delta, got {action}")
    if policy.get("type") == "forcevla":
        if policy.get("proprio_dim") != 7 or policy.get("force_dim") != 420:
            raise ValueError("ForceVLA checkpoint does not use the ManiFeel 7+420 contract")
    return policy


def validate_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Expected protocol_version={PROTOCOL_VERSION}")
    if config.get("bind_host") not in {"127.0.0.1", "localhost"}:
        raise ValueError("Policy server is restricted to a loopback bind address")
    model = config.get("models", {}).get(model_name)
    if not isinstance(model, dict):
        raise ValueError(f"Unknown model {model_name!r}")
    if model_name == "vision" and model.get("use_force") is not False:
        raise ValueError("Vision baseline must ignore force")
    if model_name != "vision" and model.get("use_force") is not True:
        raise ValueError("Force-conditioned model must require force")
    if int(config.get("action_dim", -1)) != 6:
        raise ValueError("Only the official six-dimensional EEF delta action is accepted")
    return model


def make_policy_service(
    checkpoint: Path, config: dict[str, Any], model_name: str, model: dict[str, Any], device: str
):
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_class = get_policy_class(str(model["policy_type"]))
    policy = policy_class.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    device_override = {"device": device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    return policy, preprocessor, postprocessor


def validate_act_arrays(
    arrays: dict[str, np.ndarray], config: dict[str, Any], model: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    state_key = config["state_target_key"]
    wrist_key = config["wrist_target_key"]
    force_key = config["tactile_target_key"]
    expected_keys = {state_key, wrist_key}
    if bool(model["use_force"]):
        expected_keys.add(force_key)
    if set(arrays) != expected_keys:
        raise ProtocolError(f"Observation fields must be {sorted(expected_keys)}, got {sorted(arrays)}")
    state = arrays[state_key]
    wrist = arrays[wrist_key]
    force = arrays.get(force_key)
    if state.ndim != 2 or state.shape[1] != int(config["state_dim"]):
        raise ProtocolError(f"State must be [B,{config['state_dim']}], got {state.shape}")
    expected_image = (
        state.shape[0],
        int(config["image_channels"]),
        int(config["image_height"]),
        int(config["image_width"]),
    )
    if wrist.shape != expected_image:
        raise ProtocolError(f"Wrist must be {expected_image}, got {wrist.shape}")
    if force is not None and force.shape != (state.shape[0], int(config["tactile_dim"])):
        raise ProtocolError(
            f"Force must be [B,{config['tactile_dim']}], got {force.shape}"
        )
    if state.shape[0] < 1 or state.shape[0] > int(config["n_test"]):
        raise ProtocolError(f"Batch size must be within [1,{config['n_test']}]")
    if not all(np.isfinite(value).all() for value in (state, wrist) if value is not None):
        raise ProtocolError("State or wrist contains NaN/Inf")
    if force is not None and not np.isfinite(force).all():
        raise ProtocolError("Force contains NaN/Inf")
    if wrist.size and (float(wrist.min()) < -1e-4 or float(wrist.max()) > 1.0001):
        raise ProtocolError("Wrist RGB must be float32 in [0,1]")
    return state, wrist, force


def infer(
    *,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    model: dict[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: str,
) -> np.ndarray:
    state, wrist, force = validate_act_arrays(arrays, config, model)
    batch: dict[str, Any] = {
        config["state_target_key"]: torch.from_numpy(state),
        config["wrist_target_key"]: torch.from_numpy(wrist),
        "task": [str(config["task_text"])] * state.shape[0],
    }
    if force is not None:
        batch[config["tactile_target_key"]] = torch.from_numpy(force)
    processed = preprocessor(batch)
    use_bf16 = str(getattr(policy.config, "dtype", "float32")) == "bfloat16"
    device_type = torch.device(device).type
    with torch.inference_mode(), torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=use_bf16 and device_type == "cuda",
    ):
        action = policy.select_action(processed)
    action = postprocessor(action)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().to(torch.float32).numpy()
    action = np.ascontiguousarray(action, dtype=np.float32)
    expected = (state.shape[0], int(config["action_dim"]))
    if action.shape != expected or not np.isfinite(action).all():
        raise RuntimeError(f"Policy returned invalid action; expected {expected}, got {action.shape}")
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["fcact", "rdp", "implicitrdp", "forcevla", "vision"], required=True
    )
    parser.add_argument("--checkpoint", type=Path, help="Override pretrained_model directory")
    parser.add_argument("--device", default="cuda:0", help="Logical device inside CUDA_VISIBLE_DEVICES")
    parser.add_argument("--once", action="store_true", help="Exit after one act request (test helper)")
    args = parser.parse_args()

    config = read_yaml(args.config.resolve())
    model = validate_config(config, args.model)
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else Path(model["checkpoint"]).expanduser().resolve()
    )
    checkpoint_config = validate_checkpoint(checkpoint, config, args.model, model)
    policy, preprocessor, postprocessor = make_policy_service(
        checkpoint, config, args.model, model, args.device
    )

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    configure_socket(
        socket,
        timeout_ms=int(config["timeout_ms"]),
        max_message_bytes=int(config["max_message_bytes"]),
    )
    endpoint = f"tcp://{config['bind_host']}:{int(model['port'])}"
    socket.bind(endpoint)
    running = True

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"Serving {args.model} ({checkpoint_config['type']}) from {checkpoint} at {endpoint}; "
        "action semantics remain 6-D EEF delta",
        flush=True,
    )
    act_count = 0
    try:
        while running:
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue
            try:
                header, arrays = decode_message(
                    frames, max_message_bytes=int(config["max_message_bytes"])
                )
                if header.get("model") != args.model:
                    raise ProtocolError(
                        f"Request model {header.get('model')!r} does not match server {args.model!r}"
                    )
                command = header.get("command")
                if command == "ping":
                    response = encode_message(
                        command="pong",
                        fields={
                            "model": args.model,
                            "policy_type": checkpoint_config["type"],
                            "use_force": bool(model["use_force"]),
                            "action_dim": int(config["action_dim"]),
                            "checkpoint": str(checkpoint),
                        },
                    )
                elif command == "reset":
                    policy.reset()
                    response = encode_message(command="reset_ok", fields={"model": args.model})
                elif command == "act":
                    declared_batch = header.get("batch_size")
                    state = arrays.get(config["state_target_key"])
                    if state is None or declared_batch != state.shape[0]:
                        raise ProtocolError("Header batch_size does not match observation arrays")
                    action = infer(
                        arrays=arrays,
                        config=config,
                        model=model,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=args.device,
                    )
                    response = encode_message(
                        command="act_result",
                        arrays={"action": action},
                        fields={"model": args.model},
                    )
                    act_count += 1
                elif command == "close":
                    running = False
                    response = encode_message(command="close_ok", fields={"model": args.model})
                else:
                    raise ProtocolError(f"Unsupported command: {command!r}")
            except Exception as error:  # Keep REP state valid and return a typed failure.
                response = encode_message(
                    command="error",
                    fields={"error_type": type(error).__name__, "message": str(error)},
                )
            socket.send_multipart(response, copy=False)
            if args.once and act_count:
                break
    finally:
        socket.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
