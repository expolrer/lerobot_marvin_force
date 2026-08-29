#!/usr/bin/env python3
"""Translate a reviewed YAML file into the repository's lerobot-rollout CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPECTED_POLICY_TYPES = {
    "fcact": "act",
    "rdp": "rdp",
    "implicitrdp": "implicitrdp",
    "forcevla": "forcevla",
}


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def add_group(command: list[str], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is not None:
            command.append(f"--{prefix}{key}={cli_value(value)}")


def resolve_weight_path(raw_value: str, dry_run: bool) -> tuple[str, dict[str, Any] | None]:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("policy.path must not be empty")
    local = Path(value).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"policy.path must be a pretrained_model directory: {local}")
        resolved = local.resolve()
        config_path = resolved / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
        return str(resolved), json.loads(config_path.read_text(encoding="utf-8"))
    if not HUB_ID.fullmatch(value):
        if dry_run:
            print(f"WARNING: local checkpoint does not exist yet: {local}", flush=True)
            return value, None
        raise FileNotFoundError(f"Checkpoint does not exist: {local}")
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=value, filename="config.json")
        return value, json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as error:
        if dry_run:
            print(f"WARNING: could not preflight Hub checkpoint {value}: {error}", flush=True)
            return value, None
        raise RuntimeError(f"Could not load checkpoint config for {value}: {error}") from error


def validate_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    model_name = config["model_name"]
    expected_type = EXPECTED_POLICY_TYPES[model_name]
    actual_type = checkpoint.get("type")
    if actual_type != expected_type:
        raise ValueError(
            f"Checkpoint type mismatch: {model_name} requires {expected_type}, got {actual_type}"
        )
    if model_name == "rdp" and checkpoint.get("training_stage") != "diffusion":
        raise ValueError("RDP rollout requires a stage-2 training_stage=diffusion checkpoint")
    if model_name == "implicitrdp" and checkpoint.get("training_stage") != "diffusion":
        raise ValueError("ImplicitRDP checkpoint must use its single diffusion training stage")

    selected_force = config["policy"]["force_feature_key"]
    saved_force = checkpoint.get("force_feature_key")
    if saved_force != selected_force:
        raise ValueError(
            f"force_feature_key mismatch: checkpoint={saved_force}, deployment={selected_force}"
        )
    inputs = checkpoint.get("input_features", {})
    outputs = checkpoint.get("output_features", {})
    required_shapes = {
        "observation.state": [8],
        selected_force: [7],
    }
    for key, shape in required_shapes.items():
        actual = inputs.get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Checkpoint feature mismatch for {key}: expected {shape}, got {actual}")
    action_shape = outputs.get("action", {}).get("shape")
    if action_shape != [8]:
        raise ValueError(f"Checkpoint action must be Marvin B-arm action[8], got {action_shape}")
    expected_visuals = {
        "observation.images.up_cam",
        "observation.images.left_close",
        "observation.images.right_close",
    }
    actual_visuals = {
        key for key, feature in inputs.items() if feature.get("type") == "VISUAL"
    }
    if actual_visuals != expected_visuals:
        raise ValueError(
            f"Checkpoint camera mismatch: expected {sorted(expected_visuals)}, got {sorted(actual_visuals)}"
        )
    if model_name in {"fcact", "forcevla"} and checkpoint.get("n_action_steps") != 1:
        raise ValueError(f"{model_name} requires n_action_steps=1 for per-tick force feedback")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weight-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo_root = args.config.resolve().parents[3]
    weight_path, checkpoint = resolve_weight_path(
        args.weight_path or config["policy"]["path"], args.dry_run
    )

    expected_env = config["environment_name"]
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not args.dry_run and active_env != expected_env:
        raise RuntimeError(f"Expected Conda environment {expected_env}, got {active_env}")

    if checkpoint is not None:
        validate_checkpoint(config, checkpoint)

    robot = config["robot"]
    if robot["type"] != "marvin":
        raise ValueError("This workspace deploys only robot.type=marvin")
    force_feature_key = config["policy"]["force_feature_key"]
    force_type_by_feature = {
        "observation.joint_force": "joint_force",
        "observation.joint_torque": "joint_torque",
    }
    if force_feature_key not in force_type_by_feature:
        raise ValueError(f"Unsupported force_feature_key: {force_feature_key}")
    required_force_type = force_type_by_feature[force_feature_key]
    if not robot["use_force_feedback"] or required_force_type not in robot["force_feedback_types"]:
        raise ValueError(f"Deployment must expose {force_feature_key}[7]")
    expected_cameras = {"up_cam", "left_close", "right_close"}
    if set(robot["cameras"]) != expected_cameras:
        raise ValueError(f"Expected original camera keys {sorted(expected_cameras)}")

    inference_type = config["inference"]["type"]
    if config["model_name"] in {"rdp", "implicitrdp"} and inference_type != "rdp":
        raise ValueError("RDP and ImplicitRDP require inference.type=rdp")
    if config["model_name"] in {"fcact", "forcevla"} and inference_type != "sync":
        raise ValueError("Force-conditioned ACT and ForceVLA require inference.type=sync")

    command = [
        "lerobot-rollout",
        f"--policy.path={weight_path}",
        f"--policy.force_feature_key={force_feature_key}",
        f"--strategy.type={config['strategy']['type']}",
        f"--inference.type={inference_type}",
        f"--device={config['policy']['device']}",
        f"--rename_map={cli_value(config['rename_map'])}",
    ]
    add_group(command, "", config["runtime"])
    add_group(command, "strategy.", {k: v for k, v in config["strategy"].items() if k != "type"})
    add_group(command, "robot.", robot)
    add_group(command, "dataset.", config["dataset"])
    print("+", " ".join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
