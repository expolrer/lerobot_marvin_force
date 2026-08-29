#!/usr/bin/env python3
"""Validate the Marvin dataset contract and launch stock lerobot-train."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
    return value


def validate_dataset(config: dict, repo_root: Path) -> Path:
    raw_root = expand(config["dataset_contract"]["root"], {"repo_root": str(repo_root)})
    dataset_root = Path(raw_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot dataset schema v3.0, got {info.get('codebase_version')}")
    features = info.get("features", {})
    for key, expected in config["dataset_contract"]["features"].items():
        actual = features.get(key)
        if actual is None:
            raise ValueError(f"Dataset is missing {key}")
        if actual.get("dtype") != expected["dtype"] or actual.get("shape") != expected["shape"]:
            raise ValueError(f"Dataset contract failed for {key}: {actual}")
        if "names" in expected and actual.get("names") != expected["names"]:
            raise ValueError(f"Dataset channel order failed for {key}: {actual.get('names')}")
    cart_stats = json.loads((dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))[
        "observation.cart_force"
    ]
    if any(float(value) != 0.0 for value in [*cart_stats["min"], *cart_stats["max"]]):
        raise ValueError("The reviewed contract expects observation.cart_force to be all zero")
    return dataset_root


def validate_force_selection(config: dict) -> None:
    supported = {"observation.joint_force", "observation.joint_torque"}
    selected = {
        run["args"].get("policy.force_feature_key")
        for run in config["runs"]
        if run["args"].get("policy.force_feature_key") is not None
    }
    if not selected:
        # A later RDP stage can inherit the force key from policy.path, but at
        # least one stage in a workspace must declare the contract explicitly.
        raise ValueError("No training stage declares policy.force_feature_key")
    unsupported = selected - supported
    if unsupported:
        raise ValueError(f"Unsupported force feature selection: {sorted(unsupported)}")
    missing = selected - set(config["dataset_contract"]["features"])
    if missing:
        raise ValueError(f"Selected force fields are missing from the dataset contract: {sorted(missing)}")


def validate_cli_schema(command: list[str], policy_type: str) -> None:
    """Ask Draccus to parse all non-path CLI fields without starting training."""
    import draccus

    from lerobot.configs import PreTrainedConfig, parser as lerobot_parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies import factory as _policy_factory  # noqa: F401

    args = command[1:]
    filtered = lerobot_parser.filter_path_args(
        TrainPipelineConfig.__get_path_fields__(), args
    )
    draccus.parse(config_class=TrainPipelineConfig, args=filtered)
    policy_class = PreTrainedConfig.get_choice_class(policy_type)
    allowed_policy_fields = {field.name for field in dataclasses.fields(policy_class)}
    for argument in args:
        if not argument.startswith("--policy."):
            continue
        key = argument[2:].split("=", 1)[0].removeprefix("policy.")
        if key in {"type", "path"}:
            continue
        if key not in allowed_policy_fields:
            raise ValueError(f"Unknown {policy_type} policy parameter: policy.{key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo_root = args.config.resolve().parents[3]
    validate_force_selection(config)
    dataset_root = validate_dataset(config, repo_root)
    variables = {"repo_root": str(repo_root), "dataset_root": str(dataset_root)}

    expected_env = config["environment_name"]
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env and active_env != expected_env:
        raise RuntimeError(f"Expected Conda environment {expected_env}, got {active_env}")

    active_policy_type: str | None = None
    for run in config["runs"]:
        command = ["lerobot-train"]
        for key, raw_value in run["args"].items():
            value = expand(raw_value, variables)
            command.append(f"--{key}={cli_value(value)}")
        declared_type = run["args"].get("policy.type")
        if declared_type is not None:
            active_policy_type = str(declared_type)
        if active_policy_type is None:
            raise ValueError(f"Training run {run['name']} has no policy.type or preceding policy stage")
        validate_cli_schema(command, active_policy_type)
        print(f"[{run['name']}]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
