#!/usr/bin/env python3
"""Validate ManiFeel USB contracts and launch resumable LeRobot 0.6.1 training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

EXPECTED_FEATURES = {
    "observation.state": ("float32", [7]),
    "observation.tactile_force": ("float32", [420]),
    "observation.images.wrist": ("video", [256, 256, 3]),
    "action": ("float32", [6]),
}
FORCE_KEY = "observation.tactile_force"
TRAIN_CONFIG_NAME = "train_config.json"


def _run_lerobot_entrypoint() -> None:
    """Run the native LeRobot entry point inside an Accelerate worker."""

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if len(sys.argv) > 1 and sys.argv[1] == "--lerobot-entry":
    _run_lerobot_entrypoint()
    raise SystemExit(0)


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read valid YAML from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def validate_static_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "environment_name",
        "model",
        "gpu_ids",
        "rendezvous_port",
        "num_processes",
        "mixed_precision",
        "log_dir",
        "dataset",
        "runs",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Training config is missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"Unsupported schema_version: {config['schema_version']!r}")
    if config["model"] not in {"fcact", "vision", "rdp", "implicitrdp", "forcevla"}:
        raise ValueError(f"Unsupported model: {config['model']!r}")
    if config["mixed_precision"] != "bf16":
        raise ValueError("ManiFeel H100 training contract requires mixed_precision=bf16")
    gpu_ids = [item.strip() for item in str(config["gpu_ids"]).split(",") if item.strip()]
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("gpu_ids must be a comma-separated list of non-negative integers")
    world_size = int(config["num_processes"])
    if world_size < 1 or world_size > len(gpu_ids):
        raise ValueError("num_processes must be positive and cannot exceed visible gpu_ids")
    port = int(config["rendezvous_port"])
    if not 1024 <= port <= 65535:
        raise ValueError("rendezvous_port must be between 1024 and 65535")

    dataset = config["dataset"]
    for key in ("repo_id", "root", "codebase_version", "eval_split", "return_uint8", "features"):
        if key not in dataset:
            raise ValueError(f"dataset.{key} is required")
    if dataset["codebase_version"] != "v3.0":
        raise ValueError("Only LeRobot dataset schema v3.0 is supported")
    for key, (dtype, shape) in EXPECTED_FEATURES.items():
        declared = dataset["features"].get(key)
        if not isinstance(declared, dict):
            raise ValueError(f"dataset.features.{key} is required")
        if declared.get("dtype") != dtype or list(declared.get("shape", [])) != shape:
            raise ValueError(f"Static feature contract mismatch for {key}: {declared}")

    runs = config["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    names: set[str] = set()
    for run in runs:
        required_run = {
            "name",
            "output_dir",
            "batch_size",
            "steps",
            "num_workers",
            "save_freq",
            "eval_steps",
            "log_freq",
            "seed",
            "push_to_hub",
            "wandb_enable",
            "policy",
        }
        missing_run = required_run - run.keys()
        if missing_run:
            raise ValueError(f"Run is missing keys: {sorted(missing_run)}")
        if run["name"] in names:
            raise ValueError(f"Duplicate run name: {run['name']}")
        names.add(run["name"])
        if int(run["batch_size"]) < 1 or int(run["steps"]) < 1:
            raise ValueError(f"Run {run['name']} needs positive batch_size and steps")
        if bool(run["wandb_enable"]):
            raise ValueError("W&B is fail-closed for this workspace; set wandb_enable: false")

        policy = run["policy"]
        expected_type = "act" if config["model"] in {"fcact", "vision"} else config["model"]
        if policy.get("type") != expected_type:
            raise ValueError(
                f"Run {run['name']} policy.type must be {expected_type}, got {policy.get('type')}"
            )
        expected_force = None if config["model"] == "vision" else FORCE_KEY
        if config["model"] != "rdp" or run is runs[0]:
            if policy.get("force_feature_key") != expected_force:
                raise ValueError(
                    f"Run {run['name']} force_feature_key must be {expected_force!r}"
                )
        if config["model"] == "forcevla":
            if policy.get("proprio_dim") != 7 or policy.get("force_dim") != 420:
                raise ValueError("ForceVLA requires proprio_dim=7 and force_dim=420")
            if not run.get("pretrained_path"):
                raise ValueError("ForceVLA fresh training requires runs[].pretrained_path")

    if config["model"] == "rdp":
        if len(runs) != 2:
            raise ValueError("RDP requires exactly tokenizer and diffusion runs")
        if runs[0]["policy"].get("training_stage") != "tokenizer":
            raise ValueError("The first RDP run must train the tokenizer")
        if runs[1]["policy"].get("training_stage") != "diffusion":
            raise ValueError("The second RDP run must train diffusion")
        if runs[1].get("pretrained_from_run") != runs[0]["name"]:
            raise ValueError("RDP diffusion must point to the tokenizer run by name")


def validate_dataset(config: dict[str, Any], *, allow_missing: bool) -> Path:
    dataset = config["dataset"]
    root = Path(dataset["root"]).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        if allow_missing:
            print(f"[dry-run] dataset metadata not present; deferred validation: {info_path}")
            return root
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = read_json(info_path)
    if info.get("codebase_version") != dataset["codebase_version"]:
        raise ValueError(
            f"Dataset version mismatch: expected {dataset['codebase_version']}, "
            f"got {info.get('codebase_version')}"
        )
    actual_features = info.get("features")
    if not isinstance(actual_features, dict):
        raise ValueError(f"Dataset metadata has no features mapping: {info_path}")
    for key, expected in dataset["features"].items():
        actual = actual_features.get(key)
        if not isinstance(actual, dict):
            raise ValueError(f"Dataset is missing required feature {key}")
        actual_shape = list(actual.get("shape", []))
        if actual.get("dtype") != expected["dtype"] or actual_shape != list(expected["shape"]):
            raise ValueError(
                f"Dataset feature {key} expected {expected['dtype']}{expected['shape']}, got {actual}"
            )
        names = actual.get("names")
        if names is not None and len(names) != expected["shape"][0] and len(expected["shape"]) == 1:
            raise ValueError(f"Dataset feature {key} names length does not match shape: {names}")
    fps = info.get("fps")
    if fps is None or float(fps) <= 0:
        raise ValueError("Dataset metadata must contain a positive fps")
    return root


def validate_cuda_devices(gpu_ids: str) -> None:
    """Fail before creating outputs when a requested physical GPU is unavailable."""

    import torch

    requested = [int(item.strip()) for item in gpu_ids.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError(f"gpu_ids contains duplicates: {gpu_ids}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; ManiFeel training requires an NVIDIA GPU")
    available = torch.cuda.device_count()
    invalid = [device for device in requested if device >= available]
    if invalid:
        raise RuntimeError(
            f"Requested physical GPU ids {invalid}, but torch detects only {available} CUDA devices"
        )


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    checkpoint = output_dir / "checkpoints" / "last"
    pretrained = checkpoint / "pretrained_model"
    return (
        pretrained / TRAIN_CONFIG_NAME,
        pretrained / "config.json",
        checkpoint / "training_state" / "training_step.json",
    )


def policy_shape(policy_config: dict[str, Any], collection: str, key: str) -> list[int] | None:
    feature = policy_config.get(collection, {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    return list(shape) if isinstance(shape, list) else None


def validate_policy_checkpoint(
    pretrained_dir: Path,
    *,
    expected_type: str,
    expected_force_key: str | None,
    immutable_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = pretrained_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing policy config: {config_path}")
    policy = read_json(config_path)
    if policy.get("type") != expected_type:
        raise ValueError(
            f"Checkpoint policy type mismatch at {config_path}: expected {expected_type}, got {policy.get('type')}"
        )
    if policy.get("force_feature_key") != expected_force_key:
        raise ValueError(
            f"Checkpoint force key mismatch: expected {expected_force_key!r}, "
            f"got {policy.get('force_feature_key')!r}"
        )
    expected_inputs = {
        "observation.state": [7],
        "observation.images.wrist": [3, 256, 256],
    }
    if expected_force_key is not None:
        expected_inputs[expected_force_key] = [420]
    for key, expected in expected_inputs.items():
        actual = policy_shape(policy, "input_features", key)
        if actual != expected:
            raise ValueError(f"Checkpoint input {key} expected {expected}, got {actual}")
    action_shape = policy_shape(policy, "output_features", "action")
    if action_shape != [6]:
        raise ValueError(f"Checkpoint action must be [6] EEF delta, got {action_shape}")

    if immutable_policy:
        ignored = {"type", "device", "repo_id", "pretrained_path"}
        for key, expected in immutable_policy.items():
            if key in ignored:
                continue
            if policy.get(key) != expected:
                raise ValueError(
                    f"Checkpoint policy.{key} mismatch: expected {expected!r}, got {policy.get(key)!r}"
                )
    return policy


def validate_resume_contract(
    config: dict[str, Any], run: dict[str, Any], output_dir: Path
) -> tuple[Path, int]:
    train_path, policy_path, state_path = checkpoint_paths(output_dir)
    if not (train_path.is_file() and policy_path.is_file() and state_path.is_file()):
        present = [path for path in (train_path, policy_path, state_path) if path.exists()]
        raise ValueError(
            f"Incomplete checkpoints/last in {output_dir}; present files: {[str(p) for p in present]}"
        )
    expected_type = run["policy"]["type"]
    expected_force = None if config["model"] == "vision" else FORCE_KEY
    validate_policy_checkpoint(
        policy_path.parent,
        expected_type=expected_type,
        expected_force_key=expected_force,
        immutable_policy=run["policy"],
    )
    saved = read_json(train_path)
    saved_dataset = saved.get("dataset", {})
    if saved_dataset.get("repo_id") != config["dataset"]["repo_id"]:
        raise ValueError("Resume rejected: dataset.repo_id differs from the checkpoint")
    saved_root = Path(str(saved_dataset.get("root", ""))).expanduser().resolve()
    configured_root = Path(config["dataset"]["root"]).expanduser().resolve()
    if saved_root != configured_root:
        raise ValueError(
            f"Resume rejected: dataset.root differs ({saved_root} != {configured_root})"
        )
    saved_output = Path(str(saved.get("output_dir", ""))).expanduser().resolve()
    if saved_output != output_dir:
        raise ValueError(f"Resume rejected: output_dir differs ({saved_output} != {output_dir})")
    if int(saved.get("batch_size", -1)) != int(run["batch_size"]):
        raise ValueError("Resume rejected: per-process batch_size changed")
    if bool(saved.get("wandb", {}).get("enable", False)) != bool(run["wandb_enable"]):
        raise ValueError("Resume rejected: W&B enable state changed")
    state = read_json(state_path)
    saved_processes = state.get("num_processes")
    if saved_processes is not None and int(saved_processes) != int(config["num_processes"]):
        raise ValueError("Resume rejected: num_processes changed, breaking sample-exact ordering")
    saved_batch = state.get("batch_size")
    if saved_batch is not None and int(saved_batch) != int(run["batch_size"]):
        raise ValueError("Resume rejected: recorded training batch_size changed")
    step = int(state.get("step", -1))
    if step < 0:
        raise ValueError("Resume checkpoint has an invalid training step")
    if step > int(run["steps"]):
        raise ValueError(
            f"Resume target {run['steps']} is below checkpoint step {step}; increase steps or use a new output"
        )
    return train_path, step


def validate_forcevla_base(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"ForceVLA base checkpoint directory does not exist: {path}")
    config = path / "config.json"
    weights = path / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(
            "ForceVLA base must contain config.json and one model.safetensors file; "
            f"got config={config.is_file()} weights={weights.is_file()} at {path}"
        )
    base = read_json(config)
    if base.get("type") not in {"pi0", "forcevla"}:
        raise ValueError(f"ForceVLA base must be PI0/ForceVLA, got {base.get('type')!r}")


def append_arg(args: list[str], key: str, value: Any) -> None:
    args.append(f"--{key}={cli_value(value)}")


def fresh_train_args(
    config: dict[str, Any],
    run: dict[str, Any],
    dataset_root: Path,
    completed_runs: dict[str, Path],
    *,
    dry_run: bool,
) -> list[str]:
    dataset = config["dataset"]
    args: list[str] = []
    for key, value in (
        ("dataset.repo_id", dataset["repo_id"]),
        ("dataset.root", dataset_root),
        ("dataset.eval_split", dataset["eval_split"]),
        ("dataset.return_uint8", dataset["return_uint8"]),
    ):
        append_arg(args, key, value)

    policy = dict(run["policy"])
    source_run = run.get("pretrained_from_run")
    if source_run:
        source = completed_runs.get(source_run)
        if source is None:
            source = Path("<TOKENIZER_CHECKPOINT_NOT_YET_CREATED>")
            if not dry_run:
                raise FileNotFoundError(f"No completed source run named {source_run}")
        if not dry_run:
            validate_policy_checkpoint(
                source,
                expected_type=policy["type"],
                expected_force_key=FORCE_KEY,
            )
        append_arg(args, "policy.path", source)
        policy.pop("type")
    else:
        append_arg(args, "policy.type", policy.pop("type"))

    pretrained_path = run.get("pretrained_path")
    if pretrained_path:
        pretrained = Path(pretrained_path).expanduser().resolve()
        if not dry_run:
            validate_forcevla_base(pretrained)
        append_arg(args, "policy.pretrained_path", pretrained)

    for key, value in policy.items():
        append_arg(args, f"policy.{key}", value)
    for key, value in (
        ("output_dir", Path(run["output_dir"]).expanduser().resolve()),
        ("job_name", run["name"]),
        ("batch_size", run["batch_size"]),
        ("num_workers", run["num_workers"]),
        ("steps", run["steps"]),
        ("save_freq", run["save_freq"]),
        ("eval_steps", run["eval_steps"]),
        ("log_freq", run["log_freq"]),
        ("seed", run["seed"]),
        ("save_checkpoint", True),
        ("policy.push_to_hub", run["push_to_hub"]),
        ("wandb.enable", run["wandb_enable"]),
        ("resume", False),
    ):
        append_arg(args, key, value)
    return args


def validate_cli_policy_fields(policy_type: str, run: dict[str, Any]) -> None:
    """Verify workspace policy keys against this checkout's LeRobot 0.6.1 API."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import factory as _policy_factory  # noqa: F401

    policy_class = PreTrainedConfig.get_choice_class(policy_type)
    allowed = {field.name for field in dataclasses.fields(policy_class)}
    unknown = set(run["policy"]) - allowed - {"type"}
    if unknown:
        raise ValueError(f"Unknown {policy_type} policy parameters: {sorted(unknown)}")


def launch_command(config: dict[str, Any], train_args: list[str]) -> list[str]:
    accelerate = shutil.which("accelerate") or "accelerate"
    command = [
        accelerate,
        "launch",
        "--num_processes",
        str(config["num_processes"]),
        "--mixed_precision",
        str(config["mixed_precision"]),
        "--main_process_port",
        str(config["rendezvous_port"]),
    ]
    if int(config["num_processes"]) > 1:
        command.append("--multi_gpu")
    command.extend([str(SCRIPT_PATH), "--lerobot-entry", *train_args])
    return command


def format_command(command: list[str], gpu_ids: str) -> str:
    import shlex

    return f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_ids)} " + " ".join(
        shlex.quote(part) for part in command
    )


def run_training(config: dict[str, Any], dataset_root: Path, *, dry_run: bool) -> None:
    completed_runs: dict[str, Path] = {}
    for run in config["runs"]:
        output_dir = Path(run["output_dir"]).expanduser().resolve()
        train_path, policy_path, state_path = checkpoint_paths(output_dir)
        resume_evidence = [train_path.exists(), policy_path.exists(), state_path.exists()]
        if any(resume_evidence):
            resume_config, step = validate_resume_contract(config, run, output_dir)
            if step == int(run["steps"]):
                print(f"[{run['name']}] already complete at step {step}; skipping", flush=True)
                completed_runs[run["name"]] = policy_path.parent
                continue
            train_args = [
                f"--config_path={resume_config}",
                "--resume=true",
                f"--steps={int(run['steps'])}",
            ]
            mode = f"resume from step {step}"
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(
                    f"Refusing fresh training in non-empty output without a complete checkpoint: {output_dir}"
                )
            train_args = fresh_train_args(
                config, run, dataset_root, completed_runs, dry_run=dry_run
            )
            mode = "fresh"

        expected_type = run["policy"]["type"]
        validate_cli_policy_fields(expected_type, run)
        command = launch_command(config, train_args)
        print(f"[{run['name']}] {mode}", flush=True)
        print(format_command(command, str(config["gpu_ids"])), flush=True)
        if not dry_run:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(config["gpu_ids"])
            env.setdefault("WANDB_MODE", "disabled")
            log_dir = Path(config["log_dir"]).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run['name']}.log"
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n[{mode}] {format_command(command, str(config['gpu_ids']))}\n")
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                except KeyboardInterrupt:
                    process.terminate()
                    raise
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)
            # lerobot-train always saves the final checkpoint.  Validate it before
            # allowing a dependent RDP stage to consume it.
            _, final_policy_config, _ = checkpoint_paths(output_dir)
            if not final_policy_config.is_file():
                raise RuntimeError(f"Training returned without a last checkpoint: {output_dir}")
        completed_runs[run["name"]] = output_dir / "checkpoints" / "last" / "pretrained_model"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Model training YAML")
    parser.add_argument("--gpu-ids", help="Override physical CUDA device list")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without launching")
    args = parser.parse_args()

    config = load_yaml(args.config.resolve())
    if args.gpu_ids is not None:
        config["gpu_ids"] = args.gpu_ids
    validate_static_config(config)

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not args.dry_run and active_env != config["environment_name"]:
        raise RuntimeError(
            f"Expected Conda environment {config['environment_name']!r}, got {active_env!r}"
        )
    if not args.dry_run:
        validate_cuda_devices(str(config["gpu_ids"]))
    dataset_root = validate_dataset(config, allow_missing=args.dry_run)
    run_training(config, dataset_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
