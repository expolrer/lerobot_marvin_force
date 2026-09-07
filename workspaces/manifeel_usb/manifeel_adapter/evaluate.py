#!/usr/bin/env python3
"""Run ManiFeel's official USB IsaacGym evaluator through the local ZMQ bridge."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def resolve_symbol(dotted: str) -> type:
    module_name, _, symbol_name = dotted.rpartition(".")
    if not module_name or not symbol_name:
        raise ValueError(f"Invalid class path: {dotted!r}")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def serialize_result(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "_path"):
        return str(value._path)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def validate_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    required = {
        "protocol_version",
        "bind_host",
        "timeout_ms",
        "max_message_bytes",
        "manifeel_root",
        "simulator_environment_name",
        "isaacgym_config_name",
        "force_runner_class",
        "vision_runner_class",
        "n_test",
        "n_test_vis",
        "test_start_seed",
        "max_steps",
        "n_obs_steps",
        "n_action_steps",
        "fps",
        "video_crf",
        "tqdm_interval_sec",
        "wrist_source_key",
        "tactile_source_key",
        "state_target_key",
        "wrist_target_key",
        "tactile_target_key",
        "task_text",
        "state_dim",
        "tactile_dim",
        "image_channels",
        "image_height",
        "image_width",
        "action_dim",
        "output_root",
        "models",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Evaluation config is missing keys: {sorted(missing)}")
    if config["protocol_version"] != "manifeel-lerobot-zmq-v1":
        raise ValueError("Unsupported protocol_version")
    if config["bind_host"] not in {"127.0.0.1", "localhost"}:
        raise ValueError("The unauthenticated policy bridge must bind to loopback only")
    if int(config["n_action_steps"]) != 1:
        raise ValueError("n_action_steps must be 1 for force-feedback closed loop")
    if (int(config["state_dim"]), int(config["tactile_dim"]), int(config["action_dim"])) != (
        7,
        420,
        6,
    ):
        raise ValueError("ManiFeel USB contract is state[7], tactile_force[420], action[6]")
    model = config["models"].get(model_name)
    if not isinstance(model, dict):
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(config['models'])}")
    if model_name == "vision" and bool(model.get("use_force")):
        raise ValueError("Vision baseline must set use_force=false")
    if model_name != "vision" and not bool(model.get("use_force")):
        raise ValueError(f"Force model {model_name} must set use_force=true")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["fcact", "rdp", "implicitrdp", "forcevla", "vision"], required=True
    )
    parser.add_argument("--output-dir", type=Path, help="Override timestamped result directory")
    parser.add_argument("--num-envs", type=int, help="Override n_test for a smoke or full evaluation")
    parser.add_argument("--max-steps", type=int, help="Override the episode step limit")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    if args.num_envs is not None:
        if args.num_envs < 1:
            raise ValueError("--num-envs must be positive")
        config["n_test"] = args.num_envs
        config["n_test_vis"] = min(int(config["n_test_vis"]), args.num_envs)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["max_steps"] = args.max_steps
    model = validate_config(config, args.model)
    manifeel_root = Path(config["manifeel_root"]).expanduser().resolve()
    config_dir = manifeel_root / "manifeel" / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Missing ManiFeel config directory: {config_dir}")
    sys.path.insert(0, str(manifeel_root))

    # NVIDIA IsaacGym requires importing isaacgym before torch.  Keep this
    # ordering local to the old simulator process; the LeRobot server is a
    # separate Python 3.12 process connected only over loopback ZMQ.
    try:
        importlib.import_module("isaacgym")
    except ImportError as error:
        raise RuntimeError(
            "IsaacGym is not installed in the ManiFeel simulator environment"
        ) from error

    import hydra
    from isaacgymenvs.utils.utils import set_seed

    from manifeel_adapter.zmq_policy import ZmqLeRobotPolicy

    # The public ManiFeel wrapper currently comments out this import while still
    # calling set_seed. Inject the official isaacgymenvs helper without editing
    # or forking the evaluator itself.
    wrapper_module = importlib.import_module("manifeel.envs.vistac_isaacgym_multiple_env_wrapper")
    wrapper_module.set_seed = set_seed

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(config["output_root"]).expanduser().resolve() / args.model / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    use_force = bool(model["use_force"])
    obs_meta: dict[str, Any] = {
        config["wrist_source_key"]: {
            "shape": [
                int(config["image_channels"]),
                int(config["image_height"]),
                int(config["image_width"]),
            ],
            "type": "rgb",
        },
        "state": {"shape": [int(config["state_dim"])], "type": "low_dim"},
    }
    if use_force:
        obs_meta[config["tactile_source_key"]] = {
            "shape": [3, 10, 14],
            # This matches ManiFeel's official visff_wrist.yaml. The wrapper
            # still returns raw float32 CHW data to the ZMQ proxy.
            "type": "rgb",
        }
    shape_meta = {
        "obs": obs_meta,
        "action": {"shape": [int(config["action_dim"])]},
    }
    runner_class = resolve_symbol(
        config["force_runner_class"] if use_force else config["vision_runner_class"]
    )
    policy = ZmqLeRobotPolicy(config, args.model)
    runner = None
    try:
        initialize_kwargs = {"config_dir": str(config_dir)}
        try:
            hydra_context = hydra.initialize_config_dir(version_base=None, **initialize_kwargs)
        except TypeError:  # Hydra 1.1 used by some IsaacGym installations.
            hydra_context = hydra.initialize_config_dir(**initialize_kwargs)
        with hydra_context:
            runner = runner_class(
                output_dir=str(output_dir),
                shape_meta=shape_meta,
                isaacgym_cfg_name=str(config["isaacgym_config_name"]),
                n_test=int(config["n_test"]),
                n_test_vis=int(config["n_test_vis"]),
                test_start_seed=int(config["test_start_seed"]),
                max_steps=int(config["max_steps"]),
                n_obs_steps=int(config["n_obs_steps"]),
                n_action_steps=int(config["n_action_steps"]),
                fps=int(config["fps"]),
                crf=int(config["video_crf"]),
                past_action=False,
                tqdm_interval_sec=float(config["tqdm_interval_sec"]),
            )
            result = runner.run(policy)
    finally:
        policy.close()
        if runner is not None and hasattr(runner.env, "close"):
            runner.env.close()

    serializable = {key: serialize_result(value) for key, value in result.items()}
    mean_score = serializable.get("test/mean_score")
    if not isinstance(mean_score, (int, float)):
        raise RuntimeError(f"Official runner did not return test/mean_score: {serializable}")
    payload = {
        "model": args.model,
        "policy_type": model["policy_type"],
        "checkpoint": str(policy.server_info.get("checkpoint", model["checkpoint"])),
        "action_semantics": "6D EEF delta [dx,dy,dz,d_axis_angle_x,d_axis_angle_y,d_axis_angle_z]",
        "official_success_rate": float(mean_score),
        "raw_metrics": serializable,
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved official ManiFeel metrics to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
