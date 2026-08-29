#!/usr/bin/env python3
"""Dispatch the root deployment selection to one self-contained workspace."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml


MODEL_MAP = {
    "fcact": "fcact",
    "rdp": "rdp",
    "implicitrdp": "irdp",
    "irdp": "irdp",
    "forcevla": "fvla",
    "fvla": "fvla",
}
POLICY_MODULE_MAP = {
    "fcact": "act",
    "rdp": "rdp",
    "implicitrdp": "implicitrdp",
    "forcevla": "forcevla",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = str(config["model_name"]).lower()
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model_name: {model_name}")
    workspace_name = MODEL_MAP[model_name]
    repo_root = args.config.resolve().parents[1]
    workspace = repo_root / "workspaces" / workspace_name
    deploy_config = workspace / "config" / "deploy.yaml"
    deploy_script = workspace / "scripts" / "deploy.py"
    if not deploy_config.is_file() or not deploy_script.is_file():
        raise FileNotFoundError(f"Workspace is incomplete: {workspace}")
    workspace_deploy = yaml.safe_load(deploy_config.read_text(encoding="utf-8"))
    canonical_model = "implicitrdp" if model_name == "irdp" else "forcevla" if model_name == "fvla" else model_name
    if workspace_deploy["model_name"] != canonical_model:
        raise ValueError(
            f"Root model {canonical_model} does not match workspace deploy model "
            f"{workspace_deploy['model_name']}"
        )
    environment_config = workspace / "config" / "environment.yaml"
    environment = yaml.safe_load(environment_config.read_text(encoding="utf-8"))
    environment_name = workspace_deploy["environment_name"]
    if environment["environment_name"] != environment_name:
        raise ValueError("environment.yaml and deploy.yaml select different Conda environments")

    conda = shutil.which("conda") or "/home/marvin/miniconda3/bin/conda"
    conda_state = subprocess.run(
        [conda, "env", "list", "--json"], text=True, capture_output=True, check=True
    )
    environment_names = {Path(path).name for path in json.loads(conda_state.stdout)["envs"]}
    if environment_name not in environment_names:
        raise RuntimeError(
            f"Missing environment {environment_name}; run ./run.sh env {canonical_model}"
        )
    cuda_check = (
        "assert torch.cuda.is_available(),'CUDA is required by deploy.yaml';"
        if str(workspace_deploy["policy"]["device"]).startswith("cuda")
        else ""
    )
    dependency_check = (
        "import importlib,lerobot,torch;"
        "assert lerobot.__version__=='0.6.1',lerobot.__version__;"
        f"importlib.import_module('lerobot.policies.{POLICY_MODULE_MAP[canonical_model]}');"
        f"{cuda_check}"
        "print('LeRobot',lerobot.__version__,'CUDA',torch.cuda.is_available())"
    )
    subprocess.run(
        [conda, "run", "-n", environment_name, "python", "-c", dependency_check],
        cwd=repo_root,
        check=True,
    )
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        environment_name,
        "python",
        str(deploy_script),
        "--config",
        str(deploy_config),
        "--weight-path",
        str(config["weight_path"]),
    ]
    if args.dry_run:
        command.append("--dry-run")
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
