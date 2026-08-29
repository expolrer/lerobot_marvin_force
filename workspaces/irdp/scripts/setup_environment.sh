#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?environment config is required}"
shift
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

read_scalar() {
  local key="$1"
  awk -F':[[:space:]]*' -v wanted="${key}" '
    $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
      value=$2
      gsub(/^[\047\"]|[\047\"]$/, "", value)
      print value
      exit
    }
  ' "${CONFIG_PATH}"
}

environment_name="$(read_scalar environment_name)"
python_version="$(read_scalar python_version)"
editable_extras="$(read_scalar editable_extras)"
workspace_dir="$(cd "$(dirname "${CONFIG_PATH}")/.." && pwd)"
repo_root="$(cd "${workspace_dir}/../.." && pwd)"
conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${conda_bin}" ]]; then
  conda_bin="/home/marvin/miniconda3/bin/conda"
fi

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == false ]]; then
    "$@"
  fi
}

if ! "${conda_bin}" env list | awk '{print $1}' | grep -Fxq "${environment_name}"; then
  run "${conda_bin}" create -y -n "${environment_name}" "python=${python_version}" pip
fi
run "${conda_bin}" run --no-capture-output -n "${environment_name}" \
  python -m pip install --upgrade pip
run "${conda_bin}" run --no-capture-output -n "${environment_name}" \
  python -m pip install -e "${repo_root}${editable_extras}"
run "${conda_bin}" run --no-capture-output -n "${environment_name}" \
  python -c "import lerobot; print('LeRobot environment ready:', lerobot.__version__)"
run "${conda_bin}" run --no-capture-output -n "${environment_name}" lerobot-train --help
run "${conda_bin}" run --no-capture-output -n "${environment_name}" lerobot-rollout --help
