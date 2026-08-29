#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSPACE_DIR}/../.." && pwd)"
ENV_CONFIG="${WORKSPACE_DIR}/config/environment.yaml"

read_scalar() {
  local key="$1"
  awk -F':[[:space:]]*' -v wanted="${key}" '
    $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
      value=$2
      gsub(/^[\047\"]|[\047\"]$/, "", value)
      print value
      exit
    }
  ' "${ENV_CONFIG}"
}

environment_name="$(read_scalar environment_name)"
conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${conda_bin}" ]]; then
  conda_bin="/home/marvin/miniconda3/bin/conda"
fi

usage() {
  echo "Usage: $0 {env|data|train|deploy} [--dry-run]"
}

stage="${1:-}"
if [[ -z "${stage}" ]]; then
  usage
  exit 2
fi
shift

if [[ "${stage}" == "env" ]]; then
  exec bash "${WORKSPACE_DIR}/scripts/setup_environment.sh" "${ENV_CONFIG}" "$@"
fi

if ! "${conda_bin}" env list | awk '{print $1}' | grep -Fxq "${environment_name}"; then
  echo "Missing Conda environment ${environment_name}. Run: $0 env" >&2
  exit 3
fi

case "${stage}" in
  data)
    exec "${conda_bin}" run --no-capture-output -n "${environment_name}" \
      python "${WORKSPACE_DIR}/scripts/audit_dataset.py" \
      --config "${WORKSPACE_DIR}/config/data.yaml" "$@"
    ;;
  train)
    exec "${conda_bin}" run --no-capture-output -n "${environment_name}" \
      python "${WORKSPACE_DIR}/scripts/train.py" \
      --config "${WORKSPACE_DIR}/config/train.yaml" "$@"
    ;;
  deploy)
    exec "${conda_bin}" run --no-capture-output -n "${environment_name}" \
      python "${WORKSPACE_DIR}/scripts/deploy.py" \
      --config "${WORKSPACE_DIR}/config/deploy.yaml" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
