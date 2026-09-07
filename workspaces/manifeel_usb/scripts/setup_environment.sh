#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKSPACE_ROOT}/../.." && pwd)"
DEFAULT_CONFIG="${WORKSPACE_ROOT}/config/environment.yaml"

usage() {
  cat <<'EOF'
Usage: setup_environment.sh [--config FILE] [--dry-run]

Creates the isolated LeRobot 0.6.1 policy environment.  The ManiFeel/IsaacGym
environment remains separate and is only used by the ZMQ evaluation client.
EOF
}

CONFIG="${DEFAULT_CONFIG}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "${CONFIG}" ]] || { echo "Missing environment config: ${CONFIG}" >&2; exit 2; }

# environment.yaml intentionally contains only scalar top-level values so it can
# be read before PyYAML has been installed.
yaml_scalar() {
  local key="$1"
  awk -v wanted="${key}" '
    $0 ~ "^[[:space:]]*" wanted ":[[:space:]]*" {
      sub("^[[:space:]]*" wanted ":[[:space:]]*", "")
      sub(/[[:space:]]+#.*$/, "")
      gsub(/^\047|\047$/, "")
      gsub(/^\042|\042$/, "")
      print
      exit
    }
  ' "${CONFIG}"
}

ENV_NAME="$(yaml_scalar environment_name)"
PYTHON_VERSION="$(yaml_scalar python_version)"
INSTALL_MODE="$(yaml_scalar install_mode)"
CONDA_REQUESTED="$(yaml_scalar conda_executable)"
ARCHIVE="$(yaml_scalar conda_pack_archive)"
ARCHIVE_SHA256="$(yaml_scalar conda_pack_sha256)"
ENV_DIR="$(yaml_scalar environment_prefix)"
EXTRAS="$(yaml_scalar editable_extras)"
ALLOW_NETWORK="$(yaml_scalar allow_network)"
INSTALL_CURRENT_REPO="$(yaml_scalar install_current_repo)"
REQUIRED_VERSION="$(yaml_scalar required_lerobot_version)"

for value_name in ENV_NAME PYTHON_VERSION INSTALL_MODE ENV_DIR EXTRAS REQUIRED_VERSION; do
  [[ -n "${!value_name}" ]] || { echo "Empty required config value: ${value_name}" >&2; exit 2; }
done
[[ "${INSTALL_MODE}" == controlled || "${INSTALL_MODE}" == archive ]] || {
  echo "install_mode must be controlled or archive" >&2
  exit 2
}

find_conda() {
  if [[ "${CONDA_REQUESTED}" != auto ]]; then
    [[ -x "${CONDA_REQUESTED}" ]] || { echo "Conda is not executable: ${CONDA_REQUESTED}" >&2; return 1; }
    printf '%s\n' "${CONDA_REQUESTED}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
  for candidate in /opt/conda/bin/conda "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then printf '%s\n' "${candidate}"; return; fi
  done
  echo "Cannot find conda; set conda_executable in ${CONFIG}" >&2
  return 1
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ ${DRY_RUN} -eq 0 ]]; then "$@"; fi
}

if [[ "${INSTALL_MODE}" == archive ]]; then
  [[ -f "${ARCHIVE}" ]] || { echo "Missing conda-pack archive: ${ARCHIVE}" >&2; exit 2; }
  [[ "${ARCHIVE_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "archive mode requires a 64-character conda_pack_sha256" >&2
    exit 2
  }
  actual_sha="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
  [[ "${actual_sha,,}" == "${ARCHIVE_SHA256,,}" ]] || {
    echo "Archive SHA256 mismatch: expected ${ARCHIVE_SHA256}, got ${actual_sha}" >&2
    exit 2
  }
  if [[ -e "${ENV_DIR}" && ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Refusing to extract over incomplete environment prefix: ${ENV_DIR}" >&2
    exit 2
  fi
  if [[ ! -e "${ENV_DIR}" ]]; then
    run mkdir -p "${ENV_DIR}"
    run tar -xzf "${ARCHIVE}" -C "${ENV_DIR}"
    if [[ ${DRY_RUN} -eq 0 && -x "${ENV_DIR}/bin/conda-unpack" ]]; then
      run env "PATH=${ENV_DIR}/bin:${PATH}" "${ENV_DIR}/bin/conda-unpack"
    fi
  fi
else
  CONDA_BIN="$(find_conda)"
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    run "${CONDA_BIN}" create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
  fi
  pip_args=(pip install --disable-pip-version-check -e "${REPO_ROOT}[${EXTRAS}]")
  if [[ "${ALLOW_NETWORK}" == false ]]; then
    pip_args+=(--no-index)
  elif [[ "${ALLOW_NETWORK}" != true ]]; then
    echo "allow_network must be true or false" >&2
    exit 2
  fi
  run "${ENV_DIR}/bin/python" -m "${pip_args[@]}"
fi

if [[ "${INSTALL_CURRENT_REPO}" == true ]]; then
  # The fixed archive provides all wheels. Register only this checkout and do
  # not let pip resolve or download anything during the offline archive path.
  run "${ENV_DIR}/bin/python" -m pip install --disable-pip-version-check --no-deps \
    --no-build-isolation -e "${REPO_ROOT}[${EXTRAS}]"
elif [[ "${INSTALL_CURRENT_REPO}" != false ]]; then
  echo "install_current_repo must be true or false" >&2
  exit 2
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
  "${ENV_DIR}/bin/python" - "${REQUIRED_VERSION}" "${PYTHON_VERSION}" <<'PY'
import importlib.util
import sys

import lerobot
import torch
import yaml
import zmq

required = sys.argv[1]
required_python = tuple(int(part) for part in sys.argv[2].split("."))
if sys.version_info[:2] != required_python:
    raise SystemExit(
        f"Python version mismatch: expected {required_python}, got {sys.version_info[:2]}"
    )
actual = getattr(lerobot, "__version__", None)
if actual != required:
    raise SystemExit(f"LeRobot version mismatch: expected {required}, got {actual}")
if importlib.util.find_spec("accelerate") is None:
    raise SystemExit("accelerate is missing")
print(
    f"validated python={sys.version_info.major}.{sys.version_info.minor} "
    f"lerobot={actual} torch={torch.__version__} pyzmq={zmq.__version__}"
)
PY
fi

echo "Environment ${ENV_NAME} is ready at ${ENV_DIR}. ManiFeel simulator environment was intentionally left untouched."
