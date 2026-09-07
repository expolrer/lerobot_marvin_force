#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh manifeel-usb env [--dry-run]
  ./run.sh manifeel-usb data {download|convert|audit|all}
  ./run.sh manifeel-usb train {fcact|rdp|implicitrdp|forcevla|vision|all} [--dry-run|--smoke-test]
  ./run.sh manifeel-usb eval {serve|sim|all} {fcact|rdp|implicitrdp|forcevla|vision}
  ./run.sh env [1|2|3|4|fcact|rdp|implicitrdp|forcevla] [--dry-run]
  ./run.sh data [model] [--dry-run]
  ./run.sh train <model> [--dry-run]
  ./run.sh deploy [--dry-run]

Models:
  1 / fcact       Force-conditioned ACT
  2 / rdp         Reactive Diffusion Policy
  3 / implicitrdp ImplicitRDP
  4 / forcevla    ForceVLA

Deployment model and checkpoint are selected in config/deploy.yaml.
EOF
}

normalize_model() {
  case "${1,,}" in
    1|fcact|force-conditioned-act|force_conditioned_act) echo fcact ;;
    2|rdp) echo rdp ;;
    3|irdp|implicitrdp|implicit-rdp|implicit_rdp) echo irdp ;;
    4|fvla|forcevla|force-vla|force_vla) echo fvla ;;
    *) return 1 ;;
  esac
}

select_model() {
  local choice="${1:-}"
  if [[ -z "${choice}" ]]; then
    cat <<'EOF' >&2
请选择模型 / Select a model:
  1) Force-conditioned ACT
  2) RDP
  3) ImplicitRDP
  4) ForceVLA
EOF
    read -r -p "输入 1/2/3/4: " choice
  fi
  normalize_model "${choice}" || {
    echo "Unknown model: ${choice}" >&2
    exit 2
  }
}

stage="${1:-}"
if [[ -z "${stage}" || "${stage}" == "-h" || "${stage}" == "--help" ]]; then
  usage
  exit 0
fi
shift

if [[ "${stage}" == "manifeel-usb" || "${stage}" == "manifeel_usb" ]]; then
  exec bash "${REPO_ROOT}/workspaces/manifeel_usb/run.sh" "$@"
fi

case "${stage}" in
  env|data)
    if [[ ${#} -gt 0 && "${1}" != --* ]]; then
      workspace="$(select_model "$1")"
      shift
    else
      workspace="$(select_model)"
    fi
    exec bash "${REPO_ROOT}/workspaces/${workspace}/run.sh" "${stage}" "$@"
    ;;
  train)
    if [[ ${#} -eq 0 || "${1}" == --* ]]; then
      echo "train requires a model name; run ./run.sh --help" >&2
      exit 2
    fi
    workspace="$(select_model "$1")"
    shift
    exec bash "${REPO_ROOT}/workspaces/${workspace}/run.sh" train "$@"
    ;;
  deploy)
    exec python3 "${REPO_ROOT}/scripts/launch_selected_deploy.py" \
      --config "${REPO_ROOT}/config/deploy.yaml" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
