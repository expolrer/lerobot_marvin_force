#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSPACE_ROOT}/../.." && pwd)"
CONFIG_DIR="${WORKSPACE_ROOT}/config"

usage() {
  cat <<'EOF'
Usage:
  workspaces/manifeel_usb/run.sh env [--dry-run]
  workspaces/manifeel_usb/run.sh data {download|convert|audit|all}
  workspaces/manifeel_usb/run.sh train {fcact|rdp|implicitrdp|forcevla|vision|all} [--dry-run|--smoke-test]
  workspaces/manifeel_usb/run.sh eval {serve|sim|all} {fcact|rdp|implicitrdp|forcevla|vision}

`train all` starts the four force-conditioned models concurrently on GPU 0/1/2/3.
The vision baseline is intentionally separate. `eval all MODEL` starts the LeRobot
policy server, runs the official ManiFeel IsaacGym runner, and then stops the server.
`--smoke-test` performs isolated real forward/backward/checkpoint steps under smoke/.
EOF
}

yaml_scalar() {
  local file="$1" key="$2"
  awk -v wanted="${key}" '
    $0 ~ "^[[:space:]]*" wanted ":[[:space:]]*" {
      sub("^[[:space:]]*" wanted ":[[:space:]]*", "")
      sub(/[[:space:]]+#.*$/, "")
      gsub(/^\047|\047$/, "")
      gsub(/^\042|\042$/, "")
      print
      exit
    }
  ' "${file}"
}

model_scalar() {
  local file="$1" model="$2" key="$3"
  awk -v wanted_model="${model}" -v wanted_key="${key}" '
    /^  [[:alnum:]_]+:[[:space:]]*$/ {
      section=$0
      sub(/^  /, "", section)
      sub(/:[[:space:]]*$/, "", section)
      next
    }
    section == wanted_model && $0 ~ "^    " wanted_key ":[[:space:]]*" {
      value=$0
      sub("^    " wanted_key ":[[:space:]]*", "", value)
      sub(/[[:space:]]+#.*$/, "", value)
      gsub(/^\047|\047$/, "", value)
      gsub(/^\042|\042$/, "", value)
      print value
      exit
    }
  ' "${file}"
}

find_conda() {
  if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
  for candidate in /opt/conda/bin/conda "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then printf '%s\n' "${candidate}"; return; fi
  done
  echo "Cannot find conda; run the env stage after installing Miniconda" >&2
  return 1
}

validate_model() {
  case "$1" in
    fcact|rdp|implicitrdp|forcevla|vision) ;;
    *) echo "Unknown model: $1" >&2; usage >&2; exit 2 ;;
  esac
}

train_config() {
  case "$1" in
    fcact) printf '%s\n' "${CONFIG_DIR}/train_fcact.yaml" ;;
    rdp) printf '%s\n' "${CONFIG_DIR}/train_rdp.yaml" ;;
    implicitrdp) printf '%s\n' "${CONFIG_DIR}/train_implicitrdp.yaml" ;;
    forcevla) printf '%s\n' "${CONFIG_DIR}/train_forcevla.yaml" ;;
    vision) printf '%s\n' "${CONFIG_DIR}/train_vision.yaml" ;;
  esac
}

CONDA_BIN=""
LEROBOT_ENV="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" environment_name)"
LEROBOT_PREFIX="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" environment_prefix)"
SIM_ENV="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" simulator_environment_name)"
HF_CACHE="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" hf_home)"
TORCH_CACHE="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" torch_home)"
OFFLINE_MODE="$(yaml_scalar "${CONFIG_DIR}/environment.yaml" offline_mode)"

require_conda() {
  if [[ -z "${CONDA_BIN}" ]]; then CONDA_BIN="$(find_conda)"; fi
}

conda_python() {
  [[ -x "${LEROBOT_PREFIX}/bin/python" ]] || {
    echo "Missing LeRobot environment at ${LEROBOT_PREFIX}; run the env stage first" >&2
    return 2
  }
  local hf_offline=0
  if [[ "${OFFLINE_MODE}" == true ]]; then hf_offline=1; fi
  CONDA_DEFAULT_ENV="${LEROBOT_ENV}" CONDA_PREFIX="${LEROBOT_PREFIX}" \
    HF_HOME="${HF_CACHE}" TORCH_HOME="${TORCH_CACHE}" \
    HF_HUB_OFFLINE="${hf_offline}" TRANSFORMERS_OFFLINE="${hf_offline}" \
    PATH="${LEROBOT_PREFIX}/bin:${PATH}" "${LEROBOT_PREFIX}/bin/python" "$@"
}

run_data_stage() {
  local operation="$1"
  local cfg="${CONFIG_DIR}/data.yaml"
  local raw source converted repo_id max_episodes task storage threads audit epsilon skip_compare allow_noncanonical
  raw="$(yaml_scalar "${cfg}" source_dir)"
  source="$(yaml_scalar "${cfg}" source_zarr)"
  converted="$(yaml_scalar "${cfg}" converted_root)"
  repo_id="$(yaml_scalar "${cfg}" repo_id)"
  max_episodes="$(yaml_scalar "${cfg}" audit_max_episodes)"
  task="$(yaml_scalar "${cfg}" task)"
  storage="$(yaml_scalar "${cfg}" image_storage)"
  threads="$(yaml_scalar "${cfg}" image_writer_threads)"
  audit="$(yaml_scalar "${cfg}" audit_output_dir)"
  epsilon="$(yaml_scalar "${cfg}" force_epsilon)"
  skip_compare="$(yaml_scalar "${cfg}" skip_value_compare)"
  allow_noncanonical="$(yaml_scalar "${cfg}" allow_noncanonical_source)"
  if [[ "${audit}" != /* ]]; then audit="${REPO_ROOT}/${audit}"; fi

  case "${operation}" in
    download)
      conda_python "${WORKSPACE_ROOT}/scripts/download_dataset.py" \
        --output-dir "${raw}"
      ;;
    convert)
      if [[ -f "${converted}/meta/info.json" ]]; then
        echo "Converted dataset already exists; refusing to overwrite and treating it as reusable: ${converted}"
      else
        local convert_args=(
          "${WORKSPACE_ROOT}/scripts/convert_manifeel_to_lerobot.py"
          --source "${source}"
          --output-root "${converted}"
          --repo-id "${repo_id}"
          --task "${task}"
          --image-storage "${storage}"
          --image-writer-threads "${threads}"
        )
        if [[ "${max_episodes}" != null && -n "${max_episodes}" ]]; then convert_args+=(--max-episodes "${max_episodes}"); fi
        if [[ "${allow_noncanonical}" == true ]]; then convert_args+=(--allow-noncanonical-source); fi
        conda_python "${convert_args[@]}"
      fi
      ;;
    audit)
      local audit_args=(
        "${WORKSPACE_ROOT}/scripts/audit_dataset.py"
        --source "${source}"
        --converted-root "${converted}"
        --output-dir "${audit}"
        --force-epsilon "${epsilon}"
      )
      if [[ "${max_episodes}" != null && -n "${max_episodes}" ]]; then audit_args+=(--max-episodes "${max_episodes}"); fi
      if [[ "${skip_compare}" == true ]]; then audit_args+=(--skip-value-compare); fi
      if [[ "${allow_noncanonical}" == true ]]; then audit_args+=(--allow-noncanonical-source); fi
      conda_python "${audit_args[@]}"
      ;;
    all)
      run_data_stage download
      run_data_stage convert
      run_data_stage audit
      ;;
    *) echo "data requires download, convert, audit, or all" >&2; exit 2 ;;
  esac
}

run_one_train() {
  local model="$1"
  shift
  conda_python "${WORKSPACE_ROOT}/scripts/train.py" --config "$(train_config "${model}")" "$@"
}

run_all_training() {
  local extra=("$@")
  local models=(fcact rdp implicitrdp forcevla)
  local pids=() names=()
  cleanup_training() {
    local pid
    for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  }
  trap cleanup_training INT TERM
  for model in "${models[@]}"; do
    run_one_train "${model}" "${extra[@]}" &
    pids+=("$!")
    names+=("${model}")
  done
  local failed=0 index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Training failed: ${names[$index]}" >&2
      failed=1
      cleanup_training
    fi
  done
  trap - INT TERM
  return "${failed}"
}

run_eval_server() {
  local model="$1"
  shift
  local gpu
  gpu="$(model_scalar "${CONFIG_DIR}/evaluate.yaml" "${model}" policy_gpu_id)"
  [[ -x "${LEROBOT_PREFIX}/bin/python" ]] || {
    echo "Missing LeRobot environment at ${LEROBOT_PREFIX}; run the env stage first" >&2
    return 2
  }
  local hf_offline=0
  if [[ "${OFFLINE_MODE}" == true ]]; then hf_offline=1; fi
  CUDA_VISIBLE_DEVICES="${gpu}" CONDA_DEFAULT_ENV="${LEROBOT_ENV}" CONDA_PREFIX="${LEROBOT_PREFIX}" \
    HF_HOME="${HF_CACHE}" TORCH_HOME="${TORCH_CACHE}" \
    HF_HUB_OFFLINE="${hf_offline}" TRANSFORMERS_OFFLINE="${hf_offline}" \
    PATH="${LEROBOT_PREFIX}/bin:${PATH}" \
    "${LEROBOT_PREFIX}/bin/python" "${WORKSPACE_ROOT}/scripts/serve_lerobot_act.py" \
    --config "${CONFIG_DIR}/evaluate.yaml" --model "${model}" "$@"
}

run_eval_sim() {
  local model="$1"
  shift
  local gpu
  gpu="$(model_scalar "${CONFIG_DIR}/evaluate.yaml" "${model}" simulator_gpu_id)"
  require_conda
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${SIM_ENV}" \
    python "${WORKSPACE_ROOT}/manifeel_adapter/evaluate.py" \
    --config "${CONFIG_DIR}/evaluate.yaml" --model "${model}" "$@"
}

run_full_eval() {
  local model="$1"
  local server_pid
  run_eval_server "${model}" &
  server_pid=$!
  cleanup_server() {
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  }
  trap cleanup_server EXIT INT TERM
  # The simulator proxy's initial ping waits up to timeout_ms while a large
  # ForceVLA checkpoint finishes loading, so no fragile fixed sleep is needed.
  run_eval_sim "${model}"
  trap - EXIT INT TERM
  cleanup_server
}

stage="${1:-}"
if [[ -z "${stage}" || "${stage}" == -h || "${stage}" == --help ]]; then usage; exit 0; fi
shift

case "${stage}" in
  env)
    exec bash "${WORKSPACE_ROOT}/scripts/setup_environment.sh" --config "${CONFIG_DIR}/environment.yaml" "$@"
    ;;
  data)
    [[ $# -ge 1 ]] || { usage >&2; exit 2; }
    operation="$1"; shift
    [[ $# -eq 0 ]] || { echo "Unexpected data arguments: $*" >&2; exit 2; }
    run_data_stage "${operation}"
    ;;
  train)
    [[ $# -ge 1 ]] || { usage >&2; exit 2; }
    model="$1"; shift
    if [[ "${model}" == all ]]; then
      run_all_training "$@"
    else
      validate_model "${model}"
      run_one_train "${model}" "$@"
    fi
    ;;
  eval)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    operation="$1" model="$2"; shift 2
    validate_model "${model}"
    case "${operation}" in
      serve) run_eval_server "${model}" "$@" ;;
      sim) run_eval_sim "${model}" "$@" ;;
      all)
        [[ $# -eq 0 ]] || { echo "eval all reads checkpoint/output from evaluate.yaml" >&2; exit 2; }
        run_full_eval "${model}"
        ;;
      *) echo "eval requires serve, sim, or all" >&2; exit 2 ;;
    esac
    ;;
  *) usage >&2; exit 2 ;;
esac
