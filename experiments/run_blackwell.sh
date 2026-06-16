#!/bin/bash
# ---------------------------------------------------------------------------
# run_blackwell.sh — Run AtomAgents Exp2 (real-prefetch) directly on a
# reserved Blackwell node without going through sbatch.
#
# Usage (from repo root or any directory):
#   bash experiments/run_blackwell.sh                     # single regular run
#   bash experiments/run_blackwell.sh --ablation          # all ablation conditions
#   bash experiments/run_blackwell.sh --condition no_plan # one ablation condition
#
# Override vLLM startup: set NO_START_MODELS=1 if servers are already running.
# ---------------------------------------------------------------------------

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ENV="${EXP_ENV:-atoms}"            # experiment env (AtomAgents + runtime)
PREDICTOR="${PREDICTOR:-learned}"
HW_PROFILE="${HW_PROFILE:-blackwell}"
EXP_TIMEOUT="${EXP_TIMEOUT:-180m}"
NO_START_MODELS="${NO_START_MODELS:-0}"
ABLATION="${ABLATION:-0}"
CONDITION="${CONDITION:-}"

log(){ echo "[$(date +'%F %T')] $*"; }

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --ablation)    ABLATION=1 ;;
    --condition=*) CONDITION="${arg#--condition=}" ;;
    --condition)   shift; CONDITION="$1" ;;
  esac
done

LAMMPS_SLOWDOWN_S="${LAMMPS_SLOWDOWN_S:-900}"
export LAMMPS_SLOWDOWN_S

log "Node       : $(hostname)"
log "Repo root  : $REPO_ROOT"
log "HW profile : $HW_PROFILE"
log "Predictor  : $PREDICTOR"
log "LAMMPS slow: ${LAMMPS_SLOWDOWN_S}s per relax step"

# vLLM ZMQ worker IPC sockets need a writable TMPDIR.  On PACE Blackwell hold
# jobs /tmp may be absent or unwritable.  /dev/shm is always a real tmpfs and
# works reliably for Unix domain sockets.
_JOB_ID="${SLURM_JOB_ID:-$$}"
if [[ -d /dev/shm && -w /dev/shm ]]; then
  VLLM_TMP="/dev/shm/atomagents_${USER:-vllm}_${_JOB_ID}"
else
  VLLM_TMP="${REPO_ROOT}/tmp/vllm_${_JOB_ID}"
fi
mkdir -p "$VLLM_TMP"
export TMPDIR="$VLLM_TMP"
log "TMPDIR     : $TMPDIR"

ATOMS_PYTHON="/storage/project/r-ag117-0/shared/agent_hpc/envs/${EXP_ENV}/bin/python"

# Pick experiment script: exp3 for blackwell_swap (3-model forced-swap), exp2 otherwise.
if [[ "$HW_PROFILE" == "blackwell_swap" || "$HW_PROFILE" == "rtx6000" ]]; then
  EXP_SCRIPT="${REPO_ROOT}/experiments/atomagents_exp3.py"
  log "Experiment : atomagents_exp3 (3-model forced-swap)"
else
  EXP_SCRIPT="${REPO_ROOT}/experiments/atomagents_exp2.py"
  log "Experiment : atomagents_exp2 (original)"
fi

# Build base flags
BASE_FLAGS=(
  --runtime-mode real
  --predictor "$PREDICTOR"
  --hw-profile "$HW_PROFILE"
  --swap-models
)
if [[ "$HW_PROFILE" == "blackwell_swap" || "$HW_PROFILE" == "rtx6000" ]]; then
  BASE_FLAGS+=(--lammps-slowdown "$LAMMPS_SLOWDOWN_S")
fi
[[ "$NO_START_MODELS" == "1" ]] && BASE_FLAGS+=(--no-start-models)

cleanup_vllm(){
  # Kill any lingering vLLM processes and wait for all 4 GPUs to drop below 1 GB.
  # Called before each condition so stale servers don't block new ones.
  pkill -f "vllm_clean128" 2>/dev/null || true
  local attempts=0
  while (( attempts < 60 )); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '{s+=$1} END {print s}')
    [[ "$used" -lt 2048 ]] && break
    sleep 5
    (( attempts++ )) || true
  done
  log "GPUs cleared (${attempts} polls). Continuing."
}

run_condition(){
  local label="$1"; shift
  local extra_flags=("$@")
  local ts; ts=$(date +'%Y%m%d_%H%M%S')
  local logfile="${REPO_ROOT}/logs/blackwell_${label}_${ts}.log"
  mkdir -p "${REPO_ROOT}/logs"

  cleanup_vllm

  log "===== Condition: $label ====="
  set +e
  timeout --preserve-status "$EXP_TIMEOUT" \
    "$ATOMS_PYTHON" -u "$EXP_SCRIPT" \
      "${BASE_FLAGS[@]}" \
      --condition "$label" \
      "${extra_flags[@]}" \
      2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 124 ]] && log "WARN: $label timed out" || log "OK: $label exit=$rc"
}

cd "$REPO_ROOT"

if [[ "$ABLATION" == "1" ]]; then
  CONDITIONS="${CONDITIONS:-full_system no_plan no_diverg_guard naive_prefetch no_model_prefetch no_data_prefetch mock_predictor baseline}"
  log "Running ablation conditions: $CONDITIONS"
  for COND in $CONDITIONS; do
    case "$COND" in
      baseline)         run_condition baseline --runtime-mode baseline --predictor mock ;;
      full_system)      run_condition full_system ;;
      no_plan)          run_condition no_plan --no-plan-extraction ;;
      no_diverg_guard)  run_condition no_diverg_guard --no-divergence-guard ;;
      naive_prefetch)   run_condition naive_prefetch --naive-prefetch ;;
      no_model_prefetch) run_condition no_model_prefetch --skip-resource-types vllm_model ;;
      no_data_prefetch) run_condition no_data_prefetch --skip-resource-types data_file ;;
      mock_predictor)   run_condition mock_predictor --predictor mock ;;
      *)                log "Unknown condition: $COND, skipping" ;;
    esac
    [[ "$COND" != "${CONDITIONS##* }" ]] && { log "Sleeping 30s between conditions…"; sleep 30; }
  done
  log "Ablation complete."
elif [[ -n "$CONDITION" ]]; then
  case "$CONDITION" in
    baseline)         run_condition baseline --runtime-mode baseline --predictor mock ;;
    full_system)      run_condition full_system ;;
    no_plan)          run_condition no_plan --no-plan-extraction ;;
    no_diverg_guard)  run_condition no_diverg_guard --no-divergence-guard ;;
    naive_prefetch)   run_condition naive_prefetch --naive-prefetch ;;
    no_model_prefetch) run_condition no_model_prefetch --skip-resource-types vllm_model ;;
    no_data_prefetch) run_condition no_data_prefetch --skip-resource-types data_file ;;
    mock_predictor)   run_condition mock_predictor --predictor mock ;;
    *)                run_condition "$CONDITION" ;;
  esac
else
  # Regular (non-ablation) run
  run_condition "full_system"
fi
