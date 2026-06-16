#!/bin/bash
# ---------------------------------------------------------------------------
# run_l40s.sh — Run AtomAgents Exp3 (3-model forced-swap) on a reserved
# L40S node.  Uses atomagents_exp3.py with MODELS_L40S (all models share
# GPUs 0-5, tp=6).  Only one model can be resident at a time; ModelRouter
# stops the current model and loads the next on every agent-role transition.
#
# Usage (from repo root or any directory):
#   bash experiments/run_l40s.sh                      # single full_system run
#   bash experiments/run_l40s.sh --ablation           # all ablation conditions
#   bash experiments/run_l40s.sh --condition no_plan  # one condition
#
# Tuning knobs:
#   LAMMPS_SLOWDOWN_S (default 300): seconds of sleep injected after each
#     LAMMPS relax step to simulate NFS-backed runs and create prefetch windows.
#   EXP_TIMEOUT (default 240m): per-condition wall-time limit.
# ---------------------------------------------------------------------------

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ENV="${EXP_ENV:-atoms}"
PREDICTOR="${PREDICTOR:-learned}"
HW_PROFILE="${HW_PROFILE:-l40s}"
EXP_TIMEOUT="${EXP_TIMEOUT:-240m}"
NO_START_MODELS="${NO_START_MODELS:-0}"
ABLATION="${ABLATION:-0}"
CONDITION="${CONDITION:-}"
LAMMPS_SLOWDOWN_S="${LAMMPS_SLOWDOWN_S:-300}"
export LAMMPS_SLOWDOWN_S

log(){ echo "[$(date +'%F %T')] $*"; }

for arg in "$@"; do
  case "$arg" in
    --ablation)    ABLATION=1 ;;
    --condition=*) CONDITION="${arg#--condition=}" ;;
    --condition)   shift; CONDITION="$1" ;;
  esac
done

log "Node       : $(hostname)"
log "Repo root  : $REPO_ROOT"
log "HW profile : $HW_PROFILE"
log "Predictor  : $PREDICTOR"
log "LAMMPS slow: ${LAMMPS_SLOWDOWN_S}s per relax step"

# L40S nodes have /tmp mounted; also set /dev/shm as fallback for vLLM ZMQ sockets.
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

BASE_FLAGS=(
  --runtime-mode real
  --predictor "$PREDICTOR"
  --hw-profile "$HW_PROFILE"
  --swap-models
  --lammps-slowdown "$LAMMPS_SLOWDOWN_S"
)
[[ "$NO_START_MODELS" == "1" ]] && BASE_FLAGS+=(--no-start-models)

cleanup_vllm(){
  pkill -f "vllm_clean128" 2>/dev/null || true
  local attempts=0
  while (( attempts < 60 )); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '{s+=$1} END {print s}')
    [[ "$used" -lt 4096 ]] && break
    sleep 5
    (( attempts++ )) || true
  done
  log "GPUs cleared (${attempts} polls). Continuing."
}

run_condition(){
  local label="$1"; shift
  local extra_flags=("$@")
  local ts; ts=$(date +'%Y%m%d_%H%M%S')
  local logfile="${REPO_ROOT}/logs/l40s_${label}_${ts}.log"
  mkdir -p "${REPO_ROOT}/logs"

  cleanup_vllm

  log "===== Condition: $label ====="
  set +e
  timeout --preserve-status "$EXP_TIMEOUT" \
    "$ATOMS_PYTHON" -u "${REPO_ROOT}/experiments/atomagents_exp3.py" \
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
      baseline)          run_condition baseline --runtime-mode baseline --predictor mock ;;
      full_system)       run_condition full_system ;;
      no_plan)           run_condition no_plan --no-plan-extraction ;;
      no_diverg_guard)   run_condition no_diverg_guard --no-divergence-guard ;;
      naive_prefetch)    run_condition naive_prefetch --naive-prefetch ;;
      no_model_prefetch) run_condition no_model_prefetch --skip-resource-types vllm_model ;;
      no_data_prefetch)  run_condition no_data_prefetch --skip-resource-types data_file ;;
      mock_predictor)    run_condition mock_predictor --predictor mock ;;
      *)                 log "Unknown condition: $COND, skipping" ;;
    esac
    [[ "$COND" != "${CONDITIONS##* }" ]] && { log "Sleeping 30s between conditions…"; sleep 30; }
  done
  log "Ablation complete."
elif [[ -n "$CONDITION" ]]; then
  case "$CONDITION" in
    baseline)          run_condition baseline --runtime-mode baseline --predictor mock ;;
    full_system)       run_condition full_system ;;
    no_plan)           run_condition no_plan --no-plan-extraction ;;
    no_diverg_guard)   run_condition no_diverg_guard --no-divergence-guard ;;
    naive_prefetch)    run_condition naive_prefetch --naive-prefetch ;;
    no_model_prefetch) run_condition no_model_prefetch --skip-resource-types vllm_model ;;
    no_data_prefetch)  run_condition no_data_prefetch --skip-resource-types data_file ;;
    mock_predictor)    run_condition mock_predictor --predictor mock ;;
    *)                 run_condition "$CONDITION" ;;
  esac
else
  run_condition "full_system"
fi
