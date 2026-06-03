#!/bin/bash
#SBATCH --job-name=chemgraph_mace
#SBATCH --account=gts-ag117
#SBATCH -qembers
#SBATCH --gres=gpu:H200:1              # 1 GPU for MACE calculation
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dax@gatech.edu

set -Eeuo pipefail

############### CONFIG ###############
EXP_ENV="${EXP_ENV:-chemgraph}"       # conda env with ChemGraph + MACE installed
RUNTIME_MODE="${RUNTIME_MODE:-observe_only}"   # baseline | observe_only | simulated | real
PREDICTOR="${PREDICTOR:-mock}"                 # mock | learned
MACE_DEVICE="${MACE_DEVICE:-cuda}"             # cpu | cuda
TASK="${TASK:-default}"                        # default | extended
LLM_MODEL="${LLM_MODEL:-}"                     # blank = use CHEMGRAPH_MODEL env or gpt-4o-mini
LLM_BASE_URL="${LLM_BASE_URL:-}"               # blank = use OPENAI_API_KEY or Groq
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
VLLM_ARGS="${VLLM_ARGS:- --enable-auto-tool-choice --tool-call-parser hermes --dtype=float16 --max-model-len=8192}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
EXP_TIMEOUT="${EXP_TIMEOUT:-60m}"
######################################

log(){ echo "[$(date +'%F %T')] $*"; }

PIDS_TO_KILL=()
kill_tree(){
  for pid in "${PIDS_TO_KILL[@]}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -"${pid}" >/dev/null 2>&1 || true
  done
}
trap kill_tree EXIT

wait_healthy(){
  local name="$1" port="$2" timeout_s="$3"
  log "Waiting for $name on port $port (timeout ${timeout_s}s)…"
  for ((i=0;i<timeout_s;i++)); do
    if curl -sSf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "$name is healthy."
      return 0
    fi
    sleep 1
  done
  log "ERROR: $name did not become healthy within ${timeout_s}s"
  return 1
}

# ---- Start ----
module load anaconda3
log "Node: $(hostname) | SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log "Repo root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# Optionally start a local vLLM server for the LLM reasoning component.
# Skip if LLM_BASE_URL already points to an external API (Groq, OpenAI, etc).
# ---------------------------------------------------------------------------
START_VLLM=0
if [[ -z "$LLM_BASE_URL" && -z "$LLM_MODEL" ]]; then
  # No external endpoint configured → start local vLLM
  START_VLLM=1
fi

if [[ "$START_VLLM" -eq 1 ]]; then
  log "Starting local vLLM server for $VLLM_MODEL on port $VLLM_PORT …"
  (
    source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
    conda activate "$EXP_ENV"
    export CUDA_VISIBLE_DEVICES=0
    exec python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port "$VLLM_PORT" \
      --model "$VLLM_MODEL" $VLLM_ARGS \
      >> "vllm_${SLURM_JOB_ID:-local}.log" 2>&1
  ) &
  VLLM_PID=$!
  PIDS_TO_KILL+=("$VLLM_PID")
  wait_healthy "vLLM" "$VLLM_PORT" "$HEALTH_TIMEOUT"
  export LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  export LLM_MODEL="$VLLM_MODEL"
  export OPENAI_API_KEY="dummy"
fi

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
[[ "$TASK" == "extended" ]] && EXTRA_ARGS="$EXTRA_ARGS --extended-task"
[[ -n "$LLM_MODEL" ]]       && EXTRA_ARGS="$EXTRA_ARGS --model-name $LLM_MODEL"
[[ -n "$LLM_BASE_URL" ]]    && EXTRA_ARGS="$EXTRA_ARGS --base-url $LLM_BASE_URL"

log "Launching chemgraph_exp.py  mode=$RUNTIME_MODE  predictor=$PREDICTOR  device=$MACE_DEVICE"

set +e
timeout --preserve-status "$EXP_TIMEOUT" bash -c "
  set -Eeuo pipefail
  source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
  conda activate '$EXP_ENV'
  export RUNTIME_ENABLED=1
  cd '$REPO_ROOT'
  python -u experiments/chemgraph_exp.py \
    --runtime-mode '$RUNTIME_MODE' \
    --predictor '$PREDICTOR' \
    --mace-device '$MACE_DEVICE' \
    $EXTRA_ARGS \
    >> 'chemgraph_exp_${SLURM_JOB_ID:-local}.log' 2>&1
"
RC=$?
set -e

if [[ $RC -eq 124 ]]; then
  log "Experiment timed out (${EXP_TIMEOUT})."
else
  log "Experiment finished with exit code $RC."
fi

sleep 3
log "Job complete."

# Usage examples:
#
#   # Observe-only (safe, no prefetch I/O)
#   sbatch experiments/slurm/chemgraph_exp.sh
#
#   # Baseline (no runtime overhead)
#   RUNTIME_MODE=baseline sbatch experiments/slurm/chemgraph_exp.sh
#
#   # Real MACE prefetch
#   RUNTIME_MODE=real MACE_DEVICE=cuda sbatch experiments/slurm/chemgraph_exp.sh
#
#   # With external Groq LLM (no local vLLM needed)
#   LLM_BASE_URL=https://api.groq.com/openai/v1 \
#   LLM_MODEL=llama3-70b-8192 \
#   RUNTIME_MODE=real \
#   sbatch experiments/slurm/chemgraph_exp.sh
