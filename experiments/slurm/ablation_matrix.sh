#!/bin/bash
#SBATCH --job-name=ablation_matrix
#SBATCH --account=gts-ag117
#SBATCH -qembers
#SBATCH --gres=gpu:H200:4              # 4 GPUs: 72B on 2,3  +  32B on 0,1 (speculative)
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=18:00:00               # 8 conditions × ~2h each + buffer
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dax@gatech.edu

# ---------------------------------------------------------------------------
# Ablation matrix for agent-hpc-runtime
#
# Runs 8 conditions sequentially on a single node, each time launching the
# vLLM servers fresh, running the experiment, then tearing the servers down.
# All results land in results/ablation_<JOB_ID>/ for easy comparison.
#
# Conditions
# ----------
#   1. baseline            — no runtime layer, pure AtomAgents
#   2. full_system         — all components: plan + divergence + real prefetch
#   3. no_plan             — plan extraction disabled
#   4. no_diverg_guard     — no prefetch cancellation on mismatch
#   5. naive_prefetch      — prefetch every prediction regardless of confidence
#   6. no_model_prefetch   — skip vllm_model prefetch (data prefetch only)
#   7. no_data_prefetch    — skip data_file prefetch (model prefetch only)
#   8. mock_predictor      — mock (rule-based) predictor vs learned
#
# Override config without editing this file:
#   MODEL_ENV=my_env EXP_ENV=my_atoms sbatch experiments/slurm/ablation_matrix.sh
#
# To run a subset of conditions:
#   CONDITIONS="baseline full_system" sbatch experiments/slurm/ablation_matrix.sh
# ---------------------------------------------------------------------------

set -Eeuo pipefail

############### CONFIG ###############
MODEL_ENV="${MODEL_ENV:-dummy_agent}"
EXP_ENV="${EXP_ENV:-atoms}"
HW_PROFILE="${HW_PROFILE:-l40s}"
PORT_72B="${PORT_72B:-8001}"
PORT_32B="${PORT_32B:-8002}"
MODEL_72B="${MODEL_72B:-Qwen/Qwen2.5-72B-Instruct}"
MODEL_32B="${MODEL_32B:-Qwen/Qwen2.5-32B-Instruct}"
VLLM_ARGS_72B="${VLLM_ARGS_72B:- --enable-auto-tool-choice --tool-call-parser hermes --dtype=half --tensor-parallel-size=2 --max-model-len=16384}"
VLLM_ARGS_32B="${VLLM_ARGS_32B:- --enable-auto-tool-choice --tool-call-parser hermes --dtype=half --tensor-parallel-size=2 --max-model-len=16384}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2700}"   # 45 min for 72B cold NFS load
EXP_TIMEOUT="${EXP_TIMEOUT:-90m}"         # per-condition timeout
# Conditions to run (space-separated); set to subset for quick tests
CONDITIONS="${CONDITIONS:-baseline full_system no_plan no_diverg_guard naive_prefetch no_model_prefetch no_data_prefetch mock_predictor}"
######################################

log(){ echo "[$(date +'%F %T')] $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_OUT="${REPO_ROOT}/results/ablation_${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_OUT"

log "Node: $(hostname) | SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
log "Repo root : $REPO_ROOT"
log "Output dir: $JOB_OUT"
log "Conditions: $CONDITIONS"

module load anaconda3

# ---------------------------------------------------------------------------
# Server management helpers
# ---------------------------------------------------------------------------

SERVER_PIDS=()

kill_servers(){
  log "Stopping vLLM servers…"
  for pid in "${SERVER_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -"$pid" >/dev/null 2>&1 || true
  done
  SERVER_PIDS=()
  sleep 5
}

trap kill_servers EXIT

wait_healthy(){
  local name="$1" port="$2" timeout_s="$3"
  log "Waiting for $name on port $port (timeout ${timeout_s}s)…"
  for ((i=0;i<timeout_s;i++)); do
    if curl -sSf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
       || curl -sSf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "$name is healthy."
      return 0
    fi
    sleep 1
  done
  log "ERROR: $name did not become healthy within ${timeout_s}s"
  return 1
}

start_server(){
  local name="$1" gpus="$2" port="$3" model="$4" vllm_args="$5" logfile="$6"
  log "Starting $name  GPUs=[$gpus]  port=$port"
  : > "$logfile"
  setsid bash -c "
    source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
    conda activate $MODEL_ENV
    export CUDA_VISIBLE_DEVICES=$gpus
    exec python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port $port --model $model $vllm_args \
      >>$logfile 2>&1
  " &
  echo $!
}

# ---------------------------------------------------------------------------
# Baseline runner — no vLLM servers needed (pure AtomAgents with all models
# pre-loaded via the standard orchestrator path, or --runtime-mode baseline
# which bypasses the runtime entirely).
#
# For a fair wall-time comparison, we want models pre-loaded the same way as
# in non-real conditions. The baseline uses observe_only mode + both servers
# running so AtomAgents can call them freely.
# ---------------------------------------------------------------------------

run_condition(){
  local cond="$1"
  shift
  local extra_flags="$*"

  log "========================================"
  log "Running condition: $cond"
  log "========================================"

  local logfile="$JOB_OUT/${cond}.log"
  : > "$logfile"

  set +e
  timeout --preserve-status "$EXP_TIMEOUT" bash -c "
    source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
    set +u; conda activate '$EXP_ENV'; set -u
    cd '$REPO_ROOT'
    python -u experiments/atomagents_exp2.py \
      --runtime-mode real \
      --predictor learned \
      --hw-profile '$HW_PROFILE' \
      --swap-models \
      --no-start-models \
      --results-dir '$JOB_OUT' \
      --log-dir '$JOB_OUT/traces' \
      --condition '$cond' \
      $extra_flags \
      2>&1
  " >> "$logfile"
  local rc=$?
  set -e

  if [[ $rc -eq 124 ]]; then
    log "WARN: condition $cond timed out"
  elif [[ $rc -ne 0 ]]; then
    log "WARN: condition $cond exited with code $rc"
  else
    log "OK: condition $cond finished"
  fi
}

# ---------------------------------------------------------------------------
# Special baseline: no runtime at all, use observe_only for fair comparison
# ---------------------------------------------------------------------------

run_baseline(){
  local logfile="$JOB_OUT/baseline.log"
  : > "$logfile"
  log "Running condition: baseline (mode=baseline)"

  set +e
  timeout --preserve-status "$EXP_TIMEOUT" bash -c "
    source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
    set +u; conda activate '$EXP_ENV'; set -u
    cd '$REPO_ROOT'
    python -u experiments/atomagents_exp2.py \
      --runtime-mode baseline \
      --predictor mock \
      --hw-profile '$HW_PROFILE' \
      --swap-models \
      --no-start-models \
      --results-dir '$JOB_OUT' \
      --log-dir '$JOB_OUT/traces' \
      --condition baseline \
      2>&1
  " >> "$logfile"
  local rc=$?
  set -e
  [[ $rc -ne 0 ]] && log "WARN: baseline exited with code $rc" || log "OK: baseline finished"
}

# ---------------------------------------------------------------------------
# Per-condition: start servers → run → tear down → next condition
#
# For conditions that need real model prefetch (full_system, no_plan,
# no_diverg_guard, naive_prefetch, no_data_prefetch, mock_predictor):
#   - Only start 72B; ModelPrefetchExecutor speculatively starts 32B.
#
# For conditions that skip model prefetch (no_model_prefetch, baseline):
#   - Start both servers so the experiment has models to call.
# ---------------------------------------------------------------------------

mkdir -p "$JOB_OUT/traces"

for COND in $CONDITIONS; do
  case "$COND" in

    baseline)
      # Start both servers (no speculative swap — models need to be available)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      S32=$(start_server "Qwen-32B" "0,1" "$PORT_32B" "$MODEL_32B" "$VLLM_ARGS_32B" \
             "$JOB_OUT/${COND}_qwen32b.log"); SERVER_PIDS+=("$S32")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      wait_healthy "Qwen-32B" "$PORT_32B" "300"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_baseline
      kill_servers
      ;;

    no_model_prefetch)
      # No speculative swap, start both so models are always available
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      S32=$(start_server "Qwen-32B" "0,1" "$PORT_32B" "$MODEL_32B" "$VLLM_ARGS_32B" \
             "$JOB_OUT/${COND}_qwen32b.log"); SERVER_PIDS+=("$S32")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      wait_healthy "Qwen-32B" "$PORT_32B" "300"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --skip-resource-types "vllm_model"
      kill_servers
      ;;

    full_system)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND"
      kill_servers
      ;;

    no_plan)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --no-plan-extraction
      kill_servers
      ;;

    no_diverg_guard)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --no-divergence-guard
      kill_servers
      ;;

    naive_prefetch)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --naive-prefetch
      kill_servers
      ;;

    no_data_prefetch)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --skip-resource-types "data_file"
      kill_servers
      ;;

    mock_predictor)
      S72=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
             "$JOB_OUT/${COND}_qwen72b.log"); SERVER_PIDS+=("$S72")
      wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"
      export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
      export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"
      run_condition "$COND" --predictor mock
      kill_servers
      ;;

    *)
      log "WARN: unknown condition '$COND', skipping"
      ;;
  esac

  log "Sleeping 30s between conditions…"
  sleep 30
done

# ---------------------------------------------------------------------------
# Post-run: generate ablation report
# ---------------------------------------------------------------------------

log "Generating ablation report…"
set +e
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
set +u; conda activate "$EXP_ENV"; set -u

REPORT="$JOB_OUT/ablation_report_${SLURM_JOB_ID:-local}"
python -u "$REPO_ROOT/runtime/analysis/ablation_report.py" \
       "$JOB_OUT"/summary_*.json \
       --csv "${REPORT}.csv" \
       2>&1 | tee "${REPORT}.txt"

echo ""
echo "Report : ${REPORT}.txt"
echo "CSV    : ${REPORT}.csv"
echo "Results: $JOB_OUT"

log "Ablation matrix complete."
