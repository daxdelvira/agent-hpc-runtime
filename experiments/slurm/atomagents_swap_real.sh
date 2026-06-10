#!/bin/bash
#SBATCH --job-name=atomagents_swap_real
#SBATCH --account=gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -qembers
#SBATCH --gres=gpu:4                   # 4× RTX PRO 6000 Blackwell: 72B on 2,3 + 32B on 0,1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dax@gatech.edu

# ---------------------------------------------------------------------------
# AtomAgents Exp2 — REAL prefetch with model swap
#
# Key difference from autorun_experiment.sh:
#   - Only the base 72B model is pre-loaded.
#   - The runtime (ModelPrefetchExecutor) starts 32B speculatively when it
#     predicts computation_task_screw_dislocation is coming.
#   - Overlap metric = time 32B was loading while 72B was reasoning.
#
# Override any config var without editing this file:
#   MODEL_ENV=my_env PREDICTOR=learned sbatch experiments/slurm/atomagents_swap_real.sh
# ---------------------------------------------------------------------------

set -Eeuo pipefail

############### CONFIG ###############
MODEL_ENV="${MODEL_ENV:-vllm_clean128}"     # conda env for vLLM servers (CUDA 12.8, vLLM 0.19.0)
EXP_ENV="${EXP_ENV:-atoms}"                # conda env for the experiment
PREDICTOR="${PREDICTOR:-learned}"          # mock | learned
HW_PROFILE="${HW_PROFILE:-blackwell}"      # blackwell | l40s | rtx6000
PORT_72B="${PORT_72B:-8001}"
PORT_32B="${PORT_32B:-8002}"
# Local VL snapshot paths (downloaded to ~/scratch/hf_home/hub/).
_HF_HUB="${HOME}/scratch/hf_home/hub"
MODEL_72B="${MODEL_72B:-${_HF_HUB}/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5}"
MODEL_32B="${MODEL_32B:-${_HF_HUB}/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da}"
# --served-model-name aliases the snapshot path to the HF model ID that
# AtomAgents config_list uses for routing requests.
VLLM_ARGS_72B="${VLLM_ARGS_72B:- --enable-auto-tool-choice --tool-call-parser hermes --dtype=float16 --tensor-parallel-size=2 --max-model-len=16384 --gpu-memory-utilization=0.97 --disable-custom-all-reduce --enforce-eager --served-model-name Qwen/Qwen2.5-VL-72B-Instruct}"
VLLM_ARGS_32B="${VLLM_ARGS_32B:- --enable-auto-tool-choice --tool-call-parser hermes --dtype=float16 --tensor-parallel-size=2 --max-model-len=16384 --gpu-memory-utilization=0.82 --disable-custom-all-reduce --served-model-name Qwen/Qwen2.5-VL-32B-Instruct}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2700}"   # 45 min for 72B
EXP_TIMEOUT="${EXP_TIMEOUT:-120m}"
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

track_gpus(){
  local tag="$1" gpus="$2" outfile="$3"
  log "Starting GPU tracker: $tag → $outfile"
  (
    echo "timestamp,index,util.gpu [%],util.mem [%],mem.used [MiB],mem.total [MiB]"
    while true; do
      CUDA_VISIBLE_DEVICES="$gpus" nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader
      sleep 1
    done
  ) > "$outfile" &
  echo $!
}

start_server(){
  local name="$1" gpus="$2" port="$3" model="$4" args="$5" logfile="$6"
  log "Starting $name  GPUs=[$gpus]  port=$port  model=$model"
  : > "$logfile"
  local tmpdir="${REPO_ROOT}/.vllm_tmp_${port}"
  mkdir -p "$tmpdir"
  setsid bash -c "
    source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
    conda activate $MODEL_ENV
    export CUDA_VISIBLE_DEVICES=$gpus
    export TMPDIR='$tmpdir'
    export TEMP='$tmpdir'
    export TMP='$tmpdir'
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export XALT_EXECUTABLE_TRACKING=no
    exec python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 --port $port --model $model $args \
      >>$logfile 2>&1
  " &
  echo $!
}

# ---- Start ----
module load anaconda3
log "Node: $(hostname) | SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log "Repo root: $REPO_ROOT"

# GPU trackers
T72_PID=$(track_gpus "Qwen-72B" "2,3" "gpu_track_qwen72b_${SLURM_JOB_ID}.csv"); PIDS_TO_KILL+=("$T72_PID")
T32_PID=$(track_gpus "Qwen-32B" "0,1" "gpu_track_qwen32b_${SLURM_JOB_ID}.csv"); PIDS_TO_KILL+=("$T32_PID")

# ---- Pre-load ONLY the 72B base model ----
# The runtime (ModelPrefetchExecutor) will start 32B speculatively when it
# predicts that computation_task_screw_dislocation is about to fire.
S72_PID=$(start_server "Qwen-72B" "2,3" "$PORT_72B" "$MODEL_72B" "$VLLM_ARGS_72B" \
          "qwen72b_${SLURM_JOB_ID}.log"); PIDS_TO_KILL+=("$S72_PID")

log "Waiting for 72B only…  (32B will be started speculatively by the runtime)"
wait_healthy "Qwen-72B" "$PORT_72B" "$HEALTH_TIMEOUT"

export QWEN72B_ENDPOINT="http://127.0.0.1:${PORT_72B}/v1"
export QWEN32B_ENDPOINT="http://127.0.0.1:${PORT_32B}/v1"  # 32B not yet started

log "72B is ready. Launching experiment (runtime=real, predictor=$PREDICTOR)…"

set +e
timeout --preserve-status "$EXP_TIMEOUT" bash -c "
  set -Eeuo pipefail
  source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
  set +u; conda activate '$EXP_ENV'; set -u
  cd '$REPO_ROOT'
  python -u experiments/atomagents_exp2.py \
    --runtime-mode real \
    --predictor '$PREDICTOR' \
    --hw-profile '$HW_PROFILE' \
    --swap-models \
    >> 'atomagents_swap_real_${SLURM_JOB_ID:-local}.log' 2>&1
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

# ---------------------------------------------------------------------------
# Companion baseline run (no swap, both models pre-loaded)
# Run this to establish the wall-time comparison point:
#   sbatch workloads/AtomAgents/autorun_experiment.sh
#   (or: RUNTIME_MODE=baseline sbatch workloads/AtomAgents/autorun_experiment.sh)
# ---------------------------------------------------------------------------
