#!/usr/bin/env bash
# start_models.sh — Launch both vLLM servers and wait until they are ready.
#
# Usage:
#   bash setup/start_models.sh                              # auto-detect paths
#   bash setup/start_models.sh --models-dir /data/models   # explicit weight dirs
#   bash setup/start_models.sh --72b-only                  # skip 32B
#   bash setup/start_models.sh --stop                      # kill both servers
#
# HuggingFace cache: if HF_HOME is exported in the calling shell, model names
# are passed directly to vLLM (which resolves weights from the cache).
# Otherwise --models-dir must point to a dir with qwen_72b/ and qwen_32b/ subdirs.
#
# Logs: /tmp/vllm_72b.log  /tmp/vllm_32b.log
set -euo pipefail

MODELS_DIR="${MODEL_BASE_DIR:-/workspace/models}"
ONLY_72B=0
STOP=0
LOG_DIR=/tmp

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift ;;
    --72b-only)   ONLY_72B=1 ;;
    --stop)       STOP=1 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# --stop: kill any running vLLM servers
# ---------------------------------------------------------------------------
if [[ $STOP -eq 1 ]]; then
  echo "[start_models] Stopping vLLM servers..."
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "  killed" || echo "  none running"
  exit 0
fi

# ---------------------------------------------------------------------------
# Resolve vLLM python
# ---------------------------------------------------------------------------
VLLM_PYTHON=$(conda run -n vllm_clean which python 2>/dev/null || \
              conda run -n vllm       which python 2>/dev/null || echo "")
if [[ -z "$VLLM_PYTHON" ]]; then
  echo "[start_models] ERROR: no vllm conda env found (tried: vllm_clean, vllm)." >&2
  echo "  Run: conda env create -f setup/environment_vllm.yml" >&2
  exit 1
fi
echo "[start_models] vLLM python: $VLLM_PYTHON"

# ---------------------------------------------------------------------------
# Detect GPU count
# ---------------------------------------------------------------------------
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "[start_models] GPUs detected: $GPU_COUNT"

if [[ $GPU_COUNT -lt 2 ]]; then
  echo "[start_models] ERROR: need at least 2 GPUs" >&2
  exit 1
fi

# GPU split: 72B on upper half, 32B on lower half
if [[ $GPU_COUNT -ge 4 ]]; then
  GPUS_72B="2,3"
  TP_72B=2
  GPUS_32B="0,1"
  TP_32B=2
else
  # 2 GPUs: run only 72B (or 32B if --72b-only is flipped)
  GPUS_72B="0,1"
  TP_72B=2
  GPUS_32B="0,1"
  TP_32B=2
  if [[ $ONLY_72B -eq 0 ]]; then
    echo "[start_models] Only 2 GPUs — cannot run both models simultaneously. Using --72b-only."
    ONLY_72B=1
  fi
fi

# ---------------------------------------------------------------------------
# Helper: wait until an HTTP endpoint responds
# ---------------------------------------------------------------------------
wait_ready() {
  local name="$1" port="$2" log="$3" timeout="${4:-2700}"
  echo "[start_models] Waiting for $name on port $port (timeout ${timeout}s)..."
  local t=0
  while ! curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
    if grep -q "ERROR\|Traceback\|RuntimeError\|CUDA error" "$log" 2>/dev/null; then
      echo "[start_models] ERROR: $name failed to start. Last 20 lines of $log:"
      tail -20 "$log"
      exit 1
    fi
    if [[ $t -ge $timeout ]]; then
      echo "[start_models] TIMEOUT waiting for $name after ${timeout}s. Last 20 lines of $log:"
      tail -20 "$log"
      exit 1
    fi
    sleep 10
    t=$((t + 10))
    echo "  ... ${t}s elapsed ($(tail -1 "$log" 2>/dev/null | cut -c1-100))"
  done
  echo "[start_models] $name ready after ${t}s"
}

# ---------------------------------------------------------------------------
# Resolve model identifiers: HF cache (by name) or explicit directory
# ---------------------------------------------------------------------------
if [[ -n "${HF_HOME:-}" ]]; then
  echo "[start_models] HF_HOME=$HF_HOME — using HF cache for model resolution"
  export HF_HOME
  MODEL_ID_72B="Qwen/Qwen2.5-VL-72B-Instruct"
  MODEL_ID_32B="Qwen/Qwen2.5-VL-32B-Instruct"
else
  MODEL_ID_72B="$MODELS_DIR/qwen_72b"
  MODEL_ID_32B="$MODELS_DIR/qwen_32b"
  if [[ ! -d "$MODEL_ID_72B" ]]; then
    echo "[start_models] ERROR: 72B weights not found at $MODEL_ID_72B" >&2
    echo "  Set HF_HOME to use HuggingFace cache, or --models-dir to an explicit path." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Launch 72B
# ---------------------------------------------------------------------------
LOG_72B="$LOG_DIR/vllm_72b.log"

echo "[start_models] Starting 72B on GPUs $GPUS_72B → port 8001 (log: $LOG_72B)"
CUDA_VISIBLE_DEVICES=$GPUS_72B "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID_72B" \
  --served-model-name "Qwen/Qwen2.5-VL-72B-Instruct" \
  --tensor-parallel-size "$TP_72B" \
  --gpu-memory-utilization 0.97 \
  --max-model-len 16384 \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype float16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  >"$LOG_72B" 2>&1 &
PID_72B=$!
echo "[start_models] 72B PID: $PID_72B"

# ---------------------------------------------------------------------------
# Launch 32B (unless --72b-only)
# ---------------------------------------------------------------------------
if [[ $ONLY_72B -eq 0 ]]; then
  LOG_32B="$LOG_DIR/vllm_32b.log"

  if [[ -z "${HF_HOME:-}" && ! -d "$MODEL_ID_32B" ]]; then
    echo "[start_models] ERROR: 32B weights not found at $MODEL_ID_32B" >&2
    exit 1
  fi

  echo "[start_models] Starting 32B on GPUs $GPUS_32B → port 8002 (log: $LOG_32B)"
  CUDA_VISIBLE_DEVICES=$GPUS_32B "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID_32B" \
    --served-model-name "Qwen/Qwen2.5-VL-32B-Instruct" \
    --tensor-parallel-size "$TP_32B" \
    --gpu-memory-utilization 0.82 \
    --max-model-len 16384 \
    --port 8002 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype float16 \
    --disable-custom-all-reduce \
    >"$LOG_32B" 2>&1 &
  PID_32B=$!
  echo "[start_models] 32B PID: $PID_32B"
fi

# ---------------------------------------------------------------------------
# Wait for both to be ready
# ---------------------------------------------------------------------------
wait_ready "72B" 8001 "$LOG_72B" 2700

if [[ $ONLY_72B -eq 0 ]]; then
  wait_ready "32B" 8002 "$LOG_32B" 1200
fi

echo ""
echo "================================================================"
echo "  Both models ready."
echo "  72B → http://localhost:8001/v1   (log: $LOG_72B)"
if [[ $ONLY_72B -eq 0 ]]; then
  echo "  32B → http://localhost:8002/v1   (log: $LOG_32B)"
fi
echo "  Run experiments now."
echo "================================================================"
