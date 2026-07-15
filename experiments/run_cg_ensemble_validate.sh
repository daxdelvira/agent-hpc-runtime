#!/bin/bash
# run_cg_ensemble_validate.sh — Step 1 of ChemGraph Option D.
#
# Validate the out-of-core MACE *ensemble* tool END-TO-END before building the
# compute-window model prefetch on top of it.  Confirms three things:
#   (1) the WorkerAgent (72B) actually CALLS run_mace_ensemble,
#   (2) the tool runs a genuine multi-minute batch over data/materials_ensemble
#       (200 real COD crystals), and
#   (3) the GPU sits idle during that window (MACE is CPU) — i.e. there IS a
#       compute window in which a model load could be hidden (Option D).
#
# Single text-72B server, NO swap: planner defaults to the worker LLM, so one
# server serves planner+worker+aggregator.  observe_only = runtime watches and
# emits predictions but does no prefetch I/O (nothing to prefetch here yet).
set -uo pipefail

PROJ=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
CG_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python
VLLM_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean128/bin/python
SNAP_72B=/storage/home/hcoda1/7/avandevoorde3/scratch/hf_home/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31

AA_NVIDIA=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/lib/python3.11/site-packages/nvidia
TORCH_LIB=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$AA_NVIDIA/cudnn/lib:$AA_NVIDIA/cusparselt/lib:$AA_NVIDIA/nccl/lib:$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJ/workloads/AtomAgents:$PROJ/ChemGraph/src:${PYTHONPATH:-}
export HF_HOME=$HOME/scratch/hf_home
export OPENAI_API_KEY=dummy
cd $PROJ

# Redirect temp off a broken job-private /tmp and disable XALT so vLLM's Triton
# JIT compile at engine-core init succeeds (see setup/fix_tmp.sh).
source $PROJ/setup/fix_tmp.sh

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Killing any existing vLLM servers"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 15

VLOG=/tmp/vllm_72b_ensemble.log
log "Starting text-72B on GPUs 0-3 → port 8001 (log: $VLOG)"
$VLLM_PYTHON -u -m vllm.entrypoints.openai.api_server \
  --model "$SNAP_72B" \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype float16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --served-model-name Qwen/Qwen2.5-72B-Instruct \
  > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true' EXIT

log "Waiting for 72B on port 8001 (timeout 2700s)..."
t0=$(date +%s)
while ! curl -sf "http://localhost:8001/v1/models" >/dev/null 2>&1; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then log "vLLM died during startup — see $VLOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 2700 ]; then log "timeout waiting for server"; exit 1; fi
  sleep 5
done
log "72B ready in $(( $(date +%s) - t0 ))s"

log "Launching ChemGraph ensemble run (observe_only, GPU idle expected during tool)"
$CG_PYTHON experiments/chemgraph_exp.py \
  --workflow-type multi_agent \
  --model-name Qwen/Qwen2.5-72B-Instruct \
  --base-url http://localhost:8001/v1 \
  --mace-device cpu \
  --runtime-mode observe_only \
  --ensemble-dataset data/materials_ensemble \
  --run-id cg_ensemble_validate
rc=$?
log "ChemGraph ensemble run finished (exit $rc)"
exit $rc
