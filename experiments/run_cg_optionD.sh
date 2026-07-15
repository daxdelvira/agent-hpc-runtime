#!/bin/bash
# Option D validation: hide the aggregator model load inside the ensemble MACE
# compute window.  The worker (72B, GPUs 0-3) calls run_mace_ensemble (~11 min
# GPU-idle CPU compute); the adapter prefetches the DISTINCT aggregator model
# (32B, GPUs 4-5, co-resident) during that window so it is hot when control
# reaches AggregatorAgent.
#   baseline : aggregator loads ON-DEMAND at AggregatorAgent (~full load cost)
#   real     : aggregator prefetched during compute window (aggregator_swap_wait ~ 0)
# Emits aggregator_prefetch_start / aggregator_swap_wait into the shared trace.
set -uo pipefail
PROJ=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
CG_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python
AA_NVIDIA=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/lib/python3.11/site-packages/nvidia
TORCH_LIB=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$AA_NVIDIA/cudnn/lib:$AA_NVIDIA/cusparselt/lib:$AA_NVIDIA/nccl/lib:$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJ/workloads/AtomAgents:$PROJ/ChemGraph/src:${PYTHONPATH:-}
export VLLM_PLANNER_BASE_URL=http://localhost:8002/v1
export VLLM_AGGREGATOR_BASE_URL=http://localhost:8004/v1
export HF_HOME=$HOME/scratch/hf_home

cd $PROJ
source $PROJ/setup/fix_tmp.sh

log() { echo "[$(date '+%H:%M:%S')] $*"; }
kill_servers() {
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 20
}

MODE=${1:-real}
RUNID=${2:-cg_optionD_${MODE}}

BASE_FLAGS="--workflow-type multi_agent \
  --model-name Qwen/Qwen2.5-72B-Instruct \
  --planner-model Qwen/Qwen2.5-VL-32B-Instruct \
  --aggregator-model Qwen/Qwen2.5-VL-32B-Instruct-Aggregator \
  --aggregator-base-url http://localhost:8004/v1 \
  --base-url http://localhost:8001/v1 \
  --mace-device cpu \
  --hw-profile chemgraph_swap \
  --swap-models \
  --evict-worker-cache \
  --ensemble-dataset data/materials_ensemble"

log "Option D run: mode=$MODE run-id=$RUNID"
kill_servers
$CG_PYTHON experiments/chemgraph_exp.py \
    $BASE_FLAGS \
    --runtime-mode "$MODE" \
    --run-id "$RUNID"
log "Finished Option D run (exit $?)"
kill_servers
