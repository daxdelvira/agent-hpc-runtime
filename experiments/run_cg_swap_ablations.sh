#!/bin/bash
# ChemGraph swap-mode ablation runs (L40S node, GPUs 0-3, tp=4)
# 6 conditions x 5 runs = 30 total
# Each run: planner (32B-VL) starts → plan extracted → planner stopped →
#           worker (72B-Instruct) prefetched/loaded → WorkerAgent proceeds.
# Key metric: how much of the worker swap is hidden by remaining planner-side work.

set -uo pipefail   # no -e: one failed run should not kill the whole sweep

PROJ=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
CG_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python

AA_NVIDIA=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/lib/python3.11/site-packages/nvidia
TORCH_LIB=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$AA_NVIDIA/cudnn/lib:$AA_NVIDIA/cusparselt/lib:$AA_NVIDIA/nccl/lib:$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJ/workloads/AtomAgents:$PROJ/ChemGraph/src:${PYTHONPATH:-}
export VLLM_PLANNER_BASE_URL=http://localhost:8002/v1
export HF_HOME=$HOME/scratch/hf_home

cd $PROJ

# Redirect temp off a broken job-private /tmp and disable XALT so vLLM's Triton
# JIT compile at engine-core init succeeds (see setup/fix_tmp.sh).
source $PROJ/setup/fix_tmp.sh

log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_servers() {
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 20
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "$used" -gt 5000 ]; then
        log "WARNING: GPUs still show ${used}MiB used after kill — waiting 60s more"
        sleep 60
    fi
}

# --evict-worker-cache: every run starts from a COLD worker page cache so the
#   cold baseline is truly cold and Option-A staging benefit is measured fairly.
BASE_FLAGS="--workflow-type multi_agent \
  --model-name Qwen/Qwen2.5-72B-Instruct \
  --planner-model Qwen/Qwen2.5-VL-32B-Instruct \
  --base-url http://localhost:8001/v1 \
  --mace-device cpu \
  --hw-profile chemgraph_swap \
  --swap-models \
  --evict-worker-cache"

run_one() {
    local mode=$1 label=$2 i=$3 extra=${4:-}
    log "Starting $label run $i/5"
    kill_servers
    $CG_PYTHON experiments/chemgraph_exp.py \
        $BASE_FLAGS \
        --runtime-mode "$mode" \
        $extra \
        --run-id "cg_swap_${label}_${i}"
    log "Finished $label run $i/5 (exit $?)"
}

# --- Decisive Option-A comparison first (validate staging before the full sweep) ---
#   baseline      : cold sequential swap (no runtime)
#   full_system   : Option-A page-cache staging during planning -> warm swap
#   no_cache_stage: runtime on but staging OFF -> cold swap (isolates staging)
for i in 1 2 3 4 5; do run_one baseline baseline "$i"; done
for i in 1 2 3 4 5; do run_one real full_system "$i"; done
for i in 1 2 3 4 5; do run_one real no_cache_stage "$i" "--no-cache-stage"; done

# --- Remaining ablations ---
for i in 1 2 3 4 5; do run_one real no_plan "$i" "--no-plan-extraction"; done
for i in 1 2 3 4 5; do run_one real no_diverg_guard "$i" "--no-divergence-guard"; done
for i in 1 2 3 4 5; do run_one real naive_prefetch "$i" "--naive-prefetch"; done

kill_servers
log "All 30 swap ablation runs complete."
