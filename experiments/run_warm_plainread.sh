#!/bin/bash
set -uo pipefail
PROJ=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
CG_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python
AA_NVIDIA=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/lib/python3.11/site-packages/nvidia
TORCH_LIB=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$AA_NVIDIA/cudnn/lib:$AA_NVIDIA/cusparselt/lib:$AA_NVIDIA/nccl/lib:$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJ:$PROJ/workloads/AtomAgents:$PROJ/ChemGraph/src:${PYTHONPATH:-}
export HF_HOME=$HOME/scratch/hf_home
cd $PROJ
source $PROJ/setup/fix_tmp.sh
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 12
exec $CG_PYTHON -u experiments/warm_plainread_72b.py
