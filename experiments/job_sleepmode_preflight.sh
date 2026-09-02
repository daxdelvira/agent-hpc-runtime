#!/bin/bash
#SBATCH -J sm_preflight
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o sm_preflight_%j.log
#
# Jobs 12709114 and 12714024 both died in 4-5 s with NO diagnostic reaching the
# SLURM log -- not even echo lines placed before the failing command.  Two
# 4-GPU holds were spent learning nothing, and 12714024 had landed on
# atl1-1-03-020-6-0, the exact baseline node the comparison needs.
#
# So this job asks for 1 GPU / 2 CPUs / 16 G / 20 min -- small enough to
# schedule almost immediately -- and its ONLY purpose is to find out where the
# real job dies.  Two changes make that possible:
#
#   1. `exec > file 2>&1` at the top, writing to the PROJECT filesystem rather
#      than relying on SLURM's log.  Whatever else is lost, this file survives.
#   2. A breadcrumb after every step, each one flushed by a separate process.
#
# It boots nothing, so it cannot take 6 h and cannot OOM.

BC=results/sm_preflight_${SLURM_JOB_ID}.trace
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime || exit 9
mkdir -p results
exec > "$BC" 2>&1

crumb() { echo "[$(date +%H:%M:%S)] $*"; sync; }

crumb "01 shell alive, pid=$$"
crumb "02 node=$(hostname)"
crumb "03 pwd=$(pwd)"
crumb "04 whoami=$(whoami)  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
crumb "05 nvidia-smi path=$(command -v nvidia-smi || echo MISSING)"
crumb "06 gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | tr '\n' ';')"
crumb "07 python3 path=$(command -v python3 || echo MISSING)"
crumb "08 conda path=$(command -v conda || echo MISSING)"

PY_BIN=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/bin/python
crumb "09 PY_BIN=$PY_BIN exists=$([ -x "$PY_BIN" ] && echo yes || echo NO)"
crumb "10 PY_BIN version=$("$PY_BIN" -V 2>&1)"

crumb "11 importing model_configs + orchestrator"
"$PY_BIN" -c "
import sys
sys.path.insert(0,'experiments'); sys.path.insert(0,'.')
sys.path.insert(0,'workloads/AtomAgents')
from model_configs import MODELS_BLACKWELL_SWAP
from atomagents.runtime.model_orchestrator import ModelOrchestrator
print('   imports OK, models:', sorted(MODELS_BLACKWELL_SWAP))
" 2>&1
crumb "12 import step rc=$?"

crumb "13 running the bench in --preflight mode"
"$PY_BIN" -u experiments/bench_sleepmode_boot_cost.py --preflight 2>&1
crumb "14 bench preflight rc=$?"

crumb "99 DONE -- if you are reading this line the whole path is sound"
