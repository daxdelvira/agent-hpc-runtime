#!/bin/bash
#SBATCH -J tandem_tp2
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH -t 05:00:00
#SBATCH -o tandem_tp2_%j.log
#SBATCH -e tandem_tp2_%j.log
#
# ONE TANDEM TRIAL AT tp=2 ON TWO GPUs.
#
# WHY. The production topology is tp=4 on GPUs 0-3, and three separate 4-GPU
# requests have sat all day without ever receiving a start estimate (partition
# 23/24 allocated, seven 1-GPU jobs pending ahead of them). The property the
# experiment actually depends on is CONTENTION -- all models sharing one pool so
# M=1 and every swap forces an eviction -- and that does not need four cards. At
# tp=2 all three models share GPUs [0,1], so M=1 is preserved exactly.
#
# Arithmetic (97887 MiB/card, util 0.95): 181.6 GiB usable, 72B weights
# 136.7 GiB, leaving ~45 GiB for KV at max_model_len 16384. Comfortable.
#
# THIS DATA MUST NOT POOL WITH THE tp=4 TRIALS, which is why it writes to its own
# workload key (atomagents_exp3_aligned_tp2). Cold boot, KV headroom and swap
# latency all move with tensor-parallel degree. It does not pair with the
# t09/t10/t11 baselines either; if this lands and the 4-GPU jobs do not, a tp=2
# baseline is the next thing to collect.
#
# One trial, not three: the first tandem trial is a DIAGNOSTIC. Does the actor
# wire, does eviction happen, do prefetches stop failing. Three trials answer
# that no better than one and schedule worse.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
export LAMMPS_SLOWDOWN_S=0

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
echo "[job] commit=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# HARDWARE ASSERTION -- belt and braces with the typed --gres above.
# Job 12694812 was submitted with an UNTYPED `--gres=gpu:1` and, despite naming
# -p gpu-rtxpro-blackwell, SLURM placed it in Partition=gpu-v100 on a
# Tesla V100-PCIE-32GB. A 68.28 GB model cannot load on a 32 GB card, so that
# run failed loudly -- but a job that merely ran SLOWER on unexpected hardware
# would have produced data that silently violates this project's never-pool
# rule. Never let a run reach the measurement on hardware nobody checked.
_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$_GPU" in
  *Blackwell*) echo "[job] GPU OK: $_GPU" ;;
  *) echo "[job] WRONG HARDWARE: expected Blackwell, got '${_GPU:-none}'." \
          "Releasing the hold rather than producing unpoolable data."; exit 1 ;;
esac


python3 experiments/preflight_tandem.py || {
    echo "[job] PREFLIGHT FAILED -- releasing the hold"; exit 1; }

python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned_tp2 --configs tandem --trials 1
echo "[job] tandem tp2 exit=$? end=$(date -Is)"
