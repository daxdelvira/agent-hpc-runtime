#!/bin/bash
#SBATCH -J tandem_only
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o tandem_only_%j.log
#SBATCH -e tandem_only_%j.log
#
# TANDEM ARM ONLY -- the resubmit after job 12624742 collected 3 baselines and
# zero tandem trials.
#
# WHAT WENT WRONG LAST TIME, so it is not repeated: that job ran baseline
# first, each trial took ~2.4 h rather than the 70 min est_run_s predicted, and
# three baselines consumed 7.5 h of an 8 h hold. The driver then correctly
# declined to start a tandem trial it could not finish. est_run_s is now the
# measured 7700 s.
#
# PAIRING RISK, STATED RATHER THAN HIDDEN. The three reference baselines all
# completed on atl1-1-03-020-6-0 (8616.2 / 8587.4 / 6006.5 s, mean 7736.7, sd
# 1498.5). --nodelist=atl1-1-03-020-6-0 was the intended pairing but SLURM
# rejects it under the embers QOS ("Requested nodes not in this partition"),
# even though the node is MIXED and does belong to gpu-rtxpro-blackwell.
#
# So this job may land on any of the three Blackwell nodes. All three are the
# same GPU model (RTX PRO 6000 Blackwell, 97887 MiB), which bounds the risk,
# but it does NOT eliminate it: A2 measured the same LAMMPS retention ratio at
# 10.71x and 11.29-11.39x on two different CPU nodes, so within-type variation
# is real if modest. The node this lands on is echoed below and MUST be
# compared against the baseline node before any speedup is quoted; if they
# differ, top up baselines on the tandem node rather than pooling.
#
# Idempotent: preemption costs only the trial in flight.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
export LAMMPS_SLOWDOWN_S=0

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
echo "[job] commit=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"
echo "[job] pairing against baselines t09/t10/t11 on this node (mean 7736.7 s)"
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


python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs tandem --trials 3
echo "[job] tandem arm exit=$? end=$(date -Is)"
