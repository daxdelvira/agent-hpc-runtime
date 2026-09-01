#!/bin/bash
#SBATCH -J tandem_e2e
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o tandem_e2e_%j.log
#SBATCH -e tandem_e2e_%j.log
#
# FIRST END-TO-END TRIALS OF TANDEM.
#
# WHAT IS BEING TESTED. On the aligned campaign, 16 of 16 model prefetches
# failed. 10 died within ~1 ms with "Cannot start qwen_32b: GPUs [0,1,2,3]
# occupied by qwen_72b. Call stop_model first."; the proactive-swap ones did
# NOT fail fast -- they sat in the executor's 600 s wait-for-GPUs loop and
# failed after 600.02, 600.02, 600.03 and 918.04 s. The predictor was not the
# problem: a correct prediction had nowhere to put its result. --residency
# wires VllmModelActor so a prefetch may evict the GPU incumbent, and that in
# turn licenses the proactive-swap confidence-gate bypass.
#
# BOTH ARMS RUN IN THIS ONE ALLOCATION, ON THIS ONE NODE, DELIBERATELY.
# Node-to-node variation is up to 4.0x on an identical cold boot and 2.3x on an
# identical parse, and the existing 8 baseline trials are 7 L40S + 1 Blackwell.
# Pairing within a hold is the cheapest variance reduction available and it
# costs nothing here.
#
# --mem=256G IS THE PRODUCTION ALLOCATION AND IS NOT A MISTAKE. A parked 72B is
# ~279 GB and does not fit; the measured 32B park is 114 GiB and does. So this
# run exercises eviction everywhere and parking only where it actually fits,
# which is the real regime. Raising the memory to make parking always possible
# would dissolve the constraint the system exists to manage.
#
# Idempotent and resumable: the driver tops up to the requested count and runs
# nothing if already satisfied, so preemption costs only the trial in flight.

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


# Baseline first: tops 8 -> 11, i.e. 3 fresh trials on THIS node.
echo "=== ARM 1/2: baseline (3 fresh trials, this node) ==="
python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs baseline --trials 11
echo "[job] baseline arm exit=$? at $(date -Is)"

echo "=== ARM 2/2: tandem (3 trials, same node) ==="
python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs tandem --trials 3
echo "[job] tandem arm exit=$? at $(date -Is)"

echo "[job] end=$(date -Is)"
