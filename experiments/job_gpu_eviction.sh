#!/bin/bash
#SBATCH -J gpu_eviction
#SBATCH -A gts-ag117
#SBATCH -q embers
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH -t 02:30:00
#SBATCH -o gpu_eviction_%j.log
#SBATCH -e gpu_eviction_%j.log
#
# Does VllmModelActor actually take an occupied GPU? See the script docstring.
#
# ONE GPU, DELIBERATELY. The full exp3 workload needs 4 at tp=4 and the
# partition has been saturated all morning -- 23 of 24 allocated, seven 1-GPU
# jobs pending, and no start estimate on any of our three 4-GPU requests. But
# the mechanism under test is CONTENTION, and two 68.28 GB engines cannot
# coexist on one 96 GB card. A single GPU reproduces the exact condition and
# schedules today.
#
# This does not replace the end-to-end trials; it answers the question those
# trials would answer FIRST, and answers it while there is still a day to act
# on the result.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python3 experiments/preflight_tandem.py || {
    echo "[job] PREFLIGHT FAILED -- releasing the hold"; exit 1; }

python3 experiments/bench_gpu_eviction.py \
    --out "results/bench_gpu_eviction_${SLURM_JOB_ID}.json"
echo "[job] exit=$? end=$(date -Is)"
