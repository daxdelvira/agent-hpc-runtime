#!/bin/bash
#SBATCH -J h1_fp8_tp1
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH -t 01:30:00
#SBATCH -o h1_fp8_tp1_%j.log
#SBATCH -e h1_fp8_tp1_%j.log
#
# H1 -- park a QUANTIZED 72B at R2 and measure what it costs.
#
# Replaces the one unmeasured assumption under the whole policy claim: every
# cell where arbitration is worth >=20% of wall assumes fp8 halves the parked
# footprint. Both arms run in THIS job on THIS node, deliberately -- cold boot
# varies up to 4.0x across nodes here, so a cross-node fp16/fp8 comparison
# would be uninterpretable.
#
# --mem=400G is headroom for the MEASUREMENT, not the regime. The production
# allocation is 256G and that constraint is the point of the evaluation; but a
# footprint probe that OOMs measures nothing, and the fp16 arm alone is
# expected near 279 GB.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

MODEL="$HOME/scratch/hf_home/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
echo "[job] mem.max=$(cat /sys/fs/cgroup/$(awk -F: '$1==0{print $3}' /proc/self/cgroup)/memory.max 2>/dev/null || echo n/a)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

python3 experiments/bench_h1_quantized_park.py \
    --model-path "$MODEL" \
    --gpus 0 --tp 1 --port 8243 \
    --quantization fp8 \
    --arms quant \
    --out "results/bench_h1_fp8_tp1_${SLURM_JOB_ID}.json"
rc=$?
echo "[job] exit=$rc end=$(date -Is)"
exit $rc
