#!/bin/bash
#SBATCH -J h1_quant_park
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH -t 03:00:00
#SBATCH -o h1_quant_park_%j.log
#SBATCH -e h1_quant_park_%j.log
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


python3 experiments/bench_h1_quantized_park.py \
    --model-path "$MODEL" \
    --gpus 0,1,2,3 --tp 4 --port 8241 \
    --quantization fp8 \
    --arms fp16,quant \
    --out "results/bench_h1_quantized_park_${SLURM_JOB_ID}.json"
rc=$?
echo "[job] exit=$rc end=$(date -Is)"
exit $rc
