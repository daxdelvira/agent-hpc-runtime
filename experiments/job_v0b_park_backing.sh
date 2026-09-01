#!/bin/bash
#SBATCH -J v0b_park_backing
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH -t 01:00:00
#SBATCH -o v0b_park_backing_%j.log
#SBATCH -e v0b_park_backing_%j.log
#
# V0 -- WHICH MEMORY COLUMN DOES A COHERENT L1 PARK LAND IN?
#
# Proposed by A5 after job 12561711. That run parked ~82 GiB of a 72B and the
# entire delta appeared in the cgroup's `file` column while `anon` did not move
# at all. If that is what an L1 park really does, then:
#   - the actor's charging column is wrong, AND
#   - the plan's R2 definition ("bytes in a process address space, ANONYMOUS,
#     not a file range") is wrong for the model class, AND
#   - the argument that a byte-oriented tier structurally cannot express model
#     residency is weakened, because a file-backed park IS a file range.
# That last one is a paper-level correction, not a code tweak.
#
# 12561711 cannot settle it: that engine was fp8, tp=1, and INCOHERENT (it
# answered "The capital of France is" with "は.   1111"). A footprint from a
# broken engine proves nothing. M2's 1.90x park ratio cannot settle it either --
# it was measured with host-wide MemTotal-MemAvailable, an instrument that
# cannot tell the two columns apart. So there is currently NO measurement
# anywhere showing an L1 park landing in anonymous memory.
#
# This run uses fp16 qwen_32b, the model M1 has verified coherence evidence for
# (782.27 s cold boot -> 2.076 s wake, verbatim correct output). 68.28 GB of
# weights fit one 96 GB Blackwell, so tp=1 on the free capacity; the column a
# park lands in does not depend on tensor-parallel degree.
#
# Expected if M2's ratio holds: ~120.8 GiB parked. The question is WHERE.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

MODEL="$HOME/scratch/hf_home/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da"

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
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
    --gpus 0 --tp 1 --port 8245 \
    --arms fp16 \
    --out "results/bench_v0b_park_backing_${SLURM_JOB_ID}.json"
rc=$?
echo "[job] exit=$rc end=$(date -Is)"
exit $rc
