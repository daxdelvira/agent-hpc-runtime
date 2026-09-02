#!/bin/bash
#SBATCH -J tandem_paired
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o tandem_paired_%j.log
#
# BOTH ARMS ON ONE NODE.  This exists to kill the confound that every tandem
# number so far has carried.
#
# The problem: t03 (11573.8 s) and t04 (11442.6 s) both ran on
# atl1-1-03-020-2-0, and all four Blackwell baselines ran on
# atl1-1-03-020-6-0.  The arms therefore differ by 1.563x AND by node, and
# nothing collected so far can separate the two.  That is not a small worry
# here: this project has already measured 4.0x cold-boot spread across nodes
# for one identical model.
#
# --nodelist cannot fix it: SLURM rejects it under the embers QOS
# ("Requested nodes not in this partition") even for a node that does belong to
# gpu-rtxpro-blackwell.  So instead of choosing the node, make the node
# IRRELEVANT -- run both arms inside one allocation, whichever node that is.
#
# --order roundrobin alternates the arms rather than finishing one first.  That
# is the load-bearing flag, because of how these holds actually end: job
# 12700296 was preempted after one completed trial, and 12709115 after one.  At
# ~3.2 h per trial against embers preemption, a `sequential` order would spend
# the whole hold on baselines and produce another unpaired number.  Roundrobin
# means whatever survives is balanced.
#
# Idempotent: the driver tops up to --trials and re-runs nothing, so preemption
# costs only the trial in flight and the next hold resumes the count.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime || exit 9
mkdir -p results
TRACE=results/tandem_paired_${SLURM_JOB_ID}.trace
exec > >(tee -a "$TRACE") 2>&1

export LAMMPS_SLOWDOWN_S=0

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
echo "[job] commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Captured into a variable, NOT piped into `grep -q`: under set -o pipefail
# grep -q closes the pipe, nvidia-smi takes SIGPIPE, the pipeline reports 141,
# and a leading `!` turns that into a false FATAL.  Cost two 4-GPU holds.
_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$_GPU" in
  *Blackwell*) echo "[job] GPU OK: $_GPU" ;;
  *) echo "[job] FATAL: not a Blackwell node (got '${_GPU:-empty}')"; exit 2 ;;
esac

echo "[job] BOTH ARMS, roundrobin, on THIS node -- the pairing is the point."
python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs baseline,tandem \
    --trials 3 --order roundrobin
echo "[job] paired arm exit=$? end=$(date -Is)"
