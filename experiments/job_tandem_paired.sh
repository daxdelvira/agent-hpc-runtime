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
# --trials is a TOP-UP TARGET COUNTED OVER THE WHOLE CORPUS, not a number of
# trials to run, and the driver's count POOLS HARDWARE.  Run 12720426 asked for
# 3 and exited 0 in 12 seconds:
#
#     atomagents_exp3_aligned/baseline: 11/3 completed, 0 to run
#     atomagents_exp3_aligned/tandem:    4/3 completed, 0 to run
#
# Those 11 baselines are 4 Blackwell plus 11 L40S; only the Blackwell ones are
# comparable, so a target that the L40S trials already satisfy schedules no
# work at all.  The target must therefore clear the EXISTING count, not the
# number of trials wanted.
#
# 12 leaves baseline 1 short and tandem 8 short.  Under roundrobin that yields
# baseline first, then tandem -- which is exactly the same-node pair this job
# exists to produce -- and keeps going with tandem if the hold survives.
# 13, not 12.  completed_trials() (run_eval_q1_q4.py:924) counts a trial as
# completed when its meta.json says status == "completed", and that status means
# only that the driver process exited 0 -- NOT that the workflow ran.  Baseline
# t12 was cut short by the preemption of job 12721602 at 2273.9 s with ONE model
# load (real baselines are 6006-8616 s with 5-8 loads), exited cleanly, and was
# counted.  Job 12733935 therefore saw "baseline: 12/12 completed, 0 to run" and
# collected no baseline at all -- defeating the entire purpose of a paired job.
#
# Every preemption that produces a clean exit inflates this count by one, so the
# target has to be bumped again each time until the counter itself distinguishes
# a completed WORKFLOW from a completed PROCESS.  That is the real fix and it is
# not made here, because tightening completed_trials() would re-open trials
# across every workload mid-campaign.
python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs baseline,tandem \
    --trials 15 --order roundrobin
echo "[job] paired arm exit=$? end=$(date -Is)"
