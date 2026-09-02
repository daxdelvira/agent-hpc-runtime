#!/bin/bash
#SBATCH -J sleepmode_boot
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 06:00:00
#SBATCH -o sleepmode_boot_%j.log
# NOTE: no separate -e.  Run 12709114 sent -o and -e to the SAME path;
# SLURM opens the two streams independently and they clobbered each
# other, so the python traceback never survived and a 4-second failure
# left a log that simply stopped.  One stream, with 2>&1 below.
#
# Why this job exists: tandem t03 was 1.57x SLOWER than the paired baselines
# (11573.8 s vs 7362.4 s) and the entire loss is per-boot -- 1690 s per 72B
# cold boot against 1024 s -- while the residency mechanism never fired
# (0 POST /sleep, 0 POST /wake_up in 3.2 h).  The tandem and baseline arms
# differ in TWO config axes at once (--enable-sleep-mode, and the
# expandable_segments that model_orchestrator.py:236-239 drops whenever
# sleep mode is inferred), and they ran on DIFFERENT nodes.  This job runs all
# three arms on ONE node in ONE job.
#
# 6 boots x ~1000-1700 s ~= 2.5 h, so 6 h leaves room for a slow node.
# Free/preemptible: losing it costs nothing, and the JSON is written after
# every boot so a preemption keeps whatever completed.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime || exit 9
mkdir -p results

# EVERYTHING to a file on the project filesystem, not just SLURM's log.
# Jobs 12709114 and 12714024 both died in 4-5 s having printed NOTHING past the
# nvidia-smi banner -- not even echo lines placed before the failing command --
# so two 4-GPU holds were spent learning nothing.  The 1-GPU probe 12718871
# used exactly this redirect and captured all 14 of its steps, which is how the
# path was finally shown to be sound.  Keep it.
TRACE=results/sleepmode_boot_${SLURM_JOB_ID}.trace
exec > >(tee -a "$TRACE") 2>&1

echo "[job] node=$(hostname)  jobid=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The typed --gres above should make this impossible, but an untyped one once
# placed a Blackwell-partition job on a Tesla V100, so assert it at runtime.
# DO NOT write this as `if ! nvidia-smi ... | grep -q Blackwell`.  With
# `set -o pipefail` (above), grep -q exits the instant it matches, closes the
# pipe, nvidia-smi takes SIGPIPE, and the PIPELINE reports 141 even though the
# node is Blackwell.  `if !` then inverts that into the FATAL branch -- a false
# negative that fires only when nvidia-smi is still writing, i.e. racily.
# Verified locally: forcing a long nvidia-smi output makes the guard take the
# FATAL branch every time, with `pipeline status with pipefail: 141`.
# job_tandem_only.sh:54 already avoids this by capturing into a variable; this
# now matches it.
_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[job] gpu0=${_GPU:-UNKNOWN}"
case "$_GPU" in
  *Blackwell*) ;;
  *) echo "[job] FATAL: not a Blackwell node (got '${_GPU:-empty}')"; exit 2 ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate atomagents 2>/dev/null || true

# Explicit interpreter: `conda activate` above is best-effort, and which
# python3 wins after it decides whether the imports resolve.  Name the one that
# is known to import both model_configs and the orchestrator.
PY_BIN=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/bin/python
[ -x "$PY_BIN" ] || PY_BIN=python3
echo "[job] interpreter=$PY_BIN"
echo "[job] nvidia-smi=$(command -v nvidia-smi || echo MISSING)"

# Its own file, so nothing can clobber a traceback, and tee so the job log
# still shows progress live.
BENCH_LOG="sleepmode_boot_${SLURM_JOB_ID}.bench.log"
"$PY_BIN" -u experiments/bench_sleepmode_boot_cost.py \
    --model qwen_72b --reps 2 2>&1 | tee "$BENCH_LOG"
rc=${PIPESTATUS[0]}
echo "[job] bench exit=$rc  end=$(date -Is)"
if [ "$rc" -ne 0 ]; then
  echo "[job] === last 40 lines of $BENCH_LOG ==="
  tail -40 "$BENCH_LOG"
fi
exit "$rc"
