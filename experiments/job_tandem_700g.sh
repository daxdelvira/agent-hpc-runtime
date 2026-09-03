#!/bin/bash
#SBATCH -J tandem_700g
#SBATCH -A gts-ag117
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH -q embers
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH -t 08:00:00
#SBATCH -o tandem_700g_%j.log
#
# THE BUDGET IS THE EXPERIMENT.  This is the one change that makes Tandem's
# mechanism expressible at all.
#
# At --mem=256G the arbitrator is correct and useless.  Measured in trial t06:
#
#   qwen_32b        129.7 GB   requested 1x   -> PARK   (never asked for again)
#   qwen_72b        279.0 GB   requested 4x   -> STOP   (exceeds the budget)
#   qwen_72b_text   276.3 GB   requested 2x   -> STOP   (exceeds the budget)
#
# The only model that fits is the only one that is never reused, so all three
# post-fix trials issued exactly one POST /sleep and ZERO POST /wake_up, and
# every swap was a full cold boot.
#
# At --mem=700G the arithmetic inverts.  700 GiB = 751.6 GB, and _can_park
# keeps 15% headroom, so 638.9 GB is spendable against 555.3 GB for both 72Bs
# plus ~18 GB for the resident engine's own host footprint (147.7 observed
# minus the 129.7 GB park it was holding).  Both reused models become parkable.
#
# WHAT THIS IS WORTH, computed from t06's own swap sequence: 5 618.0 s of its
# 9 767.1 s of swap-wait is REPEAT loading -- four loads of a model the runtime
# had already booted and killed.  Served from a park at the measured 2.21 s L1
# wake, that is 5 609.2 s saved: 10 469.3 s -> 4 860.1 s, or 0.660x the
# baseline mean.  A 1.51x speedup rather than a 1.52x slowdown.
#
# BOTH ARMS, ONE NODE, ONE BUDGET.  Running a 700G tandem trial against the
# existing 256G baselines would swap one confound for a worse one -- the
# comparison would differ in arm AND node AND budget.  --order roundrobin puts
# a baseline and a tandem trial in the same allocation at the same --mem, which
# is the only shape that licenses a ratio.
#
# --trials 18 is deliberately loose.  completed_trials() counts any trial whose
# process exited 0, so preemption casualties inflate it (currently baseline 13,
# tandem 8), and job 12783665 is concurrently topping up at 256G.  A tight
# target risks "0 to run" and a wasted hold, which has already happened once.
#
# PREDICTION, recorded before the data.  Expect PARK on qwen_72b and
# qwen_72b_text, at least one POST /wake_up, and swap notes reading
# residency_wake rather than residency_cold_boot.  If the parks are admitted
# but no wake follows, the fault is in the wake path, not the budget.  If a
# park is still refused, read the printed `used` figure -- the resident
# engine's own footprint may be larger than the 18 GB estimated above.
#
# Trials are labelled by allocation: meta.json now records slurm_mem_mb
# (SLURM_MEM_PER_NODE), so these must never be pooled with the 256G corpus.

set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime || exit 9
mkdir -p results
TRACE=results/tandem_700g_${SLURM_JOB_ID}.trace
exec > >(tee -a "$TRACE") 2>&1

export LAMMPS_SLOWDOWN_S=0

echo "[job] node=$(hostname) jobid=${SLURM_JOB_ID} start=$(date -Is)"
echo "[job] commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[job] SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} MB  <-- the experimental variable"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Captured into a variable, NOT piped into `grep -q`: under set -o pipefail
# grep -q closes the pipe, nvidia-smi takes SIGPIPE, the pipeline reports 141,
# and a leading `!` turns that into a false FATAL.  Cost two 4-GPU holds.
_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$_GPU" in
  *Blackwell*) echo "[job] GPU OK: $_GPU" ;;
  *) echo "[job] FATAL: not a Blackwell node (got '${_GPU:-empty}')"; exit 2 ;;
esac

# No conda source/activate anywhere in this script: `conda activate` is a shell
# function, and under set -u an unbound variable inside it kills the whole
# shell in a way `|| true` cannot catch.  It killed three 4-GPU holds.

python3 experiments/run_eval_q1_q4.py \
    --workload atomagents_exp3_aligned --configs baseline,tandem \
    --trials 18 --order roundrobin
echo "[job] 700G paired arm exit=$? end=$(date -Is)"
