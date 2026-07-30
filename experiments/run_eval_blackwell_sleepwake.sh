#!/bin/bash
# run_eval_blackwell_sleepwake.sh — first sleep_wake collection (2026-07-30).
#
# GENERATION: sleep_wake valid from commit 94fa2b2 (merge; gates: bench
# 2026-07-29 + supervised smoke attempt 5 on job 11518018).  The arm is
# explicit-only in the driver — it never runs without --configs sleep_wake.
# Baseline comparators: existing Blackwell-facet chemgraph_swap baselines
# (N=11+); cross-night Lustre variance applies — compare exposed_stall_s and
# swap_mechanism-faceted gates, not raw wall.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
nvidia-smi -L | grep -qiE "rtx pro 6000|blackwell" \
  || { log "ABORT: not a Blackwell node — facet mismatch."; exit 1; }
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

PY=python3

log "Phase 1: sleep_wake x1 (first driver-launched trial of the arm)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 1

log "Phase 2: sleep_wake to n=3"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 3

log "Phase 3 (stretch): sleep_wake to n=5"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 5

log "Campaign complete."
