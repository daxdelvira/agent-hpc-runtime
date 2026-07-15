#!/bin/bash
# run_eval_blackwell_20260713.sh — exp3 completion campaign for the 2026-07-13
# Blackwell hold chain (11118101 running until 21:17, 11118102-105 chained).
#
# State going in (post-rearm-fix data only):
#   atomagents_exp3 baseline 3 / full_system 3 (core DONE),
#   naive_prefetch 1, no_model_prefetch 1, no_plan 0.
# Value order: (1) the missing no_plan ablation, (2) core N=4 pair,
# (3) second ablation trials, (4) stretch core N=5.
# Driver is resumable, deadline-aware (reads SLURM_JOB_ID), never overwrites,
# so this script is safe to relaunch verbatim on the chained holds.
#
# exp2 still excluded (task #16 tool-call parser bug — bogus rc=0 runs).
# chemgraph_swap still excluded pending port hardening (task #15).
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

log(){ echo "[$(date +'%F %T')] === $*"; }

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ngpu" -lt 4 ]; then
  log "ABORT: only $ngpu GPUs visible (need 4 for exp3 tp=4)"
  exit 1
fi

log "Phase 1: exp3 no_plan ablation (last missing config)"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs no_plan --trials 1

log "Phase 2: exp3 core top-up to N=4"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 4

log "Phase 3: exp3 ablations to N=2"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs naive_prefetch,no_model_prefetch,no_plan --trials 2

log "Phase 4 (stretch): exp3 core to N=5"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 5

log "campaign script complete"
