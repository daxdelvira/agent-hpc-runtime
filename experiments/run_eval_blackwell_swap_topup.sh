#!/bin/bash
# run_eval_blackwell_swap_topup.sh — chemgraph_swap Blackwell-facet top-ups
# (2026-07-17, third concurrent node: job 11237288, atl1-1-03-020-2-0).
#
# The plots facet chemgraph_swap by GPU type (never blended); the Blackwell
# facet is thin on ablations: valid oracle N=0 (t01 is oracle-bug generation,
# excluded), observe_only N=0, plan_only N=1, transition_only N=1,
# no_cache_stage N=2.  Driver counts completed trials per config dir ACROSS
# GPU types, so targets below are global-count offsets (as of 2026-07-17):
#   oracle          4 completed -> --trials 6  (+2 Blackwell, valid gen)
#   observe_only    2           -> --trials 4  (+2)
#   plan_only       3           -> --trials 5  (+2)
#   transition_only 3           -> --trials 5  (+2)
#   no_cache_stage  3           -> --trials 4  (+1 -> BW N=3)
#   baseline/full_system 4 each -> --trials 5  (+1 each, facet parity)
# If counts changed, fix offsets BEFORE launching.
#
# Safe to run concurrently with run_eval_l40s_night.sh (touches only ensemble
# + megammap configs tonight) and run_eval_blackwell_postfix.sh (AtomAgents
# only) — no shared workload/config dirs.  CONFOUND: three campaigns share
# Lustre; expect elevated load-time variance (report spreads; gate metrics
# unaffected).  Known TBLite SCF hazard ~1/3 of swap trials; driver retries.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."
  exit 1
fi

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ngpu" -lt 4 ]; then
  log "ABORT: only $ngpu GPUs visible (need 4)"
  exit 1
fi

PY=python3

log "Phase 1: oracle x2 (Blackwell valid-generation oracle: none exist yet)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs oracle --trials 6

log "Phase 2: predictor-signal ablations x2 each"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs observe_only --trials 4
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs plan_only,transition_only --trials 5

log "Phase 3: no_cache_stage +1 (Blackwell facet to N=3)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs no_cache_stage --trials 4

log "Phase 4 (stretch): baseline/full_system facet parity +1 each"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,full_system --trials 5

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
