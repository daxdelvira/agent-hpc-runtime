#!/bin/bash
# run_eval_l40s_screen_20260719.sh — chemgraph_screen L40S night campaign.
#
# First collection night for the screen workload (designed 2026-07-19; see
# chemgraph_screen_DESIGN.md).  Smoke-validated on BW job 11267675: routing,
# plan-conditioned staging, per-transition staging all work at ff41fb8+.
#
# GENERATION NOTE: trials at commits <ff41fb8 ran with broken specialist
# routing (one t01 per arm, 124da8b/e34d5c4) — completed but INVALID; the
# parser must split by git hash.  Driver counts them, so targets below are
# offsets over the invalid+valid total (as of 2026-07-19 16:45):
#   full_system 2 completed (1 invalid), baseline 2 (1 invalid)
#     -> --trials 6  (+4 valid each, pairs)
#   blind_stage 0, naive_prefetch 0 -> --trials 2
#   stretch: pairs to 8
# CONCURRENCY: BW chain runs run_eval_bw_screen_20260719.sh — disjoint
# configs (guard/predictor ablations only), safe to run simultaneously.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
nvidia-smi -L | grep -qi "l40s" || { log "ABORT: not an L40S node — facet mismatch."; exit 1; }

PY=python3

log "Phase 1: screen pairs (+4 valid each) — the headline comparison"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs baseline,full_system --trials 6

log "Phase 2: blind_stage x2 — does plan-conditioning beat blind staging?"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs blind_stage --trials 2

log "Phase 3: naive_prefetch x2 — does the full system beat naive here?"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs naive_prefetch --trials 2

log "Phase 4 (stretch): pairs toward n=6 valid"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs baseline,full_system --trials 8

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
