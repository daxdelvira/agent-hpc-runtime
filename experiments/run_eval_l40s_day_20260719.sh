#!/bin/bash
# run_eval_l40s_day_20260719.sh — L40S day top-up (2026-07-19 morning).
#
# Post-overnight picture: ensemble is a null/regression across the whole
# matrix (every variant −4..−16% vs warm baseline; no more ensemble trials).
# The live fronts are chemgraph_swap facets. L40S facet after last night:
# baseline 312±80 (n=3) vs full_system 282±34 (n=3), nominal +10% — thin.
# naive_prefetch beats full_system on BOTH BW fronts (swap 249 vs 277,
# exp3 3711 vs 3922); L40S has zero naive trials — cross-facet check.
# megammap_stage still ≤ megammap_stage_rand (904 vs 827, n=5) — +2 each
# decides keep-or-drop.
#
# Global-count offsets (from --list at 2026-07-19 08:15):
#   chemgraph_swap baseline 14, full_system 14 -> --trials 18 (+4/arm, L40S)
#   chemgraph_swap naive_prefetch 8            -> --trials 10 (+2, first L40S)
#   megammap_stage 5, megammap_stage_rand 5    -> --trials 7  (+2 each)
# If counts changed, fix offsets BEFORE launching.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true

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
if ! nvidia-smi -L | grep -qi "l40s"; then
  log "ABORT: not an L40S node ($(nvidia-smi -L | head -1)) — facet mismatch."
  exit 1
fi

PY=python3

log "Phase 1: chemgraph_swap L40S facet pairs (+4/arm)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,full_system --trials 18

log "Phase 2: first L40S naive_prefetch trials (+2) — naive-vs-full cross-check"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs naive_prefetch --trials 10

log "Phase 3: megammap keep-or-drop (+2 each)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 7

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
