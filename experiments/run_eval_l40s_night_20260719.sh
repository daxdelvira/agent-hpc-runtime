#!/bin/bash
# run_eval_l40s_night_20260719.sh — L40S night campaign (2026-07-19).
#
# Priorities from the 20260719 results review:
#   * chemgraph_ensemble baseline: the +17.5% full_system headline rests on ONE
#     cold (first_in_allocation) baseline outlier (t03, 2713 s vs 1155–1248 s
#     warm); warm-only comparison flips to −8.5%. More baselines are the single
#     highest-value trials on L40S tonight — they decide the headline.
#   * ensemble plan_only (+26% raw) / transition_only (+21% raw) at n=1: best
#     raw numbers in the matrix; take both to n=3.
#   * chemgraph_swap L40S facet: baseline n=1 blocks every L40S comparison.
#   * Stretch: remaining ensemble ablations n=1 -> 2, then core toward n=6.
#
# Global-count offsets (from --list at 2026-07-19 01:50):
#   chemgraph_ensemble baseline 4                  -> --trials 7  (+3)
#   chemgraph_ensemble plan_only 1, transition_only 1 -> --trials 3 (+2 each)
#   chemgraph_swap baseline 8, full_system 8       -> --trials 14
#     (assumes the BW night script's Phase 1 has finished at 12/12; the +2 is
#      the L40S facet. Runs LAST here to stay time-disjoint from BW Phase 1.)
#   ensemble naive/no_div/observe/oracle 1 each    -> --trials 2  (stretch)
#   ensemble core baseline/full/no_cache           -> --trials 6  (stretch)
# If counts changed, fix offsets BEFORE launching.
#
# Note: tonight's first trial is first_in_allocation on a 4 h-idle node — the
# parser flags it; baseline arm absorbs the cold start (same as prior nights).
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true

log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."
  exit 1
fi

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ngpu" -lt 6 ]; then
  log "ABORT: only $ngpu GPUs visible; chemgraph_ensemble Option D needs 6."
  exit 1
fi
if ! nvidia-smi -L | grep -qi "l40s"; then
  log "ABORT: not an L40S node ($(nvidia-smi -L | head -1)) — facet mismatch."
  exit 1
fi

PY=python3

log "Phase 1: chemgraph_ensemble warm baselines (+3) — decides the headline"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline --trials 7

log "Phase 2: ensemble plan_only + transition_only to n=3"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs plan_only,transition_only --trials 3

log "Phase 3: chemgraph_swap L40S facet core (+2/arm)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,full_system --trials 14

log "Phase 4 (stretch): remaining ensemble ablations to n=2"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs naive_prefetch,no_divergence_guard,observe_only,oracle --trials 2

log "Phase 5 (stretch): ensemble core toward n=6"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 6

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
