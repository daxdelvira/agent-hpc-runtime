#!/bin/bash
# run_eval_bw_screen_20260719.sh — chemgraph_screen Blackwell ablation night.
#
# Runs configs DISJOINT from run_eval_l40s_screen_20260719.sh (pairs live on
# L40S tonight) so both chains can run concurrently.  BW facet already has the
# smoke pair (t02s at ff41fb8) as its anchor.
#   no_divergence_guard 0 -> --trials 2
#   plan_only 0, transition_only 0 -> --trials 2
#   no_plan 0 -> --trials 2
#   unpinned 0 -> --trials 2   (guard-behavior case study: TBLite failures)
#   oracle 0 -> --trials 2     (replays newest full_system screen trace)
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
nvidia-smi -L | grep -qi "blackwell" || { log "ABORT: not a Blackwell node — facet mismatch."; exit 1; }

PY=python3

log "Phase 1: no_divergence_guard x2"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs no_divergence_guard --trials 2

log "Phase 2: predictor-signal ablations x2"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs plan_only,transition_only --trials 2

log "Phase 3: no_plan x2"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs no_plan --trials 2

log "Phase 4: unpinned x2 (guard behavior on TBLite failures)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs unpinned --trials 2

log "Phase 5 (stretch): oracle x2"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen \
    --configs oracle --trials 2

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
