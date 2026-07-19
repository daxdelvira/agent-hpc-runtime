#!/bin/bash
# run_eval_l40s_ext_20260719.sh — L40S midday extension (2026-07-19).
# Day campaign left arms unbalanced (driver retry cap + TBLite-draw streak):
# L40S facet baseline n=7 vs full_system n=4; naive_prefetch n=2 (248±7,
# fastest variant again).  Use the ~2 h left on job 11267688:
#   chemgraph_swap full_system 15 -> --trials 18 (+3, rebalance L40S facet)
#   chemgraph_swap naive_prefetch 10 -> --trials 12 (+2, L40S n=4)
#   stretch: baseline,full_system -> --trials 20 (+2 each, paired)
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

log "Phase 1: rebalance full_system L40S facet (+3)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs full_system --trials 18

log "Phase 2: naive_prefetch L40S (+2)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs naive_prefetch --trials 12

log "Phase 3 (stretch): paired core (+2 each)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,full_system --trials 20

log "Extension done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
