#!/bin/bash
# run_eval_l40s_pool_20260720.sh — chemgraph_screen_pool (Option D) first
# collection night.
#
# Motivation (7/20 screen verdict): oracle ≈ full_system on the shared-pool
# screen workload — prediction is not the bottleneck, the unhideable vLLM
# spin-up is.  Disjoint pools (72B on GPUs 0-3 :8001, 32B on GPUs 4-5 :8005,
# SpecialistProxy :8006) let the next engine boot while the current one
# serves.  Pre-registered criterion carries over from
# chemgraph_screen_DESIGN.md: full_system must beat baseline by >10%
# (wall AND exposed_swap_s) or the design gets revisited again.
#
# Phase order is smoke-first: one full_system trial exercises ALL new
# machinery (proxy, pre-boot, plan-conditioned keep, background eviction);
# one baseline trial exercises the on-demand path.  Only then pairs.
# NO oracle tonight (needs a completed pool full_system trace to replay).
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 6 ] && { log "ABORT: only $ngpu GPUs visible (pool mode needs 6)"; exit 1; }
nvidia-smi -L | grep -qi "l40s" || { log "ABORT: not an L40S node — facet mismatch."; exit 1; }

PY=python3

log "Phase 0a: full_system smoke x1 (proxy + pre-boot + keep + eviction)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs full_system --trials 1

log "Phase 0b: baseline smoke x1 (proxy + on-demand pool boots)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs baseline --trials 1

log "Phase 1: pairs to n=3 — the headline comparison"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs baseline,full_system --trials 3

log "Phase 2: blind_stage x2 (plan vs alternation) + naive x2 (keep-all bound)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs blind_stage,naive_prefetch --trials 2

log "Phase 3 (stretch): pairs toward n=6"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs baseline,full_system --trials 6

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
