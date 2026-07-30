#!/bin/bash
# run_eval_l40s_megammap_topup_20260719.sh — SECONDARY L40S campaign
# (2026-07-19).  Only launched if a second L40S node is up while the primary
# night campaign still runs; configs are disjoint from every other campaign
# tonight (megammap_* only).
#
# megammap_stage (904±247 s, n=5) currently measures WORSE than
# megammap_stage_rand (827±85 s, n=5) — structured staging losing to random
# order, within noise. +2 each decides keep-or-drop for the external-system
# comparison.
#   megammap_stage 5, megammap_stage_rand 5 -> --trials 7 (+2 each)
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

log "megammap_stage / megammap_stage_rand +2 each (keep-or-drop)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 7

log "Top-up done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
