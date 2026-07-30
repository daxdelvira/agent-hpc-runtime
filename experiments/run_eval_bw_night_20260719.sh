#!/bin/bash
# run_eval_bw_night_20260719.sh — Blackwell night campaign (2026-07-19).
#
# Priorities from the 20260719 results review (post-fix facets, campaign logs
# through 2026-07-18 03:00):
#   * chemgraph_swap BW post-fix core: +3.8% full_system vs baseline with tight
#     baseline noise (±12 s) — cheapest decisive signal; top-up both arms.
#   * atomagents_exp3 post-0517f4d: only 3 valid full_system trials and ZERO
#     same-week baselines (all 4 are 7/09–7/13, other nodes). Collect fresh
#     baseline pairs + more post-fix full_system; noise ±500 s needs n.
#   * exp3 naive_prefetch (+14.7%, n=3) — if it holds it undercuts the full
#     system's machinery; must know.
#   * swap oracle: valid post-ae945e6 anchor is thin (n=2 BW).
#   * NOT running: atomagents_exp2 (GPU-idle/fixed-duration as configured —
#     fix config first), chemgraph_ensemble_noagg (0 completed full_system).
#
# Driver counts completed trials per config dir ACROSS GPU types; targets are
# global-count offsets (from --list at 2026-07-19 01:50):
#   chemgraph_swap  baseline 8, full_system 8      -> --trials 12 (+4 each, BW)
#   atomagents_exp3 baseline 4                     -> --trials 6  (+2 fresh BW)
#   atomagents_exp3 full_system 7                  -> --trials 9  (+2 post-fix)
#   atomagents_exp3 naive_prefetch 3               -> --trials 5  (+2)
#   chemgraph_swap  oracle 6                       -> --trials 8  (+2 valid BW)
#   atomagents_exp3 no_model_prefetch 1, no_plan 1 -> --trials 3  (deep stretch)
# If counts changed, fix offsets BEFORE launching.
#
# Concurrency: run_eval_l40s_night_20260719.sh touches chemgraph_swap
# baseline/full_system in its LAST phase (targets 14 = this script's 12 + 2
# L40S facet trials). Phase 1 here runs first (~1 h) so the config dirs are
# effectively disjoint in time; a brief overlap only costs extra trials.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

# PMIx strip (b7b95a8): srun --overlap steps export PMIx vars that segfault
# LAMMPS at MPI teardown in the atoms env (rc=139 after a successful run).
# No-op for chemgraph phases and plain nohup launches.
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
if ! nvidia-smi -L | grep -qi "blackwell"; then
  log "ABORT: not a Blackwell node ($(nvidia-smi -L | head -1)) — facet mismatch."
  exit 1
fi

PY=python3

log "Phase 1: chemgraph_swap BW core top-up (+4/arm, ~1 h)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,full_system --trials 12

log "Phase 2: atomagents_exp3 fresh BW baselines (+2)"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs baseline,full_system --trials 6

log "Phase 3: atomagents_exp3 post-fix full_system (+2)"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs full_system --trials 9

log "Phase 4 (stretch): exp3 naive_prefetch (+2)"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs naive_prefetch --trials 5

log "Phase 5 (stretch): chemgraph_swap valid BW oracle (+2)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs oracle --trials 8

log "Phase 6 (deep stretch): exp3 ablation singles to n=3"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs no_model_prefetch,no_plan --trials 3

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
