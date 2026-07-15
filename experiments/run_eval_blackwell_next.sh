#!/bin/bash
# run_eval_blackwell_next.sh — Blackwell campaign for holds after 2026-07-14.
#
# PRECONDITION: commit the 2026-07-14 fixes first (autogen_hook type-gated
# remap, orchestration_tools potential recovery, exp2 workflow_completed
# guard, driver VLLM::Worker orphan reaping).  Run-id git hashes are how the
# parser separates parser generations; running this on a dirty 27b7b0f tree
# would blend pre/post-fix exp2+exp3 provenance.
#
# State (2026-07-14 18:00): exp3 baseline 4, full_system 3, norearm 2,
# naive 1, no_model 1, no_plan 1.  exp2 ZERO trials (fix validated in
# supervised run debug_exp2_parserfix_v2_20260714 — real LAMMPS both legs).
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

log(){ echo "[$(date +'%F %T')] === $*"; }

# --ignore-submodules=untracked: AtomAgents workload outputs (W_screw_*,
# results/, logs/) live untracked inside the submodule; only tracked
# modifications there are a provenance hazard.  Untracked files under
# experiments/ and runtime/ still count as dirty (an untracked module the
# driver imports is uncommitted behavior — e.g. runtime/prefetch/megammap_stage.py).
if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first (see header)."
  exit 1
fi

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ngpu" -lt 4 ]; then
  log "ABORT: only $ngpu GPUs visible (need 4)"
  exit 1
fi

log "Phase 1: exp2 first-ever validation pair (fix landed 2026-07-14)"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp2 \
  --configs baseline,full_system --trials 1

log "Phase 2: exp3 full_system to N=4"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 4

log "Phase 3: exp2 core to N=3"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp2 \
  --configs baseline,full_system --trials 3

log "Phase 4: exp3 ablations to N=2"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs naive_prefetch,no_model_prefetch,no_plan --trials 2

log "Phase 5 (stretch): exp3 core to N=5"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 5

log "campaign script complete"
