#!/bin/bash
# run_eval_blackwell_postfix.sh — POST-0517f4d AtomAgents series (2026-07-17,
# per dax's decision: fresh post-fix series; pre-fix full_system will be
# reported as the plan-less variant).
#
# GENERATION SPLIT: commit 0517f4d (plan-extraction window fix) is
# behavior-changing for AtomAgents full_system + prefetch ablations.
# Trials at <=d0f85d9 must NEVER pool with >=0517f4d trials; the parser
# separates them by the run-id git hash.  Baselines are unaffected (adapter
# is inert in BASELINE mode) and pool across the split.
#
# The driver counts completed trials per config dir, so post-fix targets are
# expressed as OFFSETS on top of the pre-fix counts (as of 2026-07-17):
#   exp3 full_system       4 pre-fix + 3 post-fix -> --trials 7
#   exp3 naive_prefetch    2 pre-fix + 2 post-fix -> --trials 4
#   exp3 no_model_prefetch 1 pre-fix + 2 post-fix -> --trials 3
#   exp3 no_plan           1 pre-fix + 2 post-fix -> --trials 3
#   exp2 full_system       3 pre-fix + 3 post-fix -> --trials 6
#   exp3 baseline          4 (pools)  + 1 stretch -> --trials 5
# If pre-fix counts change, fix the offsets BEFORE launching.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

# When this script is launched via `srun --overlap` into a hold job, the step
# exports PMIx vars that the atoms env's OpenMPI picks up; the MCA mismatch
# ("find-available:not-valid") segfaults LAMMPS at MPI teardown AFTER a
# successful run (rc=139), which the lattice tool reports as a list-JSON error
# payload and every screw-dislocation leg dies (exp3 t05/t06 2026-07-17).
# Stripping them is a no-op for plain batch/nohup launches.  Verified on
# 11190655: rc 139 with vars, rc 0 without (~/scratch/latsim_test_20260717).
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

PY=python3

log "Phase 1: exp3 full_system post-fix x3 (headline: plan signal now live)"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs full_system --trials 7

log "Phase 2: exp3 prefetch ablations post-fix x2 each"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs naive_prefetch --trials 4
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs no_model_prefetch,no_plan --trials 3

log "Phase 3: exp2 full_system post-fix x3"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp2 \
    --configs full_system --trials 6

log "Phase 4 (stretch): exp3 baseline parity trial"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
    --configs baseline --trials 5

log "Campaign done (or deadline reached). Status:"
$PY experiments/run_eval_q1_q4.py --list
