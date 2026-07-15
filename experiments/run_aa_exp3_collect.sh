#!/bin/bash
# AtomAgents Exp3 (Option D) ablation data collection.
#
# Fills the data gaps in the 3-model forced-swap experiment (proactive prefetch
# during LAMMPS compute windows).  Each run_blackwell.sh --condition invocation
# is ONE run; this loops the priority conditions with a CONSISTENT
# LAMMPS_SLOWDOWN_S so the resulting walls are comparable across conditions
# (the pre-existing full_system spread of 951-6310s is suspected to come from
# inconsistent slowdown settings — hold it fixed here).
#
# Priority order fills the biggest gaps first: baseline & naive_prefetch had 0
# good runs; no_plan/no_diverg_guard/no_model_prefetch had 1 each.  full_system
# runs first as an end-to-end validation of the mechanism on this node.
#
# Runs until the SLURM allocation ends (or PASSES rounds).  Safe to relaunch.
set -uo pipefail

REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"

export HW_PROFILE=blackwell_swap
export LAMMPS_SLOWDOWN_S="${LAMMPS_SLOWDOWN_S:-300}"   # fixed 5-min windows
export EXP_TIMEOUT="${EXP_TIMEOUT:-110m}"
export PREDICTOR="${PREDICTOR:-learned}"               # match existing full_system runs

log(){ echo "[$(date +'%F %T')] $*"; }

# One validation full_system run first, then gap-fill order. Repeat PASSES times.
PASSES="${PASSES:-4}"
ORDER=(full_system baseline naive_prefetch no_plan no_diverg_guard no_model_prefetch)

log "=== AtomAgents Exp3 collection: PASSES=$PASSES LAMMPS_SLOWDOWN_S=${LAMMPS_SLOWDOWN_S}s ==="
for pass in $(seq 1 "$PASSES"); do
  for COND in "${ORDER[@]}"; do
    log ">>> pass $pass: condition $COND"
    bash experiments/run_blackwell.sh --condition "$COND" \
      || log "WARN: $COND (pass $pass) exited non-zero"
    log "<<< pass $pass: condition $COND done"
    sleep 20
  done
done
log "=== collection complete (PASSES=$PASSES) ==="
