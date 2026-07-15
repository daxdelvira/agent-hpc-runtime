#!/bin/bash
# run_eval_blackwell_evening.sh — Q1-Q4 collection for a 4x Blackwell hold
# (first used 2026-07-09 evening, job 10847702).
#
# This node class fits chemgraph_swap and AtomAgents (blackwell/blackwell_swap
# hw profiles).  It does NOT fit chemgraph_ensemble Option D (needs 6 GPUs:
# worker tp=4 on 0-3 + aggregator tp=2 on 4-5) — that phase belongs on the
# pending l40s_bigmem jobs.
#
# Priorities: AtomAgents has ZERO eval-tree trials, so validate exp3 and exp2
# end-to-end first (one baseline/full_system pair each), then top up pairs,
# then AtomAgents ablations, then chemgraph_swap toward N=10.  The driver is
# resumable, deadline-aware, and never overwrites, so this script is safe to
# relaunch verbatim on the next Blackwell hold (10892097-100 are chained).
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

log(){ echo "[$(date +'%F %T')] === $*"; }

# Phase A: first-ever driver-managed AtomAgents trials — validation pairs.
log "Phase A: atomagents_exp3 validation pair"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 1

# exp2 PULLED from tonight's rotation (2026-07-09 20:30): after the port fix
# its agents still fail at the first tool call — engineer_core emits a
# text-format call using computation_task's (task_index, prompt) convention,
# the text-fallback hook maps args positionally onto
# computation_task_screw_dislocation(potential, ...), and the tool dies on
# potential=1 (int).  The workflow then terminates in ~90 s with ZERO real
# compute but rc=0 — the driver counts it completed, silently filling the
# trial quota with bogus runs.  Needs agent/tool-schema debugging in daylight
# (see task #16) — do not re-add until a supervised exp2 run does real LAMMPS.

# Phase B: top up the exp3 core pair (roundrobin keeps N balanced).
log "Phase B: atomagents_exp3 core top-up"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs baseline,full_system --trials 3

# Phase C: AtomAgents ablations (stretch goal on this hold).
log "Phase C: atomagents_exp3 ablations"
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3 \
  --configs naive_prefetch,no_model_prefetch,no_plan --trials 1

# Phase D (chemgraph_swap top-ups) REMOVED 2026-07-09: another user's vLLM on
# this shared node owns port 8002, which chemgraph's planner (qwen_32b_vl in
# MODELS_CHEMGRAPH_SWAP) binds.  AtomAgents was re-pointed to safe ports
# (exp3: 8007/8012/8003; exp2: 8016/8017) but the chemgraph port set is shared
# with the concurrently-running L40S megammap campaign and must not change
# mid-night.  Re-add swap top-ups after chemgraph ports are collision-hardened.

log "campaign script complete"
