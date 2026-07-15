#!/bin/bash
# run_eval_l40s_eve.sh — chemgraph_ensemble (Option D) collection on an L40S
# bigmem node (first used 2026-07-09 evening, job 10865289, launched via
# `srun --overlap` from the concurrent Blackwell session).
#
# ENSEMBLE ONLY — unlike run_eval_l40s_day.sh this deliberately runs no
# chemgraph_swap phases: (a) the swap series lives on 4x Blackwell and mixing
# GPU types in one aggregate is forbidden; (b) a Blackwell campaign
# (run_eval_blackwell_evening.sh) runs chemgraph_swap concurrently tonight and
# two drivers on one workload would race trial targets.
#
# NOTE concurrency confound: the Blackwell AtomAgents campaign reads model
# weights from the same Lustre FS tonight (~17:48-00:42). Within-workload
# config comparisons stay fair via round-robin (all configs sample the same
# contention), but flag tonight's trials if cross-night absolute numbers look
# shifted.  Trials record job/node/timestamps in meta.json.
#
# Safe to rerun on any later 6+ GPU allocation: every phase resumes from the
# results tree and the driver stops itself before the SLURM deadline.
set -u
cd "$(dirname "$0")/.."
PY=python3

# Preflight: Option D needs 6 visible GPUs (worker tp=4 on 0-3 + aggregator
# tp=2 on 4-5).  Job 10865289 (2026-07-09) exposed only 5 of its 6 allocated
# GPUs and burned a 49-min baseline trial before the aggregator ValidationError
# surfaced — fail fast instead.
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 6 ]; then
    echo "ABORT: only $NGPU GPUs visible; chemgraph_ensemble Option D needs 6." >&2
    exit 1
fi

echo "=== Phase 1: chemgraph_ensemble core (baseline/full_system/no_cache_stage) x3 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 3

echo "=== Phase 2: chemgraph_ensemble ablations x1 (oracle replays the newest"
echo "    completed ensemble baseline/full_system trace via resolve_oracle_trace) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs naive_prefetch,no_divergence_guard,observe_only,plan_only,transition_only,oracle \
    --trials 1

echo "=== Phase 3: chemgraph_ensemble core top-up toward N=5 (if time remains) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 5

echo "=== Campaign done (or deadline reached). Status: ==="
$PY experiments/run_eval_q1_q4.py --list
