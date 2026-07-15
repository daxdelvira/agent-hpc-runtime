#!/bin/bash
# run_eval_l40s_swap_pivot.sh — L40S campaign for nodes with a dead GPU 5.
#
# Written 2026-07-15 on job 11118114 (atl1-1-01-002-2-0): GPU 5 has an
# uncorrectable ECC error (volatile=1, aggregate=5), which kills the Option D
# aggregator vLLM (tp=2, GPUs 4-5) at init_device — so EVERY chemgraph_ensemble
# trial fails on this node (see failure_note in ensemble full_system
# t02__20260715-131014 meta.json).  chemgraph_swap pins all models to GPUs 0-3
# and never touches GPU 5, so this script collects only chemgraph_swap data,
# ordered by marginal value:
#   1. megammap_stage to N=3 + megammap_stage_rand x3 (rand: zero trials ever)
#   2. swap ablation gaps: observe_only (zero trials), oracle/plan_only/
#      transition_only to N=2, no_cache_stage to N=3
#   3. swap baseline to N=4 (match full_system)
#   4. stretch: megammap pair toward N=5
# Resumable; relaunch verbatim after interruption.  Deadline-aware via driver.
set -u
cd "$(dirname "$0")/.."
PY=python3

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 4 ]; then
    echo "ABORT: only $NGPU GPUs visible; chemgraph_swap needs GPUs 0-3." >&2
    exit 1
fi

echo "=== Phase 1: megammap_stage to N=3 + first megammap_stage_rand x3 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 3

echo "=== Phase 2: swap ablation gaps ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs observe_only --trials 1
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs oracle,plan_only,transition_only --trials 2
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs no_cache_stage --trials 3

echo "=== Phase 3: swap baseline to N=4 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline --trials 4

echo "=== Phase 4 (stretch): megammap pair toward N=5 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 5

echo "=== Campaign done (or deadline reached). Status: ==="
$PY experiments/run_eval_q1_q4.py --list
