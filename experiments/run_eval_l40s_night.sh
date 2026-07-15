#!/bin/bash
# run_eval_l40s_night.sh — L40S night collection (first used 2026-07-10, job
# 10932526).  Orders phases by marginal data value across BOTH L40S workloads
# instead of running run_eval_l40s_eve.sh (whose Phase 3 stretch top-up could
# eat the night before megammap_stage_rand gets its first-ever trial):
#   1. ensemble core to N=3        (paper headline; baseline/full 1 each)
#   2. ensemble ablations x1       (oracle replays newest ensemble trace)
#   3. megammap_stage to N=3 + megammap_stage_rand x3 (rand: zero trials)
#   4. stretch: ensemble core N=5, megammap N=5
# Same preflight and concurrency rules as run_eval_l40s_eve.sh /
# run_eval_l40s_megammap.sh (see those headers).  Resumable; deadline-aware.
set -u
cd "$(dirname "$0")/.."
PY=python3

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 6 ]; then
    echo "ABORT: only $NGPU GPUs visible; chemgraph_ensemble Option D needs 6." >&2
    exit 1
fi

echo "=== Phase 1: chemgraph_ensemble core x3 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 3

echo "=== Phase 2: chemgraph_ensemble ablations x1 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs naive_prefetch,no_divergence_guard,observe_only,plan_only,transition_only,oracle \
    --trials 1

echo "=== Phase 3: megammap_stage to N=3 + first megammap_stage_rand x3 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 3

echo "=== Phase 4 (stretch): ensemble core toward N=5 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 5

echo "=== Phase 5 (stretch): megammap toward N=5 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 5

echo "=== Campaign done (or deadline reached). Status: ==="
$PY experiments/run_eval_q1_q4.py --list
