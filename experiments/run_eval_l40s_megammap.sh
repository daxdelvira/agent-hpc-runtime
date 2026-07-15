#!/bin/bash
# run_eval_l40s_megammap.sh — MegaMmap/Hermes staging comparison on an L40S
# bigmem node (first used 2026-07-09 evening after job 10865289 turned out to
# expose only 5 of 6 GPUs, killing the Option D ensemble plan for the night).
#
# megammap_stage / megammap_stage_rand only need GPUs 0-3 (chemgraph_swap
# topology) plus the bigmem cgroup (Hermes agentic config: 50 GB RAM tier) and
# NVMe /tmp — i.e. exactly this node class, and Blackwell holds cannot run it.
#
# Deliberately does NOT touch any config the concurrent Blackwell campaign
# (run_eval_blackwell_evening.sh, chemgraph_swap Phase D) might write: two
# drivers filling the same workload/config dir would corrupt each other's
# trial targets and mix GPU types in one aggregate.  The L40S-regime
# full_system/no_cache_stage comparison partners for these trials must be
# collected on a future L40S hold with no concurrent swap collection.
#
# Safe to rerun: resumable driver, deadline-aware, never overwrites.
set -u
cd "$(dirname "$0")/.."
PY=python3

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -lt 4 ]; then
    echo "ABORT: only $NGPU GPUs visible; chemgraph_swap topology needs 4." >&2
    exit 1
fi

echo "=== Phase 1: megammap_stage x3 (page-order = sequential, prefetch-informed) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage --trials 3

echo "=== Phase 2: megammap_stage_rand x3 (random page order — first-ever trials) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage_rand --trials 3

echo "=== Phase 3: top-up both toward N=5 (if time remains) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage,megammap_stage_rand --trials 5

echo "=== Campaign done (or deadline reached). Status: ==="
$PY experiments/run_eval_q1_q4.py --list
