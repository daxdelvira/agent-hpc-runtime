#!/bin/bash
# run_eval_l40s_day.sh — daytime campaign on an L40S bigmem node (2026-07-08).
# Node properties this exploits: cgroup RAM unlimited (503 GB → page-cache
# staging reaches FULL residency of the 145 GB worker — the sufficient-DRAM
# regime the 48 GB Blackwell holds cannot measure) and 56 TB NVMe at /tmp
# (fits the Hermes agentic config for the MegaMmap comparison).
#
# Safe to rerun on any later allocation: every phase resumes from the results
# tree and the driver stops itself before the SLURM deadline.
set -u
cd "$(dirname "$0")/.."
PY=python3

echo "=== Phase 0: MegaMmap smoke test (1 trial; validates vLLM under the"
echo "    Hermes interceptor before any batch use) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs megammap_stage --trials 1

echo "=== Phase 1: chemgraph_ensemble core (big-window mechanism) ×3 ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble \
    --configs baseline,full_system,no_cache_stage --trials 3

echo "=== Phase 2: chemgraph_swap top-up toward N=10, all configs round-robin"
echo "    (observe_only/simulated get their first trials here) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs baseline,observe_only,simulated,full_system,no_cache_stage,naive_prefetch,no_divergence_guard,plan_only,transition_only,oracle \
    --trials 10

echo "=== Phase 3: chemgraph_ensemble remaining configs (if time remains) ==="
$PY experiments/run_eval_q1_q4.py --workload chemgraph_ensemble --trials 3

echo "=== Campaign done (or deadline reached). Status: ==="
$PY experiments/run_eval_q1_q4.py --list
