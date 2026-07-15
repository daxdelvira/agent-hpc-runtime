#!/bin/bash
# run_eval_tonight.sh — prioritized Q1–Q4 collection for the current Blackwell
# hold (2026-07-07).  Phase 1: core system comparison (round-robin, balanced N).
# Phase 2: predictor-mode / oracle / observe configs, 2 trials each.
# The driver self-stops before the SLURM deadline and resumes on rerun, so this
# same script is safe to relaunch on the next allocation.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

python3 experiments/run_eval_q1_q4.py --workload chemgraph_swap \
  --configs baseline,full_system,no_cache_stage,naive_prefetch,no_divergence_guard \
  --trials 3

python3 experiments/run_eval_q1_q4.py --workload chemgraph_swap \
  --configs oracle,plan_only,transition_only,observe_only \
  --trials 2

# Top-up toward N=10 with whatever time remains
python3 experiments/run_eval_q1_q4.py --workload chemgraph_swap \
  --configs baseline,full_system,no_cache_stage,naive_prefetch,no_divergence_guard \
  --trials 10
