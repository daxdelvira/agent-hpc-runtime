#!/usr/bin/env bash
# run_atomagents.sh — Run AtomAgents Exp2 on an interactive allocation.
#
# Starts vLLM servers, runs the experiment, then kills servers on exit.
# No SLURM required — run this directly on your GPU node.
#
# Usage:
#   bash experiments/run_atomagents.sh [MODE] [PREDICTOR] [LAMMPS_SLOWDOWN_S]
#
#   MODE             : baseline | observe_only | simulated | real  (default: observe_only)
#   PREDICTOR        : mock | learned                              (default: learned)
#   LAMMPS_SLOWDOWN_S: seconds of sleep per LAMMPS relax step     (default: 0)
#
# Examples:
#   bash experiments/run_atomagents.sh                               # observe_only + learned
#   bash experiments/run_atomagents.sh baseline
#   bash experiments/run_atomagents.sh real learned
#   bash experiments/run_atomagents.sh real learned 200              # slow LAMMPS for prefetch testing
#
# Model weights:
#   Set HF_HOME to your HuggingFace cache dir, or MODEL_BASE_DIR to a directory
#   containing qwen_72b/ and qwen_32b/ subdirectories.
#
#   export HF_HOME=~/scratch/hf_home   # HF cache (recommended)
#   # or:
#   export MODEL_BASE_DIR=/path/to/models

set -euo pipefail

MODE="${1:-observe_only}"
PREDICTOR="${2:-learned}"
SLOWDOWN="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP="$REPO_ROOT/setup/start_models.sh"

# ---- Validate args -----------------------------------------------------------
case "$MODE" in
  baseline|observe_only|simulated|real) ;;
  *) echo "ERROR: unknown mode '$MODE'. Use: baseline | observe_only | simulated | real" >&2; exit 1 ;;
esac

# ---- Always kill servers on exit --------------------------------------------
trap 'echo; echo "[run] Stopping vLLM servers..."; bash "$SETUP" --stop' EXIT

echo "================================================================"
echo "  AtomAgents Exp2 — Interactive Run"
echo "  Mode      : $MODE"
echo "  Predictor : $PREDICTOR"
echo "  Repo root : $REPO_ROOT"
echo "================================================================"
echo ""

# ---- Start vLLM servers -----------------------------------------------------
# real mode: start 72B only — the runtime speculatively loads 32B.
# All other modes: start both (needed for reasoning and computation turns).
if [[ "$MODE" == "real" ]]; then
  echo "[run] Real-swap mode: starting 72B only (32B will be loaded speculatively)"
  bash "$SETUP" --72b-only
else
  echo "[run] Starting both models..."
  bash "$SETUP"
fi

# ---- Build experiment flags --------------------------------------------------
EXTRA_FLAGS=""
if [[ "$MODE" == "real" ]]; then
  # --swap-models creates the ModelOrchestrator so ModelPrefetchExecutor can
  # start 32B in the background.  --no-start-models skips re-launching 72B
  # (it's already running from start_models.sh above).
  EXTRA_FLAGS="--swap-models --no-start-models"
fi

# ---- Run the experiment -------------------------------------------------------
echo ""
echo "[run] Launching experiment..."
SLOWDOWN_FLAG=""
[[ "$SLOWDOWN" -gt 0 ]] && SLOWDOWN_FLAG="--lammps-slowdown $SLOWDOWN"

conda run -n atomagents \
  python -u "$REPO_ROOT/experiments/atomagents_exp2.py" \
    --runtime-mode "$MODE" \
    --predictor    "$PREDICTOR" \
    --hw-profile   blackwell \
    $EXTRA_FLAGS \
    $SLOWDOWN_FLAG

# ---- Post-run analysis --------------------------------------------------------
echo ""
echo "[run] Experiment complete. Analyzing latest trace..."
LATEST_TRACE=$(ls -t "$REPO_ROOT/logs/workflow_traces"/runtime_trace_*.jsonl 2>/dev/null | head -1 || true)

if [[ -n "$LATEST_TRACE" ]]; then
  PYTHONPATH="$REPO_ROOT" conda run -n atomagents \
    python "$REPO_ROOT/runtime/analysis/trace_analyzer.py" "$LATEST_TRACE"
  echo ""
  echo "[run] Full overlap report:"
  PYTHONPATH="$REPO_ROOT" conda run -n atomagents \
    python "$REPO_ROOT/runtime/analysis/overlap_report.py" "$LATEST_TRACE"
else
  echo "[run] No trace file found in logs/workflow_traces/ — skipping analysis."
fi

echo ""
echo "[run] Compare all runs with:"
echo "  PYTHONPATH=$REPO_ROOT conda run -n atomagents python $REPO_ROOT/runtime/analysis/compare_runs.py $REPO_ROOT/results/summary_*.json"
