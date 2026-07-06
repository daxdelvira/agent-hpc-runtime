#!/usr/bin/env bash
# run_chemgraph.sh — Run ChemGraph MACE experiment on an interactive allocation.
#
# Starts the 72B vLLM server (used as the LLM reasoning backend), runs the
# geometry-optimization experiment, then kills the server on exit.
# MACE runs in-process using GPUs 0,1 (free while 72B uses GPUs 2,3).
# No SLURM required — run this directly on your GPU node.
#
# Usage:
#   bash experiments/run_chemgraph.sh [MODE] [MACE_DEVICE]
#
#   MODE        : baseline | observe_only | simulated | real  (default: observe_only)
#   MACE_DEVICE : cuda | cpu                                  (default: cuda)
#
# Examples:
#   bash experiments/run_chemgraph.sh                         # observe_only, cuda
#   bash experiments/run_chemgraph.sh baseline
#   bash experiments/run_chemgraph.sh real cuda
#   bash experiments/run_chemgraph.sh real cpu               # if GPUs 0,1 are in use
#
# Model weights:
#   Set HF_HOME to your HuggingFace cache dir, or MODEL_BASE_DIR to a directory
#   containing a qwen_72b/ subdirectory.
#
#   export HF_HOME=~/scratch/hf_home   # recommended

set -euo pipefail

MODE="${1:-observe_only}"
MACE_DEVICE="${2:-cuda}"

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
echo "  ChemGraph MACE Experiment — Interactive Run"
echo "  Mode        : $MODE"
echo "  MACE device : $MACE_DEVICE"
echo "  Repo root   : $REPO_ROOT"
echo "================================================================"
echo ""

# ---- Start 72B as the LLM backend -------------------------------------------
# 72B occupies GPUs 2,3.  MACE will use GPUs 0,1 (or CPU) in-process.
echo "[run] Starting 72B server (LLM backend for ChemGraph)..."
bash "$SETUP" --72b-only

LLM_ENDPOINT="http://localhost:8001/v1"
LLM_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"

# ---- Run the experiment -------------------------------------------------------
echo ""
echo "[run] Launching ChemGraph experiment..."

# RUNTIME_ENABLED tells MaceCalc.get_calculator() to check the prefetch cache.
# It is set for all non-baseline modes so the hook is always exercised.
RUNTIME_ENABLED_FLAG=""
[[ "$MODE" != "baseline" ]] && RUNTIME_ENABLED_FLAG="RUNTIME_ENABLED=1"

env $RUNTIME_ENABLED_FLAG \
  OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
  conda run -n chemgraph \
    python -u "$REPO_ROOT/experiments/chemgraph_exp.py" \
      --runtime-mode "$MODE" \
      --mace-device  "$MACE_DEVICE" \
      --base-url     "$LLM_ENDPOINT" \
      --model-name   "$LLM_MODEL"

# ---- Post-run analysis --------------------------------------------------------
echo ""
echo "[run] Experiment complete. Analyzing latest trace..."
LATEST_TRACE=$(ls -t "$REPO_ROOT/logs/workflow_traces"/chemgraph_trace_*.jsonl 2>/dev/null | head -1 || true)

if [[ -n "$LATEST_TRACE" ]]; then
  PYTHONPATH="$REPO_ROOT" conda run -n chemgraph \
    python "$REPO_ROOT/runtime/analysis/trace_analyzer.py" "$LATEST_TRACE"
  echo ""
  echo "[run] Overlap report:"
  PYTHONPATH="$REPO_ROOT" conda run -n chemgraph \
    python "$REPO_ROOT/runtime/analysis/overlap_report.py" "$LATEST_TRACE"
else
  echo "[run] No trace file found — skipping analysis."
fi

echo ""
echo "[run] Compare all runs with:"
echo "  PYTHONPATH=$REPO_ROOT conda run -n chemgraph python $REPO_ROOT/runtime/analysis/compare_runs.py $REPO_ROOT/results/summary_*.json"
