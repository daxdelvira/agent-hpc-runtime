#!/usr/bin/env bash
# verify.sh — Smoke-test all three environments after setup_cloud.sh.
#
# Usage:
#   bash setup/verify.sh
#   bash setup/verify.sh --skip-gpu     # skip nvidia-smi / vllm checks
#   bash setup/verify.sh --env atomagents  # test one env only
#
# Exit code 0 = all checks passed.
# Exit code 1 = one or more checks failed (details printed above).
set -euo pipefail

SKIP_GPU=0
ONLY_ENV=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0; FAIL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gpu)  SKIP_GPU=1 ;;
    --env)       ONLY_ENV="$2"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

ok()   { echo "  [OK]  $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
skip() { echo "  [SKIP] $1"; }

section() { echo ""; echo "--- $1 ---"; }

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
section "GPU / CUDA"
if [[ $SKIP_GPU -eq 1 ]]; then
  skip "GPU checks (--skip-gpu)"
else
  if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_LIST=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    ok "nvidia-smi: $(echo "$GPU_LIST" | wc -l) GPU(s) found"
    echo "       $GPU_LIST"
  else
    fail "nvidia-smi not available"
  fi
fi

# ---------------------------------------------------------------------------
# Helper: run a python snippet in a conda env
# ---------------------------------------------------------------------------
run_py() {
  local env="$1"; shift
  conda run --no-capture-output -n "$env" python -c "$@" 2>&1
}

check_py() {
  local env="$1"; local desc="$2"; local code="$3"
  if conda run --no-capture-output -n "$env" python -c "$code" &>/dev/null 2>&1; then
    ok "[$env] $desc"
  else
    fail "[$env] $desc"
    conda run --no-capture-output -n "$env" python -c "$code" 2>&1 | sed 's/^/       /' || true
  fi
}

# ---------------------------------------------------------------------------
# chemgraph env
# ---------------------------------------------------------------------------
if [[ -z "$ONLY_ENV" || "$ONLY_ENV" == "chemgraph" ]]; then
  section "chemgraph env"
  check_py chemgraph "langchain importable"   "import langchain"
  check_py chemgraph "langgraph importable"   "import langgraph"
  check_py chemgraph "ase importable"         "import ase"
  check_py chemgraph "mace-torch importable"  "from mace.calculators import MACECalculator"
  check_py chemgraph "rdkit importable"       "from rdkit import Chem"
  check_py chemgraph "chemgraph package"      "from chemgraph.tools.ase_tools import run_ase"
  check_py chemgraph "runtime importable"     "import sys; sys.path.insert(0,'$REPO_ROOT'); import runtime"
fi

# ---------------------------------------------------------------------------
# atomagents env
# ---------------------------------------------------------------------------
if [[ -z "$ONLY_ENV" || "$ONLY_ENV" == "atomagents" ]]; then
  section "atomagents env"
  check_py atomagents "autogen importable"    "import autogen"
  check_py atomagents "atomman importable"    "import atomman"
  check_py atomagents "ase importable"        "import ase"
  check_py atomagents "psutil importable"     "import psutil"
  check_py atomagents "atomagents package"    "from atomagents.agents.core_execution_agents import admin_core"
  check_py atomagents "runtime importable"    "import runtime"
  check_py atomagents "runtime.predictor"     "from runtime.predictor.mock_predictor import MockPredictor; p=MockPredictor('atomagents'); assert p.predictor_id=='mock'"
  check_py atomagents "runtime.prefetch"      "from runtime.prefetch.model_prefetch import FakeModelOrchestrator"
  check_py atomagents "cluster_probes"        "from runtime.measurement.cluster_probes import ClusterProbes; ClusterProbes().check_availability()"
  check_py atomagents "/proc/self/io readable" \
    "from runtime.measurement.cluster_probes import _read_proc_io; r=_read_proc_io(); assert r is not None"

  # LAMMPS
  if conda run --no-capture-output -n atomagents python -c "import lammps" &>/dev/null 2>&1; then
    ok "[atomagents] lammps Python module importable"
  else
    fail "[atomagents] lammps not importable (run setup_cloud.sh without --skip-lammps)"
  fi
fi

# ---------------------------------------------------------------------------
# vllm env
# ---------------------------------------------------------------------------
if [[ -z "$ONLY_ENV" || "$ONLY_ENV" == "vllm" ]]; then
  section "vllm env"
  check_py vllm "vllm importable"            "import vllm; print(vllm.__version__)"
  check_py vllm "torch with CUDA"            \
    "import torch; assert torch.cuda.is_available(), 'CUDA not available in vllm env'"
  check_py vllm "huggingface_hub cli"        "from huggingface_hub import snapshot_download"

  if [[ $SKIP_GPU -eq 0 ]]; then
    VLLM_PYTHON=$(conda run -n vllm which python)
    ok "vllm python path: $VLLM_PYTHON"
    echo "       Set VLLM_PYTHON=$VLLM_PYTHON in model_config.py"
  fi
fi

# ---------------------------------------------------------------------------
# Model weights check (non-fatal)
# ---------------------------------------------------------------------------
section "Model weights"
MODELS_DIR="${MODEL_BASE_DIR:-/models}"
for model in qwen_32b qwen_72b; do
  MODEL_PATH="$MODELS_DIR/$model"
  if [[ -d "$MODEL_PATH" ]] && ls "$MODEL_PATH"/*.safetensors &>/dev/null 2>&1; then
    SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    ok "weights: $model found at $MODEL_PATH ($SIZE)"
  else
    skip "weights: $model not found at $MODEL_PATH (run: setup_cloud.sh --download-models)"
  fi
done

# ---------------------------------------------------------------------------
# Runtime unit tests
# ---------------------------------------------------------------------------
section "Runtime unit tests"
if conda run --no-capture-output -n atomagents \
    python -m pytest "$REPO_ROOT/runtime/tests/" -q --tb=short 2>&1 \
    | tee /tmp/runtime_tests.log | tail -3; then
  ok "runtime test suite passed"
else
  fail "runtime test suite: see /tmp/runtime_tests.log"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Verify summary: $PASS passed, $FAIL failed"
echo "=========================================================="

if [[ $FAIL -gt 0 ]]; then
  echo "  Fix the failures above before running cluster experiments."
  exit 1
else
  echo "  All checks passed. Ready to run."
  exit 0
fi
