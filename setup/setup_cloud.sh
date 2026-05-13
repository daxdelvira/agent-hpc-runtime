#!/usr/bin/env bash
# setup_cloud.sh — One-shot environment setup for a cloud GPU node (A100).
#
# Creates three conda environments:
#   chemgraph   — ChemGraph + MACE (uses ChemGraph/environment.yml)
#   atomagents  — AutoGen + atomman + runtime layer
#   vllm        — vLLM server (isolated to avoid torch version conflicts)
#
# Also builds LAMMPS with Python bindings (required by AtomAgents physics tools)
# and optionally downloads Qwen model weights from Hugging Face.
#
# Usage:
#   bash setup/setup_cloud.sh                         # full setup, skip model download
#   bash setup/setup_cloud.sh --download-models       # also download Qwen weights
#   bash setup/setup_cloud.sh --models-dir /data/models --download-models
#   bash setup/setup_cloud.sh --skip-lammps           # skip LAMMPS build (if already built)
#
# Prerequisites:
#   - conda or mamba on PATH  (tested with miniforge3/miniconda3)
#   - CUDA 12.x drivers installed (check: nvidia-smi)
#   - ~500 GB free disk for model weights (if --download-models)
#   - ~10 GB free disk for environments
#   - cmake >= 3.16  (for LAMMPS build)
#   - A Hugging Face token in HF_TOKEN env var (if --download-models)
#
# After running, copy the printed VLLM_PYTHON path into:
#   AtomAgents/atomagents/runtime/model_config.py  (or use model_config_cloud.py)
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DOWNLOAD_MODELS=0
SKIP_LAMMPS=0
MODELS_DIR="${MODEL_BASE_DIR:-/models}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-models)   DOWNLOAD_MODELS=1 ;;
    --skip-lammps)       SKIP_LAMMPS=1 ;;
    --models-dir)        MODELS_DIR="$2"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

echo ""
echo "=========================================================="
echo "  agent_hpc cloud setup"
echo "  Repo root  : $REPO_ROOT"
echo "  Models dir : $MODELS_DIR"
echo "  Download models: $DOWNLOAD_MODELS"
echo "  Skip LAMMPS: $SKIP_LAMMPS"
echo "=========================================================="
echo ""

# Detect conda/mamba
if command -v mamba &>/dev/null; then
  CONDA_CMD=mamba
elif command -v conda &>/dev/null; then
  CONDA_CMD=conda
else
  echo "ERROR: conda or mamba not found. Install miniforge3 first:" >&2
  echo "  https://github.com/conda-forge/miniforge/releases/latest" >&2
  exit 1
fi

echo "[conda] Using: $(command -v $CONDA_CMD)"

# Verify CUDA
if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found. Proceeding, but GPU steps may fail."
else
  echo "[gpu] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4)"
fi

# ---------------------------------------------------------------------------
# Step 1: ChemGraph environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Creating 'chemgraph' conda environment..."

if $CONDA_CMD env list | grep -q "^chemgraph "; then
  echo "  'chemgraph' already exists — skipping creation."
  echo "  To recreate: conda env remove -n chemgraph && rerun this script."
else
  $CONDA_CMD env create -f "$REPO_ROOT/ChemGraph/environment.yml" -n chemgraph
  echo "  'chemgraph' created."
fi

# Install ChemGraph as editable package
echo "  Installing ChemGraph (editable)..."
$CONDA_CMD run -n chemgraph pip install -e "$REPO_ROOT/ChemGraph" --quiet

# ---------------------------------------------------------------------------
# Step 2: AtomAgents + runtime environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Creating 'atomagents' conda environment..."

if $CONDA_CMD env list | grep -q "^atomagents "; then
  echo "  'atomagents' already exists — skipping creation."
else
  $CONDA_CMD env create -f "$REPO_ROOT/setup/environment_atomagents.yml"
  echo "  'atomagents' created."
fi

# Add repo root to PYTHONPATH inside atomagents env so 'import atomagents' and
# 'import runtime' both resolve without a package install.
ATOMAGENTS_SITE=$($CONDA_CMD run -n atomagents python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$ATOMAGENTS_SITE/agent_hpc.pth"
echo "$REPO_ROOT" > "$PTH_FILE"
echo "$REPO_ROOT/AtomAgents" >> "$PTH_FILE"
echo "  Wrote PYTHONPATH .pth: $PTH_FILE"

# ---------------------------------------------------------------------------
# Step 3: vLLM environment (isolated)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Creating 'vllm' conda environment..."

if $CONDA_CMD env list | grep -q "^vllm "; then
  echo "  'vllm' already exists — skipping creation."
else
  $CONDA_CMD env create -f "$REPO_ROOT/setup/environment_vllm.yml"
  echo "  'vllm' created."
fi

VLLM_PYTHON=$($CONDA_CMD run -n vllm which python)
echo "  VLLM_PYTHON = $VLLM_PYTHON"

# ---------------------------------------------------------------------------
# Step 4: LAMMPS with Python bindings
# ---------------------------------------------------------------------------
if [[ $SKIP_LAMMPS -eq 1 ]]; then
  echo ""
  echo "[4/5] Skipping LAMMPS build (--skip-lammps)."
  echo "  Make sure liblammps.so is on LD_LIBRARY_PATH and lammps is importable"
  echo "  in the 'atomagents' env before running AtomAgents physics tools."
else
  echo ""
  echo "[4/5] Building LAMMPS with Python bindings..."

  LAMMPS_SRC="$REPO_ROOT/setup/_lammps_src"
  LAMMPS_BUILD="$LAMMPS_SRC/build"

  if [[ -d "$LAMMPS_SRC/.git" ]]; then
    echo "  LAMMPS source already cloned at $LAMMPS_SRC"
  else
    git clone --depth 1 https://github.com/lammps/lammps.git "$LAMMPS_SRC"
  fi

  mkdir -p "$LAMMPS_BUILD"

  # Resolve the atomagents Python for LAMMPS Python bindings
  ATOMAGENTS_PYTHON=$($CONDA_CMD run -n atomagents which python)

  cmake -S "$LAMMPS_SRC/cmake" -B "$LAMMPS_BUILD" \
    -D BUILD_SHARED_LIBS=yes \
    -D LAMMPS_EXCEPTIONS=yes \
    -D PKG_PYTHON=yes \
    -D PKG_MANYBODY=yes \
    -D PKG_MISC=yes \
    -D PKG_REPLICA=yes \
    -D Python_EXECUTABLE="$ATOMAGENTS_PYTHON" \
    -D CMAKE_INSTALL_PREFIX="$LAMMPS_SRC/install" \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build "$LAMMPS_BUILD" -j "$(nproc)"
  cmake --install "$LAMMPS_BUILD"

  # Install lammps Python module into atomagents env
  $CONDA_CMD run -n atomagents pip install "$LAMMPS_SRC/python" --quiet

  # Add LAMMPS shared lib to atomagents env's LD path
  echo "$LAMMPS_SRC/install/lib" > "$ATOMAGENTS_SITE/lammps_lib.pth"
  echo "  LAMMPS built and installed."
  echo "  If you see 'liblammps.so not found' at runtime, run:"
  echo "    export LD_LIBRARY_PATH=$LAMMPS_SRC/install/lib:\$LD_LIBRARY_PATH"
fi

# ---------------------------------------------------------------------------
# Step 5: Model weights (optional)
# ---------------------------------------------------------------------------
if [[ $DOWNLOAD_MODELS -eq 1 ]]; then
  echo ""
  echo "[5/5] Downloading Qwen model weights to $MODELS_DIR ..."

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "  WARNING: HF_TOKEN not set. Download may fail for gated models."
    echo "  Set it with: export HF_TOKEN=hf_xxxx"
  fi

  mkdir -p "$MODELS_DIR/qwen_32b" "$MODELS_DIR/qwen_72b"

  echo "  Downloading Qwen2.5-VL-32B-Instruct (~64 GB)..."
  $CONDA_CMD run -n vllm huggingface-cli download \
    Qwen/Qwen2.5-VL-32B-Instruct \
    --local-dir "$MODELS_DIR/qwen_32b" \
    --local-dir-use-symlinks False \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

  echo "  Downloading Qwen2.5-VL-72B-Instruct (~144 GB)..."
  $CONDA_CMD run -n vllm huggingface-cli download \
    Qwen/Qwen2.5-VL-72B-Instruct \
    --local-dir "$MODELS_DIR/qwen_72b" \
    --local-dir-use-symlinks False \
    ${HF_TOKEN:+--token "$HF_TOKEN"}

  echo "  Models downloaded to $MODELS_DIR"
else
  echo ""
  echo "[5/5] Skipping model download (pass --download-models to enable)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat <<EOF

==========================================================
  Setup complete. Next steps:
==========================================================

1. Configure model paths
   Edit AtomAgents/atomagents/runtime/model_config.py — or copy the
   cloud version and set env vars:

     export VLLM_PYTHON="$VLLM_PYTHON"
     export MODEL_BASE_DIR="$MODELS_DIR"
     cp $REPO_ROOT/setup/model_config_cloud.py \\
        $REPO_ROOT/AtomAgents/atomagents/runtime/model_config.py

2. Add API keys
   ChemGraph reads from ChemGraph/config.toml.
   AtomAgents reads from AtomAgents/config_list (OpenAI format).
   Add your Groq / OpenAI / Anthropic keys there.

3. Verify the installation
     bash $REPO_ROOT/setup/verify.sh

4. First run (observe-only, no prefetch I/O):
   AtomAgents:
     conda run -n atomagents python runtime/demo/cluster_atomagents_exp2.py \\
       --runtime-mode observe_only --predictor mock

   Timing demo (no cluster needed):
     conda run -n atomagents python runtime/demo/demo_timing.py \\
       --compute-time 5 --load-time 3

5. When PACE is back:
   Replace --runtime-mode observe_only with --runtime-mode real
   and add --swap-models if using a single-GPU profile.

VLLM_PYTHON (save this):
  $VLLM_PYTHON

Models directory:
  $MODELS_DIR

EOF
