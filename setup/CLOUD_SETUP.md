# Cloud GPU Setup Guide

Instructions for setting up ChemGraph, AtomAgents, and the runtime prefetch layer
on a paid A100 instance (Lambda Labs, RunPod, Vast.ai, etc.).

---

## Prerequisites

**Hardware minimum**: 2× A100-80GB (to fit qwen_72b across two GPUs).
With a single A100-80GB you can test qwen_32b only.

**Disk**: ~500 GB free for model weights + ~10 GB for environments.

**Software** (usually pre-installed on cloud GPU images):
- CUDA 12.x (`nvidia-smi` to verify)
- cmake ≥ 3.16 (`cmake --version`)
- git, wget, curl

**Conda**: install miniforge3 if not present:
```bash
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash && source ~/.bashrc
```

---

## 1. Clone the repo

```bash
git clone <your-fork-or-origin> ~/agent_hpc
cd ~/agent_hpc
```

---

## 2. Run setup

```bash
# Environments only (no model download yet — do this first to catch issues early)
bash setup/setup_cloud.sh

# Then download models separately (takes ~2-4h depending on bandwidth)
export HF_TOKEN=hf_xxxx          # needed for Qwen gated models
bash setup/setup_cloud.sh --download-models --models-dir /data/models
```

Flags:
| Flag | Effect |
|---|---|
| `--download-models` | Download Qwen-32B + 72B weights from Hugging Face |
| `--models-dir PATH` | Where to store weights (default: `/models`) |
| `--skip-lammps` | Skip LAMMPS build if already compiled |

The script prints the `VLLM_PYTHON` path at the end — save it.

---

## 3. Configure model paths

```bash
export VLLM_PYTHON="$(conda run -n vllm which python)"
export MODEL_BASE_DIR="/data/models"   # wherever you put the weights

# Apply the cloud model config (adjusts GPU assignments, shorter timeouts)
cp setup/model_config_cloud.py AtomAgents/atomagents/runtime/model_config.py
```

Or set the environment variables and keep the original `model_config.py` if you
prefer — the cloud config reads both `VLLM_PYTHON` and `MODEL_BASE_DIR` from
the environment.

---

## 4. Configure API keys

**ChemGraph** (`ChemGraph/config.toml`):
```toml
[llm]
provider = "groq"          # or "openai", "anthropic", "ollama"
model    = "llama3-8b-8192"
api_key  = "gsk_xxxx"
```

**AtomAgents** (`AtomAgents/config_list`):
```json
[
  {"model": "gpt-4o", "api_key": "sk-xxxx"}
]
```

---

## 5. Verify

```bash
bash setup/verify.sh
```

Expected output:
```
--- GPU / CUDA ---
  [OK]  nvidia-smi: 2 GPU(s) found

--- chemgraph env ---
  [OK]  langchain importable
  [OK]  mace-torch importable
  ...

--- atomagents env ---
  [OK]  autogen importable
  [OK]  runtime.predictor
  [OK]  cluster_probes
  ...

--- vllm env ---
  [OK]  vllm importable
  [OK]  torch with CUDA

--- Runtime unit tests ---
  [OK]  runtime test suite passed (132 passed)

  All checks passed. Ready to run.
```

Pass `--skip-gpu` if running on a CPU-only node for initial testing.

---

## 6. What to run (in order)

### Sanity check — no GPU needed
```bash
conda run -n atomagents python runtime/demo/demo_timing.py \
  --compute-time 5 --load-time 3
```
Should show `benefit_s > 0` and `overlap_s = 3.0s`.

### Timing with real model — single GPU, qwen_32b only
```bash
export VLLM_PYTHON="$(conda run -n vllm which python)"
export MODEL_BASE_DIR=/data/models

conda run -n atomagents python runtime/demo/cluster_atomagents_exp2.py \
  --runtime-mode observe_only \
  --predictor mock \
  --hw-profile l40s          # adjust if needed
```

This run has **zero prefetch I/O** — it just records prediction events alongside
the normal AtomAgents trace. Check that `prediction_result` events appear in the
JSONL and that MockPredictor accuracy is reasonable (≥ 50%).

### First real prefetch run — 2× A100
```bash
conda run -n atomagents python runtime/demo/cluster_atomagents_exp2.py \
  --runtime-mode simulated \
  --predictor mock \
  --swap-models              # required on 2× A100 (can't hold both models at once)
```

Then upgrade to `--runtime-mode real` once simulated looks correct.

### Baseline comparison run
```bash
conda run -n atomagents python AtomAgents/run_atomagents_experiment.py \
  --mode baseline \
  --run-id baseline-$(date +%Y%m%d)
```
Run this **before** and **after** the prefetch run to get a clean comparison.

---

## 7. Artifacts to save

After each run:
```bash
# Traces
cp logs/workflow_traces/*.jsonl results/

# Analysis
conda run -n atomagents python runtime/analysis/trace_analyzer.py \
  logs/workflow_traces/runtime_trace_*.jsonl

conda run -n atomagents python runtime/analysis/overlap_report.py \
  logs/workflow_traces/runtime_trace_*.jsonl \
  --csv results/overlap_$(date +%Y%m%d).csv
```

The `probe_delta` field in `prefetch_completed` events tells you whether each
load came from NFS/disk vs. OS page cache. Check it with:
```bash
grep prefetch_completed logs/workflow_traces/*.jsonl | \
  python -c "
import sys, json
for line in sys.stdin:
  ev = json.loads(line.split('prefetch_completed')[1].lstrip(':').strip() if 'prefetch_completed' in line else '{}')
  # simpler: just pretty-print the raw line
  print(json.dumps(json.loads(line), indent=2))
" | grep -A3 "likely_source"
```

---

## Environment summary

| Env name | Purpose | Key packages |
|---|---|---|
| `chemgraph` | ChemGraph MACE runs | mace-torch, langchain, ase, rdkit |
| `atomagents` | AtomAgents + runtime | autogen, atomman, lammps, psutil |
| `vllm` | vLLM server process | vllm, torch+CUDA |

The runtime layer lives in `runtime/` and is importable from both `chemgraph`
and `atomagents` envs via the `.pth` file written by `setup_cloud.sh`.

---

## Troubleshooting

**`liblammps.so: cannot open shared object file`**
```bash
export LD_LIBRARY_PATH=$HOME/agent_hpc/setup/_lammps_src/install/lib:$LD_LIBRARY_PATH
```
Add this to `~/.bashrc` or prefix your run command with it.

**vLLM OOM on first load**
Reduce `gpu_memory_utilization` in `model_config_cloud.py` (e.g., 0.85 → 0.80).

**`prediction_result` events missing from trace**
The adapter wasn't installed. Check that `adapter.install(admin_core)` was called
before `admin_core.initiate_chat(...)` in `cluster_atomagents_exp2.py`.

**Model download fails with 401**
```bash
conda run -n vllm huggingface-cli login
```
Then re-run `setup_cloud.sh --download-models`.
