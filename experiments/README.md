# Experiments

Cluster runner scripts for the paper experiments. Each script wires the
runtime layer into a specific workload and accepts a `--runtime-mode` flag
that controls how much the runtime does:

| Mode | What it does |
|---|---|
| `baseline` | Zero runtime overhead — normal workload run for comparison |
| `observe_only` | Emits prediction events; no prefetch I/O. Safe first run. |
| `simulated` | Logs prefetch decisions with estimated timing; no actual I/O |
| `real` | Starts real background prefetches (model loads, file staging, MACE cache) |

## Prerequisites

- `workloads/AtomAgents` submodule checked out (`git submodule update --init`)
- ChemGraph available at `../agent_hpc/ChemGraph/` (for chemgraph experiments)
- `atomagents` conda environment for AtomAgents; `chemgraph` (or equivalent) for ChemGraph
- Model weights downloaded and paths configured

## Analysis

After any run, analyze with:
```bash
# Prediction accuracy + per-step table
python runtime/analysis/trace_analyzer.py logs/workflow_traces/<trace>.jsonl

# Prefetch overlap/benefit/waste timing
python runtime/analysis/overlap_report.py logs/workflow_traces/<trace>.jsonl

# Compare multiple runs side-by-side
python runtime/analysis/compare_runs.py results/summary_*.json
```

---

## AtomAgents Exp2 — screw dislocation in W

**Key measurements:**
- Predictor accuracy (does `predict(suggest_orientation)` → `computation_task` fire reliably?)
- Model-swap overlap (how much of the 32B load can be hidden behind 72B reasoning?)
- Data-file staging benefit (EAM files: ~2s, low but measurable on cold NFS)

### Option A: Standard run (both models pre-loaded, observe/simulated modes)

```bash
# Safe first run — observe predictions, no prefetch I/O
conda run -n atoms python experiments/atomagents_exp2.py \
  --runtime-mode observe_only \
  --predictor learned \
  --run-id obs-$(date +%Y%m%d)

# Baseline for comparison
conda run -n atoms python experiments/atomagents_exp2.py \
  --runtime-mode baseline \
  --run-id baseline-$(date +%Y%m%d)

# Simulated prefetch (logs decisions, no real I/O)
conda run -n atoms python experiments/atomagents_exp2.py \
  --runtime-mode simulated \
  --predictor learned \
  --run-id sim-$(date +%Y%m%d)
```

Or via SLURM:
```bash
RUNTIME_MODE=observe_only sbatch workloads/AtomAgents/autorun_experiment.sh
RUNTIME_MODE=baseline     sbatch workloads/AtomAgents/autorun_experiment.sh
```

### Option B: Real model-swap prefetch (4-GPU node, 32B loaded speculatively)

This is the key experiment: only 72B is pre-loaded; the runtime starts 32B
in the background when it predicts `computation_task_screw_dislocation` is
coming. If the prediction fires ~20 min before the tool actually executes,
the entire 32B load (20 min) is hidden.

```bash
sbatch experiments/slurm/atomagents_swap_real.sh

# Override predictor or hw profile without editing the script
PREDICTOR=mock HW_PROFILE=blackwell \
  sbatch experiments/slurm/atomagents_swap_real.sh
```

### What to collect

```bash
# Pull back traces + summaries after the job
rsync -av pace:/path/to/repo/logs/workflow_traces/ results/traces/
rsync -av pace:/path/to/repo/results/ results/

# Compare runs locally
python runtime/analysis/compare_runs.py results/summary_*.json \
  --csv results/atomagents_metrics_*.csv
```

---

## ChemGraph — MACE geometry optimization

**Key measurement:**
- MACE model load time (30–60 s from disk/NFS, or longer cold)
- Prefetch benefit: LLM reasoning time that overlaps with MACE loading

The runtime predicts `run_ase` → `mace_model` after seeing `smiles_to_coordinate_file`.
`MacePrefetchExecutor` loads the MACE calculator in a background thread.
When `run_ase` fires, `MaceCalc.get_calculator()` checks the cache and returns the
pre-loaded instance (skipping the entire load).

**Requires:** `RUNTIME_ENABLED=1` env var (set automatically by the experiment script).

### Quick local test (any machine with MACE installed)

```bash
# Observe only — no real prefetch, but shows prediction events
python experiments/chemgraph_exp.py --runtime-mode observe_only

# Baseline (no runtime layer, measures cold MACE load time)
python experiments/chemgraph_exp.py --runtime-mode baseline

# Real prefetch with CPU MACE
python experiments/chemgraph_exp.py --runtime-mode real --mace-device cpu

# Longer task to give more overlap time
python experiments/chemgraph_exp.py --runtime-mode real --extended-task
```

### Cluster run (GPU MACE, local vLLM or API)

```bash
# With a local vLLM LLM + CUDA MACE  (1 GPU node)
RUNTIME_MODE=real MACE_DEVICE=cuda \
  sbatch experiments/slurm/chemgraph_exp.sh

# Baseline for comparison
RUNTIME_MODE=baseline \
  sbatch experiments/slurm/chemgraph_exp.sh

# With Groq API for the LLM (no local vLLM needed; just needs MACE GPU)
LLM_BASE_URL=https://api.groq.com/openai/v1 \
LLM_MODEL=llama3-70b-8192 \
RUNTIME_MODE=real \
  sbatch experiments/slurm/chemgraph_exp.sh
```

### What to collect

Traces go to `logs/workflow_traces/chemgraph_trace_*.jsonl`.
Summaries go to `results/summary_*.json`.

```bash
# Per-task overlap breakdown
python runtime/analysis/overlap_report.py \
  logs/workflow_traces/chemgraph_trace_*.jsonl \
  --csv results/overlap_chemgraph.csv

# Compare baseline vs real
python runtime/analysis/compare_runs.py results/summary_*.json
```

---

## Full Comparison Experiment Suite

For the paper, run each of these in sequence and then compare:

```bash
# 1. AtomAgents baselines
RUNTIME_MODE=baseline  sbatch workloads/AtomAgents/autorun_experiment.sh
RUNTIME_MODE=observe_only sbatch workloads/AtomAgents/autorun_experiment.sh

# 2. AtomAgents real prefetch (swap mode)
sbatch experiments/slurm/atomagents_swap_real.sh

# 3. ChemGraph baseline
RUNTIME_MODE=baseline sbatch experiments/slurm/chemgraph_exp.sh

# 4. ChemGraph real prefetch
RUNTIME_MODE=real MACE_DEVICE=cuda sbatch experiments/slurm/chemgraph_exp.sh

# 5. Compare all results
python runtime/analysis/compare_runs.py results/summary_*.json \
  --csv results/atomagents_metrics_*.csv
```

Expected comparison output:

```
Run ID      Mode          Predictor  Wall(s)  Accuracy        Diverg  PF strt  benefit_s
----------  ------------  ---------  -------  --------------  ------  -------  ---------
abc123      baseline      —          3600s    N/A             —       —        —
def456      observe_only  learned    3610s    72% (8H/3M)     2       0        —
ghi789      real          learned    2400s    72% (8H/3M)     2       2        1180.0
```
