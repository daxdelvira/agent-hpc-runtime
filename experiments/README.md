# Experiments

Cluster runner scripts for the paper experiments. Each script wires the
runtime layer into a specific workload and accepts a `--runtime-mode` flag
that controls how much the runtime does:

| Mode | What it does |
|---|---|
| `baseline` | Zero runtime overhead — normal workload run for comparison |
| `observe_only` | Emits prediction events; no prefetch I/O. Safe first run. |
| `simulated` | Logs prefetch decisions with estimated timing; no actual I/O |
| `real` | Starts real background prefetches (model loads, file staging) |

## Prerequisites

- `workloads/AtomAgents` and `workloads/ChemGraph` submodules checked out
- `atomagents` conda environment created (`setup/setup_cloud.sh`)
- Model weights downloaded and `MODEL_BASE_DIR` set

## AtomAgents Exp2 — screw dislocation in W

```bash
# Recommended first run: observe only, no side effects
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode observe_only \
  --predictor mock \
  --run-id obs-$(date +%Y%m%d)

# Baseline for comparison
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode baseline \
  --run-id baseline-$(date +%Y%m%d)

# Real prefetch (2× A100, swap-models required)
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode real \
  --predictor mock \
  --swap-models \
  --run-id real-$(date +%Y%m%d)
```

Traces written to `logs/workflow_traces/`. Analyze with:
```bash
conda run -n atomagents python runtime/analysis/trace_analyzer.py \
  logs/workflow_traces/runtime_trace_*.jsonl

conda run -n atomagents python runtime/analysis/overlap_report.py \
  logs/workflow_traces/runtime_trace_*.jsonl \
  --csv results/overlap_$(date +%Y%m%d).csv
```
