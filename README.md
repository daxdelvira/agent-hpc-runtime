# agent-hpc-runtime

A prediction-driven prefetch runtime for HPC agent workflows.

The runtime sits between an LLM agent and the HPC resources it needs. After
each LLM response it predicts which model weights or data files the next tool
call will require, starts loading them in a background thread, and cancels
speculatively if the agent diverges. The result is that expensive I/O (e.g. a
45-minute vLLM model load from NFS) overlaps with ongoing LLM compute instead
of blocking it.

Tested workloads: **AtomAgents** (vLLM model prefetch) and **ChemGraph** (MACE
model prefetch). Designed to extend to **DeepDriveMD**.

---

## Repository layout

```
runtime/            Core runtime layer (the primary artifact)
  adapters/         Non-invasive hooks into existing workflow callbacks
  predictor/        Predicts upcoming resource needs (mock, oracle, llm)
  prefetch/         Speculative resource acquisition + scheduler
  guard/            Divergence detection + WAL-inspired checkpointing
  measurement/      Timing records and hardware storage-hierarchy probes
  analysis/         JSONL trace analysis and overlap reporting
  tests/            Unit tests (132 passing)
  demo/             Local development demos (no GPU required)

experiments/        Cluster runner scripts (paper experiments)
setup/              Environment setup for cloud GPU nodes
workloads/          Git submodules — workloads at tested commits
  AtomAgents/
  ChemGraph/
  DeepDriveMD/      (planned)
```

---

## Quick start (local, no GPU)

```bash
git clone --recurse-submodules https://github.com/<you>/agent-hpc-runtime
cd agent-hpc-runtime

conda run -n atomagents pip install psutil requests   # minimal deps

# Full pipeline demo with simulated 5s compute + 3s model load
python runtime/demo/demo_timing.py --compute-time 5 --load-time 3

# Unit tests
python -m pytest runtime/tests/ -q
```

Expected output from the timing demo:
```
Total wall time  : 5.03s   (vs ~8s baseline)
overlap_s        : 3.0s
benefit_s        : 2.0s    (model ready 2s before it was needed)
```

---

## Cluster setup

See [`setup/CLOUD_SETUP.md`](setup/CLOUD_SETUP.md) for full instructions.
Short version:

```bash
# 1. Create environments (chemgraph, atomagents, vllm) + build LAMMPS
bash setup/setup_cloud.sh

# 2. Download model weights (~200 GB)
export HF_TOKEN=hf_xxxx
bash setup/setup_cloud.sh --download-models --models-dir /data/models

# 3. Verify
bash setup/verify.sh
```

---

## Running experiments

```bash
# Baseline (no runtime) — run first for comparison
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode baseline --run-id baseline-$(date +%Y%m%d)

# Observe only — safe first runtime run, no prefetch I/O
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode observe_only --predictor mock

# Real prefetch
conda run -n atomagents python experiments/atomagents_exp2.py \
  --runtime-mode real --predictor mock --swap-models
```

Analyze results:
```bash
python runtime/analysis/trace_analyzer.py logs/workflow_traces/*.jsonl
python runtime/analysis/overlap_report.py logs/workflow_traces/*.jsonl
```

---

## Adding workload submodules

```bash
git submodule add https://github.com/<you>/AtomAgents workloads/AtomAgents
git submodule add https://github.com/<you>/ChemGraph  workloads/ChemGraph
git submodule add https://github.com/<you>/DeepDriveMD workloads/DeepDriveMD
git commit -m "Add workload submodules"
```

Pin to tested commits before submission:
```bash
cd workloads/AtomAgents && git checkout <tested-commit> && cd ../..
git add workloads/AtomAgents && git commit -m "Pin AtomAgents to tested commit"
```

---

## Runtime modes

| Mode | Prefetch I/O | Overhead | Use for |
|---|---|---|---|
| `baseline` | None | Zero | Clean comparison baseline |
| `observe_only` | None | Minimal | First cluster run, accuracy measurement |
| `simulated` | None | Minimal | Decision logging, estimated benefit |
| `real` | Yes | Background thread | Actual overlap measurement |

---

## How it works

```
[LLM response]
      ↓  adapter intercepts (adapters/)
      ↓  predictor.predict() → ResourceSpec list (predictor/)
      ↓  detector.on_prediction() → CheckpointRecord (guard/)
      ↓  scheduler.schedule() → PrefetchTask started in background (prefetch/)

[Tool starts]
      ↓  detector.on_tool_about_to_execute()
      ├─ HIT  → prediction_validated, overlap metrics recorded
      └─ MISS → divergence_detected, pending prefetches cancelled
```

All events are written to a JSONL trace file alongside the existing workflow
events, making each run self-contained and analyzable offline.

---

## Hardware probes

The runtime detects whether a model load came from NFS, local SSD, or OS page
cache using `/proc/self/io` byte counters and `nvidia-smi` VRAM deltas.
This validates that measured speedups reflect real I/O overlap, not cache hits.

```python
from runtime.measurement.cluster_probes import ClusterProbes, LoadProbeContext

with LoadProbeContext() as ctx:
    model.load()

print(ctx.delta.likely_source)   # "nfs" | "page_cache" | "mixed"
print(ctx.delta.summary_line())  # source=nfs  elapsed=2703.1s  storage_read=138.4GB  gpu_vram_delta=147456MiB
```

---

## Citation

*Paper in preparation.*
