# Data regimes in agentic scientific workflows: I/O-bound vs activation-bound

Survey status: **partial — truncated for budget on 2026-08-03.** Everything below is
sourced to a repo path + code line or a measurement artifact in this tree. A web/paper
survey leg was launched but did not return before cutoff; its scope is listed under
"Not found / not attempted" so the gap is explicit rather than silently missing.

Definitions used here:
- **I/O-bound** — the cost is moving bytes.
- **Activation-bound** — the cost is transforming bytes into a usable in-memory form
  *after* they have arrived (parsing, decompression, deserialisation, spline/index/mesh
  construction, dtype conversion).
- **SEARCH** — a third category: the cost is compute over a database, not a load at all,
  and therefore cannot be prefetched away.

Confidence tags: **MEASURED** (a timing exists), **STATED** (format/size given, no timing),
**INFERRED** (reasoned from format design — the basis is named), **UNKNOWN**.

---

## Table

| # | Workload | Artifact | Format | Size | Regime | Activation hardware | Confidence | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | AtomAgents (ours) | EAM potential `w_eam4_big.fs` | LAMMPS `setfl` ASCII | 3,320,490,868 B (3.32 GB) | **Activation-bound, 98.1%** | CPU (LAMMPS formatted C++ stream extraction + spline build) — thread count UNKNOWN | MEASURED | `potbench_11653714.log`; `results/bench_potential_activation_atl1-1-03-004-2-1.pace.gatech.edu.json` |
| 2 | AtomAgents (ours) | same file, raw sequential read | bytes | 3.32 GB | I/O leg only: 4.78 s @ 695.3 MB/s | n/a | MEASURED | same json, rung `raw_read_cold` |
| 3 | AtomAgents (upstream, native) | EAM/MEAM/MTP potentials actually shipped | `setfl` `.eam.alloy`/`.eam.fs`, `.meam`, `.mtp` | 62 B – 9,300,295 B (9.3 MB); 14 files | Activation-bound by format, but **too small to matter** | CPU | STATED (sizes are `ls` of the tree) | `workloads/AtomAgents/potential_repository/` |
| 4 | ChemGraph (ours + upstream) | MACE-MP-0 "medium" foundation model | `torch.save` pickle (`.model`) | 44,422,970 B (44.42 MB) | **Activation-bound, 99.4%** (warm `torch.load` 23.07 s vs warm raw read 0.147 s) | CPU — `torch.load` unpickle, single Python thread | MEASURED (this survey, `envs/chemgraph` python, warm cache) | measurement below; file `~/.cache/mace/20231203mace128L1_epoch199model` |
| 5 | ChemGraph (ours) | MACE calculator construction, live agent runs | as #4 | 44.42 MB | Activation-bound | CPU (`device: "cpu"` in the trace payload) | MEASURED, n=33 | `results/eval_q1_q4/eval_q2_breakdown.csv` col `mace_load_disk_s`; median **5.46 s**, min 5.10, max 13.94 (workload `chemgraph_ensemble`) |
| 6 | ChemGraph (upstream) | MACE checkpoint → float64 conversion | torch tensors | — | Activation (small) | CPU, ATen elementwise; measured 0.035 s | MEASURED | `mace/calculators/mace.py:229` `self.models = [model.double() for model in self.models]` |
| 7 | ChemGraph (upstream) | molecular structure input | `.xyz` / ASE-readable text | UNKNOWN (small, single molecules) | Activation-bound by format (text parse), negligible at this size | CPU, `ase.io.read` Python | INFERRED (from `ase.io.read` on text) | `ChemGraph/src/chemgraph/mcp/mcp_tools.py:216,260` `from ase.io import read` / `atoms = read(input_structure_file)` |
| 8 | DeepDriveMD | contact maps | `.npy` written by `np.save`, loaded with `allow_pickle=True` (object array of sparse maps) | UNKNOWN | **Activation-bound** — `allow_pickle` means unpickling, not a flat buffer read | CPU, single Python thread (pickle) | INFERRED (from the `allow_pickle=True` flag + `returntype="sparse"` writer) | `deepdrivemd/apps/cvae_inference/app.py:33-34`; writer `apps/openmm_simulation/app.py:474,428` |
| 9 | DeepDriveMD | RMSD arrays | `.npy`, plain `np.load` (no pickle) | UNKNOWN | **I/O-bound** — flat contiguous buffer, `.npy` is the in-memory layout + header | none | INFERRED (from `.npy` design; no `allow_pickle`) | `deepdrivemd/apps/cvae_inference/app.py:35` `[np.load(p) for p in input_data.rmsd_paths]` |
| 10 | DeepDriveMD | CVAE model weights | PyTorch `.pt` checkpoint | UNKNOWN | Activation-bound by format (pickle) | CPU unpickle, then `map_location` device copy | INFERRED (same format as #4, which is MEASURED) | `deepdrivemd/apps/cvae_inference/app.py:49-51` `torch.load(input_data.model_weight_path, map_location=trainer.device)`; also `apps/cvae_train/app.py:33` |
| 11 | DeepDriveMD | MD trajectory + topology | `.dcd` trajectory + `.pdb` topology, read via `MDAnalysis.Universe` | UNKNOWN | Mixed: PDB is text (activation); DCD is raw binary frames (closer to I/O-bound) | CPU | INFERRED (from format design; DCD is uncompressed binary, PDB is fixed-column text) | `deepdrivemd/apps/openmm_simulation/app.py:307,450` `MDAnalysis.Universe(str(pdb_file), str(traj_file))` |
| 12 | vLLM model weights (ours) | Qwen2.5-72B-Instruct shards | `safetensors` | 146.82 GB, 38 shards, tp=4 | I/O-bound (design intent: mmap, near-zero-copy) | none / DMA | **STATED for size (MEASURED artifact); the 0.737 GB/s rate is NOT verified in this tree — see caveat** | `wakecache_step.log:10` `[bench] 38 shards, 146.82 GB, tp=4` |

---

## Per-workload notes

### 1–3. AtomAgents / LAMMPS `setfl` EAM — the anchor activation-bound case

Measured on `atl1-1-03-004-2-1`, three rungs from a cold page cache
(`experiments/bench_potential_activation.py`):

```
raw sequential read (cold)     4.78 s   <- byte movement (695.3 MB/s)
LAMMPS load (cold cache)     100.10 s
LAMMPS load (warm cache)      98.23 s   <- I/O already paid
I/O share  = cold - warm =      1.87 s (1.9%)
activation = warm        =     98.23 s (98.1%)
```

**Two honesty caveats that must survive into the paper:**

(a) `w_eam4_big.fs` is **synthetic**. The repo says so itself:

> `NB w_eam4_big.fs is a LOAD GENERATOR, not physics: it is produced by
> inflate_fs_blockaware.py with interpolation plus random jitter and yields
> non-physical energies (E_pair ~ 7.7e6 eV for 16 atoms). It is legitimate for
> measuring activation cost and must be described as such in any write-up.`
> — `workloads/AtomAgents/atomagents/tools/orchestration_tools.py:198-201`

The **native** AtomAgents potentials are 145 KB – 9.3 MB (row 3). So the 98 s figure
characterises the *format's* activation rate, not a cost real AtomAgents users pay today.
The correct claim is about GB/s of activation, not about AtomAgents being slow.

(b) An earlier version of this claim ("129 s load, 5.45 s read, ≥123.5 s activation")
**had no surviving artifact** and was re-measured for exactly that reason — see the
docstring of `experiments/bench_potential_activation.py:6-17`. Cite the 100.10/98.23
numbers, not the 129/5.45 ones.

LAMMPS version in the measured env: `29 Aug 2024`
(`envs/atomagents/bin/lmp -h`). **Activation hardware: CPU.** The `setfl` reader is
formatted C++ stream extraction followed by spline construction — but I did **not**
verify from LAMMPS source whether the parse is rank-0-only-then-broadcast or per-rank,
and I did not verify thread count. Mark as **CPU, parallelism UNKNOWN.**

### 4–7. ChemGraph / MACE — a *model* that is activation-bound

This is the most useful new result, because it is a counterexample to "models are
I/O-bound, data is activation-bound" and therefore *supports* the format-design hypothesis.

Measured this session with the project's own `envs/chemgraph` interpreter, warm page cache
(so the split is unambiguous — no eviction needed, the I/O leg is already ~free):

```
file                        44,422,970 B (44.42 MB)
raw read #1                 0.175 s (254.5 MB/s)
raw read #2 (warm)          0.147 s (301.8 MB/s)
torch.load (warm cache)    23.071 s
.double()                   0.035 s
warm torch.load / warm raw read = 156.7x
```

→ **99.4% of a warm MACE load is activation.** The bytes are already in RAM; the 23 s is
unpickling and object graph reconstruction.

Discrepancy to state honestly: the live agent traces record a **median 5.46 s** for the
same nominal load (`mace_load_disk_s`, n=33, `chemgraph_ensemble`). The 23.07 s above was a
first load in a fresh interpreter on a login/compute node and includes torch/e3nn import and
constant-table work; the 5.46 s figure is the steady-state in-run cost. **Both are
activation-dominated** — 44.42 MB in 5.46 s is 8.1 MB/s, still ~37x below the measured warm
read rate — but do not present 23.07 s as the in-workflow number.

Provenance chain, all verified locally:
- ChemGraph selects MACE: `ChemGraph/src/chemgraph/tools/ase_tools.py:296-299`
  (`elif "mace" in calc_type: ... calc = MaceCalc(**calculator)`)
- `"medium"` resolves to `2023-12-03-mace-128-L1_epoch-199.model`:
  `mace/calculators/foundations_models.py:34`
- which is the cached file `~/.cache/mace/20231203mace128L1_epoch199model`, 44,422,970 B
- loaded by `mace/calculators/mace.py:143` — `torch.load(f=model_path, map_location=device)`
- then `mace.py:229` — `self.models = [model.double() for model in self.models]`
  (ChemGraph defaults `default_dtype="float64"` while the checkpoint is float32, so a
  dtype conversion is unconditional)
- a representative trace event:
  `{"event_type": "mace_load", "payload": {"calculator_type": "mace_mp", "model": "medium", "device": "cpu", "duration_s": 5.542, "source": "disk"}}`
  — `results/eval_q1_q4/runs/chemgraph_ensemble/baseline/t01__20260709-105413__27b7b0f/trace.jsonl`

**Activation hardware: CPU, and predominantly single-threaded** — `torch.load` unpickling is
serial Python/C bytecode; `device: "cpu"` in every trace payload sampled. The one
parallelisable step (`.double()`) is 0.035 s, i.e. 0.15% of the cost.

### 8–11. DeepDriveMD — a *split* within one workflow

DeepDriveMD is the sharpest test of the format-design hypothesis found, because two
artifacts in the *same function* land in opposite regimes:

```python
contact_maps = np.concatenate(
    [np.load(p, allow_pickle=True) for p in input_data.contact_map_paths]
)
_rmsds = [np.load(p) for p in input_data.rmsd_paths]
```
— `deepdrivemd/apps/cvae_inference/app.py:33-35`

Contact maps need `allow_pickle=True` because the writer stores ragged sparse maps
(`distances.self_capped_distance(..., returntype="sparse")`,
`apps/openmm_simulation/app.py:474`, saved at `:428`) — so the load is an unpickle.
RMSDs are a flat float array and take the plain `.npy` path. Same library, same call,
same directory, opposite regimes — **determined entirely by how the artifact was encoded.**

Local source: `/storage/project/r-ag117-0/shared/agent_hpc/deepdrivemd/`
(editable install behind `envs/deepdrivemd`, `deepdrivemd-0.0.2a1`).

**No sizes and no load timings were found for any DeepDriveMD artifact in this tree.**
Rows 8–11 are all size-UNKNOWN. This is a gap worth closing with a local measurement —
the files exist under `deepdrivemd/runs/experiment-*/` and could be `ls`'d and timed
in well under an hour.

### 12. safetensors — the I/O-bound anchor, with a verification failure to disclose

`146.82 GB / 38 shards / tp=4` is confirmed in `wakecache_step.log:10`.

**I could not verify the `0.737 GB/s` load rate anywhere in this repository.** I grepped
every `.py`, `.json`, `.log`, `.csv`, `.md` for `0.737` / `146.82` / `GB/s`. What `0.737`
*does* match in this tree is `"probability": 0.7377` in
`runtime/predictor/data/learned_transitions.json:526`, plus a large number of coincidental
`0.73`/`0.74` values in the metrics CSVs (LLM token latencies, GPU-utilisation fractions).
Given the 2026-08-03 audit, **this figure should be re-derived from an artifact before it
enters the draft.**

The weight-staging rates that *are* written down in this tree are:
- `runtime/prefetch/model_cache_prefetch.py:15-18` — "Measured on this cluster (Qwen2.5-72B,
  37 shards, 136 GB, Lustre): cold read ~0.9 GB/s -> ~148 s to stage the whole model /
  warm read ~8.2 GB/s -> ~17 s"
- `scripts/plot_prefetch_lifecycle.py:110` — `STAGING_GBPS = 2.78  # measured staging bandwidth, GB/s`
- `experiments/chemgraph_screen_DESIGN.md:16` — "staging 3.8-4.8 GB/s from Lustre;
  72B-Instruct ≈ 145 GB (~200-245 s cold swap incl. vLLM load)"

These disagree with each other by ~9x (different media and warm/cold states), so none of
them is a drop-in substitute for 0.737. Pick one, name its rung, or re-measure.

Note also that the claim "safetensors does near-zero-copy mmap" is **currently unsourced
here** — I did not fetch the upstream design statement. It is almost certainly true but it
is exactly the kind of load-bearing assertion this project now requires a quote for.

---

## Does the format-design hypothesis survive?

**Yes, and the strongest evidence is that it cuts across the model/data boundary in both
directions.**

- A *model* that is activation-bound: MACE-MP-0, 99.4% activation warm (row 4) — because
  it is a Python pickle.
- *Data* that is I/O-bound: DeepDriveMD RMSD `.npy` (row 9) — because `.npy` is a header
  plus the in-memory buffer.
- The decisive case is rows 8 vs 9: **two artifacts, one `np.load` call site, one directory,
  opposite regimes**, and the only variable is whether the writer emitted a flat array or a
  pickled sparse object.

So the predictive variable is the encoding, not the artifact's scientific role. **No
counterexample was found** — but note that the sample is small (4 workloads, 12 rows) and
only 3 rows are MEASURED end-to-end, so this is corroboration, not a test with power.

Practical consequence for the prefetcher, which is the point of the survey: in rows 1, 4,
5 and 8 — the artifacts an agent actually *chooses* — moving bytes earlier recovers at most
1.9% (row 1) or 0.6% (row 4) of the cost. **A byte-moving prefetcher is the wrong
instrument for the majority of the agent-selected artifacts characterised here.**

---

## Not found / not attempted

This list is deliberate: the absence is itself reportable.

**Cut for budget (launched, did not return):**
- Published papers on AtomAgents / ChemGraph / DeepDriveMD read for stated formats and sizes.
  Nothing in this report comes from a paper — it is all code and local measurement.
- ChemCrow, Coscientist, MDCrow/MDAgent, LLaMP, ChatMOF, El Agente Q, BioDiscoveryAgent and
  other agentic workloads: **not characterised at all.**
- The **SEARCH** category (MSA search over UniRef90/BFD/MGnify as an agent tool) is defined
  above but has **zero rows**. This was expected to be the third regime and is entirely
  missing.
- Format-level upstream evidence: safetensors zero-copy design statement, GGUF/llama.cpp
  mmap and dequantisation, HDF5 compressed-vs-contiguous, Zarr, Arrow IPC vs Parquet
  (the cleanest same-ecosystem opposite-regime pair), nvCOMP/cuDF **GPU-side activation**.
  The GPU-activation question — is there any hardware other than a CPU doing this work? —
  **remains completely unanswered.**

**Searched for and genuinely not present:**
- **Any size or load timing for a DeepDriveMD artifact** anywhere in this tree. Formats are
  fully determined from code; sizes are not.
- **`0.737 GB/s`** — not in this repository (see row 12).
- **LAMMPS `setfl` reader internals**: whether the parse is rank-0-then-broadcast, and
  whether spline construction is threaded. Not verified; `pair_eam.cpp` was not read. This
  matters because it decides whether the 98 s in row 1 is one core or many.
- **Thread count / parallelism for every activation in the table.** I established *which
  processor* (CPU in every determined case, 6 of 12 rows) but established *degree of
  parallelism* in **zero** rows. This is the axis the brief called least-documented, and
  that assessment is confirmed: it is not documented, and it is also not recoverable from
  the loader code alone.
- **No workload in this survey performs activation on the GPU.** In every case where the
  hardware was determined, it was the CPU. Stated as an observation over 4 workloads, not
  as a general claim.
