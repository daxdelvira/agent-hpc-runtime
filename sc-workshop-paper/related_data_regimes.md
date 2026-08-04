# Data regimes in agentic scientific workflows: I/O-bound vs activation-bound

Survey status: **partial — 2026-08-03.** Workload rows are sourced to a repo path + code line
or a measurement artifact in this tree; format rows are sourced to upstream docs/source with
verbatim quotes. The literature leg on *other* agentic workloads (ChemCrow, Coscientist,
MDCrow, bio agents) and the SEARCH regime did not return before cutoff; that scope is listed
under "Not found / not attempted" so the gap is explicit rather than silently missing.

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
| 8 | DeepDriveMD (**local fork**, `0.0.2a1`) | contact maps | `.npy` via `np.save`, loaded `allow_pickle=True` (object array of sparse maps) | UNKNOWN | **Activation-bound** — `allow_pickle` means unpickling, not a flat buffer read | CPU, single Python thread (pickle) | INFERRED (from the `allow_pickle=True` flag + `returntype="sparse"` writer) | `deepdrivemd/apps/cvae_inference/app.py:33-34`; writer `apps/openmm_simulation/app.py:474,428` |
| 9 | DeepDriveMD (**local fork**) | RMSD arrays | `.npy`, plain `np.load` (no pickle) | UNKNOWN | **I/O-bound** — flat contiguous buffer, `.npy` is the in-memory layout + header | none | INFERRED (from `.npy` design; no `allow_pickle`) | `deepdrivemd/apps/cvae_inference/app.py:35` |
| 10 | DeepDriveMD (**local fork**) | CVAE model weights | PyTorch `.pt` | UNKNOWN | Activation-bound by format (pickle) | CPU unpickle, then `map_location` device copy | INFERRED (same format as #4, which is MEASURED) | `deepdrivemd/apps/cvae_inference/app.py:49-51`; `apps/cvae_train/app.py:33` |
| 11 | DeepDriveMD (**local fork**) | MD trajectory + topology | `.dcd` + `.pdb` via `MDAnalysis.Universe` | UNKNOWN | Mixed: PDB text (activation); DCD raw binary (I/O-bound, see #15) | CPU single-thread, GIL held | INFERRED (format) + MEASURED hardware (#15) | `deepdrivemd/apps/openmm_simulation/app.py:307,450` |
| 13 | DeepDriveMD-**S** (upstream, IPDPS'22) | contact maps over ADIOS2 SST | dense `uint8`, optionally packed | 3375-residue spike | **Activation-bound, MEASURED at scale.** Training read **1464±78 s → 9±4 s**; Inference **2239±20 s → 12±2 s** after removing compression | CPU single-thread — pack/unpack is a pure-Python nested loop (`utils.py:119-139`), ~5.7M interpreted iterations/frame at 3375 residues | **MEASURED** | Brace et al. 2022 Table IV, arXiv:2104.04797 |
| 14 | DeepDriveMD-**F** (upstream) | per-sim outputs | `.dcd` + `.h5` (contact map: `vlen_dtype(int16)` COO, `fletcher32`, **no compression filter**) | per-file **NOT FOUND**; ~50 GB total avoided by streaming (UC1) | Activation-bound on read: file holds indices, consumer rebuilds dense D×D per frame | **CPU single-thread, pure Python loop** — `for raw_indices in f[dataset_name]: ... coo_matrix(...).todense()` | STATED (format) / MEASURED (hardware from code) | `braceal/MD-tools mdtools/writers.py:38-59`; `keras_cvae/utils.py:43-51` |
| 15 | MDAnalysis RMSD (generic MD analysis) | XTC trajectory | Gromacs XTC, lossy ~1e-2 Å | **30 GB, 2,512,200 frames** (0.011 MB/frame) | **I/O-bound, MEASURED**: t_comp 0.09 ms vs t_IO 0.3 ms per frame, ratio **0.3**; ~40 MB/s single process | **CPU single-thread, GIL held** — `libmdaxdr.pyx:790` `read_xtc(...)` has **no** `nogil`; zero `openmp`/`prange` in file | **MEASURED** | Khoshlessan et al. 2020, arXiv:1907.00097 |
| 16 | MDAnalysis (format contrast) | DCD vs XTC, same system | DCD raw binary vs XTC compressed | **DCD300x 47 GB vs XTC300x 15 GB** (3.1x) | DCD on SSD is the **only** format where read ≲ compute (t_IO 0.06 ms vs t_comp 0.098 ms) | CPU single-thread (`libdcd.pyx`: zero `nogil`/`openmp`/`prange`) | **MEASURED** | Khoshlessan et al., SciPy 2017 |
| 17 | MDAnalysis | XTC → HDF5 conversion | one-off re-encode | 2,512,200 frames | Pure activation: **5,400 s** | CPU | **MEASURED** | Khoshlessan et al. 2020 |
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

The "near-zero-copy mmap" claim is now sourced — see the format table below, item
*safetensors*, including the upstream README's own **self-limiting** counter-quote
("No format is really zero-copy in ML... SafeTensors is not zero-copy for the header").

---

## Format-level evidence (upstream docs and source)

This section tests the hypothesis against format *designers'* own statements, independent of
any workload. Every row carries a verbatim quote in the notes beneath.

| Format | Regime | Activation work | Hardware doing it | Measured? |
|---|---|---|---|---|
| `safetensors` | I/O-bound | JSON header only, "<<<" data time | CPU 1-thread (header); none (data) | Yes — 76.6x CPU / 2.1x GPU vs `.pt` |
| PyTorch `.pt` | Activation-bound | pickle VM + extra CPU copy | CPU 1-thread | Yes (same benchmark) |
| GGUF | **I/O-bound** | **none** — dequant deferred into the matmul | none at load | No |
| HDF5 contiguous | I/O-bound | none (filters impossible) | none | No |
| HDF5 chunked+gzip | Activation-bound | inflate per chunk | CPU 1-thread (h5py `phil` lock) | No |
| Blosc/Blosc2 | Activation-bound, parallelised | shuffle + codec | **CPU multi-thread + SIMD** | GB/s NOT FOUND (plots only) |
| NetCDF4 | — | — | — | **NOT FOUND** |
| Zarr | Activation-bound (docs say so) | decompression | CPU multi-thread, GIL released | No |
| Arrow IPC | I/O-bound | **"does not involve any decoding"** | none | No |
| Parquet | Activation-bound | "must be decoded in large chunks" | CPU multi-thread | No |
| `.npy` / memmap | I/O-bound | none | none | No |
| `np.loadtxt` | Activation-bound | ASCII to binary parse | CPU 1-thread | Partial (numpy PR 20580) |
| Arrow CSV | Activation-bound | parse | CPU multi-thread | **~100 MB/s per core** |
| simdjson | Activation-bound | parse | CPU SIMD | **3.5–13 GB/s** |
| nvCOMP / cuDF Parquet | Activation-bound, **offloaded** | decompress + decode | **GPU SM, or Blackwell DE (fixed-function ASIC)** | **up to 600 GB/s; 35% e2e** |

### The decisive case: Arrow IPC vs Parquet

Same project, same committers, two on-disk formats, opposite regimes — and the Arrow FAQ
(`https://arrow.apache.org/faq/`) names the cause explicitly:

> "Reading Parquet files generally requires efficient yet relatively complex decoding, while
> **reading Arrow IPC files does not involve any decoding because the on-disk representation
> is the same as the in-memory representation.** ... If your disk storage or network is slow,
> Parquet may be a better choice even for short-term storage or caching."

> "[Parquet] efficiency comes at the cost of relatively expensive reading into memory, as
> Parquet data cannot be directly operated on but must be decoded in large chunks."

> "Conversely, Arrow is an in-memory format... **Arrow data is typically not compressed but
> laid out in natural format for the CPU**"

That final clause of the first quote is also a statement of the **crossover condition** — the
regimes trade off, and which one wins depends on the storage rate. That is precisely the
variable our prefetcher changes, and it is worth quoting in the paper for that reason.

### GGUF: the rebuttal to "compressed implies activation-bound"

A 4-bit quantised format with a **zero-cost load**, because the decode was designed into the
kernel rather than the loader. `llama.cpp`'s `src/llama-model-loader.cpp` mmap branch is a
pointer assignment:

```cpp
if (use_mmap) {
    const auto & mapping = mappings.at(w.idx);
    if (cur->data == nullptr) {
        cur->data = (uint8_t *)mapping->addr() + w.offs;
```

and there is no dequantise call in `load_data_for`; quantised blocks are consumed in situ by
`ggml_vec_dot_q4_0_q8_0` (`ggml/src/ggml-cpu/ggml-cpu.c`). So encoding density and load cost
are **independent** — which is a sharper version of our hypothesis than "format design decides
the regime": it is specifically *where the decode is placed*, and a designer can choose.

### safetensors, with its own caveat

`https://github.com/huggingface/safetensors` README:
> "This repository implements a new simple format for storing tensors safely (as opposed to
> pickle) and that is still fast (zero-copy)."

But the same README volunteers the limit, which we should quote rather than let a reviewer find:
> "No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes
> time)... **SafeTensors is not zero-copy for the header.** ... deserialization is <<< of the
> time required to load the actual tensor data"

`https://huggingface.co/docs/safetensors/speed` (gpt2): CPU 0.004015 s vs PyTorch 0.307460 s
(**76.6x**); GPU 0.165206 s vs 0.353889 s (**2.1x**).

**Read that pair carefully before citing it.** The 76.6x CPU figure is mmap setup versus full
deserialisation — 0.004 s is not a read of the tensor bytes at all, it is the header parse plus
mapping. The collapse to 2.1x on GPU is what happens once real byte movement (`cudaMemcpy`)
dominates. Both numbers are consistent with our hypothesis, but the honest gloss is "activation
is ~0 and I/O is what remains", not "safetensors loads 76x faster".

### The answer to "which hardware", at format level

Activation is **CPU in every software format above**, and the spread within CPU is large and
purely implementation-driven: ~100 MB/s per core (Arrow CSV, general text) to 3.5–13 GB/s
(simdjson, SIMD) — a ~35x spread for the same nominal job.

**Activation on non-CPU hardware does exist**, but nowhere in our workloads. NVIDIA nvCOMP
(`https://developer.nvidia.com/nvcomp`):
> "It leverages Blackwell's dedicated hardware Decompression Engine (DE) to achieve **up to
> 600 GB/s decompression throughput for standard formats.**"

and the RAPIDS blog reports a measured end-to-end consequence:
> "on a system with an **NVIDIA B100 GPU and NVMe storage, we saw 35% faster end-to-end
> runtimes using the hardware decompression engine instead of (standard) software-based GPU
> kernel decompression**" (Snappy-compressed Parquet, cuDF 25.04, nvCOMP 4.2.0.11)

So the real axis is **transform vs transfer**, not CPU vs GPU: there are at least three
hardware classes doing activation (CPU cores, GPU SMs, fixed-function ASIC), and even after
activation has been GPU-accelerated once it is still on the critical path by 35%.

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
counterexample was found** — but note that the workload sample is small (4 workloads, 12 rows)
and only 3 rows are MEASURED end-to-end, so on its own that is corroboration, not a test with
power.

**The format-level evidence above is what gives the claim power**, and it does two things to
the hypothesis:

1. *Confirms it, via a controlled case the ecosystem ran for us.* Arrow IPC vs Parquet is one
   project, one data model, one set of committers, two on-disk formats, opposite regimes, and
   the FAQ attributes the difference entirely to encoding. Nothing about "model vs data"
   appears anywhere in that reasoning.
2. *Sharpens it.* GGUF is a 4-bit quantised format with a zero-cost load. If the hypothesis
   were "compressed/encoded formats are activation-bound", GGUF would refute it. The surviving
   formulation is narrower and better: **the regime is set by where the decode is placed, and
   that is a designer's choice.** `llama.cpp` moved dequantisation into the matmul kernel, so
   the load pays nothing and every matmul pays a little. Storing the *same* weights as a
   pickle would have made the load activation-bound instead.

The one genuine qualification: **regime is a property of format *plus configuration*, not
format alone.** HDF5 is I/O-bound contiguous and activation-bound chunked+gzip; Arrow IPC can
be pushed into the activation regime by enabling compression. So "format design" should be
stated as "encoding decisions", which includes options the *writer* chose at save time — which
is exactly what rows 8 vs 9 show inside DeepDriveMD.

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
**Now closed** (was cut, then returned — see the format-level section):
- safetensors zero-copy statement, GGUF/llama.cpp, HDF5 contiguous-vs-chunked, Zarr,
  Arrow IPC vs Parquet, and the GPU-activation question. **GPU/ASIC activation does exist**
  (nvCOMP, Blackwell DE) but appears in **none** of our four workloads.

**Still missing at format level:**
- **NetCDF4: nothing.** No measured I/O-vs-decompression split, and no statement on whether
  its decompression is threaded. It is HDF5-backed so the HDF5 findings plausibly transfer,
  but that is not asserted here.
- **No absolute GB/s** for: Zarr, Arrow IPC vs Parquet reads, llama.cpp load, Blosc in prose
  (its benchmark page publishes plots only), GPUDirect Storage from a primary NVIDIA page,
  or the nvCOMP per-algorithm table (`doc/Benchmarks.md` **404s**; the blog's figures are
  inside an image, not text). If any of these must be hard numbers, they have to be ours.
- **No like-for-like single-thread-vs-SIMD CSV pair.** The ~100 MB/s/core (Arrow CSV) and
  3.5 GB/s (simdjson NDJSON) figures are different parsers on different inputs; combining
  them into a speedup would be the "infer a number from a related number" error.
- **No direct `np.load`-mmap vs `np.loadtxt` MB/s comparison on the same array.** numpy PR
  20580 gives elapsed times at unspecified row widths, not throughput. The widely-circulated
  "loadtxt 25.7 s vs read_csv 0.78 s" figure appears only in secondary blogs and is **not**
  cited here.
- NVIDIA DALI / nvJPEG throughput: **unverified**, no primary quote confirmed.

**Searched for and genuinely not present:**
- **Any size or load timing for a DeepDriveMD artifact** anywhere in this tree. Formats are
  fully determined from code; sizes are not.
- **`0.737 GB/s`** — not in this repository (see row 12).
- **LAMMPS `setfl` reader internals**: whether the parse is rank-0-then-broadcast, and
  whether spline construction is threaded. Not verified; `pair_eam.cpp` was not read. This
  matters because it decides whether the 98 s in row 1 is one core or many.
- **Thread count / parallelism for the *workload* rows.** I established *which processor*
  (CPU in every determined case, 6 of 12 workload rows) but established *degree of
  parallelism* in **zero** workload rows. Note the contrast: at **format** level the
  parallelism is often documented (h5py serialises behind `phil`; Zarr releases the GIL;
  Arrow CSV is multi-threaded by default; Blosc is multi-thread + SIMD). So the axis is
  documented by format *designers* and undocumented by workload *authors* — which is a
  finding about the field, and probably a sentence worth putting in the paper.
- **No workload in this survey performs activation on the GPU.** In every case where the
  hardware was determined, it was the CPU — even though the hardware to do otherwise exists
  and is shipping (nvCOMP/Blackwell DE, 600 GB/s). Stated as an observation over 4
  workloads, not as a general claim.

---

# Appendix: agentic-biology systems (external survey)

Produced by a sub-agent on 2026-08-03 whose parent was terminated by an API spend
limit before it could fold this in; recovered from the session transcript.

CAVEAT ON PROVENANCE: the safety classifier was unavailable when this sub-agent's
output was reviewed, and its claims have NOT been independently re-verified.
Byte sizes are stated to come from the GitHub Trees API (`?recursive=1`, `size`)
and HTTP `HEAD` `content-length`; the regime labels are INFERRED from loader code
by the same reasoning used elsewhere in this document. Spot-check before citing.

| # | System | Repo | Committed data | Remote data | Regime | Activation hardware | Confidence |
|---|---|---|---|---|---|---|---|
| B1 | BioDiscoveryAgent | snap-stanford | 10,041,908 B (29 files) | `achilles.csv` DepMap, README says "at least 300MB"; exact size NOT FOUND (Figshare blocks HEAD) | NETWORK-API by default; **I/O then activation** with `--gene_search` | CPU: `pd.read_csv` single-thread; BLAS `.dot` parallel | STATED sizes / INFERRED regime |
| B2 | **Biomni** | snap-stanford | 3,509,010 B | **15,050,945,988 B (15.05 GB), 76/76 files resolved by HEAD** | **I/O + activation** | CPU single-thread (pandas, `pickle.load`, `sc.read_h5ad`) | **MEASURED sizes** / INFERRED regime |
| B3 | CellAgent | cellagent659 | 6,005,874 B xlsx + 760,048 B csv | input AnnData UNKNOWN (user-supplied) | activation / SEARCH | GPU via scVI; CPU capped `n_jobs=4` | STATED / INFERRED |
| B4 | Robin | Future-House | 189.7 MB — **all run OUTPUTS, not inputs**; zero `.fcs` | via Edison platform | NETWORK-API | CPU single-thread (`choix.ilsr_pairwise`) | INFERRED (high) |
| B5 | **GenoTEX / GenoMAS** | Liu-Hy | 69.4 MB / 9.11 MB | **41.5 GB input, 82.0 GB total** (README) | **activation then I/O** | CPU single-thread gzip + parse | STATED (quoted README) |
| B6 | ProtAgents | lamm-mit | 4,883,431 B JSON vector index | ProteinForceGPT `model.safetensors` **1,817,061,136 B** (HF API) | activation | **CPU pinned explicitly in 3 places** while loading a 1.8 GB transformer | MEASURED size / INFERRED regime |
| B7 | STELLA | zaixizhang | 4.21 MB; `data/` holds only a 0-byte `.gitkeep` | optional 2.0 GB zip (Drive); uncompressed size NOT FOUND | NETWORK-API by default | unknown / CPU (no `torch.device` found) | INFERRED |
| B8 | BioMANIA | batmen-lab | 181.6 MB (demo GIFs); **no `data/` dir at all** | Google Drive bundles, sizes NOT FOUND | NETWORK-API + activation | GPU for retriever; CPU for analysis | INFERRED |
| B9 | BioAgents | bio-xyz | 2.2 MB pure TypeScript | S3 object store | NETWORK-API | CPU | INFERRED (high) |

## Cross-cutting findings

1. **Most agentic-biology systems are network-API wrappers with negligible local
   data.** Robin's committed corpus is its own *outputs*; STELLA's `data/` is a
   single 0-byte `.gitkeep`; BioMANIA has no `data/` directory; BioAgents is 2.2 MB
   of TypeScript. Only three have a real disk workload: Biomni (15.05 GB measured),
   GenoTEX/GenoMAS (41.5 GB quoted), BioDiscoveryAgent (~300 MB, optional path).

2. **Nobody uses backed/lazy AnnData.** `grep -n "backed"` returns zero hits in
   `biomni/tool/genomics.py` and in CellAgent's `scOmni/codes/`. Every
   `sc.read_h5ad` fully materialises an HDF5 file that supports partial reads.
   This is the cleanest unexploited-lazy-I/O opportunity found.

3. **The heaviest activation costs are deserialisation, not transfer.** Biomni's
   three GeneBass `.pkl` files (1.52–1.66 GB each) and its 6.25 GB
   `BindingDB_All_202409.tsv`. GenoTEX reads every `.gz` cohort **twice**
   (`preprocess.py:140` line-scan, then `:149` full parse) and gzip is
   non-seekable, so the second pass re-inflates from byte 0.

4. **Access is genuinely agent-determined in exactly two systems**:
   BioDiscoveryAgent's `gene_search` (LLM picks the gene, then re-reads and
   re-parses the whole ~300 MB matrix with no cache, `tools.py:286-289`) and
   GenoMAS's cohort selection (`environment.py:274` `os.listdir` → LLM chooses).
   These two meet the three-part workload filter; the other seven do not.

5. **Hardware skews CPU-single-thread more than expected.** ProtAgents pins CPU in
   three places while loading a 1.82 GB transformer. GPU appears only for
   embedding/foundation models.

6. **Free measurement infrastructure**: GenoTEX ships `utils/resource_monitor.py`
   with per-process CPU/RSS/GPU sampling. That repo instruments itself, so MEASURED
   numbers for a real agentic-biology workload are obtainable by running it.

## Not found in this appendix
TAIS has no code repo (superseded by GenoMAS per arXiv:2402.12391 comments).
Exact `achilles.csv` size. STELLA's uncompressed resource size. BioMANIA's Drive
bundle sizes. CellAgent input AnnData sizes. ProtAgents' Chroma weights.
