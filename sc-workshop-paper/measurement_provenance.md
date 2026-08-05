# Measurement provenance — what was measured, with what, and how far to trust it

Written 2026-08-05 for someone deciding which numbers can go in the paper. Every
row names the instrument, the artifact, and a trust class. Where a number was
withdrawn, the withdrawal is recorded rather than the row deleted.

---

## 1. How to read this

| trust | meaning | may it be a paper claim? |
|---|---|---|
| **A** | measured, artifact on disk, probe independently verified as doing real work | yes |
| **B** | measured with an artifact, but one known confound not eliminated | yes, with the caveat stated |
| **C** | measured, no artifact survives, or a single small-scale point | re-run first |
| **D** | reasoned from documentation/format knowledge, never executed | **no** — a hypothesis |
| **X** | measured and then WITHDRAWN | never; recorded so it is not rediscovered |

The distinction that matters most here is **A vs B**, and it is almost always the
same question: *did the probe actually do the thing it claims to have done?* Three
numbers this project has produced looked clean and were not. Each is dissected in
§4 because the failure mode generalises.

---

## 2. The instrumentation toolkit

Six techniques do nearly all the work. Each is listed with what it measures and,
more importantly, **how it lies**.

### 2.1 Page-cache eviction — `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`
Drops a file from the page cache with no privileges, so a "cold" rung can be
created inside an unprivileged job. Used by `bench_potential_activation.py`,
`bench_activated_residency.py`, `bench_p1_consumer_retention.py`.

**How it lies:** it is a *hint*. On Lustre it returns success and evicts nothing.
Measured directly (§4.2). Every cold-vs-warm difference is then two identically
cached rungs and their difference is noise. `evict_works()` in
`bench_p1_consumer_retention.py:74` now probes the filesystem before the run and
suppresses the affected column.

### 2.2 Residency verification — `mincore()` over a direct libc `mmap`
Returns the fraction of a file actually in the page cache, so the state that
eviction *claims* to have produced can be checked rather than assumed. Implemented
via `ctypes` because Python exposes no `mincore`.

**How it lies:** returns the client's view. Confirms what this node believes, not
what the storage system did.

### 2.3 Memory accounting — three different readings, deliberately
- `VmRSS` from `/proc/self/status` — live process footprint, used for
  activated-structure size.
- `ru_maxrss` from `getrusage` — *peak*, so it never decreases; correct for "did
  this ever fit," wrong for "what is held now."
- `MemTotal − MemAvailable` from `/proc/meminfo` — host-wide, the only way to see
  what vLLM parks outside the Python process.

**How it lies:** mixing them. The host-wide reading includes every other tenant on
a shared node; the process reading misses anything vLLM allocates in C.

### 2.4 vLLM sleep/wake — HTTP `/sleep?level=N`, `/wake_up`
Needs `VLLM_SERVER_DEV_MODE=1` and `--enable-sleep-mode`.

**How it lies:** spectacularly, and it did. See §4.1 — L2 sleep reported a fast
wake while returning `"!!!!"`. **A wake is only proven by generating text and
checking the content and `finish_reason`,** which the original `sanity_generate`
did not do because it discarded output.

### 2.5 Offline replay — `replay_*.py`
Re-runs a recorded need sequence under a cost model. No GPU, deterministic,
re-runnable at zero cost.

**How it lies:** it is only as good as the cost model, and an impossible ordering
among policies is the tell. `belady` scoring worse than LRU is impossible for an
offline optimum, and that is what exposed a sentinel bug in
`replay_retention_policy.py` (`-1.0` where `float("-inf")` was needed, so "never
used again" sorted last instead of first).

### 2.6 Synthetic sweep — `sweep_policy_regime.py`
Answers "what configuration *would* discriminate," not "what does our workload do."

**How it lies:** its t-statistics measure reproducibility across random seeds, not
confidence about reality. Its sanity check is the load-bearing part: at
cost-per-byte spread 1×, value density must reduce to Belady's ranking exactly,
and it does (0.0% ± 0.0 at every budget). **Its numbers must never be reported as
measured speedups.**

---

## 3. Measurement register

### Model axis

| # | claim | instrument | artifact | trust |
|---|---|---|---|---|
| M1 | L1 sleep wakes **coherently**; 782.27 s boot → **2.076 s** wake (377×) | `bench_wake_cache_dependence.py --sleep-level 1` | `results/bench_wake_L1_coherence_32b.json` | **A** |
| M2 | park cost **120.77 GiB for a 68.28 GB model = 1.90×** | same run, `MemTotal−MemAvailable` | same | **A** |
| M3 | wake is page-cache independent (evicting 18 shards moved it 0.015 s) | same, `mincore()` verified | same | **A** |
| M4 | cold boot for ONE identical 68.3 GB model ranges **344–1372 s (4.0×)** across 8 nodes; wake stays 1.43–2.07 s, park 113.5–115.8 GiB | `bench_activation_ladder` | `results/bench_activation_ladder_*.json` ×8 | **A** |
| M5 | warm boot vs cold boot is **sign-unstable**: −7.0% to +37.1% | same 8 files | same | **B** — page-cache state across jobs not controlled |
| M6 | k=3 simultaneous L1 sleeps does not complete | `bench_residency_preflight` | `results/bench_residency_preflight_*.json` | **B** — timeout/HTTP 500, **no OOM ever observed** |

M1–M3 carry three independent trust checks, which is why they are the strongest
numbers in the project: correct generated text, unchanged post-wake throughput
(20.05–20.08 vs 20.20 tok/s steady, so no cost deferred into generation), and a
physically plausible 16.6 GB/s per GPU across two PCIe links.

### Data axis

| # | claim | instrument | artifact | trust |
|---|---|---|---|---|
| D1 | EAM potential: **98.1% activation, 1.9% I/O** | `bench_potential_activation.py`, `mincore`-verified at every rung | `results/bench_potential_activation_*.json` | **A** |
| D2 | retention **42.83 s → 4.78 s (9.0×)**; activated **16.93 GB = 5.10× expansion** | `bench_activated_residency.py` | `results/bench_activated_residency_BIG.json` | **A** |
| D3 | LAMMPS does **not** memoise (`r4_repeat_coeff` 42.84 s ≈ full re-parse) | same | same | **A** |
| D4 | 8 background parsers cost the foreground **0.0%** | `bench_preactivation_interference.py` | `results/bench_preactivation_interference.json` | **A** — verified by loadavg 7.25 and 46.41 CPU-s in 6 s wall |
| D5 | s/GB spans **65×** across formats; flat to 1.00–1.16× across a 4× size range | `bench_format_activation` | `results/bench_format_activation.csv` | **B** — all six formats are generated |
| D6 | pyhmmer activation share **~46–49%**, flat across 40×; expansion 2.27–2.31× | `bench_p1_consumer_retention.py` | `results/bench_p1_hmmer_2gb.json`, diag JSON | **B** — synthetic FASTA, random residues |
| D7 | Lustre read collapses **16.3× within one 8 GB read**; local NVMe flat at 1.00× | `diag_p1_superlinear.py` | `results/diag_p1_superlinear_8gb.json` | **B** — 2 observations, 1 node, unknown neighbours |
| D8 | Parquet activation share 73%, expansion 2.93× | inline script, login node, 82 MB | **none** | **C** → re-running as `bench_p1_parquet.py` |
| D9 | raw MRC ≈ 1× expansion, R3 collapses onto R1 | — | **none** | **D** |

D4 deserves note as a *methodological* success: a perfectly flat curve is the
signature of a probe that silently did nothing, so it was re-run with worker
liveness, loadavg and CPU-seconds instrumentation before being believed.

### Offline / replay

| # | claim | instrument | trust |
|---|---|---|---|
| R1 | ceiling: exp_3 15.3% serial / 46.1% concurrent | `replay_ceiling.py` | **A** — negative control passes (chemgraph_swap 1.6%) |
| R2 | capacity: M=1 22.2%, M=2 40.1%, saturates at M=2 | `replay_capacity.py` | **B** — Belady is offline-optimal, so an upper bound |
| R3 | retention worth **45.4%**, sharp threshold at **280 GB** | `replay_retention_policy.py` | **A** — costs all from measured constants |
| R4 | value density ties LRU and Belady **exactly** (negative result) | same | **A** |
| R5 | need **≥34 GB** activated to beat LRU, **≥68 GB** to beat Belady | `sweep_policy_regime.py` | **B** — *simulation*; access pattern synthetic, constants real |

### End-to-end

| # | claim | trust |
|---|---|---|
| E1 | exp_3 full_system 1.0953× over baseline | **X-adjacent** — t = 1.32, **not significant**; must not be reported as a speedup |
| E2 | ablations point the wrong way (`no_plan` fastest at 1.1319×) | **A** as a *finding about our variance*, not about the mechanism |

---

## 4. The three numbers that looked clean and were not

These are the reason this document exists.

### 4.1 L2 sleep — "377× faster wake" that returned `"!!!!"`
The wake timing was real; the engine was destroyed. The probe called
`sanity_generate` and **discarded the output**, so it verified that the server
responded, not that it responded *correctly*. The arithmetic also implied
33.2 GB/s through one path from storage — physically impossible on Gen4 x16, which
is the check that should have caught it first.
**Fix:** coherence is now asserted on generated text and `finish_reason`. L1 passes
this; the degeneracy is L2-only, so the plan's sleep/wake death notice was
wrongly scoped for days.

### 4.2 `io_share = 12.7%` on Lustre — a difference between two identical rungs
`posix_fadvise(DONTNEED)` is a silent no-op on Lustre. Measured on a 64 MB file,
full read then evict:

| filesystem | after read | after evict |
|---|---|---|
| `/storage/scratch1` (lustre) | 1.000 | **1.000** |
| `/storage/project` (nfs) | 1.000 | 0.000 |
| `/tmp` (node-local) | 1.000 | 0.000 |

So "cold" and "warm" were the same state and their difference was run-to-run
noise. **Fix:** `evict_works()` probes the filesystem and emits `null` instead of
a number when control cannot be established.

### 4.3 pyhmmer's 91.8% — the filesystem, not the format
An 8 GB load took 639 s (12.5 MB/s) where 2 GB took 13.65 s (146 MB/s), implying
activation was 91.8% of a tool call and promoting the candidate to lead. The
identical file on node-local NVMe loaded in **52.2 s at 153 MB/s**, with the parse
rate **flat** — 423 krec/s at the first million records and at the nineteenth. On
Lustre it decayed 304 → 18.7 krec/s *within the single read*.
**True value 47.6%. The 91.8% and 12.13× are withdrawn.**
**Fix:** all data-axis timing now runs on node-local NVMe, which is also the only
place eviction works.

**The pattern in all three:** the probe produced a plausible number while not doing
the thing it claimed. None was caught by looking at the number; all three were
caught by asking *what physical event should have occurred, and can I see it?*

---

## 5. Environment hazards — permissions, sharing, and what each cost

Non-obvious properties of this cluster that have consumed real time.

| # | hazard | how it showed up | cost |
|---|---|---|---|
| H1 | **`/tmp` is node-local, not shared** | `PYTHONPATH` pointed at the login node's `/tmp`; `import pyhmmer` failed on the compute node and job output went to a `/tmp` nobody could read — the job looked silently empty | 1 full job (~13 min + queue) |
| H2 | **Lustre ignores `fadvise`** | §4.2 | 1 rerun + a withdrawn column |
| H3 | **Lustre throughput collapses under neighbour load** | §4.3 | 1 rerun + a withdrawn headline |
| H4 | **Login nodes cap memory** | the 2 GB pyhmmer run was killed after generating its file, with no message | 1 rerun |
| H5 | **SLURM silently appends `gpu-v100`** to every hold's partition list | a `mem=1000G` request became `BadConstraints` — permanent, unlike `Resources` — because gpu-v100 is heterogeneous (9 nodes at 191 GB, 26 at 385, **5 at 772**) and nothing fits above the widest node. `sbatch --test-only` disagreed because it evaluates only the requested partition | 2 resubmissions |
| H6 | **`--constraint` makes H5 worse** | `--constraint=RTX-Pro-Blackwell` was the obvious fix and is an *independent second cause* of `BadConstraints`: no v100 node has that feature. Separated by a controlled pair — 750G without it queues, 256G with it does not | 1 resubmission |
| H7 | **Holds land idle if nothing is pointed at them** | watcher hardcoded job IDs, so holds submitted later were invisible; it also exited when its tracked jobs ended — exactly when a new batch arrives | **3 nights**, ~7.6 Blackwell node-hours on 08-04 alone |
| H8 | **The campaign refuses to run on a dirty tree** | correct behaviour (a trial that cannot be tied to a commit is worthless), but my own uncommitted probe scripts held it shut | blocked collection until noticed |
| H9 | **`results/` is gitignored** | measurement artifacts need `git add -f`, so a "committed" experiment can have no data behind it | latent; caused D8's missing artifact |
| H10 | **No DCGM anywhere** | `dcgmi` absent, no `libdcgm`, no module | wake bandwidth stays *derived* (bytes ÷ time) rather than measured at the PCIe counter |
| H11 | **Preemptible QOS** | `embers` is free but 5 of 8 recent GPU jobs were `PREEMPTED`; `inferno` costs real allocation | collection is bursty and cannot be scheduled tightly |
| H12 | **Shared nodes** | an early E4 attempt overlaid a 4-GPU bench onto a hold whose campaign already owned all 4 GPUs; it died after loading 146.82 GB | 1 run; fixed by using GPUs 4–5 on a 6-GPU L40S |

H7 is the most expensive single item in this table by an order of magnitude, and
it recurred three times because each fix was a reminder rather than a structural
change. `submit_holds.sh` + `overnight_watcher8` now make submission and
supervision one action with name-based discovery.

---

## 6. What I would still not trust

1. **Any end-to-end speedup.** σ is 12–40% of the mean and t = 1.32 on the headline
   pair. Detecting 20% at this variance needs ~100 trials/arm.
2. **Absolute seconds across nodes.** A 2.3× CPU difference was measured on the
   same parse (42.83 s vs 98.23 s). Ratios transfer; seconds do not. Facet by node.
3. **Any s/GB from a generated artifact** as evidence about a real workload. The
   format constants are believable; the sizes are ours.
4. **The 72B park cost (~279 GB).** The 1.90× ratio is measured on a 32B and
   extrapolated. Only the ratio is real.
5. **D9 (raw MRC).** Never executed.
6. **`sweep_policy_regime.py` outputs as measurements.** Design answers only.

---

## 7. If you read one thing

The instruments are mostly sound; the *environment* is where the errors came from.
Every withdrawn number in §4 was produced by a correct program running against a
filesystem, a node, or a scheduler that behaved differently than assumed. The
defence that has actually worked is not more careful code — it is asking, for each
number, **what physical event must have happened, and can I observe it
independently?** Bytes moved. Cores burned. Text generated. Pages resident.
