# agent-hpc-runtime

A residency runtime for HPC agent workflows: it decides **what to keep in
memory, at which level, under one budget**, when an LLM agent's next resource
need is not known in advance.

The repo contains **two subsystems**, built in that order:

| | what it does | status |
|---|---|---|
| **`runtime/residency/`** — *Tandem* | Holds resources at the highest rung already paid for, and arbitrates model weights against activated data under a single host-RAM budget. | current work |
| `runtime/prefetch/` + `predictor/` + `guard/` | Predicts the next resource need and starts loading it early, cancelling on divergence. | earlier; still ships and is still a baseline arm |

If you are new, read **[The idea](#the-idea-the-cost-ladder)** then
**[Navigating runtime/residency](#navigating-runtimeresidency)**. If you are
here to run something, jump to **[Running it](#running-it)**.

---

## The idea: the cost ladder

Making a resource usable is a *chain* of costs. A **residency level** is a
prefix of that chain you have already paid:

| rung | state | cost paid to reach it | cost type |
|---|---|---|---|
| `R0_DISK` | on disk | — | — |
| `R1_PAGE_CACHE` | bytes in page cache | disk → RAM | movement |
| `R2_PROCESS_BYTES` | bytes in a process address space (vLLM L1 sleep) | RAM → process | movement |
| `R3_ACTIVATED` | activated structure in a live consumer | parse / decode / construct | **transformation** |

Two measured facts make this a system rather than a taxonomy:

* A **72B model load** is 74–81% weight *movement*; engine init is ~4.7%.
  So the win is to **never fall below R2** — park the weights in host RAM —
  which skips the movement rather than overlapping it. ~1600 s → ~2 s.
* An **EAM potential load** is 1.9% I/O and **98.1% parse**. Staging its bytes
  early recovers at most 1.9%. The only win is to have already run the parse —
  hold it at **R3 in a live worker**.

So the correct action differs *in kind* by resource class, and a byte-oriented
tier (which only ever operates R0→R1) cannot express either one: R2 is process
state, not a file range, and R3 is transformed state. **Rung coverage, not
prediction accuracy, is the ceiling.** That claim is the reason this subsystem
exists.

Both classes draw on the **same host-RAM budget**, so something must arbitrate
between them. That arbitration is the contribution.

### Eq. 1 — what a held resource is worth

```
v(r) = (cold(r) − ready(r)) · D / max(Δt(r), D)          ranked by v(r) / g(r)
```

`cold − ready` is the stall avoided by one reuse; `g` is GB held; `D` is the
decay scale (default 60 s); `Δt` is time to next use. **`D` is not the
lookahead** — the estimator's reach `L` is passed separately, because setting
them equal makes the value function time-blind. See `contract.py`, invariant I3.

---

## Navigating `runtime/residency/`

Read in this order. Each file states its own contract in its module docstring.

| # | file | what it is | read it for |
|---|---|---|---|
| 1 | **`contract.py`** | the frozen interface | rungs, `ResourceSpec`, Eq. 1, the `ResidencyActor` / `Ledger` / `Arbitrator` protocols, and **invariants I1–I5** |
| 2 | `ledger.py` (T1) | one budget over every class | measured charges, confirmed releases, leak accounting |
| 3 | `arbitrator.py` (T2) | the retention policy | greedy, **chained** (up to `DEFAULT_MAX_VICTIMS=3`), ranked by `v/g`; `admit()` returns a plan and does **not** mutate the ledger |
| 4 | `horizon.py` (T3) | the estimator | `next_use_s`, the demand map, transition signal |
| 5 | `model_actor.py` (T4a) | models at R2 | vLLM L1 park/wake, GPU eviction, `MODEL_CATALOGUE` of measured costs |
| 6 | `data_worker.py` (T4b) | data at R3 | the resident, evictable LAMMPS worker |

**Start with `contract.py`.** Everything else depends only on its protocols,
and its five invariants each record a specific failure that motivated them:

| | invariant | why |
|---|---|---|
| **I1** | `held_gb` is **measured**, not declared | a declared footprint is a wish |
| **I2** | `release()` is confirmable **by independent measurement** | an actor reporting its own release is not evidence |
| **I3** | the horizon never says "never again", only "not within the lookahead" | one wrong "never" discards a resource permanently |
| **I4** | the arbitrator is **class-blind** | class-specific knowledge lives in actors, or the budget stops being one budget |
| **I5** | v1 currency is **retain-only** | mixing retain and prefetch in one score misranked them by 280× |

### The one thing that will confuse you

The residency actor sits on **two different call paths**, and they are wired
separately:

```
DEMAND path   agent needs a model now
              → ModelRouter.ensure_ready()            (workloads/AtomAgents/.../model_router.py)
              → actor.activate()                      ← parks the incumbent if the budget allows

PREFETCH path predictor says a model is coming
              → ModelPrefetchExecutor                 (runtime/prefetch/model_prefetch.py)
              → actor.activate()
```

**Almost every model change this workload performs comes through the DEMAND
path.** Wiring the actor only to the prefetch path produces a system that
looks fully configured, logs `TANDEM: VllmModelActor wired`, and never parks
anything — which is exactly what trials t03 and t04 did for 3.2 hours each.
`test_router_demand_path_residency.py` pins this, including that the no-actor
path stays byte-identical to the router that produced every earlier trial.

### The budget is read, not assumed

`_can_park` (`experiments/atomagents_exp3.py`) walks **up** the cgroup tree to
the first real `memory.max`. SLURM sets the limit on the *job* cgroup; the leaf
step cgroups inherit enforcement without carrying the file. Reading only the
leaf finds no limit and silently permits every park. The guard prints its
arithmetic on every call:

```
[tandem] cgroup limit from /sys/fs/cgroup/system.slice/slurmstepd.scope/s8FXM1H0C82200
[tandem] can_park(qwen_32b): need 129.7 GB, 220.2 GB spendable of 274.9 GB (13.4 used) -> PARK
[tandem] can_park(qwen_72b): need 279.0 GB,  85.9 GB spendable of 274.9 GB (147.7 used) -> STOP
```

If you see a park with no `can_park` line above it, the guard is not running.

---

## Running it

### No GPU — replay recorded traces through the real policy

The fastest way to exercise the policy is to replay recorded need sequences
through the **shipped** arbitrator and ledger:

```bash
python3 scripts/replay_tandem_trace.py --budgets 256,400,560,700
python3 scripts/replay_tandem_trace.py --calibrate-only          # trust gate
python3 scripts/replay_tandem_trace.py --lookahead-s 7200        # sweep L
```

This imports `GreedyArbitrator`, `ResidencyLedger` and `contract.value()`
rather than reimplementing them, so a defect in Eq. 1 or the eviction chain
shows up as a wrong number. It reproduces measured wall time to a median 2.5%
with retention off — run `--calibrate-only` first and distrust any other arm if
that gate is wide.

Note it tests the **policy**, not the mechanism: its actor is a bookkeeping
stub, so I2 passes trivially and the vLLM sleep endpoint is never touched.

### With GPUs — an end-to-end trial

```bash
# single trial
python3 experiments/atomagents_exp3.py --hw-profile blackwell_swap \
        --swap-models --residency

# through the eval driver (the `tandem` arm is explicit_only)
python3 experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
        --configs baseline,tandem --trials 4 --order roundrobin
```

`--residency` is **off by default**; with it off the runtime is unchanged.

Ready-made SLURM jobs are in `experiments/`: `job_tandem_paired.sh` (both arms
in one allocation — the only shape that licenses a ratio, since the two
Blackwell nodes differ by **1.42× on identical weights**) and
`job_tandem_700g.sh` (the budget experiment).

### Parsing a trial

```bash
python3 scripts/parse_tandem_trial.py results/eval_q1_q4/runs/.../tandem/<trial>/
```

It answers six questions in **dependency order — completeness first** — and
refuses to compute a speedup without a same-family baseline. A trial with
fewer than ~5 model swaps is truncated; its wall time is not a measurement.

### Tests

```bash
python3 -m pytest runtime/tests/ -q                        # 588 passed, 11 skipped
python3 -m pytest runtime/tests/ -q -k residency           # 219 passed
```

One test (`test_evict_then_mincore_reports_cold_on_local_tmp`) is load-flaky —
it reads real page-cache state and fails under machine load. If it is the only
failure, it is not your change.

---

## Status — what is measured and what is not

Being explicit, because the design is further along than the evidence.

**Established:**
- The mechanism runs end-to-end: the actor parks, refuses when the budget says
  so, and prints its arithmetic. First live park 2026-09-02.
- The `Cannot start … occupied by` failure that killed 10 of 13 admitted
  prefetches is **gone** — 0 occurrences across completed trials.
- Park cost is measured twice by independent paths, 0.01 GB apart.

**Not established:**
- **No speedup.** At `--mem=256G` the tandem arm measured **1.52× slower**
  (n=3, 11161.9 s vs 7362.4 s baseline, n=4). 91.6% of that gap is per-load
  cost, not policy.
- **No wake has ever occurred in a trial.** At 256 GB only `qwen_32b`
  (129.7 GB) fits the budget, and it is used once and never reused; the two
  models reused 4× and 2× are each larger than the whole spendable budget.
- Tandem and baseline trials have run on **different nodes**, so every ratio is
  arm-and-node, not arm alone.

The reuse-distance distribution says the budget threshold is **653 GB** (the
retained 72B plus the one used between its two uses), which is what
`job_tandem_700g.sh` tests.

---

## Repository layout

```
runtime/
  residency/        Tandem — one budget over every held resource class
    contract.py       the frozen interface + invariants I1-I5   (start here)
    ledger.py         T1  measured charges, confirmed releases
    arbitrator.py     T2  greedy chained retention, ranked by v/g
    horizon.py        T3  next-use estimation
    model_actor.py    T4a models at R2 (vLLM L1 park/wake, GPU eviction)
    data_worker.py    T4b data at R3 (resident, evictable worker)
  prefetch/         Speculative acquisition + scheduler (the earlier subsystem)
  predictor/        Predicts upcoming needs (mock, oracle, learned)
  adapters/         Non-invasive hooks into workflow callbacks
  guard/            Divergence detection + checkpointing
  measurement/      Timing records and storage-hierarchy probes
  analysis/         JSONL trace analysis
  tests/            Unit tests
  demo/             Local demos (no GPU)

experiments/        Cluster runners, SLURM jobs, benchmarks
scripts/            Replay, parsing, and analysis tools (mostly no GPU)
setup/              Environment setup for GPU nodes
workloads/          Git submodules pinned at tested commits
  AtomAgents/       vLLM model swapping; hosts ModelRouter (the demand path)
  ChemGraph/        MACE model prefetch
```

---

## The earlier subsystem: prediction-driven prefetch

Still present, still a baseline arm, and the source of the traces the residency
work replays.

After each LLM response an adapter predicts which model weights or data files
the next tool call needs, starts loading them in a background thread, and
cancels if the agent diverges:

```
[LLM response]
      ↓  adapter intercepts                     (adapters/)
      ↓  predictor.predict() → ResourceSpec[]   (predictor/)
      ↓  detector.on_prediction() → checkpoint  (guard/)
      ↓  scheduler.schedule() → background task (prefetch/)

[Tool starts]
      ↓  detector.on_tool_about_to_execute()
      ├─ HIT  → prediction_validated, overlap recorded
      └─ MISS → divergence_detected, pending prefetches cancelled
```

All events go to a JSONL trace alongside the workflow's own events, so each run
is self-contained and analyzable offline.

### Runtime modes

| Mode | Prefetch I/O | Overhead | Use for |
|---|---|---|---|
| `baseline` | None | Zero | Clean comparison baseline |
| `observe_only` | None | Minimal | First cluster run, accuracy measurement |
| `simulated` | None | Minimal | Decision logging, estimated benefit |
| `real` | Yes | Background thread | Actual overlap measurement |

---

## Cluster setup

```bash
# 1. Create environments (chemgraph, atomagents, vllm) + build LAMMPS
bash setup/setup_all.sh

# 2. Download model weights (~200 GB)
bash setup/download_models.sh

# 3. Verify
python3 -m pytest runtime/tests/ -q
```

### Adding workload submodules

```bash
git submodule add https://github.com/<you>/AtomAgents workloads/AtomAgents
git commit -m "Add workload submodules"
```

Pin to tested commits before submission:

```bash
cd workloads/AtomAgents && git checkout <tested-commit> && cd ../..
git add workloads/AtomAgents && git commit -m "Pin AtomAgents to tested commit"
```

---

## Hardware probes

The runtime detects whether a model load came from NFS, local SSD, or OS page
cache using `/proc/self/io` byte counters and `nvidia-smi` VRAM deltas, so a
measured speedup can be attributed to real I/O overlap rather than a cache hit.

```python
from runtime.measurement.cluster_probes import LoadProbeContext

with LoadProbeContext() as ctx:
    model.load()

print(ctx.delta.likely_source)   # "nfs" | "page_cache" | "mixed"
print(ctx.delta.summary_line())
```

---

## Conventions worth knowing before you measure anything

* **Never pool L40S with Blackwell.** Identical work differs by up to 4.0×
  across node types. `summary.json` has **no** `gpu_name` — GPU identity comes
  from `meta.json` → `gpus[0]`.
* **Facet by node even within a type.** The two Blackwell nodes differ by 1.42×
  on the same weights.
* **Record the allocation.** `meta.json` carries `slurm_mem_mb`; trials at
  different `--mem` are different configurations and must not be pooled.
* **A completed process is not a completed workflow.** `completed_trials()`
  counts `status == "completed"`, which only means the driver exited 0. Gate on
  swap count and whether the trial reached LAMMPS.
* **Any retention percentage needs the oracle-vs-LRU gap at the same budget**,
  or it is a claim about retention in general, not about this policy.

---

## Citation

*Paper in preparation.*
