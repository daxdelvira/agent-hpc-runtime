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

Two measurements make this a system rather than a taxonomy, and they point in
opposite directions:

**Models — the win is to skip the movement, not to schedule it earlier.**
Warming the page cache before a boot is an unreliable lever: across 9 nodes the
cold-minus-warm difference ranges from **−7.0% to +66.9%** and changes sign
(`results/bench_activation_ladder_*.json`), because it removes only the
disk→RAM leg and leaves RAM→GPU untouched. Parking the weights at R2 removes
both legs, and does so consistently: **782.27 s cold boot → 2.076 s wake**, with
verbatim-correct output and unchanged throughput
(`results/bench_wake_L1_coherence_32b.json`).

**Data — the win is to have already run the parse.** A 3.32 GB EAM potential
splits three ways: **2.0% disk→RAM, 3.9% RAM→process, 93.0% parse and spline
construction** (`bench_potential_activation.py`, re-measured with `getrusage`
2026-08-09). Staging the bytes early therefore recovers at most ~2%. Holding
the activated structure in a live worker takes a redundant invocation from
**93.73 s → 10.56 s**, physics bit-identical (`verify_persistent_lammps_BIG.json`).

> An older two-way "1.9% / 98.1%" split for the potential is superseded — it
> folded the RAM→process row into activation. Cite 93.0%. See the note at the
> top of `experiments/bench_activated_residency.py`.

So the correct action differs *in kind* by resource class, and a byte-oriented
tier (which only ever operates R0→R1) cannot express either one: R2 is process
state rather than a file range, and R3 is transformed state. **Rung coverage,
not prediction accuracy, is what bounds the achievable win.** That is the
argument this subsystem exists to test.

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

### Two call paths — worth knowing before you debug anything

The residency actor is reachable from **two places**, and they are wired
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
path** — the agent asks for a model it does not currently have. If the actor is
attached only to the prefetch path, the system looks fully configured and logs
`TANDEM: VllmModelActor wired`, but never actually parks anything. That is a
quiet failure mode rather than a loud one, so it is worth checking first when a
trial shows no parks.

`test_router_demand_path_residency.py` pins the wiring, including the case that
matters most for comparability: with no actor attached, the router must behave
byte-identically to the one that produced every earlier trial.

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

It answers six questions in dependency order, completeness first, and will not
compute a speedup without a same-family baseline. A trial with fewer than ~5
model swaps was cut short (usually by preemption) and its wall time should not
be used.

### Tests

```bash
python3 -m pytest runtime/tests/ -q                        # 588 passed, 11 skipped
python3 -m pytest runtime/tests/ -q -k residency           # 219 passed
```

One test (`test_evict_then_mincore_reports_cold_on_local_tmp`) is load-flaky —
it reads real page-cache state and fails under machine load. If it is the only
failure, it is not your change.

---

## Where the evidence stands

The design is further along than the measurements, which is normal at this
stage but worth knowing before you quote anything from a trial.

**Working and verified:**
- The mechanism runs end-to-end. The actor parks, declines when the budget says
  to, and prints its arithmetic either way. First live park 2026-09-02.
- The `Cannot start … occupied by` failure that used to kill most admitted
  prefetches no longer occurs — 0 across all completed trials.
- Park cost agrees to 0.01 GB when measured by two independent paths.
- 219 residency unit tests pass with no GPU.

**Still open — good places to contribute:**
- **No end-to-end speedup yet.** At `--mem=256G` the tandem arm currently
  measures slower than baseline (11161.9 s, n=3, against 7362.4 s, n=4). The
  decomposition is informative: **91.6% of that gap is per-model-load cost, and
  the two arms perform the same number of loads** — so it is not the policy
  making extra work.
- **No wake has been observed in a trial yet**, and the reason is arithmetic
  rather than mechanical. At 256 GB the only model that fits the budget
  (`qwen_32b`, 129.7 GB) is used once and never reused, while the two models
  reused 4× and 2× are each larger than the whole spendable budget. The
  reuse-distance distribution puts the threshold at **653 GB**, which is what
  `job_tandem_700g.sh` exists to test.
- **Tandem and baseline trials have mostly run on different nodes**, and the
  two Blackwell nodes differ by 1.42× on identical work — so present ratios
  carry a node term. `job_tandem_paired.sh` runs both arms in one allocation to
  remove it.

`scripts/replay_tandem_trace.py` is the cheapest way to explore any of these:
it drives the real policy over recorded traces with no GPU, and reproduces
measured wall time to a median 2.5% with retention off.

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

## Conventions for measurement

* **Never pool L40S with Blackwell.** Identical work differs by up to 4.0×
  across node types. `summary.json` has **no** `gpu_name` — GPU identity comes
  from `meta.json` → `gpus[0]`.
* **Facet by node even within a type.** The two Blackwell nodes differ by 1.42×
  on the same weights.
* **Record the allocation.** `meta.json` carries `slurm_mem_mb`; trials at
  different `--mem` are different configurations and must not be pooled.
* **A completed process is not a completed workflow.** `completed_trials()`
  counts `status == "completed"`, which only means the driver exited 0 — a
  trial cut short by SLURM preemption still counts. Gate on swap count and
  whether the trial reached LAMMPS; `parse_tandem_trial.py` does this for you.
* **Report a retention percentage alongside the LRU comparison at the same
  budget.** Plain retention (a vLLM flag plus a loop) already gets a large
  share of the win; the number that says something about *this* policy is the
  gap over LRU.

---

## Citation

*Paper in preparation.*
