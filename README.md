# agent-hpc-runtime

A residency runtime for HPC agent workflows: it decides **what to keep in
memory, at which level, under one budget**, when an LLM agent's next resource
need is not known in advance.

The repo contains **two subsystems**, built in that order:

| | what it does | status |
|---|---|---|
| **`runtime/residency/`** — *Tandem* | Holds resources at the highest rung already paid for, and arbitrates model weights against activated data under a single host-RAM budget. | current work |
| `runtime/prefetch/` + `predictor/` + `guard/` | Predicts the next resource need and starts loading it early, cancelling on divergence. | earlier; still ships and is still a baseline arm |

New here? Read **[The idea](#the-idea-the-cost-ladder)**, then
**[Navigating runtime/residency](#navigating-runtimeresidency)**.
Here to run something? Jump to **[Running it](#running-it)**.

> **This README carries no measurements on purpose.** Numbers go stale and get
> quoted out of context. Every figure lives in `results/` as JSON next to the
> script that produced it, and each benchmark's docstring states what it
> measured and what it does not. Read those, not this.

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

The design turns on one observation: **the two resource classes are dominated
by costs of different kinds**, so the right action differs in kind too.

* A **model** load is dominated by *moving* weights. The win is therefore not
  to move them earlier but to **never fall below R2** — park them in host RAM,
  so the move is skipped rather than overlapped.
* A **data artifact** load is dominated by *transforming* bytes into a
  structure. Staging the bytes early recovers almost nothing. The win is to
  have already run the transform, holding the result at **R3 in a live
  consumer process**.

A byte-oriented tier only ever operates R0→R1, so it can express neither: R2 is
process state rather than a file range, and R3 is transformed state. **Rung
coverage, not prediction accuracy, is what bounds the achievable win** — that
is the argument this subsystem exists to test.

Both classes draw on the **same host-RAM budget**, so something must arbitrate
between them. That arbitration is the contribution.

Where the evidence for each claim lives:

| claim | benchmark | raw |
|---|---|---|
| model park/wake is cheap and coherent | `experiments/bench_wake_cache_dependence.py` | `results/bench_wake_L1_coherence_*.json` |
| warming the page cache is an unreliable substitute | `experiments/bench_activation_ladder.py` | `results/bench_activation_ladder_*.json` |
| a data load is transformation-dominated | `experiments/bench_potential_activation.py` | `results/bench_potential_activation_*.json` |
| holding at R3 pays off, physics unchanged | `experiments/bench_activated_residency.py` | `results/verify_persistent_lammps_BIG.json` |

### Eq. 1 — what a held resource is worth

```
v(r) = (cold(r) − ready(r)) · D / max(Δt(r), D)          ranked by v(r) / g(r)
```

`cold − ready` is the stall avoided by one reuse; `g` is GB held; `D` is the
decay scale (`DEFAULT_DECAY_S`); `Δt` is time to next use. **`D` is not the
lookahead** — the estimator's reach `L` is passed separately, because setting
them equal makes the value function time-blind. See `contract.py`, invariant I3.

---

## Navigating `runtime/residency/`

Read in this order. Each file states its own contract in its module docstring.

| # | file | what it is | read it for |
|---|---|---|---|
| 1 | **`contract.py`** | the frozen interface | rungs, `ResourceSpec`, Eq. 1, the `ResidencyActor` / `Ledger` / `Arbitrator` protocols, and **invariants I1–I5** |
| 2 | `ledger.py` (T1) | one budget over every class | measured charges, confirmed releases, leak accounting |
| 3 | `arbitrator.py` (T2) | the retention policy | greedy and **chained** (bounded by `DEFAULT_MAX_VICTIMS`), ranked by `v/g`; `admit()` returns a plan and does **not** mutate the ledger |
| 4 | `horizon.py` (T3) | the estimator | `next_use_s`, the demand map, the transition signal |
| 5 | `model_actor.py` (T4a) | models at R2 | vLLM L1 park/wake, GPU eviction, `MODEL_CATALOGUE` of measured per-model costs |
| 6 | `data_worker.py` (T4b) | data at R3 | the resident, evictable LAMMPS worker |

**Start with `contract.py`.** Everything else depends only on its protocols,
and its five invariants each record a specific failure that motivated them:

| | invariant | why it exists |
|---|---|---|
| **I1** | `held_gb` is **measured**, not declared | a declared footprint is a wish |
| **I2** | `release()` is confirmable **by independent measurement** | an actor reporting its own release is not evidence |
| **I3** | the horizon never says "never again", only "not within the lookahead" | one wrong "never" discards a resource permanently |
| **I4** | the arbitrator is **class-blind** | class-specific knowledge belongs in actors, or the budget stops being one budget |
| **I5** | v1 currency is **retain-only** | scoring retain and prefetch on one scale ranked them incomparably |

The actors (T4a, T4b) are deliberately **not** exported from the package
`__init__`: they import vLLM and LAMMPS clients, and keeping them out is what
lets the policy be unit-tested and replayed with no GPU and no engine.

### Two call paths — worth knowing before you debug anything

The residency actor is reachable from **two places**, wired separately:

```
DEMAND path   agent needs a model now
              → ModelRouter.ensure_ready()      (workloads/AtomAgents/…/model_router.py)
              → actor.activate()                ← parks the incumbent if the budget allows

PREFETCH path predictor says a model is coming
              → ModelPrefetchExecutor           (runtime/prefetch/model_prefetch.py)
              → actor.activate()
```

**Almost every model change this workload performs comes through the DEMAND
path** — the agent asks for a model it does not currently have. If the actor is
attached only to the prefetch path, the system looks fully configured and logs
`TANDEM: VllmModelActor wired`, but never parks anything. That is a quiet
failure rather than a loud one, so check it first when a trial shows no parks.

`runtime/tests/test_router_demand_path_residency.py` pins the wiring, including
the case that matters most for comparability: with no actor attached, the
router must behave identically to the one that produced every earlier trial.

### The budget is read, not assumed

`_can_park` (in `experiments/atomagents_exp3.py`) walks **up** the cgroup tree
to the first real `memory.max`. SLURM sets the limit on the *job* cgroup, and
the leaf step cgroups inherit enforcement without carrying the file — so
reading only the leaf finds no limit and silently permits every park.

The guard logs its arithmetic on every call, in this shape:

```
[tandem] cgroup limit from <cgroup path>
[tandem] can_park(<model>): need … GB, … GB spendable of … GB limit (… used, …% reserved) -> PARK
[tandem] can_park(<model>): need … GB, … GB spendable of … GB limit (… used, …% reserved) -> STOP
```

If you see a park with no `can_park` line above it, the guard is not running.

---

## Running it

### No GPU — replay recorded traces through the real policy

The fastest way to exercise the policy is to replay recorded need sequences
through the **shipped** arbitrator and ledger:

```bash
python3 scripts/replay_tandem_trace.py --calibrate-only     # run this first
python3 scripts/replay_tandem_trace.py --budgets 256,400,560,700
python3 scripts/replay_tandem_trace.py --lookahead-s 7200   # sweep the estimator's reach
python3 scripts/replay_tandem_trace.py --beyond-l at-l,2l,zero
```

It imports `GreedyArbitrator`, `ResidencyLedger` and `contract.value()` rather
than reimplementing them, so a defect in Eq. 1 or in the eviction chain shows up
as a wrong number. `--calibrate-only` replays with retention off, where
predicted wall time should match what the trial actually took; if that gate is
wide, distrust every other arm.

It tests the **policy**, not the mechanism — its actor is a bookkeeping stub, so
I2 passes trivially and the vLLM sleep endpoint is never touched.

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

Ready-made SLURM jobs live in `experiments/`:

| job | what it is for |
|---|---|
| `job_tandem_paired.sh` | both arms in one allocation — the only shape that licenses a ratio, since nodes differ measurably even within a GPU type |
| `job_tandem_700g.sh` | the host-RAM budget experiment |
| `job_sleepmode_boot.sh` | isolates the cost of the sleep-mode launch flags |

### Reading a trial

```bash
python3 scripts/parse_tandem_trial.py results/eval_q1_q4/runs/…/tandem/<trial>/
```

It answers six questions in dependency order, completeness first, and will not
compute a speedup without a same-family baseline. Trials cut short by SLURM
preemption are common; the parser identifies them, and their wall time should
not be used.

### Tests

```bash
python3 -m pytest runtime/tests/ -q
python3 -m pytest runtime/tests/ -q -k residency      # no GPU required
```

`test_evict_then_mincore_reports_cold_on_local_tmp` reads real page-cache state
and is load-flaky. If it is the only failure, it is not your change.

---

## Current status

The mechanism runs end-to-end: the actor parks, declines when the budget says
to, and logs its reasoning either way. Currently trying to demonstrate an end-to-end speedup, and the open questions are mostly about *configuration*
rather than mechanism — whether the budget admits the resources that are
actually reused, and whether the estimator's reach covers the workload's reuse
distances.

To see where things currently stand, run the tooling rather than trusting a
summary: `scripts/replay_tandem_trace.py` for the policy with no GPU, and
`scripts/parse_tandem_trial.py` on the newest trial directory under
`results/eval_q1_q4/runs/`.

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
results/            Benchmark JSON + eval trial directories
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
# 1. Create the three conda environments (chemgraph, atomagents, vllm)
bash setup/setup_cloud.sh

# 2. Smoke-test all three
bash setup/verify.sh

# 3. Run the unit tests
python3 -m pytest runtime/tests/ -q
```

`setup/CLOUD_SETUP.md` is the fuller guide, including hardware minimums and how
to bring the model servers up (`setup/start_models.sh`). Environment
definitions are `setup/environment_atomagents.yml` and
`setup/environment_vllm.yml`.

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

Learned the hard way; they will save you a wasted run.

* **Never pool L40S with Blackwell.** Identical work differs substantially
  across node types. `summary.json` has **no** `gpu_name` — GPU identity comes
  from `meta.json` → `gpus[0]`.
* **Facet by node even within a type.** Two nodes of the same GPU model can
  differ measurably on the same weights, so compare arms within an allocation
  where you can.
* **Record the allocation.** `meta.json` carries `slurm_mem_mb`; trials run at
  different `--mem` are different configurations and should not be pooled.
* **A completed process is not a completed workflow.** `completed_trials()`
  counts `status == "completed"`, which only means the driver exited 0 — a
  trial cut short by preemption still counts. Gate on swap count and whether
  the trial reached LAMMPS; `parse_tandem_trial.py` does this for you.
* **Report a retention result alongside the LRU comparison at the same
  budget.** Plain retention already captures much of the win; the number that
  says something about *this* policy is the gap over LRU.

---

## Citation

*Paper in preparation.*
