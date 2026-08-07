#!/usr/bin/env python3
"""
bench_arbitration_harness.py — measured wall clock for retention AND prefetch
arbitrated against ONE host-RAM budget.

WHAT THIS IS FOR
----------------
Everything so far is either a mechanism measurement (park a model: 782 -> 2.08 s;
retain a parsed database: 93.73 -> 10.56 s) or an offline replay under a cost
model. Neither is a measured end-to-end speedup, which is what the paper's
acceptance bar actually requires. This harness drives the REAL resources on a
schedule and reports WALL CLOCK.

THE SYSTEM MAKES TWO KINDS OF DECISION, NOT ONE
-----------------------------------------------
An earlier design of this harness tested retention only. That understates the
problem, because the interesting system trades those decisions off:

  RETAIN   decided at EVICTION time, when the past is known. Cost: RAM held.
           A wrong retain wastes memory.
  PREFETCH decided at FETCH time, on a PREDICTION. Cost: RAM held *earlier*,
           plus bandwidth. A wrong prefetch wastes memory AND the load, and can
           evict something that was going to be used.

They draw on the SAME budget, so choosing to prefetch X may mean evicting
retained Y. One value function, two actions. That is the design claim, and the
arms below are built to separate it:

    never_retain              every need pays full cold cost         baseline
    retain_lru                recency only, PAST information only    REALIZABLE baseline
    retain_vd_pred            cost-aware, predictor-driven           REALIZABLE system
    retain_vd_pred_prefetch   the same, plus speculative loading     REALIZABLE, both actions
    retain_belady             furthest-next-use                      ORACLE bound
    retain_vd_oracle          cost-aware with TRUE next use          ORACLE bound

ORACLE VS REALIZABLE IS A DISTINCTION THIS FILE ONCE GOT WRONG. An earlier
version computed the next use by reading ahead in the schedule and handed that
to `value_density` -- making the "system" arm depend on Belady-grade knowledge
no deployed scheduler has. It also gave LRU an "is this ever used again?"
shortcut from the same source, so the baseline was not LRU either. Both arms were
stronger than reality, in opposite directions, and their comparison meant
nothing. The oracle arms are now labelled as BOUNDS and the realizable arms see
only the past or a predictor.

`--predict-accuracy` (default 0.55, the middle of this project's measured 45-62%)
tests the standing claim that such accuracy is disqualifying for PREFETCHING but
adequate for RETENTION -- because a retention decision is made at eviction time,
when the past is already known, and only the horizon is guessed.

WHY COMPUTE WINDOWS ARE IN THE SCHEDULE
---------------------------------------
Prefetching can only pay if there is something to hide behind. The schedule
therefore carries explicit compute steps, and a prefetch is issued on a
background thread so it genuinely overlaps. A real hmmsearch against UniRef90
takes ~566 s, which is a genuine window; the AtomAgents workload had to fake one
with a 900 s time.sleep, and that fake is one of the write-up's weakest points.

BUDGET REGIMES (the arbitration is real at the PRODUCTION allocation)
---------------------------------------------------------------------
    32B parked at R2   129.7 GB       UniRef50 retained    36.08 GB
    72B parked at R2   279.0 GB       UniRef90 retained   117.20 GB

At the 256 GB the campaigns actually request: a 72B CANNOT be parked at all
(E5's 280 GB threshold), 32B + UniRef90 = 246.9 just fits, and 32B + both
databases = 283.0 does not. That is a real knapsack. The production code already
knows this -- atomagents_exp3.py defaults to sleep level 2 precisely because
three L1-parked engines need ~355 GB against a 256G cgroup.

MODES
-----
  --simulate   no GPU, no data; every cost is a measured constant and time.sleep
               stands in for the work. For validating the SCHEDULER's decisions
               and the accounting. Fast. NOT a speedup measurement.
  (default)    real vLLM engines via ModelOrchestrator and real pyhmmer blocks.
               This is the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "workloads", "AtomAgents"))


# --- resource catalogue: every number here is measured ------------------------
# Models: park cost = weights x 1.90 (E4). Cold boots are medians of measured
# model_load durations from the exp3 traces. Wakes are E4-derived.
# Data: activated sizes and load times from the real UniProt runs.
CATALOGUE = {
    "qwen_32b":      dict(cls="model", held_gb=129.7, cold_s=495.2, ready_s=1.03),
    "qwen_72b":      dict(cls="model", held_gb=279.0, cold_s=800.5, ready_s=2.21),
    # THE THIRD MODEL, restored 2026-08-06. Omitting it dropped 50 of 275 real
    # needs and, worse, removed 276.3 GB of parked footprint from an experiment
    # whose entire subject is memory pressure. All three parked is 685 GB, which
    # is exactly why atomagents_exp3.py defaults to sleep level 2 in production.
    "qwen_72b_text": dict(cls="model", held_gb=276.3, cold_s=770.3, ready_s=2.19),
    "uniref50":      dict(cls="data",  held_gb=36.08, cold_s=107.1, ready_s=0.0,
                          path="/storage/scratch1/7/avandevoorde3/p1/data/uniref50.fasta"),
    "uniref90":      dict(cls="data",  held_gb=117.20, cold_s=372.6, ready_s=0.0,
                          path="/storage/scratch1/7/avandevoorde3/p1/data/uniref90.fasta"),
}

# The three LAMMPS phases in the traces -- lattice_constant, screw_initial and
# relax_screw -- are three invocations against the SAME activated potential.
# They must map to ONE data resource: that identity IS the redundancy the
# persistent worker removes ("3 invocations, 1 activated structure"). Mapping
# only lattice_constant, as an earlier version did, silently discarded two
# thirds of the data-side reuse and made retention look worthless on the data
# axis. Model cold costs above are medians over 45-109 real observations.
TRACE_DATA_MAP = {
    "lattice_constant": "uniref90",
    "screw_initial":    "uniref90",
    "relax_screw":      "uniref90",
}


def s_per_gb(name: str) -> float:
    r = CATALOGUE[name]
    return (r["cold_s"] - r["ready_s"]) / r["held_gb"]


def cgroup_limit_gb() -> float:
    """This job's REAL host-memory ceiling, read from the slurmstepd cgroup.

    NOT /proc/meminfo, which is host-wide even inside a job (196 GB reported
    against a 16 GB request) and is what demoted the park-cost measurement from
    trust A to B. The path must be resolved through /proc/self/cgroup;
    /sys/fs/cgroup/memory.max is not readable directly.
    """
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
        for f in (f"/sys/fs/cgroup{rel}/memory.max",
                  f"/sys/fs/cgroup{rel}/../memory.max"):
            try:
                v = open(f).read().strip()
                if v and v != "max":
                    return int(v) / (1024 ** 3)
            except OSError:
                continue
    except OSError:
        pass
    return float("inf")


def assert_budget_is_safe(budget_gb: float, simulate: bool) -> None:
    """Refuse a budget the allocation cannot actually hold.

    THIS GUARD EXISTS BECAUSE THE FAILURE MODE IS A DEAD NODE. The scheduler's
    budget is bookkeeping; the parks it authorises are real host memory. If the
    budget exceeds the cgroup, the harness will happily park past the limit and,
    per atomagents_exp3.py:444-446, "the failure mode is the node dying rather
    than a clean error." Gate (b) hit exactly this at k=3.
    """
    if simulate:
        return
    limit = cgroup_limit_gb()
    if limit == float("inf"):
        print("  !! cgroup limit unreadable — cannot verify the budget is safe. "
              "Proceeding only because nothing can be checked; watch host RAM.")
        return
    headroom = 0.85 * limit
    if budget_gb > headroom:
        raise SystemExit(
            f"REFUSING budget {budget_gb:.0f} GB: this job's cgroup allows "
            f"{limit:.0f} GB and the harness needs working room beyond the "
            f"retained set (load buffers, the engine's own footprint). "
            f"Safe ceiling here is {headroom:.0f} GB. Request a larger "
            f"allocation or lower --budgets; do NOT raise this threshold to "
            f"make a run fit.")


# --- schedules ----------------------------------------------------------------
# ("need", resource) | ("compute", seconds)
#
# ⚠️ THE HAND-WRITTEN SCHEDULE BELOW IS NOT REALISTIC AND MUST NOT CARRY A CLAIM.
# Measured from 37 real exp3 trials, the gaps BETWEEN resource needs -- which are
# the windows a prefetch has to hide inside -- are:
#
#       median 6.5 s     p75 36.7 s     >=60 s: 9% of gaps     >=300 s: 1%
#
# Every compute window in the hand-written schedule (60-566 s) sits in the top 9%
# of real gaps and most sit in the top 1%. A model cold boot is 440-765 s, so in
# the real workload a prefetch essentially never has room to hide -- which is the
# same `no_window` result the stall taxonomy reported for chemgraph_swap.
#
# Real need ORDER is also strongly structured: 15 distinct orders across 37
# trials, the most common appearing 9 times, versus uniform-random over 4
# resources in the synthetic generator.
#
# So: HAND_SCHEDULE is a demonstration fixture for validating scheduler logic.
# schedule_from_traces() is what any reported number must be built on.
HAND_SCHEDULE = [
    ("need", "qwen_72b"), ("compute", 120),
    ("need", "uniref90"), ("compute", 300),
    ("need", "qwen_32b"), ("compute", 120),
    ("need", "uniref90"), ("compute", 120),
    ("need", "qwen_72b"), ("compute", 60),
    ("need", "uniref50"),
]
DEFAULT_SCHEDULE = HAND_SCHEDULE     # overridden by --from-traces


def synthetic_schedule(n_needs: int = 10, window_scale: float = 1.0,
                       zipf_a: float = 0.8, seed: int = 0,
                       resources: list[str] | None = None) -> list:
    """A synthetic need sequence with every parameter's provenance stated.

    WHY A SYNTHETIC GENERATOR IS LEGITIMATE HERE. The exp3 traces are a fixed
    pre-scripted experiment: their windows are small because the workflow does
    little real computation between LLM calls. That is a property of THAT
    benchmark, not of agentic workflows generally -- tool-heavy workflows with
    genuine simulation between calls, and longer LLM iteration cycles, produce
    much larger windows. `window_scale` is the knob for asking what happens as
    workflows move in that direction, and it is a SENSITIVITY AXIS to be swept
    and reported, not a number to be picked once and buried.

    WHAT AN EARLIER VERSION OF THIS GOT WRONG, so it is not repeated. The first
    synthetic generator lived inline in a throwaway script -- unversioned and
    uninspectable -- and drew need order UNIFORMLY over the resources and windows
    UNIFORMLY from four round numbers I chose. Measured against 37 real trials
    that is wrong in three ways, all of them flattering to a prefetcher:

      order       real traces show 15 distinct orders across 37 trials with one
                  appearing 9 times -- strongly structured, not uniform. Zipf
                  popularity (zipf_a, default 0.8 as in sweep_policy_regime.py)
                  plus repeat-bias reproduces the clustering.
      windows     real gaps are BIMODAL -- median 16.3 s, p90 964 s -- not four
                  round numbers. Short gaps are the agent thinking; long gaps are
                  real computation. Drawn here as a mixture.
      correlation 43 of 45 real gaps >=300 s follow a MODEL need, because the
                  window IS the agent running inference. Long windows are
                  therefore emitted after model needs, not uniformly.

    Returns one schedule; call repeatedly with different seeds for a population.
    """
    import random as _random
    rng = _random.Random(seed)
    names = resources or list(CATALOGUE)
    weights = [1.0 / ((k + 1) ** zipf_a) for k in range(len(names))]

    sched = []
    prev = None
    for k in range(n_needs):
        # Repeat-bias: real sequences revisit a resource far more often than
        # uniform draws do (qwen_72b appears 3x in the dominant real order).
        if prev is not None and rng.random() < 0.35:
            pick = prev
        else:
            pick = rng.choices(names, weights=weights, k=1)[0]
        sched.append(("need", pick))
        prev = pick
        if k + 1 == n_needs:
            break
        # Bimodal gap. The long mode is emitted mostly after MODEL needs, since
        # in the real traces the long window is the agent doing inference.
        is_model = CATALOGUE[pick]["cls"] == "model"
        long_p = 0.45 if is_model else 0.05
        if rng.random() < long_p:
            gap = rng.lognormvariate(6.4, 0.6)      # ~600 s median, heavy tail
        else:
            gap = rng.lognormvariate(2.6, 1.1)      # ~13 s median, matches 16.3
        sched.append(("compute", round(gap * window_scale, 1)))
    return sched


def schedule_from_traces(pattern: str, data_map: dict | None = None,
                         min_needs: int = 4) -> list[list]:
    """Build schedules from the REALIZED need sequences in exp3 metrics CSVs.

    Preserves what the synthetic generator destroyed: the real order of needs and
    the real idle gap between them. Gaps become ("compute", seconds) steps, so a
    prefetch gets exactly as much room to hide as the workload actually gave it.

    `data_map` renames trace data resources onto catalogue entries (the traces
    carry LAMMPS potentials; the pivot substitutes UniProt databases). That
    substitution is a MODELLING CHOICE and must be reported as one -- the timing
    of the surrounding sequence is real, the identity of the data artifact is not.
    """
    import csv as _csv
    import datetime as _dt
    import glob as _glob

    data_map = data_map or {}
    out = []
    for f in sorted(_glob.glob(pattern)):
        try:
            rows = list(_csv.DictReader(open(f)))
        except OSError:
            continue
        if not rows:
            continue
        try:
            t0 = _dt.datetime.fromisoformat(rows[0]["timestamp"])
        except Exception:
            continue
        seq = []
        for r in rows:
            ph = r["phase"]
            if not (ph.startswith("model_load:") or ph.startswith("lammps:")):
                continue
            try:
                dur = float(r["duration_s"] or 0)
                end = (_dt.datetime.fromisoformat(r["timestamp"]) - t0).total_seconds()
            except Exception:
                continue
            raw = ph.split(":", 1)[1].split("/")[0]
            name = data_map.get(raw, raw)
            if name in CATALOGUE:
                seq.append((end - dur, name, dur))
        if len(seq) < min_needs:
            continue
        seq.sort()
        sched = []
        for k, (start, name, dur) in enumerate(seq):
            sched.append(("need", name))
            if k + 1 < len(seq):
                gap = seq[k + 1][0] - (start + dur)
                if gap > 0.5:
                    sched.append(("compute", round(gap, 1)))
        out.append(sched)
    return out


class Backend:
    """Real resources. Every readiness transition is asserted, never assumed."""

    def __init__(self, simulate: bool, log):
        self.simulate = simulate
        self.log = log
        self.blocks = {}          # data name -> retained block
        self.orch = None
        self.booted = set()       # models with a live server process

    # -- models ---------------------------------------------------------------
    def _ensure_orch(self):
        if self.orch is None and not self.simulate:
            from experiments.model_configs import MODELS_BLACKWELL_SWAP as MODELS
            from atomagents.runtime.model_orchestrator import ModelOrchestrator
            # Sleep mode must be injected BEFORE any engine starts -- neither
            # --enable-sleep-mode nor VLLM_SERVER_DEV_MODE can be turned on after
            # the fact, and vLLM 0.17.x gates /sleep, /wake_up and /is_sleeping
            # behind dev mode. Mirrors atomagents_exp3.py:456-473 deliberately;
            # do not simplify, each clause is load-bearing.
            for mc in MODELS.values():
                extra = list(mc.get("extra_args") or [])
                if "--enable-sleep-mode" not in extra:
                    extra.append("--enable-sleep-mode")
                mc["extra_args"] = extra
                env = dict(mc.get("extra_env") or {})
                env["VLLM_SERVER_DEV_MODE"] = "1"
                # expandable_segments makes /sleep free NOTHING: the allocator
                # cannot release segments it has expanded, so the park silently
                # does not happen and the next engine OOMs.
                if "expandable_segments" in env.get("PYTORCH_CUDA_ALLOC_CONF", ""):
                    env.pop("PYTORCH_CUDA_ALLOC_CONF")
                mc["extra_env"] = env
            self.orch = ModelOrchestrator(MODELS)
        return self.orch

    def make_ready(self, name: str, retained: bool) -> float:
        """Bring `name` to usable. Returns measured seconds."""
        r = CATALOGUE[name]
        t0 = time.perf_counter()
        if self.simulate:
            time.sleep((r["ready_s"] if retained else r["cold_s"]) / self.speedup)
            return time.perf_counter() - t0

        if r["cls"] == "model":
            o = self._ensure_orch()
            if retained and name in self.booted and o.is_sleeping(name):
                o.wake_model(name)
            else:
                if name in self.booted:
                    o.stop_model(name)
                    self.booted.discard(name)
                o.start_model(name)
                o.wait_until_ready(name)
                self.booted.add(name)
            # ASSERT the engine actually serves. A 200 is not evidence -- the L2
            # sleep result returned fast and generated "!!!!" for days.
            if not o._probe_completion(name):
                raise RuntimeError(f"{name} did not produce a coherent completion "
                                   f"after make_ready(retained={retained})")
        else:
            if retained and name in self.blocks:
                pass                      # already at R3: the saving
            else:
                self.blocks[name] = self._load_block(r["path"])
            n = len(self.blocks[name])
            if n <= 0:
                raise RuntimeError(f"{name} retained but holds {n} records")
        return time.perf_counter() - t0

    def _load_block(self, path: str):
        from pyhmmer.easel import SequenceFile
        with SequenceFile(path, digital=True) as sf:
            return sf.read_block()

    def evict(self, name: str) -> None:
        r = CATALOGUE[name]
        if self.simulate:
            return
        if r["cls"] == "model":
            o = self._ensure_orch()
            if name in self.booted:
                o.stop_model(name)
                self.booted.discard(name)
        else:
            self.blocks.pop(name, None)

    def park(self, name: str) -> None:
        """Move a model to R2 (weights in host RAM) rather than killing it."""
        if self.simulate or CATALOGUE[name]["cls"] != "model":
            return
        o = self._ensure_orch()
        if name in self.booted and not o.is_sleeping(name):
            o.sleep_model(name, level=1)

    def shutdown(self):
        if self.simulate or self.orch is None:
            return
        for n in list(self.booted):
            try:
                self.orch.stop_model(n)
            except Exception:
                pass


# --- the scheduler -----------------------------------------------------------
class Arbiter:
    def __init__(self, policy: str, budget_gb: float, prefetch: bool,
                 schedule, backend: Backend, log,
                 predict_accuracy: float = 1.0, seed: int = 0,
                 optimal_selection: bool = False):
        import random as _random
        self.rng = _random.Random(seed)
        self.predict_accuracy = predict_accuracy
        self.optimal_selection = optimal_selection
        self.policy = policy
        self.budget = budget_gb
        self.prefetch = prefetch
        self.schedule = schedule
        self.be = backend
        self.log = log
        self.retained: dict[str, float] = {}    # name -> last-use step index
        self.inflight: dict[str, threading.Thread] = {}
        self.events = []
        self.prefetch_hits = 0
        self.prefetch_wasted = 0
        # True while a model occupies the GPUs, i.e. from the moment a model
        # need is served until it is parked or evicted. A model prefetch during
        # that period cannot happen at tp=4 with M=1.
        self.gpu_busy = False
        # HOW MANY MODELS MAY BE RESIDENT AT ONCE. exp3 has M=1: every model
        # declares gpus=[0,1,2,3] at tp=4, so one model occupies the whole node
        # and a model prefetch during inference is impossible. More GPUs, or
        # smaller models, give M>=2 -- a QUALITATIVE change, because it is the
        # difference between model prefetch being available and forbidden.
        self.gpu_slots = 1
        self._models_resident: set = set()

    # -- the value function ---------------------------------------------------
    def _next_use(self, name: str, i: int) -> float:
        """Schedule POSITION of the next need for `name`. Oracle: reads ahead."""
        for j in range(i + 1, len(self.schedule)):
            k, v = self.schedule[j]
            if k == "need" and v == name:
                return j
        return float("inf")

    def _seconds_between(self, i: int, j: float) -> float:
        """Wall-SECONDS from position i to position j.

        FIXES A UNITS BUG. This used to be `j - i`, a count of schedule
        POSITIONS, while the value function divides by it as though it were
        time held. A one-step gap containing a 566 s hmmsearch scored identically
        to a one-step gap containing a 60 s compute -- so the ranking was wrong
        exactly when compute windows vary, which is the regime this whole
        experiment is about.

        Intervening needs are priced at their COLD cost: we are estimating how
        long the resource must be held, and at ranking time we do not know
        whether those needs will hit or miss.
        """
        if j == float("inf"):
            return float("inf")
        t = 0.0
        for k in range(i + 1, int(j)):
            kind, val = self.schedule[k]
            t += val if kind == "compute" else CATALOGUE[val]["cold_s"]
        return max(t, 1e-6)

    def _hold_seconds(self, name: str, i: int) -> float:
        """How long we believe `name` must be held. ORACLE vs REALIZABLE split.

        The oracle variants read the true next use out of the schedule. That is
        Belady-grade information and NOT something a deployed scheduler has --
        an earlier version of this file gave it to `value_density` and to `lru`
        alike, which made both stronger than reality and made their comparison
        meaningless.

        The realizable variant asks a predictor that is right `predict_accuracy`
        of the time; when wrong it returns some other resource's horizon, which
        is what a confusion between resources actually looks like. This is the
        knob that tests the project's standing claim -- 45-62% accuracy was
        disqualifying for PREFETCHING but might be adequate for RETENTION,
        because a retention decision is made at eviction time when the past is
        already known.
        """
        true_s = self._seconds_between(i, self._next_use(name, i))
        if not self.policy.endswith("_pred"):
            return true_s
        if self.rng.random() < self.predict_accuracy:
            return true_s
        others = [self._seconds_between(i, self._next_use(o, i))
                  for o in CATALOGUE if o != name]
        others = [o for o in others if o != float("inf")]
        return self.rng.choice(others) if others else true_s

    def _rank(self, name: str, i: int) -> float:
        """Lower rank is evicted first."""
        # REAL LRU SEES ONLY THE PAST. It gets no "this is never used again"
        # shortcut -- that is future knowledge, and handing it to the baseline
        # was the other half of the contaminated comparison.
        if self.policy == "lru":
            return self.retained.get(name, -1)

        if self.policy == "belady":
            nu = self._next_use(name, i)
            return float("-inf") if nu == float("inf") else -nu

        if self.policy in ("value_density", "value_density_pred"):
            held = self._hold_seconds(name, i)
            if held == float("inf"):
                return float("-inf")      # believed dead -> evict first
            r = CATALOGUE[name]
            saved = r["cold_s"] - r["ready_s"]
            return saved / (r["held_gb"] * held)

        raise ValueError(self.policy)

    # -- ONE CURRENCY FOR BOTH ACTIONS ---------------------------------------
    def action_rate(self, kind: str, name: str, i: int) -> float:
        """Stall-seconds avoided per GB-second of budget consumed.

        THIS IS THE UNIFICATION. Retaining and prefetching produce the SAME
        good -- a resource ready at its next need -- and differ only in their
        cost profile, so they can and should be ranked against each other in one
        number instead of being two mechanisms that merely share a budget.

            RETAIN x    benefit = cold(x) - ready(x)      the whole rebuild avoided
                        cost    = size(x) * dt            held from now to next use

            PREFETCH y  benefit = min(load(y), dt)        can only hide as much as
                                                          the window actually allows
                        cost    = size(y) * dt            occupied from now to need

        The `min(load, dt)` term is what makes this honest about the measured
        workload: real gaps between needs are median 6.5 s against 440-765 s cold
        boots, so for models the benefit term collapses to dt and prefetching a
        model is nearly worthless -- the value function now SAYS that rather than
        our having to remember it.
        """
        r = CATALOGUE[name]
        dt = self._hold_seconds(name, i)
        if dt == float("inf"):
            return float("-inf")
        cost = r["held_gb"] * dt
        if kind == "retain":
            benefit = r["cold_s"] - r["ready_s"]
        elif kind == "prefetch":
            benefit = min(r["cold_s"] - r["ready_s"], dt)
        else:
            raise ValueError(kind)
        return benefit / max(cost, 1e-9)

    def _held(self, extra: str | None = None) -> float:
        names = set(self.retained) | ({extra} if extra else set())
        return sum(CATALOGUE[n]["held_gb"] for n in names)

    def _best_set(self, candidates: dict, i: int) -> set:
        """The highest-value subset that fits. EXACT for small catalogues.

        Greedy single-victim eviction solves the wrong problem: it can only evict
        WHOLE items, so it frees 117-130 GB to make room for 36 GB, and it admits
        the newcomer unconditionally. On our own resources that costs 32.7% of
        retainable value in a concrete case -- optimal holds {qwen_32b, uniref90}
        = 811.8 s, while greedy's best after admitting uniref50 is 546.3 s.

        With a handful of resources the subset problem is 2^n and simply solvable,
        so the approximation is not worth its error here. `optimal_selection=False`
        keeps the greedy behaviour for comparison.

        *** OBJECTIVE BUG, FOUND AND FIXED 2026-08-07. *** This used to score a
        combination as

            sum( action_rate(x) * held_gb(x) * hold_seconds(x) )

        and `action_rate` is `benefit / (held_gb * hold_seconds)` -- so the
        denominator cancelled *exactly* and the objective collapsed to
        `sum(cold_s - ready_s)`. Verified numerically: {qwen_32b, uniref50} and
        {qwen_32b, uniref90} both scored 494.17, identical to the bare sum of
        (cold - ready). The selector was therefore BLIND TO WHEN THE REUSE
        HAPPENS: a resource needed again in 10 s and one needed again in 10,000 s
        were interchangeable, and the time term the whole value function exists to
        express was multiplied straight back out. The tell was non-monotonicity --
        the arm beat LRU by 18.16% on 24-need schedules and LOST to it by 0.81% on
        48-need ones, which no genuinely better selector can do.

        THE FIX. Hold the budget constraint in GB (that is what the cgroup limits)
        and score value per unit of TIME the space is tied up:

            value  = benefit / hold_seconds        weight = held_gb

        This also makes the exact solver consistent with the greedy one, whose
        ranking `benefit / (held_gb * hold_seconds)` is precisely value/weight for
        this pair -- i.e. the fractional-knapsack density of the same problem. The
        two arms now differ only in 0/1-vs-fractional, which is the comparison the
        arm was built to make.

        STILL NOT A BOUND. This is an exact solve of a ONE-STEP surrogate, not an
        offline optimum over the whole schedule. Do not label it "oracle."
        """
        import itertools
        names = sorted(candidates)
        if not self.optimal_selection or len(names) > 16:
            return None                      # caller falls back to greedy
        best, best_v = set(), -1.0
        for r in range(len(names), -1, -1):
            for combo in itertools.combinations(names, r):
                if sum(CATALOGUE[x]["held_gb"] for x in combo) > self.budget:
                    continue
                v = 0.0
                for x in combo:
                    dt = self._hold_seconds(x, i)
                    if dt == float("inf"):
                        continue             # never used again: worth nothing
                    r_ = CATALOGUE[x]
                    v += (r_["cold_s"] - r_["ready_s"]) / max(dt, 1e-9)
                if v > best_v:
                    best, best_v = set(combo), v
        return best

    def _make_room_for(self, name: str, i: int) -> bool:
        """Evict until `name` fits. False if it cannot fit even alone."""
        if CATALOGUE[name]["held_gb"] > self.budget:
            return False
        while self._held(name) > self.budget:
            victims = sorted(x for x in self.retained if x != name)
            if not victims:
                return False
            v = min(victims, key=lambda x: (self._rank(x, i), x))
            self.be.evict(v)
            del self.retained[v]
            self.events.append({"step": i, "op": "evict", "resource": v})
        return True

    # -- prefetch -------------------------------------------------------------
    def _predicted_next(self, i: int) -> str | None:
        """What will be needed after step i. Perfect lookahead by default; this
        is the CEILING for prefetching, and --predict-accuracy degrades it."""
        for j in range(i + 1, len(self.schedule)):
            k, v = self.schedule[j]
            if k == "need":
                return v if v not in self.retained else None
        return None

    def _start_prefetch(self, name: str, i: int) -> None:
        if name in self.inflight or name in self.retained:
            return
        # A prefetch competes with the RETAINED set in the same currency. If
        # holding what is already there is worth more per GB-second, do not
        # displace it -- the two actions now trade off inside one value
        # function rather than merely sharing a budget.
        # GPUs ARE NOT THE BUDGET, AND A MODEL PREFETCH NEEDS THEM.
        # Every model declares gpus=[0,1,2,3] at tp=4, so M=1: only one model can
        # be resident on the GPUs at a time. The windows we prefetch into are
        # overwhelmingly the agent doing 72B inference (43 of 45 gaps >=300 s
        # follow a qwen_72b need), and during that window the GPUs are BUSY --
        # loading another model onto them is physically impossible, not merely
        # expensive. Data is unaffected: it lands in host RAM and needs no GPU.
        #
        # At 256 GB this never fired because a 279 GB model cannot fit the budget
        # anyway, so the right behaviour emerged by accident. It would not at
        # larger budgets, and an accident is not a constraint.
        if CATALOGUE[name]["cls"] == "model" and self.gpu_busy:
            self.events.append({"step": i, "op": "prefetch_impossible_gpus_busy",
                                "resource": name})
            return
        rate = self.action_rate("prefetch", name, i)
        rivals = [self.action_rate("retain", x, i) for x in self.retained]
        need_gb = CATALOGUE[name]["held_gb"]
        if rivals and self._held(name) > self.budget and rate <= min(rivals):
            self.events.append({"step": i, "op": "prefetch_declined_low_value",
                                "resource": name, "rate": round(rate, 6)})
            return
        if not self._make_room_for(name, i):
            self.events.append({"step": i, "op": "prefetch_skipped_no_room",
                                "resource": name})
            return

        def work():
            try:
                self.be.make_ready(name, retained=False)
                self.retained[name] = i
            except Exception as e:
                self.log(f"    prefetch of {name} FAILED: {e}")

        t = threading.Thread(target=work, daemon=True)
        self.inflight[name] = t
        self.events.append({"step": i, "op": "prefetch_start", "resource": name})
        t.start()

    def _join_prefetch(self, name: str | None = None) -> None:
        for n, t in list(self.inflight.items()):
            if name is None or n == name:
                t.join()
                self.inflight.pop(n, None)

    # -- analytic run: simulated clock, no sleeping ---------------------------
    def run_analytic(self) -> dict:
        """Stall seconds on a SIMULATED clock, honouring how long loads take.

        THE DRY-RUN ACCOUNTING WAS FICTITIOUS AND OVERSTATED PREFETCH. In the
        thread-based path a stub backend returns instantly, so `_join_prefetch`
        never waits and EVERY prefetch counted as a hit -- even a 372.6 s
        UniRef90 load issued into a 5 s window. The tell was a window_scale sweep
        that returned identically +24.6% at 0.5x, 1x, 2x, 4x and 8x: if window
        size genuinely did not matter, prefetching would be free, which it is not.

        Here a prefetch issued at simulated time t completes at t + cold(y). If
        the need arrives first, only the elapsed portion was hidden and the
        remainder is still paid. That is the whole point of a window.
        """
        clock = 0.0
        stall = 0.0
        inflight: dict[str, float] = {}       # name -> completion time
        hits = {"retain": 0, "prefetch": 0}
        wasted = 0

        for i, (kind, val) in enumerate(self.schedule):
            if kind == "compute":
                if self.prefetch:
                    nxt = self._predicted_next(i)
                    if nxt and nxt not in inflight and nxt not in self.retained:
                        _needs_slot = CATALOGUE[nxt]["cls"] == "model"
                        if not (_needs_slot and self.gpu_busy):
                            cand = dict(self.retained); cand[nxt] = i
                            if sum(CATALOGUE[x]["held_gb"] for x in cand) <= self.budget:
                                inflight[nxt] = clock + CATALOGUE[nxt]["cold_s"]
                clock += val
                continue

            name = val
            r = CATALOGUE[name]
            if name in self.retained:
                stall += r["ready_s"]; clock += r["ready_s"]
                hits["retain"] += 1
                del self.retained[name]
            elif name in inflight:
                remaining = max(0.0, inflight.pop(name) - clock)
                # Fully hidden only if the load finished before the need.
                cost = remaining if remaining > 0 else r["ready_s"]
                if remaining <= 0:
                    hits["prefetch"] += 1
                stall += cost; clock += cost
            else:
                stall += r["cold_s"]; clock += r["cold_s"]

            if r["cls"] == "model":
                self._models_resident.add(name)
                self.gpu_busy = len(self._models_resident) >= self.gpu_slots
            if self.policy != "never_retain":
                cand = dict(self.retained); cand[name] = i
                keep = self._best_set(cand, i)
                if keep is None:
                    keep = set(cand)
                    while sum(CATALOGUE[x]["held_gb"] for x in keep) > self.budget:
                        # TIE-BREAK BY NAME, DETERMINISTICALLY. Iterating a set
                        # of strings depends on hash randomisation, which varies
                        # PER PROCESS -- so equal-ranked victims were chosen
                        # differently every run and the same computation returned
                        # 95404 / 90952 / 112506 stall seconds on three
                        # consecutive invocations. A 23% spread from PYTHONHASHSEED.
                        vic = sorted(x for x in keep if x != name)
                        if not vic:
                            keep.discard(name); break
                        keep.discard(min(vic, key=lambda x: (self._rank(x, i), x)))
                # PRESERVE EACH RESOURCE'S OWN LAST-USE STEP. Re-stamping every
                # retained item with the current step `i` -- which this line used
                # to do -- gives them all identical timestamps, so _rank() returns
                # the same value for every candidate, LRU has NO recency
                # information at all, and the alphabetical tie-break silently
                # becomes the eviction policy. The baseline was "evict whichever
                # name sorts first", not LRU, for every comparison this harness
                # has produced. Only the resource just needed gets a new stamp.
                _prev = dict(self.retained)
                self.retained = {x: (i if x == name else _prev.get(x, i))
                                 for x in sorted(keep)}
                if name in keep and r["cls"] == "model":
                    # Parking frees this model's GPUs; occupancy drops.
                    self._models_resident.discard(name)
                    self.gpu_busy = len(self._models_resident) >= self.gpu_slots
            # anything still in flight that will never be used is waste
        wasted = len(inflight)
        return {"stall_s": round(stall, 2), "sim_wall_s": round(clock, 2),
                "retained_hits": hits["retain"], "prefetch_hits": hits["prefetch"],
                "prefetch_unused": wasted}

    # -- run ------------------------------------------------------------------
    def run(self) -> dict:
        t_start = time.perf_counter()
        stall_total = 0.0
        for i, (kind, val) in enumerate(self.schedule):
            if kind == "compute":
                # The window. A prefetch issued here genuinely overlaps because
                # it runs on another thread while we sleep out the compute.
                if self.prefetch:
                    nxt = self._predicted_next(i)
                    if nxt:
                        self._start_prefetch(nxt, i)
                time.sleep(val / self.be.speedup)
                self.log(f"  [{i}] compute {val}s")
                continue

            name = val
            self._join_prefetch(name)          # if it is in flight, wait for it
            was_retained = name in self.retained
            if was_retained and self.policy != "never_retain":
                self.prefetch_hits += 1

            t0 = time.perf_counter()
            elapsed = self.be.make_ready(name, retained=was_retained)
            stall = time.perf_counter() - t0
            stall_total += stall
            self.log(f"  [{i}] need {name:10s} "
                     f"{'READY' if was_retained else 'COLD ':5s} {stall:8.2f}s")
            self.events.append({"step": i, "op": "need", "resource": name,
                                "retained": was_retained, "stall_s": round(stall, 3)})

            if CATALOGUE[name]["cls"] == "model":
                self.gpu_busy = True

            if self.policy == "never_retain":
                self.be.evict(name)
                self.retained.pop(name, None)
                if CATALOGUE[name]["cls"] == "model":
                    self.gpu_busy = False
            else:
                cand = dict(self.retained)
                cand[name] = i
                keep = self._best_set(cand, i)
                if keep is not None:
                    # EXACT selection: choose the best subset, which may decline
                    # to admit the newcomer at all. Greedy cannot express that.
                    for x in list(self.retained):
                        if x not in keep:
                            self.be.evict(x); del self.retained[x]
                    if name in keep:
                        self.retained[name] = i
                        self.be.park(name)     # parking frees the GPUs
                        if CATALOGUE[name]["cls"] == "model":
                            self.gpu_busy = False
                    else:
                        self.be.evict(name)
                        self.retained.pop(name, None)
                elif self._make_room_for(name, i):
                    self.retained[name] = i
                    self.be.park(name)
                else:
                    self.be.evict(name)
                    self.retained.pop(name, None)

        self._join_prefetch()
        return {"wall_s": round(time.perf_counter() - t_start, 3),
                "stall_s": round(stall_total, 3),
                "retained_hits": self.prefetch_hits,
                "final_retained": list(self.retained),
                "events": self.events}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="no GPU/data; measured constants + sleep. Validates the "
                         "scheduler, NOT a speedup measurement.")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="simulate-mode time compression (e.g. 50 = 50x faster)")
    ap.add_argument("--budgets", default="150,256,400,560")
    ap.add_argument("--arms",
                    default="never_retain,retain_lru,retain_vd_pred,"
                            "retain_vd_pred_prefetch,retain_belady,retain_vd_oracle")
    ap.add_argument("--predict-accuracy", type=float, default=0.55,
                    help="accuracy of the retention predictor for the *_pred "
                         "arms. Default 0.55 is the middle of this project's "
                         "measured 45-62%% predictor accuracy. 1.0 makes them "
                         "oracle-equivalent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_arbitration_harness.json")
    args = ap.parse_args()

    # ORACLE arms read the true next use out of the schedule and are BOUNDS,
    # not systems. REALIZABLE arms see only the past, or a predictor.
    ARMS = {
        "never_retain":        ("never_retain",         False),   # baseline
        "retain_lru":          ("lru",                  False),   # realizable baseline
        "retain_vd_pred":      ("value_density_pred",   False),   # REALIZABLE system
        "retain_vd_pred_prefetch": ("value_density_pred", True),  # REALIZABLE, both actions
        "retain_belady":       ("belady",               False),   # ORACLE bound (hit count)
        "retain_vd_oracle":    ("value_density",        False),   # ORACLE bound (cost aware)
    }

    print("resource catalogue (s/GB retained is the ranking key):")
    for n, r in CATALOGUE.items():
        print(f"  {n:10s} {r['cls']:6s} held {r['held_gb']:7.2f} GB  "
              f"cold {r['cold_s']:7.1f}s  ready {r['ready_s']:5.2f}s  "
              f"s/GB {s_per_gb(n):5.2f}")
    print(f"\nmode: {'SIMULATE' if args.simulate else 'REAL'}"
          f"{f' (x{args.speedup} time compression)' if args.simulate else ''}")

    out = {"simulate": args.simulate, "schedule": DEFAULT_SCHEDULE, "runs": []}
    if not args.simulate:
        print(f"cgroup limit: {cgroup_limit_gb():.0f} GB")
    for budget in [float(b) for b in args.budgets.split(",")]:
        assert_budget_is_safe(budget, args.simulate)
        print(f"\n=== budget {budget:.0f} GB ===")
        for arm in args.arms.split(","):
            policy, pf = ARMS[arm]
            be = Backend(args.simulate, print)
            be.speedup = args.speedup if args.simulate else 1.0
            arb = Arbiter(policy, budget, pf, DEFAULT_SCHEDULE, be, print,
                          predict_accuracy=args.predict_accuracy, seed=args.seed)
            print(f"  -- {arm}")
            try:
                res = arb.run()
            finally:
                be.shutdown()
            res.update(arm=arm, budget_gb=budget)
            out["runs"].append(res)
            print(f"     wall {res['wall_s']:9.2f}s   stall {res['stall_s']:9.2f}s"
                  f"   hits {res['retained_hits']}")

    # summary
    print(f"\n{'budget':>8} " + " ".join(f"{a:>20}" for a in args.arms.split(",")))
    for budget in [float(b) for b in args.budgets.split(",")]:
        row = []
        for arm in args.arms.split(","):
            r = [x for x in out["runs"]
                 if x["budget_gb"] == budget and x["arm"] == arm]
            row.append(f"{r[0]['wall_s']:>20.1f}" if r else f"{'-':>20}")
        print(f"{budget:>8.0f} " + " ".join(row))

    base = {b: next((x["wall_s"] for x in out["runs"]
                     if x["budget_gb"] == b and x["arm"] == "never_retain"), None)
            for b in [float(b) for b in args.budgets.split(",")]}
    print("\nspeedup vs never_retain:")
    for budget, bw in base.items():
        if not bw:
            continue
        parts = []
        for arm in args.arms.split(","):
            if arm == "never_retain":
                continue
            r = [x for x in out["runs"]
                 if x["budget_gb"] == budget and x["arm"] == arm]
            if r:
                parts.append(f"{arm} {bw / r[0]['wall_s']:.3f}x")
        print(f"  {budget:>6.0f} GB: " + "   ".join(parts))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
