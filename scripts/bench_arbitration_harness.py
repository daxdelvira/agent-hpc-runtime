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
    "qwen_32b":      dict(cls="model", held_gb=129.7, cold_s=440.2, ready_s=1.03),
    "qwen_72b":      dict(cls="model", held_gb=279.0, cold_s=765.3, ready_s=2.21),
    "uniref50":      dict(cls="data",  held_gb=36.08, cold_s=107.1, ready_s=0.0,
                          path="/storage/scratch1/7/avandevoorde3/p1/data/uniref50.fasta"),
    "uniref90":      dict(cls="data",  held_gb=117.20, cold_s=372.6, ready_s=0.0,
                          path="/storage/scratch1/7/avandevoorde3/p1/data/uniref90.fasta"),
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


# --- the schedule ------------------------------------------------------------
# ("need", resource) | ("compute", seconds)
DEFAULT_SCHEDULE = [
    ("need", "qwen_72b"), ("compute", 120),
    ("need", "uniref90"), ("compute", 300),
    ("need", "qwen_32b"), ("compute", 120),
    ("need", "uniref90"), ("compute", 120),
    ("need", "qwen_72b"), ("compute", 60),
    ("need", "uniref50"),
]


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
                 predict_accuracy: float = 1.0, seed: int = 0):
        import random as _random
        self.rng = _random.Random(seed)
        self.predict_accuracy = predict_accuracy
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

    def _held(self, extra: str | None = None) -> float:
        names = set(self.retained) | ({extra} if extra else set())
        return sum(CATALOGUE[n]["held_gb"] for n in names)

    def _make_room_for(self, name: str, i: int) -> bool:
        """Evict until `name` fits. False if it cannot fit even alone."""
        if CATALOGUE[name]["held_gb"] > self.budget:
            return False
        while self._held(name) > self.budget:
            victims = [x for x in self.retained if x != name]
            if not victims:
                return False
            v = min(victims, key=lambda x: self._rank(x, i))
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

            if self.policy == "never_retain":
                self.be.evict(name)
                self.retained.pop(name, None)
            else:
                if self._make_room_for(name, i):
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
