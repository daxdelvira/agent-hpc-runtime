#!/usr/bin/env python3
"""
replay_two_class.py — arbitrate MODELS and DATA against ONE host-RAM budget.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
E5 replayed model needs alone and found value_density tied LRU and Belady
EXACTLY at every budget: three models, room for one, no decision to make. The
plan's response was that the interesting regime is MIXED -- models and data
competing for the same RAM -- because that is where the measured 65x spread in
seconds-per-GB actually appears. This script is that replay.

Both mechanisms are now measured, not assumed:

  MODEL  park at R2 (vLLM L1 sleep). 68.28 GB of weights -> 120.77 GiB parked
         (1.90x), wake 2.076 s against a 782 s cold boot. First park costs
         23.692 s; later ones 2.6 s.
  DATA   retain the activated potential at R3 in a live LAMMPS worker. 3.32 GB
         file -> 16.93 GB activated (5.10x). Verified on the real potential:
         a redundant invocation drops 93.73 s -> 10.56 s (8.88x), physics
         bit-identical to the fork path (pe rel diff 0.000e+00).

  s/GB retained:  model ~2.96      activated potential ~2.53
Those are within 1.2x of each other, which is exactly what makes this an
arbitration problem rather than a fixed priority.

⚠️ THE DATA NEEDS ARE PARTLY RECONSTRUCTED, AND THAT IS NOT HIDEABLE
--------------------------------------------------------------------
Real trials issue THREE LAMMPS invocations per potential (lattice_constant,
screw_initial, relax_screw). Until 2026-08-05 only lattice_constant was wrapped
in a metrics phase, so metrics.csv records ONE of the three -- a 3x undercount.
The instrumentation is now fixed (physics/screw_dislocation.py), but every trace
on disk predates the fix.

So this script runs in two modes and reports BOTH:

  --data-mode instrumented   use only the lammps: phases actually recorded.
                             Faithful to the traces, and known to undercount.
  --data-mode reconstructed  add the two missing invocations per observed
                             potential, priced from the VERIFIED per-invocation
                             measurement. Faithful to the workload, but the
                             timings of the added needs are modelled.

Neither is "the" answer. If they disagree, the honest report is that the result
depends on an instrumentation gap and needs re-collection to settle -- which is
a finding about our evidence, not about the policy.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics

# --- measured constants ------------------------------------------------------
# Models (E4, results/bench_wake_L1_coherence_32b.json)
WAKE_BW_GB_S_PER_GPU = 16.6
PARK_RATIO = 1.90
FIRST_PARK_S = 23.692
MODEL_BYTES = {"qwen_32b": 68.28, "qwen_72b": 146.82, "qwen_72b_text": 145.41}
MODEL_GPUS = 4

# Data (results/verify_persistent_lammps_BIG.json + bench_activated_residency_BIG.json)
# The big potential: activated size and the cost of an invocation with and
# without a retained worker. Small potentials activate too cheaply to matter and
# are priced from their observed durations.
BIG_POTENTIAL_ACTIVATED_GB = 16.93
BIG_INVOCATION_COLD_S = 93.73      # fork path, parse included
BIG_INVOCATION_RETAINED_S = 10.56  # retained worker, parse skipped
SMALL_POTENTIAL_ACTIVATED_GB = 0.05
INVOCATIONS_PER_POTENTIAL = 3      # lattice_constant, screw_initial, relax_screw


def model_park_gb(m: str) -> float:
    return MODEL_BYTES[m] * PARK_RATIO


def model_wake_s(m: str) -> float:
    return MODEL_BYTES[m] / (WAKE_BW_GB_S_PER_GPU * MODEL_GPUS)


def _is_big(pot: str) -> bool:
    return "big" in pot.lower()


def data_size_gb(pot: str) -> float:
    return (BIG_POTENTIAL_ACTIVATED_GB if _is_big(pot)
            else SMALL_POTENTIAL_ACTIVATED_GB)


# --- trace loading -----------------------------------------------------------
def load_trials(pattern: str, data_mode: str) -> tuple[list[dict], dict, dict]:
    trials = []
    boots = collections.defaultdict(list)
    lam_obs = collections.defaultdict(list)
    import datetime as dt

    for f in sorted(glob.glob(pattern)):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        try:
            t0 = dt.datetime.fromisoformat(rows[0]["timestamp"])
        except Exception:
            continue
        needs = []
        for r in rows:
            ph = r["phase"]
            try:
                dur = float(r["duration_s"] or 0)
                end = (dt.datetime.fromisoformat(r["timestamp"]) - t0).total_seconds()
            except Exception:
                continue
            start = end - dur
            if ph.startswith("model_load:"):
                name = ph.split(":", 1)[1]
                if name in MODEL_BYTES:
                    needs.append({"t": start, "cls": "model", "id": name,
                                  "cold_s": dur})
                    boots[name].append(dur)
            elif ph.startswith("lammps:"):
                pot = ph.split("/")[-1]
                needs.append({"t": start, "cls": "data", "id": pot,
                              "cold_s": dur})
                lam_obs[pot].append(dur)
        if not needs:
            continue

        if data_mode == "reconstructed":
            # Each observed potential really got INVOCATIONS_PER_POTENTIAL calls.
            # Insert the missing ones just after the observed one, priced from
            # the verified measurement for the big potential and from the
            # observed duration for the small ones.
            extra = []
            for nd in [n for n in needs if n["cls"] == "data"]:
                cold = (BIG_INVOCATION_COLD_S if _is_big(nd["id"])
                        else nd["cold_s"])
                for k in range(1, INVOCATIONS_PER_POTENTIAL):
                    extra.append({"t": nd["t"] + 0.001 * k, "cls": "data",
                                  "id": nd["id"], "cold_s": cold,
                                  "reconstructed": True})
            needs.extend(extra)

        needs.sort(key=lambda n: n["t"])
        trials.append({"file": os.path.basename(f), "needs": needs})

    boot_med = {k: statistics.median(v) for k, v in boots.items()}
    lam_med = {k: statistics.median(v) for k, v in lam_obs.items()}
    return trials, boot_med, lam_med


# --- the replay --------------------------------------------------------------
def replay(needs: list[dict], boot: dict, budget_gb: float, policy: str) -> dict:
    """Total stall seconds under `policy` with `budget_gb` of retention RAM.

    ONE budget, TWO classes. A parked model and a retained potential are ranked
    by the same function and evict each other.
    """
    retained: dict[tuple, float] = {}       # (cls,id) -> last-use time
    ever_parked: set = set()
    running_model = None
    stall = 0.0
    hits = collections.Counter()
    served = collections.Counter()

    def size_of(key) -> float:
        cls, rid = key
        return model_park_gb(rid) if cls == "model" else data_size_gb(rid)

    def retained_cost(key, nd) -> float:
        cls, rid = key
        if cls == "model":
            return model_wake_s(rid)
        return (BIG_INVOCATION_RETAINED_S if _is_big(rid)
                else min(nd["cold_s"], BIG_INVOCATION_RETAINED_S))

    def cold_cost(key, nd) -> float:
        cls, rid = key
        if cls == "model":
            return boot.get(rid, nd["cold_s"])
        return nd["cold_s"]

    def next_use(key, i) -> float:
        for j in range(i + 1, len(needs)):
            if (needs[j]["cls"], needs[j]["id"]) == key:
                return needs[j]["t"]
        return float("inf")

    for i, nd in enumerate(needs):
        key = (nd["cls"], nd["id"])
        t = nd["t"]
        served[nd["cls"]] += 1

        if nd["cls"] == "model" and nd["id"] == running_model:
            pass                                  # already resident on the GPUs
        elif key in retained:
            stall += retained_cost(key, nd)
            hits[nd["cls"]] += 1
            if nd["cls"] == "model":
                del retained[key]                 # woken: leaves host RAM
        elif nd["cls"] == "model" and False:
            pass                                  # already resident on the GPUs
        else:
            stall += cold_cost(key, nd)

        # --- what actually occupies the host-RAM budget ---------------------
        # THE TWO CLASSES ARE ASYMMETRIC HERE, and collapsing them was a bug
        # that manufactured a false 14% win for value_density.
        #
        #   MODEL  while it is the running model its weights are on the GPUs.
        #          It costs host RAM only once DISPLACED and parked at R2. So a
        #          model becomes a retention candidate on displacement, never
        #          while running. Counting the running model against the budget
        #          shrinks the budget and invents eviction decisions that do not
        #          exist -- which is exactly what made this script disagree with
        #          replay_retention_policy.py (E5), where all policies tie.
        #   DATA   the activated potential lives in the worker's host RAM
        #          continuously, in use or not. It occupies budget immediately
        #          and keeps occupying it until evicted.
        cand = dict(retained)
        if nd["cls"] == "data":
            cand[key] = t
        if running_model is not None and running_model != nd.get("id"):
            cand[("model", running_model)] = t      # displaced -> park candidate

        def rank(x) -> float:
            nu = next_use(x, i)
            if nu == float("inf"):
                return float("-inf")              # dead: always first victim
            if policy == "lru":
                return cand[x]
            if policy == "belady":
                return -nu
            if policy == "value_density":
                # saved per GB per second held -- the only policy that can see
                # that a 16.93 GB potential and a 279 GB model are different
                # propositions per byte.
                nd_x = {"cold_s": (BIG_INVOCATION_COLD_S if _is_big(x[1])
                                   else boot.get(x[1], 1.0))}
                saved = cold_cost(x, nd_x) - retained_cost(x, nd_x)
                held = max(nu - t, 1e-6)
                return saved / (size_of(x) * held)
            raise ValueError(policy)

        if policy != "never_retain":
            while sum(size_of(x) for x in cand) > budget_gb and cand:
                cand.pop(min(cand, key=rank))
            for x in cand:
                if x[0] == "model" and x not in ever_parked:
                    stall += FIRST_PARK_S
                    ever_parked.add(x)
            retained = cand
        if nd["cls"] == "model":
            running_model = nd["id"]

    return {"stall_s": stall,
            "hits_model": hits["model"], "hits_data": hits["data"],
            "served_model": served["model"], "served_data": served["data"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern",
                    default="results/atomagents_metrics_eval_atomagents_exp3_aligned_*.csv")
    ap.add_argument("--budgets", default="0,130,180,280,340,420,560,700")
    ap.add_argument("--data-mode", choices=("instrumented", "reconstructed", "both"),
                    default="both")
    ap.add_argument("--out", default="results/replay_two_class.json")
    args = ap.parse_args()

    modes = (["instrumented", "reconstructed"] if args.data_mode == "both"
             else [args.data_mode])
    budgets = [float(b) for b in args.budgets.split(",")]
    policies = ["never_retain", "lru", "belady", "value_density"]
    out = {"modes": {}}

    for mode in modes:
        trials, boot, lam = load_trials(args.pattern, mode)
        nd_tot = sum(len(t["needs"]) for t in trials)
        n_data = sum(1 for t in trials for n in t["needs"] if n["cls"] == "data")
        print(f"\n=== data-mode: {mode} ===")
        print(f"{len(trials)} trials, {nd_tot} needs "
              f"({nd_tot - n_data} model, {n_data} data)")
        if n_data == 0:
            print("  NO DATA NEEDS -> this reduces to the model-only replay (E5).")
        print(f"  observed potentials: "
              f"{ {k: round(v,1) for k,v in lam.items()} }")
        print(f"\n{'budget_GB':>10} " + " ".join(f"{p:>14}" for p in policies))
        rows = []
        for b in budgets:
            vals = {}
            for p in policies:
                s = sum(replay(t["needs"], boot, b, p)["stall_s"] for t in trials)
                vals[p] = s
            rows.append({"budget_gb": b, **vals})
            print(f"{b:>10.0f} " + " ".join(f"{vals[p]:>14.0f}" for p in policies))

        base = rows[0]["never_retain"]
        best = min(r["value_density"] for r in rows)
        vd_vs_lru = [(r["budget_gb"], 100 * (r["lru"] - r["value_density"]) / r["lru"])
                     for r in rows if r["lru"] > 0]
        vd_vs_bel = [(r["budget_gb"], 100 * (r["belady"] - r["value_density"]) / r["belady"])
                     for r in rows if r["belady"] > 0]
        print(f"\n  never_retain {base:.0f} s -> best value_density {best:.0f} s "
              f"= {100*(base-best)/base:.1f}% reduction")
        worst_lru = max(vd_vs_lru, key=lambda x: abs(x[1])) if vd_vs_lru else (0, 0)
        worst_bel = max(vd_vs_bel, key=lambda x: abs(x[1])) if vd_vs_bel else (0, 0)
        print(f"  largest VD-vs-LRU gap    {worst_lru[1]:+.2f}% at {worst_lru[0]:.0f} GB")
        print(f"  largest VD-vs-Belady gap {worst_bel[1]:+.2f}% at {worst_bel[0]:.0f} GB")
        if abs(worst_lru[1]) < 0.05 and abs(worst_bel[1]) < 0.05:
            print("  => value_density is INDISTINGUISHABLE from LRU and Belady here.")
        out["modes"][mode] = {"n_trials": len(trials), "n_needs": nd_tot,
                              "n_data_needs": n_data, "rows": rows,
                              "observed_potentials": lam}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
