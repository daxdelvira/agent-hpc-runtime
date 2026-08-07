#!/usr/bin/env python3
"""
analyze_misses.py — when does LRU actually fail, and is there anything to win?

WHY THIS EXISTS
---------------
Cost-aware retention does not beat LRU on our workload; at realistic predictor
accuracy it loses 11-13%. "The policy does not help" is a conclusion, not an
explanation. This classifies every MISS -- every point where a resource was
needed and was not ready -- so we can say which misses ANY policy could have
prevented, and therefore how much room a better policy ever had.

THE TAXONOMY (adapted from the classic compulsory/capacity/conflict split)

  COMPULSORY  first use of this resource in the schedule. Nothing retained can
              help; only a PREFETCH can, and only if a window precedes it.
  CAPACITY    the resource does not fit the budget even alone. Unreachable for
              retention at this budget, whatever the ranking.
  REPLACEMENT the resource was retained earlier and evicted before its reuse.
              *** THIS IS THE ONLY CATEGORY A BETTER RANKING CAN CONVERT. ***

The headline is not the miss count but the share of STALL SECONDS sitting in the
replacement bucket. If that share is small, no ranking function can matter, and
reporting a policy contribution would be reporting noise.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "harness", os.path.join(HERE, "bench_arbitration_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


class Dry:
    speedup = 1e12
    def make_ready(self, n, retained=False): return 0.0
    def evict(self, n): pass
    def park(self, n): pass
    def shutdown(self): pass


def classify(policy: str, budget: float, scheds: list, accuracy: float = 0.55):
    """Replay and label every miss. Mirrors run_analytic's decisions exactly."""
    C = H.CATALOGUE
    misses = []
    hits = 0
    needs = 0
    for si, s in enumerate(scheds):
        arb = H.Arbiter(policy, budget, False, s, Dry(), lambda *_: None,
                        predict_accuracy=accuracy, seed=si)
        seen: set = set()
        ever_retained: set = set()
        clock = 0.0
        last_use: dict = {}
        for i, (kind, val) in enumerate(s):
            if kind == "compute":
                clock += val
                continue
            name = val
            r = C[name]
            needs += 1
            if name in arb.retained:
                hits += 1
            else:
                if name not in seen:
                    cat = "compulsory"
                elif r["held_gb"] > budget:
                    cat = "capacity"
                elif name in ever_retained:
                    cat = "replacement"       # we held it and let it go
                else:
                    cat = "never_admitted"    # fit, seen, but never retained
                misses.append(dict(schedule=si, step=i, resource=name,
                                   cls=r["cls"], held_gb=r["held_gb"],
                                   stall_s=r["cold_s"], category=cat,
                                   reuse_gap_s=(clock - last_use.get(name, clock))
                                   if name in last_use else None))
            seen.add(name)
            last_use[name] = clock
            clock += r["ready_s"] if name in arb.retained else r["cold_s"]
            # advance the arbiter's own state one step
            arb_state = arb.retained
            arb.retained = arb_state
            # replicate admission
            if name in arb.retained:
                del arb.retained[name]
            cand = dict(arb.retained); cand[name] = i
            keep = set(cand)
            while sum(C[x]["held_gb"] for x in keep) > budget:
                vic = sorted(x for x in keep if x != name)
                if not vic:
                    keep.discard(name); break
                keep.discard(min(vic, key=lambda x: (arb._rank(x, i), x)))
            prev = dict(arb.retained)
            arb.retained = {x: (i if x == name else prev.get(x, i))
                            for x in sorted(keep)}
            ever_retained |= set(arb.retained)
    return misses, hits, needs


def report(label, misses, hits, needs):
    tot_stall = sum(m["stall_s"] for m in misses)
    print(f"\n=== {label} ===")
    print(f"  {needs} needs, {hits} hits ({100*hits/needs:.1f}%), "
          f"{len(misses)} misses, {tot_stall:,.0f} stall seconds")
    by = collections.defaultdict(list)
    for m in misses:
        by[m["category"]].append(m)
    print(f"  {'category':>15} {'misses':>7} {'%misses':>8} {'stall_s':>11} {'%stall':>7}")
    for cat in ("compulsory", "capacity", "replacement", "never_admitted"):
        g = by.get(cat, [])
        st = sum(m["stall_s"] for m in g)
        print(f"  {cat:>15} {len(g):>7} {100*len(g)/max(len(misses),1):>7.1f}% "
              f"{st:>11,.0f} {100*st/max(tot_stall,1):>6.1f}%")
    fix = by.get("replacement", []) + by.get("never_admitted", [])
    if fix:
        print(f"\n  the {len(fix)} AVOIDABLE misses ({100*sum(m['stall_s'] for m in fix)/tot_stall:.1f}% of stall):")
        c = collections.Counter(m["resource"] for m in fix)
        for n, k in c.most_common():
            sub = [m for m in fix if m["resource"] == n]
            gaps = [m["reuse_gap_s"] for m in sub if m["reuse_gap_s"] is not None]
            print(f"    {n:14s} x{k:<4} {H.CATALOGUE[n]['held_gb']:>6.1f} GB  "
                  f"{sum(m['stall_s'] for m in sub):>8,.0f} s"
                  f"{'   median reuse gap %.0f s' % statistics.median(gaps) if gaps else ''}")
    else:
        print("\n  NO avoidable misses: every miss was compulsory or did not fit.")
        print("  No ranking function can improve on this. A policy contribution")
        print("  measured here would be measuring noise.")
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="256,400,560,700")
    ap.add_argument("--accuracy", type=float, default=0.55)
    ap.add_argument("--out", default="results/analyze_misses.json")
    args = ap.parse_args()

    scheds = H.schedule_from_traces(
        "results/atomagents_metrics_eval_atomagents_exp3_aligned_*.csv",
        data_map=H.TRACE_DATA_MAP)
    out = {}
    for b in [float(x) for x in args.budgets.split(",")]:
        print("\n" + "#" * 66)
        print(f"# BUDGET {b:.0f} GB")
        print("#" * 66)
        res = {}
        for pol, lbl in (("lru", "LRU"), ("value_density", "value_density (ORACLE)")):
            m, h_, n = classify(pol, b, scheds, args.accuracy)
            by = report(f"{lbl} @ {b:.0f} GB", m, h_, n)
            res[pol] = {"hits": h_, "needs": n,
                        "by_category": {k: {"n": len(v),
                                            "stall_s": sum(x["stall_s"] for x in v)}
                                        for k, v in by.items()}}
        out[str(b)] = res
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
