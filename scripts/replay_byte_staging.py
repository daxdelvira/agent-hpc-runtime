#!/usr/bin/env python3
"""
replay_byte_staging.py — what could a MegaMmap-style BYTE tier achieve on
AtomAgents, as a function of predictor quality?

WHY THIS EXISTS
---------------
`megammap_stage` was only ever collected on chemgraph_swap, where it was
3.18x SLOWER than baseline and 100% of its stalls were `window_too_small`.
That result does not transfer to AtomAgents, because the two workloads fail
differently:

    chemgraph_swap  full_system  ->  no_window          198.4 s/trial
    atomagents_exp3 full_system  ->  no_prediction     1642.2 s/trial

chemgraph_swap has no window to stage into; AtomAgents does (the LAMMPS
compute) and instead fails to identify what to stage. So "MegaMmap loses on
AtomAgents too" is NOT established by the chemgraph number, and running the
live arm needs driver work plus ~23 h of GPU we do not have. This replays the
question offline over traces already on disk.

THE MODEL
---------
Same window reconstruction as replay_ceiling.py, over the BASELINE arm (no
runtime, so its exposures are the un-prefetched reference):

    window_k = t_first_needed[k] - (t_first_needed[k-1] + exposure[k-1])

A byte tier differs from the ceiling replay in two ways, and both matter:

  1. It only removes the MOVEMENT part of a load. Making a resource usable is
     movement THEN transformation, and a tier that relocates bytes cannot
     perform the transformation. `r` below is the movement share.
  2. It must have PREDICTED the need to start staging. `a` is the probability
     a given need is correctly identified in time.

    recovered_k = a * min(r * exposure_k, window_k)

`r` IS SWEPT, NOT ASSUMED. For the data class it is measured: the EAM
potential's cold/warm LAMMPS load differs by 1.87 s of 100.10 s on one node
and 1.97 s of 99.67 s on another, so a byte tier recovers ~1.9% there
(results/bench_potential_activation_*.json). For the MODEL class we have no
equally clean measurement -- page-cache warming moved vLLM boot in both
directions across nodes -- so rather than pick a number, the sweep reports
the whole range and lets r=1.0 stand as a hard bound. r=1.0 means byte
movement is the ENTIRE cost of loading a model, which is maximally generous
to the byte tier and certainly false.

DELIBERATELY OPTIMISTIC, in four ways that all favour the byte tier:
  (a) a wrong prediction costs nothing -- no wasted bandwidth, no eviction of
      something useful, no delay to the correct stage. The live chemgraph
      arm shows real staging ADDS time; none of that is charged here.
  (b) staging is free of contention with the running workflow.
  (c) no capacity constraint -- every predicted resource can be held.
  (d) the CONCURRENT variant lets a load begin at t=0.
So the output is an upper bound. If the bound is small, the conclusion is
safe; if it is large, nothing is proven.

NOTE ON THE DATA CLASS: on AtomAgents exp3 every data_file need has exposure
0.0 in every arm -- the potentials never stalled the workflow. 100% of
measured exposed stall is model loading. The 1.9% figure therefore bounds a
class that contributes no stall here, and is reported for completeness only.

USAGE
    python3 scripts/replay_byte_staging.py
    python3 scripts/replay_byte_staging.py --config baseline --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIFECYCLE = REPO / "results/eval_q1_q4/eval_prefetch_lifecycle.csv"
SUMMARY = REPO / "results/eval_q1_q4/eval_q1_summary.csv"

# Measured movement share for the data class, from the cold/warm LAMMPS rungs
# on two independent nodes. Kept as a constant so the provenance travels with
# the number rather than living only in a comment.
R_DATA = ((100.10 - 98.23) / 100.10 + (99.67 - 97.70) / 99.67) / 2

ACC = [0.0, 0.20, 0.3333, 0.55, 0.584, 0.75, 1.0]
RSHARE = [R_DATA, 0.10, 0.25, 0.50, 0.75, 1.00]


def _f(row, key):
    v = row.get(key)
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_walls():
    walls = {}
    for r in csv.DictReader(SUMMARY.open()):
        if r.get("status") != "completed":
            continue
        w = _f(r, "wall_time_s")
        if w:
            walls[(r["workload"], r["config"], r["trial_index"])] = w
    return walls


def trial_needs(rows):
    """[(t_first_needed, exposure)] sorted, plus unplaced exposure."""
    placed, unplaced = [], 0.0
    for r in rows:
        e = _f(r, "exposure_s")
        if not e or e <= 0.1:
            continue
        t = _f(r, "t_first_needed")
        if t is None:
            unplaced += e
            continue
        placed.append((t, e))
    placed.sort()
    return placed, unplaced


def recovered(placed, a, r, concurrent=False):
    """Expected stall seconds a byte tier removes at accuracy `a`, share `r`."""
    tot, prev_release = 0.0, 0.0
    for t_need, exposure in placed:
        window = max(0.0, t_need if concurrent else t_need - prev_release)
        tot += a * min(r * exposure, window)
        prev_release = t_need + exposure
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="atomagents_exp3")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    trials = defaultdict(list)
    for r in csv.DictReader(LIFECYCLE.open()):
        if r["workload"] == args.workload and r["config"] == args.config:
            trials[r["trial_index"]].append(r)
    if not trials:
        print(f"no lifecycle rows for {args.workload}/{args.config}")
        return 2

    walls = load_walls()
    data = []
    for ti, rows in sorted(trials.items()):
        placed, unplaced = trial_needs(rows)
        w = walls.get((args.workload, args.config, ti))
        if not w or not placed:
            continue
        data.append({"trial": ti, "wall": w, "placed": placed,
                     "unplaced": unplaced,
                     "exposed": sum(e for _, e in placed) + unplaced})

    tot_wall = sum(d["wall"] for d in data)
    tot_exp = sum(d["exposed"] for d in data)
    print(f"{args.workload} / {args.config}: {len(data)} trials, "
          f"{sum(len(d['placed']) for d in data)} placed needs")
    print(f"summed wall {tot_wall:.0f} s, exposed stall {tot_exp:.0f} s "
          f"({100*tot_exp/tot_wall:.1f}% of wall)")
    print(f"measured movement share for the DATA class: {100*R_DATA:.2f}% "
          f"(contributes no stall here -- see module docstring)")
    print()

    out = {"workload": args.workload, "config": args.config,
           "n_trials": len(data), "wall_s": tot_wall, "exposed_s": tot_exp,
           "r_data_measured": R_DATA, "cells": []}

    for label, conc in (("SERIAL (stage starts when the previous need releases)", False),
                        ("CONCURRENT (stage may start at t=0)", True)):
        print(label)
        print("  recovered stall as % of summed wall")
        print("  " + "movement share r".ljust(20)
              + "".join(f"a={a:<7.2f}" for a in ACC))
        for r in RSHARE:
            cells = []
            for a in ACC:
                rec = sum(recovered(d["placed"], a, r, conc) for d in data)
                cells.append(100.0 * rec / tot_wall)
                out["cells"].append({"concurrent": conc, "r": r, "a": a,
                                     "recovered_s": rec,
                                     "pct_of_wall": 100.0 * rec / tot_wall})
            tag = f"{r:.3f}" + (" (measured, data)" if abs(r - R_DATA) < 1e-9 else "")
            print("  " + tag.ljust(20) + "".join(f"{c:<9.2f}" for c in cells))
        print()

    print("READING THIS TABLE")
    print("  a = predictor accuracy. a=0 is no predictor at all; a=1/3 is the")
    print("      expected hit rate of random staging over the 3 distinct models;")
    print("      a=0.584 is the measured modal-action agreement for this workload.")
    print("  r = share of a load a byte tier can remove. r=1.00 asserts byte")
    print("      movement is the ENTIRE cost of loading a model, which is false")
    print("      and is included only as a hard bound.")
    print("  Every cell is optimistic: wrong predictions are charged nothing.")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
