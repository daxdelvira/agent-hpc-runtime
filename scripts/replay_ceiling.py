#!/usr/bin/env python3
"""
replay_ceiling.py — Gate M: what end-to-end speedup is achievable AT BEST?

WHY THIS EXISTS
---------------
`eval_stall_taxonomy.csv` says atomagents_exp3/full_system loses 1943 s per trial
and that 85% of that is `no_prediction` — the runtime never predicted the resource,
so it never even tried to hide the load.  That is a large number, but "stall" is not
the same as "recoverable stall": you can only hide a load behind computation that is
actually happening.  Before committing days of GPU collection to a workload we need
to know whether the stall is *hideable*, and that question is answerable offline from
traces already on disk.

THE MODEL
---------
For each trial, take every resource need that exposed stall, ordered by the time it
was first needed.  After need k-1 is satisfied (at t_first_needed[k-1] +
exposure[k-1]) the workflow computes until need k arrives at t_first_needed[k].  That
gap is the window a perfect predictor could have prefetched need k into:

    window_k    = t_first_needed[k] - (t_first_needed[k-1] + exposure[k-1])
    hideable_k  = min(exposure_k, max(0, window_k))

The first need in a trial gets the interval from t=0, which is the workflow's own
start-up computation.

Summing `hideable_k` over a trial gives the best case: every load started the instant
its predecessor released, running fully concurrently with compute.

THIS IS DELIBERATELY OPTIMISTIC, and the paper must say so.  It assumes (a) perfect
prediction with zero lead-time error, (b) unlimited concurrency between staging and
compute, and (c) *no capacity constraint* — which Stage-0 gate (b) shows is false
(level-1 sleep holds 108-128 GiB per engine and the node dies at k=3, so a real
system cannot keep every predicted resource resident).  Treat the output as an upper
bound on the achievable speedup, not a prediction of it.

NEGATIVE CONTROL
----------------
chemgraph_swap must come back near zero.  Its stall is already classified 100%
`no_window` — even the oracle arm — so if this script reports a large ceiling there,
the window reconstruction is wrong and the exp_3 number cannot be trusted either.

USAGE
    python3 scripts/replay_ceiling.py
    python3 scripts/replay_ceiling.py --workload atomagents_exp3 --verbose
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIFECYCLE = REPO / "results/eval_q1_q4/eval_prefetch_lifecycle.csv"
SUMMARY = REPO / "results/eval_q1_q4/eval_q1_summary.csv"

# Arms with no runtime at all: their stall is the un-prefetched reference, and a
# ceiling over them describes what the *workload* offers, not what a config recovered.
NO_RUNTIME = {"baseline", "observe_only"}


def _f(row: dict, key: str):
    """Float or None — the CSV uses '' for absent, which float() rejects."""
    v = row.get(key)
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_walls() -> dict[tuple, float]:
    """(workload, config, trial_index) -> wall_time_s, completed trials only."""
    walls: dict[tuple, float] = {}
    if not SUMMARY.exists():
        return walls
    for r in csv.DictReader(SUMMARY.open()):
        if r.get("status") != "completed":
            continue
        w = _f(r, "wall_time_s")
        if w:
            walls[(r.get("workload"), r.get("config"), r.get("trial_index"))] = w
    return walls


def trial_ceiling(rows: list[dict]) -> tuple[float, float, int, int]:
    """
    -> (hideable_s, exposed_s, n_needs, n_fully_hideable) for one trial.

    Needs without a t_first_needed cannot be placed on the timeline and are counted
    into exposed_s but never into hideable_s — omitting them would silently inflate
    the recoverable fraction.
    """
    placed, unplaced = [], 0.0
    for r in rows:
        exposure = _f(r, "exposure_s")
        if not exposure or exposure <= 0.1:
            continue
        t_need = _f(r, "t_first_needed")
        if t_need is None:
            unplaced += exposure
            continue
        placed.append((t_need, exposure))

    placed.sort(key=lambda x: x[0])

    hideable = 0.0                # SERIAL bound: one staging at a time
    hideable_conc = 0.0           # CONCURRENT bound: unlimited parallel staging
    exposed = unplaced
    n_full = 0
    prev_release = 0.0            # workflow start; the first need hides behind startup
    for t_need, exposure in placed:
        window = max(0.0, t_need - prev_release)
        h = min(exposure, window)
        hideable += h
        # With unlimited concurrency and perfect foresight a load could begin at
        # t=0, so the whole elapsed time before the need is available to hide in.
        hideable_conc += min(exposure, max(0.0, t_need))
        exposed += exposure
        if h >= exposure - 0.01:
            n_full += 1
        prev_release = t_need + exposure

    return hideable, exposed, len(placed), n_full, hideable_conc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--verbose", action="store_true", help="per-trial detail")
    args = ap.parse_args()

    if not LIFECYCLE.exists():
        print(f"MISSING: {LIFECYCLE}\nRun scripts/extract_prefetch_lifecycle.py first.")
        return 2

    rows = list(csv.DictReader(LIFECYCLE.open()))
    walls = load_walls()

    trials: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if args.workload and r.get("workload") != args.workload:
            continue
        trials[(r.get("workload"), r.get("config"), r.get("trial_index"))].append(r)

    per_arm: dict[tuple, list[dict]] = defaultdict(list)
    for (wl, cfg, ti), rs in trials.items():
        wall = walls.get((wl, cfg, ti))
        if not wall:
            continue                      # not completed, or no wall time recorded
        hideable, exposed, n_needs, n_full, hid_conc = trial_ceiling(rs)
        per_arm[(wl, cfg)].append({
            "trial": ti, "wall": wall, "hideable": hideable, "exposed": exposed,
            "n_needs": n_needs, "n_full": n_full, "hid_conc": hid_conc,
            "pct": 100.0 * hideable / wall if wall else 0.0,
            "pct_conc": 100.0 * hid_conc / wall if wall else 0.0,
        })

    # Everything is reported against the workload's OWN baseline wall — the "X" the
    # paper compares to.  Reporting a bare "ceiling %" per arm invites a serious
    # misreading: those percentages are each relative to that ARM's wall, so a faster
    # arm shows a *smaller* percentage purely because its denominator shrank, which
    # looks (wrongly) like the baseline beating the system.  Projected wall times are
    # unambiguous, so lead with those.
    base_wall: dict[str, float] = {}
    for (wl, cfg), ts in per_arm.items():
        if cfg == "baseline":
            base_wall[wl] = st.mean(t["wall"] for t in ts)

    print(f"{'workload':<20}{'config':<22}{'N':>3} {'wall_s':>9} {'proj_wall_s':>12} "
          f"{'now_vs_base':>12} {'best_vs_base':>13} {'serial_vs_base':>15}")
    print("-" * 122)

    for (wl, cfg) in sorted(per_arm):
        ts = per_arm[(wl, cfg)]
        wall = st.mean(t["wall"] for t in ts)
        proj = st.mean(t["wall"] - t["hid_conc"] for t in ts)
        proj_ser = st.mean(t["wall"] - t["hideable"] for t in ts)
        bw = base_wall.get(wl)
        if bw:
            now = f"{100.0 * (bw - wall) / bw:>10.1f}%"
            best = f"{100.0 * (bw - proj) / bw:>11.1f}%"
            ser = f"{100.0 * (bw - proj_ser) / bw:>13.1f}%"
        else:
            now = best = ser = "         n/a"
        flag = "  <-- this is X" if cfg == "baseline" else ""
        print(f"{wl:<20}{cfg:<22}{len(ts):>3} {wall:>9.1f} {proj:>12.1f} "
              f"{now:>12} {best:>13} {ser:>15}{flag}")
        if args.verbose:
            for t in sorted(ts, key=lambda x: str(x["trial"])):
                print(f"      trial {str(t['trial']):>3}  wall={t['wall']:>8.1f} "
                      f"proj={t['wall'] - t['hid_conc']:>8.1f} "
                      f"exposed={t['exposed']:>8.1f} needs={t['n_needs']}")

    print("\nHOW TO READ THIS — every percentage is against the workload's own baseline")
    print("wall (the 'X'), so the columns ARE comparable across arms:")
    print("  wall_s        actual measured wall time for this arm")
    print("  proj_wall_s   wall minus all hideable stall = the achievable floor")
    print("  now_vs_base   what this arm achieves TODAY vs baseline")
    print("  best_vs_base  what it could achieve if every hideable load were hidden")
    print("  serial_vs_base  same, but if a load may only start once the previous")
    print("                  need releases — i.e. today's ONE-STEP lookahead")
    print("\nSanity check: every arm of a workload should project to roughly the SAME")
    print("proj_wall_s, because the floor is a property of the workload, not the arm.")
    print("A large spread there means the replay is wrong.")
    print("\nOptimistic by construction: assumes perfect prediction and no capacity")
    print("limit. Gate (b) measured 108-128 GiB of host RAM per slept engine, so a real")
    print("system cannot hold everything. Read best_vs_base as an upper bound.")
    print("\nGate M: proceed if atomagents_exp3 best_vs_base >=20% AND chemgraph_swap")
    print("is near zero (negative control — its stall is already 100% no_window, so a")
    print("large ceiling there would mean the window reconstruction is broken).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
