#!/usr/bin/env python3
"""
replay_capacity.py — capacity-honest ceiling: what is achievable with M resident slots?

WHY THIS EXISTS (and why replay_ceiling.py's CONCUR column is NOT the answer)
----------------------------------------------------------------------------
`replay_ceiling.py` reports a CONCUR bound that lets a resource begin loading at t=0
and be held until needed. That silently assumes **unlimited residency**: to hold a
pre-loaded model you need a free slot, and every exp_3 model in `MODELS_BLACKWELL_SWAP`
declares `gpus: [0,1,2,3]` at tp=4 with gpu_memory_utilization 0.82-0.95 — they are
mutually exclusive. **M = 1.** Under M=1 a "pre-load" would evict the running model and
stall the workflow immediately, so the CONCUR number compares an infinite-resource
system against a finite-resource baseline. That comparison is invalid and must not
reach the paper.

This script fixes it by simulating M slots explicitly, with **Belady** eviction (evict
whatever is next needed furthest in the future — the offline optimum). Baseline and
system are evaluated at the SAME M, so any improvement is attributable to scheduling,
never to extra hardware.

M is a real knob, not a hypothetical:
  MODELS_BLACKWELL_SWAP  all 3 models on [0,1,2,3]        -> M = 1
  MODELS_BLACKWELL       32b on [0,1], 72b on [2,3]       -> M = 2  (disjoint pools)
With N = 3 distinct models, M = 2 gives both a free slot to prefetch into AND a real
eviction choice. M = 3 is the uninteresting "enough GPUs for everything" case and is
reported only to show where the curve saturates.

THE MODEL
---------
From each trial's realized trace we recover the alternating structure

    [need r_1, cost c_1] compute g_1 [need r_2, cost c_2] compute g_2 ...

where g_k is the pure compute between need k-1 releasing and need k arriving — a
property of the workload, invariant to how loads are scheduled. Cost c(r) is taken
from the BASELINE arm, where nothing is hidden, so it is the true cold activation cost.

We then replay that sequence under M slots. A load may proceed during compute if a
slot is free; on a need the workflow stalls for whatever load time remains. Belady
picks the victim. Projected wall = sum(compute) + sum(residual stalls).

LIMITS — state these if the numbers are used
  * Belady is offline-optimal, so this is an UPPER bound on any online policy.
  * Costs are state-independent (a load costs c(r) regardless of what else happened),
    which ignores the sleep/wake ladder: gate (b) measured L1 wake at 1.5-2.1 s versus
    a 500-975 s cold boot. So this UNDERSTATES what a sleep-aware system can do.
  * Slots are homogeneous; in reality a 72B and a 32B do not cost the same to hold.

USAGE
    python3 scripts/replay_capacity.py --workload atomagents_exp3
    python3 scripts/replay_capacity.py --workload atomagents_exp3 --slots 1,2,3 -v
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


def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def trial_sequence(rows: list[dict]) -> list[tuple[float, str, float]]:
    """-> [(t_need, resource, exposure)] sorted by need time, stalls only."""
    seq = []
    for r in rows:
        exp = _f(r, "exposure_s")
        t = _f(r, "t_first_needed")
        if not exp or exp <= 0.1 or t is None:
            continue
        name = r.get("resource_name") or r.get("resource_id") or "(unnamed)"
        seq.append((t, name, exp))
    seq.sort(key=lambda x: x[0])
    return seq


def simulate(seq: list[tuple[float, str, float]], cost: dict[str, float],
             slots: int) -> tuple[float, float]:
    """
    Replay the need sequence under `slots` resident slots with Belady eviction.

    -> (projected_wall_s, residual_stall_s)

    Compute gaps come from the observed timeline and are held fixed; only the stall
    portion is affected by scheduling, which is the whole point.
    """
    if not seq:
        return 0.0, 0.0

    # Pure compute between need k-1 releasing and need k arriving.
    gaps, prev_release = [], 0.0
    for t, _, exp in seq:
        gaps.append(max(0.0, t - prev_release))
        prev_release = t + exp

    names = [n for _, n, _ in seq]

    def next_use(res: str, after: int) -> int:
        for j in range(after, len(names)):
            if names[j] == res:
                return j
        return len(names) + 1        # never again -> best victim

    resident: dict[str, float] = {}  # resource -> load time still outstanding
    clock = 0.0
    stall_total = 0.0

    for k, res in enumerate(names):
        # --- compute phase before this need: spend it loading, if a slot is free ---
        budget = gaps[k]
        clock += budget

        # Prefetch the upcoming need during this compute window when possible.
        if res not in resident:
            if len(resident) < slots:
                resident[res] = cost.get(res, 0.0)
            else:
                # Belady: evict whatever is needed furthest out (or never).
                victim = max(resident, key=lambda r: next_use(r, k))
                # Never evict something needed right now.
                if next_use(victim, k) > k:
                    del resident[victim]
                    resident[res] = cost.get(res, 0.0)

        # Apply the compute window to whatever is loading.
        if res in resident:
            resident[res] = max(0.0, resident[res] - budget)

        # --- the need itself ---
        if res in resident:
            stall = resident[res]        # residual load not covered by the window
            resident[res] = 0.0
        else:
            stall = cost.get(res, 0.0)   # no slot was ever free: pay it all
        stall_total += stall
        clock += stall

    return clock, stall_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="atomagents_exp3")
    ap.add_argument("--slots", default="1,2,3", help="comma-separated M values")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    slot_vals = [int(s) for s in args.slots.split(",")]

    rows = [r for r in csv.DictReader(LIFECYCLE.open())
            if r.get("workload") == args.workload]
    walls = {}
    for r in csv.DictReader(SUMMARY.open()):
        if r.get("workload") == args.workload and r.get("status") == "completed":
            w = _f(r, "wall_time_s")
            if w:
                walls[(r.get("config"), r.get("trial_index"))] = w

    trials = defaultdict(list)
    for r in rows:
        trials[(r.get("config"), r.get("trial_index"))].append(r)

    # Cost model from the BASELINE arm: nothing is hidden there, so the observed
    # exposure IS the cold activation cost.
    cost_samples = defaultdict(list)
    for (cfg, ti), rs in trials.items():
        if cfg != "baseline":
            continue
        for _, name, exp in trial_sequence(rs):
            cost_samples[name].append(exp)
    cost = {k: st.median(v) for k, v in cost_samples.items()}

    print(f"workload: {args.workload}")
    print("cold activation cost per resource (median of baseline exposures):")
    for k, v in sorted(cost.items(), key=lambda x: -x[1]):
        print(f"    {k:<22} {v:>8.1f} s   (n={len(cost_samples[k])})")
    if not cost:
        print("  !! no baseline exposures — cannot build a cost model")
        return 2

    base_wall = st.mean(w for (c, _), w in walls.items() if c == "baseline")
    print(f"\nbaseline wall (X) = {base_wall:.1f} s\n")

    hdr = f"{'config':<24}{'N':>3} {'wall_s':>9}"
    for m in slot_vals:
        hdr += f"  {'M=' + str(m) + ' wall':>11}{'  vs X':>8}"
    print(hdr)
    print("-" * len(hdr))

    for cfg in sorted({c for c, _ in trials}):
        ts = [(ti, rs) for (c, ti), rs in trials.items() if c == cfg
              and (c, ti) in walls]
        if not ts:
            continue
        actual = st.mean(walls[(cfg, ti)] for ti, _ in ts)
        line = f"{cfg:<24}{len(ts):>3} {actual:>9.1f}"
        for m in slot_vals:
            projs = [simulate(trial_sequence(rs), cost, m)[0] for _, rs in ts]
            p = st.mean(projs)
            line += f"  {p:>11.1f}{100.0 * (base_wall - p) / base_wall:>7.1f}%"
        print(line)
        if args.verbose:
            for ti, rs in sorted(ts, key=lambda x: str(x[0])):
                s = trial_sequence(rs)
                order = " ".join(n.replace("qwen_", "") for _, n, _ in s)
                print(f"      t{ti}: {len(s)} needs  [{order}]")

    print("\nEvery column is measured at the SAME M for every arm, so an improvement")
    print("cannot come from giving one arm more hardware. Belady is offline-optimal,")
    print("so these are UPPER bounds on any online policy.")
    print("\nM=1 is what all collected exp_3 trials actually ran (MODELS_BLACKWELL_SWAP:")
    print("every model on gpus [0,1,2,3]). M=2 requires --hw-profile blackwell")
    print("(disjoint pools: 32b on [0,1], 72b on [2,3]). M>=N is the uninteresting")
    print("'enough GPUs for everything' case and only shows where the curve saturates.")
    print("\nCosts here are state-independent, so the sleep/wake ladder is NOT modelled")
    print("(gate (b): L1 wake 1.5-2.1 s vs 500-975 s cold boot). These numbers therefore")
    print("UNDERSTATE what a sleep-aware system can achieve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
