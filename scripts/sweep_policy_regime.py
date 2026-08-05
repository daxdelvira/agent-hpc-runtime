#!/usr/bin/env python3
"""
sweep_policy_regime.py — when is a cost-aware retention policy worth building?

CONTEXT
-------
E5 replayed the real model-need sequences and found value_density ties LRU and Belady
EXACTLY at every budget. Three models, room for one: there is no decision to make.
That is a property of the configuration, not a verdict on the policy. This sweep asks
what configuration WOULD discriminate, so the paper can state the boundary rather than
generalise from one degenerate point.

THE HYPOTHESIS BEING TESTED
---------------------------
Value density ranks by  saved / (size * time_until_next_use).
If reload cost is proportional to size, then saved/size is a CONSTANT and the ranking
collapses to a monotone function of 1/time_until_next_use -- which is exactly Belady's
ranking. So:

    value_density can only differ from Belady to the extent that COST PER BYTE varies.

That is not incidental: cost-per-byte is precisely the s/GB quantity Insight B measures,
and we measured a 65x spread across data formats (ASCII 22.0 -> npy 0.34) against ~1.2x
across our three models. The prediction is therefore that the policy matters in the
MIXED-CLASS regime and nowhere else.

Note also that Belady is NOT the ceiling here. Furthest-next-use is optimal for HIT COUNT
in a uniform-cost, uniform-size cache. Ours is neither, and minimising misses is not
minimising stall seconds when one miss costs 455 s and another 798 s. A cost-aware policy
can legitimately BEAT Belady on stall time; Belady is a strong baseline, not a bound.

METHOD
------
Synthetic resources with controlled size and cost-per-byte spread; a request stream with
controlled burstiness (which produces the tight-cluster / long-gap reuse structure the
real traces show); replay under a size budget. R independent seeds per cell, reported as
mean +/- std with a paired comparison against LRU.

NO GPU. Pure simulation -- this answers a DESIGN question, and its numbers must never be
presented as measurements of the real workload.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics


# --------------------------------------------------------------------------
def make_resources(n: int, size_spread: float, cpb_spread: float,
                   rng: random.Random) -> list[dict]:
    """n resources with log-uniform size and log-uniform cost-per-byte."""
    out = []
    for i in range(n):
        # log-uniform over [1, spread]
        s = math.exp(rng.uniform(0, math.log(size_spread))) if size_spread > 1 else 1.0
        c = math.exp(rng.uniform(0, math.log(cpb_spread))) if cpb_spread > 1 else 1.0
        out.append({"id": i, "size": s, "cpb": c, "cost": s * c})
    return out


def make_stream(res: list[dict], length: int, burst: float,
                zipf_a: float, rng: random.Random) -> list[int]:
    """Request stream: Zipf popularity plus bursts (repeat the previous id).

    Bursts are what create the tight-cluster reuse we measured on the data side
    (three LAMMPS invocations inside one compute window) against the long gaps
    on the model side (median 848 s).
    """
    n = len(res)
    weights = [1.0 / ((i + 1) ** zipf_a) for i in range(n)]
    ids = list(range(n))
    stream = []
    prev = None
    for _ in range(length):
        if prev is not None and rng.random() < burst:
            stream.append(prev)
        else:
            pick = rng.choices(ids, weights=weights, k=1)[0]
            stream.append(pick)
            prev = pick
    return stream


# --------------------------------------------------------------------------
def replay(stream: list[int], res: list[dict], budget: float, policy: str) -> float:
    """Total stall cost under `policy`. Resident set constrained by `budget`."""
    resident: dict[int, int] = {}          # id -> last used index
    stall = 0.0
    nxt = _next_use_table(stream)

    for i, rid in enumerate(stream):
        if rid in resident:
            resident[rid] = i
        else:
            stall += res[rid]["cost"]
            resident[rid] = i
            # admit, then evict until within budget
            while sum(res[x]["size"] for x in resident) > budget and len(resident) > 1:
                victim = min(
                    (x for x in resident if x != rid),
                    key=lambda x: _rank(x, i, policy, res, resident, nxt),
                    default=None,
                )
                if victim is None:
                    break
                del resident[victim]
            if sum(res[x]["size"] for x in resident) > budget:
                del resident[rid]          # does not fit even alone
    return stall


def _next_use_table(stream: list[int]) -> list[float]:
    """nxt[i] = index of the next occurrence of stream[i] after i, else inf."""
    n = len(stream)
    nxt = [math.inf] * n
    last: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        rid = stream[i]
        nxt[i] = last.get(rid, math.inf)
        last[rid] = i
    return nxt


def _rank(x: int, i: int, policy: str, res, resident, nxt) -> float:
    """Lower rank = evicted first."""
    # next use of x after position i
    nu = math.inf
    j = resident[x]
    # walk forward from x's last use via the precomputed table
    k = nxt[j] if j < len(nxt) else math.inf
    while k is not math.inf and k <= i:
        k = nxt[int(k)]
    nu = k

    if nu is math.inf:
        return float("-inf")               # never used again: always first victim
    if policy == "lru":
        return resident[x]
    if policy == "belady":
        return -nu
    if policy == "value_density":
        held = max(nu - i, 1e-9)
        return res[x]["cost"] / (res[x]["size"] * held)
    raise ValueError(policy)


# --------------------------------------------------------------------------
def cell(n_res, size_spread, cpb_spread, budget_frac, burst, zipf_a,
         length, reps, seed0) -> dict:
    gains_lru, gains_bel = [], []
    for r in range(reps):
        rng = random.Random(seed0 + r)
        res = make_resources(n_res, size_spread, cpb_spread, rng)
        stream = make_stream(res, length, burst, zipf_a, rng)
        budget = budget_frac * sum(x["size"] for x in res)
        s_lru = replay(stream, res, budget, "lru")
        s_bel = replay(stream, res, budget, "belady")
        s_vd = replay(stream, res, budget, "value_density")
        if s_lru > 0:
            gains_lru.append(100 * (s_lru - s_vd) / s_lru)
        if s_bel > 0:
            gains_bel.append(100 * (s_bel - s_vd) / s_bel)
    return {
        "vd_vs_lru_mean": statistics.mean(gains_lru),
        "vd_vs_lru_std": statistics.pstdev(gains_lru),
        "vd_vs_bel_mean": statistics.mean(gains_bel),
        "vd_vs_bel_std": statistics.pstdev(gains_bel),
        "n": len(gains_lru),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-res", type=int, default=12)
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--burst", type=float, default=0.35)
    ap.add_argument("--zipf", type=float, default=0.8)
    ap.add_argument("--out", default="results/sweep_policy_regime.json")
    args = ap.parse_args()

    print("Cost-per-byte spread is the hypothesised discriminating axis.")
    print("If cost is proportional to size (spread = 1x), value_density MUST reduce")
    print("to Belady's ranking. Confirming that is the sanity check on the whole sweep.\n")

    cpbs = [1.0, 2.0, 5.0, 10.0, 25.0, 65.0]
    fracs = [0.20, 0.35, 0.50]
    rows = []

    for bf in fracs:
        print(f"--- budget = {bf:.0%} of total resident size, "
              f"{args.n_res} resources, size spread 20x ---")
        print(f"{'cost/byte spread':>17s} {'VD vs LRU':>20s} {'VD vs Belady':>20s}")
        for cs in cpbs:
            r = cell(args.n_res, 20.0, cs, bf, args.burst, args.zipf,
                     args.length, args.reps, seed0=1000)
            rows.append({"budget_frac": bf, "cpb_spread": cs, **r})
            print(f"{cs:14.0f}x   {r['vd_vs_lru_mean']:9.1f}% ± {r['vd_vs_lru_std']:5.1f}"
                  f"   {r['vd_vs_bel_mean']:9.1f}% ± {r['vd_vs_bel_std']:5.1f}")
        print()

    with open(args.out, "w") as f:
        json.dump({"params": vars(args), "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
