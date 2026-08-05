#!/usr/bin/env python3
"""
replay_retention_policy.py — E5: is "value density" actually a POLICY, or just a
description?

THE QUESTION
------------
E1-E4 established that resources can be held at an intermediate rung (a model parked
at R2 via vLLM L1 sleep, 2.06 s to wake against a 782 s cold boot) and that the value
of holding one is s/GB divided by how long you must hold it. That is a ranking
function. It is only a contribution if ranking by it beats what you would do without
it. This replays the realized resource-need sequences from collected trials under a
finite host-RAM budget and scores four policies:

  never_park     today's runtime: every eviction goes to R0, every need is a cold boot
  lru            park what fits, evict least-recently-used  (no cost or size model)
  belady         evict whose next use is furthest away      (offline optimal for HITS)
  value_density  evict lowest  saved / (GB * time_until_next_use)   <- the proposal

belady and value_density both use oracle knowledge of the next use, so neither is
implementable online; they bound what a perfect predictor could achieve and, more
importantly, they differ from each other only in whether COST and SIZE enter the
ranking. If value_density ties belady, the cost model adds nothing. If it beats
belady, then evicting by "furthest next use" is the wrong objective when resources
have wildly different sizes and reload costs -- which is exactly the claim.

INPUT is the realized need sequence, which is policy-independent: the agent's
behaviour does not change with the eviction policy, so the ORDER of model
requirements is a fixed replay input. Each model_load event in metrics.csv is one
need that missed under the current never-park policy, so the load sequence IS the
decision sequence.

COST MODEL (every constant measured, provenance in-line)
  cold boot     per-model median of the measured model_load durations
  wake          bytes_per_gpu / 16.6 GB/s, from E4: 68.28 GB over 2 GPUs in 2.06 s
  park RAM      1.90 x weight bytes, from E4: 120.77 GiB held for a 68.28 GB model

NO GPU. Pure replay over recorded traces.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import statistics
from collections import defaultdict

# --- measured constants ------------------------------------------------------
WAKE_BW_GB_S_PER_GPU = 16.6      # E4: 68.28 GB / 2 GPUs / 2.06 s
PARK_RATIO = 1.90                # E4: 120.77 GiB parked for 68.28 GB of weights
FIRST_PARK_S = 23.692            # E4: first sleep allocates the host backup
SUBSEQ_PARK_S = 2.611            # E4: later sleeps reuse it

MODEL_BYTES = {                  # measured on disk
    "qwen_32b": 68.28e9,
    "qwen_72b": 146.82e9,
    "qwen_72b_text": 145.41e9,
}
MODEL_TP = {"qwen_32b": 4, "qwen_72b": 4, "qwen_72b_text": 4}


def wake_s(m: str) -> float:
    gb = MODEL_BYTES[m] / 1e9
    return gb / MODEL_TP[m] / WAKE_BW_GB_S_PER_GPU


def park_gb(m: str) -> float:
    return MODEL_BYTES[m] / 1e9 * PARK_RATIO


# --- load the realized need sequences ----------------------------------------
def load_traces(root: str) -> tuple[list[dict], dict]:
    trials, boots = [], defaultdict(list)
    for meta_p in sorted(glob.glob(os.path.join(root, "*/*/meta.json"))):
        try:
            meta = json.load(open(meta_p))
        except Exception:
            continue
        if meta.get("status") != "completed":
            continue
        csv_p = os.path.join(os.path.dirname(meta_p), "metrics.csv")
        if not os.path.exists(csv_p):
            continue
        rows = list(csv.DictReader(open(csv_p)))
        if not rows:
            continue
        t0 = dt.datetime.fromisoformat(rows[0]["timestamp"])
        needs = []
        for r in rows:
            if not r["phase"].startswith("model_load:"):
                continue
            name = r["phase"].split(":", 1)[1]
            if name not in MODEL_BYTES:
                continue
            d = float(r["duration_s"] or 0)
            end = (dt.datetime.fromisoformat(r["timestamp"]) - t0).total_seconds()
            needs.append({"t": end - d, "model": name, "measured_boot_s": d})
            boots[name].append(d)
        if len(needs) >= 2:
            needs.sort(key=lambda n: n["t"])
            trials.append({"config": meta.get("config"),
                           "wall": meta.get("wall_time_s", 0),
                           "trial": os.path.basename(os.path.dirname(meta_p)),
                           "needs": needs})
    return trials, {k: statistics.median(v) for k, v in boots.items()}


# --- the replay --------------------------------------------------------------
def replay(needs: list[dict], boot: dict, budget_gb: float, policy: str) -> dict:
    """Return total stall seconds under `policy` at `budget_gb` of park RAM."""
    parked: dict[str, float] = {}     # model -> last-used time
    ever_parked: set[str] = set()
    running: str | None = None
    stall = 0.0
    hits = 0

    def next_use(m: str, i: int) -> float:
        for j in range(i + 1, len(needs)):
            if needs[j]["model"] == m:
                return needs[j]["t"]
        return float("inf")

    for i, nd in enumerate(needs):
        m, t = nd["model"], nd["t"]

        # --- serve the need ---
        if m in parked:
            stall += wake_s(m)
            del parked[m]
            hits += 1
        elif m != running:
            stall += boot.get(m, nd["measured_boot_s"])
        # if m == running the need is free and generates no load event anyway

        # --- the incumbent is displaced; decide whether to park it ---
        if running is not None and running != m:
            cand = running
            trial_set = dict(parked)
            trial_set[cand] = t

            def rank(x: str) -> float:
                """Lower = evict first."""
                nu = next_use(x, i)
                if nu == float("inf"):
                    # Never used again -> always the first victim. MUST be -inf,
                    # not a small constant: belady's rank is -next_use, i.e.
                    # NEGATIVE, so a sentinel of -1.0 sorted AFTER every finite
                    # rank and made belady evict a soon-to-be-reused model in
                    # preference to a dead one. That inversion made "optimal"
                    # score worse than LRU at budget=280, which is impossible
                    # and is what exposed the bug.
                    return float("-inf")
                if policy == "lru":
                    return trial_set[x]              # oldest last-use first
                if policy == "belady":
                    return -nu                       # furthest next use first
                if policy == "value_density":
                    saved = boot.get(x, 0.0) - wake_s(x)
                    held = max(nu - t, 1e-6)
                    return saved / (park_gb(x) * held)
                raise ValueError(policy)

            if policy != "never_park":
                while (sum(park_gb(x) for x in trial_set) > budget_gb
                       and trial_set):
                    victim = min(trial_set, key=rank)
                    del trial_set[victim]
                parked = trial_set
                for x in parked:
                    if x not in ever_parked:
                        stall += FIRST_PARK_S
                        ever_parked.add(x)
                    else:
                        stall += 0.0   # subsequent parks are cheap; see note below
        running = m

    return {"stall_s": stall, "wake_hits": hits, "n_needs": len(needs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="results/eval_q1_q4/runs/atomagents_exp3_aligned")
    ap.add_argument("--budgets",
                    default="0,130,260,280,420,560,700,900")
    ap.add_argument("--out", default="results/replay_retention_policy.json")
    args = ap.parse_args()

    trials, boot = load_traces(args.root)
    if not trials:
        print("no usable trials"); return 2

    print(f"trials with >=2 model needs: {len(trials)}")
    print("median measured cold boot per model:")
    for k, v in sorted(boot.items()):
        print(f"  {k:16s} {v:8.1f} s   weights {MODEL_BYTES[k]/1e9:6.2f} GB   "
              f"park {park_gb(k):6.1f} GB   wake {wake_s(k):5.2f} s")

    budgets = [float(b) for b in args.budgets.split(",")]
    policies = ["never_park", "lru", "belady", "value_density"]
    out = []

    print(f"\n{'budget_GB':>10s} " + "".join(f"{p:>16s}" for p in policies)
          + f"{'VD vs LRU':>12s}{'VD vs Belady':>14s}")
    for b in budgets:
        tot = {}
        for p in policies:
            s = sum(replay(t["needs"], boot, b, p)["stall_s"] for t in trials)
            tot[p] = s
        vd_lru = 100 * (tot["lru"] - tot["value_density"]) / tot["lru"] if tot["lru"] else 0
        vd_bel = 100 * (tot["belady"] - tot["value_density"]) / tot["belady"] if tot["belady"] else 0
        print(f"{b:10.0f} " + "".join(f"{tot[p]:16.0f}" for p in policies)
              + f"{vd_lru:11.1f}%{vd_bel:13.1f}%")
        row = {"budget_gb": b, **{f"stall_{p}": round(tot[p], 1) for p in policies},
               "vd_vs_lru_pct": round(vd_lru, 2), "vd_vs_belady_pct": round(vd_bel, 2),
               "saved_vs_never_pct": round(
                   100 * (tot["never_park"] - tot["value_density"]) / tot["never_park"], 2)}
        out.append(row)

    with open(args.out, "w") as f:
        json.dump({"constants": {"wake_bw_gb_s_per_gpu": WAKE_BW_GB_S_PER_GPU,
                                 "park_ratio": PARK_RATIO,
                                 "first_park_s": FIRST_PARK_S},
                   "median_boot_s": boot, "n_trials": len(trials),
                   "rows": out}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
