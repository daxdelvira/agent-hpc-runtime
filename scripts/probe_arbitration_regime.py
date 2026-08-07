#!/usr/bin/env python3
"""
probe_arbitration_regime.py — what would have to change for ARBITRATION to matter?

CONTEXT (2026-08-07). The 57.7% "retention reduces stall" headline is retracted:
it lives at a 700 GB budget where the entire 838 GB footprint nearly fits, so the
correct policy there is "park everything and never evict" -- a vLLM flag, not a
scheduler. At the production 256 GB allocation retention is worth 8.4% and an
oracle recovers +0.00% over LRU. So the open question is not "does retention pay"
but "is there ANY configuration where deciding WHAT to retain pays."

The miss taxonomy says exactly why 256 GB is dead, and it is structural, not a
tuning problem:

    at 256 GB, of 209 misses / 142,238 stall seconds
      compulsory   107 misses  46.0% of stall   first use; no ranking can help
      capacity      92 misses  51.4% of stall   does not fit AT ALL at this budget
      replacement   10 misses   2.6% of stall   <-- the only convertible category

51.4% is capacity because qwen_72b (279.0 GB) and qwen_72b_text (276.3 GB) each
EXCEED a 256 GB budget on their own. The entire model class is therefore
unrankable at the production allocation: there is no decision, only an
impossibility. 46.0% is compulsory because the schedules are short -- most
resources are used once, and a first use cannot be a retention hit.

THIS SCRIPT tests the three candidate fixes, one at a time and then together, and
reports the ONLY number that matters: the gap between LRU and an optimal selector
at the same budget. Everything else (never-retain reduction) is vLLM's.

  H1  QUANTIZE THE MODELS so they individually fit the budget.
      Converts capacity misses into replacement misses -- i.e. into decisions.
      fp8 halves weights, and because parking cost is 1.90x weights (measured,
      bench_wake_L1_coherence_32b.json) the parked footprint halves too. Boot is
      74-81% weight movement, so cold cost roughly halves as well.
      *** THE MODELLED NUMBERS ARE PROJECTIONS, NOT MEASUREMENTS. *** They assume
      fp8 is exactly half of bf16 and that movement scales linearly. Validating
      H1 requires parking one real quantized 72B and reading held_gb / cold_s.

  H2  GIVE THE DATA CLASS INTERNAL SPREAD. Today all three LAMMPS steps map to
      one artifact (uniref90), so data never competes with data. UniRef50 and
      UniRef90 differ 3.2x in size, which LRU (size-blind) should lose to -- but
      their s/GB is nearly identical (2.968 vs 3.179), and value density can only
      differ from Belady to the extent s/GB VARIES. A profile-HMM artifact is the
      cheap source of that spread: Pfam-A activates far cheaper per GB.

  H3  LONGER SCHEDULES. 46% compulsory means most resources are used once. Reuse
      is what creates a retention decision at all.

Run: python3 scripts/probe_arbitration_regime.py
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Catalogues. MEASURED rows are marked; PROJECTED rows carry their derivation.
# ---------------------------------------------------------------------------
MEASURED = dict(H.CATALOGUE)

# H1: fp8 quantization. weights/2 -> parked footprint/2 (park ratio 1.90x is a
# ratio, so it survives); boot is 74-81% movement so cold ~/2 with engine init
# (~10 s, measured median) held FIXED because it does not scale with weights.
QUANT = {}
for _n, _r in MEASURED.items():
    if _r["cls"] == "model":
        QUANT[_n] = dict(cls="model", held_gb=round(_r["held_gb"] / 2, 1),
                         cold_s=round((_r["cold_s"] - 10.0) / 2 + 10.0, 1),
                         ready_s=round(_r["ready_s"] / 2, 2))
    else:
        QUANT[_n] = dict(_r)

# H2: a third data artifact with genuinely different s/GB. Pfam-A.hmm is 0.42 GB
# compressed / ~1.5 GB raw; profile HMMs activate cheaply per GB relative to a
# sequence block. Sized up to compete (a full profile library at Pfam+PDB+CATH
# scale). s/GB 0.9 vs uniref's 2.97-3.18 -> a 3.5x spread where there is none now.
PFAM = dict(cls="data", held_gb=88.0, cold_s=79.2, ready_s=0.0)


def add_pfam(cat):
    c = dict(cat)
    c["pfam_profiles"] = dict(PFAM)
    return c


def stall(cat, policy, budget, scheds, prefetch=False, optimal=False,
          accuracy=1.0):
    old = H.CATALOGUE
    H.CATALOGUE = cat
    try:
        tot = 0.0
        for k, s in enumerate(scheds):
            a = H.Arbiter(policy, budget, prefetch, s, Dry(), lambda *_: None,
                          predict_accuracy=accuracy, seed=k,
                          optimal_selection=optimal)
            tot += a.run_analytic()["stall_s"]
        return tot
    finally:
        H.CATALOGUE = old


# ---------------------------------------------------------------------------
# POPULARITY ORDER -- and why passing sorted(CATALOGUE) was a real error.
#
# synthetic_schedule() assigns Zipf popularity weights by POSITION in the list it
# is handed, so whichever resource is passed first becomes the most-requested one.
# Passing sorted(CATALOGUE) therefore made qwen_32b most popular for no reason
# beyond string collation ("q" < "u", "32" < "72") -- and qwen_32b at 129.7 GB is
# the ONLY model that fits a 256 GB budget, so the arbitrary choice handed the
# most requests to the one resource retention can actually serve.
#
# Measured cost of that accident, over all 120 orderings of the 5 resources at
# 256 GB / 12 needs / 40 schedules:
#
#     opt-vs-LRU gap   min 3.21%   median 6.42%   max 10.18%       (3.2x swing)
#     most popular = qwen_32b      -> 9.17%   <-- what sorted() picked
#     most popular = qwen_72b      -> 4.86%   <-- what the TRACES say
#
# The real exp3 traces request qwen_72b 99 times, qwen_72b_text 47, uniref90 71,
# qwen_32b 27. So the trace-faithful ordering is the LEAST flattering one, and the
# sorted() default was inflating the headline by ~1.9x. Ordering is now taken from
# the traces, and any new resource is inserted by its measured request share.
TRACE_NEED_COUNTS = {"qwen_72b": 99, "uniref90": 71, "qwen_72b_text": 47,
                     "qwen_32b": 27}


def popularity_order(cat):
    """Most-requested first, from the traces; unseen resources rank last."""
    return sorted(cat, key=lambda n: (-TRACE_NEED_COUNTS.get(n, 0), n))


def scheds_for(cat, n_needs, n_sched=40, window_scale=1.0, seed0=7000):
    old = H.CATALOGUE
    H.CATALOGUE = cat
    try:
        return [H.synthetic_schedule(n_needs=n_needs, window_scale=window_scale,
                                     seed=seed0 + k, resources=popularity_order(cat))
                for k in range(n_sched)]
    finally:
        H.CATALOGUE = old


def cell(label, cat, budget, n_needs, n_sched=40):
    """One configuration. Reports ratio-of-totals, never mean-of-ratios."""
    scheds = scheds_for(cat, n_needs, n_sched)
    never = stall(cat, "never_retain", budget, scheds)
    lru = stall(cat, "lru", budget, scheds)
    opt = stall(cat, "value_density", budget, scheds, optimal=True)
    bel = stall(cat, "belady", budget, scheds)
    foot = sum(r["held_gb"] for r in cat.values())
    biggest = max(r["held_gb"] for r in cat.values())
    ret = 100 * (never - lru) / never if never else 0.0
    gap = 100 * (lru - opt) / lru if lru else 0.0
    bgap = 100 * (lru - bel) / lru if lru else 0.0
    print(f"  {label:<34} {foot:>6.0f} {biggest:>6.0f} {never:>9.0f} "
          f"{lru:>9.0f} {ret:>7.1f}% {gap:>8.2f}% {bgap:>8.2f}%")
    return dict(label=label, budget_gb=budget, n_needs=n_needs,
                footprint_gb=round(foot, 1), largest_gb=round(biggest, 1),
                never_s=round(never), lru_s=round(lru), opt_s=round(opt),
                belady_s=round(bel), retention_pct=round(ret, 2),
                opt_vs_lru_pct=round(gap, 2), belady_vs_lru_pct=round(bgap, 2))


def main() -> int:
    B = 256.0          # the production allocation; the only budget that is real
    rows = []

    print("\nAll cells at the PRODUCTION 256 GB budget. The column that matters is")
    print("opt-LRU: what a scheduler adds OVER the size-blind baseline. 'retain%'")
    print("is what vLLM's sleep flag would get on its own and is NOT a contribution.\n")
    hdr = (f"  {'configuration':<34} {'foot':>6} {'largest':>6} {'never':>9} "
           f"{'LRU':>9} {'retain%':>8} {'opt-LRU':>9} {'bel-LRU':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    print("\n  [baseline: what we have today]")
    rows.append(cell("measured, 12 needs", MEASURED, B, 12))

    print("\n  [H1  quantized models -- each one now FITS 256 GB]")
    rows.append(cell("fp8 models, 12 needs", QUANT, B, 12))

    print("\n  [H2  a third data artifact with different s/GB]")
    rows.append(cell("measured + pfam, 12 needs", add_pfam(MEASURED), B, 12))

    print("\n  [H3  longer schedules -- more reuse, fewer compulsory misses]")
    for n in (24, 48):
        rows.append(cell(f"measured, {n} needs", MEASURED, B, n))

    print("\n  [H1+H2+H3 together]")
    for n in (12, 24, 48):
        rows.append(cell(f"fp8 + pfam, {n} needs", add_pfam(QUANT), B, n))

    # ---- which single change moves the gap most? --------------------------
    base = rows[0]["opt_vs_lru_pct"]
    print(f"\n  marginal effect on the opt-vs-LRU gap (baseline {base:.2f}%):")
    for r in rows[1:]:
        print(f"    {r['label']:<34} {r['opt_vs_lru_pct']:>7.2f}% "
              f"({r['opt_vs_lru_pct'] - base:+.2f} pts)")

    best = max(rows, key=lambda r: r["opt_vs_lru_pct"])
    print(f"\n  best cell: {best['label']} -> {best['opt_vs_lru_pct']:.2f}% "
          f"over LRU, {best['belady_vs_lru_pct']:.2f}% for Belady")
    if best["opt_vs_lru_pct"] < 3.0:
        print("\n  VERDICT: no tested change opens a gap worth reporting. The")
        print("  arbitration claim does not survive on this resource set, and the")
        print("  paper should say so rather than sweep for a flattering cell.")
    else:
        print("\n  VERDICT: a gap exists. It is a PROJECTION until the quantized")
        print("  park footprint and the profile-HMM s/GB are measured, not modelled.")

    out = "results/probe_arbitration_regime.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"budget_gb": B, "rows": rows,
                   "caveat": "fp8 and pfam rows are PROJECTIONS; see docstring"},
                  f, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
