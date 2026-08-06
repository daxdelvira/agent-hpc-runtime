#!/usr/bin/env python3
"""
sweep_arbitration_regimes.py — WHEN does a cost-aware, prefetch-aware policy matter?

THE QUESTION
------------
On the real exp3 traces, cost-aware retention beats LRU by ~2% and prefetching
buys under 1%. Read narrowly that says "the policy is not worth building." Read
correctly it says "this configuration does not exercise the policy" -- the same
thing E5 and the two-class replay each concluded before it. So rather than
report one point, sweep the axes that plausibly control the answer and state
where the policy earns its keep and where it does not.

THE AXES, AND WHY EACH IS PLAUSIBLE
-----------------------------------
  n_models      2-5 backing models. Real agentic stacks are trending toward more
                specialised models, not fewer.
  n_data        1-4 activated data artifacts.
  gpu_slots M   how many models can be RESIDENT at once. exp3 has M=1 (every
                model declares all 4 GPUs at tp=4), but a node with more GPUs, or
                smaller models, gives M=2+ -- which changes everything, because
                with M>=2 a prefetch of a model becomes physically possible.
  budget_frac   host-RAM budget as a fraction of the total footprint. This is the
                pressure knob and the one the paper's threshold results are about.
  window_scale  how much real computation happens between tool calls. exp3's
                windows are small because it is a pre-scripted benchmark; a
                workflow doing genuine simulation between calls has far larger
                ones. This is the axis that decides whether prefetching matters.
  accuracy      predictor accuracy. This project measured 45-62%.

WHAT IS REPORTED
----------------
For each cell, stall seconds under: never_retain, lru (realizable baseline),
value_density_pred (realizable system), the same plus prefetch, and belady
(oracle bound). The headline per cell is the two gaps that matter:

    vd_pred vs lru      is cost-awareness worth it?
    +prefetch vs vd     is prefetching worth it?

DETERMINISM. Set-iteration order was a 23% source of run-to-run variance in this
harness until 2026-08-06; every selection here is sorted. If a cell's numbers
move between runs, that is a bug, not noise.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "harness", os.path.join(HERE, "bench_arbitration_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


class Dry:
    """Analytic backend: run_analytic() advances a simulated clock itself."""
    speedup = 1e12

    def make_ready(self, n, retained=False):
        return 0.0

    def evict(self, n):
        pass

    def park(self, n):
        pass

    def shutdown(self):
        pass


def build_catalogue(n_models: int, n_data: int) -> dict:
    """Synthesise a catalogue anchored on the MEASURED resources.

    The first 3 models and 2 data artifacts are the real ones. Beyond that we
    interpolate, keeping the measured relationships: park cost 1.90x weights,
    wake ~ size / (16.6 GB/s x 4 GPUs), data expansion ~2x with s/GB near 3.
    Extra resources are labelled synthetic so no one mistakes them for measured.
    """
    real_models = [
        ("qwen_32b", 129.7, 495.2, 1.03),
        ("qwen_72b", 279.0, 800.5, 2.21),
        ("qwen_72b_text", 276.3, 770.3, 2.19),
    ]
    real_data = [
        ("uniref50", 36.08, 107.1, 0.0),
        ("uniref90", 117.20, 372.6, 0.0),
    ]
    cat = {}
    for k in range(n_models):
        if k < len(real_models):
            n, gb, cold, ready = real_models[k]
        else:
            gb = 150.0 + 60.0 * (k - len(real_models) + 1)
            cold = 500.0 + 150.0 * (k - len(real_models) + 1)
            ready = gb / (16.6 * 4)
            n = f"model_syn{k}"
        cat[n] = dict(cls="model", held_gb=gb, cold_s=cold, ready_s=ready)
    for k in range(n_data):
        if k < len(real_data):
            n, gb, cold, ready = real_data[k]
        else:
            gb = 40.0 + 45.0 * (k - len(real_data) + 1)
            cold = gb * 3.1
            ready = 0.0
            n = f"data_syn{k}"
        cat[n] = dict(cls="data", held_gb=gb, cold_s=cold, ready_s=ready)
    return cat


def make_schedules(cat: dict, n_sched: int, n_needs: int,
                   window_scale: float, seed0: int) -> list:
    old = H.CATALOGUE
    H.CATALOGUE = cat
    try:
        return [H.synthetic_schedule(n_needs=n_needs, window_scale=window_scale,
                                     seed=seed0 + k, resources=list(cat))
                for k in range(n_sched)]
    finally:
        H.CATALOGUE = old


def run_cell(cat, scheds, budget, accuracy, gpu_slots) -> dict:
    """One configuration, all arms. Returns stall seconds per arm."""
    old = H.CATALOGUE
    H.CATALOGUE = cat
    out = {}
    try:
        for arm, pol, pf in (("never", "never_retain", False),
                             ("lru", "lru", False),
                             ("vd_pred", "value_density_pred", False),
                             ("vd_pf", "value_density_pred", True),
                             ("belady", "belady", False)):
            tot = 0.0
            for k, s in enumerate(scheds):
                a = H.Arbiter(pol, budget, pf, s, Dry(), lambda *_: None,
                              predict_accuracy=accuracy, seed=k)
                # gpu_slots > 1 means a model prefetch is physically possible
                # while another model serves; M=1 forbids it.
                a.gpu_slots = gpu_slots
                tot += a.run_analytic()["stall_s"]
            out[arm] = tot
    finally:
        H.CATALOGUE = old
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sched", type=int, default=60)
    ap.add_argument("--n-needs", type=int, default=12)
    ap.add_argument("--accuracy", type=float, default=0.55)
    ap.add_argument("--out", default="results/sweep_arbitration_regimes.json")
    args = ap.parse_args()

    MODELS = (2, 3, 5)
    DATA = (1, 2, 4)
    SLOTS = (1, 2)
    FRACS = (0.25, 0.5, 0.75)
    WINDOWS = (1.0, 4.0, 16.0)

    rows = []
    print(f"predictor accuracy {args.accuracy}, {args.n_sched} schedules x "
          f"{args.n_needs} needs per cell\n")
    hdr = (f"{'M':>2} {'D':>2} {'slots':>5} {'bud%':>5} {'win':>4} "
           f"{'never':>9} {'lru':>9} {'vd_pred':>9} {'vd+PF':>9} {'belady':>9}"
           f" {'vd-lru':>8} {'PF gain':>8}")
    print(hdr); print("-" * len(hdr))
    for nm, nd, slots, frac, win in itertools.product(MODELS, DATA, SLOTS,
                                                      FRACS, WINDOWS):
        cat = build_catalogue(nm, nd)
        total = sum(r["held_gb"] for r in cat.values())
        budget = frac * total
        scheds = make_schedules(cat, args.n_sched, args.n_needs, win, seed0=1000)
        r = run_cell(cat, scheds, budget, args.accuracy, slots)
        vd_gap = 100 * (r["lru"] - r["vd_pred"]) / r["lru"] if r["lru"] else 0
        pf_gap = 100 * (r["vd_pred"] - r["vd_pf"]) / r["vd_pred"] if r["vd_pred"] else 0
        rows.append(dict(n_models=nm, n_data=nd, gpu_slots=slots,
                         budget_frac=frac, window_scale=win,
                         budget_gb=round(budget, 1), **{k: round(v, 1) for k, v in r.items()},
                         vd_vs_lru_pct=round(vd_gap, 2),
                         prefetch_gain_pct=round(pf_gap, 2)))
        print(f"{nm:>2} {nd:>2} {slots:>5} {frac:>5.2f} {win:>4.0f} "
              f"{r['never']:>9.0f} {r['lru']:>9.0f} {r['vd_pred']:>9.0f} "
              f"{r['vd_pf']:>9.0f} {r['belady']:>9.0f} "
              f"{vd_gap:>7.2f}% {pf_gap:>7.2f}%")

    # --- which axis actually controls each gap? -----------------------------
    print("\n=== marginal effect of each axis ===")
    for axis in ("n_models", "n_data", "gpu_slots", "budget_frac", "window_scale"):
        vals = sorted({r[axis] for r in rows})
        print(f"\n  {axis}:")
        print(f"    {'value':>8} {'vd vs lru':>12} {'prefetch gain':>15}")
        for v in vals:
            sub = [r for r in rows if r[axis] == v]
            print(f"    {v:>8} {statistics.mean(r['vd_vs_lru_pct'] for r in sub):>11.2f}% "
                  f"{statistics.mean(r['prefetch_gain_pct'] for r in sub):>14.2f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"accuracy": args.accuracy, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
