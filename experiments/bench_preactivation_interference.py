#!/usr/bin/env python3
"""
bench_preactivation_interference.py — E3: does speculative pre-activation steal
throughput from the agent's foreground work?

WHY IT MATTERS
--------------
Insight B says small, transformation-bound data resources are worth holding at R3,
and getting them there speculatively means running a parse in the background while
the agent works. That is only free if the parse does not contend with the foreground.

`results/bench_format_activation.csv` reports warm_cpu_per_wall = 0.996-0.998 for
every format at every size, i.e. activation is essentially SINGLE-THREADED. On a
12-CPU allocation that predicts one background pre-activation costs 1 core of 12 and
the foreground should be untouched. Predicted, not measured — this measures it.

METHOD
------
Foreground: one ASCII parse (the most CPU-bound format in the sweep, 22.0 s/GB).
Background: K concurrent ASCII parses of a SEPARATE file, started first and kept
running for the whole foreground measurement.
Report foreground wall at K = 0, 1, 2, 4, 8 and the slowdown vs K=0.

A flat curve up to K = ncpu-1 means pre-activation is free and the policy question is
purely about RAM. A rising curve means pre-activation has a CPU price that the
scheduler must also budget, which would be a genuinely different system.

CPU-ONLY. No GPU, no allocation needed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def make_ascii(path: Path, n_values: int) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    rng = np.random.default_rng(12345)
    chunk = 2_000_000
    with open(path, "w") as f:
        written = 0
        while written < n_values:
            k = min(chunk, n_values - written)
            np.savetxt(f, rng.random(k, dtype=np.float64), fmt="%.10f")
            written += k


def parse_once(path: str) -> float:
    t0 = time.perf_counter()
    a = np.loadtxt(path, dtype=np.float64)
    el = time.perf_counter() - t0
    del a
    return el


def _bg_worker(path: str, stop: mp.Event) -> None:  # type: ignore[name-defined]
    """Parse in a loop until told to stop — a stand-in for continuous staging."""
    while not stop.is_set():
        try:
            parse_once(path)
        except Exception:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/preact_interference")
    ap.add_argument("--n-values", type=int, default=6_000_000,
                    help="values per file (~114 MB of ASCII at the default)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--levels", default="0,1,2,4,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    fg = wd / "fg.txt"
    bg = wd / "bg.txt"
    print("generating test data…", flush=True)
    make_ascii(fg, args.n_values)
    make_ascii(bg, args.n_values)

    ncpu = len(os.sched_getaffinity(0))
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_preactivation_interference_{host}.json")
    rows: list[dict] = []

    def rec(**kw):
        rows.append(kw)
        print(json.dumps(kw), flush=True)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2))
        except OSError:
            pass

    rec(rung="env", host=host, cpus_available=ncpu,
        fg_bytes=fg.stat().st_size, bg_bytes=bg.stat().st_size,
        n_values=args.n_values, reps=args.reps)

    # Warm the page cache for both files so we measure ACTIVATION contention,
    # not I/O contention — the whole point of Insight A is that these are
    # different costs and must not be conflated.
    parse_once(str(fg)); parse_once(str(bg))

    baseline = None
    for k in [int(x) for x in args.levels.split(",")]:
        stop = mp.Event()
        procs = [mp.Process(target=_bg_worker, args=(str(bg), stop), daemon=True)
                 for _ in range(k)]
        for p in procs:
            p.start()
        if k:
            time.sleep(3.0)   # let the background reach steady state

        times = [parse_once(str(fg)) for _ in range(args.reps)]

        stop.set()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        med = statistics.median(times)
        if baseline is None:
            baseline = med
        rec(rung=f"bg_{k}", background_parsers=k,
            fg_median_s=round(med, 3),
            fg_min_s=round(min(times), 3), fg_max_s=round(max(times), 3),
            slowdown_vs_idle=round(med / baseline, 3),
            oversubscribed=bool(k + 1 > ncpu))

    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
