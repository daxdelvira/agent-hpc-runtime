#!/usr/bin/env python3
"""Turn the bench_p1_* artifacts into the two tables the paper needs.

TABLE 1 is compute-INDEPENDENT: size, load_warm, activated, expansion, s/GB,
io_share. These are the only numbers that rank candidates, because s/GB has no
`compute` term and is exactly what the value-density retention policy consumes.
io_share prints as `n/a` unless `rungs_verified_distinct` is true -- a cold and
a warm rung that cannot be PROVEN to be in different cache states differ by
run-to-run noise, and a plausible number from that is worse than an admitted
gap.

TABLE 2 is compute-DEPENDENT and exists to make the trap unmissable: activation
share is quoted per named compute, never alone.

TABLE 3 is the length-matched query pairs, which is the actual test of the
claim that HMMER's cost is hit-count independent.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def esc(v):
    """UniProt names contain `|`, which silently eats markdown table cells."""
    return str(v).replace("|", "\\|")


def verdict(rows):
    for r in rows:
        if r.get("rung") == "VERDICT":
            return r
    return None


def main(paths) -> int:
    runs = []
    for p in sorted(paths):
        try:
            rows = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        v = verdict(rows)
        if v:
            runs.append((os.path.basename(p), rows, v))
    if not runs:
        print("no artifacts found", file=sys.stderr)
        return 1

    print("## Table 1 — compute-independent (the ranking metrics)\n")
    print("| run | host | file GB | load_cold s | load_warm s | activated GB "
          "| expansion | **s/GB** | io_share | rungs distinct |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, rows, v in runs:
        label = v.get("label", name)
        io = v.get("io_share_of_cold")
        io = f"{100*io:.1f}%" if io is not None else "n/a"
        file_gb = round(v["activated_gb"] / v["expansion"], 3)
        print(f"| {label} | {v['host'].split('.')[0]} | {file_gb} "
              f"| {v['load_cold_s']} | {v['load_warm_s']} | {v['activated_gb']} "
              f"| {v['expansion']}x | **{v['s_per_gb_retained']}** | {io} "
              f"| {v.get('rungs_verified_distinct')} |")

    print("\n## Table 2 — compute-DEPENDENT (never quote one of these alone)\n")
    print("| run | compute | hits | compute s | activation share | "
          "retention speedup |")
    print("|---|---|---|---|---|---|")
    for name, rows, v in runs:
        label = v.get("label", name)
        cs = v.get("compute_s_by_compute") or {"phmmer_random200":
                                               v.get("compute_s")}
        hits = v.get("n_hits_by_compute") or {}
        sh = v.get("activation_share_by_compute") or {}
        sp = v.get("retention_speedup_by_compute") or {}
        for k in cs:
            print(f"| {label} | {esc(k)} | {hits.get(k, '?')} | {cs[k]} "
                  f"| {100*sh.get(k, float('nan')):.1f}% "
                  f"| {sp.get(k, '?')}x |")

    print("\n## Table 3 — length-matched query pairs "
          "(is search cost hit-count independent?)\n")
    print("| run | query | len | real hits | real s | rand hits | rand s | "
          "real/rand |")
    print("|---|---|---|---|---|---|---|---|")
    any_pair = False
    for name, rows, v in runs:
        for r in rows:
            if not str(r.get("rung", "")).startswith("r3s_pair"):
                continue
            any_pair = True
            print(f"| {r.get('label', name)} | {esc(r['query'])} "
                  f"| {r['query_len']} "
                  f"| {r['real_hits']} | {r['real_s']} | {r['randmatched_hits']} "
                  f"| {r['randmatched_s']} | {r['real_over_random']}x |")
    if not any_pair:
        print("| (none — run with --n-sampled-queries) | | | | | | | |")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("results/bench_p1_*.json"))
    raise SystemExit(main(args))
