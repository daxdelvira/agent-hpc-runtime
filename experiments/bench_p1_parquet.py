#!/usr/bin/env python3
"""
bench_p1_parquet.py — P1/P2 for the Parquet -> Arrow consumer.

WHY A SEPARATE FILE. bench_p1_consumer_retention.py is written around
SequenceFile/DigitalSequenceBlock. Rather than refactor a working instrument
mid-flight, this imports its page-cache helpers (evict, evict_works,
resident_fraction, rss_gb) so there is exactly ONE implementation of the tricky
parts -- the same reuse pattern bench_activated_residency.py already follows.

WHAT IS BEING RE-MEASURED. An earlier reading gave Parquet a 73% activation
share, the best of any non-incumbent candidate. That number came from an 82 MB
file, run inline on a LOGIN NODE, with no artifact written -- it is recorded in
the candidate register as `measured-unpersisted` and cannot currently be cited.
It also predates the finding that shared-filesystem timings are untrustworthy.

TWO METHODOLOGY RULES THIS FILE OBEYS, BOTH PAID FOR IN RERUNS
--------------------------------------------------------------
1. NODE-LOCAL NVMe ONLY. posix_fadvise(DONTNEED) is a silent no-op on Lustre,
   and Lustre read throughput was measured collapsing 16x WITHIN a single 8 GB
   read (304 -> 18.7 krec/s), which manufactured a fake 91.8% activation share
   for pyhmmer. All I/O here happens under --work-dir, which must be node-local.
2. THE FILE IS GENERATED WHERE IT IS MEASURED. Copying it in from shared storage
   reintroduces the same contention on the write side.

THE HONEST DIFFICULTY WITH THIS METRIC
--------------------------------------
Activation share is `load / (load + compute)`, and `compute` is WHATEVER THE TOOL
CALL DOES. That is not a property of the format. A tool that sums one column
makes activation look enormous; a tool that sorts the table makes it look small.
LAMMPS's 90% is against a force evaluation, HMMER's 48% against a database
search -- those are not the same denominator, and comparing them as if they were
is the main way this metric can mislead.

So this measures THREE compute intensities and reports the share as a RANGE, not
a point. A candidate whose range straddles the interesting region is a candidate
whose verdict depends on what the agent actually asks it to do, and that has to
be said out loud rather than resolved by picking a flattering denominator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_p1_consumer_retention import (          # noqa: E402
    evict, evict_works, resident_fraction, rss_gb,
)


def make_parquet(path: str, target_bytes: int, seed: int = 5) -> None:
    """A Parquet file of roughly `target_bytes`, written in row-group chunks.

    Column mix is deliberately heterogeneous, because Parquet's decode cost is
    per-encoding: dictionary-encoded strings, RLE-friendly low-cardinality ints,
    and incompressible floats exercise different paths. A single-dtype file would
    measure one decoder and generalise to nothing.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    cats = np.array([f"category_{i:03d}" for i in range(200)])
    ROWS = 4_000_000                      # per row group
    writer = None
    try:
        while True:
            tbl = pa.table({
                "id":      pa.array(rng.integers(0, 2**40, ROWS)),
                "value":   pa.array(rng.random(ROWS)),               # incompressible
                "bucket":  pa.array(rng.integers(0, 50, ROWS).astype("int32")),
                "label":   pa.array(cats[rng.integers(0, len(cats), ROWS)]),
                "flag":    pa.array(rng.integers(0, 2, ROWS).astype("bool_")),
            })
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema,
                                          compression="snappy",
                                          use_dictionary=True)
            writer.write_table(tbl)
            if os.path.getsize(path) >= target_bytes:
                break
    finally:
        if writer is not None:
            writer.close()
    os.sync()


def load_table(path: str):
    """Parquet on disk -> Arrow Table in memory. The R1->R3 transformation:
    decompress, undo dictionary/RLE encoding, materialise columnar buffers."""
    import pyarrow.parquet as pq
    return pq.read_table(path)


def compute(tbl, kind: str):
    """One tool call's worth of work against the retained table."""
    import pyarrow.compute as pc
    if kind == "scan":                    # cheapest: one aggregate
        return pc.sum(tbl.column("value"))
    if kind == "groupby":                 # moderate: hash aggregation
        return tbl.group_by("label").aggregate([("value", "mean"),
                                                ("bucket", "sum")])
    if kind == "sort":                    # expensive: full sort
        return tbl.sort_by([("value", "ascending")])
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.environ.get("TMPDIR", "/tmp"),
                    help="MUST be node-local; see the module docstring")
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the generated file")
    args = ap.parse_args()

    path = os.path.join(args.work_dir, f"p1_parquet_{args.size_gb:g}gb.parquet")
    recs = []

    def rec(**kw):
        kw["t"] = time.time(); kw["host"] = os.uname().nodename
        recs.append(kw)
        print("  " + json.dumps({k: v for k, v in kw.items()
                                 if k not in ("t", "host")}), flush=True)

    print(f"host      {os.uname().nodename}")
    print(f"work dir  {args.work_dir}")
    if not os.path.exists(path):
        print(f"generating ~{args.size_gb} GB parquet ...", flush=True)
        make_parquet(path, int(args.size_gb * 1e9))
    nbytes = os.path.getsize(path)
    print(f"parquet   {nbytes/1e9:.3f} GB", flush=True)

    cache_control = evict_works(path)
    print(f"page-cache control: {'WORKING' if cache_control else 'UNAVAILABLE'}",
          flush=True)
    if not cache_control:
        print("  !! io_share will be suppressed. Is --work-dir really "
              "node-local? On Lustre this is expected and the run is still "
              "valid for retention, just not for the I/O split.", flush=True)

    # --- r1 cold -------------------------------------------------------------
    evict(path)
    frac = resident_fraction(path)
    rss0 = rss_gb(); t0 = time.time()
    tbl = load_table(path)
    t_cold = time.time() - t0
    rss1 = rss_gb()
    inmem = tbl.nbytes / 1e9
    rec(rung="r1_load_cold", elapsed_s=round(t_cold, 3),
        resident_frac_before=round(frac, 4),
        n_rows=tbl.num_rows, n_cols=tbl.num_columns,
        file_gb=round(nbytes / 1e9, 3),
        arrow_nbytes_gb=round(inmem, 3),
        rss_delta_gb=round(rss1 - rss0, 3),
        expansion=round(inmem / (nbytes / 1e9), 3))

    # --- r3 reuse: compute against the RETAINED table, three intensities -----
    # One intensity failing must not cost the whole size point. `sort` raises
    # inside Arrow's take() at 448M rows (8 GB), which killed the 8 and 32 GB
    # runs of the first attempt after the cheap intensities had already
    # succeeded -- their results were lost with the process.
    comp = {}
    for kind in ("scan", "groupby", "sort"):
        try:
            t0 = time.time(); compute(tbl, kind); dt = time.time() - t0
            comp[kind] = dt
            rec(rung=f"r3_reuse_live__{kind}", elapsed_s=round(dt, 3),
                note="no reload: the Arrow table from r1 is still resident")
        except Exception as e:
            rec(rung=f"r3_reuse_live__{kind}", elapsed_s=None,
                error=f"{type(e).__name__}: {str(e)[:160]}",
                note="intensity unavailable at this size; other intensities stand")
    if not comp:
        print("no compute intensity succeeded — cannot form a verdict")
        return 1

    # --- r2 warm rebuild: what a per-call subprocess pays --------------------
    del tbl
    frac_warm = resident_fraction(path)
    t0 = time.time()
    tbl2 = load_table(path)
    t_warm = time.time() - t0
    rec(rung="r2_load_warm", elapsed_s=round(t_warm, 3),
        resident_frac_before=round(frac_warm, 4),
        note="fresh decode, page cache hot — what a per-call subprocess pays")

    # --- verdict: a RANGE over compute intensity, not a point ----------------
    shares = {k: t_warm / (t_warm + v) for k, v in comp.items()}
    speedups = {k: (t_warm + v) / v for k, v in comp.items()}
    io_share = ((t_cold - t_warm) / t_cold
                if (cache_control and t_cold > 0) else None)
    verdict = {
        "rung": "VERDICT",
        "load_cold_s": round(t_cold, 3),
        "load_warm_s": round(t_warm, 3),
        "compute_s": {k: round(v, 3) for k, v in comp.items()},
        "page_cache_control": cache_control,
        "io_share_of_cold": (round(io_share, 4) if io_share is not None else None),
        "activation_share_by_compute": {k: round(v, 4) for k, v in shares.items()},
        "retention_speedup_by_compute": {k: round(v, 2) for k, v in speedups.items()},
        "activation_share_range": [round(min(shares.values()), 4),
                                   round(max(shares.values()), 4)],
        "arrow_gb": round(inmem, 3),
        "expansion": round(inmem / (nbytes / 1e9), 3),
        "s_per_gb_retained": round(t_warm / max(inmem, 1e-9), 3),
        "P1a_expressible": "PASS",
    }
    rec(**verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\nwrote {args.out}")
    lo, hi = verdict["activation_share_range"]
    print(f"activation share {100*lo:.1f}% (vs sort) .. {100*hi:.1f}% (vs scan) "
          f"— the spread IS the finding, not noise")
    if not args.keep:
        try:
            os.remove(path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
