#!/usr/bin/env python3
"""
diag_p1_superlinear.py — why does pyhmmer's FASTA load scale 47x for 4x data?

THE OBSERVATION
---------------
bench_p1_consumer_retention.py on Lustre scratch:

    size    load(warm)   compute   activation share   speedup
    0.2 GB     1.02 s     1.36 s        42.8%          1.75x
    2   GB    13.65 s    14.46 s        48.6%          1.94x
    8   GB   639.43 s    57.44 s        91.8%         12.13x

Compute scales cleanly (3.97x for 4x data). LOAD scales 47x for 4x data --
146 MB/s collapsing to 12.5 MB/s. Taken at face value this promotes pyhmmer from
a 49% candidate to a 92% one, level with LAMMPS, and makes it the lead candidate.

It is not being taken at face value. A 12x collapse in parse rate is far more
likely to be an artifact than a property of the format, and the paper cannot rest
on a number whose mechanism is unknown. Three hypotheses, each with a signature:

  H1  LUSTRE READ CONTENTION on a shared node. The 8 GB run read from
      /storage/scratch1 while other jobs hammered the same OSTs.
      SIGNATURE: staging the identical file to node-local NVMe restores ~146 MB/s.
  H2  ALLOCATOR / STRUCTURE DEGRADATION at ~20M live DigitalSequence objects.
      SIGNATURE: seconds-per-million-records climbs steadily WITHIN a single load,
      independent of where the bytes came from.
  H3  MEMORY PRESSURE against the cgroup limit.
      SIGNATURE: RSS approaches the allocation and the climb starts there.

H1 and H3 are environmental and mean the 91.8% is spurious. H2 would be REAL and
is the more interesting outcome: activation cost superlinear in artifact size is
exactly the regime where a cost-aware policy beats a size-aware one, and it would
be measured rather than assumed.

METHOD
------
Same file, two filesystems, incremental instrumentation on both. Reads records in
batches so per-batch rate and RSS are visible, rather than one opaque total.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def cgroup_limit_gb() -> float:
    """The allocation we are actually held to, for H3."""
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v not in ("max", ""):
                return int(v) / (1024 ** 3)
        except Exception:
            pass
    return float("inf")


def timed_load(path: str, batch: int, label: str) -> dict:
    """Load a FASTA into a DigitalSequenceBlock, timing every `batch` records.

    Uses SequenceFile.read() in a loop rather than read_block() so the cost can be
    attributed incrementally -- read_block() is one opaque C call and cannot
    distinguish H2 from H1.
    """
    from pyhmmer.easel import SequenceFile, DigitalSequenceBlock, Alphabet

    abc = Alphabet.amino()
    seqs = []
    marks = []
    t0 = time.time()
    last_t, last_n = t0, 0

    with SequenceFile(path, digital=True, alphabet=abc) as sf:
        for seq in sf:
            seqs.append(seq)
            if len(seqs) % batch == 0:
                now = time.time()
                marks.append({
                    "records": len(seqs),
                    "cum_s": round(now - t0, 3),
                    "batch_s": round(now - last_t, 3),
                    "rate_krec_s": round((len(seqs) - last_n) / max(now - last_t, 1e-9) / 1e3, 1),
                    "rss_gb": round(rss_gb(), 3),
                })
                last_t, last_n = now, len(seqs)

    total = time.time() - t0
    block = DigitalSequenceBlock(abc, seqs)
    return {
        "label": label,
        "path": path,
        "total_load_s": round(total, 3),
        "n_records": len(seqs),
        "file_gb": round(os.path.getsize(path) / 1e9, 3),
        "throughput_mb_s": round(os.path.getsize(path) / 1e6 / max(total, 1e-9), 1),
        "rss_after_gb": round(rss_gb(), 3),
        "marks": marks,
        "_block": block,          # kept alive so RSS reflects retention
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="source file (on shared storage)")
    ap.add_argument("--batch", type=int, default=1_000_000)
    ap.add_argument("--local-dir", default=os.environ.get("TMPDIR", "/tmp"),
                    help="node-local staging dir; must be where fadvise works")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"host          {os.uname().nodename}")
    print(f"cgroup limit  {cgroup_limit_gb():.1f} GB")
    print(f"source        {args.fasta}")

    recs = {"host": os.uname().nodename,
            "cgroup_limit_gb": round(cgroup_limit_gb(), 1),
            "runs": []}

    # --- run A: straight from shared storage (reproduces the observation) ----
    print("\n=== A: load from SHARED storage (as the original sweep did) ===",
          flush=True)
    a = timed_load(args.fasta, args.batch, "shared")
    del a["_block"]
    print(f"  {a['total_load_s']:.1f} s, {a['throughput_mb_s']} MB/s, "
          f"RSS {a['rss_after_gb']} GB", flush=True)
    recs["runs"].append(a)

    # --- run B: identical file staged to node-local NVMe (tests H1) ---------
    local = os.path.join(args.local_dir, os.path.basename(args.fasta))
    print(f"\n=== B: staging to {local} ===", flush=True)
    t = time.time()
    shutil.copyfile(args.fasta, local)
    os.sync()
    print(f"  copied in {time.time()-t:.1f} s", flush=True)
    print("=== B: load from NODE-LOCAL NVMe ===", flush=True)
    b = timed_load(local, args.batch, "local_nvme")
    del b["_block"]
    print(f"  {b['total_load_s']:.1f} s, {b['throughput_mb_s']} MB/s, "
          f"RSS {b['rss_after_gb']} GB", flush=True)
    recs["runs"].append(b)
    try:
        os.remove(local)
    except OSError:
        pass

    # --- verdict ------------------------------------------------------------
    speedup = a["total_load_s"] / max(b["total_load_s"], 1e-9)
    first = a["marks"][0]["rate_krec_s"] if a["marks"] else 0
    last = a["marks"][-1]["rate_krec_s"] if a["marks"] else 0
    decay = first / max(last, 1e-9)
    verdict = {
        "local_vs_shared_speedup": round(speedup, 2),
        "shared_rate_first_batch_krec_s": first,
        "shared_rate_last_batch_krec_s": last,
        "within_load_rate_decay": round(decay, 2),
        "peak_rss_gb": round(max(a["rss_after_gb"], b["rss_after_gb"]), 3),
        "cgroup_limit_gb": round(cgroup_limit_gb(), 1),
    }
    # H1 if local is much faster; H2 if the rate decays WITHIN a load regardless.
    if speedup > 3:
        verdict["conclusion"] = ("H1 — storage. The 91.8% was Lustre contention; "
                                 "pyhmmer's real activation share is the ~49% "
                                 "measured at 2 GB.")
    elif decay > 3:
        verdict["conclusion"] = ("H2 — structure. Parse rate degrades with live "
                                 "object count on BOTH filesystems. Superlinear "
                                 "activation is real and is a finding.")
    else:
        verdict["conclusion"] = ("neither clean — inspect marks[] before "
                                 "concluding anything.")
    recs["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\n{json.dumps(verdict, indent=2)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
