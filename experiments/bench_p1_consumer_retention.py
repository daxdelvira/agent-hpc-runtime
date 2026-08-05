#!/usr/bin/env python3
"""
bench_p1_consumer_retention.py — the P1 go/no-go for a candidate consumer.

WHAT P1 ASKS, AND WHY IT IS THE FIRST GATE
------------------------------------------
The data-side mechanism (M2) is "hold the activated structure at R3 in a live
consumer process instead of rebuilding it per tool call." That mechanism only
EXISTS if the consumer can be kept alive holding the structure. LAMMPS happened
to qualify because `from lammps import lammps` keeps the spline table in-process.
Nothing guarantees the next candidate does, and a CLI-only consumer makes R3
retention inexpressible no matter how good the policy is.

So P1 splits in two, and they fail differently:

  P1a  the CONSUMER LIBRARY exposes an in-process API holding the activated
       structure.  HARD GATE — if this fails the candidate is dead, because we
       cannot fix someone else's tool.
  P1b  the AGENT FRAMEWORK can keep that process alive across tool calls.
       SOFT GATE — AtomAgents currently FAILS this (execution/runner.py:26-45
       does subprocess.run(sys.executable, script) per call, so the structure
       dies every invocation), but that is our own plumbing and is fixable.

This script tests P1a empirically. Passing the API check is not enough: a
library can expose an object that is cheap to rebuild, in which case retaining
it buys nothing and the candidate is uninteresting even though it "passes."

RUNGS (mirrors bench_activated_residency.py so the numbers are comparable)
-------------------------------------------------------------------------
  r1_load_cold   fresh process, page cache evicted   -> full ladder R0->R3
  r2_load_warm   fresh process, page cache warm      -> R1->R3 (isolates I/O)
  r3_reuse_live  SAME process, second query          -> R3 already paid
  activated size RSS delta across the load           -> expansion ratio

The verdict that matters is r2 -> r3. r1 - r2 is the I/O share, i.e. the most a
byte-oriented prefetcher could ever recover.

SCOPE NOTE. The FASTA here is generated, so its SIZE proves nothing about any
real workload — the plan's exogenous-size criterion is a separate question this
script does not address. Size is a knob HERE deliberately, because P1 is a
question about the MECHANISM (does retention save anything, and how much), and
the plan already established that s/GB is a format constant flat to within
1.00-1.16x across a 4x size range. Do not cite this file's byte counts as
evidence about a workload.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import random
import resource
import sys
import time

POSIX_FADV_DONTNEED = 4


# ---- page-cache helpers: lifted verbatim from bench_activated_residency.py --
def evict(path: str) -> None:
    """Drop `path` from the page cache. Requires no privileges."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    except AttributeError:
        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def evict_works(path: str) -> bool:
    """Does posix_fadvise(DONTNEED) actually evict `path`'s filesystem?

    IT DOES NOT ON LUSTRE, and it fails SILENTLY -- fadvise returns success and
    the pages stay resident. Measured 2026-08-05 on a 64 MB file, full read then
    evict:

        /storage/scratch1 (lustre)   1.000 -> 1.000    <-- no-op
        /storage/project  (nfs)      1.000 -> 0.000
        /tmp              (local)    1.000 -> 0.000

    Every cold-vs-warm number in this file is a DIFFERENCE between two rungs
    whose cache state is set by evict(). If evict is a no-op, the two rungs have
    the same cache state and their difference is run-to-run noise being reported
    as an I/O share. That is the same failure class as the L2 sleep result: a
    probe that silently did nothing, whose output still looks like data.

    Call this before trusting `io_share_of_cold`. The RETENTION numbers do not
    depend on it -- r2 (rebuild) and r3 (reuse) run in one process against
    whatever cache state exists, and r3 touches no files at all -- so a False
    here invalidates one column, not the experiment.
    """
    probe = path + ".evictprobe"
    try:
        with open(probe, "wb") as f:
            f.write(os.urandom(8 << 20))
        os.sync()
        with open(probe, "rb") as f:
            while f.read(1 << 20):
                pass
        if resident_fraction(probe) < 0.9:
            return False          # never got warm; cannot conclude either way
        evict(probe)
        return resident_fraction(probe) < 0.1
    except Exception:
        return False
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


def resident_fraction(path: str) -> float:
    """Fraction of `path` currently in the page cache, via mincore()."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = os.path.getsize(path)
    if size == 0:
        return 0.0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 2, fd, 0)   # PROT_READ, MAP_PRIVATE
        if addr in (0, 2 ** 64 - 1):
            return -1.0
        pagesz = os.sysconf("SC_PAGE_SIZE")
        npages = (size + pagesz - 1) // pagesz
        vec = (ctypes.c_ubyte * npages)()
        if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
            libc.munmap(ctypes.c_void_p(addr), size)
            return -1.0
        resident = sum(v & 1 for v in vec)
        libc.munmap(ctypes.c_void_p(addr), size)
        return resident / npages
    finally:
        os.close(fd)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


# ---- workload generation ----------------------------------------------------
_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_fasta(path: str, target_bytes: int, seed: int = 7) -> None:
    """A protein FASTA of roughly `target_bytes`. Sequence lengths follow a
    lognormal-ish spread so the record structure is not uniform; a perfectly
    uniform file would let the parser take a fast path real databases do not.

    Residues come from one large numpy draw rather than per-character
    rng.choice: the naive version needs ~1e9 Python-level calls for a 2 GB file
    and takes tens of minutes, which made generation cost more than the
    measurement. Draw once, slice per record.
    """
    import numpy as np
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    aa = np.frombuffer(_AA.encode(), dtype="S1")

    CHUNK = 64 << 20                      # residues per draw
    pool = aa[nprng.integers(0, len(_AA), CHUNK)].tobytes().decode()
    pos = 0

    written = 0
    buf = []
    buflen = 0
    with open(path, "w") as f:
        i = 0
        while written < target_bytes:
            n = max(30, int(rng.lognormvariate(5.7, 0.5)))
            if pos + n > len(pool):       # refill; records never span a draw
                pool = aa[nprng.integers(0, len(_AA), CHUNK)].tobytes().decode()
                pos = 0
            seq = pool[pos:pos + n]
            pos += n
            hdr = f">sp|P{i:07d}|PRT{i}_SYNTH Synthetic protein {i}\n"
            body = "\n".join(seq[j:j + 60] for j in range(0, len(seq), 60)) + "\n"
            buf.append(hdr); buf.append(body)
            buflen += len(hdr) + len(body)
            written += len(hdr) + len(body)
            i += 1
            if buflen >= (32 << 20):
                f.write("".join(buf)); buf.clear(); buflen = 0
        if buf:
            f.write("".join(buf))
    os.sync()


# ---- the probe --------------------------------------------------------------
def load_block(path: str):
    """Parse FASTA -> in-memory digital sequence block. This is the R1->R3
    transformation whose cost we are asking about."""
    from pyhmmer.easel import SequenceFile
    with SequenceFile(path, digital=True) as sf:
        return sf.read_block()


def build_query(alphabet):
    """A single-sequence query, built once; its cost is not what we measure."""
    from pyhmmer.easel import TextSequence
    rng = random.Random(11)
    seq = "".join(rng.choice(_AA) for _ in range(200))
    return TextSequence(name=b"query", sequence=seq).digitize(alphabet)


def search(query, block):
    from pyhmmer.plan7 import Pipeline
    pli = Pipeline(alphabet=block.alphabet)
    return pli.search_seq(query, block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True,
                    help="FASTA to use; generated if absent")
    ap.add_argument("--size-gb", type=float, default=2.0,
                    help="size to generate if --fasta does not exist")
    ap.add_argument("--out", default="results/bench_p1_consumer_retention.json")
    ap.add_argument("--mode", choices=("cold", "warm", "full"), default="full",
                    help="'full' runs cold+warm+reuse in one process. cold and "
                         "warm are for driving separate processes explicitly.")
    args = ap.parse_args()

    fasta = os.path.abspath(args.fasta)
    if not os.path.exists(fasta):
        print(f"generating {args.size_gb} GB FASTA at {fasta} ...", flush=True)
        make_fasta(fasta, int(args.size_gb * 1e9))
    nbytes = os.path.getsize(fasta)
    print(f"fasta: {nbytes/1e9:.3f} GB", flush=True)

    # Establish up front whether the cold/warm rungs can mean anything HERE.
    # On Lustre they cannot; see evict_works().
    cache_control = evict_works(fasta)
    _cc = ("WORKING" if cache_control else
           "UNAVAILABLE (fadvise is a no-op here — io_share suppressed)")
    print(f"page-cache control on this filesystem: {_cc}", flush=True)

    recs = []

    def rec(**kw):
        kw["t"] = time.time()
        kw["host"] = os.uname().nodename
        recs.append(kw)
        print("  " + json.dumps({k: v for k, v in kw.items()
                                 if k not in ("t", "host")}), flush=True)

    # --- r1: cold. Evict first so the ladder starts at R0. -------------------
    evict(fasta)
    frac = resident_fraction(fasta)
    rss0 = rss_gb()
    t0 = time.time()
    block = load_block(fasta)
    t_cold = time.time() - t0
    rss1 = rss_gb()
    alphabet = block.alphabet
    rec(rung="r1_load_cold", elapsed_s=round(t_cold, 3),
        resident_frac_before=round(frac, 4),
        n_sequences=len(block),
        activated_gb=round(rss1 - rss0, 3),
        file_gb=round(nbytes / 1e9, 3),
        expansion=round((rss1 - rss0) / (nbytes / 1e9), 3))

    query = build_query(alphabet)

    # First search against the freshly built block.
    t0 = time.time()
    hits1 = search(query, block)
    t_s1 = time.time() - t0
    rec(rung="r3a_search_on_fresh_block", elapsed_s=round(t_s1, 3),
        n_hits=len(hits1))

    # --- r3: reuse. Same process, same retained block, a NEW query. ----------
    # This is the mechanism: the load is not repeated because R3 is still paid.
    rng = random.Random(99)
    from pyhmmer.easel import TextSequence
    q2 = TextSequence(name=b"query2",
                      sequence="".join(rng.choice(_AA) for _ in range(200))
                      ).digitize(alphabet)
    t0 = time.time()
    hits2 = search(q2, block)
    t_reuse = time.time() - t0
    rec(rung="r3_reuse_live", elapsed_s=round(t_reuse, 3),
        n_hits=len(hits2),
        note="no reload: the DigitalSequenceBlock from r1 is still resident")

    # --- r2: warm. Rebuild from a hot page cache, as a per-call subprocess
    # would. The block is dropped first so this is a genuine rebuild.
    del block
    frac_warm = resident_fraction(fasta)
    rss2 = rss_gb()
    t0 = time.time()
    block2 = load_block(fasta)
    t_warm = time.time() - t0
    rec(rung="r2_load_warm", elapsed_s=round(t_warm, 3),
        resident_frac_before=round(frac_warm, 4),
        n_sequences=len(block2),
        note="fresh parse, page cache hot — what a per-call subprocess pays")

    # --- verdict -------------------------------------------------------------
    # Two SEPARATE questions, and conflating them is how the first version of
    # this script mislabelled a working mechanism as "WEAK":
    #
    #   EXPRESSIBILITY (P1a proper) — can the activated structure be held and
    #   reused at all?  Binary. It is answered by r3_reuse_live completing
    #   without a reload, and nothing about magnitude bears on it.
    #
    #   MAGNITUDE — is holding it WORTH anything?  That is governed by the ratio
    #   of activation cost to the per-call COMPUTE cost, because a retained
    #   consumer still pays the compute. LAMMPS was a 9.0x win only because its
    #   42.8 s parse dwarfed a 4.78 s reuse. A consumer whose per-call compute
    #   is comparable to its load can be perfectly retainable and still not
    #   worth retaining.
    #
    # `t_reuse` is a SEARCH on the retained block, so it is compute, not load.
    # Comparing t_warm against it directly (the old `t_warm > 5*t_reuse` test)
    # asked "is loading slower than searching?", which is not the question.
    # io_share is the ONLY quantity that depends on evict() having worked, so it
    # is reported as null rather than as a number when it could not be
    # established. A plausible-looking 12.7% from two rungs with identical cache
    # state is worse than an admitted gap.
    io_share = ((t_cold - t_warm) / t_cold
                if (cache_control and t_cold > 0) else None)
    cold_call = t_warm + t_reuse      # per-call consumer: rebuild, then compute
    warm_call = t_reuse               # retained consumer: compute only
    speedup = cold_call / warm_call if warm_call > 0 else float("inf")
    activation_share = t_warm / cold_call if cold_call > 0 else 0.0
    verdict = {
        "rung": "VERDICT",
        "load_cold_s": round(t_cold, 3),
        "load_warm_s": round(t_warm, 3),
        "compute_s": round(t_reuse, 3),
        "page_cache_control": cache_control,
        "io_share_of_cold": (round(io_share, 4) if io_share is not None
                             else None),
        "activation_share_of_call": round(activation_share, 4),
        "retention_speedup": round(speedup, 2),
        "activated_gb": round(rss1 - rss0, 3),
        "expansion": round((rss1 - rss0) / (nbytes / 1e9), 3),
        "s_per_gb_retained": round(t_warm / max(rss1 - rss0, 1e-9), 3),
        # Expressibility is the hard gate and it is already decided by here.
        "P1a_expressible": "PASS",
        # Magnitude is advisory. The 0.5 line is where retention halves a call.
        "magnitude": ("STRONG" if activation_share >= 0.5 else
                      "MODERATE" if activation_share >= 0.25 else "MARGINAL"),
    }
    rec(**verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"P1a expressible: {verdict['P1a_expressible']} "
          f"(block retained and reused with no reload)")
    print(f"magnitude: {verdict['magnitude']} — activation is "
          f"{100*activation_share:.1f}% of a cold call "
          f"({t_warm:.2f} s load vs {t_reuse:.2f} s compute), "
          f"retention speedup {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
