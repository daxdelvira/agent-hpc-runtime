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
    # MUST be set: without argtypes ctypes marshals the length as c_int, which
    # OVERFLOWS above 2 GB and makes mincore fail, returning -1.0 for exactly the
    # large files this benchmark exists to measure. Copied from
    # bench_activated_residency.py:81 without this line, and it went unnoticed
    # because -1.0 is only visible in a field nothing downstream consumes.
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_ubyte)]
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
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            libc.munmap(ctypes.c_void_p(addr), size)
            return -1.0
        resident = sum(v & 1 for v in vec)
        libc.munmap(ctypes.c_void_p(addr), size)
        return resident / npages
    finally:
        os.close(fd)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def storage_bytes() -> int:
    """Bytes this process has really caused to be fetched from the STORAGE layer.

    `/proc/self/io: read_bytes` is the one instrument here that does not depend on
    any other instrument telling the truth. `rchar` counts bytes delivered to
    read(2) -- satisfied from page cache or not -- while `read_bytes` counts only
    what actually came off the device. Their ratio IS the cache hit rate.

    THIS IS THE CHECK THAT WAS MISSING. Every withdrawn number in this project
    came from a cold rung that was not cold, and both existing instruments can be
    fooled: fadvise is a hint the kernel may ignore, and mincore reports the
    client's belief about its own mapping. read_bytes is downstream of both --
    if the bytes did not come off the device, the rung was warm, whatever
    fadvise returned and whatever mincore believes.

    Measured 2026-08-05 on a 256 MB file, cold then warm:

        local NVMe   mincore 0.000 -> 268.4 MB from storage;  warm -> 0.0 MB
        project NFS  mincore 0.000 -> 268.4 MB from storage;  warm -> 0.0 MB
        lustre       mincore 0.562 ->  83.9 MB from storage;  warm -> 0.0 MB

    Note the Lustre row: eviction there is PARTIAL AND NONDETERMINISTIC, not the
    clean no-op recorded earlier from a 64 MB probe (1.000 -> 1.000). That is
    worse than a no-op, because a partial eviction still produces a plausible
    cold-vs-warm difference -- just an unreproducible one.

    Works on all three filesystems, so it validates shared-storage runs too.
    Returns -1 if unavailable (the field is Linux-specific and can be restricted).

    NOT usable for mmap-based consumers' page faults -- read()-based readers show
    majflt=0 throughout (verified). Use ru_majflt for mmap consumers instead.
    """
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split(":")[1])
    except OSError:
        pass
    return -1


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


def build_query(alphabet, seed: int = 11, length: int = 200, name: str = "query"):
    """A single-sequence query, built once; its cost is not what we measure."""
    from pyhmmer.easel import TextSequence
    rng = random.Random(seed)
    seq = "".join(rng.choice(_AA) for _ in range(length))
    return TextSequence(name=name.encode(), sequence=seq).digitize(alphabet)


def real_queries(path: str, ids, alphabet):
    """Pull named REAL protein sequences out of `path` to use as queries.

    WHY THIS EXISTS. Every pyhmmer number in this project was measured with a
    RANDOM 200-residue query, which finds 8-75 hits in a database of millions.
    The claim that this does not matter -- that HMMER's cost is dominated by the
    MSV/Viterbi filter sweeping every sequence and is therefore largely
    hit-count independent -- was `asserted`, never measured, and the whole ~48%
    activation share rests on it. A real conserved protein queried against a
    real database produces hit counts orders of magnitude larger, on the SAME
    retained block, which turns the assertion into a controlled comparison.

    Matching is on the accession or the entry name as they appear in a UniProt
    FASTA header (`>sp|P69905|HBA_HUMAN ...`), so either form works. Missing IDs
    are skipped rather than fatal -- a query that is absent from the release
    must not cost the run.
    """
    from pyhmmer.easel import SequenceFile
    wanted = set(ids)
    found = {}
    with SequenceFile(path, digital=False) as sf:
        for seq in sf:
            # pyhmmer 0.12 returns `str` here; older builds returned `bytes`.
            # Normalise, because getting this wrong silently yields zero
            # matches and the real-query comparison quietly does not happen.
            name = seq.name
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            name = name or ""
            parts = name.split("|")
            keys = (parts[1], parts[2], name) if len(parts) >= 3 else (name,)
            for key in keys:
                if key in wanted and key not in found:
                    found[key] = seq.digitize(alphabet)
            if len(found) == len(wanted):
                break
    return found


def sampled_queries(path: str, k: int, stride: int, alphabet):
    """`k` real sequences spread through `path`, as an UNBIASED query sample.

    The named-accession list is deliberately biased -- hemoglobin, ubiquitin and
    ATP synthase beta are among the most conserved proteins there are, so they
    sit at the expensive end of the search-cost distribution. Quoting only those
    would replace one wrong number with another. This draws every `stride`-th
    record instead, which is what an arbitrary real query looks like.
    """
    from pyhmmer.easel import SequenceFile
    out = []
    with SequenceFile(path, digital=False) as sf:
        for i, seq in enumerate(sf):
            if i % stride:
                continue
            nm = seq.name
            if isinstance(nm, bytes):
                nm = nm.decode(errors="replace")
            out.append((nm, seq.digitize(alphabet)))
            if len(out) >= k:
                break
    return out


def load_hmm(path: str, name: str | None):
    """First HMM in `path`, or the one whose NAME/ACC matches `name`.

    An HMM query is what `hmmsearch` actually does, so it is the realistic
    per-call compute for this consumer; a single sequence query is `phmmer`.
    They are DIFFERENT COMPUTES and therefore different denominators for
    activation share -- which is the whole point of reporting several.
    """
    from pyhmmer.plan7 import HMMFile

    def s(v):
        if v is None:
            return ""
        return v.decode(errors="replace") if isinstance(v, bytes) else str(v)

    with HMMFile(path) as hf:
        first = None
        for hmm in hf:
            if first is None:
                first = hmm
            if name is None:
                return hmm
            if s(hmm.name) == name or s(hmm.accession).split(".")[0] == name:
                return hmm
        return first          # requested profile absent: use one, don't fail
    return None


def search(query, block):
    from pyhmmer.plan7 import Pipeline
    pli = Pipeline(alphabet=block.alphabet)
    return pli.search_seq(query, block)


def search_hmm(hmm, block):
    from pyhmmer.plan7 import Pipeline
    pli = Pipeline(alphabet=block.alphabet)
    return pli.search_hmm(hmm, block)


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
    ap.add_argument("--label", default=None,
                    help="free-text tag recorded in every row, e.g. "
                         "'uniref50_real' vs 'synthetic'")
    ap.add_argument("--query-fasta", default=None,
                    help="FASTA to draw REAL query sequences from (Swiss-Prot). "
                         "Each becomes an extra named compute on the SAME "
                         "retained block, so hit-count sensitivity is measured "
                         "rather than assumed.")
    ap.add_argument("--query-ids", default="P69905,P0CG48,P06576",
                    help="comma-separated UniProt accessions or entry names")
    ap.add_argument("--n-sampled-queries", type=int, default=0,
                    help="how many arbitrary real sequences to draw from "
                         "--query-fasta as queries, each paired with a "
                         "length-matched random control")
    ap.add_argument("--sample-stride", type=int, default=97391,
                    help="take every Nth record when sampling queries")
    ap.add_argument("--hmm", default=None,
                    help="Pfam-A.hmm to draw a profile query from; an HMM "
                         "search is hmmsearch's real compute")
    ap.add_argument("--hmm-name", default="Pkinase",
                    help="HMM NAME or accession to use; first HMM if absent")
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
        if args.label:
            kw["label"] = args.label
        recs.append(kw)
        print("  " + json.dumps({k: v for k, v in kw.items()
                                 if k not in ("t", "host")}), flush=True)

    # --- r1: cold. Evict first so the ladder starts at R0. -------------------
    evict(fasta)
    frac = resident_fraction(fasta)
    rss0 = rss_gb()
    sb0 = storage_bytes()
    t0 = time.time()
    block = load_block(fasta)
    t_cold = time.time() - t0
    sb_cold = storage_bytes() - sb0
    rss1 = rss_gb()
    alphabet = block.alphabet
    rec(rung="r1_load_cold", elapsed_s=round(t_cold, 3),
        resident_frac_before=round(frac, 4),
        storage_mb=round(sb_cold / 1e6, 1),
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

    # --- extra computes on the SAME retained block ---------------------------
    # Two questions, both answered here because they need the same expensive
    # block and would be uncomparable across processes:
    #   (a) is per-call search cost hit-count dependent? (the asserted claim)
    #   (b) what is the SPREAD of activation share across plausible computes?
    #       Parquet showed 28x on identical data, so a single share is not a
    #       property of the format and must never be quoted alone.
    computes = {"phmmer_random200": (round(t_reuse, 3), len(hits2))}

    if args.query_fasta and os.path.exists(args.query_fasta):
        ids = [i for i in args.query_ids.split(",") if i]
        try:
            qs = real_queries(args.query_fasta, ids, alphabet)
        except Exception as e:                       # pragma: no cover
            qs = {}
            rec(rung="r3q_ERROR", error=repr(e))
        for nm, q in qs.items():
            t0 = time.time()
            h = search(q, block)
            dt = time.time() - t0
            rec(rung=f"r3q_phmmer_real:{nm}", elapsed_s=round(dt, 3),
                n_hits=len(h), query_len=len(q),
                note="real conserved protein as query; same retained block")
            computes[f"phmmer_real:{nm}"] = (round(dt, 3), len(h))
        missing = [i for i in ids if i not in qs]
        if missing:
            rec(rung="r3q_missing", ids=missing)

        # Unbiased sample, each paired with a LENGTH-MATCHED random query.
        # Query length and homology both drive HMMER's cost and they are
        # confounded in any single comparison: a real query is usually longer
        # than 200 aa, so a bare real-vs-random gap could be pure length. The
        # matched pair holds length fixed, so whatever gap remains is the part
        # attributable to sequences surviving the MSV/Viterbi filters -- which
        # is exactly the "hit-count independence" claim under test.
        if args.n_sampled_queries > 0:
            try:
                samples = sampled_queries(args.query_fasta,
                                          args.n_sampled_queries,
                                          args.sample_stride, alphabet)
            except Exception as e:                   # pragma: no cover
                samples = []
                rec(rung="r3s_ERROR", error=repr(e))
            for j, (nm, q) in enumerate(samples):
                L = len(q)
                t0 = time.time()
                h = search(q, block)
                dt_real = time.time() - t0
                rq = build_query(alphabet, seed=1000 + j, length=L,
                                 name=f"rand{L}")
                t0 = time.time()
                hr = search(rq, block)
                dt_rand = time.time() - t0
                rec(rung=f"r3s_pair[{j}]", query=nm, query_len=L,
                    real_s=round(dt_real, 3), real_hits=len(h),
                    randmatched_s=round(dt_rand, 3), randmatched_hits=len(hr),
                    real_over_random=(round(dt_real / dt_rand, 3)
                                      if dt_rand > 0 else None))
                computes[f"phmmer_sampled[{j}]:{nm}"] = (round(dt_real, 3),
                                                         len(h))
                computes[f"phmmer_randlen{L}[{j}]"] = (round(dt_rand, 3),
                                                       len(hr))

    if args.hmm and os.path.exists(args.hmm):
        try:
            hmm = load_hmm(args.hmm, args.hmm_name)
        except Exception as e:                       # pragma: no cover
            hmm = None
            rec(rung="r3h_ERROR", error=repr(e))
        if hmm is not None:
            nm = hmm.name or "?"
            nm = nm.decode(errors="replace") if isinstance(nm, bytes) else str(nm)
            t0 = time.time()
            h = search_hmm(hmm, block)
            dt = time.time() - t0
            rec(rung=f"r3h_hmmsearch:{nm}", elapsed_s=round(dt, 3),
                n_hits=len(h), hmm_len=hmm.M,
                note="profile HMM query — this is what hmmsearch actually does")
            computes[f"hmmsearch:{nm}"] = (round(dt, 3), len(h))

    # --- r2: warm. Rebuild from a hot page cache, as a per-call subprocess
    # would. The block is dropped first so this is a genuine rebuild.
    del block
    frac_warm = resident_fraction(fasta)
    rss2 = rss_gb()
    sb1 = storage_bytes()
    t0 = time.time()
    block2 = load_block(fasta)
    t_warm = time.time() - t0
    sb_warm = storage_bytes() - sb1
    rec(rung="r2_load_warm", elapsed_s=round(t_warm, 3),
        resident_frac_before=round(frac_warm, 4),
        storage_mb=round(sb_warm / 1e6, 1),
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
    # Gate io_share on what the DEVICE actually did, not on what fadvise
    # claimed. The cold rung must have pulled most of the file off storage and
    # the warm rung must have pulled almost none; otherwise the two rungs were
    # not in different cache states and their difference is noise.
    cold_ok = sb_cold >= 0.5 * nbytes
    warm_ok = 0 <= sb_warm <= 0.1 * nbytes
    rungs_distinct = bool(cache_control and cold_ok and warm_ok)
    io_share = ((t_cold - t_warm) / t_cold
                if (rungs_distinct and t_cold > 0) else None)
    cold_call = t_warm + t_reuse      # per-call consumer: rebuild, then compute
    warm_call = t_reuse               # retained consumer: compute only
    speedup = cold_call / warm_call if warm_call > 0 else float("inf")
    activation_share = t_warm / cold_call if cold_call > 0 else 0.0
    # The compute-INDEPENDENT metrics (s/GB, expansion) are single numbers; the
    # compute-DEPENDENT ones are a dict keyed by the compute, because there is
    # no such thing as "the" activation share.
    share_by_compute = {k: round(t_warm / (t_warm + v[0]), 4)
                        for k, v in computes.items() if t_warm + v[0] > 0}
    speedup_by_compute = {k: round((t_warm + v[0]) / v[0], 2)
                          for k, v in computes.items() if v[0] > 0}
    verdict = {
        "rung": "VERDICT",
        "compute_s_by_compute": {k: v[0] for k, v in computes.items()},
        "n_hits_by_compute": {k: v[1] for k, v in computes.items()},
        "activation_share_by_compute": share_by_compute,
        "retention_speedup_by_compute": speedup_by_compute,
        "activation_share_range": ([min(share_by_compute.values()),
                                    max(share_by_compute.values())]
                                   if share_by_compute else None),
        "load_cold_s": round(t_cold, 3),
        "load_warm_s": round(t_warm, 3),
        "compute_s": round(t_reuse, 3),
        "page_cache_control": cache_control,
        "storage_mb_cold": round(sb_cold / 1e6, 1),
        "storage_mb_warm": round(sb_warm / 1e6, 1),
        "rungs_verified_distinct": rungs_distinct,
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
