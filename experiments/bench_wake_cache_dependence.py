#!/usr/bin/env python3
"""
bench_wake_cache_dependence.py — is an L2 wake fast because of the page cache,
and does it defer cost into the first inferences?

THE TWO QUESTIONS
-----------------
A live eval trial recorded a level-2 wake of 1.093 s against a 1600.6 s cold
boot of the same model on the same node.  vLLM reported that sleep as

    sleep freed 41.21 GiB ... 0.00 GiB is backed up in CPU
                and the rest 41.21 GiB is discarded directly

so nothing was held in vLLM's own host-side buffer.  Yet 146.82 GB of weights
at tp=4 is 36.7 GB per GPU, and 36.7 GB in 1.105 s is 33.2 GB/s per GPU --
ABOVE PCIe Gen4 x16's ~31.5 GB/s theoretical.  The reported wake therefore
cannot be a complete host-to-device copy.  Two explanations survive:

  Q1  the weights are served from the OS PAGE CACHE (warm because the cold boot
      just read them), so a wake on a cold cache would be far slower; or
  Q2  the restore is LAZY -- the wake returns after re-establishing mappings and
      pages fault in on demand, so the cost is deferred into later inference.

These are not mutually exclusive and the experiment separates them.

WHY THE EXISTING BENCH CANNOT ANSWER THIS
-----------------------------------------
bench_sleep_wake.py runs its L1 cycles BEFORE its L2 cycles in the same process
and never releases L1's host-side backup.  Its host RAM reads 218.2 GiB at the
L1 sleep and *still* 218.2 GiB at the L2 sleep -- so its "L2" woke with L1's
copy resident.  Its L2 numbers are contaminated and must not be used.

It also measures host RAM as (MemTotal - MemAvailable), and MemAvailable counts
reclaimable page cache as AVAILABLE.  Weights sitting in page cache are
therefore invisible to that metric -- which is exactly why L2 appeared to hold
"+0.0 GiB".  This script measures page-cache residency of the weight files
directly with mincore() instead of inferring it.

DESIGN
------
One process, one engine, four rungs, each preceded by an explicit cache state:

  1 wake_warm    sleep L2, wake.  Page cache left as the boot left it.
  2 wake_cold    sleep L2, fadvise(DONTNEED) every weight shard, wake.
  3 gen_after_wake     sustained generation immediately after a wake.
  4 gen_steady         the same generation once the engine has been serving.

Rungs 1 vs 2 answer Q1.  Rungs 3 vs 4 answer Q2: if the restore is lazy, the
first burst after a wake is slower than the steady state.

mincore() residency is recorded at every rung so a failed eviction is visible
rather than silently reported as "cold".  NOTE the known hazard: on Lustre,
fadvise was measured leaving 56.2% of a file resident.  These weights live on
the user's scratch; the script reports what it achieved and does not assume.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.bench_sleep_wake import (  # noqa: E402
    _get_json, _post, gpu_mem_used_mib, host_ram_used_gib, sanity_generate,
    wait_ready,
)
from experiments.bench_activation_ladder import SERVED, kill, launch  # noqa: E402

POSIX_FADV_DONTNEED = 4


def weight_files(model_dir: str) -> list[str]:
    out = []
    for root, _, fs in os.walk(model_dir):
        for f in fs:
            if f.endswith((".safetensors", ".bin")):
                out.append(os.path.join(root, f))
    return sorted(out)


def evict(paths: list[str]) -> int:
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return n


def resident_fraction(paths: list[str]) -> float:
    """Fraction of all weight bytes currently in the page cache, via mincore().

    Direct libc mmap: Python's mmap object refuses a ctypes pointer on a
    PROT_READ mapping ("underlying buffer is not writable").
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_ubyte)]
    PROT_READ, MAP_SHARED = 0x1, 0x01
    MAP_FAILED = ctypes.c_void_p(-1).value
    pagesize = os.sysconf("SC_PAGE_SIZE")

    res = tot = 0
    for p in paths:
        try:
            size = os.path.getsize(p)
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        addr = None
        try:
            addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
            if addr in (None, MAP_FAILED):
                continue
            npages = (size + pagesize - 1) // pagesize
            vec = (ctypes.c_ubyte * npages)()
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                continue
            res += sum(v & 1 for v in vec)
            tot += npages
        finally:
            if addr not in (None, MAP_FAILED):
                libc.munmap(ctypes.c_void_p(addr), size)
            os.close(fd)
    return (res / tot) if tot else float("nan")


def wait_sleeping(port: int, want: bool, timeout: float = 600.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if bool((_get_json(port, "/is_sleeping") or {}).get("is_sleeping")) == want:
                return
        except Exception:
            pass
        time.sleep(0.5)


def timed_generation(port: int, n: int, max_tokens: int) -> tuple[float, float]:
    """Return (total_s, tokens_per_s) for n sequential generations."""
    t0 = time.perf_counter()
    for _ in range(n):
        sanity_generate(port, SERVED)
    el = time.perf_counter() - t0
    return el, (n * max_tokens) / el if el > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=8231)
    ap.add_argument("--gen-n", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_wake_cache_{host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    shards = weight_files(args.model_path)
    nbytes = sum(os.path.getsize(p) for p in shards)
    rows: list[dict] = []

    def rec(**kw):
        kw["t"] = time.time(); kw["host"] = host
        rows.append(kw); out.write_text(json.dumps(rows, indent=2))
        print(f"  -> {json.dumps({k: v for k, v in kw.items() if k != 't'})}", flush=True)

    print(f"[bench] {len(shards)} shards, {nbytes/1e9:.2f} GB, tp={args.tp}", flush=True)
    proc = None
    try:
        boot0 = time.perf_counter()
        proc = launch(args.model_path, args.port, gpus, args.tp, Path("logs/wakecache"))
        boot = wait_ready(args.port, timeout=3600)
        sanity_generate(args.port, SERVED)
        rec(rung="cold_boot", boot_s=round(boot, 2),
            resident_frac_after=round(resident_fraction(shards), 4),
            ram_used_gib=host_ram_used_gib(), gpu_mib=gpu_mem_used_mib(gpus),
            bytes=nbytes, tp=args.tp)

        # ---- rung 3/4 baseline: steady-state generation BEFORE any sleep ----
        el, tps = timed_generation(args.port, args.gen_n, 4)
        rec(rung="gen_steady", total_s=round(el, 3), tok_per_s=round(tps, 2),
            n=args.gen_n)

        for label, do_evict in (("wake_warm", False), ("wake_cold", True)):
            t0 = time.perf_counter()
            _post(args.port, "/sleep?level=2")
            wait_sleeping(args.port, True)
            sleep_s = time.perf_counter() - t0
            frac_slept = resident_fraction(shards)
            ram_slept = host_ram_used_gib()
            gpu_slept = gpu_mem_used_mib(gpus)

            n_ev = 0
            if do_evict:
                n_ev = evict(shards)
            frac_before_wake = resident_fraction(shards)

            t1 = time.perf_counter()
            _post(args.port, "/wake_up")
            wait_sleeping(args.port, False)
            wake_s = time.perf_counter() - t1

            # Immediately after the wake — this is the lazy-restore probe.
            el_a, tps_a = timed_generation(args.port, args.gen_n, 4)

            rec(rung=label, sleep_s=round(sleep_s, 3), wake_s=round(wake_s, 3),
                evicted_shards=n_ev,
                resident_frac_when_slept=round(frac_slept, 4),
                resident_frac_before_wake=round(frac_before_wake, 4),
                resident_frac_after_wake=round(resident_fraction(shards), 4),
                ram_used_gib_slept=ram_slept, gpu_mib_slept=gpu_slept,
                gpu_mib_awake=gpu_mem_used_mib(gpus),
                gen_after_wake_total_s=round(el_a, 3),
                gen_after_wake_tok_per_s=round(tps_a, 2))
    finally:
        if proc is not None:
            kill(proc)

    print("\n" + "=" * 72)
    print("WAKE: PAGE-CACHE DEPENDENCE AND LAZY RESTORE")
    print("=" * 72)
    g = {r["rung"]: r for r in rows}
    warm, cold = g.get("wake_warm"), g.get("wake_cold")
    steady = g.get("gen_steady")
    if warm and cold:
        print(f"  wake, cache WARM   {warm['wake_s']:7.3f} s   "
              f"(resident before wake {warm['resident_frac_before_wake']:.1%})")
        print(f"  wake, cache COLD   {cold['wake_s']:7.3f} s   "
              f"(resident before wake {cold['resident_frac_before_wake']:.1%})")
        if warm["wake_s"] > 0:
            r = cold["wake_s"] / warm["wake_s"]
            print(f"\n  cold/warm ratio = {r:.2f}x")
            print("  => L2's speed DEPENDS on page-cache residency; report it as"
                  if r >= 1.5 else
                  "  => L2 is fast even from a cold cache; the page cache is NOT"
                  " the mechanism.")
            if r >= 1.5:
                print("     contingent on the weights still being cached.")
    if steady and warm:
        print(f"\n  generation tok/s steady-state      {steady['tok_per_s']:7.2f}")
        print(f"  generation tok/s just after wake   {warm['gen_after_wake_tok_per_s']:7.2f}")
        print("  A materially LOWER rate after the wake means the restore is"
              " lazy and\n  the reported wake_s understates the true cost.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
