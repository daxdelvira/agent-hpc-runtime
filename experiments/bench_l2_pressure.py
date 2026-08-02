#!/usr/bin/env python3
"""
bench_l2_pressure.py — is L2 sleep's speed an artifact of an idle page cache?

THE CLAIM UNDER TEST
--------------------
bench_activation_ladder measured, on an A100 node with 2015 GB of host RAM:

    L1 wake  3.87 s   holding +27.1 GiB of process RAM
    L2 wake  3.64 s   holding  +0.0 GiB

Read naively that says L2 strictly dominates L1 — same latency, no standing
cost. But L2 DISCARDS the weights and re-reads them on wake, so "+0.0 GiB" only
means the bytes moved somewhere `free` does not count as used: the OS page
cache. On a node with 2 TB of RAM and a 15 GB model there is no pressure, so
that cache never gets evicted and L2 looks free.

Under real pressure — several large models competing for host RAM, which is
exactly the regime gate (b) probed when the node died at k=3 — those pages go
away and an L2 wake becomes a disk read.

This isolates that with posix_fadvise(DONTNEED) instead of trying to fill 2 TB
of RAM: it is precise, instant, and uses the same eviction primitive the runtime
already relies on (measured effective: 8.68x read slowdown on project NFS).

    L2 wake, cache WARM  -> should reproduce ~3.6 s
    L2 wake, cache COLD  -> if this stays ~3.6 s, L2 really is cheap and the
                            caveat is wrong. If it jumps toward a warm-boot
                            shard read, L2's low cost is contingent on cache
                            residency and must be reported that way.

L1 is measured in the same loop as a control: L1 holds weights in PROCESS
memory, so evicting the page cache should NOT affect it. If L1 also slows down,
the eviction is hitting something else and neither number means what we think.
"""
from __future__ import annotations

import argparse
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
from runtime.prefetch.model_cache_prefetch import evict_model_cache  # noqa: E402


def wait_sleeping(port: int, want: bool, timeout: float = 300.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if bool((_get_json(port, "/is_sleeping") or {}).get("is_sleeping")) == want:
                return
        except Exception:
            pass
        time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--port", type=int, default=8207)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_l2_pressure_{host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def rec(**kw):
        kw["t"] = time.time(); kw["host"] = host
        rows.append(kw); out.write_text(json.dumps(rows, indent=2))
        print(f"  -> {json.dumps({k: v for k, v in kw.items() if k != 't'})}", flush=True)

    proc = None
    try:
        print("[boot] starting engine (page cache is warm from prior runs)", flush=True)
        proc = launch(args.model_path, args.port, gpus, args.tp, Path("logs/ladder"))
        boot = wait_ready(args.port, timeout=1800)
        sanity_generate(args.port, SERVED)
        rec(phase="boot", boot_s=round(boot, 2), ram_gib=host_ram_used_gib())

        # (level, evict_before_wake, label)
        for level, evict, label in [
            (2, False, "L2_cache_warm"),
            (2, True,  "L2_cache_EVICTED"),
            (1, True,  "L1_cache_EVICTED"),   # control: L1 keeps weights in RSS
        ]:
            before = host_ram_used_gib()
            t0 = time.perf_counter()
            _post(args.port, f"/sleep?level={level}")
            wait_sleeping(args.port, True)
            sleep_s = time.perf_counter() - t0
            slept_ram = host_ram_used_gib()

            n = freed = 0
            if evict:
                n, freed = evict_model_cache(args.model_path)

            t1 = time.perf_counter()
            _post(args.port, "/wake_up")
            wait_sleeping(args.port, False)
            wake_s = time.perf_counter() - t1
            gen = sanity_generate(args.port, SERVED)

            rec(phase=label, level=level, evicted=bool(evict),
                sleep_s=round(sleep_s, 2), wake_s=round(wake_s, 2),
                first_gen_s=round(gen, 2),
                ram_held_gib=round(slept_ram - before, 1),
                evicted_shards=n, evicted_bytes=freed,
                gpu_mib=gpu_mem_used_mib(gpus))
    finally:
        if proc is not None:
            kill(proc)

    print("\n" + "=" * 68)
    print("L2 UNDER PAGE-CACHE PRESSURE")
    print("=" * 68)
    g = {r["phase"]: r for r in rows if "wake_s" in r}
    warm = g.get("L2_cache_warm"); cold = g.get("L2_cache_EVICTED")
    l1 = g.get("L1_cache_EVICTED")
    for k in ("L2_cache_warm", "L2_cache_EVICTED", "L1_cache_EVICTED"):
        if k in g:
            print(f"  {k:<20} wake {g[k]['wake_s']:7.2f} s   "
                  f"RAM held {g[k]['ram_held_gib']:+7.1f} GiB")
    if warm and cold and warm["wake_s"] > 0:
        r = cold["wake_s"] / warm["wake_s"]
        print(f"\n  L2 cold/warm wake ratio = {r:.2f}x")
        if r < 1.5:
            print("  => L2 is genuinely cheap; its low RAM cost is NOT a cache artifact.")
        else:
            print("  => L2's speed DEPENDS on page-cache residency. Its '+0.0 GiB' is")
            print("     bookkeeping, not a free lunch: the weights still occupy RAM,")
            print("     just as evictable cache. Report it as contingent.")
    if l1:
        print(f"\n  L1 control (weights in process RSS, cache evicted): "
              f"{l1['wake_s']:.2f} s")
        print("  L1 should be unaffected by eviction. If it moved, the eviction hit")
        print("  something else and neither number can be trusted.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
