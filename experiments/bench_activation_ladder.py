#!/usr/bin/env python3
"""
bench_activation_ladder.py — measure the FULL activation ladder on one model.

THE OPEN QUESTION
-----------------
We have hard numbers for three rungs and a hole in the middle:

    cold boot        ~1000 s   (measured: qwen_72b 1000.4 s; gate (a) 978.5 s)
    warm-cache boot   ~190 s   (INFERRED: 88 s p05 shard read + ~100 s init)
    L1 sleep -> wake  1.5-2.1 s (gate (b))
    L2 sleep -> wake  UNKNOWN   <-- gate (c) never ran; the n=4 fleet died at k=3

L2 matters because it is the only rung that might dominate page-cache warming:
the process stays alive (so CUDA init, memory profiling and graph capture are
all skipped) while the weights are discarded (so it should cost little standing
host RAM). If that holds, L2 is the right middle tier and page-cache warming is
never the best choice for a model you are willing to keep a process for. If it
does not hold, warming wins the middle and the residency policy is different.

WHY A SMALL MODEL ON A CHEAP GPU
--------------------------------
The absolute seconds here will NOT transfer to 32B/72B. The RATIOS and the
ORDERING will, and that is what is actually unknown. Running this on 2 idle
RTX 6000s for an hour costs nothing we need elsewhere, whereas spending a
contended Blackwell hold on a measurement that never appears in a figure is a
bad trade.

WHAT IT MEASURES, per rung: time to serve a token again, host RAM held while in
that state, and GPU memory held. Cold is forced with posix_fadvise(DONTNEED)
via the existing evict_model_cache() helper (verified effective: 8.68x read
slowdown on project NFS, 7.85x on the L40S node).

USAGE
    python3 experiments/bench_activation_ladder.py \
        --model-path /path/to/snapshot --gpus 0 --tp 1 --repeat 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.bench_sleep_wake import (  # noqa: E402
    _get_json, _post, gpu_mem_used_mib, host_ram_used_gib, sanity_generate,
    wait_ready,
)
from runtime.prefetch.model_cache_prefetch import (  # noqa: E402
    evict_model_cache, list_model_shards,
)

VLLM_PY = "/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean128/bin/python"
SERVED = "ladder-model"


def launch(model_path: str, port: int, gpus: list[int], tp: int,
           logdir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["VLLM_SERVER_DEV_MODE"] = "1"        # exposes /sleep /wake_up /is_sleeping
    # Sleep mode and the expandable-segments allocator are mutually exclusive
    # ("Expandable segments are not compatible with memory pool"). Strip it
    # unconditionally — this cost four failed smoke attempts on 2026-07-29.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    cmd = [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--port", str(port),
           "--tensor-parallel-size", str(tp),
           "--gpu-memory-utilization", "0.85",
           "--max-model-len", "4096",
           "--dtype", "float16", "--enforce-eager",
           "--enable-sleep-mode",
           "--served-model-name", SERVED]
    logdir.mkdir(parents=True, exist_ok=True)
    log = (logdir / f"vllm_{port}.log").open("a")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    subprocess.run(["pkill", "-u", os.environ.get("USER", ""), "-f", "VLLM::"],
                   check=False)
    time.sleep(8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the sleep/wake rungs N times (they are cheap)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    logdir = Path("logs/ladder")
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_activation_ladder_{host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def rec(**kw):
        # Persist after EVERY record: this runs under a preemptible embers
        # allocation and a partial ladder is still worth having.
        kw["t"] = time.time()
        kw["host"] = host
        rows.append(kw)
        out.write_text(json.dumps(rows, indent=2))
        print(f"  -> {json.dumps({k: v for k, v in kw.items() if k != 't'})}",
              flush=True)

    shards = list_model_shards(args.model_path)
    total_b = sum(p.stat().st_size for p in shards)
    print(f"model: {args.model_path}")
    print(f"  {len(shards)} shards, {total_b/1e9:.2f} GB, gpus={gpus} tp={args.tp}")
    rec(rung="env", n_shards=len(shards), bytes=total_b,
        gpus=gpus, tp=args.tp, ram_idle_gib=host_ram_used_gib())

    proc = None
    try:
        # ---- rung 1: COLD (page cache dropped) --------------------------
        n, freed = evict_model_cache(args.model_path)
        print(f"\n[cold] evicted {n} shards ({freed/1e9:.2f} GB) from page cache")
        ram0 = host_ram_used_gib()
        t0 = time.perf_counter()
        proc = launch(args.model_path, args.port, gpus, args.tp, logdir)
        ready = wait_ready(args.port, timeout=1800)
        gen = sanity_generate(args.port, SERVED)
        rec(rung="cold_boot", boot_s=round(ready, 2), first_gen_s=round(gen, 2),
            ram_gib=host_ram_used_gib(), ram_delta_gib=round(host_ram_used_gib()-ram0, 1),
            gpu_mib=gpu_mem_used_mib(gpus), evicted_shards=n, evicted_bytes=freed)

        # ---- rung 2: WARM boot (same process torn down, cache now hot) ---
        kill(proc); proc = None
        ram0 = host_ram_used_gib()
        t0 = time.perf_counter()
        proc = launch(args.model_path, args.port, gpus, args.tp, logdir)
        ready = wait_ready(args.port, timeout=1800)
        gen = sanity_generate(args.port, SERVED)
        rec(rung="warm_boot", boot_s=round(ready, 2), first_gen_s=round(gen, 2),
            ram_gib=host_ram_used_gib(), gpu_mib=gpu_mem_used_mib(gpus),
            note="new process, weights served from page cache")

        # ---- rungs 3 & 4: L1 and L2 sleep/wake ---------------------------
        for level in (1, 2):
            for i in range(args.repeat):
                base_ram = host_ram_used_gib()
                base_gpu = gpu_mem_used_mib(gpus)
                t0 = time.perf_counter()
                _post(args.port, f"/sleep?level={level}")
                # /sleep returns before the transition is fully settled; poll.
                for _ in range(600):
                    try:
                        if (_get_json(args.port, "/is_sleeping") or {}).get("is_sleeping"):
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)
                sleep_s = time.perf_counter() - t0
                slept_ram = host_ram_used_gib()
                slept_gpu = gpu_mem_used_mib(gpus)

                t1 = time.perf_counter()
                _post(args.port, "/wake_up")
                wake_s = time.perf_counter() - t1
                gen = sanity_generate(args.port, SERVED)

                rec(rung=f"sleep_l{level}", iteration=i,
                    sleep_s=round(sleep_s, 2), wake_s=round(wake_s, 2),
                    first_gen_s=round(gen, 2),
                    ram_before_gib=base_ram, ram_slept_gib=slept_ram,
                    # POSITIVE = RAM held while asleep (L1 parks weights in host
                    # RAM). NEGATIVE/zero = RAM returned (expected for L2).
                    ram_held_gib=round(slept_ram - base_ram, 1),
                    gpu_mib_before=base_gpu, gpu_mib_slept=slept_gpu)
    finally:
        if proc is not None:
            kill(proc)

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 72)
    print("ACTIVATION LADDER")
    print("=" * 72)
    cold = next((r for r in rows if r["rung"] == "cold_boot"), None)
    warm = next((r for r in rows if r["rung"] == "warm_boot"), None)
    for lvl in (1, 2):
        rs = [r for r in rows if r["rung"] == f"sleep_l{lvl}"]
        if not rs:
            continue
        w = min(r["wake_s"] for r in rs)
        held = max(r["ram_held_gib"] for r in rs)
        gpu = rs[0]["gpu_mib_slept"]
        print(f"  L{lvl} wake        {w:8.2f} s   host RAM held {held:+7.1f} GiB   "
              f"gpu {gpu}")
    if warm:
        print(f"  warm boot     {warm['boot_s']:8.2f} s   (new process, cache hot)")
    if cold:
        print(f"  cold boot     {cold['boot_s']:8.2f} s   "
              f"({cold['evicted_bytes']/1e9:.1f} GB evicted first)")
    if cold and warm and warm["boot_s"] > 0:
        print(f"\n  cold/warm ratio = {cold['boot_s']/warm['boot_s']:.2f}x "
              "(what page-cache warming buys)")
    l2 = [r for r in rows if r["rung"] == "sleep_l2"]
    if warm and l2:
        b = min(r["wake_s"] for r in l2)
        print(f"  warm/L2 ratio   = {warm['boot_s']/b:.1f}x "
              "(what keeping the PROCESS alive buys on top)")
        print("\n  If L2 wake << warm boot AND L2 holds little RAM, L2 dominates")
        print("  page-cache warming for the middle tier and the residency policy")
        print("  should never choose warming for a model it can keep a process for.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
