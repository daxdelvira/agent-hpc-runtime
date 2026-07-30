#!/usr/bin/env python3
"""
bench_residency_preflight.py — Stage-0 gates for the constrained-residency regime.

The new experimental regime (N backing models, room for only M < N) rests on
four hardware assumptions that have never been measured.  Every one of them can
invalidate the topology, and three of them are cheap to test.  This benchmark
runs them all in one allocation and prints a pass/fail table.  Nothing in
Stages 1-7 of the residency plan should be built before gate (a) and gate (b)
pass, because a failure there forces a different topology, not a code change.

Gates
-----
(a) SINGLE-GPU FEASIBILITY.  Can a 32B-class model serve at tp=1 on one 96 GB
    Blackwell GPU (or tp=2 on 2x L40S)?  The whole point is N=4 engines with a
    sweepable resident count M; that needs one model per GPU.  tp=1 also
    sidesteps the 64-attention-head divisibility constraint that has been
    forcing 4-GPU pools.  FAIL here => N=4 is not achievable on this hardware.

(b) N SIMULTANEOUS LEVEL-1 SLEEPS.  Exactly one slept engine has ever been
    measured (bench_sleep_wake.py, 2026-07-29: wake 1.51 s vs 524.7 s cold
    boot).  Level-1 sleep parks weights in host RAM: ~64 GB per 32B engine, so
    4 of them want ~256 GB against a 256 GB cgroup.  This gate sleeps k = 1..N
    engines at once and reports host-RAM growth and wake time vs k.  FAIL (or a
    wake time that degrades with k) => the residency ladder needs level 2 as
    the default parked state, not level 1.

(c) LEVEL-2 ACTUALLY RETURNS HOST RAM.  Level 2 is supposed to discard weights
    entirely and re-read them on wake.  If L1 and L2 have the same host-RAM
    footprint, the sleep-level dimension of the policy is vacuous and should be
    deleted from the plan rather than swept.

(d) posix_fadvise(DONTNEED) EFFICACY ON PROJECT NFS.  The data-tier ablation
    needs to evict page cache to measure a cold tier honestly.  fadvise is
    verified on Lustre; on NFS it is advisory and may be a no-op, in which case
    every "cold" data measurement is silently a page-cache hit.

Gate (d) needs no GPU and runs first so a GPU-side failure still leaves it
measured.

Usage (on a compute node, inside the allocation):
    python experiments/bench_residency_preflight.py                # all gates
    python experiments/bench_residency_preflight.py --gates d      # NFS only
    python experiments/bench_residency_preflight.py --n 4 --tp 1
    python experiments/bench_residency_preflight.py --n 3 --tp 2   # L40S

NOT part of the eval tree — measurement only, no trace/meta artifacts.
Results: results/bench_residency_preflight_<host>.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model_configs import MODELS_CHEMGRAPH_SWAP  # noqa: E402
from bench_sleep_wake import (  # noqa: E402
    _get_json,
    _post,
    gpu_mem_used_mib,
    host_ram_used_gib,
    sanity_generate,
    wait_ready,
)

# Wake-time ceiling (s) above which gate (b) is considered degraded.  The
# single-engine measurement was 1.51 s; 30 s still beats a 130-700 s cold boot
# by an order of magnitude, so anything under it keeps the regime interesting.
WAKE_CEILING_S = 30.0
# Fraction of a level-1 footprint that level-2 must give back to count as
# genuinely freeing host RAM (gate c).
L2_RETURN_FRACTION = 0.5
# Gate (d): a DONTNEED'd re-read must be at least this many times slower than a
# page-cache-warm re-read to count as effective eviction.
FADVISE_SLOWDOWN = 3.0


# --------------------------------------------------------------------------
# environment probes
# --------------------------------------------------------------------------
def cgroup_mem_limit_gib() -> float | None:
    """Memory ceiling of this job's cgroup, or None if unlimited/undetectable."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = ""
            for line in f:
                parts = line.strip().split(":")
                if parts[0] == "0":            # cgroup v2 unified
                    rel = parts[2]
                    break
        for path in (Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.max",
                     Path("/sys/fs/cgroup/memory.max")):
            if path.exists():
                raw = path.read_text().strip()
                return None if raw == "max" else int(raw) / 1024 ** 3
        v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if v1.exists():
            val = int(v1.read_text().strip())
            # v1 reports a sentinel near 2^63 for "unlimited"
            return None if val > 2 ** 60 else val / 1024 ** 3
    except Exception:
        pass
    return None


def _nvidia_smi(argv: list[str]) -> str:
    """nvidia-smi output, or '' where the tool is absent (e.g. a login node,
    where gate (d) is still perfectly runnable)."""
    try:
        return subprocess.run(["nvidia-smi", *argv], capture_output=True,
                              text=True, check=False).stdout
    except FileNotFoundError:
        return ""


def gpu_total_mib() -> list[int]:
    out = _nvidia_smi(["--query-gpu=memory.total",
                       "--format=csv,noheader,nounits"])
    return [int(x) for x in out.split()] if out.strip() else []


def gpu_names() -> list[str]:
    out = _nvidia_smi(["-L"])
    return [l.strip() for l in out.strip().splitlines() if l.strip()]


# --------------------------------------------------------------------------
# multi-engine harness
# --------------------------------------------------------------------------
def engine_cfg(base: dict, idx: int, tp: int, base_port: int) -> dict:
    """One engine of the fleet: its own contiguous GPU block and its own port."""
    cfg = dict(base)
    cfg["gpus"] = list(range(idx * tp, idx * tp + tp))
    cfg["tensor_parallel_size"] = tp
    cfg["port"] = base_port + idx
    return cfg


def launch_engine(cfg: dict, idx: int, logdir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg["gpus"])
    env["VLLM_SERVER_DEV_MODE"] = "1"      # exposes /sleep /wake_up /is_sleeping
    # Sleep mode and the expandable-segments allocator are mutually exclusive
    # ("Expandable segments are not compatible with memory pool") — this bit
    # cost four failed smoke attempts on 2026-07-29.  Strip it unconditionally.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    # Each engine needs a distinct served name so a misrouted request is an
    # error rather than a silent hit on the wrong engine.
    extra = list(cfg.get("extra_args", []))
    for i, a in enumerate(extra):
        if a == "--served-model-name":
            extra[i + 1] = f"{extra[i + 1]}-r{idx}"
    cmd = [
        cfg["python_bin"], "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model_name"],
        "--port", str(cfg["port"]),
        "--tensor-parallel-size", str(cfg["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--enable-sleep-mode",
        *extra,
    ]
    log = logdir / f"bench_residency_engine{idx}.log"
    return subprocess.Popen(cmd, env=env, stdout=open(log, "ab"),
                            stderr=subprocess.STDOUT)


def served_name(cfg: dict, idx: int) -> str:
    ex = cfg.get("extra_args", [])
    for i, a in enumerate(ex):
        if a == "--served-model-name":
            return f"{ex[i + 1]}-r{idx}"
    return cfg["model_name"]


def sleep_engine(port: int, level: int) -> float:
    t0 = time.perf_counter()
    _post(port, f"/sleep?level={level}")
    deadline = time.perf_counter() + 300
    while not _get_json(port, "/is_sleeping").get("is_sleeping"):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"engine :{port} never reported is_sleeping")
        time.sleep(0.5)
    return time.perf_counter() - t0


def wake_engine(port: int) -> float:
    t0 = time.perf_counter()
    _post(port, "/wake_up", timeout=1800)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# gate (d) — fadvise on NFS
# --------------------------------------------------------------------------
def gate_d_fadvise(target_dir: Path, size_gib: float, rec) -> dict:
    """Warm the page cache, DONTNEED it, and see whether the re-read got slower.

    Three reads of the same file: cold (just written, partially cached), warm
    (should be page-cache speed), and post-DONTNEED.  Effective eviction means
    the third read falls back toward device speed.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"fadvise_canary_{os.getpid()}.bin"
    nbytes = int(size_gib * 1024 ** 3)
    chunk = b"\xa5" * (1 << 20)

    def read_all() -> float:
        t0 = time.perf_counter()
        with open(path, "rb", buffering=0) as f:
            while f.read(1 << 22):
                pass
        return time.perf_counter() - t0

    def dontneed() -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    try:
        with open(path, "wb", buffering=0) as f:
            for _ in range(nbytes >> 20):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

        dontneed()
        cold_s = read_all()
        warm_s = read_all()
        dontneed()
        evicted_s = read_all()

        gib = nbytes / 1024 ** 3
        warm_bw = gib / warm_s if warm_s else 0.0
        evicted_bw = gib / evicted_s if evicted_s else 0.0
        slowdown = evicted_s / warm_s if warm_s else 0.0
        ok = slowdown >= FADVISE_SLOWDOWN
        res = {
            "gate": "d_fadvise_nfs", "pass": ok,
            "target_dir": str(target_dir), "size_gib": round(gib, 2),
            "cold_read_s": round(cold_s, 2),
            "warm_read_s": round(warm_s, 2),
            "evicted_read_s": round(evicted_s, 2),
            "warm_gb_per_s": round(warm_bw, 2),
            "evicted_gb_per_s": round(evicted_bw, 2),
            "slowdown_x": round(slowdown, 2),
            "threshold_x": FADVISE_SLOWDOWN,
            "note": ("DONTNEED evicts — cold-tier data measurements are honest"
                     if ok else
                     "DONTNEED appears to be a NO-OP on this filesystem: every "
                     "'cold' data-tier measurement would silently be a page-cache "
                     "hit.  Cold reads must be forced another way (O_DIRECT, or "
                     "a distinct uncached copy per trial)."),
        }
    except Exception as exc:                                  # noqa: BLE001
        res = {"gate": "d_fadvise_nfs", "pass": False,
               "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    rec(res)
    return res


# --------------------------------------------------------------------------
# gates (a) (b) (c) — the GPU fleet
# --------------------------------------------------------------------------
def gates_abc(args, rec) -> list[dict]:
    base = MODELS_CHEMGRAPH_SWAP[args.model]
    out: list[dict] = []
    procs: list[subprocess.Popen] = []
    cfgs = [engine_cfg(base, i, args.tp, args.base_port) for i in range(args.n)]
    needed = args.n * args.tp
    have = len(gpu_names())
    if have < needed:
        res = {"gate": "a_single_gpu_feasibility", "pass": False,
               "error": f"need {needed} GPUs (n={args.n} x tp={args.tp}), "
                        f"{have} visible"}
        rec(res)
        return [res]

    ram_idle = host_ram_used_gib()
    logdir = _HERE.parent / "logs"
    logdir.mkdir(exist_ok=True)

    # ---- gate (a): boot the whole fleet in parallel -----------------------
    # Parallel, not serial: serial cold boots at 130-700 s each would eat the
    # hold before gate (b) ran, and the fleet has to coexist anyway.
    t0 = time.perf_counter()
    for i, cfg in enumerate(cfgs):
        procs.append(launch_engine(cfg, i, logdir))
    boots: list[dict] = []
    fleet_ok = True
    try:
        for i, cfg in enumerate(cfgs):
            try:
                boot_s = wait_ready(cfg["port"], cfg.get("load_timeout", 3600))
                gen_s = sanity_generate(cfg["port"], served_name(cfg, i))
                boots.append({"engine": i, "gpus": cfg["gpus"],
                              "boot_s": round(boot_s, 1),
                              "sanity_gen_s": round(gen_s, 2), "ok": True})
            except Exception as exc:                          # noqa: BLE001
                fleet_ok = False
                boots.append({"engine": i, "gpus": cfg["gpus"], "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"})
        fleet_boot_s = time.perf_counter() - t0
        ram_resident = host_ram_used_gib()
        res_a = {
            "gate": "a_single_gpu_feasibility", "pass": fleet_ok,
            "model": args.model, "n": args.n, "tp": args.tp,
            "gpu_total_mib": gpu_total_mib()[:needed],
            "fleet_boot_s_wallclock": round(fleet_boot_s, 1),
            "engines": boots,
            "gpu_mem_mib_resident": gpu_mem_used_mib(list(range(needed))),
            "host_ram_gib_idle": round(ram_idle, 1),
            "host_ram_gib_all_resident": round(ram_resident, 1),
            "note": ("all engines serve at this topology"
                     if fleet_ok else
                     "at least one engine failed to boot/serve — N=%d at tp=%d "
                     "is NOT achievable on this hardware and the residency "
                     "topology must be re-derived" % (args.n, args.tp)),
        }
        rec(res_a)
        out.append(res_a)
        if not fleet_ok:
            return out

        live = [i for i in range(args.n)]

        # ---- gate (b): k simultaneous level-1 sleeps ----------------------
        b_rows = []
        b_ok = True
        for k in range(1, args.n + 1):
            try:
                ram_before = host_ram_used_gib()
                sleeps = [sleep_engine(cfgs[i]["port"], 1) for i in live[:k]]
                time.sleep(3)                       # let nvidia-smi settle
                ram_slept = host_ram_used_gib()
                vram = gpu_mem_used_mib(list(range(needed)))
                wakes, gens = [], []
                for i in live[:k]:
                    wakes.append(wake_engine(cfgs[i]["port"]))
                    gens.append(sanity_generate(cfgs[i]["port"],
                                                served_name(cfgs[i], i)))
                row = {
                    "k_slept": k,
                    "sleep_s": [round(s, 2) for s in sleeps],
                    "wake_s": [round(w, 2) for w in wakes],
                    "wake_s_max": round(max(wakes), 2),
                    "first_gen_s": [round(g, 2) for g in gens],
                    "host_ram_gib_before": round(ram_before, 1),
                    "host_ram_gib_slept": round(ram_slept, 1),
                    "host_ram_gib_delta": round(ram_slept - ram_before, 1),
                    "gpu_mem_mib_while_slept": vram,
                    "ok": max(wakes) <= WAKE_CEILING_S,
                }
            except Exception as exc:                          # noqa: BLE001
                row = {"k_slept": k, "ok": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            b_ok = b_ok and row.get("ok", False)
            b_rows.append(row)
            rec({"gate": "b_progress", **row})
            if not row.get("ok", False):
                break                       # a failure at k won't improve at k+1
        res_b = {
            "gate": "b_n_simultaneous_l1_sleep", "pass": b_ok,
            "n": args.n, "wake_ceiling_s": WAKE_CEILING_S,
            "cgroup_mem_limit_gib": cgroup_mem_limit_gib(),
            "rows": b_rows,
            "note": ("level-1 sleep scales to N engines with wake under the "
                     "ceiling — the state ladder can park at L1"
                     if b_ok else
                     "level-1 sleep does not scale to N engines here; the "
                     "ladder must park at L2 (or M must shrink)"),
        }
        rec(res_b)
        out.append(res_b)

        # ---- gate (c): does level 2 give host RAM back? -------------------
        try:
            l1_delta = next((r["host_ram_gib_delta"] for r in reversed(b_rows)
                             if r.get("ok") and "host_ram_gib_delta" in r), None)
            ram_before = host_ram_used_gib()
            for i in live:
                sleep_engine(cfgs[i]["port"], 2)
            time.sleep(3)
            ram_l2 = host_ram_used_gib()
            wakes = [wake_engine(cfgs[i]["port"]) for i in live]
            gens = [sanity_generate(cfgs[i]["port"], served_name(cfgs[i], i))
                    for i in live]
            l2_delta = ram_l2 - ram_before
            # L2 passes if it holds substantially less host RAM than L1 did at
            # the same k.  If l1_delta is unavailable the gate is inconclusive.
            if l1_delta is None:
                c_ok, verdict = False, "inconclusive — no usable level-1 row"
            elif l1_delta <= 1.0:
                c_ok, verdict = False, (
                    "inconclusive — level-1 host-RAM growth was already ~0, so "
                    "there is nothing for level 2 to give back (suspect the "
                    "RAM probe or the cgroup accounting)")
            else:
                c_ok = l2_delta <= (1.0 - L2_RETURN_FRACTION) * l1_delta
                verdict = ("level 2 genuinely frees host RAM — the sleep-level "
                           "dimension is real and worth sweeping" if c_ok else
                           "level 2 holds about as much host RAM as level 1: "
                           "the sleep-level policy dimension is vacuous and "
                           "should be dropped from the plan")
            res_c = {
                "gate": "c_l2_returns_host_ram", "pass": c_ok,
                "n_slept": len(live),
                "l1_host_ram_gib_delta": l1_delta,
                "l2_host_ram_gib_delta": round(l2_delta, 1),
                "return_fraction_required": L2_RETURN_FRACTION,
                "wake_s": [round(w, 2) for w in wakes],
                "first_gen_s": [round(g, 2) for g in gens],
                "note": verdict,
            }
        except Exception as exc:                              # noqa: BLE001
            res_c = {"gate": "c_l2_returns_host_ram", "pass": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        rec(res_c)
        out.append(res_c)
    finally:
        for p in procs:
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                p.kill()
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen_32b_vl",
                    choices=sorted(MODELS_CHEMGRAPH_SWAP),
                    help="32B-class model to fleet-boot (gate a/b/c)")
    ap.add_argument("--n", type=int, default=4, help="engines in the fleet")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel per engine")
    ap.add_argument("--base-port", type=int, default=8021,
                    help="engine i listens on base_port+i (kept clear of the "
                         "8001-8012 range the eval driver uses)")
    ap.add_argument("--gates", default="a,b,c,d",
                    help="comma-separated subset of gates to run")
    ap.add_argument("--fadvise-dir", default=None,
                    help="directory for the gate-d canary file "
                         "(default: <repo>/results/_fadvise_canary)")
    ap.add_argument("--fadvise-gib", type=float, default=2.0)
    args = ap.parse_args()

    gates = {g.strip() for g in args.gates.split(",") if g.strip()}
    results: list[dict] = []
    host = socket.gethostname()
    outfile = (_HERE.parent / "results" /
               f"bench_residency_preflight_{host}.json")
    outfile.parent.mkdir(exist_ok=True)

    def rec(row: dict) -> None:
        row = {"t": time.time(), "host": host, **row}
        results.append(row)
        print(json.dumps(row), flush=True)
        # Persist after every record: this runs on a preemptible allocation and
        # a partial result table is far better than none.
        outfile.write_text(json.dumps(results, indent=2) + "\n")

    rec({"gate": "env", "pass": True,
         "gpus": gpu_names(), "gpu_total_mib": gpu_total_mib(),
         "cgroup_mem_limit_gib": cgroup_mem_limit_gib(),
         "host_ram_gib_used_at_start": round(host_ram_used_gib(), 1),
         "requested": {"model": args.model, "n": args.n, "tp": args.tp,
                       "gates": sorted(gates)}})

    # (d) first: no GPU needed, so a GPU-side failure still leaves it measured.
    if "d" in gates:
        target = (Path(args.fadvise_dir) if args.fadvise_dir
                  else _HERE.parent / "results" / "_fadvise_canary")
        gate_d_fadvise(target, args.fadvise_gib, rec)

    if gates & {"a", "b", "c"}:
        gates_abc(args, rec)

    # ---- summary table ---------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("STAGE-0 RESIDENCY PREFLIGHT — %s" % host, flush=True)
    print("=" * 72, flush=True)
    headline = [r for r in results
                if r.get("gate", "").startswith(("a_", "b_", "c_", "d_"))]
    for r in headline:
        mark = "PASS" if r.get("pass") else "FAIL"
        print(f"[{mark}] {r['gate']}", flush=True)
        if r.get("error"):
            print(f"        error: {r['error']}", flush=True)
        if r.get("note"):
            print(f"        {r['note']}", flush=True)
    ran_blocking = [r for r in headline if r["gate"].startswith(("a_", "b_"))]
    blocking = [r for r in ran_blocking if not r.get("pass")]
    print("-" * 72, flush=True)
    if blocking:
        print("BLOCKING FAILURE — the N-model residency topology is not "
              "achievable as designed on this hardware.  Do not build "
              "Stages 1-7 until the topology is re-derived.", flush=True)
    elif ran_blocking:
        print("Gates (a)/(b) clear: N-model residency topology is achievable.",
              flush=True)
    else:
        print("Blocking gates (a)/(b) were NOT run — topology feasibility is "
              "still unmeasured.", flush=True)
    print(f"results: {outfile}", flush=True)
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
