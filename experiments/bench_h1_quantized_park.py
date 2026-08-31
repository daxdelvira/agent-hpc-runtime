#!/usr/bin/env python3
"""
bench_h1_quantized_park.py — H1: what does a QUANTIZED 72B cost to park at R2?

WHY THIS IS THE HIGHEST-VALUE MEASUREMENT AVAILABLE
---------------------------------------------------
Every cell in which retention arbitration is worth >=20% of wall is a
PROJECTION that assumes fp8 halves the parked footprint (weights/2, park ratio
1.90x carried over unchanged because it is a ratio, engine init held fixed at
~10 s because it does not scale with weights).  Nothing else in the policy
claim rests on an unmeasured assumption.  This script replaces it with a
number.

The reason it matters is capacity, not tidiness.  At the production 256 GB
allocation, qwen_72b (279.0 GB parked) and qwen_72b_text (276.3 GB) each exceed
the budget ALONE, so 51.4% of stall is capacity misses -- the model class is
not badly ranked there, it is UNRANKABLE, because there is no decision, only an
impossibility.  Quantization is what converts capacity misses into replacement
misses, which is the one category a ranking can convert.  If fp8 does not fit
a 72B under 256 GB, the arbitration claim has no regime on this hardware.

DESIGN -- BOTH ARMS IN ONE JOB ON ONE NODE
------------------------------------------
    arm fp16   the fp16 baseline, re-measured here
    arm fp8    the same checkpoint, --quantization fp8

Paired within a single allocation deliberately.  Node-to-node variation on this
cluster is up to 4.0x on an identical cold boot and 2.3x on an identical parse,
so a cross-node fp8-vs-fp16 comparison would be uninterpretable.  Pairing is
free and it is the cheapest variance reduction available.

WHAT IS MEASURED, AND THE TWO TRAPS
-----------------------------------
  held_gb   host RAM delta across /sleep?level=1, read from the process's OWN
            CGROUP, not /proc/meminfo.  MemTotal-MemAvailable is a HOST-WIDE
            reading on a SHARED node, so a neighbour's allocation lands in our
            number; it also counts reclaimable page cache as available, which
            is how an earlier L2 sleep appeared to hold "+0.0 GiB".  L1 parks
            weights in ANONYMOUS memory, which is what the cgroup counts.
  cold_s    time to a serving engine from a cold start.

  TRAP 1 -- ONLINE QUANTIZATION IS NOT A QUANTIZED CHECKPOINT.  With
  --quantization fp8, vLLM reads the fp16 weights and quantizes at load.  So
  the fp8 arm's cold_s INCLUDES the full fp16 read and is an UPPER BOUND on a
  real fp8 checkpoint's cold boot.  held_gb is unaffected -- what gets parked is
  the quantized tensor -- and held_gb is the term that drives the arbitration
  projection.  Report cold_s with this caveat attached or not at all.

  TRAP 2 -- A WAKE THAT RETURNS 200 IS NOT A WAKE THAT WORKS.  Level-2 sleep
  produced verbatim "!!!!" degeneracy on this cluster while returning success.
  Every arm here asserts generated TEXT and a finish_reason, and the fp8 arm
  additionally compares its output against the fp16 arm's on the same prompt.
  A footprint number from an engine that wakes incoherent is worthless.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VLLM_PY = "/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean128/bin/python"
SERVED = "h1-model"
PROBE_PROMPT = "The capital of France is"
# A SEMANTIC ANCHOR, not a style check. Measured 2026-08-30: online fp8 of
# Qwen2.5-72B at tp=1 answered this prompt with "\u306f.   1111" -- which has
# alphanumerics and three distinct characters, so BOTH the original rule and my
# first degeneracy fix passed it. No heuristic over character statistics can
# separate broken output from terse-but-correct output. Only knowing the answer
# can. If the probe prompt changes, this must change with it.
PROBE_MUST_CONTAIN = "Paris"


# ---------------------------------------------------------------- memory


def _cgroup_dir() -> tuple[Path | None, str]:
    """This process's cgroup v2 directory, walking up to one with memory.stat.

    Works under a SLURM step cgroup on a compute node and under user.slice on a
    login node. Returns (None, reason) rather than guessing -- a silently wrong
    budget reading is how the L2 sleep result got misread.
    """
    try:
        rel = ""
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2]
                    break
        if not rel:
            return None, "no cgroup v2 entry in /proc/self/cgroup"
        node = Path("/sys/fs/cgroup") / rel.lstrip("/")
        while True:
            if (node / "memory.stat").exists():
                return node, str(node)
            if node == Path("/sys/fs/cgroup"):
                return None, "no memory.stat found walking to cgroup root"
            node = node.parent
    except Exception as exc:  # noqa: BLE001
        return None, f"cgroup read failed: {exc!r}"


def cgroup_mem() -> dict:
    """anon / file / current for this cgroup, in GiB.

    **anon is the number that matters, and this distinction is the whole
    measurement.** memory.current includes PAGE CACHE, and getting a model
    parked means first READING ~146 GB of weight shards -- so a delta taken on
    memory.current would count the entire file cache as though it were parked
    weights. An L1 sleep holds the weights in ANONYMOUS memory (which is
    precisely why wake is independent of page-cache state: evicting 18 shards
    changed a measured wake by 0.015 s). So anon is what a park costs the
    budget, and file is recorded alongside only to make the confound visible
    rather than invisible.
    """
    d, path = _cgroup_dir()
    if d is None:
        return {"anon_gib": -1.0, "file_gib": -1.0, "current_gib": -1.0,
                "path": path}
    out = {"path": str(d)}
    try:
        stat = {}
        for line in (d / "memory.stat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in ("anon", "file"):
                stat[k] = int(v)
        out["anon_gib"] = stat.get("anon", 0) / 1024 ** 3
        out["file_gib"] = stat.get("file", 0) / 1024 ** 3
    except Exception as exc:  # noqa: BLE001
        out["anon_gib"] = out["file_gib"] = -1.0
        out["error"] = repr(exc)
    try:
        out["current_gib"] = int(
            (d / "memory.current").read_text().strip()) / 1024 ** 3
    except Exception:  # noqa: BLE001
        out["current_gib"] = -1.0
    return out


def cgroup_anon_gib() -> float:
    return cgroup_mem()["anon_gib"]


def cgroup_mem_bytes() -> tuple[int, str]:
    m = cgroup_mem()
    b = -1 if m["current_gib"] < 0 else int(m["current_gib"] * 1024 ** 3)
    return b, m["path"]


def cgroup_mem_gib() -> float:
    return cgroup_mem()["current_gib"]


def host_ram_used_gib() -> float:
    """Kept ONLY as a cross-check against the cgroup reading, never as the
    primary number. Host-wide on a shared node; see the docstring."""
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def gpu_mem_used_mib(gpus: list[int]) -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True).stdout
        by_idx = {int(a): int(b) for a, b in
                  (l.split(",") for l in out.strip().splitlines())}
        return [by_idx.get(g, -1) for g in gpus]
    except Exception:  # noqa: BLE001
        return [-1] * len(gpus)


# ---------------------------------------------------------------- engine


def launch(model_path: str, port: int, gpus: list[int], tp: int,
           quantization: str | None, logdir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["VLLM_SERVER_DEV_MODE"] = "1"      # exposes /sleep /wake_up /is_sleeping
    # Sleep mode and the expandable-segments allocator are mutually exclusive
    # ("Expandable segments are not compatible with memory pool"). Stripping
    # this unconditionally cost four failed smoke attempts on 2026-07-29.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    cmd = [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--port", str(port),
           "--tensor-parallel-size", str(tp),
           "--gpu-memory-utilization", "0.85",
           "--max-model-len", "4096",
           "--enforce-eager",
           "--enable-sleep-mode",
           "--served-model-name", SERVED]
    if quantization:
        cmd += ["--quantization", quantization]
    else:
        cmd += ["--dtype", "float16"]
    logdir.mkdir(parents=True, exist_ok=True)
    log = (logdir / f"vllm_{port}_{quantization or 'fp16'}.log").open("a")
    print(f"[launch] {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def kill(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    subprocess.run(["pkill", "-u", os.environ.get("USER", ""), "-f", "VLLM::"],
                   check=False)
    time.sleep(8)


def _get(port: int, path: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return r.read().decode()


def _post(port: int, path: str, timeout: float = 900.0) -> str:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"",
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def wait_ready(port: int, timeout: float = 3600.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            _get(port, "/health", timeout=10)
            return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise TimeoutError(f"engine on :{port} not ready within {timeout}s")


def wait_sleeping(port: int, want: bool, timeout: float = 900.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if json.loads(_get(port, "/is_sleeping")).get("is_sleeping") is want:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise TimeoutError(f"is_sleeping != {want} within {timeout}s")


def coherence_probe(port: int, max_tokens: int = 24) -> dict:
    """Assert generated TEXT and a finish_reason, never just a 200.

    This is the check that separated the real L1 result from the fake L2 one.
    """
    body = json.dumps({
        "model": SERVED, "prompt": PROBE_PROMPT,
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    text = ch.get("text", "")
    return {
        "text": text,
        "finish_reason": ch.get("finish_reason"),
        "n_tokens": d.get("usage", {}).get("completion_tokens"),
        "degenerate": is_degenerate(text),
        "expected_substring": PROBE_MUST_CONTAIN,
        "has_expected": PROBE_MUST_CONTAIN.lower() in text.lower(),
        "ok": (bool(text.strip())
               and ch.get("finish_reason") is not None
               and not is_degenerate(text)
               and PROBE_MUST_CONTAIN.lower() in text.lower()),
    }


def is_degenerate(text: str) -> bool:
    """True for output that is non-empty but carries no information.

    FIXED 2026-08-30, caught by A5 reviewing this file. The original rule was
    `bool(text.strip()) and finish_reason is not None` -- and the verbatim
    "!!!!" degeneracy that L2 sleep produced on this cluster satisfies BOTH.
    So the check this script exists to perform would have passed the exact
    failure it was written to catch. The L2 result was actually caught by
    comparing text ACROSS arms, not by this rule; unattended, nothing caught it.
    """
    t = text.strip()
    if not t:
        return True
    if not any(c.isalnum() for c in t):        # "!!!!", "....", "$$$$"
        return True
    if len(set(t.replace(" ", ""))) <= 2:      # "aaaa", "abababab"
        return True
    return False


# ---------------------------------------------------------------- one arm


def run_arm(name: str, model_path: str, port: int, gpus: list[int], tp: int,
            quantization: str | None, rec) -> dict:
    proc = None
    try:
        mem_before = cgroup_mem()
        host_before = host_ram_used_gib()
        proc = launch(model_path, port, gpus, tp, quantization,
                      Path("logs/h1_quantized"))
        cold_s = wait_ready(port)
        probe_boot = coherence_probe(port)
        if not probe_boot["ok"]:
            raise RuntimeError(f"{name}: engine serves but generates nothing: "
                               f"{probe_boot!r}")
        mem_awake = cgroup_mem()
        gpu_awake = gpu_mem_used_mib(gpus)
        rec(arm=name, rung="cold_boot", quantization=quantization or "fp16",
            cold_s=round(cold_s, 2),
            anon_gib_before=round(mem_before["anon_gib"], 2),
            file_gib_before=round(mem_before["file_gib"], 2),
            anon_gib_awake=round(mem_awake["anon_gib"], 2),
            file_gib_awake=round(mem_awake["file_gib"], 2),
            host_gib_awake=round(host_ram_used_gib(), 2),
            gpu_mib=gpu_awake, probe=probe_boot)

        # ---- park at R2 (L1 sleep) ------------------------------------
        t0 = time.perf_counter()
        _post(port, "/sleep?level=1")
        wait_sleeping(port, True)
        sleep_s = time.perf_counter() - t0
        mem_slept = cgroup_mem()
        gpu_slept = gpu_mem_used_mib(gpus)

        t1 = time.perf_counter()
        _post(port, "/wake_up")
        wait_sleeping(port, False)
        wake_s = time.perf_counter() - t1
        probe_wake = coherence_probe(port)

        # A first park allocates the host-side backup; a second reuses it, and
        # was ~9x cheaper when this was measured on a 32B. Recording both so a
        # scheduler amortising over a workflow has the real number.
        t2 = time.perf_counter()
        _post(port, "/sleep?level=1")
        wait_sleeping(port, True)
        sleep2_s = time.perf_counter() - t2
        mem_slept2 = cgroup_mem()
        _post(port, "/wake_up")
        wait_sleeping(port, False)

        # THE park cost: memory that appeared when the weights left the GPU,
        # taken against the AWAKE reading so neither the engine's baseline nor
        # the 145 GB of shards read during loading is charged to the park.
        #
        # MEASURED 2026-08-30, job 12561711, and it corrected me. I originally
        # read `anon` alone, reasoning that an L1 park lands in ANONYMOUS
        # memory and that `current` would wrongly absorb the load's page cache.
        # The first half is false on this build: across the sleep, anon stayed
        # at 2.77 GiB while `file` went 1.32 -> 83.36 GiB, host-wide RAM rose
        # 86.67 GiB and GPU fell 86 GB. The weights moved to host and landed
        # FILE-BACKED. Reading anon alone reported held_gb = 0.0 for an 82 GiB
        # park.
        #
        # The delta-from-awake decision was the right one and is what makes
        # `current` safe here: the load's page cache is already in the awake
        # baseline, so it cancels. Report all three columns -- the split is
        # itself a finding, because a file-backed park is in principle
        # expressible as a file range and an anonymous one is not.
        held_anon_gib = mem_slept["anon_gib"] - mem_awake["anon_gib"]
        held_file_gib = mem_slept["file_gib"] - mem_awake["file_gib"]
        held_gib = held_anon_gib + held_file_gib
        held_gb = held_gib * 1024 ** 3 / 1e9 if mem_slept["anon_gib"] >= 0 else -1.0
        row = dict(
            arm=name, rung="park_L1", quantization=quantization or "fp16",
            sleep_s=round(sleep_s, 3), sleep2_s=round(sleep2_s, 3),
            wake_s=round(wake_s, 3),
            held_anon_gib=round(held_anon_gib, 2),
            held_file_gib=round(held_file_gib, 2),
            anon_gib_slept=round(mem_slept["anon_gib"], 2),
            file_gib_slept=round(mem_slept["file_gib"], 2),
            anon_gib_slept2=round(mem_slept2["anon_gib"], 2),
            host_gib_slept=round(host_ram_used_gib(), 2),
            gpu_mib_slept=gpu_slept,
            held_gib=round(held_gib, 2),
            held_gb=round(held_gb, 2),
            probe_after_wake=probe_wake,
        )
        rec(**row)
        if not probe_wake["ok"]:
            raise RuntimeError(f"{name}: incoherent after L1 wake: {probe_wake!r}")
        return row
    finally:
        kill(proc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=8241)
    ap.add_argument("--quantization", default="fp8",
                    help="vLLM --quantization value for the quantized arm")
    ap.add_argument("--arms", default="fp16,quant",
                    help="comma list; 'fp16' and/or 'quant'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_h1_quantized_park_{host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    mem0 = cgroup_mem()
    mem_b, mem_path = int(max(mem0["current_gib"], 0) * 1024 ** 3), mem0["path"]
    shards = sorted(Path(args.model_path).glob("*.safetensors"))
    nbytes = sum(p.stat().st_size for p in shards)
    meta = {
        "host": host, "tp": args.tp, "gpus": gpus,
        "model_path": args.model_path,
        "weight_bytes": nbytes, "n_shards": len(shards),
        "cgroup_path": mem_path,
        "cgroup_anon_gib_at_start": round(mem0["anon_gib"], 2),
        "cgroup_mem_max": (lambda p: p.read_text().strip()
                           if p.exists() else "n/a")(Path(mem_path) / "memory.max")
        if mem0["anon_gib"] >= 0 else "n/a",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rows: list[dict] = []

    def rec(**kw):
        kw["t"] = time.time()
        rows.append(kw)
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print(f"  -> {json.dumps({k: v for k, v in kw.items() if k != 't'})}",
              flush=True)

    print(f"[h1] {len(shards)} shards, {nbytes/1e9:.2f} GB fp16 on disk", flush=True)
    print(f"[h1] cgroup memory.current: {mem_path} ({mem_b/1024**3:.2f} GiB)",
          flush=True)
    if mem0["anon_gib"] < 0:
        print("[h1] WARNING: no cgroup reading available; held_gb will be -1 "
              "and the host-wide figure is NOT a substitute on a shared node.",
              flush=True)

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    results: dict[str, dict] = {}
    for arm in wanted:
        quant = None if arm == "fp16" else args.quantization
        print(f"\n=== arm {arm} (quantization={quant}) ===", flush=True)
        try:
            results[arm] = run_arm(arm, args.model_path, args.port, gpus,
                                   args.tp, quant, rec)
        except Exception as exc:  # noqa: BLE001
            # One arm failing must not lose the other arm's data.
            rec(arm=arm, rung="ERROR", error=repr(exc))
            print(f"[h1] arm {arm} FAILED: {exc!r}", file=sys.stderr, flush=True)

    if "fp16" in results and "quant" in results:
        a, b = results["fp16"], results["quant"]
        ratio = b["held_gb"] / a["held_gb"] if a["held_gb"] > 0 else -1.0
        # Cross-arm text identity at temperature 0 is the check that actually
        # caught L2's degeneracy. Keep it, but it is now a second line of
        # defence rather than the only one -- is_degenerate() runs per arm.
        same = (a["probe_after_wake"]["text"] == b["probe_after_wake"]["text"])
        rec(arm="COMPARISON", rung="summary",
            fp16_held_gb=a["held_gb"], quant_held_gb=b["held_gb"],
            held_ratio=round(ratio, 4),
            projection_assumed=0.5,
            note=("held_ratio is the number the >=20% projection assumed to be "
                  "0.50. cold_s for the quantized arm INCLUDES the fp16 read "
                  "(online quantization) and is an upper bound."),
            same_text_as_fp16=same)
    print(f"\n[h1] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
