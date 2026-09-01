#!/usr/bin/env python3
"""Does the residency actor actually take a GPU that is already occupied?

THE FAILURE THIS TESTS. On the aligned campaign, 16 of 16 model prefetches
failed. Ten died in under a millisecond with

    Cannot start qwen_32b: GPUs [0,1,2,3] occupied by qwen_72b. Call stop_model first.

and the proactive-swap ones did not fail fast at all -- they sat in the
executor's 600 s wait-for-GPUs loop and failed after 600.02, 600.02, 600.03 and
918.04 s. The predictor was not the problem: a correct prediction had nowhere to
put its result. VllmModelActor.activate() is the replacement for
orchestrator.start_model_measured() on that path, and this measures whether it
works against a real engine rather than a fake.

WHY ONE GPU. The full exp3 workload needs 4 GPUs at tp=4 and the partition has
been saturated all day. But the mechanism under test is contention, and
contention does not need four cards: two 68.28 GB engines cannot coexist on one
96 GB card, so a single GPU reproduces the exact condition -- the target cannot
start until the incumbent is evicted. Both entries point at the same 32B
snapshot, the model with verified coherence evidence (782.27 s cold boot ->
2.076 s wake, verbatim-correct output), so a wake here is trustworthy.

TWO ARMS, AND THE CONTROL IS THE POINT.
  control : boot A, then call the orchestrator directly for B.
            EXPECTED TO FAIL with the "occupied" error. If it succeeds, the
            premise is wrong and nothing below means anything.
  tandem  : boot A, then actor.activate(B).
            Expected to evict A and serve B coherently.

A green tandem arm without a red control arm would prove nothing -- it could
mean the GPUs were free all along.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workloads/AtomAgents"))

VLLM_PY = "/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean128/bin/python"
SNAP_32B = os.path.expanduser(
    "~/scratch/hf_home/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/"
    "7cfb30d71a1f4f49a57592323337a4a4727301da")


def _entry(port: int) -> dict:
    return {
        "python_bin": VLLM_PY,
        "model_name": SNAP_32B,
        "port": port,
        "gpus": [0],                      # BOTH on GPU 0 -- the contention
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "load_timeout": 2700,
        "extra_args": ["--dtype", "float16", "--enforce-eager",
                       "--enable-sleep-mode",
                       "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct"],
        # BOTH are required and neither can be turned on after launch.
        # --enable-sleep-mode builds the engine with CuMemAllocator;
        # VLLM_SERVER_DEV_MODE=1 is what EXPOSES /sleep, /wake_up and
        # /is_sleeping. Run 12684990 set only the first, so park() got a 404 and
        # the actor downgraded to stop -- forcing a 1256.63 s cold boot where a
        # wake is ~2 s. The same omission was live in the production tandem arm.
        "extra_env": {"VLLM_SERVER_DEV_MODE": "1"},
    }


MODELS = {"model_a": _entry(8311), "model_b": _entry(8312)}


def vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"results/bench_gpu_eviction_{os.environ.get('SLURM_JOB_ID','local')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec: dict = {"meta": {"node": os.uname().nodename,
                          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "model": SNAP_32B, "gpus": [0], "tp": 1}, "arms": {}}

    def save():
        out.write_text(json.dumps(rec, indent=1))

    from atomagents.runtime.model_orchestrator import ModelOrchestrator
    from runtime.residency.model_actor import VllmModelActor, GpusNotFreed

    orch = ModelOrchestrator(MODELS)

    # ---------------- CONTROL: reproduce the documented failure -----------
    print("\n=== CONTROL: boot A, then ask the orchestrator directly for B ===",
          flush=True)
    t0 = time.perf_counter()
    orch.start_model("model_a")
    orch.wait_until_ready("model_a")
    boot_a = time.perf_counter() - t0
    vram_a = vram_mib()
    print(f"[control] model_a serving after {boot_a:.1f}s, VRAM {vram_a} MiB", flush=True)

    ctl = {"boot_a_s": round(boot_a, 2), "vram_after_a_mib": vram_a}
    t1 = time.perf_counter()
    try:
        orch.start_model("model_b")
        orch.wait_until_ready("model_b", timeout=180)
        ctl["result"] = "UNEXPECTED SUCCESS — the premise is wrong"
        ctl["failed_as_expected"] = False
    except Exception as exc:
        ctl["result"] = f"{type(exc).__name__}: {exc}"[:400]
        ctl["failed_as_expected"] = True
        ctl["occupied_error"] = "occupied by" in str(exc) or "stop_model" in str(exc)
    ctl["elapsed_s"] = round(time.perf_counter() - t1, 3)
    ctl["path"] = "orchestrator.start_model"
    rec["arms"]["control"] = ctl

    # ---- CONTROL 2: the EXACT path the prefetcher used -------------------
    # start_model() has no occupancy guard, so it launches vLLM and lets it
    # crash on OOM (~30 s). The documented failure came from
    # start_model_measured(), whose guard at model_orchestrator.py:597-598
    # refuses BEFORE launching -- which is why those 10 prefetches died in
    # under a millisecond rather than after a crash. Both prove contention;
    # only this one reproduces the recorded error string.
    print("\n=== CONTROL 2: start_model_measured (the prefetch path) ===",
          flush=True)
    c2: dict = {"path": "orchestrator.start_model_measured"}
    t1b = time.perf_counter()
    try:
        orch.start_model_measured("model_b")
        c2["result"] = "UNEXPECTED SUCCESS"
        c2["failed_as_expected"] = False
    except Exception as exc:
        c2["result"] = f"{type(exc).__name__}: {exc}"[:400]
        c2["failed_as_expected"] = True
        c2["reproduces_recorded_error"] = (
            "occupied by" in str(exc) and "stop_model" in str(exc))
    c2["elapsed_s"] = round(time.perf_counter() - t1b, 4)
    rec["arms"]["control_measured"] = c2
    save()
    print(f"[control2] {c2['result'][:170]}  ({c2['elapsed_s']}s)", flush=True)
    save()
    print(f"[control] {ctl['result'][:160]}  ({ctl['elapsed_s']}s)", flush=True)

    # ---------------- TANDEM: the actor takes the GPU ---------------------
    print("\n=== TANDEM: actor.activate(model_b) with A holding GPU 0 ===",
          flush=True)
    actor = VllmModelActor(orch, verbose=True)
    tan: dict = {"vram_before_mib": vram_mib()}
    t2 = time.perf_counter()
    try:
        res = actor.activate("model_b")
        tan.update({
            "ok": True,
            "mechanism": res.get("mechanism"),
            "activate_s": round(res.get("elapsed_s", 0.0), 2),
            "evicted": res.get("evicted"),
            "probe_text": (res.get("probe") or {}).get("text"),
            "probe_ok": (res.get("probe") or {}).get("ok"),
            "gpu_path": {k: v for k, v in (res.get("gpu_path") or {}).items()
                         if k in ("ok", "reason", "evicted", "blocked_by", "action")},
        })
    except GpusNotFreed as exc:
        tan.update({"ok": False, "error": f"GpusNotFreed: {exc}"[:500]})
    except Exception as exc:
        tan.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
    tan["wall_s"] = round(time.perf_counter() - t2, 2)
    tan["vram_after_mib"] = vram_mib()
    try:
        tan["a_alive"] = orch.get_running_model() is not None
        tan["a_sleeping"] = orch.is_sleeping("model_a")
    except Exception as exc:
        tan["a_state_error"] = repr(exc)
    rec["arms"]["tandem"] = tan
    save()
    print(f"[tandem] {json.dumps({k: v for k, v in tan.items() if k != 'gpu_path'})[:300]}",
          flush=True)

    contended = ctl.get("failed_as_expected") or c2.get("failed_as_expected")
    # ---------------- ROUND TRIP: is the parked model cheap to get back? ---
    # Taking the GPU is only half of it. The point of parking rather than
    # stopping is that the return trip costs ~2 s instead of a ~1500 s boot.
    # This arm measures that directly: activate A again, now that B holds the
    # card and A is (hopefully) parked.
    print("\n=== ROUND TRIP: activate(model_a) again ===", flush=True)
    rt: dict = {}
    t3 = time.perf_counter()
    try:
        res2 = actor.activate("model_a")
        rt.update({"ok": True, "mechanism": res2.get("mechanism"),
                   "activate_s": round(res2.get("elapsed_s", 0.0), 2),
                   "evicted": res2.get("evicted"),
                   "probe_ok": (res2.get("probe") or {}).get("ok"),
                   "probe_text": (res2.get("probe") or {}).get("text")})
    except Exception as exc:
        rt.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]})
    rt["wall_s"] = round(time.perf_counter() - t3, 2)
    # THE number this arm exists for: wake vs cold boot on the return trip.
    rt["was_a_wake"] = rt.get("mechanism") == "wake"
    rt["speedup_vs_cold_boot"] = (
        round(ctl["boot_a_s"] / rt["activate_s"], 1)
        if rt.get("activate_s") else None)
    rec["arms"]["round_trip"] = rt
    save()
    print(f"[roundtrip] {json.dumps(rt)[:300]}", flush=True)

    verdict = ("MECHANISM WORKS" if tan.get("ok") and contended
               else "INCONCLUSIVE — GPUs were not actually contended"
               if not contended else "MECHANISM FAILED")
    rec["verdict"] = verdict
    rec["retention_verdict"] = (
        "PARK+WAKE — retention benefit realised" if rt.get("was_a_wake")
        else f"NO PARK — evicted by "
             f"{(tan.get('evicted') or [{}])[0].get('action')}, "
             f"return trip was {rt.get('mechanism')}")
    save()
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"wrote {out}", flush=True)

    for n in ("model_a", "model_b"):
        try:
            orch.stop_model(n)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
