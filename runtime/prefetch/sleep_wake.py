"""
prefetch/sleep_wake.py — sleep/wake swap sequencing (RuntimeConfig.sleep_wake_swaps).

Diagnosis (results/eval_q1_q4/eval_stall_taxonomy.csv): the swap-family exposed
stall is dominated by "no_window" vLLM engine bring-up (~185 s on chemgraph_swap)
against a ~0-12 s workflow window — a cold boot can never be hidden by any
trigger.  Sleep mode fixes the MECHANISM instead of the trigger: each engine
boots ONCE with --enable-sleep-mode; between uses it is SLEPT (level 1: weights
offloaded to CPU RAM, VRAM freed) instead of killed, and WOKEN on demand (H2D
copy, ~6-15 s for the 145 GB worker; measured by experiments/bench_sleep_wake.py).

Invariant: the planner and worker share one GPU pool and their weights do NOT
fit in VRAM together — the sequence is always sleep-the-other THEN
wake/boot-the-target, never the reverse.

RAM budget: level-1 sleep parks weights in CPU RAM — 32B planner (~65 GB) +
72B worker (~145 GB) ≈ 210 GB simultaneously resident, inside the 256G hold
cgroup but with little headroom.  If a node is RAM-constrained, pass level=2:
weights are discarded (VRAM still freed) and wake re-reads them from the page
cache at staging bandwidth instead of copying H2D.

This module is shared by BOTH swap call sites so the unit tests
(runtime/tests/test_sleep_wake.py, against FakeModelOrchestrator) exercise the
exact production sequencing:
  - runtime/adapters/chemgraph.py on_chain_start (on-demand path)
  - runtime/prefetch/model_prefetch.py ModelPrefetchExecutor._load_model
    (prefetch path — wake-if-slept preferred over cold boot)
"""
from __future__ import annotations

import time
from typing import Any


def _proc_alive(proc: Any) -> bool:
    """True if a Popen-like object represents a live process.  Objects without
    poll() (FakeModelOrchestrator sentinels) count as alive."""
    if proc is None:
        return False
    poll = getattr(proc, "poll", None)
    if callable(poll):
        return poll() is None
    return True


def has_live_process(orchestrator, name: str) -> bool:
    """True if the orchestrator holds a live server process for `name`
    (serving OR slept — a slept engine keeps its process alive)."""
    procs = getattr(orchestrator, "processes", {}) or {}
    return _proc_alive(procs.get(name))


def record_mechanism(orchestrator, name: str, mechanism: str) -> None:
    """Remember how `name` most recently became serving ("sleep_wake" or
    "cold_boot") on the orchestrator, creating the dict if absent (works for
    FakeModelOrchestrator and older orchestrator versions alike)."""
    lt = getattr(orchestrator, "last_transition", None)
    if not isinstance(lt, dict):
        lt = {}
        try:
            setattr(orchestrator, "last_transition", lt)
        except Exception:
            return
    lt[name] = mechanism


def last_mechanism(orchestrator, name: str, default: str = "cold_boot") -> str:
    lt = getattr(orchestrator, "last_transition", None)
    if isinstance(lt, dict):
        return lt.get(name, default)
    return default


def release_gpus_for(orchestrator, target: str, sleep_wake: bool,
                     level: int = 1) -> None:
    """
    Free the shared GPU pool for `target` ahead of a scheduled prefetch.

    sleep_wake=False: legacy behaviour, byte-identical to the pre-existing
    inline code — stop_model(running) (process killed, weights lost).

    sleep_wake=True: the running model is SLEPT, never stopped — its weights
    stay in CPU RAM so a later wake avoids a cold boot.  Falls back to
    stop_model if the sleep endpoint fails (e.g. dev mode missing), which
    keeps the run alive at the cost of the wake benefit.
    """
    running = orchestrator.get_running_model()
    if not running or running == target:
        return
    if not sleep_wake:
        orchestrator.stop_model(running)
        return
    is_sleeping = getattr(orchestrator, "is_sleeping", None)
    if callable(is_sleeping) and orchestrator.is_sleeping(running):
        return   # already asleep — VRAM already free
    try:
        orchestrator.sleep_model(running, level=level)
    except Exception as exc:
        print(f"[sleep_wake] WARNING: could not sleep {running} ({exc}); "
              f"stopping it instead.", flush=True)
        orchestrator.stop_model(running)


def swap_to_model(orchestrator, target: str, level: int = 1,
                  metrics=None) -> dict[str, Any]:
    """
    Sleep/wake swap: make `target` the serving model.

    Sequence (order matters — both models' weights never fit in VRAM):
      1. SLEEP every other live, awake model whose GPU set overlaps the
         target's pool (never stop: weights stay in CPU RAM for later wakes;
         falls back to stop_model if the sleep endpoint fails).
      2. Then bring up the target:
         - live process, engine asleep    -> wake_model  ("sleep_wake")
         - live process, engine awake     -> wait_until_serving only
                                             ("already_serving": a concurrent
                                             boot/wake — e.g. the prefetch
                                             thread — owns the transition)
         - no process (never booted yet)  -> cold boot via
                                             start_model_measured, which
                                             launches WITH sleep mode enabled
                                             so later swaps can sleep it
                                             ("cold_boot")

    Blocks until the target actually serves.  Returns
      {"mechanism": "sleep_wake"|"cold_boot"|"already_serving",
       "elapsed_s": float, "slept_models": [...]}.
    """
    t0 = time.perf_counter()
    models = getattr(orchestrator, "models", {}) or {}
    target_gpus = set(models.get(target, {}).get("gpus", []))
    procs = getattr(orchestrator, "processes", {}) or {}
    target_alive = _proc_alive(procs.get(target))

    # 1) Free VRAM: sleep conflicting engines (default: everything, when no
    #    GPU sets are declared — the shared-pool case).
    slept: list[str] = []
    for other, proc in list(procs.items()):
        if other == target or not _proc_alive(proc):
            continue
        other_gpus = set(models.get(other, {}).get("gpus", []))
        if target_gpus and other_gpus and not (target_gpus & other_gpus):
            continue   # disjoint pools — no VRAM conflict
        if orchestrator.is_sleeping(other):
            continue
        try:
            orchestrator.sleep_model(other, level=level)
            slept.append(other)
        except Exception as exc:
            print(f"[sleep_wake] WARNING: could not sleep {other} ({exc}); "
                  f"stopping it instead.", flush=True)
            orchestrator.stop_model(other)

    # 2) Wake / verify / cold-boot the target.
    # Serving-wait budget: the "already_serving" branch may be waiting out a
    # full cold boot owned by the prefetch thread — give it the model's
    # load_timeout (5400 s for the 72B worker on degraded Lustre; a slow swap
    # is valid data, a timeout is a lost trial), not the 1800 s default.
    serve_timeout = int(models.get(target, {}).get("load_timeout", 1800))
    if target_alive and orchestrator.is_sleeping(target):
        orchestrator.wake_model(target)
        mechanism = "sleep_wake"
        record_mechanism(orchestrator, target, "sleep_wake")
    elif target_alive:
        orchestrator.wait_until_serving(target, timeout=serve_timeout)
        mechanism = "already_serving"
    else:
        orchestrator.start_model_measured(target, metrics=metrics)
        orchestrator.wait_until_serving(target, timeout=serve_timeout)
        mechanism = "cold_boot"
        record_mechanism(orchestrator, target, "cold_boot")

    return {
        "mechanism": mechanism,
        "elapsed_s": time.perf_counter() - t0,
        "slept_models": slept,
    }
