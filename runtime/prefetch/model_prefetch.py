"""
prefetch/model_prefetch.py — ModelPrefetchExecutor for vLLM model servers.

Wraps ModelOrchestrator.start_model_measured() in a background thread so
model weight loading overlaps with ongoing agent compute.

Key design constraint: vLLM weight loading is NOT safely interruptible once
started (half-loaded shards leave the GPU in an undefined state). When the
divergence guard calls cancel(), we:
  1. Record the cancellation intent in _cancelled_tasks
  2. Return False (cannot interrupt the load)
  3. Let the background load finish; mark the task WASTED when it completes

The scheduler already accounts for WASTED tasks in the overlap metrics.

Swap to this executor from SimulatedPrefetchExecutor by changing one line
in the cluster script constructor:

    # Before (simulated):
    executor = SimulatedPrefetchExecutor()

    # After (real):
    from atomagents.runtime.model_orchestrator import ModelOrchestrator
    from atomagents.runtime.model_config import MODELS
    executor = ModelPrefetchExecutor(ModelOrchestrator(MODELS))

FakeModelOrchestrator lets you run the full timing pipeline locally
with configurable sleep-based "load" times — no GPUs required.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask


# ---------------------------------------------------------------------------
# FakeModelOrchestrator — for local testing and demos
# ---------------------------------------------------------------------------

class FakeModelOrchestrator:
    """
    Drop-in replacement for ModelOrchestrator that sleeps instead of
    loading real model weights.

    Parameters
    ----------
    load_times : dict mapping model name → simulated load seconds.
                 Unknown models default to 1.0s.
    failure_models : set of model names that should raise RuntimeError.
                     Used to test error handling paths.
    wake_times : dict mapping model name → simulated wake seconds
                 (sleep/wake swap arm; default 0.0).
    models : optional dict mapping model name → config dict (e.g.
             {"gpus": [0, 1]}), mirroring ModelOrchestrator.models so
             GPU-overlap logic can be tested.

    Sleep/wake state (mirrors the real orchestrator's sleep-mode API):
      sleeping        — set of model names currently slept (weights "in CPU RAM")
      calls           — chronological op log, e.g. ("sleep_model", "planner", 1),
                        for sequencing assertions in tests
      last_transition — model name → "sleep_wake" | "cold_boot"
    """

    def __init__(
        self,
        load_times: dict[str, float] | None = None,
        failure_models: set[str] | None = None,
        wake_times: dict[str, float] | None = None,
        models: dict | None = None,
    ) -> None:
        self.load_times = load_times or {}
        self.failure_models = failure_models or set()
        self.wake_times = wake_times or {}
        self.models = models or {}
        self.processes: dict[str, object] = {}   # mimics ModelOrchestrator.processes
        self.sleeping: set[str] = set()
        self.calls: list[tuple] = []
        self.last_transition: dict[str, str] = {}

    def start_model_measured(
        self,
        name: str,
        metrics=None,
    ) -> float:
        """Simulate model loading: sleep for load_times[name] seconds."""
        self.calls.append(("start_model_measured", name))
        if name in self.failure_models:
            raise RuntimeError(f"FakeModelOrchestrator: simulated failure for {name}")
        elapsed = self.load_times.get(name, 1.0)
        time.sleep(elapsed)
        self.processes[name] = object()   # non-None sentinel
        self.sleeping.discard(name)       # a fresh boot is awake
        self.last_transition[name] = "cold_boot"
        return elapsed

    def stop_model(self, name: str, wait_s: float = 5.0) -> None:
        self.calls.append(("stop_model", name))
        self.processes.pop(name, None)
        self.sleeping.discard(name)

    def wait_until_ready(self, name: str, timeout: int = 60) -> None:
        pass   # already "ready" after start_model_measured returns

    def ensure_all_models_ready(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Sleep/wake API (mirrors ModelOrchestrator's sleep-mode methods)
    # ------------------------------------------------------------------

    def get_running_model(self) -> str | None:
        """First model with a live process (sleeping or not) — mirrors the
        real orchestrator, whose slept processes stay alive."""
        return next(iter(self.processes), None)

    def is_sleeping(self, name: str) -> bool:
        return name in self.processes and name in self.sleeping

    def sleep_model(self, name: str, level: int = 1, timeout: float = 900.0) -> float:
        self.calls.append(("sleep_model", name, level))
        if name in self.failure_models:
            raise RuntimeError(f"FakeModelOrchestrator: simulated sleep failure for {name}")
        if name not in self.processes or name in self.sleeping:
            return 0.0
        self.sleeping.add(name)
        return 0.0

    def wake_model(self, name: str, timeout: float = 900.0) -> float:
        self.calls.append(("wake_model", name))
        if name not in self.processes:
            raise RuntimeError(f"Cannot wake {name}: no live server process.")
        elapsed = self.wake_times.get(name, 0.0)
        if elapsed:
            time.sleep(elapsed)
        self.sleeping.discard(name)
        self.last_transition[name] = "sleep_wake"
        return elapsed

    def wait_until_serving(self, name: str, timeout: int = 60) -> None:
        self.calls.append(("wait_until_serving", name))
        if name not in self.processes:
            raise RuntimeError(f"{name} has no server process.")
        if name in self.sleeping:
            raise RuntimeError(f"{name} is asleep — wake_model() it first.")


# ---------------------------------------------------------------------------
# ModelPrefetchExecutor
# ---------------------------------------------------------------------------

class ModelPrefetchExecutor(PrefetchExecutor):
    """
    Prefetch executor that loads vLLM model servers in a background thread.

    - One active load at a time (max_concurrent=1 default; vLLM's tensor
      parallelism claims all assigned GPUs, so concurrent loads would conflict).
    - Cancellation records intent but does NOT interrupt the load thread.
    - Background thread marks the task WASTED (not COMPLETED) if cancelled.
    """

    executor_id = "model_prefetch"

    def __init__(
        self,
        orchestrator,   # ModelOrchestrator | FakeModelOrchestrator
        max_concurrent: int = 1,
        probes=None,    # ClusterProbes | None — if set, snapshots before/after load
        stop_wasted_models: bool = False,
        evict_conflicting: bool = False,
        sleep_wake: bool = False,
    ) -> None:
        self._orchestrator = orchestrator
        self._probes = probes
        # Sleep/wake swap arm (RuntimeConfig.sleep_wake_swaps): prefer WAKING
        # a slept engine (weights already in CPU RAM — H2D copy, seconds) over
        # cold-booting a new server process (~185 s).  Cold boot remains the
        # fallback for engines that have never been booted this run; either
        # way conflicting engines on the shared pool are SLEPT first, never
        # stopped.  See runtime/prefetch/sleep_wake.py for the sequencing and
        # the CPU-RAM budget note (level-2 fallback when RAM-constrained).
        self._sleep_wake = sleep_wake
        # Disjoint-pool mode: before booting the target engine, stop (and VRAM-
        # drain) any running engine whose GPU set overlaps the target's pool —
        # e.g. the planner still holds GPUs 0-3 when the first advanced
        # specialist pre-boots.  All of it happens in this executor's
        # background thread, off the workflow's critical path.
        self._evict_conflicting = evict_conflicting
        # Disjoint-pool mode: a cancelled pre-boot finishes loading on its own
        # pool (interrupting vLLM mid-load is unsafe), but the engine serves
        # nobody — stop it as soon as the load completes so the wasted
        # residency window is bounded and visible in the trace.
        self._stop_wasted = stop_wasted_models
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="model_prefetch",
        )
        self._futures: dict[str, Future] = {}
        self._cancelled_tasks: set[str] = set()
        self._lock = threading.Lock()

    def start(self, task: PrefetchTask) -> None:
        """Launch model loading in the background thread pool."""
        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        future = self._pool.submit(self._load_model, task)
        with self._lock:
            self._futures[task.task_id] = future

    def cancel(self, task: PrefetchTask) -> bool:
        """
        Record cancellation intent. The background load continues uninterrupted
        because stopping a half-loaded vLLM server corrupts GPU memory state.

        Returns False: the load cannot be physically interrupted.
        The scheduler will mark the task WASTED via the event it emits.
        """
        with self._lock:
            self._cancelled_tasks.add(task.task_id)
        return False

    def is_complete(self, task: PrefetchTask) -> bool:
        """True once the background thread has finished (success, failure, or wasted)."""
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            return future.done()
        # No future: task never started or was never tracked
        return task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS)

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        """Return timing metadata. Blocks briefly if the future is nearly done."""
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            try:
                return future.result(timeout=5.0)
            except Exception as exc:
                return {"elapsed_s": 0.0, "success": False, "error": str(exc)}
        elapsed = (
            (task.completed_at - task.started_at)
            if task.started_at is not None and task.completed_at is not None
            else 0.0
        )
        return {"elapsed_s": elapsed, "success": task.status == PrefetchStatus.COMPLETED}

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the thread pool."""
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, task: PrefetchTask) -> dict[str, Any]:
        """
        Run in the background thread. Calls orchestrator.start_model_measured().
        Snapshots hardware probes before/after if self._probes is set.
        After the load finishes, checks for a cancellation that arrived mid-load
        and marks the task WASTED instead of COMPLETED.

        Proactive-swap path (task.proactive_swap=True):
          Used when a long compute window (e.g. LAMMPS) provides GPU idle time.
          The compute tool (computation_task_screw_dislocation) stops the current
          model right before LAMMPS starts.  This background thread waits until
          the GPUs are free, then loads the next model concurrently with LAMMPS.
          This avoids stopping qwen_72b too early (before a bad-arg retry fires).
        """
        probe_before = self._probes.snapshot() if self._probes else None
        try:
            if task.proactive_swap:
                # Wait for the current model to be stopped by the compute tool.
                # Timeout after 10 min; fall back to stopping it ourselves.
                deadline = time.perf_counter() + 600.0
                while time.perf_counter() < deadline:
                    current = self._orchestrator.get_running_model()
                    if current is None:
                        break
                    time.sleep(5.0)
                else:
                    current = self._orchestrator.get_running_model()
                    if current and current != task.resource.name:
                        print(
                            f"[model_prefetch] Proactive swap timeout: stopping {current} "
                            f"(fallback) to load {task.resource.name}.",
                            flush=True,
                        )
                        self._orchestrator.stop_model(current)
                print(
                    f"[model_prefetch] Proactive swap: GPUs free — loading {task.resource.name}.",
                    flush=True,
                )

            if self._evict_conflicting:
                target_gpus = set(
                    self._orchestrator.models.get(task.resource.name, {})
                    .get("gpus", []))
                for other, proc in list(
                        getattr(self._orchestrator, "processes", {}).items()):
                    if other == task.resource.name:
                        continue
                    alive = proc.poll() is None if hasattr(proc, "poll") else True
                    other_gpus = set(
                        self._orchestrator.models.get(other, {}).get("gpus", []))
                    if alive and target_gpus & other_gpus:
                        print(f"[model_prefetch] Evicting {other} "
                              f"(GPUs {sorted(target_gpus & other_gpus)} needed "
                              f"for {task.resource.name} pre-boot).", flush=True)
                        self._orchestrator.stop_model(other)

            mechanism = None
            if self._sleep_wake:
                # Sleep/wake arm: wake-if-slept preferred over cold boot; the
                # planner (or any conflicting engine) is slept, never stopped.
                from runtime.prefetch.sleep_wake import swap_to_model
                swap_info = swap_to_model(self._orchestrator, task.resource.name)
                elapsed = swap_info["elapsed_s"]
                mechanism = swap_info["mechanism"]
            else:
                elapsed = self._orchestrator.start_model_measured(
                    task.resource.name,
                    metrics=None,
                )
            task.completed_at = time.perf_counter()

            probe_after = self._probes.snapshot() if self._probes else None
            probe_delta = (
                self._probes.delta(probe_before, probe_after).to_dict()
                if (probe_before and probe_after)
                else None
            )

            with self._lock:
                was_cancelled = task.task_id in self._cancelled_tasks

            if was_cancelled:
                task.status = PrefetchStatus.WASTED
                if self._stop_wasted:
                    try:
                        self._orchestrator.stop_model(task.resource.name)
                        print(f"[model_prefetch] Stopped wasted pre-boot "
                              f"{task.resource.name} (cancelled mid-load).",
                              flush=True)
                    except Exception as _exc:
                        print(f"[model_prefetch] WARNING: could not stop wasted "
                              f"{task.resource.name}: {_exc}", flush=True)
            elif task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED

            result: dict[str, Any] = {
                "elapsed_s": elapsed,
                "success": True,
                "wasted": was_cancelled,
            }
            # Sleep/wake arm: record HOW the engine became serving so the
            # parser can facet prefetch_completed on mechanism
            # ("sleep_wake" wake vs. "cold_boot").  Only emitted when the arm
            # is on — existing configs' traces stay byte-identical.
            if mechanism is not None:
                result["mechanism"] = mechanism
            # A successful boot read the full weight snapshot: report it as
            # measured bytes so Q4/lifecycle byte provenance is not "estimated"
            # for vllm_model tasks (size itself comes from the snapshot dir's
            # st_size sum — see _model_size_bytes in the chemgraph adapter).
            # A wake copies weights H2D from CPU RAM and reads no bytes from
            # storage, so bytes_staged only applies to cold boots.
            size_b = getattr(task.resource, "estimated_size_bytes", None)
            if size_b and mechanism in (None, "cold_boot"):
                result["bytes_staged"] = int(size_b)
                if elapsed and elapsed > 0:
                    result["gb_per_s"] = round(size_b / elapsed / 1e9, 3)
            if probe_delta:
                result["probe_delta"] = probe_delta
            return result

        except Exception as exc:
            if task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.FAILED
            task.error = str(exc)
            return {
                "elapsed_s": 0.0,
                "success": False,
                "error": str(exc),
            }
