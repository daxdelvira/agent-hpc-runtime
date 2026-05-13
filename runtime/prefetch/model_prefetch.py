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
    """

    def __init__(
        self,
        load_times: dict[str, float] | None = None,
        failure_models: set[str] | None = None,
    ) -> None:
        self.load_times = load_times or {}
        self.failure_models = failure_models or set()
        self.processes: dict[str, object] = {}   # mimics ModelOrchestrator.processes

    def start_model_measured(
        self,
        name: str,
        metrics=None,
    ) -> float:
        """Simulate model loading: sleep for load_times[name] seconds."""
        if name in self.failure_models:
            raise RuntimeError(f"FakeModelOrchestrator: simulated failure for {name}")
        elapsed = self.load_times.get(name, 1.0)
        time.sleep(elapsed)
        self.processes[name] = object()   # non-None sentinel
        return elapsed

    def stop_model(self, name: str, wait_s: float = 5.0) -> None:
        self.processes.pop(name, None)

    def wait_until_ready(self, name: str, timeout: int = 60) -> None:
        pass   # already "ready" after start_model_measured returns

    def ensure_all_models_ready(self) -> None:
        pass


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
    ) -> None:
        self._orchestrator = orchestrator
        self._probes = probes
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
        """
        probe_before = self._probes.snapshot() if self._probes else None
        try:
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
            elif task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED

            result: dict[str, Any] = {
                "elapsed_s": elapsed,
                "success": True,
                "wasted": was_cancelled,
            }
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
