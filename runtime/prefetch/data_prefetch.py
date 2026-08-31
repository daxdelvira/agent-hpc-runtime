"""
prefetch/data_prefetch.py — File staging and composite prefetch executors.

FileStagingExecutor  — copy files to $SCRATCH in a background thread.
CompositeExecutor    — route prefetch tasks to different executors by resource_type.

Example for a real-mode cluster run:

    from runtime.prefetch.data_prefetch import FileStagingExecutor, CompositeExecutor
    from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
    from runtime.prefetch.mace_prefetch import MacePrefetchExecutor
    from runtime.prefetch.simulated import SimulatedPrefetchExecutor

    executor = CompositeExecutor(
        executors={
            "vllm_model": ModelPrefetchExecutor(orchestrator),
            "mace_model": MacePrefetchExecutor(device="cuda"),
            "data_file":  FileStagingExecutor(),
        },
        default=SimulatedPrefetchExecutor(),
    )
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask


class FileStagingExecutor(PrefetchExecutor):
    """
    Prefetch executor that copies files to a fast scratch directory.

    Uses shutil.copy2 (user-space, no root required) in a background thread
    pool.  After a successful copy, task.resource.path is updated to point to
    the staged location so the consuming tool reads from $SCRATCH instead of
    the original (possibly slow NFS) path.

    On PACE set the SCRATCH env var to node-local or project scratch:
        export SCRATCH=/storage/scratch/...
    """

    executor_id = "file_staging"

    def __init__(
        self,
        scratch_dir: str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.scratch_dir = scratch_dir or os.environ.get(
            "SCRATCH", "/tmp/runtime_prefetch"
        )
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="file_staging",
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def start(self, task: PrefetchTask) -> None:
        if not task.resource.path:
            task.status = PrefetchStatus.FAILED
            task.error = "resource.path is None; cannot stage file"
            return

        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        future = self._pool.submit(self._stage, task)
        with self._lock:
            self._futures[task.task_id] = future

    def cancel(self, task: PrefetchTask) -> bool:
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None and future.cancel():
            task.status = PrefetchStatus.CANCELLED
            task.cancelled_at = time.perf_counter()
            return True
        # Already running — can't safely interrupt shutil.copy2 mid-stream.
        # Mark cancelled; the copy result will be ignored by the consumer.
        task.status = PrefetchStatus.CANCELLED
        task.cancelled_at = time.perf_counter()
        return False

    def is_complete(self, task: PrefetchTask) -> bool:
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            return future.done()
        return task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS)

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            try:
                return future.result(timeout=5.0)
            except Exception as exc:
                return {"elapsed_s": 0.0, "success": False, "error": str(exc),
                        "bytes_staged": 0}
        elapsed = (
            (task.completed_at - task.started_at)
            if task.started_at is not None and task.completed_at is not None
            else 0.0
        )
        # Fallback path (no future recorded): still report bytes so a data
        # prefetch never silently contributes 0 to the byte ledger.
        try:
            nbytes = os.path.getsize(task.resource.path) if task.resource.path else 0
        except OSError:
            nbytes = 0
        ok = task.status == PrefetchStatus.COMPLETED
        return {"elapsed_s": elapsed, "success": ok,
                "bytes_staged": nbytes if ok else 0}

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def _stage(self, task: PrefetchTask) -> dict[str, Any]:
        src = task.resource.path
        os.makedirs(self.scratch_dir, exist_ok=True)
        dst = str(Path(self.scratch_dir) / Path(src).name)
        # Size the source BEFORE the copy: it is what we are accountable for
        # moving, and it is still correct if the copy fails partway.
        try:
            nbytes = os.path.getsize(src)
        except OSError:
            nbytes = 0
        t0 = time.perf_counter()
        try:
            shutil.copy2(src, dst)
            elapsed = time.perf_counter() - t0
            task.completed_at = time.perf_counter()
            if task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED
                # NOTE: this redirect is currently INERT for every workload we
                # run. Nothing reads resource.path — AtomAgents rebuilds
                # "../potential_repository/<name>" itself
                # (screw_dislocation.py:54), so the consumer reads the ORIGINAL
                # file. The measurable benefit therefore comes only from the
                # side effect of having read src (its bytes are now in the OS
                # page cache), NOT from the copy landing anywhere useful. That
                # caps recovery at the read time (~5.5 s of a 129 s EAM
                # activation); the ~123.5 s of LAMMPS parse + spline
                # construction happens in the consumer process regardless of
                # where the bytes came from. Recovering that needs a pre-parsed
                # object handed over, e.g. a warm LAMMPS pool mirroring
                # mace_prefetch.py's _MACE_CACHE.
                task.resource.path = dst
            return {
                "elapsed_s": elapsed,
                "src": src,
                "dst": dst,
                "success": True,
                # bytes_staged/gb_per_s mirror ModelCacheStagingExecutor and
                # MegaMmapStagingExecutor so the speculation-cost ledger can
                # finally attribute bytes to data_file. Without these, every
                # data prefetch showed byte_source=unknown and contributed 0 to
                # useful/wasted byte totals — data I/O was structurally
                # invisible in the analysis even when it happened.
                "bytes_staged": nbytes,
                "gb_per_s": round(nbytes / 1e9 / elapsed, 3) if elapsed > 0 else None,
            }
        except Exception as exc:
            task.status = PrefetchStatus.FAILED
            task.error = str(exc)
            return {
                "elapsed_s": time.perf_counter() - t0,
                "success": False,
                "error": str(exc),
                "bytes_staged": 0,
            }


class CompositeExecutor(PrefetchExecutor):
    """
    Route prefetch tasks to different executors based on resource_type.

    Parameters
    ----------
    executors : dict mapping resource_type → PrefetchExecutor
    default   : executor for resource types not in the dict
                (defaults to SimulatedPrefetchExecutor if None)

    Example
    -------
        CompositeExecutor(
            executors={
                "vllm_model": ModelPrefetchExecutor(orchestrator),
                "mace_model": MacePrefetchExecutor(device="cuda"),
                "data_file":  FileStagingExecutor(),
            },
        )
    """

    executor_id = "composite"

    def __init__(
        self,
        executors: dict[str, PrefetchExecutor],
        default: PrefetchExecutor | None = None,
    ) -> None:
        self._executors = executors
        if default is None:
            from runtime.prefetch.simulated import SimulatedPrefetchExecutor
            default = SimulatedPrefetchExecutor()
        self._default = default
        self._routing: dict[str, PrefetchExecutor] = {}   # task_id → executor
        self._lock = threading.Lock()

    def _route(self, task: PrefetchTask) -> PrefetchExecutor:
        return self._executors.get(task.resource.resource_type, self._default)

    def executor_for_resource_type(self, resource_type: str) -> PrefetchExecutor:
        """Which executor would handle this resource type.

        The scheduler needs this to ask a CAPABILITY question about the
        executor that will actually run a given prediction — e.g. "can you
        evict the GPU incumbent?" — rather than about the router. Answering
        with the composite's own capabilities would let one resource type be
        admitted on the strength of another type's executor.
        """
        return self._executors.get(resource_type, self._default)

    @property
    def can_evict_gpu_occupants(self) -> bool:
        """True if ANY routed executor can. Callers that know the resource type
        should ask `executor_for_resource_type()` instead; this exists so the
        composite is not silently less capable than its parts."""
        return any(getattr(ex, "can_evict_gpu_occupants", False)
                   for ex in list(self._executors.values()) + [self._default])

    def start(self, task: PrefetchTask) -> None:
        ex = self._route(task)
        with self._lock:
            self._routing[task.task_id] = ex
        ex.start(task)

    def cancel(self, task: PrefetchTask) -> bool:
        with self._lock:
            ex = self._routing.get(task.task_id)
        return ex.cancel(task) if ex else False

    def is_complete(self, task: PrefetchTask) -> bool:
        with self._lock:
            ex = self._routing.get(task.task_id)
        return ex.is_complete(task) if ex else True

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        with self._lock:
            ex = self._routing.get(task.task_id)
        return ex.get_result(task) if ex else {}

    def shutdown(self, wait: bool = True) -> None:
        seen: set[int] = set()
        for ex in list(self._executors.values()) + [self._default]:
            if id(ex) not in seen and hasattr(ex, "shutdown"):
                ex.shutdown(wait=wait)
                seen.add(id(ex))
