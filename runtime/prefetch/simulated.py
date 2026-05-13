"""
prefetch/simulated.py — SimulatedPrefetchExecutor: logs decisions, no real I/O.

Used in SIMULATED mode (and for all local development) to exercise the full
decision pipeline — predictor → scheduler → guard — without touching any
real models or files.

Every method records timing metadata so that `overlap_report.py` can show
what the timing numbers *would* be if this were a real prefetch executor.
The estimated_load_s from ResourceSpec is used to project "simulated" benefit.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask


class SimulatedPrefetchExecutor(PrefetchExecutor):
    """
    Records "would prefetch" decisions without performing any real I/O.

    Behaviour:
    - start()        : marks task IN_PROGRESS, records started_at, immediately
                       schedules a near-instant "completion" in a thread pool
                       (so is_complete() becomes True quickly)
    - cancel()       : marks task CANCELLED if not yet complete; always succeeds
    - is_complete()  : True once the background no-op has run
    - get_result()   : returns timing metadata + projected benefit_s
    """

    executor_id = "simulated"

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sim_prefetch")
        self._futures: dict[str, Future] = {}

    def start(self, task: PrefetchTask) -> None:
        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        def _noop():
            # Simulate the resource becoming available "instantly"
            # (real executors would do actual I/O here)
            task.completed_at = time.perf_counter()
            if task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED

        future = self._pool.submit(_noop)
        self._futures[task.task_id] = future

    def cancel(self, task: PrefetchTask) -> bool:
        future = self._futures.get(task.task_id)
        cancelled = False
        if future is not None:
            cancelled = future.cancel()
        task.status = PrefetchStatus.CANCELLED
        task.cancelled_at = time.perf_counter()
        return True   # simulated executor always reports successful cancellation

    def is_complete(self, task: PrefetchTask) -> bool:
        future = self._futures.get(task.task_id)
        if future is None:
            return task.status in (PrefetchStatus.COMPLETED, PrefetchStatus.CANCELLED)
        return future.done()

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        future = self._futures.get(task.task_id)
        if future is not None:
            future.result(timeout=1.0)   # should be instant
        elapsed = (
            (task.completed_at - task.started_at)
            if task.started_at is not None and task.completed_at is not None
            else 0.0
        )
        projected_benefit = task.resource.estimated_load_s or 0.0
        return {
            "elapsed_s": elapsed,
            "projected_benefit_s": projected_benefit,
            "simulated": True,
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
