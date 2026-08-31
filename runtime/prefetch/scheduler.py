"""
prefetch/scheduler.py — PrefetchScheduler: decides when to start/cancel prefetches.

The scheduler connects the predictor output to the prefetch executor.
It is responsible for:
  - Deciding whether to start a prefetch (policy check)
  - Launching the executor in a non-blocking way
  - Tracking all in-flight PrefetchTasks
  - Cancelling tasks on divergence
  - Recording consumption events (for overlap metrics)
  - Emitting EventBus events for every decision

The scheduler holds no workflow-specific logic — that belongs in the predictor.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.events import (
    PredictionResult,
    ResourceSpec,
    make_prefetch_cancelled_event,
    make_prefetch_completed_event,
    make_prefetch_decision_event,
    make_prefetch_started_event,
    make_resource_consumed_event,
)
from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask

if TYPE_CHECKING:
    from runtime.event_bus import EventBus


class PrefetchScheduler:
    """
    Stateful scheduler that bridges predictor output → executor → event bus.

    Thread-safe: all public methods acquire _lock.
    """

    def __init__(
        self,
        executor: PrefetchExecutor,
        config: RuntimeConfig,
        bus: "EventBus | None" = None,
    ) -> None:
        self._executor = executor
        self._config = config
        self._bus = bus
        self._tasks: dict[str, PrefetchTask] = {}     # keyed by task_id
        self._resource_to_task: dict[str, str] = {}   # resource_id → task_id
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        resource: ResourceSpec,
        current_step: int,
        checkpoint_id: str,
        estimated_remaining_compute_s: float = 0.0,
    ) -> PrefetchTask | None:
        """
        Decide whether to prefetch `resource`; if yes, start the executor.

        Returns the PrefetchTask if started, None if skipped.
        """
        if self._config.mode in (RuntimeMode.BASELINE, RuntimeMode.OBSERVE_ONLY):
            return None

        # Deduplication: don't re-prefetch a resource already in flight
        with self._lock:
            existing_id = self._resource_to_task.get(resource.resource_id)
            if existing_id and self._tasks.get(existing_id, None):
                existing = self._tasks[existing_id]
                if existing.status in (PrefetchStatus.IN_PROGRESS, PrefetchStatus.COMPLETED):
                    self._emit("prefetch_decision", make_prefetch_decision_event(
                        self._config.run_id, current_step, resource.resource_id,
                        "skip", "already_in_flight_or_complete",
                    ).payload, current_step)
                    return None

        # Policy check
        should, reason = self._should_prefetch(resource, estimated_remaining_compute_s)
        self._emit("prefetch_decision", make_prefetch_decision_event(
            self._config.run_id, current_step, resource.resource_id,
            "start" if should else "skip", reason,
            estimated_load_s=resource.estimated_load_s,
        ).payload, current_step)

        if not should:
            return None

        task = PrefetchTask(
            resource=resource,
            status=PrefetchStatus.PENDING,
            checkpoint_id=checkpoint_id,
            workflow_step_at_start=current_step,
            predicted_at_step=current_step,
            proactive_swap=resource.proactive_swap,
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._resource_to_task[resource.resource_id] = task.task_id

        self._executor.start(task)
        self._emit("prefetch_started", make_prefetch_started_event(
            self._config.run_id, current_step, task.task_id,
            resource.resource_id, self._executor.executor_id,
        ).payload, current_step)

        # Poll for completion in a background thread to emit prefetch_completed
        threading.Thread(
            target=self._wait_for_completion,
            args=(task, current_step),
            daemon=True,
        ).start()

        return task

    def _wait_for_completion(self, task: PrefetchTask, step: int) -> None:
        """Background thread: wait for executor to finish; emit completed event."""
        timeout = 7200.0   # 2 hours max (covers 72B model load)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._executor.is_complete(task):
                break
            time.sleep(2.0)

        # Emit prefetch_completed for any terminal state (COMPLETED, WASTED, FAILED).
        # The condition intentionally covers all non-pending states so the event is
        # emitted even when the executor thread already transitioned the status
        # (e.g., ModelPrefetchExecutor sets COMPLETED inside _load_model before
        # this poll thread wakes up).
        if task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS):
            extras: dict = {}
            try:
                result = self._executor.get_result(task)
                elapsed = result.get("elapsed_s", 0.0)
                extras = {k: v for k, v in result.items() if k != "elapsed_s"}
            except Exception as e:
                task.status = PrefetchStatus.FAILED
                task.error = str(e)
                elapsed = 0.0

            self._emit("prefetch_completed", make_prefetch_completed_event(
                self._config.run_id, step, task.task_id, elapsed,
                status=task.status.value, extras=extras,
            ).payload, step)

    # ------------------------------------------------------------------
    # Divergence: cancel all pending/in-progress tasks
    # ------------------------------------------------------------------

    def cancel_all_pending(
        self,
        reason: str,
        checkpoint_id: str,
        current_step: int,
    ) -> list[PrefetchTask]:
        cancelled = []
        with self._lock:
            tasks_snapshot = list(self._tasks.values())

        for task in tasks_snapshot:
            cancellable_statuses = (
                PrefetchStatus.PENDING,
                PrefetchStatus.IN_PROGRESS,
                PrefetchStatus.COMPLETED,   # completed but not yet consumed → wasted
            )
            if task.checkpoint_id == checkpoint_id and task.status in cancellable_statuses:
                status_before = task.status.value
                was_completed = task.status == PrefetchStatus.COMPLETED
                self._executor.cancel(task)
                wasted = was_completed   # completed but diverged → wasted
                if wasted:
                    task.status = PrefetchStatus.WASTED
                self._emit("prefetch_cancelled", make_prefetch_cancelled_event(
                    self._config.run_id, current_step, task.task_id,
                    reason, status_before, wasted,
                ).payload, current_step)
                cancelled.append(task)

        return cancelled

    # ------------------------------------------------------------------
    # Resource consumption
    # ------------------------------------------------------------------

    def on_resource_consumed(
        self,
        resource_id: str,
        consumed_at: float,
        current_step: int,
    ) -> PrefetchTask | None:
        """
        Called when the workflow step that needs `resource_id` starts.
        Marks the task USED and emits resource_consumed.
        """
        with self._lock:
            task_id = self._resource_to_task.get(resource_id)
            task = self._tasks.get(task_id) if task_id else None

        if task is None or task.status in (PrefetchStatus.CANCELLED, PrefetchStatus.WASTED):
            self._emit("resource_consumed", make_resource_consumed_event(
                self._config.run_id, current_step, resource_id, None, "no_prefetch",
            ).payload, current_step)
            return None

        task.consumed_at = consumed_at
        task.status = PrefetchStatus.USED
        self._emit("resource_consumed", make_resource_consumed_event(
            self._config.run_id, current_step, resource_id, task.task_id, "used",
        ).payload, current_step)
        return task

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _should_prefetch(
        self,
        resource: ResourceSpec,
        estimated_remaining_compute_s: float,
    ) -> tuple[bool, str]:
        cfg = self._config

        # Ablation: resource-type filter
        if cfg.skip_resource_types and resource.resource_type in cfg.skip_resource_types:
            return False, f"resource_type_skipped ({resource.resource_type})"

        # Ablation: naive mode bypasses confidence and cancellable checks entirely
        if cfg.naive_prefetch:
            return True, "naive_prefetch"

        # ------------------------------------------------------------------
        # The confidence gate is structurally unreachable for a plan-only
        # prediction, and this is the fix — but ONLY when something downstream
        # can act on the admission.
        #
        # THE ARITHMETIC.  _PLAN_CONFIDENCE_DEFAULT = 0.80
        # (runtime/predictor/learned_predictor.py) is a FLOOR, applied as
        # max(entry.probability, 0.80); confidence_threshold = 0.85
        # (runtime/config.py).  So a plan-only prediction can never clear the
        # gate.  On the L40S rows of atomagents_exp3_aligned that is 34
        # `confidence_below_threshold (0.80 < 0.85)` skips against 10 admits.
        #
        # WHY THE CAPABILITY CHECK IS PART OF THE FIX AND NOT A KNOB.  Moving
        # this above the confidence gates on its own converts silent skips into
        # instant orchestrator failures: all three exp3 models declare
        # gpus [0,1,2,3] at tp=4, so staging one requires EVICTING the
        # incumbent, and without an actor that can do that the admission dies
        # with "Cannot start X: GPUs [0,1,2,3] occupied by Y. Call stop_model
        # first."  On the L40S exp3 rows that killed ALL 16 admitted model
        # prefetches — ten of them in <=1 ms, and four only after 600-918 s in
        # the executor's proactive-swap wait loop, which is the more expensive
        # half of the failure.  So the
        # bypass fires only when the executor says it can evict — the ordering
        # (T4a before the gate) is enforced by construction rather than by a
        # comment or a config flag someone can set early.
        #
        # NOTE ON THE ABSTRACTION, not on the tuning.  This is a local repair,
        # not the answer.  A fixed threshold asks "is this prediction likely
        # enough?" in isolation, when the question is whether the expected
        # saving is worth the GB — which is Eq. 1's job
        # (runtime/residency/contract.py).  The intended end state is that
        # admissions are priced by the arbitrator and this gate disappears, so
        # nothing here should grow into a second threshold to tune.
        # SCOPE, deliberately narrow. Only the confidence gates are bypassed:
        # the horizon condition is repeated here rather than jumped over, so a
        # proactive-swap prediction that is too far ahead still falls through
        # to the existing `horizon_exceeded` skip and reports it under that
        # name. No decision_reason string in any trace gains a new value or
        # changes which check produced it.
        if (resource.proactive_swap
                and resource.consumer_step_offset <= cfg.max_horizon
                and self._can_evict_gpu_occupants(resource)):
            return True, "proactive_swap_compute_window"

        if resource.confidence < cfg.confidence_threshold:
            return False, f"confidence_below_threshold ({resource.confidence:.2f} < {cfg.confidence_threshold})"
        if resource.consumer_step_offset > cfg.max_horizon:
            return False, f"horizon_exceeded ({resource.consumer_step_offset} > {cfg.max_horizon})"
        # For non-cancellable resources, require higher confidence
        if not resource.cancellation_safe and resource.confidence < min(0.9, cfg.confidence_threshold + 0.1):
            return False, "non_cancellable_resource_requires_higher_confidence"
        # Proactive-swap tasks: load during a known compute window (e.g. LAMMPS).
        # The window length is enforced via compute_window_min_s in tool_resources.json
        # rather than the generic overlap check, so skip the overlap heuristic here.
        if resource.proactive_swap:
            return True, "proactive_swap_compute_window"
        # If we know the load time and there's not enough compute left to overlap: skip
        if (
            resource.estimated_load_s is not None
            and estimated_remaining_compute_s > 0
            and resource.estimated_load_s > estimated_remaining_compute_s * 2
        ):
            return False, "insufficient_compute_time_for_overlap"
        return True, "confidence_above_threshold"

    def _can_evict_gpu_occupants(self, resource: ResourceSpec) -> bool:
        """Can whatever will execute THIS prediction take the GPUs off the
        incumbent?

        Asked of the executor that would actually run it, not of the router: a
        CompositeExecutor holding a GPU-capable model executor must not license
        a data_file admission. Anything that does not answer is treated as not
        capable, so an executor that has never heard of the question keeps the
        old behaviour exactly.
        """
        ex = self._executor
        router = getattr(ex, "executor_for_resource_type", None)
        if callable(router):
            try:
                ex = router(resource.resource_type) or ex
            except Exception:      # noqa: BLE001 - a router must never gate
                pass
        return bool(getattr(ex, "can_evict_gpu_occupants", False))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> PrefetchTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_for_resource(self, resource_id: str) -> PrefetchTask | None:
        with self._lock:
            tid = self._resource_to_task.get(resource_id)
            return self._tasks.get(tid) if tid else None

    def all_tasks(self) -> list[PrefetchTask]:
        with self._lock:
            return list(self._tasks.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict, step: int) -> None:
        if self._bus is not None:
            self._bus.emit(event_type, payload, step=step)
