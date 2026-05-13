"""
prefetch/base.py — Abstract PrefetchExecutor interface and PrefetchTask lifecycle.

Implementations:
  simulated.py       — logs "would prefetch"; no real I/O
  model_prefetch.py  — wraps ModelOrchestrator.start_model_measured() in a thread
  data_prefetch.py   — shutil.copy2 to $SCRATCH for file staging
  mace_prefetch.py   — preloads MACE model into in-process cache
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.events import ResourceSpec


class PrefetchStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    CANCELLED   = "cancelled"   # intent cancelled; actual load may still be running
    FAILED      = "failed"
    WASTED      = "wasted"      # completed but divergence made the resource unnecessary
    USED        = "used"        # consumed successfully by workflow


@dataclass
class PrefetchTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource: ResourceSpec = field(default_factory=ResourceSpec)
    status: PrefetchStatus = PrefetchStatus.PENDING
    checkpoint_id: str = ""
    workflow_step_at_start: int = 0
    predicted_at_step: int = 0

    # Populated by executor:
    started_at: float | None = None
    completed_at: float | None = None
    cancelled_at: float | None = None
    error: str | None = None

    # Populated when workflow step consumes this resource:
    consumed_at: float | None = None
    # For research comparison: load latency that would have occurred without prefetch
    load_latency_without_prefetch: float | None = None

    def overlap_s(self) -> float:
        """Seconds of load time hidden behind concurrent compute."""
        if self.started_at is None or self.consumed_at is None:
            return 0.0
        end = self.completed_at or self.consumed_at
        return max(0.0, min(end, self.consumed_at) - self.started_at)

    def benefit_s(self) -> float:
        """Seconds saved: positive if prefetch completed before resource was needed."""
        if self.completed_at is None or self.consumed_at is None:
            return 0.0
        return max(0.0, self.consumed_at - self.completed_at)

    def waste_s(self) -> float:
        """Seconds wasted: prefetch arrived after resource was needed (no saving)."""
        if self.completed_at is None or self.consumed_at is None:
            return 0.0
        return max(0.0, self.completed_at - self.consumed_at)


class PrefetchExecutor(ABC):
    """
    Abstract base for all prefetch implementations.

    All methods must be thread-safe. `start()` must return immediately
    (kick off background work); callers check `is_complete()` or await
    `get_result()` at task consumption time.
    """

    @abstractmethod
    def start(self, task: PrefetchTask) -> None:
        """Launch background prefetch. Sets task.started_at and task.status=IN_PROGRESS."""

    @abstractmethod
    def cancel(self, task: PrefetchTask) -> bool:
        """
        Request cancellation. Returns True if successfully cancelled before starting.
        For resources where cancellation_safe=False (vLLM models), returns False and
        marks task CANCELLED but lets the underlying load continue; caller must handle
        the wasted work accounting.
        """

    @abstractmethod
    def is_complete(self, task: PrefetchTask) -> bool:
        """True if the prefetch has finished (completed or failed)."""

    @abstractmethod
    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        """
        Return timing metadata and any resource handle.
        Blocks briefly if the task is nearly complete; raises if failed.
        """

    @property
    @abstractmethod
    def executor_id(self) -> str:
        """Short string identifying this executor type in events."""
