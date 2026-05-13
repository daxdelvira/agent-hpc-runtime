"""
guard/checkpoint.py — WAL-inspired checkpoint records for speculative prefetch.

Each CheckpointRecord is created when a prediction is made and speculative
I/O may be started. It links:
  - the JSONL log position at prediction time
  - the prediction itself
  - all prefetch task IDs started under this prediction
  - the resolution status (validated / diverged / expired)

This gives us an audit trail of every speculative decision and its outcome,
which is the data we need to answer: "did prefetching help or hurt?"
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from runtime.events import PredictionResult


@dataclass
class CheckpointRecord:
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step: int = 0
    log_position: int = 0               # byte offset in JSONL at prediction time
    prediction: Optional[PredictionResult] = None
    prefetch_task_ids: list[str] = field(default_factory=list)
    status: str = "pending"             # "pending" | "validated" | "diverged" | "expired"
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    divergence_action: str | None = None


class CheckpointStore:
    """
    In-memory store for pending CheckpointRecords.

    Thread-safe. Records expire automatically after `max_horizon` steps
    (they are not needed for divergence detection beyond that window).
    """

    def __init__(self, max_horizon: int = 3) -> None:
        self._max_horizon = max_horizon
        self._records: dict[str, CheckpointRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: CheckpointRecord) -> None:
        with self._lock:
            self._records[record.checkpoint_id] = record

    def get(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._lock:
            return self._records.get(checkpoint_id)

    def get_pending_for_step(self, step: int) -> list[CheckpointRecord]:
        """Return all pending checkpoints whose expected step matches."""
        with self._lock:
            return [
                r for r in self._records.values()
                if r.status == "pending"
                and r.prediction is not None
                and r.step + r.prediction.horizon >= step
            ]

    def resolve(
        self,
        checkpoint_id: str,
        status: str,
        divergence_action: str | None = None,
    ) -> None:
        with self._lock:
            r = self._records.get(checkpoint_id)
            if r is not None:
                r.status = status
                r.resolved_at = time.time()
                r.divergence_action = divergence_action

    def expire_old(self, current_step: int) -> list[CheckpointRecord]:
        """Mark and return checkpoints whose horizon has passed without resolution."""
        expired = []
        with self._lock:
            for r in list(self._records.values()):
                if (
                    r.status == "pending"
                    and r.prediction is not None
                    and current_step > r.step + r.prediction.horizon + 1
                ):
                    r.status = "expired"
                    r.resolved_at = time.time()
                    expired.append(r)
        return expired

    def all_records(self) -> list[CheckpointRecord]:
        with self._lock:
            return list(self._records.values())
