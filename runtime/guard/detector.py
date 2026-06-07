"""
guard/detector.py — DivergenceDetector + PrefetchPolicy.

The DivergenceDetector is called at two moments:
  1. on_prediction(result) — creates a CheckpointRecord; returns checkpoint_id
  2. on_tool_about_to_execute(tool_name) — compares actual tool to prediction;
     triggers cancellation and conservative mode on mismatch

Accuracy is tracked per predictor_id so that the policy can adapt thresholds.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from runtime.config import RuntimeConfig
from runtime.events import PredictionResult
from runtime.guard.checkpoint import CheckpointRecord, CheckpointStore

if TYPE_CHECKING:
    from runtime.event_bus import EventBus
    from runtime.prefetch.scheduler import PrefetchScheduler


class DivergenceAction(str, Enum):
    CONTINUE              = "CONTINUE"
    INVALIDATE_PARTIAL    = "INVALIDATE_PARTIAL"
    INVALIDATE_ALL        = "INVALIDATE_ALL"
    FALLBACK_CONSERVATIVE = "FALLBACK_CONSERVATIVE"


@dataclass
class AccuracyTracker:
    """Rolling hit/miss counts per predictor."""
    hits: int = 0
    misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1

    @property
    def accuracy(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total(self) -> int:
        return self.hits + self.misses


class DivergenceDetector:
    """
    Maintains checkpoint state and detects divergence between predicted and
    actual tool sequences. Thread-safe.
    """

    def __init__(
        self,
        scheduler: "PrefetchScheduler | None",
        config: RuntimeConfig,
        bus: "EventBus | None" = None,
    ) -> None:
        self._scheduler = scheduler
        self._config = config
        self._bus = bus
        self._store = CheckpointStore(max_horizon=config.max_horizon)
        self._accuracy: dict[str, AccuracyTracker] = {}
        self._conservative_until_step = 0
        # Ordered queue of pending checkpoints (oldest first).
        # "Next tool call" matching: each tool execution consumes the oldest
        # pending prediction regardless of how many LLM calls (including
        # sub-conversation ones) happened in between.
        self._pending_queue: list[CheckpointRecord] = []
        self._lock = threading.Lock()

    def on_prediction(
        self,
        result: PredictionResult,
        step: int,
        log_position: int = 0,
    ) -> CheckpointRecord:
        """Create and store a checkpoint; return it so the adapter can attach prefetch IDs."""
        ckpt = CheckpointRecord(
            step=step,
            log_position=log_position,
            prediction=result,
        )
        self._store.add(ckpt)
        with self._lock:
            self._pending_queue.append(ckpt)
        return ckpt

    def on_tool_about_to_execute(
        self,
        tool_name: str,
        step: int,
    ) -> tuple[bool, DivergenceAction, CheckpointRecord | None]:
        """
        Compare actual tool to the oldest pending prediction.

        Uses "next tool call" matching: the prediction is validated against
        whatever tool actually fires next, not against an exact target step.
        This handles sub-conversation LLM calls that inflate the step counter
        without producing tool executions.

        Checkpoints expire after max_horizon * 8 steps (generous window so
        predictions survive through deep sub-conversations).

        Returns (hit, action, checkpoint_or_None).
        """
        with self._lock:
            # Expire checkpoints whose step window has passed.
            # 60 steps covers the longest observed AtomAgents workflow
            # (step 1 outer prediction → step 22+ second computation_task)
            # without expiring model-prefetch checkpoints early.
            max_age = max(self._config.max_horizon * 8, 60)
            self._pending_queue = [
                c for c in self._pending_queue
                if c.status == "pending" and (step - c.step) <= max_age
            ]
            # Only validate predictions made at an EARLIER step. A prediction
            # created at step N is about what comes AFTER the tool that fired
            # at step N — not the tool itself. Without this guard, the detector
            # would pop the freshly-created checkpoint and immediately mark it
            # as DIVERGE against the very tool that triggered the prediction.
            eligible = [c for c in self._pending_queue if c.step < step]
            if not eligible:
                return True, DivergenceAction.CONTINUE, None

            # Only pop a checkpoint if its consumer_tool matches the actual tool
            # (exact or prefix, e.g. "computation_task" matches
            # "computation_task_screw_dislocation").
            #
            # The original "oldest wins" strategy caused outer-level model
            # predictions (consumer_tool="computation_task") to be consumed and
            # diverged by inner-level tools like "create_working_folder" that
            # operate in a completely different sub-conversation scope.
            # Unmatched checkpoints stay in the queue until they expire via
            # max_age, so nothing is lost.
            def _matches(c: CheckpointRecord) -> bool:
                if not c.prediction or not c.prediction.resources:
                    return False
                predicted = c.prediction.resources[0].consumer_tool
                return tool_name == predicted or tool_name.startswith(predicted)

            matched = [c for c in eligible if _matches(c)]
            if not matched:
                return True, DivergenceAction.CONTINUE, None
            ckpt = matched[0]
            self._pending_queue.remove(ckpt)

        if ckpt.prediction is None or not ckpt.prediction.resources:
            return True, DivergenceAction.CONTINUE, None

        predicted_tool = ckpt.prediction.resources[0].consumer_tool
        hit = tool_name == predicted_tool

        predictor_id = ckpt.prediction.predictor_id
        if predictor_id not in self._accuracy:
            self._accuracy[predictor_id] = AccuracyTracker()
        self._accuracy[predictor_id].record(hit)

        if hit:
            self._store.resolve(ckpt.checkpoint_id, "validated")
            return True, DivergenceAction.CONTINUE, ckpt

        # Divergence — choose action based on policy
        action = self._choose_action(ckpt, step)
        self._store.resolve(ckpt.checkpoint_id, "diverged", action.value)

        if action in (DivergenceAction.INVALIDATE_ALL, DivergenceAction.FALLBACK_CONSERVATIVE):
            if self._scheduler:
                self._scheduler.cancel_all_pending(
                    reason="divergence",
                    checkpoint_id=ckpt.checkpoint_id,
                    current_step=step,
                )
            with self._lock:
                self._conservative_until_step = step + self._config.conservative_mode_steps

        return False, action, ckpt

    def is_conservative(self, step: int) -> bool:
        with self._lock:
            return step <= self._conservative_until_step

    def accuracy_for(self, predictor_id: str) -> float:
        tracker = self._accuracy.get(predictor_id)
        return tracker.accuracy if tracker else 0.0

    def accuracy_summary(self) -> dict[str, dict]:
        return {
            pid: {"accuracy": t.accuracy, "hits": t.hits, "misses": t.misses}
            for pid, t in self._accuracy.items()
        }

    def _choose_action(self, ckpt: CheckpointRecord, step: int) -> DivergenceAction:
        """Conservative default: always INVALIDATE_ALL on divergence.

        When disable_divergence_cancellation is set (ablation), record the
        miss but take no action — in-flight prefetches are left running.
        """
        if self._config.disable_divergence_cancellation:
            return DivergenceAction.CONTINUE
        return DivergenceAction.INVALIDATE_ALL
