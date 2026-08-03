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

        Returns (hit, action, checkpoint_or_None) with three distinct outcomes:

          (True,  CONTINUE,       ckpt)  HIT   — ckpt's predicted tool fired.
                                                 Recorded as a hit.
          (False, <policy action>, ckpt) MISS  — ckpt's target step arrived and
                                                 a different tool fired.
                                                 Recorded as a miss.
          (True,  CONTINUE,       None)  NO OPINION — nothing pending, or the
                                                 only pending checkpoints are
                                                 still short of their target
                                                 step (out-of-scope inner-level
                                                 tool call).  Nothing consumed,
                                                 nothing recorded.

        NOTE: the boolean is only meaningful when the checkpoint is not None.
        Callers must branch on `ckpt is None` FIRST; treating the NO OPINION
        case as a hit is exactly the bug this method carried from e68d52b.
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

            # Only checkpoints that actually carry a prediction can be scored.
            eligible = [
                c for c in eligible
                if c.prediction is not None and c.prediction.resources
            ]
            if not eligible:
                return True, DivergenceAction.CONTINUE, None

            # ----------------------------------------------------------------
            # Selection has TWO independent questions, and conflating them is
            # what broke this detector between 2026-06-02 (e68d52b) and now:
            #
            #   (a) Which pending checkpoint is this tool call answering?
            #   (b) Given that checkpoint, was the prediction right or wrong?
            #
            # e68d52b answered (a) with "the checkpoint whose consumer_tool
            # matches the actual tool" and then fell through to `return True`
            # when nothing matched.  That makes "the agent did something
            # completely different from what we predicted" — the definition of
            # a divergence — indistinguishable from a hit, and keeps it out of
            # the accuracy accounting entirely.  It also made a legitimate
            # prefix match ("computation_task" predicted, the concrete
            # "computation_task_screw_dislocation" observed) pass the filter and
            # then fail the `==` test below, reporting a MATCH as a divergence.
            # Both directions were inverted.
            #
            # Scope is answered by the prediction's own declared target step,
            # not by the tool name.  A prediction created at step S says
            # "consumer_tool will run at expected_at_step".  Tool calls that
            # fire BEFORE that target step are, by the prediction's own claim,
            # not the call it was talking about: they are the inner-level tools
            # (create_working_folder, ...) that run inside a sub-conversation
            # while an outer-level model prediction is still in flight.  Those
            # neither consume nor diverge the checkpoint.  This is the same
            # gate the chemgraph adapter already uses (chemgraph.py:963).
            # ----------------------------------------------------------------
            def _predicted_tool(c: CheckpointRecord) -> str:
                return c.prediction.resources[0].consumer_tool

            def _target_step(c: CheckpointRecord) -> int:
                r = c.prediction.resources[0]
                # Every predictor stamps expected_at_step = step + offset.
                # Fall back to the same formula when it was left unset (0).
                return r.expected_at_step or (c.step + max(r.consumer_step_offset, 1))

            def _matches(c: CheckpointRecord) -> bool:
                """Exact, or a concrete specialisation of a generic prediction.

                "computation_task" predicted / "computation_task_screw_dislocation"
                observed is a MATCH: the predictor named the tool family and the
                agent called a member of it, so the prefetched resource is the
                right one.  The separator is required so that a prediction of
                "run_a" cannot claim "run_ase".
                """
                predicted = _predicted_tool(c)
                if not predicted:
                    return False
                return tool_name == predicted or tool_name.startswith(predicted + "_")

            # (a) A checkpoint whose predicted tool actually fired — regardless
            #     of whether its target step has arrived yet — is a HIT.
            matched = [c for c in eligible if _matches(c)]
            if matched:
                ckpt, hit = matched[0], True
            else:
                # (b) Nothing predicted this tool.  Any checkpoint whose target
                #     step has arrived was genuinely wrong: the tool it named
                #     did not run, this one did.  That is a real divergence and
                #     MUST be recorded as a miss.
                due = [c for c in eligible if step >= _target_step(c)]
                if not due:
                    # Out of scope: an inner-level tool firing while an
                    # outer-level prediction is still pending its target step.
                    # Consume nothing, score nothing; the checkpoint stays in
                    # the queue until it matches or ages out via max_age.
                    return True, DivergenceAction.CONTINUE, None
                ckpt, hit = due[0], False

            self._pending_queue.remove(ckpt)

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
