"""
adapters/chemgraph.py — Runtime adapter for ChemGraph (LangGraph/LangChain).

Extends ChemGraphCallbackHandler without modifying ChemGraph source code.
Adds prediction and divergence-guard hooks at two key interception points:

  on_llm_end   → after LLM response: extract tool_calls, call predictor,
                  create checkpoint, schedule prefetches
  on_tool_start → before tool executes: compare to prediction, detect divergence

When config.mode == BASELINE the adapter behaves identically to the parent
class (zero additional overhead).

Usage
-----
    from runtime.adapters.chemgraph import make_runtime_callback

    cb = make_runtime_callback(
        predictor=MockPredictor("chemgraph"),
        scheduler=scheduler,
        guard=guard,
        config=cfg,
    )
    langgraph_config = {"configurable": {"thread_id": "1"}, "callbacks": [cb]}
    await cg.workflow.astream(inputs, stream_mode="values", config=langgraph_config)
"""
from __future__ import annotations

import sys
import os
import time
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ChemGraph", "src"))

from langchain_core.outputs import LLMResult

from chemgraph.instrumentation.langgraph_hook import ChemGraphCallbackHandler

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import (
    PredictionResult,
    make_checkpoint_created_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prediction_validated_event,
    make_conservative_mode_event,
)
from runtime.guard.checkpoint import CheckpointRecord, CheckpointStore
from runtime.predictor.base import Predictor
from runtime.prefetch.scheduler import PrefetchScheduler


class RuntimeChemGraphCallback(ChemGraphCallbackHandler):
    """
    ChemGraphCallbackHandler subclass that adds the runtime prediction/prefetch
    pipeline. All parent instrumentation (metrics CSV, WorkflowTracker JSONL)
    continues to work unchanged.

    New behaviour (skipped when mode=BASELINE):
    - on_llm_end  : call predictor → create checkpoint → schedule prefetches
    - on_tool_start: check divergence guard
    """

    def __init__(
        self,
        metrics_logger=None,
        workflow_tracker=None,
        gpu_index: int = 0,
        predictor: Predictor | None = None,
        scheduler: PrefetchScheduler | None = None,
        config: RuntimeConfig | None = None,
        bus: EventBus | None = None,
        task_description: str = "",
    ) -> None:
        super().__init__(metrics_logger, workflow_tracker, gpu_index)
        self._predictor = predictor
        self._scheduler = scheduler
        self._config = config or RuntimeConfig(mode=RuntimeMode.BASELINE)
        self._bus = bus
        self._task_description = task_description
        self._step = 0
        self._checkpoint_store = CheckpointStore(max_horizon=self._config.max_horizon)
        self._conservative_until_step = 0   # step at which conservative mode expires
        # Pending checkpoint for the current step (set in on_llm_end)
        self._pending_checkpoint: CheckpointRecord | None = None

    @property
    def _is_active(self) -> bool:
        return self._config.mode != RuntimeMode.BASELINE and self._predictor is not None

    # ------------------------------------------------------------------
    # on_llm_end — fires after every LLM response
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        # Always call parent first (records metrics CSV, WorkflowTracker)
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        if not self._is_active:
            return

        self._step += 1
        if self._bus:
            self._bus.set_step(self._step)

        # Extract tool_calls from the LLM response
        tool_calls = _extract_tool_calls(response)

        # Gather recent WorkflowTracker events as predictor context
        recent_events = _read_recent_events(
            getattr(self._tracker, "log_path", None),
            n=self._config.predictor_context_events,
        )

        # Skip prediction during conservative mode
        in_conservative = self._step <= self._conservative_until_step
        if in_conservative:
            return

        # Call predictor
        try:
            result: PredictionResult = self._predictor.predict(
                step=self._step,
                recent_events=recent_events,
                current_tool_calls=tool_calls,
                task_description=self._task_description,
            )
        except Exception as e:
            # Predictor failure must never abort the workflow
            if self._bus:
                self._bus.emit("prediction_error", {"error": str(e)}, step=self._step)
            return

        if not result.resources:
            return

        # Emit prediction event
        if self._bus:
            self._bus.emit_event(make_prediction_result_event(self._config.run_id, self._step, result))

        # Create checkpoint (WAL record before any speculative I/O)
        log_pos = self._bus.current_log_position() if self._bus else 0
        checkpoint = CheckpointRecord(
            step=self._step,
            log_position=log_pos,
            prediction=result,
        )
        self._checkpoint_store.add(checkpoint)
        self._pending_checkpoint = checkpoint

        if self._bus:
            self._bus.emit_event(make_checkpoint_created_event(
                self._config.run_id, self._step, checkpoint.checkpoint_id, log_pos,
            ))

        # Schedule prefetches for each predicted resource
        if self._scheduler is not None and self._config.mode != RuntimeMode.OBSERVE_ONLY:
            for resource in result.resources:
                self._scheduler.schedule(
                    resource=resource,
                    current_step=self._step,
                    checkpoint_id=checkpoint.checkpoint_id,
                )

    # ------------------------------------------------------------------
    # on_tool_start — fires before every tool execution
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags=None,
        **kwargs,
    ) -> None:
        # Always call parent (records metrics)
        super().on_tool_start(serialized, input_str, run_id=run_id,
                               parent_run_id=parent_run_id, tags=tags, **kwargs)

        if not self._is_active or self._pending_checkpoint is None:
            return

        tool_name = serialized.get("name") or kwargs.get("name", "unknown_tool")
        ckpt = self._pending_checkpoint
        pred = ckpt.prediction

        if not pred or not pred.resources:
            return

        predicted_tool = pred.resources[0].consumer_tool
        hit = tool_name == predicted_tool

        if hit:
            self._checkpoint_store.resolve(ckpt.checkpoint_id, "validated")
            if self._bus:
                self._bus.emit_event(make_prediction_validated_event(
                    self._config.run_id, self._step, True,
                    ckpt.checkpoint_id, tool_name,
                ))
            # Notify scheduler the resource is being consumed now
            if self._scheduler:
                self._scheduler.on_resource_consumed(
                    pred.resources[0].resource_id,
                    consumed_at=time.perf_counter(),
                    current_step=self._step,
                )
        else:
            self._checkpoint_store.resolve(ckpt.checkpoint_id, "diverged", "INVALIDATE_ALL")
            if self._bus:
                self._bus.emit_event(make_divergence_detected_event(
                    self._config.run_id, self._step, predicted_tool, tool_name,
                    ckpt.checkpoint_id, "INVALIDATE_ALL",
                ))
            # Cancel all prefetches for this checkpoint
            if self._scheduler:
                self._scheduler.cancel_all_pending(
                    reason="divergence",
                    checkpoint_id=ckpt.checkpoint_id,
                    current_step=self._step,
                )
            # Enter conservative mode
            self._conservative_until_step = self._step + self._config.conservative_mode_steps
            if self._bus:
                self._bus.emit_event(make_conservative_mode_event(
                    self._config.run_id, self._step, "divergence",
                    self._config.conservative_mode_steps,
                ))

        self._pending_checkpoint = None

        # Expire old checkpoints
        self._checkpoint_store.expire_old(self._step)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: LLMResult) -> list[dict]:
    """Extract tool_calls from a LangChain LLMResult."""
    tool_calls = []
    try:
        for gen_list in (response.generations or []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                calls = getattr(msg, "tool_calls", None) or []
                for call in calls:
                    if isinstance(call, dict):
                        tool_calls.append(call)
                    else:
                        # ToolCall pydantic object
                        tool_calls.append({"name": getattr(call, "name", ""), "args": getattr(call, "args", {})})
    except Exception:
        pass
    return tool_calls


def _read_recent_events(log_path: str | None, n: int) -> list[dict]:
    """Read the last `n` events from a WorkflowTracker JSONL file."""
    if not log_path or not os.path.exists(log_path):
        return []
    import json
    try:
        with open(log_path) as f:
            lines = f.readlines()
        events = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events
    except Exception:
        return []


def make_runtime_callback(
    predictor: Predictor | None = None,
    scheduler: PrefetchScheduler | None = None,
    guard=None,                   # reserved; divergence is handled inside the callback
    config: RuntimeConfig | None = None,
    bus: EventBus | None = None,
    metrics_logger=None,
    workflow_tracker=None,
    gpu_index: int = 0,
    task_description: str = "",
) -> RuntimeChemGraphCallback:
    """
    Convenience factory for RuntimeChemGraphCallback.

    If metrics_logger / workflow_tracker are not supplied, falls back to the
    ChemGraph module-level singletons (same as make_callback() in langgraph_hook).
    """
    if metrics_logger is None:
        try:
            from chemgraph.instrumentation.metrics_logger import get_metrics_logger
            metrics_logger = get_metrics_logger()
        except Exception:
            pass

    if workflow_tracker is None:
        try:
            from chemgraph.instrumentation.workflow_tracker import tracker
            workflow_tracker = tracker
        except Exception:
            pass

    if config is None:
        config = RuntimeConfig(mode=RuntimeMode.SIMULATED)

    # Create an EventBus that shares the WorkflowTracker's file if possible
    if bus is None and workflow_tracker is not None:
        shared_file = getattr(workflow_tracker, "_file", None)
        bus = EventBus(run_id=config.run_id, shared_file=shared_file)

    return RuntimeChemGraphCallback(
        metrics_logger=metrics_logger,
        workflow_tracker=workflow_tracker,
        gpu_index=gpu_index,
        predictor=predictor,
        scheduler=scheduler,
        config=config,
        bus=bus,
        task_description=task_description,
    )
