"""
adapters/chemgraph.py — Runtime adapter for ChemGraph (LangGraph/LangChain).

Extends ChemGraphCallbackHandler without modifying ChemGraph source code.

LangGraph callback propagation note
------------------------------------
In LangGraph 1.x, graph-level config callbacks propagate to *chain* and *tool*
callbacks (on_chain_start, on_tool_start) but NOT to LLM callbacks inside node
functions.  Node functions call llm.invoke(messages) without forwarding the
config, so on_llm_end / on_chat_model_start never fire.

We therefore use on_tool_start as the primary prediction/divergence hook:

  on_tool_start(A):
    1. Check _pending_checkpoint: was A the predicted consumer?  → HIT or MISS
    2. Make a new prediction for what comes AFTER A → schedule prefetch

  on_llm_end: passive — just calls super() for metrics.  When on_llm_end does
    fire (future LangGraph versions or direct LangChain invocations) the
    tool-start logic remains correct because it uses expected_at_step gating
    instead of a raw step comparison.

When config.mode == BASELINE the adapter behaves identically to the parent
class (zero additional overhead).

Usage
-----
    from runtime.adapters.chemgraph import make_runtime_callback

    cb = make_runtime_callback(
        predictor=MockPredictor("chemgraph"),
        scheduler=scheduler,
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
from runtime.predictor.plan_extractor import KNOWN_TOOLS, extract_plan
from runtime.prefetch.scheduler import PrefetchScheduler


class RuntimeChemGraphCallback(ChemGraphCallbackHandler):
    """
    ChemGraphCallbackHandler subclass that adds the runtime prediction/prefetch
    pipeline. All parent instrumentation (metrics CSV, WorkflowTracker JSONL)
    continues to work unchanged.

    New behaviour (skipped when mode=BASELINE):
    - on_tool_start: (1) check whether this tool matches the previous prediction,
                     (2) make a new prediction for what comes after this tool,
                     (3) schedule prefetches for predicted resources
    - on_llm_end: passive — just calls super(); no runtime logic here because
                  LangGraph 1.x does not propagate graph-level callbacks to
                  node-internal LLM invocations (llm.invoke without config=).
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
        self._conservative_until_step = 0
        self._pending_checkpoint: CheckpointRecord | None = None
        # Set once when the PlannerAgent node emits its plan (multi_agent workflow).
        self._plan_context = None

    @property
    def _is_active(self) -> bool:
        return self._config.mode != RuntimeMode.BASELINE and self._predictor is not None

    # ------------------------------------------------------------------
    # on_chain_start — guard against LangGraph passing None for serialized
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags=None,
        **kwargs,
    ) -> None:
        # LangGraph 1.x occasionally passes serialized=None for internal chain
        # nodes.  The parent handler calls serialized.get(...) which raises
        # AttributeError.  Substitute an empty dict so the parent can proceed.
        super().on_chain_start(
            serialized or {},
            inputs or {},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # on_chain_end — extract plan when PlannerAgent node finishes
    # ------------------------------------------------------------------

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        # Peek at the node name before super() pops it from _chain_names.
        node_name = self._chain_names.get(str(run_id), "")
        super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        if node_name != "PlannerAgent" or self._plan_context is not None:
            return
        if self._step > self._config.plan_extraction_horizon:
            return
        if not self._bus or not isinstance(outputs, dict):
            return

        # PlannerAgent returns {"messages": ["<PlannerResponse JSON>"]}
        # LangGraph may pass the raw string OR an AIMessage object (after reducer).
        messages = outputs.get("messages", [])
        if not messages:
            return
        raw = messages[-1]
        if isinstance(raw, str):
            raw_content = raw
        elif hasattr(raw, "content") and isinstance(raw.content, str):
            raw_content = raw.content
        else:
            return

        import json as _json
        try:
            plan_data = _json.loads(raw_content)
            tasks = plan_data.get("worker_tasks", [])
            combined_text = " ".join(t.get("prompt", "") for t in tasks)
        except Exception:
            return

        ctx = extract_plan(combined_text, KNOWN_TOOLS, step=0, source="planner_agent")
        if ctx is not None:
            self._plan_context = ctx
            self._bus.emit("plan_extracted", {
                "step": 0,
                "tool_sequence": ctx.tool_sequence,
                "n_mentions": ctx.n_mentions,
                "source": ctx.source,
            }, step=0)

    # ------------------------------------------------------------------
    # on_llm_end — passive; just records metrics via parent
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        # No runtime logic here: LangGraph 1.x does not propagate graph-level
        # config callbacks into node-internal llm.invoke() calls, so this hook
        # does not fire reliably.  All prediction logic lives in on_tool_start.

    # ------------------------------------------------------------------
    # on_tool_start — primary prediction / divergence hook
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
        # Always call parent (records metrics, WorkflowTracker tool_call event)
        super().on_tool_start(serialized, input_str, run_id=run_id,
                               parent_run_id=parent_run_id, tags=tags, **kwargs)

        if not self._is_active:
            return

        tool_name = serialized.get("name") or kwargs.get("name", "unknown_tool")
        self._step += 1
        if self._bus:
            self._bus.set_step(self._step)

        # ------------------------------------------------------------------
        # Phase 1: check the PREVIOUS prediction against this tool.
        # The pending checkpoint says "expect consumer_tool at expected_at_step".
        # Gate by expected_at_step so multi-step lookahead predictions (step_offset>1)
        # are not invalidated by intermediate tool calls.
        # ------------------------------------------------------------------
        if self._pending_checkpoint is not None:
            ckpt = self._pending_checkpoint
            if ckpt.prediction.resources:
                predicted_tool = ckpt.prediction.resources[0].consumer_tool
                expected_at_step = ckpt.prediction.resources[0].expected_at_step or 0

                if self._step >= expected_at_step:
                    hit = (tool_name == predicted_tool)
                    if hit:
                        self._checkpoint_store.resolve(ckpt.checkpoint_id, "validated")
                        if self._bus:
                            self._bus.emit_event(make_prediction_validated_event(
                                self._config.run_id, self._step, True,
                                ckpt.checkpoint_id, tool_name,
                            ))
                        if self._scheduler:
                            self._scheduler.on_resource_consumed(
                                ckpt.prediction.resources[0].resource_id,
                                consumed_at=time.perf_counter(),
                                current_step=self._step,
                            )
                    else:
                        self._checkpoint_store.resolve(
                            ckpt.checkpoint_id, "diverged", "INVALIDATE_ALL"
                        )
                        if self._bus:
                            self._bus.emit_event(make_divergence_detected_event(
                                self._config.run_id, self._step, predicted_tool,
                                tool_name, ckpt.checkpoint_id, "INVALIDATE_ALL",
                            ))
                        if not self._config.disable_divergence_cancellation:
                            if self._scheduler:
                                self._scheduler.cancel_all_pending(
                                    reason="divergence",
                                    checkpoint_id=ckpt.checkpoint_id,
                                    current_step=self._step,
                                )
                            self._conservative_until_step = (
                                self._step + self._config.conservative_mode_steps
                            )
                            if self._bus:
                                self._bus.emit_event(make_conservative_mode_event(
                                    self._config.run_id, self._step, "divergence",
                                    self._config.conservative_mode_steps,
                                ))
                    self._pending_checkpoint = None
                    self._checkpoint_store.expire_old(self._step)
                # else: too early — multi-step lookahead, keep pending

        # ------------------------------------------------------------------
        # Phase 2: make a new prediction for what comes AFTER this tool
        # ------------------------------------------------------------------
        in_conservative = self._step <= self._conservative_until_step
        if in_conservative:
            return

        recent_events = _read_recent_events(
            getattr(self._tracker, "log_path", None),
            n=self._config.predictor_context_events,
        )

        try:
            result: PredictionResult = self._predictor.predict(
                step=self._step,
                recent_events=recent_events,
                current_tool_calls=[{"name": tool_name}],
                task_description=self._task_description,
            )
        except Exception as e:
            if self._bus:
                self._bus.emit("prediction_error", {"error": str(e)}, step=self._step)
            return

        if not result.resources:
            return

        if self._bus:
            self._bus.emit_event(make_prediction_result_event(
                self._config.run_id, self._step, result
            ))

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

        if self._scheduler is not None and self._config.mode != RuntimeMode.OBSERVE_ONLY:
            for resource in result.resources:
                self._scheduler.schedule(
                    resource=resource,
                    current_step=self._step,
                    checkpoint_id=checkpoint.checkpoint_id,
                )


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
