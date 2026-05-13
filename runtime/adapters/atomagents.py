"""
adapters/atomagents.py — Runtime adapter for AtomAgents / AutoGen workflows.

Layers prediction + divergence detection on top of the existing autogen_hook
monkey-patches without modifying AtomAgents source code.

Interception points
-------------------
1. OpenAIWrapper.create (class-level, wraps the atomagents-patched version)
   Fires after every LLM response: extract tool_calls → predict → schedule prefetch.

2. Reply handler registered on admin_agent at position=1
   Fires before tool execution (before text_tool_call_fallback at position=2):
   extract tool_calls from message → divergence check.

Timing
------
The step counter increments inside _on_llm_response (point 1).  The reply
handler (point 2) reads the same counter, so both see the same step number for
a given LLM ↔ tool-execution pair:

  engineer LLM call  →  _on_llm_response(step=N)  →  prediction stored at target N+offset
  admin receives msg →  _divergence_reply(step=N)  →  on_tool_about_to_execute("tool", step=N)
                         → finds pending from step N-offset predicting target N → HIT or DIVERGE

Usage
-----
    from atomagents.instrumentation.autogen_hook import patch_autogen
    from runtime.adapters.atomagents import AtomAgentsRuntimeAdapter, make_atomagents_adapter

    patch_autogen()   # existing instrumentation (idempotent)

    adapter = make_atomagents_adapter(
        config=RuntimeConfig(mode=RuntimeMode.OBSERVE_ONLY, run_id="exp2-001"),
        predictor=MockPredictor("atomagents"),
    )
    adapter.install(admin_core)   # patches OpenAIWrapper + registers reply handler

    admin_core.initiate_chat(engineer_core, message=task_prompt)
    adapter.close()
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from typing import TYPE_CHECKING

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import (
    make_checkpoint_created_event,
    make_conservative_mode_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prediction_validated_event,
)
from runtime.guard.detector import DivergenceDetector
from runtime.predictor.base import Predictor
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor

if TYPE_CHECKING:
    pass


class AtomAgentsRuntimeAdapter:
    """
    Non-invasive runtime adapter for AtomAgents / AutoGen.

    Stateful: maintains a step counter and pending prediction state.
    Thread-safe: step counter and divergence state are lock-protected.

    The two public methods below (_on_llm_response, _run_divergence_check)
    are intentionally internal-but-testable: tests can call them directly
    with mock response objects, bypassing the autogen patching machinery.
    """

    def __init__(
        self,
        predictor: Predictor,
        scheduler: PrefetchScheduler,
        detector: DivergenceDetector,
        bus: EventBus,
        config: RuntimeConfig,
    ) -> None:
        self._predictor = predictor
        self._scheduler = scheduler
        self._detector = detector
        self._bus = bus
        self._config = config
        self._step = 0
        self._lock = threading.Lock()
        self._installed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self, admin_agent) -> None:
        """
        Activate the adapter.

        1. Ensures patch_autogen() has been called (idempotent).
        2. Wraps OpenAIWrapper.create at the class level to add prediction.
        3. Registers a divergence-check reply handler on admin_agent.

        Must be called after patch_autogen() and before initiate_chat().
        """
        if self._installed:
            return
        if not self._is_active:
            return

        # Ensure base atomagents instrumentation is applied first
        try:
            from atomagents.instrumentation.autogen_hook import patch_autogen
            patch_autogen()
        except ImportError:
            pass

        self._patch_oai_wrapper()
        self._install_reply_handler(admin_agent)
        self._installed = True

    def close(self) -> None:
        """Flush and close the event bus (only if we own it)."""
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Core logic (testable without autogen)
    # ------------------------------------------------------------------

    def _on_llm_response(self, response, messages: list) -> None:
        """
        Called after OpenAIWrapper.create returns.

        Increments the step counter, runs the predictor, creates a
        checkpoint, and schedules prefetches.
        """
        with self._lock:
            self._step += 1
            step = self._step

        if self._config.mode == RuntimeMode.BASELINE:
            return

        tool_calls = _extract_tool_calls_from_response(response)
        model_name = getattr(response, "model", "") or ""

        self._bus.set_step(step)
        self._bus.emit("llm_call", {
            "step": step,
            "tool_calls": [tc.get("name", "") for tc in tool_calls],
            "model": model_name,
        }, step=step)

        if not tool_calls:
            return

        if self._detector.is_conservative(step):
            return

        recent = _read_recent_events(
            self._bus.log_path,
            n=self._config.predictor_context_events,
        )

        try:
            result = self._predictor.predict(
                step=step,
                recent_events=recent,
                current_tool_calls=tool_calls,
            )
        except Exception as exc:
            self._bus.emit("prediction_error", {"error": str(exc)}, step=step)
            return

        if not result.resources:
            return

        self._bus.emit_event(make_prediction_result_event(
            self._config.run_id, step, result,
        ))

        log_pos = self._bus.current_log_position()
        ckpt = self._detector.on_prediction(result, step=step, log_position=log_pos)
        self._bus.emit_event(make_checkpoint_created_event(
            self._config.run_id, step, ckpt.checkpoint_id, log_pos,
        ))

        if self._config.mode not in (RuntimeMode.OBSERVE_ONLY,):
            for resource in result.resources:
                self._scheduler.schedule(
                    resource=resource,
                    current_step=step,
                    checkpoint_id=ckpt.checkpoint_id,
                )

    def _run_divergence_check(self, tool_name: str, step: int) -> None:
        """
        Called just before a tool executes at the given step.
        Checks the pending prediction targeting this step.
        """
        self._bus.set_step(step)
        hit, action, ckpt_out = self._detector.on_tool_about_to_execute(
            tool_name, step=step,
        )

        if ckpt_out is None:
            return

        if hit:
            self._bus.emit_event(make_prediction_validated_event(
                self._config.run_id, step, True,
                ckpt_out.checkpoint_id, tool_name,
            ))
        else:
            predicted = (
                ckpt_out.prediction.resources[0].consumer_tool
                if ckpt_out.prediction and ckpt_out.prediction.resources
                else "?"
            )
            self._bus.emit_event(make_divergence_detected_event(
                self._config.run_id, step, predicted, tool_name,
                ckpt_out.checkpoint_id, action.value,
            ))
            self._bus.emit_event(make_conservative_mode_event(
                self._config.run_id, step, "divergence",
                self._config.conservative_mode_steps,
            ))

    # ------------------------------------------------------------------
    # Internal: patching + handler installation
    # ------------------------------------------------------------------

    @property
    def _is_active(self) -> bool:
        return self._config.mode != RuntimeMode.BASELINE

    def _patch_oai_wrapper(self) -> None:
        """Wrap OpenAIWrapper.create to call _on_llm_response after each call."""
        try:
            from autogen.oai.client import OpenAIWrapper
        except ImportError:
            return

        if getattr(OpenAIWrapper.create, "_runtime_patched", False):
            return

        original_create = OpenAIWrapper.create
        adapter = self

        @functools.wraps(original_create)
        def runtime_create(self_wrapper, *args, **kwargs):
            response = original_create(self_wrapper, *args, **kwargs)
            try:
                adapter._on_llm_response(response, list(kwargs.get("messages", [])))
            except Exception:
                pass
            return response

        runtime_create._runtime_patched = True
        OpenAIWrapper.create = runtime_create

    def _install_reply_handler(self, admin_agent) -> None:
        """
        Register an observe-only reply handler on admin_agent.

        Fires at position=1 (before text_tool_call_fallback at position=2).
        Always returns (False, None) — it never intercepts the reply.
        """
        try:
            import autogen
        except ImportError:
            return

        adapter = self

        def _divergence_reply(recipient, messages, sender, config):
            if not messages:
                return False, None
            last = messages[-1]
            tool_calls_raw = last.get("tool_calls") or []
            if not tool_calls_raw:
                return False, None

            tool_name = _tool_name_from_call(tool_calls_raw[0])
            if not tool_name:
                return False, None

            with adapter._lock:
                step = adapter._step

            try:
                adapter._run_divergence_check(tool_name, step)
            except Exception:
                pass

            return False, None   # observe-only; let normal execution proceed

        admin_agent.register_reply(
            trigger=autogen.ConversableAgent,
            reply_func=_divergence_reply,
            position=1,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        with self._lock:
            return self._step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_response(response) -> list[dict]:
    """
    Normalise an OpenAI ChatCompletion response to [{name, args}, ...].

    Handles both the openai SDK object (response.choices[0].message.tool_calls)
    and dict-based representations.
    """
    result = []
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if not choices:
            return result
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message", {})
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return result
        for tc in tool_calls:
            name = _tool_name_from_call(tc)
            args: dict = {}
            fn = getattr(tc, "function", None)
            if fn is not None:
                raw_args = getattr(fn, "arguments", "{}") or "{}"
            elif isinstance(tc, dict):
                fn_d = tc.get("function", {})
                raw_args = fn_d.get("arguments", "{}") or "{}"
            else:
                raw_args = "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            if name:
                result.append({"name": name, "args": args})
    except Exception:
        pass
    return result


def _tool_name_from_call(tool_call) -> str:
    """Extract the function name from a tool_call (object or dict)."""
    if isinstance(tool_call, dict):
        return tool_call.get("function", {}).get("name", "") or tool_call.get("name", "")
    fn = getattr(tool_call, "function", None)
    if fn is not None:
        return getattr(fn, "name", "") or ""
    return getattr(tool_call, "name", "") or ""


def _read_recent_events(log_path: str, n: int) -> list[dict]:
    """Read the last n JSONL events from the EventBus / WorkflowTracker file."""
    if not log_path or not os.path.exists(log_path):
        return []
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_atomagents_adapter(
    config: RuntimeConfig | None = None,
    predictor: Predictor | None = None,
    scheduler: PrefetchScheduler | None = None,
    detector: DivergenceDetector | None = None,
    bus: EventBus | None = None,
    log_path: str | None = None,
) -> AtomAgentsRuntimeAdapter:
    """
    Convenience factory that wires all components together.

    Provide only the parts you want to customise; defaults are filled in:
    - config      : RuntimeConfig(mode=OBSERVE_ONLY)
    - predictor   : MockPredictor("atomagents")
    - scheduler   : PrefetchScheduler with SimulatedPrefetchExecutor
    - detector    : DivergenceDetector
    - bus         : standalone EventBus writing to logs/workflow_traces/
    """
    from runtime.predictor.mock_predictor import MockPredictor

    if config is None:
        config = RuntimeConfig(mode=RuntimeMode.OBSERVE_ONLY)

    if bus is None:
        if log_path is None:
            os.makedirs("logs/workflow_traces", exist_ok=True)
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = (
                f"logs/workflow_traces/runtime_trace_{ts}_{config.run_id}.jsonl"
            )
        bus = EventBus(run_id=config.run_id, log_path=log_path)

    # Try to share an already-open WorkflowTracker file so runtime events
    # appear interleaved with agent/tool/LLM events in a single JSONL.
    if getattr(bus, "_owns_file", True):
        try:
            from atomagents.instrumentation.workflow_tracker import tracker
            shared_file = getattr(tracker, "_file", None)
            if shared_file is not None and not shared_file.closed:
                bus = EventBus(run_id=config.run_id, shared_file=shared_file)
        except ImportError:
            pass

    if predictor is None:
        predictor = MockPredictor("atomagents")

    if scheduler is None:
        executor = SimulatedPrefetchExecutor()
        scheduler = PrefetchScheduler(executor=executor, config=config, bus=bus)

    if detector is None:
        detector = DivergenceDetector(scheduler=scheduler, config=config, bus=bus)

    return AtomAgentsRuntimeAdapter(
        predictor=predictor,
        scheduler=scheduler,
        detector=detector,
        bus=bus,
        config=config,
    )
