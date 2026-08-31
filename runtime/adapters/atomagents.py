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
    ResourceSpec,
    make_checkpoint_created_event,
    make_conservative_mode_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prediction_validated_event,
)
from runtime.guard.detector import DivergenceDetector
from runtime.predictor.base import Predictor
from runtime.predictor.plan_extractor import KNOWN_TOOLS, PlanContext, extract_plan
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
        self._tool_call_count = 0
        self._plan_context: PlanContext | None = None
        self._plan_window_open = True
        self._lock = threading.Lock()
        self._installed = False
        # Model names already given a model_cache staging task. Staging the same
        # snapshot twice is pure waste: the shards are immutable, so a second
        # warm of the same path can only re-read bytes that are already cached.
        self._staged_models: set[str] = set()

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

        # Also watch the inner admin agent used inside computation_task
        # sub-conversations so tool calls there (e.g. create_screw_dislocation
        # loading an EAM potential file) are also tracked and validated.
        try:
            from atomagents.agents import admin as inner_admin  # type: ignore
            if inner_admin is not admin_agent:
                self._install_reply_handler(inner_admin)
                # The inner admin's default is_termination_msg fires on any message
                # whose last line is "TERMINATE" — including the model's bash-style
                # multi-call blocks that pack all tool calls + TERMINATE in one reply.
                # That kills the conversation before the text fallback can execute any
                # tool.  Override with a stricter check: only terminate when the entire
                # message content is exactly "TERMINATE" (no other tool calls present).
                def _inner_admin_terminate(msg: dict) -> bool:
                    if msg.get("tool_calls"):
                        return False
                    return str(msg.get("content", "")).strip() == "TERMINATE"
                inner_admin._is_termination_msg = _inner_admin_terminate
        except ImportError:
            pass

        # Override AutoGen's termination check with a state-aware closure.
        # Problem: the model uses TERMINATE_ALL both (a) prematurely before any
        # tool runs (bad) and (b) legitimately at the end of successful work (good).
        # Solution: allow any "TERMINATE*" string only after at least one tool call
        # has been observed; before that, only allow the bare word "TERMINATE"
        # (which the model never sends mid-task, only at a clean end).
        import re as _re
        _adapter_ref = self

        def _smart_terminate(msg: dict) -> bool:
            content = str(msg.get("content", ""))
            # Never terminate a message that still has pending structured tool calls
            if msg.get("tool_calls"):
                return False
            # Bare TERMINATE (word boundary — does not match TERMINATE_ALL because
            # '_' is a word character so there is no \b between E and _)
            if _re.search(r"\bTERMINATE\b", content):
                return True
            # TERMINATE_ALL / TERMINATE_PLAN / similar — only after tools have run,
            # AND only when this message does not also contain text-based tool calls
            # to execute (the model sometimes packs all work + TERMINATE_ALL into one
            # message; terminate only once the tool calls have been processed).
            if "TERMINATE" in content and _adapter_ref._tool_call_count > 0:
                if _extract_text_tool_name(content):
                    return False  # still has tool calls — let the fallback run first
                return True
            return False

        admin_agent._is_termination_msg = _smart_terminate

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

        # Attempt plan extraction while the planning phase is still running,
        # i.e. until the first real (non-plan_task) tool executes. The plan text
        # is written by the planner inside the plan_task sub-chat, whose nested
        # LLM calls also pass through this hook and inflate `step` — a small
        # step horizon alone never reaches it (plan_extracted was False on every
        # exp2/exp3 trial through 2026-07-16). The step-horizon check is kept as
        # a fallback for workflows without a planning sub-chat; horizon == 0
        # disables extraction entirely (the no_plan ablation).
        if (self._plan_context is None
                and self._config.plan_extraction_horizon > 0
                and (self._plan_window_open
                     or step <= self._config.plan_extraction_horizon)):
            content = _extract_response_content(response)
            if content:
                ctx = extract_plan(content, KNOWN_TOOLS, step=step)
                if ctx is not None:
                    self._plan_context = ctx
                    self._bus.emit("plan_extracted", {
                        "step": step,
                        "tool_sequence": ctx.tool_sequence,
                        "n_mentions": ctx.n_mentions,
                        "source": ctx.source,
                    }, step=step)

        self._bus.set_step(step)
        self._bus.emit("llm_call", {
            "step": step,
            "tool_calls": [tc.get("name", "") for tc in tool_calls],
            "model": model_name,
        }, step=step)

        if not tool_calls:
            return

        # Stamp consumption BEFORE scheduling: the dedup in schedule() skips a
        # resource whose previous task is still COMPLETED, so a repeatable
        # proactive-swap resource (e.g. reload qwen_72b every LAMMPS window)
        # must be marked USED here — the tool_call hook fires only after
        # scheduling and is too late to re-arm this window's prefetch.
        for tc in tool_calls:
            self._stamp_consumed(tc.get("name", ""), step)

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
                plan_context=self._plan_context,
            )
        except Exception as exc:
            self._bus.emit("prediction_error", {"error": str(exc)}, step=step)
            return

        if not result.resources:
            return

        # Pair each predicted engine with a host-side weight-staging task. Done
        # BEFORE the prediction event is emitted so the trace, the checkpoint
        # and the scheduler all see one prediction carrying both resources.
        self._pair_model_cache(result)

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

    def _pair_model_cache(self, result) -> int:
        """Append a `model_cache` resource for each predicted `vllm_model`.

        WHY THIS EXISTS. Making a served model usable is two costs, not one:
        moving its weight shards from storage into host memory, and bringing the
        engine up on the accelerator. AtomAgents only ever expressed the second
        -- it emitted `vllm_model` and `data_file` and nothing else -- so the
        host-I/O half of a model load had no resource to attach to. That is what
        blocked an external-tier comparison here: MegaMmapStagingExecutor is a
        drop-in for `model_cache` (see runtime/prefetch/megammap_stage.py), and
        with no such resource there was nothing for it to replace. ChemGraph has
        carried this since adapters/chemgraph.py:710; this is the AtomAgents
        counterpart, deliberately written to match it.

        THE PAIRING IS BY MODEL NAME, AND THAT IS LOAD-BEARING.
        extract_prefetch_lifecycle.py groups a staging task with its engine task
        by `name` (`g["model"] == name`, :194), and aggregations take max() over
        a gate_group rather than summing, because the two gate the SAME wall
        interval. Emit a cache resource whose `name` differs from its engine's
        and the two stop sharing a gate, at which point one stall is counted
        twice and every taxonomy number for this workload is inflated. The
        `name` below is therefore copied from the engine resource verbatim.

        Returns the number of resources added.
        """
        if (
            not self._config.stage_worker_cache
            or self._config.mode == RuntimeMode.BASELINE
            or "model_cache" in self._config.skip_resource_types
        ):
            return 0

        import hashlib

        added = []
        for res in list(result.resources):
            if res.resource_type != "vllm_model" or not res.name:
                continue
            if res.name in self._staged_models:
                continue
            path = self._config.model_paths.get(res.name)
            if not path:
                # No snapshot dir means nothing to warm. Staging a resource we
                # cannot name a path for would schedule a task that can only
                # fail, and a FAILED task is not the same as no task in the
                # lifecycle extractor -- it would surface as an unattributed
                # stall rather than as an absent mechanism.
                continue
            self._staged_models.add(res.name)
            added.append(ResourceSpec(
                resource_id="cache_" + hashlib.md5(res.name.encode()).hexdigest()[:12],
                resource_type="model_cache",
                name=res.name,
                path=path,
                estimated_size_bytes=_model_shard_bytes(path),
                # Host-side warm only: no GPU is touched and no running engine
                # is stopped, so this is always safe to cancel mid-flight.
                cancellation_safe=True,
                confidence=res.confidence,
                consumer_tool=res.consumer_tool,
                consumer_step_offset=res.consumer_step_offset,
                expected_at_step=res.expected_at_step,
            ))

        result.resources.extend(added)
        if added and self._bus:
            self._bus.emit("model_cache_paired", {
                "models": [r.name for r in added],
            }, step=self._step)
        return len(added)

    def _run_divergence_check(self, tool_name: str, step: int) -> None:
        """
        Called just before a tool executes at the given step.
        Checks the pending prediction targeting this step.
        """
        with self._lock:
            self._tool_call_count += 1
        if tool_name != "plan_task":
            # A real tool is about to run: the planning phase is over.
            self._plan_window_open = False
        self._bus.set_step(step)
        self._bus.emit("tool_call", {"tool": tool_name}, step=step)
        self._stamp_consumed(tool_name, step)
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

    def _stamp_consumed(self, tool_name: str, step: int) -> None:
        """Mark prefetched resources whose consumer tool is firing as consumed.

        The chemgraph adapter stamps consumption at its WorkerAgent gate;
        AtomAgents had no stamping at all (added 2026-07-10), so every
        completed prefetch was scored WASTED and — because the scheduler's
        dedup only re-arms on USED — a proactive-swap resource could fire at
        most once per run (observed: exp3 full_system t02 hid the first
        LAMMPS-window reload, then skipped the second window's).

        Consumption = the consumer tool is about to run.  For proactive-swap
        resources the consumer IS the compute tool that opens the window, so
        stamping here both credits the previous window's reload and re-arms
        the resource for this window's.
        """
        if self._scheduler is None:
            return
        try:
            from runtime.prefetch.base import PrefetchStatus
            for task in self._scheduler.all_tasks():
                res = task.resource
                if res is None or res.consumer_tool != tool_name:
                    continue
                # COMPLETED only: an IN_PROGRESS task here is usually the one
                # admitted moments ago by the prediction hook for THIS tool
                # firing — stamping it would score a load as consumed before
                # it finished.  A load still in flight when its consumer fires
                # is late; it stays unconsumed (honest) until a later firing.
                if task.status is PrefetchStatus.COMPLETED:
                    self._scheduler.on_resource_consumed(
                        res.resource_id,
                        consumed_at=time.perf_counter(),
                        current_step=step,
                    )
        except Exception:
            pass  # consumption accounting must never break the workflow

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
            except Exception as _exc:
                import sys
                print(f"[runtime] _on_llm_response error: {_exc}", file=sys.stderr)
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
            tool_name = None

            if tool_calls_raw:
                tool_name = _tool_name_from_call(tool_calls_raw[0])
            else:
                # vLLM hermes parser often emits tool calls as plain text in content
                content = last.get("content") or ""
                if content:
                    tool_name = _extract_text_tool_name(content)

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

def _model_shard_bytes(path: str | None) -> int | None:
    """Total weight-shard bytes under a model snapshot dir, or None if unknown.

    Uses the same shard enumeration the staging executors use, so the size the
    scheduler budgets against is the size that will actually be moved. None on
    any failure -- an absent estimate is handled downstream, a wrong one is not.
    """
    if not path:
        return None
    try:
        from runtime.prefetch.model_cache_prefetch import list_model_shards
        return sum(p.stat().st_size for p in list_model_shards(path)) or None
    except Exception:
        return None


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
            # vLLM hermes parser sometimes emits tool calls as plain text content
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content") or ""
            name = _extract_text_tool_name(content or "")
            if name:
                result.append({"name": name, "args": {}})
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


def _extract_text_tool_name(content: str) -> str:
    """
    Extract a function name from plain-text tool call content produced by the
    vLLM hermes parser.

    Handles two formats autogen_hook also handles:
    - JSON: ```json\\n{"function": "name", ...}\\n``` or bare {"name": ...}
    - Python call: ```python\\nfunc(key=val)\\n``` or bare func(key=val)

    NOTE: _parse_json_call / _parse_python_call are nested inside
    install_text_tool_call_fallback in autogen_hook and cannot be imported;
    this function reimplements the same logic inline.
    """
    if not content:
        return ""
    import ast
    import json as _json
    import re

    # --- JSON format ---
    for m in re.finditer(r"```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```", content):
        try:
            data = _json.loads(m.group(1))
            name = data.get("function") or data.get("name")
            if name:
                return str(name)
        except _json.JSONDecodeError:
            pass
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = _json.loads(stripped)
            name = data.get("function") or data.get("name")
            if name:
                return str(name)
        except _json.JSONDecodeError:
            pass

    # --- Python call syntax ---
    candidates: list[str] = []
    for m in re.finditer(r"```(?:python)?\s*\n?([\s\S]*?)\n?```", content):
        candidates.append(m.group(1).strip())
    candidates.append(stripped)
    for code in candidates:
        if not code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                    # Reject Python builtins / stdlib names — the model sometimes
                    # emits print(result) or json.dumps(...) which look like tool
                    # calls but are not registered workflow tools.
                    if name not in _PYTHON_BUILTINS and "_" in name:
                        return name
    return ""


# Common Python names that are NOT workflow tools.
_PYTHON_BUILTINS: frozenset[str] = frozenset({
    "print", "len", "str", "int", "float", "list", "dict", "tuple", "set",
    "range", "type", "input", "open", "format", "map", "filter", "zip",
    "enumerate", "sorted", "reversed", "sum", "min", "max", "abs", "round",
    "repr", "vars", "dir", "help", "id", "hash", "bool", "bytes", "hex",
    "oct", "bin", "chr", "ord", "eval", "exec", "compile", "isinstance",
    "issubclass", "hasattr", "getattr", "setattr", "delattr", "callable",
    "iter", "next", "super", "object", "classmethod", "staticmethod",
    "property", "Exception", "ValueError", "TypeError", "KeyError",
})


def _extract_response_content(response) -> str:
    """
    Extract the text content from an OpenAI ChatCompletion response.
    Returns empty string on any failure.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message", {})
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        return content or ""
    except Exception:
        return ""


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
