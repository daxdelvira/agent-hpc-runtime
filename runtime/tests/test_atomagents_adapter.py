"""
test_atomagents_adapter.py — Unit tests for AtomAgentsRuntimeAdapter.

Tests bypass the autogen patching machinery entirely: they call
adapter._on_llm_response() and adapter._run_divergence_check() directly
with lightweight mock objects, then verify the right events appear in
the EventBus JSONL trace.

No autogen, no vLLM, no GPUs required.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from runtime.adapters.atomagents import (
    AtomAgentsRuntimeAdapter,
    _extract_tool_calls_from_response,
    _tool_name_from_call,
)
from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.guard.detector import DivergenceDetector
from runtime.prefetch.base import PrefetchStatus
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.predictor.mock_predictor import MockPredictor


# ---------------------------------------------------------------------------
# Mock objects — minimal ChatCompletion-like structures
# ---------------------------------------------------------------------------

class _Fn:
    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str):
        self.function = _Fn(name)
        self.id = f"call_{name}"


class _Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, tool_calls=None):
        self.message = _Message(tool_calls)


class MockChatCompletion:
    """Mimics openai.ChatCompletion with a single tool call."""
    def __init__(self, tool_name: str, model: str = "qwen-72b"):
        self.choices = [_Choice([_ToolCall(tool_name)])]
        self.model = model
        self.usage = None


class MockChatCompletionNoTool:
    """LLM response with no tool call (plain text)."""
    def __init__(self, model: str = "qwen-72b"):
        self.choices = [_Choice([])]
        self.model = model
        self.usage = None


class MockAdminAgent:
    """Minimal UserProxyAgent mock that records registered reply handlers."""
    def __init__(self):
        self._reply_funcs: list[tuple] = []

    def register_reply(self, trigger, reply_func, position=2, **kwargs):
        self._reply_funcs.append((trigger, reply_func, position))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_trace(tmp_path):
    return str(tmp_path / "trace.jsonl")


@pytest.fixture
def cfg():
    return RuntimeConfig(
        mode=RuntimeMode.SIMULATED,
        run_id="test-aa",
        confidence_threshold=0.85,
        max_horizon=2,
        conservative_mode_steps=2,
    )


@pytest.fixture
def observe_cfg():
    return RuntimeConfig(
        mode=RuntimeMode.OBSERVE_ONLY,
        run_id="test-aa-obs",
        confidence_threshold=0.85,
        max_horizon=2,
    )


@pytest.fixture
def baseline_cfg():
    return RuntimeConfig(mode=RuntimeMode.BASELINE, run_id="test-aa-base")


def _make_components(cfg, trace_path):
    bus = EventBus(run_id=cfg.run_id, log_path=trace_path)
    executor = SimulatedPrefetchExecutor()
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)
    predictor = MockPredictor("atomagents")
    adapter = AtomAgentsRuntimeAdapter(
        predictor=predictor,
        scheduler=scheduler,
        detector=detector,
        bus=bus,
        config=cfg,
    )
    return adapter, bus, scheduler, detector


def _read_events(path: str) -> list[dict]:
    events = []
    if not os.path.exists(path):
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events


def _event_types(events: list[dict]) -> list[str]:
    return [e["event_type"] for e in events]


# ---------------------------------------------------------------------------
# _extract_tool_calls_from_response
# ---------------------------------------------------------------------------

class TestExtractToolCalls:
    def test_extract_from_object(self):
        resp = MockChatCompletion("plan_task")
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "plan_task"

    def test_extract_no_tool_call(self):
        resp = MockChatCompletionNoTool()
        calls = _extract_tool_calls_from_response(resp)
        assert calls == []

    def test_extract_from_dict(self):
        resp = {
            "choices": [
                {"message": {"tool_calls": [
                    {"function": {"name": "computation_task_screw_dislocation", "arguments": "{}"}}
                ]}}
            ],
            "model": "qwen-72b",
        }
        calls = _extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "computation_task_screw_dislocation"

    def test_extract_malformed_returns_empty(self):
        calls = _extract_tool_calls_from_response(None)
        assert calls == []
        calls = _extract_tool_calls_from_response("not_a_response")
        assert calls == []


class TestToolNameFromCall:
    def test_from_object(self):
        tc = _ToolCall("plan_task")
        assert _tool_name_from_call(tc) == "plan_task"

    def test_from_dict_function_key(self):
        tc = {"function": {"name": "run_ase"}}
        assert _tool_name_from_call(tc) == "run_ase"

    def test_from_dict_name_key(self):
        tc = {"name": "plan_task"}
        assert _tool_name_from_call(tc) == "plan_task"

    def test_empty_dict(self):
        assert _tool_name_from_call({}) == ""


# ---------------------------------------------------------------------------
# AtomAgentsRuntimeAdapter — core logic
# ---------------------------------------------------------------------------

class TestOnLLMResponse:
    def test_emits_llm_call_event(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        events = _read_events(tmp_trace)
        assert "llm_call" in _event_types(events)

    def test_increments_step(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        assert adapter.step == 0
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        assert adapter.step == 1
        adapter._on_llm_response(MockChatCompletion("computation_task_screw_dislocation"), [])
        assert adapter.step == 2

    def test_emits_prediction_for_plan_task(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        events = _read_events(tmp_trace)
        et = _event_types(events)
        assert "prediction_result" in et
        assert "checkpoint_created" in et

    def test_no_prediction_for_unknown_tool(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("unknown_xyz_tool"), [])
        events = _read_events(tmp_trace)
        assert "prediction_result" not in _event_types(events)

    def test_no_prediction_for_no_tool_call(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletionNoTool(), [])
        events = _read_events(tmp_trace)
        assert "prediction_result" not in _event_types(events)

    def test_baseline_mode_emits_nothing(self, baseline_cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(baseline_cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        events = _read_events(tmp_trace)
        assert events == []

    def test_observe_only_no_prefetch_started(self, observe_cfg, tmp_trace):
        adapter, bus, scheduler, _ = _make_components(observe_cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        time.sleep(0.05)
        assert len(scheduler.all_tasks()) == 0

    def test_simulated_prefetch_started_for_plan_task(self, cfg, tmp_trace):
        adapter, bus, scheduler, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        time.sleep(0.05)
        # plan_task → W_Zhou04.eam.alloy + w_eam4.fs (conf=0.87 >= 0.85)
        assert len(scheduler.all_tasks()) >= 1


class TestRunDivergenceCheck:
    def test_hit_emits_prediction_validated(self, cfg, tmp_trace):
        adapter, bus, scheduler, detector = _make_components(cfg, tmp_trace)
        # Step 1: LLM response for plan_task → prediction stored at target step 2
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        # Step 2: computation_task_screw_dislocation executes (= predicted consumer_tool)
        adapter._run_divergence_check("computation_task_screw_dislocation", step=2)
        events = _read_events(tmp_trace)
        et = _event_types(events)
        assert "prediction_validated" in et
        assert "divergence_detected" not in et

    def test_miss_emits_divergence_detected(self, cfg, tmp_trace):
        adapter, bus, scheduler, detector = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        # Wrong tool executes at the consumer step
        adapter._run_divergence_check("wrong_tool", step=2)
        events = _read_events(tmp_trace)
        et = _event_types(events)
        assert "divergence_detected" in et
        assert "conservative_mode" in et
        assert "prediction_validated" not in et

    def test_no_pending_is_silent(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        # No prediction made → divergence check is a no-op
        adapter._run_divergence_check("any_tool", step=5)
        events = _read_events(tmp_trace)
        assert "divergence_detected" not in _event_types(events)
        assert "prediction_validated" not in _event_types(events)

    def test_miss_cancels_prefetch(self, cfg, tmp_trace):
        adapter, bus, scheduler, _ = _make_components(cfg, tmp_trace)
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        time.sleep(0.05)   # let simulated executor complete
        tasks_before = scheduler.all_tasks()
        assert len(tasks_before) >= 1

        adapter._run_divergence_check("wrong_tool", step=2)
        time.sleep(0.05)
        for t in scheduler.all_tasks():
            assert t.status in (PrefetchStatus.CANCELLED, PrefetchStatus.WASTED)


# ---------------------------------------------------------------------------
# Integration: two-step sequence matching the real workflow
# ---------------------------------------------------------------------------

class TestIntegrationAtomAgentsSequence:
    """
    Simulates the AtomAgents Exp2 tool sequence:
      step 1: plan_task          → predict W_Zhou04.eam.alloy for computation_task (target step 2)
      step 2: computation_task   → HIT; predict qwen_32b for next plan_task (target step 3)
      step 3: plan_task          → HIT for qwen_32b? (conf=0.65 < 0.85 → no prefetch,
                                    but prediction still made and checkpoint stored)
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trace = os.path.join(self.tmpdir, "trace.jsonl")
        self.cfg = RuntimeConfig(
            mode=RuntimeMode.SIMULATED,
            run_id="int-test",
            confidence_threshold=0.85,
            max_horizon=2,
        )
        self.adapter, self.bus, self.scheduler, self.detector = _make_components(
            self.cfg, self.trace,
        )

    def test_full_two_step_sequence(self):
        # Step 1: plan_task → predicts EAM files for computation at step 2
        self.adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        # Step 2: computation executes → check pending → HIT
        self.adapter._on_llm_response(
            MockChatCompletion("computation_task_screw_dislocation"), [],
        )
        self.adapter._run_divergence_check("computation_task_screw_dislocation", step=2)

        events = _read_events(self.trace)
        et = _event_types(events)

        assert "prediction_result" in et
        assert "checkpoint_created" in et
        assert "prediction_validated" in et
        assert "divergence_detected" not in et

    def test_diverge_at_consumer_step_cancels_prefetch(self):
        # Step 1: plan_task → predict + schedule prefetch
        self.adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        time.sleep(0.05)
        assert len(self.scheduler.all_tasks()) >= 1

        # Step 2: wrong tool executes → cancel prefetch
        self.adapter._on_llm_response(MockChatCompletion("wrong_tool"), [])
        self.adapter._run_divergence_check("wrong_tool", step=2)
        time.sleep(0.05)

        events = _read_events(self.trace)
        et = _event_types(events)
        assert "divergence_detected" in et
        assert "prefetch_cancelled" in et

    def test_conservative_mode_suppresses_prediction(self):
        # Trigger divergence at step 2
        self.adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        self.adapter._run_divergence_check("wrong_tool", step=2)
        # conservative_mode_steps defaults to 3 → conservative_until = 2+3=5
        assert self.detector.is_conservative(3)
        assert self.detector.is_conservative(4)
        assert self.detector.is_conservative(5)
        assert not self.detector.is_conservative(6)

        # Step 3: LLM call fires but should be suppressed by conservative mode
        event_count_before = len(_read_events(self.trace))
        self.adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        events = _read_events(self.trace)
        # llm_call event is still emitted, but no new prediction_result
        new_et = _event_types(events[event_count_before:])
        assert "llm_call" in new_et
        assert "prediction_result" not in new_et


# ---------------------------------------------------------------------------
# Reply handler installation (no real autogen needed)
# ---------------------------------------------------------------------------

class TestReplyHandlerInstallation:
    def test_install_registers_handler(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        admin_mock = MockAdminAgent()

        # Manually install just the reply handler (no real autogen patch)
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {
            "autogen": mock.MagicMock(ConversableAgent=object),
        }):
            adapter._install_reply_handler(admin_mock)

        assert len(admin_mock._reply_funcs) == 1
        _, reply_fn, position = admin_mock._reply_funcs[0]
        assert position == 1   # before text_tool_call_fallback at position=2

    def test_reply_handler_returns_false_none_always(self, cfg, tmp_trace):
        """The reply handler must never intercept the reply."""
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        admin_mock = MockAdminAgent()

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {
            "autogen": mock.MagicMock(ConversableAgent=object),
        }):
            adapter._install_reply_handler(admin_mock)

        _, reply_fn, _ = admin_mock._reply_funcs[0]

        # Simulate admin receiving a message with tool_calls
        messages = [{"tool_calls": [{"function": {"name": "plan_task"}}]}]
        result = reply_fn(admin_mock, messages, None, {})
        assert result == (False, None)

    def test_reply_handler_triggers_divergence_check(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        # Create a pending prediction at target step 2
        adapter._on_llm_response(MockChatCompletion("plan_task"), [])
        # step is now 1; pending targets step 2

        admin_mock = MockAdminAgent()
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {
            "autogen": mock.MagicMock(ConversableAgent=object),
        }):
            adapter._install_reply_handler(admin_mock)

        _, reply_fn, _ = admin_mock._reply_funcs[0]

        # Manually bump step to 2 (simulating second LLM call having already fired)
        with adapter._lock:
            adapter._step = 2

        messages = [{"tool_calls": [
            {"function": {"name": "computation_task_screw_dislocation"}}
        ]}]
        reply_fn(admin_mock, messages, None, {})

        events = _read_events(tmp_trace)
        # Should have emitted prediction_validated
        assert "prediction_validated" in _event_types(events)

    def test_reply_handler_no_tool_calls_is_silent(self, cfg, tmp_trace):
        adapter, bus, _, _ = _make_components(cfg, tmp_trace)
        admin_mock = MockAdminAgent()
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {
            "autogen": mock.MagicMock(ConversableAgent=object),
        }):
            adapter._install_reply_handler(admin_mock)

        _, reply_fn, _ = admin_mock._reply_funcs[0]
        # Message with no tool_calls
        messages = [{"content": "Here is the plan: ..."}]
        result = reply_fn(admin_mock, messages, None, {})
        assert result == (False, None)
