"""
test_adapter_tool_emission.py — tool_call emission mode for the AtomAgents adapter.

The defect: _divergence_reply (position=1) fires on every OBSERVED message that
names a tool and emits tool_call there, but execution happens later at
position=2.  Two consecutive LLM completions naming the same tool therefore
produce two tool_call events for one execution.

The fix cannot use a time threshold.  In the recorded corpus a duplicated
observation and a genuine back-to-back repeat abut without overlapping:

    duplicate observation   obs -> obs -> exec        obs gap ~0.35 s
    genuine repeat          obs -> exec -> obs -> exec  exec gap 0.947 s
      (results/eval_q1_q4/runs/atomagents_exp2/full_system/
       t06__20260729-073419__bf51974 — analyze_screw_core, 0.484 s and 0.458 s
       executions 0.947 s apart)

so any rule that collapses "same tool within N seconds" destroys real repeats.
The rule under test counts EXECUTIONS instead: each execution consumes the
nearest unmatched observation and emits one event; unmatched observations are
dropped.  No clock is consulted anywhere on that path, which is what these
tests pin down.

No autogen, no vLLM, no GPUs required.
"""
from __future__ import annotations

import inspect
import json
import os

import pytest

from runtime.adapters import atomagents as aa
from runtime.adapters.atomagents import (
    TOOL_CALL_EMISSION_ENV,
    TOOL_CALL_EMISSION_EXECUTION,
    TOOL_CALL_EMISSION_OBSERVATION,
    TOOL_CALL_SCHEMA_VERSION,
    AtomAgentsRuntimeAdapter,
)
from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.guard.detector import DivergenceDetector
from runtime.predictor.mock_predictor import MockPredictor
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class MockToolAgent:
    """Minimal UserProxyAgent stand-in exposing _function_map and register_reply."""

    def __init__(self, tools: dict):
        self._function_map = dict(tools)
        self._reply_funcs: list[tuple] = []
        self.calls: list[str] = []

    def register_reply(self, trigger, reply_func, position=2, **kwargs):
        self._reply_funcs.append((trigger, reply_func, position))


def _tool(agent: MockToolAgent, name: str):
    """A tool body that records that it really ran."""
    def fn(query: str = "", iter_num: int = 1) -> str:
        agent.calls.append(name)
        return f"[{name} result]"
    fn.__name__ = name
    return fn


def _build(tmp_path, mode: str | None, tools=("plan_task", "analyze_screw_core")):
    """Adapter + bus + a mock agent whose _function_map is hooked (if enabled)."""
    cfg = RuntimeConfig(
        mode=RuntimeMode.SIMULATED,
        run_id="test-emission",
        confidence_threshold=0.85,
        max_horizon=2,
        conservative_mode_steps=2,
    )
    if mode is not None:
        cfg.tool_call_emission = mode
    trace = str(tmp_path / "trace.jsonl")
    bus = EventBus(run_id=cfg.run_id, log_path=trace)
    scheduler = PrefetchScheduler(
        executor=SimulatedPrefetchExecutor(), config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)
    adapter = AtomAgentsRuntimeAdapter(
        predictor=MockPredictor("atomagents"),
        scheduler=scheduler,
        detector=detector,
        bus=bus,
        config=cfg,
    )
    agent = MockToolAgent({})
    agent._function_map = {n: _tool(agent, n) for n in tools}
    adapter._install_execution_hook(agent)
    return adapter, bus, agent, trace


def _tool_calls(trace: str) -> list[dict]:
    out = []
    with open(trace) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("event_type") == "tool_call":
                out.append(e)
    return out


def _events(trace: str, event_type: str) -> list[dict]:
    with open(trace) as fh:
        return [json.loads(l) for l in fh
                if l.strip() and json.loads(l).get("event_type") == event_type]


def _observe(adapter, tool: str, step: int) -> None:
    """One position=1 observation of `tool`."""
    adapter._run_divergence_check(tool, step=step)


def _execute(agent: MockToolAgent, tool: str) -> None:
    """One real execution through _function_map (what position=2 does)."""
    agent._function_map[tool](query="x")


# ---------------------------------------------------------------------------
# Flag off (default) — legacy behaviour must be untouched
# ---------------------------------------------------------------------------

class TestFlagOffIsLegacy:
    def test_default_is_observation_mode(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TOOL_CALL_EMISSION_ENV, raising=False)
        adapter, _bus, _agent, _trace = _build(tmp_path, mode=None)
        assert adapter.emits_on_execution is False
        assert (adapter.tool_call_emission_stats["mode"]
                == TOOL_CALL_EMISSION_OBSERVATION)

    def test_duplicate_observation_still_emits_twice(self, tmp_path):
        """The defect is preserved exactly when the flag is off.

        Byte-identical instrumentation is a corpus requirement: the Blackwell
        t09/t10/t11 baselines were collected this way and the tandem arm is
        queued against them.
        """
        adapter, bus, agent, trace = _build(tmp_path, mode=None)
        _observe(adapter, "plan_task", 1)
        _observe(adapter, "plan_task", 2)
        _execute(agent, "plan_task")          # one real execution
        bus.close()

        evs = _tool_calls(trace)
        assert len(evs) == 2, "flag off must reproduce the double emission"
        assert [e["step"] for e in evs] == [1, 2]
        # payload shape unchanged — no new keys leak into a legacy trace
        assert [e["payload"] for e in evs] == [
            {"tool": "plan_task"}, {"tool": "plan_task"}]

    def test_no_schema_marker_when_off(self, tmp_path):
        """A pre-fix trace stays identifiable by the ABSENCE of the marker."""
        adapter, bus, agent, trace = _build(tmp_path, mode=None)
        _observe(adapter, "plan_task", 1)
        bus.close()
        assert _events(trace, "runtime_schema") == []

    def test_execution_hook_is_inert_when_off(self, tmp_path):
        """_function_map is left completely alone in observation mode."""
        adapter, bus, agent, trace = _build(tmp_path, mode=None)
        assert not getattr(agent._function_map["plan_task"],
                           "_runtime_exec_hooked", False)
        _execute(agent, "plan_task")
        bus.close()
        assert _tool_calls(trace) == []       # execution emits nothing
        assert agent.calls == ["plan_task"]   # ...but the tool still ran


# ---------------------------------------------------------------------------
# Flag on — one event per execution
# ---------------------------------------------------------------------------

class TestExecutionMode:
    def test_duplicate_observation_emits_once(self, tmp_path):
        """obs, obs, exec  ->  ONE event.

        This is the plan_task pattern: 21 of the 41 Blackwell trials in the
        adjudication set emit two events and execute once, with both
        observations preceding the single execution.
        """
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        _observe(adapter, "plan_task", 1)
        _observe(adapter, "plan_task", 2)
        _execute(agent, "plan_task")
        bus.close()

        evs = _tool_calls(trace)
        assert len(evs) == 1
        # attributed to the observation that actually led to the run
        assert evs[0]["step"] == 2
        assert evs[0]["payload"]["tool"] == "plan_task"
        assert evs[0]["payload"]["emitted_on"] == TOOL_CALL_EMISSION_EXECUTION
        assert "observed" not in evs[0]["payload"]
        assert adapter.tool_call_emission_stats["observations"] == 2
        assert adapter.tool_call_emission_stats["events_emitted"] == 1
        assert adapter.tool_call_emission_stats["observations_unmatched"] == 1

    def test_two_real_executions_emit_twice(self, tmp_path):
        """obs, exec, obs, exec  ->  TWO events.

        computation_task_screw_dislocation runs twice per trial on purpose:
        W_screw_Zhou04 then W_screw_w_eam4, the two-potential comparison the
        workload exists to perform.
        """
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION,
            tools=("computation_task_screw_dislocation",))
        tool = "computation_task_screw_dislocation"
        _observe(adapter, tool, 8)
        _execute(agent, tool)
        _observe(adapter, tool, 11)
        _execute(agent, tool)
        bus.close()

        evs = _tool_calls(trace)
        assert len(evs) == 2
        assert [e["step"] for e in evs] == [8, 11]
        assert agent.calls == [tool, tool]

    def test_duplicate_and_genuine_repeat_together(self, tmp_path):
        """obs, obs, exec, obs, exec  ->  TWO events.

        The hardest recorded shape: 7 trials show 3 observations and 2 real
        executions of computation_task_screw_dislocation (e.g.
        results/eval_q1_q4/runs/atomagents_exp3/full_system/
        t01__20260710-194731__27b7b0f, steps 8, 9, 11).  The duplicate must
        collapse and the genuine repeat must survive, in the same trial.
        """
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION,
            tools=("computation_task_screw_dislocation",))
        tool = "computation_task_screw_dislocation"
        _observe(adapter, tool, 8)
        _observe(adapter, tool, 9)
        _execute(agent, tool)
        _observe(adapter, tool, 11)
        _execute(agent, tool)
        bus.close()

        evs = _tool_calls(trace)
        assert len(evs) == 2
        assert [e["step"] for e in evs] == [9, 11]

    def test_genuine_repeat_at_0_946s_is_not_collapsed(self, tmp_path,
                                                       monkeypatch):
        """The 0.947 s analyze_screw_core repeat survives.

        Replays the recorded timing of
        results/eval_q1_q4/runs/atomagents_exp2/full_system/
        t06__20260729-073419__bf51974 with a fake clock: observation at t,
        execution at t+0.086, second observation at t+0.947, second execution
        at t+1.033.  An earlier 1-second same-tool collapse rule destroyed
        this pair; the execution-counting rule must not.
        """
        clock = {"t": 1000.0}
        monkeypatch.setattr(aa.time, "monotonic", lambda: clock["t"])
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION,
            tools=("analyze_screw_core",))
        tool = "analyze_screw_core"

        clock["t"] = 1819.070; _observe(adapter, tool, 3)
        clock["t"] = 1819.156; _execute(agent, tool)
        clock["t"] = 1820.017; _observe(adapter, tool, 5)   # +0.947 s
        clock["t"] = 1820.103; _execute(agent, tool)
        bus.close()

        evs = _tool_calls(trace)
        assert len(evs) == 2, "a 0.947 s genuine repeat must not be collapsed"
        assert [e["step"] for e in evs] == [3, 5]

    def test_zero_gap_repeat_is_not_collapsed(self, tmp_path, monkeypatch):
        """No time threshold exists at all: even a 0 s gap yields two events."""
        monkeypatch.setattr(aa.time, "monotonic", lambda: 5.0)  # frozen clock
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION,
            tools=("analyze_screw_core",))
        tool = "analyze_screw_core"
        _observe(adapter, tool, 3)
        _execute(agent, tool)
        _observe(adapter, tool, 4)
        _execute(agent, tool)
        bus.close()
        assert len(_tool_calls(trace)) == 2

    def test_observation_without_execution_emits_nothing(self, tmp_path):
        """A message that names a tool the fallback then refuses to run."""
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        _observe(adapter, "plan_task", 1)
        bus.close()
        assert _tool_calls(trace) == []
        assert adapter.tool_call_emission_stats["observations_unmatched"] == 1

    def test_execution_without_observation_is_flagged(self, tmp_path):
        """An execution the position=1 handler never parsed is marked, not hidden."""
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        _execute(agent, "plan_task")
        bus.close()
        evs = _tool_calls(trace)
        assert len(evs) == 1
        assert evs[0]["payload"]["observed"] is False
        assert (adapter.tool_call_emission_stats[
            "executions_without_observation"] == 1)

    def test_distinct_tools_do_not_cross_match(self, tmp_path):
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        _observe(adapter, "plan_task", 1)
        _execute(agent, "analyze_screw_core")
        _execute(agent, "plan_task")
        bus.close()
        evs = _tool_calls(trace)
        assert [(e["payload"]["tool"], e["step"]) for e in evs] == [
            ("analyze_screw_core", 0),   # unmatched -> current step
            ("plan_task", 1),
        ]
        assert evs[0]["payload"]["observed"] is False
        assert "observed" not in evs[1]["payload"]

    def test_pending_observations_are_bounded(self, tmp_path):
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        for step in range(1, 40):
            _observe(adapter, "plan_task", step)
        bus.close()
        assert (adapter.tool_call_emission_stats["observations_unmatched"]
                <= aa._MAX_PENDING_OBS_PER_TOOL)


# ---------------------------------------------------------------------------
# Divergence-check semantics must not move
# ---------------------------------------------------------------------------

class TestDetectorSemanticsUnchanged:
    @pytest.mark.parametrize(
        "mode", [None, TOOL_CALL_EMISSION_EXECUTION])
    def test_detector_fires_on_observation_in_both_modes(self, tmp_path, mode):
        """on_tool_about_to_execute must keep firing per OBSERVATION.

        B1 validated all 24 recorded divergences tuple-for-tuple against the
        shipped detector; shifting this call point to the execution edge would
        invalidate that.  Only the bus emission moves.
        """
        adapter, bus, agent, trace = _build(tmp_path, mode=mode)
        seen: list[tuple] = []
        real = adapter._detector.on_tool_about_to_execute

        def spy(tool_name, step):
            seen.append((tool_name, step))
            return real(tool_name, step=step)

        adapter._detector.on_tool_about_to_execute = spy
        _observe(adapter, "plan_task", 1)
        _observe(adapter, "plan_task", 2)      # duplicate observation
        _execute(agent, "plan_task")           # single execution
        bus.close()

        # Two observations -> two detector calls, in BOTH modes.
        assert seen == [("plan_task", 1), ("plan_task", 2)]

    @pytest.mark.parametrize(
        "mode", [None, TOOL_CALL_EMISSION_EXECUTION])
    def test_tool_call_count_tracks_observations_in_both_modes(self, tmp_path,
                                                               mode):
        """_tool_call_count gates _smart_terminate; it must not change."""
        adapter, bus, agent, trace = _build(tmp_path, mode=mode)
        _observe(adapter, "plan_task", 1)
        _observe(adapter, "plan_task", 2)
        bus.close()
        assert adapter._tool_call_count == 2

    @pytest.mark.parametrize(
        "mode", [None, TOOL_CALL_EMISSION_EXECUTION])
    def test_plan_window_closes_on_observation_in_both_modes(self, tmp_path,
                                                             mode):
        adapter, bus, agent, trace = _build(tmp_path, mode=mode)
        assert adapter._plan_window_open is True
        _observe(adapter, "plan_task", 1)
        assert adapter._plan_window_open is True   # plan_task is not "real"
        _observe(adapter, "analyze_screw_core", 2)
        assert adapter._plan_window_open is False
        bus.close()


# ---------------------------------------------------------------------------
# Flag plumbing and trace marker
# ---------------------------------------------------------------------------

class TestFlagAndMarker:
    def test_env_var_enables_execution_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TOOL_CALL_EMISSION_ENV, TOOL_CALL_EMISSION_EXECUTION)
        adapter, bus, agent, trace = _build(tmp_path, mode=None)
        assert adapter.emits_on_execution is True
        bus.close()

    def test_config_field_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TOOL_CALL_EMISSION_ENV, TOOL_CALL_EMISSION_EXECUTION)
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_OBSERVATION)
        assert adapter.emits_on_execution is False
        bus.close()

    def test_unrecognised_value_falls_back_to_legacy(self, tmp_path):
        adapter, bus, agent, trace = _build(tmp_path, mode="excution")  # typo
        assert adapter.emits_on_execution is False
        bus.close()

    def test_schema_marker_written_in_execution_mode(self, tmp_path):
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        # install() is what writes the marker; drive it with a mock agent.
        adapter._installed = False
        adapter._patch_oai_wrapper = lambda: None
        adapter._install_reply_handler = lambda a: None
        adapter.install(agent)
        bus.close()

        markers = _events(trace, "runtime_schema")
        assert len(markers) == 1
        assert markers[0]["payload"] == {
            "tool_call_emission": TOOL_CALL_EMISSION_EXECUTION,
            "tool_call_schema_version": TOOL_CALL_SCHEMA_VERSION,
            "adapter": "atomagents",
        }

    def test_execution_hook_preserves_signature_and_return(self, tmp_path):
        """autogen_hook's _remap_args and type validation inspect the callable."""
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        hooked = agent._function_map["plan_task"]
        assert hooked.__name__ == "plan_task"
        assert list(inspect.signature(hooked).parameters) == ["query", "iter_num"]
        assert hooked(query="q") == "[plan_task result]"
        bus.close()

    def test_execution_hook_is_idempotent(self, tmp_path):
        """A second install must not double-wrap and double-count."""
        adapter, bus, agent, trace = _build(
            tmp_path, mode=TOOL_CALL_EMISSION_EXECUTION)
        adapter._install_execution_hook(agent)   # again
        _observe(adapter, "plan_task", 1)
        _execute(agent, "plan_task")
        bus.close()
        assert len(_tool_calls(trace)) == 1
        assert agent.calls == ["plan_task"]
