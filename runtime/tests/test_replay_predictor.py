"""
Tests for scripts/replay_predictor.py.

The point of these tests is NOT to check that the predictor is good — it is to
check that the *scoring arithmetic* is right, using a stub predictor whose
output is fully controlled.  If coverage/lead/precision are computed wrongly,
the ablation table is wrong in a way that no amount of reading the numbers would
reveal, so the arithmetic is pinned here against hand-worked examples.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "replay_predictor", REPO / "scripts" / "replay_predictor.py"
)
rp = importlib.util.module_from_spec(_spec)
# Must be registered before exec_module: @dataclass resolves string annotations
# via sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["replay_predictor"] = rp
_spec.loader.exec_module(rp)

from runtime.events import ResourceSpec, PredictionResult  # noqa: E402


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class StubRegistry:
    """tool -> [ResourceSpec]."""

    def __init__(self, mapping: dict[str, list[str]]):
        self._m = {
            t: [ResourceSpec(resource_id=n, resource_type="vllm_model", name=n)
                for n in names]
            for t, names in mapping.items()
        }

    def get(self, tool):
        return list(self._m.get(tool, []))

    def all_tools(self):
        return list(self._m)


class ScriptedPredictor:
    """
    Emits a fixed set of (resource, offset) per prediction point.

    `script` is keyed by the prediction point index i, where i = -1 is the
    pre-first-tool_call point.  Missing keys emit nothing.
    """

    def __init__(self, script: dict[int, list[tuple[str, int]]]):
        self.script = script
        self._calls = 0

    def predict(self, step, recent_events, current_tool_calls,
                task_description="", plan_context=None):
        # Recover the prediction point: the replay passes current_tool_calls=[]
        # only at i = -1, and step=max(i,0) otherwise.
        i = -1 if not current_tool_calls else step
        self._calls += 1
        specs = []
        for name, off in self.script.get(i, []):
            specs.append(ResourceSpec(
                resource_id=name, resource_type="vllm_model", name=name,
                confidence=0.9, consumer_step_offset=off,
            ))
        return PredictionResult(step=max(step, 0), resources=specs,
                                confidence=0.9, predictor_id="stub")


def make_trace(tools, t0=1000.0, dt=10.0, plan=None):
    """Build an rp.Trace with one tool_call event per tool, 1 event each."""
    events = []
    if plan:
        events.append({"event_type": "plan_extracted", "epoch_time": t0 - 1.0,
                       "payload": {"tool_sequence": list(plan)}})
    steps = []
    for k, tool in enumerate(tools):
        pos = len(events)
        ep = t0 + k * dt
        events.append({"event_type": "tool_call", "epoch_time": ep,
                       "payload": {"tool": tool}})
        steps.append(rp.Step(k, tool, ep, pos))
    plan_points = [(0, list(plan))] if plan else []
    return rp.Trace(Path("stub.jsonl"), rp.UNLABELED, "none",
                    events, steps, plan_points, 0)


# ---------------------------------------------------------------------------


class TestRealizedNeeds(unittest.TestCase):

    def test_needs_are_registry_expansion_of_tool_calls(self):
        reg = StubRegistry({"a": ["R1"], "b": ["R1", "R2"], "c": []})
        tr = make_trace(["a", "b", "c"])
        sc = rp.replay_trace(tr, ScriptedPredictor({}), reg, 10)
        # a->{R1}, b->{R1,R2}, c->{} = 3 need instances
        self.assertEqual(sc.n_needs, 3)
        self.assertEqual(sc.n_steps, 3)
        # c contributes no needs and is invisible to coverage
        self.assertEqual(sc.n_steps_registry_covered, 2)

    def test_unmapped_tools_contribute_nothing(self):
        reg = StubRegistry({})
        tr = make_trace(["x", "y", "z"])
        sc = rp.replay_trace(tr, ScriptedPredictor({}), reg, 10)
        self.assertEqual(sc.n_needs, 0)
        self.assertEqual(sc.n_steps_registry_covered, 0)


class TestCoverageAndLead(unittest.TestCase):

    def test_no_predictions_gives_zero_coverage(self):
        reg = StubRegistry({"a": ["R1"], "b": ["R1"]})
        sc = rp.replay_trace(make_trace(["a", "b"]), ScriptedPredictor({}), reg, 10)
        self.assertEqual(sc.n_needs, 2)
        self.assertEqual(sc.n_needs_covered, 0)

    def test_prediction_before_first_tool_covers_step_zero(self):
        """i=-1 is a real prediction point; without it a plan-driven predictor
        could never cover the first need."""
        reg = StubRegistry({"a": ["R1"]})
        sc = rp.replay_trace(make_trace(["a"]),
                             ScriptedPredictor({-1: [("R1", 1)]}), reg, 10)
        self.assertEqual(sc.n_needs, 1)
        self.assertEqual(sc.n_needs_covered, 1)
        self.assertEqual(sc.lead_steps, [1])

    def test_lead_in_steps_and_seconds(self):
        reg = StubRegistry({"a": [], "b": [], "c": ["R1"]})
        # predict R1 at point i=0 (after tool a at t=1000); need lands at k=2 (t=1020)
        sc = rp.replay_trace(make_trace(["a", "b", "c"], t0=1000.0, dt=10.0),
                             ScriptedPredictor({0: [("R1", 2)]}), reg, 10)
        self.assertEqual(sc.n_needs_covered, 1)
        self.assertEqual(sc.lead_steps, [2])
        self.assertEqual(sc.lead_s, [20.0])

    def test_lead_uses_earliest_live_prediction(self):
        """Two warnings for the same need -> the lead is the EARLIER one."""
        reg = StubRegistry({"a": [], "b": [], "c": ["R1"]})
        sc = rp.replay_trace(make_trace(["a", "b", "c"]),
                             ScriptedPredictor({0: [("R1", 2)], 1: [("R1", 1)]}),
                             reg, 10)
        self.assertEqual(sc.lead_steps, [2])

    def test_credit_is_confined_to_the_consumption_interval(self):
        """
        THE KEY ASYMMETRY between cov% and covL%.

        R1 is needed at steps 0 and 2.  A single prediction at i=-1 covers the
        step-0 instance only; the step-2 instance got no fresh warning.  Strict
        coverage must be 1/2, loose coverage 2/2.  Without the confinement a
        predictor that fires once would score 100%.
        """
        reg = StubRegistry({"a": ["R1"], "b": []})
        tr = make_trace(["a", "b", "a"])
        sc = rp.replay_trace(tr, ScriptedPredictor({-1: [("R1", 1)]}), reg, 10)
        self.assertEqual(sc.n_needs, 2)
        self.assertEqual(sc.n_needs_covered, 1)
        self.assertEqual(sc.n_needs_covered_loose, 2)

    def test_re_prediction_recovers_strict_credit(self):
        reg = StubRegistry({"a": ["R1"], "b": []})
        tr = make_trace(["a", "b", "a"])
        sc = rp.replay_trace(tr, ScriptedPredictor({-1: [("R1", 1)],
                                                    1: [("R1", 1)]}), reg, 10)
        self.assertEqual(sc.n_needs_covered, 2)

    def test_prediction_at_or_after_the_need_does_not_count(self):
        """A warning issued at the step that needs it is not a warning."""
        reg = StubRegistry({"a": [], "b": ["R1"]})
        sc = rp.replay_trace(make_trace(["a", "b"]),
                             ScriptedPredictor({1: [("R1", 1)]}), reg, 10)
        # point i=1 is not < k=1, so the need at step 1 is uncovered
        self.assertEqual(sc.n_needs, 1)
        self.assertEqual(sc.n_needs_covered, 0)


class TestPrecision(unittest.TestCase):

    def test_exact_within_and_wasted(self):
        reg = StubRegistry({"a": [], "b": [], "c": ["R1"]})
        # claim R1 at offset 2 from point 0 -> lands exactly on step 2. correct.
        sc = rp.replay_trace(make_trace(["a", "b", "c"]),
                             ScriptedPredictor({0: [("R1", 2)]}), reg, 10)
        self.assertEqual(sc.n_pred_instances, 1)
        self.assertEqual(sc.n_hit_exact, 1)
        self.assertEqual(sc.n_hit_within, 1)
        self.assertEqual(sc.n_hit_ever, 1)

    def test_right_resource_wrong_step_is_within_but_not_exact(self):
        reg = StubRegistry({"a": [], "b": ["R1"], "c": []})
        # claim R1 at offset 2 from point 0 -> step 2, but R1 is really at step 1
        sc = rp.replay_trace(make_trace(["a", "b", "c"]),
                             ScriptedPredictor({0: [("R1", 2)]}), reg, 10)
        self.assertEqual(sc.n_hit_exact, 0)
        self.assertEqual(sc.n_hit_within, 1)   # inside (0, 2]
        self.assertEqual(sc.n_hit_ever, 1)

    def test_never_used_prediction_is_wasted(self):
        reg = StubRegistry({"a": [], "b": []})
        sc = rp.replay_trace(make_trace(["a", "b"]),
                             ScriptedPredictor({0: [("R9", 1)]}), reg, 10)
        self.assertEqual(sc.n_pred_instances, 1)
        self.assertEqual(sc.n_hit_ever, 0)     # -> wasted

    def test_specs_counts_raw_emissions_predictions_dedup_by_resource(self):
        """Volume must not be hidden by the dedup: same resource at two offsets
        is 2 specs but 1 prediction instance."""
        reg = StubRegistry({"a": [], "b": ["R1"]})
        sc = rp.replay_trace(make_trace(["a", "b"]),
                             ScriptedPredictor({0: [("R1", 1), ("R1", 2)]}), reg, 10)
        self.assertEqual(sc.n_specs, 2)
        self.assertEqual(sc.n_pred_instances, 1)


class TestFaceting(unittest.TestCase):

    def test_gpu_class_never_pools(self):
        self.assertEqual(rp._gpu_class({"gpus": ["NVIDIA L40S, 45 MiB"]}), "l40s")
        self.assertEqual(
            rp._gpu_class({"gpus": ["NVIDIA RTX PRO 6000 Blackwell Server Edition, 9 MiB"]}),
            "blackwell")
        self.assertEqual(rp._gpu_class({"gpus": []}), "gpu_unknown")
        self.assertEqual(rp._gpu_class({}), "gpu_unknown")

    def test_unlabeled_facet_is_its_own_bucket(self):
        self.assertEqual(rp.UNLABELED.workload, "UNLABELED")
        self.assertNotEqual(rp.UNLABELED, rp.Facet("chemgraph_swap", "l40s"))


class TestVariantGating(unittest.TestCase):

    def test_unsupported_kwargs_are_refused_not_silently_dropped(self):
        """
        A variant asking for a knob the installed predictor does not have must
        be reported UNSUPPORTED.  Silently dropping the kwarg would publish the
        baseline's numbers under a label claiming the knob was applied — exactly
        the class of error this harness exists to avoid.
        """
        pred, reason = rp.build_predictor(
            "bogus", {"signals": "full", "definitely_not_a_param": 3}, None)
        self.assertIsNone(pred)
        self.assertIn("definitely_not_a_param", reason)

    def test_supported_variant_builds(self):
        pred, reason = rp.build_predictor("full", {"signals": "full"}, None)
        self.assertIsNotNone(pred)
        self.assertEqual(reason, "")


class TestTraceParsing(unittest.TestCase):

    def test_only_tool_call_advances_a_step(self):
        """tool_end must not double-count: it is a completion record for a
        tool_call that was already counted."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text("\n".join(json.dumps(o) for o in [
                {"event_type": "tool_call", "epoch_time": 1.0,
                 "payload": {"tool": "a"}},
                {"event_type": "tool_end", "epoch_time": 2.0,
                 "payload": {"tool": "a"}},
                {"event_type": "llm_call", "epoch_time": 3.0,
                 "payload": {"model": "m"}},
                {"event_type": "tool_call", "epoch_time": 4.0,
                 "payload": {"tool": "b"}},
            ]) + "\n")
            tr = rp.load_trace(p, {}, {})
        self.assertEqual([s.tool for s in tr.steps], ["a", "b"])

    def test_malformed_lines_are_counted_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text('{"event_type":"tool_call","epoch_time":1.0,'
                         '"payload":{"tool":"a"}}\n'
                         'this is not json\n'
                         '\n')
            tr = rp.load_trace(p, {}, {})
        self.assertEqual(len(tr.steps), 1)
        self.assertEqual(tr.n_bad_lines, 1)

    def test_plan_context_is_causal(self):
        """A plan emitted after a prediction point must not be visible to it."""
        tr = make_trace(["a", "b"])
        tr.plan_points = [(5, ["a", "b"])]
        self.assertIsNone(tr.plan_before(0))
        self.assertIsNotNone(tr.plan_before(5))


class TestTableIntrospection(unittest.TestCase):

    def test_shipped_table_max_offset_is_reported(self):
        """The lookahead saturation ceiling must come from the file, not a
        constant in this script."""
        m = rp.table_max_offset()
        self.assertIsNotNone(m)
        self.assertGreaterEqual(m, 1)


if __name__ == "__main__":
    unittest.main()
