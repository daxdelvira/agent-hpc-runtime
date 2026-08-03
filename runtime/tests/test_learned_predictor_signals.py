"""
test_learned_predictor_signals.py — LearnedPredictor signal combination (A1)
and parameterised lookahead horizon (A2).

Covers the 2026-08-03 change that

  * removed the `and not resources` guard so the plan signal and the learned
    transition signal run SIMULTANEOUSLY in "full" mode,
  * unioned the two candidate sets with dedup on
    (resource_id, consumer_step_offset) keeping the max confidence,
  * recorded the combination provenance in PredictionResult.predictor_id,
  * replaced the hardcoded `(1, 2)` offsets with range(1, lookahead + 1) and
    removed the `if resources: break` that made the table signal a
    "offset 1, or offset 2 if 1 was empty" lookup rather than a horizon.

Everything here runs offline against the shipped transition table
(runtime/predictor/data/learned_transitions.json) and small synthetic
registries; no GPU and no workflow run is required.
"""
import json
import statistics
from pathlib import Path

import pytest

from runtime.events import ResourceSpec
from runtime.predictor.learned_predictor import (
    _LEGACY_HORIZON,
    LearnedPredictor,
    _derive_offset_decay,
)
from runtime.predictor.plan_extractor import PlanContext
from runtime.predictor.resource_registry import ResourceRegistry
from runtime.predictor.transition_learner import TransitionTable

TABLE_PATH = Path(__file__).resolve().parents[1] / "predictor" / "data" / "learned_transitions.json"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _spec(name: str, consumer: str, rid: str | None = None) -> ResourceSpec:
    return ResourceSpec(
        resource_id=rid or f"rid_{name}",
        resource_type="data_file",
        name=name,
        consumer_tool=consumer,
        consumer_step_offset=1,
    )


@pytest.fixture()
def registry() -> ResourceRegistry:
    """
    Registry covering the ChemGraph tools used below.

    NOTE: the SHIPPED registry (runtime/predictor/data/tool_resources.json +
    MockPredictor tables) has no entry for extract_output_json, so the
    predictor can name that transition but cannot emit a ResourceSpec for it.
    These tests register one explicitly so the predictor logic is testable
    independently of that data gap.
    """
    reg = ResourceRegistry()
    reg.register("run_ase", _spec("mace_mp:medium", "run_ase"))
    reg.register("extract_output_json", _spec("ase_output_json", "extract_output_json"))
    reg.register("code_task", _spec("qwen_72b_text", "code_task"))
    reg.register("file_to_atomsdata", _spec("atoms_data", "file_to_atomsdata"))
    return reg


def _predict(pred: LearnedPredictor, tool: str, plan: PlanContext | None, step: int = 3):
    return pred.predict(
        step=step,
        recent_events=[],
        current_tool_calls=[{"name": tool}],
        plan_context=plan,
    )


def _keys(result) -> set[tuple[str, int]]:
    return {(r.resource_id, r.consumer_step_offset) for r in result.resources}


def _make(registry, signals="full", lookahead=None) -> LearnedPredictor:
    return LearnedPredictor(transitions_path=TABLE_PATH, registry=registry,
                            signals=signals, lookahead=lookahead)


# A plan whose next entry after run_ase is a tool the registry covers, so the
# plan signal DOES produce candidates.  This is the configuration in which the
# old `and not resources` guard silently suppressed the transition signal.
PLAN_AFTER_RUN_ASE = PlanContext(tool_sequence=["run_ase", "code_task"])


# ---------------------------------------------------------------------------
# The canonical missed transition
# ---------------------------------------------------------------------------

class TestCanonicalGap:
    def test_table_probability_is_read_from_the_json_not_assumed(self):
        raw = json.loads(TABLE_PATH.read_text())
        entries = raw["tool_transitions"]["run_ase"]["1"]
        entry = next(e for e in entries if e["target"] == "extract_output_json")
        assert entry["probability"] == pytest.approx(0.4045)
        assert entry["count"] == 36
        # It clears the predictor's default gate, so the only thing that could
        # have suppressed it was the signal guard.
        assert entry["probability"] >= LearnedPredictor(
            transitions_path=TABLE_PATH)._min_confidence

    def test_extract_output_json_predicted_while_plan_also_fires(self, registry):
        """The regression test for the removed `and not resources` guard."""
        result = _predict(_make(registry), "run_ase", PLAN_AFTER_RUN_ASE)
        names = {r.name for r in result.resources}
        assert "ase_output_json" in names       # from the transition signal
        assert "qwen_72b_text" in names         # from the plan signal
        eoj = next(r for r in result.resources if r.name == "ase_output_json")
        assert eoj.consumer_step_offset == 1
        assert eoj.confidence == pytest.approx(0.4045)

    def test_shipped_registry_cannot_emit_extract_output_json(self):
        """
        Documents a DATA gap, not a predictor gap: the shipped registry maps no
        resource to extract_output_json, so the prediction has nothing to
        prefetch.  If someone adds an entry to
        runtime/predictor/data/tool_resources.json this test should be updated,
        not deleted.
        """
        shipped = ResourceRegistry.merged(ResourceRegistry.from_json(),
                                          ResourceRegistry.from_mock_predictor())
        assert shipped.get("extract_output_json") == []


# ---------------------------------------------------------------------------
# A1: simultaneous signals, union + dedup, provenance
# ---------------------------------------------------------------------------

class TestSignalCombination:
    def test_full_is_superset_of_plan_only(self, registry):
        full = _predict(_make(registry, "full"), "run_ase", PLAN_AFTER_RUN_ASE)
        plan = _predict(_make(registry, "plan_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        assert _keys(plan)                      # the ablation arm predicts something
        assert _keys(plan) < _keys(full)        # strict superset

    def test_full_is_superset_of_transition_only(self, registry):
        full = _predict(_make(registry, "full"), "run_ase", PLAN_AFTER_RUN_ASE)
        tran = _predict(_make(registry, "transition_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        assert _keys(tran)
        assert _keys(tran) <= _keys(full)

    def test_plan_only_confidences_are_preserved_in_full(self, registry):
        """Dedup keeps the max, and the plan calibration is >= table prob."""
        plan = _predict(_make(registry, "plan_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        full = _predict(_make(registry, "full"), "run_ase", PLAN_AFTER_RUN_ASE)
        full_by_key = {(r.resource_id, r.consumer_step_offset): r.confidence
                       for r in full.resources}
        for r in plan.resources:
            key = (r.resource_id, r.consumer_step_offset)
            assert full_by_key[key] >= r.confidence

    def test_overlapping_candidate_is_not_double_counted(self, registry):
        """
        Both signals name run_ase at offset 1 (plan: sequence, table:
        smiles_to_coordinate_file +1 -> run_ase p=0.9538).  The union must emit
        ONE ResourceSpec for it, at the max confidence.
        """
        plan = PlanContext(tool_sequence=["smiles_to_coordinate_file", "run_ase"])
        result = _predict(_make(registry), "smiles_to_coordinate_file", plan)
        mace = [r for r in result.resources
                if r.name == "mace_mp:medium" and r.consumer_step_offset == 1]
        assert len(mace) == 1
        assert mace[0].confidence == pytest.approx(0.9538)
        assert result.predictor_id == "learned+both_agree"

    def test_tag_both_disagree(self, registry):
        """Plan and table name disjoint (resource, offset) pairs."""
        result = _predict(_make(registry), "run_ase", PLAN_AFTER_RUN_ASE)
        assert result.predictor_id == "learned+both_disagree"

    def test_tag_transition_only_when_plan_silent(self, registry):
        result = _predict(_make(registry), "run_ase", plan=None)
        assert result.resources
        assert result.predictor_id == "learned+transition_only"

    def test_tag_plan_only_when_table_silent(self, registry):
        """A tool absent from the transition table leaves only the plan signal."""
        plan = PlanContext(tool_sequence=["not_a_real_tool", "code_task"])
        result = _predict(_make(registry), "not_a_real_tool", plan)
        assert {r.name for r in result.resources} == {"qwen_72b_text"}
        assert result.predictor_id == "learned+plan_only"

    def test_tag_reaches_the_emitted_trace_payload(self, registry):
        """predictor_id is serialised into the prediction_result event payload."""
        from runtime.events import make_prediction_result_event
        result = _predict(_make(registry), "run_ase", PLAN_AFTER_RUN_ASE)
        ev = make_prediction_result_event("run-x", 3, result)
        assert ev.payload["predictor_id"] == "learned+both_disagree"

    def test_restricted_modes_do_not_borrow_the_other_signal(self, registry):
        plan_only = _predict(_make(registry, "plan_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        tran_only = _predict(_make(registry, "transition_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        assert {r.name for r in plan_only.resources} == {"qwen_72b_text"}
        assert "qwen_72b_text" not in {r.name for r in tran_only.resources}
        assert "ase_output_json" in {r.name for r in tran_only.resources}

    def test_plan_only_keeps_its_legacy_tag(self, registry):
        """The ablation arm's traces stay comparable with pre-change trials."""
        result = _predict(_make(registry, "plan_only"), "run_ase", PLAN_AFTER_RUN_ASE)
        assert result.predictor_id == "learned+plan"


# ---------------------------------------------------------------------------
# A2: lookahead horizon
# ---------------------------------------------------------------------------

class TestLookahead:
    def test_default_is_two(self, registry):
        assert _make(registry).lookahead == 2

    def test_table_signal_accumulates_across_offsets(self, registry):
        """
        The old code broke out of the offset loop as soon as offset 1 produced
        something, so offset 2 was unreachable for run_ase.  run_ase has
        entries at BOTH offsets (+1 run_ase p=0.5056, +2 run_ase p=0.7073).
        """
        result = _predict(_make(registry, "transition_only"), "run_ase", None)
        offsets = {r.consumer_step_offset for r in result.resources}
        assert offsets == {1, 2}

    def test_lookahead_one_stops_at_offset_one(self, registry):
        result = _predict(_make(registry, "transition_only", lookahead=1), "run_ase", None)
        assert {r.consumer_step_offset for r in result.resources} == {1}

    def test_longer_horizon_only_adds(self, registry):
        two = _predict(_make(registry, "full", lookahead=2), "run_ase", PLAN_AFTER_RUN_ASE)
        three = _predict(_make(registry, "full", lookahead=3), "run_ase", PLAN_AFTER_RUN_ASE)
        assert _keys(two) < _keys(three)
        # ... and never changes what the shorter horizon already emitted
        conf2 = {(r.resource_id, r.consumer_step_offset): r.confidence for r in two.resources}
        conf3 = {(r.resource_id, r.consumer_step_offset): r.confidence for r in three.resources}
        for key, conf in conf2.items():
            assert conf3[key] == pytest.approx(conf)

    def test_rejects_bad_lookahead(self, registry):
        with pytest.raises(ValueError):
            _make(registry, lookahead=0)

    def test_env_var_sets_the_default(self, registry, monkeypatch):
        monkeypatch.setenv("RUNTIME_PREDICTOR_LOOKAHEAD", "4")
        assert _make(registry).lookahead == 4
        monkeypatch.setenv("RUNTIME_PREDICTOR_LOOKAHEAD", "nonsense")
        assert _make(registry).lookahead == 2       # falls back, does not crash
        # an explicit argument always wins over the environment
        monkeypatch.setenv("RUNTIME_PREDICTOR_LOOKAHEAD", "4")
        assert _make(registry, lookahead=1).lookahead == 1


# ---------------------------------------------------------------------------
# A2: confidence decay derived from the table
# ---------------------------------------------------------------------------

class TestOffsetDecay:
    def test_decay_matches_an_independent_recomputation(self):
        """Recompute the median ratio straight from the JSON, not from code."""
        raw = json.loads(TABLE_PATH.read_text())["tool_transitions"]
        ratios = []
        for _src, offset_map in raw.items():
            base = {e["target"]: e["probability"] for e in offset_map.get("1", [])
                    if e["probability"] > 0}
            if not base:
                continue
            for off_str, entries in offset_map.items():
                off = int(off_str)
                if off <= 1:
                    continue
                for e in entries:
                    p1 = base.get(e["target"])
                    if p1 and e["probability"] > 0:
                        ratios.append((e["probability"] / p1) ** (1.0 / (off - 1)))
        assert len(ratios) == 9
        expected = statistics.median(ratios)
        assert expected == pytest.approx(0.8404, abs=1e-4)
        pred = LearnedPredictor(transitions_path=TABLE_PATH)
        assert pred.offset_decay == pytest.approx(expected)
        assert "n=9" in pred.offset_decay_provenance

    def test_decay_is_a_damping_factor(self):
        pred = LearnedPredictor(transitions_path=TABLE_PATH)
        assert 0.0 < pred.offset_decay <= 1.0

    def test_no_damping_inside_the_legacy_horizon(self, registry):
        """
        Offsets 1..2 must keep the exact confidences the pre-change code
        produced, so --lookahead 2 adds events without changing any.
        """
        pred = _make(registry, "transition_only", lookahead=3)
        result = _predict(pred, "run_ase", None)
        by_offset = {r.consumer_step_offset: r.confidence for r in result.resources
                     if r.name == "mace_mp:medium"}
        assert by_offset[1] == pytest.approx(0.5056)     # raw table probability
        assert by_offset[2] == pytest.approx(0.7073)     # raw table probability
        assert by_offset[3] == pytest.approx(0.5714 * pred.offset_decay)
        assert _LEGACY_HORIZON == 2

    def test_decay_falls_back_when_the_table_has_no_paired_evidence(self):
        empty = TransitionTable()
        decay, provenance = _derive_offset_decay(empty)
        assert decay == pytest.approx(0.84)
        assert "fallback" in provenance

    def test_decay_can_drop_a_far_out_candidate_below_the_gate(self, registry):
        """Damping is what stops a long horizon from flooding the scheduler."""
        pred = _make(registry, "transition_only", lookahead=6)
        result = _predict(pred, "run_ase", None)
        for r in result.resources:
            assert r.confidence >= pred._min_confidence
        # offset 3 survives (0.5714 * 0.84 = 0.48); nothing beyond offset 3
        # exists in the table, so the horizon is bounded by the data too.
        assert max(r.consumer_step_offset for r in result.resources) == 3


# ---------------------------------------------------------------------------
# Replay against a recorded trace (no GPU needed)
# ---------------------------------------------------------------------------

def _recorded_traces(limit: int = 8) -> list[Path]:
    """Recorded traces that carry BOTH a plan and tool calls (else the replay
    exercises at most one signal and proves nothing about their union)."""
    root = Path(__file__).resolve().parents[2] / "results" / "eval_q1_q4" / "runs"
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("chemgraph_*/*/*/trace.jsonl")):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if '"plan_extracted"' in text and '"tool_call"' in text:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def _replay(trace: Path, predictor: LearnedPredictor) -> list[dict]:
    """Mirror runtime/adapters/chemgraph.py:on_tool_start over a recorded trace."""
    plan_ctx = None
    step = 0
    seen: list[dict] = []
    preds: list[dict] = []
    for line in trace.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        seen.append(ev)
        if ev.get("event_type") == "plan_extracted":
            seq = (ev.get("payload") or {}).get("tool_sequence") or []
            if seq:
                plan_ctx = PlanContext(tool_sequence=list(seq), source="replay")
        elif ev.get("event_type") == "tool_call":
            tool = (ev.get("payload") or {}).get("tool")
            if not tool:
                continue
            step += 1
            res = predictor.predict(step=step, recent_events=seen[-10:],
                                    current_tool_calls=[{"name": tool}],
                                    plan_context=plan_ctx)
            preds.append({
                "step": step,
                "keys": {(r.resource_id, r.consumer_step_offset) for r in res.resources},
            })
    return preds


@pytest.mark.skipif(not _recorded_traces(), reason="no recorded eval traces on disk")
def test_full_is_superset_of_plan_only_on_recorded_traces():
    """Definition-of-done #1 on real traces rather than synthetic inputs."""
    shipped = ResourceRegistry.merged(ResourceRegistry.from_json(),
                                      ResourceRegistry.from_mock_predictor())
    n_strictly_larger = 0
    for trace in _recorded_traces():
        full = _replay(trace, LearnedPredictor(registry=shipped, signals="full"))
        plan = _replay(trace, LearnedPredictor(registry=shipped, signals="plan_only"))
        assert len(full) == len(plan)
        for f, p in zip(full, plan):
            assert p["keys"] <= f["keys"], f"{trace}: step {p['step']} lost a resource"
            if p["keys"] < f["keys"]:
                n_strictly_larger += 1
    # If this is 0 the change did not take: full would be a clone of plan_only.
    assert n_strictly_larger > 0
