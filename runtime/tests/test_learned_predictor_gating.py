"""
test_learned_predictor_gating.py — plan-gated signal combination.

Covers the 2026-08-03 `signals="plan_gated"` mode, in which the PLAN filters
the TRANSITION TABLE's candidates (the opposite direction from
`_plan_confidence()`, where the table scores the plan's candidates).

The load-bearing test here is `test_existing_modes_bit_identical`: every
already-collected trial and every existing replay number is expressed in terms
of `plan_only` / `transition_only` / `full`, so if any of those three changed
behaviour, all of that data becomes incomparable and must be rerun on a GPU
allocation that this project does not have.  The gate knobs must be inert
outside `plan_gated`.

Offline only: shipped transition table + synthetic registries, no GPU.
"""
import copy
import itertools
from pathlib import Path

import pytest

from runtime.events import ResourceSpec
from runtime.predictor.learned_predictor import LearnedPredictor
from runtime.predictor.plan_extractor import PlanContext
from runtime.predictor.resource_registry import ResourceRegistry

TABLE_PATH = (Path(__file__).resolve().parents[1]
              / "predictor" / "data" / "learned_transitions.json")


# ---------------------------------------------------------------------------
# Helpers
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
    Registry where several TOOLS map onto the SAME RESOURCE.

    This mirrors the property that motivates the whole gating experiment: in
    the shipped registry 29 distinct tools in the transition table collapse
    onto only 7 distinct resources, so the plan and the table disagree about
    tools while naming the same resources.  A registry with a bijection between
    tools and resources would not exercise gate_scope="resource" at all.
    """
    reg = ResourceRegistry()
    reg.register("run_ase", _spec("shared_model", "run_ase", rid="R_shared"))
    reg.register("extract_output_json",
                 _spec("shared_model", "extract_output_json", rid="R_shared"))
    reg.register("smiles_to_coordinate_file",
                 _spec("geom_data", "smiles_to_coordinate_file", rid="R_geom"))
    reg.register("molecule_name_to_smiles",
                 _spec("name_db", "molecule_name_to_smiles", rid="R_name"))
    return reg


def _fingerprint(res) -> tuple:
    """Everything about a PredictionResult that any consumer can observe."""
    return (
        res.step,
        res.confidence,
        res.horizon,
        res.predictor_id,
        res.reasoning,
        res.context_events_used,
        tuple(sorted(
            (r.resource_id, r.name, r.resource_type, r.consumer_tool,
             r.consumer_step_offset, r.expected_at_step, r.confidence)
            for r in (res.resources or [])
        )),
    )


def _prediction_points(registry: ResourceRegistry):
    """
    A small corpus of (kwargs) prediction points spanning the interesting
    cases: plan present / absent, current_tool in the plan / not in the plan /
    missing entirely, and both real tools from the shipped table.
    """
    plan = PlanContext(
        tool_sequence=["molecule_name_to_smiles", "smiles_to_coordinate_file",
                       "run_ase", "extract_output_json"],
        source="test",
    )
    other_plan = PlanContext(tool_sequence=["run_ase", "run_ase"], source="test")
    events = [
        {"event_type": "tool_call", "payload": {"tool": "run_ase"}},
        {"event_type": "llm_call", "payload": {"model": "gpt-4o-mini"}},
    ]
    points = []
    for tool in (None, "run_ase", "smiles_to_coordinate_file",
                 "molecule_name_to_smiles", "a_tool_nobody_has_heard_of"):
        for plan_ctx in (None, plan, other_plan):
            points.append({
                "step": 3,
                "recent_events": events,
                "current_tool_calls": ([{"name": tool}] if tool else []),
                "task_description": "",
                "plan_context": plan_ctx,
            })
    return points


# ---------------------------------------------------------------------------
# THE REGRESSION GUARD
# ---------------------------------------------------------------------------

# Every gate knob, at a non-default value.  If any of these leaks into a
# non-gated mode the test below fails.
_GATE_KNOB_SETS = [
    {},
    {"gate_mode": "hard"},
    {"gate_mode": "soft", "gate_factor": 0.31},
    {"gate_mode": "cap", "gate_k": 1},
    {"gate_mode": "tail", "gate_tail": 0.99},
    {"gate_scope": "key"},
    {"gate_no_plan": "suppress"},
    {"gate_mode": "cap", "gate_k": 0, "gate_scope": "key",
     "gate_no_plan": "suppress", "gate_factor": 0.05, "gate_tail": 1.5},
]


@pytest.mark.parametrize("mode", ["plan_only", "transition_only", "full"])
def test_existing_modes_bit_identical(registry, mode):
    """
    plan_only / transition_only / full must be UNAFFECTED by every gate knob.

    Rationale in the module docstring: these three labels index all previously
    collected trials and all existing replay numbers.
    """
    base = LearnedPredictor(registry=registry, signals=mode, lookahead=2)
    baseline = [_fingerprint(base.predict(**pt)) for pt in _prediction_points(registry)]

    for knobs in _GATE_KNOB_SETS:
        p = LearnedPredictor(registry=registry, signals=mode, lookahead=2, **knobs)
        got = [_fingerprint(p.predict(**pt)) for pt in _prediction_points(registry)]
        assert got == baseline, f"signals={mode!r} changed under gate knobs {knobs!r}"
        assert p.predictor_id == base.predictor_id


def test_predictor_id_unchanged_for_existing_modes(registry):
    assert LearnedPredictor(registry=registry, signals="full").predictor_id == "learned"
    assert (LearnedPredictor(registry=registry, signals="plan_only").predictor_id
            == "learned_plan_only")
    assert (LearnedPredictor(registry=registry, signals="transition_only").predictor_id
            == "learned_transition_only")


# ---------------------------------------------------------------------------
# Gate semantics
# ---------------------------------------------------------------------------

def test_plan_gated_is_a_selectable_mode(registry):
    p = LearnedPredictor(registry=registry, signals="plan_gated")
    assert p.signal_mode == "plan_gated"
    assert p.predictor_id.startswith("learned_plan_gated[")
    # The id must name the RULE and its PARAMETERS -- a bare mode label would be
    # unreproducible.
    assert "hard" in p.predictor_id
    assert "scope=resource" in p.predictor_id
    assert "no_plan=pass" in p.predictor_id


def test_gate_spec_reports_factor_and_effective_threshold(registry):
    p = LearnedPredictor(registry=registry, signals="plan_gated",
                         gate_mode="soft", gate_factor=0.5, min_confidence=0.30)
    # soft(f) drops unsupported candidates below min_confidence / f.  Both
    # numbers jointly determine what survives, so both must be reported.
    assert "factor=0.5" in p.gate_spec
    assert "min_conf=0.3" in p.gate_spec
    assert "eff_thresh=0.6" in p.gate_spec


@pytest.mark.parametrize("bad", [
    {"signals": "nonsense"},
    {"signals": "plan_gated", "gate_mode": "nonsense"},
    {"signals": "plan_gated", "gate_scope": "nonsense"},
    {"signals": "plan_gated", "gate_no_plan": "nonsense"},
    {"signals": "plan_gated", "gate_factor": 0.0},
    {"signals": "plan_gated", "gate_factor": 1.5},
    {"signals": "plan_gated", "gate_k": -1},
])
def test_invalid_params_rejected(registry, bad):
    with pytest.raises(ValueError):
        LearnedPredictor(registry=registry, **bad)


def _names(res) -> set[str]:
    return {r.name for r in (res.resources or [])}


def test_hard_gate_output_is_a_subset_of_the_union(registry):
    """The gate can only REMOVE table candidates, never invent any."""
    pts = _prediction_points(registry)
    full = LearnedPredictor(registry=registry, signals="full", lookahead=2)
    gated = LearnedPredictor(registry=registry, signals="plan_gated",
                             gate_mode="hard", lookahead=2)
    for pt in pts:
        u = {(r.resource_id, r.consumer_step_offset)
             for r in (full.predict(**pt).resources or [])}
        g = {(r.resource_id, r.consumer_step_offset)
             for r in (gated.predict(**pt).resources or [])}
        # `full` may add MockPredictor fallback candidates that plan_gated
        # (a restricted mode) never emits, so compare only where full did not
        # fall back.
        if "mock_fallback" in full.predict(**pt).predictor_id:
            continue
        assert g <= u


def test_tail_zero_is_a_noop_gate(registry):
    """gate_tail=0.0 must not drop anything -- the control point of the sweep."""
    pts = _prediction_points(registry)
    noop = LearnedPredictor(registry=registry, signals="plan_gated",
                            gate_mode="tail", gate_tail=0.0, lookahead=2)
    passthru = LearnedPredictor(registry=registry, signals="plan_gated",
                                gate_mode="soft", gate_factor=1.0, lookahead=2)
    for pt in pts:
        assert _names(noop.predict(**pt)) == _names(passthru.predict(**pt))


def test_tail_above_one_equals_hard_gate(registry):
    """No confidence exceeds 1.0, so tail=1.01 suppresses every unsupported."""
    pts = _prediction_points(registry)
    hard = LearnedPredictor(registry=registry, signals="plan_gated",
                            gate_mode="hard", lookahead=2)
    tail = LearnedPredictor(registry=registry, signals="plan_gated",
                            gate_mode="tail", gate_tail=1.01, lookahead=2)
    for pt in pts:
        assert _names(hard.predict(**pt)) == _names(tail.predict(**pt))


def test_soft_equals_tail_at_the_effective_threshold(registry):
    """
    Offline, soft(f) and tail(min_confidence/f) select the SAME candidates --
    replay scoring ignores confidence, so soft's only observable effect is
    which candidates fall under min_confidence.  This equivalence is the reason
    the two rules are not independent evidence.
    """
    pts = _prediction_points(registry)
    for factor in (0.9, 0.75, 0.6, 0.5, 0.4):
        soft = LearnedPredictor(registry=registry, signals="plan_gated",
                                gate_mode="soft", gate_factor=factor,
                                min_confidence=0.30, lookahead=2)
        tail = LearnedPredictor(registry=registry, signals="plan_gated",
                                gate_mode="tail", gate_tail=0.30 / factor,
                                min_confidence=0.30, lookahead=2)
        for pt in pts:
            assert _names(soft.predict(**pt)) == _names(tail.predict(**pt)), factor


def test_cap_limits_prediction_volume(registry):
    pts = _prediction_points(registry)
    for k in (0, 1, 2, 3):
        cap = LearnedPredictor(registry=registry, signals="plan_gated",
                               gate_mode="cap", gate_k=k, lookahead=2)
        for pt in pts:
            assert len(cap.predict(**pt).resources or []) <= k


def test_cap_prefers_candidates_both_signals_name(registry):
    """
    At k=1 the survivor must be a resource the plan AND the table both name,
    whenever such a candidate exists.
    """
    plan = PlanContext(
        tool_sequence=["run_ase", "extract_output_json", "run_ase"], source="test")
    pt = {"step": 3,
          "recent_events": [{"event_type": "tool_call", "payload": {"tool": "run_ase"}}],
          "current_tool_calls": [{"name": "run_ase"}],
          "task_description": "",
          "plan_context": plan}
    plan_p = LearnedPredictor(registry=registry, signals="plan_only", lookahead=2)
    tab_p = LearnedPredictor(registry=registry, signals="transition_only", lookahead=2)
    agreed = ({r.resource_id for r in (plan_p.predict(**pt).resources or [])}
              & {r.resource_id for r in (tab_p.predict(**pt).resources or [])})
    if not agreed:
        pytest.skip("shipped table + fixture registry produce no agreed candidate")
    cap = LearnedPredictor(registry=registry, signals="plan_gated",
                           gate_mode="cap", gate_k=1, lookahead=2)
    got = cap.predict(**pt).resources
    assert got and got[0].resource_id in agreed


def test_no_plan_policies_differ_when_plan_is_silent(registry):
    """
    With no plan_context there is nothing to gate WITH.  "pass" must leave the
    table's candidates alone; "suppress" must emit nothing.
    """
    pt = {"step": 3,
          "recent_events": [{"event_type": "tool_call", "payload": {"tool": "run_ase"}}],
          "current_tool_calls": [{"name": "run_ase"}],
          "task_description": "",
          "plan_context": None}
    tab = LearnedPredictor(registry=registry, signals="transition_only", lookahead=2)
    expected = _names(tab.predict(**pt))
    assert expected, "fixture must produce table candidates for this test to mean anything"

    p_pass = LearnedPredictor(registry=registry, signals="plan_gated",
                              gate_mode="hard", gate_no_plan="pass", lookahead=2)
    p_supp = LearnedPredictor(registry=registry, signals="plan_gated",
                              gate_mode="hard", gate_no_plan="suppress", lookahead=2)
    assert _names(p_pass.predict(**pt)) == expected
    assert _names(p_supp.predict(**pt)) == set()


def test_gate_scope_key_is_at_least_as_strict_as_resource(registry):
    """(resource, offset) support is strictly stronger than resource support."""
    pts = _prediction_points(registry)
    res_scope = LearnedPredictor(registry=registry, signals="plan_gated",
                                 gate_mode="hard", gate_scope="resource",
                                 lookahead=2)
    key_scope = LearnedPredictor(registry=registry, signals="plan_gated",
                                 gate_mode="hard", gate_scope="key", lookahead=2)
    for pt in pts:
        a = {(r.resource_id, r.consumer_step_offset)
             for r in (res_scope.predict(**pt).resources or [])}
        b = {(r.resource_id, r.consumer_step_offset)
             for r in (key_scope.predict(**pt).resources or [])}
        assert b <= a


def test_plan_gated_does_not_use_the_mock_fallback(registry):
    """
    Restricted mode, like plan_only/transition_only: an empty gate result must
    stay empty rather than being backfilled by MockPredictor, or the arm would
    no longer isolate the gated signal.
    """
    pt = {"step": 0, "recent_events": [], "current_tool_calls": [],
          "task_description": "", "plan_context": None}
    p = LearnedPredictor(registry=registry, signals="plan_gated",
                         gate_mode="hard", gate_no_plan="suppress")
    res = p.predict(**pt)
    assert res.resources == []
    assert "mock_fallback" not in res.predictor_id


def test_gate_does_not_mutate_the_registry_templates(registry):
    """
    soft downweight rewrites confidence; it must do so on a copy, or the
    registry's shared ResourceSpec templates would drift across calls.
    """
    before = copy.deepcopy({t: registry.get(t) for t in registry.all_tools()})
    p = LearnedPredictor(registry=registry, signals="plan_gated",
                         gate_mode="soft", gate_factor=0.9, lookahead=2)
    for pt in _prediction_points(registry):
        p.predict(**pt)
    after = {t: registry.get(t) for t in registry.all_tools()}
    for tool, specs in after.items():
        for a, b in zip(specs, before[tool]):
            assert a.confidence == b.confidence
            assert a.consumer_step_offset == b.consumer_step_offset


# ---------------------------------------------------------------------------
# The replay harness must still refuse to mislabel an unsupported variant
# ---------------------------------------------------------------------------

def test_replay_harness_still_rejects_unknown_kwargs():
    import sys
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    import replay_predictor as rp

    reg = ResourceRegistry.merged(ResourceRegistry.from_json(),
                                  ResourceRegistry.from_mock_predictor())
    pred, reason = rp.build_predictor(
        "bogus", {"signals": "full", "not_a_real_knob": 1}, reg)
    assert pred is None and "not_a_real_knob" in reason

    # ...and every gate variant registered in the sweep IS constructible, so a
    # silent UNSUPPORTED row cannot hide a whole rule from the sweep.
    for name, spec in rp.VARIANTS.items():
        p, why = rp.build_predictor(name, spec, reg)
        assert p is not None, f"{name}: {why}"


def test_cap_control_ignores_the_plan(registry):
    """
    gate_cap_use_plan=False must make the cap's ranking independent of the
    plan.  It is the control that decides whether any cap improvement is a
    plan-gating effect or just a volume effect.
    """
    plan = PlanContext(
        tool_sequence=["run_ase", "extract_output_json", "run_ase"], source="test")
    base = {"step": 3,
            "recent_events": [{"event_type": "tool_call", "payload": {"tool": "run_ase"}}],
            "current_tool_calls": [{"name": "run_ase"}],
            "task_description": ""}
    ctl = LearnedPredictor(registry=registry, signals="plan_gated",
                           gate_mode="cap", gate_k=1, gate_cap_use_plan=False,
                           lookahead=2)
    # Changing the plan may change WHICH candidates exist, so compare the
    # ranking on a fixed candidate set instead: same candidates, plan present
    # vs a different plan -> the control must not reorder on plan agreement.
    tab = LearnedPredictor(registry=registry, signals="transition_only", lookahead=2)
    cands = tab.predict(**base, plan_context=None).resources or []
    if len(cands) < 2:
        pytest.skip("need >= 2 table candidates to observe a ranking")
    ranked = ctl._cap_candidates(list(cands), list(cands[:1]), list(cands))
    by_conf = sorted(cands, key=lambda r: -r.confidence)[:1]
    assert [r.resource_id for r in ranked] == [r.resource_id for r in by_conf]
    assert "use_plan=False" in ctl.gate_spec
