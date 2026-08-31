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

Everything here runs offline against a FROZEN transition table
(runtime/tests/fixtures/learned_transitions_20260707.json) and small synthetic
registries; no GPU and no workflow run is required.

Why a frozen fixture rather than the shipped table
--------------------------------------------------
These tests assert predictor LOGIC — union, dedup, provenance tagging, the
lookahead horizon, the decay derivation.  Every one of those assertions is
parameterised by table content, so pinning them to
`runtime/predictor/data/learned_transitions.json` coupled the logic suite to a
data artifact that is *supposed* to be regenerated as traces accumulate.  When
the table was regenerated on 2026-08-30 (165 -> 490 traces) six tests here went
red without a single line of predictor code changing, and the only way to make
them green was to not regenerate — exactly backwards.

So: logic is tested against the frozen 2026-07-07 fixture, which never moves.
The SHIPPED table is still tested, in `TestShippedTableIsWellFormed`, but by
PROPERTY rather than by memorised constant.  That is what keeps the
anti-fabrication guarantee alive: a made-up entry (the "p = 0.964, n = 111"
that was once recorded for run_ase +1 -> extract_output_json and never existed
in any table) violates the internal-consistency property no matter what the
real numbers are, whereas an assertion that the probability equals 0.4045
stops protecting anything the moment 0.4045 legitimately becomes 0.4948.
"""
import json
import math
import statistics
from datetime import datetime
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
from runtime.predictor.transition_learner import (
    LEGACY_OFFSET_BASIS,
    OFFSET_BASIS,
    TABLE_VERSION,
    TransitionTable,
)

# Frozen copy of the 2026-07-07 table (n_traces=165).  Byte-identical to what
# runtime/predictor/data/learned_transitions.json held before the 2026-08-30
# regeneration.  NEVER regenerate this file: its whole purpose is to be the
# unmoving input that makes the logic assertions below deterministic.
FIXTURE_TABLE_PATH = (Path(__file__).resolve().parent / "fixtures"
                      / "learned_transitions_20260707.json")

# The live artifact.  Only TestShippedTableIsWellFormed reads this, and only
# for properties that must hold for ANY regeneration.
SHIPPED_TABLE_PATH = (Path(__file__).resolve().parents[1] / "predictor" / "data"
                      / "learned_transitions.json")


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
    return LearnedPredictor(transitions_path=FIXTURE_TABLE_PATH, registry=registry,
                            signals=signals, lookahead=lookahead)


# A plan whose next entry after run_ase is a tool the registry covers, so the
# plan signal DOES produce candidates.  This is the configuration in which the
# old `and not resources` guard silently suppressed the transition signal.
PLAN_AFTER_RUN_ASE = PlanContext(tool_sequence=["run_ase", "code_task"])


# ---------------------------------------------------------------------------
# The canonical missed transition
# ---------------------------------------------------------------------------

class TestCanonicalGap:
    def test_fixture_table_probability_is_read_from_the_json_not_assumed(self):
        """Pinned against the FROZEN fixture, where a constant cannot drift.

        The anti-fabrication duty this test used to carry now lives on the
        SHIPPED table as a property check — see
        TestShippedTableIsWellFormed::
        test_shipped_table_probability_is_read_from_the_json_not_assumed.
        """
        raw = json.loads(FIXTURE_TABLE_PATH.read_text())
        entries = raw["tool_transitions"]["run_ase"]["1"]
        entry = next(e for e in entries if e["target"] == "extract_output_json")
        assert entry["probability"] == pytest.approx(0.4045)
        assert entry["count"] == 36
        # It clears the predictor's default gate, so the only thing that could
        # have suppressed it was the signal guard.
        assert entry["probability"] >= LearnedPredictor(
            transitions_path=FIXTURE_TABLE_PATH)._min_confidence

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
# The SHIPPED table: properties that must hold for ANY regeneration
# ---------------------------------------------------------------------------

def _buckets(raw: dict):
    """Yield (kind, source, offset, entries) for every bucket in a table."""
    for kind in ("tool_transitions", "model_transitions"):
        for source, offset_map in raw.get(kind, {}).items():
            for offset, entries in offset_map.items():
                yield kind, source, int(offset), entries


def _consistent_denominators(entries: list[dict]) -> list[int]:
    """Integer denominators D consistent with EVERY (count, probability) here.

    transition_learner._build normalises within one (source, offset) bucket:
    `prob = cnt / denom[(src, offset)]`, rounded to 4 dp, with the SAME denom
    for every entry.  So a well-formed bucket admits at least one integer D
    with `round(count / D, 4) == probability` for all of its entries.  D can
    exceed the surviving counts, because entries below --min-count are dropped
    after the denominator is fixed.

    This is the anti-fabrication property.  A hand-written entry cannot satisfy
    it unless its author solved for the bucket's denominator, and it does not
    care what the real probabilities are — which is exactly why it survives
    regeneration where a pinned constant does not.
    """
    counts = sum(e["count"] for e in entries)
    probs = sum(e["probability"] for e in entries)
    if probs <= 0:
        return []
    est = counts / probs                     # exact in real arithmetic
    lo = max(counts, int(math.floor(est * 0.95)) - 2)
    hi = int(math.ceil(est * 1.05)) + 2
    return [D for D in range(lo, hi + 1)
            if all(round(e["count"] / D, 4) == e["probability"] for e in entries)]


class TestShippedTableIsWellFormed:
    """Property checks on runtime/predictor/data/learned_transitions.json.

    Deliberately asserts NO specific probability.  The table is meant to be
    regenerated as traces accumulate (165 traces on 2026-07-07 -> 490 on
    2026-08-30), and a suite that goes red on regeneration is a suite that
    argues against regenerating.  What must never change is that every number
    in the file was *computed from traces*, and these properties are what a
    fabricated entry fails.

    Regenerate with:
        python3 -m runtime.predictor.transition_learner \
            logs/workflow_traces/*.jsonl \
            --output runtime/predictor/data/learned_transitions.json
    """

    @pytest.fixture(scope="class")
    def raw(self) -> dict:
        return json.loads(SHIPPED_TABLE_PATH.read_text())

    def test_provenance_header_is_present(self, raw):
        assert raw["version"]
        assert isinstance(raw["n_traces"], int) and raw["n_traces"] > 0
        assert isinstance(raw["n_tool_events"], int) and raw["n_tool_events"] > 0
        # parses as ISO-8601; a hand-edited table tends to lose this
        datetime.fromisoformat(raw["generated_at"])

    def test_the_offset_basis_is_declared(self, raw):
        """An `offset` is meaningless until you know what it counts.

        Before 2026-08-30 learn_from_traces built ONE list holding both
        tool_call and llm_call events and used that list's index delta, so a
        tool offset counted EVENTS.  ChemGraph traces have no llm_call and
        were unaffected; AtomAgents traces interleave one per tool and were
        wrecked — `plan_task +2 -> plan_task p=1.0 n=45` was the whole
        plan_task row, while the real successor `code_task` (78 of 143) sat
        outside max_offset entirely.

        The basis is now written into the file so the two can never be
        confused and a pre-fix table is detectable rather than silently
        misread.
        """
        basis = raw.get("offset_basis")
        assert basis, (
            "no offset_basis in the shipped table — it predates the "
            "2026-08-30 fix and its tool offsets count raw events. "
            "Regenerate: python3 -m runtime.predictor.transition_learner "
            "logs/workflow_traces/*.jsonl "
            "--output runtime/predictor/data/learned_transitions.json")
        assert basis == OFFSET_BASIS
        assert raw["version"] == TABLE_VERSION

    def test_a_pre_fix_table_is_detectable_as_legacy(self):
        """The frozen 2026-07-07 fixture is such a table.  Loading it must
        report the legacy basis, never silently inherit the current one."""
        legacy = json.loads(FIXTURE_TABLE_PATH.read_text())
        assert "offset_basis" not in legacy
        assert legacy.get("version") == "1"
        loaded = TransitionTable.load(FIXTURE_TABLE_PATH)
        assert loaded.offset_basis == LEGACY_OFFSET_BASIS
        assert TransitionTable.load(SHIPPED_TABLE_PATH).offset_basis == OFFSET_BASIS

    def test_probabilities_are_in_range_and_sorted(self, raw):
        seen = 0
        for kind, source, offset, entries in _buckets(raw):
            assert entries, f"{kind}/{source}/+{offset} is empty"
            probs = [e["probability"] for e in entries]
            for e in entries:
                assert 0.0 < e["probability"] <= 1.0, (kind, source, offset, e)
                assert isinstance(e["count"], int) and e["count"] >= 1
                assert e["offset"] == offset
                assert e["target"]
            assert probs == sorted(probs, reverse=True), (
                f"{kind}/{source}/+{offset} is not sorted by probability")
            targets = [e["target"] for e in entries]
            assert len(set(targets)) == len(targets), (
                f"{kind}/{source}/+{offset} repeats a target")
            seen += 1
        assert seen > 0

    def test_each_offset_distribution_sums_to_at_most_one(self, raw):
        """<= 1, not == 1: --min-count drops entries AFTER normalisation."""
        for kind, source, offset, entries in _buckets(raw):
            total = sum(e["probability"] for e in entries)
            assert total <= 1.0 + 1e-3, (kind, source, offset, total)

    def test_shipped_table_probability_is_read_from_the_json_not_assumed(self, raw):
        """Anti-fabrication guard — the reason this test exists.

        A "p = 0.964, n = 111" was once recorded for run_ase +1 ->
        extract_output_json and existed in no table, shipped or regenerated.
        Pinning the true value could not catch that, because the true value is
        allowed to move (0.4045 at 165 traces, 0.4948 at 490).  What cannot
        move is that every entry in a bucket shares one integer denominator.
        The fabricated pair implies a denominator of ~115 among siblings that
        imply 289, and it pushes the bucket's probabilities to a sum of 1.47.
        """
        for kind, source, offset, entries in _buckets(raw):
            assert _consistent_denominators(entries), (
                f"{kind}/{source}/+{offset}: no single integer denominator "
                f"explains {[(e['target'], e['count'], e['probability']) for e in entries]}"
                " — an entry here was not computed by transition_learner")

    def test_the_fabricated_entry_would_be_caught(self, raw):
        """Executable proof that the guard above has teeth."""
        entries = [dict(e) for e in raw["tool_transitions"]["run_ase"]["1"]]
        assert _consistent_denominators(entries)          # the real bucket is fine
        for e in entries:
            if e["target"] == "extract_output_json":
                e["probability"], e["count"] = 0.964, 111
        assert not _consistent_denominators(entries)
        assert sum(e["probability"] for e in entries) > 1.0 + 1e-3

    def test_the_frozen_fixture_also_satisfies_every_property(self):
        """The properties describe transition_learner output, not one table."""
        raw = json.loads(FIXTURE_TABLE_PATH.read_text())
        for kind, source, offset, entries in _buckets(raw):
            assert _consistent_denominators(entries), (kind, source, offset)
            assert sum(e["probability"] for e in entries) <= 1.0 + 1e-3


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
        raw = json.loads(FIXTURE_TABLE_PATH.read_text())["tool_transitions"]
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
        pred = LearnedPredictor(transitions_path=FIXTURE_TABLE_PATH)
        assert pred.offset_decay == pytest.approx(expected)
        assert "n=9" in pred.offset_decay_provenance

    def test_decay_is_a_damping_factor(self):
        pred = LearnedPredictor(transitions_path=FIXTURE_TABLE_PATH)
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

    def test_characterisation_shipped_table_gates_out_the_run_ase_self_loop(self, registry):
        """CHARACTERISATION — asserts a KNOWN REGRESSION, not a desired state.

        Against the frozen 165-trace fixture, run_ase's transition signal emits
        mace_mp:medium at offsets 1 and 2 (p 0.5056 / 0.7073 — see
        test_no_damping_inside_the_legacy_horizon).  Against the regenerated
        490-trace table it emits NEITHER, because the corpus tripled and the
        run_ase self-loop diffused:

            run_ase +1 -> run_ase   0.5056 (n=45)  ->  0.1730 (n=50)
            run_ase +2 -> run_ase   0.7073 (n=29)  ->  0.2197 (n=29)

        Both now sit under `_min_confidence` (0.3).  The counts barely moved;
        the DENOMINATOR grew (89 -> 289 at offset 1), so this is dilution, not
        a workflow that changed.  Offset 3 survives at 0.8957 but is outside
        the default lookahead of 2.

        This is the fixed-threshold problem in miniature: 0.3 was calibrated on
        a small homogeneous corpus, and a broader corpus walks candidates under
        it without anything getting worse in the world.  Argument for the
        value-density arbitrator, not a tuning knob to turn.

        Reproduce the before-state:
            cp runtime/predictor/data/_preA3_20260830/learned_transitions.json \
               runtime/predictor/data/learned_transitions.json
        (also frozen at runtime/tests/fixtures/learned_transitions_20260707.json)

        NOT affected by the 2026-08-30 offset-basis fix: run_ase occurs only
        in ChemGraph traces, which carry no llm_call events, so the
        mixed-event offset and the tool-only offset coincide there and every
        run_ase bucket is byte-identical across that change.  The AtomAgents
        rows moved a great deal — see test_the_offset_basis_is_declared.

        If this test goes red, the shipped table's run_ase distribution moved
        again — update the numbers here deliberately, do not delete the test.
        """
        shipped = LearnedPredictor(transitions_path=SHIPPED_TABLE_PATH,
                                   registry=registry, signals="transition_only",
                                   lookahead=3)
        by_offset = {r.consumer_step_offset: r.confidence
                     for r in _predict(shipped, "run_ase", None).resources
                     if r.name == "mace_mp:medium"}
        assert 1 not in by_offset, "offset-1 self-loop is back above the gate"
        assert 2 not in by_offset, "offset-2 self-loop is back above the gate"
        assert by_offset[3] == pytest.approx(0.8957, abs=1e-4)

        raw = json.loads(SHIPPED_TABLE_PATH.read_text())["tool_transitions"]
        p1 = next(e["probability"] for e in raw["run_ase"]["1"]
                  if e["target"] == "run_ase")
        assert p1 < shipped._min_confidence

        # ...and the signal itself is alive: at the same step it still names a
        # registry-covered resource for the NEW top successor.
        names = {r.name for r in _predict(shipped, "run_ase", None).resources}
        assert "ase_output_json" in names

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


def _n_strictly_larger(registry, table_path=None) -> int:
    """Steps at which `full` emits a resource `plan_only` does not.

    Asserts the hard invariant on the way past: `full` may never LOSE a
    resource that `plan_only` emitted.  That part is not a characterisation —
    it is the union's contract and must never be relaxed.
    """
    kw = {"registry": registry}
    if table_path is not None:
        kw["transitions_path"] = table_path
    n = 0
    for trace in _recorded_traces():
        full = _replay(trace, LearnedPredictor(signals="full", **kw))
        plan = _replay(trace, LearnedPredictor(signals="plan_only", **kw))
        assert len(full) == len(plan)
        for f, p in zip(full, plan):
            assert p["keys"] <= f["keys"], f"{trace}: step {p['step']} lost a resource"
            if p["keys"] < f["keys"]:
                n += 1
    return n


@pytest.mark.skipif(not _recorded_traces(), reason="no recorded eval traces on disk")
def test_full_is_superset_of_plan_only_on_recorded_traces():
    """Definition-of-done #1 on real traces rather than synthetic inputs.

    CHARACTERISATION as of 2026-08-30.  This test used to assert
    `n_strictly_larger > 0` — that the union genuinely adds something on real
    traces.  Against the shipped 490-trace table with the SHIPPED registry it
    is now 0, and that number is asserted here rather than relaxed away.

    The cause is NOT that complementarity got weaker.  It is a registry data
    gap meeting a corpus shift, and the two halves are separable:

      * all 29 previously-contributing steps were `run_ase` steps where the
        plan was silent, and the table carried the prediction alone
        (predictor_id `learned+transition_only`);
      * the shipped table's top run_ase successor moved from `run_ase`
        (self-loop, registry-covered) to `extract_output_json`, for which the
        shipped registry holds NO resource — see
        TestCanonicalGap::test_shipped_registry_cannot_emit_extract_output_json;
      * the residual self-loop fell under the confidence gate — see
        TestOffsetDecay::
        test_characterisation_shipped_table_gates_out_the_run_ase_self_loop.

    So the predictor names the right tool and the registry cannot turn it into
    anything prefetchable.  The companion assertion below proves the union
    mechanism is intact: give the registry one `extract_output_json` entry and
    the SAME shipped table yields 58 contributing steps — twice the 29 the
    165-trace table managed on the un-patched registry.

    Fixing this for real means adding an entry to
    runtime/predictor/data/tool_resources.json, which changes what the runtime
    prefetches at run time.  That is a product decision, not a test fix, and it
    is why this test characterises rather than relaxes.

    Reproduce the before-state:
        cp runtime/predictor/data/_preA3_20260830/learned_transitions.json \
           runtime/predictor/data/learned_transitions.json
    """
    shipped = ResourceRegistry.merged(ResourceRegistry.from_json(),
                                      ResourceRegistry.from_mock_predictor())
    assert _n_strictly_larger(shipped) == 0, (
        "the shipped registry now contributes transition-only resources again "
        "— the registry gap may have been closed; update this characterisation")

    # Control: the union itself works.  Same table, same traces, one extra
    # registry entry for the successor the table actually predicts.
    patched = ResourceRegistry.merged(ResourceRegistry.from_json(),
                                      ResourceRegistry.from_mock_predictor())
    patched.register("extract_output_json",
                     _spec("ase_output_json", "extract_output_json"))
    assert _n_strictly_larger(patched) == 58

    # ...and the frozen 165-trace table, on that same patched registry, is not
    # better than the regenerated one — the corpus growth did not cost us
    # complementarity, the registry gap did.
    assert _n_strictly_larger(patched, FIXTURE_TABLE_PATH) == 58
    assert _n_strictly_larger(shipped, FIXTURE_TABLE_PATH) == 29
