"""test_residency_horizon.py — T3, the horizon estimator.

WHAT MUST NOT REGRESS, and why each one is here:

  * I3, IN BOTH DIRECTIONS. `next_use_s` returns None for "not within the
    lookahead L" and NEVER inf, NEVER a negative, NEVER a sentinel that means
    "never again". A resource that genuinely never recurs in a recording is
    indistinguishable, to this estimator, from one whose next use is far away —
    and it must stay indistinguishable, because excluding predicted-dead
    resources from the pool cost ~9 points in simulation and produced a finding
    that had to be retracted. Every test that calls `next_use_s` pushes the
    answer back through `check_horizon`.
  * L IS NOT D. `horizon_s` is the lookahead. Eq. 1's decay scale belongs to
    the arbitrator. There is a source-level test that this module never names a
    decay parameter, never imports one, and never imports the value function —
    because the way this defect comes back is somebody "tidying up" by passing
    `horizon_s` where `decay_s` was wanted, and the two are within a factor of
    30 of each other so the result would look plausible.
  * NO CONFIDENCE THRESHOLD. §1.4: a fixed 0.85 gate calibrated on 165
    homogeneous traces admits nothing on 490 diverse ones. Confidence must
    enter as a continuous shift in the reported distance, with no cliff. There
    is a test that sweeps probability and asserts the distance moves smoothly
    and monotonically, and a source-level test that no threshold constant
    exists to be tuned.
  * THE TWO SIGNALS ARE SIMULTANEOUS, NOT A FALLBACK CHAIN. A test asserts that
    adding the second signal changes the answer while the first is still
    speaking — i.e. that there is no "if the plan answered, ignore the table"
    branch.

No GPU, no vLLM, no LAMMPS, no SLURM. The recorded-trace tests skip cleanly
when the (gitignored) recordings are not present.
"""
from __future__ import annotations

import io
import json
import math
import tokenize
from pathlib import Path

import pytest

from runtime.residency import horizon as hz_mod
from runtime.residency.contract import (
    HorizonEstimator,
    ResourceClass,
    ResourceSpec,
    Rung,
    check_horizon,
    value,
)
from runtime.residency.horizon import (
    DEFAULT_LOOKAHEAD_S,
    EXPECTED_MODEL_BASIS,
    EXPECTED_TOOL_BASIS,
    MEASURED_LLM_STEP_S,
    MEASURED_TOOL_STEP_S,
    Arrival,
    DemandMap,
    PlanTransitionHorizon,
    ReplayHorizon,
    PHASE_TO_TOOL,
    TransitionBasisMismatch,
    TransitionSignal,
    _read_tool_calls,
    adjudicate_needs,
    read_tool_executions,
    certainty_equivalent_distance,
    mean_tool_step_s,
    reuse_distances,
)

REPO = Path(__file__).resolve().parents[2]
EVAL_RUNS = REPO / "results" / "eval_q1_q4" / "runs"

L = 600.0
STEP = 60.0


# --------------------------------------------------------------------------
# Fixtures — a tiny two-resource world, no I/O
# --------------------------------------------------------------------------


DEMAND = DemandMap(
    by_tool={
        "plan_task": frozenset({"qwen_32b"}),
        "code_task": frozenset({"qwen_72b_text"}),
        "computation_task_screw_dislocation": frozenset(
            {"qwen_72b", "w_eam4_big_activated"}),
    },
).with_models({"Qwen/Qwen2.5-VL-72B-Instruct": ["qwen_72b"]})


class FakePlan:
    """Duck-types `plan_extractor.PlanContext.tool_at_offset`."""

    def __init__(self, seq):
        self.tool_sequence = list(seq)

    def tool_at_offset(self, current_index: int, offset: int):
        t = current_index + offset
        if 0 <= t < len(self.tool_sequence):
            return self.tool_sequence[t]
        return None


def a_horizon(**kw) -> PlanTransitionHorizon:
    kw.setdefault("horizon_s", L)
    h = PlanTransitionHorizon(DEMAND, STEP, **kw)
    return h


UNIREF90 = ResourceSpec(
    resource_id="uniref90", resource_class=ResourceClass.DATA_PATTERN_A,
    held_rung=Rung.R3_ACTIVATED, held_gb=117.20, cold_s=372.6 + 12.0,
    ready_s=12.0,
)


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


def test_both_estimators_satisfy_the_protocol():
    assert isinstance(ReplayHorizon({}, horizon_s=L), HorizonEstimator)
    assert isinstance(a_horizon(), HorizonEstimator)


def test_horizon_s_is_the_lookahead_and_is_positive():
    assert a_horizon(horizon_s=1234.0).horizon_s == 1234.0
    assert ReplayHorizon({}, horizon_s=1234.0).horizon_s == 1234.0
    for ctor in (lambda: PlanTransitionHorizon(DEMAND, STEP, horizon_s=0.0),
                 lambda: ReplayHorizon({}, horizon_s=-1.0)):
        with pytest.raises(ValueError):
            ctor()


def test_tool_step_s_has_no_default():
    """A default would be wrong by 220x between the measured facets."""
    with pytest.raises(TypeError):
        PlanTransitionHorizon(DEMAND)          # type: ignore[call-arg]


# --------------------------------------------------------------------------
# I3 — None means "not within L", never "never again"
# --------------------------------------------------------------------------


def test_beyond_the_lookahead_returns_None_not_inf():
    """The headline invariant. A far-away use is None, and None is not a float
    the caller could mistake for a distance."""
    h = ReplayHorizon({"uniref90": [10_000.0]}, horizon_s=L)
    dt = h.next_use_s("uniref90", 0.0)
    assert dt is None
    assert not isinstance(dt, float)
    # and the same resource inside L is a real number
    assert h.next_use_s("uniref90", 9_500.0) == pytest.approx(500.0)


def test_a_resource_that_never_recurs_yields_None_and_no_dead_marker():
    """A single use, then nothing, ever. The estimator is NOT entitled to say
    'dead' — it must return the same None it returns for 'far away'."""
    h = ReplayHorizon({"uniref90": [5.0]}, horizon_s=L)
    after_its_only_use = h.next_use_s("uniref90", 50.0)
    never_seen_at_all = h.next_use_s("no_such_resource", 50.0)
    far_away = ReplayHorizon(
        {"uniref90": [1e6]}, horizon_s=L).next_use_s("uniref90", 0.0)

    assert after_its_only_use is None
    assert never_seen_at_all is None
    assert far_away is None
    # The three cases must be INDISTINGUISHABLE. If a future change makes
    # "exhausted" reportable, the arbitrator can act on "dead" again, and that
    # is exactly the 9-point regression I3 exists to prevent.
    assert after_its_only_use is never_seen_at_all is far_away


def test_no_return_path_can_produce_inf_or_negative():
    """Every answer from every estimator, over a grid of times, survives
    check_horizon — which is what the arbitrator applies at `_dt`."""
    replay = ReplayHorizon(
        {"qwen_32b": [0.0, 30.0, 900.0], "qwen_72b": [5.0], "w_eam4_big_activated": []},
        horizon_s=L)
    plan = a_horizon()
    plan.set_plan(FakePlan(["plan_task", "code_task",
                            "computation_task_screw_dislocation"]), index=0)
    plan.observe(0.0, tool="plan_task", model="Qwen/Qwen2.5-VL-72B-Instruct")

    for est in (replay, plan):
        for rid in ("qwen_32b", "qwen_72b", "qwen_72b_text",
                    "w_eam4_big_activated", "not_a_resource"):
            for now in (0.0, 0.5, 29.9, 30.0, 120.0, 599.0, 600.0, 5000.0):
                dt = est.next_use_s(rid, now)
                assert check_horizon(dt) is dt      # raises on inf/nan/negative
                if dt is not None:
                    assert 0.0 <= dt <= est.horizon_s
                    assert math.isfinite(dt)


def test_an_infinite_use_time_is_refused_at_construction():
    """inf smuggled in as data is 'never again' by another name."""
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ReplayHorizon({"uniref90": [bad]}, horizon_s=L)


def test_arrival_refuses_inf_and_out_of_range_probability():
    for bad in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError):
            Arrival(bad, 0.5, "test")
    for p in (-0.01, 1.01):
        with pytest.raises(ValueError):
            Arrival(10.0, p, "test")


def test_the_lookahead_boundary_is_value_neutral():
    """A certain need exactly at L collapses to None. That is not a loss: Eq. 1
    prices dt=None at L and dt=L at L, so the two are the same number. This
    pins the claim in the module docstring."""
    at_L = certainty_equivalent_distance(
        [Arrival(L, 1.0, "test")], lookahead_s=L, resolution_s=1.0)
    assert at_L is None
    D = 60.0
    assert value(UNIREF90, None, D, L) == pytest.approx(value(UNIREF90, L, D, L))


# --------------------------------------------------------------------------
# The certainty-equivalent distance
# --------------------------------------------------------------------------


def test_no_signal_at_all_is_None_and_falls_out_of_the_formula():
    assert certainty_equivalent_distance([], lookahead_s=L, resolution_s=1.0) is None
    assert certainty_equivalent_distance(
        [Arrival(10.0, 0.0, "test")], lookahead_s=L, resolution_s=1.0) is None


def test_a_certain_near_arrival_is_reported_at_its_own_distance():
    dt = certainty_equivalent_distance(
        [Arrival(42.0, 1.0, "test")], lookahead_s=L, resolution_s=1.0)
    assert dt == pytest.approx(42.0)


def test_confidence_shifts_the_distance_smoothly_with_no_threshold():
    """The §1.4 lesson, as a test. Sweeping probability must move the reported
    distance CONTINUOUSLY and MONOTONICALLY — no probability at which the answer
    jumps from 'nothing' to 'certain', because that jump is what a fixed gate
    is and what stopped admitting anything as the corpus grew."""
    def sweep(n):
        return [certainty_equivalent_distance(
            [Arrival(30.0, i / n, "test")], lookahead_s=L, resolution_s=1.0)
            for i in range(1, n + 1)]

    coarse, fine = sweep(100), sweep(1000)
    assert all(d is not None for d in coarse)
    # monotone: more confident -> nearer
    assert all(b <= a + 1e-9 for a, b in zip(coarse, coarse[1:]))
    # continuous: refining the grid tenfold shrinks the largest step roughly
    # tenfold. A THRESHOLD would keep one fixed jump at every resolution, which
    # is exactly the shape this test exists to exclude.
    big = max(abs(a - b) for a, b in zip(coarse, coarse[1:]))
    small = max(abs(a - b) for a, b in zip(fine, fine[1:]))
    assert small < big / 5.0, (
        f"the confidence response does not refine away: {big} -> {small}; "
        "that is the signature of a gate")
    # the endpoints are what they should be
    assert coarse[-1] == pytest.approx(30.0)           # p=1 -> its own distance
    assert coarse[0] > 300.0                           # p=0.01 -> discounted


def test_a_low_probability_near_need_is_not_discarded():
    """1% at 30 s is worth something, and 'something' is a distance inside L,
    not a None. A gate at any threshold above 0.01 would throw this away."""
    dt = certainty_equivalent_distance(
        [Arrival(30.0, 0.01, "test")], lookahead_s=L, resolution_s=1.0)
    assert dt is not None and dt < L


def test_resolution_floors_the_divergence_and_is_not_a_decay_scale():
    """1/t diverges as t -> 0, which is the same divergence that makes the
    shared prefetch currency broken (I5). The floor is the step duration."""
    lots = certainty_equivalent_distance(
        [Arrival(0.0, 0.02, "test")], lookahead_s=L, resolution_s=60.0)
    assert lots is not None
    # floored at 60 s, 2% of the rate: 0.02/60 + 0.98/600 -> ~1/0.0019667
    assert lots == pytest.approx(1.0 / (0.02 / 60.0 + 0.98 / 600.0))
    # A smaller resolution reports a nearer need. It changes RESOLUTION, and
    # nothing in it is a decay: it never multiplies a benefit.
    finer = certainty_equivalent_distance(
        [Arrival(0.0, 0.02, "test")], lookahead_s=L, resolution_s=1.0)
    assert finer < lots


def test_two_signals_agreeing_do_not_exceed_certainty():
    """First-need chaining, not summation: two 0.8 claims about the same need
    must not add to 1.6 and report an impossibly near distance."""
    one = certainty_equivalent_distance(
        [Arrival(60.0, 0.8, "a")], lookahead_s=L, resolution_s=1.0)
    two = certainty_equivalent_distance(
        [Arrival(60.0, 0.8, "a"), Arrival(60.0, 0.8, "b")],
        lookahead_s=L, resolution_s=1.0)
    assert two <= one
    assert two >= 60.0 - 1e-9      # can never be nearer than the nearest claim


def test_mass_beyond_the_lookahead_is_discounted_not_discarded():
    """Arrivals past L are dropped from the sum, and their mass is priced at L
    — the most discounted thing the estimator may say, and not zero."""
    near_only = certainty_equivalent_distance(
        [Arrival(100.0, 0.3, "a")], lookahead_s=L, resolution_s=1.0)
    with_far = certainty_equivalent_distance(
        [Arrival(100.0, 0.3, "a"), Arrival(5000.0, 0.7, "b")],
        lookahead_s=L, resolution_s=1.0)
    assert with_far == pytest.approx(near_only)


# --------------------------------------------------------------------------
# ReplayHorizon
# --------------------------------------------------------------------------


def test_distances_are_monotone_along_a_replayed_trace():
    """As the clock advances toward a known use, the reported distance must
    fall by exactly the elapsed time, and must never increase between two uses.
    This is the property a stub can get wrong and a replay cannot."""
    uses = [100.0, 250.0, 251.0, 900.0]
    h = ReplayHorizon({"uniref90": uses}, horizon_s=L)
    prev_dt = None
    prev_target = None
    for i in range(0, 1000):
        now = float(i)
        dt = h.next_use_s("uniref90", now)
        if dt is None:
            prev_dt, prev_target = None, None
            continue
        assert dt >= 0.0
        # the absolute arrival time implied is always one of the recorded uses
        target = now + dt
        assert any(abs(target - u) < 1e-6 for u in uses)
        if prev_target is not None and abs(target - prev_target) < 1e-6:
            # same target still ahead: the distance must have fallen by exactly
            # the second that elapsed. Monotone, and by the right amount.
            assert dt == pytest.approx(prev_dt - 1.0)
        prev_dt, prev_target = dt, target
    # a use being consumed is the ONLY thing allowed to move the distance back
    # out, and it moves it out to the next recorded use, never to a sentinel.
    assert h.next_use_s("uniref90", 100.0) == pytest.approx(0.0)
    assert h.next_use_s("uniref90", 101.0) == pytest.approx(149.0)


def test_replay_from_next_use_is_a_drop_in_for_a_stub():
    """A1's tests take {resource_id: dt}. Same shape, but it cannot store inf."""
    h = ReplayHorizon.from_next_use(
        {"uniref90": 30.0, "uniref50": 200.0}, horizon_s=L)
    assert h.next_use_s("uniref90", 0.0) == pytest.approx(30.0)
    assert h.next_use_s("uniref50", 0.0) == pytest.approx(200.0)
    assert h.next_use_s("eam_potential", 0.0) is None
    with pytest.raises(ValueError):
        ReplayHorizon.from_next_use({"uniref90": float("inf")}, horizon_s=L)


def test_replay_is_deterministic():
    a = ReplayHorizon({"r": [1.0, 2.0, 900.0]}, horizon_s=L)
    b = ReplayHorizon({"r": [900.0, 2.0, 1.0]}, horizon_s=L)   # unsorted input
    for now in (0.0, 1.5, 3.0, 400.0, 899.0):
        assert a.next_use_s("r", now) == b.next_use_s("r", now)


# --------------------------------------------------------------------------
# PlanTransitionHorizon — the two signals
# --------------------------------------------------------------------------


def test_the_plan_signal_reports_a_distance_scaled_by_the_step_duration():
    h = a_horizon(plan_confidence=1.0)
    h.set_plan(FakePlan(["plan_task", "code_task",
                         "computation_task_screw_dislocation"]), index=0)
    h.observe(0.0, tool="plan_task")
    # cursor now sits after plan_task; code_task is one step out, the
    # computation two.
    assert h.next_use_s("qwen_72b_text", 0.0) == pytest.approx(STEP)
    assert h.next_use_s("qwen_72b", 0.0) == pytest.approx(2 * STEP)
    # and the resource nothing ahead needs is beyond the lookahead
    assert h.next_use_s("qwen_32b", 0.0) is None


def test_it_works_against_the_real_PlanContext():
    """Not just the FakePlan above: the actual `plan_extractor.PlanContext`,
    which already supports arbitrary offsets, driven from real plan text."""
    from runtime.predictor.plan_extractor import extract_plan

    ctx = extract_plan(
        "1. plan_task to lay out the study\n"
        "2. then computation_task_screw_dislocation to run it\n"
        "3. finally analyze_screw_core on the output\n")
    assert ctx is not None
    assert ctx.tool_sequence[:2] == [
        "plan_task", "computation_task_screw_dislocation"]
    h = a_horizon(plan_confidence=1.0)
    h.set_plan(ctx, index=0)
    h.observe(0.0, tool="plan_task")
    assert h.next_use_s("qwen_72b", 0.0) == pytest.approx(STEP)
    assert h.next_use_s("w_eam4_big_activated", 0.0) == pytest.approx(STEP)
    # before anything is observed the cursor sits at the head of the plan
    fresh = a_horizon(plan_confidence=1.0)
    fresh.set_plan(ctx, index=0)
    assert fresh.next_use_s("qwen_32b", 0.0) == pytest.approx(STEP)


def test_the_plan_signal_is_structurally_blind_to_code_task():
    """A defect in an upstream file, pinned here rather than worked around.

    `code_task` is the ONLY tool that needs a registered resource and is absent
    from `plan_extractor.KNOWN_TOOLS`. `extract_plan` matches on that frozenset,
    so `code_task` can never appear in a PlanContext, so the plan signal can
    never predict a need for `qwen_72b_text` — 276.3 GB, 450 s to load, 2.78
    s/GB, the second-largest resource in the catalogue and the bottom of the
    model band. It is a live tool: 138 tool->tool pairs in the 490-trace corpus
    have it as source.

    Adding it is a one-line change to `runtime/predictor/plan_extractor.py`,
    which T3 does not own; this test fails the moment somebody makes it, which
    is the point.
    """
    from runtime.predictor.plan_extractor import KNOWN_TOOLS, extract_plan

    tr = json.loads(
        (REPO / "runtime" / "predictor" / "data" / "tool_resources.json").read_text())
    needs_a_resource = {e["consumer_tool"] for e in tr
                        if isinstance(e, dict) and e.get("consumer_tool")}
    missing = needs_a_resource - set(KNOWN_TOOLS)
    assert missing == {"code_task"}, (
        f"the set of resource-needing tools invisible to the plan changed: {missing}")

    ctx = extract_plan("1. plan_task\n2. code_task\n"
                       "3. computation_task_screw_dislocation\n")
    assert ctx is not None and "code_task" not in ctx.tool_sequence
    h = a_horizon(plan_confidence=1.0)
    h.set_plan(ctx, index=0)
    h.observe(0.0, tool="plan_task")
    assert h.next_use_s("qwen_72b_text", 0.0) is None


def test_plan_confidence_is_a_probability_not_a_gate():
    """At 0.80 the same plan claim reports a farther distance than at 1.0, and
    the answer is still a distance. Nothing is suppressed."""
    far = a_horizon(plan_confidence=0.80)
    near = a_horizon(plan_confidence=1.0)
    for h in (far, near):
        h.set_plan(FakePlan(["plan_task", "code_task"]), index=0)
        h.observe(0.0, tool="plan_task")
    a = far.next_use_s("qwen_72b_text", 0.0)
    b = near.next_use_s("qwen_72b_text", 0.0)
    assert a is not None and b is not None
    assert a > b


def test_elapsed_time_inside_a_step_brings_the_need_nearer():
    """The reported distance must track the clock inside a step. It does only
    because `resolution_s` is 1 s and not the step duration: flooring at a step
    would flatten the entire final approach, and whether that mattered would
    depend on the arbitrator's D."""
    h = a_horizon(plan_confidence=1.0)
    h.set_plan(FakePlan(["plan_task", "code_task"]), index=0)
    h.observe(0.0, tool="plan_task")
    assert h.next_use_s("qwen_72b_text", 0.0) == pytest.approx(STEP)
    assert h.next_use_s("qwen_72b_text", 20.0) == pytest.approx(STEP - 20.0)
    # a step that overruns reports "imminent", never a negative distance
    dt = h.next_use_s("qwen_72b_text", 5 * STEP)
    assert dt is not None and dt >= 0.0
    assert check_horizon(dt) is dt


def test_the_two_signals_are_simultaneous_not_a_fallback_chain():
    """With the plan already answering, switching the model-transition signal on
    must still change the answer. If the code short-circuited on the plan, this
    would be equal."""
    demand = DEMAND
    # A hand-built signal, enabled explicitly. 0.9893 is the corrected
    # offset-1 self-loop; it is a fixture here, not a claim about the table.
    tbl = TransitionSignal(
        model_transitions={"Qwen/Qwen2.5-VL-72B-Instruct":
                           {1: [{"target": "Qwen/Qwen2.5-VL-72B-Instruct",
                                 "target_type": "model", "probability": 0.9893}]}},
        use_model_transitions=True,
    )
    plan = FakePlan(["plan_task", "code_task",
                     "computation_task_screw_dislocation"])

    plan_only = PlanTransitionHorizon(
        demand, STEP, horizon_s=L, transitions=TransitionSignal.disabled())
    both = PlanTransitionHorizon(
        demand, STEP, horizon_s=L, transitions=tbl, llm_step_s=STEP / 4.0)
    for h in (plan_only, both):
        h.set_plan(plan, index=0)
        h.observe(0.0, tool="plan_task", model="Qwen/Qwen2.5-VL-72B-Instruct")

    a = plan_only.next_use_s("qwen_72b", 0.0)
    b = both.next_use_s("qwen_72b", 0.0)
    assert a is not None and b is not None
    assert b < a, "the transition signal contributed nothing on top of the plan"

    # and losing the plan is a degradation, not a switch: the table alone still
    # answers.
    table_only = PlanTransitionHorizon(
        demand, STEP, horizon_s=L, transitions=tbl, llm_step_s=STEP / 4.0)
    table_only.observe(0.0, model="Qwen/Qwen2.5-VL-72B-Instruct")
    assert table_only.next_use_s("qwen_72b", 0.0) is not None


def test_explain_shows_the_working_for_every_signal():
    # A hand-built signal, enabled explicitly. 0.9893 is the corrected
    # offset-1 self-loop; it is a fixture here, not a claim about the table.
    tbl = TransitionSignal(
        model_transitions={"Qwen/Qwen2.5-VL-72B-Instruct":
                           {1: [{"target": "Qwen/Qwen2.5-VL-72B-Instruct",
                                 "probability": 0.9893}]}},
        use_model_transitions=True)
    h = PlanTransitionHorizon(DEMAND, STEP, horizon_s=L, transitions=tbl)
    h.set_plan(FakePlan(["plan_task", "computation_task_screw_dislocation"]), 0)
    h.observe(0.0, tool="plan_task", model="Qwen/Qwen2.5-VL-72B-Instruct")
    out = h.explain("qwen_72b", 0.0)
    srcs = {a["source"] for a in out["arrivals"]}
    assert "plan" in srcs and "transition:model" in srcs
    assert 0.0 < out["mass_within_lookahead"] <= 1.0
    assert out["next_use_s"] == h.next_use_s("qwen_72b", 0.0)
    assert out["lookahead_s"] == L


def test_both_halves_of_the_table_are_off_by_default():
    """Kept off, but for entirely different reasons than before A3's fix.

    The old justification — that a table offset was not a plan step — is GONE:
    the corrected file declares `tool_call_subsequence`, and on that basis an
    offset IS a plan step. What keeps the default False now is what is in the
    corpus, not how it is indexed:

      * 88.8% of `tool_call` events and 84.6% of `llm_call` events in
        `logs/workflow_traces/runtime_trace_*.jsonl` come from traces whose
        median inter-`llm_call` gap is under a second, and nothing filters them.
      * `plan_task`'s n=45 offset-1 self-loop is 42 sub-second logging pairs and
        3 genuine ones.
      * the model self-loop (p=0.9893) barely beats its own base rate (0.9524
        pooled, 0.7805 on real traces only) and is a SELF-loop, so it says
        nothing about switching — the only thing a residency decision needs
        from a model predictor.

    Every one of those is fixable, which is why the machinery and the flags
    stay. This test exists so the default is re-examined deliberately rather
    than drifting.
    """
    sig = TransitionSignal.load()
    assert sig.use_tool_transitions is False
    assert sig.use_model_transitions is False
    assert sig.tool_offsets("run_ase") == ()
    assert sig.model_offsets("Qwen/Qwen2.5-VL-72B-Instruct") == ()
    on = TransitionSignal.load(use_tool_transitions=True,
                               use_model_transitions=True)
    assert on.tool_offsets("run_ase") != ()
    assert on.model_offsets("Qwen/Qwen2.5-VL-72B-Instruct") != ()


def test_enabling_a_half_requires_its_declared_offset_basis(tmp_path):
    """A pre-fix table's offsets were counted over a MIXED tool_call+llm_call
    sequence, so they are on NEITHER basis, and converting them with either step
    constant is an arithmetic error. Loading such a file with a half enabled
    must raise, not approximate."""
    path = REPO / "runtime" / "predictor" / "data" / "learned_transitions.json"
    if not path.exists():
        pytest.skip("learned_transitions.json not present")
    raw = json.loads(path.read_text())
    assert raw["offset_basis"]["tool_transitions"] == EXPECTED_TOOL_BASIS
    assert raw["offset_basis"]["model_transitions"] == EXPECTED_MODEL_BASIS

    stale = dict(raw)
    stale.pop("offset_basis")
    q = tmp_path / "prefix_table.json"
    q.write_text(json.dumps(stale))
    # reading it is fine; ENABLING a half of it is not
    assert TransitionSignal.load(q).use_tool_transitions is False
    for kw in ({"use_tool_transitions": True}, {"use_model_transitions": True}):
        with pytest.raises(TransitionBasisMismatch):
            TransitionSignal.load(q, **kw)


def test_the_corrected_table_indexes_the_tool_subsequence():
    """The fix, pinned to its reason rather than to the symptom it removed.

    Before A3's fix `plan_task` had one row — a self-transition at offset 2 —
    because the learner counted offsets over a mixed tool_call+llm_call sequence
    and `plan_task`'s real successor fell outside `max_offset=3`. On the
    corrected basis the entry tool of the workload the paper evaluates has its
    real successors back."""
    path = REPO / "runtime" / "predictor" / "data" / "learned_transitions.json"
    if not path.exists():
        pytest.skip("learned_transitions.json not present")
    raw = json.loads(path.read_text())
    assert raw.get("offset_basis", {}).get("tool_transitions") == EXPECTED_TOOL_BASIS, (
        "no declared tool offset basis — this is a PRE-FIX table")
    pt = raw["tool_transitions"]["plan_task"]
    assert sorted(pt) == ["1", "2", "3"]
    top = pt["1"][0]
    assert top["target"] == "code_task"
    assert top["probability"] == pytest.approx(0.5455, abs=5e-4)
    assert top["count"] == 78


def test_the_model_self_loop_does_not_beat_its_base_rate():
    """The measurement that removed the reason for enabling model_transitions.

    The corrected offset-1 self-loop is p=0.9893 (n=11311); the p=0.9973 once
    quoted came from the mixed-basis file and is void. Against an unconditional
    base rate of 0.9524 for the same model among `llm_call` events, knowing the
    current model is worth about three points."""
    path = REPO / "runtime" / "predictor" / "data" / "learned_transitions.json"
    if not path.exists():
        pytest.skip("learned_transitions.json not present")
    raw = json.loads(path.read_text())
    mt = raw["model_transitions"]["Qwen/Qwen2.5-VL-72B-Instruct"]["1"]
    loop = next(r for r in mt if r["target"] == "Qwen/Qwen2.5-VL-72B-Instruct")
    assert loop["probability"] == pytest.approx(0.9893, abs=5e-4)
    assert loop["count"] == 11311
    # if this ever rises materially above the base rate, revisit the default
    assert loop["probability"] - 0.9524 < 0.10


def test_resync_keeps_the_cursor_from_drifting_on_divergence():
    h = a_horizon(plan_confidence=1.0)
    h.set_plan(FakePlan(["plan_task", "code_task",
                         "computation_task_screw_dislocation"]), index=0)
    h.observe(0.0, tool="plan_task")
    # the agent skips code_task and jumps to the computation
    h.observe(10.0, tool="computation_task_screw_dislocation")
    # the cursor is past the computation, so nothing in the plan needs qwen_72b
    assert h.next_use_s("qwen_72b", 10.0) is None


def test_a_missing_plan_is_a_degradation_not_an_error():
    h = a_horizon()
    h.observe(0.0, tool="plan_task")
    assert h.next_use_s("qwen_72b_text", 0.0) is None       # no signal -> None
    h.set_plan(None)
    assert h.next_use_s("qwen_72b_text", 0.0) is None


# --------------------------------------------------------------------------
# L is not D — enforced at the source level
# --------------------------------------------------------------------------


def _names_in(module) -> set:
    src = Path(module.__file__).read_text()
    return {t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.NAME}


def test_the_module_never_names_a_decay_scale():
    """L and D are within a factor of 30 of each other, so conflating them
    produces a plausible-looking number rather than a crash. The defence is
    that the decay scale is not nameable in this file at all."""
    names = _names_in(hz_mod)
    for forbidden in ("decay_s", "DEFAULT_DECAY_S", "HORIZON_H", "H"):
        assert forbidden not in names, (
            f"{forbidden} appears in runtime/residency/horizon.py — the decay "
            "scale D belongs to the arbitrator, the lookahead L belongs here")


def test_the_module_does_not_import_the_value_function_or_the_arbitrator():
    """The estimator reports a distance. Pricing it is Eq. 1's job, and Eq. 1
    needs D, which this module must not have."""
    src = Path(hz_mod.__file__).read_text()
    assert "from runtime.residency.arbitrator" not in src
    assert "import arbitrator" not in src
    names = _names_in(hz_mod)
    assert "value" not in names
    assert "value_density" not in names
    # check_horizon is the ONE thing it takes from the contract.
    assert "check_horizon" in names


def test_the_module_has_no_confidence_threshold_to_tune():
    """§1.4: a fixed gate calibrated on 165 traces admits nothing on 490. The
    absence is the design; assert the absence."""
    names = _names_in(hz_mod)
    for forbidden in ("confidence_threshold", "min_confidence",
                      "CONFIDENCE_THRESHOLD", "MIN_CONFIDENCE"):
        assert forbidden not in names


def test_horizon_s_is_not_used_as_a_step_or_decay_constant():
    """L must only ever price the unseen tail and bound the window. If it were
    also the step duration or the resolution, changing L would silently change
    every reported distance rather than only the boundary."""
    h = a_horizon(plan_confidence=1.0, horizon_s=600.0)
    h2 = a_horizon(plan_confidence=1.0, horizon_s=6000.0)
    for x in (h, h2):
        x.set_plan(FakePlan(["plan_task", "code_task"]), index=0)
        x.observe(0.0, tool="plan_task")
    # a CERTAIN plan claim is at k*step regardless of L
    assert h.next_use_s("qwen_72b_text", 0.0) == pytest.approx(STEP)
    assert h2.next_use_s("qwen_72b_text", 0.0) == pytest.approx(STEP)


def test_L_only_prices_the_unseen_tail():
    """With uncertainty present, a larger L discounts the unseen mass less
    harshly, so the reported distance moves — but it moves through the tail
    term only, which is the single place L is allowed to appear."""
    near = certainty_equivalent_distance(
        [Arrival(30.0, 0.5, "a")], lookahead_s=600.0, resolution_s=1.0)
    wide = certainty_equivalent_distance(
        [Arrival(30.0, 0.5, "a")], lookahead_s=6000.0, resolution_s=1.0)
    assert near is not None and wide is not None
    assert wide > near


# --------------------------------------------------------------------------
# Measured constants and the demand map
# --------------------------------------------------------------------------


def test_measured_step_tables_are_faceted_and_never_pooled():
    """Every key is (workload, gpu_family). A pooled entry would be a constant
    that describes neither node type — the standing rule."""
    for table in (MEASURED_TOOL_STEP_S, MEASURED_LLM_STEP_S):
        for key, v in table.items():
            assert isinstance(key, tuple) and len(key) == 2
            workload, gpu = key
            assert gpu in ("L40S", "Blackwell"), gpu
            assert v > 0
    # the same workload on two node types differs enough that pooling is wrong
    a = MEASURED_TOOL_STEP_S[("chemgraph_swap", "Blackwell")]
    b = MEASURED_TOOL_STEP_S[("chemgraph_swap", "L40S")]
    assert b / a > 2.0


def test_the_lookahead_default_is_far_larger_than_the_arbitrators_decay():
    """Not imported from the arbitrator — restated, so that if either moves the
    separation is checked. D is 60 s; if L ever approaches it, Eq. 1 collapses
    to a static s/GB ranking with no time term (I3)."""
    assert DEFAULT_LOOKAHEAD_S >= 10 * 60.0


def test_demand_map_reads_both_shapes_of_tool_resources_json():
    """Including the residency_artifact block, which has deliberately no
    `consumer_tool` key and so is invisible to ResourceRegistry.from_json."""
    path = REPO / "runtime" / "predictor" / "data" / "tool_resources.json"
    if not path.exists():
        pytest.skip("tool_resources.json not present")
    dm = DemandMap.from_tool_resources(path)
    assert "qwen_32b" in dm.resources_for_tool("plan_task")
    assert "w_eam4_big_activated" in dm.resources_for_tool(
        "computation_task_screw_dislocation")
    assert "mace_mp:medium" in dm.resources_for_tool("run_ase")
    assert dm.resources_for_tool("no_such_tool") == frozenset()


# --------------------------------------------------------------------------
# Recorded traces (skipped when the gitignored recordings are absent)
# --------------------------------------------------------------------------


def _traces(workload: str, gpu: str, limit: int = 40):
    """[(trace.jsonl, metrics.csv|None)] for COMPLETED runs of one facet.

    `status == "completed"` is part of the selection, not a nicety: a failed or
    preempted run contributes a truncated need sequence, and 18 of the 83
    AtomAgents runs carrying both files are failed.
    """
    out = []
    for meta_path in sorted(EVAL_RUNS.glob(f"{workload}/*/*/meta.json")):
        trace = meta_path.parent / "trace.jsonl"
        if not trace.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "completed":
            continue
        if gpu not in " ".join(meta.get("gpus") or []):
            continue
        metrics = meta_path.parent / "metrics.csv"
        out.append((trace, metrics if metrics.exists() else None))
        if len(out) >= limit:
            break
    return out


@pytest.mark.parametrize("workload,gpu", [("chemgraph_swap", "Blackwell"),
                                          ("atomagents_exp3", "Blackwell")])
def test_replay_over_a_real_trace_obeys_I3_everywhere(workload, gpu):
    rows = _traces(workload, gpu, limit=8)
    if not rows:
        pytest.skip(f"no completed {workload}/{gpu} traces on this checkout")
    dm = DemandMap.from_tool_resources()
    for trace, metrics in rows:
        h = ReplayHorizon.from_trace(trace, dm, horizon_s=DEFAULT_LOOKAHEAD_S,
                                     metrics_csv=metrics)
        for rid in dm.resource_ids():
            uses = h.uses(rid)
            end = (uses[-1] + 2 * DEFAULT_LOOKAHEAD_S) if uses else 100.0
            now = 0.0
            while now <= end:
                dt = h.next_use_s(rid, now)
                assert check_horizon(dt) is dt
                if dt is not None:
                    assert 0.0 <= dt <= DEFAULT_LOOKAHEAD_S
                now += 37.0


def test_a_real_trace_gives_monotone_distances_between_uses():
    rows = _traces("atomagents_exp3", "Blackwell", limit=30)
    if not rows:
        pytest.skip("no completed atomagents_exp3/Blackwell traces")
    dm = DemandMap.from_tool_resources()
    checked = 0
    for trace, metrics in rows:
        h = ReplayHorizon.from_trace(trace, dm, horizon_s=DEFAULT_LOOKAHEAD_S,
                                     metrics_csv=metrics)
        for rid in h.resource_ids():
            uses = h.uses(rid)
            if len(uses) < 2:
                continue
            checked += 1
            # sample strictly INSIDE one inter-use interval: at the endpoints
            # the target flips to the following use, which is the one thing
            # allowed to move the distance back out.
            lo, hi = uses[0], uses[1]
            prev = None
            for i in range(1, 10):
                now = lo + (hi - lo) * i / 10.0
                dt = h.next_use_s(rid, now)
                if dt is None:
                    prev = None
                    continue
                assert abs(now + dt - hi) < 1e-6
                if prev is not None:
                    assert dt < prev
                prev = dt
    if checked == 0:
        pytest.skip("no resource in these traces was used twice")


def test_duplicates_are_adjudicated_against_metrics_not_against_a_clock():
    """The rule that replaced the 1 s collapse, and the reason it had to.

    `plan_task` emits twice and executes once, so its second event has no
    execution and is dropped. `analyze_screw_core` emits twice and executes
    twice — the two-potential comparison the workload exists to perform — so
    BOTH events survive even though they are 0.946-1.084 s apart, which any
    threshold near a second would have merged. The two populations abut in time
    and are separated only by the independent instrumentation.
    """
    rows = _traces("atomagents_exp3_aligned", "Blackwell", limit=40) \
        + _traces("atomagents_exp2", "Blackwell", limit=40) \
        + _traces("atomagents_exp3", "Blackwell", limit=40)
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        pytest.skip("no completed AtomAgents runs with a metrics.csv")

    dropped_plan = kept_close = 0
    for trace, metrics in rows:
        raw = _read_tool_calls(trace)
        kept, unadj = adjudicate_needs(raw, read_tool_executions(metrics))
        n_raw = sum(1 for _, t in raw if t == "plan_task")
        n_kept = sum(1 for _, t in kept if t == "plan_task")
        dropped_plan += n_raw - n_kept
        asc = sorted(t for t, n in kept if n == "analyze_screw_core")
        kept_close += sum(1 for a, b in zip(asc, asc[1:]) if b - a < 1.1)
        # a tool with no execution record is named, never silently dropped
        assert "code_task" in unadj or all(t != "code_task" for _, t in raw)

    assert dropped_plan > 0, "no plan_task logging duplicate was adjudicated away"
    assert kept_close > 0, (
        "every close analyze_screw_core repeat was lost — that is the failure "
        "mode a time threshold has and this rule must not")


def test_code_task_is_unadjudicable_and_says_so():
    """`code_task` has no `agent:` phase in any metrics.csv, so `qwen_72b_text`
    cannot be adjudicated and its reuse distance is a BRACKET (764.3-1297.3 s),
    not an estimate. The helper must name the tool rather than quietly keep or
    quietly drop it."""
    rows = [r for r in _traces("atomagents_exp3", "Blackwell", limit=40)
            if r[1] is not None]
    if not rows:
        pytest.skip("no completed atomagents_exp3/Blackwell runs with metrics")
    assert "code_task" not in PHASE_TO_TOOL.values()
    dm = DemandMap.from_tool_resources()
    _, unadj = reuse_distances([t for t, _ in rows], dm,
                               metrics_csvs=[m for _, m in rows])
    assert "code_task" in unadj


def test_reuse_distance_helper_reports_per_resource_gaps():
    rows = _traces("atomagents_exp3", "Blackwell", limit=30)
    if not rows:
        pytest.skip("no completed atomagents_exp3/Blackwell traces")
    dm = DemandMap.from_tool_resources()
    d, unadj = reuse_distances([t for t, _ in rows], dm,
                               metrics_csvs=[m for _, m in rows])
    assert d, "expected at least one resource with a repeated need"
    for rid, gaps in d.items():
        assert all(g > 0 for g in gaps)
    step = mean_tool_step_s([t for t, _ in rows], [m for _, m in rows])
    assert step is not None and step > 0
