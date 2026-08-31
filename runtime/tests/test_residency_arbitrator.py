"""test_residency_arbitrator.py — T2, greedy single-victim retention.

THE FOUR THINGS THAT MUST NOT REGRESS, each with the reason:

  * It is GREEDY, meaning CHAINED and BOUNDED eviction: least-dense holder
    first, re-ranked after each step, more than one victim allowed, capped at
    max_victims. Greedy at 0.55 predictor accuracy reaches essentially the
    exact-solve ceiling (+20.6% of wall) and is nearly flat across accuracy;
    exact goes NEGATIVE at the accuracy we measure. A one-victim rule is a
    different, strictly more conservative policy that the +20.6% was not
    measured on. There is a source-level test below that the module contains no
    subset enumeration, because "add an exact arm for comparison" is exactly
    how it would come back.
  * It keeps D and L APART. D is Eq. 1's decay scale; L is the estimator's
    lookahead. Setting them equal makes the value function time-blind, which is
    a defect this file pins in two directions.
  * It is CLASS-BLIND (I4). Relabelling every resource's class must not move a
    decision.
  * It is RETAIN-ONLY (I5). `admit()` takes one already-needed candidate; a
    prefetch candidate never enters this ranking, because the shared currency
    caps prefetch benefit at min(load, dt) and a 5-second prefetch would
    outrank a 798-second retention.
  * DECLINING IS A DECISION. The EAM potential at 2.25 s/GB is below the model
    band (2.78-3.81) and loses to any model at any size. The arbitrator must
    decline it, must say which ranking produced that, and must record it.

The horizon estimator (T3) is somebody else's work; StubHorizon here is a
deterministic dict so these tests pin the arbitrator and nothing else.

No GPU, no vLLM, no LAMMPS, no SLURM.
"""
from __future__ import annotations

import inspect
import io
import tokenize

import pytest

from runtime.residency import arbitrator as arb_mod
from runtime.residency.arbitrator import (
    GreedyArbitrator,
    evict_until_fits,
    greedy_pack,
)
from runtime.residency.contract import (
    Arbitrator,
    HorizonEstimator,
    ResourceClass,
    ResourceSpec,
    Rung,
    value,
    value_density,
)
from runtime.residency.ledger import ResidencyLedger
from runtime.tests.test_residency_ledger import (
    EAM_POT,
    QWEN_32B,
    QWEN_72B,
    UNIREF50,
    UNIREF90,
    FakeActor,
)

QWEN_72B_TEXT = ResourceSpec("qwen_72b_text", ResourceClass.MODEL,
                             Rung.R2_PROCESS_BYTES, 276.3, 770.3, 2.19)


class StubHorizon:
    """Deterministic T3 stand-in. `None` means 'not within H' (I3) -- there is
    deliberately no way to express 'never again'."""

    def __init__(self, table: dict, horizon_s: float = 600.0):
        self._table = dict(table)
        self._h = float(horizon_s)

    @property
    def horizon_s(self) -> float:
        return self._h

    def next_use_s(self, resource_id, now_s):
        return self._table.get(resource_id, None)


class BrokenHorizon(StubHorizon):
    """An estimator that says 'never again'. I3 exists to reject this."""

    def next_use_s(self, resource_id, now_s):
        return float("inf")


def _ledger(budget_gb, held=()):
    led = ResidencyLedger(budget_gb)
    for cls in ResourceClass:
        led.register_actor(FakeActor(cls))
    for spec in held:
        a = led.actor_for(spec.resource_class)
        led.charge(spec, a.stage(spec), 0.0)
    return led


# --------------------------------------------------------------------------
# Shape, and the measured band the whole argument rests on
# --------------------------------------------------------------------------


def test_implements_the_contract_protocols():
    assert isinstance(GreedyArbitrator(), Arbitrator)
    assert isinstance(StubHorizon({}), HorizonEstimator)


def test_the_measured_value_density_band():
    """Provenance: tandem_build_plan_v2_20260829.md §0.1. If these move, every
    decline below is arguing about a different workload."""
    assert QWEN_32B.static_density == pytest.approx(3.81, abs=5e-3)
    assert QWEN_72B.static_density == pytest.approx(2.86, abs=5e-3)
    assert QWEN_72B_TEXT.static_density == pytest.approx(2.78, abs=5e-3)
    assert UNIREF90.static_density == pytest.approx(3.18, abs=5e-3)
    assert UNIREF50.static_density == pytest.approx(2.97, abs=5e-3)
    assert EAM_POT.static_density == pytest.approx(2.25, abs=5e-3)


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


def test_admits_into_slack_without_evicting_anything():
    led = _ledger(400.0, held=[UNIREF50])
    plan = GreedyArbitrator().admit(
        UNIREF90, led, StubHorizon({"uniref90": 30.0, "uniref50": 200.0}), 0.0)
    assert plan.admit == "uniref90"
    assert plan.evict == () and plan.freed_gb == 0.0
    assert "slack" in plan.rationale
    assert "value-density" in plan.rationale


def test_the_chain_evicts_least_dense_first_and_may_take_more_than_one():
    """GREEDY MEANS CHAINED. budget 150: uniref50 (36.08 GB) + eam (16.93 GB)
    held, so 96.99 GB of slack against a 117.20 GB candidate -- 20.21 GB short.
    The EAM potential is the least dense, so it goes first; it frees only 16.93
    GB, so the chain continues to uniref50. A one-victim rule would instead
    have taken uniref50 alone, which is a different policy and not the one the
    +20.6% was measured on."""
    led = _ledger(150.0, held=[UNIREF50, EAM_POT])
    hor = StubHorizon({"uniref90": 20.0, "uniref50": 400.0, "eam_potential": 400.0})
    plan = GreedyArbitrator().admit(UNIREF90, led, hor, 0.0)
    assert plan.admit == "uniref90"
    assert plan.evict == ("eam_potential", "uniref50")      # in eviction order
    assert plan.freed_gb == pytest.approx(16.93 + 36.08)


def test_the_chain_is_recorded_in_the_rationale():
    led = _ledger(150.0, held=[UNIREF50, EAM_POT])
    hor = StubHorizon({"uniref90": 20.0, "uniref50": 400.0, "eam_potential": 400.0})
    r = GreedyArbitrator().admit(UNIREF90, led, hor, 0.0).rationale
    assert "chain of 2" in r and "bound 3" in r
    assert "eam_potential@" in r and "-> uniref50@" in r     # order and prices
    assert "20.21 GB" in r and "53.01 GB" in r               # needed, freed
    assert "greedy-chained/value-density" in r
    assert "D=60 s, L=600 s" in r                            # both constants named


def test_the_chain_is_bounded_by_max_victims():
    """The bound exists so a pathological ranking cannot empty the ledger for
    one admission. Same scenario, same ranking, only the bound differs."""
    hor = StubHorizon({"uniref90": 20.0, "uniref50": 400.0, "eam_potential": 400.0})

    one = GreedyArbitrator(max_victims=1).admit(
        UNIREF90, _ledger(150.0, held=[UNIREF50, EAM_POT]), hor, 0.0)
    assert one.admit is None and one.evict == ()
    assert "chain bound max_victims=1 reached" in one.rationale
    assert "having freed 16.93 GB" in one.rationale          # it did try

    two = GreedyArbitrator(max_victims=2).admit(
        UNIREF90, _ledger(150.0, held=[UNIREF50, EAM_POT]), hor, 0.0)
    assert two.admit == "uniref90" and len(two.evict) == 2

    with pytest.raises(ValueError):
        GreedyArbitrator(max_victims=0)


def test_the_chain_stops_at_the_first_holder_that_is_not_cheaper():
    """Chaining never evicts something denser than the candidate, however much
    room is still missing."""
    led = _ledger(300.0, held=[QWEN_72B_TEXT, EAM_POT])       # 2.780 and 2.247
    hor = StubHorizon({"qwen_32b": 1.0, "qwen_72b_text": 1.0, "eam_potential": 1.0})
    # qwen_32b at 3.810 needs 129.7 - 6.77 = 122.93 GB. The EAM potential is
    # cheaper and goes first (16.93 GB), qwen_72b_text is cheaper too, so the
    # chain completes at two.
    plan = GreedyArbitrator().admit(QWEN_32B, led, hor, 0.0)
    assert plan.evict == ("eam_potential", "qwen_72b_text")
    # Now make the model dearer than the candidate: only the potential goes,
    # which is not enough, so the whole admission is declined.
    led2 = _ledger(300.0, held=[QWEN_72B_TEXT, EAM_POT])
    hor2 = StubHorizon({"qwen_32b": 300.0, "qwen_72b_text": 1.0,
                        "eam_potential": 1.0})
    plan2 = GreedyArbitrator().admit(QWEN_32B, led2, hor2, 0.0)
    assert plan2.admit is None
    assert "no remaining holder is cheaper" in plan2.rationale


def test_a_below_band_resource_is_declined_and_the_decline_is_a_decision():
    """The EAM potential at 2.25 s/GB against the cheapest model in the band
    (qwen_72b_text, 2.78). Expected to lose. This is the arbitrator working."""
    led = _ledger(280.0, held=[QWEN_72B_TEXT])
    hor = StubHorizon({"eam_potential": 10.0, "qwen_72b_text": 10.0})
    a = GreedyArbitrator()
    plan = a.admit(EAM_POT, led, hor, 7.0)
    assert plan.admit is None
    assert plan.evict == () and plan.freed_gb == 0.0
    assert "DECLINED" in plan.rationale
    assert "greedy-chained/value-density" in plan.rationale
    assert "qwen_72b_text" in plan.rationale
    # recorded, not dropped
    assert a.decisions and a.decisions[-1] == (7.0, plan)
    assert a.declines() == [(7.0, plan)]


def test_below_band_loses_at_any_size():
    """Size was the wrong diagnosis; value density is the right one. Shrink the
    potential to a tenth and it still loses to the cheapest model."""
    tiny = ResourceSpec("eam_tiny", EAM_POT.resource_class, EAM_POT.held_rung,
                        1.693, 4.283, 0.478)
    assert tiny.static_density == pytest.approx(EAM_POT.static_density, rel=1e-9)
    led = _ledger(277.0, held=[QWEN_72B_TEXT])
    hor = StubHorizon({"eam_tiny": 10.0, "qwen_72b_text": 10.0})
    plan = GreedyArbitrator().admit(tiny, led, hor, 0.0)
    assert plan.admit is None and "DECLINED" in plan.rationale


def test_a_candidate_larger_than_the_whole_allocation_is_declined():
    led = _ledger(100.0)
    plan = GreedyArbitrator().admit(UNIREF90, led, StubHorizon({"uniref90": 5.0}), 0.0)
    assert plan.admit is None
    assert "does not fit" in plan.rationale and "even empty" in plan.rationale


def test_chaining_admits_what_a_one_victim_rule_would_have_declined():
    """Two victims together fit the candidate; neither does alone. The first
    draft of the contract said decline here. It now says chain, because that
    is the arm the +20.6% was measured on. Synthetic sizes, chosen to isolate
    the rule."""
    a1 = ResourceSpec("syn_a", ResourceClass.DATA_PATTERN_A, Rung.R3_ACTIVATED,
                      25.0, 50.0, 0.0)                     # 2.00 s/GB static
    a2 = ResourceSpec("syn_b", ResourceClass.DATA_PATTERN_A, Rung.R3_ACTIVATED,
                      25.0, 50.0, 0.0)                     # 2.00 s/GB static
    cand = ResourceSpec("syn_c", ResourceClass.DATA_PATTERN_A, Rung.R3_ACTIVATED,
                        90.0, 360.0, 0.0)                  # 4.00 s/GB static
    led = _ledger(100.0, held=[a1, a2])
    hor = StubHorizon({"syn_a": 300.0, "syn_b": 300.0, "syn_c": 10.0})
    plan = GreedyArbitrator().admit(cand, led, hor, 0.0)
    assert plan.admit == "syn_c"
    assert plan.evict == ("syn_a", "syn_b")                # ties break by id
    assert plan.freed_gb == pytest.approx(50.0)
    assert "40.00 GB" in plan.rationale                    # what was needed
    # and the bound still governs: one victim is not enough here.
    assert GreedyArbitrator(max_victims=1).admit(
        cand, _ledger(100.0, held=[a1, a2]), hor, 0.0).admit is None


def test_nothing_cheaper_than_the_candidate_means_decline():
    led = _ledger(280.0, held=[QWEN_72B_TEXT])
    hor = StubHorizon({"qwen_32b": 10.0, "qwen_72b_text": 10.0})
    # qwen_32b (3.81) IS denser than the holder (2.78), so this must admit...
    assert GreedyArbitrator().admit(QWEN_32B, led, hor, 0.0).evict == ("qwen_72b_text",)
    # ...and the reverse must not.
    led2 = _ledger(280.0, held=[QWEN_32B])
    plan = GreedyArbitrator().admit(QWEN_72B_TEXT, led2, hor, 0.0)
    assert plan.admit is None
    assert "no remaining holder is cheaper" in plan.rationale


def test_an_already_held_candidate_is_a_no_op_decision():
    led = _ledger(400.0, held=[UNIREF90])
    plan = GreedyArbitrator().admit(UNIREF90, led, StubHorizon({"uniref90": 5.0}), 0.0)
    assert plan.admit == "uniref90" and plan.evict == ()
    assert "already held" in plan.rationale


def test_min_density_ratio_is_hysteresis_not_policy():
    led = _ledger(280.0, held=[QWEN_72B_TEXT])       # 2.78 s/GB
    hor = StubHorizon({"qwen_32b": 10.0, "qwen_72b_text": 10.0})
    assert GreedyArbitrator(min_density_ratio=1.0).admit(
        QWEN_32B, led, hor, 0.0).admit == "qwen_32b"
    # 3.81 / 2.78 = 1.37, so a 1.5x churn margin refuses the swap.
    assert GreedyArbitrator(min_density_ratio=1.5).admit(
        QWEN_32B, led, hor, 0.0).admit is None
    with pytest.raises(ValueError):
        GreedyArbitrator(min_density_ratio=0.9)


# --------------------------------------------------------------------------
# I3 — the horizon never says "never again"
# --------------------------------------------------------------------------


def test_dt_none_is_scored_at_the_lookahead_not_discarded():
    """'Not within L' is not 'worthless'. It is priced AT the lookahead and
    discounted -- the most discounted thing the estimator is entitled to say --
    rather than dropped. Excluding predicted-dead resources cost ~9 points in
    simulation and produced a finding that had to be retracted."""
    D, L = 60.0, 600.0
    assert value(UNIREF90, None, D, L) == pytest.approx(UNIREF90.benefit_s * D / L)
    assert value_density(UNIREF90, None, D, L) == pytest.approx(
        UNIREF90.static_density * D / L)
    assert value(UNIREF90, None, D, L) > 0.0            # NOT discarded

    # and it survives a real decision. Every resource here is beyond L, so all
    # three are discounted equally and the static band decides: uniref90
    # (3.179) outranks uniref50 (2.968) and the EAM potential (2.247), and is
    # admitted on a chain of both. Were dt=None scored as worthless, the
    # candidate would be the cheapest thing in the ranking and declined.
    led = _ledger(150.0, held=[UNIREF50, EAM_POT])
    hor = StubHorizon({})                               # everything -> None
    a = GreedyArbitrator()
    assert a.score(UNIREF90, hor, 0.0) == pytest.approx(
        UNIREF90.static_density * D / L)
    plan = a.admit(UNIREF90, led, hor, 0.0)
    assert plan.admit == "uniref90"
    assert plan.evict == ("eam_potential", "uniref50")
    assert "beyond L=600 s" in plan.rationale


def test_eq1_is_time_blind_when_the_decay_scale_equals_the_lookahead():
    """THE DEFECT THAT RATIFIED THE TWO-CONSTANT SIGNATURE, kept as a guard.

    With D == L, every dt an I3-conforming estimator can produce is <= D, so
    max(dt, D) == D and Eq. 1 collapses to benefit_s: the ranking degenerates
    to the static s/GB band with no time term at all. Numbers verified against
    contract.value on 2026-08-30 and now recorded in I3.
    """
    D = L = 600.0
    reachable = [0.0, 1.0, 60.0, 599.0, 600.0, None]
    scores = [value(UNIREF90, dt, D, L) for dt in reachable]
    assert scores == [pytest.approx(UNIREF90.benefit_s)] * len(reachable)
    # the same collapse when the lookahead is not passed at all
    assert value(UNIREF90, None, D) == pytest.approx(UNIREF90.benefit_s)

    # At the decision level: imminent and distant become indistinguishable.
    led = _ledger(150.0, held=[UNIREF50, EAM_POT])
    soon = StubHorizon({"uniref90": 1.0, "uniref50": 599.0, "eam_potential": 599.0})
    late = StubHorizon({"uniref90": 599.0, "uniref50": 1.0, "eam_potential": 1.0})
    blind = GreedyArbitrator(decay_s=600.0)             # D == L == 600
    assert blind.admit(UNIREF90, led, soon, 0.0).evict == \
           blind.admit(UNIREF90, led, late, 0.0).evict


def test_separating_the_decay_scale_from_the_lookahead_restores_the_time_term():
    """The ratified fix, from the other side: D=60 against L=600 discriminates
    at every reachable dt. Values checked against contract.value_density."""
    D, L = 60.0, 600.0
    assert value_density(UNIREF90, 0.0, D, L) == pytest.approx(3.1792, abs=5e-4)
    assert value_density(UNIREF90, 120.0, D, L) == pytest.approx(1.5896, abs=5e-4)
    assert value_density(UNIREF90, 599.0, D, L) == pytest.approx(0.3184, abs=5e-4)
    assert value_density(UNIREF90, None, D, L) == pytest.approx(0.3179, abs=5e-4)

    a = GreedyArbitrator()                              # D defaults to 60 s
    assert a.decay_s == 60.0
    hor = StubHorizon({"uniref90": 120.0}, horizon_s=L)
    assert a.score(UNIREF90, hor, 0.0) == pytest.approx(1.5896, abs=5e-4)
    with pytest.raises(ValueError):
        GreedyArbitrator(decay_s=0.0)


def test_the_decay_scale_can_flip_a_decision():
    """The point of keeping D short: WHEN a resource is needed matters again.

    uniref90 (3.179 s/GB static) is not needed for 590 s; uniref50 (2.968) and
    the EAM potential (2.247) are needed in 1 s. At D=600 the candidate wins on
    the static band alone and evicts both. At the paper's D=60 s it is
    discounted to 0.323 s/GB, nothing held is cheaper, and it is declined.
    """
    hor = StubHorizon({"uniref90": 590.0, "uniref50": 1.0, "eam_potential": 1.0})
    wide = GreedyArbitrator(decay_s=600.0).admit(
        UNIREF90, _ledger(150.0, held=[UNIREF50, EAM_POT]), hor, 0.0)
    assert wide.admit == "uniref90" and len(wide.evict) == 2

    plan = GreedyArbitrator().admit(
        UNIREF90, _ledger(150.0, held=[UNIREF50, EAM_POT]), hor, 0.0)
    assert plan.admit is None
    assert "DECLINED" in plan.rationale


def test_an_estimator_that_says_never_again_is_rejected_at_runtime():
    led = _ledger(400.0, held=[UNIREF50])
    with pytest.raises(ValueError, match="I3 forbids"):
        GreedyArbitrator().admit(UNIREF90, led, BrokenHorizon({}), 0.0)


def test_a_negative_horizon_is_rejected():
    led = _ledger(400.0)
    with pytest.raises(ValueError):
        GreedyArbitrator().admit(UNIREF90, led, StubHorizon({"uniref90": -1.0}), 0.0)


# --------------------------------------------------------------------------
# I4 / I5 — structural guards
# --------------------------------------------------------------------------


def test_relabelling_the_class_does_not_change_the_decision():
    """I4: the arbitrator ranks ResourceSpec and never learns what a class is."""
    hor = StubHorizon({"uniref90": 20.0, "uniref50": 400.0, "eam_potential": 400.0})
    base = GreedyArbitrator().admit(
        UNIREF90, _ledger(150.0, held=[UNIREF50, EAM_POT]), hor, 0.0)

    def relabel(spec, cls, rung):
        return ResourceSpec(spec.resource_id, cls, rung, spec.held_gb,
                            spec.cold_s, spec.ready_s)

    swapped = [relabel(UNIREF50, ResourceClass.MODEL, Rung.R2_PROCESS_BYTES),
               relabel(EAM_POT, ResourceClass.MODEL, Rung.R2_PROCESS_BYTES)]
    other = GreedyArbitrator().admit(
        relabel(UNIREF90, ResourceClass.DATA_PATTERN_B, Rung.R3_ACTIVATED),
        _ledger(150.0, held=swapped), hor, 0.0)
    assert (other.admit, other.evict, other.freed_gb) == \
           (base.admit, base.evict, base.freed_gb)


def test_the_module_contains_no_exact_solve_and_no_class_knowledge():
    """I4 and the greedy constraint, guarded at the source level, because both
    would come back as 'just an option for comparison'."""
    # Comments and docstrings are stripped first -- the prose in that module
    # names vLLM and LAMMPS precisely to say that it never touches them.
    src = inspect.getsource(arb_mod)
    code = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        code.append(tok.string)
    code = " ".join(code).lower()
    for banned in ("combinations", "itertools", "vllm", "lammps", "subprocess",
                   "resourceclass"):
        assert banned not in code, f"{banned!r} appears in arbitrator.py code"


def test_admit_does_not_mutate_the_ledger():
    """Only the caller can tell whether the release was honoured (I2), so the
    plan is a recommendation and the ledger is untouched."""
    led = _ledger(150.0, held=[UNIREF50, EAM_POT])
    before = {e.spec.resource_id: e.charged_gb for e in led.entries()}
    GreedyArbitrator().admit(
        UNIREF90, led, StubHorizon({"uniref90": 20.0}), 0.0)
    assert {e.spec.resource_id: e.charged_gb for e in led.entries()} == before


# --------------------------------------------------------------------------
# The primitives the simulator shares
# --------------------------------------------------------------------------


def test_evict_until_fits_is_deterministic_and_ties_break_by_id():
    sizes = {"a": 40.0, "b": 40.0, "c": 40.0}
    keep = evict_until_fits(sizes, lambda x: 1.0, 80.0)     # all keys equal
    assert keep == {"b", "c"}                               # "a" evicted first
    keep2 = evict_until_fits(sizes, lambda x: {"a": 9, "b": 1, "c": 5}[x], 80.0)
    assert keep2 == {"a", "c"}
    assert evict_until_fits(sizes, lambda x: 1.0, 200.0) == {"a", "b", "c"}
    assert evict_until_fits(sizes, lambda x: 1.0, 1.0) == set()


def test_evict_until_fits_accepts_any_currency():
    """The primitive is key-agnostic on purpose: the simulator's LRU arm hands
    it a last-use step and the cost-aware arm hands it Eq. 1 density. That
    difference IS the comparison the paper makes."""
    sizes = {"old": 50.0, "new": 50.0}
    assert evict_until_fits(sizes, lambda x: {"old": 1, "new": 9}[x], 50.0) == {"new"}
    dens = {"old": 3.8, "new": 2.2}
    assert evict_until_fits(sizes, lambda x: dens[x], 50.0) == {"old"}


def test_greedy_pack_takes_the_densest_that_fit_and_skips_the_rest():
    items = {"big": (100.0, 90.0),      # 1.11 /GB
             "dense": (60.0, 10.0),     # 6.00 /GB
             "mid": (40.0, 20.0)}       # 2.00 /GB
    assert greedy_pack(items, 30.0) == {"dense", "mid"}
    assert greedy_pack(items, 100.0) == {"dense", "mid"}
    assert greedy_pack(items, 120.0) == {"dense", "mid", "big"}
    assert greedy_pack({}, 10.0) == set()
