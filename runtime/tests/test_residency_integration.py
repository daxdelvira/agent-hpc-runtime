"""test_residency_integration.py — T1 + T2 against the REAL horizon estimator.

WHY THIS IS A TEST AND NOT A SCRATCH SCRIPT. Everything else in the residency
suite exercises one component against doubles: the ledger against FakeActor,
the arbitrator against a StubHorizon. This is the only thing that runs the
ledger, the arbitrator and A4's `ReplayHorizon` together, so it is the only
place a signature drift between them shows up. It lived in a scratchpad for
one afternoon, which is exactly how that kind of check disappears.

WHAT IT PINS, beyond "it runs":

  * The two time constants stay apart across the seam. D (Eq. 1's decay scale)
    comes from the arbitrator; L (the lookahead) comes from the estimator. If
    they ever coincide, Eq. 1 goes time-blind -- see I3 and
    test_residency_arbitrator.py. This file reads L from A4's module rather
    than hardcoding it, so a change to A4's default cannot silently make D
    equal L or silently move what a beyond-lookahead resource is worth.

  * The plan the arbitrator returns is EXECUTABLE against the ledger: every
    id it names is held, releasing them makes exactly enough room, and the
    caller -- not the arbitrator -- is the one that confirms each release (I2).

  * A resource beyond the lookahead is scored at L and discounted, never
    discarded (I3), through a real estimator that obeys the invariant by
    construction rather than a stub that could be wrong about it.

No GPU, no SLURM, no vLLM, no LAMMPS.
"""
from __future__ import annotations

import pytest

from runtime.residency.arbitrator import GreedyArbitrator
from runtime.residency.contract import ResourceClass, value_density
from runtime.residency.ledger import ResidencyLedger
from runtime.tests.test_residency_ledger import (
    EAM_POT,
    QWEN_32B,
    UNIREF50,
    UNIREF90,
    FakeActor,
)

horizon = pytest.importorskip("runtime.residency.horizon",
                              reason="T3 (A4's horizon estimator) not present")

PRODUCTION_BUDGET_GB = 256.0        # the allocation the paper's cells assume


def _ledger(budget_gb=PRODUCTION_BUDGET_GB):
    led = ResidencyLedger(budget_gb)
    for cls in ResourceClass:
        led.register_actor(FakeActor(cls))
    return led


def _stage(led, spec, now_s=0.0):
    actor = led.actor_for(spec.resource_class)
    return led.charge(spec, actor.stage(spec), now_s)


def test_the_two_time_constants_stay_apart_across_the_seam():
    """D from the arbitrator, L from the estimator, and D < L. Read from A4's
    module, so if the lookahead default moves this test says so rather than
    quietly changing what every beyond-lookahead resource is worth."""
    L = horizon.DEFAULT_LOOKAHEAD_S
    arb = GreedyArbitrator()
    hor = horizon.ReplayHorizon({}, horizon_s=L)
    assert hor.horizon_s == L
    assert arb.decay_s < L, (
        f"D={arb.decay_s} s must stay well below L={L} s; at D == L Eq. 1 "
        f"collapses to the static s/GB band with no time term at all (I3)")
    # what "beyond the lookahead" is worth, stated as arithmetic so a change
    # in either constant is visible in the diff of this file
    assert arb.score(UNIREF90, hor, 0.0) == pytest.approx(
        UNIREF90.static_density * arb.decay_s / L)


def test_the_plan_is_executable_against_the_ledger():
    """End to end: charge, ask, execute the eviction chain through the actors,
    charge the admitted resource, reconcile. The arbitrator never touches the
    ledger itself -- only the caller can confirm a release (I2)."""
    L = horizon.DEFAULT_LOOKAHEAD_S
    led = _ledger()
    for spec in (UNIREF50, EAM_POT):
        _stage(led, spec)
    hor = horizon.ReplayHorizon(
        {"uniref90": [30.0], "uniref50": [500.0], "eam_potential": [500.0]},
        horizon_s=L)
    arb = GreedyArbitrator()

    before = {e.spec.resource_id for e in led.entries()}
    plan = arb.admit(UNIREF90, led, hor, 0.0)
    assert {e.spec.resource_id for e in led.entries()} == before, \
        "admit() must not mutate the ledger"

    if plan.admit is None:                       # a decline is a valid outcome
        assert "DECLINED" in plan.rationale
        return
    for victim in plan.evict:
        assert led.is_held(victim)               # every named id is really held
        led.release(victim, 1.0)                 # and every release is confirmed
    _stage(led, UNIREF90, now_s=1.0)

    assert led.held_gb <= led.budget_gb
    assert led.release_shortfalls == []
    assert led.drift() == {}
    charged, measured = led.reconcile()["uniref90"]
    assert charged == pytest.approx(measured)


def test_a_resource_beyond_the_lookahead_is_discounted_not_discarded():
    """Through a real estimator: uniref90 has no recorded use, so next_use_s
    returns None -- 'not within L', never 'never again' (I3). It must still be
    worth something, and it must still be able to win budget against cheaper
    holders."""
    L = horizon.DEFAULT_LOOKAHEAD_S
    hor = horizon.ReplayHorizon({"uniref50": [], "eam_potential": []},
                                horizon_s=L)
    assert hor.next_use_s("uniref90", 0.0) is None
    arb = GreedyArbitrator()
    d = arb.score(UNIREF90, hor, 0.0)
    assert d > 0.0
    assert d == pytest.approx(value_density(UNIREF90, None, arb.decay_s, L))

    led = _ledger(150.0)
    for spec in (UNIREF50, EAM_POT):
        _stage(led, spec)
    plan = arb.admit(UNIREF90, led, hor, 0.0)
    # every resource here is beyond L, so all three are discounted by the same
    # factor and the static band decides: 3.179 > 2.968 > 2.247.
    assert plan.admit == "uniref90"
    assert plan.evict == ("eam_potential", "uniref50")


def test_the_below_band_resource_is_still_declined_through_a_real_estimator():
    """The EAM potential at 2.25 s/GB against the cheapest model in the band.
    The decline is the arbitrator working, and it must survive the swap from a
    stub horizon to a real one."""
    L = horizon.DEFAULT_LOOKAHEAD_S
    led = _ledger(140.0)
    _stage(led, QWEN_32B)                        # 129.7 GB, 3.81 s/GB
    hor = horizon.ReplayHorizon({"qwen_32b": [10.0], "eam_potential": [10.0]},
                                horizon_s=L)
    plan = GreedyArbitrator().admit(EAM_POT, led, hor, 0.0)
    assert plan.admit is None
    assert "DECLINED" in plan.rationale
    assert "no remaining holder is cheaper" in plan.rationale
