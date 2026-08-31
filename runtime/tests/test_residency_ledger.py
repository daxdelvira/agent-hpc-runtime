"""test_residency_ledger.py — T1, the one budget.

WHAT THESE TESTS ARE FOR. Two of the ledger's properties are invariants that
have already cost this project real time, and both fail SILENTLY if they
regress:

  I1  charges are MEASURED. A budget over numbers a caller asserted is a
      fiction, and every percentage computed against it is meaningless.
  I2  releases are CONFIRMED. Item D2 of the build plan -- "a worker that
      retains but cannot truly release" -- is the most likely silent failure
      in this build. `test_release_not_honoured_*` is the assertion that
      catches it, and it is the reason this file exists.

FakeActor lives here and NOT in ledger.py on purpose: a test double that can be
configured to lie is exactly the kind of thing that ends up imported by
production code by accident.

No GPU, no vLLM, no LAMMPS, no SLURM.
"""
from __future__ import annotations

import pytest

from runtime.residency.contract import (
    Ledger,
    LedgerEntry,
    ReleaseNotHonoured,
    ResidencyActor,
    ResourceClass,
    ResourceSpec,
    Rung,
)
from runtime.residency.ledger import (
    DEFAULT_RELEASE_TOLERANCE_GB,
    BudgetExceeded,
    ResidencyLedger,
)

# --------------------------------------------------------------------------
# The measured catalogue (provenance: sc-workshop-paper/tandem_build_plan_v2
# _20260829.md §0.1, which cites bench_arbitration_harness.py). Sizes and
# times are measured; nothing here is invented.
# --------------------------------------------------------------------------

QWEN_32B = ResourceSpec("qwen_32b", ResourceClass.MODEL,
                        Rung.R2_PROCESS_BYTES, 129.7, 495.2, 1.03)
QWEN_72B = ResourceSpec("qwen_72b", ResourceClass.MODEL,
                        Rung.R2_PROCESS_BYTES, 279.0, 800.5, 2.21)
UNIREF90 = ResourceSpec("uniref90", ResourceClass.DATA_PATTERN_A,
                        Rung.R3_ACTIVATED, 117.20, 372.6, 0.0)
UNIREF50 = ResourceSpec("uniref50", ResourceClass.DATA_PATTERN_A,
                        Rung.R3_ACTIVATED, 36.08, 107.1, 0.0)
EAM_POT = ResourceSpec("eam_potential", ResourceClass.DATA_PATTERN_B,
                       Rung.R3_ACTIVATED, 16.93, 42.83, 4.78)


# --------------------------------------------------------------------------
# The test double
# --------------------------------------------------------------------------


class FakeActor:
    """A ResidencyActor whose REPORT and whose REALITY are independent.

    THE FIRST VERSION OF THIS DOUBLE COULD NOT EXPRESS THE BUG THE LEDGER
    EXISTS TO CATCH, and that is the more important half of the 2026-08-30
    finding. It computed `freed = gb * release_fraction`, returned exactly
    that, and put the remainder back into `_held` -- so its return value and
    its memory agreed BY CONSTRUCTION. It modelled an actor that under-frees
    and says so. The real failure is an actor that under-frees and does not
    know: glibc arena retention returned 0.084 of 0.364 GB while an honest
    implementation would have reported the full 0.364. A suite built on the
    old double was structurally blind to that, not merely incomplete.

    So the knobs are split:
      release_fraction  what actually comes back  (reality)
      residue_gb        an absolute leftover, for the slow drip
      report_gb         what release() RETURNS: None = the truth, "full" =
                        claims everything it was holding, or any float
                        (including nan) for an actor that reports nonsense
      raise_on_release  an exception to raise instead of returning
      witness_delta_gb  what release_witness() returns: None = "I have no
                        witness" (the default), a float = the GB the enclosing
                        allocation gave back, or an exception instance to
                        raise. Independent of every other knob, because the
                        whole point of the witness is that it does not come
                        from the same accounting as the report.
      teardown          the actor KILLS THE PROCESS rather than freeing inside
                        it, so measure_held_gb() afterwards is 0.0 BY
                        CONSTRUCTION whatever the OS actually did. This is the
                        shape of A5's VllmModelActor and the one that defeats a
                        before/after comparison, because the ledger cannot tell
                        it apart from an honest in-process release.
    """

    def __init__(self, resource_class: ResourceClass,
                 release_fraction: float = 1.0,
                 residue_gb: float = 0.0,
                 report_gb=None,
                 raise_on_release: BaseException = None,
                 witness_delta_gb=None,
                 teardown: bool = False,
                 stage_rung: Rung = None):
        self._cls = resource_class
        self._held: dict[str, float] = {}
        self.release_fraction = release_fraction
        self.residue_gb = residue_gb
        self.report_gb = report_gb
        self.raise_on_release = raise_on_release
        self.witness_delta_gb = witness_delta_gb
        self.teardown = teardown
        self.witness_calls: list[str] = []
        self.stage_rung = stage_rung
        self.released: list[str] = []
        self.measured: list[str] = []

    @property
    def resource_class(self) -> ResourceClass:
        return self._cls

    def stage(self, spec: ResourceSpec) -> Rung:
        self._held[spec.resource_id] = spec.held_gb
        return self.stage_rung or spec.held_rung

    def release(self, resource_id: str) -> float:
        self.released.append(resource_id)
        if self.raise_on_release is not None:
            raise self.raise_on_release
        gb = self._held.get(resource_id, 0.0)
        # REALITY: what the OS actually gets back.
        retained = max(gb * (1.0 - self.release_fraction), self.residue_gb)
        retained = min(retained, gb)
        freed = gb - retained
        if self.teardown:
            # The process is gone. Whatever the OS kept, this actor can no
            # longer see it -- and neither can the ledger, through this actor.
            self._held.pop(resource_id, None)
        elif retained > 0:
            self._held[resource_id] = retained
        else:
            self._held.pop(resource_id, None)
        # THE REPORT: unconstrained by the above, which is the whole point.
        if self.report_gb is None:
            return freed
        if self.report_gb == "full":
            return gb
        return float(self.report_gb)

    def release_witness(self, resource_id: str):
        """The contract's optional independent witness (added 2026-08-30).

        Returning None is honest and allowed -- most actors have no way to
        measure their enclosing allocation -- and the ledger then falls back to
        its own callable, or counts the release as unwitnessed.
        """
        self.witness_calls.append(resource_id)
        if isinstance(self.witness_delta_gb, BaseException):
            raise self.witness_delta_gb
        return self.witness_delta_gb

    def measure_held_gb(self, resource_id: str) -> float:
        self.measured.append(resource_id)
        return self._held.get(resource_id, 0.0)

    def is_resident(self, resource_id: str) -> bool:
        return resource_id in self._held

    # -- knobs the tests use to move reality out from under the ledger ----
    def set_measured(self, resource_id: str, gb: float) -> None:
        self._held[resource_id] = gb


class FakeCgroupWitness:
    """An outside reader of the enclosing allocation's footprint.

    The ledger reads it twice per release -- once before it asks the actor to
    do anything, once after -- so this returns the pre-teardown total on the
    first read and the post-teardown total on every read after that. That is
    how a real `memory.stat anon` read behaves across a teardown, and it is
    the only witness that survives the process going away.
    """

    def __init__(self, before_gb: float, after_gb: float):
        self.before_gb, self.after_gb = before_gb, after_gb
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.before_gb if self.reads == 1 else self.after_gb


def _ledger(budget_gb=400.0, **kw):
    led = ResidencyLedger(budget_gb, **kw)
    led.register_actor(FakeActor(ResourceClass.MODEL))
    led.register_actor(FakeActor(ResourceClass.DATA_PATTERN_A))
    led.register_actor(FakeActor(ResourceClass.DATA_PATTERN_B))
    return led


def _stage_and_charge(led, spec, now_s=0.0):
    actor = led.actor_for(spec.resource_class)
    rung = actor.stage(spec)
    return led.charge(spec, rung, now_s)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_implements_the_contract_protocols():
    led = _ledger()
    assert isinstance(led, Ledger)
    assert isinstance(FakeActor(ResourceClass.MODEL), ResidencyActor)


def test_budget_is_an_allocation_not_a_constant():
    """It is a constructor argument and a swept variable; nothing reads the
    machine's RAM. Two ledgers at two budgets differ only by that argument."""
    small, large = _ledger(256.0), _ledger(1024.0)
    _stage_and_charge(small, QWEN_32B)
    _stage_and_charge(large, QWEN_32B)
    assert small.budget_gb == 256.0 and large.budget_gb == 1024.0
    assert small.slack_gb == pytest.approx(256.0 - 129.7)
    assert large.slack_gb == pytest.approx(1024.0 - 129.7)
    with pytest.raises(ValueError):
        ResidencyLedger(0.0)


def test_held_and_slack_track_every_class_in_one_budget():
    led = _ledger(400.0)
    _stage_and_charge(led, QWEN_32B)
    _stage_and_charge(led, UNIREF90)
    _stage_and_charge(led, EAM_POT)
    assert led.held_gb == pytest.approx(129.7 + 117.20 + 16.93)
    assert led.slack_gb == pytest.approx(400.0 - 263.83)
    assert [e.spec.resource_id for e in led.entries()] == [
        "eam_potential", "qwen_32b", "uniref90"]          # sorted, not hash order


# --------------------------------------------------------------------------
# I1 — charges are measured
# --------------------------------------------------------------------------


def test_charge_books_the_measured_footprint_not_the_declared_one():
    led = _ledger()
    actor = led.actor_for(ResourceClass.MODEL)
    actor.stage(QWEN_32B)
    actor.set_measured("qwen_32b", 141.2)      # reality disagrees with the spec
    e = led.charge(QWEN_32B, Rung.R2_PROCESS_BYTES, 0.0)
    assert e.charged_gb == pytest.approx(141.2)
    assert led.held_gb == pytest.approx(141.2)
    assert led.declared_charges == []


def test_an_unmeasurable_charge_falls_back_to_declared_and_says_so():
    """Charging something the actor cannot measure yet is allowed -- silently
    calling it measured is not."""
    led = _ledger()
    e = led.charge(QWEN_32B, Rung.R2_PROCESS_BYTES, 0.0)   # never staged
    assert e.charged_gb == pytest.approx(129.7)
    assert led.declared_charges == ["qwen_32b"]
    assert "DECLARED, not measured" in led.audit()


def test_charge_records_the_rung_actually_reached_not_the_one_requested():
    led = _ledger()
    led.actor_for(ResourceClass.MODEL).stage_rung = Rung.R1_PAGE_CACHE
    e = _stage_and_charge(led, QWEN_32B)
    assert e.rung is Rung.R1_PAGE_CACHE
    assert e.spec.held_rung is Rung.R2_PROCESS_BYTES


def test_double_charge_and_unknown_class_are_errors():
    led = ResidencyLedger(400.0)
    with pytest.raises(KeyError):
        led.charge(QWEN_32B, Rung.R2_PROCESS_BYTES, 0.0)   # no actor registered
    led.register_actor(FakeActor(ResourceClass.MODEL))
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ValueError):
        _stage_and_charge(led, QWEN_32B)


def test_overcommit_is_refused_by_default_and_recordable_on_request():
    led = _ledger(300.0)
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(BudgetExceeded):
        _stage_and_charge(led, QWEN_72B)
    assert led.held_gb == pytest.approx(129.7)             # nothing booked

    loose = ResidencyLedger(300.0, allow_overcommit=True)
    loose.register_actor(FakeActor(ResourceClass.MODEL))
    _stage_and_charge(loose, QWEN_32B)
    _stage_and_charge(loose, QWEN_72B)
    assert loose.slack_gb < 0                              # honest, not clamped


# --------------------------------------------------------------------------
# I2 — releases are confirmed. THE POINT OF THE FILE.
# --------------------------------------------------------------------------


def test_an_honoured_release_returns_the_gb_and_frees_the_budget():
    led = _ledger()
    _stage_and_charge(led, UNIREF90)
    freed = led.release("uniref90", 10.0)
    assert freed == pytest.approx(117.20)
    assert led.held_gb == 0.0
    assert led.slack_gb == pytest.approx(400.0)
    assert led.release_shortfalls == []


def test_release_not_honoured_fires_on_an_under_releasing_actor():
    """The D2 failure, caught. A worker that gives back half the pages must not
    be able to leave the budget believing it gave back all of them."""
    led = _ledger()
    led.actor_for(ResourceClass.DATA_PATTERN_A).release_fraction = 0.5
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("uniref90", 10.0)
    msg = str(ei.value)
    assert "117.200" in msg and "58.600" in msg            # charged and freed
    assert "fiction" in msg


def test_a_release_that_returns_nothing_is_the_loudest_case():
    led = _ledger()
    led.actor_for(ResourceClass.DATA_PATTERN_B).release_fraction = 0.0
    _stage_and_charge(led, EAM_POT)
    with pytest.raises(ReleaseNotHonoured):
        led.release("eam_potential", 5.0)


def test_a_failed_release_keeps_the_charge_and_is_recorded():
    """The memory was not returned, so the charge is still TRUE. Dropping the
    entry here would convert a loud failure into a quiet budget leak."""
    led = _ledger()
    led.actor_for(ResourceClass.DATA_PATTERN_A).release_fraction = 0.25
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured):
        led.release("uniref90", 10.0)
    assert led.is_held("uniref90")
    assert led.held_gb == pytest.approx(117.20)
    assert len(led.release_shortfalls) == 1
    rec = led.release_shortfalls[0]
    assert rec.resource_id == "uniref90"
    assert rec.shortfall_gb == pytest.approx(117.20 * 0.75)
    assert "I2 VIOLATION" in led.audit()


def test_the_tolerance_is_an_argument_with_a_documented_default():
    """Both directions of the same shortfall, decided only by the argument."""
    assert DEFAULT_RELEASE_TOLERANCE_GB == 0.25

    # A 0.1 GB shortfall is inside the default and must NOT fire.
    led = _ledger()
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.stage(UNIREF50)
    led.charge(UNIREF50, Rung.R3_ACTIVATED, 0.0)
    a.set_measured("uniref50", UNIREF50.held_gb - 0.1)
    assert led.release("uniref50", 1.0) == pytest.approx(35.98)

    # The same 0.1 GB against a 0.05 GB tolerance MUST fire -- and the
    # RELATIVE part has to be switched off explicitly, because 1% of
    # 36.08 GB is 0.36 GB and would otherwise absorb it.
    strict = ResidencyLedger(400.0, release_tolerance_gb=0.05,
                             release_tolerance_frac=0.0)
    strict.register_actor(FakeActor(ResourceClass.DATA_PATTERN_A))
    b = strict.actor_for(ResourceClass.DATA_PATTERN_A)
    b.stage(UNIREF50)
    strict.charge(UNIREF50, Rung.R3_ACTIVATED, 0.0)
    b.set_measured("uniref50", UNIREF50.held_gb - 0.1)
    with pytest.raises(ReleaseNotHonoured):
        strict.release("uniref50", 1.0)

    # And a caller with a noisier probe may widen it explicitly rather than
    # edit a literal -- but widening the PER-RELEASE tolerance alone does not
    # open the leak path, because the cumulative bound is a separate argument.
    loose = ResidencyLedger(400.0, release_tolerance_gb=100.0)
    loose.register_actor(FakeActor(ResourceClass.DATA_PATTERN_A,
                                   release_fraction=0.5))
    _stage_and_charge(loose, UNIREF90)
    with pytest.raises(ReleaseNotHonoured, match="accumulated residue"):
        loose.release("uniref90", 1.0)

    wide = ResidencyLedger(400.0, release_tolerance_gb=100.0,
                           leak_budget_frac=0.25)          # 100 GB of leak
    wide.register_actor(FakeActor(ResourceClass.DATA_PATTERN_A,
                                  release_fraction=0.5))
    _stage_and_charge(wide, UNIREF90)
    assert wide.release("uniref90", 1.0) == pytest.approx(58.6)
    assert wide.cumulative_residual_gb == pytest.approx(58.6)


def test_freeing_more_than_charged_is_reported_but_is_not_an_error():
    led = _ledger()
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.stage(UNIREF50)
    led.charge(UNIREF50, Rung.R3_ACTIVATED, 0.0)
    a.set_measured("uniref50", 60.0)                # the charge was an undercount
    freed = led.release("uniref50", 1.0)
    assert freed == pytest.approx(60.0)
    assert len(led.over_releases) == 1
    assert led.release_shortfalls == []
    assert "over-release" in led.audit()


# --------------------------------------------------------------------------
# I2, ADVERSARIAL. Every one of these passed the first implementation, which
# compared charged_gb against the actor's RETURN VALUE and never measured.
# --------------------------------------------------------------------------


def test_an_honest_report_that_frees_nothing_is_caught_by_measurement():
    """THE HOLE. The actor reports the full amount in good faith and the pages
    never come back -- glibc arena retention, an engine that drops a reference
    the allocator does not return. Nothing about the return value is wrong, so
    only an independent measurement can see it."""
    led = _ledger(256.0)
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.release_fraction = 0.0            # reality: nothing comes back
    a.report_gb = "full"                # report: everything did
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("uniref90", 1.0)
    msg = str(ei.value)
    assert "MEASURED footprint fell only 0.000 GB" in msg
    assert "reported 117.200 GB" in msg
    assert led.is_held("uniref90")                       # still charged
    assert led.release_shortfalls[-1].reason == "measured-retention"
    assert "measured-retention" in led.audit()


def test_the_88_percent_overrun_cannot_happen_silently():
    """V's scenario, as a regression test: four iterations of stage/release
    against a 256 GB allocation with an honest actor that frees nothing left
    480 GB actually resident, 88% over budget, zero recorded violations and a
    clean audit(). It must now fail on the FIRST release."""
    led = _ledger(256.0)
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.release_fraction, a.report_gb = 0.0, "full"
    violations = 0
    for _ in range(4):
        try:
            if not led.is_held("uniref90"):
                _stage_and_charge(led, UNIREF90)
            led.release("uniref90", 1.0)
        except ReleaseNotHonoured:
            violations += 1
    assert violations == 4
    assert led.held_gb == pytest.approx(117.20)          # one entry, still charged
    assert led.held_gb <= led.budget_gb
    # and what is really resident is visible, not hidden by a deleted entry
    assert led.reconcile()["uniref90"] == (pytest.approx(117.20),
                                           pytest.approx(117.20))


def test_a_nan_report_is_rejected_before_it_is_compared():
    """nan compares False against every bound -- `nan > tol` and `-nan > tol`
    are both False -- so it passed the shortfall check, the over-release check
    and the tolerance, recorded nothing, and was handed to the caller."""
    led = _ledger()
    led.actor_for(ResourceClass.DATA_PATTERN_A).report_gb = float("nan")
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured, match="non-finite"):
        led.release("uniref90", 1.0)
    assert led.is_held("uniref90")
    assert led.release_shortfalls[-1].reason == "non-finite-report"

    for bogus in (float("inf"), float("-inf"), -5.0):
        l2 = _ledger()
        l2.actor_for(ResourceClass.DATA_PATTERN_A).report_gb = bogus
        _stage_and_charge(l2, UNIREF90)
        with pytest.raises(ReleaseNotHonoured):
            l2.release("uniref90", 1.0)
        assert l2.is_held("uniref90")


def test_an_actor_that_raises_mid_release_is_recorded_not_just_propagated():
    """The entry staying charged was already right. What was missing is the
    record: audit() showed nothing and the only trace was the actor's own
    exception type."""
    boom = RuntimeError("vLLM engine died during sleep")
    led = _ledger()
    led.actor_for(ResourceClass.DATA_PATTERN_A).raise_on_release = boom
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("uniref90", 1.0)
    assert ei.value.__cause__ is boom                    # the original survives
    assert "vLLM engine died" in str(ei.value)
    assert led.is_held("uniref90")
    assert led.release_shortfalls[-1].reason == "actor-raised"
    assert "actor-raised" in led.audit()


def test_a_report_larger_than_the_measured_drop_is_a_violation():
    """The actor's word is evidence, not proof: half comes back, it claims all
    of it, and the two disagreeing is itself the failure."""
    led = _ledger()
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.release_fraction, a.report_gb = 0.5, "full"
    _stage_and_charge(led, UNIREF90)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("uniref90", 1.0)
    assert led.release_shortfalls[-1].reason in (
        "measured-retention", "report-exceeds-measurement")
    assert "58.600" in str(ei.value)


def test_the_slow_drip_is_bounded_by_the_allocation_not_by_cycle_count():
    """Each release leaves 0.25 GB behind, inside any per-release tolerance.
    Over 100 cycles that leaked 25.00 GB -- 10% of a 256 GB allocation -- with
    zero exceptions. The cumulative bound has to end it, and end it at a
    fraction of the BUDGET rather than after some number of cycles."""
    led = _ledger(256.0)                        # leak budget 2% = 5.12 GB
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    a.residue_gb = 0.25                         # honest report, 0.25 GB stays
    assert led.leak_budget_gb == pytest.approx(5.12)

    cycles = 0
    with pytest.raises(ReleaseNotHonoured, match="accumulated residue"):
        for _ in range(100):
            a._held.pop("uniref50", None)       # a fresh stage each cycle
            _stage_and_charge(led, UNIREF50)
            led.release("uniref50", float(cycles))
            cycles += 1
    assert cycles == 20                         # 20 x 0.25 = 5.0, the 21st crosses
    assert led.cumulative_residual_gb == pytest.approx(5.25)
    assert led.cumulative_residual_gb < 0.10 * led.budget_gb   # never near 25 GB
    assert "cumulative unexplained residue" in led.audit()
    assert led.reset_leak_accounting() == pytest.approx(5.25)
    assert led.cumulative_residual_gb == 0.0


def test_a_teardown_actor_cannot_hide_behind_the_measured_drop():
    """A5'S FINDING. The actor kills the process, so measure_held_gb()
    afterwards is 0.0 by construction and `before - after` is trivially the
    whole charge -- it confirms nothing. The actor's own reported delta is then
    the ONLY independent evidence, and the ledger has to hold it to the charge.

    The first version of the measured-drop fix checked the report only for
    being too LARGE, so an honest short give-back sailed through.
    """
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True                    # process gone: nothing left to measure
    a.report_gb = 22.302                 # what the cgroup actually gave back
    _stage_and_charge(led, QWEN_32B)     # 129.7 GB charged
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("qwen_32b", 1.0)
    msg = str(ei.value)
    assert "the actor itself reports only 22.302 GB" in msg
    assert "129.676" in msg or "129.700" in msg
    assert led.is_held("qwen_32b")                       # still charged
    assert led.release_shortfalls[-1].reason == "report-below-charge"

    # and the tautology is real: the per-resource measurement saw a full drop
    assert a.measure_held_gb("qwen_32b") == 0.0


def test_a_teardown_actor_that_really_did_free_it_still_passes():
    """The other half: this must not reject an honest teardown, or the model
    actor can never release anything."""
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    _stage_and_charge(led, QWEN_32B)
    assert led.release("qwen_32b", 1.0) == pytest.approx(129.7)
    assert led.release_shortfalls == []
    assert led.unwitnessed_releases == 1                 # counted, not hidden
    assert "no actor-independent witness" in led.audit()


def test_an_independent_witness_catches_what_the_actor_cannot_see():
    """The witness the ledger can have that the actor cannot: an outside
    reader of the enclosing allocation. Here the actor tears down and reports
    the full charge in good faith -- its own accounting says the weights went
    -- while the allocation as a whole gave back 2 GB of the 129.7 charged."""
    w = FakeCgroupWitness(400.0, 398.0)
    led = ResidencyLedger(512.0, witness_gb=w)
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True,
                                 report_gb="full"))
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("qwen_32b", 1.0)
    assert led.release_shortfalls[-1].reason == "witness-contradicts-release"
    assert "witness (ledger-witness) fell only 2.000 GB" in str(ei.value)
    assert led.is_held("qwen_32b")
    assert w.reads == 2                                  # before AND after

    # the same release with the witness agreeing is confirmed, and is counted
    # as witnessed rather than taken on the actor's word
    ok = ResidencyLedger(512.0, witness_gb=FakeCgroupWitness(400.0, 400.0 - 129.7))
    ok.register_actor(FakeActor(ResourceClass.MODEL, teardown=True,
                                report_gb="full"))
    _stage_and_charge(ok, QWEN_32B)
    assert ok.release("qwen_32b", 1.0) == pytest.approx(129.7, rel=1e-6)
    assert ok.unwitnessed_releases == 0
    assert ok.release_log[-1]["witnesses"] == (
        "actor-report", "resource-measurement", "independent-witness")


def test_the_actors_own_witness_is_asked_first():
    """`release_witness()` is the primary source. A witness that has to be
    wired in at the call site is the one that goes missing in production --
    `unwitnessed_releases` would count up silently and nobody would look -- so
    the witness travels with the actor that knows how to take it."""
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    a.witness_delta_gb = 129.7                  # the cgroup agrees
    _stage_and_charge(led, QWEN_32B)
    proved = led.release("qwen_32b", 1.0)
    assert proved == pytest.approx(129.7)
    assert a.witness_calls == ["qwen_32b"]
    rec = led.release_log[-1]
    assert rec["witness_source"] == "actor-witness"
    assert rec["witness_drop_gb"] == pytest.approx(129.7)
    assert led.unwitnessed_releases == 0        # no callable was needed


def test_the_actors_witness_catches_a_teardown_leak_with_no_wiring_at_all():
    """The production case: no witness_gb configured anywhere, and the leak is
    still caught, because the actor brought its own witness."""
    led = _ledger(512.0)                        # note: no witness_gb
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    a.report_gb = "full"                        # honest by its own accounting
    a.witness_delta_gb = 22.302                 # what the cgroup really gave back
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("qwen_32b", 1.0)
    assert led.release_shortfalls[-1].reason == "witness-contradicts-release"
    assert "(actor-witness)" in str(ei.value)
    assert led.is_held("qwen_32b")


def test_the_callable_is_the_fallback_when_the_actor_has_no_witness():
    """Returning None is honest and allowed; existing wiring keeps working."""
    led = ResidencyLedger(512.0, witness_gb=FakeCgroupWitness(400.0, 270.3))
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True))
    a = led.actor_for(ResourceClass.MODEL)
    assert a.release_witness("qwen_32b") is None       # no witness of its own
    _stage_and_charge(led, QWEN_32B)
    assert led.release("qwen_32b", 1.0) == pytest.approx(129.7)
    assert led.release_log[-1]["witness_source"] == "ledger-witness"
    assert led.unwitnessed_releases == 0


def test_the_actors_witness_takes_precedence_over_the_callable():
    led = ResidencyLedger(512.0, witness_gb=FakeCgroupWitness(400.0, 270.3))
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True,
                                 witness_delta_gb=22.302))
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured) as ei:
        led.release("qwen_32b", 1.0)
    # the callable would have vouched for the full 129.7; the actor's own
    # witness is the one that governs, and it does not
    assert "(actor-witness)" in str(ei.value)
    assert led.release_shortfalls[-1].reason == "witness-contradicts-release"


def test_a_witness_that_raises_is_recorded_and_does_not_block_the_release():
    """A broken witness must not fail a release that is otherwise confirmable
    -- but it must not vanish either, or the ledger goes back to believing a
    tautology without saying so."""
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    a.witness_delta_gb = OSError("cgroup memory.stat unreadable")
    _stage_and_charge(led, QWEN_32B)
    assert led.release("qwen_32b", 1.0) == pytest.approx(129.7)
    assert led.release_shortfalls == []                 # not fatal
    assert led.unwitnessed_releases == 1                # but not witnessed
    assert led.witness_errors == [("qwen_32b", "OSError: cgroup memory.stat unreadable")]
    assert "release_witness() FAILED for qwen_32b" in led.audit()
    assert led.release_log[-1]["witness_source"] is None


def test_a_non_finite_witness_is_rejected_like_every_other_nan():
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    a.witness_delta_gb = float("nan")
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured, match="non-finite"):
        led.release("qwen_32b", 1.0)
    assert led.release_shortfalls[-1].reason == "non-finite-witness"
    assert led.is_held("qwen_32b")


def test_an_actor_witness_satisfies_the_requirement_with_no_callable():
    led = ResidencyLedger(512.0, require_independent_witness=True)
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True,
                                 witness_delta_gb=129.7))
    _stage_and_charge(led, QWEN_32B)
    assert led.release("qwen_32b", 1.0) == pytest.approx(129.7)

    # ...and an actor that HAS the method but returns None does not satisfy it
    strict = ResidencyLedger(512.0, require_independent_witness=True)
    strict.register_actor(FakeActor(ResourceClass.MODEL, teardown=True))
    _stage_and_charge(strict, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured, match="witness is required"):
        strict.release("qwen_32b", 1.0)
    assert strict.release_shortfalls[-1].reason == "no-independent-witness"


def test_an_actor_predating_the_witness_method_still_works():
    """`release_witness` was added to the contract on 2026-08-30. An actor
    written before it -- and there are two in this tree -- must keep working:
    the ledger probes for the method rather than assuming it, falls back to
    the callable, and counts the release as unwitnessed when there is neither.

    NOTE the isinstance consequence, which is real and not this test's fault:
    ResidencyActor is runtime_checkable, so isinstance() now returns False for
    an actor lacking the method even though the docstring calls it optional.
    The LEDGER does not use isinstance, so such an actor still releases.
    """
    class LegacyActor:                       # no release_witness at all
        resource_class = ResourceClass.MODEL

        def __init__(self):
            self.held = {}

        def stage(self, spec):
            self.held[spec.resource_id] = spec.held_gb
            return spec.held_rung

        def release(self, resource_id):
            return self.held.pop(resource_id, 0.0)

        def measure_held_gb(self, resource_id):
            return self.held.get(resource_id, 0.0)

        def is_resident(self, resource_id):
            return resource_id in self.held

    legacy = LegacyActor()
    assert not hasattr(legacy, "release_witness")
    assert not isinstance(legacy, ResidencyActor)     # the isinstance cost
    led = ResidencyLedger(512.0)
    led.register_actor(legacy)
    legacy.stage(QWEN_32B)
    led.charge(QWEN_32B, Rung.R2_PROCESS_BYTES, 0.0)
    assert led.release("qwen_32b", 1.0) == pytest.approx(129.7)   # still works
    assert led.unwitnessed_releases == 1
    assert led.release_log[-1]["witness_source"] is None

    # and the fallback callable still reaches such an actor
    l2 = ResidencyLedger(512.0, witness_gb=FakeCgroupWitness(400.0, 270.3))
    l2.register_actor(LegacyActor())
    l2.actor_for(ResourceClass.MODEL).stage(QWEN_32B)
    l2.charge(QWEN_32B, Rung.R2_PROCESS_BYTES, 0.0)
    assert l2.release("qwen_32b", 1.0) == pytest.approx(129.7)
    assert l2.release_log[-1]["witness_source"] == "ledger-witness"


def test_requiring_a_witness_refuses_to_confirm_without_one():
    """Off by default -- an in-process actor is legitimately confirmable
    without it -- but a campaign whose numbers depend on the budget being real
    can demand it."""
    led = ResidencyLedger(512.0, require_independent_witness=True)
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True))
    _stage_and_charge(led, QWEN_32B)
    with pytest.raises(ReleaseNotHonoured, match="independent witness is required"):
        led.release("qwen_32b", 1.0)
    assert led.is_held("qwen_32b")
    assert led.release_shortfalls[-1].reason == "no-independent-witness"


def test_the_weakest_witness_is_what_gets_returned():
    """`release()` returns what can be PROVED came back, not the largest
    number available. Rounding up to the charge is how a budget becomes
    fiction."""
    led = ResidencyLedger(512.0, witness_gb=FakeCgroupWitness(400.0, 274.0),
                          release_tolerance_frac=0.0, release_tolerance_gb=5.0)
    led.register_actor(FakeActor(ResourceClass.MODEL, teardown=True,
                                 report_gb=127.0))       # actor says 127.0
    _stage_and_charge(led, QWEN_32B)                     # charged 129.7
    proved = led.release("qwen_32b", 1.0)                # witness says 126.0
    assert proved == pytest.approx(126.0)                # the weakest one
    rec = led.release_log[-1]
    assert rec["reported_gb"] == pytest.approx(127.0)
    assert rec["measured_drop_gb"] == pytest.approx(129.7)   # the tautology
    assert rec["witness_drop_gb"] == pytest.approx(126.0)
    assert led.cumulative_residual_gb == pytest.approx(3.7)


def test_a_teardown_that_returns_more_than_charged_is_benign():
    """A teardown gives back the parked weights AND the engine's own baseline
    (interpreter, CUDA context, allocator arenas), and only the weights were
    charged. A5's actor subtracts the baseline, but when the baseline is
    unknown it returns the whole delta rather than guessing. That legitimately
    exceeds the charge: the budget is safe in that direction, so it is
    recorded, not raised."""
    led = _ledger(512.0)
    a = led.actor_for(ResourceClass.MODEL)
    a.teardown = True
    a.report_gb = 137.7                  # weights 129.7 + an 8 GB baseline
    _stage_and_charge(led, QWEN_32B)
    proved = led.release("qwen_32b", 1.0)
    assert proved == pytest.approx(129.7)              # never more than measured
    assert led.release_shortfalls == []                 # NOT an error
    reasons = {r.reason for r in led.over_releases}
    assert "over-release" in reasons or "report-exceeds-measurement" in reasons
    assert led.cumulative_residual_gb == 0.0


def test_a_clean_release_measures_before_and_after():
    """The confirmation is not optional bookkeeping: measure_held_gb must
    actually be consulted on the release path."""
    led = _ledger()
    a = led.actor_for(ResourceClass.DATA_PATTERN_A)
    _stage_and_charge(led, UNIREF90)
    a.measured.clear()
    led.release("uniref90", 1.0)
    assert a.measured.count("uniref90") >= 2             # before and after
    assert led.cumulative_residual_gb == 0.0


def test_the_ledger_serialises_its_own_mutations():
    """Not a proof of thread safety -- it is a statement that the assumption is
    explicit. The lock covers ledger state only; actors are their own problem.
    """
    import threading as _t
    led = _ledger(4000.0)
    assert isinstance(led._lock, type(_t.RLock()))
    errors = []

    def churn(spec):
        try:
            for _ in range(50):
                _stage_and_charge(led, spec)
                led.note_use(spec.resource_id, 1.0)
                led.reconcile()
                led.release(spec.resource_id, 2.0)
        except Exception as exc:                          # noqa: BLE001
            errors.append(exc)

    ts = [_t.Thread(target=churn, args=(sp,))
          for sp in (QWEN_32B, UNIREF90, EAM_POT)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    assert led.held_gb == 0.0
    assert led.release_shortfalls == []


def test_release_routes_to_the_actor_that_owns_the_class():
    led = _ledger()
    _stage_and_charge(led, QWEN_32B)
    _stage_and_charge(led, UNIREF90)
    led.release("uniref90", 1.0)
    assert led.actor_for(ResourceClass.DATA_PATTERN_A).released == ["uniref90"]
    assert led.actor_for(ResourceClass.MODEL).released == []
    with pytest.raises(KeyError):
        led.release("nothing_here", 1.0)
    with pytest.raises(KeyError):
        led.release("uniref90", 2.0)          # already released: no second charge


def test_two_actors_cannot_own_one_class():
    led = ResidencyLedger(400.0)
    led.register_actor(FakeActor(ResourceClass.MODEL))
    with pytest.raises(ValueError):
        led.register_actor(FakeActor(ResourceClass.MODEL))


# --------------------------------------------------------------------------
# reconcile — report drift, never correct it
# --------------------------------------------------------------------------


def test_reconcile_surfaces_drift_and_does_not_silently_correct_it():
    led = _ledger()
    _stage_and_charge(led, QWEN_32B)
    _stage_and_charge(led, UNIREF90)
    assert led.reconcile() == {
        "qwen_32b": (pytest.approx(129.7), pytest.approx(129.7)),
        "uniref90": (pytest.approx(117.20), pytest.approx(117.20)),
    }
    assert led.drift() == {}

    # reality moves under the ledger: the worker grew by 8 GB.
    led.actor_for(ResourceClass.DATA_PATTERN_A).set_measured("uniref90", 125.2)
    charged, measured = led.reconcile()["uniref90"]
    assert charged == pytest.approx(117.20)      # NOT rewritten to 125.2
    assert measured == pytest.approx(125.2)
    assert "uniref90" in led.drift() and "qwen_32b" not in led.drift()
    assert led.held_gb == pytest.approx(129.7 + 117.20)
    assert "DRIFT" in led.audit()


def test_drift_tolerance_is_also_an_argument():
    led = _ledger(drift_tolerance_gb=10.0)
    _stage_and_charge(led, UNIREF90)
    led.actor_for(ResourceClass.DATA_PATTERN_A).set_measured("uniref90", 125.2)
    assert led.drift() == {}                     # 8.0 GB is inside 10.0
    assert led.reconcile()["uniref90"][1] == pytest.approx(125.2)


def test_note_use_tracks_recency_and_count():
    led = _ledger()
    _stage_and_charge(led, UNIREF90, now_s=5.0)
    led.note_use("uniref90", 12.0)
    led.note_use("uniref90", 40.0)
    e = led.entry("uniref90")
    assert isinstance(e, LedgerEntry)
    assert e.staged_at_s == 5.0 and e.last_use_s == 40.0 and e.use_count == 2
    with pytest.raises(KeyError):
        led.note_use("uniref50", 1.0)
