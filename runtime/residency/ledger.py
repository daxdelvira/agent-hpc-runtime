"""T1 — the residency ledger. One budget over every held class.

WHAT THIS IS. A single account of everything Tandem is holding resident, in
GB, against one allocation. `runtime/prefetch/scheduler.py` has no budget, no
ledger and no eviction; this is the thing that gives the arbitrator something
to arbitrate over.

THREE PROPERTIES THAT ARE NOT DECORATION.

1. The budget is an ALLOCATION, not a hardware constant. It is the cgroup
   limit of the job and it is a swept variable in the evaluation, so it is a
   constructor argument and nothing here may read a machine's total RAM.

2. Charges are MEASURED (invariant I1). `charge()` asks the owning actor for
   `measure_held_gb()` and books that number. `ResourceSpec.held_gb` is used
   only as a fallback when the actor cannot measure yet, and every such charge
   is recorded in `declared_charges` so a run can be audited for them. A budget
   over numbers a caller asserted is a fiction and every percentage computed
   against it is meaningless.

3. Releases are CONFIRMED BY MEASUREMENT (invariant I2). `release()` measures
   the footprint, calls the actor, measures again, and requires the MEASURED
   footprint to have fallen by what was charged. The actor's return value is
   evidence, not proof.

   THE FIRST VERSION OF THIS FILE GOT THAT WRONG and the hole is worth keeping
   in front of whoever edits it next (found by V, 2026-08-30). It compared
   `charged_gb` against the float the actor RETURNED and never called
   `measure_held_gb()`. An actor that reports honestly and frees nothing
   therefore passed cleanly, and because the entry is deleted on the success
   path, `reconcile()` could never see the leak afterwards: four iterations
   against a 256 GB allocation left 480 GB actually resident -- 88% over budget
   -- with zero recorded violations and a clean `audit()`. This needs no
   malice. A2 measured glibc arena retention freeing 0.084 of 0.364 GB while an
   honest actor would have reported the full 0.364. That is D2 exactly, the
   failure this design exists to prevent.

   AND A SECOND HOLE, IN THE FIX ITSELF (found by A5, 2026-08-30). Confirming
   with `measure_held_gb()` before and after is not enough on its own, because
   for a TEARDOWN actor -- one that kills the process rather than freeing
   inside it -- the post-release measurement is 0.0 BY CONSTRUCTION: there is
   no process left to read. `before - after` is then trivially the whole
   charge and confirms nothing. Worse, the ledger cannot tell that case apart
   from an honest in-process release, which also measures 0 afterwards.

   So the rule is not "the measurement is the witness". It is: EVERY witness
   available must independently clear the charge, and the WEAKEST one governs.
   There are up to three:
     1. the actor's returned delta          -- always present;
     2. the per-resource measured drop      -- present, but tautological for a
                                               teardown, and indistinguishable
                                               from an honest release;
     3. an actor-independent witness        -- the enclosing allocation's
                                               give-back, asked of the ACTOR
                                               first (`release_witness()`,
                                               added to the contract on
                                               2026-08-30) and falling back to
                                               a caller-supplied `witness_gb`
                                               callable. The ledger never
                                               learns what either one reads,
                                               which keeps this class-blind.

   THE ACTOR IS ASKED FIRST, AND THAT ORDERING IS THE POINT. A safety
   mechanism that only works when production wiring remembers to pass a
   callable is inert in production: `unwitnessed_releases` would count up
   silently and nobody would look. An actor knows how to measure its own
   enclosing allocation -- a cgroup for a vLLM engine, a process tree for a
   data worker -- so the witness travels with the thing that knows how to take
   it. The constructor callable stays as a fallback for actors that have no
   witness of their own, and for wiring that predates the method.
   The first version of the fix checked (2) against the charge and (1) only
   for being too LARGE, so an actor that honestly reported a SHORT give-back
   sailed through. That is A5's `VllmModelActor` exactly: it tears the engine
   down and returns the cgroup delta, which is the only real evidence there is.

   Three further requirements come from the same probe set:
     * non-finite returns are rejected. `nan` compares False against every
       bound, so it sailed through the shortfall check, the over-release check
       and the tolerance, recorded nothing, and was handed back to the caller.
     * when the actor RAISES, the entry stays charged AND the failure is
       recorded, so `audit()` shows it instead of showing nothing.
     * the tolerance is not purely absolute, and there is a cumulative bound.
       An actor freeing exactly `charged - 0.25` every cycle leaked 25.00 GB
       over 100 cycles -- 10% of a 256 GB allocation -- with 0 exceptions.

WHAT A FAILED RELEASE DOES TO THE ENTRY. Nothing. The entry stays charged. The
memory demonstrably was not returned, so dropping the charge would be the
ledger lying to itself in exactly the direction that makes the budget fiction.
The exception is the event; the retained charge is the truthful state.

THREAD SAFETY, STATED RATHER THAN ASSUMED. This ledger is designed for a
single arbitration loop. `charge()` reads `held_gb` and writes `_entries`
non-atomically, and `release()` now performs TWO blocking `measure_held_gb()`
calls (a cgroup or smaps_rollup read on a real actor), which widens that
window considerably compared with the first version. An `RLock` serialises
every mutation and every reconcile below, which is enough to keep the ledger's
own state consistent under threads. It does NOT make an actor thread-safe, and
it does not stop a caller from staging behind the ledger's back: the invariant
"nothing is resident that was not charged" is the caller's to keep.

This module is class-blind in the same sense as the arbitrator: it routes by
`ResourceClass` to a registered actor and never learns what the actor does.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from runtime.residency.contract import (
    LedgerEntry,
    ReleaseNotHonoured,
    ResidencyActor,
    ResourceClass,
    ResourceSpec,
    Rung,
)

# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------

DEFAULT_RELEASE_TOLERANCE_GB = 0.25
"""Absolute floor of the release tolerance, in GB.

Chosen, not tuned. It has to sit above the noise of a real measurement --
allocator arenas, glibc trim behaviour, and the page granularity of a
smaps_rollup read all move a footprint by tens to a couple of hundred MB
between two reads of the same idle process -- and well below any real partial
release. The smallest resource in the measured catalogue is the 16.93 GB EAM
potential, so 0.25 GB is 1.5% of the cheapest thing we can hold.

It is a floor, not the whole tolerance: see DEFAULT_RELEASE_TOLERANCE_FRAC.
"""

DEFAULT_RELEASE_TOLERANCE_FRAC = 0.01
"""Relative part of the release tolerance, as a fraction of what was charged.

The effective tolerance is max(absolute, frac * charged_gb), because a purely
absolute bound does not scale with the thing being measured: 0.25 GB is 1.5%
of the EAM potential but 0.09% of a 279 GB park, so one number cannot be both
above measurement noise on the large resource and tight on the small one.
1% of the charge is comfortably inside the arena/page noise A2 measured and far
below any partial release.
"""

DEFAULT_LEAK_BUDGET_FRAC = 0.02
"""Cumulative unexplained residue allowed, as a fraction of the ALLOCATION.

The per-release tolerance is necessarily non-zero, which by itself lets a slow
drip run forever: an actor freeing exactly `charged - 0.25` every cycle stays
inside any fixed per-release bound and leaked 25.00 GB over 100 cycles, 10% of
a 256 GB allocation, with zero exceptions raised (V, 2026-08-30). So every
release's residue -- charged minus the MEASURED drop, when positive -- is also
accumulated, and crossing this fraction of the budget raises. That bounds the
undetectable leak by the allocation (5.12 GB on 256 GB) instead of by the
number of cycles, which is the property the absolute tolerance cannot give.

Reset deliberately with `reset_leak_accounting()` after a real reconciliation,
never to quiet a failure.
"""

DEFAULT_DRIFT_TOLERANCE_GB = 0.25
"""How far charged may sit from measured before `drift()` reports the entry.

Same reasoning as the release tolerance floor. `reconcile()` itself reports
every entry unconditionally; this only decides what `drift()` flags.
"""


class BudgetExceeded(RuntimeError):
    """Raised when a charge would take held_gb past the allocation.

    Not part of the contract's exception set, and deliberately separate from
    ReleaseNotHonoured: this is a caller error (the arbitrator was supposed to
    make room first), not a broken actor. Pass allow_overcommit=True to record
    the overcommit instead of refusing it -- the ledger will then report a
    negative slack, which is the honest thing for it to say.
    """


@dataclass(frozen=True)
class ReleaseShortfall:
    """One I2 event, kept for the run report.

    `freed_gb` is the MEASURED drop where one exists; for an actor that raised
    or returned a non-finite number there is no measured drop to speak of and
    it is 0.0, with `reason` carrying what happened. `reported_gb` is what the
    actor claimed, kept beside it precisely because the two disagreeing IS the
    failure mode.
    """

    resource_id: str
    charged_gb: float
    freed_gb: float
    now_s: float
    reason: str = "under-release"
    reported_gb: float = float("nan")

    @property
    def shortfall_gb(self) -> float:
        return self.charged_gb - self.freed_gb


class ResidencyLedger:
    """The one budget. Implements the `Ledger` protocol from contract.py."""

    def __init__(
        self,
        budget_gb: float,
        release_tolerance_gb: float = DEFAULT_RELEASE_TOLERANCE_GB,
        release_tolerance_frac: float = DEFAULT_RELEASE_TOLERANCE_FRAC,
        leak_budget_frac: float = DEFAULT_LEAK_BUDGET_FRAC,
        drift_tolerance_gb: float = DEFAULT_DRIFT_TOLERANCE_GB,
        allow_overcommit: bool = False,
        witness_gb: Optional[Callable[[], float]] = None,
        require_independent_witness: bool = False,
    ) -> None:
        """witness_gb: a FALLBACK actor-independent reader of the enclosing
        allocation's resident footprint, in GB, sampled by the ledger either
        side of a release. The obvious one is the job cgroup's `anon` total,
        but the ledger does not know or care what it reads -- it is a callable
        the caller supplies, which is what keeps this class-blind (I4).

        THE ACTOR IS ASKED FIRST. `ResidencyActor.release_witness()` is the
        primary source; this callable is used only when the actor does not
        implement it or returns None. Wiring a witness in by hand at the call
        site is the arrangement that goes missing in production, which is
        precisely when it is needed.

        require_independent_witness: refuse to confirm any release that has no
        such witness, from either source. Off by default because an actor that
        measures in-process is legitimately confirmable without one, and
        turning it on where nothing provides a witness would fail every
        release. Turn it on for a campaign whose numbers depend on the budget
        being real.
        """
        if budget_gb <= 0:
            raise ValueError("budget_gb must be > 0 -- it is an allocation")
        for nm, v in (("release_tolerance_gb", release_tolerance_gb),
                      ("release_tolerance_frac", release_tolerance_frac),
                      ("leak_budget_frac", leak_budget_frac),
                      ("drift_tolerance_gb", drift_tolerance_gb)):
            if v < 0:
                raise ValueError(f"{nm} must be >= 0")
        self._budget_gb = float(budget_gb)
        self.release_tolerance_gb = float(release_tolerance_gb)
        self.release_tolerance_frac = float(release_tolerance_frac)
        self.leak_budget_frac = float(leak_budget_frac)
        self.drift_tolerance_gb = float(drift_tolerance_gb)
        self.allow_overcommit = bool(allow_overcommit)
        self.witness_gb = witness_gb
        self.require_independent_witness = bool(require_independent_witness)

        self._lock = threading.RLock()   # see THREAD SAFETY in the module docstring
        self._entries: Dict[str, LedgerEntry] = {}
        self._actors: Dict[ResourceClass, ResidencyActor] = {}

        # Audit trails. All three exist because a number that came out of a
        # campaign has to be traceable to the decisions and measurements
        # behind it.
        self.declared_charges: List[str] = []      # charged from spec, not measured
        self.release_shortfalls: List[ReleaseShortfall] = []
        self.over_releases: List[ReleaseShortfall] = []   # freed MORE than charged
        # Sum of the positive residues left behind by releases that each passed
        # the per-release tolerance. This is the drip counter (I2).
        self.cumulative_residual_gb: float = 0.0
        # Releases confirmed WITHOUT an actor-independent witness. Not an
        # error by itself; it is the count of times the budget was taken on
        # evidence the ledger cannot tell apart from a teardown's tautology.
        self.unwitnessed_releases: int = 0
        # An actor's release_witness() that RAISED. Not fatal -- a broken
        # witness must not block an otherwise confirmable release -- but it is
        # a witness that silently stopped working, so it is published.
        self.witness_errors: List[Tuple[str, str]] = []
        self.release_log: List[dict] = []

    # -- tolerances --------------------------------------------------------

    def release_tolerance_for(self, charged_gb: float) -> float:
        """max(absolute, relative) -- see the two DEFAULT_RELEASE_TOLERANCE_*
        docstrings. Exposed so a test or a report can state the bound it was
        actually judged against instead of re-deriving it."""
        return max(self.release_tolerance_gb,
                   self.release_tolerance_frac * abs(charged_gb))

    @property
    def leak_budget_gb(self) -> float:
        """The cumulative residue allowed before the drip becomes an error."""
        return self.leak_budget_frac * self._budget_gb

    def reset_leak_accounting(self) -> float:
        """Clear the drip counter after a real reconciliation. Returns what was
        cleared. Never call this to quiet a failure -- the counter existing is
        the only thing standing between a per-release tolerance and an
        unbounded leak."""
        with self._lock:
            was = self.cumulative_residual_gb
            self.cumulative_residual_gb = 0.0
            return was

    # -- actors ------------------------------------------------------------

    def register_actor(self, actor: ResidencyActor) -> None:
        """Register the actor that owns one ResourceClass.

        Routing is by class and nothing else; the ledger never learns whether
        the actor behind MODEL is vLLM or a stub.
        """
        cls = actor.resource_class
        if cls in self._actors and self._actors[cls] is not actor:
            raise ValueError(
                f"an actor is already registered for {cls}; two actors for one "
                f"class means release() has no unambiguous owner"
            )
        self._actors[cls] = actor

    def actor_for(self, resource_class: ResourceClass) -> ResidencyActor:
        try:
            return self._actors[resource_class]
        except KeyError:
            raise KeyError(
                f"no residency actor registered for {resource_class}; the "
                f"ledger cannot charge what it cannot later release or measure"
            ) from None

    # -- budget ------------------------------------------------------------

    @property
    def budget_gb(self) -> float:
        return self._budget_gb

    @property
    def held_gb(self) -> float:
        return sum(e.charged_gb for e in self._entries.values())

    @property
    def slack_gb(self) -> float:
        """budget - held. May be NEGATIVE, and that is deliberate.

        A clamp at zero would hide an overcommit from the only component that
        would notice it.
        """
        return self._budget_gb - self.held_gb

    def entries(self) -> Sequence[LedgerEntry]:
        return tuple(self._entries[k] for k in sorted(self._entries))

    def entry(self, resource_id: str) -> Optional[LedgerEntry]:
        return self._entries.get(resource_id)

    def is_held(self, resource_id: str) -> bool:
        return resource_id in self._entries

    # -- charging ----------------------------------------------------------

    def charge(self, spec: ResourceSpec, rung: Rung, now_s: float) -> LedgerEntry:
        """Book a resource as held. The charged GB is MEASURED (I1).

        `rung` is the rung the actor actually reached, which may be lower than
        spec.held_rung -- the caller passes what `stage()` returned, not what
        it asked for.
        """
        with self._lock:
            return self._charge_locked(spec, rung, now_s)

    def _charge_locked(self, spec: ResourceSpec, rung: Rung,
                       now_s: float) -> LedgerEntry:
        if spec.resource_id in self._entries:
            raise ValueError(
                f"{spec.resource_id} is already charged; note_use() a held "
                f"resource, do not charge it twice"
            )
        actor = self.actor_for(spec.resource_class)

        measured = float(actor.measure_held_gb(spec.resource_id))
        if not math.isfinite(measured):
            raise ValueError(
                f"{spec.resource_id}: measure_held_gb returned a non-finite "
                f"{measured!r}; a budget cannot be kept in nan"
            )
        if measured > 0:
            charged = measured
        else:
            # The actor cannot measure it yet (or claims nothing is resident).
            # Fall back to the declared footprint so the budget is not silently
            # free, and record it: a run full of declared charges has no I1.
            charged = float(spec.held_gb)
            self.declared_charges.append(spec.resource_id)

        prospective = self.held_gb + charged
        if prospective > self._budget_gb and not self.allow_overcommit:
            raise BudgetExceeded(
                f"charging {spec.resource_id} at {charged:.2f} GB would take "
                f"held to {prospective:.2f} GB against a {self._budget_gb:.2f} "
                f"GB allocation; the arbitrator is expected to free room first"
            )

        e = LedgerEntry(
            spec=spec,
            rung=rung,
            charged_gb=charged,
            staged_at_s=float(now_s),
            last_use_s=float(now_s),
            use_count=0,
            actor_class=spec.resource_class,
        )
        self._entries[spec.resource_id] = e
        return e

    def note_use(self, resource_id: str, now_s: float) -> None:
        with self._lock:
            e = self._entries.get(resource_id)
            if e is None:
                raise KeyError(f"note_use on unheld resource {resource_id!r}")
            e.last_use_s = float(now_s)
            e.use_count += 1

    # -- releasing (INVARIANT I2) ------------------------------------------

    def _fail(self, rec: ReleaseShortfall, msg: str, cause: BaseException = None):
        """Record an I2 event and raise. The entry is NEVER dropped on this
        path: the memory was not returned, so the charge is still true, and
        tidying up here would convert a loud failure into a quiet budget leak.
        """
        self.release_shortfalls.append(rec)
        err = ReleaseNotHonoured(
            f"{rec.resource_id}: {msg}. The budget is fiction from here: every "
            f"percentage computed against it after an unconfirmed release is "
            f"meaningless. Entry kept charged."
        )
        if cause is not None:
            raise err from cause
        raise err

    def release(self, resource_id: str, now_s: float) -> float:
        """Release through the owning actor and CONFIRM IT, weakest witness first.

        Returns the SMALLEST amount any witness can vouch for -- the most that
        can actually be proved came back. Rounding that up to what was charged
        is precisely how a budget becomes fiction.

        Up to three witnesses (see the module docstring):
          1. the actor's returned delta;
          2. the per-resource measured drop, which is TAUTOLOGICAL for a
             teardown actor and which the ledger cannot distinguish from an
             honest in-process release;
          3. `witness_gb`, if the caller configured one -- the only one that
             survives a teardown.
        Every witness present must clear `charged - tolerance`. The first
        version of this method checked (2) against the charge and (1) only for
        being too large, which let an honest short give-back through (A5).

        Raises ReleaseNotHonoured when the actor raises, when the report is
        non-finite or negative, when ANY witness falls short, when an
        independent witness is required and absent, or when the accumulated
        residue crosses the leak budget.
        """
        with self._lock:
            e = self._entries.get(resource_id)
            if e is None:
                raise KeyError(f"release of unheld resource {resource_id!r}")
            actor = self.actor_for(e.actor_class or e.spec.resource_class)
            charged = e.charged_gb
            tol = self.release_tolerance_for(charged)
            now = float(now_s)

            actor_can_witness = callable(getattr(actor, "release_witness", None))
            if (self.require_independent_witness and self.witness_gb is None
                    and not actor_can_witness):
                self._fail(
                    ReleaseShortfall(resource_id, charged, 0.0, now,
                                     "no-independent-witness", float("nan")),
                    "an actor-independent witness is required, the actor does "
                    "not implement release_witness() and no witness_gb "
                    "callable is configured; a teardown actor's post-release "
                    "measurement is 0.0 by construction and proves nothing")

            # (0) INDEPENDENT readings before the actor is asked to do
            #     anything, so a drop is measured rather than inferred.
            before = float(actor.measure_held_gb(resource_id))
            w_before = None if self.witness_gb is None else float(self.witness_gb())

            # (1) the actor may fail outright. Record it; do not let the
            #     actor's own exception type be the only trace.
            try:
                reported = float(actor.release(resource_id))
            except Exception as exc:                       # noqa: BLE001
                self._fail(
                    ReleaseShortfall(resource_id, charged, 0.0, now,
                                     "actor-raised", float("nan")),
                    f"release() raised {type(exc).__name__}: {exc}; "
                    f"{charged:.3f} GB stays charged", cause=exc)

            # (2) nan defeats every bound it is compared against -- `nan > tol`
            #     and `-nan > tol` are both False -- so it must be rejected
            #     before any comparison, not by one.
            if not math.isfinite(reported):
                self._fail(
                    ReleaseShortfall(resource_id, charged, 0.0, now,
                                     "non-finite-report", reported),
                    f"release() returned a non-finite {reported!r} against "
                    f"{charged:.3f} GB charged")
            if reported < 0:
                self._fail(
                    ReleaseShortfall(resource_id, charged, 0.0, now,
                                     "negative-report", reported),
                    f"release() returned a negative {reported:.3f} GB")

            # (3) measure again, independently, and take the outside witness.
            after = float(actor.measure_held_gb(resource_id))
            w_after = None if self.witness_gb is None else float(self.witness_gb())
            for label, v in (("before", before), ("after", after),
                             ("witness-before", w_before), ("witness-after", w_after)):
                if v is not None and not math.isfinite(v):
                    self._fail(
                        ReleaseShortfall(resource_id, charged, 0.0, now,
                                         "non-finite-measure", reported),
                        f"the {label} measurement was non-finite ({v!r})")
            measured_drop = before - after

            # THE INDEPENDENT WITNESS. The actor is asked first, because it is
            # the thing that knows how to measure its own enclosing allocation
            # (a cgroup for an engine, a process tree for a data worker) and
            # because a witness that must be wired in at the call site is the
            # one that goes missing in production. The callable is the
            # fallback, for actors with no witness of their own and for wiring
            # that predates the method.
            witness_drop = None
            witness_source = None
            witness_error = None
            if actor_can_witness:
                try:
                    w = actor.release_witness(resource_id)
                except Exception as exc:                   # noqa: BLE001
                    # A broken witness must not block a release that is
                    # otherwise confirmable -- but it must not vanish either.
                    witness_error = f"{type(exc).__name__}: {exc}"
                else:
                    if w is not None:
                        w = float(w)
                        if not math.isfinite(w):
                            self._fail(
                                ReleaseShortfall(resource_id, charged, 0.0, now,
                                                 "non-finite-witness", reported),
                                f"release_witness() returned a non-finite "
                                f"{w!r}; nan compares False against every "
                                f"bound and would confirm anything")
                        witness_drop, witness_source = w, "actor-witness"
            if witness_drop is None and w_before is not None:
                witness_drop = w_before - w_after
                witness_source = "ledger-witness"

            if self.require_independent_witness and witness_drop is None:
                self._fail(
                    ReleaseShortfall(resource_id, charged, 0.0, now,
                                     "no-independent-witness", reported),
                    f"an actor-independent witness is required and none was "
                    f"available (actor witness: "
                    f"{witness_error or 'returned None' if actor_can_witness else 'not implemented'})")

            # (4) THE RULE: every witness present must clear the charge, and
            #     the weakest one governs. Ordered so the reason names the
            #     witness that actually failed.
            floor = charged - tol
            if reported < floor:
                self._fail(
                    ReleaseShortfall(resource_id, charged, reported, now,
                                     "report-below-charge", reported),
                    f"ledger charged {charged:.3f} GB and the actor itself "
                    f"reports only {reported:.3f} GB came back (tolerance "
                    f"{tol:.3f} GB). For a teardown actor that delta is the "
                    f"only independent evidence there is, and the "
                    f"per-resource measurement ({before:.3f} -> {after:.3f}) "
                    f"cannot contradict it: it reads 0 once the process is gone")
            if measured_drop < floor:
                self._fail(
                    ReleaseShortfall(resource_id, charged, measured_drop, now,
                                     "measured-retention", reported),
                    f"ledger charged {charged:.3f} GB and release() reported "
                    f"{reported:.3f} GB, but the MEASURED footprint fell only "
                    f"{measured_drop:.3f} GB ({before:.3f} -> {after:.3f}); "
                    f"tolerance {tol:.3f} GB")
            if witness_drop is not None and witness_drop < floor:
                self._fail(
                    ReleaseShortfall(resource_id, charged, witness_drop, now,
                                     "witness-contradicts-release", reported),
                    f"ledger charged {charged:.3f} GB, the actor reported "
                    f"{reported:.3f} GB and its own measurement fell "
                    f"{measured_drop:.3f} GB, but the independent witness "
                    f"({witness_source}) fell only {witness_drop:.3f} GB; "
                    f"tolerance {tol:.3f} GB")

            proved = min([reported, measured_drop]
                         + ([witness_drop] if witness_drop is not None else []))

            # (5) the drip. Each of these passed on its own; the sum need not.
            residue = charged - proved
            if residue > 0:
                self.cumulative_residual_gb += residue
                if self.cumulative_residual_gb > self.leak_budget_gb:
                    self._fail(
                        ReleaseShortfall(resource_id, charged, proved, now,
                                         "cumulative-leak", reported),
                        f"this release left {residue:.3f} GB behind, within "
                        f"the {tol:.3f} GB per-release tolerance, but the "
                        f"accumulated residue is now "
                        f"{self.cumulative_residual_gb:.3f} GB against a leak "
                        f"budget of {self.leak_budget_gb:.3f} GB "
                        f"({self.leak_budget_frac:.1%} of the allocation)")

            # (6) recorded, NOT raised: more came back than was charged.
            #     A teardown has a BENIGN STRUCTURAL CAUSE for this and it is
            #     not always an I1 undercount: stopping an engine returns the
            #     parked weights AND the engine's own baseline (interpreter,
            #     CUDA context, allocator arenas), and only the weights were
            #     ever charged. A5's actor subtracts the baseline and reports
            #     both halves; when the baseline is unknown it returns the
            #     whole delta rather than guessing, and that legitimately
            #     exceeds the charge. The budget is safe in this direction, so
            #     it is evidence to publish, not an error to raise.
            if measured_drop - charged > tol or reported - charged > tol:
                self.over_releases.append(
                    ReleaseShortfall(resource_id, charged, measured_drop, now,
                                     "over-release", reported))
            if reported - measured_drop > tol:
                self.over_releases.append(
                    ReleaseShortfall(resource_id, charged, measured_drop, now,
                                     "report-exceeds-measurement", reported))

            if witness_drop is None:
                self.unwitnessed_releases += 1
            if witness_error is not None:
                self.witness_errors.append((resource_id, witness_error))
            self.release_log.append(dict(
                resource_id=resource_id, now_s=now, charged_gb=charged,
                reported_gb=reported, measured_drop_gb=measured_drop,
                witness_drop_gb=witness_drop, witness_source=witness_source,
                witness_error=witness_error, proved_gb=proved,
                tolerance_gb=tol,
                witnesses=("actor-report", "resource-measurement")
                + (("independent-witness",) if witness_drop is not None else ())))

            del self._entries[resource_id]
            return proved

    # -- the verification hook ---------------------------------------------

    def reconcile(self) -> Dict[str, Tuple[float, float]]:
        """{resource_id: (charged_gb, measured_gb)} for every entry.

        Reports, never corrects. Silently rewriting charged_gb to whatever the
        last measurement said would erase the only evidence that the budget and
        reality have parted company.
        """
        with self._lock:
            out: Dict[str, Tuple[float, float]] = {}
            for rid in sorted(self._entries):
                e = self._entries[rid]
                actor = self.actor_for(e.actor_class or e.spec.resource_class)
                out[rid] = (e.charged_gb, float(actor.measure_held_gb(rid)))
            return out

    def drift(self) -> Dict[str, Tuple[float, float]]:
        """The subset of reconcile() that is off by more than the tolerance."""
        return {
            rid: (c, m)
            for rid, (c, m) in self.reconcile().items()
            if abs(c - m) > self.drift_tolerance_gb
        }

    def audit(self) -> str:
        """One human-readable block for a run report."""
        lines = [
            f"ledger: {self.held_gb:.2f} / {self._budget_gb:.2f} GB held "
            f"({len(self._entries)} entries, slack {self.slack_gb:.2f} GB)"
        ]
        for rid, (c, m) in self.reconcile().items():
            flag = "  DRIFT" if abs(c - m) > self.drift_tolerance_gb else ""
            lines.append(f"  {rid:<20} charged {c:8.2f}  measured {m:8.2f}{flag}")
        if self.declared_charges:
            lines.append(
                f"  {len(self.declared_charges)} charge(s) were DECLARED, not "
                f"measured (I1): {sorted(set(self.declared_charges))}"
            )
        for r in self.release_shortfalls:
            lines.append(
                f"  I2 VIOLATION [{r.reason}] {r.resource_id}: charged "
                f"{r.charged_gb:.2f}, actor reported {r.reported_gb:.2f}, "
                f"measured drop {r.freed_gb:.2f}"
            )
        for r in self.over_releases:
            lines.append(
                f"  over-release {r.resource_id}: charged {r.charged_gb:.2f}, "
                f"measured drop {r.freed_gb:.2f} (charge was an undercount)"
            )
        for rid, err in self.witness_errors:
            lines.append(f"  release_witness() FAILED for {rid}: {err}")
        if self.unwitnessed_releases:
            lines.append(
                f"  {self.unwitnessed_releases} release(s) confirmed with no "
                f"actor-independent witness (a teardown actor's post-release "
                f"measurement is 0.0 by construction)"
            )
        if self.cumulative_residual_gb > 0:
            lines.append(
                f"  cumulative unexplained residue "
                f"{self.cumulative_residual_gb:.2f} GB against a leak budget of "
                f"{self.leak_budget_gb:.2f} GB"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ResidencyLedger(budget_gb={self._budget_gb:.1f}, "
            f"held_gb={self.held_gb:.1f}, entries={len(self._entries)})"
        )
