"""Tandem residency contract — the frozen interface between the ledger,
the arbitrator, and the per-class residency actors.

WRITTEN 2026-08-30 AS A FIXED INPUT TO THE BUILD. Agents implementing T1/T2
(ledger, arbitrator) and T4a/T4b (model actor, data worker) both build against
this file. Do not change a signature here without saying so explicitly in your
report — a silent change to this file is the one thing that makes the two
halves unmergeable.

Five invariants are load-bearing. Each exists because of a specific failure
this project has already had:

I1. held_gb is MEASURED, not declared.
    Every actor implements measure_held_gb(). A budget over numbers a caller
    asserted is a fiction, and every percentage computed against it is
    meaningless. Cf. the COW benchmark, where Rss would have made a failed
    copy-on-write look successful and Private_Dirty did not.

I2. release() is confirmable BY INDEPENDENT MEASUREMENT, not by the actor's
    word. It returns the GB the OS actually gave back, AND the ledger must
    confirm that with its own post-release measure_held_gb() call before it
    drops the entry. A worker that retains but cannot truly release is the
    most likely silent failure in this build (item D2).

    THE FIRST IMPLEMENTATION OF THIS GOT IT WRONG, and the hole is instructive
    (found by V, 2026-08-30). It compared charged_gb against the float the
    actor RETURNED and never called measure_held_gb(). An actor that reports
    honestly and frees nothing therefore passed cleanly -- and because the
    entry is deleted on the success path, reconcile() could never see the leak
    afterwards. Iterated four times against a 256 GB allocation that produced
    480 GB actually resident, 88% over budget, with zero recorded violations
    and a clean audit(). This needs no malice: glibc arena retention alone
    freed only 0.084 of 0.364 GB at small scale in A2's measurements while an
    honest actor would have reported the full amount.

    So the enforcement is: call the actor's release(), then measure, and
    require that the MEASURED footprint fell by what was charged. The actor's
    return value is evidence, not proof. Three further requirements learned
    from the same probe set:
      - reject non-finite returns (nan compares False against every bound and
        sailed through every check);
      - when the actor raises, keep the entry charged AND record it, so
        audit() shows the failure rather than showing nothing;
      - the tolerance must not be purely absolute, or the undetectable leak
        grows with cycle count rather than with budget (freeing charged-0.25
        each time leaked 25.00 GB over 100 cycles, silently).

I3. The horizon never says "never again", only "not within the lookahead".
    next_use_s returns Optional[float]; None means "not within the lookahead
    L". float('inf') is rejected at runtime. Excluding predicted-dead
    resources from the pool entirely cost ~9 points in simulation and
    produced a finding that had to be retracted.

    NOTE THE TWO DIFFERENT TIME CONSTANTS -- conflating them was a real defect
    in the first draft of this file, found by measurement on 2026-08-30:
      L, the LOOKAHEAD  -- how far the estimator can see. Beyond it: None.
      D, the DECAY SCALE -- Eq. 1's H, how fast value falls off with distance.
    If D == L, then every reachable dt satisfies dt <= L == D, so
    max(dt, D) == D for all of them and Eq. 1 collapses to benefit_s: the
    ranking degenerates to the static s/GB band with NO time term at all.
    Verified for uniref90 (117.20 GB, 372.6 s): dt of 0, 1, 60, 599, 600 and
    None all score 372.600. D is the parameter the paper sweeps (fig h-sweep)
    and is typically much SHORTER than the lookahead. Keep them separate.

I4. The arbitrator is class-blind.
    It ranks ResourceSpec by value density and never learns that one thing is
    a vLLM engine and another is a LAMMPS instance. All class-specific
    knowledge lives behind ResidencyActor.

I5. v1 is RETAIN-ONLY currency.
    The shared retain-vs-prefetch currency is known broken: benefit is capped
    at min(load, dt) for a prefetch, so as dt -> 0 the rate tends to 1/GB and
    a 5-second prefetch outranks a 798-second retention. Retain-only arms
    never touch that cap. Do not add a prefetch candidate to the same ranking
    until that objective is redesigned.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------
# Rungs of the cost ladder
# --------------------------------------------------------------------------


class Rung(enum.IntEnum):
    """A residency level is a prefix of the cost chain already paid."""

    R0_DISK = 0            # on disk
    R1_PAGE_CACHE = 1      # bytes in page cache (what a DMSH can express)
    R2_PROCESS_BYTES = 2   # bytes in a process address space (vLLM L1 sleep)
    R3_ACTIVATED = 3       # activated structure in a live consumer

    # R2 and R3 are ANONYMOUS memory. Neither is a file range, which is
    # precisely why a byte-oriented tier cannot express either.


class ResourceClass(enum.Enum):
    MODEL = "model"                    # class 1: weights parked at R2
    DATA_PATTERN_A = "data_pattern_a"  # class 2: load/query split, held by reference
    DATA_PATTERN_B = "data_pattern_b"  # class 3: engine with installed state


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSpec:
    """Static description of one holdable resource.

    cold_s and ready_s are per-use costs, both measured. The difference is
    what a retention saves on one reuse.
    """

    resource_id: str
    resource_class: ResourceClass
    held_rung: Rung          # the rung this resource is held AT when retained
    held_gb: float           # measured footprint when held (see I1)
    cold_s: float            # cost to make usable from R0
    ready_s: float           # cost to make usable when held at held_rung

    def __post_init__(self) -> None:
        if self.held_gb <= 0:
            raise ValueError(f"{self.resource_id}: held_gb must be > 0")
        if self.cold_s < self.ready_s:
            raise ValueError(
                f"{self.resource_id}: cold_s ({self.cold_s}) < ready_s "
                f"({self.ready_s}) — a retention that costs more than a cold "
                f"load is a measurement error, not a policy input"
            )

    @property
    def benefit_s(self) -> float:
        """Seconds of stall avoided by one reuse of a held copy."""
        return self.cold_s - self.ready_s

    @property
    def static_density(self) -> float:
        """Seconds saved per GB held, ignoring time. A format constant.

        Reference values measured for this project:
            qwen_32b        3.81      uniref90   3.18
            qwen_72b        2.86      uniref50   2.97
            qwen_72b_text   2.78      EAM pot.   2.25
        The model band is 2.78-3.81. A resource below 2.78 cannot win budget
        against any model, at any size. That is a fact about the workload, not
        a bug to engineer around.
        """
        return self.benefit_s / self.held_gb


# --------------------------------------------------------------------------
# Horizon (T3) — invariant I3
# --------------------------------------------------------------------------


@runtime_checkable
class HorizonEstimator(Protocol):
    """Estimates time until a resource is next needed.

    INVARIANT I3: returns None for "not within the horizon", NEVER inf and
    NEVER a sentinel meaning "never again". A single wrong "never" discards a
    resource even when the budget has room.
    """

    @property
    def horizon_s(self) -> float:
        """L — the LOOKAHEAD window, in seconds. Beyond this, next_use_s
        returns None. This is NOT Eq. 1's decay scale; see I3."""
        ...

    def next_use_s(self, resource_id: str, now_s: float) -> Optional[float]:
        ...


def check_horizon(dt: Optional[float]) -> Optional[float]:
    """Runtime guard for I3. Call this on anything a HorizonEstimator returns."""
    if dt is None:
        return None
    if math.isinf(dt) or math.isnan(dt):
        raise ValueError(
            "horizon returned inf/nan — I3 forbids 'never again'; "
            "return None to mean 'not within H'"
        )
    if dt < 0:
        raise ValueError(f"horizon returned negative dt: {dt}")
    return dt


# --------------------------------------------------------------------------
# The value function — Eq. 1
# --------------------------------------------------------------------------


def value(
    spec: ResourceSpec,
    dt_s: Optional[float],
    decay_s: float,
    lookahead_s: Optional[float] = None,
) -> float:
    """Eq. 1:  v(r) = (cold(r) - ready(r)) * D / max(dt(r), D)

    decay_s is D, Eq. 1's decay scale -- the distance at which value starts
    falling off. It is the parameter the paper sweeps.

    lookahead_s is L, how far the estimator can see. It is used ONLY to price
    dt_s=None: "not within the lookahead" means the resource is at least L
    away, so it is scored AT L and discounted accordingly. Passing None for
    lookahead_s prices an unseen resource at D, which makes it indistinguishable
    from an imminent one -- that is the degenerate case described in I3 and it
    is almost never what you want.

    A resource beyond the lookahead is NOT worthless (that would be the
    "never again" claim I3 forbids); it is merely the most discounted thing
    the estimator is entitled to describe.
    """
    if decay_s <= 0:
        raise ValueError("decay_s must be > 0")
    if lookahead_s is not None and lookahead_s <= 0:
        raise ValueError("lookahead_s must be > 0 when given")
    if dt_s is None:
        dt = decay_s if lookahead_s is None else lookahead_s
    else:
        dt = max(dt_s, 0.0)
    # NOTE THE PARENTHESES -- they are load-bearing, not style. Written as
    # `benefit * decay_s / max(dt, decay_s)` the multiply happens first, so the
    # saturated case (dt <= decay_s) is (benefit*60.0)/60.0, which is NOT
    # bit-exactly benefit: 34 of 154 catalogue values differ in the last ULP
    # (max 1.14e-13 abs, 1.4e-16 rel). That is invisible to any ranking, but
    # sim_residency_v2._knapsack tie-breaks on exact float equality
    # (`v == best_v and gb < best_gb`), so one ULP selects a different subset
    # and moves wall time by up to 15.4%. Grouping the ratio first makes the
    # factor exactly 1.0 when saturated, so value(dt<=D) is bit-exactly
    # benefit_s. Found by V, 2026-08-30.
    return spec.benefit_s * (decay_s / max(dt, decay_s))


def value_density(
    spec: ResourceSpec,
    dt_s: Optional[float],
    decay_s: float,
    lookahead_s: Optional[float] = None,
) -> float:
    """v(r) / g(r) — seconds of stall avoided per GB held. The ranking key."""
    return value(spec, dt_s, decay_s, lookahead_s) / spec.held_gb


# --------------------------------------------------------------------------
# Residency actors — all class-specific knowledge lives here (I4)
# --------------------------------------------------------------------------


@runtime_checkable
class ResidencyActor(Protocol):
    """Makes one class of resource resident, and gives it back on demand.

    The arbitrator never imports vLLM or LAMMPS. It calls stage/release and
    trusts measure_held_gb().
    """

    @property
    def resource_class(self) -> ResourceClass:
        ...

    def stage(self, spec: ResourceSpec) -> Rung:
        """Make the resource resident. Returns the rung actually reached.

        May return a LOWER rung than spec.held_rung if the full climb was not
        possible (e.g. GPUs occupied). Returning a lower rung is not an error;
        silently reporting the requested rung is.
        """
        ...

    def release(self, resource_id: str) -> float:
        """Drop the resource and return the GB the OS actually gave back.

        INVARIANT I2. This return value is EVIDENCE, NOT PROOF: the ledger
        will independently call measure_held_gb() after this returns and will
        believe the measurement over this number. Implementations should
        measure rather than compute the return value (A2's LammpsDataWorker
        reads Private_Dirty from the worker's smaps_rollup immediately before
        teardown), but must assume they are not trusted.

        A return of 0.0 for a resource the ledger believed was held is the
        signal that retention is not real, and must fail loudly rather than be
        rounded away.
        """
        ...

    def measure_held_gb(self, resource_id: str) -> float:
        """Measured footprint, right now. INVARIANT I1.

        Models: host-RAM delta read from the cgroup path, not /proc/meminfo.
        Data:   Private_Dirty (or Pss) from the worker's smaps_rollup. Rss
                double-counts shared pages and will flatter a failure.
        """
        ...

    def is_resident(self, resource_id: str) -> bool:
        ...

    def release_witness(self, resource_id: str) -> Optional[float]:
        """GB the ENCLOSING allocation gave back, measured independently of
        this resource's own process. Return None if you have no such witness.

        THIS METHOD IS REQUIRED, THE RETURN VALUE IS OPTIONAL. ResidencyActor
        is @runtime_checkable, so a Protocol method is structurally mandatory
        for isinstance() regardless of what its docstring calls it -- an
        "optional method" on a runtime-checkable Protocol is a contradiction,
        and describing it that way (mine, briefly, on 2026-08-30) silently
        broke every actor's protocol test. Every actor implements it; an actor
        with nothing independent to report returns None and is counted in the
        ledger's `unwitnessed_releases`.

        ADDED 2026-08-30, proposed by A1 after A5 found the teardown gap.
        WHY IT EXISTS: for a TEARDOWN actor -- one that kills the process
        rather than freeing inside it -- measure_held_gb() afterwards is 0.0
        BY CONSTRUCTION, so the ledger's before/after drop is tautologically
        the whole charge and confirms nothing. The ledger cannot tell that 0.0
        apart from an honest in-process release's 0.0, and it must not try:
        sniffing for "is this a teardown actor" is exactly the class-specific
        knowledge I4 forbids.

        So the actor declares its own independent witness instead. In practice
        that is the job cgroup's `anon` total sampled either side of the
        release -- a reading that does not depend on the released process still
        existing. Returning None is honest and allowed; the ledger counts such
        releases in `unwitnessed_releases` and audit() reports them, because a
        release confirmed only by a tautology is evidence the reader should be
        able to discount.

        This is deliberately a method on the ACTOR and not a field on
        ResourceSpec. How to measure an enclosing allocation is class-specific
        (a cgroup for a vLLM engine, a process tree for a data worker), and
        ResourceSpec is the structure the arbitrator ranks -- putting it there
        would leak class knowledge into exactly the place I4 keeps clean.

        STALENESS IS THE HAZARD. Implementations typically cache the value
        during release. A cached figure returned on a later call looks exactly
        like a fresh witness and would vouch for a release it never observed,
        which is the same tautology this method exists to break. Stamp the
        cached detail with the release it belongs to and return None when the
        stamp does not match.
        """
        ...


class ReleaseNotHonoured(RuntimeError):
    """Raised when release() frees materially less than the ledger charged.

    This is invariant I2 firing. It means the budget is fiction from this
    point on, so it is an error and not a warning: every downstream
    percentage computed after a silent partial release is meaningless.
    """


# --------------------------------------------------------------------------
# Ledger (T1) and arbitrator (T2)
# --------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    spec: ResourceSpec
    rung: Rung
    charged_gb: float          # what the ledger believes it holds
    staged_at_s: float
    last_use_s: float
    use_count: int = 0
    actor_class: Optional[ResourceClass] = None


@runtime_checkable
class Ledger(Protocol):
    """One budget over every held class. There is exactly one of these.

    The budget is an ALLOCATION, not a hardware constant — it is the cgroup
    limit of the job, and it is a swept variable in the evaluation.
    """

    @property
    def budget_gb(self) -> float:
        ...

    @property
    def held_gb(self) -> float:
        """Sum of charged_gb over all entries."""
        ...

    @property
    def slack_gb(self) -> float:
        """budget_gb - held_gb. What a prefetcher may stage into (T5)."""
        ...

    def entries(self) -> Sequence[LedgerEntry]:
        ...

    def charge(self, spec: ResourceSpec, rung: Rung, now_s: float) -> LedgerEntry:
        ...

    def release(self, resource_id: str, now_s: float) -> float:
        """Release via the owning actor, reconcile against charged_gb.

        Raises ReleaseNotHonoured if the measured free is materially below
        what was charged (I2). Tolerance is an explicit constructor argument,
        not a magic number buried here.
        """
        ...

    def note_use(self, resource_id: str, now_s: float) -> None:
        ...

    def reconcile(self) -> dict[str, tuple[float, float]]:
        """{resource_id: (charged_gb, measured_gb)} for every entry.

        The verification hook. Drift between the two columns is the budget
        drifting away from reality; report it, do not silently correct it.
        """
        ...


@dataclass(frozen=True)
class EvictionPlan:
    """What the arbitrator decided, and why — the 'why' is not optional.

    rationale must name the ranking that produced the decision, so that any
    number derived from a campaign can be traced to the decisions behind it.
    """

    evict: tuple[str, ...]   # in eviction order; may be >1 (see Arbitrator)
    admit: Optional[str]
    freed_gb: float
    rationale: str


@runtime_checkable
class Arbitrator(Protocol):
    """Decides what to hold under the budget. Class-blind (I4).

    v1 is GREEDY, deliberately, not exact subset selection. Greedy at 0.55
    predictor accuracy reaches essentially the exact-solve ceiling (+20.6% of
    wall) and is nearly flat across accuracy, while exact collapses and goes
    NEGATIVE at the accuracy we actually have. Do not "improve" this to an
    exact solve.

    WHAT "GREEDY" MEANS HERE, precisely -- the first draft of this file got it
    wrong and A1 caught it on 2026-08-30. Greedy evicts the least-dense holder
    REPEATEDLY, re-ranking after each step, until the candidate fits or no
    victim is cheaper than the candidate. It is allowed more than one victim
    per admission. That is the arm that produced +20.6%; a one-victim-per-
    admission rule is a DIFFERENT and strictly more conservative policy, and
    claiming the measured number for it would be claiming a result for code we
    did not run.

    The property that distinguishes greedy from exact is not the victim count.
    It is that each eviction is an independent, re-evaluated step, so a wrong
    horizon perturbs one decision -- whereas exact commits to a whole subset at
    once, so a wrong horizon restructures the entire retained set. Chaining
    preserves that property; subset enumeration does not.

    Implementations MUST bound the chain (max_victims) so a pathological rank
    cannot empty the ledger to admit one item, and MUST record the chain in
    EvictionPlan.rationale.
    """

    def admit(
        self,
        candidate: ResourceSpec,
        ledger: Ledger,
        horizon: HorizonEstimator,
        now_s: float,
    ) -> EvictionPlan:
        """Decide whether candidate can be held, and what to evict for it.

        Returns a plan with admit=None if the candidate does not earn its
        budget. Declining is a correct outcome and must be recorded as a
        decision, not dropped — the EAM potential at 2.25 s/GB is expected to
        be declined against any model, and that is the arbitrator working.
        """
        ...
