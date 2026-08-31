"""T2 — the retention arbitrator, plus the two ranking primitives it shares
with `scripts/sim_residency_v2.py`.

GREEDY, DELIBERATELY, AND NOT AN APPROXIMATION WE APOLOGISE FOR. Greedy
ranking by value density at 0.55 predictor accuracy reaches essentially the
exact-solve ceiling (+20.6% of wall) and is nearly flat across the accuracy
range, while exact subset selection collapses and goes NEGATIVE at the accuracy
we actually measure. Exact optimisation over estimates this noisy optimises the
noise. There is deliberately no exact solve in this module and no flag to turn
one on: the simulator carries that arm, which is where a comparison belongs.

GREEDY MEANS CHAINED, BOUNDED EVICTION. The least-dense holder is evicted
REPEATEDLY, with the ranking recomputed after each step, until the candidate
fits or no remaining holder is cheaper than it. More than one victim per
admission is allowed, because that is the arm that produced +20.6% -- a
one-victim rule is a different and strictly more conservative policy, and
claiming the measured number for it would be claiming a result for code we did
not run. What separates greedy from exact is not the victim count but that each
eviction is an independent, re-evaluated step: a wrong horizon perturbs one
decision instead of restructuring the whole retained set. The chain is bounded
by `max_victims` so a pathological ranking cannot empty the ledger to admit one
item, and the whole chain is recorded in the rationale.

TWO TIME CONSTANTS, NEVER ONE (I3). D, the decay scale, is Eq. 1's H and the
parameter the paper sweeps. L, the lookahead, is `HorizonEstimator.horizon_s`
-- how far the estimator can see, and the distance at which a dt of None is
priced. If D == L every reachable dt satisfies dt <= D, Eq. 1 collapses to
benefit_s, and the ranking degenerates to the static s/GB band with no time
term at all. This module always passes both, separately.

CLASS-BLIND (invariant I4). Nothing here imports vLLM or LAMMPS, branches on a
resource class, or knows what a rung physically is. It ranks `ResourceSpec` by
value density and calls the ledger. Every class-specific fact lives behind
`ResidencyActor`.

RETAIN-ONLY CURRENCY (invariant I5). A prefetch candidate must not enter this
ranking. The shared retain-vs-prefetch currency is known broken: prefetch
benefit is capped at min(load, dt), so as dt -> 0 the rate tends to 1/GB and a
5-second prefetch outranks and evicts a 798-second retention. `admit()` takes
ONE candidate that is already needed and asks only whether holding it beats
holding something else. T5's slack-first prefetcher is subordinate to the
ledger's slack and does not come through here.

DECLINING IS A DECISION. `admit()` never returns None and never raises to mean
"no". It returns a plan with `admit=None` and a rationale naming the ranking
that produced it. The EAM potential at 2.25 s/GB sits below the model band
(2.78-3.81) and is expected to lose to any model at any size; an arbitrator
that declines it is working, and the decline has to be visible in the record
rather than inferred from an absence.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Set, Tuple

from runtime.residency.contract import (
    EvictionPlan,
    HorizonEstimator,
    Ledger,
    ResourceSpec,
    check_horizon,
    value,
    value_density,
)

__all__ = [
    "DEFAULT_DECAY_S",
    "DEFAULT_MAX_VICTIMS",
    "evict_until_fits",
    "greedy_pack",
    "GreedyArbitrator",
    "value",
    "value_density",
]


DEFAULT_DECAY_S = 60.0
"""D, Eq. 1's decay scale, in seconds. Sourced, not invented.

This is `HORIZON_H` at `scripts/make_results_tables.py:44`, the value behind
every table in `sc-workshop-paper/results_tables/`. The H sweep in
`07_objective_check.md` is the evidence for it: smaller D is better because the
pathology lives at small dt, and any D above the typical inter-need gap (median
16.3 s in the real traces) fixes it, so 60 s sits comfortably above that gap
while staying far below any plausible lookahead. D is a swept parameter -- pass
it explicitly whenever a campaign sweeps it.
"""

DEFAULT_MAX_VICTIMS = 3
"""How long an eviction chain may get for ONE admission.

The bound exists so a pathological ranking cannot empty the ledger to admit one
item. 3 is chosen against the measured catalogue at the production allocation:
at 256 GB the largest sets that fit are 2-3 resources (qwen_32b 129.7 +
uniref90 117.2 = 246.9 GB; adding uniref50 at 36.08 GB overruns), so a bound of
3 never binds at our budgets -- the runtime chains exactly as far as the
simulator's unbounded loop does there -- while still refusing an unbounded
cascade on a larger allocation holding many more resources. Raise it explicitly
for a budget that holds more, and say so.
"""


# --------------------------------------------------------------------------
# Ranking primitives — shared with the simulator so the two cannot drift
# --------------------------------------------------------------------------
#
# These are mechanics, not policy. They take a key function and know nothing
# about what the key means; the CURRENCY is the caller's responsibility. The
# runtime only ever feeds them value density over retain candidates (I5). The
# simulator additionally uses them for its LRU arm (key = last-use step) and
# for its prefetch arm (a mixed currency that is known broken and exists as a
# control). Sharing the mechanics is the point: the eviction loop the paper
# describes and the eviction loop the runtime executes are one function.


def evict_until_fits(
    sizes: Mapping[str, float],
    key_of: Callable[[str], Any],
    budget_gb: float,
) -> Set[str]:
    """Drop the lowest-keyed holder until the set fits the budget.

    Ties break toward the lexicographically smaller id, so the result does not
    depend on dict or hash order. Returns the SET KEPT.

    The total is re-summed each round rather than decremented, so that a set
    sitting exactly on the budget is decided by the same arithmetic every time.
    """
    keep = set(sizes)
    while keep and sum(sizes[x] for x in keep) > budget_gb:
        keep.discard(min(sorted(keep), key=lambda x: (key_of(x), x)))
    return keep


def greedy_pack(
    items: Mapping[str, Tuple[float, float]],
    budget_gb: float,
) -> Set[str]:
    """Fill the budget by descending value density; skip what does not fit.

    `items` is {id: (value, gb)}. First-fit by density, which is the standard
    greedy knapsack: an item that does not fit is skipped, not terminal, so a
    small high-value item behind a large one is still taken.
    """
    order = sorted(
        items, key=lambda x: (-items[x][0] / max(items[x][1], 1e-9), x)
    )
    keep: Set[str] = set()
    used = 0.0
    for n in order:
        if used + items[n][1] <= budget_gb:
            keep.add(n)
            used += items[n][1]
    return keep


# --------------------------------------------------------------------------
# The arbitrator
# --------------------------------------------------------------------------


class GreedyArbitrator:
    """Implements the `Arbitrator` protocol. Greedy, chained, retain-only.

    The rule, in full:

        1. Score the candidate at v/g using Eq. 1 over the predicted horizon,
           with D the decay scale and L the estimator's lookahead.
        2. If it cannot fit even an empty budget, decline.
        3. If it fits the ledger's slack, admit with no eviction.
        4. Otherwise, repeatedly: re-rank the remaining holders least-dense
           first, take the least dense one that is cheaper than the candidate,
           and count its GB as freed. Stop when the candidate fits, when no
           remaining holder is cheaper, or when the chain reaches max_victims.
        5. Admit if the chain made room; otherwise decline, naming which of the
           three stops ended it.

    Step 4 recomputes the ranking on every pass. With today's scoring that is a
    no-op -- a resource's density does not depend on what else is held -- but
    the loop is written as the contract specifies it, because the moment a
    score depends on the held set (a shared page cache, a co-resident engine)
    the re-rank stops being cosmetic.
    """

    def __init__(
        self,
        decay_s: float = DEFAULT_DECAY_S,
        max_victims: int = DEFAULT_MAX_VICTIMS,
        min_density_ratio: float = 1.0,
    ) -> None:
        """decay_s: D, Eq. 1's decay scale. NOT the estimator's lookahead --
        the lookahead is read from `horizon.horizon_s` and passed to Eq. 1
        separately, because setting them equal makes the value function
        time-blind (I3). Defaults to DEFAULT_DECAY_S, the paper's 60 s.

        max_victims: the bound on one eviction chain. See DEFAULT_MAX_VICTIMS.

        min_density_ratio: how much denser the candidate must be than a victim
        before a swap is worth the churn. 1.0 (the default) means "strictly
        denser" and changes nothing about the rule above; raising it buys
        hysteresis at the price of some admissions. It is exposed because
        thrashing between two near-equal resources is a real failure mode, not
        because it should be tuned per workload.
        """
        if decay_s <= 0:
            raise ValueError("decay_s (D) must be > 0")
        if max_victims < 1:
            raise ValueError("max_victims must be >= 1")
        if min_density_ratio < 1.0:
            raise ValueError(
                "min_density_ratio < 1.0 would evict something DENSER than the "
                "candidate, which inverts the ranking"
            )
        self.decay_s = float(decay_s)
        self.max_victims = int(max_victims)
        self.min_density_ratio = float(min_density_ratio)
        # Every decision, including every decline. A number derived from a
        # campaign has to be traceable to the decisions behind it.
        self.decisions: List[Tuple[float, EvictionPlan]] = []

    # -- scoring -----------------------------------------------------------

    def _dt(self, resource_id: str, horizon: HorizonEstimator, now_s: float):
        """Ask the horizon and enforce I3 on the answer.

        None means "not within the lookahead L", never "never again";
        check_horizon rejects inf and nan at runtime. `value()` then prices a
        None at L and discounts it accordingly -- the most discounted thing the
        estimator is entitled to describe, and not worthless.
        """
        return check_horizon(horizon.next_use_s(resource_id, now_s))

    def score(
        self, spec: ResourceSpec, horizon: HorizonEstimator, now_s: float
    ) -> float:
        """v(r)/g(r) for one resource. Class-blind, and always given both time
        constants: D from this arbitrator, L from the estimator."""
        return value_density(
            spec,
            self._dt(spec.resource_id, horizon, now_s),
            self.decay_s,
            horizon.horizon_s,
        )

    def ranking(
        self,
        ledger: Ledger,
        horizon: HorizonEstimator,
        now_s: float,
        exclude: Sequence[str] = (),
    ) -> List[Tuple[str, float, float]]:
        """[(resource_id, density, charged_gb)] over everything held, least
        dense first. This is the ranking the rationale names. `exclude` drops
        holders already chosen as victims earlier in a chain."""
        skip = set(exclude)
        rows = [
            (e.spec.resource_id, self.score(e.spec, horizon, now_s), e.charged_gb)
            for e in ledger.entries()
            if e.spec.resource_id not in skip
        ]
        rows.sort(key=lambda r: (r[1], r[2], r[0]))
        return rows

    # -- the decision ------------------------------------------------------

    def admit(
        self,
        candidate: ResourceSpec,
        ledger: Ledger,
        horizon: HorizonEstimator,
        now_s: float,
    ) -> EvictionPlan:
        """Decide whether `candidate` earns budget, and what pays for it.

        Returns a plan. Does NOT mutate the ledger: the caller executes the
        plan through the actors, because only the caller can tell whether each
        release was honoured (I2).
        """
        L = horizon.horizon_s
        dt = self._dt(candidate.resource_id, horizon, now_s)
        cand_v = value(candidate, dt, self.decay_s, L)
        cand_d = cand_v / candidate.held_gb
        dt_txt = f"beyond L={L:.0f} s" if dt is None else f"{dt:.1f} s"
        head = (
            f"greedy-chained/value-density (D={self.decay_s:.0f} s, L={L:.0f} s): "
            f"{candidate.resource_id} at {cand_d:.3f} s/GB "
            f"({candidate.held_gb:.2f} GB, dt={dt_txt}, v={cand_v:.1f} s)"
        )

        # The contract's Ledger protocol has no membership test, so this is a
        # scan over entries() rather than a lookup. Held sets are single digits.
        if any(e.spec.resource_id == candidate.resource_id
               for e in ledger.entries()):
            return self._record(now_s, EvictionPlan(
                evict=(), admit=candidate.resource_id, freed_gb=0.0,
                rationale=f"{head}; already held, no decision to make"))

        if candidate.held_gb > ledger.budget_gb:
            return self._record(now_s, EvictionPlan(
                evict=(), admit=None, freed_gb=0.0,
                rationale=(f"{head}; DECLINED: does not fit the "
                           f"{ledger.budget_gb:.2f} GB allocation even empty")))

        if candidate.held_gb <= ledger.slack_gb:
            return self._record(now_s, EvictionPlan(
                evict=(), admit=candidate.resource_id, freed_gb=0.0,
                rationale=(f"{head}; admitted into {ledger.slack_gb:.2f} GB of "
                           f"slack, no eviction required")))

        # -- the chain ----------------------------------------------------
        threshold = cand_d / self.min_density_ratio
        opening = self.ranking(ledger, horizon, now_s)
        needed_gb = candidate.held_gb - ledger.slack_gb
        victims: List[Tuple[str, float, float]] = []
        freed_gb = 0.0
        stop = "fits"
        while freed_gb < needed_gb:
            if len(victims) >= self.max_victims:
                stop = f"chain bound max_victims={self.max_victims} reached"
                break
            rows = self.ranking(ledger, horizon, now_s,
                                exclude=[v[0] for v in victims])
            cheaper = [r for r in rows if r[1] < threshold]
            if not cheaper:
                stop = ("no remaining holder is cheaper than "
                        f"{cand_d:.3f} s/GB")
                break
            vid, vdens, vgb = cheaper[0]
            victims.append((vid, vdens, vgb))
            freed_gb += vgb

        chain_txt = " -> ".join(
            f"{n}@{d:.3f} s/GB (+{g:.2f} GB)" for n, d, g in victims
        ) or "none"

        if freed_gb >= needed_gb:
            return self._record(now_s, EvictionPlan(
                evict=tuple(v[0] for v in victims),
                admit=candidate.resource_id,
                freed_gb=freed_gb,
                rationale=(
                    f"{head}; needed {needed_gb:.2f} GB, chain of "
                    f"{len(victims)} (bound {self.max_victims}) freed "
                    f"{freed_gb:.2f} GB: {chain_txt}. Opening ranking "
                    f"least-dense first: {self._ranking_txt(opening)}")))

        return self._record(now_s, EvictionPlan(
            evict=(), admit=None, freed_gb=0.0,
            rationale=(
                f"{head}; DECLINED: needed {needed_gb:.2f} GB, chain stopped "
                f"after {len(victims)} victim(s) having freed {freed_gb:.2f} GB "
                f"({stop}); chain considered: {chain_txt}. Opening ranking "
                f"least-dense first: {self._ranking_txt(opening)}")))

    # -- bookkeeping -------------------------------------------------------

    @staticmethod
    def _ranking_txt(rows: Sequence[Tuple[str, float, float]]) -> str:
        return ", ".join(f"{n}={d:.3f}@{g:.1f}GB" for n, d, g in rows) or "empty"

    def _record(self, now_s: float, plan: EvictionPlan) -> EvictionPlan:
        self.decisions.append((float(now_s), plan))
        return plan

    def declines(self) -> List[Tuple[float, EvictionPlan]]:
        return [(t, p) for t, p in self.decisions if p.admit is None]
