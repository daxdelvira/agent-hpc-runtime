"""T3 — the horizon estimator. Implements `contract.HorizonEstimator`.

INVARIANT I3 IS THE WHOLE POINT OF THIS FILE. `next_use_s` returns
`Optional[float]`, and `None` means *"not within the lookahead L"*. It never
means "never again". Nothing in this module can produce `inf`, a negative
distance, or a sentinel: every return path goes through
`contract.check_horizon`, which raises on inf/nan/negative. Excluding
predicted-dead resources from the retention pool cost ~9 points in simulation
at our measured accuracy and produced a finding that had to be retracted. This
estimator is not entitled to claim a resource is dead; the most it may say is
that it cannot see the next use.

TWO TIME CONSTANTS, AND THIS MODULE OWNS EXACTLY ONE OF THEM.

    L — the LOOKAHEAD. `horizon_s`. How far this estimator can see. Ours.
    D — the DECAY SCALE. Eq. 1's H. How fast value falls off with distance.
        The arbitrator's (`GreedyArbitrator(decay_s=...)`, default 60 s from
        `scripts/make_results_tables.py:44`). NOT ours.

Nothing here takes D as an argument, imports `DEFAULT_DECAY_S`, or calls
`contract.value`. If D and L were equal, every reachable dt would satisfy
dt <= L == D, `max(dt, D)` would be D for all of them, and Eq. 1 would collapse
to `benefit_s` — a static s/GB ranking with no time term. The estimator must
not be the place that quietly makes them equal. There is a test that asserts
this module never reads `horizon_s` as a decay scale, and one that asserts the
module's source contains no reference to a decay parameter at all.

NO CONFIDENCE THRESHOLD. There is deliberately no `min_confidence` here, and
adding one would be a regression. A gate asks "is this prediction likely
enough?" in isolation; the right question is whether the expected saving is
worth the GB, and that is Eq. 1's job, in the arbitrator. §1.4 of
`sc-workshop-paper/tandem_build_plan_v2_20260829.md` is the evidence: the fixed
0.85 gate was calibrated on 165 homogeneous traces and admits nothing on 490
diverse ones. So this module folds confidence into the *distance* it reports —
a low-probability near-term need reports as a farther-away need, continuously,
with no cliff — and reports the raw arrival distribution alongside it
(`explain()`) for anything that wants to see the working.

HOW CONFIDENCE BECOMES A DISTANCE (the one piece of arithmetic worth reading).

Eq. 1 is v = benefit * D / max(dt, D). Outside the saturated region that is
v ∝ 1/dt, so value is linear in the *rate* 1/dt and the expected value of an
uncertain arrival distribution {(t_k, q_k)} — plus mass (1 - Σq) that is not
within L and is therefore priced at L, exactly as `contract.value` prices a
None — is proportional to

    rate = Σ_k q_k / max(t_k, resolution) + (1 - Σ_k q_k) / L

and the single distance with that same value is dt* = 1 / rate. So dt* is the
*certainty-equivalent distance*: the distance at which a certain need would be
worth what this uncertain one is worth. It is derived without D and does not
contain D — the D-dependence of Eq. 1 lives entirely in the saturating
`max(dt, D)`, which is the arbitrator's to apply.

Two consequences that are properties, not accidents:

  * With no signal at all, Σq = 0, rate = 1/L, dt* = L, and `next_use_s`
    returns None — because dt* >= L is exactly the definition of "not within
    the lookahead". The uninformed case falls out of the formula rather than
    being special-cased.
  * A *certain* need exactly at L also yields dt* = L and therefore None. That
    is value-neutral, not a loss: `contract.value(dt=None, lookahead_s=L)` and
    `contract.value(dt=L, ...)` are the same number.

`resolution_s` IS NOT A DECAY SCALE. It is the floor on t_k inside that sum,
and it exists only because 1/t_k diverges as t_k -> 0, so a 1% chance of a need
half a second out would otherwise dominate the whole estimate — the same
divergence that makes the shared prefetch currency broken (I5). It is a
bound on how far a near-zero predicted distance may dominate an aggregation
(`DEFAULT_RESOLUTION_S`, 1 s), it is chosen from the recordings rather than
against a value function, and it prices nothing. It is not a duplicate filter:
that is `adjudicate_needs`, which uses an independent instrumentation path and
no time threshold at all.

It is deliberately NOT set to the step duration, which was the first thing tried
and is wrong: a step is 60-1330 s depending on the facet, so flooring at a step
would flatten the whole final approach to a need, and whether that flattening
mattered would depend on where D happened to sit. The estimator must not have a
knob whose harmlessness depends on the arbitrator's D. Over-weighting a near
arrival is in any case corrected downstream and by design: Eq. 1 saturates below
D, so a dt* of 3 s and a dt* of D score identically.

SIGNALS. Two, combined simultaneously and not as a fallback chain, in the
*time* domain rather than the offset domain — see `_arrivals`. Both are
converted to seconds before they meet, which is what makes it legitimate to
combine a plan offset (a step of the tool sequence) with a transition-table
offset (a step of something else entirely; see the warning on
`TransitionSignal`).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from runtime.residency.contract import check_horizon

__all__ = [
    "DEFAULT_LOOKAHEAD_S",
    "DEFAULT_RESOLUTION_S",
    "MEASURED_TOOL_STEP_S",
    "MEASURED_LLM_STEP_S",
    "Arrival",
    "DemandMap",
    "certainty_equivalent_distance",
    "mean_tool_step_s",
    "reuse_distances",
    "adjudicate_needs",
    "read_tool_executions",
    "PHASE_TO_TOOL",
    "TransitionSignal",
    "TransitionBasisMismatch",
    "EXPECTED_TOOL_BASIS",
    "EXPECTED_MODEL_BASIS",
    "PlanTransitionHorizon",
    "ReplayHorizon",
]


# --------------------------------------------------------------------------
# L — the lookahead. Measured, not invented.
# --------------------------------------------------------------------------

DEFAULT_LOOKAHEAD_S = 1800.0
"""L, in seconds. Chosen from the reuse-distance distribution, not from D.

Measured over `results/eval_q1_q4/runs/*/*/*/trace.jsonl` restricted to
`meta.json` `status == "completed"` (241 runs), faceted by GPU family and NEVER
pooled across it, with duplicate emissions adjudicated per tool against each
run's `metrics.csv` (`adjudicate_needs`, no time threshold):

    Blackwell  models  n=159  median  810.7 s   46% within 600 s, 84% within 1800 s
    Blackwell  data    n=148  median 1668.7 s    3% within 600 s, 68% within 1800 s
    L40S       models  n= 56  median  472.6 s   54% within 600 s, 82% within 1800 s
    L40S       data    n=  8  median 2734.9 s    0% within 600 s,  0% within 1800 s

A 600 s lookahead sees 3% of Blackwell data reuses. 1800 s sees 68%, and is the
shortest round value that keeps both classes in view on both node types with
enough samples to say anything. Note what that makes L relative to the
arbitrator's D=60 s: a factor of 30. That is the separation invariant I3
insists on, arrived at from the data rather than asserted — the typical reuse
sits an order of magnitude beyond the decay scale, so Eq. 1's time term is live
across almost the whole distribution rather than saturated.

THE CHOICE OF L DID NOT MOVE WHEN THE DUPLICATE RULE WAS FIXED, and that is
worth stating because the fix was substantial. On every resource the
instrumentation can adjudicate, the discarded 1 s-collapse rule and this one
agree: Blackwell data 1669.0 -> 1668.7 s with identical 3% / 68% fractions;
Blackwell models restricted to adjudicable resources 844.0 -> 833.5 s with
identical 45% / 84%; L40S identical in both classes. The whole of the
class-level movement (models 986.8 -> 810.7 s, 36% -> 46% within 600 s) is
`qwen_72b_text`, and that is not a disagreement between rules but a question the
instrumentation cannot answer: its `code_task` has no execution record, so its
median is a BRACKET of 764.3-1297.3 s. See `adjudicate_needs` and
`PHASE_TO_TOOL`.

The L40S data row is n=8 and is not evidence of anything; it is printed so that
nobody quietly pools it with the Blackwell row to get a bigger n.
"""


# --------------------------------------------------------------------------
# Step durations. MEASURED, per (workload, GPU family), and never pooled.
# --------------------------------------------------------------------------

DEFAULT_RESOLUTION_S = 1.0
"""The floor on t_k inside `certainty_equivalent_distance`, so that 1/t_k
cannot diverge. One second.

It is NOT a duplicate filter — that job belongs to `adjudicate_needs`, and a
time threshold was measured to be unsafe for it (genuine `analyze_screw_core`
repeats are 0.946-1.084 s apart while `plan_task` logging duplicates are
0.330-0.714 s; the populations abut). This constant never decides whether an
event happened. It only bounds how much a near-zero predicted distance may
dominate an aggregation.

It is NOT a decay scale, it does not multiply a benefit, and it must not be set
to the step duration — see the module docstring.
"""


MEASURED_TOOL_STEP_S: Dict[Tuple[str, str], float] = {
    ("atomagents_exp2", "Blackwell"): 608.09,
    ("atomagents_exp3", "Blackwell"): 751.55,
    ("atomagents_exp3_aligned", "Blackwell"): 1330.37,
    ("atomagents_exp3_aligned", "L40S"): 1069.46,
    ("chemgraph_screen", "Blackwell"): 87.94,
    ("chemgraph_screen", "L40S"): 181.07,
    ("chemgraph_screen_pool", "L40S"): 76.82,
    ("chemgraph_swap", "Blackwell"): 6.04,
    ("chemgraph_swap", "L40S"): 16.89,
}
"""MEAN seconds between consecutive tool needs. One step of the PLAN signal.

MEAN, not median, deliberately: the estimator needs E[time to reach step k],
which is k times the mean; the median of a heavy-tailed gap distribution
understates that badly. For `atomagents_exp3`/Blackwell the median is 603 s and
the mean 752 s; for `chemgraph_screen`/Blackwell the median is 5.8 s and the
mean 87.9 s, a factor of 15.

There is NO pooled entry and no default. `chemgraph_swap` on Blackwell is
6.04 s and `atomagents_exp3_aligned` on Blackwell is 1330 s — a factor of 220.
A single constant would be wrong by that factor for one of them, and the
standing rule against pooling L40S with Blackwell applies here as much as it
does to a speedup. Callers pass the number for the facet they are in, or
measure their own with `mean_tool_step_s()`.
"""

MEASURED_LLM_STEP_S: Dict[Tuple[str, str], float] = {
    ("atomagents_exp2", "Blackwell"): 303.71,
    ("atomagents_exp3", "Blackwell"): 374.46,
    ("atomagents_exp3_aligned", "Blackwell"): 633.43,
    ("atomagents_exp3_aligned", "L40S"): 659.88,
}
"""MEAN seconds between consecutive `llm_call` events. One step of the
`model_transitions` half of the table, whose offsets are on the
`llm_call_subsequence` basis.

REPLACES an earlier `MEASURED_EVENT_STEP_S` that measured the MIXED
tool_call+llm_call sequence. That constant was correct for the pre-fix table and
is wrong for the corrected one, and using it would have been the same arithmetic
error this module refused to make for tool transitions: an offset counted in one
sequence converted with the step duration of another. The difference is not
small — the mixed-sequence mean for `atomagents_exp3`/Blackwell was 222.78 s
against 374.46 s here, so an offset-1 model self-loop would have been reported
as 40% nearer than it is.

Measured over `results/eval_q1_q4/runs/*/*/*/trace.jsonl` restricted to
`meta.json` `status == "completed"`, 2026-08-30. Only the AtomAgents workloads
appear: the chemgraph traces contain no `llm_call` events at all, which is why
their tool offsets were never affected by any of this.

Tool transitions now need NO separate constant. On the corrected
`tool_call_subsequence` basis a table offset IS a plan step, so
`MEASURED_TOOL_STEP_S` converts both signals.
"""


# --------------------------------------------------------------------------
# What a resource is needed BY
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandMap:
    """resource_id -> the tool names, and model names, that need it.

    This is the only place the estimator knows anything about what a resource
    IS, and it knows only which symbols in a trace imply a need for it. The
    arbitrator stays class-blind (I4); so does everything below this line.
    """

    by_tool: Mapping[str, frozenset] = field(default_factory=dict)
    by_model: Mapping[str, frozenset] = field(default_factory=dict)

    def resources_for_tool(self, tool: str) -> frozenset:
        return self.by_tool.get(tool, frozenset())

    def resources_for_model(self, model: str) -> frozenset:
        return self.by_model.get(model, frozenset())

    def resource_ids(self) -> frozenset:
        out: set = set()
        for s in self.by_tool.values():
            out |= set(s)
        for s in self.by_model.values():
            out |= set(s)
        return frozenset(out)

    @classmethod
    def from_tool_resources(
        cls, path: Optional[Path] = None, resource_key: str = "name"
    ) -> "DemandMap":
        """Build from `runtime/predictor/data/tool_resources.json`.

        Two shapes are read, because the file carries two:
          * entries with a `consumer_tool`, whose `name` is the resource;
          * the `residency_artifact` block, whose `resource_id` is the resource
            and whose `consumer_tools` is a list. That block has deliberately
            no `consumer_tool` key so the legacy byte-oriented registry skips
            it (it is an R3 activated structure, not a file range), which means
            `ResourceRegistry.from_json` does NOT see it and this method must.

        `resource_key` selects which field names a resource for the first
        shape. It defaults to `name` (`qwen_32b`, `w_eam4_big.fs`) because that
        is the id the ledger and the arbitrator's docstrings use. Pass
        `resource_id` if a caller is keying on the registry's md5 instead.
        """
        if path is None:
            path = (Path(__file__).resolve().parents[1]
                    / "predictor" / "data" / "tool_resources.json")
        raw = json.loads(Path(path).read_text())
        by_tool: Dict[str, set] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            tool = entry.get("consumer_tool")
            if tool:
                rid = entry.get(resource_key) or entry.get("name")
                if rid:
                    by_tool.setdefault(tool, set()).add(rid)
            art = entry.get("residency_artifact")
            if isinstance(art, dict):
                rid = art.get("resource_id")
                for t in art.get("consumer_tools", ()):
                    if rid:
                        by_tool.setdefault(t, set()).add(rid)
        return cls(by_tool={k: frozenset(v) for k, v in by_tool.items()},
                   by_model={})

    def with_models(self, by_model: Mapping[str, Iterable[str]]) -> "DemandMap":
        """A copy carrying a model-name -> resource-id map as well.

        Kept separate from the tool map because the transition table keeps
        `model_transitions` separate from `tool_transitions`, and because a
        model name in a trace (`Qwen/Qwen2.5-VL-72B-Instruct`) is not the
        resource id the catalogue uses (`qwen_72b`).
        """
        return DemandMap(
            by_tool=self.by_tool,
            by_model={k: frozenset(v) for k, v in by_model.items()},
        )


# --------------------------------------------------------------------------
# Arrivals and the certainty-equivalent distance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arrival:
    """One signal's claim: "this resource is needed in `distance_s` seconds,
    with probability `probability`". Carried as evidence, not as a decision.

    `source` names which signal said it, so `explain()` can show the working
    and so a campaign can attribute a decision to a signal.
    """

    distance_s: float
    probability: float
    source: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.distance_s < 0:
            raise ValueError(f"arrival distance must be >= 0: {self.distance_s}")
        if math.isinf(self.distance_s) or math.isnan(self.distance_s):
            raise ValueError("arrival distance must be finite (I3)")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"arrival probability out of [0,1]: {self.probability}")


def certainty_equivalent_distance(
    arrivals: Sequence[Arrival],
    lookahead_s: float,
    resolution_s: float,
) -> Optional[float]:
    """The distance a CERTAIN need would have to sit at to be worth as much as
    this uncertain arrival distribution. Returns None when that is >= L.

    See the module docstring for the derivation. In short: value is linear in
    1/dt outside Eq. 1's saturated region, so the expected value of the
    distribution is proportional to a rate, and dt* is the reciprocal of it.
    Mass beyond L is priced at L, which is exactly how `contract.value` prices
    a dt of None, so the unseen tail is discounted rather than discarded — I3.

    D DOES NOT APPEAR HERE AND MUST NOT. `resolution_s` floors t_k so that a
    tiny probability at a tiny distance cannot diverge; it is a measurement
    resolution in units of one workload step, not a decay scale, and it never
    prices anything.

    Overlapping arrivals are chained into FIRST-need mass in ascending distance
    order — q_k = p_k * prod_{j<k}(1 - p_j) — so two signals both predicting the
    same need do not sum to more than certainty, and the nearer claim is the one
    that keeps its weight.
    """
    if lookahead_s <= 0:
        raise ValueError("lookahead_s (L) must be > 0")
    if resolution_s <= 0:
        raise ValueError("resolution_s must be > 0")
    # A resolution above L would floor every arrival to beyond the lookahead,
    # which would silently turn every answer into None. Clamp and carry on:
    # the estimator can never resolve better than its own lookahead anyway.
    resolution = min(float(resolution_s), float(lookahead_s))

    within = sorted(
        (a for a in arrivals if a.distance_s <= lookahead_s and a.probability > 0.0),
        key=lambda a: (a.distance_s, -a.probability, a.source),
    )
    rate = 0.0
    survive = 1.0          # probability the need has NOT yet occurred
    for a in within:
        q = survive * a.probability
        rate += q / max(a.distance_s, resolution)
        survive *= (1.0 - a.probability)
    # Everything not accounted for is, as far as this estimator is entitled to
    # say, at least L away. Priced at L. Never at infinity.
    rate += survive / lookahead_s

    dt = 1.0 / rate
    if dt >= lookahead_s:
        return None
    return check_horizon(dt)


# --------------------------------------------------------------------------
# The transition signal, and the reason it is off by default
# --------------------------------------------------------------------------


EXPECTED_TOOL_BASIS = "tool_call_subsequence"
EXPECTED_MODEL_BASIS = "llm_call_subsequence"


class TransitionBasisMismatch(RuntimeError):
    """Raised when a half of the transition table is enabled but its declared
    offset basis is missing or unexpected.

    An error and not a warning, for the same reason `ReleaseNotHonoured` is: a
    pre-fix table's offsets were counted over a MIXED tool_call+llm_call
    sequence, so converting them with either step constant produces a distance
    that is wrong by a factor nobody can bound afterwards. Every horizon derived
    from it, and every retention decision derived from that, would be fiction.
    """


@dataclass(frozen=True)
class TransitionSignal:
    """The learned transition table, wrapped with its offset basis attached.

    TWO THINGS ARE DECLARED IN THE FILE, AND THIS CLASS REFUSES TO GUESS EITHER.

    `offset_basis` says what an offset COUNTS
    (`tool_call_subsequence` / `llm_call_subsequence`); `synthetic_filter` says
    whether replay-harness traces were excluded before counting. A file missing
    either key is a pre-fix or pre-filter table, and enabling a half of one
    RAISES rather than converting numbers whose meaning nobody recorded. Both
    checks exist because both defects were live in this file within two days,
    both produced plausible-looking numbers, and neither announced itself.

    THE TWO HALVES NOW GO DIFFERENT WAYS, and the reason is a measured
    asymmetry rather than a preference.

    -- model_transitions: ON. --------------------------------------------
    Disabled here on 2026-08-31 on the finding that the offset-1 self-loop
    barely beat its own base rate. That finding was drawn from an unfiltered
    corpus and it inverted once the synthetic traces came out, because the
    harness runs were driving the base rate toward 1.0 -- precisely the regime
    in which conditioning on the current model CANNOT buy anything:

        corpus         72B +1 -> 72B     72B share of llm_call     advantage
        unfiltered     0.9893 (n=11311)  0.9685                    +2.1 pts
        filtered       0.8752 (n=  652)  0.7433                    +13.2 pts

    Re-derived independently rather than taken on report: reimplementing
    `llm_turn_plausibility_v1` from the thresholds in the file header
    (min_llm_turn_seconds 1.0, max_consecutive_fast_turns 7,
    min_seconds_per_tool_call 3.0, scope >= 2 llm_call events) over
    `logs/workflow_traces/*.jsonl` reproduces the header exactly -- scanned 490,
    out_of_scope 340, kept 99, excluded 51 (burst 40 / rate 11), excluded
    tool_events 5365, llm_events 10826 -- and gives self-loop 652/745 = 0.8752
    against base rate 837/1126 = 0.7433. The minority model is the stronger
    case: 32B +1 -> 32B is 0.7000 against a 0.2567 base rate, +44.3 points.

    Two further checks, because three of the four reasons for the old default
    had to be retested and it would have been easy to keep the conclusion:

      * IT DOES CARRY SWITCHING, which the unfiltered table did not. The
        off-diagonal is 72B -> 32B 0.1248 / 0.2057 / 0.2329 at offsets 1-3 and
        32B -> 72B 0.3000 / 0.5961 / 0.9558. That matters more than the
        diagonal: the serving model needs no prediction, while a PARKED model
        with 12-23% mass inside three LLM steps gets a finite distance instead
        of a None and therefore competes for budget. At exp3/Blackwell that
        moves a parked 32B from L=1800 s to about 1220 s, a 1.5x in Eq. 1 -- in
        the direction I3 exists to allow.
      * THE PLAN DOES NOT ALREADY COVER IT. Predicting the next llm_call's
        model on the filtered corpus (n=1025 events with a predecessor):
        majority class 0.7180, model self-loop 0.8273, tool->model via
        `tool_resources.json` 0.9352 BUT ONLY on the 69% of events whose
        preceding tool has a model mapping, and silent on the other 31%. The
        registry is better where it applies and absent where it does not, so
        the two are complementary. "The plan already names each tool's model"
        was true and was not the whole picture.

    And the self-loop is not itself a logging artifact: adjudicated against
    `metrics.csv`, `llm_call` is UNDER-recorded, never over-recorded -- 439
    trace events against 621 `llm:inference` rows over 66 completed AtomAgents
    runs, with zero runs where the trace has more. The double emission is a
    `tool_call` phenomenon and does not reach this half.

    -- tool_transitions: OFF, for one sharp reason that survived the filter. --
    The top self-loop is 93% instrumentation artifact and it distorts the whole
    row. `plan_task -> plan_task` is n=43 on the filtered corpus, of which 40
    are sub-second pairs and 3 are real; the synthetic filter could not remove
    them because the double emission happens in REAL traces too. So the table
    reports `plan_task -> plan_task` at 0.3613 where the truth is nearer 3/119,
    and `code_task` at 0.5882 where its true share of real successors is
    70/79 = 0.886. The mass is not merely diluted, it is misallocated, and a
    horizon that reads it holds the wrong resource rather than merely holding
    the right one with less confidence.

    This one is fixable and the fix is already in this module: run the corpus
    through `adjudicate_needs` against each run's `metrics.csv` before
    learning, and the row is repaired. Until someone does, the half stays off.
    `code_task` has no execution record, so it cannot be adjudicated either --
    the same gap that brackets `qwen_72b_text`'s reuse distance.

    Each half converts in ITS OWN basis: tool offsets are plan steps and use
    `tool_step_s`; model offsets are LLM-call steps and use `llm_step_s`
    (`MEASURED_LLM_STEP_S`). Those differ by a factor of two on the same facet
    -- 751.55 s against 374.46 s at atomagents_exp3/Blackwell -- so enabling
    the model half without passing `llm_step_s` is refused rather than defaulted.
    """

    tool_transitions: Mapping[str, Mapping[int, Sequence[Mapping]]] = field(
        default_factory=dict)
    model_transitions: Mapping[str, Mapping[int, Sequence[Mapping]]] = field(
        default_factory=dict)
    n_traces: int = 0
    offset_basis: Mapping[str, str] = field(default_factory=dict)
    synthetic_filter: Mapping[str, object] = field(default_factory=dict)
    use_tool_transitions: bool = False
    use_model_transitions: bool = True

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        use_tool_transitions: bool = False,
        use_model_transitions: bool = True,
    ) -> "TransitionSignal":
        """Load the table.

        Enabling a half of a table that does not declare BOTH its offset basis
        and an applied synthetic filter raises. A pre-fix file's offsets are on
        neither basis; a pre-filter file's probabilities are dominated by
        replay-harness traces. Either one yields a plausible number that is
        wrong, which is why this is an exception and not a warning."""
        if path is None:
            path = (Path(__file__).resolve().parents[1]
                    / "predictor" / "data" / "learned_transitions.json")
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())

        def _norm(d):
            return {src: {int(k): v for k, v in offs.items()}
                    for src, offs in (d or {}).items()}

        basis = raw.get("offset_basis") or {}
        sfilter = raw.get("synthetic_filter") or {}
        if (use_tool_transitions or use_model_transitions) and not sfilter.get("applied"):
            raise TransitionBasisMismatch(
                f"{p}: synthetic_filter is "
                f"{'absent' if not sfilter else 'applied=False'}. An unfiltered "
                "table counts replay-harness traces alongside real ones; on this "
                "corpus that drove the model base rate from 0.7433 to 0.9685 and "
                "turned a +13.2 point conditional advantage into +2.1. Regenerate "
                "with the filter before enabling either half.")
        for enabled, key, expect in (
            (use_tool_transitions, "tool_transitions", EXPECTED_TOOL_BASIS),
            (use_model_transitions, "model_transitions", EXPECTED_MODEL_BASIS),
        ):
            if enabled and basis.get(key) != expect:
                raise TransitionBasisMismatch(
                    f"{p}: {key} declares offset_basis "
                    f"{basis.get(key)!r}, expected {expect!r}. A table with no "
                    "declared basis is a pre-fix file whose offsets were "
                    "counted over a MIXED tool_call+llm_call sequence and are "
                    "on neither basis; converting them with a step duration "
                    "would be an arithmetic error, not an approximation.")

        return cls(
            tool_transitions=_norm(raw.get("tool_transitions")),
            model_transitions=_norm(raw.get("model_transitions")),
            n_traces=int(raw.get("n_traces", 0)),
            offset_basis=dict(basis),
            synthetic_filter=dict(sfilter),
            use_tool_transitions=use_tool_transitions,
            use_model_transitions=use_model_transitions,
        )

    @classmethod
    def disabled(cls) -> "TransitionSignal":
        return cls(use_tool_transitions=False, use_model_transitions=False)

    def tool_offsets(self, source: str) -> Sequence[int]:
        if not self.use_tool_transitions:
            return ()
        return tuple(sorted(self.tool_transitions.get(source, {})))

    def model_offsets(self, source: str) -> Sequence[int]:
        if not self.use_model_transitions:
            return ()
        return tuple(sorted(self.model_transitions.get(source, {})))

    def tool_mass(self, source: str, offset: int, targets: frozenset) -> float:
        """Total probability, at this offset, on any target in `targets`."""
        if not self.use_tool_transitions:
            return 0.0
        rows = self.tool_transitions.get(source, {}).get(offset, ())
        return _clamp01(sum(float(r.get("probability", 0.0)) for r in rows
                            if r.get("target") in targets))

    def model_mass(self, source: str, offset: int, targets: frozenset) -> float:
        if not self.use_model_transitions:
            return 0.0
        rows = self.model_transitions.get(source, {}).get(offset, ())
        return _clamp01(sum(float(r.get("probability", 0.0)) for r in rows
                            if r.get("target") in targets))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# --------------------------------------------------------------------------
# The concrete estimator
# --------------------------------------------------------------------------


class PlanTransitionHorizon:
    """`HorizonEstimator` over the plan and the learned transitions.

    Both signals are converted to SECONDS before they are combined, each in its
    OWN basis: plan offsets and (since A3's learner fix) tool-transition offsets
    are both tool steps and go through `tool_step_s`, while model-transition
    offsets are LLM-call steps and go through `llm_step_s`. They are then merged
    into one arrival distribution and collapsed to a certainty-equivalent
    distance. That is what "simultaneous, not a fallback
    chain" means concretely: there is no branch anywhere below that says "if the
    plan has an answer, ignore the table". If both speak, both contribute; if
    neither does, the answer is None, which means "beyond L" and nothing more.

    Position is tracked by `observe()`. The estimator does not read the clock
    itself: every method takes `now_s` from the caller, so a replay and a live
    run take the same code path.
    """

    def __init__(
        self,
        demand: DemandMap,
        tool_step_s: float,
        *,
        horizon_s: float = DEFAULT_LOOKAHEAD_S,
        transitions: Optional[TransitionSignal] = None,
        llm_step_s: Optional[float] = None,
        plan_confidence: float = 0.80,
        plan_offset_decay: float = 1.0,
        max_plan_offset: int = 8,
        resolution_s: Optional[float] = None,
    ) -> None:
        """`tool_step_s` has NO default on purpose. It is the mean seconds per
        tool step of the facet being run, it ranges from 6.04 s
        (chemgraph_swap/Blackwell) to 1330 s (atomagents_exp3_aligned/Blackwell)
        across the measured facets, and a default would be wrong by a factor of
        220 for one of them. Take it from `MEASURED_TOOL_STEP_S[(workload, gpu)]`
        or measure it with `mean_tool_step_s()`.

        `plan_confidence` is the probability that the plan's named tool is the
        one that actually runs. 0.80 mirrors `_PLAN_CONFIDENCE_DEFAULT` at
        `runtime/predictor/learned_predictor.py:172`. It is a PROBABILITY fed to
        the value function, not a gate: nothing here compares it to a threshold,
        and there is no threshold to compare it to.

        `plan_offset_decay` damps confidence with distance into the plan
        (offset k gets `plan_confidence * decay**(k-1)`). 1.0 — undamped — is
        the current derived value; `offset_decay` moved 0.8404 -> 1.0 as a side
        effect of the corpus regeneration (§1.4), which is noted there as a knob
        that should not be silently accepted. It is exposed so a sweep can move
        it deliberately.

        `resolution_s` defaults to `DEFAULT_RESOLUTION_S` (1 s, the separation
        at which the recordings stop distinguishing two needs). It is NOT a
        decay scale, and it is deliberately not the step duration: a step is
        60-1330 s across the measured facets, so flooring there would flatten
        the entire final approach to a need and whether that mattered would
        depend on the arbitrator's D. This module must not own a knob like that.
        """
        if tool_step_s <= 0:
            raise ValueError("tool_step_s must be > 0")
        if horizon_s <= 0:
            raise ValueError("horizon_s (L) must be > 0")
        if not 0.0 <= plan_confidence <= 1.0:
            raise ValueError("plan_confidence must be in [0, 1]")
        if not 0.0 < plan_offset_decay <= 1.0:
            raise ValueError("plan_offset_decay must be in (0, 1]")
        if max_plan_offset < 1:
            raise ValueError("max_plan_offset must be >= 1")

        self._demand = demand
        self._tool_step_s = float(tool_step_s)
        self._transitions = transitions or TransitionSignal.disabled()
        if self._transitions.use_model_transitions and llm_step_s is None:
            raise ValueError(
                "model transitions are enabled but llm_step_s was not given. "
                "There is deliberately no fallback to tool_step_s: an LLM step "
                "and a tool step differ by a factor of two on the same facet "
                "(374.46 s against 751.55 s at atomagents_exp3/Blackwell), so "
                "the fallback would convert an llm_call_subsequence offset with "
                "a tool-step constant and be wrong by that factor. Take it from "
                "MEASURED_LLM_STEP_S[(workload, gpu)].")
        self._llm_step_s = float(
            llm_step_s if llm_step_s is not None else tool_step_s)
        if self._llm_step_s <= 0:
            raise ValueError("llm_step_s must be > 0")
        self._horizon_s = float(horizon_s)
        self._plan_confidence = float(plan_confidence)
        self._plan_offset_decay = float(plan_offset_decay)
        self._max_plan_offset = int(max_plan_offset)
        self._resolution_s = float(
            resolution_s if resolution_s is not None else DEFAULT_RESOLUTION_S)
        if self._resolution_s <= 0:
            raise ValueError("resolution_s must be > 0")

        self._plan = None            # PlanContext | None
        self._plan_index = 0
        self._last_tool: Optional[str] = None
        self._last_model: Optional[str] = None
        self._last_step_s: float = 0.0

    # -- the protocol ------------------------------------------------------

    @property
    def horizon_s(self) -> float:
        """L, the lookahead. NOT Eq. 1's decay scale (I3)."""
        return self._horizon_s

    def next_use_s(self, resource_id: str, now_s: float) -> Optional[float]:
        """Seconds until `resource_id` is next needed, or None for "not within
        L". Never inf, never negative, never a "dead" marker."""
        return certainty_equivalent_distance(
            self._arrivals(resource_id, now_s),
            self._horizon_s,
            self._resolution_s,
        )

    # -- observation -------------------------------------------------------

    def set_plan(self, plan, index: int = 0) -> None:
        """Attach a `plan_extractor.PlanContext` and the current position in it.

        Duck-typed on `tool_at_offset(current_index, offset)`, which
        `PlanContext` already supports for arbitrary offsets, so a caller can
        pass any object with that method. Passing None drops the plan signal
        and leaves the transition signal alone — the point of the two being
        simultaneous is that losing one is a degradation, not a switch.
        """
        self._plan = plan
        self._plan_index = int(index)

    def observe(
        self,
        now_s: float,
        tool: Optional[str] = None,
        model: Optional[str] = None,
        advance_plan: bool = True,
    ) -> None:
        """Record that a tool ran and/or a model was called at `now_s`.

        `advance_plan` re-syncs the plan cursor to the observed tool where the
        plan contains it ahead of the cursor, and otherwise just steps forward.
        A resync is the honest response to divergence: it does not claim the
        plan was right, it just stops the cursor drifting.
        """
        self._last_step_s = float(now_s)
        if tool:
            self._last_tool = tool
            if advance_plan and self._plan is not None:
                self._plan_index = self._resync(tool, self._plan_index)
        if model:
            self._last_model = model

    def _resync(self, tool: str, index: int) -> int:
        seq = getattr(self._plan, "tool_sequence", None) or []
        for k in range(0, self._max_plan_offset + 1):
            j = index + k
            if 0 <= j < len(seq) and seq[j] == tool:
                return j + 1
        return index + 1

    # -- the two signals ---------------------------------------------------

    def _arrivals(self, resource_id: str, now_s: float) -> List[Arrival]:
        """Every claim both signals make about `resource_id`, in seconds.

        Distances are measured from `now_s`, and the time already spent inside
        the current step is subtracted, so a resource does not appear to get
        farther away as a step runs long. The result is floored at 0 rather
        than allowed to go negative: a need that should already have happened is
        "imminent", not "in the past", and `check_horizon` rejects negatives on
        the way out anyway.
        """
        elapsed = max(0.0, float(now_s) - self._last_step_s)
        out: List[Arrival] = []

        # --- signal 1: the plan --------------------------------------------
        if self._plan is not None:
            for k in range(1, self._max_plan_offset + 1):
                tool = self._plan.tool_at_offset(self._plan_index - 1, k)
                if tool is None:
                    break
                if resource_id not in self._demand.resources_for_tool(tool):
                    continue
                # plan_offset_decay damps confidence with distance. It is a
                # CONSTRUCTOR ARGUMENT here and never read from the transition
                # table, deliberately: the table's `offset_decay` is derived
                # TABLE-WIDE and is unstable at these sample sizes in both
                # directions -- 0.8404 -> 1.0 when the corpus grew (1.0 meaning
                # no damping at all, because harness self-loops sit at p=1.0 at
                # every offset), then 1.0 -> 0.9870 over 46 pairs when the
                # synthetic traces came out. B4 measured the consequence:
                # run_ase +3 moved 0.8957 -> 0.8841 although its own row did not
                # change. So any offset>=3 confidence quoted as a property of
                # one workload is not one -- it is a property of every other
                # workload in the corpus. Anything sweeping this must set it
                # explicitly and say what it set.
                p = self._plan_confidence * (self._plan_offset_decay ** (k - 1))
                d = max(0.0, k * self._tool_step_s - elapsed)
                out.append(Arrival(d, _clamp01(p), "plan",
                                   f"offset {k} -> {tool}"))

        # --- signal 2: the learned transitions ------------------------------
        # Offsets here are the TABLE's own basis, converted with the table's own
        # step constant. They are never treated as plan offsets; see
        # TransitionSignal's docstring for the measurement behind that.
        if self._last_tool:
            for k in self._transitions.tool_offsets(self._last_tool):
                targets = self._tools_needing(resource_id)
                m = self._transitions.tool_mass(self._last_tool, k, targets)
                if m <= 0.0:
                    continue
                # tool_call_subsequence basis: an offset IS a tool step.
                d = max(0.0, k * self._tool_step_s - elapsed)
                out.append(Arrival(d, m, "transition:tool",
                                   f"offset {k} (tool_call_subsequence basis)"))

        if self._last_model:
            for k in self._transitions.model_offsets(self._last_model):
                targets = self._models_needing(resource_id)
                m = self._transitions.model_mass(self._last_model, k, targets)
                if m <= 0.0:
                    continue
                # llm_call_subsequence basis: an offset is an LLM step, which is
                # NOT a tool step. Converting one with the other's constant is
                # the error TransitionBasisMismatch exists to prevent.
                d = max(0.0, k * self._llm_step_s - elapsed)
                out.append(Arrival(d, m, "transition:model",
                                   f"offset {k} (llm_call_subsequence basis)"))

        return out

    def _tools_needing(self, resource_id: str) -> frozenset:
        return frozenset(
            t for t, rs in self._demand.by_tool.items() if resource_id in rs)

    def _models_needing(self, resource_id: str) -> frozenset:
        return frozenset(
            m for m, rs in self._demand.by_model.items() if resource_id in rs)

    # -- the working -------------------------------------------------------

    def explain(self, resource_id: str, now_s: float) -> dict:
        """The arrival distribution and the distance it collapses to.

        This is the "report a distance and, where you can, a calibrated
        confidence" half of the design. The arbitrator only takes a distance,
        so the confidence is folded into it; this is where a campaign, or a
        reviewer, can see what was folded.
        """
        arrivals = self._arrivals(resource_id, now_s)
        within = [a for a in arrivals if a.distance_s <= self._horizon_s]
        seen = 0.0
        survive = 1.0
        for a in sorted(within, key=lambda a: a.distance_s):
            seen += survive * a.probability
            survive *= (1.0 - a.probability)
        return {
            "resource_id": resource_id,
            "now_s": float(now_s),
            "lookahead_s": self._horizon_s,
            "resolution_s": self._resolution_s,
            "arrivals": [
                {"distance_s": a.distance_s, "probability": a.probability,
                 "source": a.source, "detail": a.detail}
                for a in sorted(arrivals, key=lambda a: a.distance_s)
            ],
            "mass_within_lookahead": seen,
            "next_use_s": self.next_use_s(resource_id, now_s),
        }


# --------------------------------------------------------------------------
# The deterministic replay estimator
# --------------------------------------------------------------------------


class ReplayHorizon:
    """`HorizonEstimator` that reads a recorded need schedule. Deterministic.

    Two uses. Offline: replay a recorded trace and get the horizon the estimator
    WOULD have had with a perfect predictor, so a campaign can bracket the
    ceiling before trusting a number from the learned one. In tests: a stub with
    no stub's licence to be wrong about I3 — it obeys the invariant by
    construction, so anything downstream can be tested against a real
    implementation instead of a mock that might quietly return inf.

    THE ONE THING IT MUST NOT DO is report "never again". A resource with no
    remaining use in the recording returns None — the same None as a resource
    whose next use is 10000 s away. The estimator cannot distinguish those two
    cases and is not entitled to; that is I3, and it is why the exhausted case
    below is a plain `return None` and not a distinguishable value.
    """

    def __init__(
        self,
        schedule: Mapping[str, Sequence[float]],
        horizon_s: float = DEFAULT_LOOKAHEAD_S,
    ) -> None:
        if horizon_s <= 0:
            raise ValueError("horizon_s (L) must be > 0")
        self._horizon_s = float(horizon_s)
        self._schedule: Dict[str, Tuple[float, ...]] = {}
        for rid, times in schedule.items():
            ts = tuple(sorted(float(t) for t in times))
            for t in ts:
                if math.isinf(t) or math.isnan(t):
                    raise ValueError(
                        f"{rid}: a use time must be finite — an infinite use "
                        "time is 'never again' by another name (I3)")
            self._schedule[rid] = ts

    @property
    def horizon_s(self) -> float:
        """L, the lookahead. NOT Eq. 1's decay scale (I3)."""
        return self._horizon_s

    def next_use_s(self, resource_id: str, now_s: float) -> Optional[float]:
        times = self._schedule.get(resource_id)
        if not times:
            # Unknown resource, or one with no recorded use at all. NOT dead.
            return None
        now = float(now_s)
        for t in times:
            if t >= now:
                dt = t - now
                if dt > self._horizon_s:
                    return None
                return check_horizon(dt)
        # Every recorded use is in the past. Still not dead: the recording
        # simply ends, and "the trace stopped" is not evidence about the future.
        return None

    def uses(self, resource_id: str) -> Tuple[float, ...]:
        return self._schedule.get(resource_id, ())

    def resource_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._schedule))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_next_use(
        cls,
        table: Mapping[str, float],
        horizon_s: float = DEFAULT_LOOKAHEAD_S,
        now_s: float = 0.0,
    ) -> "ReplayHorizon":
        """A fixed {resource_id: dt} table anchored at `now_s`.

        The drop-in replacement for a hand-written stub: same shape as the
        dict `test_residency_arbitrator.StubHorizon` takes, but it cannot
        return inf, because the constructor refuses to store one.
        """
        return cls({rid: [float(now_s) + float(dt)] for rid, dt in table.items()},
                   horizon_s=horizon_s)

    @classmethod
    def from_trace(
        cls,
        path,
        demand: DemandMap,
        horizon_s: float = DEFAULT_LOOKAHEAD_S,
        origin: str = "trace_start",
        metrics_csv=None,
    ) -> "ReplayHorizon":
        """Build the need schedule from a recorded `trace.jsonl`.

        `origin="trace_start"` puts t=0 at the first tool call, so replayed
        times are elapsed seconds; `origin="epoch"` keeps raw epoch times.

        `metrics_csv` is the run's sibling `metrics.csv`. Pass it. Without it
        the events are read RAW, and raw over-counts: the runtime emits some
        AtomAgents tool calls twice, about 0.35 s apart, and counting the second
        emission as a fresh need invents a reuse at 0.35 s — the single most
        valuable-looking distance there is.

        THERE IS DELIBERATELY NO TIME THRESHOLD HERE ANY MORE. An earlier
        version collapsed same-tool events inside 1 s. That rule is not safe,
        and the reason is measured rather than argued: adjudicated against
        `metrics.csv`, `plan_task`'s repeats are logging duplicates (21 of the
        41 completed trials that call it emit 2 events and execute 1, both
        events preceding the single execution, gap median 0.367 s) while
        `analyze_screw_core`'s repeats are GENUINE — each event has its own
        execution and the two runs do different work, W_screw_Zhou04 then
        W_screw_w_eam4, which is the two-potential comparison the workload
        exists to perform. Those genuine repeats are 0.946-1.084 s apart. The
        two populations abut, so no threshold separates them; only the
        independent instrumentation does. A 1 s rule destroyed 2 of the 36
        genuine adjacent repeats in the completed AtomAgents runs, at the short
        end, which is the end retention value is most sensitive to.

        See `adjudicate_needs` for the rule and for which tools it cannot reach.
        """
        events = _read_tool_calls(path)
        if metrics_csv is not None:
            events, _ = adjudicate_needs(events, read_tool_executions(metrics_csv))
        if not events:
            return cls({}, horizon_s=horizon_s)
        t0 = events[0][0] if origin == "trace_start" else 0.0
        sched: Dict[str, List[float]] = {}
        for ts, tool in events:
            for rid in demand.resources_for_tool(tool):
                sched.setdefault(rid, []).append(ts - t0)
        return cls(sched, horizon_s=horizon_s)


# --------------------------------------------------------------------------
# Trace helpers (read-only; nothing here writes anywhere)
# --------------------------------------------------------------------------


PHASE_TO_TOOL: Dict[str, str] = {
    "agent:plan_task": "plan_task",
    "agent:compute_screw_dislocation": "computation_task_screw_dislocation",
    "agent:analyze_screw_core": "analyze_screw_core",
}
"""metrics.csv `phase` -> the tool name the trace uses.

This is the whole of the independent instrumentation available. `code_task` has
NO `agent:` phase in any of the 63 completed AtomAgents runs that carry a
metrics.csv — 39 of them call it, none records an execution — so `code_task` is
UNADJUDICABLE, and so is `qwen_72b_text`, the only resource that depends on it.
Every chemgraph tool is likewise unrecorded, but those traces carry no duplicate
emissions at all, so raw and adjudicated agree there by construction.
"""


def _read_tool_calls(path) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event_type") != "tool_call":
                continue
            payload = ev.get("payload") or {}
            tool = payload.get("tool")
            ts = ev.get("epoch_time")
            if tool and ts is not None:
                out.append((float(ts), tool))
    out.sort(key=lambda r: r[0])
    return out


def read_tool_executions(metrics_csv) -> Dict[str, List[float]]:
    """{tool: [execution start times]} from a run's `metrics.csv`.

    An instrumentation path independent of the trace being diagnosed, which is
    the only reason it can settle the question at all.
    """
    out: Dict[str, List[float]] = {}
    with open(metrics_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            tool = PHASE_TO_TOOL.get((row.get("phase") or "").strip())
            if not tool:
                continue
            stamp = (row.get("timestamp") or "").strip()
            try:
                out.setdefault(tool, []).append(
                    datetime.fromisoformat(stamp).timestamp())
            except ValueError:
                continue
    for tool in out:
        out[tool].sort()
    return out


def adjudicate_needs(
    events: Sequence[Tuple[float, str]],
    executions: Mapping[str, Sequence[float]],
) -> Tuple[List[Tuple[float, str]], frozenset]:
    """Keep the `tool_call` events that correspond to a real execution.

    THE RULE, and it contains no time constant: an event counts as a need iff an
    execution of that tool starts at or after it and before the NEXT event of
    that tool. A logging duplicate has no execution attributable to it and is
    dropped; a genuine repeat has its own execution and is kept, however close
    the two events are.

    Returns `(kept_events, unadjudicable_tools)`. A tool with no execution
    record at all is kept RAW and named in the second element — the caller must
    decide what to do about it, because silently keeping it and silently
    dropping it are both wrong and the difference is large: for `qwen_72b_text`,
    whose `code_task` is unadjudicable, the median reuse distance is 764.3 s if
    its duplicates are retained and 1297.3 s if a 1 s rule removes them. That is
    a bracket, not an estimate, and it should be reported as one.
    """
    by_tool: Dict[str, List[float]] = {}
    for ts, tool in events:
        by_tool.setdefault(tool, []).append(ts)

    kept: List[Tuple[float, str]] = []
    unadjudicable: set = set()
    for tool, times in by_tool.items():
        starts = executions.get(tool)
        if not starts:
            unadjudicable.add(tool)
            kept.extend((t, tool) for t in times)
            continue
        for i, ts in enumerate(times):
            nxt = times[i + 1] if i + 1 < len(times) else float("inf")
            if any(ts <= x < nxt for x in starts):
                kept.append((ts, tool))
    kept.sort(key=lambda r: r[0])
    return kept, frozenset(unadjudicable)


def mean_tool_step_s(paths: Iterable, metrics_csvs: Optional[Iterable] = None
                     ) -> Optional[float]:
    """Mean seconds between consecutive tool needs over recorded traces.

    MEAN, because the estimator needs E[time to reach step k] = k * mean. Gaps
    are only taken WITHIN a trace, never across two, and the caller is
    responsible for passing traces from ONE hardware facet: the same tool step
    is 6.04 s on chemgraph_swap/Blackwell and 16.89 s on chemgraph_swap/L40S,
    and pooling those produces a constant that describes neither.

    Pass `metrics_csvs` positionally aligned with `paths` to adjudicate the
    duplicate emissions; without it the events are read raw.
    """
    gaps: List[float] = []
    mcs = list(metrics_csvs) if metrics_csvs is not None else []
    for i, p in enumerate(paths):
        ev = _read_tool_calls(p)
        if i < len(mcs) and mcs[i] is not None:
            ev, _ = adjudicate_needs(ev, read_tool_executions(mcs[i]))
        gaps.extend(b[0] - a[0] for a, b in zip(ev, ev[1:]))
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def reuse_distances(
    paths: Iterable,
    demand: DemandMap,
    metrics_csvs: Optional[Iterable] = None,
) -> Tuple[Dict[str, List[float]], frozenset]:
    """({resource_id: [seconds between consecutive needs]}, unadjudicable_tools).

    The measurement behind `DEFAULT_LOOKAHEAD_S`. Pass `metrics_csvs` aligned
    with `paths`; without it the events are raw and the short end is inflated by
    logging duplicates.

    Caveats that travel with every number it produces:

      * It is RIGHT-CENSORED. A gap is only observed when a second use happens
        inside the same recording, and most model resource-instances are used
        exactly once in their trace. The true tail is longer, so a "fraction
        within L" computed from it is an upper bound.
      * It is the need sequence OUR MAPPING implies, not one the agent chose.
        `tool_resources.json` says which tool needs which resource; the agent
        does not yet select artifacts (plan §0.3 item P4).
      * Anything derived from an UNADJUDICABLE tool is a bracket, not a point.
      * Filter the corpus first. Restrict to `meta.json` `status == "completed"`,
        and pass traces from ONE hardware facet.
    """
    out: Dict[str, List[float]] = {}
    unadj: set = set()
    mcs = list(metrics_csvs) if metrics_csvs is not None else []
    for i, p in enumerate(paths):
        ev = _read_tool_calls(p)
        if i < len(mcs) and mcs[i] is not None:
            ev, u = adjudicate_needs(ev, read_tool_executions(mcs[i]))
            unadj |= set(u)
        last: Dict[str, float] = {}
        for ts, tool in ev:
            for rid in demand.resources_for_tool(tool):
                if rid in last:
                    out.setdefault(rid, []).append(ts - last[rid])
                last[rid] = ts
    return out, frozenset(unadj)
