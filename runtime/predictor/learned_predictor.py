"""
predictor/learned_predictor.py — Data-driven predictor using learned transition tables.

Combines two prediction signals.  In "full" mode they run SIMULTANEOUSLY and
their candidate sets are unioned; in the ablation modes exactly one runs:

1. Plan context (if available): the planner's text outlined the full tool sequence
   upfront.  If we know where we are in the plan, the next N tools are known.
   We use the learned table to calibrate the confidence; fall back to 0.80 if
   no learned data exists for this transition.

2. Learned transitions: empirical co-occurrence probabilities from JSONL traces.
   The TransitionTable maps (current_tool, offset) → [(next_tool, probability), ...]
   We look up ResourceSpecs for predicted tools via the ResourceRegistry.

3. MockPredictor fallback: when neither signal produces a prediction (e.g. first
   run on a new workflow before enough traces exist), delegate to the rule-based
   predictor so there is always some output.

Signal combination (2026-08-03)
-------------------------------
Signal 2 used to be gated on `not resources`, i.e. it only fired when Signal 1
produced nothing.  Since the plan nearly always yields something, "full" was
empirically identical to "plan_only".  Both signals now always run in full mode
and their outputs are UNIONED with deduplication on
(resource_id, consumer_step_offset), keeping the max confidence — without the
dedup a naive union double-counts every resource both signals name and silently
inflates prefetch volume.

The provenance of the combination is recorded in PredictionResult.predictor_id
(which is serialised verbatim into the prediction_result trace event), so the
ablation figure can be built from the trace alone:

    learned+plan_only        only Signal 1 produced candidates
    learned+transition_only  only Signal 2 produced candidates
    learned+both_agree       both fired and share >= 1 (resource, offset) key
    learned+both_disagree    both fired and share no (resource, offset) key

The restricted modes keep their pre-existing tags ("learned+plan" /
"learned_transition_only") so the ablation arms stay comparable with trials
recorded before this change.

Lookahead horizon
-----------------
Both signals used to iterate a hardcoded `(1, 2)`, and the table signal broke
out of the loop as soon as offset 1 produced anything — so it was never a
2-step horizon, just "offset 1, or offset 2 if 1 was empty".  The horizon is now
`range(1, lookahead + 1)` with no break, so candidates accumulate across
offsets.  `lookahead` defaults to 2 (env override: RUNTIME_PREDICTOR_LOOKAHEAD).

Confidence decay
----------------
A long horizon without damping floods the scheduler, so far-out candidates are
multiplied by `offset_decay ** (offset - _LEGACY_HORIZON)`.

The factor is DERIVED FROM THE TABLE, never hardcoded, and it is a property of
whichever table is loaded rather than a constant of this module.  Derivation:
for every (source, target) pair observed at BOTH offset 1 and some offset k>1,
take the per-step ratio (p_k / p_1) ** (1/(k-1)); the factor is the median over
those pairs.  The median rather than the mean, because the ratio distribution
has a long right tail driven by pairs whose offset-1 probability is tiny (a
denominator near zero), and one such pair dominates any mean.
`_FALLBACK_DECAY` applies only when a table has no paired evidence at all.

**Do not read a specific value out of this docstring, and do not tune one in.**
The value moves when the table is regenerated, and it has moved a lot:

    table                                   pairs   median   effect
    2026-07-07, n_traces=165                n=9     0.8404   damps beyond offset 2
    2026-08-30, n_traces=490                n=11    1.0000   NO damping at all

That second row deserves attention rather than acceptance.  At n=11 the median
is the 6th sorted ratio, and it landed exactly on a single pair whose p_k
equals its p_1 — so the damping term is currently the identity, and a
`--lookahead` beyond 2 is undamped.  The sample is also more dispersed than
before, not less (ratios span 0.084-96.6 at 490 traces vs 0.194-11.4 at 165),
so this estimator is fragile at the sample sizes the corpus actually provides.
If a long horizon is ever turned on in anger, re-derive this deliberately
instead of trusting the median of ~10 ratios.

The 165-trace table is frozen at
runtime/tests/fixtures/learned_transitions_20260707.json (and backed up at
runtime/predictor/data/_preA3_20260830/) if you need to reproduce the old
behaviour.  Recompute the live value with:

    python3 -c "from runtime.predictor.learned_predictor import LearnedPredictor as L; \
p=L(); print(p.offset_decay, p.offset_decay_provenance)"

The decay is applied only BEYOND the legacy horizon (offset > 2).  Offsets 1
and 2 keep the exact confidences the shipped code produced, which is what makes
`--lookahead 2` a pure superset of the old behaviour (added events only, none
changed) — a hard requirement for diffing new traces against archived ones.

Plan-gated combination (2026-08-03, `signals="plan_gated"`)
-----------------------------------------------------------
The union above is strictly dominated: it inherits the TABLE's recall AND the
TABLE's precision, because on all labelled facets
`cov%(full) == max(cov%(plan_only), cov%(transition_only))` exactly.  The
structural reason is that 29 distinct tools in the transition table collapse
onto only 7 distinct resources in the registry, so the two signals disagree
about *tools* while naming the *same resources*; coverage is measured in
resource space, so the union has almost no room to grow.

`signals="plan_gated"` therefore uses the plan in the OPPOSITE direction from
`_plan_confidence()`: the plan filters the TABLE's candidates instead of the
table scoring the PLAN's candidates.  Four rules, selected by `gate_mode`:

    hard   emit a table candidate only if the plan also names that resource
    soft   multiply an unsupported table candidate's confidence by
           `gate_factor`, then re-apply `min_confidence`
    cap    emit at most `gate_k` candidates per prediction point, ranked by
           confidence with a +1.0 bonus for candidates BOTH signals name
    tail   drop an unsupported table candidate only if its confidence is below
           `gate_tail`; supported candidates and the high-confidence tail of
           unsupported ones pass untouched

Two orthogonal knobs apply to every rule:

    gate_scope    "resource" (default) — plan support is checked on the
                  resource identity alone, which is the granularity `wasted%`
                  is defined at; "key" — support must match
                  (resource, consumer_step_offset).
    gate_no_plan  what to do at a prediction point where Signal 1 produced
                  NOTHING (no plan_extracted event yet, or current_tool is not
                  locatable in the plan).  "pass" (default) lets the table
                  through ungated — gate only where there is a plan to gate
                  with; "suppress" drops everything.

NOTE ON `soft` VS `tail`: replay scoring ignores confidence, so in
scripts/replay_predictor.py the ONLY effect of `soft` is which candidates fall
below `min_confidence`.  soft(f) is therefore observationally identical to
tail(min_confidence / f) offline; they differ live, where the surviving
candidate's confidence feeds scheduler priority.  Report the factor AND the
threshold together — neither alone determines what survives.

`plan_gated` is a RESTRICTED mode: like plan_only/transition_only it does NOT
fall back to MockPredictor, so its numbers isolate the gated signal.  The
comparator that holds recall fixed is `transition_only`, not `full`.

All gating code is reached only when `signals == "plan_gated"`, so
`plan_only`, `transition_only` and `full` are bit-identical to the pre-gating
checkout by construction.  Verified explicitly by
runtime/tests/test_learned_predictor_gating.py::test_existing_modes_bit_identical.

Activation
----------
    from runtime.predictor.learned_predictor import LearnedPredictor
    predictor = LearnedPredictor()          # loads default JSON files
    # or
    predictor = LearnedPredictor(
        transitions_path="path/to/learned_transitions.json",
        registry=ResourceRegistry.from_json("path/to/tool_resources.json"),
        lookahead=3,
    )
"""
from __future__ import annotations

import copy
import os
import statistics
from pathlib import Path
from typing import Any

from runtime.events import PredictionResult, ResourceSpec
from runtime.predictor.base import Predictor
from runtime.predictor.plan_extractor import PlanContext
from runtime.predictor.resource_registry import ResourceRegistry
from runtime.predictor.transition_learner import TransitionTable


_DEFAULT_TRANSITIONS_PATH = Path(__file__).parent / "data" / "learned_transitions.json"
_PLAN_CONFIDENCE_DEFAULT = 0.80   # used when plan says "next tool is X" but no learned prob
# THIS CONSTANT IS A FLOOR (see _plan_confidence: max(entry.probability, it)),
# and it sits BELOW RuntimeConfig.confidence_threshold = 0.85 — so a plan-only
# prediction can never clear the prefetch gate, by arithmetic.  On the L40S rows
# of atomagents_exp3_aligned that is 34 `confidence_below_threshold
# (0.80 < 0.85)` skips against 10 admits.
#
# It is deliberately NOT retuned here.  Raising it would buy admissions by
# asserting a confidence the data does not support, and the gate is the wrong
# abstraction anyway: the question is whether the expected saving is worth the
# GB (Eq. 1, runtime/residency/contract.py), not whether one probability clears
# one line.  The repair is in PrefetchScheduler._should_prefetch, where a
# proactive-swap prediction bypasses the gate ONLY when the executor can
# actually evict the incumbent from the GPUs.

# Horizon the pre-2026-08-03 code hardcoded.  Offsets up to and including this
# are emitted with un-decayed confidence so --lookahead 2 reproduces the old
# confidences exactly; only offsets beyond it are damped.
_LEGACY_HORIZON = 2
_DEFAULT_LOOKAHEAD = 2
_LOOKAHEAD_ENV = "RUNTIME_PREDICTOR_LOOKAHEAD"
# Used ONLY when the transition table contains no (source, target) pair observed
# at two different offsets, so no ratio can be derived.  Value = the median
# derived from the shipped table (see module docstring); documented, not tuned.
_FALLBACK_DECAY = 0.84

# Provenance tags written to PredictionResult.predictor_id in "full" mode.
_TAG_PLAN_ONLY = "learned+plan_only"
_TAG_TRANSITION_ONLY = "learned+transition_only"
_TAG_BOTH_AGREE = "learned+both_agree"
_TAG_BOTH_DISAGREE = "learned+both_disagree"
# Pre-existing tag, kept for the plan_only ablation arm so its traces stay
# comparable with trials recorded before the combination change.
_TAG_LEGACY_PLAN = "learned+plan"

# --- plan-gated mode (see module docstring) --------------------------------
_SIGNAL_MODES = ("full", "plan_only", "transition_only", "plan_gated")
_GATE_MODES = ("hard", "soft", "cap", "tail")
_GATE_SCOPES = ("resource", "key")
_GATE_NO_PLAN = ("pass", "suppress")
# Rank bonus given to a candidate BOTH signals name, in gate_mode="cap".
# 1.0 > any confidence, so agreed candidates always outrank disagreed ones and
# confidence only breaks ties within each group.  Not a tuning knob.
_CAP_AGREE_BONUS = 1.0


class LearnedPredictor(Predictor):
    """
    Predictor that uses learned transition probabilities and optional plan context.

    Falls back to MockPredictor when no learned data is available.
    """

    def __init__(
        self,
        transitions_path: str | Path | None = None,
        registry: ResourceRegistry | None = None,
        min_confidence: float = 0.30,   # discard very weak learned transitions
        signals: str = "full",          # full|plan_only|transition_only|plan_gated
        lookahead: int | None = None,   # steps ahead; None → env or _DEFAULT_LOOKAHEAD
        # --- plan_gated knobs; INERT unless signals == "plan_gated" ---------
        gate_mode: str = "hard",        # hard | soft | cap | tail
        gate_factor: float = 0.5,       # soft: multiplier for unsupported cands
        gate_k: int = 2,                # cap: max candidates per prediction point
        gate_tail: float = 0.0,         # tail: drop unsupported below this conf
        gate_scope: str = "resource",   # resource | key
        gate_no_plan: str = "pass",     # pass | suppress
        gate_cap_use_plan: bool = True,  # cap: rank with the plan-agreement bonus
    ) -> None:
        if signals not in _SIGNAL_MODES:
            raise ValueError(f"signals must be one of {_SIGNAL_MODES}, got {signals!r}")
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"gate_mode must be one of {_GATE_MODES}, got {gate_mode!r}")
        if gate_scope not in _GATE_SCOPES:
            raise ValueError(f"gate_scope must be one of {_GATE_SCOPES}, got {gate_scope!r}")
        if gate_no_plan not in _GATE_NO_PLAN:
            raise ValueError(f"gate_no_plan must be one of {_GATE_NO_PLAN}, got {gate_no_plan!r}")
        if not 0.0 < gate_factor <= 1.0:
            raise ValueError(f"gate_factor must be in (0, 1], got {gate_factor!r}")
        if int(gate_k) < 0:
            raise ValueError(f"gate_k must be >= 0, got {gate_k!r}")
        self._gate_mode = gate_mode
        self._gate_factor = float(gate_factor)
        self._gate_k = int(gate_k)
        self._gate_tail = float(gate_tail)
        self._gate_scope = gate_scope
        self._gate_no_plan = gate_no_plan
        # CONTROL for gate_mode="cap": with False the ranking ignores the plan
        # entirely and the cap becomes a pure volume cap.  Any improvement that
        # survives this control is NOT attributable to the plan signal.
        self._gate_cap_use_plan = bool(gate_cap_use_plan)
        if lookahead is None:
            lookahead = _lookahead_from_env()
        lookahead = int(lookahead)
        if lookahead < 1:
            raise ValueError(f"lookahead must be >= 1, got {lookahead}")
        self._lookahead = lookahead
        transitions_path = Path(transitions_path) if transitions_path else _DEFAULT_TRANSITIONS_PATH
        self._table = TransitionTable.load(transitions_path)
        self._registry = registry or ResourceRegistry.merged(
            ResourceRegistry.from_json(),
            ResourceRegistry.from_mock_predictor(),
        )
        self._min_confidence = min_confidence
        # Predictor-mode ablation: plan_only uses only Signal 1 (plan context);
        # transition_only uses only Signal 2 (learned table).  The mock fallback
        # (Signal 3) is disabled in both restricted modes so measured prediction
        # quality reflects the isolated signal.
        self._signals = signals
        self._has_learned_data = bool(
            self._table.tool_transitions or self._table.model_transitions
        )
        # Derived from the loaded table — see module docstring.
        self._offset_decay, self._offset_decay_provenance = _derive_offset_decay(self._table)

        # Lazy-load fallback
        self._mock: Predictor | None = None

    @property
    def predictor_id(self) -> str:
        if self._signals == "full":
            return "learned"
        if self._signals == "plan_gated":
            return f"learned_plan_gated[{self.gate_spec}]"
        return f"learned_{self._signals}"

    @property
    def gate_spec(self) -> str:
        """
        Compact, complete description of the active gating rule.

        Written into predictor_id so a trace recorded under a gated arm names
        the exact rule AND its parameters — a bare 'plan_gated' label would be
        unreproducible, since gate_mode/gate_tail/gate_factor change the
        behaviour completely.  min_confidence is included because soft and tail
        are only meaningful jointly with it.
        """
        parts = [self._gate_mode]
        if self._gate_mode == "soft":
            parts.append(f"factor={self._gate_factor:g}")
            parts.append(f"min_conf={self._min_confidence:g}")
            parts.append(f"eff_thresh={self._min_confidence / self._gate_factor:.4g}")
        elif self._gate_mode == "tail":
            parts.append(f"tail={self._gate_tail:g}")
            parts.append(f"min_conf={self._min_confidence:g}")
        elif self._gate_mode == "cap":
            parts.append(f"k={self._gate_k}")
            parts.append(f"use_plan={self._gate_cap_use_plan}")
        parts.append(f"scope={self._gate_scope}")
        parts.append(f"no_plan={self._gate_no_plan}")
        return ",".join(parts)

    @property
    def signal_mode(self) -> str:
        """Signal-combination mode: full | plan_only | transition_only | plan_gated."""
        return self._signals

    @property
    def lookahead(self) -> int:
        return self._lookahead

    @property
    def offset_decay(self) -> float:
        return self._offset_decay

    @property
    def offset_decay_provenance(self) -> str:
        """How offset_decay was obtained (which table, how many pairs)."""
        return self._offset_decay_provenance

    def _decay_for(self, offset: int) -> float:
        """Confidence multiplier for `offset`; 1.0 within the legacy horizon."""
        if offset <= _LEGACY_HORIZON:
            return 1.0
        return self._offset_decay ** (offset - _LEGACY_HORIZON)

    def predict(
        self,
        step: int,
        recent_events: list[dict[str, Any]],
        current_tool_calls: list[dict],
        task_description: str = "",
        plan_context: Any = None,
    ) -> PredictionResult:
        current_tool = _latest_tool(current_tool_calls, recent_events)
        reasoning_parts: list[str] = []
        predictor_tag = self.predictor_id

        # ------------------------------------------------------------------
        # Signal 1: Plan context
        # ------------------------------------------------------------------
        plan_resources: list[ResourceSpec] = []
        plan_reasoning = ""
        if (
            self._signals in ("full", "plan_only", "plan_gated")
            and isinstance(plan_context, PlanContext)
            and plan_context.tool_sequence
        ):
            plan_resources, plan_reasoning = self._predict_from_plan(
                plan_context, current_tool, step,
            )

        # ------------------------------------------------------------------
        # Signal 2: Learned transitions — runs SIMULTANEOUSLY with Signal 1.
        # The old `and not resources` guard made this a fallback chain, so
        # "full" was empirically identical to "plan_only".
        # ------------------------------------------------------------------
        learned_resources: list[ResourceSpec] = []
        learned_reasoning = ""
        if (
            self._signals in ("full", "transition_only", "plan_gated")
            and current_tool and self._has_learned_data
        ):
            learned_resources, learned_reasoning = self._predict_from_table(
                current_tool, step, recent_events,
            )

        # ------------------------------------------------------------------
        # Plan gating (plan_gated only).  Runs BEFORE the union so the plan's
        # own candidates are never gated by themselves.  No effect whatsoever
        # on full / plan_only / transition_only.
        # ------------------------------------------------------------------
        gate_reasoning = ""
        if self._signals == "plan_gated":
            learned_resources, gate_reasoning = self._gate_table_candidates(
                plan_resources, learned_resources,
            )
            if not learned_resources:
                learned_reasoning = ""

        # ------------------------------------------------------------------
        # Union with dedup on (resource_id, consumer_step_offset), max conf.
        # ------------------------------------------------------------------
        resources, n_shared = _union_dedup(plan_resources, learned_resources)
        if self._signals == "plan_gated" and self._gate_mode == "cap":
            resources = self._cap_candidates(resources, plan_resources, learned_resources)
        if plan_resources:
            reasoning_parts.append(plan_reasoning)
        if learned_resources:
            reasoning_parts.append(learned_reasoning)

        if self._signals == "full":
            if plan_resources and learned_resources:
                predictor_tag = _TAG_BOTH_AGREE if n_shared else _TAG_BOTH_DISAGREE
                reasoning_parts.append(
                    f"Combine[{predictor_tag.split('+')[-1]}]: plan={len(plan_resources)} "
                    f"table={len(learned_resources)} shared={n_shared} "
                    f"union={len(resources)}"
                )
            elif plan_resources:
                predictor_tag = _TAG_PLAN_ONLY
            elif learned_resources:
                predictor_tag = _TAG_TRANSITION_ONLY
        elif self._signals == "plan_gated":
            if gate_reasoning:
                reasoning_parts.append(gate_reasoning)
        elif plan_resources:
            # plan_only ablation arm: keep the tag it emitted before this change
            predictor_tag = _TAG_LEGACY_PLAN

        # ------------------------------------------------------------------
        # Signal 3: MockPredictor fallback (full mode only)
        # ------------------------------------------------------------------
        if not resources and self._signals == "full":
            mock = self._get_mock()
            result = mock.predict(
                step=step,
                recent_events=recent_events,
                current_tool_calls=current_tool_calls,
                task_description=task_description,
                plan_context=plan_context,
            )
            if result.resources:
                return PredictionResult(
                    step=result.step,
                    resources=result.resources,
                    confidence=result.confidence,
                    horizon=result.horizon,
                    predictor_id=f"{predictor_tag}(mock_fallback)",
                    reasoning=result.reasoning,
                    context_events_used=result.context_events_used,
                )
            return result

        if not resources:
            # Restricted-signal mode with no prediction: return empty result.
            return PredictionResult(
                step=step,
                resources=[],
                confidence=0.0,
                predictor_id=predictor_tag,
                context_events_used=len(recent_events),
            )

        overall_confidence = max(r.confidence for r in resources)
        return PredictionResult(
            step=step,
            resources=resources,
            confidence=overall_confidence,
            horizon=min(r.consumer_step_offset for r in resources),
            predictor_id=predictor_tag,
            reasoning="; ".join(reasoning_parts),
            context_events_used=len(recent_events),
        )

    # ------------------------------------------------------------------
    # Plan-based prediction
    # ------------------------------------------------------------------

    def _predict_from_plan(
        self,
        plan_context: PlanContext,
        current_tool: str | None,
        step: int,
    ) -> tuple[list[ResourceSpec], str]:
        """
        Use the plan sequence to predict upcoming tools.

        Tries to locate `current_tool` in the plan sequence; if found, emits
        ResourceSpecs for the tools at offsets 1..lookahead.  Offsets beyond
        the legacy horizon are damped by the table-derived decay; offsets 1-2
        keep _plan_confidence()'s calibration untouched.
        """
        seq = plan_context.tool_sequence
        if not seq:
            return [], ""

        # Find where we are in the plan.
        # If current_tool is not in the sequence (e.g. plan_task is never listed
        # in the plan steps), treat position as -1 so that offset +1 resolves to
        # the very first planned tool (index 0).
        current_idx = -1
        if current_tool:
            current_idx = plan_context.find_tool(current_tool)

        resources: list[ResourceSpec] = []
        for offset in range(1, self._lookahead + 1):
            next_tool = plan_context.tool_at_offset(current_idx, offset)
            if not next_tool:
                continue
            # Calibrate confidence from learned table if available
            conf = self._plan_confidence(current_tool or "", next_tool, offset)
            conf *= self._decay_for(offset)
            specs = self._registry.get(next_tool)
            for spec in specs:
                r = _copy_resource(spec, conf, step, offset)
                resources.append(r)

        if not resources:
            return [], ""

        src_name = seq[current_idx] if 0 <= current_idx < len(seq) else "(before plan)"
        tgt_name = seq[current_idx + 1] if 0 <= current_idx + 1 < len(seq) else seq[0] if seq else "?"
        reasoning = f"Plan[{current_idx}→{current_idx+1}]: {src_name} → {tgt_name}"
        return resources, reasoning

    def _plan_confidence(self, source: str, target: str, offset: int) -> float:
        """
        Look up learned probability for (source → target at offset).
        Fall back to _PLAN_CONFIDENCE_DEFAULT if no data exists.
        """
        entries = self._table.top_tools(source, offset, n=10)
        for entry in entries:
            if entry.target == target:
                return max(entry.probability, _PLAN_CONFIDENCE_DEFAULT)
        return _PLAN_CONFIDENCE_DEFAULT

    # ------------------------------------------------------------------
    # Table-based prediction
    # ------------------------------------------------------------------

    def _predict_from_table(
        self,
        current_tool: str,
        step: int,
        recent_events: list[dict],
    ) -> tuple[list[ResourceSpec], str]:
        """
        Use the learned transition table to predict next resources.

        Candidates ACCUMULATE across offsets 1..lookahead: the old
        `if resources: break` meant offset 2 fired only when offset 1 was
        empty, which is not a lookahead horizon at all.
        """
        resources: list[ResourceSpec] = []

        # Tool transitions across the whole horizon
        for offset in range(1, self._lookahead + 1):
            entries = self._table.top_tools(current_tool, offset, n=3)
            for entry in entries:
                conf = entry.probability * self._decay_for(offset)
                if conf < self._min_confidence:
                    continue
                specs = self._registry.get(entry.target)
                for spec in specs:
                    r = _copy_resource(spec, conf, step, offset)
                    resources.append(r)

        # Also check model-based transitions if the tool table gave nothing
        if not resources:
            current_model = _latest_model(recent_events)
            if current_model:
                for offset in range(1, self._lookahead + 1):
                    entries = self._table.top_models(current_model, offset, n=2)
                    for entry in entries:
                        conf = entry.probability * self._decay_for(offset)
                        if conf < self._min_confidence:
                            continue
                        specs = self._registry.get(entry.target)
                        for spec in specs:
                            r = _copy_resource(spec, conf, step, offset)
                            resources.append(r)

        if not resources:
            return [], ""

        targets = [r.name for r in resources]
        reasoning = f"Learned: {current_tool} → {targets}"
        return resources, reasoning

    # ------------------------------------------------------------------
    # Plan gating (signals="plan_gated" only)
    # ------------------------------------------------------------------

    def _support_keys(self, specs: list[ResourceSpec]) -> set:
        """
        The identities a candidate set 'names', at the active gate_scope.

        scope="resource": resource identity alone.  This is the granularity
        `wasted%` is defined at (a prediction is wasted if that RESOURCE is
        never realized again), so it is the granularity the gate should act on.
        scope="key": (resource, consumer_step_offset) — strictly stronger, and
        the same key `_union_dedup` uses.
        """
        if self._gate_scope == "key":
            return {_resource_key(s) for s in specs}
        return {(s.resource_id or s.name) for s in specs}

    def _spec_identity(self, spec: ResourceSpec):
        if self._gate_scope == "key":
            return _resource_key(spec)
        return spec.resource_id or spec.name

    def _gate_table_candidates(
        self,
        plan_resources: list[ResourceSpec],
        table_resources: list[ResourceSpec],
    ) -> tuple[list[ResourceSpec], str]:
        """
        Filter/downweight Signal 2's candidates using Signal 1 as the gate.

        Returns (kept, reasoning).  See module docstring for the four rules.
        """
        if not table_resources:
            return [], ""
        if not plan_resources:
            # No plan at this prediction point -> nothing to gate WITH.
            if self._gate_no_plan == "suppress":
                return [], (f"Gate[{self._gate_mode}]: plan silent, "
                            f"suppressed {len(table_resources)} table candidates")
            return list(table_resources), (
                f"Gate[{self._gate_mode}]: plan silent, "
                f"passed {len(table_resources)} table candidates through")

        supported = self._support_keys(plan_resources)
        kept: list[ResourceSpec] = []
        n_dropped = 0
        n_downweighted = 0
        for spec in table_resources:
            if self._spec_identity(spec) in supported:
                kept.append(spec)
                continue
            # --- unsupported by the plan ---
            if self._gate_mode == "hard":
                n_dropped += 1
            elif self._gate_mode == "soft":
                new_conf = spec.confidence * self._gate_factor
                if new_conf < self._min_confidence:
                    n_dropped += 1
                    continue
                damped = copy.copy(spec)
                damped.confidence = new_conf
                kept.append(damped)
                n_downweighted += 1
            elif self._gate_mode == "tail":
                # Asymmetric: only the LOW-confidence tail is suppressed;
                # a high-confidence unsupported candidate survives untouched.
                if spec.confidence < self._gate_tail:
                    n_dropped += 1
                    continue
                kept.append(spec)
            else:  # "cap": per-candidate filtering happens after the union
                kept.append(spec)

        reasoning = (f"Gate[{self.gate_spec}]: table={len(table_resources)} "
                     f"kept={len(kept)} dropped={n_dropped} "
                     f"downweighted={n_downweighted}")
        return kept, reasoning

    def _cap_candidates(
        self,
        resources: list[ResourceSpec],
        plan_resources: list[ResourceSpec],
        table_resources: list[ResourceSpec],
    ) -> list[ResourceSpec]:
        """
        Keep at most gate_k candidates, ranked by confidence with a +1.0 bonus
        for candidates BOTH signals name.

        The cap is on prediction instances the runtime would act on, so it is
        applied to the post-dedup union.  `sorted` is stable, so ties keep the
        union's plan-first ordering and the result is deterministic.
        """
        if self._gate_k <= 0:
            return []
        if len(resources) <= self._gate_k:
            return resources
        agreed: set = set()
        if self._gate_cap_use_plan:
            agreed = (self._support_keys(plan_resources)
                      & self._support_keys(table_resources))
        ranked = sorted(
            resources,
            key=lambda r: -(r.confidence
                            + (_CAP_AGREE_BONUS
                               if self._spec_identity(r) in agreed else 0.0)),
        )
        return ranked[: self._gate_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mock(self) -> Predictor:
        if self._mock is None:
            from runtime.predictor.mock_predictor import MockPredictor
            self._mock = MockPredictor("auto")
        return self._mock


# ---------------------------------------------------------------------------
# Module-level helpers (shared with mock_predictor logic)
# ---------------------------------------------------------------------------

def _resource_key(spec: ResourceSpec) -> tuple[str, int]:
    """Dedup identity: the resource itself plus WHEN it is expected."""
    return (spec.resource_id or spec.name, spec.consumer_step_offset)


def _lookahead_from_env() -> int:
    raw = os.environ.get(_LOOKAHEAD_ENV, "").strip()
    if not raw:
        return _DEFAULT_LOOKAHEAD
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_LOOKAHEAD
    return value if value >= 1 else _DEFAULT_LOOKAHEAD


def _union_dedup(
    plan_resources: list[ResourceSpec],
    table_resources: list[ResourceSpec],
) -> tuple[list[ResourceSpec], int]:
    """
    Union the two signals' candidates, deduplicating ACROSS signals on
    (resource_id, consumer_step_offset) and keeping the max confidence.

    Returns (merged, n_shared_keys).

    Cross-signal only, deliberately: each signal's own list is passed through
    untouched so the plan_only / transition_only ablation arms behave exactly
    as they did before the signals were made simultaneous.  Order is
    plan-first, then table-only extras, so a full-mode result is a superset of
    the plan-only result with identical leading entries.
    """
    merged: list[ResourceSpec] = list(plan_resources)
    if not table_resources:
        return merged, 0
    if not plan_resources:
        return list(table_resources), 0

    by_key: dict[tuple[str, int], list[ResourceSpec]] = {}
    for spec in merged:
        by_key.setdefault(_resource_key(spec), []).append(spec)

    shared: set[tuple[str, int]] = set()
    for spec in table_resources:
        key = _resource_key(spec)
        existing = by_key.get(key)
        if existing is None:
            merged.append(spec)
            by_key.setdefault(key, []).append(spec)
            continue
        shared.add(key)
        for kept in existing:
            if spec.confidence > kept.confidence:
                kept.confidence = spec.confidence
    return merged, len(shared)


def _derive_offset_decay(table: TransitionTable) -> tuple[float, str]:
    """
    Derive the per-offset confidence decay FROM THE TABLE (see module docstring).

    For every (source, target) pair observed at both offset 1 and some offset
    k > 1, the per-step ratio is (p_k / p_1) ** (1/(k-1)).  We take the median
    over all such pairs — the mean is destroyed by pairs whose offset-1
    probability is near zero (one pair in the shipped table has ratio 11.4).

    Returns (decay, provenance) where provenance names the estimator and n.
    """
    ratios: list[float] = []
    for _source, offset_map in table.tool_transitions.items():
        base = {e.target: e.probability
                for e in offset_map.get(1, []) if e.probability > 0.0}
        if not base:
            continue
        for offset, entries in offset_map.items():
            if offset <= 1:
                continue
            for entry in entries:
                p1 = base.get(entry.target)
                if not p1 or entry.probability <= 0.0:
                    continue
                ratios.append((entry.probability / p1) ** (1.0 / (offset - 1)))

    if not ratios:
        return _FALLBACK_DECAY, (
            f"fallback constant {_FALLBACK_DECAY} (table has no (source,target) "
            "pair observed at two different offsets)"
        )
    decay = statistics.median(ratios)
    # A ratio > 1 would AMPLIFY far-out predictions, which defeats the purpose;
    # clamp to (0, 1].  Not a tuning knob — a guard on a degenerate table.
    decay = min(1.0, max(0.05, decay))
    return decay, (
        f"median per-step p_k/p_1 ratio over n={len(ratios)} (source,target) "
        f"pairs in the loaded transition table"
    )


def _latest_tool(tool_calls: list[dict], recent_events: list[dict]) -> str | None:
    for call in tool_calls:
        name = call.get("name") or call.get("function", {}).get("name")
        if name:
            return name
    for ev in reversed(recent_events):
        if ev.get("event_type") in ("tool_call", "tool_end"):
            name = (ev.get("payload") or {}).get("tool")
            if name:
                return name
    return None


def _latest_model(recent_events: list[dict]) -> str | None:
    for ev in reversed(recent_events):
        if ev.get("event_type") == "llm_call":
            model = (ev.get("payload") or {}).get("model")
            if model:
                return model
    return None


def _copy_resource(
    template: ResourceSpec,
    confidence: float,
    step: int,
    offset: int | None = None,
) -> ResourceSpec:
    r = copy.copy(template)
    r.confidence = confidence
    r.consumer_step_offset = offset if offset is not None else template.consumer_step_offset
    r.expected_at_step = step + r.consumer_step_offset
    return r
