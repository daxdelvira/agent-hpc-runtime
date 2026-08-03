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
multiplied by `offset_decay ** (offset - _LEGACY_HORIZON)`.  The factor is
DERIVED FROM THE TABLE, not hardcoded: for every (source, target) pair observed
at both offset 1 and offset k>1 we take the per-step ratio
(p_k / p_1) ** (1/(k-1)) and use the median.  On the shipped table
(runtime/predictor/data/learned_transitions.json, 165 traces) that is 0.8404
from n=9 pairs (e.g. run_ase→extract_output_json: 0.1951/0.4045 = 0.4823 at
k=2, 0.2857/0.4045 → 0.8404 per step at k=3).  The median rather than the mean
is used because one pair — smiles_to_coordinate_file→extract_output_json, whose
offset-1 probability is only 0.0462 — has a ratio of 11.4 and dominates any
mean.  `_FALLBACK_DECAY` is used only when a table has no paired evidence at
all.  Recompute with:

    python3 -c "from runtime.predictor.learned_predictor import LearnedPredictor as L; \
p=L(); print(p.offset_decay, p.offset_decay_provenance)"

The decay is applied only BEYOND the legacy horizon (offset > 2).  Offsets 1
and 2 keep the exact confidences the shipped code produced, which is what makes
`--lookahead 2` a pure superset of the old behaviour (added events only, none
changed) — a hard requirement for diffing new traces against archived ones.

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
        signals: str = "full",          # "full" | "plan_only" | "transition_only"
        lookahead: int | None = None,   # steps ahead; None → env or _DEFAULT_LOOKAHEAD
    ) -> None:
        if signals not in ("full", "plan_only", "transition_only"):
            raise ValueError(f"signals must be full|plan_only|transition_only, got {signals!r}")
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
        return "learned" if self._signals == "full" else f"learned_{self._signals}"

    @property
    def signal_mode(self) -> str:
        """Signal-combination mode: full | plan_only | transition_only."""
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
            self._signals in ("full", "plan_only")
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
            self._signals in ("full", "transition_only")
            and current_tool and self._has_learned_data
        ):
            learned_resources, learned_reasoning = self._predict_from_table(
                current_tool, step, recent_events,
            )

        # ------------------------------------------------------------------
        # Union with dedup on (resource_id, consumer_step_offset), max conf.
        # ------------------------------------------------------------------
        resources, n_shared = _union_dedup(plan_resources, learned_resources)
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
