"""
predictor/learned_predictor.py — Data-driven predictor using learned transition tables.

Combines two prediction signals in priority order:

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

Activation
----------
    from runtime.predictor.learned_predictor import LearnedPredictor
    predictor = LearnedPredictor()          # loads default JSON files
    # or
    predictor = LearnedPredictor(
        transitions_path="path/to/learned_transitions.json",
        registry=ResourceRegistry.from_json("path/to/tool_resources.json"),
    )
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from runtime.events import PredictionResult, ResourceSpec
from runtime.predictor.base import Predictor
from runtime.predictor.plan_extractor import PlanContext
from runtime.predictor.resource_registry import ResourceRegistry
from runtime.predictor.transition_learner import TransitionTable


_DEFAULT_TRANSITIONS_PATH = Path(__file__).parent / "data" / "learned_transitions.json"
_PLAN_CONFIDENCE_DEFAULT = 0.80   # used when plan says "next tool is X" but no learned prob


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
    ) -> None:
        transitions_path = Path(transitions_path) if transitions_path else _DEFAULT_TRANSITIONS_PATH
        self._table = TransitionTable.load(transitions_path)
        self._registry = registry or ResourceRegistry.merged(
            ResourceRegistry.from_json(),
            ResourceRegistry.from_mock_predictor(),
        )
        self._min_confidence = min_confidence
        self._has_learned_data = bool(
            self._table.tool_transitions or self._table.model_transitions
        )

        # Lazy-load fallback
        self._mock: Predictor | None = None

    @property
    def predictor_id(self) -> str:
        return "learned"

    def predict(
        self,
        step: int,
        recent_events: list[dict[str, Any]],
        current_tool_calls: list[dict],
        task_description: str = "",
        plan_context: Any = None,
    ) -> PredictionResult:
        current_tool = _latest_tool(current_tool_calls, recent_events)
        resources: list[ResourceSpec] = []
        reasoning_parts: list[str] = []
        predictor_tag = "learned"

        # ------------------------------------------------------------------
        # Signal 1: Plan context — highest priority
        # ------------------------------------------------------------------
        if isinstance(plan_context, PlanContext) and plan_context.tool_sequence:
            plan_resources, plan_reasoning = self._predict_from_plan(
                plan_context, current_tool, step,
            )
            if plan_resources:
                resources.extend(plan_resources)
                reasoning_parts.append(plan_reasoning)
                predictor_tag = "learned+plan"

        # ------------------------------------------------------------------
        # Signal 2: Learned transitions (fill gaps not covered by plan)
        # ------------------------------------------------------------------
        if not resources and current_tool and self._has_learned_data:
            learned_resources, learned_reasoning = self._predict_from_table(
                current_tool, step, recent_events,
            )
            if learned_resources:
                resources.extend(learned_resources)
                reasoning_parts.append(learned_reasoning)

        # ------------------------------------------------------------------
        # Signal 3: MockPredictor fallback
        # ------------------------------------------------------------------
        if not resources:
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

        Tries to locate `current_tool` in the plan sequence; if found,
        emits ResourceSpecs for the tools at offsets 1 and 2.
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
        for offset in (1, 2):
            next_tool = plan_context.tool_at_offset(current_idx, offset)
            if not next_tool:
                continue
            # Calibrate confidence from learned table if available
            conf = self._plan_confidence(current_tool or "", next_tool, offset)
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
        """Use the learned transition table to predict next resources."""
        resources: list[ResourceSpec] = []

        # Tool transitions at offset 1 (and 2 if nothing at offset 1)
        for offset in (1, 2):
            entries = self._table.top_tools(current_tool, offset, n=3)
            for entry in entries:
                if entry.probability < self._min_confidence:
                    continue
                specs = self._registry.get(entry.target)
                for spec in specs:
                    r = _copy_resource(spec, entry.probability, step, offset)
                    resources.append(r)
            if resources:
                break

        # Also check model-based transitions if still empty
        if not resources:
            current_model = _latest_model(recent_events)
            if current_model:
                for offset in (1, 2):
                    entries = self._table.top_models(current_model, offset, n=2)
                    for entry in entries:
                        if entry.probability < self._min_confidence:
                            continue
                        specs = self._registry.get(entry.target)
                        for spec in specs:
                            r = _copy_resource(spec, entry.probability, step, offset)
                            resources.append(r)
                    if resources:
                        break

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
