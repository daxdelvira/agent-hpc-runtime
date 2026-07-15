"""
predictor/oracle_predictor.py — Oracle predictor: perfect hindsight from a reference trace.

Reads a completed JSONL workflow trace and replays it with confidence=1.0 to
establish an upper-bound estimate: "if predictions were always correct, what
would the maximum achievable prefetch benefit be?"

Usage
-----
    from runtime.predictor.oracle_predictor import OraclePredictor

    # Replay from a baseline trace collected on the cluster
    predictor = OraclePredictor("logs/workflow_traces/baseline_trace.jsonl")
    result = predictor.predict(step=2, recent_events=[], current_tool_calls=[])
    # result.resources contains the actual resource consumed at step 3 in the trace

    # Compare with:
    #   MockPredictor accuracy  → real accuracy achievable with rule-based prediction
    #   OraclePredictor         → upper bound (100% prediction accuracy)
    #   Baseline (no prefetch)  → lower bound

How it works
------------
At construction, the oracle reads all `tool_call` events from the reference trace
and builds a step-ordered list: [(step, tool_name), ...].

At predict(step=N), it looks up what tool executed at step N+1 (the consumer step)
in the reference trace, finds the matching ResourceSpec, and returns it with
confidence=1.0.

If the current run differs from the reference trace (different tool sequence),
the oracle returns an empty prediction rather than a misleading one.

Limitations
-----------
- Only useful for replaying a trace that is structurally identical to the run being
  evaluated. For novel runs, use MockPredictor or LLMPredictor.
- Does not model the timing of the original run; it only knows tool order.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from runtime.events import PredictionResult, ResourceSpec
from runtime.predictor.base import Predictor
from runtime.predictor.mock_predictor import (
    _CHEMGRAPH_TRANSITIONS,
    _ATOMAGENTS_TRANSITIONS,
    _copy_resource,
)


class OraclePredictor(Predictor):
    """
    Perfect-hindsight predictor that reads tool execution order from a
    reference JSONL trace.

    Parameters
    ----------
    trace_path : path to a completed JSONL trace file.
    workflow   : "chemgraph" | "atomagents" | "auto" — used to look up which
                 ResourceSpec corresponds to each tool name.
    """

    def __init__(self, trace_path: str, workflow: str = "auto") -> None:
        self._trace_path = str(trace_path)
        self._workflow = workflow
        self._step_to_tool: dict[int, str] = {}
        self._load_trace()

    def _load_trace(self) -> None:
        path = Path(self._trace_path)
        if not path.exists():
            return
        seq = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("event_type") == "tool_call":
                    tool = ev.get("payload", {}).get("tool", "")
                    if not tool:
                        continue
                    seq += 1
                    # WorkflowTracker tool_call events carry no "step" field —
                    # only runtime events do.  The runtime adapter increments its
                    # step counter once per tool start, so the k-th tool_call in
                    # file order IS step k.  Use the explicit step when present,
                    # else fall back to the sequential index.
                    step = ev.get("step") or seq
                    self._step_to_tool[step] = tool

    @property
    def predictor_id(self) -> str:
        return "oracle"

    def predict(
        self,
        step: int,
        recent_events: list[dict],
        current_tool_calls: list[dict],
        task_description: str = "",
        plan_context=None,   # accepted for interface parity; oracle ignores it
    ) -> PredictionResult:
        """
        Return a perfect prediction for what resource will be needed next.

        Looks up what tool executes at step+1 in the reference trace, then
        finds the corresponding ResourceSpec using the same transition tables
        as MockPredictor. Returns an empty result if the next step is unknown.
        """
        # Determine next consumer step: default offset=1
        next_step = step + 1
        consumer_tool = self._step_to_tool.get(next_step)

        if consumer_tool is None:
            return PredictionResult(
                step=step,
                resources=[],
                confidence=0.0,
                horizon=1,
                predictor_id=self.predictor_id,
                context_events_used=len(recent_events),
            )

        # Infer workflow if auto
        workflow = self._workflow
        if workflow == "auto":
            workflow = _infer_workflow(recent_events, current_tool_calls)

        # Find the resource that consumer_tool consumes
        resources = _lookup_resource_for_consumer(
            consumer_tool=consumer_tool,
            workflow=workflow,
            step=step,
        )

        if not resources:
            return PredictionResult(
                step=step,
                resources=[],
                confidence=0.0,
                horizon=1,
                predictor_id=self.predictor_id,
                context_events_used=len(recent_events),
            )

        # Oracle always has confidence=1.0 (by definition)
        for r in resources:
            r.confidence = 1.0

        return PredictionResult(
            step=step,
            resources=resources,
            confidence=1.0,
            horizon=1,
            predictor_id=self.predictor_id,
            reasoning=f"Oracle: tool at step {next_step} is '{consumer_tool}'",
            context_events_used=len(recent_events),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def trace_length(self) -> int:
        """Number of tool_call events parsed from the reference trace."""
        return len(self._step_to_tool)

    def tool_at_step(self, step: int) -> str | None:
        """Return the tool name at the given step (or None)."""
        return self._step_to_tool.get(step)

    def all_steps(self) -> list[tuple[int, str]]:
        """Return all (step, tool_name) pairs in step order."""
        return sorted(self._step_to_tool.items())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_workflow(recent_events: list[dict], current_tool_calls: list[dict]) -> str:
    """Infer workflow from event/tool names (mirrors MockPredictor logic)."""
    chemgraph_tools = {"smiles_to_coordinate_file", "molecule_name_to_smiles", "run_ase"}
    atomagents_tools = {"plan_task", "computation_task_screw_dislocation"}

    for ev in recent_events:
        tool = ev.get("payload", {}).get("tool", "")
        if tool in chemgraph_tools:
            return "chemgraph"
        if tool in atomagents_tools:
            return "atomagents"

    for call in current_tool_calls:
        name = call.get("name", "")
        if name in chemgraph_tools:
            return "chemgraph"
        if name in atomagents_tools:
            return "atomagents"

    return "chemgraph"   # safe default


def _lookup_resource_for_consumer(
    consumer_tool: str,
    workflow: str,
    step: int,
) -> list[ResourceSpec]:
    """
    Find ResourceSpec entries whose consumer_tool matches.

    Searches ChemGraph or AtomAgents transition tables for entries where
    the predicted resource has consumer_tool == consumer_tool.
    Returns copies with correct step metadata.
    """
    table = (
        _ATOMAGENTS_TRANSITIONS if workflow == "atomagents"
        else _CHEMGRAPH_TRANSITIONS
    )

    seen_ids: set[str] = set()
    results: list[ResourceSpec] = []

    for _trigger_tool, entries in table.items():
        for template, _ in entries:
            if template.consumer_tool != consumer_tool:
                continue
            if template.resource_id in seen_ids:
                continue
            seen_ids.add(template.resource_id)
            r = _copy_resource(template, confidence=1.0, step=step)
            results.append(r)

    return results
