"""
predictor/mock_predictor.py — Rule-based stub predictor; zero external deps.

Encodes known tool-sequence patterns for ChemGraph and AtomAgents as static
transition tables. This lets us exercise the full prediction → prefetch →
divergence pipeline locally without any LLM calls.

ChemGraph standard sequences (single_agent):
  molecule_name_to_smiles      → smiles_to_coordinate_file (mace_model: mace-mp-0)
  smiles_to_coordinate_file    → run_ase                   (mace_model: mace-mp-0)
  run_ase                      → extract_output_json | done

AtomAgents Exp2 sequences:
  plan_task                    → computation_task_screw_dislocation (vllm_model: qwen_72b + data: EAM files)
  computation_task_screw_dislocation → plan_task | done
"""
from __future__ import annotations

import hashlib
from typing import Any

from runtime.events import PredictionResult, ResourceSpec
from runtime.predictor.base import Predictor


# ---------------------------------------------------------------------------
# Known resource descriptors
# ---------------------------------------------------------------------------

def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ChemGraph: MACE model loaded by run_ase
_MACE_MP0 = ResourceSpec(
    resource_id=_hash("mace-mp-0"),
    resource_type="mace_model",
    name="mace-mp-0",
    path=None,           # populated from config at runtime if available
    estimated_load_s=40.0,
    confidence=0.0,      # set per-prediction
    cancellation_safe=True,
    consumer_tool="run_ase",
    consumer_step_offset=1,
)

# Same resource, 2-step lookahead (molecule→smiles→run_ase)
_MACE_MP0_2 = ResourceSpec(
    resource_id=_hash("mace-mp-0"),
    resource_type="mace_model",
    name="mace-mp-0",
    path=None,
    estimated_load_s=40.0,
    confidence=0.0,
    cancellation_safe=True,
    consumer_tool="run_ase",
    consumer_step_offset=2,
)

# AtomAgents: qwen_72b (engineer/admin) — primary reasoning model
_QWEN_72B = ResourceSpec(
    resource_id=_hash("qwen_72b"),
    resource_type="vllm_model",
    name="qwen_72b",
    model_endpoint="http://localhost:8001",
    estimated_load_s=2700.0,   # up to 45 min cold NFS load
    confidence=0.0,
    cancellation_safe=False,   # can't abort mid-load
    consumer_tool="llm_call",
    consumer_step_offset=1,
)

# AtomAgents: qwen_32b (planner/critic) — planning model
_QWEN_32B = ResourceSpec(
    resource_id=_hash("qwen_32b"),
    resource_type="vllm_model",
    name="qwen_32b",
    model_endpoint="http://localhost:8002",
    estimated_load_s=1200.0,   # ~20 min cold NFS load
    confidence=0.0,
    cancellation_safe=False,
    consumer_tool="llm_call",
    consumer_step_offset=2,
)

# AtomAgents: EAM potential files for screw dislocation
_W_ZHOU04 = ResourceSpec(
    resource_id=_hash("W_Zhou04.eam.alloy"),
    resource_type="data_file",
    name="W_Zhou04.eam.alloy",
    path=None,           # populated from config or env at runtime
    estimated_load_s=2.0,
    confidence=0.0,
    cancellation_safe=True,
    consumer_tool="computation_task_screw_dislocation",
    consumer_step_offset=1,
)

_W_EAM4 = ResourceSpec(
    resource_id=_hash("w_eam4.fs"),
    resource_type="data_file",
    name="w_eam4.fs",
    path=None,
    estimated_load_s=1.5,
    confidence=0.0,
    cancellation_safe=True,
    consumer_tool="computation_task_screw_dislocation",
    consumer_step_offset=1,
)


# ---------------------------------------------------------------------------
# Transition tables
# ---------------------------------------------------------------------------

# ChemGraph: tool_name → list of (resource, confidence)
_CHEMGRAPH_TRANSITIONS: dict[str, list[tuple[ResourceSpec, float]]] = {
    "molecule_name_to_smiles":   [(_MACE_MP0_2, 0.80)],  # smiles_to_coords → run_ase (2 steps ahead)
    "smiles_to_coordinate_file": [(_MACE_MP0, 0.92)],   # almost always followed by run_ase
    "smiles_to_atomsdata":       [(_MACE_MP0, 0.90)],
    "file_to_atomsdata":         [(_MACE_MP0, 0.88)],
}

# AtomAgents: tool_name → list of (resource, confidence)
_ATOMAGENTS_TRANSITIONS: dict[str, list[tuple[ResourceSpec, float]]] = {
    "plan_task": [
        (_W_ZHOU04, 0.87),
        (_W_EAM4,   0.87),
    ],
    # After screw dislocation calc completes, planner may want 32b again
    "computation_task_screw_dislocation": [
        (_QWEN_32B, 0.65),
    ],
}

# Mapping from model names observed in LLM events to likely next model
_MODEL_TRANSITIONS: dict[str, list[tuple[ResourceSpec, float]]] = {
    "Qwen/Qwen2.5-VL-72B-Instruct": [(_QWEN_32B, 0.70)],
    "Qwen/Qwen2.5-VL-32B-Instruct": [(_QWEN_72B, 0.60)],
}


class MockPredictor(Predictor):
    """
    Rule-based stub predictor.

    Uses the last tool_call names from recent events and the current tool_calls
    to look up the next likely resources in the transition tables.
    """

    def __init__(self, workflow: str = "chemgraph") -> None:
        """
        workflow: "chemgraph" | "atomagents" | "auto"
          "auto" inspects recent events to decide which table to use.
        """
        assert workflow in ("chemgraph", "atomagents", "auto")
        self._workflow = workflow

    @property
    def predictor_id(self) -> str:
        return "mock"

    def predict(
        self,
        step: int,
        recent_events: list[dict[str, Any]],
        current_tool_calls: list[dict],
        task_description: str = "",
    ) -> PredictionResult:
        workflow = self._workflow
        if workflow == "auto":
            workflow = _infer_workflow(recent_events)

        table = _CHEMGRAPH_TRANSITIONS if workflow == "chemgraph" else _ATOMAGENTS_TRANSITIONS

        # Find the most recent tool name from current_tool_calls or recent events
        current_tool = _latest_tool(current_tool_calls, recent_events)

        resources: list[ResourceSpec] = []

        # Tool-based lookup
        if current_tool and current_tool in table:
            for template, conf in table[current_tool]:
                r = _copy_resource(template, conf, step)
                resources.append(r)

        # Model-based lookup (for AtomAgents: predict next model based on current)
        if workflow == "atomagents" and not resources:
            current_model = _latest_model(recent_events)
            if current_model and current_model in _MODEL_TRANSITIONS:
                for template, conf in _MODEL_TRANSITIONS[current_model]:
                    r = _copy_resource(template, conf, step)
                    resources.append(r)

        if not resources:
            return PredictionResult(
                step=step,
                resources=[],
                confidence=0.0,
                horizon=1,
                predictor_id=self.predictor_id,
                context_events_used=len(recent_events),
            )

        overall_confidence = max(r.confidence for r in resources)
        return PredictionResult(
            step=step,
            resources=resources,
            confidence=overall_confidence,
            horizon=min(r.consumer_step_offset for r in resources),
            predictor_id=self.predictor_id,
            reasoning=f"Rule: {current_tool} → {[r.name for r in resources]}",
            context_events_used=len(recent_events),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_resource(template: ResourceSpec, confidence: float, step: int) -> ResourceSpec:
    import copy
    r = copy.copy(template)
    r.confidence = confidence
    r.expected_at_step = step + r.consumer_step_offset
    return r


def _latest_tool(tool_calls: list[dict], recent_events: list[dict]) -> str | None:
    """Return the name of the most recently called tool."""
    # From current LLM response
    for call in tool_calls:
        name = call.get("name") or call.get("function", {}).get("name")
        if name:
            return name
    # From recent JSONL events
    for ev in reversed(recent_events):
        if ev.get("event_type") in ("tool_call", "tool_end"):
            name = (ev.get("payload") or {}).get("tool")
            if name:
                return name
    return None


def _latest_model(recent_events: list[dict]) -> str | None:
    """Return the model name from the most recent llm_call event."""
    for ev in reversed(recent_events):
        if ev.get("event_type") == "llm_call":
            model = (ev.get("payload") or {}).get("model")
            if model:
                return model
    return None


def _infer_workflow(recent_events: list[dict]) -> str:
    """Guess workflow type from event payloads."""
    for ev in recent_events:
        payload = ev.get("payload") or {}
        tool = payload.get("tool", "")
        if tool in ("run_ase", "molecule_name_to_smiles", "smiles_to_coordinate_file"):
            return "chemgraph"
        if tool in ("plan_task", "computation_task_screw_dislocation"):
            return "atomagents"
        sender = payload.get("sender", "")
        if sender in ("engineer_core", "admin_core"):
            return "atomagents"
    return "chemgraph"
