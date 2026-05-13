"""
events.py — Typed event dataclasses for the runtime JSONL event bus.

All runtime events are interleaved with existing WorkflowTracker events
in the same JSONL file, identified by their event_type field.

Existing event types (from WorkflowTracker):
  llm_start, llm_call, chain_start, chain_end, chain_error,
  tool_call, tool_end, tool_error, agent_send, phase_start, phase_end

New runtime event types (emitted by EventBus):
  prediction_result     — predictor output for this step
  checkpoint_created    — WAL record created before speculative I/O
  prefetch_decision     — scheduler decided to start or skip a prefetch
  prefetch_started      — prefetch task launched in background
  prefetch_completed    — prefetch task finished
  prediction_validated  — actual tool matched prediction
  divergence_detected   — actual tool did not match prediction
  resource_consumed     — prefetched resource was used by workflow step
  prefetch_cancelled    — in-flight prefetch cancelled due to divergence
  conservative_mode     — guard entered conservative mode after divergence
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base envelope — every runtime event is wrapped in this
# ---------------------------------------------------------------------------

@dataclass
class HpcEvent:
    run_id: str
    step: int
    epoch_time: float
    event_type: str
    payload: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "HpcEvent":
        return cls(
            run_id=d["run_id"],
            step=d["step"],
            epoch_time=d["epoch_time"],
            event_type=d["event_type"],
            payload=d["payload"],
        )

    @classmethod
    def from_jsonl(cls, line: str) -> "HpcEvent":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# ResourceSpec — describes a resource the predictor thinks will be needed
# ---------------------------------------------------------------------------

@dataclass
class ResourceSpec:
    resource_id: str                     # stable hash; used for deduplication
    resource_type: str                   # "vllm_model" | "mace_model" | "data_file" | "artifact"
    name: str                            # human-readable (e.g. "qwen_72b", "W_Zhou04.eam.alloy")
    path: str | None = None             # filesystem path for files/models
    model_endpoint: str | None = None   # vLLM endpoint URL for served models
    estimated_size_bytes: int | None = None
    estimated_load_s: float | None = None  # historical load latency; None = unknown
    expected_at_step: int = 0           # which future step will consume this
    confidence: float = 0.0             # 0.0–1.0
    cancellation_safe: bool = True      # False for vLLM models (can't abort mid-load)
    consumer_tool: str = ""             # e.g. "run_ase", "computation_task_screw_dislocation"
    consumer_step_offset: int = 1       # steps from now until needed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResourceSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# PredictionResult — output of any Predictor implementation
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    step: int
    resources: list[ResourceSpec] = field(default_factory=list)
    confidence: float = 0.0
    horizon: int = 1                    # steps ahead
    predictor_id: str = "unknown"       # "mock" | "llm:groq/llama3-8b" | "oracle"
    reasoning: str = ""                 # free-text (for LLM predictor)
    context_events_used: int = 0        # how many recent events were used

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionResult":
        resources = [ResourceSpec.from_dict(r) for r in d.get("resources", [])]
        return cls(
            step=d["step"],
            resources=resources,
            confidence=d.get("confidence", 0.0),
            horizon=d.get("horizon", 1),
            predictor_id=d.get("predictor_id", "unknown"),
            reasoning=d.get("reasoning", ""),
            context_events_used=d.get("context_events_used", 0),
        )


# ---------------------------------------------------------------------------
# Helpers to build HpcEvent payloads for each runtime event type
# ---------------------------------------------------------------------------

def make_prediction_result_event(
    run_id: str,
    step: int,
    result: PredictionResult,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prediction_result",
        payload=result.to_dict(),
    )


def make_checkpoint_created_event(
    run_id: str,
    step: int,
    checkpoint_id: str,
    log_position: int,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="checkpoint_created",
        payload={"checkpoint_id": checkpoint_id, "step": step, "log_position": log_position},
    )


def make_prefetch_decision_event(
    run_id: str,
    step: int,
    resource_id: str,
    action: str,          # "start" | "skip"
    reason: str,
    estimated_load_s: float | None = None,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prefetch_decision",
        payload={
            "resource_id": resource_id,
            "action": action,
            "reason": reason,
            "estimated_load_s": estimated_load_s,
        },
    )


def make_prefetch_started_event(
    run_id: str,
    step: int,
    task_id: str,
    resource_id: str,
    executor: str,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prefetch_started",
        payload={"task_id": task_id, "resource_id": resource_id, "executor": executor},
    )


def make_prefetch_completed_event(
    run_id: str,
    step: int,
    task_id: str,
    elapsed_s: float,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prefetch_completed",
        payload={"task_id": task_id, "elapsed_s": elapsed_s},
    )


def make_prediction_validated_event(
    run_id: str,
    step: int,
    hit: bool,
    checkpoint_id: str,
    actual_tool: str,
    overlap_s: float = 0.0,
    benefit_s: float = 0.0,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prediction_validated",
        payload={
            "hit": hit,
            "checkpoint_id": checkpoint_id,
            "actual_tool": actual_tool,
            "overlap_s": overlap_s,
            "benefit_s": benefit_s,
        },
    )


def make_divergence_detected_event(
    run_id: str,
    step: int,
    expected_tool: str,
    actual_tool: str,
    checkpoint_id: str,
    action: str,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="divergence_detected",
        payload={
            "expected_tool": expected_tool,
            "actual_tool": actual_tool,
            "checkpoint_id": checkpoint_id,
            "action": action,
        },
    )


def make_resource_consumed_event(
    run_id: str,
    step: int,
    resource_id: str,
    task_id: str | None,
    status: str,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="resource_consumed",
        payload={"resource_id": resource_id, "task_id": task_id, "status": status},
    )


def make_prefetch_cancelled_event(
    run_id: str,
    step: int,
    task_id: str,
    reason: str,
    status_before: str,
    wasted: bool,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="prefetch_cancelled",
        payload={
            "task_id": task_id,
            "reason": reason,
            "status_before": status_before,
            "wasted": wasted,
        },
    )


def make_conservative_mode_event(
    run_id: str,
    step: int,
    reason: str,
    duration_steps: int,
) -> HpcEvent:
    return HpcEvent(
        run_id=run_id,
        step=step,
        epoch_time=time.time(),
        event_type="conservative_mode",
        payload={"reason": reason, "duration_steps": duration_steps},
    )
