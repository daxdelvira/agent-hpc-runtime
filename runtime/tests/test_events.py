"""
test_events.py — Phase 0 acceptance tests: all dataclasses serialize and round-trip.
"""
import json
import time

import pytest

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.events import (
    HpcEvent,
    PredictionResult,
    ResourceSpec,
    make_checkpoint_created_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prefetch_cancelled_event,
    make_prefetch_completed_event,
    make_prefetch_decision_event,
    make_prefetch_started_event,
    make_prediction_validated_event,
    make_resource_consumed_event,
    make_conservative_mode_event,
)
from runtime.guard.checkpoint import CheckpointRecord, CheckpointStore
from runtime.measurement.timings import PrefetchTimingRecord
from runtime.prefetch.base import PrefetchStatus, PrefetchTask


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

def test_runtime_config_defaults():
    cfg = RuntimeConfig()
    assert cfg.mode == RuntimeMode.SIMULATED
    assert cfg.confidence_threshold == 0.85
    assert len(cfg.run_id) > 0


def test_runtime_config_baseline():
    cfg = RuntimeConfig(mode=RuntimeMode.BASELINE)
    assert cfg.mode == RuntimeMode.BASELINE


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------

def test_resource_spec_round_trip():
    r = ResourceSpec(
        resource_id="mace-mp0",
        resource_type="mace_model",
        name="mace-mp-0",
        path="/models/mace-mp-0.model",
        estimated_load_s=35.0,
        expected_at_step=3,
        confidence=0.9,
        cancellation_safe=True,
        consumer_tool="run_ase",
        consumer_step_offset=1,
    )
    d = r.to_dict()
    r2 = ResourceSpec.from_dict(d)
    assert r2.resource_id == r.resource_id
    assert r2.estimated_load_s == 35.0
    assert r2.confidence == 0.9


def test_resource_spec_minimal():
    r = ResourceSpec(resource_id="x", resource_type="data_file", name="file.xyz")
    d = r.to_dict()
    assert "resource_id" in d
    assert d["path"] is None


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

def test_prediction_result_empty():
    p = PredictionResult(step=1)
    d = p.to_dict()
    assert d["step"] == 1
    assert d["resources"] == []
    p2 = PredictionResult.from_dict(d)
    assert p2.step == 1
    assert p2.resources == []


def test_prediction_result_with_resources():
    r = ResourceSpec(resource_id="qwen72b", resource_type="vllm_model", name="qwen_72b",
                     model_endpoint="http://localhost:8001", estimated_load_s=2700.0,
                     confidence=0.8, cancellation_safe=False, consumer_tool="llm_call")
    p = PredictionResult(
        step=2,
        resources=[r],
        confidence=0.8,
        horizon=2,
        predictor_id="mock",
        reasoning="After plan_task, computation tool follows",
        context_events_used=5,
    )
    d = p.to_dict()
    p2 = PredictionResult.from_dict(d)
    assert len(p2.resources) == 1
    assert p2.resources[0].resource_id == "qwen72b"
    assert p2.resources[0].cancellation_safe is False
    assert p2.predictor_id == "mock"


# ---------------------------------------------------------------------------
# HpcEvent round-trip (JSONL)
# ---------------------------------------------------------------------------

def test_hpc_event_jsonl_round_trip():
    ev = HpcEvent(
        run_id="run-001",
        step=3,
        epoch_time=time.time(),
        event_type="prediction_result",
        payload={"test": True},
    )
    line = ev.to_jsonl()
    ev2 = HpcEvent.from_jsonl(line)
    assert ev2.run_id == "run-001"
    assert ev2.step == 3
    assert ev2.event_type == "prediction_result"
    assert ev2.payload["test"] is True


# ---------------------------------------------------------------------------
# All make_*_event helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def run_id():
    return "test-run-42"


def test_make_prediction_result_event(run_id):
    r = ResourceSpec(resource_id="mace-mp0", resource_type="mace_model", name="mace-mp-0")
    p = PredictionResult(step=1, resources=[r], confidence=0.9, predictor_id="mock")
    ev = make_prediction_result_event(run_id, 1, p)
    assert ev.event_type == "prediction_result"
    assert ev.run_id == run_id
    parsed = json.loads(ev.to_jsonl())
    assert parsed["payload"]["resources"][0]["resource_id"] == "mace-mp0"


def test_make_checkpoint_created_event(run_id):
    ev = make_checkpoint_created_event(run_id, 1, "ckpt-abc", 8192)
    assert ev.event_type == "checkpoint_created"
    assert ev.payload["checkpoint_id"] == "ckpt-abc"
    assert ev.payload["log_position"] == 8192


def test_make_prefetch_decision_event(run_id):
    ev = make_prefetch_decision_event(run_id, 1, "mace-mp0", "start", "confidence_above_threshold", 35.0)
    assert ev.event_type == "prefetch_decision"
    assert ev.payload["action"] == "start"
    assert ev.payload["estimated_load_s"] == 35.0


def test_make_prefetch_started_event(run_id):
    ev = make_prefetch_started_event(run_id, 1, "pf-001", "mace-mp0", "simulated")
    assert ev.event_type == "prefetch_started"
    assert ev.payload["executor"] == "simulated"


def test_make_prefetch_completed_event(run_id):
    ev = make_prefetch_completed_event(run_id, 1, "pf-001", 0.001)
    assert ev.event_type == "prefetch_completed"


def test_make_prediction_validated_event(run_id):
    ev = make_prediction_validated_event(run_id, 2, True, "ckpt-abc", "run_ase", 31.2, 28.1)
    assert ev.event_type == "prediction_validated"
    assert ev.payload["hit"] is True
    assert ev.payload["overlap_s"] == 31.2


def test_make_divergence_detected_event(run_id):
    ev = make_divergence_detected_event(run_id, 2, "run_ase", "molecule_name_to_smiles", "ckpt-abc", "INVALIDATE_ALL")
    assert ev.event_type == "divergence_detected"
    assert ev.payload["action"] == "INVALIDATE_ALL"


def test_make_resource_consumed_event(run_id):
    ev = make_resource_consumed_event(run_id, 3, "mace-mp0", "pf-001", "used")
    assert ev.event_type == "resource_consumed"
    assert ev.payload["status"] == "used"


def test_make_prefetch_cancelled_event(run_id):
    ev = make_prefetch_cancelled_event(run_id, 2, "pf-001", "divergence", "in_progress", True)
    assert ev.event_type == "prefetch_cancelled"
    assert ev.payload["wasted"] is True


def test_make_conservative_mode_event(run_id):
    ev = make_conservative_mode_event(run_id, 2, "divergence", 3)
    assert ev.event_type == "conservative_mode"
    assert ev.payload["duration_steps"] == 3


# ---------------------------------------------------------------------------
# PrefetchTask + PrefetchStatus
# ---------------------------------------------------------------------------

def test_prefetch_task_defaults():
    r = ResourceSpec(resource_id="f", resource_type="data_file", name="file.xyz")
    t = PrefetchTask(resource=r)
    assert t.status == PrefetchStatus.PENDING
    assert t.started_at is None
    assert t.overlap_s() == 0.0
    assert t.benefit_s() == 0.0
    assert t.waste_s() == 0.0


def test_prefetch_task_overlap_computation():
    r = ResourceSpec(resource_id="f", resource_type="data_file", name="file.xyz")
    t = PrefetchTask(resource=r)
    t.started_at = 100.0
    t.completed_at = 130.0     # 30s load
    t.consumed_at = 150.0      # needed 20s after completion
    # benefit: prefetch finished 20s before needed → saved 20s
    assert t.benefit_s() == pytest.approx(20.0)
    # waste: 0 (prefetch completed before needed)
    assert t.waste_s() == pytest.approx(0.0)


def test_prefetch_task_late_prefetch():
    r = ResourceSpec(resource_id="f", resource_type="data_file", name="file.xyz")
    t = PrefetchTask(resource=r)
    t.started_at = 100.0
    t.consumed_at = 120.0      # needed before prefetch could finish
    t.completed_at = 140.0     # prefetch arrived 20s late
    assert t.benefit_s() == pytest.approx(0.0)
    assert t.waste_s() == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# CheckpointRecord + CheckpointStore
# ---------------------------------------------------------------------------

def test_checkpoint_store_add_and_get():
    store = CheckpointStore(max_horizon=3)
    r = PredictionResult(step=2, confidence=0.9, horizon=1, predictor_id="mock")
    ckpt = CheckpointRecord(step=2, prediction=r)
    store.add(ckpt)
    found = store.get(ckpt.checkpoint_id)
    assert found is not None
    assert found.step == 2


def test_checkpoint_store_resolve():
    store = CheckpointStore()
    ckpt = CheckpointRecord(step=1)
    store.add(ckpt)
    store.resolve(ckpt.checkpoint_id, "validated")
    found = store.get(ckpt.checkpoint_id)
    assert found.status == "validated"
    assert found.resolved_at is not None


def test_checkpoint_store_expire():
    store = CheckpointStore(max_horizon=2)
    p = PredictionResult(step=1, horizon=1)
    ckpt = CheckpointRecord(step=1, prediction=p)
    store.add(ckpt)
    expired = store.expire_old(current_step=5)
    assert len(expired) == 1
    assert expired[0].status == "expired"


# ---------------------------------------------------------------------------
# PrefetchTimingRecord derived metrics
# ---------------------------------------------------------------------------

def test_timing_record_overlap():
    rec = PrefetchTimingRecord(
        run_id="r1",
        resource_id="qwen72b",
        prefetch_start_t=0.0,
        prefetch_end_t=1200.0,    # 20 min load
        resource_needed_t=2700.0, # needed 45 min into job
        hit=True,
    )
    assert rec.benefit_s == pytest.approx(1500.0)
    assert rec.waste_s == pytest.approx(0.0)
    assert rec.overlap_s == pytest.approx(1200.0)


def test_timing_record_cancelled():
    rec = PrefetchTimingRecord(
        run_id="r1",
        resource_id="qwen72b",
        prefetch_start_t=0.0,
        prefetch_end_t=1200.0,
        resource_needed_t=2700.0,
        cancelled=True,
    )
    assert rec.benefit_s == pytest.approx(0.0)
    assert rec.overlap_s == pytest.approx(0.0)
