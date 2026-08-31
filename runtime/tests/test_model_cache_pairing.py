"""
test_model_cache_pairing.py — AtomAgentsRuntimeAdapter._pair_model_cache.

AtomAgents historically emitted only `vllm_model` and `data_file` resources, so
the host-I/O half of a model load had no resource to attach to. That is what
blocked an external-tier comparison on this workload: MegaMmapStagingExecutor is
a drop-in for `model_cache`, and with no such resource there was nothing for it
to replace.

The tests below pin the invariants whose violation is SILENT -- each would keep
producing plausible traces while corrupting the stall taxonomy:

  * name equality with the paired engine. extract_prefetch_lifecycle.py groups a
    staging task with its engine task by model name and takes max() over the
    group rather than summing, because both gate the same wall interval. A
    mismatched name splits the group and counts one stall twice.
  * no duplicate staging for a model already staged. Shards are immutable; a
    second warm can only re-read cached bytes, and it would add a spurious task
    to the taxonomy.
  * the ablation and baseline switches actually suppress it, so
    skip_resource_types=["model_cache"] means what it says.

No autogen, no vLLM, no GPU.
"""
from __future__ import annotations

import pytest

from runtime.adapters.atomagents import AtomAgentsRuntimeAdapter, _model_shard_bytes
from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import PredictionResult, ResourceSpec
from runtime.guard.detector import DivergenceDetector
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.predictor.mock_predictor import MockPredictor


def _adapter(tmp_path, **cfg_kw):
    kw = dict(mode=RuntimeMode.SIMULATED, run_id="test-cache-pair",
              stage_worker_cache=True)
    kw.update(cfg_kw)
    cfg = RuntimeConfig(**kw)
    bus = EventBus(run_id=cfg.run_id, log_path=str(tmp_path / "trace.jsonl"))
    executor = SimulatedPrefetchExecutor()
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)
    return AtomAgentsRuntimeAdapter(
        predictor=MockPredictor("atomagents"), scheduler=scheduler,
        detector=detector, bus=bus, config=cfg,
    )


def _engine(name="qwen_72b", **kw):
    d = dict(resource_id=f"m_{name}", resource_type="vllm_model", name=name,
             confidence=0.9, consumer_tool="computation_task_screw_dislocation",
             consumer_step_offset=2, expected_at_step=7)
    d.update(kw)
    return ResourceSpec(**d)


def _result(*resources):
    return PredictionResult(step=1, resources=list(resources), confidence=0.9)


def test_pairs_one_cache_per_engine(tmp_path):
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)})
    r = _result(_engine())
    assert a._pair_model_cache(r) == 1
    kinds = [x.resource_type for x in r.resources]
    assert kinds.count("vllm_model") == 1
    assert kinds.count("model_cache") == 1


def test_cache_name_matches_engine_name(tmp_path):
    """The gate_group join key. A mismatch here double-counts every stall."""
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)})
    r = _result(_engine("qwen_72b"))
    a._pair_model_cache(r)
    eng = [x for x in r.resources if x.resource_type == "vllm_model"][0]
    cache = [x for x in r.resources if x.resource_type == "model_cache"][0]
    assert cache.name == eng.name
    # and the ids must still differ, or the scheduler dedup drops one of them
    assert cache.resource_id != eng.resource_id


def test_consumer_fields_are_inherited(tmp_path):
    """Both halves gate the same need, so both must point at the same consumer."""
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)})
    r = _result(_engine())
    a._pair_model_cache(r)
    eng = [x for x in r.resources if x.resource_type == "vllm_model"][0]
    cache = [x for x in r.resources if x.resource_type == "model_cache"][0]
    assert cache.consumer_tool == eng.consumer_tool
    assert cache.consumer_step_offset == eng.consumer_step_offset
    assert cache.expected_at_step == eng.expected_at_step


def test_cache_is_cancellation_safe(tmp_path):
    """Host warm touches no GPU and stops no engine, unlike the vllm_model task."""
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)})
    r = _result(_engine())
    a._pair_model_cache(r)
    cache = [x for x in r.resources if x.resource_type == "model_cache"][0]
    assert cache.cancellation_safe is True


def test_no_duplicate_staging_across_predictions(tmp_path):
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)})
    assert a._pair_model_cache(_result(_engine())) == 1
    assert a._pair_model_cache(_result(_engine())) == 0


def test_distinct_models_each_get_one(tmp_path):
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path),
                                        "qwen_32b": str(tmp_path)})
    r = _result(_engine("qwen_72b"), _engine("qwen_32b"))
    assert a._pair_model_cache(r) == 2


def test_skipped_without_a_known_snapshot_path(tmp_path):
    """A staging task with no path can only fail, and a FAILED task is not the
    same as an absent one -- it surfaces as unattributed stall."""
    a = _adapter(tmp_path, model_paths={})
    assert a._pair_model_cache(_result(_engine())) == 0


def test_data_file_resources_are_not_paired(tmp_path):
    a = _adapter(tmp_path, model_paths={"w_eam4.fs": str(tmp_path)})
    r = _result(ResourceSpec(resource_id="d1", resource_type="data_file",
                             name="w_eam4.fs"))
    assert a._pair_model_cache(r) == 0


def test_ablation_switch_suppresses_it(tmp_path):
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)},
                 skip_resource_types=["model_cache"])
    assert a._pair_model_cache(_result(_engine())) == 0


def test_disabled_when_stage_worker_cache_is_off(tmp_path):
    a = _adapter(tmp_path, model_paths={"qwen_72b": str(tmp_path)},
                 stage_worker_cache=False)
    assert a._pair_model_cache(_result(_engine())) == 0


def test_baseline_mode_emits_nothing(tmp_path):
    a = _adapter(tmp_path, mode=RuntimeMode.BASELINE,
                 model_paths={"qwen_72b": str(tmp_path)})
    assert a._pair_model_cache(_result(_engine())) == 0


def test_shard_bytes_is_none_for_unknown_path():
    assert _model_shard_bytes(None) is None
    assert _model_shard_bytes("/nonexistent/snapshot/dir") is None


# ---------------------------------------------------------------------------
# The downstream half of the contract: extract_prefetch_lifecycle must put the
# pair in ONE gate_group. If it does not, its max()-dedup cannot fire and a
# single stall is counted once per resource -- inflating every taxonomy number
# for this workload while every individual row still looks correct.
# ---------------------------------------------------------------------------

def _match_gate():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "extract_prefetch_lifecycle.py"
    spec = importlib.util.spec_from_file_location("_epl", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.match_gate


def test_paired_tasks_land_in_the_same_gate_group():
    match_gate = _match_gate()
    gates = [
        {"kind": "swap", "model": "qwen_72b", "idx": 0, "t_exit": 900.0, "wait_s": 700.0},
        {"kind": "swap", "model": "qwen_72b", "idx": 1, "t_exit": 2400.0, "wait_s": 650.0},
    ]
    # Both tasks are scheduled off ONE prediction, so their start times differ
    # only by dispatch order -- a few hundred ms in practice.
    g_engine = match_gate(gates, "qwen_72b", 210.0)
    g_cache = match_gate(gates, "qwen_72b", 210.4)
    assert g_engine is not None and g_cache is not None
    key = lambda g: f"{g['kind']}:{g['model']}:{g['idx']}"
    assert key(g_engine) == key(g_cache)


def test_a_mismatched_name_splits_the_gate_group():
    """Guards the failure mode: this is what a typo'd name would produce."""
    match_gate = _match_gate()
    gates = [
        {"kind": "swap", "model": "qwen_72b", "idx": 0, "t_exit": 900.0, "wait_s": 700.0},
    ]
    assert match_gate(gates, "qwen_72b", 210.0) is not None
    assert match_gate(gates, "cache_qwen_72b", 210.0) is None
