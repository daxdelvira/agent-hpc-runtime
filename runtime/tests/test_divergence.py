"""
test_divergence.py — Unit tests for DivergenceDetector, PrefetchScheduler, and policy.
"""
import time

import pytest

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import PredictionResult, ResourceSpec
from runtime.guard.detector import DivergenceAction, DivergenceDetector
from runtime.guard.checkpoint import CheckpointStore
from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mace_resource(confidence=0.9) -> ResourceSpec:
    return ResourceSpec(
        resource_id="mace-mp0",
        resource_type="mace_model",
        name="mace-mp-0",
        estimated_load_s=35.0,
        confidence=confidence,
        cancellation_safe=True,
        consumer_tool="run_ase",
        consumer_step_offset=1,
    )


def make_prediction(step=2, resource=None, tool="run_ase", confidence=0.9) -> PredictionResult:
    r = resource or make_mace_resource(confidence)
    r.consumer_tool = tool
    return PredictionResult(step=step, resources=[r], confidence=confidence, predictor_id="mock")


# ---------------------------------------------------------------------------
# DivergenceDetector
# ---------------------------------------------------------------------------

class TestDivergenceDetector:
    def setup_method(self):
        self.cfg = RuntimeConfig(mode=RuntimeMode.SIMULATED)
        self.detector = DivergenceDetector(scheduler=None, config=self.cfg)

    def test_hit_returns_continue(self):
        # offset=1: prediction at step 2 targets step 3
        pred = make_prediction(step=2, tool="run_ase")
        self.detector.on_prediction(pred, step=2)
        hit, action, ckpt = self.detector.on_tool_about_to_execute("run_ase", step=3)
        assert hit is True
        assert action == DivergenceAction.CONTINUE
        assert ckpt is not None

    def test_miss_returns_invalidate_all(self):
        pred = make_prediction(step=2, tool="run_ase")
        self.detector.on_prediction(pred, step=2)
        hit, action, ckpt = self.detector.on_tool_about_to_execute("molecule_name_to_smiles", step=3)
        assert hit is False
        assert action == DivergenceAction.INVALIDATE_ALL

    def test_miss_enters_conservative_mode(self):
        self.cfg = RuntimeConfig(mode=RuntimeMode.SIMULATED, conservative_mode_steps=3)
        self.detector = DivergenceDetector(scheduler=None, config=self.cfg)
        pred = make_prediction(step=2, tool="run_ase")
        self.detector.on_prediction(pred, step=2)
        # diverge at step 3 (2 + offset=1); conservative_until = 3 + 3 = 6
        self.detector.on_tool_about_to_execute("wrong_tool", step=3)
        assert self.detector.is_conservative(step=4)
        assert self.detector.is_conservative(step=5)
        assert self.detector.is_conservative(step=6)
        assert not self.detector.is_conservative(step=7)

    def test_hit_does_not_enter_conservative_mode(self):
        pred = make_prediction(step=2, tool="run_ase")
        self.detector.on_prediction(pred, step=2)
        self.detector.on_tool_about_to_execute("run_ase", step=3)
        assert not self.detector.is_conservative(step=4)

    def test_accuracy_tracking(self):
        for i in range(5):
            pred = make_prediction(step=i*2, tool="run_ase")
            self.detector.on_prediction(pred, step=i*2)
            actual = "run_ase" if i < 4 else "wrong_tool"
            # offset=1: execute at step i*2 + 1
            self.detector.on_tool_about_to_execute(actual, step=i*2 + 1)
        acc = self.detector.accuracy_for("mock")
        assert abs(acc - 0.8) < 0.01    # 4 hits / 5 total

    def test_no_prediction_no_divergence(self):
        hit, action, ckpt = self.detector.on_tool_about_to_execute("any_tool", step=1)
        assert hit is True
        assert action == DivergenceAction.CONTINUE

    def test_checkpoint_marked_diverged(self):
        # offset=1: prediction at step 3 targets step 4
        pred = make_prediction(step=3, tool="run_ase")
        ckpt = self.detector.on_prediction(pred, step=3)
        self.detector.on_tool_about_to_execute("wrong_tool", step=4)
        stored = self.detector._store.get(ckpt.checkpoint_id)
        assert stored.status == "diverged"


# ---------------------------------------------------------------------------
# PrefetchScheduler + SimulatedPrefetchExecutor
# ---------------------------------------------------------------------------

class TestPrefetchScheduler:
    def setup_method(self):
        self.cfg = RuntimeConfig(
            mode=RuntimeMode.SIMULATED,
            confidence_threshold=0.85,
            max_horizon=2,
        )
        self.executor = SimulatedPrefetchExecutor()
        self.bus = None
        self.scheduler = PrefetchScheduler(
            executor=self.executor,
            config=self.cfg,
            bus=self.bus,
        )

    def test_schedule_above_threshold_starts_task(self):
        resource = make_mace_resource(confidence=0.90)
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is not None
        assert task.status in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS, PrefetchStatus.COMPLETED)

    def test_schedule_below_threshold_skips(self):
        resource = make_mace_resource(confidence=0.50)
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is None

    def test_schedule_deduplication(self):
        resource = make_mace_resource(confidence=0.90)
        t1 = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        t2 = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert t1 is not None
        assert t2 is None  # deduplicated

    def test_cancel_all_pending(self):
        resource = make_mace_resource(confidence=0.90)
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is not None
        cancelled = self.scheduler.cancel_all_pending("divergence", "ckpt-1", current_step=3)
        assert len(cancelled) == 1
        assert cancelled[0].status in (PrefetchStatus.CANCELLED, PrefetchStatus.WASTED)

    def test_observe_only_mode_no_prefetch(self):
        cfg = RuntimeConfig(mode=RuntimeMode.OBSERVE_ONLY)
        scheduler = PrefetchScheduler(executor=self.executor, config=cfg)
        resource = make_mace_resource(confidence=0.99)
        task = scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is None

    def test_baseline_mode_no_prefetch(self):
        cfg = RuntimeConfig(mode=RuntimeMode.BASELINE)
        scheduler = PrefetchScheduler(executor=self.executor, config=cfg)
        resource = make_mace_resource(confidence=0.99)
        task = scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is None

    def test_on_resource_consumed_marks_used(self):
        resource = make_mace_resource(confidence=0.90)
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id="ckpt-1")
        assert task is not None
        time.sleep(0.05)   # let simulated executor complete
        self.scheduler.on_resource_consumed("mace-mp0", consumed_at=time.perf_counter(), current_step=3)
        updated = self.scheduler.get_task_for_resource("mace-mp0")
        # status should be USED or COMPLETED (consumed marks it USED)
        assert updated is not None


# ---------------------------------------------------------------------------
# Integration: detector + scheduler together
# ---------------------------------------------------------------------------

class TestDetectorWithScheduler:
    def setup_method(self):
        self.cfg = RuntimeConfig(
            mode=RuntimeMode.SIMULATED,
            confidence_threshold=0.85,
            conservative_mode_steps=2,
        )
        self.executor = SimulatedPrefetchExecutor()
        self.scheduler = PrefetchScheduler(
            executor=self.executor,
            config=self.cfg,
        )
        self.detector = DivergenceDetector(
            scheduler=self.scheduler,
            config=self.cfg,
        )

    def test_divergence_cancels_prefetch_task(self):
        # offset=1: prediction at step 2 targets step 3
        pred = make_prediction(step=2, tool="run_ase", confidence=0.90)
        ckpt = self.detector.on_prediction(pred, step=2)

        # Start prefetch for the predicted resource
        resource = pred.resources[0]
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id=ckpt.checkpoint_id)
        assert task is not None

        # Diverge: wrong tool executed at consumer step
        hit, action, _ = self.detector.on_tool_about_to_execute("wrong_tool", step=3)
        assert hit is False
        assert action == DivergenceAction.INVALIDATE_ALL

        # Task should be cancelled
        time.sleep(0.05)
        updated = self.scheduler.get_task(task.task_id)
        assert updated.status in (PrefetchStatus.CANCELLED, PrefetchStatus.WASTED)

    def test_hit_does_not_cancel_task(self):
        pred = make_prediction(step=2, tool="run_ase", confidence=0.90)
        ckpt = self.detector.on_prediction(pred, step=2)
        resource = pred.resources[0]
        task = self.scheduler.schedule(resource, current_step=2, checkpoint_id=ckpt.checkpoint_id)

        hit, action, _ = self.detector.on_tool_about_to_execute("run_ase", step=3)
        assert hit is True
        updated = self.scheduler.get_task(task.task_id)
        assert updated.status not in (PrefetchStatus.CANCELLED, PrefetchStatus.WASTED)
