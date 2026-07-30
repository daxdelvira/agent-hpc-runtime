"""
tests/test_sleep_wake.py — Unit tests for the vLLM sleep/wake swap arm
(RuntimeConfig.sleep_wake_swaps), no GPUs required.

Covers, against FakeModelOrchestrator (which mirrors the real orchestrator's
sleep-mode API):
  - wake preferred over cold boot when the target engine is slept
  - the planner (conflicting engine) is SLEPT, never stopped
  - strict sleep-then-wake ordering (both models never awake in VRAM together)
  - cold-boot fallback when the target was never booted this run
  - off-flag (sleep_wake=False) reproduces the legacy stop/start behaviour
  - prefetch_completed extras carry mechanism ("sleep_wake" | "cold_boot")

The sequencing under test (runtime/prefetch/sleep_wake.py) is the exact code
both production call sites use: the chemgraph adapter's on-demand path and
ModelPrefetchExecutor._load_model.
"""
from __future__ import annotations

import time
import unittest

from runtime.events import ResourceSpec
from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.prefetch.model_prefetch import FakeModelOrchestrator, ModelPrefetchExecutor
from runtime.prefetch.sleep_wake import (
    has_live_process,
    last_mechanism,
    release_gpus_for,
    swap_to_model,
)

PLANNER = "qwen_32b_vl"
WORKER = "qwen_72b_instruct"

SHARED_POOL = {
    PLANNER: {"gpus": [0, 1, 2, 3]},
    WORKER: {"gpus": [0, 1, 2, 3]},
}


def _make_task(name: str = WORKER, size_bytes: int | None = None) -> PrefetchTask:
    import hashlib
    rid = hashlib.sha1(name.encode()).hexdigest()[:16]
    resource = ResourceSpec(
        resource_id=rid,
        resource_type="vllm_model",
        name=name,
        estimated_load_s=0.1,
        cancellation_safe=False,
        consumer_tool="",
    )
    if size_bytes is not None:
        resource.estimated_size_bytes = size_bytes
    return PrefetchTask(
        resource=resource,
        status=PrefetchStatus.PENDING,
        checkpoint_id="ckpt-test",
        workflow_step_at_start=1,
    )


def _orch(**kwargs) -> FakeModelOrchestrator:
    defaults = dict(
        load_times={PLANNER: 0.02, WORKER: 0.02},
        models=dict(SHARED_POOL),
    )
    defaults.update(kwargs)
    return FakeModelOrchestrator(**defaults)


def _boot_both_then_sleep_worker(orch: FakeModelOrchestrator) -> None:
    """Reach the mid-run state: worker booted earlier and slept; planner awake."""
    orch.start_model_measured(WORKER)
    orch.sleep_model(WORKER)
    orch.start_model_measured(PLANNER)
    orch.calls.clear()


class TestFakeOrchestratorSleepWakeAPI(unittest.TestCase):

    def test_sleep_then_wake_roundtrip(self):
        orch = _orch()
        orch.start_model_measured(WORKER)
        self.assertFalse(orch.is_sleeping(WORKER))
        orch.sleep_model(WORKER)
        self.assertTrue(orch.is_sleeping(WORKER))
        self.assertIn(WORKER, orch.processes)   # process stays alive while slept
        orch.wake_model(WORKER)
        self.assertFalse(orch.is_sleeping(WORKER))

    def test_wake_without_process_raises(self):
        orch = _orch()
        with self.assertRaises(RuntimeError):
            orch.wake_model(WORKER)

    def test_stop_clears_sleep_state(self):
        orch = _orch()
        orch.start_model_measured(WORKER)
        orch.sleep_model(WORKER)
        orch.stop_model(WORKER)
        self.assertFalse(orch.is_sleeping(WORKER))
        self.assertNotIn(WORKER, orch.processes)

    def test_wait_until_serving_rejects_slept_engine(self):
        orch = _orch()
        orch.start_model_measured(WORKER)
        orch.sleep_model(WORKER)
        with self.assertRaises(RuntimeError):
            orch.wait_until_serving(WORKER)


class TestSwapToModel(unittest.TestCase):

    def test_wake_preferred_over_boot_when_slept(self):
        """Worker process exists and is slept -> wake path, no cold boot."""
        orch = _orch()
        _boot_both_then_sleep_worker(orch)

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["mechanism"], "sleep_wake")
        self.assertIn(("wake_model", WORKER), orch.calls)
        self.assertNotIn(("start_model_measured", WORKER), orch.calls)
        self.assertFalse(orch.is_sleeping(WORKER))

    def test_planner_slept_not_stopped(self):
        """The conflicting planner is slept (weights kept), never stopped."""
        orch = _orch()
        _boot_both_then_sleep_worker(orch)

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["slept_models"], [PLANNER])
        self.assertTrue(orch.is_sleeping(PLANNER))
        self.assertIn(PLANNER, orch.processes)   # process survives the swap
        stop_calls = [c for c in orch.calls if c[0] == "stop_model"]
        self.assertEqual(stop_calls, [])

    def test_sleep_before_wake_ordering(self):
        """VRAM safety: the planner sleeps BEFORE the worker wakes/boots."""
        orch = _orch()
        _boot_both_then_sleep_worker(orch)

        swap_to_model(orch, WORKER)

        sleep_idx = orch.calls.index(("sleep_model", PLANNER, 1))
        wake_idx = orch.calls.index(("wake_model", WORKER))
        self.assertLess(sleep_idx, wake_idx)

    def test_cold_boot_fallback_when_never_booted(self):
        """Worker has no process yet -> planner slept, worker cold-boots."""
        orch = _orch()
        orch.start_model_measured(PLANNER)
        orch.calls.clear()

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["mechanism"], "cold_boot")
        self.assertIn(("sleep_model", PLANNER, 1), orch.calls)
        self.assertIn(("start_model_measured", WORKER), orch.calls)
        self.assertNotIn(("wake_model", WORKER), orch.calls)
        self.assertTrue(orch.is_sleeping(PLANNER))
        # And the sleep happened before the boot.
        self.assertLess(orch.calls.index(("sleep_model", PLANNER, 1)),
                        orch.calls.index(("start_model_measured", WORKER)))

    def test_already_serving_short_circuits(self):
        """Worker awake -> no wake, no boot; serving is just verified."""
        orch = _orch()
        orch.start_model_measured(WORKER)
        orch.calls.clear()

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["mechanism"], "already_serving")
        self.assertEqual(
            [c for c in orch.calls if c[0] in ("wake_model", "start_model_measured")],
            [])
        self.assertIn(("wait_until_serving", WORKER), orch.calls)

    def test_disjoint_pools_not_slept(self):
        """An engine on non-overlapping GPUs is not touched by the swap."""
        models = {
            WORKER: {"gpus": [0, 1, 2, 3]},
            "qwen_32b_aggregator": {"gpus": [4, 5]},
        }
        orch = _orch(models=models)
        orch.start_model_measured("qwen_32b_aggregator")
        orch.calls.clear()

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["mechanism"], "cold_boot")
        self.assertEqual(info["slept_models"], [])
        self.assertFalse(orch.is_sleeping("qwen_32b_aggregator"))

    def test_sleep_failure_falls_back_to_stop(self):
        """If the sleep endpoint fails, the conflicting engine is stopped so
        the run survives (at the cost of the wake benefit)."""
        orch = _orch(failure_models={PLANNER})
        orch.processes[PLANNER] = object()   # planner running (bypass boot failure)
        orch.calls.clear()

        info = swap_to_model(orch, WORKER)

        self.assertEqual(info["mechanism"], "cold_boot")
        self.assertIn(("stop_model", PLANNER), orch.calls)
        self.assertNotIn(PLANNER, orch.processes)

    def test_last_mechanism_recorded(self):
        orch = _orch()
        orch.start_model_measured(PLANNER)
        swap_to_model(orch, WORKER)
        self.assertEqual(last_mechanism(orch, WORKER), "cold_boot")
        orch.sleep_model(WORKER)
        swap_to_model(orch, WORKER)
        self.assertEqual(last_mechanism(orch, WORKER), "sleep_wake")


class TestReleaseGpusFor(unittest.TestCase):
    """The adapter's plan-extraction hook (_schedule_worker_model_prefetch)."""

    def test_sleep_wake_on_sleeps_planner(self):
        orch = _orch()
        orch.start_model_measured(PLANNER)
        orch.calls.clear()

        release_gpus_for(orch, WORKER, sleep_wake=True)

        self.assertIn(("sleep_model", PLANNER, 1), orch.calls)
        self.assertNotIn(("stop_model", PLANNER), orch.calls)
        self.assertTrue(orch.is_sleeping(PLANNER))
        self.assertIn(PLANNER, orch.processes)

    def test_off_flag_stops_planner_legacy(self):
        """sleep_wake=False reproduces the legacy stop exactly."""
        orch = _orch()
        orch.start_model_measured(PLANNER)
        orch.calls.clear()

        release_gpus_for(orch, WORKER, sleep_wake=False)

        self.assertEqual(orch.calls, [("stop_model", PLANNER)])
        self.assertNotIn(PLANNER, orch.processes)

    def test_noop_when_target_already_running(self):
        orch = _orch()
        orch.start_model_measured(WORKER)
        orch.calls.clear()
        release_gpus_for(orch, WORKER, sleep_wake=True)
        self.assertEqual(orch.calls, [])

    def test_already_slept_planner_untouched(self):
        orch = _orch()
        orch.start_model_measured(PLANNER)
        orch.sleep_model(PLANNER)
        orch.calls.clear()
        release_gpus_for(orch, PLANNER, sleep_wake=True)   # target == running
        release_gpus_for(orch, WORKER, sleep_wake=True)
        self.assertEqual([c for c in orch.calls if c[0] != "is_sleeping"], [])


class TestModelPrefetchExecutorSleepWake(unittest.TestCase):
    """ModelPrefetchExecutor._load_model with sleep_wake on/off."""

    def _run(self, executor: ModelPrefetchExecutor, task: PrefetchTask,
             timeout: float = 3.0) -> dict:
        executor.start(task)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)
        else:
            raise TimeoutError(f"task did not complete: {task.status}")
        return executor.get_result(task)

    def test_prefetch_wakes_slept_worker(self):
        """Wake preferred over boot in the background prefetch path."""
        orch = _orch()
        _boot_both_then_sleep_worker(orch)
        executor = ModelPrefetchExecutor(orch, sleep_wake=True)
        try:
            result = self._run(executor, _make_task(WORKER))
        finally:
            executor.shutdown(wait=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["mechanism"], "sleep_wake")
        self.assertIn(("wake_model", WORKER), orch.calls)
        self.assertNotIn(("start_model_measured", WORKER), orch.calls)
        # Planner slept, not stopped.
        self.assertTrue(orch.is_sleeping(PLANNER))
        self.assertEqual([c for c in orch.calls if c[0] == "stop_model"], [])

    def test_prefetch_cold_boots_when_never_booted(self):
        orch = _orch()
        orch.start_model_measured(PLANNER)
        orch.calls.clear()
        executor = ModelPrefetchExecutor(orch, sleep_wake=True)
        try:
            result = self._run(executor, _make_task(WORKER))
        finally:
            executor.shutdown(wait=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["mechanism"], "cold_boot")
        self.assertIn(("start_model_measured", WORKER), orch.calls)
        self.assertTrue(orch.is_sleeping(PLANNER))

    def test_bytes_staged_only_on_cold_boot(self):
        """A wake reads no bytes from storage -> no bytes_staged claim."""
        size = 145_000_000_000
        orch = _orch()
        _boot_both_then_sleep_worker(orch)
        executor = ModelPrefetchExecutor(orch, sleep_wake=True)
        try:
            wake_result = self._run(executor, _make_task(WORKER, size_bytes=size))
        finally:
            executor.shutdown(wait=False)
        self.assertNotIn("bytes_staged", wake_result)

        orch2 = _orch()
        executor2 = ModelPrefetchExecutor(orch2, sleep_wake=True)
        try:
            boot_result = self._run(executor2, _make_task(WORKER, size_bytes=size))
        finally:
            executor2.shutdown(wait=False)
        self.assertEqual(boot_result["mechanism"], "cold_boot")
        self.assertEqual(boot_result.get("bytes_staged"), size)

    def test_off_flag_old_behavior(self):
        """sleep_wake=False: legacy start_model_measured, no wake, no
        mechanism key in the result (byte-identical extras)."""
        orch = _orch()
        _boot_both_then_sleep_worker(orch)
        # Legacy load requires free GPUs; the real conflict is exercised on
        # hardware — here we verify the code path taken, not GPU state.
        executor = ModelPrefetchExecutor(orch, sleep_wake=False)
        try:
            result = self._run(executor, _make_task(WORKER))
        finally:
            executor.shutdown(wait=False)

        self.assertTrue(result["success"])
        self.assertNotIn("mechanism", result)
        self.assertIn(("start_model_measured", WORKER), orch.calls)
        self.assertNotIn(("wake_model", WORKER), orch.calls)
        self.assertEqual([c for c in orch.calls if c[0] == "sleep_model"], [])
        # bytes_staged behaviour unchanged when the flag is off.
        orch3 = _orch()
        executor3 = ModelPrefetchExecutor(orch3, sleep_wake=False)
        try:
            r3 = self._run(executor3, _make_task(WORKER, size_bytes=1000))
        finally:
            executor3.shutdown(wait=False)
        self.assertEqual(r3.get("bytes_staged"), 1000)


if __name__ == "__main__":
    unittest.main()
