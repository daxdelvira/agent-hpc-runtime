"""
tests/test_model_prefetch.py — Unit tests for ModelPrefetchExecutor + FakeModelOrchestrator.

All tests use short sleep times (0.1–0.3s) so the suite runs in under 5s.
No GPUs, no vLLM, no real model loading required.
"""
from __future__ import annotations

import time
import unittest

from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.events import ResourceSpec
from runtime.prefetch.model_prefetch import FakeModelOrchestrator, ModelPrefetchExecutor


def _make_resource(name: str = "qwen_32b", cancellation_safe: bool = False) -> ResourceSpec:
    from runtime.events import ResourceSpec
    import hashlib
    rid = hashlib.sha1(name.encode()).hexdigest()[:16]
    return ResourceSpec(
        resource_id=rid,
        resource_type="vllm_model",
        name=name,
        estimated_load_s=0.1,
        cancellation_safe=cancellation_safe,
        consumer_tool="computation_task_screw_dislocation",
    )


def _make_task(name: str = "qwen_32b") -> PrefetchTask:
    return PrefetchTask(
        resource=_make_resource(name),
        status=PrefetchStatus.PENDING,
        checkpoint_id="ckpt-test",
        workflow_step_at_start=1,
    )


class TestFakeModelOrchestrator(unittest.TestCase):

    def test_returns_elapsed_seconds(self):
        orch = FakeModelOrchestrator(load_times={"model_a": 0.1})
        t0 = time.perf_counter()
        elapsed = orch.start_model_measured("model_a")
        wall = time.perf_counter() - t0
        self.assertAlmostEqual(elapsed, 0.1, delta=0.05)
        self.assertGreaterEqual(wall, 0.09)

    def test_unknown_model_defaults_to_1s_reduced_for_test(self):
        orch = FakeModelOrchestrator(load_times={})
        # Default is 1.0s — too slow for unit tests; override via load_times
        orch.load_times["default_model"] = 0.05
        elapsed = orch.start_model_measured("default_model")
        self.assertAlmostEqual(elapsed, 0.05, delta=0.05)

    def test_failure_model_raises(self):
        orch = FakeModelOrchestrator(
            load_times={"bad_model": 0.1},
            failure_models={"bad_model"},
        )
        with self.assertRaises(RuntimeError):
            orch.start_model_measured("bad_model")

    def test_processes_dict_populated_after_load(self):
        orch = FakeModelOrchestrator(load_times={"m": 0.05})
        self.assertNotIn("m", orch.processes)
        orch.start_model_measured("m")
        self.assertIn("m", orch.processes)

    def test_stop_model_removes_from_processes(self):
        orch = FakeModelOrchestrator(load_times={"m": 0.05})
        orch.start_model_measured("m")
        orch.stop_model("m")
        self.assertNotIn("m", orch.processes)

    def test_wait_until_ready_noop(self):
        orch = FakeModelOrchestrator()
        orch.wait_until_ready("any_model")  # must not raise

    def test_ensure_all_models_ready_noop(self):
        orch = FakeModelOrchestrator()
        orch.ensure_all_models_ready()  # must not raise


class TestModelPrefetchExecutorLifecycle(unittest.TestCase):

    def setUp(self):
        self.orch = FakeModelOrchestrator(load_times={"qwen_32b": 0.1})
        self.executor = ModelPrefetchExecutor(self.orch)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def _wait_complete(self, task: PrefetchTask, timeout: float = 2.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.executor.is_complete(task):
                return
            time.sleep(0.02)
        raise TimeoutError(f"Task did not complete in {timeout}s: status={task.status}")

    def test_start_transitions_to_in_progress(self):
        task = _make_task()
        self.executor.start(task)
        self.assertEqual(task.status, PrefetchStatus.IN_PROGRESS)

    def test_start_sets_started_at(self):
        task = _make_task()
        t0 = time.perf_counter()
        self.executor.start(task)
        self.assertIsNotNone(task.started_at)
        self.assertGreaterEqual(task.started_at, t0)

    def test_completes_with_completed_status(self):
        task = _make_task()
        self.executor.start(task)
        self._wait_complete(task)
        self.assertEqual(task.status, PrefetchStatus.COMPLETED)

    def test_completed_at_set_after_load(self):
        task = _make_task()
        self.executor.start(task)
        self._wait_complete(task)
        self.assertIsNotNone(task.completed_at)
        self.assertGreater(task.completed_at, task.started_at)

    def test_get_result_returns_elapsed(self):
        task = _make_task()
        self.executor.start(task)
        self._wait_complete(task)
        result = self.executor.get_result(task)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["elapsed_s"], 0.1, delta=0.1)

    def test_is_complete_false_while_running(self):
        orch = FakeModelOrchestrator(load_times={"slow_model": 0.3})
        executor = ModelPrefetchExecutor(orch)
        task = PrefetchTask(
            resource=_make_resource("slow_model"),
            status=PrefetchStatus.PENDING,
            checkpoint_id="ckpt-test",
            workflow_step_at_start=1,
        )
        executor.start(task)
        # Immediately after start, should still be running
        self.assertFalse(executor.is_complete(task))
        # Wait for it to finish
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)
        self.assertTrue(executor.is_complete(task))
        executor.shutdown(wait=False)


class TestModelPrefetchExecutorCancellation(unittest.TestCase):

    def setUp(self):
        # Use longer load to ensure we can cancel before completion
        self.orch = FakeModelOrchestrator(load_times={"qwen_72b": 0.3})
        self.executor = ModelPrefetchExecutor(self.orch)

    def tearDown(self):
        self.executor.shutdown(wait=False)

    def _wait_status(
        self,
        task: PrefetchTask,
        target_statuses: set[PrefetchStatus],
        timeout: float = 1.5,
    ) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if task.status in target_statuses:
                return
            time.sleep(0.02)

    def test_cancel_returns_false(self):
        """vLLM loads are non-interruptible; cancel() always returns False."""
        task = _make_task("qwen_72b")
        self.executor.start(task)
        result = self.executor.cancel(task)
        self.assertFalse(result)

    def test_cancel_mid_load_marks_wasted(self):
        """Cancel during load → task ends as WASTED, not COMPLETED."""
        task = _make_task("qwen_72b")
        self.executor.start(task)
        time.sleep(0.05)   # let load start
        self.executor.cancel(task)
        # Wait for load to finish
        self._wait_status(task, {PrefetchStatus.WASTED, PrefetchStatus.COMPLETED})
        self.assertEqual(task.status, PrefetchStatus.WASTED)

    def test_get_result_wasted_flag(self):
        task = _make_task("qwen_72b")
        self.executor.start(task)
        time.sleep(0.05)
        self.executor.cancel(task)
        self._wait_status(task, {PrefetchStatus.WASTED, PrefetchStatus.COMPLETED})
        result = self.executor.get_result(task)
        self.assertTrue(result.get("wasted", False))

    def test_double_cancel_is_safe(self):
        task = _make_task("qwen_72b")
        self.executor.start(task)
        self.executor.cancel(task)
        self.executor.cancel(task)   # second cancel must not raise


class TestModelPrefetchExecutorFailure(unittest.TestCase):

    def test_failure_sets_failed_status(self):
        orch = FakeModelOrchestrator(failure_models={"broken_model"})
        executor = ModelPrefetchExecutor(orch)
        task = PrefetchTask(
            resource=_make_resource("broken_model"),
            status=PrefetchStatus.PENDING,
            checkpoint_id="ckpt-test",
            workflow_step_at_start=1,
        )
        executor.start(task)
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS):
                break
            time.sleep(0.02)
        self.assertEqual(task.status, PrefetchStatus.FAILED)
        self.assertIsNotNone(task.error)
        self.assertIn("broken_model", task.error)
        executor.shutdown(wait=False)

    def test_failure_result_success_false(self):
        orch = FakeModelOrchestrator(failure_models={"broken"})
        executor = ModelPrefetchExecutor(orch)
        task = PrefetchTask(
            resource=_make_resource("broken"),
            status=PrefetchStatus.PENDING,
            checkpoint_id="ckpt-test",
            workflow_step_at_start=1,
        )
        executor.start(task)
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)
        result = executor.get_result(task)
        self.assertFalse(result["success"])
        executor.shutdown(wait=False)


class TestModelPrefetchExecutorTiming(unittest.TestCase):
    """Verify that timing metadata is set accurately."""

    def test_overlap_is_measurable(self):
        """
        If we start a prefetch (0.2s) and wait only 0.1s before consuming,
        the load overlaps with our simulated compute window.
        """
        orch = FakeModelOrchestrator(load_times={"model": 0.2})
        executor = ModelPrefetchExecutor(orch)

        task = _make_task("model")
        t_start = time.perf_counter()
        executor.start(task)

        # Simulate 0.1s of compute happening concurrently
        time.sleep(0.1)
        consumed_at = time.perf_counter()

        # Wait for the load to finish
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)

        self.assertEqual(task.status, PrefetchStatus.COMPLETED)

        # The prefetch ran while we were sleeping → overlap_s > 0
        from runtime.measurement.timings import PrefetchTimingRecord
        tr = PrefetchTimingRecord(
            prefetch_start_t=task.started_at,
            prefetch_end_t=task.completed_at,
            resource_needed_t=consumed_at,
        )
        self.assertGreater(tr.overlap_s, 0.0)
        executor.shutdown(wait=False)

    def test_benefit_positive_when_prefetch_finishes_first(self):
        """Prefetch completes, then consumer starts → benefit_s > 0."""
        orch = FakeModelOrchestrator(load_times={"model": 0.1})
        executor = ModelPrefetchExecutor(orch)

        task = _make_task("model")
        executor.start(task)

        # Wait for load to finish before consuming
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)

        # Consumer arrives after prefetch finished
        time.sleep(0.05)
        consumed_at = time.perf_counter()

        from runtime.measurement.timings import PrefetchTimingRecord
        tr = PrefetchTimingRecord(
            prefetch_start_t=task.started_at,
            prefetch_end_t=task.completed_at,
            resource_needed_t=consumed_at,
        )
        self.assertGreater(tr.benefit_s, 0.0)
        executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
