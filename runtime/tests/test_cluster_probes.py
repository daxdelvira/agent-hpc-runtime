"""
tests/test_cluster_probes.py — Unit tests for ClusterProbes.

Tests verify:
  - ProbeSnapshot captures proc/io fields when available
  - ProbeDelta computes correct deltas and cache_hit_ratio
  - Source inference covers page_cache / nfs / mixed / unknown cases
  - LoadProbeContext context manager snapshots before/after
  - Graceful degradation when probes are unavailable (no crashes)
  - Integration: ModelPrefetchExecutor embeds probe_delta in result dict
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from runtime.measurement.cluster_probes import (
    ClusterProbes,
    LoadProbeContext,
    ProbeSnapshot,
    ProbeDelta,
    _infer_source,
    _read_proc_io,
)
from runtime.prefetch.model_prefetch import FakeModelOrchestrator, ModelPrefetchExecutor
from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.events import ResourceSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(name: str = "qwen_32b") -> PrefetchTask:
    import hashlib
    rid = hashlib.md5(name.encode()).hexdigest()[:12]
    return PrefetchTask(
        resource=ResourceSpec(
            resource_id=rid,
            resource_type="vllm_model",
            name=name,
            estimated_load_s=0.1,
            cancellation_safe=False,
            consumer_tool="computation_task_screw_dislocation",
        ),
        status=PrefetchStatus.PENDING,
        checkpoint_id="ckpt-test",
        workflow_step_at_start=1,
    )


def _make_snapshot(**kwargs) -> ProbeSnapshot:
    defaults = dict(
        timestamp=time.time(),
        proc_rchar=None,
        proc_read_bytes=None,
        net_bytes_recv=None,
        net_bytes_sent=None,
        gpu_memory_mib=None,
    )
    defaults.update(kwargs)
    return ProbeSnapshot(**defaults)


# ---------------------------------------------------------------------------
# ProbeSnapshot
# ---------------------------------------------------------------------------

class TestProbeSnapshot(unittest.TestCase):

    def test_proc_io_read_on_linux(self):
        """On Linux, /proc/self/io should be readable."""
        import platform
        if platform.system() != "Linux":
            self.skipTest("Not Linux")
        snap = ClusterProbes().snapshot()
        self.assertIsNotNone(snap.proc_rchar)
        self.assertIsNotNone(snap.proc_read_bytes)
        self.assertGreater(snap.proc_rchar, 0)

    def test_timestamp_is_recent(self):
        t0 = time.time()
        snap = ClusterProbes().snapshot()
        self.assertGreaterEqual(snap.timestamp, t0)

    def test_snapshot_no_crash_without_nvidia_smi(self):
        """gpu_memory_mib is None when nvidia-smi not available — must not raise."""
        snap = ClusterProbes().snapshot()
        # No assertion on value; just must not crash
        _ = snap.gpu_memory_mib


# ---------------------------------------------------------------------------
# ProbeDelta computation
# ---------------------------------------------------------------------------

class TestProbeDelta(unittest.TestCase):

    def _delta_from(self, before_kwargs, after_kwargs) -> ProbeDelta:
        before = _make_snapshot(**before_kwargs)
        after = _make_snapshot(**after_kwargs)
        return ClusterProbes().delta(before, after)

    def test_proc_rchar_delta(self):
        d = self._delta_from(
            {"proc_rchar": 1000, "proc_read_bytes": 200},
            {"proc_rchar": 5000, "proc_read_bytes": 600},
        )
        self.assertEqual(d.proc_rchar_delta, 4000)
        self.assertEqual(d.proc_read_bytes_delta, 400)

    def test_cache_hit_ratio_all_cache(self):
        """read_bytes delta = 0 → everything from page cache."""
        d = self._delta_from(
            {"proc_rchar": 0, "proc_read_bytes": 0},
            {"proc_rchar": 10_000_000, "proc_read_bytes": 0},
        )
        self.assertIsNotNone(d.cache_hit_ratio)
        self.assertAlmostEqual(d.cache_hit_ratio, 1.0)

    def test_cache_hit_ratio_all_storage(self):
        """rchar ≈ read_bytes → everything from storage (NFS/disk)."""
        d = self._delta_from(
            {"proc_rchar": 0, "proc_read_bytes": 0},
            {"proc_rchar": 10_000_000, "proc_read_bytes": 9_900_000},
        )
        self.assertIsNotNone(d.cache_hit_ratio)
        self.assertAlmostEqual(d.cache_hit_ratio, 0.01, delta=0.02)

    def test_cache_hit_ratio_none_when_rchar_zero(self):
        """Avoid divide-by-zero: ratio is None when rchar delta = 0."""
        d = self._delta_from(
            {"proc_rchar": 100, "proc_read_bytes": 0},
            {"proc_rchar": 100, "proc_read_bytes": 0},
        )
        self.assertIsNone(d.cache_hit_ratio)

    def test_net_delta(self):
        d = self._delta_from(
            {"net_bytes_recv": 1_000_000},
            {"net_bytes_recv": 3_500_000_000},
        )
        self.assertEqual(d.net_recv_bytes_delta, 3_499_000_000)

    def test_gpu_vram_delta(self):
        d = self._delta_from(
            {"gpu_memory_mib": {0: 2000, 1: 1800}},
            {"gpu_memory_mib": {0: 38000, 1: 37500}},
        )
        self.assertEqual(d.gpu_vram_delta_mib, {0: 36000, 1: 35700})
        self.assertEqual(d.gpu_vram_total_delta_mib, 71700)

    def test_none_fields_dont_crash(self):
        """All-None snapshot must still produce a delta without exceptions."""
        d = self._delta_from({}, {})
        self.assertIsNone(d.proc_rchar_delta)
        self.assertIsNone(d.cache_hit_ratio)
        self.assertEqual(d.likely_source, "unknown")

    def test_elapsed_s(self):
        t = time.time()
        before = _make_snapshot(timestamp=t)
        after = _make_snapshot(timestamp=t + 30.0)
        d = ClusterProbes().delta(before, after)
        self.assertAlmostEqual(d.elapsed_s, 30.0, delta=0.01)


# ---------------------------------------------------------------------------
# Source inference
# ---------------------------------------------------------------------------

class TestInferSource(unittest.TestCase):

    def _make_delta(self, **kwargs) -> ProbeDelta:
        d = ProbeDelta(elapsed_s=60.0)
        for k, v in kwargs.items():
            setattr(d, k, v)
        return d

    def test_page_cache_from_high_hit_ratio(self):
        d = self._make_delta(
            proc_rchar_delta=10_000_000,
            proc_read_bytes_delta=0,
            cache_hit_ratio=1.0,
        )
        source, notes = _infer_source(d)
        self.assertEqual(source, "page_cache")
        self.assertTrue(any("page cache" in n for n in notes))

    def test_nfs_from_low_hit_ratio(self):
        d = self._make_delta(
            proc_rchar_delta=10_000_000,
            proc_read_bytes_delta=9_800_000,
            cache_hit_ratio=0.02,
        )
        source, notes = _infer_source(d)
        self.assertEqual(source, "nfs")
        self.assertTrue(any("storage" in n for n in notes))

    def test_nfs_corroborated_by_network(self):
        d = self._make_delta(
            proc_rchar_delta=10_000_000,
            proc_read_bytes_delta=9_800_000,
            cache_hit_ratio=0.02,
            net_recv_bytes_delta=9_500_000_000,  # 9.5 GB recv
        )
        source, notes = _infer_source(d)
        self.assertEqual(source, "nfs")
        self.assertTrue(any("NFS-over-TCP" in n for n in notes))

    def test_page_cache_with_low_network(self):
        d = self._make_delta(
            cache_hit_ratio=0.99,
            net_recv_bytes_delta=5_000,   # tiny network delta
        )
        source, _ = _infer_source(d)
        self.assertEqual(source, "page_cache")

    def test_mixed(self):
        d = self._make_delta(cache_hit_ratio=0.50)
        source, _ = _infer_source(d)
        self.assertEqual(source, "mixed")

    def test_unknown_when_no_data(self):
        d = ProbeDelta(elapsed_s=5.0)
        source, notes = _infer_source(d)
        self.assertEqual(source, "unknown")
        self.assertTrue(any("unavailable" in n for n in notes))

    def test_gpu_vram_note_large_delta(self):
        d = self._make_delta(
            cache_hit_ratio=0.02,
            gpu_vram_delta_mib={0: 40000, 1: 40000},
            gpu_vram_total_delta_mib=80000,
        )
        _, notes = _infer_source(d)
        self.assertTrue(any("GPU memory" in n or "VRAM" in n for n in notes))

    def test_gpu_vram_note_small_delta(self):
        d = self._make_delta(
            cache_hit_ratio=0.02,
            gpu_vram_delta_mib={0: 10},
            gpu_vram_total_delta_mib=10,
        )
        _, notes = _infer_source(d)
        self.assertTrue(any("deferred" in n or "not yet loaded" in n for n in notes))


# ---------------------------------------------------------------------------
# LoadProbeContext
# ---------------------------------------------------------------------------

class TestLoadProbeContext(unittest.TestCase):

    def test_context_manager_sets_before_after_delta(self):
        probes = ClusterProbes()
        with LoadProbeContext(probes) as ctx:
            time.sleep(0.05)   # tiny work
        self.assertIsNotNone(ctx.before)
        self.assertIsNotNone(ctx.after)
        self.assertIsNotNone(ctx.delta)
        self.assertGreater(ctx.delta.elapsed_s, 0.0)

    def test_default_probes_created_if_none(self):
        with LoadProbeContext() as ctx:
            pass
        self.assertIsNotNone(ctx.delta)

    def test_summary_line_no_crash(self):
        with LoadProbeContext() as ctx:
            pass
        line = ctx.delta.summary_line()
        self.assertIn("source=", line)
        self.assertIn("elapsed=", line)


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------

class TestCheckAvailability(unittest.TestCase):

    def test_returns_dict_with_expected_keys(self):
        avail = ClusterProbes().check_availability()
        self.assertIn("proc_io", avail)
        self.assertIn("psutil", avail)
        self.assertIn("nvidia_smi", avail)
        self.assertIn("vmtouch", avail)

    def test_all_values_are_bool(self):
        avail = ClusterProbes().check_availability()
        for key, val in avail.items():
            self.assertIsInstance(val, bool, f"{key} should be bool")

    def test_psutil_available(self):
        avail = ClusterProbes().check_availability()
        self.assertTrue(avail["psutil"])

    def test_proc_io_available_on_linux(self):
        import platform
        if platform.system() != "Linux":
            self.skipTest("Not Linux")
        avail = ClusterProbes().check_availability()
        self.assertTrue(avail["proc_io"])


# ---------------------------------------------------------------------------
# page_cache_residency
# ---------------------------------------------------------------------------

class TestPageCacheResidency(unittest.TestCase):

    def test_returns_none_for_nonexistent_path(self):
        probes = ClusterProbes()
        result = probes.page_cache_residency("/nonexistent/path/to/weights.pt")
        self.assertIsNone(result)

    def test_returns_none_when_vmtouch_absent(self):
        probes = ClusterProbes()
        with patch("runtime.measurement.cluster_probes._vmtouch_available", return_value=False):
            result = probes.page_cache_residency("/tmp")
        # vmtouch not available → subprocess won't find it → returns None
        # (path exists but vmtouch absent → _run_vmtouch returns None via exception)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Integration: ModelPrefetchExecutor with probes
# ---------------------------------------------------------------------------

class TestModelPrefetchWithProbes(unittest.TestCase):

    def test_probe_delta_in_result_when_probes_set(self):
        """probe_delta key should appear in get_result() when probes are provided."""
        orch = FakeModelOrchestrator(load_times={"model": 0.1})
        probes = ClusterProbes()
        executor = ModelPrefetchExecutor(orch, probes=probes)
        task = _make_task("model")
        executor.start(task)

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)

        result = executor.get_result(task)
        self.assertIn("probe_delta", result)
        delta_dict = result["probe_delta"]
        self.assertIn("likely_source", delta_dict)
        self.assertIn("elapsed_s", delta_dict)
        executor.shutdown(wait=False)

    def test_no_probe_delta_when_probes_not_set(self):
        """Without probes, probe_delta must not appear in result."""
        orch = FakeModelOrchestrator(load_times={"model": 0.1})
        executor = ModelPrefetchExecutor(orch)   # no probes=
        task = _make_task("model")
        executor.start(task)

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)

        result = executor.get_result(task)
        self.assertNotIn("probe_delta", result)
        executor.shutdown(wait=False)

    def test_probe_delta_contains_inference(self):
        """likely_source should be a non-empty string."""
        orch = FakeModelOrchestrator(load_times={"model": 0.1})
        executor = ModelPrefetchExecutor(orch, probes=ClusterProbes())
        task = _make_task("model")
        executor.start(task)

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            if executor.is_complete(task):
                break
            time.sleep(0.02)

        result = executor.get_result(task)
        source = result["probe_delta"]["likely_source"]
        self.assertIn(source, {"nfs", "page_cache", "mixed", "unknown"})
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# to_dict / summary_line
# ---------------------------------------------------------------------------

class TestProbeDeltaOutput(unittest.TestCase):

    def test_to_dict_is_serializable(self):
        import json
        d = ProbeDelta(
            elapsed_s=45.0,
            proc_rchar_delta=1_000_000,
            proc_read_bytes_delta=900_000,
            cache_hit_ratio=0.10,
            net_recv_bytes_delta=800_000,
            gpu_vram_delta_mib={0: 35000},
            gpu_vram_total_delta_mib=35000,
            likely_source="nfs",
            inference_notes=["note 1", "note 2"],
        )
        as_dict = d.to_dict()
        # Must be JSON-serialisable
        json.dumps(as_dict)

    def test_summary_line_contains_source(self):
        d = ProbeDelta(
            elapsed_s=10.0,
            cache_hit_ratio=0.05,
            likely_source="nfs",
        )
        d.likely_source, d.inference_notes = _infer_source(d)
        line = d.summary_line()
        self.assertIn("source=", line)


if __name__ == "__main__":
    unittest.main()
