"""tests/test_format_activation_bench.py — guard the invariants that make
experiments/bench_format_activation.py's numbers mean what the paper says.

No GPU, no SLURM, no LAMMPS.  Everything here runs on a few MB in /tmp.

The failure modes these tests exist to catch are the ones that silently turn a
measurement into a fabrication:
  * the "same logical content" claim degrading into "similar content", so a
    format comparison is really a payload comparison;
  * ASCII bytes/line drifting off 19, which would break the read-scales-with-
    bytes / parse-scales-with-lines argument;
  * evict() quietly not evicting, so a warm rung gets reported as cold;
  * the io/activation split arithmetic disagreeing with its own inputs.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import bench_format_activation as bfa  # noqa: E402


class TestAsciiLayout(unittest.TestCase):
    def test_ascii_is_exactly_19_bytes_per_value(self):
        """19.0 B/line is measured from w_eam4_big.fs (3320490868 B /
        174762677 lines).  The synthetic ASCII must match it exactly, or the
        bytes-vs-lines scaling argument compares two different things."""
        for v in (1.0, 1.5, 9.999999999999, 99.99999999999, 10.0, 99.0):
            s = bfa.ASCII_FMT % v
            self.assertEqual(len(s), bfa.ASCII_BYTES_PER_VALUE, repr(s))

    def test_generated_file_size_is_exact(self):
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_"))
        try:
            gen = bfa.generate(d, 5000, seed=1, formats=["npy"])
            p = Path(gen["paths"]["ascii_loadtxt"])
            self.assertEqual(p.stat().st_size,
                             5000 * bfa.ASCII_BYTES_PER_VALUE)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestContentIdentity(unittest.TestCase):
    def test_all_float64_formats_hold_bit_identical_content(self):
        """The whole experiment rests on this: if the formats do not hold the
        same bits, any timing difference could be a payload difference."""
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_"))
        try:
            fmts = ["ascii_loadtxt", "ascii_pandas", "npy", "npz_deflate"]
            try:
                import h5py  # noqa: F401
                fmts.append("hdf5")
            except ImportError:
                pass
            gen = bfa.generate(d, 20000, seed=7, formats=fmts)
            import hashlib
            for fmt in fmts:
                arr = bfa.LOADERS[fmt](gen["paths"][fmt])
                self.assertEqual(arr.dtype, np.float64, fmt)
                self.assertEqual(arr.size, 20000, fmt)
                md5 = hashlib.md5(memoryview(arr).cast("B")).hexdigest()
                self.assertEqual(md5, gen["ref_md5"],
                                 f"{fmt} does not hold the reference bits")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_raw_f32_is_declared_lossy(self):
        """raw_f32 is deliberately NOT bit-identical -- it is the 'cheap
        transform' rung.  Assert the loss so nobody later reports it as an
        identical-content format."""
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_"))
        try:
            gen = bfa.generate(d, 20000, seed=7, formats=["raw_f32"])
            arr = bfa.LOADERS["raw_f32"](gen["paths"]["raw_f32"])
            ref = bfa.LOADERS["ascii_loadtxt"](gen["paths"]["ascii_loadtxt"])
            self.assertEqual(arr.dtype, np.float64)
            np.testing.assert_allclose(arr, ref, rtol=1e-6)
            self.assertFalse(np.array_equal(arr, ref))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCacheControl(unittest.TestCase):
    def test_evict_then_mincore_reports_cold_on_local_tmp(self):
        """A rung is only 'cold' if mincore says so.  If this fails on the
        machine under test, fadvise is not evicting there (Lustre was measured
        leaving 56.2% resident) and cold rungs from that host are invalid."""
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_", dir="/tmp"))
        try:
            p = d / "blob.bin"
            p.write_bytes(os.urandom(8 << 20))
            self.assertGreater(bfa.resident_fraction(str(p)), 0.5,
                               "just-written file should be cached")
            bfa.evict(str(p))
            self.assertLess(bfa.resident_fraction(str(p)), 0.05,
                            "fadvise(DONTNEED) did not evict on /tmp")
            t = bfa.timed_read(str(p))
            self.assertGreater(t, 0.0)
            self.assertGreater(bfa.resident_fraction(str(p)), 0.5,
                               "reading should repopulate the page cache")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestChildMeasurement(unittest.TestCase):
    def test_child_reports_own_cpu_and_rss(self):
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_", dir="/tmp"))
        try:
            gen = bfa.generate(d, 200000, seed=3, formats=["npy"])
            rec = bfa.run_child("npy", gen["paths"]["npy"], d)
            self.assertNotIn("error", rec)
            self.assertEqual(rec["n"], 200000)
            self.assertEqual(rec["nbytes"], 200000 * 8)
            self.assertGreater(rec["wall_s"], 0.0)
            # the inner bracket must exclude interpreter startup
            self.assertLess(rec["wall_s"], rec["proc_wall_s"])
            self.assertGreaterEqual(rec["utime_s"] + rec["stime_s"], 0.0)
            self.assertGreater(rec["child_maxrss_kb"], 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestSplitArithmetic(unittest.TestCase):
    def test_measure_split_is_internally_consistent(self):
        d = Path(tempfile.mkdtemp(prefix="fmtbench_t_", dir="/tmp"))
        try:
            gen = bfa.generate(d, 300000, seed=5, formats=["ascii_loadtxt"])
            row = bfa.measure("ascii_loadtxt",
                              gen["paths"]["ascii_loadtxt"], d)
            self.assertIsNone(row["error"])
            self.assertEqual(row["md5"], gen["ref_md5"])
            self.assertAlmostEqual(row["io_share_s"],
                                   max(row["load_cold_s"] - row["load_warm_s"],
                                       0.0), places=9)
            self.assertAlmostEqual(row["io_share_s_raw"],
                                   row["load_cold_s"] - row["load_warm_s"],
                                   places=9)
            if row["warm_exceeds_cold"]:
                # I/O share is below the noise floor for this rung; the shares
                # do NOT sum to 100 and the script must say so rather than
                # quietly clamping.
                self.assertGreater(row["activation_pct"], 100.0)
                self.assertEqual(row["io_share_pct"], 0.0)
            else:
                self.assertAlmostEqual(
                    row["io_share_pct"] + row["activation_pct"], 100.0,
                    places=6)
            # the cold rung must actually have been cold, and the warm rung warm
            self.assertLess(row["load_cold_resident_before"], 0.05)
            self.assertGreater(row["load_warm_resident_before"], 0.95)
            # cpu/wall is what answers "can more cores hide this?"
            self.assertGreater(row["warm_cpu_per_wall"], 0.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
