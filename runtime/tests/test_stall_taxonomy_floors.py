"""
tests/test_stall_taxonomy_floors.py — regime-independence of the prefetch stall
taxonomy (scripts/extract_prefetch_lifecycle.py), no GPUs required.

The taxonomy's window floors used to be the fixed constants W_MIN_S = 15 s and
SPINUP_FLOOR_S = 30 s, calibrated when a vLLM engine could only be brought up by
a 500-1300 s cold boot.  The sleep-mode arm wakes a parked engine in 0.8-2.1 s;
against cold-boot-sized floors every such wake would be stamped `no_window` —
the instrument would report the arm's success case as a failure.

These tests pin the two properties that make the taxonomy trustworthy across
both regimes:

  1. COLD-BOOT REGIME IS UNCHANGED.  With a bring-up of hundreds of seconds the
     bring-up-relative terms exceed their caps, min() selects the cap, and every
     classification is identical to the pre-2026-08 constants.  This is the
     guard against silently reclassifying the existing corpus, on which the
     paper's central diagnostic rests.

  2. WAKE REGIME IS CLASSIFIED ON ITS OWN SCALE.  A 1.5 s wake with a 5 s window
     is `residual_partial`/`late_start` (we had ample room), never `no_window`.

Measured bring-up costs used as fixtures (recomputed 2026-08-03; never pooled
across platforms — L40S and Blackwell figures are kept separate by construction
because the classifier reads each row's OWN transfer_s):
    cold engine boot   690.8 s (L40S)      975.5 s (Blackwell, first engine)
    L1 sleep wake        2.02-2.10 s (L40S)  1.49-1.54 s (Blackwell)
    L2 sleep wake        0.77 s (32B, L40S)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from extract_prefetch_lifecycle import (  # noqa: E402
    NO_WINDOW_FRAC,
    SPINUP_CAP_S,
    SPINUP_FRAC,
    W_MIN_CAP_S,
    _stall_class,
    bringup_cost_s,
    window_floors,
)

# Measured bring-up costs (seconds).
COLD_BOOT_L40S = 690.8
COLD_BOOT_BLACKWELL = 975.5
WAKE_L1_BLACKWELL = 1.49
WAKE_L1_L40S = 2.10
WAKE_L2_L40S = 0.77


def row(**kw):
    """A window-branch row: exposed stall, prefetch task existed and was used."""
    r = {
        "outcome": "useful",
        "resource_type": "vllm_model",
        "exposure_s": 5.0,
        "gate_group": "",
        "start_delay_s": 0.0,
    }
    r.update(kw)
    return r


def classify(**kw):
    return _stall_class(row(**kw), set())


class TestBringupCostProvenance(unittest.TestCase):
    """The floors must be driven by a MEASURED cost wherever one exists."""

    def test_measured_transfer_preferred_over_estimate(self):
        cost, prov = bringup_cost_s(
            {"transfer_s": WAKE_L1_BLACKWELL, "estimated_load_s": 120.0})
        self.assertEqual(prov, "measured")
        self.assertAlmostEqual(cost, WAKE_L1_BLACKWELL)

    def test_estimate_used_when_transfer_absent(self):
        cost, prov = bringup_cost_s(
            {"transfer_s": None, "estimated_load_s": 130.0})
        self.assertEqual(prov, "estimated")
        self.assertAlmostEqual(cost, 130.0)

    def test_unknown_when_neither_available(self):
        cost, prov = bringup_cost_s({"transfer_s": None,
                                     "estimated_load_s": None})
        self.assertIsNone(cost)
        self.assertEqual(prov, "unknown")

    def test_zero_transfer_is_not_a_bringup_cost(self):
        """A 0 s elapsed is missing data, not a free bring-up; it must not
        collapse the floors to zero and make everything look hideable."""
        cost, prov = bringup_cost_s({"transfer_s": 0.0,
                                     "estimated_load_s": 120.0})
        self.assertEqual(prov, "estimated")
        self.assertAlmostEqual(cost, 120.0)


class TestColdBootRegimeUnchanged(unittest.TestCase):
    """Guard: the existing (cold-boot) corpus must not be reclassified."""

    def test_floors_saturate_at_the_legacy_caps(self):
        for bringup in (COLD_BOOT_L40S, COLD_BOOT_BLACKWELL, 1131.8, 154.27):
            w_min, spinup, got, prov = window_floors(
                {"transfer_s": bringup, "resource_type": "vllm_model"})
            self.assertEqual(w_min, W_MIN_CAP_S, f"bringup={bringup}")
            self.assertEqual(spinup, SPINUP_CAP_S, f"bringup={bringup}")
            self.assertEqual(prov, "measured")
            self.assertAlmostEqual(got, bringup)

    def test_cold_boot_at_need_time_is_no_window(self):
        """The dominant chemgraph_swap shape: engine task starts at need time.
        window 0.03 s against a 690.8 s boot — genuinely no window."""
        self.assertEqual(
            classify(window_s=0.03, transfer_s=COLD_BOOT_L40S,
                     estimated_load_s=120.0, exposure_s=690.8),
            "no_window")

    def test_cold_boot_short_window_still_no_window(self):
        """A 12 s window is below the 15 s cap AND far below the boot cost."""
        self.assertEqual(
            classify(window_s=12.0, resource_type="model_cache",
                     transfer_s=48.08, estimated_load_s=130.0),
            "no_window")

    def test_megammap_stage_shape_is_window_too_small(self):
        """The chemgraph_swap/megammap_stage reference shape: a real 21.9 s
        window, but the 130 s staging load cannot fit inside it."""
        self.assertEqual(
            classify(window_s=21.94, resource_type="model_cache",
                     transfer_s=None, estimated_load_s=130.0),
            "window_too_small")

    def test_boundary_just_below_cap_is_no_window(self):
        self.assertEqual(
            classify(window_s=W_MIN_CAP_S - 0.01, transfer_s=COLD_BOOT_L40S),
            "no_window")

    def test_boundary_just_above_cap_is_not_no_window(self):
        self.assertNotEqual(
            classify(window_s=W_MIN_CAP_S + 0.01, transfer_s=COLD_BOOT_L40S),
            "no_window")


class TestWakeRegime(unittest.TestCase):
    """The sleep-mode arm must be scored on its own scale."""

    def test_floors_scale_down_to_the_wake_cost(self):
        for wake in (WAKE_L2_L40S, WAKE_L1_BLACKWELL, WAKE_L1_L40S):
            w_min, spinup, _, _ = window_floors(
                {"transfer_s": wake, "resource_type": "vllm_model"})
            self.assertAlmostEqual(w_min, NO_WINDOW_FRAC * wake)
            self.assertAlmostEqual(spinup, SPINUP_FRAC * wake)
            self.assertLess(w_min, W_MIN_CAP_S)
            self.assertLess(spinup, SPINUP_CAP_S)

    def test_ample_window_over_a_wake_is_never_no_window(self):
        """THE REGRESSION THIS MODULE EXISTS FOR.

        A 5 s window against a 1.49 s Blackwell L1 wake is 3.4x the entire cost
        of the operation.  Under the old fixed floors (15 s / 30 s) this was
        `no_window` — the arm's success case reported as a failure."""
        for wake in (WAKE_L2_L40S, WAKE_L1_BLACKWELL, WAKE_L1_L40S):
            cls = classify(window_s=5.0, transfer_s=wake, exposure_s=0.4)
            self.assertNotEqual(cls, "no_window", f"wake={wake}")
            self.assertNotEqual(cls, "window_too_small", f"wake={wake}")

    def test_wake_with_genuinely_no_window_is_still_no_window(self):
        """Regime-independence cuts both ways: a 0.2 s window against a 1.49 s
        wake really is no window, and must still be labelled one."""
        self.assertEqual(
            classify(window_s=0.2, transfer_s=WAKE_L1_BLACKWELL,
                     exposure_s=1.3),
            "no_window")

    def test_wake_window_between_the_two_floors_is_window_too_small(self):
        """window 2.0 s: longer than the 1.49 s wake (so not `no_window`) but
        shorter than wake + spin-up margin (2.98 s) — unhideable."""
        self.assertEqual(
            classify(window_s=2.0, transfer_s=WAKE_L1_BLACKWELL,
                     exposure_s=0.6),
            "window_too_small")

    def test_late_start_detected_in_the_wake_regime(self):
        """Ample window (10 s vs a 1.49 s wake) but the task started 9 s late,
        so the remaining 1 s could not cover the wake."""
        self.assertEqual(
            classify(window_s=10.0, transfer_s=WAKE_L1_BLACKWELL,
                     start_delay_s=9.0, exposure_s=0.5),
            "late_start")


class TestCheapResourceRegression(unittest.TestCase):
    """The same defect already bit warm mace loads in the existing corpus."""

    def test_warm_mace_load_with_huge_window_is_not_no_window(self):
        """Observed shape (chemgraph_swap/naive_prefetch, mace_model): a 13.1 s
        window against a 0.12 s warm load — 109x — yet the task started 326 s
        late.  That is `late_start`, not `no_window`."""
        self.assertEqual(
            classify(window_s=13.14, resource_type="mace_model",
                     transfer_s=0.12, start_delay_s=326.27, exposure_s=313.29),
            "late_start")

    def test_mace_window_shorter_than_its_load_stays_no_window(self):
        """The chemgraph_swap/full_system reference rows: window 2.59 s against
        a 4.91 s load. Shorter than the load, so still `no_window` — these must
        NOT move, or the 198.4 s reference value breaks."""
        self.assertEqual(
            classify(window_s=2.59, resource_type="mace_model",
                     transfer_s=4.91, exposure_s=2.33),
            "no_window")


class TestNonWindowClassesUnaffected(unittest.TestCase):
    """Earlier branches must keep short-circuiting before the floors apply."""

    def test_baseline_and_no_prediction_and_skip_and_direct(self):
        for outcome, expect in (("no_prefetch_config", "baseline_no_prefetch"),
                                ("no_prediction", "no_prediction"),
                                ("direct_prefetch", "residual_partial")):
            self.assertEqual(classify(outcome=outcome, window_s=0.01,
                                      transfer_s=WAKE_L1_BLACKWELL), expect)
        self.assertEqual(
            classify(outcome="skipped", decision_reason="low_confidence",
                     window_s=0.01),
            "policy_skip:low_confidence")

    def test_negligible_exposure_is_unclassified(self):
        self.assertEqual(classify(exposure_s=0.05, window_s=0.01), "")

    def test_missing_window_is_unattributed(self):
        self.assertEqual(classify(window_s=None, exposure_s=3.0),
                         "unattributed")

    def test_gpu_serialization_residual_precedes_the_floors(self):
        r = row(window_s=0.01, transfer_s=WAKE_L1_BLACKWELL,
                gate_group="swap_wait:m:0")
        self.assertEqual(_stall_class(r, {"swap_wait:m:0"}),
                         "gpu_serialization_residual")

    def test_unknown_bringup_falls_back_to_the_legacy_caps(self):
        w_min, spinup, got, prov = window_floors(
            {"transfer_s": None, "estimated_load_s": None,
             "resource_type": "vllm_model"})
        self.assertEqual((w_min, spinup), (W_MIN_CAP_S, SPINUP_CAP_S))
        self.assertIsNone(got)
        self.assertEqual(prov, "unknown")


class TestAuditTrail(unittest.TestCase):
    """A class must be re-derivable from the CSV without the trace."""

    def test_floors_and_provenance_recorded_on_the_row(self):
        r = row(window_s=5.0, transfer_s=WAKE_L1_BLACKWELL, exposure_s=0.4)
        _stall_class(r, set())
        self.assertAlmostEqual(r["bringup_cost_s"], WAKE_L1_BLACKWELL)
        self.assertEqual(r["bringup_provenance"], "measured")
        self.assertAlmostEqual(r["w_min_s"], NO_WINDOW_FRAC * WAKE_L1_BLACKWELL)
        self.assertAlmostEqual(r["spinup_s"], SPINUP_FRAC * WAKE_L1_BLACKWELL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
