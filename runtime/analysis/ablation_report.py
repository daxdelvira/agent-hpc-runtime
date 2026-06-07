"""
analysis/ablation_report.py — Side-by-side ablation comparison table.

Reads summary JSON files produced by atomagents_exp2.py and prints a table
with one row per condition. Designed to answer: which component contributes
most to system benefit?

Conditions expected (one summary JSON per condition):
  baseline            — no runtime, pure AtomAgents
  full_system         — all components active
  no_plan             — plan extraction disabled
  no_diverg_guard     — divergence cancellation disabled
  naive_prefetch      — confidence threshold = 0
  no_vllmmodel        — model prefetch disabled
  no_datafile         — data prefetch disabled
  pred_learned        — learned predictor vs mock

Metrics reported per condition:
  wall_time_s         — end-to-end wallclock time
  total_stall_s       — sum of time consumers waited for resources
  total_overlap_s     — sum of time prefetches ran during compute
  total_benefit_s     — estimated time saved vs cold-start
  total_waste_s       — time spent on completed prefetches never consumed
  precision           — P(correct | validated)
  honest_accuracy     — P(correct | all predictions)
  divergence_count    — number of mispredictions that triggered cancellation
  wasted_prefetch     — number of prefetches cancelled after completion

Usage
-----
    python runtime/analysis/ablation_report.py results/summary_*.json
    python runtime/analysis/ablation_report.py results/summary_*.json --json
    python runtime/analysis/ablation_report.py results/summary_*.json --csv ablation.csv
    python runtime/analysis/ablation_report.py results/summary_*.json --baseline results/summary_baseline.json
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_summaries(paths: list[str]) -> list[dict[str, Any]]:
    records = []
    for pattern in paths:
        for p in sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]:
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    data.setdefault("_source", str(p.name))
                    records.append(data)
                except Exception as exc:
                    print(f"  WARNING: could not read {p}: {exc}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _fmt_s(v, decimals=1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}s"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v) -> str:
    if v is None:
        return "—"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "—"


def _speedup(baseline_wall: float | None, run_wall: float | None) -> str:
    if baseline_wall is None or run_wall is None or run_wall <= 0:
        return "—"
    try:
        ratio = baseline_wall / run_wall
        delta = baseline_wall - run_wall
        sign = "+" if delta >= 0 else ""
        return f"{ratio:.2f}×  ({sign}{delta:.0f}s)"
    except (TypeError, ValueError, ZeroDivisionError):
        return "—"


def _condition_label(rec: dict) -> str:
    """Prefer explicit condition field; fall back to mode or run_id."""
    c = rec.get("condition")
    if c:
        return c
    mode = rec.get("mode", "")
    if mode == "baseline":
        return "baseline"
    return rec.get("run_id", "unknown")[:12]


CONDITION_ORDER = [
    "baseline",
    "full_system",
    "no_plan",
    "no_diverg_guard",
    "naive_prefetch",
    "no_vllmmodel",
    "no_datafile",
    "pred_mock",
    "pred_learned",
]


def _sort_key(rec: dict) -> tuple:
    label = _condition_label(rec)
    try:
        idx = CONDITION_ORDER.index(label)
    except ValueError:
        idx = len(CONDITION_ORDER)
    return (idx, label)


# ---------------------------------------------------------------------------
# Main table printer
# ---------------------------------------------------------------------------

def print_ablation_table(records: list[dict], baseline_wall: float | None = None) -> None:
    if not records:
        print("  No records.")
        return

    records = sorted(records, key=_sort_key)

    # If no external baseline_wall, try to find it from records
    if baseline_wall is None:
        for r in records:
            if _condition_label(r) == "baseline" or r.get("mode") == "baseline":
                baseline_wall = r.get("wall_time_s")
                if baseline_wall:
                    break

    cols: list[tuple[str, int]] = [
        ("Condition",        18),
        ("Mode",              9),
        ("Wall(s)",           9),
        ("Stall(s)",          9),
        ("Overlap(s)",       10),
        ("Benefit(s)",       10),
        ("Waste(s)",          9),
        ("Precision",        10),
        ("HonestAcc",        10),
        ("Diverg",            6),
        ("Speedup vs BL",    16),
    ]

    print(f"\n{'='*120}")
    print("  Ablation Study — AtomAgents Exp2")
    print(f"{'='*120}")
    header = "  " + "  ".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("  " + "  ".join("-" * w for _, w in cols))

    for r in records:
        cond    = _condition_label(r)
        mode    = r.get("mode", "")[:9]
        wall    = r.get("wall_time_s")
        stall   = r.get("total_stall_s")
        overlap = r.get("total_overlap_s")
        benefit = r.get("total_benefit_s")
        waste   = r.get("total_waste_s")
        prec    = r.get("precision") or r.get("accuracy")
        hacc    = r.get("honest_accuracy")
        diverg  = r.get("divergence_count")
        spdup   = _speedup(baseline_wall, wall)

        vals = [
            f"{cond:<18}",
            f"{mode:<9}",
            f"{_fmt_s(wall):<9}",
            f"{_fmt_s(stall):<9}",
            f"{_fmt_s(overlap):<10}",
            f"{_fmt_s(benefit):<10}",
            f"{_fmt_s(waste):<9}",
            f"{_fmt_pct(prec):<10}",
            f"{_fmt_pct(hacc):<10}",
            f"{_fmt_int(diverg):<6}",
            f"{spdup:<16}",
        ]
        print("  " + "  ".join(vals))

    print()
    _print_sensitivity_summary(records, baseline_wall)


def _print_sensitivity_summary(records: list[dict], baseline_wall: float | None) -> None:
    """
    Print a ranked sensitivity list: which ablation hurts the most?
    Ranked by wall-time regression relative to full_system.
    """
    full = next(
        (r for r in records if _condition_label(r) == "full_system"), None
    )
    if full is None:
        return

    full_wall = full.get("wall_time_s")
    if not full_wall:
        return

    regressions: list[tuple[float, str]] = []
    for r in records:
        cond = _condition_label(r)
        if cond in ("baseline", "full_system"):
            continue
        w = r.get("wall_time_s")
        if w:
            regressions.append((w - full_wall, cond))

    if not regressions:
        return

    regressions.sort(reverse=True)   # worst regression first
    print("  Sensitivity ranking (wall-time regression vs full_system):")
    for delta, cond in regressions:
        sign = "+" if delta >= 0 else ""
        note = "  ← worst" if delta == regressions[0][0] else ""
        print(f"    {cond:<22}  {sign}{delta:+.0f}s{note}")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], path: str) -> None:
    fields = [
        "condition", "mode", "predictor", "run_id",
        "wall_time_s", "total_stall_s", "total_overlap_s",
        "total_benefit_s", "total_waste_s",
        "precision", "honest_accuracy", "divergence_count",
        "prefetch_started", "prefetch_completed", "wasted_prefetch",
        "prediction_count", "hit_count", "miss_count",
    ]
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=_sort_key):
            row = {k: r.get(k) for k in fields}
            row["condition"] = _condition_label(r)
            w.writerow(row)
    print(f"  CSV written to: {path}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json(records: list[dict]) -> dict:
    return {
        "n_conditions": len(records),
        "conditions": [
            {
                "condition": _condition_label(r),
                "mode": r.get("mode"),
                "predictor": r.get("predictor"),
                "run_id": r.get("run_id"),
                "wall_time_s": r.get("wall_time_s"),
                "total_stall_s": r.get("total_stall_s"),
                "total_overlap_s": r.get("total_overlap_s"),
                "total_benefit_s": r.get("total_benefit_s"),
                "total_waste_s": r.get("total_waste_s"),
                "precision": r.get("precision") or r.get("accuracy"),
                "honest_accuracy": r.get("honest_accuracy"),
                "divergence_count": r.get("divergence_count"),
                "prefetch_started": r.get("prefetch_started"),
                "wasted_prefetch": r.get("wasted_prefetch"),
            }
            for r in sorted(records, key=_sort_key)
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Print an ablation comparison table from summary JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="summary_*.json files (glob patterns supported)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline summary JSON (used for speedup column; "
             "auto-detected from files if omitted)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of table",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Write results to CSV file at PATH",
    )
    args = parser.parse_args(argv)

    records = load_summaries(args.files)
    if not records:
        print("No summary JSON files found.", file=sys.stderr)
        sys.exit(1)

    baseline_wall: float | None = None
    if args.baseline:
        try:
            with open(args.baseline) as f:
                bl = json.load(f)
            baseline_wall = bl.get("wall_time_s")
        except Exception as exc:
            print(f"  WARNING: could not read baseline {args.baseline}: {exc}",
                  file=sys.stderr)

    if args.json:
        print(json.dumps(build_json(records), indent=2))
        return

    print_ablation_table(records, baseline_wall)

    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
