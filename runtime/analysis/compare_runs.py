"""
analysis/compare_runs.py — Compare multiple experiment runs side-by-side.

Reads summary JSON files produced by atomagents_exp2.py and/or chemgraph_exp.py
and prints a comparison table of the key metrics:

  run_id | workflow | mode | predictor | wall_time_s | accuracy | benefit_s | wasted_s

Also reads optional metrics CSV files to include token/GPU-cost data.

Usage
-----
    # Compare all summaries in results/
    python runtime/analysis/compare_runs.py results/summary_*.json

    # Include metrics CSVs
    python runtime/analysis/compare_runs.py results/summary_*.json \
        --csv results/atomagents_metrics_*.csv

    # JSON output (for programmatic use)
    python runtime/analysis/compare_runs.py results/summary_*.json --json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Summary JSON loader
# ---------------------------------------------------------------------------

def load_summary(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    data["_source"] = str(Path(path).name)
    return data


# ---------------------------------------------------------------------------
# Metrics CSV loader (reads the SUMMARY row from each CSV)
# ---------------------------------------------------------------------------

def load_metrics_csv(path: str) -> dict[str, Any]:
    """
    Read the SUMMARY row from an atomagents_metrics_*.csv file.
    Returns a dict with token/cost/timing data merged back into a run record.
    """
    result: dict[str, Any] = {"_csv_source": str(Path(path).name)}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("phase") == "SUMMARY":
                    result["run_id_csv"] = row.get("run_id", "")
                    result["mode_csv"] = row.get("mode", "")
                    result["total_tokens"] = _safe_int(row.get("total_tokens"))
                    result["prompt_tokens"] = _safe_int(row.get("prompt_tokens"))
                    result["completion_tokens"] = _safe_int(row.get("completion_tokens"))
                    result["cost_usd"] = _safe_float(row.get("cost_usd"))
                    result["wall_time_csv_s"] = _safe_float(row.get("duration_s"))
                    notes = row.get("notes", "")
                    # Parse key=value pairs from notes column
                    for kv in notes.split():
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            result[f"notes_{k}"] = v
                    break
    except Exception:
        pass
    return result


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Merge summary + CSV data by run_id
# ---------------------------------------------------------------------------

def merge_records(
    summaries: list[dict],
    csv_data: list[dict],
) -> list[dict]:
    """Join summaries and CSV data on run_id."""
    csv_by_run: dict[str, dict] = {d["run_id_csv"]: d for d in csv_data if "run_id_csv" in d}
    merged = []
    for s in summaries:
        rid = s.get("run_id", "")
        rec = dict(s)
        if rid in csv_by_run:
            rec.update({k: v for k, v in csv_by_run[rid].items() if k not in rec})
        merged.append(rec)
    return merged


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt_acc(s: dict) -> str:
    acc = s.get("accuracy")
    hits = s.get("hit_count", 0)
    misses = s.get("miss_count", 0)
    if acc is None:
        return "N/A"
    return f"{acc:.0%} ({hits}H/{misses}M)"


def _fmt_time(v) -> str:
    if v is None:
        return "—"
    return f"{float(v):.0f}s"


def _fmt_float(v, fmt=".1f") -> str:
    if v is None:
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def print_comparison_table(records: list[dict], verbose: bool = True) -> None:
    if not records:
        print("  No records to compare.")
        return

    # Sort: baseline first, then by mode, then by run_id
    mode_order = {"baseline": 0, "observe_only": 1, "simulated": 2, "real": 3}
    records = sorted(
        records,
        key=lambda r: (mode_order.get(r.get("mode", ""), 9), r.get("run_id", "")),
    )

    # Group by workflow if multiple workflows present
    workflows = {r.get("workflow", "atomagents") for r in records}
    for wf in sorted(workflows):
        wf_records = [r for r in records if r.get("workflow", "atomagents") == wf]
        _print_workflow_table(wf_records, wf, verbose)


def _print_workflow_table(records: list[dict], workflow: str, verbose: bool) -> None:
    cols = [
        ("Run ID",       10, lambda r: (r.get("run_id") or "")[:10]),
        ("Mode",          12, lambda r: r.get("mode", "")[:12]),
        ("Predictor",      8, lambda r: r.get("predictor", "mock")[:8]),
        ("Wall(s)",        7, lambda r: _fmt_time(r.get("wall_time_s"))),
        ("Accuracy",      14, _fmt_acc),
        ("Diverg",         6, lambda r: str(r.get("divergence_count", "—"))),
        ("PF strt",        7, lambda r: str(r.get("prefetch_started", "—"))),
        ("PF done",        7, lambda r: str(r.get("prefetch_completed", "—"))),
        ("PF wast",        7, lambda r: str(r.get("wasted_prefetch", "—"))),
        ("benefit_s",      9, lambda r: _fmt_float(
            r.get("total_benefit_s") or r.get("estimated_total_benefit_s"), ".1f",
        )),
        ("waste_s",        7, lambda r: _fmt_float(r.get("total_waste_s"), ".1f")),
    ]

    if any(r.get("total_tokens") for r in records):
        cols.append(("Tokens", 8, lambda r: _fmt_float(r.get("total_tokens"), ".0f")))

    print(f"\n{'='*72}")
    print(f"  Run Comparison — {workflow}")
    print(f"{'='*72}")

    header = "  " + "  ".join(f"{name:<{w}}" for name, w, _ in cols)
    print(header)
    print("  " + "  ".join("-" * w for _, w, _ in cols))

    for r in records:
        row = "  " + "  ".join(
            f"{fn(r):<{w}}" for _, w, fn in cols
        )
        print(row)

    print()

    # Compute speedup vs baseline
    baseline = next(
        (r for r in records if r.get("mode") == "baseline"), None
    )
    if baseline and len(records) > 1:
        bw = _safe_float(baseline.get("wall_time_s"))
        if bw:
            print("  Speedup vs baseline:")
            for r in records:
                if r.get("mode") == "baseline":
                    continue
                rw = _safe_float(r.get("wall_time_s"))
                if rw and rw > 0:
                    speedup = bw / rw
                    saved = bw - rw
                    rid = (r.get("run_id") or "")[:10]
                    mode = r.get("mode", "")
                    print(f"    {rid:<10}  {mode:<12}  {speedup:.2f}×  ({saved:+.0f}s)")
        print()


# ---------------------------------------------------------------------------
# Recommendations / quick summary
# ---------------------------------------------------------------------------

def print_quick_summary(records: list[dict]) -> None:
    """Print a 3-line research summary."""
    baseline = next((r for r in records if r.get("mode") == "baseline"), None)
    real_runs = [r for r in records if r.get("mode") == "real"]

    print("  Quick summary:")
    if baseline:
        print(f"    Baseline wall time : {_fmt_time(baseline.get('wall_time_s'))}")
    if real_runs:
        best = min(real_runs, key=lambda r: _safe_float(r.get("wall_time_s")) or 1e9)
        bw = _safe_float(best.get("wall_time_s"))
        bb = _safe_float(best.get("total_benefit_s") or best.get("estimated_total_benefit_s"))
        bbl = _safe_float(baseline.get("wall_time_s")) if baseline else None
        print(f"    Best real run      : {_fmt_time(bw)}  "
              f"(benefit_s={_fmt_float(bb)})")
        if bbl and bw:
            print(f"    Wall-time saving   : {bbl - bw:+.0f}s  ({(bbl-bw)/bbl:.1%} of baseline)")

    obs_runs = [r for r in records if r.get("mode") == "observe_only"]
    if obs_runs:
        avg_acc = sum(
            _safe_float(r.get("accuracy")) or 0.0 for r in obs_runs
        ) / len(obs_runs)
        print(f"    Avg predictor acc  : {avg_acc:.0%}  (observe_only runs)")
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(records: list[dict]) -> dict:
    return {
        "n_runs": len(records),
        "runs": [
            {
                "run_id": r.get("run_id"),
                "workflow": r.get("workflow", "atomagents"),
                "mode": r.get("mode"),
                "predictor": r.get("predictor"),
                "wall_time_s": r.get("wall_time_s"),
                "accuracy": r.get("accuracy"),
                "hit_count": r.get("hit_count"),
                "miss_count": r.get("miss_count"),
                "divergence_count": r.get("divergence_count"),
                "prefetch_started": r.get("prefetch_started"),
                "prefetch_completed": r.get("prefetch_completed"),
                "wasted_prefetch": r.get("wasted_prefetch"),
                "total_benefit_s": r.get("total_benefit_s"),
                "estimated_total_benefit_s": r.get("estimated_total_benefit_s"),
                "total_waste_s": r.get("total_waste_s"),
                "total_tokens": r.get("total_tokens"),
                "cost_usd": r.get("cost_usd"),
            }
            for r in records
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple experiment run summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="summary_*.json files (glob patterns supported)",
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[],
        help="atomagents_metrics_*.csv files to merge for token/cost data",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of table",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Summary row only, no full table",
    )
    args = parser.parse_args(argv)

    # Load summaries
    summaries: list[dict] = []
    for pattern in args.files:
        for p in sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]:
            if p.exists():
                try:
                    summaries.append(load_summary(str(p)))
                except Exception as exc:
                    print(f"  WARNING: could not read {p}: {exc}", file=sys.stderr)

    if not summaries:
        print("No summary JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Load optional CSV files
    csv_data: list[dict] = []
    for pattern in (args.csv or []):
        for p in sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]:
            if p.exists():
                csv_data.append(load_metrics_csv(str(p)))

    records = merge_records(summaries, csv_data)

    if args.json:
        print(json.dumps(build_json_output(records), indent=2))
        return

    print_comparison_table(records, verbose=not args.quiet)
    print_quick_summary(records)


if __name__ == "__main__":
    main()
