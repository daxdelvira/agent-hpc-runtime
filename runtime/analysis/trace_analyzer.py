"""
analysis/trace_analyzer.py — Parse a runtime JSONL trace and print a summary.

Usage
-----
    python runtime/analysis/trace_analyzer.py logs/workflow_traces/*.jsonl
    python runtime/analysis/trace_analyzer.py logs/workflow_traces/runtime_trace_*.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_events(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def analyze(events: list[dict]) -> dict:
    """Return a structured summary of prediction and prefetch outcomes."""
    steps: dict[int, dict] = {}
    predictions: dict[str, dict] = {}   # checkpoint_id → prediction record
    prefetch_tasks: dict[str, dict] = {}  # task_id → prefetch record

    run_ids = set()
    prediction_count = 0
    hit_count = 0
    miss_count = 0
    prefetch_started_count = 0
    prefetch_completed_count = 0
    prefetch_cancelled_count = 0
    wasted_count = 0
    divergence_count = 0

    for ev in events:
        et = ev.get("event_type", "")
        p = ev.get("payload", {})
        step = ev.get("step", 0)
        run_id = ev.get("run_id", "")
        run_ids.add(run_id)

        if et == "prediction_result":
            prediction_count += 1
            predictions[p.get("checkpoint_id", "")] = ev
            steps.setdefault(step, {})["prediction"] = p

        elif et == "prediction_validated":
            if p.get("hit"):
                hit_count += 1
            else:
                miss_count += 1
            steps.setdefault(step, {})["validated"] = p

        elif et == "divergence_detected":
            divergence_count += 1
            steps.setdefault(step, {})["divergence"] = p

        elif et == "prefetch_started":
            prefetch_started_count += 1
            prefetch_tasks[p["task_id"]] = {"started": ev}

        elif et == "prefetch_completed":
            prefetch_completed_count += 1
            tid = p["task_id"]
            if tid in prefetch_tasks:
                prefetch_tasks[tid]["completed"] = ev

        elif et == "prefetch_cancelled":
            prefetch_cancelled_count += 1
            if p.get("wasted"):
                wasted_count += 1
            tid = p["task_id"]
            if tid in prefetch_tasks:
                prefetch_tasks[tid]["cancelled"] = ev

        elif et == "tool_call":
            steps.setdefault(step, {})["actual_tool"] = p.get("tool")

        elif et == "llm_call":
            steps.setdefault(step, {})["model"] = p.get("model")

    total_predictions = hit_count + miss_count
    accuracy = hit_count / total_predictions if total_predictions > 0 else None

    return {
        "run_ids": sorted(run_ids),
        "total_events": len(events),
        "prediction_count": prediction_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "accuracy": accuracy,
        "prefetch_started": prefetch_started_count,
        "prefetch_completed": prefetch_completed_count,
        "prefetch_cancelled": prefetch_cancelled_count,
        "wasted_prefetch": wasted_count,
        "divergence_count": divergence_count,
        "steps": steps,
        "prefetch_tasks": prefetch_tasks,
    }


def print_report(summary: dict, verbose: bool = True) -> None:
    run_ids = summary["run_ids"]
    acc = summary["accuracy"]
    acc_str = f"{acc:.1%}" if acc is not None else "N/A (no validated predictions yet)"

    print(f"\n{'='*60}")
    print(f"  Runtime Trace Analysis")
    print(f"  Run ID(s): {', '.join(run_ids) or 'unknown'}")
    print(f"{'='*60}")
    print(f"  Total events        : {summary['total_events']}")
    print(f"  Predictions made    : {summary['prediction_count']}")
    print(f"  Prediction accuracy : {acc_str} ({summary['hit_count']} hits / {summary['miss_count']} misses)")
    print(f"  Divergences         : {summary['divergence_count']}")
    print(f"  Prefetch started    : {summary['prefetch_started']}")
    print(f"  Prefetch completed  : {summary['prefetch_completed']}")
    print(f"  Prefetch cancelled  : {summary['prefetch_cancelled']}")
    print(f"  Wasted prefetches   : {summary['wasted_prefetch']}")
    print()

    if verbose and summary["steps"]:
        print("  Per-step breakdown:")
        print(f"  {'Step':>5}  {'Actual Tool':<35}  {'Prediction Hit':<15}  {'Prefetch'}")
        print(f"  {'-'*5}  {'-'*35}  {'-'*15}  {'-'*20}")
        for step in sorted(summary["steps"].keys()):
            s = summary["steps"][step]
            actual = s.get("actual_tool", "—")
            if "validated" in s:
                hit_str = "HIT" if s["validated"].get("hit") else "MISS"
            elif "divergence" in s:
                hit_str = "DIVERGE"
            else:
                hit_str = "—"
            pred = s.get("prediction", {})
            resources = pred.get("resources", [])
            pred_name = resources[0]["name"] if resources else "—"
            print(f"  {step:>5}  {actual:<35}  {hit_str:<15}  → {pred_name}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a runtime JSONL trace")
    parser.add_argument("files", nargs="+", help="JSONL trace file(s)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--quiet", action="store_true", help="Summary only, no per-step table")
    args = parser.parse_args(argv)

    all_events = []
    for path in args.files:
        for p in sorted(Path(".").glob(path)) if "*" in path else [Path(path)]:
            if p.exists():
                all_events.extend(load_events(str(p)))

    if not all_events:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    summary = analyze(all_events)

    if args.json:
        # Omit steps dict for cleaner JSON output
        out = {k: v for k, v in summary.items() if k not in ("steps", "prefetch_tasks")}
        print(json.dumps(out, indent=2))
    else:
        print_report(summary, verbose=not args.quiet)


if __name__ == "__main__":
    main()
