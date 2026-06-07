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
    # resource_id → {started_t, completed_t, consumed_t, estimated_load_s}
    resource_timings: dict[str, dict] = {}

    run_ids = set()
    prediction_count = 0
    hit_count = 0
    miss_count = 0
    prefetch_started_count = 0
    prefetch_completed_count = 0
    prefetch_cancelled_count = 0
    wasted_count = 0
    divergence_count = 0

    # Pass 1: index prefetch decisions for estimated_load_s per resource_id
    decision_load_s: dict[str, float] = {}
    for ev in events:
        if ev.get("event_type") == "prefetch_decision":
            p = ev.get("payload", {})
            rid = p.get("resource_id", "")
            els = p.get("estimated_load_s")
            if rid and els is not None:
                decision_load_s[rid] = float(els)

    for ev in events:
        et = ev.get("event_type", "")
        p = ev.get("payload", {})
        step = ev.get("step", 0)
        t = ev.get("epoch_time", 0.0)
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
            miss_count += 1          # divergence = prediction was wrong
            steps.setdefault(step, {})["divergence"] = p

        elif et == "prefetch_started":
            prefetch_started_count += 1
            tid = p["task_id"]
            rid = p.get("resource_id", "")
            prefetch_tasks[tid] = {"started": ev}
            rec = resource_timings.setdefault(rid, {})
            rec["started_t"] = t
            rec["task_id"] = tid
            rec["estimated_load_s"] = decision_load_s.get(rid)

        elif et == "prefetch_completed":
            prefetch_completed_count += 1
            tid = p["task_id"]
            if tid in prefetch_tasks:
                prefetch_tasks[tid]["completed"] = ev
            # Find resource_id for this task
            for rid, rec in resource_timings.items():
                if rec.get("task_id") == tid:
                    rec["completed_t"] = t
                    rec["actual_load_s"] = p.get("elapsed_s")
                    break

        elif et == "prefetch_cancelled":
            prefetch_cancelled_count += 1
            if p.get("wasted"):
                wasted_count += 1
            tid = p["task_id"]
            if tid in prefetch_tasks:
                prefetch_tasks[tid]["cancelled"] = ev

        elif et == "resource_consumed":
            rid = p.get("resource_id", "")
            status = p.get("status", "")
            rec = resource_timings.setdefault(rid, {})
            rec["consumed_t"] = t
            rec["consumed_status"] = status

        elif et == "tool_call":
            steps.setdefault(step, {})["actual_tool"] = p.get("tool")

        elif et == "llm_call":
            steps.setdefault(step, {})["model"] = p.get("model")

    validated_total = hit_count + miss_count
    # precision  = hits / validated  (quality of predictions that were testable)
    # coverage   = validated / made   (fraction of predictions that could be tested)
    # honest_acc = hits / all_made    (counts unvalidated as wrong — conservative)
    precision   = hit_count / validated_total if validated_total > 0 else None
    coverage    = validated_total / prediction_count if prediction_count > 0 else None
    honest_acc  = hit_count / prediction_count if prediction_count > 0 else None
    # Legacy field kept for backwards compatibility with summary JSONs
    accuracy    = precision

    # Compute stall and overlap times per resource.
    #
    # For prefetched resources:
    #   overlap_s  = time the prefetch ran before the resource was needed
    #                = min(completed_t, consumed_t) - started_t  (clipped to ≥ 0)
    #   stall_s    = time the consumer waited AFTER arriving until prefetch finished
    #                = max(0, completed_t - consumed_t)
    #
    # For resources consumed without any prefetch (status == "no_prefetch"):
    #   stall_s    = estimated_load_s  (consumer would have waited this long)
    #   overlap_s  = 0
    #
    # benefit_s  = estimated_load_s - stall_s  (time saved vs cold-start)
    total_stall_s = 0.0
    total_overlap_s = 0.0
    total_benefit_s = 0.0
    total_waste_s = 0.0

    for rid, rec in resource_timings.items():
        status = rec.get("consumed_status", "")
        # estimated_load_s is set by the prefetch_started handler; fall back to
        # the prefetch_decision value for resources that were skipped/never started.
        est = (rec.get("estimated_load_s")
               or rec.get("actual_load_s")
               or decision_load_s.get(rid, 0.0))
        started = rec.get("started_t")
        completed = rec.get("completed_t")
        consumed = rec.get("consumed_t")

        if status == "no_prefetch" or started is None:
            # No prefetch was running — consumer stalled for full load time
            total_stall_s += est
            continue

        if consumed is not None and completed is not None:
            overlap = max(0.0, min(completed, consumed) - started)
            stall = max(0.0, completed - consumed)
            benefit = max(0.0, est - stall)
            total_overlap_s += overlap
            total_stall_s += stall
            total_benefit_s += benefit
        elif completed is not None and consumed is None:
            # Prefetch completed but resource was never consumed (wasted)
            actual = rec.get("actual_load_s") or (completed - started)
            total_waste_s += actual

    return {
        "run_ids": sorted(run_ids),
        "total_events": len(events),
        "prediction_count": prediction_count,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "unvalidated_count": prediction_count - validated_total,
        "accuracy": accuracy,           # precision over validated subset (legacy)
        "precision": precision,         # same as accuracy — quality over testable set
        "coverage": coverage,           # fraction of predictions that were validated
        "honest_accuracy": honest_acc,  # conservative: unvalidated = wrong
        "prefetch_started": prefetch_started_count,
        "prefetch_completed": prefetch_completed_count,
        "prefetch_cancelled": prefetch_cancelled_count,
        "wasted_prefetch": wasted_count,
        "divergence_count": divergence_count,
        "total_stall_s": total_stall_s,
        "total_overlap_s": total_overlap_s,
        "total_benefit_s": total_benefit_s,
        "total_waste_s": total_waste_s,
        "steps": steps,
        "prefetch_tasks": prefetch_tasks,
    }


def print_report(summary: dict, verbose: bool = True) -> None:
    run_ids = summary["run_ids"]
    n_made      = summary["prediction_count"]
    n_hit       = summary["hit_count"]
    n_miss      = summary["miss_count"]
    n_unval     = summary.get("unvalidated_count", n_made - n_hit - n_miss)
    precision   = summary.get("precision",   summary.get("accuracy"))
    coverage    = summary.get("coverage")
    honest_acc  = summary.get("honest_accuracy")

    def _pct(v): return f"{v:.1%}" if v is not None else "N/A"
    def _s(v):   return f"{v:.1f}s" if v is not None else "N/A"

    print(f"\n{'='*60}")
    print(f"  Runtime Trace Analysis")
    print(f"  Run ID(s): {', '.join(run_ids) or 'unknown'}")
    print(f"{'='*60}")
    print(f"  Total events        : {summary['total_events']}")
    print(f"  Predictions made    : {n_made}")
    print(f"    Validated         : {n_hit + n_miss}  ({n_hit} hits / {n_miss} misses)")
    print(f"    Unvalidated       : {n_unval}  (expired or no consumer_tool match)")
    print(f"  Precision           : {_pct(precision)}"
          f"  (hits / validated — quality over testable set)")
    print(f"  Coverage            : {_pct(coverage)}"
          f"  (validated / made — how many predictions were testable)")
    print(f"  Honest accuracy     : {_pct(honest_acc)}"
          f"  (hits / made — conservative; treats unvalidated as wrong)")
    print(f"  Divergences         : {summary['divergence_count']}")
    print(f"  Prefetch started    : {summary['prefetch_started']}")
    print(f"  Prefetch completed  : {summary['prefetch_completed']}")
    print(f"  Prefetch cancelled  : {summary['prefetch_cancelled']}")
    print(f"  Wasted prefetches   : {summary['wasted_prefetch']}")
    stall   = summary.get("total_stall_s")
    overlap = summary.get("total_overlap_s")
    benefit = summary.get("total_benefit_s")
    waste   = summary.get("total_waste_s")
    if any(v is not None for v in (stall, overlap, benefit, waste)):
        print(f"  --- Timing (real-mode) ---")
        print(f"  Total stall time    : {_s(stall)}"
              f"  (consumer waited for resource after needing it)")
        print(f"  Total overlap time  : {_s(overlap)}"
              f"  (prefetch ran while compute was in progress)")
        print(f"  Total benefit       : {_s(benefit)}"
              f"  (estimated time saved vs cold-start)")
        print(f"  Total waste         : {_s(waste)}"
              f"  (completed prefetches never consumed)")
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
