"""
analysis/overlap_report.py — Per-prefetch timing breakdown from a runtime JSONL trace.

Joins prefetch_started / prefetch_completed / resource_consumed events by
task_id to compute overlap_s, benefit_s, and waste_s for each prefetch task.

For simulated runs (elapsed_s ≈ 0), estimated metrics are shown using the
estimated_load_s from prediction_result events. The "(est)" flag in the output
marks values derived from estimates rather than measured timing.

Usage
-----
    python runtime/analysis/overlap_report.py logs/workflow_traces/*.jsonl
    python runtime/analysis/overlap_report.py <file> --csv results/overlap.csv
    python runtime/analysis/overlap_report.py <file> --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.measurement.timings import PrefetchTimingRecord, write_timing_csv

# Threshold below which we treat elapsed_s as simulated (no real I/O happened).
_SIMULATED_ELAPSED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Intermediate join record
# ---------------------------------------------------------------------------

@dataclass
class _TaskRecord:
    task_id: str
    resource_id: str = ""
    resource_name: str = ""
    resource_type: str = ""
    executor: str = ""
    predictor_id: str = ""
    checkpoint_id: str = ""
    predicted_at_step: int = 0
    consumed_at_step: int = 0

    start_epoch: float | None = None          # epoch_time of prefetch_started
    end_epoch: float | None = None            # epoch_time of prefetch_completed
    consumed_epoch: float | None = None       # epoch_time of resource_consumed
    elapsed_s: float = 0.0                    # reported elapsed (may be simulated ≈ 0)
    estimated_load_s: float | None = None     # from prediction_result / prefetch_decision

    cancelled: bool = False
    wasted: bool = False
    hit: bool = False
    status: str = "unknown"
    failed: bool = False          # prefetch_completed carried status=failed

    def is_simulated(self) -> bool:
        return self.elapsed_s < _SIMULATED_ELAPSED_THRESHOLD


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_events(events: list[dict]) -> list[_TaskRecord]:
    """
    Walk the event list and build one _TaskRecord per prefetch task_id.
    Also collects resource metadata from prediction_result events.
    """
    tasks: dict[str, _TaskRecord] = {}

    # resource_id → metadata from prediction_result
    resource_meta: dict[str, dict[str, Any]] = {}
    # resource_id → estimated_load_s from prefetch_decision
    decision_meta: dict[str, float] = {}
    # checkpoint_id per resource_id from prediction_result
    resource_checkpoint: dict[str, str] = {}
    # predictor_id per step
    step_predictor: dict[int, str] = {}

    run_id: str = ""

    for ev in events:
        et = ev.get("event_type", "")
        p = ev.get("payload", {})
        epoch = ev.get("epoch_time", 0.0)
        step = ev.get("step", 0)
        run_id = run_id or ev.get("run_id", "")

        if et == "prediction_result":
            pid = p.get("predictor_id", "mock")
            step_predictor[step] = pid
            ckpt_id = p.get("checkpoint_id", "")
            for r in p.get("resources", []):
                rid = r.get("resource_id", "")
                resource_meta[rid] = r
                if ckpt_id:
                    resource_checkpoint[rid] = ckpt_id

        elif et == "prefetch_decision":
            rid = p.get("resource_id", "")
            est = p.get("estimated_load_s")
            if est is not None:
                decision_meta[rid] = float(est)

        elif et == "prefetch_started":
            tid = p.get("task_id", "")
            rid = p.get("resource_id", "")
            if not tid:
                continue
            rec = _TaskRecord(task_id=tid, resource_id=rid)
            rec.start_epoch = epoch
            rec.predicted_at_step = step
            rec.executor = p.get("executor", "")
            tasks[tid] = rec

        elif et == "prefetch_completed":
            tid = p.get("task_id", "")
            if tid in tasks:
                elapsed = float(p.get("elapsed_s", 0.0))
                tasks[tid].elapsed_s = elapsed
                # payload.status is the executor's verdict; the event is
                # emitted for failures too (scheduler.py:135-151).  Marking a
                # failed staging "completed" made it look like a prefetch that
                # had really landed and simply gone unused.
                if str(p.get("status") or "").lower() == "failed":
                    tasks[tid].failed = True
                    tasks[tid].status = "failed"
                elif tasks[tid].status not in ("used", "wasted", "cancelled"):
                    tasks[tid].status = "completed"
                # Use start_epoch + elapsed_s so end_epoch reflects when the
                # load actually finished, not when the poll thread logged it.
                if tasks[tid].start_epoch is not None and elapsed > 0:
                    tasks[tid].end_epoch = tasks[tid].start_epoch + elapsed
                else:
                    tasks[tid].end_epoch = epoch

        elif et == "prefetch_cancelled":
            tid = p.get("task_id", "")
            if tid in tasks:
                tasks[tid].cancelled = True
                tasks[tid].wasted = bool(p.get("wasted", False))
                tasks[tid].status = "wasted" if tasks[tid].wasted else "cancelled"

        elif et == "resource_consumed":
            tid = p.get("task_id")
            rid = p.get("resource_id", "")
            if tid and tid in tasks:
                tasks[tid].consumed_epoch = epoch
                tasks[tid].consumed_at_step = step
                if p.get("status") == "used":
                    tasks[tid].hit = True
                    tasks[tid].status = "used"

        elif et == "prediction_validated":
            # Correlate hit flag back to in-flight tasks by checkpoint_id
            ckpt_id = p.get("checkpoint_id", "")
            hit = bool(p.get("hit", False))
            for rec in tasks.values():
                if rec.checkpoint_id == ckpt_id:
                    rec.hit = hit

    # Enrich records with resource metadata
    for rec in tasks.values():
        meta = resource_meta.get(rec.resource_id, {})
        rec.resource_name = meta.get("name", rec.resource_id[:12])
        rec.resource_type = meta.get("resource_type", "")
        rec.estimated_load_s = (
            meta.get("estimated_load_s")
            or decision_meta.get(rec.resource_id)
        )
        ckpt = resource_checkpoint.get(rec.resource_id, "")
        if ckpt:
            rec.checkpoint_id = ckpt
        rec.predictor_id = step_predictor.get(rec.predicted_at_step, "mock")

    return list(tasks.values())


# ---------------------------------------------------------------------------
# Build PrefetchTimingRecords
# ---------------------------------------------------------------------------

def build_timing_records(
    task_records: list[_TaskRecord],
    run_id: str = "",
) -> list[PrefetchTimingRecord]:
    """Convert joined task records into PrefetchTimingRecord instances."""
    results = []
    for r in task_records:
        tr = PrefetchTimingRecord(
            run_id=run_id,
            resource_id=r.resource_id,
            resource_name=r.resource_name,
            resource_type=r.resource_type,
            predictor_id=r.predictor_id,
            checkpoint_id=r.checkpoint_id,
            predicted_at_step=r.predicted_at_step,
            consumed_at_step=r.consumed_at_step,
            prefetch_start_t=r.start_epoch,
            prefetch_end_t=r.end_epoch,
            resource_needed_t=r.consumed_epoch,
            cancelled=r.cancelled,
            wasted=r.wasted,
            hit=r.hit,
        )
        results.append(tr)
    return results


# ---------------------------------------------------------------------------
# Estimated metrics for simulated runs
# ---------------------------------------------------------------------------

def _estimated_overlap(rec: _TaskRecord) -> float | None:
    """
    For simulated runs, estimate overlap_s as estimated_load_s (the entire
    load would be hidden if the prefetch started early enough).
    Returns None if we can't estimate.
    """
    if not rec.is_simulated():
        return None
    if rec.cancelled or rec.wasted or rec.failed:
        return None
    return rec.estimated_load_s


def _estimated_benefit(rec: _TaskRecord) -> float | None:
    """
    For simulated runs, benefit_s ≈ estimated_load_s if the prefetch would
    complete before the resource is needed (optimistic upper bound).
    Returns None if we can't estimate.
    """
    if not rec.is_simulated():
        return None
    if rec.cancelled or rec.wasted or rec.failed:
        return None
    return rec.estimated_load_s


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_overlap_report(
    task_records: list[_TaskRecord],
    run_id: str = "",
    verbose: bool = True,
) -> None:
    if not task_records:
        print("\n  No prefetch tasks found in trace.\n")
        return

    print(f"\n{'='*72}")
    print(f"  Prefetch Overlap Report")
    if run_id:
        print(f"  Run ID: {run_id}")
    print(f"{'='*72}")

    # Determine if this is a simulated run
    simulated_count = sum(1 for r in task_records if r.is_simulated())
    is_simulated = simulated_count > len(task_records) / 2
    if is_simulated:
        print("  NOTE: Most tasks have near-zero elapsed_s → simulated run.")
        print("        Overlap/benefit values marked (est) use estimated_load_s.\n")

    col_w = [8, 22, 12, 6, 10, 10, 10, 8, 6]
    hdr = (
        f"  {'TaskID':<{col_w[0]}}  "
        f"{'Resource':<{col_w[1]}}  "
        f"{'Type':<{col_w[2]}}  "
        f"{'Step':>{col_w[3]}}  "
        f"{'overlap_s':>{col_w[4]}}  "
        f"{'benefit_s':>{col_w[5]}}  "
        f"{'waste_s':>{col_w[6]}}  "
        f"{'Status':<{col_w[7]}}  "
        f"{'Est?':<{col_w[8]}}"
    )
    print(hdr)
    print("  " + "-" * (sum(col_w) + 2 * len(col_w)))

    total_overlap = 0.0
    total_benefit = 0.0
    total_waste = 0.0
    total_overlap_est = 0.0
    total_benefit_est = 0.0
    used_count = 0
    wasted_count = 0
    cancelled_count = 0

    for rec in sorted(task_records, key=lambda r: r.predicted_at_step):
        timing = PrefetchTimingRecord(
            prefetch_start_t=rec.start_epoch,
            prefetch_end_t=rec.end_epoch,
            resource_needed_t=rec.consumed_epoch,
            cancelled=rec.cancelled,
            wasted=rec.wasted,
        )

        real_overlap = timing.overlap_s
        real_benefit = timing.benefit_s
        real_waste = timing.waste_s

        est_overlap = _estimated_overlap(rec)
        est_benefit = _estimated_benefit(rec)
        is_est = rec.is_simulated() and est_overlap is not None

        display_overlap = est_overlap if is_est else real_overlap
        display_benefit = est_benefit if is_est else real_benefit
        display_waste = 0.0 if is_est else real_waste

        total_overlap += real_overlap
        total_benefit += real_benefit
        total_waste += real_waste
        if is_est:
            total_overlap_est += est_overlap or 0.0
            total_benefit_est += est_benefit or 0.0

        if rec.status == "used":
            used_count += 1
        elif rec.wasted:
            wasted_count += 1
        elif rec.cancelled:
            cancelled_count += 1

        tid_short = rec.task_id[:col_w[0]]
        name_short = (rec.resource_name or rec.resource_id)[:col_w[1]]
        rtype_short = rec.resource_type[:col_w[2]]
        status_short = rec.status[:col_w[7]]
        est_flag = "(est)" if is_est else ""

        row = (
            f"  {tid_short:<{col_w[0]}}  "
            f"{name_short:<{col_w[1]}}  "
            f"{rtype_short:<{col_w[2]}}  "
            f"{rec.predicted_at_step:>{col_w[3]}}  "
            f"{display_overlap or 0.0:>{col_w[4]}.1f}  "
            f"{display_benefit or 0.0:>{col_w[5]}.1f}  "
            f"{display_waste or 0.0:>{col_w[6]}.1f}  "
            f"{status_short:<{col_w[7]}}  "
            f"{est_flag:<{col_w[8]}}"
        )
        print(row)

    print()
    print(f"  {'—'*40}")
    print(f"  Tasks total   : {len(task_records)}")
    print(f"    Used        : {used_count}")
    print(f"    Wasted      : {wasted_count}  (completed but unused due to divergence)")
    print(f"    Cancelled   : {cancelled_count}")

    if is_simulated and total_benefit_est > 0:
        print(f"\n  Estimated benefit (simulated):")
        print(f"    Total overlap_s   : {total_overlap_est:.1f}s")
        print(f"    Total benefit_s   : {total_benefit_est:.1f}s  (time saved if prefetch ready in time)")
        print(f"    Total waste_s     : 0.0s   (unknown without real timing)")
        print(f"    NOTE: Actual cluster run needed for measured overlap.")
    else:
        print(f"\n  Measured timing:")
        print(f"    Total overlap_s   : {total_overlap:.1f}s")
        print(f"    Total benefit_s   : {total_benefit:.1f}s  (time saved)")
        print(f"    Total waste_s     : {total_waste:.1f}s  (prefetch arrived too late)")

    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_summary(
    task_records: list[_TaskRecord],
    run_id: str = "",
) -> dict:
    tasks_out = []
    for rec in task_records:
        timing = PrefetchTimingRecord(
            prefetch_start_t=rec.start_epoch,
            prefetch_end_t=rec.end_epoch,
            resource_needed_t=rec.consumed_epoch,
            cancelled=rec.cancelled,
            wasted=rec.wasted,
        )
        is_est = rec.is_simulated()
        tasks_out.append({
            "task_id": rec.task_id,
            "resource_id": rec.resource_id,
            "resource_name": rec.resource_name,
            "resource_type": rec.resource_type,
            "predicted_at_step": rec.predicted_at_step,
            "consumed_at_step": rec.consumed_at_step,
            "elapsed_s": rec.elapsed_s,
            "estimated_load_s": rec.estimated_load_s,
            "overlap_s": timing.overlap_s,
            "benefit_s": timing.benefit_s,
            "waste_s": timing.waste_s,
            "estimated_overlap_s": _estimated_overlap(rec) if is_est else None,
            "estimated_benefit_s": _estimated_benefit(rec) if is_est else None,
            "cancelled": rec.cancelled,
            "wasted": rec.wasted,
            "failed": rec.failed,
            "hit": rec.hit,
            "status": rec.status,
            "simulated": is_est,
        })
    return {
        "run_id": run_id,
        "task_count": len(task_records),
        "tasks": tasks_out,
        "total_benefit_s": sum(t["benefit_s"] for t in tasks_out),
        "total_waste_s": sum(t["waste_s"] for t in tasks_out),
        "estimated_total_benefit_s": sum(
            t["estimated_benefit_s"] or 0 for t in tasks_out
        ),
    }


# ---------------------------------------------------------------------------
# Top-level API (importable by demo scripts)
# ---------------------------------------------------------------------------

def report_from_events(
    events: list[dict],
    run_id: str = "",
    verbose: bool = True,
    write_csv: str | None = None,
) -> dict:
    """
    Parse events and print the overlap report. Returns the JSON summary dict.
    Optionally write timing records to a CSV file.
    """
    task_records = parse_events(events)
    if not run_id and events:
        run_id = events[0].get("run_id", "")
    print_overlap_report(task_records, run_id=run_id, verbose=verbose)
    if write_csv:
        timing_records = build_timing_records(task_records, run_id=run_id)
        write_timing_csv(timing_records, write_csv)
    return build_json_summary(task_records, run_id=run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Per-prefetch overlap/benefit/waste breakdown from a runtime JSONL trace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="JSONL trace file(s)")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    parser.add_argument("--quiet", action="store_true", help="Skip per-task table")
    parser.add_argument("--csv", default=None, metavar="PATH", help="Write timing CSV")
    args = parser.parse_args(argv)

    all_events: list[dict] = []
    for pattern in args.files:
        for p in sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]:
            if p.exists():
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass

    if not all_events:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    summary = report_from_events(
        all_events,
        verbose=not args.quiet,
        write_csv=args.csv,
    )

    if args.json:
        print(json.dumps(summary, indent=2))

    if args.csv:
        print(f"Timing CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
