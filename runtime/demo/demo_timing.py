"""
demo/demo_timing.py — End-to-end timing demo: real prefetch overlap with FakeModelOrchestrator.

Runs a simulated 3-step AtomAgents workflow with two model loads:
  Step 1 — LLM decides: plan_task
  Step 2 — LLM decides: computation_task_screw_dislocation (needs qwen_32b)
  Step 3 — (done)

The runtime layer predicts at step 1 that qwen_32b will be needed at step 2,
immediately starts loading it (FakeModelOrchestrator sleeps 3s to simulate I/O),
while step 1 "compute" takes 5s. The model finishes loading during the compute
step → overlap_s > 0, benefit_s > 0.

Usage
-----
    python runtime/demo/demo_timing.py
    python runtime/demo/demo_timing.py --compute-time 5 --load-time 3
    python runtime/demo/demo_timing.py --no-prefetch      # baseline, no prefetch
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import (
    ResourceSpec,
    PredictionResult,
)
from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.prefetch.model_prefetch import FakeModelOrchestrator, ModelPrefetchExecutor
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.guard.detector import DivergenceDetector
from runtime.measurement.timings import PrefetchTimingRecord
from runtime.analysis.overlap_report import parse_events, print_overlap_report


# ---------------------------------------------------------------------------
# Demo parameters
# ---------------------------------------------------------------------------

DEFAULT_COMPUTE_TIME = 5.0   # seconds for the LLM compute step
DEFAULT_LOAD_TIME = 3.0      # seconds for model load (FakeModelOrchestrator)


def _make_prediction(load_time: float) -> tuple[PredictionResult, ResourceSpec]:
    """Build a hand-crafted PredictionResult so we control the resource type."""
    import hashlib
    rid = hashlib.md5(b"qwen_32b").hexdigest()[:12]
    resource = ResourceSpec(
        resource_id=rid,
        resource_type="vllm_model",
        name="qwen_32b",
        estimated_load_s=load_time,
        cancellation_safe=False,
        consumer_tool="computation_task_screw_dislocation",
        consumer_step_offset=1,
        confidence=0.90,
    )
    result = PredictionResult(
        step=1,
        resources=[resource],
        confidence=0.90,
        horizon=1,
        predictor_id="mock",
    )
    return result, resource


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_demo(
    compute_time: float,
    load_time: float,
    enable_prefetch: bool,
) -> None:
    run_id = str(uuid.uuid4())[:8]
    import tempfile, os
    log_path = os.path.join(tempfile.gettempdir(), f"runtime_timing_{run_id}.jsonl")

    label = "WITH PREFETCH" if enable_prefetch else "BASELINE (no prefetch)"
    print(f"\n{'='*60}")
    print(f"  Timing Demo — {label}")
    print(f"  Run ID        : {run_id}")
    print(f"  Compute time  : {compute_time}s  (simulated LLM reasoning)")
    print(f"  Model load    : {load_time}s   (FakeModelOrchestrator)")
    print(f"  Trace         : {log_path}")
    print(f"{'='*60}\n")

    cfg = RuntimeConfig(
        mode=RuntimeMode.REAL if enable_prefetch else RuntimeMode.BASELINE,
        run_id=run_id,
        confidence_threshold=0.85,
        max_horizon=2,
    )
    bus = EventBus(run_id=run_id, log_path=log_path)
    orch = FakeModelOrchestrator(load_times={"qwen_32b": load_time})
    executor = ModelPrefetchExecutor(orch)
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)

    result, resource = _make_prediction(load_time)

    t_wall_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Step 1: LLM decides to call plan_task
    # -----------------------------------------------------------------
    step = 1
    print(f"[Step {step}] LLM reasoning…  (plan_task, {compute_time}s)")
    bus.emit("tool_call", {"tool": "plan_task", "step": step}, step=step)

    # Predict at step 1: qwen_32b needed at step 2
    bus.emit("prediction_result", result.to_dict(), step=step)
    print(f"         Predicted: {[r.name for r in result.resources]} (conf={result.confidence})")

    if enable_prefetch and result.resources:
        ckpt = detector.on_prediction(result, step=step)
        scheduler.schedule(
            resource=resource,
            current_step=step,
            checkpoint_id=ckpt.checkpoint_id if ckpt else "ckpt-1",
            estimated_remaining_compute_s=compute_time,
        )
        print(f"         Prefetch : STARTED (qwen_32b loading in background)")

    # Simulate compute
    t_compute_start = time.perf_counter()
    time.sleep(compute_time)
    t_compute_end = time.perf_counter()
    print(f"         Compute  : {t_compute_end - t_compute_start:.2f}s elapsed")

    # -----------------------------------------------------------------
    # Step 2: Tool starts (computation_task_screw_dislocation)
    # -----------------------------------------------------------------
    step = 2
    t_tool_start = time.perf_counter()
    print(f"\n[Step {step}] Tool: computation_task_screw_dislocation")

    # Check divergence
    hit, action, ckpt_out = detector.on_tool_about_to_execute(
        "computation_task_screw_dislocation", step=step
    )
    if hit:
        print(f"         Prediction: HIT ✓")
    else:
        print(f"         Prediction: MISS (action={action})")

    if enable_prefetch:
        # Wait for prefetch to complete (it may already be done)
        task = scheduler.get_task_for_resource(resource.resource_id)
        if task is not None:
            wait_start = time.perf_counter()
            deadline = wait_start + load_time + 5.0
            while time.perf_counter() < deadline:
                if executor.is_complete(task):
                    break
                time.sleep(0.1)
            t_consumed = time.perf_counter()

            waited_extra = t_consumed - t_tool_start
            if task.status == PrefetchStatus.COMPLETED:
                print(f"         Model     : ready (waited {waited_extra:.2f}s extra after tool start)")
                scheduler.on_resource_consumed(
                    resource.resource_id,
                    consumed_at=t_consumed,
                    current_step=step,
                )
            else:
                print(f"         Model     : status={task.status.value}")
        else:
            print(f"         Model     : no prefetch task found")
    else:
        # Baseline: load the model now (blocking)
        print(f"         Model     : loading now (no prefetch, blocking for {load_time}s)")
        t_load_start = time.perf_counter()
        orch.start_model_measured("qwen_32b")
        t_load_end = time.perf_counter()
        print(f"         Model     : loaded in {t_load_end - t_load_start:.2f}s")

    t_wall_end = time.perf_counter()

    # Grab task timing before closing (task object has real perf_counter timestamps)
    task_obj = scheduler.get_task_for_resource(resource.resource_id) if enable_prefetch else None

    # Wait briefly so the scheduler's background poll thread emits prefetch_completed
    time.sleep(2.5)
    bus.close()
    executor.shutdown(wait=False)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    total_wall = t_wall_end - t_wall_start
    print(f"\n{'—'*40}")
    print(f"  Total wall time : {total_wall:.2f}s")
    if enable_prefetch:
        expected_baseline = compute_time + load_time
        expected_prefetch = compute_time + max(0, load_time - compute_time)
        print(f"  Expected baseline  : ~{expected_baseline:.1f}s  (no overlap)")
        print(f"  Expected w/prefetch: ~{expected_prefetch:.1f}s  (full overlap if load < compute)")

    # Show measured timing directly from the task object (perf_counter precision)
    if task_obj and task_obj.started_at and task_obj.completed_at and task_obj.consumed_at:
        tr_direct = PrefetchTimingRecord(
            prefetch_start_t=task_obj.started_at,
            prefetch_end_t=task_obj.completed_at,
            resource_needed_t=task_obj.consumed_at,
            cancelled=task_obj.status.value == "cancelled",
            wasted=task_obj.status.value == "wasted",
        )
        print(f"\n  Measured timing (from task object):")
        print(f"    Load duration : {task_obj.completed_at - task_obj.started_at:.3f}s")
        print(f"    overlap_s     : {tr_direct.overlap_s:.3f}s  (load time hidden behind compute)")
        print(f"    benefit_s     : {tr_direct.benefit_s:.3f}s  (model was ready before it was needed)")
        if tr_direct.waste_s > 0:
            print(f"    waste_s       : {tr_direct.waste_s:.3f}s  (had to wait after tool started)")

    # Parse JSONL and print overlap report
    import json
    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    task_records = parse_events(events)
    if task_records:
        print_overlap_report(task_records, run_id=run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end timing demo with FakeModelOrchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--compute-time",
        type=float,
        default=DEFAULT_COMPUTE_TIME,
        help="Duration of simulated LLM compute step (seconds)",
    )
    parser.add_argument(
        "--load-time",
        type=float,
        default=DEFAULT_LOAD_TIME,
        help="Duration of simulated model load (seconds)",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Baseline run: no prefetch, sequential load after tool starts",
    )
    args = parser.parse_args()

    run_demo(
        compute_time=args.compute_time,
        load_time=args.load_time,
        enable_prefetch=not args.no_prefetch,
    )


if __name__ == "__main__":
    main()
