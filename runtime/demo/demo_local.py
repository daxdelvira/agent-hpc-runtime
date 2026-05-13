"""
demo_local.py — End-to-end local demo with no real LLM, no GPU, no HPC tools.

Simulates a ChemGraph-style workflow by replaying synthetic events through
the runtime pipeline:
  LLM response (tool_calls) → predictor → scheduler → divergence guard → report

Run
---
    python runtime/demo/demo_local.py
    python runtime/demo/demo_local.py --diverge      # inject a divergence
    python runtime/demo/demo_local.py --mode observe_only
    python runtime/demo/demo_local.py --workflow atomagents
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import (
    make_checkpoint_created_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prediction_validated_event,
    make_conservative_mode_event,
)
from runtime.guard.checkpoint import CheckpointRecord
from runtime.guard.detector import DivergenceDetector, DivergenceAction
from runtime.predictor.mock_predictor import MockPredictor
from runtime.prefetch.base import PrefetchStatus
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.analysis.trace_analyzer import analyze, print_report


# ---------------------------------------------------------------------------
# Synthetic workflow steps
# ---------------------------------------------------------------------------

CHEMGRAPH_STEPS = [
    # (tool_calls_from_llm, actual_tool_executed)
    ([{"name": "molecule_name_to_smiles"}],   "molecule_name_to_smiles"),
    ([{"name": "smiles_to_coordinate_file"}], "smiles_to_coordinate_file"),
    ([{"name": "run_ase"}],                   "run_ase"),
    ([{"name": "extract_output_json"}],        "extract_output_json"),
]

ATOMAGENTS_STEPS = [
    ([{"name": "plan_task"}],                                      "plan_task"),
    ([{"name": "computation_task_screw_dislocation"}],             "computation_task_screw_dislocation"),
    ([{"name": "plan_task"}],                                      "plan_task"),
    ([{"name": "computation_task_screw_dislocation"}],             "computation_task_screw_dislocation"),
]


def run_demo(
    mode: RuntimeMode,
    workflow: str,
    inject_divergence: bool,
    run_id: str,
) -> str:
    """Run the simulated workflow and return the path to the JSONL trace."""

    cfg = RuntimeConfig(
        mode=mode,
        run_id=run_id,
        confidence_threshold=0.85,
        max_horizon=2,
        conservative_mode_steps=2,
    )

    # Use a temp dir so we don't pollute logs/ during demo
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, f"demo_{run_id}.jsonl")

        bus = EventBus(run_id=run_id, log_path=trace_path)
        executor = SimulatedPrefetchExecutor()
        scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
        predictor = MockPredictor(workflow=workflow)
        detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)

        steps = ATOMAGENTS_STEPS if workflow == "atomagents" else CHEMGRAPH_STEPS

        print(f"\n{'='*60}")
        print(f"  Runtime Demo")
        print(f"  Mode     : {mode.value}")
        print(f"  Workflow : {workflow}")
        print(f"  Run ID   : {run_id}")
        print(f"  Diverge  : {inject_divergence}")
        print(f"{'='*60}\n")

        for step_idx, (tool_calls, actual_tool) in enumerate(steps):
            step = step_idx + 1
            bus.set_step(step)

            # Synthesize a "recent_events" context from already-written events
            recent = _read_recent(trace_path, n=5)
            in_conservative = detector.is_conservative(step)

            # --- Tool execution phase (checks PREVIOUS step's prediction) ---
            # Inject divergence at step 3: run_ase should execute, but wrong tool runs.
            # This is the consumer step for the mace-mp-0 prefetch started at step 2.
            if inject_divergence and step == 3:
                execute_tool = "wrong_tool_injected"
            else:
                execute_tool = actual_tool

            # Log the "tool_call" event (simulates what WorkflowTracker writes)
            bus.emit("tool_call", {"tool": execute_tool, "step": step}, step=step)

            # Divergence check: look for a pending prediction that targeted this step
            hit, action, ckpt_out = detector.on_tool_about_to_execute(execute_tool, step=step)

            if ckpt_out is not None:
                predicted = ckpt_out.prediction.resources[0].consumer_tool if ckpt_out.prediction and ckpt_out.prediction.resources else "?"
                predicted_name = ckpt_out.prediction.resources[0].name if ckpt_out.prediction and ckpt_out.prediction.resources else "?"
                if hit:
                    bus.emit_event(make_prediction_validated_event(
                        run_id, step, True, ckpt_out.checkpoint_id, execute_tool,
                    ))
                    print(f"  Step {step:2d}  tool={execute_tool:<40}  → HIT  (predicted: {predicted_name})")
                else:
                    bus.emit_event(make_divergence_detected_event(
                        run_id, step, predicted, execute_tool,
                        ckpt_out.checkpoint_id, action.value,
                    ))
                    bus.emit_event(make_conservative_mode_event(
                        run_id, step, "divergence", cfg.conservative_mode_steps,
                    ))
                    print(f"  Step {step:2d}  tool={execute_tool:<40}  → DIVERGE  (expected: {predicted})  action={action.value}")
            else:
                mode_note = "conservative" if in_conservative else "no_prediction"
                print(f"  Step {step:2d}  tool={execute_tool:<40}  → [{mode_note}]")

            # --- LLM response phase (prediction for FUTURE steps) ---
            result = None

            if not in_conservative and mode not in (RuntimeMode.BASELINE,):
                result = predictor.predict(
                    step=step,
                    recent_events=recent,
                    current_tool_calls=tool_calls,
                )

                if result.resources:
                    bus.emit_event(make_prediction_result_event(run_id, step, result))

                    # WAL checkpoint
                    log_pos = bus.current_log_position()
                    ckpt = detector.on_prediction(result, step=step, log_position=log_pos)
                    bus.emit_event(make_checkpoint_created_event(run_id, step, ckpt.checkpoint_id, log_pos))

                    # Schedule prefetches (SIMULATED: no real I/O)
                    if mode not in (RuntimeMode.OBSERVE_ONLY,):
                        for resource in result.resources:
                            scheduler.schedule(
                                resource=resource,
                                current_step=step,
                                checkpoint_id=ckpt.checkpoint_id,
                            )

        # Small sleep to ensure background completion thread fires
        time.sleep(0.1)
        bus.close()

        # --- Analysis ---
        print()
        events = _read_all(trace_path)
        summary = analyze(events)
        print_report(summary, verbose=True)

        # Print prefetch task summary
        tasks = scheduler.all_tasks()
        if tasks:
            print(f"  Prefetch tasks ({len(tasks)} total):")
            for t in tasks:
                proj = t.resource.estimated_load_s or 0
                print(f"    {t.task_id[:8]}  {t.resource.name:<25}  status={t.status.value:<12}  "
                      f"projected_benefit={proj:.0f}s")
        print()

        # Copy trace to a persistent location for follow-up analysis
        persistent_dir = "logs/demo_traces"
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_path = os.path.join(persistent_dir, f"demo_{run_id}.jsonl")
        import shutil
        shutil.copy2(trace_path, persistent_path)
        print(f"  Trace saved to: {persistent_path}")
        print(f"  Analyze with:  python runtime/analysis/trace_analyzer.py {persistent_path}\n")
        return persistent_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_recent(path: str, n: int) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    events = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def _read_all(path: str) -> list[dict]:
    return _read_recent(path, n=100_000)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local runtime demo (no LLM, no GPU)")
    parser.add_argument("--mode", choices=[m.value for m in RuntimeMode], default="simulated")
    parser.add_argument("--workflow", choices=["chemgraph", "atomagents"], default="chemgraph")
    parser.add_argument("--diverge", action="store_true", help="Inject a divergence at step 2")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or str(uuid.uuid4())[:8]
    mode = RuntimeMode(args.mode)
    run_demo(mode=mode, workflow=args.workflow, inject_divergence=args.diverge, run_id=run_id)


if __name__ == "__main__":
    main()
