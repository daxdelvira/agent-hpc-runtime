"""
atomagents_exp2.py — AtomAgents Exp2 runner with runtime layer.

Runs the screw-dislocation experiment with the runtime layer active so that
prediction, prefetch, and divergence events appear in the JSONL trace alongside
the normal AtomAgents agent/tool/LLM events.

Designed to be run on a GPU cluster node with AtomAgents installed.
All AtomAgents / autogen / CUDA imports are guarded so the file is safe
to inspect locally.

Usage
-----
    # Observe predictions, no prefetch I/O (safe for first cluster run)
    python experiments/atomagents_exp2.py --runtime-mode observe_only

    # Simulated prefetch (logs decisions, no actual I/O)
    python experiments/atomagents_exp2.py --runtime-mode simulated

    # Real model prefetch (starts vLLM background load)
    python experiments/atomagents_exp2.py --runtime-mode real --swap-models

    # Baseline (zero runtime overhead, normal AtomAgents run)
    python experiments/atomagents_exp2.py --runtime-mode baseline

Flags
-----
    --runtime-mode    observe_only | simulated | real | baseline (default: observe_only)
    --predictor       mock | oracle (default: mock)
    --run-id          custom run ID (default: auto-generated)
    --hw-profile      l40s | rtx6000 (default: l40s)
    --swap-models     enable model swapping (requires --hw-profile rtx6000 or explicit gpu config)
    --task-prompt     override the default Exp2 task prompt
    --results-dir     directory for output CSV / JSONL (default: results/)
    --log-dir         directory for JSONL traces (default: logs/workflow_traces/)
    --confidence      predictor confidence threshold (default: 0.85)
    --horizon         max lookahead in steps (default: 2)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure the project root (repo root) and workloads/AtomAgents are importable
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "workloads" / "AtomAgents"))

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.guard.detector import DivergenceDetector
from runtime.predictor.mock_predictor import MockPredictor
from runtime.analysis.trace_analyzer import analyze, print_report
from runtime.adapters.atomagents import (
    AtomAgentsRuntimeAdapter,
    make_atomagents_adapter,
)


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

DEFAULT_TASK_PROMPT = """\
Compare the structure of 1/2<111> screw dislocation in W using \
"W_Zhou04.eam.alloy" and "w_eam4.fs" EAM potentials.
The dislocation line is aligned along [-1,1,1] direction, and dislocation \
glide and glide plane normal directions are [1,-1,2] and [1,1,0], respectively.
"""


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_exp2(args: argparse.Namespace) -> None:
    run_id = args.run_id or str(uuid.uuid4())[:12]
    mode = RuntimeMode(args.runtime_mode)

    # Resolve output paths to absolute now, before we chdir below.
    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir)
    if not results_dir.is_absolute():
        results_dir = (_PROJECT_ROOT / results_dir).resolve()
    if not log_dir.is_absolute():
        log_dir = (_PROJECT_ROOT / log_dir).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = str(log_dir / f"runtime_trace_{ts}_{run_id}.jsonl")

    cfg = RuntimeConfig(
        mode=mode,
        run_id=run_id,
        confidence_threshold=args.confidence,
        max_horizon=args.horizon,
        conservative_mode_steps=3,
        log_dir=str(log_dir),
        results_dir=str(results_dir),
    )

    print(f"\n{'='*64}")
    print(f"  AtomAgents Exp2 — Runtime Layer")
    print(f"  Mode          : {mode.value}")
    print(f"  Predictor     : {args.predictor}")
    print(f"  Run ID        : {run_id}")
    print(f"  HW Profile    : {args.hw_profile}")
    print(f"  Swap models   : {args.swap_models}")
    print(f"  Trace file    : {trace_path}")
    print(f"{'='*64}\n")

    # AtomAgents LAMMPS scripts reference '../potential_repository/' which is
    # relative to the working folder they create.  They expect to be launched
    # from the AtomAgents directory so that '../potential_repository/' resolves
    # correctly.  Our experiment lives one level up in agent-hpc-runtime/, so
    # we must chdir before any LAMMPS work starts.  All output paths were made
    # absolute above, so log/results files still land in the right place.
    _atomagents_dir = _PROJECT_ROOT / "workloads" / "AtomAgents"
    os.chdir(str(_atomagents_dir))
    print(f"[runtime] Working directory set to: {_atomagents_dir}\n")

    # ------------------------------------------------------------------
    # Import AtomAgents components (cluster-only; guarded)
    # ------------------------------------------------------------------
    try:
        from atomagents.instrumentation.autogen_hook import (
            patch_autogen,
            install_text_tool_call_fallback,
        )
        from atomagents.agents.core_execution_agents import admin_core, engineer_core, admin as inner_admin
        import atomagents.tools.registry   # registers tools on admin_core / inner_admin
    except ImportError as e:
        print(f"[cluster] ERROR: AtomAgents not available: {e}")
        print("  This script must be run on a cluster node with AtomAgents installed.")
        print("  For local testing use: python runtime/demo/demo_local.py\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Model orchestration (only if --swap-models)
    # ------------------------------------------------------------------
    orchestrator = None
    if args.swap_models:
        try:
            from atomagents.runtime.model_orchestrator import ModelOrchestrator
            if args.hw_profile == "rtx6000":
                from atomagents.runtime.model_config import MODELS_RTX6000 as MODELS
            elif args.hw_profile == "blackwell":
                from experiments.model_configs import MODELS_BLACKWELL as MODELS
            else:
                from atomagents.runtime.model_config import MODELS
            orchestrator = ModelOrchestrator(MODELS)
            print("[cluster] Starting model orchestrator…")
            orchestrator.ensure_all_models_ready()
        except Exception as e:
            print(f"[cluster] WARNING: Could not start model orchestrator: {e}")
            orchestrator = None

    # ------------------------------------------------------------------
    # Runtime layer setup
    # ------------------------------------------------------------------

    # Apply existing instrumentation first
    patch_autogen()
    install_text_tool_call_fallback(admin_core)
    install_text_tool_call_fallback(inner_admin)

    # Build predictor
    if args.predictor == "oracle":
        print("[cluster] Oracle predictor not yet implemented; falling back to mock.")
        predictor = MockPredictor("atomagents")
    else:
        predictor = MockPredictor("atomagents")

    # Build runtime components
    bus = EventBus(run_id=run_id, log_path=trace_path)
    executor = SimulatedPrefetchExecutor()
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)

    adapter = AtomAgentsRuntimeAdapter(
        predictor=predictor,
        scheduler=scheduler,
        detector=detector,
        bus=bus,
        config=cfg,
    )

    # Patch OpenAIWrapper + register reply handler on admin_core
    adapter.install(admin_core)

    print("[runtime] Adapter installed. Starting experiment…\n")

    # ------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------
    task_prompt = args.task_prompt or DEFAULT_TASK_PROMPT
    t_start = __import__("time").perf_counter()

    try:
        admin_core.initiate_chat(engineer_core, message=task_prompt)
    except KeyboardInterrupt:
        print("\n[cluster] Interrupted by user.")
    except Exception as exc:
        print(f"[cluster] Experiment error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = __import__("time").perf_counter() - t_start
        bus.close()

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------
    print(f"\n[cluster] Experiment finished in {elapsed:.1f}s.")

    try:
        events = _load_events(trace_path)
        summary = analyze(events)
        print_report(summary, verbose=True)

        # Save summary JSON
        summary_path = results_dir / f"summary_{run_id}.json"
        out = {k: v for k, v in summary.items() if k not in ("steps", "prefetch_tasks")}
        out["wall_time_s"] = elapsed
        out["run_id"] = run_id
        out["mode"] = mode.value
        out["predictor"] = args.predictor
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[cluster] Summary written to: {summary_path}")
    except Exception as e:
        print(f"[cluster] Analysis failed: {e}")

    print(f"[cluster] Trace file: {trace_path}")
    print(f"[cluster] Analyze with:")
    print(f"    python runtime/analysis/trace_analyzer.py {trace_path}\n")

    # Print prefetch task summary if any prefetch tasks were started
    tasks = scheduler.all_tasks()
    if tasks:
        print(f"  Prefetch tasks ({len(tasks)} total):")
        for t in tasks:
            proj = t.resource.estimated_load_s or 0
            print(
                f"    {t.task_id[:8]}  {t.resource.name:<25}  "
                f"status={t.status.value:<12}  projected_benefit={proj:.0f}s"
            )
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_events(path: str) -> list[dict]:
    events = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AtomAgents Exp2 with runtime prediction/prefetch layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runtime-mode",
        choices=[m.value for m in RuntimeMode],
        default="observe_only",
        help="Runtime operating mode",
    )
    parser.add_argument(
        "--predictor",
        choices=["mock", "oracle"],
        default="mock",
        help="Predictor implementation",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (auto-generated if omitted)",
    )
    parser.add_argument(
        "--hw-profile",
        choices=["l40s", "rtx6000", "blackwell"],
        default="l40s",
        help="Hardware profile for GPU assignment",
    )
    parser.add_argument(
        "--swap-models",
        action="store_true",
        help="Enable model swapping via ModelOrchestrator (requires GPUs)",
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help="Override the default Exp2 task prompt",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for output CSV / JSON summaries",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/workflow_traces",
        help="Directory for JSONL trace files",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="Predictor confidence threshold for prefetch decisions",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Maximum lookahead steps for prefetch scheduling",
    )

    args = parser.parse_args()
    run_exp2(args)


if __name__ == "__main__":
    main()
