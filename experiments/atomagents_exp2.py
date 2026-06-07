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
    --runtime-mode         observe_only | simulated | real | baseline (default: observe_only)
    --predictor            mock | learned | oracle (default: mock)
    --run-id               custom run ID (default: auto-generated)
    --hw-profile           l40s | rtx6000 | blackwell (default: l40s)
    --swap-models          enable model swapping
    --task-prompt          override the default Exp2 task prompt
    --results-dir          directory for output CSV / JSONL (default: results/)
    --log-dir              directory for JSONL traces (default: logs/workflow_traces/)
    --confidence           predictor confidence threshold (default: 0.85)
    --horizon              max lookahead in steps (default: 2)

Ablation flags (deactivate one component at a time)
----------------------------------------------------
    --no-plan-extraction   disable plan extraction (sets plan_extraction_horizon=0)
    --no-divergence-guard  disable prefetch cancellation on mismatch
    --naive-prefetch       prefetch every prediction regardless of confidence
    --skip-resource-types  comma-separated resource types to never prefetch
                           (e.g. "vllm_model" or "data_file,vllm_model")
    --condition            human-readable label stored in summary JSON
                           (auto-inferred from ablation flags if omitted)
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

# Real-prefetch executors — imported lazily inside run_exp2() so the file
# is safe to import locally without atomagents / psutil installed.
_REAL_EXECUTORS_AVAILABLE = True


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

DEFAULT_TASK_PROMPT = """\
Compare the core structure of the 1/2<111> screw dislocation in W using \
"W_Zhou04.eam.alloy" and "w_eam4.fs" EAM potentials.
The dislocation line is along [-1,1,1], glide direction [1,-1,2], glide plane normal [1,1,0].

Required steps (do these exactly, in order, then TERMINATE):
1. Call computation_task_screw_dislocation for W_Zhou04.eam.alloy to get its DD map path.
2. Call computation_task_screw_dislocation for w_eam4.fs to get its DD map path.
3. Call analyze_screw_core with the DD map from step 1 to classify the core structure.
4. Call analyze_screw_core with the DD map from step 2 to classify the core structure.
5. Report whether each potential gives a polarized or unpolarized core, and summarize the comparison.

Do NOT compute surface energy, elastic constants, stacking fault energy, NEB barriers, \
or any other property beyond the DD maps and core structure classification.
"""


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _build_executor(mode: RuntimeMode, orchestrator, metrics=None):
    """
    Select the appropriate prefetch executor for the given runtime mode.

    REAL mode with an active orchestrator → CompositeExecutor:
      - vllm_model  → ModelPrefetchExecutor  (background model server start)
      - data_file   → FileStagingExecutor     (copy EAM files to $SCRATCH)
      - everything else → SimulatedPrefetchExecutor (log-only fallback)

    All other modes → SimulatedPrefetchExecutor (no real I/O).
    """
    if mode != RuntimeMode.REAL:
        return SimulatedPrefetchExecutor()

    try:
        from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
        from runtime.prefetch.data_prefetch import FileStagingExecutor, CompositeExecutor

        executors = {}
        if orchestrator is not None:
            executors["vllm_model"] = ModelPrefetchExecutor(
                orchestrator, probes=None
            )
            print("[runtime] Real executor: ModelPrefetchExecutor for vllm_model")
        executors["data_file"] = FileStagingExecutor()
        print(f"[runtime] Real executor: FileStagingExecutor → {executors['data_file'].scratch_dir}")

        return CompositeExecutor(executors=executors, default=SimulatedPrefetchExecutor())
    except Exception as exc:
        print(f"[runtime] WARNING: Could not build real executor ({exc}); falling back to simulated.")
        return SimulatedPrefetchExecutor()


def run_exp2(args: argparse.Namespace) -> None:
    run_id = args.run_id or str(uuid.uuid4())[:12]
    mode = RuntimeMode(args.runtime_mode)

    # Propagate slowdown before AtomAgents tools are imported so the env var
    # is visible when screw_dislocation.py reads it at call time.
    if args.lammps_slowdown > 0:
        os.environ["LAMMPS_SLOWDOWN_S"] = str(args.lammps_slowdown)
        print(f"[runtime] LAMMPS slowdown : {args.lammps_slowdown}s per relax step")

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

    # Resolve ablation flags → condition label
    skip_types = [t.strip() for t in (args.skip_resource_types or "").split(",") if t.strip()]
    condition = args.condition or _infer_condition(args, skip_types)

    cfg = RuntimeConfig(
        mode=mode,
        run_id=run_id,
        confidence_threshold=args.confidence,
        max_horizon=args.horizon,
        conservative_mode_steps=3,
        plan_extraction_horizon=0 if args.no_plan_extraction else 3,
        disable_divergence_cancellation=args.no_divergence_guard,
        naive_prefetch=args.naive_prefetch,
        skip_resource_types=skip_types,
        condition=condition,
        log_dir=str(log_dir),
        results_dir=str(results_dir),
    )

    print(f"\n{'='*64}")
    print(f"  AtomAgents Exp2 — Runtime Layer")
    print(f"  Mode          : {mode.value}")
    print(f"  Predictor     : {args.predictor}")
    print(f"  Condition     : {condition}")
    print(f"  Run ID        : {run_id}")
    print(f"  HW Profile    : {args.hw_profile}")
    print(f"  Swap models   : {args.swap_models}")
    print(f"  No-start-mdls : {getattr(args, 'no_start_models', False)}")
    print(f"  LAMMPS slowdn : {args.lammps_slowdown}s per relax")
    print(f"  Plan horizon  : {cfg.plan_extraction_horizon} steps")
    print(f"  Naive prefetch: {cfg.naive_prefetch}")
    print(f"  No diverg grd : {cfg.disable_divergence_cancellation}")
    print(f"  Skip types    : {cfg.skip_resource_types or 'none'}")
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
        from atomagents.instrumentation.metrics_logger import init_logger as _init_metrics
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

            if args.no_start_models:
                print("[cluster] --no-start-models set: skipping server startup "
                      "(assuming models already running at configured ports).")
            elif mode == RuntimeMode.REAL:
                # Real-prefetch mode: only pre-load the base (72B) model.
                # ModelPrefetchExecutor will speculatively load qwen_32b
                # while qwen_72b is still reasoning.
                base_model = next(
                    (n for n in MODELS if "72b" in n.lower()), list(MODELS.keys())[-1]
                )
                print(f"[cluster] Starting base model ({base_model}) only — "
                      f"runtime will speculatively load others.")
                orchestrator.start_model_measured(base_model, metrics=None)
            else:
                print("[cluster] Starting model orchestrator (all models)…")
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

    # Initialise per-phase metrics logger (disk/net/GPU/token CSV).
    # Must come after patch_autogen() so the autogen hook can call
    # get_metrics_logger() for every LLM response it intercepts.
    metrics_csv = str(results_dir / f"atomagents_metrics_{run_id}.csv")
    ml = _init_metrics(
        run_id=run_id,
        mode="baseline" if mode == RuntimeMode.BASELINE else "agent",
        csv_path=metrics_csv,
    )
    print(f"[runtime] Metrics CSV     : {metrics_csv}")

    # Build predictor
    if args.predictor == "learned":
        from runtime.predictor.learned_predictor import LearnedPredictor
        predictor = LearnedPredictor()
        print(f"[cluster] Predictor: LearnedPredictor "
              f"(transitions={predictor._table.n_traces} traces, "
              f"plan_horizon={cfg.plan_extraction_horizon} steps)")
    elif args.predictor == "oracle":
        print("[cluster] Oracle predictor not yet implemented; falling back to mock.")
        predictor = MockPredictor("atomagents")
    else:
        predictor = MockPredictor("atomagents")

    # Build runtime components
    bus = EventBus(run_id=run_id, log_path=trace_path)
    executor = _build_executor(mode, orchestrator, ml)
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
    # System profiler — continuous CPU+GPU sampling across all processes
    # (vLLM 72B, vLLM 32B, LAMMPS, orchestration) for CPU:GPU ratio analysis
    # ------------------------------------------------------------------
    from runtime.measurement.system_profiler import SystemProfiler
    profiler = SystemProfiler(
        run_id=run_id,
        results_dir=str(results_dir),
        interval_s=3.0,
        port_72b=8001,
        port_32b=8002,
    )
    profiler.start()

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
        profiler.stop()
        bus.close()
        ml.write_summary(total_wall_s=elapsed)
        ml.close()

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------
    print(f"\n[cluster] Experiment finished in {elapsed:.1f}s.")
    print(f"[cluster] System profile  : {profiler.csv_path}")

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
        out["condition"] = condition
        out["ablation"] = {
            "no_plan_extraction": args.no_plan_extraction,
            "no_divergence_guard": args.no_divergence_guard,
            "naive_prefetch": args.naive_prefetch,
            "skip_resource_types": skip_types,
        }
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[cluster] Summary written to: {summary_path}")
    except Exception as e:
        print(f"[cluster] Analysis failed: {e}")

    print(f"[cluster] Trace file    : {trace_path}")
    print(f"[cluster] Metrics CSV   : {metrics_csv}")
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

def _infer_condition(args: argparse.Namespace, skip_types: list[str]) -> str:
    """Auto-generate a condition label from active ablation flags."""
    if args.runtime_mode == "baseline":
        return "baseline"
    parts = []
    if args.no_plan_extraction:
        parts.append("no_plan")
    if args.no_divergence_guard:
        parts.append("no_diverg_guard")
    if args.naive_prefetch:
        parts.append("naive_prefetch")
    for t in skip_types:
        parts.append(f"no_{t.replace('_', '')}")
    if args.predictor != "mock":
        parts.append(f"pred_{args.predictor}")
    return "_".join(parts) if parts else "full_system"


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
        choices=["mock", "learned", "oracle"],
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
        "--no-start-models",
        action="store_true",
        dest="no_start_models",
        help="Skip vLLM server startup — use when servers are already running "
             "(e.g. started manually before the experiment). "
             "The orchestrator is still created so ModelPrefetchExecutor can "
             "speculatively start additional models.",
    )
    parser.add_argument(
        "--lammps-slowdown",
        type=int,
        default=0,
        dest="lammps_slowdown",
        help="Add N seconds of sleep after each LAMMPS relax step to simulate "
             "a slower NFS-backed run. Useful for prefetch overlap testing on "
             "fast local hardware. Default 0 (no slowdown).",
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

    # ---- Ablation flags ----
    parser.add_argument(
        "--no-plan-extraction",
        action="store_true",
        dest="no_plan_extraction",
        help="Ablation: disable plan extraction (sets plan_extraction_horizon=0)",
    )
    parser.add_argument(
        "--no-divergence-guard",
        action="store_true",
        dest="no_divergence_guard",
        help="Ablation: do not cancel prefetches or enter conservative mode on mismatch",
    )
    parser.add_argument(
        "--naive-prefetch",
        action="store_true",
        dest="naive_prefetch",
        help="Ablation: prefetch every prediction ignoring confidence threshold",
    )
    parser.add_argument(
        "--skip-resource-types",
        default="",
        dest="skip_resource_types",
        help="Ablation: comma-separated resource types to never prefetch "
             "(e.g. 'vllm_model' or 'data_file,vllm_model')",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Human-readable condition label stored in summary JSON "
             "(auto-inferred from ablation flags if omitted)",
    )

    args = parser.parse_args()
    run_exp2(args)


if __name__ == "__main__":
    main()
