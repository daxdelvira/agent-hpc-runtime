"""
atomagents_exp3.py — AtomAgents Exp3: 3-model forced-swap experiment.

Key differences from exp2:
  - Three models share the same GPU pool (only one resident at a time).
  - ModelRouter is activated via init_router(), so every LLM call triggers
    an automatic model swap if the required model is not currently running.
  - A `code_task` tool is registered on engineer_core; it routes to the
    text-72B specialist (port 8003) for LAMMPS script generation.
  - Task prompt elicits 2× plan_task + 2× code_task + 2× LAMMPS cycles,
    creating 6+ model swap events and meaningful prefetch windows.
  - LAMMPS_SLOWDOWN_S (default 300 via runner) creates ~5-min compute
    windows for the prefetcher to overlap model loads.

Usage
-----
    # Single condition on L40S node:
    bash experiments/run_l40s.sh --condition full_system

    # Full ablation on L40S:
    bash experiments/run_l40s.sh --ablation

    # Single condition on Blackwell (swap profile):
    HW_PROFILE=blackwell_swap bash experiments/run_blackwell.sh --condition full_system

Flags (same as exp2 plus):
    --hw-profile   l40s | blackwell | blackwell_swap (default: l40s)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

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
# Exp3 task prompt — 3-model, 2-LAMMPS-cycle workflow
#
# Designed to elicit:
#   engineer (72B-VL)  →  plan_task (32B-VL)  →  code_task (72B-text)
#   →  computation_task (LAMMPS 1, 5-min window)
#   →  plan_task again (32B-VL)  →  code_task (72B-text)
#   →  computation_task (LAMMPS 2, 5-min window)
#   →  analyze_screw_core ×2  →  report
#
# Model swap sequence (with all models sharing GPUs):
#   start 72B-VL → swap to 32B (plan1) → swap to 72B-text (code1)
#   → swap to 72B-VL (LAMMPS1) → swap to 32B (plan2) → swap to 72B-text (code2)
#   → swap to 72B-VL (LAMMPS2 + analyze + report)
# ---------------------------------------------------------------------------
DEFAULT_TASK_PROMPT_EXP3 = """\
Compare the core structure of the 1/2<111> screw dislocation in W using \
"W_Zhou04.eam.alloy" and "w_eam4.fs" EAM potentials.
The dislocation line is along [-1,1,1], glide direction [1,-1,2], glide plane normal [1,1,0].

Required steps (do these exactly, in this order, then TERMINATE):
1. Call plan_task to develop a detailed step-by-step simulation plan for both potentials.
2. Call code_task to get the LAMMPS input parameters and script outline verified by the \
code specialist for potential 1 (W_Zhou04.eam.alloy). Pass a clear description of what \
the script should do (create crystal, introduce screw dislocation, relax with CG minimizer).
3. Call computation_task_screw_dislocation for W_Zhou04.eam.alloy using guidance from step 2.
4. Call plan_task again to review the intermediate result from step 3 and re-plan the \
approach for potential 2 (w_eam4.fs), noting any adjustments needed.
5. Call code_task to get the LAMMPS input parameters for potential 2 (w_eam4.fs) reviewed \
by the code specialist, incorporating any adjustments from step 4.
6. Call computation_task_screw_dislocation for w_eam4.fs using guidance from step 5.
7. Call analyze_screw_core with the DD map path returned in step 3 to classify the core structure.
8. Call analyze_screw_core with the DD map path returned in step 6 to classify the core structure.
9. Report whether each potential gives a polarized or unpolarized core and summarize the \
structural comparison.

Critical rules:
- Always call plan_task before each set of computations (steps 1 and 4).
- Always call code_task before each computation_task_screw_dislocation (steps 2 and 5).
- Do NOT compute surface energy, elastic constants, stacking fault energy, NEB barriers, \
or any other property.
- Do NOT invent tool names. Only call tools that are registered with you.
"""


# ---------------------------------------------------------------------------
# Build prefetch executor
# ---------------------------------------------------------------------------

def _build_executor(mode: RuntimeMode, orchestrator, metrics=None):
    if mode != RuntimeMode.REAL:
        return SimulatedPrefetchExecutor()
    try:
        from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
        from runtime.prefetch.data_prefetch import FileStagingExecutor, CompositeExecutor
        executors = {}
        if orchestrator is not None:
            executors["vllm_model"] = ModelPrefetchExecutor(orchestrator, probes=None)
            print("[runtime] Real executor: ModelPrefetchExecutor for vllm_model")
        executors["data_file"] = FileStagingExecutor()
        print(f"[runtime] Real executor: FileStagingExecutor → {executors['data_file'].scratch_dir}")
        return CompositeExecutor(executors=executors, default=SimulatedPrefetchExecutor())
    except Exception as exc:
        print(f"[runtime] WARNING: Could not build real executor ({exc}); falling back to simulated.")
        return SimulatedPrefetchExecutor()


# ---------------------------------------------------------------------------
# code_task — routes to text-72B specialist (port 8003)
# ---------------------------------------------------------------------------

def _make_code_task(text72b_url: str):
    """
    Factory returning a `code_task` function that calls the text-72B specialist.

    Before making the HTTP call, it invokes ModelRouter.ensure_ready() so the
    router automatically stops whichever model is currently loaded and starts
    qwen_72b_text on the shared GPU pool.
    """
    import json as _json
    import urllib.request as _urllib

    def code_task(
        task: Annotated[str, "Scripting or parameter-verification task for the code specialist"],
        context: Annotated[str, "Additional context, constraints, or prior results"] = "",
    ) -> str:
        """
        Delegate a LAMMPS input-script or code task to the text-72B specialist model.

        Returns the specialist's response (parameters, pseudocode, or a script outline).
        Use the output as guidance when calling computation_task_screw_dislocation.
        """
        from atomagents.runtime.model_router import get_router
        router = get_router()
        if router is not None:
            router.ensure_ready(text72b_url)

        prompt = task
        if context:
            prompt = f"{task}\n\nContext / prior results:\n{context}"

        payload = _json.dumps({
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a specialist in writing LAMMPS input scripts and Python "
                        "code for molecular dynamics simulations. Provide clear, correct, "
                        "concise guidance. Focus on the specific parameters and script "
                        "structure requested. Do not repeat the full task description back."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.05,
        }).encode()

        req = _urllib.Request(
            f"{text72b_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token-abc123",
            },
        )
        try:
            with _urllib.urlopen(req, timeout=300) as resp:
                data = _json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"[code_task ERROR] {exc}"

    return code_task


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_exp3(args: argparse.Namespace) -> None:
    run_id = args.run_id or str(uuid.uuid4())[:12]
    mode = RuntimeMode(args.runtime_mode)

    if args.lammps_slowdown > 0:
        os.environ["LAMMPS_SLOWDOWN_S"] = str(args.lammps_slowdown)
        print(f"[runtime] LAMMPS slowdown : {args.lammps_slowdown}s per relax step")

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
    print(f"  AtomAgents Exp3 — 3-Model Forced-Swap Experiment")
    print(f"  Mode          : {mode.value}")
    print(f"  Predictor     : {args.predictor}")
    print(f"  Condition     : {condition}")
    print(f"  Run ID        : {run_id}")
    print(f"  HW Profile    : {args.hw_profile}")
    print(f"  Swap models   : {args.swap_models}")
    print(f"  LAMMPS slowdn : {args.lammps_slowdown}s per relax")
    print(f"  Plan horizon  : {cfg.plan_extraction_horizon} steps")
    print(f"{'='*64}\n")

    _atomagents_dir = _PROJECT_ROOT / "workloads" / "AtomAgents"
    os.chdir(str(_atomagents_dir))
    print(f"[runtime] Working directory: {_atomagents_dir}\n")

    # ------------------------------------------------------------------
    # Import AtomAgents components
    # ------------------------------------------------------------------
    try:
        from atomagents.instrumentation.autogen_hook import (
            patch_autogen,
            install_text_tool_call_fallback,
        )
        from atomagents.agents.core_execution_agents import admin_core, engineer_core, admin as inner_admin
        import atomagents.tools.registry
        from atomagents.instrumentation.metrics_logger import init_logger as _init_metrics
    except ImportError as e:
        print(f"[cluster] ERROR: AtomAgents not available: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Model orchestration
    # ------------------------------------------------------------------
    orchestrator = None
    MODELS = None
    if args.swap_models:
        try:
            from atomagents.runtime.model_orchestrator import ModelOrchestrator
            if args.hw_profile == "l40s":
                from experiments.model_configs import MODELS_L40S as MODELS
            elif args.hw_profile == "blackwell_swap":
                from experiments.model_configs import MODELS_BLACKWELL_SWAP as MODELS
            elif args.hw_profile == "blackwell":
                from experiments.model_configs import MODELS_BLACKWELL as MODELS
            elif args.hw_profile == "rtx6000":
                from atomagents.runtime.model_config import MODELS_RTX6000 as MODELS
            else:
                from atomagents.runtime.model_config import MODELS
            orchestrator = ModelOrchestrator(MODELS)

            if args.no_start_models:
                print("[cluster] --no-start-models: skipping server startup.")
            else:
                base_model = next(
                    (n for n in MODELS if "72b" in n.lower() and "text" not in n.lower()),
                    list(MODELS.keys())[0],
                )
                if mode == RuntimeMode.REAL:
                    print(f"[cluster] Starting base model ({base_model}) — "
                          f"runtime will speculatively load others.")
                else:
                    # Shared-GPU profiles cannot have all models resident simultaneously;
                    # start only the base model even in non-real modes.
                    print(f"[cluster] Starting base model ({base_model}) in {mode} mode "
                          f"(shared GPUs: other models load on demand, no speculation).")
                orchestrator.start_model_measured(base_model, metrics=None)
        except Exception as e:
            print(f"[cluster] WARNING: Could not start model orchestrator: {e}")
            orchestrator = None

    # ------------------------------------------------------------------
    # Activate ModelRouter — enables automatic model swapping on every LLM call.
    # This is the critical difference from exp2: without init_router() the router
    # singleton is None and ensure_ready() is a no-op, so no swaps occur even
    # when models share GPUs.
    # ------------------------------------------------------------------
    if orchestrator is not None and MODELS is not None:
        try:
            from atomagents.runtime.model_router import init_router
            router = init_router(orchestrator, MODELS)
            print(f"[cluster] ModelRouter activated — "
                  f"{len(MODELS)} models, port map: "
                  f"{ {cfg['port']: name for name, cfg in MODELS.items()} }")
        except Exception as e:
            print(f"[cluster] WARNING: Could not init ModelRouter: {e}")

    # ------------------------------------------------------------------
    # Runtime layer setup
    # ------------------------------------------------------------------
    patch_autogen()
    install_text_tool_call_fallback(admin_core)
    install_text_tool_call_fallback(inner_admin)

    metrics_csv = str(results_dir / f"atomagents_metrics_{run_id}.csv")
    ml = _init_metrics(
        run_id=run_id,
        mode="baseline" if mode == RuntimeMode.BASELINE else "agent",
        csv_path=metrics_csv,
    )
    print(f"[runtime] Metrics CSV     : {metrics_csv}")

    if args.predictor == "learned":
        from runtime.predictor.learned_predictor import LearnedPredictor
        predictor = LearnedPredictor()
        print(f"[cluster] Predictor: LearnedPredictor "
              f"(transitions={predictor._table.n_traces} traces)")
    elif args.predictor == "oracle":
        print("[cluster] Oracle predictor not yet implemented; falling back to mock.")
        predictor = MockPredictor("atomagents")
    else:
        predictor = MockPredictor("atomagents")

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
    adapter.install(admin_core)

    # ------------------------------------------------------------------
    # Register code_task tool on engineer_core / admin_core
    # ------------------------------------------------------------------
    text72b_url = "http://localhost:8003"
    code_task_fn = _make_code_task(text72b_url)

    try:
        from autogen import register_function
        register_function(
            code_task_fn,
            caller=engineer_core,
            executor=admin_core,
            name="code_task",
            description=(
                "Delegate a LAMMPS input-script writing or parameter-verification task "
                "to the text-72B code specialist model. "
                "Call this BEFORE each computation_task_screw_dislocation to get "
                "the script parameters reviewed and confirmed. "
                "Returns guidance, parameters, or a script outline."
            ),
        )
        print("[cluster] code_task registered on engineer_core via register_function")
    except Exception as e:
        print(f"[cluster] WARNING: register_function failed ({e}); "
              "trying legacy API…")
        try:
            admin_core.register_function(function_map={"code_task": code_task_fn})
            engineer_core.update_function_signature(
                {
                    "name": "code_task",
                    "description": (
                        "Delegate a LAMMPS input-script or code task to the text-72B specialist."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The scripting task",
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context",
                                "default": "",
                            },
                        },
                        "required": ["task"],
                    },
                },
                is_remove=False,
            )
            print("[cluster] code_task registered via legacy update_function_signature")
        except Exception as e2:
            print(f"[cluster] WARNING: code_task registration failed ({e2}); "
                  "engineer_core will not be able to call it.")

    # ------------------------------------------------------------------
    # System profiler
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
    task_prompt = args.task_prompt or DEFAULT_TASK_PROMPT_EXP3
    t_start = __import__("time").perf_counter()

    print("[runtime] Adapter installed. Starting experiment…\n")

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

        summary_path = results_dir / f"summary_{run_id}.json"
        out = {k: v for k, v in summary.items() if k not in ("steps", "prefetch_tasks")}
        out["wall_time_s"] = elapsed
        out["run_id"] = run_id
        out["mode"] = mode.value
        out["predictor"] = args.predictor
        out["condition"] = condition
        out["experiment"] = "exp3"
        out["hw_profile"] = args.hw_profile
        out["lammps_slowdown_s"] = args.lammps_slowdown
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
        description="AtomAgents Exp3 — 3-model forced-swap with real prefetch benefit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runtime-mode",
        choices=[m.value for m in RuntimeMode],
        default="observe_only",
    )
    parser.add_argument(
        "--predictor",
        choices=["mock", "learned", "oracle"],
        default="mock",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--hw-profile",
        choices=["l40s", "blackwell", "blackwell_swap", "rtx6000"],
        default="l40s",
        help="Hardware profile: l40s and blackwell_swap use shared GPU pool (forced swaps); "
             "blackwell uses original separate-pair layout (no forced swaps).",
    )
    parser.add_argument(
        "--swap-models",
        action="store_true",
        help="Enable model swapping via ModelOrchestrator + ModelRouter",
    )
    parser.add_argument(
        "--no-start-models",
        action="store_true",
        dest="no_start_models",
    )
    parser.add_argument(
        "--lammps-slowdown",
        type=int,
        default=0,
        dest="lammps_slowdown",
        help="Seconds of sleep after each LAMMPS relax step (simulates NFS load).",
    )
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-dir", default="logs/workflow_traces")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--horizon", type=int, default=2)

    # Ablation flags
    parser.add_argument("--no-plan-extraction", action="store_true", dest="no_plan_extraction")
    parser.add_argument("--no-divergence-guard", action="store_true", dest="no_divergence_guard")
    parser.add_argument("--naive-prefetch", action="store_true", dest="naive_prefetch")
    parser.add_argument("--skip-resource-types", default="", dest="skip_resource_types")
    parser.add_argument("--condition", default=None)

    args = parser.parse_args()
    run_exp3(args)


if __name__ == "__main__":
    main()
