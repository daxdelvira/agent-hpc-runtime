"""
chemgraph_exp.py — ChemGraph MACE geometry-optimization experiment with
runtime prediction/prefetch layer.

Measures whether pre-loading the MACE calculator in a background thread
(while the LLM is reasoning) can hide the 30–60 s model-load latency.

Runtime modes
-------------
baseline     — no runtime layer; normal ChemGraph run for comparison
observe_only — emits prediction events; no prefetch I/O
simulated    — logs "would prefetch MACE" with estimated timing; no I/O
real         — MacePrefetchExecutor loads MACE in background

Usage
-----
    # Safe first run: observe predictions only
    python experiments/chemgraph_exp.py --runtime-mode observe_only

    # Baseline for wall-time comparison
    python experiments/chemgraph_exp.py --runtime-mode baseline

    # Real prefetch (CPU MACE, any machine)
    python experiments/chemgraph_exp.py --runtime-mode real

    # Real prefetch on cluster with CUDA MACE
    python experiments/chemgraph_exp.py --runtime-mode real --mace-device cuda

Flags
-----
    --runtime-mode   observe_only | simulated | real | baseline  (default: observe_only)
    --predictor      mock | learned  (default: mock)
    --run-id         custom run ID  (default: auto-generated)
    --task-prompt    override the default geometry-optimization task
    --model-name     LLM model name for ChemGraph  (default: gpt-4o-mini or env CHEMGRAPH_MODEL)
    --base-url       LLM API base URL  (optional, for local vLLM or Groq)
    --api-key        LLM API key  (optional; reads OPENAI_API_KEY if omitted)
    --mace-device    Device for MACE calculations  (default: cpu)
    --results-dir    directory for output JSON summaries  (default: results/)
    --log-dir        directory for JSONL traces  (default: logs/workflow_traces/)
    --confidence     predictor confidence threshold  (default: 0.85)
    --horizon        max lookahead in steps  (default: 2)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

# Make runtime importable
sys.path.insert(0, str(_PROJECT_ROOT))
# Make ChemGraph importable (agent_hpc/ChemGraph/src)
_CG_SRC = _PROJECT_ROOT.parent / "agent_hpc" / "ChemGraph" / "src"
if _CG_SRC.exists():
    sys.path.insert(0, str(_CG_SRC))

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.prefetch.scheduler import PrefetchScheduler
from runtime.prefetch.simulated import SimulatedPrefetchExecutor
from runtime.guard.detector import DivergenceDetector
from runtime.predictor.mock_predictor import MockPredictor
from runtime.analysis.trace_analyzer import analyze, print_report
from runtime.analysis.overlap_report import report_from_events


# ---------------------------------------------------------------------------
# Default task prompt — optimise a small molecule with MACE-MP
# ---------------------------------------------------------------------------

DEFAULT_TASK_PROMPT = """\
Optimize the geometry of a water molecule (H2O) using the MACE-MP force field.
Use the following steps:
1. Convert "water" to a SMILES string.
2. Generate a 3D coordinate file from the SMILES.
3. Run ASE geometry optimization using the mace_mp calculator (medium model).
4. Report the final optimized energy (in eV) and confirm the O-H bond lengths.
"""

# Multi-step task that exercises the full prediction window:
# smiles_to_coordinate → run_ase (MACE loads here) → extract_output_json
EXTENDED_TASK_PROMPT = """\
Compare the equilibrium geometry of ammonia (NH3) and water (H2O) using the
MACE-MP force field.

For each molecule:
1. Convert the molecule name to SMILES.
2. Generate a 3D coordinate file.
3. Run ASE geometry optimization with the mace_mp calculator, fmax=0.01.
4. Report the final energy (eV), bond lengths (Å), and bond angles (°).
Then compare the two results and summarise key geometric differences.
"""


# ---------------------------------------------------------------------------
# Executor builder
# ---------------------------------------------------------------------------

def _build_executor(mode: RuntimeMode, mace_device: str):
    """Return the appropriate PrefetchExecutor for the runtime mode."""
    if mode != RuntimeMode.REAL:
        return SimulatedPrefetchExecutor()

    try:
        from runtime.prefetch.mace_prefetch import MacePrefetchExecutor
        from runtime.prefetch.data_prefetch import CompositeExecutor
        executor = CompositeExecutor(
            executors={"mace_model": MacePrefetchExecutor(device=mace_device)},
            default=SimulatedPrefetchExecutor(),
        )
        print(f"[runtime] Real executor: MacePrefetchExecutor(device={mace_device!r})")
        return executor
    except Exception as exc:
        print(f"[runtime] WARNING: Could not build MacePrefetchExecutor ({exc}); "
              "falling back to simulated.")
        return SimulatedPrefetchExecutor()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_chemgraph_exp(args: argparse.Namespace) -> None:
    run_id = args.run_id or str(uuid.uuid4())[:12]
    mode = RuntimeMode(args.runtime_mode)

    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir)
    if not results_dir.is_absolute():
        results_dir = (_PROJECT_ROOT / results_dir).resolve()
    if not log_dir.is_absolute():
        log_dir = (_PROJECT_ROOT / log_dir).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = str(log_dir / f"chemgraph_trace_{ts}_{run_id}.jsonl")

    print(f"\n{'='*64}")
    print(f"  ChemGraph MACE Experiment — Runtime Layer")
    print(f"  Mode          : {mode.value}")
    print(f"  Predictor     : {args.predictor}")
    print(f"  Run ID        : {run_id}")
    print(f"  MACE device   : {args.mace_device}")
    print(f"  Trace file    : {trace_path}")
    print(f"{'='*64}\n")

    # Signal ChemGraph's ase_tools to check the MACE prefetch cache
    if mode != RuntimeMode.BASELINE:
        os.environ["RUNTIME_ENABLED"] = "1"
    else:
        os.environ.pop("RUNTIME_ENABLED", None)

    # ------------------------------------------------------------------
    # Import ChemGraph components (guarded)
    # ------------------------------------------------------------------
    try:
        from chemgraph.agent.llm_agent import ChemGraph
    except ImportError as exc:
        print(f"[chemgraph] ERROR: ChemGraph not available: {exc}")
        print(f"  Searched: {_CG_SRC}")
        print("  Install ChemGraph or set PYTHONPATH to include its src/ directory.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Runtime layer setup
    # ------------------------------------------------------------------
    cfg = RuntimeConfig(
        mode=mode,
        run_id=run_id,
        confidence_threshold=args.confidence,
        max_horizon=args.horizon,
        conservative_mode_steps=3,
        log_dir=str(log_dir),
        results_dir=str(results_dir),
    )

    bus = EventBus(run_id=run_id, log_path=trace_path)
    executor = _build_executor(mode, args.mace_device)
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)

    if args.predictor == "learned":
        try:
            from runtime.predictor.learned_predictor import LearnedPredictor
            predictor = LearnedPredictor()
            print(f"[chemgraph] Predictor: LearnedPredictor")
        except Exception as exc:
            print(f"[chemgraph] WARNING: LearnedPredictor failed ({exc}); using mock.")
            predictor = MockPredictor("chemgraph")
    else:
        predictor = MockPredictor("chemgraph")

    # Build the runtime callback (replaces ChemGraph's standard callback)
    task_prompt = args.task_prompt or DEFAULT_TASK_PROMPT
    callback = None
    if mode != RuntimeMode.BASELINE:
        try:
            from runtime.adapters.chemgraph import make_runtime_callback
            callback = make_runtime_callback(
                predictor=predictor,
                scheduler=scheduler,
                config=cfg,
                bus=bus,
                task_description=task_prompt,
            )
            print("[runtime] RuntimeChemGraphCallback installed.")
        except Exception as exc:
            print(f"[runtime] WARNING: Could not create runtime callback ({exc}); "
                  "running without runtime instrumentation.")

    # ------------------------------------------------------------------
    # Build ChemGraph agent
    # ------------------------------------------------------------------
    model_name = args.model_name or os.environ.get("CHEMGRAPH_MODEL", "gpt-4o-mini")
    cg_kwargs: dict = {"model_name": model_name, "workflow_type": "single_agent"}
    if args.base_url:
        cg_kwargs["base_url"] = args.base_url
    if args.api_key:
        cg_kwargs["api_key"] = args.api_key

    try:
        agent = ChemGraph(**cg_kwargs)
        print(f"[chemgraph] Agent created: model={model_name!r}\n")
    except Exception as exc:
        print(f"[chemgraph] ERROR: Could not create ChemGraph agent: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------
    langgraph_config: dict = {"configurable": {"thread_id": run_id}}
    if callback is not None:
        langgraph_config["callbacks"] = [callback]

    t_start = __import__("time").perf_counter()
    try:
        result = asyncio.run(agent.run(task_prompt, config=langgraph_config))
        print("\n[chemgraph] Workflow completed.")
        if result:
            try:
                content = getattr(result, "content", str(result))
                print(f"\n--- Result ---\n{content[:2000]}\n")
            except Exception:
                pass
    except KeyboardInterrupt:
        print("\n[chemgraph] Interrupted by user.")
    except Exception as exc:
        print(f"[chemgraph] Experiment error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = __import__("time").perf_counter() - t_start
        bus.close()

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------
    print(f"\n[chemgraph] Experiment finished in {elapsed:.1f}s.")

    try:
        events = _load_events(trace_path)

        summary = analyze(events)
        print_report(summary, verbose=True)

        overlap = report_from_events(events, run_id=run_id)

        summary_path = results_dir / f"summary_{run_id}.json"
        out = {k: v for k, v in summary.items() if k not in ("steps", "prefetch_tasks")}
        out.update({
            "wall_time_s": elapsed,
            "run_id": run_id,
            "mode": mode.value,
            "predictor": args.predictor,
            "workflow": "chemgraph_mace",
            "model_name": model_name,
            "mace_device": args.mace_device,
            "total_benefit_s": overlap.get("total_benefit_s", 0.0),
            "total_waste_s": overlap.get("total_waste_s", 0.0),
            "estimated_total_benefit_s": overlap.get("estimated_total_benefit_s", 0.0),
        })
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[chemgraph] Summary written to: {summary_path}")
    except Exception as exc:
        print(f"[chemgraph] Analysis failed: {exc}")

    print(f"[chemgraph] Trace file : {trace_path}")
    print(f"[chemgraph] Analyze with:")
    print(f"    python runtime/analysis/trace_analyzer.py {trace_path}")
    print(f"    python runtime/analysis/overlap_report.py {trace_path}\n")


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
        description="ChemGraph MACE geometry-optimization experiment with runtime layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runtime-mode",
        choices=[m.value for m in RuntimeMode],
        default="observe_only",
    )
    parser.add_argument(
        "--predictor",
        choices=["mock", "learned"],
        default="mock",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--task-prompt",
        default=None,
        help="Override the default geometry-optimization task",
    )
    parser.add_argument(
        "--extended-task",
        action="store_true",
        help="Use the extended multi-molecule task (NH3 + H2O)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="LLM model name for ChemGraph (default: gpt-4o-mini or CHEMGRAPH_MODEL env)",
    )
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument(
        "--mace-device",
        default="cpu",
        help="PyTorch device for MACE calculations (cpu | cuda | cuda:0 ...)",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-dir", default="logs/workflow_traces")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="Predictor confidence threshold",
    )
    parser.add_argument("--horizon", type=int, default=2)

    args = parser.parse_args()

    if args.extended_task and not args.task_prompt:
        args.task_prompt = EXTENDED_TASK_PROMPT

    run_chemgraph_exp(args)


if __name__ == "__main__":
    main()
