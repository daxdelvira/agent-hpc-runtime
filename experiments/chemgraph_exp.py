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
_CG_SRC = _PROJECT_ROOT.parent / "ChemGraph" / "src"
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

# Out-of-core / high-compute task: run MACE over an entire directory of crystal
# structures.  Produces a genuine multi-minute CPU compute window (GPU idle),
# used to hide a model prefetch/swap.  {dataset} is filled from --ensemble-dataset.
ENSEMBLE_TASK_PROMPT = """\
Screen an entire dataset of crystal structures with the MACE-MP force field.
The dataset is a DIRECTORY containing many (~200) .cif files: {dataset}

You must process the WHOLE directory in ONE batch call. Do this in a single step:
1. Call run_mace_ensemble with input_structure_directory="{dataset}",
   driver="opt", model="medium", device="cpu"{limit_arg}. This relaxes every structure in
   the directory and returns a summary (n_structures, energies, timing).
2. Report how many structures were processed and the range of energies.

Do NOT call any single-structure tool, do NOT loop over individual files, and do
NOT invent file names. The only structure input is the directory path above, and
run_mace_ensemble is the only calculation tool you have.
"""

# Screen workload: heterogeneous molecule batch with per-molecule SPECIALIST
# routing.  The planner tags every worker task with "[SPECIALIST: advanced]"
# (complex organics — multi-ring, many heteroatoms, drug-like) or
# "[SPECIALIST: standard]" (small/simple molecules).  The runtime reads those
# tags from the extracted plan to decide which specialist model to stage next;
# planner misjudgements on borderline molecules are genuine divergences.
# {molecules} is filled from --molecules (default below).
SCREEN_TASK_PROMPT = """\
Screen the following molecules with the MACE-MP force field, one worker task
per molecule, in the given order: {molecules}

When you create the plan, tag EVERY worker task prompt with exactly one
specialist marker based on the molecule's complexity:
  [SPECIALIST: advanced]  — complex organics: multi-ring systems, drug-like
                            molecules, or more than 12 heavy atoms.
  [SPECIALIST: standard]  — small or simple molecules (12 heavy atoms or fewer).

Each worker task must, for its molecule:
1. Convert the molecule name to a SMILES string.
2. Generate a 3D coordinate file from the SMILES.
3. Run ASE geometry optimization with the mace_mp calculator (medium model),
   fmax=0.005.
4. Report the final optimized energy (in eV).
"""

SCREEN_DEFAULT_MOLECULES = (
    "aspirin, water, caffeine, methane, ibuprofen, ammonia"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_condition(args: argparse.Namespace, mode, skip_types: list[str]) -> str:
    if mode.value == "baseline":
        return "baseline"
    parts = []
    if getattr(args, "no_plan_extraction", False):
        parts.append("no_plan")
    if getattr(args, "no_divergence_guard", False):
        parts.append("no_diverg_guard")
    if getattr(args, "naive_prefetch", False):
        parts.append("naive_prefetch")
    for t in skip_types:
        parts.append(f"no_{t.replace('_', '')}")
    if getattr(args, "predictor", "") in ("plan_only", "transition_only", "oracle"):
        parts.append(f"pred_{args.predictor}")
    if not parts:
        return mode.value if mode.value in ("observe_only", "simulated") else "full_system"
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Executor builder
# ---------------------------------------------------------------------------

def _megammap_settings(args) -> dict:
    """
    Validate the MegaMmap/Hermes environment for --megammap-stage and return
    {binary, interceptor, hermes_conf}.  Fails hard when anything is missing:
    a silent fallback would produce data labelled "megammap" that measured
    nothing of the sort.
    """
    binary = os.environ.get(
        "MEGA_MODEL_PRELOAD",
        "/storage/project/r-ag117-0/shared/agent_hpc/mega_mmap_integration/"
        "megammap_tests/build/bin/mm_model_preload",
    )
    interceptor = os.environ.get("HERMES_INTERCEPTOR", "")
    hermes_conf = os.environ.get("HERMES_CONF", "")
    problems = []
    if not Path(binary).exists():
        problems.append(f"mm_model_preload not found: {binary!r} "
                        "(set MEGA_MODEL_PRELOAD)")
    if not interceptor or not Path(interceptor).exists():
        problems.append(f"libhermes_posix.so not found: {interceptor!r} "
                        "(set HERMES_INTERCEPTOR)")
    if problems:
        for p in problems:
            print(f"[megammap] ERROR: {p}")
        print("[megammap] --megammap-stage requires the Hermes/MegaMmap stack "
              "(source ~/scratch/mega_stack/mega_env.sh) and a running daemon "
              "(hrun_start_runtime). Aborting rather than faking the condition.")
        sys.exit(2)
    return {"binary": binary, "interceptor": interceptor,
            "hermes_conf": hermes_conf}


def _build_executor(mode: RuntimeMode, mace_device: str, orchestrator=None,
                    model_paths=None, megammap: dict | None = None,
                    megammap_window: str = "4g", megammap_tx: str = "seq"):
    """Return the appropriate PrefetchExecutor for the runtime mode."""
    if mode != RuntimeMode.REAL:
        return SimulatedPrefetchExecutor()

    executors = {}

    try:
        from runtime.prefetch.mace_prefetch import MacePrefetchExecutor
        executors["mace_model"] = MacePrefetchExecutor(device=mace_device)
        print(f"[runtime] Real executor: MacePrefetchExecutor(device={mace_device!r})")
    except Exception as exc:
        print(f"[runtime] WARNING: Could not build MacePrefetchExecutor ({exc}); "
              "mace_model will use simulated executor.")

    if orchestrator is not None:
        try:
            from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
            executors["vllm_model"] = ModelPrefetchExecutor(orchestrator, probes=None)
            print("[runtime] Real executor: ModelPrefetchExecutor for vllm_model")
        except Exception as exc:
            print(f"[runtime] WARNING: Could not build ModelPrefetchExecutor ({exc}).")

    # Option A: warm-cache staging for the worker weights during planning.
    # Backend is either our page-cache read-ahead or, for the external-system
    # comparison, MegaMmap/Hermes (mm_model_preload into the Hermes pool).
    if model_paths and megammap:
        from runtime.prefetch.megammap_stage import MegaMmapStagingExecutor
        executors["model_cache"] = MegaMmapStagingExecutor(
            model_paths=model_paths,
            binary=megammap["binary"],
            window=megammap_window,
            tx_type=megammap_tx,
        )
        print(f"[runtime] Real executor: MegaMmapStagingExecutor "
              f"(tx={megammap_tx}, window={megammap_window}, "
              f"models={list(model_paths)})")
    elif model_paths:
        try:
            from runtime.prefetch.model_cache_prefetch import ModelCacheStagingExecutor
            executors["model_cache"] = ModelCacheStagingExecutor(model_paths=model_paths)
            print(f"[runtime] Real executor: ModelCacheStagingExecutor "
                  f"(models={list(model_paths)})")
        except Exception as exc:
            print(f"[runtime] WARNING: Could not build ModelCacheStagingExecutor ({exc}).")

    if executors:
        from runtime.prefetch.data_prefetch import CompositeExecutor
        return CompositeExecutor(executors=executors, default=SimulatedPrefetchExecutor())

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

    # Toggleable calculator pin (screen workload defaults it ON via
    # run_eval_q1_q4).  Applied in ChemGraph's ASEInputSchema validator —
    # constrains EXECUTION only; the agent's decision-making is untouched and
    # divergences from failed unpinned choices remain observable.
    if getattr(args, "pin_calculator", ""):
        os.environ["CHEMGRAPH_PIN_CALCULATOR"] = args.pin_calculator
        print(f"[chemgraph] run_ase calculator pinned to {args.pin_calculator!r}")
    else:
        os.environ.pop("CHEMGRAPH_PIN_CALCULATOR", None)

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
    skip_types = [s.strip() for s in getattr(args, "skip_resource_types", "").split(",") if s.strip()]
    condition = getattr(args, "condition", None) or _infer_condition(args, mode, skip_types)

    # ------------------------------------------------------------------
    # Model orchestration (swap mode only)
    # ------------------------------------------------------------------
    orchestrator = None
    vllm_worker_model = ""
    vllm_aggregator_model = ""  # Option D: orchestrator key of co-resident aggregator
    model_paths = {}          # model key -> snapshot dir, for page-cache staging
    stage_worker_cache = False
    megammap = None
    if getattr(args, "megammap_stage", False):
        if getattr(args, "no_cache_stage", False):
            print("[megammap] ERROR: --megammap-stage and --no-cache-stage "
                  "are mutually exclusive.")
            sys.exit(2)
        megammap = _megammap_settings(args)   # exits if the stack is missing
    specialist_models: dict = {}
    if getattr(args, "swap_models", False):
        try:
            from atomagents.runtime.model_orchestrator import ModelOrchestrator
            from experiments.model_configs import MODELS_CHEMGRAPH_SWAP
            orchestrator = ModelOrchestrator(MODELS_CHEMGRAPH_SWAP)
            vllm_worker_model = "qwen_72b_instruct"
            planner_model = "qwen_32b_vl"
            # Screen workload: per-molecule specialist routing.  Marker ->
            # orchestrator model key; the adapter swaps specialists per task.
            if getattr(args, "screen", False):
                specialist_models = {
                    "advanced": "qwen_72b_instruct",
                    "standard": "qwen_32b_standard",
                }
            # Option D: orchestrator key of the distinct co-resident aggregator
            # model.  Set only when --aggregator-model is requested; the adapter
            # start_model()s this key on the ensemble tool_start.  NOT pre-started
            # here — that is the whole point (it loads during the compute window).
            vllm_aggregator_model = (
                "qwen_32b_aggregator"
                if getattr(args, "aggregator_model", None)
                and "qwen_32b_aggregator" in MODELS_CHEMGRAPH_SWAP
                else ""
            )
            if megammap:
                # Worker server must read its weights through Hermes so the
                # MegaMmap-staged cache is actually consumed.
                from runtime.prefetch.megammap_stage import build_hermes_preload
                worker_env = {
                    "LD_PRELOAD": build_hermes_preload(megammap["interceptor"]),
                }
                if megammap["hermes_conf"]:
                    worker_env["HERMES_CONF"] = megammap["hermes_conf"]
                MODELS_CHEMGRAPH_SWAP[vllm_worker_model]["extra_env"] = worker_env
                print(f"[megammap] Worker vLLM will run with "
                      f"LD_PRELOAD={worker_env['LD_PRELOAD']}")
            # Snapshot dirs for staging (model_name is the HF snapshot path).
            model_paths = {
                k: v["model_name"] for k, v in MODELS_CHEMGRAPH_SWAP.items()
            }
            # Option A: enable page-cache staging unless the ablation skips it.
            stage_worker_cache = not getattr(args, "no_cache_stage", False)

            # Controlled cache state: evict the worker weights so every run starts
            # cold.  Otherwise a warm cache from the previous run makes both the
            # cold baseline and the staging benefit meaningless.
            if getattr(args, "evict_worker_cache", False):
                try:
                    from runtime.prefetch.model_cache_prefetch import evict_model_cache
                    evict_keys = ([vllm_worker_model] if not specialist_models
                                  else sorted(set(specialist_models.values())))
                    for key in evict_keys:
                        snap = model_paths.get(key, "")
                        if snap:
                            n, nbytes = evict_model_cache(snap)
                            print(f"[cluster] Evicted {key} cache: {n} shards, "
                                  f"{nbytes/1e9:.1f} GB dropped from page cache.")
                except Exception as exc:
                    print(f"[cluster] WARNING: cache eviction failed: {exc}")

            if mode == RuntimeMode.REAL or getattr(args, "no_start_models", False) is False:
                print(f"[cluster] Starting planner model ({planner_model}) — "
                      f"runtime will stage + prefetch worker during planning.")
                orchestrator.start_model_measured(planner_model, metrics=None)
        except Exception as exc:
            print(f"[cluster] WARNING: Could not start model orchestrator: {exc}")
            orchestrator = None
            vllm_worker_model = ""
            vllm_aggregator_model = ""

    cfg = RuntimeConfig(
        mode=mode,
        run_id=run_id,
        confidence_threshold=args.confidence,
        max_horizon=args.horizon,
        conservative_mode_steps=3,
        # -1 disables extraction entirely: it fires at _step==0 and the adapter
        # gate is `_step > horizon`, so horizon=0 would still allow it.
        plan_extraction_horizon=-1 if getattr(args, "no_plan_extraction", False) else 3,
        disable_divergence_cancellation=getattr(args, "no_divergence_guard", False),
        naive_prefetch=getattr(args, "naive_prefetch", False),
        skip_resource_types=skip_types,
        condition=condition,
        log_dir=str(log_dir),
        results_dir=str(results_dir),
        vllm_worker_model=vllm_worker_model,
        stage_worker_cache=stage_worker_cache,
        worker_model_path=model_paths.get(vllm_worker_model, ""),
        vllm_aggregator_model=vllm_aggregator_model,
        specialist_models=specialist_models,
        early_plan_conditioned_stage=getattr(args, "early_plan_stage", False),
        model_paths=model_paths,
    )

    # Share ChemGraph's WorkflowTracker file so runtime events and workflow
    # events (llm_call, tool_call) are interleaved in one JSONL for analysis.
    try:
        from chemgraph.instrumentation.workflow_tracker import tracker as _cg_tracker
        bus = EventBus(run_id=run_id, shared_file=_cg_tracker._file)
        trace_path = _cg_tracker.log_path
        print(f"[runtime] Sharing WorkflowTracker log: {trace_path}")
    except Exception as _e:
        print(f"[runtime] Could not share WorkflowTracker file ({_e}); using standalone log.")
        bus = EventBus(run_id=run_id, log_path=trace_path)

    executor = _build_executor(mode, args.mace_device, orchestrator=orchestrator,
                               model_paths=model_paths, megammap=megammap,
                               megammap_window=args.megammap_window,
                               megammap_tx=args.megammap_tx)
    scheduler = PrefetchScheduler(executor=executor, config=cfg, bus=bus)
    detector = DivergenceDetector(scheduler=scheduler, config=cfg, bus=bus)

    if args.predictor in ("learned", "plan_only", "transition_only"):
        try:
            from runtime.predictor.learned_predictor import LearnedPredictor
            signals = "full" if args.predictor == "learned" else args.predictor
            predictor = LearnedPredictor(signals=signals)
            print(f"[chemgraph] Predictor: LearnedPredictor(signals={signals!r})")
        except Exception as exc:
            print(f"[chemgraph] WARNING: LearnedPredictor failed ({exc}); using mock.")
            predictor = MockPredictor("chemgraph")
    elif args.predictor == "oracle":
        try:
            from runtime.predictor.oracle_predictor import OraclePredictor
            if not args.oracle_trace:
                print("[chemgraph] ERROR: --predictor oracle requires --oracle-trace")
                sys.exit(1)
            predictor = OraclePredictor(args.oracle_trace, workflow="chemgraph")
            print(f"[chemgraph] Predictor: OraclePredictor(trace={args.oracle_trace}, "
                  f"{predictor.trace_length} reference tool steps)")
        except Exception as exc:
            print(f"[chemgraph] WARNING: OraclePredictor failed ({exc}); using mock.")
            predictor = MockPredictor("chemgraph")
    else:
        predictor = MockPredictor("chemgraph")

    # Build the runtime callback (replaces ChemGraph's standard callback).
    # Also needed in BASELINE when swap mode is active: the adapter's
    # on_chain_start gate performs the on-demand (sequential) worker swap.
    # With mode=BASELINE all prediction/prefetch logic inside the callback
    # is disabled, so this adds no runtime behaviour beyond the swap.
    if getattr(args, "ensemble_dataset", None) and not args.task_prompt:
        lim = getattr(args, "ensemble_limit", 0)
        task_prompt = ENSEMBLE_TASK_PROMPT.format(
            dataset=args.ensemble_dataset,
            limit_arg=f", limit={lim}" if lim else "",
        )
    elif getattr(args, "screen", False) and not args.task_prompt:
        task_prompt = SCREEN_TASK_PROMPT.format(
            molecules=getattr(args, "molecules", "") or SCREEN_DEFAULT_MOLECULES,
        )
    else:
        task_prompt = args.task_prompt or DEFAULT_TASK_PROMPT
    callback = None
    if mode != RuntimeMode.BASELINE or orchestrator is not None:
        try:
            from runtime.adapters.chemgraph import make_runtime_callback
            callback = make_runtime_callback(
                predictor=predictor,
                scheduler=scheduler,
                config=cfg,
                bus=bus,
                task_description=task_prompt,
                orchestrator=orchestrator,
            )
            print("[runtime] RuntimeChemGraphCallback installed.")
        except Exception as exc:
            print(f"[runtime] WARNING: Could not create runtime callback ({exc}); "
                  "running without runtime instrumentation.")

    # ------------------------------------------------------------------
    # Build ChemGraph agent
    # ------------------------------------------------------------------
    model_name = args.model_name or os.environ.get("CHEMGRAPH_MODEL", "gpt-4o-mini")
    workflow_type = args.workflow_type
    cg_kwargs: dict = {"model_name": model_name, "workflow_type": workflow_type}
    # Screen workload: 6 worker tasks x ~8 graph supersteps blows through
    # LangGraph's default recursion_limit=50 (t01 died at molecule 6).
    if getattr(args, "screen", False):
        cg_kwargs["recursion_limit"] = 200
    if args.planner_model:
        cg_kwargs["planner_model_name"] = args.planner_model
        print(f"[chemgraph] Planner model: {args.planner_model!r}  "
              f"Worker model: {model_name!r}")
    if getattr(args, "aggregator_model", None):
        cg_kwargs["aggregator_model_name"] = args.aggregator_model
        # The aggregator LLM client must point at the SEPARATE aggregator server
        # (its own GPUs / port), not the worker's base_url.  llm_agent reads this
        # env when building the aggregator ChatOpenAI client.
        agg_url = args.aggregator_base_url or "http://localhost:8004/v1"
        os.environ["VLLM_AGGREGATOR_BASE_URL"] = agg_url
        print(f"[chemgraph] Aggregator model: {args.aggregator_model!r} @ {agg_url}  "
              f"(prefetched during the ensemble compute window; NOT pre-started)")

    if workflow_type == "multi_agent":
        # Augment the planner prompt so it names ChemGraph tool functions
        # explicitly — this lets the runtime extract a structured plan sequence
        # from the planner's output via plan_extractor.py.
        from chemgraph.prompt.multi_agent_prompt import planner_prompt as _base_prompt
        _tool_names = (
            "  molecule_name_to_smiles, smiles_to_coordinate_file, smiles_to_atomsdata,\n"
            "  file_to_atomsdata, run_ase, extract_output_json"
        )
        # Out-of-core ensemble: inject the directory-batch MACE tool into the
        # worker toolset and advertise it to the planner.
        if getattr(args, "ensemble_dataset", None):
            from experiments.ensemble_tools import make_ensemble_tool
            from chemgraph.tools.ase_tools import extract_output_json
            # Give the worker ONLY the batch tool (+ json extractor).  Leaving the
            # single-structure run_ase in the set makes the LLM default to the
            # familiar per-file pattern and hallucinate filenames (structure_1.xyz)
            # that aren't in the directory — so the ensemble/compute window never
            # forms.  Dropping it forces the directory-batch path.
            cg_kwargs["tools"] = [make_ensemble_tool(), extract_output_json]
            _tool_names = "  run_mace_ensemble, extract_output_json"
            print(f"[chemgraph] Injected run_mace_ensemble tool "
                  f"(dataset={args.ensemble_dataset}); run_ase removed to force batch path")
        _tool_addendum = (
            "\n\nIMPORTANT: In each task prompt, explicitly name which Python "
            "tool function the worker should call. The available tool names are:\n"
            f"{_tool_names}\n"
            "Always include the exact function name (e.g. 'call molecule_name_to_smiles "
            "to convert the molecule name to a SMILES string')."
        )
        cg_kwargs["planner_prompt"] = _base_prompt + _tool_addendum
    if args.base_url:
        cg_kwargs["base_url"] = args.base_url
        # Local vLLM servers don't check the API key; use a dummy value so
        # LangChain/OpenAI SDK doesn't prompt for one or raise a missing-key error.
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        cg_kwargs["api_key"] = api_key
    elif args.api_key:
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

    # Optional continuous CPU/GPU profiler (results/system_profile_<run_id>.csv).
    # Gives measured GPU idle time (Q1) and memory-occupancy overhead (Q4).
    profiler = None
    if getattr(args, "profile", False):
        try:
            from runtime.measurement.system_profiler import SystemProfiler
            profiler = SystemProfiler(run_id=run_id, results_dir=str(results_dir))
            profiler.start()
        except Exception as exc:
            print(f"[runtime] WARNING: SystemProfiler unavailable ({exc}).")
            profiler = None

    workflow_completed = False
    start_iso = datetime.now().astimezone().isoformat()
    t_start = __import__("time").perf_counter()
    try:
        result = asyncio.run(agent.run(task_prompt, config=langgraph_config))
        workflow_completed = True
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
        end_iso = datetime.now().astimezone().isoformat()
        if profiler is not None:
            try:
                profiler.stop()
            except Exception:
                pass
        bus.close()
        # Swap mode: stop any running model so the next run starts from a clean state.
        if orchestrator is not None:
            running = orchestrator.get_running_model()
            if running:
                try:
                    orchestrator.stop_model(running)
                except Exception as _e:
                    print(f"[cluster] Warning: could not stop {running}: {_e}")

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
        if getattr(args, "swap_models", False):
            workload_variant = "swap"
        elif getattr(args, "ensemble_dataset", None):
            workload_variant = "ensemble"
        else:
            workload_variant = "plain"
        out.update({
            "wall_time_s": elapsed,
            "workflow_completed": workflow_completed,
            "start_time": start_iso,
            "end_time": end_iso,
            "run_id": run_id,
            "mode": mode.value,
            "predictor": args.predictor,
            "workflow": "chemgraph_mace",
            "workload_variant": workload_variant,
            "model_name": model_name,
            "mace_device": args.mace_device,
            "condition": condition,
            "node": __import__("socket").gethostname(),
            "git_commit": _git_commit(),
            "gpu_name": _gpu_name(),
            "trace_path": str(trace_path),
            "system_profile_csv": getattr(profiler, "csv_path", None),
            "ablation": {
                "no_plan_extraction": getattr(args, "no_plan_extraction", False),
                "no_divergence_guard": getattr(args, "no_divergence_guard", False),
                "naive_prefetch": getattr(args, "naive_prefetch", False),
                "skip_resource_types": skip_types,
                "no_cache_stage": getattr(args, "no_cache_stage", False),
                "stage_worker_cache": stage_worker_cache,
                "evict_worker_cache": getattr(args, "evict_worker_cache", False),
                "cache_stage_backend": ("megammap" if megammap else
                                        "page_cache" if stage_worker_cache else "none"),
                "megammap_tx": args.megammap_tx if megammap else None,
                "megammap_window": args.megammap_window if megammap else None,
            },
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

def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_PROJECT_ROOT),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def _gpu_name() -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:
        return ""


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
        choices=["mock", "learned", "plan_only", "transition_only", "oracle"],
        default="mock",
        help="mock = rule-based; learned = plan+transition-table+mock-fallback; "
             "plan_only / transition_only = LearnedPredictor restricted to one "
             "signal (predictor-mode ablation); oracle = perfect hindsight from "
             "a reference trace (--oracle-trace)",
    )
    parser.add_argument(
        "--oracle-trace",
        default=None,
        dest="oracle_trace",
        help="Reference JSONL trace for --predictor oracle",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run the continuous CPU/GPU SystemProfiler during the workflow "
             "(writes results/system_profile_<run_id>.csv)",
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
        help="LLM model name for workers (default: gpt-4o-mini or CHEMGRAPH_MODEL env)",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Smaller LLM for the PlannerAgent in multi_agent mode (default: same as --model-name)",
    )
    parser.add_argument(
        "--aggregator-model",
        default=None,
        help="Distinct LLM for the AggregatorAgent (Option D). Served on its own "
             "GPUs (port 8004); the runtime prefetches it co-resident during the "
             "ensemble MACE compute window so it is hot at aggregation time.",
    )
    parser.add_argument(
        "--aggregator-base-url",
        default=None,
        help="Base URL for the aggregator vLLM server (default: http://localhost:8004/v1)",
    )
    parser.add_argument(
        "--workflow-type",
        default="single_agent",
        choices=["single_agent", "multi_agent", "python_relp", "graspa", "mock_agent"],
        help="ChemGraph workflow type (default: single_agent)",
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
    parser.add_argument("--no-plan-extraction", action="store_true", dest="no_plan_extraction",
        help="Disable LLM plan extraction (plan_extraction_horizon=0)")
    parser.add_argument("--no-divergence-guard", action="store_true", dest="no_divergence_guard",
        help="Track divergences but do not cancel pending prefetches")
    parser.add_argument("--naive-prefetch", action="store_true", dest="naive_prefetch",
        help="Prefetch on every prediction regardless of confidence")
    parser.add_argument("--skip-resource-types", default="", dest="skip_resource_types",
        help="Comma-separated resource types to never prefetch (e.g. mace_mp)")
    parser.add_argument("--condition", default=None,
        help="Override the auto-inferred ablation condition label in the summary JSON")
    parser.add_argument(
        "--hw-profile",
        default="standard",
        choices=["standard", "chemgraph_swap"],
        dest="hw_profile",
        help="Hardware profile: standard = separate GPU pools; "
             "chemgraph_swap = shared GPU pool, forced model swaps (default: standard)",
    )
    parser.add_argument(
        "--swap-models",
        action="store_true",
        dest="swap_models",
        help="Enable model swapping via ModelOrchestrator (requires --hw-profile chemgraph_swap)",
    )
    parser.add_argument(
        "--no-start-models",
        action="store_true",
        dest="no_start_models",
        help="Skip initial model server startup (assume servers already running)",
    )
    parser.add_argument(
        "--ensemble-dataset",
        default=None,
        dest="ensemble_dataset",
        help="Directory of structure files (CIF/XYZ). Injects the run_mace_ensemble "
             "tool and runs a directory-batch MACE job — a genuine multi-minute CPU "
             "compute window (GPU idle) for hiding a model prefetch/swap.",
    )
    parser.add_argument(
        "--ensemble-limit",
        type=int,
        default=0,
        dest="ensemble_limit",
        help="Process only the first N structures of --ensemble-dataset "
             "(0 = all). Controls the CPU compute-window length for the "
             "window-size sensitivity sweep.",
    )
    parser.add_argument(
        "--no-cache-stage",
        action="store_true",
        dest="no_cache_stage",
        help="Swap mode ablation: disable Option-A page-cache staging of the "
             "worker weights during planning (cold swap on the critical path)",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Screen workload: heterogeneous molecule batch with per-molecule "
             "specialist routing (advanced=72B, standard=32B; requires "
             "--swap-models --workflow-type multi_agent)",
    )
    parser.add_argument(
        "--molecules",
        default="",
        help="Screen workload: comma-separated molecule list (default: "
             "aspirin, water, caffeine, methane, ibuprofen, ammonia)",
    )
    parser.add_argument(
        "--early-plan-stage",
        action="store_true",
        dest="early_plan_stage",
        help="Screen workload: stage the page cache at plan_extracted time, "
             "choosing WHICH specialist from the plan (plan-conditioned early "
             "trigger). Off = legacy blind staging at first chain start.",
    )
    parser.add_argument(
        "--pin-calculator",
        default="",
        dest="pin_calculator",
        help="Pin run_ase to one calculator family (e.g. mace_mp) via "
             "CHEMGRAPH_PIN_CALCULATOR; empty = agent free choice (may pick "
             "TBLite, which fails ~90%% of the time).",
    )
    parser.add_argument(
        "--megammap-stage",
        action="store_true",
        dest="megammap_stage",
        help="External-system comparison: replace page-cache staging with "
             "MegaMmap/Hermes (mm_model_preload into the Hermes buffer pool "
             "during planning; worker vLLM launched with libhermes_posix.so "
             "LD_PRELOAD). Requires a running Hermes daemon and "
             "HERMES_INTERCEPTOR/MEGA_MODEL_PRELOAD in the environment.",
    )
    parser.add_argument(
        "--megammap-window",
        default=os.environ.get("MEGA_WINDOW", "4g"),
        dest="megammap_window",
        help="MegaMmap DRAM window per shard for --megammap-stage (e.g. 4g)",
    )
    parser.add_argument(
        "--megammap-tx",
        default="seq",
        choices=["seq", "rand"],
        dest="megammap_tx",
        help="MegaMmap transaction mode: seq = deterministic prefetch (known "
             "model), rand = no prefetch signal (unknown model)",
    )
    parser.add_argument(
        "--evict-worker-cache",
        action="store_true",
        dest="evict_worker_cache",
        help="Swap mode: posix_fadvise(DONTNEED) the worker weight shards before "
             "the run so every run starts from a cold page cache (fair comparison)",
    )

    args = parser.parse_args()

    if args.extended_task and not args.task_prompt:
        args.task_prompt = EXTENDED_TASK_PROMPT

    run_chemgraph_exp(args)


if __name__ == "__main__":
    main()
