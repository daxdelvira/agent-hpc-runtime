"""
atomagents_exp3.py — AtomAgents Exp3: 3-model forced-swap experiment.

Key differences from exp2:
  - Three models share the same GPU pool (only one resident at a time).
  - ModelRouter is activated via init_router(), so every LLM call triggers
    an automatic model swap if the required model is not currently running.
  - A `code_task` tool is registered on engineer_core; it routes to the
    text-72B specialist (port 8003) for LAMMPS script generation.
  - Task prompt elicits 1× plan_task + 2× code_task + 2× LAMMPS cycles;
    plan_task is also called internally by each computation_task_screw_dislocation.
  - LAMMPS_SLOWDOWN_S (default 600 via run_l40s.sh) creates 10-min compute
    windows for proactive model swap: evict current model, load next during LAMMPS.

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
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "workloads" / "AtomAgents"))

# ---------------------------------------------------------------------------
# AutoGen's disk cache directory must exist before ANY autogen import.
#
# autogen/agentchat/contrib/capabilities/transforms.py evaluates `Cache.disk()`
# as a DEFAULT ARGUMENT at class-definition time, i.e. during import, and that
# opens a diskcache SQLite DB at ./.cache/<seed> relative to CWD. Two separate
# outages came from this:
#   1. CWD is workloads/AtomAgents on project NFS, and SQLite locking over NFS
#      is broken -> `sqlite3.OperationalError: locking protocol` killed 12/12
#      trials once two campaigns ran concurrently.
#   2. The fix (symlink .cache -> /tmp/autogen_cache) put the SYMLINK on NFS
#      where every node sees it, while the TARGET is node-local. Any node that
#      had not created /tmp/autogen_cache saw a dangling link and died with
#      `FileNotFoundError: ./.cache/42` -> 10 more trials lost.
# Doing it here, in the entry point, covers every launch path (campaign script,
# manual driver run, interactive debugging) instead of only the one shell
# script that happened to have the mkdir.
os.makedirs("/tmp/autogen_cache", exist_ok=True)

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
#   →  computation_task_screw_dislocation (LAMMPS 1, 10-min window;
#       proactive prefetch loads 32B during LAMMPS, then calls plan_task internally)
#   →  code_task (72B-text)
#   →  computation_task_screw_dislocation (LAMMPS 2, 10-min window; same pattern)
#   →  report
#
# Model swap sequence (all models share one GPU pool):
#   start 72B-VL → swap to 32B (plan1) → swap to 72B-text (code1)
#   → swap to 72B-VL → [LAMMPS1: proactive: stop 72B-VL, load 32B] → plan_task (32B hot)
#   → swap to 72B-text (code2) → swap to 72B-VL
#   → [LAMMPS2: proactive: stop 72B-VL, load 32B] → plan_task (32B hot) → report
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
3. Call computation_task_screw_dislocation for W_Zhou04.eam.alloy with \
analysis_query="Classify the screw dislocation core (polarized or unpolarized?) and \
identify any adjustments needed before running w_eam4.fs." \
This function runs LAMMPS and then automatically performs structural analysis — \
do NOT call plan_task again after this step.
4. Call code_task to get the LAMMPS input parameters for potential 2 (w_eam4.fs) reviewed \
by the code specialist, incorporating any structural insights from step 3.
5. Call computation_task_screw_dislocation for w_eam4.fs with \
analysis_query="Classify the screw dislocation core (polarized or unpolarized?) and \
compare the energetics with the W_Zhou04.eam.alloy result." \
This function runs LAMMPS and then automatically performs structural analysis — \
do NOT call plan_task again after this step.
6. Report whether each potential gives a polarized or unpolarized core and summarize the \
structural and energetic comparison between the two potentials.

Critical rules:
- Call plan_task ONLY in step 1. computation_task_screw_dislocation handles its own \
  structural analysis via the analysis_query argument — do NOT call plan_task after steps 3 or 5.
- Always call code_task before each computation_task_screw_dislocation (steps 2 and 4).
- Do NOT compute surface energy, elastic constants, stacking fault energy, NEB barriers, \
or any other property.
- Do NOT invent tool names. Only call tools that are registered with you.
"""


# ---------------------------------------------------------------------------
# Prompt variants — two INDEPENDENT changes, so their effects are separable.
#
# WHY THESE EXIST.  Measured on the 26 collected exp3 trials (2026-08-02):
# `plan_extracted` fires in only 12/32 traces, always at step 4 of ~8, and the
# plan it recovers names `analyze_screw_core` and `computation_task_surface_energy`
# — tools that are executed ZERO times — while missing `code_task` (77 calls)
# and `plan_task` (52).  The planner is not malfunctioning: the engineer composes
# the plan_task query from the scientific goal ALONE and drops the scope rules,
# so the planner legitimately plans a full 7-step workflow (surface energy, save
# data, analyze plots) while the executor runs a narrow 3-tool script.  The plan
# is coherent and simply never followed, so insight (1) — "these workflows have a
# central planner" — contributes nothing on this workload.
#
#   pinned    exactly DEFAULT_TASK_PROMPT_EXP3.  The default, byte-identical to
#             what all 28 collected trials ran.  Never change this.
#   aligned   CHANGE 1 — the executor must forward the scope rules into
#             plan_task.  Nothing else differs from `pinned`, so any movement is
#             attributable to planner/executor alignment alone.
#   unpinned  CHANGE 1 + CHANGE 2 — additionally drops the numbered step
#             enumeration, so tool order and potential order become the agent's
#             choice.  Builds ON `aligned`: un-pinning without alignment would
#             leave the plan unfollowed AND the executor unbounded, confounding
#             both effects.
#
# The scope rules are deliberately KEPT in every variant.  They bound runtime
# (no surface energy / NEB) so trials stay comparable to what is collected;
# removing them is a different experiment.  Note the retained "at most once"
# guard on plan_task in `unpinned` — without an ordering rule the agent can loop
# on plan_task (a 2,265-step runaway is on record in logs/workflow_traces/).
#
# WARNING for anyone tuning this: if `unpinned` raises the failure rate, DO NOT
# restore step enumeration to recover reliability.  That silently destroys both
# the entropy this measures and the plan signal insight (1) depends on.  Escalate
# the backing model instead, and say so in the write-up.
# ---------------------------------------------------------------------------
_EXP3_PLAN_ALIGNMENT = """\
The query you pass to plan_task MUST also state the scope constraints from this \
message — which tools are available, and that surface energy, elastic constants, \
stacking fault energy and NEB barriers are out of scope. The planner runs in a \
separate sub-chat and CANNOT see this message, so a plan built without those \
constraints will propose steps you are not permitted to execute."""

# CHANGE 1 only: identical to DEFAULT_TASK_PROMPT_EXP3 except that step 1 tells
# the executor to forward the scope rules into plan_task.
TASK_PROMPT_EXP3_ALIGNED = DEFAULT_TASK_PROMPT_EXP3.replace(
    "1. Call plan_task to develop a detailed step-by-step simulation plan for both potentials.",
    "1. Call plan_task to develop a detailed step-by-step simulation plan for both "
    "potentials. " + _EXP3_PLAN_ALIGNMENT,
)

# CHANGE 1 + CHANGE 2: scope preserved, step enumeration removed.
TASK_PROMPT_EXP3_UNPINNED = """\
Compare the core structure of the 1/2<111> screw dislocation in W using \
"W_Zhou04.eam.alloy" and "w_eam4.fs" EAM potentials.
The dislocation line is along [-1,1,1], glide direction [1,-1,2], glide plane normal [1,1,0].

Begin by calling plan_task once to produce your own simulation plan, then carry that \
plan out and report the comparison. {alignment}

Scope constraints:
- Call plan_task at most once.
- Do NOT compute surface energy, elastic constants, stacking fault energy, NEB barriers, \
or any other property.
- Do NOT invent tool names. Only call tools that are registered with you.
- When the comparison is reported, TERMINATE.
""".format(alignment=_EXP3_PLAN_ALIGNMENT)

TASK_PROMPT_VARIANTS_EXP3 = {
    "pinned": DEFAULT_TASK_PROMPT_EXP3,
    "aligned": TASK_PROMPT_EXP3_ALIGNED,
    "unpinned": TASK_PROMPT_EXP3_UNPINNED,
}

# Names baked into every prompt above, in the order the workload offers them.
_PROMPT_POTENTIALS = ("W_Zhou04.eam.alloy", "w_eam4.fs")


def retarget_prompt_potentials(prompt: str, offered: tuple) -> str:
    """Rewrite hardcoded potential filenames to whatever `--potentials` offers.

    WHY THIS IS NECESSARY, not cosmetic. Every prompt variant names
    "w_eam4.fs" literally. When the driver offers `w_eam4_big.fs` instead, the
    agent dutifully asks for a potential that is NOT on offer, the tool rejects
    it, and the argument-recovery path runs on EVERY call. Recovery is a
    best-effort heuristic over free text; making it the normal path rather than
    the exceptional one is what let the wrong potential through on 2026-08-03
    (the second simulation silently re-ran W_Zhou04). Naming the real file in
    the prompt keeps recovery exceptional, which is the only regime it is
    designed for.

    This does NOT narrow the agent's choice — it still picks which potential to
    run when, and in which order. It only ensures the names it is shown are the
    names that exist, which is a bug fix, not a hint. (See the standing rule
    above: reliability may not be bought with prompt structure.)

    A no-op when the offered set is the default, so `pinned` stays
    byte-identical to what the 28 already-collected trials ran.
    """
    if tuple(offered) == _PROMPT_POTENTIALS:
        return prompt
    if len(offered) != len(_PROMPT_POTENTIALS):
        print(f"[runtime] WARNING: --potentials has {len(offered)} entries but the "
              f"prompt names {len(_PROMPT_POTENTIALS)}; leaving prompt text alone. "
              f"The agent will be shown names that may not be on offer.", flush=True)
        return prompt
    # Single simultaneous pass: chained str.replace could re-substitute a name
    # that a previous replacement just introduced.
    mapping = dict(zip(_PROMPT_POTENTIALS, offered))
    pattern = re.compile("|".join(re.escape(k) for k in _PROMPT_POTENTIALS))
    return pattern.sub(lambda m: mapping[m.group(0)], prompt)


# ---------------------------------------------------------------------------
# Build prefetch executor
# ---------------------------------------------------------------------------

def _megammap_settings(args) -> dict:
    """Validate the MegaMmap/Hermes environment and return its paths.

    Ported verbatim in spirit from experiments/chemgraph_exp.py:179, including
    the decision to ABORT rather than degrade: a silent fallback to page-cache
    staging would produce trials labelled "megammap" that measured nothing of
    the sort, and the label is what the comparison rests on.
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


def _build_executor(mode: RuntimeMode, orchestrator, metrics=None,
                    megammap: dict | None = None,
                    megammap_window: str = "4g", megammap_tx: str = "seq",
                    model_paths: dict | None = None):
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
        # model_cache is the host-I/O half of a model load, emitted by the
        # adapter alongside each vllm_model (see adapters/atomagents.py
        # _pair_model_cache). MegaMmap replaces the page-cache warm here; that
        # substitution IS the external-system comparison.
        if megammap:
            from runtime.prefetch.megammap_stage import MegaMmapStagingExecutor
            extra_env = {}
            if megammap.get("hermes_conf"):
                extra_env["HERMES_CONF"] = megammap["hermes_conf"]
            executors["model_cache"] = MegaMmapStagingExecutor(
                model_paths=model_paths,
                binary=megammap["binary"],
                window=megammap_window,
                tx_type=megammap_tx,
                extra_env=extra_env or None,
            )
            print(f"[runtime] Real executor: MegaMmapStagingExecutor for "
                  f"model_cache (window={megammap_window}, tx={megammap_tx})")
        else:
            from runtime.prefetch.model_cache_prefetch import ModelCacheStagingExecutor
            executors["model_cache"] = ModelCacheStagingExecutor(
                model_paths=model_paths)
            print("[runtime] Real executor: ModelCacheStagingExecutor for model_cache")
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

    # Must be set BEFORE AtomAgents tools are imported: orchestration_tools reads
    # it at import time to build both the LLM-visible tool description and the
    # runtime validator from one source. Setting it later would leave the agent
    # being told about one potential set while another is accepted.
    if args.potentials:
        os.environ["ATOMAGENTS_POTENTIALS"] = args.potentials

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

    # Validate the Hermes/MegaMmap stack up front, before any model is loaded:
    # a missing interceptor discovered 700 s into a boot wastes the hold, and
    # _megammap_settings() exits rather than degrading to page-cache staging.
    _megammap = _megammap_settings(args) if getattr(args, "megammap_stage", False) else None

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
            # ----------------------------------------------------------
            # Sleep/wake arm (--sleep-wake): park engines instead of killing
            # them. vLLM 0.17.x gates /sleep, /wake_up and /is_sleeping behind
            # VLLM_SERVER_DEV_MODE=1, and the engine must have been LAUNCHED
            # with --enable-sleep-mode, so both are injected here — before any
            # engine starts, since neither can be turned on after the fact.
            #
            # WHY LEVEL 2 IS THE DEFAULT HERE (chemgraph uses level 1).
            # Level 1 offloads weights to host RAM: measured at 108-128 GiB per
            # slept engine (Stage-0 gate (b)), which died at k=3 on a 2 TB node.
            # exp3 carries THREE large models, so level 1 would need roughly
            # 355 GB against a 256G hold cgroup — it cannot fit, and the failure
            # mode is the node dying rather than a clean error. Level 2 discards
            # the weights (VRAM still freed, ~0.3 GiB standing host RAM
            # measured) and re-reads them on wake, while still keeping the
            # process, CUDA context, captured graphs and profiling results —
            # which is where the bulk of a cold boot actually goes.
            #
            # Caveat to carry into the write-up: level 2's wake cost depends on
            # page-cache residency, and fadvise on Lustre was measured leaving
            # 56.2% of shards resident, so a "cold" L2 wake here is not fully
            # cold. Report it as contingent, not free.
            # ----------------------------------------------------------
            if args.sleep_wake:
                for _mc in MODELS.values():
                    _extra = list(_mc.get("extra_args") or [])
                    if "--enable-sleep-mode" not in _extra:
                        _extra.append("--enable-sleep-mode")
                    _mc["extra_args"] = _extra
                    _env = dict(_mc.get("extra_env") or {})
                    _env["VLLM_SERVER_DEV_MODE"] = "1"
                    # Sleep mode is incompatible with expandable_segments: the
                    # allocator cannot release segments it has expanded, so
                    # /sleep frees nothing and the next engine still OOMs.
                    _alloc = _env.get("PYTORCH_CUDA_ALLOC_CONF", "")
                    if "expandable_segments" in _alloc:
                        _env.pop("PYTORCH_CUDA_ALLOC_CONF")
                        print("[cluster] sleep-wake: dropped "
                              "PYTORCH_CUDA_ALLOC_CONF=expandable_segments "
                              "(incompatible with vLLM sleep mode)")
                    _mc["extra_env"] = _env
                print(f"[cluster] Sleep/wake swaps ENABLED (level "
                      f"{args.sleep_level}) for {len(MODELS)} engines.")

            # Every engine must read its weights THROUGH Hermes, or the staged
            # cache is never consumed and the arm measures page-cache staging
            # with extra overhead bolted on. ModelOrchestrator applies
            # extra_env wholesale at launch (model_orchestrator.py:255), so
            # setting it here is sufficient -- but it must happen before the
            # orchestrator is constructed below.
            if _megammap and MODELS:
                from runtime.prefetch.megammap_stage import build_hermes_preload
                _pre = build_hermes_preload(_megammap["interceptor"])
                for _mc in MODELS.values():
                    _env = dict(_mc.get("extra_env") or {})
                    _env["LD_PRELOAD"] = _pre
                    if _megammap.get("hermes_conf"):
                        _env["HERMES_CONF"] = _megammap["hermes_conf"]
                    _mc["extra_env"] = _env
                print(f"[megammap] {len(MODELS)} engines will launch with "
                      f"LD_PRELOAD={_pre}")

            # Turn on the host-I/O half of a model load, and tell the adapter
            # where each model's shards live. Both are read by
            # AtomAgentsRuntimeAdapter._pair_model_cache; without them it emits
            # nothing and a staging executor has no work.
            #
            # OFF BY DEFAULT ON PURPOSE. Every exp3 trial collected before this
            # change ran without model_cache staging, so switching it on for the
            # existing arms would make new runs incomparable with the recorded
            # baseline (n=6) and full_system (n=9). It is enabled only when an
            # arm explicitly asks for staging.
            if MODELS and (_megammap or getattr(args, "stage_model_cache", False)):
                cfg.model_paths = {k: v["model_name"] for k, v in MODELS.items()}
                cfg.stage_worker_cache = True
                print(f"[runtime] model_cache staging ENABLED for "
                      f"{len(cfg.model_paths)} models")

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
            router = init_router(
                orchestrator, MODELS,
                sleep_level=(args.sleep_level if args.sleep_wake else 0),
            )
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

    # Hand the metrics logger to the router so agent-blocking model swaps are
    # recorded as model_swap_wait:<name> phases (the exposed-stall number the
    # Q2 breakdown needs; before 2026-07-09 swap time was only in stdout prose).
    if orchestrator is not None:
        try:
            from atomagents.runtime.model_router import get_router
            _r = get_router()
            if _r is not None:
                _r.set_metrics(ml)
        except Exception as e:
            print(f"[cluster] WARNING: could not attach metrics to router: {e}")

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
    # Snapshot dirs keyed by the name the adapter puts on a ResourceSpec, so a
    # staging executor can find the shards for a model it was handed by name.
    _model_paths = ({k: v["model_name"] for k, v in MODELS.items()}
                    if MODELS else {})
    executor = _build_executor(
        mode, orchestrator, ml,
        megammap=_megammap,
        megammap_window=getattr(args, "megammap_window", "4g"),
        megammap_tx=getattr(args, "megammap_tx", "seq"),
        model_paths=_model_paths,
    )
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
        port_72b=8007,
        port_32b=8012,
    )
    profiler.start()

    # ------------------------------------------------------------------
    # Run the experiment
    # ------------------------------------------------------------------
    # An explicit --task-prompt always wins; otherwise the variant selects one.
    # Default is "pinned", i.e. byte-identical to every trial collected so far.
    task_prompt = args.task_prompt or TASK_PROMPT_VARIANTS_EXP3[args.prompt_variant]
    # Show the agent the potential names that are actually on offer. No-op
    # unless --potentials overrides the default pair.
    from atomagents.tools.orchestration_tools import offered_potentials
    _offered = offered_potentials()
    task_prompt = retarget_prompt_potentials(task_prompt, _offered)
    print(f"[runtime] Potentials on offer : {list(_offered)}")
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
        # Faceting key: results MUST NOT be pooled across prompt variants — they
        # are different workloads, not different configs of one workload.
        out["prompt_variant"] = ("custom" if args.task_prompt
                                 else args.prompt_variant)
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
        "--sleep-wake",
        action="store_true",
        dest="sleep_wake",
        help="Swap-mechanism arm: boot each engine once with --enable-sleep-mode "
             "(+ VLLM_SERVER_DEV_MODE=1) and park the outgoing engine with "
             "/sleep instead of killing it. Measured on this workload, a cold "
             "72B tp=4 boot costs 990-1315 s against a 0.8-2.1 s wake.",
    )
    parser.add_argument(
        "--sleep-level",
        type=int,
        default=2,
        choices=[1, 2],
        dest="sleep_level",
        help="vLLM sleep level for --sleep-wake. 1 offloads weights to host RAM "
             "(108-128 GiB per engine; three engines do NOT fit a 256G hold). "
             "2 discards weights, keeps the process/CUDA context/graphs, and "
             "holds ~0.3 GiB. Default 2 because exp3 carries three models.",
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
    parser.add_argument(
        "--potentials",
        default=None,
        dest="potentials",
        help="Comma-separated EAM potential filenames the agent may choose from "
             "(sets ATOMAGENTS_POTENTIALS). Default keeps the workload's own pair "
             "(W_Zhou04.eam.alloy, w_eam4.fs). Use "
             "'W_Zhou04.eam.alloy,w_eam4_big.fs' to make the data side cost real "
             "time: w_eam4_big.fs is 3.32 GB and ~129 s to activate, of which "
             ">=123.5 s is parse/spline construction rather than I/O. NOTE it is a "
             "synthetic LOAD GENERATOR (non-physical energies) and must be "
             "described as such in any write-up.",
    )
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(TASK_PROMPT_VARIANTS_EXP3),
        default="pinned",
        dest="prompt_variant",
        help="pinned = the collected-trials prompt (default, unchanged); "
             "aligned = CHANGE 1, executor forwards scope rules into plan_task so "
             "the plan is one it can actually execute; "
             "unpinned = CHANGE 1 + CHANGE 2, additionally drops the numbered step "
             "enumeration so tool and potential order become agent choices. "
             "Ignored if --task-prompt is given.",
    )
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

    # External-system comparison: replace host-side weight staging with
    # MegaMmap/Hermes. Mirrors experiments/chemgraph_exp.py. Requires the
    # Hermes stack on PATH and a running daemon; the eval driver starts and
    # stops one per trial for arms flagged needs_hermes.
    parser.add_argument(
        "--megammap-stage", action="store_true", dest="megammap_stage",
        help="stage model weights through MegaMmap into the Hermes buffer pool "
             "instead of warming the OS page cache; the worker vLLM servers are "
             "launched with LD_PRELOAD=libhermes_posix.so so their reads hit it")
    parser.add_argument(
        "--stage-model-cache", action="store_true", dest="stage_model_cache",
        help="warm model weight shards into the OS page cache during compute "
             "windows. The page-cache control for --megammap-stage; off by "
             "default so previously collected exp3 arms stay comparable.")
    parser.add_argument(
        "--megammap-window", default="4g", dest="megammap_window",
        help="MegaMmap bounded DRAM window per staged model")
    parser.add_argument(
        "--megammap-tx", default="seq", choices=("seq", "rand"),
        dest="megammap_tx",
        help="page order for the staging transaction. 'rand' is the "
             "no-prefetch-signal control: it models staging with no idea which "
             "model comes next, and is what isolates window from prediction.")

    args = parser.parse_args()
    run_exp3(args)


if __name__ == "__main__":
    main()
