"""
run_eval_q1_q4.py — unified evaluation driver for the paper's Q1–Q4 experiments.

Runs (workload × configuration × trial) sweeps, saving every artifact needed by
scripts/parse_eval_traces.py into a resumable, never-overwritten result tree:

    results/eval_q1_q4/
      commands.log                     — every command launched, timestamped
      runs/<workload>/<config>/
        t03__20260707-181501__27b7b0f/ — one dir per trial (never reused)
          meta.json                    — command, env, node, GPUs, git, timestamps, status
          stdout.log                   — full run output
          summary.json                 — copied workflow summary
          trace.jsonl                  — copied runtime trace
          system_profile.csv           — copied CPU/GPU profiler samples (if any)

Resume: completed trials are counted per (workload, config); re-running the same
command only launches the missing remainder.  Failed/timeout trials are kept on
disk (marked in meta.json) but do not count toward N.

Workloads
---------
chemgraph_swap   ChemGraph multi-agent split-model workflow: 32B-VL planner →
                 72B-Instruct worker on a shared 4-GPU pool (forced swap).
                 Runtime hides the swap via plan-triggered worker prefetch +
                 page-cache staging.  ~10–15 min/trial on 4× Blackwell.
atomagents_exp2  AtomAgents screw-dislocation (2-model, simultaneous residency).
atomagents_exp3  AtomAgents 3-model forced-swap (LAMMPS compute windows).
                 ~1–3 h/trial — budget accordingly.
deepdrivemd      NOT integrated yet (no runtime adapter/entrypoint).  Listed so
                 the result tree and parser are ready; selecting it errors out.

Usage
-----
    # Everything Q1–Q4 needs on the ChemGraph swap workload, 10 trials each,
    # round-robin so partial sweeps stay balanced:
    python experiments/run_eval_q1_q4.py --workload chemgraph_swap --trials 10

    # Specific configs only
    python experiments/run_eval_q1_q4.py --workload chemgraph_swap \
        --configs baseline,full_system,naive_prefetch --trials 5

    # Show completion status of the result tree
    python experiments/run_eval_q1_q4.py --list

    # Print the commands without running
    python experiments/run_eval_q1_q4.py --workload chemgraph_swap --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
EVAL_ROOT = PROJECT_ROOT / "results" / "eval_q1_q4"
RUNS_ROOT = EVAL_ROOT / "runs"

# Default predictor horizon for --lookahead.  Imported from the predictor so
# the two cannot drift; the literal is only a fallback for the case where the
# runtime package is not importable from the driver's sys.path.
try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from runtime.predictor.learned_predictor import _DEFAULT_LOOKAHEAD
except Exception:                                   # pragma: no cover
    _DEFAULT_LOOKAHEAD = 2

CG_PYTHON = "/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python"
ATOMS_PYTHON = "/storage/project/r-ag117-0/shared/agent_hpc/envs/atoms/bin/python"
_AA_NVIDIA = ("/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/"
              "lib/python3.11/site-packages/nvidia")
_TORCH_LIB = ("/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/"
              "lib/python3.10/site-packages/torch/lib")

# Fallback oracle reference trace (validated full_system swap run, 2026-07-06).
_FALLBACK_ORACLE_TRACE = str(
    PROJECT_ROOT / "logs/workflow_traces/"
    "chemgraph_trace_20260706_142854_d61a4951-6e15-4275-8869-10547ca4b6bd.jsonl"
)

# MegaMmap/Hermes stack (external-system comparison, config "megammap_stage").
# Paths from ~/scratch/mega_stack/mega_env.sh; every binary carries an RPATH to
# mega_stack/lib, so only PATH (mpirun) and the HERMES_* vars are injected.
MEGA_STACK = os.path.expanduser("~/scratch/mega_stack")
MEGA_MPI_BIN = ("/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/"
                "gcc-12.3.0/openmpi-4.1.5-ahgvv7r3aju6cty4nlmcd5hihsckie7j/bin")
MEGA_MODEL_PRELOAD = ("/storage/project/r-ag117-0/shared/agent_hpc/"
                      "mega_mmap_integration/megammap_tests/build/bin/mm_model_preload")
HERMES_INTERCEPTOR = f"{MEGA_STACK}/lib/libhermes_posix.so"
# Agentic daemon config (50 GB RAM + 200 GB NVMe tier). NOTE: sized for the
# bigmem holds — on a 48 GB-cgroup job override with HERMES_CONF_EVAL.
HERMES_CONF_EVAL = os.environ.get(
    "HERMES_CONF_EVAL",
    os.path.expanduser("~/scratch/mega_src/hermes/config/hermes_agentic.yaml"),
)


# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------

# Flags shared by every chemgraph_swap run.  Mirrors the validated recipe in
# experiments/run_cg_swap_ablations.sh + run_cg_ensemble_validate.sh:
#   - planner = Qwen2.5-VL-32B on port 8002, worker = Qwen2.5-72B on port 8001
#   - --evict-worker-cache so every trial starts from a cold Lustre page cache
#   - --profile so GPU idle time / memory overhead are measured per run
CHEMGRAPH_SWAP_BASE = [
    "--workflow-type", "multi_agent",
    "--model-name", "Qwen/Qwen2.5-72B-Instruct",
    "--planner-model", "Qwen/Qwen2.5-VL-32B-Instruct",
    "--base-url", "http://localhost:8001/v1",
    "--mace-device", "cpu",
    "--hw-profile", "chemgraph_swap",
    "--swap-models",
    "--evict-worker-cache",
    "--profile",
]

# config name -> dict(mode, predictor, flags, needs_oracle_trace)
# Names follow the paper's configuration list; the runner's --condition label is
# set to the config name so summaries/traces are self-describing.
CHEMGRAPH_SWAP_CONFIGS: dict[str, dict] = {
    "baseline": {
        "mode": "baseline", "predictor": "mock", "flags": [],
        "desc": "vanilla workflow; on-demand sequential model swap",
    },
    "observe_only": {
        "mode": "observe_only", "predictor": "learned", "flags": [],
        "desc": "predictor enabled, no prefetch I/O",
    },
    "simulated": {
        "mode": "simulated", "predictor": "learned", "flags": [],
        "desc": "predictor + simulated prefetch decisions, no real I/O",
    },
    "full_system": {
        "mode": "real", "predictor": "learned", "flags": [],
        "desc": "full SystemName: plan-triggered worker prefetch + cache staging"
                " + confidence/divergence gates",
    },
    "sleep_wake": {
        "mode": "real", "predictor": "learned", "flags": ["--sleep-wake"],
        # Behavior-changing arm (different vLLM launch args + swap mechanism):
        # opt-in only — excluded from the implicit "all configs" set so
        # campaigns run without --configs never pick it up by accident.
        "explicit_only": True,
        "desc": "full system with vLLM sleep-mode swaps: engines boot once "
                "(--enable-sleep-mode); swaps sleep the planner (weights -> "
                "CPU RAM, VRAM freed) and wake the worker (H2D copy) instead "
                "of kill + cold boot — attacks the no_window bring-up stall",
    },
    "no_cache_stage": {
        "mode": "real", "predictor": "learned", "flags": ["--no-cache-stage"],
        "desc": "full system minus page-cache staging (cold swap on critical path)",
    },
    "naive_prefetch": {
        "mode": "real", "predictor": "learned", "flags": ["--naive-prefetch"],
        "desc": "prefetch every prediction; no confidence/safety gates",
    },
    "no_divergence_guard": {
        "mode": "real", "predictor": "learned", "flags": ["--no-divergence-guard"],
        "desc": "divergences tracked but in-flight prefetches never cancelled",
    },
    "no_plan": {
        "mode": "real", "predictor": "learned", "flags": ["--no-plan-extraction"],
        "desc": "system ablation: no plan extraction (also disables plan-triggered"
                " worker prefetch)",
    },
    "plan_only": {
        "mode": "real", "predictor": "plan_only", "flags": [],
        "desc": "predictor ablation: plan-context signal only",
    },
    "transition_only": {
        "mode": "real", "predictor": "transition_only", "flags": [],
        "desc": "predictor ablation: learned transition table only",
    },
    "oracle": {
        "mode": "real", "predictor": "oracle", "flags": [],
        "needs_oracle_trace": True,
        "desc": "upper bound: perfect-hindsight predictor replaying a reference trace",
    },
    # External-system comparison, not an ablation: staging backend replaced by
    # MegaMmap/Hermes (mm_model_preload into the Hermes buffer pool during
    # planning; worker vLLM reads weights through libhermes_posix.so).
    # Needs the Hermes daemon — the driver starts/stops it per trial.
    "megammap_stage": {
        "mode": "real", "predictor": "learned", "flags": ["--megammap-stage"],
        "needs_hermes": True,
        "desc": "full system with MegaMmap/Hermes as the staging tier "
                "(seq preload; comparison against page-cache staging)",
    },
    "megammap_stage_rand": {
        "mode": "real", "predictor": "learned",
        "flags": ["--megammap-stage", "--megammap-tx", "rand"],
        "needs_hermes": True,
        "desc": "MegaMmap staging with random page order (no prefetch signal — "
                "models an unknown next model)",
    },
}

# chemgraph_screen: heterogeneous molecule batch with per-molecule specialist
# routing (advanced=72B / standard=32B, same port+GPUs — every class
# alternation is a real swap).  Designed 2026-07-19 so every runtime component
# has a falsifiable job: plan analysis names each task's specialist,
# transitions sharpen timing, the guard cancels wrongly-staged specialists.
# Calculator pinned to mace_mp by default (the TBLite draw fails ~90%);
# the "unpinned" config re-frees the agent to study guard behavior on failures.
CHEMGRAPH_SCREEN_CONFIGS: dict[str, dict] = {
    "baseline": {
        "mode": "baseline", "predictor": "mock", "flags": [],
        "desc": "vanilla batch; on-demand sequential specialist swaps",
    },
    "full_system": {
        "mode": "real", "predictor": "learned", "flags": ["--early-plan-stage"],
        "desc": "plan-conditioned specialist staging at plan_extracted + "
                "per-transition staging + confidence/divergence gates",
    },
    "blind_stage": {
        "mode": "real", "predictor": "learned", "flags": [],
        "desc": "trigger ablation: legacy blind staging of the default worker "
                "at first chain start (plan does not choose the model)",
    },
    "no_cache_stage": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--no-cache-stage"],
        "desc": "full system minus page-cache staging",
    },
    "naive_prefetch": {
        "mode": "real", "predictor": "learned", "flags": ["--naive-prefetch"],
        "desc": "prefetch every prediction immediately; no gates, no plan choice",
    },
    "no_divergence_guard": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--no-divergence-guard"],
        "desc": "wrongly-staged specialists are never cancelled",
    },
    "no_plan": {
        "mode": "real", "predictor": "learned", "flags": ["--no-plan-extraction"],
        "desc": "no plan extraction: no specialist sequence, routing on-demand",
    },
    "plan_only": {
        "mode": "real", "predictor": "plan_only", "flags": ["--early-plan-stage"],
        "desc": "predictor ablation: plan signal only",
    },
    "transition_only": {
        "mode": "real", "predictor": "transition_only",
        "flags": ["--early-plan-stage"],
        "desc": "predictor ablation: transition table only",
    },
    "unpinned": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--pin-calculator", ""],
        "desc": "agent free calculator choice (TBLite failures become live "
                "divergence-guard case studies)",
    },
    "oracle": {
        "mode": "real", "predictor": "oracle",
        "flags": ["--early-plan-stage"],
        "needs_oracle_trace": True,
        "desc": "upper bound: perfect-hindsight predictor replaying a "
                "reference screen trace",
    },
}

# chemgraph_screen_pool: Option D — specialists on DISJOINT GPU pools (72B on
# GPUs 0-3 port 8001, 32B on GPUs 4-5 port 8005, SpecialistProxy on 8006 in
# front of both).  Motivated by the 7/20 screen verdict: oracle ≈ full_system,
# i.e. prediction was never the bottleneck — the ~100-140 s vLLM spin-up on a
# SHARED pool is, and no shared-pool trigger can hide it.  With disjoint pools
# the next engine boots while the current one serves; the residency policy
# (evict idle engines) is identical in every arm, and the plan decides per
# transition whether the other engine is kept resident or pre-booted.
# Molecule order is deliberately NON-alternating (adv,adv,std,adv,std,std) so
# plan-conditioning is distinguishable from a blind alternation heuristic.
# L40S 6-GPU facet ONLY (Blackwell nodes have 4 GPUs).
CHEMGRAPH_SCREEN_POOL_CONFIGS: dict[str, dict] = {
    "baseline": {
        "mode": "baseline", "predictor": "mock", "flags": [],
        "desc": "evict-idle policy + on-demand engine boots (spin-up exposed "
                "at every specialist alternation)",
    },
    "full_system": {
        "mode": "real", "predictor": "learned", "flags": ["--early-plan-stage"],
        "desc": "plan-conditioned keep/pre-boot per transition + cache staging "
                "+ divergence guard (spin-up overlaps the serving window)",
    },
    "blind_stage": {
        "mode": "real", "predictor": "learned", "flags": ["--blind-preboot"],
        "desc": "trigger ablation: always prepare the OTHER specialist "
                "(alternation heuristic; wastes boots on same-class runs)",
    },
    "naive_prefetch": {
        "mode": "real", "predictor": "learned", "flags": ["--naive-prefetch"],
        "desc": "resource upper bound: boot every specialist, never evict "
                "(time-optimal, residency-maximal)",
    },
    "no_divergence_guard": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--no-divergence-guard"],
        "desc": "wrongly kept/pre-booted engines are never cancelled/evicted",
    },
    "no_plan": {
        "mode": "real", "predictor": "learned", "flags": ["--no-plan-extraction"],
        "desc": "no plan: no keep/pre-boot decisions — evict-idle + on-demand "
                "boots plus generic prefetch machinery only",
    },
    "no_cache_stage": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--no-cache-stage"],
        "desc": "full system minus page-cache staging (pre-boot reads cold "
                "Lustre)",
    },
    "plan_only": {
        "mode": "real", "predictor": "plan_only", "flags": ["--early-plan-stage"],
        "desc": "predictor ablation: plan signal only",
    },
    "transition_only": {
        "mode": "real", "predictor": "transition_only",
        "flags": ["--early-plan-stage"],
        "desc": "predictor ablation: transition table only",
    },
    "unpinned": {
        "mode": "real", "predictor": "learned",
        "flags": ["--early-plan-stage", "--pin-calculator", ""],
        "desc": "agent free calculator choice (failure-path guard case study)",
    },
    "oracle": {
        "mode": "real", "predictor": "oracle",
        "flags": ["--early-plan-stage"],
        "needs_oracle_trace": True,
        "desc": "upper bound: perfect-hindsight predictor replaying a "
                "reference pool trace",
    },
}

# AtomAgents conditions mirror experiments/run_blackwell.sh.
ATOMAGENTS_CONFIGS: dict[str, dict] = {
    "baseline":            {"mode": "baseline", "predictor": "mock", "flags": []},
    "observe_only":        {"mode": "observe_only", "predictor": "learned", "flags": []},
    "full_system":         {"mode": "real", "predictor": "learned", "flags": []},
    "no_plan":             {"mode": "real", "predictor": "learned", "flags": ["--no-plan-extraction"]},
    "no_divergence_guard": {"mode": "real", "predictor": "learned", "flags": ["--no-divergence-guard"]},
    "naive_prefetch":      {"mode": "real", "predictor": "learned", "flags": ["--naive-prefetch"]},
    "no_model_prefetch":   {"mode": "real", "predictor": "learned", "flags": ["--skip-resource-types", "vllm_model"]},
    "no_data_prefetch":    {"mode": "real", "predictor": "learned", "flags": ["--skip-resource-types", "data_file"]},
    # Sleep/wake swap mechanism. Level 2 (not 1): three engines at 108-128 GiB
    # each cannot fit a 256G hold, and gate (b) showed the node DIES at k=3
    # rather than erroring cleanly. Behaviour-changing (different vLLM launch
    # args + swap path), so explicit_only keeps it out of the implicit
    # "all configs" set — a campaign run without --configs must never pick it
    # up and silently pool its trials with the kill+cold-boot arms.
    # ---- host-side weight staging -------------------------------------
    # All three are explicit_only: they enable model_cache staging, which no
    # previously collected exp3 trial had, so a campaign run without --configs
    # must never pick them up and pool them with the existing arms.
    #
    # page_cache_stage IS THE CONTROL AND IS NOT OPTIONAL. Comparing
    # megammap_stage straight to baseline confounds two changes at once --
    # "MegaMmap replaced the page cache" and "weight staging now happens at
    # all". Only the page-cache arm separates them: baseline -> page_cache_stage
    # is the cost of staging, page_cache_stage -> megammap_stage is the cost of
    # MegaMmap. The ChemGraph result never had this arm, which is why its
    # 3.18x is a statement about staging-with-MegaMmap rather than about
    # MegaMmap.
    # ---- TANDEM ---------------------------------------------------------
    # explicit_only, and that is not caution: this arm wires a residency actor
    # into the model prefetch executor, which changes what a prefetch is
    # ALLOWED TO DO (it may now evict the GPU incumbent) and licenses the
    # proactive-swap confidence-gate bypass. No previously collected exp3 trial
    # had either, so a campaign run without --configs must never pick it up and
    # pool it with the kill+cold-boot arms.
    #
    # The comparison this arm exists for: on the aligned campaign 16 of 16
    # model prefetches failed, 10 of them within ~1 ms with "Cannot start
    # qwen_32b: GPUs occupied", and the proactive-swap ones sat in a 600 s
    # wait-for-GPUs loop before failing. Nothing about the predictor changes
    # here -- only whether a correct prediction has somewhere to put its
    # result.
    "tandem": {
        "mode": "real", "predictor": "learned",
        "flags": ["--residency"],
        "explicit_only": True,
    },
    "page_cache_stage": {
        "mode": "real", "predictor": "learned",
        "flags": ["--stage-model-cache"],
        "explicit_only": True,
        "desc": "warm weight shards into the OS page cache during compute "
                "windows; the control arm for the MegaMmap comparison",
    },
    "megammap_stage": {
        "mode": "real", "predictor": "learned",
        "flags": ["--megammap-stage"],
        "needs_hermes": True,
        "explicit_only": True,
        "desc": "weight staging through MegaMmap into the Hermes buffer pool; "
                "engines read weights via libhermes_posix.so",
    },
    "megammap_stage_rand": {
        "mode": "real", "predictor": "learned",
        "flags": ["--megammap-stage", "--megammap-tx", "rand"],
        "needs_hermes": True,
        "explicit_only": True,
        "desc": "MegaMmap staging with random page order — no prefetch signal, "
                "so it isolates window from prediction quality",
    },
    "sleep_wake": {
        "mode": "real", "predictor": "learned",
        "flags": ["--sleep-wake", "--sleep-level", "2"],
        "explicit_only": True,
        "desc": "engines boot once and park via /sleep level 2 instead of "
                "kill + cold boot; attacks the 990-1315 s bring-up stall",
    },
    "sleep_wake_baseline": {
        "mode": "baseline", "predictor": "mock",
        "flags": ["--sleep-wake", "--sleep-level", "2"],
        "explicit_only": True,
        # The honest comparator for the sleep_wake arm. Without it, any gain
        # from sleep/wake is indistinguishable from the prefetcher's gain,
        # because the kill+cold-boot baseline differs in TWO ways at once.
        "desc": "sleep/wake swaps with the runtime OFF — isolates the swap "
                "mechanism from the prefetcher",
    },
}

WORKLOADS: dict[str, dict] = {
    "chemgraph_swap": {
        "script": "experiments/chemgraph_exp.py",
        "python": CG_PYTHON,
        "base_flags": CHEMGRAPH_SWAP_BASE,
        "configs": CHEMGRAPH_SWAP_CONFIGS,
        # Sized for worst-case Lustre (~40 MB/s observed): planner alone can
        # need ~30 min and the worker swap ~1 h.
        "timeout_s": 9000,
        "est_run_s": 1800,
    },
    # Out-of-core variant: run_mace_ensemble screens ~200 CIF structures on CPU,
    # creating a genuine multi-minute GPU-idle window in which model staging /
    # swap can be fully hidden (Option D).  Same configs as chemgraph_swap.
    "chemgraph_ensemble": {
        "script": "experiments/chemgraph_exp.py",
        "python": CG_PYTHON,
        # Option D: the worker's run_mace_ensemble call is a long GPU-idle CPU
        # window; a DISTINCT aggregator model (32B, GPUs 4-5, port 8004) is
        # prefetched during that window so it is hot at AggregatorAgent.
        # Without these flags the ensemble degenerates to the swap topology's
        # ~11 s window (measured: chemgraph_ensemble_noagg baseline t01,
        # 2026-07-09 — worker swap 384 s exposed BEFORE the MACE screen).
        "base_flags": CHEMGRAPH_SWAP_BASE + [
            "--ensemble-dataset", "data/materials_ensemble",
            "--aggregator-model", "Qwen/Qwen2.5-VL-32B-Instruct-Aggregator",
            "--aggregator-base-url", "http://localhost:8004/v1",
        ],
        "configs": CHEMGRAPH_SWAP_CONFIGS,
        "timeout_s": 10800,
        "est_run_s": 2400,
    },
    # Screening batch with specialist routing — see CHEMGRAPH_SCREEN_CONFIGS.
    # ~6 molecules x (MACE opt window + specialist analysis) ≈ 20-30 min.
    "chemgraph_screen": {
        "script": "experiments/chemgraph_exp.py",
        "python": CG_PYTHON,
        "base_flags": CHEMGRAPH_SWAP_BASE + [
            "--screen",
            "--pin-calculator", "mace_mp",
        ],
        "configs": CHEMGRAPH_SCREEN_CONFIGS,
        "timeout_s": 7200,
        "est_run_s": 1800,
    },
    # Option D disjoint-pool variant — see CHEMGRAPH_SCREEN_POOL_CONFIGS.
    # L40S 6-GPU nodes only.  Non-alternating molecule order (adv,adv,std,
    # adv,std,std) makes plan-conditioning falsifiable against blind
    # alternation.  The worker client endpoint is forced to the
    # SpecialistProxy inside chemgraph_exp.py (--base-url is overridden).
    "chemgraph_screen_pool": {
        "script": "experiments/chemgraph_exp.py",
        "python": CG_PYTHON,
        "base_flags": CHEMGRAPH_SWAP_BASE + [
            "--screen",
            "--pin-calculator", "mace_mp",
            "--disjoint-pools",
            "--molecules", "aspirin, caffeine, water, ibuprofen, methane, ammonia",
        ],
        "configs": CHEMGRAPH_SCREEN_POOL_CONFIGS,
        "timeout_s": 7200,
        "est_run_s": 1800,
    },
    "atomagents_exp2": {
        "script": "experiments/atomagents_exp2.py",
        "python": ATOMS_PYTHON,
        # --lammps-slowdown mirrors exp3: without it the two screw-dislocation
        # relaxes finish in seconds (~700-atom EAM cells), the whole workflow
        # walls at ~20 s, and the wall>60 completion rule rejects a run that
        # actually completed (exp2 baseline/full_system t01 2026-07-16).
        # est_run_s=5400 was always sized for the slowed-down workload.
        "base_flags": ["--hw-profile", "blackwell", "--swap-models",
                       "--lammps-slowdown", os.environ.get("LAMMPS_SLOWDOWN_S", "900")],
        "configs": ATOMAGENTS_CONFIGS,
        "timeout_s": 10800,
        "est_run_s": 5400,
    },
    "atomagents_exp3": {
        "script": "experiments/atomagents_exp3.py",
        "python": ATOMS_PYTHON,
        "base_flags": ["--hw-profile", "blackwell_swap", "--swap-models",
                       "--lammps-slowdown", os.environ.get("LAMMPS_SLOWDOWN_S", "900")],
        "configs": ATOMAGENTS_CONFIGS,
        "timeout_s": 14400,
        "est_run_s": 7200,
    },
    # A SEPARATE WORKLOAD KEY, not a config under atomagents_exp3 — the two are
    # different workloads and their results must never pool. Three differences,
    # each deliberate:
    #   --prompt-variant aligned  the executor forwards its scope rules into
    #       plan_task, so the planner's plan is one the executor can actually
    #       run. Without this the plan names tools executed ZERO times
    #       (analyze_screw_core, computation_task_surface_energy) and the
    #       "central planner" signal contributes nothing.
    #   --potentials ...,w_eam4_big.fs  swaps the 9.3 MB potential for the
    #       3.32 GB one so loading scientific data costs real time (~129 s,
    #       >=123.5 s of it parse/spline rather than I/O). Without this the
    #       data axis is literally absent: 100% of measured stall is vllm_model.
    #   --lammps-slowdown 0  the compute window now comes from REAL potential
    #       activation instead of time.sleep(900). Override LAMMPS_SLOWDOWN_S
    #       to reintroduce the sleep as a window-size sensitivity axis.
    # est_run_s is lower than exp3's because 900 s of sleep per relax is gone.
    "atomagents_exp3_aligned": {
        "script": "experiments/atomagents_exp3.py",
        "python": ATOMS_PYTHON,
        "base_flags": ["--hw-profile", "blackwell_swap", "--swap-models",
                       "--prompt-variant", "aligned",
                       "--potentials", "W_Zhou04.eam.alloy,w_eam4_big.fs",
                       "--lammps-slowdown", os.environ.get("LAMMPS_SLOWDOWN_S", "0")],
        "configs": ATOMAGENTS_CONFIGS,
        "timeout_s": 14400,
        # MEASURED 2026-08-31, not estimated. Three completed Blackwell
        # baselines on atl1-1-03-020-6-0 ran 8616.2 / 8587.4 / 6006.5 s, mean
        # 7736.7. The old 4200 was a 2x underestimate and it cost a whole
        # campaign: the driver packed three baselines into an 8 h hold on that
        # arithmetic, then correctly declined to start the tandem arm with
        # 35 min left. Budget ~2.4 h per trial, i.e. 3 trials per 8 h hold.
        "est_run_s": 7700,
    },
    # A SEPARATE WORKLOAD KEY FOR tp=2, and the separation is the point: cold
    # boot, KV headroom and swap latency all change with tensor-parallel degree,
    # so these trials must never pool with the tp=4 ones. Identical to
    # atomagents_exp3_aligned in every other respect.
    #
    # It exists because the production tp=4 topology needs a 4-GPU gap that the
    # partition would not give us on 2026-09-01 (23/24 allocated, no start
    # estimate on three separate 4-GPU requests). At tp=2 all three models still
    # share one pool, so M=1 and forced eviction -- the property under test --
    # are preserved exactly on half the hardware.
    "atomagents_exp3_aligned_tp2": {
        "script": "experiments/atomagents_exp3.py",
        "python": ATOMS_PYTHON,
        "base_flags": ["--hw-profile", "blackwell_swap_tp2", "--swap-models",
                       "--prompt-variant", "aligned",
                       "--potentials", "W_Zhou04.eam.alloy,w_eam4_big.fs",
                       "--lammps-slowdown", os.environ.get("LAMMPS_SLOWDOWN_S", "0")],
        "configs": ATOMAGENTS_CONFIGS,
        "timeout_s": 14400,
        "est_run_s": 7700,
    },
    # DeepDriveMD is not integrated with the runtime yet: there is no
    # runtime/adapters/deepdrivemd.py and no experiment entrypoint.  The key
    # exists so the eval tree/parser are structured for it; selecting it fails
    # loudly instead of fabricating data.
    "deepdrivemd": {
        "script": None,
        "configs": {},
        "not_runnable": ("DeepDriveMD has no runtime adapter/entrypoint yet "
                         "(TODO: runtime/adapters/deepdrivemd.py + "
                         "experiments/deepdrivemd_exp.py)."),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


# ---------------------------------------------------------------------------
# Predictor provenance measured by IMPORT, not by git stamp.
#
# Why this exists (2026-08-03).  meta.json's `git_commit` is `git rev-parse
# HEAD` at trial launch, and `signal_mode` is a lookup in _PREDICTOR_SIGNAL_MODE
# keyed by the ARM NAME.  Neither says anything about the bytes Python actually
# imports: the runner is a separate process that imports whatever is ON DISK at
# its own import time, and the working tree is routinely dirty for minutes while
# other agents edit runtime/predictor.  A trial started in such a window gets a
# git_commit that does not describe its code, and a signal_mode that states the
# arm's INTENT rather than what ran.  The `learned` arm changed meaning on
# 2026-08-03 (a6283a2 made the plan and transition signals combine, roughly
# doubling prefetch volume), so a wrong stamp silently pools two predictors.
#
# The gap is NOT a few seconds of race.  LearnedPredictor is constructed only
# after the model router comes up, so the module is imported LAZILY, tens of
# minutes into a trial.  Confirmed on the sleep_wake t01 of 2026-08-03:
#   trial start / git_commit stamp .............. 12:30:32  (stamped aa5c8f7)
#   a6283a2 lands on disk ....................... 12:37:39
#   qwen_72b ready after 1585.7 s, predictor built  ~12:57
#   first prediction_result ..................... 12:57:41
# and that trace carries predictor_id="learned+both_disagree", a tag that exists
# in exactly one commit in all of history (a6283a2) -- so the trial ran a6283a2
# while its meta.json says aa5c8f7.  The exposure window was 27 minutes wide.
#
# That is why the fingerprint is taken TWICE.  A single launch-time sample would
# have recorded the pre-change blob for that trial and been just as wrong as the
# git stamp; only the before/after PAIR reveals it, as a mismatch.  The durable
# version of this check belongs at the construction site (have the predictor
# factory emit its module fingerprint into the trace), which lives in a module
# this driver does not own; until then the pair is the reliable signal.
#
# `git_blob` is the git object id of the imported bytes, so a recorded trial maps
# onto a commit mechanically:
#     git rev-parse <commit>:runtime/predictor/learned_predictor.py
# and the fingerprint is taken TWICE, before and after the runner. If the two
# differ the file was edited mid-trial and the trial must be quarantined, not
# pooled — that is the case a launch-time stamp cannot detect at all.
_PREDICTOR_PROBE = r'''
import hashlib, json, sys
out = {}
try:
    sys.path.insert(0, sys.argv[1])
    import runtime.predictor.learned_predictor as m
    path = getattr(m, "__file__", None)
    out["module_path"] = path
    if path:
        with open(path, "rb") as fh:
            data = fh.read()
        out["sha256"] = hashlib.sha256(data).hexdigest()
        out["bytes"] = len(data)
        out["git_blob"] = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
    if hasattr(m, "_DEFAULT_LOOKAHEAD"):
        out["default_lookahead"] = m._DEFAULT_LOOKAHEAD
    cls = getattr(m, "LearnedPredictor", None)
    # Presence of these properties is the structural signature of the
    # simultaneous-signals predictor; absence means the pre-change module.
    out["has_signal_mode_property"] = isinstance(
        getattr(cls, "signal_mode", None), property)
    out["has_lookahead_property"] = isinstance(
        getattr(cls, "lookahead", None), property)
except Exception as exc:
    out["error"] = repr(exc)
print(json.dumps(out))
'''


def predictor_fingerprint(python_exe: str) -> dict:
    """Import the predictor with the TRIAL's interpreter and describe it.

    Uses the same executable and sys.path the runner will use, so shadowing by
    another entry on sys.path shows up as a module_path that is not the repo's.
    Never raises: provenance must not be able to kill a campaign.
    """
    try:
        proc = subprocess.run(
            [python_exe, "-c", _PREDICTOR_PROBE, str(PROJECT_ROOT)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120)
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return {"error": f"no output (rc={proc.returncode}): "
                             f"{(proc.stderr or '')[-300:]}"}
        return json.loads(line[-1])
    except Exception as exc:
        return {"error": repr(exc)}


def gpu_info() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def slurm_deadline_epoch() -> float | None:
    """EndTime of the surrounding SLURM job, or None outside a job."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["scontrol", "show", "job", job],
                             capture_output=True, text=True, timeout=15).stdout
        for tok in out.split():
            if tok.startswith("EndTime="):
                val = tok.split("=", 1)[1]
                if val in ("Unknown", "N/A"):
                    return None
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S").timestamp()
    except Exception:
        pass
    return None


def kill_vllm_servers(wait_gpu_clear: bool = True) -> bool:
    """Kill our vLLM servers and wait for GPU memory to drain.

    Returns True if the GPUs are clean (or we were told not to wait), False if
    memory is still held after the timeout. Callers MUST honour a False: a trial
    started on a dirty GPU does not degrade, it dies — vLLM raises
    "Free memory on device cuda:N (8.81/44.39 GiB) on startup is less than
    desired GPU memory utilization" and the whole trial is lost. Eight trials
    burned this way overnight on 2026-08-02 because this function warned and
    returned anyway, so the driver marched on into a guaranteed failure and
    then repeated it for every remaining trial on that node.
    """
    user = os.environ.get("USER", "")
    subprocess.run(["pkill", "-u", user, "-f", "vllm.entrypoints.openai.api_server"],
                   capture_output=True)
    # An api_server killed mid-startup can leave orphaned tensor-parallel
    # workers (setproctitle "VLLM::Worker_TPn") holding the full KV-cache
    # allocation (seen 2026-07-14: 4x93 GiB after a timed-out qwen_72b load).
    subprocess.run(["pkill", "-u", user, "-f", "VLLM::"],
                   capture_output=True)
    time.sleep(10)
    if not wait_gpu_clear:
        return True
    used = 0.0
    for i in range(60):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout
            used = sum(float(x) for x in out.split() if x.strip())
        except Exception:
            used = 0
        if used < 4096:
            return True
        # Half way through, escalate: something is holding memory that our
        # pattern-based pkill did not match (an orphaned worker whose title was
        # rewritten, a process re-parented after its step exited). Ask the
        # driver itself which PIDs hold GPU memory and signal those directly.
        if i == 30:
            log(f"GPUs still holding {used:.0f} MiB after 150 s; "
                f"escalating to PID-targeted kill.")
            _kill_gpu_pids(user)
        time.sleep(5)
    log(f"ERROR: GPUs still holding {used:.0f} MiB after 300 s. "
        f"Refusing to start a trial on a dirty GPU.")
    return False


def _kill_gpu_pids(user: str) -> None:
    """SIGKILL our own processes that nvidia-smi reports as holding GPU memory."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return
    for tok in out.split():
        if not tok.strip().isdigit():
            continue
        pid = int(tok)
        try:
            # Only ours: killing another user's job would be both rude and,
            # on a shared node, a genuine outage for them.
            owner = subprocess.run(["ps", "-o", "user=", "-p", str(pid)],
                                   capture_output=True, text=True,
                                   timeout=10).stdout.strip()
            if owner and owner == user:
                os.kill(pid, signal.SIGKILL)
                log(f"  SIGKILLed stale GPU process {pid}")
        except Exception:
            pass


def hermes_stack_available() -> str | None:
    """Return None if the MegaMmap/Hermes stack is usable, else the reason."""
    if not Path(MEGA_MODEL_PRELOAD).exists():
        return f"mm_model_preload missing: {MEGA_MODEL_PRELOAD}"
    if not Path(HERMES_INTERCEPTOR).exists():
        return f"libhermes_posix.so missing: {HERMES_INTERCEPTOR}"
    if not Path(f"{MEGA_STACK}/bin/hrun_start_runtime").exists():
        return f"hrun_start_runtime missing: {MEGA_STACK}/bin"
    if not Path(HERMES_CONF_EVAL).exists():
        return f"Hermes config missing: {HERMES_CONF_EVAL}"
    return None


def start_hermes_daemon() -> subprocess.Popen | None:
    """Start hrun_start_runtime for one trial; returns None on failure."""
    stop_hermes_daemon(None)   # clear any stale daemon first
    env = dict(os.environ)
    env["HERMES_CONF"] = HERMES_CONF_EVAL
    env["LD_LIBRARY_PATH"] = (f"{MEGA_STACK}/lib:"
                              + env.get("LD_LIBRARY_PATH", ""))
    logf = open(EVAL_ROOT / "hermes_daemon.log", "a")
    logf.write(f"\n==== daemon start {datetime.now().isoformat()} "
               f"conf={HERMES_CONF_EVAL} ====\n")
    try:
        proc = subprocess.Popen([f"{MEGA_STACK}/bin/hrun_start_runtime"],
                                env=env, stdout=logf, stderr=subprocess.STDOUT)
    except OSError as exc:
        log(f"Hermes daemon failed to launch: {exc}")
        return None
    time.sleep(8)
    if proc.poll() is not None:
        log(f"Hermes daemon exited immediately (rc={proc.returncode}); "
            f"see {EVAL_ROOT / 'hermes_daemon.log'}")
        return None
    log("Hermes daemon running.")
    return proc


def stop_hermes_daemon(proc: subprocess.Popen | None) -> None:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (f"{MEGA_STACK}/lib:"
                              + env.get("LD_LIBRARY_PATH", ""))
    stopper = Path(f"{MEGA_STACK}/bin/hrun_stop_runtime")
    if stopper.exists():
        try:
            subprocess.run([str(stopper)], env=env, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            log("hrun_stop_runtime hung >30 s; falling back to pkill.")
    subprocess.run(["pkill", "-f", "hrun_start_runtime"], capture_output=True)
    if proc is not None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# Signal-combination mode of LearnedPredictor per --predictor value.  Mirrors
# experiments/chemgraph_exp.py:523 (`"full" if args.predictor == "learned"`).
# mock/oracle are not LearnedPredictors, so they have no signal mode.
_PREDICTOR_SIGNAL_MODE: dict[str, str | None] = {
    "learned": "full",
    "plan_only": "plan_only",
    "transition_only": "transition_only",
    "mock": None,
    "oracle": None,
}


def build_env(workload: str, needs_hermes: bool = False,
              lookahead: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    # LearnedPredictor reads its horizon from this env var (the workload
    # runners take --predictor but not --lookahead, so the driver cannot pass
    # it on the command line without changing every runner's argparse).
    if lookahead is not None:
        env["RUNTIME_PREDICTOR_LOOKAHEAD"] = str(lookahead)
    ld_parts = [f"{_AA_NVIDIA}/cudnn/lib", f"{_AA_NVIDIA}/cusparselt/lib",
                f"{_AA_NVIDIA}/nccl/lib", _TORCH_LIB, "/usr/local/cuda/lib64"]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    py_parts = [str(PROJECT_ROOT / "workloads" / "AtomAgents"),
                str(PROJECT_ROOT.parent / "ChemGraph" / "src")]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_parts)
    env.setdefault("HF_HOME", os.path.expanduser("~/scratch/hf_home"))
    env.setdefault("OPENAI_API_KEY", "dummy")
    if needs_hermes:
        env["PATH"] = f"{MEGA_STACK}/bin:{MEGA_MPI_BIN}:" + env.get("PATH", "")
        env["HERMES_INTERCEPTOR"] = HERMES_INTERCEPTOR
        env["HERMES_CONF"] = HERMES_CONF_EVAL
        env["MEGA_MODEL_PRELOAD"] = MEGA_MODEL_PRELOAD
    # Both ChemGraph workloads run the split planner/worker topology; without
    # this the planner client falls back to the worker port (8001) and dies
    # with connection-refused before the worker exists.
    if workload.startswith("chemgraph"):
        env.setdefault("VLLM_PLANNER_BASE_URL", "http://localhost:8002/v1")
    # The ensemble workload runs Option D (distinct aggregator on port 8004,
    # GPUs 4-5); the aggregator ChatOpenAI client reads this env var.
    if workload == "chemgraph_ensemble":
        env.setdefault("VLLM_AGGREGATOR_BASE_URL", "http://localhost:8004/v1")
    return env


def resolve_oracle_trace(workload: str) -> str | None:
    """Newest trace from a completed baseline/full_system trial of this workload."""
    candidates: list[Path] = []
    for cfg in ("baseline", "full_system"):
        for trial_dir in sorted((RUNS_ROOT / workload / cfg).glob("t*__*")):
            trace = trial_dir / "trace.jsonl"
            meta = trial_dir / "meta.json"
            if trace.exists() and meta.exists():
                try:
                    if json.loads(meta.read_text()).get("status") == "completed":
                        candidates.append(trace)
                except Exception:
                    pass
    if candidates:
        return str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
    if workload == "chemgraph_swap" and Path(_FALLBACK_ORACLE_TRACE).exists():
        return _FALLBACK_ORACLE_TRACE
    return None


# ---------------------------------------------------------------------------
# Trial bookkeeping
# ---------------------------------------------------------------------------

def completed_trials(workload: str, config: str) -> int:
    n = 0
    for meta_path in (RUNS_ROOT / workload / config).glob("t*__*/meta.json"):
        try:
            if json.loads(meta_path.read_text()).get("status") == "completed":
                n += 1
        except Exception:
            pass
    return n


def trial_status_table() -> str:
    lines = [f"{'workload':<18} {'config':<22} {'completed':>9}  {'failed':>6}"]
    for wl_dir in sorted(RUNS_ROOT.glob("*")):
        for cfg_dir in sorted(wl_dir.glob("*")):
            done = fail = 0
            for meta_path in cfg_dir.glob("t*__*/meta.json"):
                try:
                    st = json.loads(meta_path.read_text()).get("status")
                except Exception:
                    st = "unreadable"
                if st == "completed":
                    done += 1
                elif st != "running":
                    fail += 1
            lines.append(f"{wl_dir.name:<18} {cfg_dir.name:<22} {done:>9}  {fail:>6}")
    return "\n".join(lines)


def run_one_trial(
    workload: str,
    config: str,
    trial_idx: int,
    dry_run: bool = False,
    lookahead: int = _DEFAULT_LOOKAHEAD,
) -> bool:
    """Launch one trial; returns True if it completed successfully."""
    wl = WORKLOADS[workload]
    cfg = wl["configs"][config]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_commit()
    run_id = f"eval_{workload}_{config}_t{trial_idx:02d}_{ts}_{commit}"
    trial_dir = RUNS_ROOT / workload / config / f"t{trial_idx:02d}__{ts}__{commit}"

    flags = list(wl["base_flags"]) + list(cfg["flags"])
    flags += ["--runtime-mode", cfg["mode"], "--predictor", cfg["predictor"],
              "--condition", config, "--run-id", run_id]
    if cfg.get("needs_oracle_trace"):
        oracle_trace = resolve_oracle_trace(workload)
        if not oracle_trace:
            log(f"SKIP {workload}/{config}: no completed reference trace for the "
                "oracle yet (run baseline/full_system first).")
            return False
        flags += ["--oracle-trace", oracle_trace]

    cmd = [wl["python"], "-u", str(PROJECT_ROOT / wl["script"])] + flags
    # Source fix_tmp.sh first: PACE job-private /tmp may be a dangling mount,
    # which breaks vLLM engine-core init (see setup/fix_tmp.sh).
    shell_cmd = ["bash", "-c",
                 'source "$1"/setup/fix_tmp.sh; shift; exec "$@"',
                 "eval_run", str(PROJECT_ROOT)] + cmd

    if dry_run:
        log(f"DRY-RUN {workload}/{config} t{trial_idx:02d}: {' '.join(cmd)}")
        return True

    hermes_proc = None
    if cfg.get("needs_hermes"):
        reason = hermes_stack_available()
        if reason:
            log(f"SKIP {workload}/{config}: MegaMmap/Hermes stack unavailable "
                f"({reason}) — config skipped, not faked.")
            return False
        hermes_proc = start_hermes_daemon()
        if hermes_proc is None:
            log(f"SKIP {workload}/{config}: Hermes daemon failed to start.")
            return False

    trial_dir.mkdir(parents=True, exist_ok=False)
    signal_mode = _PREDICTOR_SIGNAL_MODE.get(cfg["predictor"])
    # Only LearnedPredictor arms have a horizon; recording a number for a
    # mock/oracle arm would imply the knob did something there.
    effective_lookahead = lookahead if signal_mode else None
    env = build_env(workload, needs_hermes=bool(cfg.get("needs_hermes")),
                    lookahead=effective_lookahead)
    meta = {
        "workload": workload,
        "config": config,
        "trial_index": trial_idx,
        "run_id": run_id,
        "command": " ".join(cmd),
        "runtime_mode": cfg["mode"],
        "predictor": cfg["predictor"],
        # Predictor provenance (added 2026-08-03).  The `learned` arms changed
        # meaning when the two predictor signals were made simultaneous, so
        # trials recorded before/after that change must never be pooled.  These
        # two keys make the split mechanical: signal_mode is which signals the
        # predictor combined, lookahead is the horizon it searched.  Trials
        # predating this key carry neither and are the pre-change population.
        "signal_mode": signal_mode,
        "lookahead": effective_lookahead,
        # signal_mode/lookahead above are the arm's INTENT, resolved from a
        # table at launch; git_commit below is HEAD at launch.  Neither is
        # evidence about the code that ran.  predictor_import IS: it is the
        # module the trial's own interpreter resolves and its bytes' git blob
        # id.  Prefer it over git_commit whenever the two disagree.
        "predictor_import": predictor_fingerprint(wl["python"]),
        "extra_flags": cfg["flags"],
        "git_commit": commit,
        "node": socket.gethostname(),
        "gpus": gpu_info(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        # THE HOST-RAM ALLOCATION IS AN EXPERIMENTAL VARIABLE, NOT AN
        # ENVIRONMENT DETAIL.  Tandem's whole mechanism is "what fits in the
        # budget": at --mem=256G only qwen_32b (129.7 GB) is parkable and the
        # two reused 72Bs (279.0 / 276.3 GB) are refused, so retention can buy
        # nothing; at --mem=700G both fit and the same code should park and
        # wake them.  Trials from those two allocations are therefore DIFFERENT
        # CONFIGURATIONS and must never be pooled -- and until now the only
        # record of which was which was slurm_job_id, recoverable solely via
        # `sacct -j <id> --format=ReqMem` while that job stayed in the
        # accounting DB.  Record it in the trial itself.
        "slurm_mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "conda_python": wl["python"],
        "start_time": datetime.now().astimezone().isoformat(),
        "status": "running",
    }
    (trial_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(EVAL_ROOT / "commands.log", "a") as f:
        f.write(f"{meta['start_time']}  {run_id}\n  {' '.join(cmd)}\n")

    if not kill_vllm_servers():
        # Do not spend an hour of a preemptible hold on a trial that cannot
        # start. Reported as its own status so these are never silently pooled
        # with genuine workload failures when the results are aggregated.
        log(f"SKIP {workload}/{config} t{trial_idx:02d} — GPUs not clean")
        meta["status"] = "skipped_dirty_gpu"
        meta["end_time"] = datetime.now().astimezone().isoformat()
        (trial_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        # False, not the status string: this function's contract is "did the
        # trial complete", and a truthy return would count a skip as a success
        # and decrement the remaining-trials target. It also counts toward
        # --max-failures, which is what stops the campaign from retrying a
        # node-level fault for the rest of the hold.
        return False

    log(f"START {workload}/{config} t{trial_idx:02d}  run_id={run_id}")

    status = "failed"
    rc: int | None = None
    try:
        with open(trial_dir / "stdout.log", "w") as logf:
            try:
                proc = subprocess.run(
                    shell_cmd, env=env, cwd=str(PROJECT_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT,
                    timeout=wl["timeout_s"],
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
                log(f"TIMEOUT after {wl['timeout_s']} s")
                kill_vllm_servers(wait_gpu_clear=False)
    finally:
        if hermes_proc is not None:
            stop_hermes_daemon(hermes_proc)

    # Collect artifacts by run_id
    summary_src = PROJECT_ROOT / "results" / f"summary_{run_id}.json"
    summary = None
    if summary_src.exists():
        shutil.copy2(summary_src, trial_dir / "summary.json")
        try:
            summary = json.loads(summary_src.read_text())
        except Exception:
            summary = None
        # Stamp the predictor provenance into the ARCHIVED summary too, next to
        # the runner's own descriptors (condition/hw_profile/prompt_variant/...),
        # so any tool reading summary.json alone can split pre-/post-change
        # `learned` trials without cross-referencing meta.json.  Additive keys
        # only; the runner-owned results/summary_<run_id>.json is left as-is.
        if isinstance(summary, dict):
            summary.setdefault("signal_mode", signal_mode)
            summary.setdefault("lookahead", effective_lookahead)
            try:
                (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            except OSError as exc:
                log(f"WARN: could not stamp provenance into summary.json: {exc!r}")
    # trace_path/system_profile_csv may be None or "" even when a summary
    # exists (atomagents_exp2/3 write trace_path=None): Path("") is '.', which
    # passes .exists() and made copy2 raise IsADirectoryError, killing the
    # driver mid-campaign (2026-07-09).  Guard the raw strings and fall back to
    # a run_id search whenever the summary doesn't yield a usable path.
    try:
        trace_p = (summary or {}).get("trace_path")
        if trace_p and Path(trace_p).is_file():
            shutil.copy2(trace_p, trial_dir / "trace.jsonl")
        else:
            # Runtime trace filenames embed the run_id — match by name first,
            # since a content grep can never match AtomAgents baseline traces
            # (they are empty by design).
            by_name = sorted(
                (PROJECT_ROOT / "logs/workflow_traces").glob(f"*{run_id}*.jsonl"))
            if by_name:
                shutil.copy2(by_name[-1], trial_dir / "trace.jsonl")
            else:
                for trace in sorted(
                        (PROJECT_ROOT / "logs/workflow_traces").glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                    try:
                        head = trace.read_text(errors="ignore")[:200000]
                    except Exception:
                        continue
                    if run_id in head:
                        shutil.copy2(trace, trial_dir / "trace.jsonl")
                        break
        prof_p = (summary or {}).get("system_profile_csv")
        if not (prof_p and Path(prof_p).is_file()):
            cand = PROJECT_ROOT / "results" / f"system_profile_{run_id}.csv"
            prof_p = str(cand) if cand.is_file() else None
        if prof_p:
            shutil.copy2(prof_p, trial_dir / "system_profile.csv")
        # AtomAgents runners put their per-phase timing in a metrics CSV (the
        # only structured source of the Q2 breakdown there — baseline trials
        # write an empty trace.jsonl).
        mcsv = PROJECT_ROOT / "results" / f"atomagents_metrics_{run_id}.csv"
        if mcsv.is_file():
            shutil.copy2(mcsv, trial_dir / "metrics.csv")
    except Exception as exc:  # never let artifact copying kill the campaign
        log(f"WARN: artifact collection failed for {run_id}: {exc!r}")

    if status != "timeout":
        wall = (summary or {}).get("wall_time_s", 0)
        wf_ok = (summary or {}).get("workflow_completed")
        # workflow_completed is only written by runners that carry the new field;
        # for older runners fall back to exit code + plausible wall time.
        if rc == 0 and summary is not None and wall > 60 and wf_ok is not False:
            status = "completed"
        else:
            status = "failed"

    # Second fingerprint: a commit landing DURING the trial leaves the launch
    # stamp describing code the runner never ran.  Comparing the two blob ids is
    # the only way to see that from the record afterwards.  A True here means the
    # trial straddled an edit and must be quarantined, not pooled.
    pred_after = predictor_fingerprint(wl["python"])
    pred_before = meta.get("predictor_import") or {}
    before_blob, after_blob = pred_before.get("git_blob"), pred_after.get("git_blob")
    changed = bool(before_blob and after_blob and before_blob != after_blob)
    if changed:
        log(f"WARN {run_id}: predictor source changed mid-trial "
            f"({before_blob[:12]} -> {after_blob[:12]}) — QUARANTINE, do not pool")
    meta.update({
        "status": status,
        "exit_code": rc,
        "end_time": datetime.now().astimezone().isoformat(),
        "wall_time_s": (summary or {}).get("wall_time_s"),
        "predictor_import_after": pred_after,
        "predictor_changed_mid_trial": changed,
    })
    (trial_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    # Mirror into the archived summary so a tool reading summary.json alone can
    # make the same pre-/post-change split, matching the signal_mode stamp above.
    summary_p = trial_dir / "summary.json"
    if isinstance(summary, dict) and summary_p.exists():
        summary.setdefault("predictor_import", pred_before)
        summary.setdefault("predictor_changed_mid_trial", changed)
        try:
            summary_p.write_text(json.dumps(summary, indent=2))
        except OSError as exc:
            log(f"WARN: could not stamp predictor_import into summary.json: {exc!r}")
    log(f"END   {workload}/{config} t{trial_idx:02d}  status={status} "
        f"rc={rc} wall={(summary or {}).get('wall_time_s')}")
    return status == "completed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q1–Q4 evaluation driver (resumable, never overwrites)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workload", default="chemgraph_swap",
                        choices=sorted(WORKLOADS.keys()))
    parser.add_argument("--configs", default="",
                        help="Comma-separated config subset (default: all for workload)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Target completed trials per (workload, config)")
    parser.add_argument("--order", default="roundrobin",
                        choices=["roundrobin", "sequential"],
                        help="roundrobin keeps trial counts balanced if interrupted")
    parser.add_argument("--deadline-margin-s", type=int, default=900,
                        help="Stop launching runs this close to the SLURM job end")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="Give up on a config after this many consecutive failures")
    parser.add_argument("--lookahead", type=int, default=_DEFAULT_LOOKAHEAD,
                        help="LearnedPredictor horizon in steps (offsets "
                             "1..N are searched by both signals).  Recorded in "
                             "meta.json/summary.json; ignored by mock/oracle arms")
    parser.add_argument("--list", action="store_true", help="Show status and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.lookahead < 1:
        print(f"ERROR: --lookahead must be >= 1, got {args.lookahead}")
        sys.exit(2)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(trial_status_table())
        return

    wl = WORKLOADS[args.workload]
    if wl.get("not_runnable"):
        print(f"ERROR: workload '{args.workload}' is not runnable: {wl['not_runnable']}")
        sys.exit(2)

    # Default (no --configs): every config EXCEPT explicit_only ones —
    # behavior-changing arms (e.g. sleep_wake: different vLLM launch args and
    # swap mechanism) must be opted into by name, never picked up implicitly.
    config_names = ([c.strip() for c in args.configs.split(",") if c.strip()]
                    or [c for c, spec in wl["configs"].items()
                        if not spec.get("explicit_only")])
    unknown = [c for c in config_names if c not in wl["configs"]]
    if unknown:
        print(f"ERROR: unknown configs for {args.workload}: {unknown}\n"
              f"Available: {sorted(wl['configs'].keys())}")
        sys.exit(2)

    deadline = slurm_deadline_epoch()
    if deadline:
        log(f"SLURM deadline: {datetime.fromtimestamp(deadline)} "
            f"(margin {args.deadline_margin_s} s + est. run length)")

    # Build worklist: (config, trial_index) for missing trials only
    todo: dict[str, list[int]] = {}
    for cfg in config_names:
        have = completed_trials(args.workload, cfg)
        todo[cfg] = list(range(have + 1, args.trials + 1))
        log(f"{args.workload}/{cfg}: {have}/{args.trials} completed, "
            f"{len(todo[cfg])} to run")

    consecutive_failures = {c: 0 for c in config_names}

    def out_of_time() -> bool:
        if not deadline:
            return False
        return time.time() + wl["est_run_s"] + args.deadline_margin_s > deadline

    def eligible(cfg: str) -> bool:
        return bool(todo[cfg]) and consecutive_failures[cfg] < args.max_failures

    stopped_for_time = False
    if args.order == "roundrobin":
        while any(eligible(c) for c in config_names):
            progressed = False
            for cfg in config_names:
                if not eligible(cfg):
                    continue
                if out_of_time():
                    stopped_for_time = True
                    break
                idx = todo[cfg].pop(0)
                ok = run_one_trial(args.workload, cfg, idx, dry_run=args.dry_run,
                                   lookahead=args.lookahead)
                consecutive_failures[cfg] = 0 if ok else consecutive_failures[cfg] + 1
                if not ok and not args.dry_run:
                    # Failed trial dirs stay on disk; schedule a replacement index
                    todo[cfg].append(todo[cfg][-1] + 1 if todo[cfg] else idx + 1)
                progressed = True
            if stopped_for_time or not progressed:
                break
    else:
        for cfg in config_names:
            while eligible(cfg):
                if out_of_time():
                    stopped_for_time = True
                    break
                idx = todo[cfg].pop(0)
                ok = run_one_trial(args.workload, cfg, idx, dry_run=args.dry_run,
                                   lookahead=args.lookahead)
                consecutive_failures[cfg] = 0 if ok else consecutive_failures[cfg] + 1
                if not ok and not args.dry_run:
                    todo[cfg].append(todo[cfg][-1] + 1 if todo[cfg] else idx + 1)
            if stopped_for_time:
                break

    if stopped_for_time:
        log("Stopped before SLURM deadline; rerun the same command on the next "
            "allocation to resume.")
    kill_vllm_servers(wait_gpu_clear=False)
    log("Driver finished.\n")
    print(trial_status_table())


if __name__ == "__main__":
    main()
