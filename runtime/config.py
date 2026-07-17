"""
config.py — Runtime configuration and mode enum.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class RuntimeMode(str, Enum):
    BASELINE     = "baseline"      # no runtime layer; zero overhead
    OBSERVE_ONLY = "observe_only"  # emit prediction events; no prefetch I/O
    SIMULATED    = "simulated"     # log "would prefetch" decisions; no actual I/O
    REAL         = "real"          # start actual prefetch tasks
    ORACLE       = "oracle"        # replay known trace for upper-bound estimate


@dataclass
class RuntimeConfig:
    mode: RuntimeMode = RuntimeMode.SIMULATED
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Predictor settings
    predictor_context_events: int = 10    # recent JSONL events fed to predictor
    confidence_threshold: float = 0.85   # minimum confidence to trigger prefetch
    max_horizon: int = 2                  # ignore predictions further than this

    # Divergence guard settings
    conservative_mode_steps: int = 3     # steps to stay conservative after divergence

    # Prefetch scheduler
    min_remaining_compute_s: float = 5.0  # don't prefetch if less compute time expected

    # Plan extraction
    plan_extraction_horizon: int = 3    # step-horizon fallback for plan extraction; 0 disables it
                                        # (AtomAgents also extracts until the first real tool runs)

    # Logging
    log_dir: str = "logs/workflow_traces"
    results_dir: str = "results"

    # ----------------------------------------------------------------
    # Ablation knobs — set via CLI flags in experiment runners
    # ----------------------------------------------------------------

    # Skip divergence cancellation: on mismatch, track accuracy but don't
    # cancel in-flight prefetches or enter conservative mode.
    disable_divergence_cancellation: bool = False

    # Naive prefetch: ignore confidence threshold and cancellable checks —
    # prefetch every prediction regardless of confidence.
    naive_prefetch: bool = False

    # Resource-type filter: resource types in this set are never prefetched.
    # E.g. {"vllm_model"} disables model prefetch; {"data_file"} disables
    # data staging. Empty set = no filtering (default behaviour).
    skip_resource_types: list = field(default_factory=list)

    # Human-readable label stored in summary JSON for ablation bookkeeping.
    condition: str = "full_system"

    # Swap mode: if non-empty, the adapter schedules a vllm_model prefetch for
    # this model name immediately after plan extraction (before WorkerAgent runs).
    vllm_worker_model: str = ""

    # Swap mode, Option A: warm the worker model's weight shards into the OS page
    # cache during the planner phase (pure host I/O, concurrent with GPU-bound
    # planner inference).  Scheduled at the first chain start as a "model_cache"
    # resource; disable per-ablation via skip_resource_types=["model_cache"].
    stage_worker_cache: bool = False

    # HF snapshot dir of the worker model (swap mode).  Used only to compute
    # estimated_size_bytes for staged/prefetched model resources so the trace
    # carries byte-level speculation-cost data.
    worker_model_path: str = ""

    # Option D: if non-empty, the adapter starts this (distinct, co-resident)
    # aggregator model on the ensemble tool's tool_start — during the long
    # GPU-idle MACE compute window — so it is hot when control reaches
    # AggregatorAgent.  Lives on its own GPUs, so no worker stop/swap is needed.
    vllm_aggregator_model: str = ""
