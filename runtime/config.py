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
    plan_extraction_horizon: int = 3    # only attempt plan extraction on steps ≤ this

    # Logging
    log_dir: str = "logs/workflow_traces"
    results_dir: str = "results"
