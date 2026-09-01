"""
model_configs.py — cluster-specific hardware profiles that live outside the
AtomAgents submodule so they can be updated without touching the submodule.

Import from atomagents_exp2.py / atomagents_exp3.py via --hw-profile flag.

Exp3 design: all models share the same GPU pool so only one can be resident at
a time.  This forces real model swaps and creates meaningful prefetch windows.
"""
import os

# vLLM python (shared project env, same across all PACE nodes in r-ag117-0)
VLLM_PYTHON = "/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean128/bin/python"

# HF hub snapshot paths (weights live in ~/scratch/hf_home/hub/)
_HF_HUB = os.path.expanduser("~/scratch/hf_home/hub")
_SNAPSHOT_72B_VL = os.path.join(
    _HF_HUB,
    "models--Qwen--Qwen2.5-VL-72B-Instruct",
    "snapshots",
    "89c86200743eec961a297729e7990e8f2ddbc4c5",
)
_SNAPSHOT_32B_VL = os.path.join(
    _HF_HUB,
    "models--Qwen--Qwen2.5-VL-32B-Instruct",
    "snapshots",
    "7cfb30d71a1f4f49a57592323337a4a4727301da",
)
_SNAPSHOT_72B_TEXT = os.path.join(
    _HF_HUB,
    "models--Qwen--Qwen2.5-72B-Instruct",
    "snapshots",
    "495f39366efef23836d0cfae4fbe635880d2be31",
)

# Keep old name as alias so exp2.py's `MODELS_BLACKWELL` import still works.
_SNAPSHOT_72B = _SNAPSHOT_72B_VL
_SNAPSHOT_32B = _SNAPSHOT_32B_VL

# ---------------------------------------------------------------------------
# Blackwell profile (exp2 — original, simultaneous residency)
#   qwen_32b : GPUs 0-1  (2 × 96 GB)  tensor_parallel=2
#   qwen_72b : GPUs 2-3  (2 × 96 GB)  tensor_parallel=2
#   Both fit simultaneously; no forced swapping.
# ---------------------------------------------------------------------------
MODELS_BLACKWELL = {
    # Ports fixed 2026-07-09: these MUST match workloads/AtomAgents/config_list
    # (the AutoGen agents' endpoints: 72B→8007, 32B→8012 — the same ports the
    # SWAP profile uses), or the router's port_map never matches the agents'
    # base_urls and every LLM call gets Connection refused.  The historical
    # 8001/8002 never matched AND collided with another user's vLLM on :8002
    # (Blackwell nodes are shared).
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8012,
        "gpus": [0, 1],
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.82,
        "max_model_len": 16384,
        "load_timeout": 1200,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct",
        ],
    },
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_VL,
        "port": 8007,
        "gpus": [2, 3],
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.97,
        "max_model_len": 16384,
        "load_timeout": 2700,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
    },
}

# ---------------------------------------------------------------------------
# Blackwell profile (exp3 — shared GPU pool, forced model swapping)
#   All three models share GPUs 0-3 (tp=4, 4 × 96 GB = 384 GB).
#   Only ONE model can be resident at a time; ModelRouter stops the current
#   model and loads the next one on every agent-role transition.
#
#   Model roles in exp3:
#     qwen_72b     (port 8007) — main engineer agent  (vision + reasoning)
#     qwen_32b     (port 8012) — planner/critic agents (vision + planning)
#     qwen_72b_text(port 8003) — code specialist agent (text, scripts)
# ---------------------------------------------------------------------------
MODELS_BLACKWELL_SWAP = {
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_VL,
        "port": 8007,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 16384,
        "load_timeout": 2700,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
    },
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8012,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.82,
        "max_model_len": 16384,
        "load_timeout": 1200,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct",
        ],
    },
    "qwen_72b_text": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_TEXT,
        "port": 8003,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 16384,
        "load_timeout": 2700,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
}

# ---------------------------------------------------------------------------
# ChemGraph swap profile — shared GPU pool, forced model swapping
#   planner (32B-VL) and worker (72B-Instruct) share all 4 GPUs (tp=4).
#   32B-VL at 0.90 utilization claims ~346 GB; 72B-Instruct needs ~144 GB →
#   cannot be co-resident.  Only ONE model fits at a time.
#   Workflow: planner loaded first → plan extracted → runtime prefetches worker
#   (stops planner, loads worker in background) → worker runs hot.
# ---------------------------------------------------------------------------
MODELS_CHEMGRAPH_SWAP = {
    "qwen_32b_vl": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8002,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 16384,
        # Lustre bandwidth swings 40-300 MB/s across the day; at the low end
        # the 65 GB planner alone needs >25 min (observed 2026-07-08, L40S).
        "load_timeout": 3600,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct",
        ],
    },
    "qwen_72b_instruct": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_TEXT,
        "port": 8001,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        # 0.95 -> 0.92 (2026-07-11): a ~1.7 GiB non-drainable context on GPU 0
        # (L40S job 10932526) left free memory at 94.8% and vLLM refuses to
        # start when free < requested utilization — two ensemble trials died
        # at worker boot.  0.92 leaves ~3.5 GiB/GPU headroom; weights need
        # ~36.3 GiB/GPU so KV space is still ample.  Weight-load/stall
        # measurements are unaffected by the utilization fraction.
        "gpu_memory_utilization": 0.92,
        "max_model_len": 16384,
        # See planner note: sized for worst-case Lustre (145 GB worker).
        "load_timeout": 5400,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
    # Distinct AGGREGATOR model for the Option D compute-window prefetch demo.
    # Lives on the otherwise-idle GPUs 4-5 (tp=2) so it can be loaded
    # CO-RESIDENT with the 72B worker (GPUs 0-3) during the long GPU-idle MACE
    # ensemble compute window — no swap/stop of the worker required.  A 32B-VL
    # fp16 ≈ 64 GB; 2 × 46 GB × 0.92 ≈ 85 GB leaves headroom for KV cache.
    # 32B has 64 query / 8 KV heads → tp=2 divides both.
    "qwen_32b_aggregator": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8004,
        "gpus": [4, 5],
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 8192,
        "load_timeout": 1800,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct-Aggregator",
        ],
    },
    # STANDARD specialist for the chemgraph_screen workload.  Deliberately on
    # GPUs 0-3 (same pool as the 72B "advanced" specialist and the planner) so
    # every advanced<->standard class alternation is a REAL swap on both the
    # 4-GPU Blackwell and 6-GPU L40S facets — the per-transition load is the
    # cost the plan-conditioned staging must hide.  tp=4 divides 64 heads.
    # Same PORT and SERVED NAME as the 72B worker: the ChemGraph worker client
    # is built once against one base_url/model-name, so specialists must be
    # interchangeable behind it.  Which specialist answers is decided solely by
    # which vLLM process the orchestrator has running (never both — same port,
    # same GPUs).
    "qwen_32b_standard": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8001,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.92,
        # 16384 to match the 72B worker it stands in for: late-batch worker
        # contexts (accumulated tool outputs) overflow 8192 (baseline t02,
        # 2026-07-19).  tp=4 leaves ample KV room.
        "max_model_len": 16384,
        "load_timeout": 1800,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
    # STANDARD specialist, DISJOINT-POOL variant (chemgraph_screen_pool /
    # Option D).  Lives on GPUs 4-5 with its own port so it can boot
    # CO-RESIDENT with the 72B advanced specialist (GPUs 0-3) — the pre-boot
    # overlaps the other pool's serving window, hiding the vLLM spin-up that
    # the shared-pool swap exposes.  L40S 6-GPU facet only (Blackwell nodes
    # have 4 GPUs).  Same served name as the 72B worker; the client reaches
    # whichever specialist is current through the SpecialistProxy.
    # 32B-VL fp16 ≈ 64 GB over 2 × 45 GB × 0.92 ≈ 83 GB; tp=2 divides
    # 64 query / 8 KV heads.
    "qwen_32b_standard_pool": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8005,
        "gpus": [4, 5],
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 16384,
        "load_timeout": 1800,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
}

# ---------------------------------------------------------------------------
# L40S profile (exp3 — shared GPU pool, forced model swapping)
#   All three models share GPUs 0-3 (tp=4, 4 × 48 GB = 192 GB usable).
#   Qwen2.5-VL-72B has 64 attention heads → tp must divide 64; tp=6 fails.
#   72B fp16 ≈ 144 GB; 32B fp16 ≈ 64 GB; 72B-text fp16 ≈ 144 GB.
#   4 × 46GB × 0.94 = 173 GB > 144 GB — all three models fit one at a time.
#   GPUs 4-5 are intentionally left idle (no valid tp for 64-head model).
#   Only ONE model can be resident at a time.
# ---------------------------------------------------------------------------
MODELS_L40S = {
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_VL,
        "port": 8007,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.94,
        "max_model_len": 16384,
        "load_timeout": 3600,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-72B-Instruct",
        ],
    },
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8012,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.82,
        "max_model_len": 16384,
        "load_timeout": 1800,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-VL-32B-Instruct",
        ],
    },
    "qwen_72b_text": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_TEXT,
        "port": 8003,
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.94,
        "max_model_len": 16384,
        "load_timeout": 3600,
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--served-model-name", "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
}


# ---------------------------------------------------------------------------
# Blackwell profile, TWO GPUs (exp3 tp=2 shared pool)
#
# WHY THIS EXISTS. MODELS_BLACKWELL_SWAP puts all three models on GPUs 0-3 at
# tp=4, which is the production topology but needs a 4-GPU gap. On 2026-09-01
# the partition sat at 23/24 GPUs allocated with seven 1-GPU jobs pending, and
# none of three 4-GPU requests ever got a start estimate.
#
# The property the experiment depends on is CONTENTION -- all models sharing one
# pool so that M=1 and every swap forces an eviction. That property does not
# need four cards. At tp=2 all three models share GPUs [0,1], so M=1 is
# preserved exactly, on hardware that actually schedules.
#
# The arithmetic (97887 MiB/card, gpu_memory_utilization 0.95):
#   usable across 2 cards      181.6 GiB
#   qwen_72b / qwen_72b_text   136.7 GiB weights  ->  ~45 GiB for KV
#   qwen_32b                    63.6 GiB weights  -> ~118 GiB for KV
# Comfortable at max_model_len 16384.
#
# THIS IS A DIFFERENT HARDWARE CONFIGURATION AND MUST NOT POOL WITH tp=4 DATA.
# Cold-boot cost, KV headroom and swap latency all change with tp. It is wired
# to its own workload key (atomagents_exp3_aligned_tp2) for that reason, the
# same way exp3_aligned was split from exp3.
# ---------------------------------------------------------------------------
MODELS_BLACKWELL_SWAP_TP2 = {
    name: {**cfg, "gpus": [0, 1], "tensor_parallel_size": 2}
    for name, cfg in MODELS_BLACKWELL_SWAP.items()
}
