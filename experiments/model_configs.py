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
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B_VL,
        "port": 8002,
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
        "port": 8001,
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
#     qwen_72b     (port 8001) — main engineer agent  (vision + reasoning)
#     qwen_32b     (port 8002) — planner/critic agents (vision + planning)
#     qwen_72b_text(port 8003) — code specialist agent (text, scripts)
# ---------------------------------------------------------------------------
MODELS_BLACKWELL_SWAP = {
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_VL,
        "port": 8001,
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
        "port": 8002,
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
# L40S profile (exp3 — shared GPU pool, forced model swapping)
#   All three models share GPUs 0-5 (tp=6, 6 × 48 GB = 288 GB).
#   72B fp16 ≈ 144 GB; 32B fp16 ≈ 64 GB; 72B-text fp16 ≈ 144 GB.
#   Only ONE model can be resident at a time.
# ---------------------------------------------------------------------------
MODELS_L40S = {
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B_VL,
        "port": 8001,
        "gpus": [0, 1, 2, 3, 4, 5],
        "tensor_parallel_size": 6,
        "gpu_memory_utilization": 0.92,
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
        "port": 8002,
        "gpus": [0, 1, 2, 3, 4, 5],
        "tensor_parallel_size": 6,
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
        "gpus": [0, 1, 2, 3, 4, 5],
        "tensor_parallel_size": 6,
        "gpu_memory_utilization": 0.92,
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
