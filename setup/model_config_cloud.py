"""
model_config_cloud.py — Drop-in replacement for atomagents/runtime/model_config.py
on a cloud A100 node (no NFS; models on local SSD).

Usage: copy this file over model_config.py after running setup_cloud.sh,
or set ATOMAGENTS_MODEL_CONFIG=setup/model_config_cloud.py and patch the
import in model_orchestrator.py to respect that env var.

The only required edit is MODEL_BASE_DIR — set it to wherever
setup_cloud.sh downloaded the Qwen weights.
"""
import os

# ---- Edit this to match your download location ----
MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", "/models")

# ---- Edit this after setup_cloud.sh reports the vllm env path ----
VLLM_PYTHON = os.environ.get(
    "VLLM_PYTHON",
    os.path.expanduser("~/miniconda3/envs/vllm/bin/python"),
)

# ---------------------------------------------------------------------------
# A100 profile (80 GB VRAM each)
#
# Single A100-80GB:
#   qwen_32b fits alone (32B fp16 ≈ 64 GB < 80 GB)
#   qwen_72b does NOT fit alone (72B fp16 ≈ 144 GB) — needs 2× A100
#
# 2× A100-80GB (recommended minimum):
#   qwen_32b : GPU 0       tensor_parallel=1  (leaves GPU 1 free)
#   qwen_72b : GPU 0+1     tensor_parallel=2  (uses both)
#   Both cannot run simultaneously on 2× A100; use --swap-models.
#
# 4× A100-80GB:
#   qwen_32b : GPU 0       tensor_parallel=1
#   qwen_72b : GPU 1-3     tensor_parallel=3  (adjust tensor_parallel to 4 for symmetry)
#   Can run simultaneously; remove --swap-models flag.
# ---------------------------------------------------------------------------

_N_GPUS = int(os.environ.get("N_GPUS", "2"))

if _N_GPUS >= 4:
    # 4× A100: simultaneous load possible
    _32B_GPUS = [0]
    _32B_TP   = 1
    _72B_GPUS = [1, 2, 3]
    _72B_TP   = 3
else:
    # 2× A100: must swap (--swap-models)
    _32B_GPUS = [0]
    _32B_TP   = 1
    _72B_GPUS = [0, 1]
    _72B_TP   = 2

# A100-80GB is fast local NVMe; load times are far shorter than PACE NFS.
# 32B ≈ 3-5 min from NVMe; 72B ≈ 8-12 min.
MODELS = {
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": os.path.join(MODEL_BASE_DIR, "qwen_32b"),
        "port": 8002,
        "gpus": _32B_GPUS,
        "tensor_parallel_size": _32B_TP,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 16384,
        "load_timeout": 600,           # 10 min; fast NVMe
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
        ],
    },
    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": os.path.join(MODEL_BASE_DIR, "qwen_72b"),
        "port": 8001,
        "gpus": _72B_GPUS,
        "tensor_parallel_size": _72B_TP,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 16384,
        "load_timeout": 900,           # 15 min; fast NVMe
        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
        ],
    },
}

# Single-GPU fallback for smoke-testing with qwen_32b only
MODELS_SINGLE_GPU = {
    "qwen_32b": {**MODELS["qwen_32b"], "gpus": [0], "tensor_parallel_size": 1},
}
