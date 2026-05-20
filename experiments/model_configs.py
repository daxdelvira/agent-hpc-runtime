"""
model_configs.py — cluster-specific hardware profiles that live outside the
AtomAgents submodule so they can be updated without touching the submodule.

Import from atomagents_exp2.py via --hw-profile flag.
"""
import os

# vLLM python (shared project env, same across all PACE nodes in r-ag117-0)
VLLM_PYTHON = "/storage/project/r-ag117-0/shared/agent_hpc/envs/vllm_clean/bin/python"

# HF hub snapshot paths for Blackwell node
# Weights live in ~/scratch/hf_home/hub/ (user-local scratch, not project storage)
_HF_HUB = os.path.expanduser("~/scratch/hf_home/hub")
_SNAPSHOT_72B = os.path.join(
    _HF_HUB,
    "models--Qwen--Qwen2.5-VL-72B-Instruct",
    "snapshots",
    "89c86200743eec961a297729e7990e8f2ddbc4c5",
)
_SNAPSHOT_32B = os.path.join(
    _HF_HUB,
    "models--Qwen--Qwen2.5-VL-32B-Instruct",
    "snapshots",
    "7cfb30d71a1f4f49a57592323337a4a4727301da",
)

# ---------------------------------------------------------------------------
# Blackwell profile — 4× NVIDIA RTX PRO 6000 Blackwell (96 GB each)
#   qwen_32b : GPUs 0-1  (2 × 96 GB = 192 GB)  tensor_parallel=2
#   qwen_72b : GPUs 2-3  (2 × 96 GB = 192 GB)  tensor_parallel=2
#   Both models run simultaneously; 72B fp16 ≈ 144 GB → 48 GB KV headroom.
# ---------------------------------------------------------------------------
MODELS_BLACKWELL = {
    "qwen_32b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_32B,
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
        ],
    },

    "qwen_72b": {
        "python_bin": VLLM_PYTHON,
        "model_name": _SNAPSHOT_72B,
        "port": 8001,

        "gpus": [2, 3],
        "tensor_parallel_size": 2,

        "gpu_memory_utilization": 0.97,
        # 2× 96 GB = 192 GB total; 72B fp16 weights ≈ 144 GB → ~48 GB KV headroom.
        # If vLLM OOMs during init, reduce max_model_len to 8192.
        "max_model_len": 16384,
        "load_timeout": 2700,

        "extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--dtype", "float16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
        ],
    },
}
