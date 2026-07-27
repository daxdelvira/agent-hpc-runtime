#!/usr/bin/env python3
"""
bench_sleep_wake.py — vLLM sleep/wake latency microbenchmark (track (b) prep).

Motivation: the stall taxonomy (results/eval_q1_q4/eval_stall_taxonomy.csv)
shows the swap-family exposed stall is dominated by no_window engine bring-up
(~200 s/trial chemgraph_swap, ~380 s/trial screen first-boot+repeat).  Sleep
mode converts a cold re-boot into a wake: level 1 keeps weights in CPU RAM
(wake = H2D copy, expected 6-15 s for 145 GB over PCIe), level 2 discards
weights (wake = re-read from page cache, expected 30-50 s at staging
bandwidth).  This benchmark measures, on real hardware, for one model config:

    cold boot -> [sanity gen] -> sleep(L) -> VRAM check -> wake -> [sanity gen]
    repeated for level 1 and level 2, N cycles each.

Output: JSON lines to stdout + results/bench_sleep_wake_<model>_<host>.json.

Requires: a GPU allocation, vllm >= 0.7 (installed: 0.17.1 in vllm_clean),
and the dev endpoints (/sleep, /wake_up, /is_sleeping) which need
VLLM_SERVER_DEV_MODE=1 — exported here at server launch.

Usage (on a compute node):
    python experiments/bench_sleep_wake.py --model qwen_72b_instruct
    python experiments/bench_sleep_wake.py --model qwen_32b_vl --cycles 3
    python experiments/bench_sleep_wake.py --model qwen_72b_instruct \
        --levels 2            # level-2 only (RAM-constrained node)

NOT part of the eval tree — measurement only, no trace/meta artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model_configs import MODELS_CHEMGRAPH_SWAP  # noqa: E402


def _url(port: int, path: str) -> str:
    return f"http://localhost:{port}{path}"


def _post(port: int, path: str, timeout: float = 600.0) -> None:
    req = urllib.request.Request(_url(port, path), data=b"", method="POST")
    urllib.request.urlopen(req, timeout=timeout).read()


def _get_json(port: int, path: str, timeout: float = 30.0):
    with urllib.request.urlopen(_url(port, path), timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port: int, timeout: float) -> float:
    """Poll /v1/models until the server answers; returns elapsed seconds."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            _get_json(port, "/v1/models", timeout=5)
            return time.perf_counter() - t0
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2.0)
    raise TimeoutError(f"server on :{port} not ready after {timeout}s")


def sanity_generate(port: int, served_name: str, timeout: float = 120.0) -> float:
    """One short completion; returns latency. Proves the engine truly serves."""
    body = json.dumps({
        "model": served_name,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 4,
    }).encode()
    req = urllib.request.Request(
        _url(port, "/v1/chat/completions"), data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    urllib.request.urlopen(req, timeout=timeout).read()
    return time.perf_counter() - t0


def gpu_mem_used_mib(gpus: list[int]) -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False).stdout
    by_idx = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(",")
        by_idx[int(idx)] = int(mem)
    return [by_idx.get(g, -1) for g in gpus]


def host_ram_used_gib() -> float:
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def launch_server(cfg: dict) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg["gpus"])
    env["VLLM_SERVER_DEV_MODE"] = "1"   # exposes /sleep /wake_up /is_sleeping
    cmd = [
        cfg["python_bin"], "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model_name"],
        "--port", str(cfg["port"]),
        "--tensor-parallel-size", str(cfg["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--enable-sleep-mode",
        *cfg.get("extra_args", []),
    ]
    print(json.dumps({"event": "launch", "cmd": " ".join(cmd)}), flush=True)
    return subprocess.Popen(cmd, env=env,
                            stdout=open("bench_sleep_wake_server.log", "ab"),
                            stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen_72b_instruct",
                    choices=sorted(MODELS_CHEMGRAPH_SWAP))
    ap.add_argument("--cycles", type=int, default=2,
                    help="sleep/wake cycles per level")
    ap.add_argument("--levels", default="1,2",
                    help="comma-separated sleep levels to test")
    args = ap.parse_args()

    cfg = MODELS_CHEMGRAPH_SWAP[args.model]
    port = cfg["port"]
    served = next((cfg["extra_args"][i + 1]
                   for i, a in enumerate(cfg["extra_args"])
                   if a == "--served-model-name"), cfg["model_name"])
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    results: list[dict] = []

    def rec(event: str, **kw):
        row = {"event": event, "model": args.model, "t": time.time(), **kw}
        results.append(row)
        print(json.dumps(row), flush=True)

    proc = launch_server(cfg)
    try:
        t_boot = wait_ready(port, cfg.get("load_timeout", 3600))
        rec("cold_boot", boot_s=round(t_boot, 1),
            gpu_mem_mib=gpu_mem_used_mib(cfg["gpus"]),
            host_ram_gib=round(host_ram_used_gib(), 1))
        rec("sanity", gen_s=round(sanity_generate(port, served), 2))

        for level in levels:
            for cycle in range(args.cycles):
                t0 = time.perf_counter()
                _post(port, f"/sleep?level={level}")
                # poll until asleep
                while not _get_json(port, "/is_sleeping").get("is_sleeping"):
                    time.sleep(0.5)
                sleep_s = time.perf_counter() - t0
                time.sleep(3)   # let nvidia-smi settle
                rec("sleep", level=level, cycle=cycle,
                    sleep_s=round(sleep_s, 2),
                    gpu_mem_mib=gpu_mem_used_mib(cfg["gpus"]),
                    host_ram_gib=round(host_ram_used_gib(), 1))

                t0 = time.perf_counter()
                _post(port, "/wake_up", timeout=1800)
                wake_s = time.perf_counter() - t0
                gen_s = sanity_generate(port, served)
                rec("wake", level=level, cycle=cycle,
                    wake_s=round(wake_s, 2),
                    first_gen_s=round(gen_s, 2),
                    wake_plus_gen_s=round(wake_s + gen_s, 2),
                    gpu_mem_mib=gpu_mem_used_mib(cfg["gpus"]),
                    host_ram_gib=round(host_ram_used_gib(), 1))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        out = (_HERE.parent / "results" /
               f"bench_sleep_wake_{args.model}_{socket.gethostname()}.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"event": "done", "results_file": str(out)}),
              flush=True)


if __name__ == "__main__":
    main()
