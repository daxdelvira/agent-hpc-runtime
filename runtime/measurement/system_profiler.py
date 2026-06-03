"""
runtime/measurement/system_profiler.py
----------------------------------------
Continuous background profiler that samples CPU and GPU utilization
across ALL relevant process trees during an AtomAgents experiment run.

Captures every INTERVAL_S seconds:
  - System-wide CPU %  (all cores, all processes)
  - vLLM 72B process tree CPU %  (API server + engine workers)
  - vLLM 32B process tree CPU %  (same, once loaded)
  - Main orchestration process CPU %
  - Per-GPU utilization % and memory (via nvidia-smi)

This produces a time-series CSV that can answer:
  "How many CPU cores vs GPU compute does the agentic workflow need,
   and how does that vary between reasoning, tool execution, and idle?"

Usage
-----
    from runtime.measurement.system_profiler import SystemProfiler

    profiler = SystemProfiler(run_id="abc123")
    profiler.start()
    # ... run experiment ...
    profiler.stop()
    print(profiler.csv_path)
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# ── Columns ───────────────────────────────────────────────────────────────────

COLUMNS = [
    "t_rel_s",          # seconds since profiler.start()
    "wall_time",        # ISO-like timestamp string

    # System-wide (all cores, all processes)
    "sys_cpu_pct",      # % of all logical CPUs combined (100% = 1 core fully used)
    "sys_cpu_cores",    # number of logical CPU cores on the node

    # vLLM 72B process tree (API server PID + all children)
    "vllm_72b_cpu_pct",   # sum of cpu_percent() across the tree
    "vllm_72b_cpu_s",     # cumulative CPU seconds (user+system) of the tree
    "vllm_72b_rss_mb",    # RSS of the tree in MB
    "vllm_72b_n_procs",   # number of live processes in the tree

    # vLLM 32B process tree (available once prefetch completes)
    "vllm_32b_cpu_pct",
    "vllm_32b_cpu_s",
    "vllm_32b_rss_mb",
    "vllm_32b_n_procs",

    # Main orchestration process (the Python experiment script)
    "orch_cpu_pct",
    "orch_rss_mb",

    # Per-GPU (one column set per GPU, added dynamically at start)
    # gpu0_util_pct, gpu0_mem_used_mb, gpu0_power_w, ...
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_vllm_pids(port_72b: int = 8001, port_32b: int = 8002) -> dict[str, int]:
    """
    Locate vLLM API server PIDs by matching command-line --port argument.
    Returns e.g. {"vllm_72b": 12345, "vllm_32b": 67890}.
    Missing entries are omitted (no error).
    """
    if not _HAVE_PSUTIL:
        return {}
    result: dict[str, int] = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "vllm.entrypoints.openai.api_server" not in cmd:
                continue
            parts = cmd.split()
            for i, p in enumerate(parts):
                if p == "--port" and i + 1 < len(parts):
                    port = int(parts[i + 1])
                    if port == port_72b:
                        result["vllm_72b"] = proc.info["pid"]
                    elif port == port_32b:
                        result["vllm_32b"] = proc.info["pid"]
        except Exception:
            pass
    return result


def _sample_tree(root_pid: int) -> tuple[float, float, float, int]:
    """
    Return (cpu_pct, cpu_s, rss_mb, n_procs) for a process and all descendants.
    cpu_pct uses psutil's interval=None (measures since last call on each process).
    """
    if not _HAVE_PSUTIL:
        return 0.0, 0.0, 0.0, 0
    try:
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0, 0.0, 0.0, 0

    total_cpu_pct = 0.0
    total_cpu_s = 0.0
    total_rss = 0.0
    n = 0
    for p in procs:
        try:
            total_cpu_pct += p.cpu_percent(interval=None)
            ct = p.cpu_times()
            total_cpu_s += ct.user + ct.system
            total_rss += p.memory_info().rss
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total_cpu_pct, total_cpu_s, total_rss / 1024**2, n


def _query_nvidia_smi() -> list[dict[str, Any]]:
    """
    Run nvidia-smi and return per-GPU stats.
    Returns [] if nvidia-smi is unavailable or fails.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "index":       int(parts[0]),
                "util_pct":    float(parts[1]),
                "mem_util_pct": float(parts[2]),
                "mem_used_mb": float(parts[3]),
                "mem_total_mb": float(parts[4]),
                "power_w":     float(parts[5]) if parts[5] != "[N/A]" else -1.0,
            })
        except (ValueError, IndexError):
            pass
    return gpus


# ── Main class ─────────────────────────────────────────────────────────────────

class SystemProfiler:
    """
    Continuous background profiler.  Call start() before the experiment,
    stop() after.  Writes a time-series CSV to results_dir/.

    Parameters
    ----------
    run_id      : experiment run identifier (used in CSV filename)
    results_dir : directory to write the CSV (created if missing)
    interval_s  : sampling interval in seconds (default 3)
    port_72b    : vLLM 72B server port for PID discovery
    port_32b    : vLLM 32B server port for PID discovery
    """

    def __init__(
        self,
        run_id: str = "unknown",
        results_dir: str = "results",
        interval_s: float = 3.0,
        port_72b: int = 8001,
        port_32b: int = 8002,
    ) -> None:
        self.run_id = run_id
        self.interval_s = interval_s
        self.port_72b = port_72b
        self.port_32b = port_32b

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = str(Path(results_dir) / f"system_profile_{run_id}.csv")

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background sampling thread."""
        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="system_profiler",
        )
        self._thread.start()
        print(f"[profiler] System profiler started → {self.csv_path}", flush=True)

    def stop(self, timeout: float = 15.0) -> None:
        """Stop the background thread and flush the CSV."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        print(f"[profiler] System profiler stopped. CSV: {self.csv_path}", flush=True)

    # ── background thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        # Discover vLLM PIDs (retry for up to 10s in case they haven't started yet)
        vllm_pids: dict[str, int] = {}
        for _ in range(5):
            vllm_pids = _find_vllm_pids(self.port_72b, self.port_32b)
            if vllm_pids:
                break
            time.sleep(2)

        if not vllm_pids:
            print("[profiler] WARNING: no vLLM servers found on ports "
                  f"{self.port_72b}/{self.port_32b}. CPU columns will be zero.",
                  flush=True)

        orch_pid = os.getpid()

        # Prime cpu_percent counters (first call always returns 0.0)
        if _HAVE_PSUTIL:
            psutil.cpu_percent(interval=None)
            for pid in vllm_pids.values():
                _sample_tree(pid)
            try:
                psutil.Process(orch_pid).cpu_percent(interval=None)
            except Exception:
                pass

        # Probe GPU count once
        gpus_initial = _query_nvidia_smi()
        gpu_indices = [g["index"] for g in gpus_initial]

        # Build CSV columns dynamically (add GPU columns)
        columns = list(COLUMNS)
        for idx in gpu_indices:
            columns += [
                f"gpu{idx}_util_pct",
                f"gpu{idx}_mem_util_pct",
                f"gpu{idx}_mem_used_mb",
                f"gpu{idx}_mem_total_mb",
                f"gpu{idx}_power_w",
            ]

        with open(self.csv_path, "w", newline="", buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()

            while not self._stop_event.is_set():
                t_rel = time.perf_counter() - self._t0
                row: dict[str, Any] = {
                    "t_rel_s":      f"{t_rel:.2f}",
                    "wall_time":    time.strftime("%H:%M:%S"),
                    "sys_cpu_cores": psutil.cpu_count(logical=True) if _HAVE_PSUTIL else -1,
                }

                # System-wide CPU (all cores)
                row["sys_cpu_pct"] = psutil.cpu_percent(interval=None) if _HAVE_PSUTIL else -1

                # vLLM process trees
                for name in ("vllm_72b", "vllm_32b"):
                    pid = vllm_pids.get(name)
                    if pid is None:
                        # Try to discover now (32B may have loaded since start)
                        new_pids = _find_vllm_pids(self.port_72b, self.port_32b)
                        if name in new_pids:
                            vllm_pids[name] = new_pids[name]
                            pid = new_pids[name]
                            # Prime the counter for the new process
                            _sample_tree(pid)
                    if pid is not None:
                        cpu_pct, cpu_s, rss_mb, n = _sample_tree(pid)
                        row[f"{name}_cpu_pct"]  = f"{cpu_pct:.2f}"
                        row[f"{name}_cpu_s"]    = f"{cpu_s:.2f}"
                        row[f"{name}_rss_mb"]   = f"{rss_mb:.1f}"
                        row[f"{name}_n_procs"]  = n
                    else:
                        row[f"{name}_cpu_pct"]  = 0
                        row[f"{name}_cpu_s"]    = 0
                        row[f"{name}_rss_mb"]   = 0
                        row[f"{name}_n_procs"]  = 0

                # Orchestration process
                try:
                    orch = psutil.Process(orch_pid) if _HAVE_PSUTIL else None
                    if orch:
                        row["orch_cpu_pct"] = f"{orch.cpu_percent(interval=None):.2f}"
                        row["orch_rss_mb"]  = f"{orch.memory_info().rss / 1024**2:.1f}"
                    else:
                        row["orch_cpu_pct"] = -1
                        row["orch_rss_mb"]  = -1
                except Exception:
                    row["orch_cpu_pct"] = -1
                    row["orch_rss_mb"]  = -1

                # GPU stats
                for g in _query_nvidia_smi():
                    idx = g["index"]
                    row[f"gpu{idx}_util_pct"]      = g["util_pct"]
                    row[f"gpu{idx}_mem_util_pct"]  = g["mem_util_pct"]
                    row[f"gpu{idx}_mem_used_mb"]   = g["mem_used_mb"]
                    row[f"gpu{idx}_mem_total_mb"]  = g["mem_total_mb"]
                    row[f"gpu{idx}_power_w"]        = g["power_w"]

                writer.writerow(row)
                self._stop_event.wait(self.interval_s)
