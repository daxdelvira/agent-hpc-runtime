"""
prefetch/megammap_stage.py — MegaMmapStagingExecutor: stage model shards into
the Hermes buffer pool via MegaMmap (external-system comparison for Option A).

Drop-in alternative to ModelCacheStagingExecutor for resource_type
"model_cache": instead of warming the OS page cache with sequential reads,
it runs the mm_model_preload shim (mega_mmap_integration/megammap_tests),
which pulls every safetensors shard through MegaMmap's bounded DRAM window
into the Hermes buffer pool.  The worker vLLM server must then be launched
with LD_PRELOAD=libhermes_posix.so so its reads hit the warm Hermes cache
(chemgraph_exp.py --megammap-stage injects that env into the orchestrator's
worker entry).

Requires a running Hermes daemon (hrun_start_runtime); mm_model_preload
exits with an error immediately if none is found — the task is then marked
FAILED, never silently skipped.

Environment expected by the shim binary (mpirun on PATH, mega_stack libs on
LD_LIBRARY_PATH) is supplied via `extra_env` by the caller; typically the
eval driver injects the values from ~/scratch/mega_stack/mega_env.sh.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask
from runtime.prefetch.model_cache_prefetch import list_model_shards

# libhermes_posix.so's RPATH pulls in spack GCC-12 libstdc++ (no CXXABI_1.3.15);
# preloading GCC-14's libstdc++ FIRST makes the linker keep the newer one.
# Same fix as mega_mmap_integration/common/hermes_env.py; override via env.
_GCC14_STDCXX = os.environ.get(
    "MEGA_GCC14_STDCXX",
    "/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/"
    "gcc-runtime-14.2.0-c6fqq2mtuqpgb4jcaolketxppd7bht37/lib/libstdc++.so.6",
)


def build_hermes_preload(interceptor: str, existing: str = "") -> str:
    """LD_PRELOAD value with GCC-14 libstdc++ before the Hermes interceptor."""
    parts = []
    if os.path.exists(_GCC14_STDCXX):
        parts.append(_GCC14_STDCXX)
    parts.append(interceptor)
    if existing:
        parts.append(existing)
    return ":".join(parts)


class MegaMmapStagingExecutor(PrefetchExecutor):
    """
    Parameters
    ----------
    model_paths : model key (== ResourceSpec.name) -> snapshot dir; an explicit
                  task.resource.path wins (same contract as page-cache staging).
    binary      : path to mm_model_preload.
    mpirun      : mpirun launcher (the shim is MPI-linked; run with -n 1).
    window      : MegaMmap DRAM window per shard, e.g. "4g".
    tx_type     : "seq" (known model -> deterministic prefetch) or "rand".
    extra_env   : env vars merged over os.environ for the shim subprocess.
    """

    executor_id = "megammap_staging"

    def __init__(
        self,
        model_paths: dict[str, str] | None = None,
        binary: str = "",
        mpirun: str = "mpirun",
        window: str = "4g",
        tx_type: str = "seq",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._model_paths = dict(model_paths or {})
        self._binary = binary
        self._mpirun = mpirun
        self._window = window
        self._tx_type = tx_type
        self._extra_env = dict(extra_env or {})
        self._procs: dict[str, subprocess.Popen] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def start(self, task: PrefetchTask) -> None:
        snapshot = task.resource.path or self._model_paths.get(task.resource.name, "")
        shards = list_model_shards(snapshot) if snapshot else []
        if not self._binary or not os.path.exists(self._binary):
            task.status = PrefetchStatus.FAILED
            task.error = f"mm_model_preload binary not found: {self._binary!r}"
            return
        if not shards:
            task.status = PrefetchStatus.FAILED
            task.error = (
                f"no shards for model '{task.resource.name}' (path={snapshot!r})"
            )
            return

        env = dict(os.environ)
        env.update(self._extra_env)
        cmd = [
            self._mpirun, "-n", "1", self._binary,
            "--shard-dir", snapshot,
            "--tx-type", self._tx_type,
            "--window", self._window,
        ]
        task.started_at = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except OSError as exc:
            task.status = PrefetchStatus.FAILED
            task.error = f"failed to launch mm_model_preload: {exc}"
            return
        task.status = PrefetchStatus.IN_PROGRESS
        total_bytes = sum(p.stat().st_size for p in shards)
        with self._lock:
            self._procs[task.task_id] = proc
            self._meta[task.task_id] = {
                "n_shards": len(shards),
                "shard_bytes": total_bytes,
                "t0": task.started_at,
            }
        print(
            f"[megammap_stage] staging {total_bytes/1e9:.1f} GB "
            f"({len(shards)} shards, tx={self._tx_type}, window={self._window}) "
            f"for {task.resource.name}",
            flush=True,
        )

    def cancel(self, task: PrefetchTask) -> bool:
        with self._lock:
            proc = self._procs.get(task.task_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        task.status = PrefetchStatus.CANCELLED
        task.cancelled_at = time.perf_counter()
        return True

    def is_complete(self, task: PrefetchTask) -> bool:
        with self._lock:
            proc = self._procs.get(task.task_id)
        if proc is None:
            return task.status not in (
                PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS
            )
        if proc.poll() is None:
            return False
        if task.status == PrefetchStatus.IN_PROGRESS:
            task.completed_at = time.perf_counter()
            task.status = (
                PrefetchStatus.COMPLETED if proc.returncode == 0
                else PrefetchStatus.FAILED
            )
        return True

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        with self._lock:
            proc = self._procs.get(task.task_id)
            meta = self._meta.get(task.task_id, {})
        if proc is None:
            return {"elapsed_s": 0.0, "success": False, "error": task.error or ""}
        out = ""
        try:
            out, _ = proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
        elapsed = (
            (task.completed_at or time.perf_counter()) - meta.get("t0", 0.0)
            if meta.get("t0") else 0.0
        )
        ok = proc.returncode == 0
        nbytes = meta.get("shard_bytes", 0) if ok else 0
        result: dict[str, Any] = {
            "elapsed_s": elapsed,
            # Shim reads each shard end-to-end, so staged bytes == shard file
            # bytes on success (0 reported on failure/cancel: actual partial
            # progress is not observable from outside the shim).
            "bytes_staged": nbytes,
            "gb_per_s": (nbytes / 1e9 / elapsed) if (ok and elapsed > 0) else 0.0,
            "n_shards": meta.get("n_shards", 0),
            "backend": "megammap",
            "tx_type": self._tx_type,
            "success": ok,
        }
        if not ok:
            tail = "\n".join((out or "").strip().splitlines()[-5:])
            result["error"] = f"mm_model_preload rc={proc.returncode}: {tail}"
            task.error = result["error"]
        print(
            f"[megammap_stage] {'staged' if ok else 'FAILED'} "
            f"{nbytes/1e9:.1f} GB in {elapsed:.1f}s for {task.resource.name}",
            flush=True,
        )
        return result

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        if wait:
            for proc in procs:
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
