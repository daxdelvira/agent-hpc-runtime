"""
prefetch/model_cache_prefetch.py — Page-cache staging for vLLM model weights.

Option A ("stage-during-planning") for swap workflows.  When the planner and
worker share the GPU pool (tp=4) they cannot co-reside, so the worker's *GPU*
load cannot begin until the planner releases the GPUs.  But the worker's weight
*I/O* (reading ~136 GB of safetensors shards off Lustre) does NOT need the GPU
and can run fully concurrently with GPU-bound planner inference.

This executor reads the worker's shard files into the OS page cache in the
background while the planner is still running.  When the real vLLM swap load
runs afterwards, it reads from warm cache instead of cold Lustre — cutting the
read portion of the load out of the critical path.

Measured on this cluster (Qwen2.5-72B, 37 shards, 136 GB, Lustre):
    cold read  ~0.9 GB/s  -> ~148 s to stage the whole model
    warm read  ~8.2 GB/s  -> ~17 s
so the I/O ceiling that staging can hide is ~130 s.

Unlike ModelPrefetchExecutor (vLLM load, cancellation_safe=False), page-cache
staging IS safely interruptible: cancel() just stops reading.  Staging also
does NOT stop the running planner and does NOT touch the GPU, so it overlaps
freely with the planner phase.  Leaving it running through the swap is fine and
even beneficial: it reads the same blocks vLLM's loader will read, in order,
staying just ahead — the page cache dedupes, so vLLM gets warm hits.

Helpers
-------
list_model_shards(snapshot_dir) -> [Path, ...]
evict_model_cache(snapshot_dir) -> (n_files, bytes) — posix_fadvise DONTNEED
"""
from __future__ import annotations

import glob
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask

# Weight-shard globs, in the order vLLM's loader consumes them.
_SHARD_GLOBS = ("*.safetensors", "*.bin")


def list_model_shards(snapshot_dir: str) -> list[Path]:
    """Return sorted weight-shard files under a HF snapshot dir (symlinks resolved)."""
    out: list[Path] = []
    for pat in _SHARD_GLOBS:
        out.extend(Path(p) for p in glob.glob(os.path.join(snapshot_dir, pat)))
    # Resolve HF-cache symlinks (snapshots/<rev>/*.safetensors -> ../../blobs/<sha>)
    # so posix_fadvise / os.read hit the real blob file.
    return sorted({p.resolve() for p in out})


def evict_model_cache(snapshot_dir: str) -> tuple[int, int]:
    """
    Drop the model's shards from the OS page cache via posix_fadvise(DONTNEED).

    User-space, no root required (works on the Lustre client cache here).  Used
    before a run so every run starts from a genuinely cold cache, otherwise a
    warm cache from a previous run makes both the cold baseline and the staging
    benefit meaningless.

    Returns (n_files_evicted, total_bytes).
    """
    n = 0
    total = 0
    for path in list_model_shards(snapshot_dir):
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            size = os.fstat(fd).st_size
            # Flush any dirty pages first so DONTNEED can drop them, then evict.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
            total += size
        finally:
            os.close(fd)
    return n, total


class ModelCacheStagingExecutor(PrefetchExecutor):
    """
    Warm the OS page cache with a model's weight shards during a GPU-occupied
    compute window (the planner phase of a swap workflow).

    Parameters
    ----------
    model_paths : dict mapping model key (== ResourceSpec.name) -> snapshot dir.
                  If a task's resource carries an explicit .path, that wins.
    read_bufsize : bytes per os.read() call.
    max_workers : parallel shard readers.  Lustre stripes files across OSTs, so a
                  few concurrent streams reach higher aggregate bandwidth and warm
                  the full model faster (more likely to finish inside the planner
                  window).  Each worker reads a disjoint slice of the shard list.
    """

    executor_id = "model_cache_staging"

    def __init__(
        self,
        model_paths: dict[str, str] | None = None,
        read_bufsize: int = 32 * 1024 * 1024,
        max_workers: int = 4,
    ) -> None:
        self._model_paths = dict(model_paths or {})
        self._bufsize = read_bufsize
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers + 1,  # +1 coordinator future
            thread_name_prefix="cache_stage",
        )
        self._futures: dict[str, Future] = {}
        self._stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _resolve_shards(self, task: PrefetchTask) -> list[Path]:
        snapshot = task.resource.path or self._model_paths.get(task.resource.name, "")
        if not snapshot:
            return []
        return list_model_shards(snapshot)

    def start(self, task: PrefetchTask) -> None:
        shards = self._resolve_shards(task)
        if not shards:
            task.status = PrefetchStatus.FAILED
            task.error = (
                f"no shards for model '{task.resource.name}' "
                f"(path={task.resource.path!r})"
            )
            return

        stop = threading.Event()
        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        future = self._pool.submit(self._stage_all, task, shards, stop)
        with self._lock:
            self._futures[task.task_id] = future
            self._stops[task.task_id] = stop

    def cancel(self, task: PrefetchTask) -> bool:
        """Page-cache staging is safely interruptible: signal the readers to stop."""
        with self._lock:
            stop = self._stops.get(task.task_id)
        if stop is not None:
            stop.set()
        task.status = PrefetchStatus.CANCELLED
        task.cancelled_at = time.perf_counter()
        return True

    def is_complete(self, task: PrefetchTask) -> bool:
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            return future.done()
        return task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS)

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            try:
                return future.result(timeout=5.0)
            except Exception as exc:
                return {"elapsed_s": 0.0, "success": False, "error": str(exc)}
        elapsed = (
            (task.completed_at - task.started_at)
            if task.started_at is not None and task.completed_at is not None
            else 0.0
        )
        return {"elapsed_s": elapsed, "success": task.status == PrefetchStatus.COMPLETED}

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            for stop in self._stops.values():
                stop.set()
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------

    def _read_into_cache(self, path: Path, stop: threading.Event) -> int:
        """Sequentially read one shard so its pages land in the page cache."""
        nbytes = 0
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            return 0
        try:
            # Hint the kernel to read ahead aggressively.
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass
            while not stop.is_set():
                chunk = os.read(fd, self._bufsize)
                if not chunk:
                    break
                nbytes += len(chunk)
        finally:
            os.close(fd)
        return nbytes

    def _stage_all(
        self, task: PrefetchTask, shards: list[Path], stop: threading.Event
    ) -> dict[str, Any]:
        """Read every shard into page cache using a small pool of reader threads."""
        t0 = time.perf_counter()
        total = 0
        # Round-robin shards across worker slices so concurrent streams hit
        # different files (Lustre stripes -> higher aggregate bandwidth).
        slices = [shards[i :: self._max_workers] for i in range(self._max_workers)]

        def _run_slice(sl: list[Path]) -> int:
            n = 0
            for p in sl:
                if stop.is_set():
                    break
                n += self._read_into_cache(p, stop)
            return n

        try:
            with ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="cache_rd"
            ) as inner:
                for n in inner.map(_run_slice, slices):
                    total += n

            task.completed_at = time.perf_counter()
            elapsed = task.completed_at - t0
            if stop.is_set():
                if task.status == PrefetchStatus.IN_PROGRESS:
                    task.status = PrefetchStatus.CANCELLED
            elif task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED

            gbps = (total / 1e9 / elapsed) if elapsed > 0 else 0.0
            print(
                f"[cache_stage] staged {total/1e9:.1f} GB in {elapsed:.1f}s "
                f"({gbps:.2f} GB/s) for {task.resource.name} "
                f"[{task.status.value}]",
                flush=True,
            )
            return {
                "elapsed_s": elapsed,
                "bytes_staged": total,
                "gb_per_s": gbps,
                "n_shards": len(shards),
                "success": True,
                "cancelled": stop.is_set(),
            }
        except Exception as exc:
            if task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.FAILED
            task.error = str(exc)
            return {
                "elapsed_s": time.perf_counter() - t0,
                "bytes_staged": total,
                "success": False,
                "error": str(exc),
            }
