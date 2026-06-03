"""
prefetch/mace_prefetch.py — MacePrefetchExecutor: preloads MACE model into memory.

Loads the MACE calculator into a module-level cache in a background thread so
that run_ase() can find it ready on arrival instead of loading from disk.

Integration with ChemGraph
--------------------------
Add one import guard and four lines to run_ase() in ase_tools.py:

    from runtime.prefetch.mace_prefetch import get_cached_calculator

    def run_ase(smiles: str, ...):
        ...
        calc = get_cached_calculator(model_path)   # returns cached or None
        if calc is None:
            calc = MACECalculator(model_paths=model_path, device=device, ...)
        ...

This is the only invasive change to ChemGraph source; it is safe to gate behind
`if os.environ.get("RUNTIME_ENABLED")` if you want zero-change baseline runs.

Cancellation
------------
MACE loads are relatively fast (~30-60s from SSD, 2-5min from NFS) and the
MACECalculator constructor does not expose a cancellation mechanism. We try to
cancel the Future before PyTorch starts loading; if the future is already
running we mark the task CANCELLED but let it finish (the cache entry will
just go unused). cancellation_safe=True in the ResourceSpec means the scheduler
will mark it WASTED (not leave it hanging) if cancelled after completion.

Local testing without MACE installed
-------------------------------------
MacePrefetchExecutor handles ImportError gracefully: if `mace` is not installed
it sets task.status = FAILED with a descriptive error. The rest of the pipeline
continues unaffected. Use SimulatedPrefetchExecutor for local dev.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask


# Module-level cache shared between MacePrefetchExecutor and MaceCalc.
# Key: (calculator_type, model_str) tuple.  Value: loaded MACECalculator instance.
_MACE_CACHE: dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()


def _mace_cache_key(calculator_type: str, model: Any) -> tuple[str, str]:
    """Canonical cache key: (calculator_type, str(model or 'default'))."""
    return (calculator_type, str(model) if model is not None else "default")


def get_cached_mace_calculator(calculator_type: str, model: Any) -> Any | None:
    """
    Return a pre-loaded MACECalculator for the given spec, or None if not cached.
    Pops the entry — each cached instance is consumed exactly once.

    Used in MaceCalc.get_calculator() to bypass the model load when prefetch
    has already completed.

    Parameters
    ----------
    calculator_type : str   e.g. "mace_mp", "mace_off", "mace_anicc"
    model           : str | Path | None   model variant or None for default
    """
    key = _mace_cache_key(calculator_type, model)
    with _CACHE_LOCK:
        return _MACE_CACHE.pop(key, None)


# Legacy alias — kept for any existing callers that use the path-based API.
def get_cached_calculator(model_path: str) -> Any | None:
    """Deprecated: use get_cached_mace_calculator(). Returns None for unknown paths."""
    return None


def cache_size() -> int:
    """Return number of cached calculators (for testing/monitoring)."""
    with _CACHE_LOCK:
        return len(_MACE_CACHE)


class MacePrefetchExecutor(PrefetchExecutor):
    """
    Prefetch executor that preloads a MACE calculator into _MACE_CACHE.

    The model path is taken from task.resource.path. If the path is None,
    the task fails immediately (path must be set in the ResourceSpec before
    scheduling).

    Parameters
    ----------
    device  : PyTorch device string passed to MACECalculator ("cpu", "cuda", ...).
              Default "cpu" for safety; override to "cuda" on cluster nodes.
    cache   : dict to store loaded calculators. Defaults to module-level
              _MACE_CACHE so run_ase() can find them.
    """

    executor_id = "mace_prefetch"

    def __init__(
        self,
        device: str = "cpu",
        cache: dict[tuple, Any] | None = None,
    ) -> None:
        self._device = device
        self._cache = cache if cache is not None else _MACE_CACHE
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mace_prefetch",
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def start(self, task: PrefetchTask) -> None:
        if not task.resource.path:
            task.status = PrefetchStatus.FAILED
            task.error = "resource.path is None; cannot load MACE model"
            return

        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        future = self._pool.submit(self._load_mace, task)
        with self._lock:
            self._futures[task.task_id] = future

    def cancel(self, task: PrefetchTask) -> bool:
        """
        Try to cancel the Future before loading starts. If already running,
        mark CANCELLED and let the load finish (result goes into cache but
        will be ignored by run_ase() since task is CANCELLED).
        """
        with self._lock:
            future = self._futures.get(task.task_id)

        if future is not None and future.cancel():
            # Cancelled before the thread started
            task.status = PrefetchStatus.CANCELLED
            task.cancelled_at = time.perf_counter()
            return True

        # Already running — mark intent; let load finish
        task.status = PrefetchStatus.CANCELLED
        task.cancelled_at = time.perf_counter()
        return False

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
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cache key helpers  (same logic as get_cached_mace_calculator above)
    # ------------------------------------------------------------------

    @staticmethod
    def _key_from_resource(resource) -> tuple[str, str]:
        """Derive (calculator_type, model_str) from a ResourceSpec.

        Interprets resource.name as "<calc_type>:<model>" or a legacy name
        like "mace-mp-0".  Defaults to mace_mp with the default model.
        """
        name = resource.name or ""
        if ":" in name:
            calc_type, model_str = name.split(":", 1)
        elif "mace_mp" in name.lower() or "mace-mp" in name.lower():
            calc_type, model_str = "mace_mp", "default"
        elif "mace_off" in name.lower() or "mace-off" in name.lower():
            calc_type, model_str = "mace_off", "default"
        elif "mace_ani" in name.lower() or "mace-ani" in name.lower():
            calc_type, model_str = "mace_anicc", "default"
        else:
            calc_type, model_str = "mace_mp", "default"
        return calc_type.strip(), model_str.strip()

    def _load_mace(self, task: PrefetchTask) -> dict[str, Any]:
        """
        Background thread: load a MACE calculator using the same mace_mp /
        mace_off entry points that ChemGraph uses, then store in _MACE_CACHE.

        The cache key is (calculator_type, model_str) — the same key that
        MaceCalc.get_calculator() checks before doing its own load.
        """
        calc_type, model_str = self._key_from_resource(task.resource)
        model_arg: Any = None if model_str == "default" else model_str
        cache_key = _mace_cache_key(calc_type, model_arg)
        t0 = time.perf_counter()

        try:
            if calc_type == "mace_mp":
                from mace.calculators import mace_mp  # type: ignore
                calc = mace_mp(
                    model=model_arg,
                    device=self._device,
                    default_dtype="float32",
                )
            elif calc_type == "mace_off":
                from mace.calculators import mace_off  # type: ignore
                calc = mace_off(
                    model=model_arg,
                    device=self._device,
                    default_dtype="float32",
                )
            elif calc_type == "mace_anicc":
                from mace.calculators import mace_anicc  # type: ignore
                calc = mace_anicc(device=self._device, model_path=model_arg)
            else:
                raise ValueError(f"Unknown MACE calculator type: {calc_type!r}")

        except ImportError:
            task.status = PrefetchStatus.FAILED
            task.error = "mace package not installed; run: pip install mace-torch"
            return {"elapsed_s": 0.0, "success": False, "error": task.error}
        except Exception as exc:
            task.status = PrefetchStatus.FAILED
            task.error = str(exc)
            return {"elapsed_s": time.perf_counter() - t0, "success": False, "error": str(exc)}

        elapsed = time.perf_counter() - t0
        task.completed_at = time.perf_counter()

        if task.status not in (PrefetchStatus.CANCELLED,):
            with _CACHE_LOCK:
                self._cache[cache_key] = calc
            task.status = PrefetchStatus.COMPLETED

        return {"elapsed_s": elapsed, "success": True, "cache_key": str(cache_key)}
