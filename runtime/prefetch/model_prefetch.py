"""
prefetch/model_prefetch.py — ModelPrefetchExecutor for vLLM model servers.

Wraps ModelOrchestrator.start_model_measured() in a background thread so
model weight loading overlaps with ongoing agent compute.

Key design constraint: vLLM weight loading is NOT safely interruptible once
started (half-loaded shards leave the GPU in an undefined state). When the
divergence guard calls cancel(), we:
  1. Record the cancellation intent in _cancelled_tasks
  2. Return False (cannot interrupt the load)
  3. Let the background load finish; mark the task WASTED when it completes

The scheduler already accounts for WASTED tasks in the overlap metrics.

Swap to this executor from SimulatedPrefetchExecutor by changing one line
in the cluster script constructor:

    # Before (simulated):
    executor = SimulatedPrefetchExecutor()

    # After (real):
    from atomagents.runtime.model_orchestrator import ModelOrchestrator
    from atomagents.runtime.model_config import MODELS
    executor = ModelPrefetchExecutor(ModelOrchestrator(MODELS))

FakeModelOrchestrator lets you run the full timing pipeline locally
with configurable sleep-based "load" times — no GPUs required.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from runtime.prefetch.base import PrefetchExecutor, PrefetchStatus, PrefetchTask


# How long the proactive-swap path waits for a handover that is ALREADY UNDER
# WAY before it hands the GPUs to the residency actor.  Read off the
# orchestrator's own constants rather than picked:
#   workloads/AtomAgents/atomagents/runtime/model_orchestrator.py
#     :311  stop_model(name, wait_s: float = 30.0)   SIGINT grace
#     :335  process.wait(timeout=10)                 SIGKILL reap
#     :352  drain_timeout = 240                      VRAM-drain poll ceiling
# 30 + 10 + 240 = 280 s, rounded up to 300.  The 240 s term is not nominal: all
# three failed Tandem trials printed a drain that hit exactly that ceiling —
# "GPU VRAM drained in 242.4s" (…_tp2/tandem/t01__20260901-151756__f9b64ab/
# stdout.log:394), 240.2s (…_aligned/tandem/t01__20260901-144241__f9b64ab/
# stdout.log:436) and 240.3s (…/t02__20260901-152547__f9b64ab/stdout.log:433).
# Beyond this the incumbent is not handing over, and forcing it is the actor's
# decision to make, not this executor's.
#
# This bound is used ONLY on the residency-actor path.  The legacy path's 600 s
# is left exactly where it is.
HANDOVER_GRACE_S = 300.0


class ProactiveSwapDeclined(RuntimeError):
    """The swap was not attempted, and nothing was touched.

    Raised instead of evicting an incumbent that is still COMING UP.  A
    prefetch is worth less than the step the workflow is currently waiting on,
    and this is the failure mode that reads as "we did nothing" in the trace.
    """


# ---------------------------------------------------------------------------
# FakeModelOrchestrator — for local testing and demos
# ---------------------------------------------------------------------------

class FakeModelOrchestrator:
    """
    Drop-in replacement for ModelOrchestrator that sleeps instead of
    loading real model weights.

    Parameters
    ----------
    load_times : dict mapping model name → simulated load seconds.
                 Unknown models default to 1.0s.
    failure_models : set of model names that should raise RuntimeError.
                     Used to test error handling paths.
    wake_times : dict mapping model name → simulated wake seconds
                 (sleep/wake swap arm; default 0.0).
    models : optional dict mapping model name → config dict (e.g.
             {"gpus": [0, 1]}), mirroring ModelOrchestrator.models so
             GPU-overlap logic can be tested.

    Sleep/wake state (mirrors the real orchestrator's sleep-mode API):
      sleeping        — set of model names currently slept (weights "in CPU RAM")
      calls           — chronological op log, e.g. ("sleep_model", "planner", 1),
                        for sequencing assertions in tests
      last_transition — model name → "sleep_wake" | "cold_boot"
    """

    def __init__(
        self,
        load_times: dict[str, float] | None = None,
        failure_models: set[str] | None = None,
        wake_times: dict[str, float] | None = None,
        models: dict | None = None,
    ) -> None:
        self.load_times = load_times or {}
        self.failure_models = failure_models or set()
        self.wake_times = wake_times or {}
        self.models = models or {}
        self.processes: dict[str, object] = {}   # mimics ModelOrchestrator.processes
        self.sleeping: set[str] = set()
        self.calls: list[tuple] = []
        self.last_transition: dict[str, str] = {}

    def start_model_measured(
        self,
        name: str,
        metrics=None,
    ) -> float:
        """Simulate model loading: sleep for load_times[name] seconds."""
        self.calls.append(("start_model_measured", name))
        if name in self.failure_models:
            raise RuntimeError(f"FakeModelOrchestrator: simulated failure for {name}")
        elapsed = self.load_times.get(name, 1.0)
        time.sleep(elapsed)
        self.processes[name] = object()   # non-None sentinel
        self.sleeping.discard(name)       # a fresh boot is awake
        self.last_transition[name] = "cold_boot"
        return elapsed

    def stop_model(self, name: str, wait_s: float = 5.0) -> None:
        self.calls.append(("stop_model", name))
        self.processes.pop(name, None)
        self.sleeping.discard(name)

    def wait_until_ready(self, name: str, timeout: int = 60) -> None:
        pass   # already "ready" after start_model_measured returns

    def ensure_all_models_ready(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Sleep/wake API (mirrors ModelOrchestrator's sleep-mode methods)
    # ------------------------------------------------------------------

    def get_running_model(self) -> str | None:
        """First model with a live process (sleeping or not) — mirrors the
        real orchestrator, whose slept processes stay alive."""
        return next(iter(self.processes), None)

    def is_sleeping(self, name: str) -> bool:
        return name in self.processes and name in self.sleeping

    def sleep_model(self, name: str, level: int = 1, timeout: float = 900.0) -> float:
        self.calls.append(("sleep_model", name, level))
        if name in self.failure_models:
            raise RuntimeError(f"FakeModelOrchestrator: simulated sleep failure for {name}")
        if name not in self.processes or name in self.sleeping:
            return 0.0
        self.sleeping.add(name)
        return 0.0

    def wake_model(self, name: str, timeout: float = 900.0) -> float:
        self.calls.append(("wake_model", name))
        if name not in self.processes:
            raise RuntimeError(f"Cannot wake {name}: no live server process.")
        elapsed = self.wake_times.get(name, 0.0)
        if elapsed:
            time.sleep(elapsed)
        self.sleeping.discard(name)
        self.last_transition[name] = "sleep_wake"
        return elapsed

    def wait_until_serving(self, name: str, timeout: int = 60) -> None:
        self.calls.append(("wait_until_serving", name))
        if name not in self.processes:
            raise RuntimeError(f"{name} has no server process.")
        if name in self.sleeping:
            raise RuntimeError(f"{name} is asleep — wake_model() it first.")


# ---------------------------------------------------------------------------
# ModelPrefetchExecutor
# ---------------------------------------------------------------------------

class ModelPrefetchExecutor(PrefetchExecutor):
    """
    Prefetch executor that loads vLLM model servers in a background thread.

    - One active load at a time (max_concurrent=1 default; vLLM's tensor
      parallelism claims all assigned GPUs, so concurrent loads would conflict).
    - Cancellation records intent but does NOT interrupt the load thread.
    - Background thread marks the task WASTED (not COMPLETED) if cancelled.
    """

    executor_id = "model_prefetch"

    def __init__(
        self,
        orchestrator,   # ModelOrchestrator | FakeModelOrchestrator
        max_concurrent: int = 1,
        probes=None,    # ClusterProbes | None — if set, snapshots before/after load
        stop_wasted_models: bool = False,
        evict_conflicting: bool = False,
        sleep_wake: bool = False,
        residency_actor=None,   # runtime.residency.model_actor.VllmModelActor
        handover_grace_s: float = HANDOVER_GRACE_S,
    ) -> None:
        self._orchestrator = orchestrator
        self._probes = probes
        # Only consulted on the residency-actor proactive-swap path; the legacy
        # path keeps its own hard-coded 600 s. Constructor-visible so the tests
        # can drive the timeout without sleeping for five minutes.
        self._handover_grace_s = handover_grace_s
        # T4a.  When a model residency actor is wired in, bring-up goes through
        # it: it PARKS (L1 sleep) or stops whoever holds the GPUs, confirms the
        # VRAM actually came back, probes the engine for coherent text, and
        # measures the parked footprint.  The bare orchestrator path instead
        # raises "Cannot start X: GPUs [...] occupied by Y. Call stop_model
        # first." — which is how all 16 admitted model prefetches on the L40S
        # exp3 rows failed: ten of them in <=1 ms, and four of them only after
        # 600-918 s in the proactive-swap wait loop below.  Default None keeps
        # every existing arm byte-identical.
        self._residency_actor = residency_actor
        # Sleep/wake swap arm (RuntimeConfig.sleep_wake_swaps): prefer WAKING
        # a slept engine (weights already in CPU RAM — H2D copy, seconds) over
        # cold-booting a new server process (~185 s).  Cold boot remains the
        # fallback for engines that have never been booted this run; either
        # way conflicting engines on the shared pool are SLEPT first, never
        # stopped.  See runtime/prefetch/sleep_wake.py for the sequencing and
        # the CPU-RAM budget note (level-2 fallback when RAM-constrained).
        self._sleep_wake = sleep_wake
        # Disjoint-pool mode: before booting the target engine, stop (and VRAM-
        # drain) any running engine whose GPU set overlaps the target's pool —
        # e.g. the planner still holds GPUs 0-3 when the first advanced
        # specialist pre-boots.  All of it happens in this executor's
        # background thread, off the workflow's critical path.
        self._evict_conflicting = evict_conflicting
        # Disjoint-pool mode: a cancelled pre-boot finishes loading on its own
        # pool (interrupting vLLM mid-load is unsafe), but the engine serves
        # nobody — stop it as soon as the load completes so the wasted
        # residency window is bounded and visible in the trace.
        self._stop_wasted = stop_wasted_models
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="model_prefetch",
        )
        self._futures: dict[str, Future] = {}
        self._cancelled_tasks: set[str] = set()
        self._lock = threading.Lock()

    @property
    def can_evict_gpu_occupants(self) -> bool:
        """True when this executor can take the GPUs off an incumbent.

        `PrefetchScheduler._should_prefetch` reads this before letting a
        proactive-swap prediction past the confidence gate. The two ship
        together on purpose: admitting a swap that the executor cannot
        actually perform converts a silent skip into an instant failure.
        """
        return self._residency_actor is not None or bool(self._evict_conflicting)

    def start(self, task: PrefetchTask) -> None:
        """Launch model loading in the background thread pool."""
        task.started_at = time.perf_counter()
        task.status = PrefetchStatus.IN_PROGRESS

        future = self._pool.submit(self._load_model, task)
        with self._lock:
            self._futures[task.task_id] = future

    def cancel(self, task: PrefetchTask) -> bool:
        """
        Record cancellation intent. The background load continues uninterrupted
        because stopping a half-loaded vLLM server corrupts GPU memory state.

        Returns False: the load cannot be physically interrupted.
        The scheduler will mark the task WASTED via the event it emits.
        """
        with self._lock:
            self._cancelled_tasks.add(task.task_id)
        return False

    def is_complete(self, task: PrefetchTask) -> bool:
        """True once the background thread has finished (success, failure, or wasted)."""
        with self._lock:
            future = self._futures.get(task.task_id)
        if future is not None:
            return future.done()
        # No future: task never started or was never tracked
        return task.status not in (PrefetchStatus.PENDING, PrefetchStatus.IN_PROGRESS)

    def get_result(self, task: PrefetchTask) -> dict[str, Any]:
        """Return timing metadata. Blocks briefly if the future is nearly done."""
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
        """Gracefully shut down the thread pool."""
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Proactive swap
    # ------------------------------------------------------------------

    def _proactive_swap_legacy(self, task: PrefetchTask) -> None:
        """The pre-residency proactive swap. DO NOT CHANGE.

        This is the code as it stood before 2026-09-01, moved verbatim out of
        `_load_model` and nothing else.  `full_system`, `naive_prefetch` and
        `baseline` run without a residency actor and must keep behaving exactly
        as they did, including the 600 s bound, the `while/else`, the two print
        strings and the bare `stop_model`.  The defect this fallback carries
        predates Tandem and is NOT fixed here on purpose: fixing it silently
        would change arms whose numbers are already in the paper.

        The defect, for the record — it is the subject of
        `_proactive_swap_via_actor` below:  `stop_model` SIGKILLs the launcher
        after a 30 s grace, but a vLLM engine that is mid-load leaves its
        EngineCore and Worker children running.  They finish loading and
        allocate the KV cache AFTER "stopped" is printed, so the drain poll can
        never reach its >0.95-free condition and falls out at its 240 s
        ceiling — printing "GPU VRAM drained in 240.2s", which reads like
        success and is not.  ~80 GiB stays held and every later boot on those
        GPUs dies with vLLM's "Free memory on device cuda:N (14.28/94.97 GiB)".
        """
        # Wait for the current model to be stopped by the compute tool.
        # Timeout after 10 min; fall back to stopping it ourselves.
        deadline = time.perf_counter() + 600.0
        while time.perf_counter() < deadline:
            current = self._orchestrator.get_running_model()
            if current is None:
                break
            time.sleep(5.0)
        else:
            current = self._orchestrator.get_running_model()
            if current and current != task.resource.name:
                print(
                    f"[model_prefetch] Proactive swap timeout: stopping {current} "
                    f"(fallback) to load {task.resource.name}.",
                    flush=True,
                )
                self._orchestrator.stop_model(current)
        print(
            f"[model_prefetch] Proactive swap: GPUs free — loading {task.resource.name}.",
            flush=True,
        )

    def _incumbent_is_serving(self, name: str) -> bool | None:
        """Is `name`'s HTTP endpoint answering? None when we cannot tell.

        Deliberately NOT `orchestrator.wait_until_ready`: that one blocks,
        prints "[orchestrator] X is ready on :PORT." as a side effect, and
        raises on a port mismatch.  This is the same GET it makes
        (model_orchestrator.py:270-272) with none of that.
        """
        cfg = (getattr(self._orchestrator, "models", {}) or {}).get(name) or {}
        port = cfg.get("port")
        if not port:
            return None                     # no port to ask (e.g. the fakes)
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(
                    f"http://localhost:{port}/v1/models", timeout=5) as r:
                return 200 <= r.status < 300
        except urllib.error.URLError:
            # Connection refused / timeout / HTTP error. The engine holds the
            # GPUs but is not serving: it is still coming up (or wedged).
            return False
        except Exception:                             # noqa: BLE001
            return None                     # instrument problem, not an answer

    def _proactive_swap_via_actor(self, task: PrefetchTask) -> None:
        """Proactive swap WITH a residency actor wired.

        Three things differ from the legacy path, all of them the point:

        1. It NEVER calls `stop_model` itself.  Every eviction goes through
           `VllmModelActor.activate()` -> `_free_gpus_for()`, which prefers an
           L1 park, falls back to `_stop_and_measure()` (which waits for the
           whole process TREE to leave /proc and raises ReleaseNotHonoured if
           any pid survives — the orphaned EngineCore/Worker that wedged these
           trials), and then CONFIRMS with a VRAM read before anything is
           launched.  That is the whole difference between an eviction that is
           claimed and one that is evidenced.

        2. The wait is bounded by HANDOVER_GRACE_S (300 s, derived above from
           the orchestrator's own stop_model constants) rather than 600 s,
           because expiry is no longer an action.  It is a handoff: the actor
           decides whether the occupant can be moved.

        3. Expiry against an incumbent that is still COMING UP declines the
           swap instead of taking its GPUs.  This is the case all three failed
           trials were in — qwen_32b was 12/18 shards into a cold boot the
           ROUTER had asked for on the workflow's critical path, and no
           voluntary handover was ever coming.  Evicting it throws away a
           581-773 s weight load the current step is blocked on
           ("Model loading took 15.83 GiB memory and 581.144760 seconds",
           …_aligned/tandem/t01__20260901-144241__f9b64ab/stdout.log) to win a
           prefetch.  A step is worth more than a prefetch, so we decline.
        """
        target = task.resource.name
        deadline = time.perf_counter() + self._handover_grace_s
        while time.perf_counter() < deadline:
            current = self._orchestrator.get_running_model()
            if current is None or current == target:
                print(f"[model_prefetch] Proactive swap: GPUs free — "
                      f"loading {target}.", flush=True)
                return
            time.sleep(5.0)

        current = self._orchestrator.get_running_model()
        if current is None or current == target:
            print(f"[model_prefetch] Proactive swap: GPUs free — "
                  f"loading {target}.", flush=True)
            return

        serving = self._incumbent_is_serving(current)
        if serving is False:
            raise ProactiveSwapDeclined(
                f"declining the proactive swap to {target}: {current} still "
                f"holds GPUs "
                f"{sorted((getattr(self._orchestrator, 'models', {}) or {}).get(current, {}).get('gpus', []))} "
                f"after {self._handover_grace_s:.0f}s and is NOT serving yet — "
                f"it is mid bring-up for the workflow. Taking its GPUs would "
                f"discard a cold boot the current step is blocked on. Nothing "
                f"was touched; the incumbent is still coming up.")

        print(f"[model_prefetch] Proactive swap: no handover after "
              f"{self._handover_grace_s:.0f}s and {current} is serving — "
              f"delegating eviction to the residency actor (confirmed "
              f"park-or-stop), not stopping it blind.", flush=True)

    def _activate_or_restore(self, task: PrefetchTask) -> dict[str, Any]:
        """`actor.activate()`, but never leaving the GPUs without a server.

        `restore_on_failure` exists on the actor and does NOT cover this path:
        `_restore()` is called from `stage()` alone (model_actor.py:1270 and
        :1296).  `activate()` has no try/except around its wake/boot, so a
        target that fails to come up after the incumbent was parked leaves the
        incumbent parked.  This puts the same restore on the serving path.
        """
        target = task.resource.name
        try:
            return self._residency_actor.activate(target)
        except BaseException:
            self._restore_service(target)
            raise

    def _restore_service(self, target: str) -> None:
        """Undo what the failed activation cost. Never raises.

        PARKED victims are woken back — the actor's own restore (2.076 s), just
        on the path that was missing it.  STOPPED victims are reported and not
        cold-booted: a boot from this background thread would run 581-773 s
        alongside whatever `ModelRouter.ensure_ready` is already starting on
        the same GPUs (model_router.py:163), which is the collision that caused
        this defect.  The executor's obligation is narrower and sufficient —
        leave the GPUs in a state where the router's own boot can succeed.  The
        actor's stop is tree-confirmed and VRAM-read, so it does; the legacy
        `stop_model` did not, which is why those three runs could never
        recover.
        """
        actor = self._residency_actor
        try:
            key = actor.model_for(target)
        except Exception:                             # noqa: BLE001
            key = target
        try:
            ev = dict(getattr(actor, "last_eviction_detail", {}).get(key) or {})
        except Exception:                             # noqa: BLE001
            ev = {}

        parked = list(ev.get("parked") or [])
        stopped = list(ev.get("stopped") or [])
        if not parked and not stopped:
            print(f"[model_prefetch] Proactive swap to {target} failed before "
                  f"any eviction — the incumbent is untouched and still "
                  f"serving. Nothing to restore.", flush=True)
            return

        # `last_eviction_detail` PERSISTS across swaps, so a stale entry could
        # name a victim that is serving again. Only wake what is parked NOW.
        try:
            currently_parked = set(actor.parked_models())
        except Exception:                             # noqa: BLE001
            currently_parked = set()
        for victim in parked:
            if victim not in currently_parked:
                continue
            try:
                actor.wake(victim)
                print(f"[model_prefetch] Restored {victim}: woken back after "
                      f"{target} failed to come up.", flush=True)
            except Exception as exc:                  # noqa: BLE001
                print(f"[model_prefetch] WARNING: {victim} was parked for "
                      f"{target} and would not wake back: {exc}", flush=True)

        if stopped:
            print(f"[model_prefetch] {target} failed after STOPPING {stopped}; "
                  f"a stop cannot be undone cheaply and this thread will not "
                  f"race the router by cold-booting it. The teardown was "
                  f"confirmed (process tree gone + VRAM read), so the GPUs are "
                  f"free and ensure_ready() can boot whatever is asked next.",
                  flush=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, task: PrefetchTask) -> dict[str, Any]:
        """
        Run in the background thread. Calls orchestrator.start_model_measured().
        Snapshots hardware probes before/after if self._probes is set.
        After the load finishes, checks for a cancellation that arrived mid-load
        and marks the task WASTED instead of COMPLETED.

        Proactive-swap path (task.proactive_swap=True):
          Used when a long compute window (e.g. LAMMPS) provides GPU idle time.
          The compute tool (computation_task_screw_dislocation) stops the current
          model right before LAMMPS starts.  This background thread waits until
          the GPUs are free, then loads the next model concurrently with LAMMPS.
          This avoids stopping qwen_72b too early (before a bad-arg retry fires).

          There are TWO of these now, split on whether a residency actor is
          wired.  See _proactive_swap_legacy (unchanged; the arms without
          --residency) and _proactive_swap_via_actor (the fix).
        """
        probe_before = self._probes.snapshot() if self._probes else None
        try:
            if task.proactive_swap:
                if self._residency_actor is not None:
                    self._proactive_swap_via_actor(task)
                else:
                    self._proactive_swap_legacy(task)

            if self._evict_conflicting:
                target_gpus = set(
                    self._orchestrator.models.get(task.resource.name, {})
                    .get("gpus", []))
                for other, proc in list(
                        getattr(self._orchestrator, "processes", {}).items()):
                    if other == task.resource.name:
                        continue
                    alive = proc.poll() is None if hasattr(proc, "poll") else True
                    other_gpus = set(
                        self._orchestrator.models.get(other, {}).get("gpus", []))
                    if alive and target_gpus & other_gpus:
                        print(f"[model_prefetch] Evicting {other} "
                              f"(GPUs {sorted(target_gpus & other_gpus)} needed "
                              f"for {task.resource.name} pre-boot).", flush=True)
                        self._orchestrator.stop_model(other)

            mechanism = None
            evicted: list = []
            if self._residency_actor is not None:
                # T4a: the GPU-occupancy path. Raises GpusNotFreed — naming the
                # occupant and why it could not be moved — instead of the
                # orchestrator's generic "Call stop_model first."
                # Wrapped: the actor's own `restore_on_failure` is wired into
                # stage() only (model_actor.py:1270,1296 are its ONLY callers),
                # so an activate() that evicts and then fails to boot leaves
                # the victims down. _activate_or_restore closes that.
                info = self._activate_or_restore(task)
                elapsed = info["elapsed_s"]
                mechanism = info["mechanism"]
                evicted = [e["model"] for e in info.get("evicted", [])]
            elif self._sleep_wake:
                # Sleep/wake arm: wake-if-slept preferred over cold boot; the
                # planner (or any conflicting engine) is slept, never stopped.
                from runtime.prefetch.sleep_wake import swap_to_model
                swap_info = swap_to_model(self._orchestrator, task.resource.name)
                elapsed = swap_info["elapsed_s"]
                mechanism = swap_info["mechanism"]
            else:
                elapsed = self._orchestrator.start_model_measured(
                    task.resource.name,
                    metrics=None,
                )
            task.completed_at = time.perf_counter()

            probe_after = self._probes.snapshot() if self._probes else None
            probe_delta = (
                self._probes.delta(probe_before, probe_after).to_dict()
                if (probe_before and probe_after)
                else None
            )

            with self._lock:
                was_cancelled = task.task_id in self._cancelled_tasks

            if was_cancelled:
                task.status = PrefetchStatus.WASTED
                if self._stop_wasted:
                    try:
                        self._orchestrator.stop_model(task.resource.name)
                        print(f"[model_prefetch] Stopped wasted pre-boot "
                              f"{task.resource.name} (cancelled mid-load).",
                              flush=True)
                    except Exception as _exc:
                        print(f"[model_prefetch] WARNING: could not stop wasted "
                              f"{task.resource.name}: {_exc}", flush=True)
            elif task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.COMPLETED

            result: dict[str, Any] = {
                "elapsed_s": elapsed,
                "success": True,
                "wasted": was_cancelled,
            }
            # Sleep/wake arm: record HOW the engine became serving so the
            # parser can facet prefetch_completed on mechanism
            # ("sleep_wake" wake vs. "cold_boot").  Only emitted when the arm
            # is on — existing configs' traces stay byte-identical.
            if mechanism is not None:
                result["mechanism"] = mechanism
            if evicted:
                # What the admission COST: whose GPUs were taken, and by which
                # mechanism. Without this the trace shows a successful prefetch
                # and not the eviction that paid for it.
                result["evicted"] = evicted
            # A successful boot read the full weight snapshot: report it as
            # measured bytes so Q4/lifecycle byte provenance is not "estimated"
            # for vllm_model tasks (size itself comes from the snapshot dir's
            # st_size sum — see _model_size_bytes in the chemgraph adapter).
            # A wake copies weights H2D from CPU RAM and reads no bytes from
            # storage, so bytes_staged only applies to cold boots.
            size_b = getattr(task.resource, "estimated_size_bytes", None)
            if size_b and mechanism in (None, "cold_boot"):
                result["bytes_staged"] = int(size_b)
                if elapsed and elapsed > 0:
                    result["gb_per_s"] = round(size_b / elapsed / 1e9, 3)
            if probe_delta:
                result["probe_delta"] = probe_delta
            return result

        except Exception as exc:
            if task.status == PrefetchStatus.IN_PROGRESS:
                task.status = PrefetchStatus.FAILED
            task.error = str(exc)
            return {
                "elapsed_s": 0.0,
                "success": False,
                "error": str(exc),
            }
