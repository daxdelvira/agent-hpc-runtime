"""
adapters/chemgraph.py — Runtime adapter for ChemGraph (LangGraph/LangChain).

Extends ChemGraphCallbackHandler without modifying ChemGraph source code.

LangGraph callback propagation note
------------------------------------
In LangGraph 1.x, graph-level config callbacks propagate to *chain* and *tool*
callbacks (on_chain_start, on_tool_start) but NOT to LLM callbacks inside node
functions.  Node functions call llm.invoke(messages) without forwarding the
config, so on_llm_end / on_chat_model_start never fire.

We therefore use on_tool_start as the primary prediction/divergence hook:

  on_tool_start(A):
    1. Check _pending_checkpoint: was A the predicted consumer?  → HIT or MISS
    2. Make a new prediction for what comes AFTER A → schedule prefetch

  on_llm_end: passive — just calls super() for metrics.  When on_llm_end does
    fire (future LangGraph versions or direct LangChain invocations) the
    tool-start logic remains correct because it uses expected_at_step gating
    instead of a raw step comparison.

When config.mode == BASELINE the adapter behaves identically to the parent
class (zero additional overhead).

Usage
-----
    from runtime.adapters.chemgraph import make_runtime_callback

    cb = make_runtime_callback(
        predictor=MockPredictor("chemgraph"),
        scheduler=scheduler,
        config=cfg,
    )
    langgraph_config = {"configurable": {"thread_id": "1"}, "callbacks": [cb]}
    await cg.workflow.astream(inputs, stream_mode="values", config=langgraph_config)
"""
from __future__ import annotations

import re
import sys
import os
import time
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ChemGraph", "src"))

from langchain_core.outputs import LLMResult

from chemgraph.instrumentation.langgraph_hook import ChemGraphCallbackHandler

from runtime.config import RuntimeConfig, RuntimeMode
from runtime.event_bus import EventBus
from runtime.events import (
    PredictionResult,
    make_checkpoint_created_event,
    make_divergence_detected_event,
    make_prediction_result_event,
    make_prediction_validated_event,
    make_conservative_mode_event,
)
from runtime.guard.checkpoint import CheckpointRecord, CheckpointStore
from runtime.predictor.base import Predictor
from runtime.predictor.plan_extractor import KNOWN_TOOLS, extract_plan
from runtime.prefetch.scheduler import PrefetchScheduler

# Screen workload: planner tags each worker task "[SPECIALIST: <marker>]".
_SPECIALIST_RE = re.compile(r"\[SPECIALIST:\s*(\w+)\]", re.IGNORECASE)


class RuntimeChemGraphCallback(ChemGraphCallbackHandler):
    """
    ChemGraphCallbackHandler subclass that adds the runtime prediction/prefetch
    pipeline. All parent instrumentation (metrics CSV, WorkflowTracker JSONL)
    continues to work unchanged.

    New behaviour (skipped when mode=BASELINE):
    - on_tool_start: (1) check whether this tool matches the previous prediction,
                     (2) make a new prediction for what comes after this tool,
                     (3) schedule prefetches for predicted resources
    - on_llm_end: passive — just calls super(); no runtime logic here because
                  LangGraph 1.x does not propagate graph-level callbacks to
                  node-internal LLM invocations (llm.invoke without config=).
    """

    def __init__(
        self,
        metrics_logger=None,
        workflow_tracker=None,
        gpu_index: int = 0,
        predictor: Predictor | None = None,
        scheduler: PrefetchScheduler | None = None,
        config: RuntimeConfig | None = None,
        bus: EventBus | None = None,
        task_description: str = "",
        orchestrator=None,
        specialist_proxy=None,
    ) -> None:
        super().__init__(metrics_logger, workflow_tracker, gpu_index)
        self._predictor = predictor
        self._scheduler = scheduler
        self._config = config or RuntimeConfig(mode=RuntimeMode.BASELINE)
        self._bus = bus
        self._task_description = task_description
        self._orchestrator = orchestrator  # ModelOrchestrator | None — set in swap mode
        self._proxy = specialist_proxy    # SpecialistProxy | None — disjoint-pool mode
        self._step = 0
        self._checkpoint_store = CheckpointStore(max_horizon=self._config.max_horizon)
        self._conservative_until_step = 0
        self._pending_checkpoint: CheckpointRecord | None = None
        # Set once when the PlannerAgent node emits its plan (multi_agent workflow).
        self._plan_context = None
        self._worker_prefetch_scheduled = False
        self._cache_stage_scheduled = False
        self._cache_resource_id = ""
        self._worker_consumed_recorded = False
        # Option D: aggregator (co-resident) prefetch bookkeeping.
        self._aggregator_prefetch_started = False
        self._aggregator_consumed_recorded = False
        # Screen workload: per-task specialist routing state.
        self._specialist_plan: list[str] = []  # planned marker per worker task
        self._worker_task_idx = -1             # current WorkerAgent task index
        self._current_specialist_marker = ""   # routing established for current task
        self._transition_staged_for = -1       # task idx whose successor is staged
        self._transition_ckpt_id = ""          # checkpoint of in-flight transition stage
        self._staged_models: set[str] = set()  # models already cache-staged this run
        # Disjoint-pool mode: per-transition pre-boot bookkeeping.  Pre-boot
        # resource ids are unique per transition (the scheduler dedupes on
        # resource_id, and the same engine boots several times per run).
        self._pool_pending_boot: dict[str, str] = {}  # model -> in-flight boot rid

    @property
    def _is_active(self) -> bool:
        return self._config.mode != RuntimeMode.BASELINE and self._predictor is not None

    def _running_model_set(self) -> set[str]:
        """Names of all managed models whose vLLM process is alive (co-resident
        aware, unlike get_running_model() which returns only the first)."""
        if self._orchestrator is None:
            return set()
        procs = getattr(self._orchestrator, "processes", {}) or {}
        return {name for name, p in procs.items() if p.poll() is None}

    # ------------------------------------------------------------------
    # Screen workload helpers — specialist routing
    # ------------------------------------------------------------------

    def _specialist_model_for(self, marker: str) -> str:
        """Model key for a specialist marker; falls back to the legacy worker."""
        return self._config.specialist_models.get(
            (marker or "").lower(), self._config.vllm_worker_model)

    @staticmethod
    def _marker_in_text(text: str) -> str:
        m = _SPECIALIST_RE.search(text or "")
        return m.group(1).lower() if m else ""

    # ------------------------------------------------------------------
    # Disjoint-pool helpers (chemgraph_screen_pool / Option D)
    # ------------------------------------------------------------------

    def _pool_port(self, model: str) -> int:
        return int(self._orchestrator.models[model]["port"])

    def _pool_evict_async(self, models: list[str], reason: str) -> None:
        """Stop engines in a background thread (VRAM drain blocks for tens of
        seconds — never on the workflow's critical path)."""
        import threading
        for m in models:
            self._pool_pending_boot.pop(m, None)
        if self._bus:
            for m in models:
                self._bus.emit("pool_evict", {
                    "model": m, "reason": reason,
                }, step=self._step)

        def _stop():
            for m in models:
                try:
                    self._orchestrator.stop_model(m)
                except Exception as exc:
                    print(f"[pool] WARNING: evict {m} failed: {exc}", flush=True)

        threading.Thread(target=_stop, name="pool_evict", daemon=True).start()

    def _pool_maintain_after_flip(self, current_model: str) -> None:
        """
        Residency policy after routing task k to `current_model`:

        - Every arm (baseline included): engines outside the keep set are
          evicted — idle residency must be justified.
        - REAL mode: the keep set is plan-conditioned.  If the plan names the
          OTHER specialist for task k+1, that engine is kept resident (if up)
          or cache-staged + pre-booted on its own pool (if down) — the boot
          overlaps task k's serving window instead of blocking task k+1.
        - pool_blind_preboot: the trigger ablation — always prepare the other
          specialist, ignoring the plan (wrong on same-specialist runs).
        - pool_keep_all_resident (naive): boot everything, evict nothing.
        """
        cfg = self._config
        k = self._worker_task_idx
        specialists = set(cfg.specialist_models.values())
        running = self._running_model_set()

        if cfg.pool_keep_all_resident and cfg.mode == RuntimeMode.REAL:
            for m in sorted(specialists - running):
                self._stage_model_cache(m, trigger=f"pool_naive_task{k}")
                self._schedule_pool_preboot(m, trigger=f"pool_naive_task{k}",
                                            source="keep_all_resident")
            return

        needed_next = ""
        source = ""
        if cfg.mode == RuntimeMode.REAL:
            if cfg.pool_blind_preboot:
                others = sorted(specialists - {current_model})
                needed_next = others[0] if others else ""
                source = "blind_alternation"
            elif self._specialist_plan and k + 1 < len(self._specialist_plan):
                needed_next = self._specialist_model_for(
                    self._specialist_plan[k + 1])
                source = "plan"

        keep = {current_model} | ({needed_next} if needed_next else set())
        # Engines on needed_next's pool are left for the pre-boot executor,
        # which evicts them right before booting (keeps stop/start ordered).
        needed_gpus: set[int] = set()
        if needed_next and needed_next not in running:
            needed_gpus = set(self._orchestrator.models.get(
                needed_next, {}).get("gpus", []))
        to_stop = []
        for m in sorted(running - keep):
            m_gpus = set(self._orchestrator.models.get(m, {}).get("gpus", []))
            if needed_gpus and (m_gpus & needed_gpus):
                continue
            to_stop.append(m)
        if to_stop:
            self._pool_evict_async(to_stop, reason="idle_policy")

        if not needed_next or needed_next == current_model:
            return
        if needed_next in running:
            if self._bus:
                self._bus.emit("pool_keep", {
                    "model": needed_next, "task_index": k, "source": source,
                }, step=self._step)
            return
        self._stage_model_cache(needed_next, trigger=f"pool_transition_task{k}")
        self._schedule_pool_preboot(
            needed_next, trigger=f"pool_transition_task{k}", source=source)

    def _schedule_pool_preboot(self, model_name: str, trigger: str,
                               source: str = "") -> None:
        """
        Schedule a vllm_model prefetch that BOOTS `model_name` on its own GPU
        pool in the executor's background thread (evicting that pool's old
        occupant first).  Resource id is unique per transition — the same
        engine boots several times per run and the scheduler dedupes on id.
        """
        if (
            self._scheduler is None
            or self._config.mode != RuntimeMode.REAL
            or "vllm_model" in self._config.skip_resource_types
            or model_name in self._pool_pending_boot
        ):
            return

        from runtime.events import PredictionResult, ResourceSpec
        import hashlib

        resource_id = (hashlib.md5(model_name.encode()).hexdigest()[:16]
                       + f"_t{max(self._worker_task_idx, 0) + 1}")
        resource = ResourceSpec(
            resource_id=resource_id,
            resource_type="vllm_model",
            name=model_name,
            estimated_size_bytes=_model_size_bytes(
                self._config.model_paths.get(model_name), self._config),
            estimated_load_s=120.0,
            confidence=1.0,
            cancellation_safe=False,
            consumer_tool="",
            consumer_step_offset=1,
        )
        result = PredictionResult(
            step=self._step,
            resources=[resource],
            confidence=1.0,
            predictor_id=(f"pool_preboot:{trigger}"
                          + (f":{source}" if source else "")),
        )
        ckpt = CheckpointRecord(step=self._step, log_position=0,
                                prediction=result)
        self._checkpoint_store.add(ckpt)
        if self._bus:
            self._bus.emit_event(make_prediction_result_event(
                self._config.run_id, self._step, result))
        task = self._scheduler.schedule(
            resource=resource,
            current_step=self._step,
            checkpoint_id=ckpt.checkpoint_id,
        )
        if task is not None:
            self._pool_pending_boot[model_name] = resource_id
            self._transition_ckpt_id = ckpt.checkpoint_id

    # ------------------------------------------------------------------
    # on_chain_start — guard against LangGraph passing None for serialized
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags=None,
        **kwargs,
    ) -> None:
        # LangGraph 1.x occasionally passes serialized=None for internal chain
        # nodes.  The parent handler calls serialized.get(...) which raises
        # AttributeError.  Substitute an empty dict so the parent can proceed.
        super().on_chain_start(
            serialized or {},
            inputs or {},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )

        # Swap mode, Option A: begin warming the worker model's weight shards into
        # the OS page cache as early as possible — at the very first chain start,
        # concurrent with GPU-bound planner inference.  Pure host I/O; does not stop
        # the planner or touch the GPU.  Fires once per run.
        # Screen workload (early_plan_conditioned_stage + specialists): skipped —
        # staging instead fires at plan_extracted, conditioned on WHICH specialist
        # the plan names first.
        if not (self._config.early_plan_conditioned_stage
                and self._config.specialist_models):
            self._maybe_stage_worker_cache()

        # Swap mode: ensure the worker LLM is loaded before WorkerAgent executes.
        # In REAL mode the prefetch thread is already loading it (may already be done).
        # In BASELINE/non-REAL mode no prefetch was scheduled, so we trigger the swap
        # on-demand here — this is the sequential (non-overlapped) baseline path.
        node_name = self._chain_names.get(str(run_id), "")
        if (
            node_name == "WorkerAgent"
            and self._orchestrator is not None
            and self._config.vllm_worker_model
        ):
            worker_model = self._config.vllm_worker_model
            if self._config.specialist_models:
                # Screen workload: route this task to its tagged specialist.
                # WorkerAgent fires once per LLM TURN, not per task — the
                # authoritative task index is state["current_task_index"], and
                # the current task's prompt (with its [SPECIALIST: x] marker)
                # is the last worker_messages entry on the task's FIRST turn.
                idx = None
                content = ""
                if isinstance(inputs, dict):
                    idx = inputs.get("current_task_index")
                    wm = inputs.get("worker_messages") or []
                    if wm:
                        last = wm[-1]
                        content = (last.get("content", "")
                                   if isinstance(last, dict)
                                   else getattr(last, "content", "") or "")
                if idx is None:
                    idx = max(self._worker_task_idx, 0)
                new_task = idx != self._worker_task_idx
                self._worker_task_idx = idx
                k = idx
                planned = (self._specialist_plan[k]
                           if k < len(self._specialist_plan) else "")
                if new_task:
                    # Marker read from the CURRENT task prompt only; later
                    # turns of the same task keep the established routing.
                    self._current_specialist_marker = (
                        self._marker_in_text(content) or planned)
                    self._worker_consumed_recorded = False
                actual = getattr(self, "_current_specialist_marker", "") or planned
                worker_model = self._specialist_model_for(actual)
                if new_task and planned and actual != planned:
                    # Routing divergence: the specialist staged from the plan
                    # is not the one this task actually requires.
                    if self._bus:
                        self._bus.emit("specialist_divergence", {
                            "task_index": k,
                            "planned": planned,
                            "actual": actual,
                            "planned_model": self._specialist_model_for(planned),
                            "actual_model": worker_model,
                        }, step=self._step)
                    if (not self._config.disable_divergence_cancellation
                            and self._scheduler is not None
                            and self._transition_ckpt_id):
                        self._scheduler.cancel_all_pending(
                            reason="specialist_divergence",
                            checkpoint_id=self._transition_ckpt_id,
                            current_step=self._step,
                        )
                        self._transition_ckpt_id = ""
                    # Disjoint pools: a READY-but-wrong engine (kept or
                    # pre-booted from the bad plan entry) serves nobody —
                    # evict it in the background to reclaim its pool.  A boot
                    # still in flight was cancelled above; the executor stops
                    # it as soon as its load completes (stop_wasted_models).
                    if (self._config.disjoint_pools
                            and not self._config.disable_divergence_cancellation):
                        wrong_model = self._specialist_model_for(planned)
                        mid_boot = wrong_model in self._pool_pending_boot
                        self._pool_pending_boot.pop(wrong_model, None)
                        if (wrong_model != worker_model
                                and not mid_boot
                                and wrong_model in self._running_model_set()):
                            self._pool_evict_async(
                                [wrong_model], reason="specialist_divergence")
            t_wait0 = time.perf_counter()
            on_demand_swap = False
            swap_mechanism = ""
            import hashlib as _hashlib
            cache_rid = "cache_" + _hashlib.md5(worker_model.encode()).hexdigest()[:12]
            cache_known = (self._cache_resource_id == cache_rid
                           or worker_model in self._staged_models)
            if self._config.disjoint_pools:
                # Disjoint pools: the engine may already be resident on its own
                # pool (kept, or pre-booted during the previous task).  On-demand
                # boot only when it is not — after synchronously evicting any
                # engine that overlaps its pool (the planner holds GPUs 0-3
                # before the first advanced task).  This sync eviction+boot is
                # the exposed cost the pre-boot path avoids.
                running_set = self._running_model_set()
                # A pre-boot may be IN FLIGHT but not yet visible in the
                # running set (its executor thread is still draining the old
                # occupant's VRAM — with the ~0 s planner->worker transition
                # this is the common case, not the exception).  Booting again
                # here races two engines onto the same GPUs/port and kills
                # both (observed 2026-07-28, pool smoke t01-t03: two :8001
                # APIServers in the same second, rc=1).  If a pre-boot is
                # pending, fall through to wait_until_ready — its fail-fast
                # still catches a genuinely crashed pre-boot.
                preboot_in_flight = worker_model in self._pool_pending_boot
                if worker_model not in running_set and not preboot_in_flight:
                    on_demand_swap = True
                    if cache_known and self._scheduler is not None:
                        self._scheduler.on_resource_consumed(
                            cache_rid,
                            consumed_at=time.perf_counter(),
                            current_step=self._step,
                        )
                    target_gpus = set(self._orchestrator.models.get(
                        worker_model, {}).get("gpus", []))
                    for other in sorted(running_set):
                        other_gpus = set(self._orchestrator.models.get(
                            other, {}).get("gpus", []))
                        if target_gpus & other_gpus:
                            self._orchestrator.stop_model(other)
                    self._orchestrator.start_model(worker_model)
            elif self._config.sleep_wake_swaps:
                # Sleep/wake swaps (RuntimeConfig.sleep_wake_swaps): the worker
                # engine may already EXIST but be asleep (weights parked in CPU
                # RAM by a previous sleep).  Never stop the planner — sleep it
                # first (frees VRAM), THEN wake or (first use only) cold-boot
                # the worker.  Strict sleep-then-wake ordering: planner +
                # worker weights do not fit in VRAM together.
                # get_running_model() is not meaningful here (a slept process
                # is still "running"), so decide from the worker's own
                # process + sleep state.
                from runtime.prefetch.sleep_wake import (
                    has_live_process, last_mechanism, swap_to_model)
                worker_alive = has_live_process(self._orchestrator, worker_model)
                if (not worker_alive) or self._orchestrator.is_sleeping(worker_model):
                    on_demand_swap = True
                    # Only a cold boot reads weights from storage — the staged
                    # page cache is consumed by the first boot, not by a wake
                    # (which copies H2D from CPU RAM).
                    if not worker_alive and cache_known and self._scheduler is not None:
                        self._scheduler.on_resource_consumed(
                            cache_rid,
                            consumed_at=time.perf_counter(),
                            current_step=self._step,
                        )
                # Blocks until the worker actually serves (sleep others ->
                # wake/boot -> serving verified); wait_s below spans the whole
                # sleep+wake sequence because t_wait0 was taken above.
                swap_info = swap_to_model(self._orchestrator, worker_model)
                swap_mechanism = swap_info["mechanism"]
                if swap_mechanism == "already_serving":
                    # The prefetch thread owned the transition — report the
                    # mechanism it used.
                    swap_mechanism = last_mechanism(self._orchestrator, worker_model)
            else:
                running = self._orchestrator.get_running_model()
                if running != worker_model:
                    on_demand_swap = True
                    # The vLLM load is about to read the worker weights: this is
                    # the point the staged page cache gets consumed.  Record it so
                    # the overlap trace attributes the staging benefit to this swap.
                    if cache_known and self._scheduler is not None:
                        self._scheduler.on_resource_consumed(
                            cache_rid,
                            consumed_at=time.perf_counter(),
                            current_step=self._step,
                        )
                    # Worker not yet loaded: swap now (stop planner, start worker).
                    if running:
                        self._orchestrator.stop_model(running)
                    self._orchestrator.start_model(worker_model)
            # Record consumption of the prefetched worker-LLM (and its staged
            # cache) at NEED time — i.e. now, before the readiness wait.  This
            # marks the prefetch tasks used instead of wasted, and lets the
            # trace distinguish "ready in time" (completed < consumed) from
            # "late" (completed after consumed ≈ the residual wait below).
            if not self._worker_consumed_recorded and self._scheduler is not None:
                self._worker_consumed_recorded = True
                worker_rid = _hashlib.md5(worker_model.encode()).hexdigest()[:16]
                self._scheduler.on_resource_consumed(
                    worker_rid, consumed_at=t_wait0, current_step=self._step)
                # Disjoint pools: pre-boot resource ids are per-transition —
                # consume the in-flight boot for THIS engine, if any.
                pool_rid = self._pool_pending_boot.pop(worker_model, "")
                if pool_rid:
                    self._scheduler.on_resource_consumed(
                        pool_rid, consumed_at=t_wait0, current_step=self._step)
                if cache_known and not on_demand_swap:
                    self._scheduler.on_resource_consumed(
                        cache_rid, consumed_at=t_wait0,
                        current_step=self._step)
            # Cold Lustre loads of 72B: ~45 min typical worst case, but the
            # evening-degraded window (40 MB/s observed) exceeded 2700 s and
            # failed an otherwise-healthy trial on 2026-07-10.  Match the
            # orchestrator's load_timeout (5400) — a slow swap is valid data
            # (it IS the exposed stall); a timeout is a lost trial.
            self._orchestrator.wait_until_ready(worker_model, timeout=5400)
            # Disjoint pools: route the (single-endpoint) worker client to this
            # specialist's port.  Always after wait_until_ready — requests
            # never race a booting engine.
            if self._config.disjoint_pools and self._proxy is not None:
                self._proxy.set_target(self._pool_port(worker_model))
            # Exposed staging time on the critical path: how long the workflow
            # sat blocked here waiting for the worker LLM to become ready.
            # ≈ full swap time in baseline (on-demand), ≈ residual in real mode
            # (prefetch started earlier), ≈ 0 when the prefetch finished in time.
            wait_s = time.perf_counter() - t_wait0
            if self._bus is not None and (on_demand_swap or wait_s > 0.1):
                payload = {
                    "model": worker_model,
                    "wait_s": wait_s,
                    "on_demand_swap": on_demand_swap,
                    "prefetch_scheduled": self._worker_prefetch_scheduled,
                    "cache_stage_scheduled": self._cache_stage_scheduled,
                    "mode": self._config.mode.value,
                    "pool": self._config.disjoint_pools,
                }
                if self._config.sleep_wake_swaps:
                    # Parser facet key: how this swap was served.  Only added
                    # in the sleep_wake arm so pre-existing configs' trace
                    # payloads stay byte-identical.
                    payload["swap_mechanism"] = swap_mechanism or "cold_boot"
                self._bus.emit("worker_swap_wait", payload, step=self._step)
            # Disjoint pools: post-flip residency maintenance — evict idle
            # engines (policy, every arm) and, in REAL mode, decide from the
            # plan whether the next task's engine is kept resident or
            # pre-booted on its own pool (all off the critical path).
            if self._config.disjoint_pools:
                self._pool_maintain_after_flip(worker_model)

        # Option D: ensure the (co-resident) aggregator LLM is ready before the
        # AggregatorAgent runs.  In active/REAL mode the prefetch thread was
        # started at run_mace_ensemble's tool_start (during the GPU-idle compute
        # window) and is likely already READY, so this wait is ~0.  In baseline
        # (no prefetch) we start it on-demand here — the sequential cost the
        # overlap is meant to hide.  The aggregator sits on its own GPUs, so no
        # worker stop/swap is needed regardless of path.
        if (
            node_name == "AggregatorAgent"
            and self._orchestrator is not None
            and self._config.vllm_aggregator_model
        ):
            agg_model = self._config.vllm_aggregator_model
            t_wait0 = time.perf_counter()
            on_demand_start = False
            running_models = self._running_model_set()
            if agg_model not in running_models:
                on_demand_start = True
                self._orchestrator.start_model(agg_model)
            self._orchestrator.wait_until_ready(agg_model, timeout=1800)
            wait_s = time.perf_counter() - t_wait0
            if self._bus is not None:
                self._bus.emit("aggregator_swap_wait", {
                    "model": agg_model,
                    "wait_s": wait_s,
                    "on_demand_start": on_demand_start,
                    "prefetch_started": self._aggregator_prefetch_started,
                    "mode": self._config.mode.value,
                }, step=self._step)

    # ------------------------------------------------------------------
    # on_chain_end — extract plan when PlannerAgent node finishes
    # ------------------------------------------------------------------

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        # Peek at the node name before super() pops it from _chain_names.
        node_name = self._chain_names.get(str(run_id), "")
        super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        if node_name != "PlannerAgent" or self._plan_context is not None:
            return
        if self._config.mode == RuntimeMode.BASELINE:
            return
        if self._step > self._config.plan_extraction_horizon:
            return
        if not self._bus or not isinstance(outputs, dict):
            return

        # PlannerAgent returns {"messages": ["<PlannerResponse JSON>"]}
        # LangGraph may pass the raw string OR an AIMessage object (after reducer).
        messages = outputs.get("messages", [])
        if not messages:
            return
        raw = messages[-1]
        if isinstance(raw, str):
            raw_content = raw
        elif hasattr(raw, "content") and isinstance(raw.content, str):
            raw_content = raw.content
        else:
            return

        import json as _json
        try:
            plan_data = _json.loads(raw_content)
            tasks = plan_data.get("worker_tasks", [])
            combined_text = " ".join(t.get("prompt", "") for t in tasks)
        except Exception:
            return

        # Screen workload: record the planner's per-task specialist assignments.
        if self._config.specialist_models:
            self._specialist_plan = [
                self._marker_in_text(t.get("prompt", "")) for t in tasks
            ]

        ctx = extract_plan(combined_text, KNOWN_TOOLS, step=0, source="planner_agent")
        if ctx is not None:
            self._plan_context = ctx
            payload = {
                "step": 0,
                "tool_sequence": ctx.tool_sequence,
                "n_mentions": ctx.n_mentions,
                "source": ctx.source,
            }
            if self._specialist_plan:
                payload["specialist_sequence"] = self._specialist_plan
            self._bus.emit("plan_extracted", payload, step=0)

            # Screen workload: plan-conditioned scheduling — the plan's FIRST
            # specialist decides which model gets staged and vLLM-prefetched.
            if (self._config.early_plan_conditioned_stage
                    and self._config.specialist_models
                    and self._specialist_plan):
                first_model = self._specialist_model_for(self._specialist_plan[0])
                self._stage_model_cache(first_model, trigger="plan_extracted")
                self._schedule_worker_model_prefetch(model_name=first_model)
            else:
                # Swap mode: schedule the worker LLM model prefetch now.
                # The model load begins in a background thread; on_chain_start
                # will block WorkerAgent until wait_until_ready() returns.
                self._schedule_worker_model_prefetch()

    # ------------------------------------------------------------------
    # _maybe_stage_worker_cache — Option A: page-cache staging (swap mode)
    # ------------------------------------------------------------------

    def _maybe_stage_worker_cache(self) -> None:
        """
        Schedule a "model_cache" prefetch that warms the worker model's weight
        shards into the OS page cache during the planner phase.

        Fires once, at the first chain start, so the host-side read overlaps the
        full planner inference (which occupies the GPUs).  When the WorkerAgent
        swap later loads the worker into vLLM, it reads from warm cache — hiding
        the ~130 s of cold Lustre I/O that the cold swap pays on the critical path.

        Guards mirror the vllm_model prefetch: needs a worker model + scheduler,
        never in BASELINE, disabled when stage_worker_cache is False or when
        "model_cache" is in skip_resource_types (the no_cache_stage ablation).
        """
        if (
            self._cache_stage_scheduled
            or not self._config.vllm_worker_model
        ):
            return
        ckpt_id = self._stage_model_cache(
            self._config.vllm_worker_model, trigger="first_chain_start")
        if ckpt_id is not None:
            self._cache_stage_scheduled = True

    def _stage_model_cache(self, model_name: str, trigger: str) -> str | None:
        """
        Schedule a "model_cache" prefetch that warms `model_name`'s weight
        shards into the OS page cache.  Host I/O only — does not stop any
        running model or touch the GPU; safely cancellable.  Returns the
        checkpoint id, or None if staging is disabled/duplicate.
        """
        if (
            not self._config.stage_worker_cache
            or not model_name
            or model_name in self._staged_models
            or self._scheduler is None
            or self._config.mode == RuntimeMode.BASELINE
            or "model_cache" in self._config.skip_resource_types
        ):
            return None

        from runtime.events import PredictionResult, ResourceSpec
        from runtime.guard.checkpoint import CheckpointRecord
        import hashlib

        resource_id = "cache_" + hashlib.md5(model_name.encode()).hexdigest()[:12]
        self._cache_resource_id = resource_id
        self._staged_models.add(model_name)

        path = (self._config.model_paths.get(model_name)
                or (self._config.worker_model_path
                    if model_name == self._config.vllm_worker_model else None))
        resource = ResourceSpec(
            resource_id=resource_id,
            resource_type="model_cache",
            name=model_name,
            path=path or None,
            estimated_size_bytes=_model_size_bytes(path, self._config),
            estimated_load_s=130.0,
            confidence=1.0,
            cancellation_safe=True,
            consumer_tool="",
            consumer_step_offset=1,
        )
        result = PredictionResult(
            step=self._step,
            resources=[resource],
            confidence=1.0,
            predictor_id=f"swap_cache_stage:{trigger}",
        )
        ckpt = CheckpointRecord(step=self._step, log_position=0, prediction=result)
        self._checkpoint_store.add(ckpt)

        if self._bus:
            self._bus.emit_event(make_prediction_result_event(
                self._config.run_id, self._step, result))

        self._scheduler.schedule(
            resource=resource,
            current_step=self._step,
            checkpoint_id=ckpt.checkpoint_id,
        )
        return ckpt.checkpoint_id

    # ------------------------------------------------------------------
    # _schedule_worker_model_prefetch — swap-mode helper
    # ------------------------------------------------------------------

    def _schedule_worker_model_prefetch(self, model_name: str | None = None) -> None:
        """
        Schedule a vllm_model prefetch for the worker LLM after plan extraction.
        Only fires once per run (guarded by _worker_prefetch_scheduled).
        `model_name` overrides config.vllm_worker_model (screen workload: the
        plan's first specialist).
        """
        if (
            self._worker_prefetch_scheduled
            or not self._config.vllm_worker_model
            or self._scheduler is None
            or self._config.mode == RuntimeMode.BASELINE
            or "vllm_model" in self._config.skip_resource_types
        ):
            return

        from runtime.events import PredictionResult, ResourceSpec
        import hashlib

        model_name = model_name or self._config.vllm_worker_model
        resource_id = hashlib.md5(model_name.encode()).hexdigest()[:16]

        # Shared GPU pool: the planner must be stopped before the worker can
        # load.  The planner LLM is only used by the PlannerAgent entry node,
        # which has just finished — safe to evict now.
        # Disjoint pools: skip the synchronous stop — the pre-boot executor
        # evicts conflicting engines in its background thread instead, so the
        # workflow never blocks on the planner's VRAM drain.
        # Sleep/wake arm: release_gpus_for SLEEPS the planner instead of
        # stopping it (weights -> CPU RAM, VRAM freed) so it can be woken
        # later; with the flag off it reproduces the legacy stop exactly.
        if self._orchestrator is not None and not self._config.disjoint_pools:
            from runtime.prefetch.sleep_wake import release_gpus_for
            release_gpus_for(self._orchestrator, model_name,
                             sleep_wake=self._config.sleep_wake_swaps)

        resource = ResourceSpec(
            resource_id=resource_id,
            resource_type="vllm_model",
            name=model_name,
            estimated_size_bytes=_model_size_bytes(
                self._config.model_paths.get(model_name)
                or self._config.worker_model_path, self._config),
            estimated_load_s=120.0,
            confidence=1.0,
            cancellation_safe=False,
            consumer_tool="",
            consumer_step_offset=1,
        )

        import uuid as _uuid
        from runtime.guard.checkpoint import CheckpointRecord
        result = PredictionResult(
            step=0,
            resources=[resource],
            confidence=1.0,
            predictor_id="plan_extraction",
        )
        ckpt = CheckpointRecord(step=0, log_position=0, prediction=result)
        self._checkpoint_store.add(ckpt)

        if self._bus:
            self._bus.emit_event(make_prediction_result_event(self._config.run_id, 0, result))

        self._scheduler.schedule(
            resource=resource,
            current_step=0,
            checkpoint_id=ckpt.checkpoint_id,
        )
        self._worker_prefetch_scheduled = True

    # ------------------------------------------------------------------
    # on_llm_end — passive; just records metrics via parent
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        # No runtime logic here: LangGraph 1.x does not propagate graph-level
        # config callbacks into node-internal llm.invoke() calls, so this hook
        # does not fire reliably.  All prediction logic lives in on_tool_start.

    # ------------------------------------------------------------------
    # on_tool_start — primary prediction / divergence hook
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags=None,
        **kwargs,
    ) -> None:
        # Always call parent (records metrics, WorkflowTracker tool_call event)
        super().on_tool_start(serialized, input_str, run_id=run_id,
                               parent_run_id=parent_run_id, tags=tags, **kwargs)

        if not self._is_active:
            return

        tool_name = serialized.get("name") or kwargs.get("name", "unknown_tool")
        self._step += 1
        if self._bus:
            self._bus.set_step(self._step)

        # ------------------------------------------------------------------
        # Option D: the long-running ensemble tool just started — its MACE
        # compute occupies the CPU for ~11 min while the worker GPUs sit idle.
        # Kick off the (co-resident) aggregator model load NOW so it overlaps
        # that window and is READY by the time control reaches AggregatorAgent.
        # Non-blocking: start_model() spawns the server and returns; the
        # readiness wait happens in on_chain_start("AggregatorAgent").
        # ------------------------------------------------------------------
        if (
            tool_name == "run_mace_ensemble"
            and self._orchestrator is not None
            and self._config.vllm_aggregator_model
            and not self._aggregator_prefetch_started
            and self._config.mode != RuntimeMode.OBSERVE_ONLY
        ):
            agg_model = self._config.vllm_aggregator_model
            if agg_model not in self._running_model_set():
                self._aggregator_prefetch_started = True
                t0 = time.perf_counter()
                try:
                    self._orchestrator.start_model(agg_model)
                    if self._bus:
                        self._bus.emit("aggregator_prefetch_start", {
                            "model": agg_model,
                            "trigger_tool": tool_name,
                            "mode": self._config.mode.value,
                        }, step=self._step)
                except Exception as exc:
                    self._aggregator_prefetch_started = False
                    if self._bus:
                        self._bus.emit("aggregator_prefetch_error", {
                            "model": agg_model, "error": str(exc),
                        }, step=self._step)

        # ------------------------------------------------------------------
        # Screen workload: this task's heavyweight compute (run_ase → MACE)
        # just started — the window in which the NEXT task's specialist can be
        # staged.  Plan-conditioned: the plan's marker for task k+1 decides
        # WHICH model gets warmed.  Host I/O only; the exposed vLLM spin-up at
        # the task boundary is the (bounded) residual the trace measures.
        # ------------------------------------------------------------------
        if (
            self._config.specialist_models
            and tool_name == "run_ase"
            and self._worker_task_idx >= 0
            and self._transition_staged_for < self._worker_task_idx
            and self._config.mode == RuntimeMode.REAL
            and self._scheduler is not None
        ):
            k = self._worker_task_idx
            self._transition_staged_for = k
            nxt = k + 1
            if nxt < len(self._specialist_plan):
                cur_marker = (self._specialist_plan[k]
                              if k < len(self._specialist_plan) else "")
                next_model = self._specialist_model_for(self._specialist_plan[nxt])
                cur_model = self._specialist_model_for(cur_marker)
                if next_model != cur_model:
                    ckpt_id = self._stage_model_cache(
                        next_model, trigger=f"transition_task{k}")
                    if ckpt_id is not None:
                        self._transition_ckpt_id = ckpt_id

        # ------------------------------------------------------------------
        # Phase 1: check the PREVIOUS prediction against this tool.
        # The pending checkpoint says "expect consumer_tool at expected_at_step".
        # Gate by expected_at_step so multi-step lookahead predictions (step_offset>1)
        # are not invalidated by intermediate tool calls.
        # ------------------------------------------------------------------
        if self._pending_checkpoint is not None:
            ckpt = self._pending_checkpoint
            if ckpt.prediction.resources:
                predicted_tool = ckpt.prediction.resources[0].consumer_tool
                expected_at_step = ckpt.prediction.resources[0].expected_at_step or 0

                if self._step >= expected_at_step:
                    hit = (tool_name == predicted_tool)
                    if hit:
                        self._checkpoint_store.resolve(ckpt.checkpoint_id, "validated")
                        if self._bus:
                            self._bus.emit_event(make_prediction_validated_event(
                                self._config.run_id, self._step, True,
                                ckpt.checkpoint_id, tool_name,
                            ))
                        if self._scheduler:
                            self._scheduler.on_resource_consumed(
                                ckpt.prediction.resources[0].resource_id,
                                consumed_at=time.perf_counter(),
                                current_step=self._step,
                            )
                    else:
                        self._checkpoint_store.resolve(
                            ckpt.checkpoint_id, "diverged", "INVALIDATE_ALL"
                        )
                        if self._bus:
                            self._bus.emit_event(make_divergence_detected_event(
                                self._config.run_id, self._step, predicted_tool,
                                tool_name, ckpt.checkpoint_id, "INVALIDATE_ALL",
                            ))
                        if not self._config.disable_divergence_cancellation:
                            if self._scheduler:
                                self._scheduler.cancel_all_pending(
                                    reason="divergence",
                                    checkpoint_id=ckpt.checkpoint_id,
                                    current_step=self._step,
                                )
                            self._conservative_until_step = (
                                self._step + self._config.conservative_mode_steps
                            )
                            if self._bus:
                                self._bus.emit_event(make_conservative_mode_event(
                                    self._config.run_id, self._step, "divergence",
                                    self._config.conservative_mode_steps,
                                ))
                    self._pending_checkpoint = None
                    self._checkpoint_store.expire_old(self._step)
                # else: too early — multi-step lookahead, keep pending

        # ------------------------------------------------------------------
        # Phase 2: make a new prediction for what comes AFTER this tool
        # ------------------------------------------------------------------
        in_conservative = self._step <= self._conservative_until_step
        if in_conservative:
            return

        recent_events = _read_recent_events(
            getattr(self._tracker, "log_path", None),
            n=self._config.predictor_context_events,
        )

        try:
            result: PredictionResult = self._predictor.predict(
                step=self._step,
                recent_events=recent_events,
                current_tool_calls=[{"name": tool_name}],
                task_description=self._task_description,
                plan_context=self._plan_context,
            )
        except Exception as e:
            if self._bus:
                self._bus.emit("prediction_error", {"error": str(e)}, step=self._step)
            return

        if not result.resources:
            return

        if self._bus:
            self._bus.emit_event(make_prediction_result_event(
                self._config.run_id, self._step, result
            ))

        log_pos = self._bus.current_log_position() if self._bus else 0
        checkpoint = CheckpointRecord(
            step=self._step,
            log_position=log_pos,
            prediction=result,
        )
        self._checkpoint_store.add(checkpoint)
        self._pending_checkpoint = checkpoint

        if self._bus:
            self._bus.emit_event(make_checkpoint_created_event(
                self._config.run_id, self._step, checkpoint.checkpoint_id, log_pos,
            ))

        if self._scheduler is not None and self._config.mode != RuntimeMode.OBSERVE_ONLY:
            for resource in result.resources:
                self._scheduler.schedule(
                    resource=resource,
                    current_step=self._step,
                    checkpoint_id=checkpoint.checkpoint_id,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worker_model_size_bytes(config: RuntimeConfig) -> int | None:
    """Total shard bytes of the worker model snapshot, or None if unknown."""
    return _model_size_bytes(config.worker_model_path, config)


def _model_size_bytes(path: str | None, config: RuntimeConfig) -> int | None:
    """Total shard bytes of a model snapshot dir, or None if unknown."""
    if not path:
        return None
    try:
        from runtime.prefetch.model_cache_prefetch import list_model_shards
        shards = list_model_shards(path)
        return sum(p.stat().st_size for p in shards) or None
    except Exception:
        return None


def _extract_tool_calls(response: LLMResult) -> list[dict]:
    """Extract tool_calls from a LangChain LLMResult."""
    tool_calls = []
    try:
        for gen_list in (response.generations or []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                calls = getattr(msg, "tool_calls", None) or []
                for call in calls:
                    if isinstance(call, dict):
                        tool_calls.append(call)
                    else:
                        # ToolCall pydantic object
                        tool_calls.append({"name": getattr(call, "name", ""), "args": getattr(call, "args", {})})
    except Exception:
        pass
    return tool_calls


def _read_recent_events(log_path: str | None, n: int) -> list[dict]:
    """Read the last `n` events from a WorkflowTracker JSONL file."""
    if not log_path or not os.path.exists(log_path):
        return []
    import json
    try:
        with open(log_path) as f:
            lines = f.readlines()
        events = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events
    except Exception:
        return []


def make_runtime_callback(
    predictor: Predictor | None = None,
    scheduler: PrefetchScheduler | None = None,
    guard=None,                   # reserved; divergence is handled inside the callback
    config: RuntimeConfig | None = None,
    bus: EventBus | None = None,
    metrics_logger=None,
    workflow_tracker=None,
    gpu_index: int = 0,
    task_description: str = "",
    orchestrator=None,            # ModelOrchestrator | None — set in swap mode
    specialist_proxy=None,        # SpecialistProxy | None — disjoint-pool mode
) -> RuntimeChemGraphCallback:
    """
    Convenience factory for RuntimeChemGraphCallback.

    If metrics_logger / workflow_tracker are not supplied, falls back to the
    ChemGraph module-level singletons (same as make_callback() in langgraph_hook).
    """
    if metrics_logger is None:
        try:
            from chemgraph.instrumentation.metrics_logger import get_metrics_logger
            metrics_logger = get_metrics_logger()
        except Exception:
            pass

    if workflow_tracker is None:
        try:
            from chemgraph.instrumentation.workflow_tracker import tracker
            workflow_tracker = tracker
        except Exception:
            pass

    if config is None:
        config = RuntimeConfig(mode=RuntimeMode.SIMULATED)

    # Create an EventBus that shares the WorkflowTracker's file if possible
    if bus is None and workflow_tracker is not None:
        shared_file = getattr(workflow_tracker, "_file", None)
        bus = EventBus(run_id=config.run_id, shared_file=shared_file)

    return RuntimeChemGraphCallback(
        metrics_logger=metrics_logger,
        workflow_tracker=workflow_tracker,
        gpu_index=gpu_index,
        predictor=predictor,
        scheduler=scheduler,
        config=config,
        bus=bus,
        task_description=task_description,
        orchestrator=orchestrator,
        specialist_proxy=specialist_proxy,
    )
