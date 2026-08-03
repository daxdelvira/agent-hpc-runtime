#!/usr/bin/env python3
"""
extract_prefetch_lifecycle.py — per-prefetch-object lifecycle rows from eval runs.

Input : the result tree written by experiments/run_eval_q1_q4.py
        (results/eval_q1_q4/runs/<workload>/<config>/<trial>/...)
Output: results/eval_q1_q4/eval_prefetch_lifecycle.csv
          one row per (trial x prefetch-relevant object), where objects are the
          union of (a) admitted prefetch tasks, (b) skipped prefetch_decision
          intents, and (c) needs with no prediction at all (gate waits /
          no_prefetch consumption stamps / disk mace loads in baseline or
          inert trials) so stall totals reconcile across configs.
        results/eval_q1_q4/eval_stall_taxonomy.csv
          seconds of exposed stall per (workload, config, stall_class).

Timeline semantics
------------------
All t_* columns are seconds relative to the trace's first event; absolute
epochs are kept in *_epoch columns for cross-file joins.

t_first_needed derivation, in priority order (need_source column):
  swap_wait / agg_wait  worker_swap_wait / aggregator_swap_wait gate events.
                        Both are emitted at gate EXIT with wait_s measured from
                        gate entry (runtime/adapters/chemgraph.py), so
                        t_first_needed = event_t - wait_s and exposure_s =
                        wait_s (authoritative measured stall).  NOTE
                        worker_swap_wait is only emitted when on_demand_swap or
                        wait_s > 0.1 — a fully hidden swap leaves no gate event
                        and falls through to 'consumed'.
  consumed              resource_consumed stamp (recorded at need time by the
                        adapters); exposure_s = max(0, completed - consumed),
                        the residual-stall formula used by parse_eval_traces.
  consumer_tool_call    first tool_call of the resource's consumer_tool after
                        t_predicted (registry fallback).
  none                  never needed (pure speculation).

gate_group: the vllm_model server task and the model_cache staging task for the
same model gate the SAME wall interval (consumed together at t_wait0).
Aggregations must max() exposure within a gate_group, never sum (this mirrors
the max()-dedup in parse_eval_traces.py's exposed_stall_s).

stall_class (rows with exposure_s > 0), decision order:
  baseline_no_prefetch      need in a config that never prefetches (reference)
  no_prediction             need occurred, nothing ever predicted the resource
  policy_skip:<reason>      scheduler declined (prefetch_decision skip)
  gpu_serialization_residual vllm_model gate stalled although the model's I/O
                            (model_cache task in the same gate_group) landed on
                            time — the engine bring-up itself is the exposed
                            cost.  Checked before the window classes: when I/O
                            was on time the window argument is moot.
  no_window                 window_s < w_min  (prediction arrived at need time)
  window_too_small          window_s < transfer + spin-up (unhideable)
  late_start                window sufficient but start_delay_s consumed it
  residual_partial          transfer started early, finished after need
  unattributed              exposure with no reconstructable window

Bring-up-relative window floors
-------------------------------
w_min and spin-up used to be the fixed constants 15 s and 30 s.  They were
calibrated when the only way to make a vLLM engine serve was a cold boot of
500-1300 s.  The sleep-mode arm (RuntimeConfig.sleep_wake_swaps) wakes a parked
engine in 0.8-2.1 s — three orders of magnitude cheaper — and against fixed
cold-boot floors every such wake would be stamped `no_window`: a 5 s window
would be called "no window at all" even though it is >3x the entire cost of the
operation.  The instrument would report the success case as a failure, silently
and plausibly.

A fixed threshold is the wrong instrument when the quantity it gates spans
three orders of magnitude.  Both floors are therefore expressed relative to the
resource's OWN measured bring-up cost (bringup_cost_s()), capped by the historic
constants:

    w_min  = min(W_MIN_CAP_S,  NO_WINDOW_FRAC * bringup)     # 15 s cap
    spinup = min(SPINUP_CAP_S, SPINUP_FRAC   * bringup)      # 30 s cap, vllm

Reading: a floor may never exceed the cost of the thing it is gating.  You
cannot call a window "too small to act in" when it is longer than the entire
action.  With NO_WINDOW_FRAC = SPINUP_FRAC = 1.0 the floors are not tuned
parameters — `no_window` means literally "the window was shorter than the
bring-up", which is regime-independent by construction.

Consequences:
  cold boot (bringup 500-1300 s)  fractions exceed the caps, min() picks the
                                  cap, behaviour identical to the old code.
  L1/L2 wake (bringup 0.8-2.1 s)  caps are ~10-20x the whole operation; min()
                                  picks the relative term and the arm is scored
                                  on whether it actually hid the wake.

Using each row's own transfer_s (rather than a table of calibrated per-platform
bring-up costs) is deliberate: it is per-trial, per-platform and per-mechanism,
so it never pools L40S with Blackwell figures and cannot go stale when a new
bring-up mechanism lands.  Every window-branch row carries bringup_cost_s,
bringup_provenance, w_min_s and spinup_s so its class can be re-derived from the
CSV alone; swap_mechanism carries vLLM's own report of how the swap was served
("sleep_wake" / "cold_boot" / "already_serving") where the arm emits it.

Usage
-----
    python scripts/extract_prefetch_lifecycle.py
    python scripts/extract_prefetch_lifecycle.py --workload chemgraph_swap
    python scripts/extract_prefetch_lifecycle.py --include-failed   # debug only
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from parse_eval_traces import (  # noqa: E402
    DEFAULT_EVAL_ROOT,
    classify_task,
    discover_runs,
    load_consumer_map,
    load_events,
    parse_trace,
    task_bytes,
    used_resource_ids,
)

# ---------------------------------------------------------------------------
# Window floors — see "Bring-up-relative window floors" in the module docstring
# ---------------------------------------------------------------------------
# These two constants were calibrated in the cold-boot-only era, when the ONLY
# way to make a vLLM engine serve was a 500-1300 s boot.  They are retained as
# CAPS, not as the floors themselves: a floor may never exceed the bring-up cost
# it is supposed to gate (see window_floors()).
W_MIN_CAP_S = 15.0        # cold-boot-era "no window at all" cap
SPINUP_CAP_S = 30.0       # cold-boot-era engine init/compile cap, added to
                          # transfer when judging window_too_small for
                          # vllm_model resources
# Fractions of the measured bring-up cost.  1.0 == "the floor is the bring-up
# cost itself"; both are deliberately NOT tuned constants (see docstring).
NO_WINDOW_FRAC = 1.0
SPINUP_FRAC = 1.0

# Backwards-compatible aliases (external callers / older notebooks import these).
W_MIN_S = W_MIN_CAP_S
SPINUP_FLOOR_S = SPINUP_CAP_S

SIM_ELAPSED_S = 0.5       # elapsed below this on a >1 GB object = simulated

NO_PREFETCH_CONFIGS = {"baseline", "observe_only"}


def _r(x, nd=2):
    return round(x, nd) if isinstance(x, (int, float)) else None


# ---------------------------------------------------------------------------
# Gate extraction: one record per worker/aggregator readiness gate
# ---------------------------------------------------------------------------

def collect_gates(trace: dict) -> list[dict]:
    gates = []
    for i, w in enumerate(trace["swap_waits"]):
        gates.append({
            "kind": "swap_wait", "idx": i, "model": w.get("model", ""),
            "t_exit": w.get("t", 0.0), "wait_s": float(w.get("wait_s") or 0.0),
            "on_demand": bool(w.get("on_demand_swap")),
            "prefetch_scheduled": bool(w.get("prefetch_scheduled")),
            # Only emitted by the sleep_wake arm (runtime/adapters/chemgraph.py
            # adds it under RuntimeConfig.sleep_wake_swaps); "" elsewhere.
            "mechanism": w.get("swap_mechanism", "") or "",
            "matched": False,
        })
    for i, w in enumerate(trace["agg_waits"]):
        gates.append({
            "kind": "agg_wait", "idx": i, "model": w.get("model", ""),
            "t_exit": w.get("t", 0.0), "wait_s": float(w.get("wait_s") or 0.0),
            "on_demand": bool(w.get("on_demand_start")),
            "prefetch_scheduled": bool(w.get("prefetch_started")),
            "mechanism": w.get("swap_mechanism", "") or "",
            "matched": False,
        })
    gates.sort(key=lambda g: g["t_exit"])
    return gates


def match_gate(gates: list[dict], name: str,
               anchor_t: float | None) -> dict | None:
    """Earliest gate for this model at/after the task's anchor time.

    The vllm_model and model_cache tasks of one swap share a gate; the pool
    workloads boot the same engine several times (resource ids suffixed _tN),
    so we match by model name in time order WITHOUT consuming the gate —
    gate_group keeps the shared-interval bookkeeping honest downstream.
    """
    cands = [g for g in gates if g["model"] == name]
    if not cands:
        return None
    if anchor_t is not None:
        after = [g for g in cands if g["t_exit"] >= anchor_t - 1.0]
        if after:
            return after[0]
    return cands[-1]


# ---------------------------------------------------------------------------
# Per-trial row construction
# ---------------------------------------------------------------------------

def trial_rows(base: dict, trace: dict, config: str, mode: str) -> list[dict]:
    rows: list[dict] = []

    # trace zero = earliest timestamp across the collections we use
    t0 = None
    for coll in ("tool_calls", "swap_waits", "agg_waits", "mace_loads"):
        for e in trace[coll]:
            if e.get("t"):
                t0 = e["t"] if t0 is None else min(t0, e["t"])
    for p in trace["predictions"]:
        if p.get("t"):
            t0 = p["t"] if t0 is None else min(t0, p["t"])
    for rec in trace["tasks"].values():
        if rec.get("started_t"):
            t0 = rec["started_t"] if t0 is None else min(t0, rec["started_t"])
    if t0 is None:
        t0 = 0.0

    def rel(t):
        return round(t - t0, 3) if t else None

    used_rids = used_resource_ids(trace["tasks"])
    gates = collect_gates(trace)

    # earliest prediction / decision per resource_id
    first_pred: dict[str, dict] = {}
    for p in trace["predictions"]:
        rid = p["resource_id"]
        if rid not in first_pred or p["t"] < first_pred[rid]["t"]:
            first_pred[rid] = p
    first_dec: dict[str, dict] = {}
    for d in trace["decisions"]:
        rid = d.get("resource_id", "")
        if rid and (rid not in first_dec or d["t"] < first_dec[rid]["t"]):
            first_dec[rid] = d

    def consumer_need_t(consumer_tool: str, after_t: float | None):
        if not consumer_tool:
            return None
        for tc in trace["tool_calls"]:
            if tc["tool"] == consumer_tool and (after_t is None
                                                or tc["t"] >= after_t):
                return tc["t"]
        return None

    # ---- (a) admitted prefetch tasks -------------------------------------
    task_rids = set()
    for rec in trace["tasks"].values():
        rid = rec.get("resource_id", "")
        task_rids.add(rid)
        name = rec.get("resource_name", "")
        rtype = rec.get("resource_type", "")
        pred = first_pred.get(rid)
        dec = first_dec.get(rid)
        bts, prov = task_bytes(rec)
        outcome = classify_task(rec, used_rids)

        consumed_t = rec.get("consumed_t")
        completed_t = rec.get("completed_t")
        gate = (match_gate(gates, name, rec.get("started_t"))
                if rtype in ("vllm_model", "model_cache") else None)
        if gate is not None:
            gate["matched"] = True
            need_t = gate["t_exit"] - gate["wait_s"]
            need_source = gate["kind"]
            exposure = gate["wait_s"]
            gate_group = f"{gate['kind']}:{gate['model']}:{gate['idx']}"
        elif consumed_t is not None:
            need_t = consumed_t
            need_source = "consumed"
            exposure = (max(0.0, completed_t - consumed_t)
                        if completed_t is not None else None)
            gate_group = ""
        else:
            need_t = consumer_need_t(rec.get("consumer_tool")
                                     or (pred or {}).get("consumer_tool", ""),
                                     (pred or {}).get("t"))
            if need_t is not None:
                need_source = "consumer_tool_call"
                exposure = (max(0.0, completed_t - need_t)
                            if completed_t is not None else None)
            else:
                need_source = "none"
                exposure = 0.0
            gate_group = ""

        t_pred = (pred or {}).get("t") or rec.get("started_t")
        window = (need_t - t_pred) if (need_t and t_pred) else None
        transfer = rec.get("elapsed_s")
        start_delay = (rec["started_t"] - t_pred
                       if rec.get("started_t") and t_pred else None)
        simulated = bool(mode == "simulated"
                         or (transfer is not None
                             and 0 < transfer < SIM_ELAPSED_S
                             and bts > 1e9))
        rows.append({
            **base,
            "resource_id": rid, "resource_name": name, "resource_type": rtype,
            "consumer_tool": (rec.get("consumer_tool")
                              or (pred or {}).get("consumer_tool", "")),
            "executor": rec.get("executor", ""),
            "backend": rec.get("backend", ""),
            "n_shards": rec.get("n_shards"),
            "predictor_id": (pred or {}).get("predictor_id", ""),
            "confidence": (pred or {}).get("confidence"),
            "bytes": int(bts) if bts else None,
            "bytes_provenance": prov,
            "t_predicted": rel(t_pred),
            "t_decision": rel((dec or {}).get("t")),
            "decision_action": (dec or {}).get("action", ""),
            "decision_reason": (dec or {}).get("reason", ""),
            "t_started": rel(rec.get("started_t")),
            "t_completed": rel(completed_t),
            "t_cancelled": rel(rec.get("cancelled_t")),
            "t_first_needed": rel(need_t),
            "need_source": need_source,
            "t_consumed": rel(consumed_t),
            "t_predicted_epoch": t_pred,
            "t_first_needed_epoch": need_t,
            "transfer_s": _r(transfer),
            "achieved_gbps": (_r(bts / transfer / 1e9, 3)
                              if bts and transfer and transfer >= SIM_ELAPSED_S
                              else None),
            "event_gbps": rec.get("gb_per_s"),
            "window_s": _r(window),
            "load_vs_window": (_r((transfer or rec.get("estimated_load_s")
                                   or 0.0) / window, 3)
                               if window and window > 0 else None),
            "lead_time_s": (_r(need_t - completed_t)
                            if need_t and completed_t is not None else None),
            "lateness_s": (_r(max(0.0, completed_t - need_t))
                           if need_t and completed_t is not None else None),
            "exposure_s": _r(exposure),
            "start_delay_s": _r(start_delay),
            "outcome": outcome,
            "gate_group": gate_group,
            "gate_on_demand": gate["on_demand"] if gate else None,
            "gate_prefetch_scheduled": (gate["prefetch_scheduled"]
                                        if gate else None),
            "swap_mechanism": gate["mechanism"] if gate else "",
            "simulated": simulated,
            "estimated_load_s": rec.get("estimated_load_s"),
        })

    # ---- (b) skipped decisions (no task ever existed) ---------------------
    for rid, dec in first_dec.items():
        if dec.get("action") != "skip" or rid in task_rids:
            continue
        pred = first_pred.get(rid, {})
        name = pred.get("resource_name", rid[:12])
        need_t = consumer_need_t(pred.get("consumer_tool", ""), pred.get("t"))
        est_load = pred.get("estimated_load_s") or dec.get("estimated_load_s")
        rows.append({
            **base,
            "resource_id": rid, "resource_name": name,
            "resource_type": pred.get("resource_type", ""),
            "consumer_tool": pred.get("consumer_tool", ""),
            "predictor_id": pred.get("predictor_id", ""),
            "confidence": pred.get("confidence"),
            "bytes": (int(pred["estimated_size_bytes"])
                      if pred.get("estimated_size_bytes") else None),
            "bytes_provenance": ("estimated"
                                 if pred.get("estimated_size_bytes")
                                 else "unknown"),
            "t_predicted": rel(pred.get("t")),
            "t_decision": rel(dec.get("t")),
            "decision_action": "skip",
            "decision_reason": dec.get("reason", ""),
            "t_first_needed": rel(need_t),
            "need_source": ("consumer_tool_call" if need_t is not None
                            else "none"),
            "t_predicted_epoch": pred.get("t"),
            "t_first_needed_epoch": need_t,
            "window_s": (_r(need_t - pred["t"])
                         if need_t and pred.get("t") else None),
            # skipped => nothing staged; if the need materialized, the full
            # estimated load sat on the critical path
            "exposure_s": _r(est_load) if need_t is not None else 0.0,
            "outcome": "skipped",
            "estimated_load_s": est_load,
            "simulated": mode == "simulated",
        })

    # ---- (c) needs with no prediction ------------------------------------
    no_pref = ("no_prefetch_config" if config in NO_PREFETCH_CONFIGS
               else "no_prediction")
    for gate in gates:
        if gate["matched"]:
            continue
        # Option D aggregator (and any adapter-fired prefetch thread) never
        # goes through the scheduler, so no task exists to match — the gate's
        # own prefetch_started flag is the ground truth for those.  Its start
        # time is the nearest preceding aggregator_prefetch_start stamp; the
        # completion instant is NOT instrumented (the thread ends silently),
        # so t_completed stays null — plots must render an open-ended span.
        direct = gate["prefetch_scheduled"] and not gate["on_demand"]
        t_direct_start = None
        if direct and gate["kind"] == "agg_wait":
            need_epoch = gate["t_exit"] - gate["wait_s"]
            starts = [s["t"] for s in trace.get("agg_prefetch_starts", [])
                      if s.get("t") and s["t"] <= need_epoch]
            t_direct_start = max(starts) if starts else None
        rows.append({
            "t_started": rel(t_direct_start),
            **base,
            "resource_name": gate["model"],
            "resource_type": "vllm_model",
            "t_first_needed": rel(gate["t_exit"] - gate["wait_s"]),
            "need_source": gate["kind"],
            "t_first_needed_epoch": gate["t_exit"] - gate["wait_s"],
            "exposure_s": _r(gate["wait_s"]),
            "gate_group": f"{gate['kind']}:{gate['model']}:{gate['idx']}",
            "gate_on_demand": gate["on_demand"],
            "gate_prefetch_scheduled": gate["prefetch_scheduled"],
            "swap_mechanism": gate["mechanism"],
            "outcome": "direct_prefetch" if direct else no_pref,
            "bytes_provenance": "unknown",
            "simulated": mode == "simulated",
        })
    for m in trace["mace_loads"]:
        if m.get("source") != "disk":
            continue
        rows.append({
            **base,
            "resource_name": m.get("model", "mace"),
            "resource_type": "mace_model",
            "t_first_needed": rel(m.get("t")),
            "need_source": "mace_load",
            "t_first_needed_epoch": m.get("t"),
            "exposure_s": _r(m.get("duration_s")),
            "outcome": no_pref,
            "bytes_provenance": "unknown",
            "simulated": mode == "simulated",
        })
    consumed_task_rids = {rec.get("resource_id")
                          for rec in trace["tasks"].values()
                          if rec.get("consumed_t") is not None}
    for c in trace["consumed"]:
        if c.get("status") != "no_prefetch":
            continue
        rid = c.get("resource_id", "")
        if rid in consumed_task_rids:
            continue
        pred = first_pred.get(rid, {})
        rows.append({
            **base,
            "resource_id": rid,
            "resource_name": pred.get("resource_name", rid[:12]),
            "resource_type": pred.get("resource_type", ""),
            "t_first_needed": rel(c.get("t")),
            "need_source": "consumed",
            "t_first_needed_epoch": c.get("t"),
            "exposure_s": _r(pred.get("estimated_load_s")),
            "outcome": no_pref,
            "bytes": (int(pred["estimated_size_bytes"])
                      if pred.get("estimated_size_bytes") else None),
            "bytes_provenance": ("estimated"
                                 if pred.get("estimated_size_bytes")
                                 else "unknown"),
            "simulated": mode == "simulated",
        })

    # ---- stall classification (needs the trial's full row set) ------------
    cache_ontime_groups = {
        r.get("gate_group") for r in rows
        if r.get("resource_type") == "model_cache" and r.get("gate_group")
        and r.get("t_completed") is not None
        and r.get("t_first_needed") is not None
        and r["t_completed"] <= r["t_first_needed"] + 0.5
    }
    for r in rows:
        r["stall_class"] = _stall_class(r, cache_ontime_groups)
    return rows


def bringup_cost_s(r: dict) -> tuple[float | None, str]:
    """The measured cost of making THIS resource ready, in THIS trial.

    Returns (cost_s, provenance).  Priority:
      measured   transfer_s — the task's own elapsed_s.  For a vllm_model task
                 this is the end-to-end bring-up (cold boot OR sleep-mode wake),
                 already inclusive of engine init, so it is the ground truth for
                 which regime the row is in.  Preferred over any table because
                 it is per-trial, per-platform and per-mechanism: it never
                 requires pooling L40S with Blackwell numbers, and it cannot go
                 stale when a new bring-up mechanism is added.
      estimated  estimated_load_s — the resource's declared load latency, used
                 when the object was never staged (skips, megammap_stage rows
                 whose transfer is not instrumented).
      unknown    neither available; caller falls back to the cold-boot-era caps.
    """
    t = r.get("transfer_s")
    if isinstance(t, (int, float)) and t > 0:
        return float(t), "measured"
    e = r.get("estimated_load_s")
    if isinstance(e, (int, float)) and e > 0:
        return float(e), "estimated"
    return None, "unknown"


def window_floors(r: dict) -> tuple[float, float, float | None, str]:
    """(w_min_s, spinup_s, bringup_s, provenance) for one row.

    A floor may never exceed the bring-up cost it gates.  That single rule is
    what makes the taxonomy regime-independent:

      w_min  = min(W_MIN_CAP_S,  NO_WINDOW_FRAC * bringup)
      spinup = min(SPINUP_CAP_S, SPINUP_FRAC   * bringup)   [vllm_model only]

    Cold boot (bringup 500-1300 s): both fractions exceed their caps, min()
    selects the cap, and the result is byte-identical to the pre-2026-08 code.
    Sleep-mode wake (bringup 0.8-2.1 s): the caps are ~10-20x the entire cost
    of the operation being judged, so min() selects the bring-up-relative term
    and a 5 s window is correctly read as ample rather than as "no window".
    """
    bringup, prov = bringup_cost_s(r)
    if bringup is None:
        # No measurement and no estimate — keep the historical constants rather
        # than invent a cost.  These rows are reported under provenance
        # 'unknown' so the fallback is never silent.
        w_min = W_MIN_CAP_S
        spinup = (SPINUP_CAP_S if r.get("resource_type") == "vllm_model"
                  else 0.0)
        return w_min, spinup, None, prov
    w_min = min(W_MIN_CAP_S, NO_WINDOW_FRAC * bringup)
    spinup = (min(SPINUP_CAP_S, SPINUP_FRAC * bringup)
              if r.get("resource_type") == "vllm_model" else 0.0)
    return w_min, spinup, bringup, prov


def _stall_class(r: dict, cache_ontime_groups: set) -> str:
    exposure = r.get("exposure_s")
    if not exposure or exposure <= 0.1:
        return ""
    if r.get("outcome") == "no_prefetch_config":
        return "baseline_no_prefetch"
    if r.get("outcome") == "no_prediction":
        return "no_prediction"
    if r.get("outcome") == "skipped":
        return f"policy_skip:{r.get('decision_reason', '')}"
    if r.get("outcome") == "direct_prefetch":
        return "residual_partial"
    if (r.get("resource_type") == "vllm_model"
            and r.get("gate_group") in cache_ontime_groups):
        return "gpu_serialization_residual"
    window = r.get("window_s")
    if window is None:
        return "unattributed"
    w_min, spinup, bringup, prov = window_floors(r)
    # Audit trail: every window-branch decision records the floors it used and
    # where the bring-up cost came from, so a class can be re-derived from the
    # CSV without re-reading the trace.
    r["bringup_cost_s"] = _r(bringup)
    r["bringup_provenance"] = prov
    r["w_min_s"] = _r(w_min, 3)
    r["spinup_s"] = _r(spinup, 3)
    if window < w_min:
        return "no_window"
    transfer = r.get("transfer_s") or r.get("estimated_load_s") or 0.0
    if window < transfer + spinup:
        return "window_too_small"
    delay = r.get("start_delay_s")
    if delay is not None and (window - delay) < transfer + spinup:
        return "late_start"
    return "residual_partial"


# ---------------------------------------------------------------------------
# AtomAgents: per-interval swap waits from the runner's metrics CSV
# ---------------------------------------------------------------------------

def atomagents_need_rows(base: dict, trial_dir: Path, config: str,
                         trace_tasks: dict[str, dict]) -> list[dict]:
    """model_swap_wait:* intervals as need rows.

    The metrics CSV is the ONLY stall source for AtomAgents (trace-task
    residuals cover the same wall intervals — parse_eval_traces uses metrics
    exclusively too, so lifecycle trace-task rows carry exposure_s=None for
    these workloads).  Each interval is classified against the trace's
    prefetch tasks: a task for the same model already in flight when the wait
    began means the wait is the prefetch's residual, not an unpredicted swap.

    Bytes are left unknown (the metrics CSV carries no sizes; inventing them
    from static tables would contaminate provenance).  Pre-instrumentation
    trials (no model_swap_wait rows) emit nothing — their stall is real but
    unmeasured, matching parse_eval_traces's None-not-zero rule.
    """
    path = trial_dir / "metrics.csv"
    if not path.is_file():
        return []
    outcome = ("no_prefetch_config" if config in NO_PREFETCH_CONFIGS
               else "no_prediction")
    t0 = None
    intervals: list[tuple[float, float, str]] = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                phase = (r.get("phase") or "").strip()
                if not phase or phase == "SUMMARY":
                    continue
                try:
                    dur = float(r.get("duration_s") or 0.0)
                    end = datetime.fromisoformat(r["timestamp"]).timestamp()
                except (ValueError, KeyError, TypeError):
                    continue
                if t0 is None or end - dur < t0:
                    t0 = end - dur
                if phase.startswith("model_swap_wait:"):
                    intervals.append((end - dur, end, phase.split(":", 1)[1]))
    except OSError:
        return []
    rows = []
    # tasks by model name, for interval-vs-prefetch matching
    tasks_by_name: dict[str, list[dict]] = defaultdict(list)
    for rec in trace_tasks.values():
        if rec.get("resource_name"):
            tasks_by_name[rec["resource_name"]].append(rec)
    for start, end, model in intervals:
        in_flight = next(
            (rec for rec in tasks_by_name.get(model, [])
             if rec.get("started_t") is not None
             and rec["started_t"] <= end
             and (rec.get("completed_t") or end) >= start - 5.0),
            None)
        if outcome == "no_prefetch_config":
            row_outcome, cls = outcome, "baseline_no_prefetch"
        elif in_flight is not None:
            row_outcome = "prefetch_residual"
            cls = ("late_start" if in_flight["started_t"] > start
                   else "residual_partial")
        else:
            row_outcome, cls = "no_prediction", "no_prediction"
        rows.append({
            **base,
            "resource_name": model,
            "resource_type": "vllm_model",
            "t_first_needed": round(start - (t0 or start), 3),
            "need_source": "atomagents_metrics",
            "t_first_needed_epoch": start,
            "exposure_s": round(end - start, 2),
            "outcome": row_outcome,
            "bytes_provenance": "unknown",
            "stall_class": cls,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--config", default=None, help="restrict to one config")
    ap.add_argument("--include-failed", action="store_true",
                    help="debug only — plots must not use failed trials")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    runs = discover_runs(eval_root)
    if not runs:
        print(f"No runs under {eval_root}/runs", file=sys.stderr)
        sys.exit(1)

    # first trial per SLURM allocation (same rule as parse_eval_traces)
    first_in_alloc: dict[str, tuple[str, str]] = {}
    for run in runs:
        job = run["meta"].get("slurm_job_id")
        start = run["meta"].get("start_time", "")
        if not job or not start:
            continue
        cur = first_in_alloc.get(job)
        if cur is None or start < cur[0]:
            first_in_alloc[job] = (start, run["meta"].get("run_id", ""))
    first_run_ids = {v[1] for v in first_in_alloc.values()}

    all_rows: list[dict] = []
    n_included = n_excluded = 0
    for run in runs:
        meta, summary = run["meta"], run["summary"]
        wl = meta.get("workload", "?")
        cfg = meta.get("config", "?")
        if args.workload and wl != args.workload:
            continue
        if args.config and cfg != args.config:
            continue
        status = meta.get("status", "?")
        if status != "completed" and not args.include_failed:
            n_excluded += 1
            continue
        trace = (parse_trace(load_events(run["trace_path"]))
                 if run["trace_path"] else parse_trace([]))
        n_included += 1
        run_id = meta.get("run_id", "?")
        mode = summary.get("mode", meta.get("runtime_mode")) or ""
        base = {
            "workload": wl, "config": cfg,
            "trial_index": meta.get("trial_index"),
            "run_id": run_id, "trial_dir": str(run["trial_dir"].name),
            "status": status, "mode": mode,
            "git_commit": meta.get("git_commit"),
            "node": meta.get("node"),
            "gpu_name": (summary.get("gpu_name")
                         or (meta.get("gpus") or [""])[0].split(",")[0]),
            "slurm_job_id": meta.get("slurm_job_id"),
            "first_in_allocation": run_id in first_run_ids,
        }
        t_rows = trial_rows(base, trace, cfg, mode)
        if wl.startswith("atomagents"):
            # metrics.csv is the sole stall source for AtomAgents; blank the
            # trace-task exposures so the two sources never double-count the
            # same wall interval (parse_eval_traces does the same).
            for r in t_rows:
                r["exposure_s"] = None
                r["stall_class"] = ""
            all_rows.extend(t_rows)
            all_rows.extend(
                atomagents_need_rows(base, run["trial_dir"], cfg,
                                     trace["tasks"]))
        else:
            all_rows.extend(t_rows)

    # union of columns, stable order from first occurrence
    cols: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = eval_root / "eval_prefetch_lifecycle.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"{out}  ({len(all_rows)} rows; trials included={n_included} "
          f"excluded={n_excluded})")

    # ---- taxonomy aggregate ----------------------------------------------
    # gate_group max-dedup: within one (run_id, gate_group) only the largest
    # exposure counts (server + cache rows share the same wall interval).
    agg: dict[tuple[str, str, str], float] = defaultdict(float)
    seen_gate: dict[tuple[str, str], tuple[str, float]] = {}
    for r in all_rows:
        cls = r.get("stall_class") or ""
        if not cls:
            continue
        exp = float(r.get("exposure_s") or 0.0)
        gg = r.get("gate_group") or ""
        if gg:
            key = (r["run_id"], gg)
            prev = seen_gate.get(key)
            if prev is not None:
                prev_cls, prev_exp = prev
                if exp <= prev_exp:
                    continue
                agg[(r["workload"], r["config"], prev_cls)] -= prev_exp
            seen_gate[key] = (cls, exp)
        agg[(r["workload"], r["config"], cls)] += exp
    n_trials: dict[tuple[str, str], set] = defaultdict(set)
    for r in all_rows:
        n_trials[(r["workload"], r["config"])].add(r["run_id"])
    tax = eval_root / "eval_stall_taxonomy.csv"
    with open(tax, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "config", "stall_class", "exposed_stall_s",
                    "n_trials", "stall_s_per_trial"])
        for (wl, cfg, cls), s in sorted(agg.items()):
            n = len(n_trials[(wl, cfg)]) or 1
            w.writerow([wl, cfg, cls, round(s, 2), n, round(s / n, 2)])
    print(f"{tax}  ({len(agg)} cells)")


if __name__ == "__main__":
    main()
