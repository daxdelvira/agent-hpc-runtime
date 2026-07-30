#!/usr/bin/env python3
"""
parse_eval_traces.py — normalize Q1–Q4 evaluation runs into analysis CSVs.

Input : the result tree written by experiments/run_eval_q1_q4.py
        (results/eval_q1_q4/runs/<workload>/<config>/<trial>/{meta.json,
         summary.json,trace.jsonl,system_profile.csv})
Output: results/eval_q1_q4/
          eval_q1_summary.csv          one row per successful run (wall time, GPU idle)
          eval_q2_breakdown.csv        one row per run (time-breakdown buckets)
          eval_q3_prediction_quality.csv  one row per run (prediction outcome counts)
          eval_q3_predictions_long.csv one row per admitted/emitted prediction
          eval_q4_speculation_cost.csv one row per run (bytes, cancellations, memory)
          eval_validation_report.txt   data-quality assertions & excluded runs

Failed / timed-out runs are NEVER averaged into the CSVs; they are listed in the
validation report with the reason for exclusion.

Derived-metric definitions (documented in results/eval_q1_q4/README.md):
  exposed_stall_s   = worker_swap_wait.wait_s (workflow blocked waiting for the
                      worker LLM) + per-resource residual stall
                      (prefetch_completed after resource_consumed)
  overlapped_io_s   = Σ per used prefetch task of
                      min(completed_t, consumed_t) − started_t  (clipped ≥ 0)
  residual_load_s   = Σ max(0, completed_t − consumed_t) over used tasks
  overlap_recovered = overlapped_io_s / (overlapped_io_s + exposed_stall_s)
  agent reasoning   = Σ chain spans of LLM agent nodes − exposed_stall inside them
  tool execution    = Σ tool_end.duration_s
  other             = wall − reasoning − tools − exposed stall − residual

Usage
-----
    python scripts/parse_eval_traces.py                       # default tree
    python scripts/parse_eval_traces.py --eval-root results/eval_q1_q4
    python scripts/parse_eval_traces.py --include-failed      # debug only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "results" / "eval_q1_q4"

# LLM-inference nodes per workload graph (chain spans counted as agent reasoning).
LLM_NODES = {"PlannerAgent", "WorkerAgent", "AggregatorAgent",
             "engineer", "planner", "critic", "code_specialist"}

# Fallback consumer→resource map (from runtime/predictor/data/tool_resources.json)
# used to count missed prefetch opportunities: a consumer tool fired but no
# prefetch task existed for its resource.
def load_consumer_map() -> dict[str, list[str]]:
    path = PROJECT_ROOT / "runtime" / "predictor" / "data" / "tool_resources.json"
    out: dict[str, list[str]] = defaultdict(list)
    try:
        for e in json.loads(path.read_text()):
            tool = e.get("consumer_tool")
            if tool:
                out[tool].append(e.get("name", ""))
    except Exception:
        pass
    return dict(out)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def parse_trace(events: list[dict], consumer_of: dict[str, set] | None = None) -> dict:
    """
    One pass over a runtime trace → everything Q2/Q3/Q4 need.

    consumer_of maps resource_name -> set of tools that really consume it;
    consumption stamps coinciding with a non-consumer tool are ignored
    (legacy traces validated MACE at smiles_to_coordinate_file due to a bad
    tool_resources.json entry, mislabelling in-time prefetches as late).
    """
    consumer_of = consumer_of or {}
    # --- chain spans (agent reasoning) -----------------------------------
    open_chains: dict[str, tuple[str, float]] = {}   # lc run_id -> (node, t)
    node_spans: dict[str, float] = defaultdict(float)
    # --- tools -------------------------------------------------------------
    tool_calls: list[dict] = []          # {tool, t, step?}
    tool_exec_s = 0.0
    # --- swap waits ---------------------------------------------------------
    swap_waits: list[dict] = []
    # Option D: AggregatorAgent blocking on the co-resident aggregator model.
    # A SEPARATE gate from the worker swap (different wall interval), so it is
    # never deduped against worker_swap_wait.
    agg_waits: list[dict] = []
    # Adapter-fired (non-scheduler) aggregator prefetch thread start stamps —
    # the only start-time record a direct prefetch leaves in the trace.
    agg_prefetch_starts: list[dict] = []
    # --- MACE calculator loads (emitted in ALL modes incl. baseline) --------
    mace_loads: list[dict] = []
    # --- predictions (join by step via checkpoint_created) ------------------
    predictions: list[dict] = []         # per prediction_result resource
    step_checkpoint: dict[int, str] = {}
    validations: list[dict] = []
    divergences: list[dict] = []
    conservative_events: list[dict] = []
    plan_extracted = False
    # --- prefetch task lifecycle --------------------------------------------
    tasks: dict[str, dict] = {}
    resource_meta: dict[str, dict] = {}
    resource_to_task: dict[str, str] = {}
    decisions: list[dict] = []
    consumed: list[dict] = []
    last_tool: tuple[str, float] | None = None
    t_first = None
    t_last = None

    for ev in events:
        et = ev.get("event_type", "")
        p = ev.get("payload", {}) or {}
        t = ev.get("epoch_time", 0.0)
        step = ev.get("step", 0)
        if t:
            t_first = t if t_first is None else min(t_first, t)
            t_last = t if t_last is None else max(t_last, t)

        if et == "chain_start":
            open_chains[str(p.get("run_id"))] = (p.get("node", ""), t)
        elif et == "chain_end":
            key = str(p.get("run_id"))
            if key in open_chains:
                node, t0 = open_chains.pop(key)
                node_spans[node] += max(0.0, t - t0)
        elif et == "tool_call":
            tool_calls.append({"tool": p.get("tool", ""), "t": t})
            last_tool = (p.get("tool", ""), t)
        elif et == "tool_end":
            try:
                tool_exec_s += float(p.get("duration_s", 0.0))
            except (TypeError, ValueError):
                pass
        elif et == "worker_swap_wait":
            swap_waits.append({**p, "t": t, "step": step})
        elif et == "aggregator_swap_wait":
            agg_waits.append({**p, "t": t, "step": step})
        elif et == "aggregator_prefetch_start":
            agg_prefetch_starts.append({**p, "t": t, "step": step})
        elif et == "mace_load":
            mace_loads.append({**p, "t": t})
        elif et == "plan_extracted":
            plan_extracted = True
        elif et == "prediction_result":
            for r in p.get("resources", []):
                rid = r.get("resource_id", "")
                resource_meta[rid] = r
                predictions.append({
                    "step": step,
                    "t": t,
                    "resource_id": rid,
                    "resource_type": r.get("resource_type", ""),
                    "resource_name": r.get("name", ""),
                    "confidence": r.get("confidence", 0.0),
                    "consumer_tool": r.get("consumer_tool", ""),
                    "expected_at_step": r.get("expected_at_step", 0),
                    "estimated_load_s": r.get("estimated_load_s"),
                    "estimated_size_bytes": r.get("estimated_size_bytes"),
                    "predictor_id": p.get("predictor_id", ""),
                })
        elif et == "checkpoint_created":
            step_checkpoint[step] = p.get("checkpoint_id", "")
        elif et == "prediction_validated":
            validations.append({**p, "step": step, "t": t})
        elif et == "divergence_detected":
            divergences.append({**p, "step": step, "t": t})
        elif et == "conservative_mode":
            conservative_events.append({**p, "step": step, "t": t})
        elif et == "prefetch_decision":
            decisions.append({**p, "step": step, "t": t})
        elif et == "prefetch_started":
            tid = p.get("task_id", "")
            rid = p.get("resource_id", "")
            tasks[tid] = {"task_id": tid, "resource_id": rid, "started_t": t,
                          "started_step": step,
                          "executor": p.get("executor", "")}
            resource_to_task[rid] = tid
        elif et == "prefetch_completed":
            tid = p.get("task_id", "")
            rec = tasks.setdefault(tid, {"task_id": tid})
            elapsed = float(p.get("elapsed_s", 0.0) or 0.0)
            rec["elapsed_s"] = elapsed
            st = rec.get("started_t")
            rec["completed_t"] = (st + elapsed) if (st and elapsed > 0) else t
            rec["completion_status"] = p.get("status", "")
            if "bytes_staged" in p:
                rec["bytes_measured"] = p.get("bytes_staged")
            for extra in ("gb_per_s", "n_shards", "backend"):
                if extra in p:
                    rec[extra] = p.get(extra)
        elif et == "prefetch_cancelled":
            tid = p.get("task_id", "")
            rec = tasks.setdefault(tid, {"task_id": tid})
            rec["cancelled"] = True
            rec["cancelled_t"] = t
            rec["wasted_flag"] = bool(p.get("wasted", False))
        elif et == "resource_consumed":
            consumed.append({**p, "step": step, "t": t})
            tid = p.get("task_id")
            if tid and tid in tasks:
                # Drop stamps emitted by a tool that is not a real consumer of
                # this resource (validation fired on the wrong tool).
                rname = resource_meta.get(p.get("resource_id", ""), {}).get("name", "")
                allowed = consumer_of.get(rname)
                if (allowed and last_tool is not None
                        and abs(t - last_tool[1]) < 0.25
                        and last_tool[0] not in allowed):
                    continue
                tasks[tid]["consumed_t"] = t
                tasks[tid]["consumed_step"] = step
                tasks[tid]["consumed_status"] = p.get("status", "")

    # Enrich tasks with resource metadata
    for rec in tasks.values():
        meta = resource_meta.get(rec.get("resource_id", ""), {})
        rec["resource_type"] = meta.get("resource_type", "")
        rec["resource_name"] = meta.get("name", rec.get("resource_id", "")[:12])
        rec["estimated_load_s"] = meta.get("estimated_load_s")
        rec["estimated_size_bytes"] = meta.get("estimated_size_bytes")
        rec["cancellation_safe"] = meta.get("cancellation_safe", True)

    return {
        "node_spans": dict(node_spans),
        "tool_calls": tool_calls,
        "tool_exec_s": tool_exec_s,
        "swap_waits": swap_waits,
        "agg_waits": agg_waits,
        "agg_prefetch_starts": agg_prefetch_starts,
        "mace_loads": mace_loads,
        "predictions": predictions,
        "step_checkpoint": step_checkpoint,
        "validations": validations,
        "divergences": divergences,
        "conservative_events": conservative_events,
        "plan_extracted": plan_extracted,
        "tasks": tasks,
        "decisions": decisions,
        "consumed": consumed,
        "trace_span_s": (t_last - t_first) if (t_first and t_last) else None,
    }


# ---------------------------------------------------------------------------
# Per-task classification (Q3/Q4)
# ---------------------------------------------------------------------------

_LATE_EPS = 0.5   # seconds of grace before a hit counts as "late"


def classify_task(rec: dict, used_resources: set[str] | None = None) -> str:
    """useful | late | wasted | cancelled | duplicate | pending

    `used_resources`: resource_ids that were consumed via *some* task.  A
    completed-but-unconsumed task whose resource WAS served by another task is
    a scheduler dedup artifact (duplicate), not genuinely wasted speculation.
    """
    consumed_t = rec.get("consumed_t")
    completed_t = rec.get("completed_t")
    if rec.get("consumed_status") == "used" and consumed_t is not None:
        if completed_t is not None and completed_t > consumed_t + _LATE_EPS:
            return "late"
        return "useful"
    if rec.get("wasted_flag"):
        return "wasted"
    if rec.get("cancelled"):
        return "cancelled"
    if completed_t is not None:
        if used_resources and rec.get("resource_id") in used_resources:
            return "duplicate"
        return "wasted"          # completed but never consumed
    return "pending"             # never completed nor consumed (run ended first)


def used_resource_ids(tasks: dict[str, dict]) -> set[str]:
    return {rec.get("resource_id", "") for rec in tasks.values()
            if rec.get("consumed_status") == "used"}


def task_bytes(rec: dict) -> tuple[float, str]:
    if rec.get("bytes_measured") is not None:
        return float(rec["bytes_measured"]), "measured"
    if rec.get("estimated_size_bytes"):
        return float(rec["estimated_size_bytes"]), "estimated"
    return 0.0, "unknown"


def task_overlap_stall(rec: dict) -> tuple[float, float]:
    """(overlapped_io_s, residual_stall_s) for a used task."""
    st, ct, ut = rec.get("started_t"), rec.get("completed_t"), rec.get("consumed_t")
    if st is None or ut is None:
        return 0.0, 0.0
    end = min(ct, ut) if ct is not None else ut
    overlap = max(0.0, end - st)
    stall = max(0.0, (ct - ut)) if ct is not None else 0.0
    return overlap, stall


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(eval_root: Path) -> list[dict]:
    runs = []
    for meta_path in sorted(eval_root.glob("runs/*/*/*/meta.json")):
        trial_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        summary = {}
        sp = trial_dir / "summary.json"
        if sp.exists():
            try:
                summary = json.loads(sp.read_text())
            except Exception:
                pass
        runs.append({
            "trial_dir": trial_dir,
            "meta": meta,
            "summary": summary,
            "trace_path": (trial_dir / "trace.jsonl"
                           if (trial_dir / "trace.jsonl").exists() else None),
            "profile_path": (trial_dir / "system_profile.csv"
                             if (trial_dir / "system_profile.csv").exists() else None),
        })
    return runs


def _merge_intervals(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(iv):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _interval_len(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def _interval_overlap(x: list[tuple[float, float]],
                      y: list[tuple[float, float]]) -> float:
    total = 0.0
    for a, b in x:
        for c, d in y:
            total += max(0.0, min(b, d) - max(a, c))
    return total


def atomagents_q2(trial_dir: Path, run_id: str,
                  results_dir: Path) -> dict[str, float] | None:
    """Q2 components from the AtomAgents runner's metrics CSV.

    Prefers the trial-dir copy (driver archives it as metrics.csv since
    2026-07-09); falls back to the shared results/ file for older trials.
    Each row's timestamp is written at phase END, so [end-duration, end]
    reconstructs the phase interval.  model_swap_wait spans can sit INSIDE
    agent:* spans (the router fires on LLM calls mid-conversation), so tool
    time is agent-span time MINUS its overlap with swap waits — summing the
    two double-counts (observed +255 s on exp3 baseline t02).  The SUMMARY
    row (whole-wall) is dropped; lammps:* rows nest inside agent spans and
    are not added separately.
    """
    cands = [trial_dir / "metrics.csv",
             results_dir / f"atomagents_metrics_{run_id}.csv"]
    path = next((c for c in cands if c.is_file()), None)
    if path is None:
        return None
    agent_iv: list[tuple[float, float]] = []
    swap_iv: list[tuple[float, float]] = []
    reasoning_s = 0.0
    has_model_rows = False
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
            if phase.startswith(("model_swap_wait:", "model_load:")):
                has_model_rows = True
            if phase.startswith("agent:"):
                agent_iv.append((end - dur, end))
            elif phase.startswith("model_swap_wait:"):
                swap_iv.append((end - dur, end))
            elif phase == "llm:inference":
                reasoning_s += dur
    agent_iv = _merge_intervals(agent_iv)
    swap_iv = _merge_intervals(swap_iv)
    # swap_s=None marks "unavailable", not zero: runs collected before the
    # router metrics wiring (2026-07-09) have neither model_swap_wait nor
    # model_load rows, yet every exp2/exp3 run swaps — their stall is real but
    # unmeasured and must not enter stall aggregates as 0.
    swap_s = _interval_len(swap_iv) if has_model_rows else None
    tool_s = _interval_len(agent_iv) - _interval_overlap(agent_iv, swap_iv)
    return {"tool_s": tool_s, "swap_s": swap_s, "reasoning_s": reasoning_s}


# ---------------------------------------------------------------------------
# GPU idle from system_profile.csv
# ---------------------------------------------------------------------------

def gpu_stats(profile_path: Path) -> dict:
    """GPU idle/util/memory stats over the profiled window."""
    try:
        with open(profile_path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    if len(rows) < 2:
        return {}
    gpu_cols = sorted({c.split("_")[0] for c in rows[0]
                       if c.startswith("gpu") and c.endswith("_util_pct")})
    if not gpu_cols:
        return {}
    idle_s = 0.0
    util_sum = 0.0
    mem_peak = 0.0
    n = 0
    prev_t = float(rows[0]["t_rel_s"])
    for row in rows[1:]:
        t = float(row["t_rel_s"])
        dt = max(0.0, t - prev_t)
        prev_t = t
        try:
            utils = [float(row[f"{g}_util_pct"]) for g in gpu_cols]
            mems = [float(row[f"{g}_mem_used_mb"]) for g in gpu_cols]
        except (KeyError, ValueError):
            continue
        if max(utils) == 0.0:
            idle_s += dt
        util_sum += sum(utils) / len(utils)
        mem_peak = max(mem_peak, sum(mems))
        n += 1
    total_s = float(rows[-1]["t_rel_s"]) - float(rows[0]["t_rel_s"])
    return {
        "gpu_profile_span_s": round(total_s, 1),
        "gpu_all_idle_s": round(idle_s, 1),
        "gpu_all_idle_frac": round(idle_s / total_s, 4) if total_s > 0 else None,
        "gpu_mean_util_pct": round(util_sum / n, 2) if n else None,
        "gpu_peak_mem_mb": round(mem_peak, 1),
    }


# ---------------------------------------------------------------------------
# Main normalization
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    ap.add_argument("--include-failed", action="store_true",
                    help="Also emit rows for failed runs (marked status!=completed); "
                    "for debugging only — plots must not use them")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    runs = discover_runs(eval_root)
    if not runs:
        print(f"No runs found under {eval_root}/runs — run "
              "experiments/run_eval_q1_q4.py first.", file=sys.stderr)
        sys.exit(1)

    consumer_map = load_consumer_map()
    # Inverse map: resource_name -> {tools that really consume it}
    consumer_of: dict[str, set] = defaultdict(set)
    for tool, res_names in consumer_map.items():
        for rn in res_names:
            consumer_of[rn].add(tool)
    consumer_of = dict(consumer_of)

    q1_rows, q2_rows, q3_rows, q3_long, q4_rows = [], [], [], [], []
    report: list[str] = []
    excluded: list[str] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)

    # First trial of each SLURM allocation: closest-to-cold Lustre server-side
    # cache (round-robin keeps later trials comparably server-warm). Tagged so
    # cold-vs-warm sensitivity can be checked; runs without a job id are never
    # tagged rather than guessed.
    first_in_alloc: dict[str, tuple[str, str]] = {}  # job_id -> (start, run_id)
    for run in runs:
        job = run["meta"].get("slurm_job_id")
        start = run["meta"].get("start_time", "")
        if not job or not start:
            continue
        cur = first_in_alloc.get(job)
        if cur is None or start < cur[0]:
            first_in_alloc[job] = (start, run["meta"].get("run_id", ""))
    first_run_ids = {v[1] for v in first_in_alloc.values()}

    for run in runs:
        meta, summary = run["meta"], run["summary"]
        wl = meta.get("workload", "?")
        cfg = meta.get("config", "?")
        run_id = meta.get("run_id", "?")
        status = meta.get("status", "?")
        label = f"{wl}/{cfg}/{run_id}"

        # ---- validation ---------------------------------------------------
        problems = []
        if status != "completed":
            problems.append(f"status={status}")
        if not meta.get("start_time") or not meta.get("end_time"):
            problems.append("missing start/end timestamps")
        wall = summary.get("wall_time_s") or meta.get("wall_time_s")
        if not wall or wall <= 0:
            problems.append("missing/invalid wall_time_s")
        if run["trace_path"] is None and not (
                wl.startswith("atomagents")
                and (run["trial_dir"] / "metrics.csv").is_file()):
            # AtomAgents baseline trials write an EMPTY trace (runtime adapter
            # only logs in real/observe modes); their Q2 source is metrics.csv.
            problems.append("missing trace.jsonl")
        if problems and not args.include_failed:
            excluded.append(f"  EXCLUDED {label}: {', '.join(problems)}")
            continue

        counts[(wl, cfg)] += 1
        trace = (parse_trace(load_events(run["trace_path"]), consumer_of)
                 if run["trace_path"] else parse_trace([]))
        gstats = gpu_stats(run["profile_path"]) if run["profile_path"] else {}
        used_rids = used_resource_ids(trace["tasks"])

        # ---- Q2 breakdown ---------------------------------------------------
        swap_wait_s = sum(w.get("wait_s", 0.0) for w in trace["swap_waits"])
        agg_wait_s = sum(w.get("wait_s", 0.0) for w in trace["agg_waits"])
        overlap_s = 0.0
        residual_swap_s = 0.0    # residual stall on worker-LLM/cache tasks —
        residual_agg_s = 0.0     # aggregator-model tasks gate AggregatorAgent,
        residual_other_s = 0.0   # a different interval than the worker gate
        for rec in trace["tasks"].values():
            if classify_task(rec, used_rids) in ("useful", "late"):
                o, s = task_overlap_stall(rec)
                overlap_s += o
                if rec.get("resource_type") in ("vllm_model", "model_cache"):
                    # max, not sum: the server task and the cache-staging task
                    # stall the SAME worker-gate interval (both consumed at
                    # t_wait0). Summing counted staging's residual (~38 s) on
                    # top of the server's, inflating every real-config run.
                    if "aggregator" in (rec.get("resource_name") or ""):
                        residual_agg_s = max(residual_agg_s, s)
                    else:
                        residual_swap_s = max(residual_swap_s, s)
                else:
                    residual_other_s += s
        # Legacy stall (no prefetch): consumer arrived, no task → estimated full load
        no_prefetch_stall_s = 0.0
        for c in trace["consumed"]:
            if c.get("status") == "no_prefetch":
                meta_r = None
                for p in trace["predictions"]:
                    if p["resource_id"] == c.get("resource_id"):
                        meta_r = p
                        break
                no_prefetch_stall_s += (meta_r or {}).get("estimated_load_s") or 0.0

        # max() dedupes the worker-swap wait: with a prefetch in flight it is
        # reported both as worker_swap_wait and as the worker task's residual.
        # The aggregator gate (Option D) is deduped the same way but summed with
        # the worker gate — they block distinct wall intervals.
        residual_s = residual_swap_s + residual_agg_s + residual_other_s
        exposed_stall_s = (max(swap_wait_s, residual_swap_s)
                           + max(agg_wait_s, residual_agg_s)
                           + residual_other_s + no_prefetch_stall_s)
        reasoning_s = sum(v for k, v in trace["node_spans"].items() if k in LLM_NODES)
        # Swap/aggregator waits happen inside LLM-agent chain spans; don't
        # double count them as reasoning.
        reasoning_s = max(0.0, reasoning_s - swap_wait_s - agg_wait_s)
        tool_s = trace["tool_exec_s"]
        wall_f = float(wall or 0.0)
        other_s = max(0.0, wall_f - reasoning_s - tool_s - exposed_stall_s)

        # ---- AtomAgents: decompose from the runner's metrics CSV -----------
        # AtomAgents traces don't carry chemgraph's node/tool span events
        # (baseline trials write an empty trace), so the trace-derived buckets
        # above are all zero there.  The runner's per-phase metrics CSV is the
        # structured source instead: agent:* spans are the tool/agent work
        # (lammps:* rows are nested inside them and excluded to avoid double
        # counting), model_swap_wait:* rows (router instrumentation added
        # 2026-07-09) are the agent-blocking swap stalls, and llm:inference
        # rows time the API wrappers.  Trials predating the swap-wait rows get
        # no exposed_stall_s — the stall stays inside other_s rather than
        # being guessed.
        if wl.startswith("atomagents"):
            aa = atomagents_q2(run["trial_dir"], run_id, eval_root.parent)
            if aa:
                tool_s = aa["tool_s"]
                reasoning_s = aa["reasoning_s"]
                if aa["swap_s"] is None:
                    # Pre-instrumentation run: swaps happened but were not
                    # measured.  Emit stall as missing (never 0) so stall
                    # aggregates only pool measured trials; the unattributed
                    # remainder (incl. the real stall) stays in other_s.
                    swap_wait_s = exposed_stall_s = None
                    other_s = max(0.0, wall_f - reasoning_s - tool_s)
                else:
                    swap_wait_s = aa["swap_s"]
                    exposed_stall_s = swap_wait_s + no_prefetch_stall_s
                    other_s = max(0.0, wall_f - reasoning_s - tool_s
                                  - exposed_stall_s)

        denom = overlap_s + (exposed_stall_s or 0.0)
        overlap_recovered = (overlap_s / denom) if denom > 0 else None
        # MACE calculator loads (mace_load events exist in every mode, incl.
        # baseline — separates model-load time from the rest of tool time).
        # None (not 0) for traces collected before the event existed.
        if trace["mace_loads"]:
            mace_disk_s = sum(m.get("duration_s", 0.0) for m in trace["mace_loads"]
                              if m.get("source") == "disk")
            mace_cache_s = sum(m.get("duration_s", 0.0) for m in trace["mace_loads"]
                               if m.get("source") == "prefetch_cache")
        else:
            mace_disk_s = mace_cache_s = None

        # ---- Q3 prediction outcomes ------------------------------------------
        n_pred = len(trace["predictions"])
        by_class = defaultdict(int)
        lead_times = []
        for rec in trace["tasks"].values():
            c = classify_task(rec, used_rids)
            by_class[c] += 1
            if c in ("useful", "late") and rec.get("started_t") and rec.get("consumed_t"):
                lead_times.append(rec["consumed_t"] - rec["started_t"])
        n_admitted = len(trace["tasks"])
        n_skipped = sum(1 for d in trace["decisions"] if d.get("action") == "skip")
        n_hits = sum(1 for v in trace["validations"] if v.get("hit"))
        n_miss_validated = sum(1 for v in trace["validations"] if not v.get("hit"))
        n_diverg = len(trace["divergences"])
        # Missed opportunities: a known consumer tool fired but no prefetch task
        # was ever admitted for its resource.  Counted once per tool invocation.
        admitted_names = {rec.get("resource_name") for rec in trace["tasks"].values()}
        missed_names = set()
        for tc in trace["tool_calls"]:
            for res_name in consumer_map.get(tc["tool"], []):
                if res_name not in admitted_names:
                    missed_names.add(res_name)
        n_missed = len(missed_names)   # distinct resources, matching recall's basis
        useful = by_class["useful"]
        # expired = predictions that were never validated nor admitted
        n_validated = len(trace["validations"]) + n_diverg
        n_expired = max(0, n_pred - n_validated - n_admitted)

        # One coherent scheme (open-world speculation has no countable TN, so
        # no "accuracy" is reported — only precision/recall + a timeliness rate):
        #   positive        = admitted prefetch task (cancelled excluded: the
        #                     guard revoking speculation is scored in Q4)
        #   TP (per task)   = task whose resource was consumed (useful, late,
        #                     or duplicate); FP = wasted
        #   FN (per need)   = consumer fired with no admitted prefetch (missed)
        #   on_time_rate    = of resources served by prefetch, share whose
        #                     prefetch completed before the consumer needed it
        n_consumed_tasks = useful + by_class["late"] + by_class["duplicate"]
        denom_adm = n_admitted - by_class["cancelled"]
        precision = n_consumed_tasks / denom_adm if denom_adm else None
        served_rids = used_rids
        n_served = len(served_rids)
        recall = (n_served / (n_served + n_missed)
                  if (n_served + n_missed) else None)
        ontime_rids = {rec.get("resource_id") for rec in trace["tasks"].values()
                       if classify_task(rec, used_rids) == "useful"}
        on_time_rate = (len(ontime_rids & served_rids) / n_served
                        if n_served else None)

        # ---- Q4 bytes ---------------------------------------------------------
        useful_b = wasted_b = cancelled_b = 0.0
        byte_sources = set()
        n_wasted_noncancellable = 0
        spec_read_s = 0.0
        wrong_after_diverg = 0
        first_diverg_t = min((d["t"] for d in trace["divergences"]), default=None)
        for rec in trace["tasks"].values():
            b, src = task_bytes(rec)
            byte_sources.add(src)
            c = classify_task(rec, used_rids)
            if c in ("useful", "late"):
                useful_b += b
            elif c == "wasted":
                wasted_b += b
                if not rec.get("cancellation_safe", True):
                    n_wasted_noncancellable += 1
            elif c == "cancelled":
                # measured bytes_staged reflects what was actually read before stop
                cancelled_b += b if rec.get("bytes_measured") is not None else 0.0
            if c in ("wasted", "cancelled") and first_diverg_t \
                    and (rec.get("started_t") or 0) >= first_diverg_t:
                wrong_after_diverg += 1
            if rec.get("elapsed_s"):
                spec_read_s += rec["elapsed_s"]

        conservative_steps = sum(e.get("duration_steps", 0)
                                 for e in trace["conservative_events"])

        base = {
            "workload": wl,
            "config": cfg,
            "trial_index": meta.get("trial_index"),
            "run_id": run_id,
            "status": status,
            "mode": summary.get("mode", meta.get("runtime_mode")),
            "predictor": summary.get("predictor", meta.get("predictor")),
            "git_commit": meta.get("git_commit"),
            "node": meta.get("node"),
            "gpu_name": (summary.get("gpu_name")
                         or (meta.get("gpus") or [""])[0].split(",")[0]),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "slurm_job_id": meta.get("slurm_job_id"),
            "first_in_allocation": run_id in first_run_ids,
        }

        q1_rows.append({
            **base,
            "wall_time_s": round(wall_f, 2),
            **gstats,
        })
        q2_rows.append({
            **base,
            "wall_time_s": round(wall_f, 2),
            "agent_reasoning_s": round(reasoning_s, 2),
            "tool_exec_s": round(tool_s, 2),
            "exposed_stall_s": (round(exposed_stall_s, 2)
                                if exposed_stall_s is not None else None),
            "swap_wait_s": (round(swap_wait_s, 2)
                            if swap_wait_s is not None else None),
            "aggregator_wait_s": round(agg_wait_s, 2),
            "aggregator_prefetch_started": any(
                w.get("prefetch_started") for w in trace["agg_waits"]) or None,
            "residual_load_s": round(residual_s, 2),
            "no_prefetch_stall_s": round(no_prefetch_stall_s, 2),
            "overlapped_io_s": round(overlap_s, 2),
            "other_s": round(other_s, 2),
            "mace_load_disk_s": (round(mace_disk_s, 2)
                                 if mace_disk_s is not None else None),
            "mace_load_cache_s": (round(mace_cache_s, 2)
                                  if mace_cache_s is not None else None),
            "overlap_recovered": (round(overlap_recovered, 4)
                                  if overlap_recovered is not None else None),
            "plan_extracted": trace["plan_extracted"],
        })
        q3_rows.append({
            **base,
            "predictions_emitted": n_pred,
            "prefetches_admitted": n_admitted,
            "prefetches_skipped": n_skipped,
            "useful": useful,
            "late": by_class["late"],
            "wasted": by_class["wasted"],
            "cancelled": by_class["cancelled"],
            "pending_at_end": by_class["pending"],
            "duplicate_prefetches": by_class["duplicate"],
            "expired_unvalidated": n_expired,
            "missed_opportunities": n_missed,
            "validated_hits": n_hits,
            "validated_misses": n_miss_validated,
            "divergences": n_diverg,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "on_time_rate": (round(on_time_rate, 4)
                             if on_time_rate is not None else None),
            "mean_lead_time_s": (round(sum(lead_times) / len(lead_times), 2)
                                 if lead_times else None),
        })
        # long format: one row per emitted prediction resource, with outcome
        for p in trace["predictions"]:
            tid = None
            for rec in trace["tasks"].values():
                if rec.get("resource_id") == p["resource_id"]:
                    tid = rec["task_id"]
                    break
            rec = trace["tasks"].get(tid, {}) if tid else {}
            outcome = classify_task(rec, used_rids) if rec else "not_admitted"
            # was this specific prediction validated? join via checkpoint at step
            ckpt = trace["step_checkpoint"].get(p["step"], "")
            val = next((v for v in trace["validations"]
                        if v.get("checkpoint_id") == ckpt), None)
            div = next((d for d in trace["divergences"]
                        if d.get("checkpoint_id") == ckpt), None)
            hit = (True if (val and val.get("hit")) else
                   (False if (div or (val and not val.get("hit"))) else None))
            lead = (rec.get("consumed_t") - rec.get("started_t")
                    if rec.get("consumed_t") and rec.get("started_t") else None)
            q3_long.append({
                **base,
                "step": p["step"],
                "predictor_id": p["predictor_id"],
                "resource_name": p["resource_name"],
                "resource_type": p["resource_type"],
                "consumer_tool": p["consumer_tool"],
                "confidence": p["confidence"],
                "admitted": bool(rec),
                "outcome": outcome,
                "validated_hit": hit,
                "lead_time_s": round(lead, 2) if lead is not None else None,
                "step_lead": ((rec.get("consumed_step") or 0) - p["step"]
                              if rec.get("consumed_step") else None),
            })
        q4_rows.append({
            **base,
            "useful_bytes": int(useful_b),
            "wasted_bytes": int(wasted_b),
            "cancelled_bytes": int(cancelled_b),
            "byte_source": "+".join(sorted(byte_sources - {"unknown"})) or "unknown",
            "n_tasks": n_admitted,
            "n_useful": useful,
            "n_late": by_class["late"],
            "n_wasted": by_class["wasted"],
            "n_cancelled": by_class["cancelled"],
            "n_wasted_noncancellable": n_wasted_noncancellable,
            "n_wrong_after_divergence": wrong_after_diverg,
            "speculative_read_s": round(spec_read_s, 2),
            "divergences": n_diverg,
            "conservative_mode_events": len(trace["conservative_events"]),
            "conservative_mode_steps": conservative_steps,
            "gpu_peak_mem_mb": gstats.get("gpu_peak_mem_mb"),
        })

    # ---- write CSVs ----------------------------------------------------------
    def write_csv(name: str, rows: list[dict]) -> None:
        path = eval_root / name
        if not rows:
            # Remove any stale file from a previous parse so downstream plots
            # can never silently use outdated rows.
            if path.exists():
                path.unlink()
            print(f"  {name}: no rows (stale file removed)" if not rows else "")
            return
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  {path}  ({len(rows)} rows)")

    # Aggregate Q1 table: mean/std/N wall time, normalized to the same
    # workload's baseline mean, and speedup (baseline / config).
    q1_agg_rows = []
    by_cfg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in q1_rows:
        if r.get("wall_time_s"):
            by_cfg[(r["workload"], r["config"])].append(float(r["wall_time_s"]))
    base_means = {wl: (sum(v) / len(v))
                  for (wl, cfg), v in by_cfg.items() if cfg == "baseline"}
    for (wl, cfg), vals in sorted(by_cfg.items()):
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        bm = base_means.get(wl)
        q1_agg_rows.append({
            "workload": wl, "config": cfg, "n_trials": n,
            "wall_time_mean_s": round(mean, 2), "wall_time_std_s": round(std, 2),
            "normalized_wall_time": round(mean / bm, 4) if bm else None,
            "speedup_vs_baseline": round(bm / mean, 4) if bm else None,
            "baseline_available": bm is not None,
        })

    eval_root.mkdir(parents=True, exist_ok=True)
    print("Writing normalized CSVs:")
    write_csv("eval_q1_summary.csv", q1_rows)
    write_csv("eval_q1_agg.csv", q1_agg_rows)
    write_csv("eval_q2_breakdown.csv", q2_rows)
    write_csv("eval_q3_prediction_quality.csv", q3_rows)
    write_csv("eval_q3_predictions_long.csv", q3_long)
    write_csv("eval_q4_speculation_cost.csv", q4_rows)

    # ---- validation report -----------------------------------------------------
    report.append("Q1–Q4 evaluation data validation report")
    report.append(f"eval root : {eval_root}")
    report.append(f"runs found: {len(runs)}, included: {sum(counts.values())}, "
                  f"excluded: {len(excluded)}")
    report.append("")
    report.append("Trials per (workload, config) [successful only]:")
    for (wl, cfg), n in sorted(counts.items()):
        flag = "" if n >= 10 else f"   << fewer than 10 trials (N={n})"
        report.append(f"  {wl:<18} {cfg:<22} N={n}{flag}")
    report.append("")
    # Mixed-hardware guard: aggregate bars must not silently blend GPU types.
    hw_by_key: dict[tuple[str, str], set] = defaultdict(set)
    for row in q1_rows:
        if row.get("gpu_name"):
            hw_by_key[(row["workload"], row["config"])].add(row["gpu_name"])
    mixed = {k: v for k, v in hw_by_key.items() if len(v) > 1}
    if mixed:
        report.append("WARNING — configs mixing GPU types (aggregate means "
                      "blend hardware; facet by gpu_name before publishing):")
        for (wl, cfg), names in sorted(mixed.items()):
            report.append(f"  {wl}/{cfg}: {sorted(names)}")
        report.append("")
    if excluded:
        report.append("Excluded runs (never averaged into CSVs):")
        report.extend(excluded)
    rp = eval_root / "eval_validation_report.txt"
    rp.write_text("\n".join(report) + "\n")
    print(f"  {rp}")
    print("\n".join(report[:40]))


if __name__ == "__main__":
    main()
