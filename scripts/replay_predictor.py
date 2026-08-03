#!/usr/bin/env python3
"""
replay_predictor.py — offline predictor ablation by replaying recorded traces.

WHY THIS EXISTS
---------------
The end-to-end wall-clock arms have n = 5 trials and a variance so large that no
arm is statistically distinguishable from another (Welch's t = 1.32 for the
headline comparison).  An ablation built on those numbers cannot support a claim.

But *prediction quality* does not need a GPU allocation to measure.  The predictor
is a pure function of the event prefix, so it can be re-run over traces already on
disk.  That raises the sample size from 5 trials to hundreds of traces and
thousands of individual prediction decisions, and it isolates the predictor from
every downstream confound (scheduler, capacity, page cache, node noise).

This script replays each recorded trace step by step, calls the predictor with the
same inputs the live adapter would have given it, and scores the predictions
against what the workflow actually went on to do.

WHAT A "REALIZED NEED" IS  -- THE LOAD-BEARING ASSUMPTION, READ THIS
-------------------------------------------------------------------
A realized need is a pair (step k, resource R) where:

    step k    is the k-th `tool_call` event in the trace, in file order, and
    R         is a resource that `ResourceRegistry` declares the tool at step k
              requires (runtime/predictor/data/tool_resources.json, merged with
              MockPredictor's built-in table).

So "the workflow needed qwen_72b" is operationalised as "the workflow invoked a
tool that the registry says consumes qwen_72b".

Why this definition and not something else:

  * It is the *same* map the predictor's output is expressed in.  The predictor
    emits ResourceSpecs keyed by resource name; scoring against anything else
    would require a second, unvalidated translation layer.
  * It is total over the tool sequence: every tool_call either contributes needs
    or contributes none, deterministically, with no thresholds or heuristics.
  * The obvious alternative -- the `resource_consumed` event -- is NOT usable as
    the ground truth here.  Run this script with --audit to reproduce the exact
    figures; as of 2026-08-03 they are: 1092 such events in the corpus versus
    7142 tool_calls, of which (over the 396 replayed traces) 1062 are seen and
    only 255 resolve to a registry resource name -- 807, i.e. 76%, do not
    resolve at all, being cache-layer and per-task ids such as
    `cache_a38637c036d9` and `8b76ae884f7deaa1_t2`.  They are also emitted only
    by *instrumented* arms, so using them would score the predictor only on runs
    where the runtime was already active -- a selection effect that flatters the
    system.  `resource_consumed` is therefore a cross-check here, printed under
    --audit, and never the denominator.

HOW TO CHALLENGE IT
  The definition inherits the registry's coverage.  The merged registry maps only
  9 distinct tools; the corpus contains 32.  Tool_calls whose tool has no registry
  entry generate NO realized needs and are invisible to coverage.  The exact
  covered/uncovered split is printed in the preamble every run, so the reader can
  see how much of the trace the definition actually speaks for.  If the registry
  is wrong about a tool, every number below is wrong in the same direction for
  every variant -- the ablation *ranking* is robust to registry error, the
  absolute levels are not.

WHAT IS MEASURED
----------------
  coverage   fraction of realized needs that some earlier prediction named.
             Denominator: realized need instances (step, resource) in the facet.
  lead       for covered needs, how far ahead the warning came, in steps and in
             seconds (seconds from `epoch_time`, which is present on 100% of
             tool_call events in this corpus -- verified, not assumed).
  precision  of the predictions actually emitted, how many were used.
             Denominator: prediction instances (step, resource) in the facet.
  volume     how many predictions were emitted at all.  A variant can buy
             coverage by predicting everything, and `preds/need` is the column
             that exposes it.  Never read coverage without reading volume.

EVERY PERCENTAGE STATES ITS DENOMINATOR IN THE COLUMN LEGEND.  Coverage and
precision have *different* denominators (needs vs predictions) and must not be
averaged together.

FACETING -- NEVER POOLED
------------------------
Results are reported per (workload, gpu_class).  L40S and Blackwell traces are
never pooled, and workloads are never pooled, because both change the tool mix
and therefore the achievable coverage.  Traces that cannot be attributed to a
workload are reported under the literal facet `UNLABELED/?` and are never folded
into a named workload.

USAGE
    python3 scripts/replay_predictor.py
    python3 scripts/replay_predictor.py --workload chemgraph_swap --audit
    python3 scripts/replay_predictor.py --variants plan_only,transition_only,full
    python3 scripts/replay_predictor.py --limit 40          # quick smoke run

Artifacts are written to results/replay_predictor/ (JSON + CSV) so that any
number quoted in the paper has a file behind it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.predictor.learned_predictor import LearnedPredictor  # noqa: E402
from runtime.predictor.plan_extractor import PlanContext          # noqa: E402
from runtime.predictor.resource_registry import ResourceRegistry  # noqa: E402

TRACE_DIR = REPO / "logs/workflow_traces"
RUNS_DIR = REPO / "results/eval_q1_q4/runs"
OUT_DIR = REPO / "results/replay_predictor"

# Tool events.  VERIFIED against all 531 trace files: the only tool event types
# present are `tool_call` (7142) and `tool_end` (1156), and the tool name is
# under payload["tool"] in 100% of them -- payload["tool_name"] and
# payload["name"] never occur.  `tool_start` and `tool_invocation` do not exist
# in this corpus.  Only tool_call advances a step; tool_end is a completion
# record for the same call and would double-count.
STEP_EVENT = "tool_call"
TOOL_KEYS = ("tool", "tool_name", "name")   # checked in order; only "tool" fires today

# Below this denominator a cell's mean/median is not reported as a bare number.
SMALL_N = 30

# Variant table -- data-driven so a new variant is one line.
# `kwargs` are passed to LearnedPredictor(...).  A variant whose kwargs are not
# accepted by the installed LearnedPredictor is reported as UNSUPPORTED rather
# than silently falling back to a different variant's numbers.
VARIANTS: dict[str, dict[str, Any]] = {
    "plan_only":       {"signals": "plan_only"},
    "transition_only": {"signals": "transition_only"},
    "full":            {"signals": "full"},
    "full_la2":        {"signals": "full", "lookahead": 2},
    "full_la4":        {"signals": "full", "lookahead": 4},
    "full_la6":        {"signals": "full", "lookahead": 6},
}


# ---------------------------------------------------------------------------
# Facet resolution
# ---------------------------------------------------------------------------

def _gpu_class(meta: dict) -> str:
    """Coarse GPU class.  Kept coarse deliberately: the standing rule is only
    that L40S and Blackwell are never pooled, not that every SKU is separate."""
    g = (meta.get("gpus") or [""])
    g = g[0] if g else ""
    if "Blackwell" in g:
        return "blackwell"
    if "L40S" in g:
        return "l40s"
    return "gpu_unknown"


@dataclass(frozen=True)
class Facet:
    workload: str
    gpu: str

    def __str__(self) -> str:
        return f"{self.workload}/{self.gpu}"


UNLABELED = Facet("UNLABELED", "?")


def build_facet_index() -> tuple[dict[str, Facet], dict[str, Facet], int]:
    """
    -> (by_content_hash, by_run_id, n_meta)

    The eval driver writes results/eval_q1_q4/runs/<workload>/<config>/<trial>/
    containing meta.json AND a byte-identical copy of the trace.  That copy is
    the authoritative workload/GPU label, so content hash is the primary join
    (it survives run_id rewrites and trace re-emission); run_id is the fallback.
    """
    by_hash: dict[str, Facet] = {}
    by_run: dict[str, Facet] = {}
    n_meta = 0
    if not RUNS_DIR.exists():
        return by_hash, by_run, n_meta
    for meta_path in RUNS_DIR.glob("*/*/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        n_meta += 1
        facet = Facet(meta.get("workload") or "UNLABELED", _gpu_class(meta))
        trace = meta_path.parent / "trace.jsonl"
        if trace.exists():
            by_hash.setdefault(_md5(trace), facet)
        rid = meta.get("run_id")
        if rid:
            by_run.setdefault(rid, facet)
    return by_hash, by_run, n_meta


def _rel(p: Path) -> str:
    """Repo-relative when possible, absolute otherwise (--out-dir may be elsewhere)."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def table_max_offset(path: Path | None = None) -> int | None:
    """
    Largest step offset the shipped transition table actually contains.

    This is a hard ceiling on what --lookahead can buy from the transition
    signal: asking for lookahead 6 against a table built to offset 3 cannot
    produce anything at offsets 4-6, so those arms saturate.  Surfaced in the
    preamble so a flat lookahead curve is read as a property of the TABLE and
    not mistaken for a finding about the workload.
    """
    p = path or (REPO / "runtime/predictor/data/learned_transitions.json")
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    offs: set[int] = set()
    for key in ("tool_transitions", "model_transitions"):
        for _src, per_off in (d.get(key) or {}).items():
            if isinstance(per_off, dict):
                for k in per_off:
                    try:
                        offs.add(int(k))
                    except (TypeError, ValueError):
                        pass
    return max(offs) if offs else None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

@dataclass
class Step:
    idx: int
    tool: str
    epoch: float | None
    event_pos: int          # index of this event within Trace.events


@dataclass
class Trace:
    path: Path
    facet: Facet
    facet_source: str        # "hash" | "run_id" | "none"
    events: list[dict]
    steps: list[Step]
    plan_points: list[tuple[int, list[str]]] = field(default_factory=list)
    n_bad_lines: int = 0

    def plan_before(self, event_pos: int) -> PlanContext | None:
        """Most recent plan_extracted at or before `event_pos`.  Causal: a plan
        emitted later in the trace is not visible to an earlier prediction."""
        best = None
        for pos, seq in self.plan_points:
            if pos <= event_pos:
                best = seq
            else:
                break
        if not best:
            return None
        return PlanContext(tool_sequence=list(best), source="replay")


def _tool_name(payload: dict) -> str | None:
    for k in TOOL_KEYS:
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def load_trace(path: Path, by_hash: dict[str, Facet],
               by_run: dict[str, Facet]) -> Trace:
    events: list[dict] = []
    steps: list[Step] = []
    plans: list[tuple[int, list[str]]] = []
    run_ids: list[str] = []
    n_bad = 0
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            if not isinstance(obj, dict):
                n_bad += 1
                continue
            pos = len(events)
            events.append(obj)
            rid = obj.get("run_id")
            if isinstance(rid, str) and rid not in run_ids:
                run_ids.append(rid)
            et = obj.get("event_type")
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            if et == STEP_EVENT:
                tool = _tool_name(payload)
                if tool:
                    ep = obj.get("epoch_time")
                    steps.append(Step(len(steps), tool,
                                      float(ep) if isinstance(ep, (int, float)) else None,
                                      pos))
            elif et == "plan_extracted":
                seq = payload.get("tool_sequence")
                if isinstance(seq, list) and seq:
                    plans.append((pos, [s for s in seq if isinstance(s, str)]))

    facet, src = UNLABELED, "none"
    h = _md5(path)
    if h in by_hash:
        facet, src = by_hash[h], "hash"
    else:
        for rid in run_ids:
            if rid in by_run:
                facet, src = by_run[rid], "run_id"
                break
    return Trace(path, facet, src, events, steps, plans, n_bad)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass
class TraceScore:
    """Raw per-trace tallies.  Aggregation is pure summation of these, so a
    facet total is always literally the count of underlying instances."""
    n_needs: int = 0
    n_needs_covered: int = 0
    n_needs_covered_loose: int = 0
    lead_steps: list[int] = field(default_factory=list)
    lead_s: list[float] = field(default_factory=list)
    n_lead_s_unavailable: int = 0

    n_pred_instances: int = 0     # distinct (step, resource)
    n_specs: int = 0              # raw ResourceSpec objects emitted
    n_predict_calls: int = 0
    n_calls_nonempty: int = 0
    n_hit_exact: int = 0
    n_hit_within: int = 0
    n_hit_ever: int = 0

    n_steps: int = 0
    n_steps_registry_covered: int = 0


def replay_trace(trace: Trace, predictor: LearnedPredictor,
                 registry: ResourceRegistry, context_events: int) -> TraceScore:
    """
    Replay one trace through one predictor variant.

    Prediction points are i = -1 (the moment before the first tool_call, which is
    where a plan-driven predictor legitimately gets its first shot) and then
    i = 0 .. n-2, immediately after tool_call i fires.  This mirrors the live
    adapters: chemgraph calls predict() right after a tool completes with
    current_tool_calls=[{"name": tool}] and the last `predictor_context_events`
    raw JSONL events as context.

    Every variant is given byte-identical inputs -- including plan_context.  The
    variants differ ONLY in the predictor's internal signal gating, so any
    difference in the numbers is attributable to the signal, not the input.
    """
    sc = TraceScore()
    steps = trace.steps
    sc.n_steps = len(steps)
    if not steps:
        return sc

    # ---- realized needs -------------------------------------------------
    realized: list[set[str]] = []
    for s in steps:
        specs = registry.get(s.tool)
        names = {sp.name for sp in specs}
        realized.append(names)
        if names:
            sc.n_steps_registry_covered += 1

    occ: dict[str, list[int]] = defaultdict(list)
    for k, names in enumerate(realized):
        for r in names:
            occ[r].append(k)
    sc.n_needs = sum(len(v) for v in occ.values())

    # ---- run the predictor ---------------------------------------------
    # pred_offsets[i][resource] = set of offsets claimed at prediction point i
    pred_offsets: dict[int, dict[str, set[int]]] = {}
    pred_time: dict[int, float | None] = {}

    for i in range(-1, len(steps) - 1):
        if i < 0:
            end = steps[0].event_pos           # strictly before the first tool_call
            ctc: list[dict] = []
            prev = trace.events[end - 1] if end > 0 else None
            t = prev.get("epoch_time") if isinstance(prev, dict) else None
        else:
            end = steps[i].event_pos + 1       # inclusive of tool_call i
            ctc = [{"name": steps[i].tool}]
            t = steps[i].epoch
        recent = trace.events[max(0, end - context_events):end]
        plan = trace.plan_before(end - 1)
        try:
            res = predictor.predict(
                step=max(i, 0),
                recent_events=recent,
                current_tool_calls=ctc,
                task_description="",
                plan_context=plan,
            )
        except Exception:
            # A variant that raises on some prefix is a real result, not a crash
            # to hide: count the call, credit it no predictions.
            sc.n_predict_calls += 1
            pred_offsets[i] = {}
            pred_time[i] = float(t) if isinstance(t, (int, float)) else None
            continue

        sc.n_predict_calls += 1
        specs = list(res.resources or [])
        sc.n_specs += len(specs)
        if specs:
            sc.n_calls_nonempty += 1
        m: dict[str, set[int]] = defaultdict(set)
        for sp in specs:
            off = sp.consumer_step_offset
            m[sp.name].add(int(off) if isinstance(off, int) else 1)
        pred_offsets[i] = dict(m)
        pred_time[i] = float(t) if isinstance(t, (int, float)) else None

    pred_at: dict[str, list[int]] = defaultdict(list)
    for i in sorted(pred_offsets):
        for r in pred_offsets[i]:
            pred_at[r].append(i)

    # ---- coverage + lead -------------------------------------------------
    # A need instance is credited only to a prediction made inside its OWN
    # consumption interval, i.e. after the previous realization of the same
    # resource.  Without that confinement a single prediction at step 2 would
    # be credited for every later re-use of the resource, and a variant that
    # fires once and never again would score 100% coverage.
    for r, ks in occ.items():
        preds_r = pred_at.get(r, [])
        for j, k in enumerate(ks):
            lo = ks[j - 1] if j > 0 else -2     # -2 so i=-1 is live for the first
            live = [i for i in preds_r if lo < i < k]
            if live:
                sc.n_needs_covered += 1
                first = min(live)
                sc.lead_steps.append(k - first)
                t0, t1 = pred_time.get(first), steps[k].epoch
                if t0 is not None and t1 is not None and t1 >= t0:
                    sc.lead_s.append(t1 - t0)
                else:
                    sc.n_lead_s_unavailable += 1
            if any(i < k for i in preds_r):
                sc.n_needs_covered_loose += 1

    # ---- precision -------------------------------------------------------
    # Unit = one (prediction point, resource) pair, which is what the runtime
    # would dedup to before issuing a prefetch.  n_specs is reported separately
    # so raw emission volume is not hidden by the dedup.
    for i, m in pred_offsets.items():
        for r, offsets in m.items():
            sc.n_pred_instances += 1
            future = occ.get(r, [])
            if any(k > i for k in future):
                sc.n_hit_ever += 1
            if any(0 <= i + o < len(realized) and r in realized[i + o] for o in offsets):
                sc.n_hit_exact += 1
            horizon = i + max(offsets)
            if any(i < k <= horizon for k in future):
                sc.n_hit_within += 1

    return sc


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def _fmt_pct(num: int, den: int) -> str:
    if not den:
        return "   n/a"
    v = 100.0 * num / den
    mark = "†" if den < SMALL_N else " "
    return f"{v:5.1f}{mark}"


def _fmt_stat(vals: list[float], fmt: str = "{:5.1f}") -> str:
    if not vals:
        return "   n/a"
    mark = "†" if len(vals) < SMALL_N else " "
    return fmt.format(st.median(vals)) + mark


def aggregate(scores: dict[tuple[str, Facet], list[TraceScore]]) -> list[dict]:
    rows = []
    for (variant, facet), lst in scores.items():
        agg = TraceScore()
        for s in lst:
            agg.n_needs += s.n_needs
            agg.n_needs_covered += s.n_needs_covered
            agg.n_needs_covered_loose += s.n_needs_covered_loose
            agg.lead_steps += s.lead_steps
            agg.lead_s += s.lead_s
            agg.n_lead_s_unavailable += s.n_lead_s_unavailable
            agg.n_pred_instances += s.n_pred_instances
            agg.n_specs += s.n_specs
            agg.n_predict_calls += s.n_predict_calls
            agg.n_calls_nonempty += s.n_calls_nonempty
            agg.n_hit_exact += s.n_hit_exact
            agg.n_hit_within += s.n_hit_within
            agg.n_hit_ever += s.n_hit_ever
            agg.n_steps += s.n_steps
            agg.n_steps_registry_covered += s.n_steps_registry_covered
        rows.append({
            "variant": variant,
            "workload": facet.workload,
            "gpu": facet.gpu,
            "n_traces": len(lst),
            "n_steps": agg.n_steps,
            "n_steps_with_registry_needs": agg.n_steps_registry_covered,
            # --- coverage: denominator = realized need instances ---
            "n_realized_needs": agg.n_needs,
            "n_needs_covered": agg.n_needs_covered,
            "coverage_pct_of_realized_needs": _pct(agg.n_needs_covered, agg.n_needs),
            "n_needs_covered_loose": agg.n_needs_covered_loose,
            "coverage_loose_pct_of_realized_needs":
                _pct(agg.n_needs_covered_loose, agg.n_needs),
            # --- lead: denominator = covered needs ---
            "n_lead_samples": len(agg.lead_steps),
            "lead_steps_median": st.median(agg.lead_steps) if agg.lead_steps else None,
            "lead_steps_mean": st.mean(agg.lead_steps) if agg.lead_steps else None,
            "n_lead_s_samples": len(agg.lead_s),
            "lead_s_median": st.median(agg.lead_s) if agg.lead_s else None,
            "lead_s_mean": st.mean(agg.lead_s) if agg.lead_s else None,
            "n_lead_s_unavailable": agg.n_lead_s_unavailable,
            # --- precision: denominator = prediction instances ---
            "n_prediction_instances": agg.n_pred_instances,
            "n_resource_specs_emitted": agg.n_specs,
            "n_predict_calls": agg.n_predict_calls,
            "n_predict_calls_nonempty": agg.n_calls_nonempty,
            "precision_exact_pct_of_predictions":
                _pct(agg.n_hit_exact, agg.n_pred_instances),
            "precision_within_horizon_pct_of_predictions":
                _pct(agg.n_hit_within, agg.n_pred_instances),
            "used_ever_pct_of_predictions": _pct(agg.n_hit_ever, agg.n_pred_instances),
            "n_never_used": agg.n_pred_instances - agg.n_hit_ever,
            "wasted_pct_of_predictions":
                _pct(agg.n_pred_instances - agg.n_hit_ever, agg.n_pred_instances),
            # --- volume ---
            "predictions_per_realized_need":
                (agg.n_pred_instances / agg.n_needs) if agg.n_needs else None,
            "predictions_per_step": (agg.n_pred_instances / agg.n_steps)
                if agg.n_steps else None,
            "small_n_coverage": agg.n_needs < SMALL_N,
            "small_n_precision": agg.n_pred_instances < SMALL_N,
            "_raw": agg,
        })
    return rows


def print_tables(rows: list[dict], variant_order: list[str],
                 unsupported: dict[str, str]) -> None:
    by_facet: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_facet[(r["workload"], r["gpu"])].append(r)

    hdr = (f"{'variant':<17}{'trc':>4}{'needs':>7}{'cov%':>7}{'covL%':>7}"
           f"{'leadΔk':>8}{'leadΔs':>9}{'preds':>7}{'specs':>7}"
           f"{'exact%':>8}{'horiz%':>8}{'wasted%':>9}{'p/need':>8}")

    def sort_key(fk):
        wl, gpu = fk
        return (wl == "UNLABELED", wl, gpu)

    for fk in sorted(by_facet, key=sort_key):
        wl, gpu = fk
        lst = {r["variant"]: r for r in by_facet[fk]}
        any_row = next(iter(by_facet[fk]))
        print()
        print(f"=== workload={wl}   gpu={gpu} "
              f"  traces={any_row['n_traces']}"
              f"  tool_calls={any_row['n_steps']}"
              f"  of which registry-mapped={any_row['n_steps_with_registry_needs']}"
              f" ===")
        print(hdr)
        print("-" * len(hdr))
        for v in variant_order:
            if v in unsupported:
                print(f"{v:<17}{'UNSUPPORTED — ' + unsupported[v]}")
                continue
            r = lst.get(v)
            if r is None:
                continue
            raw: TraceScore = r["_raw"]
            print(
                f"{v:<17}"
                f"{r['n_traces']:>4}"
                f"{r['n_realized_needs']:>7}"
                f"{_fmt_pct(raw.n_needs_covered, raw.n_needs):>7}"
                f"{_fmt_pct(raw.n_needs_covered_loose, raw.n_needs):>7}"
                f"{_fmt_stat([float(x) for x in raw.lead_steps], '{:6.1f}'):>8}"
                f"{_fmt_stat(raw.lead_s, '{:7.1f}'):>9}"
                f"{r['n_prediction_instances']:>7}"
                f"{r['n_resource_specs_emitted']:>7}"
                f"{_fmt_pct(raw.n_hit_exact, raw.n_pred_instances):>8}"
                f"{_fmt_pct(raw.n_hit_within, raw.n_pred_instances):>8}"
                f"{_fmt_pct(raw.n_pred_instances - raw.n_hit_ever, raw.n_pred_instances):>9}"
                + (f"{r['predictions_per_realized_need']:>8.2f}"
                   if r["predictions_per_realized_need"] is not None else f"{'n/a':>8}")
            )


LEGEND = """
COLUMN LEGEND — note that coverage and precision have DIFFERENT denominators
  trc      traces in this facet (same for every variant: identical input set)
  needs    realized need instances = sum over tool_calls of the resources the
           ResourceRegistry says that tool consumes.  THIS IS THE COVERAGE
           DENOMINATOR.
  cov%     % OF NEEDS whose resource was named by a prediction made after the
           previous realization of that same resource (per-instance credit)
  covL%    % OF NEEDS whose resource was named by ANY earlier prediction (loose;
           one early prediction gets credit for every later re-use).  The gap
           cov% -> covL% is how much of the coverage is stale credit.
  leadΔk   median steps of warning, over COVERED needs only
  leadΔs   median seconds of warning, over covered needs with usable timestamps.
           Recoverable here: epoch_time is present on 100% of tool_call events.
           CAVEAT: the gap is measured in the RECORDED timeline, which already
           contains whatever stalls that run suffered.  It is the warning the
           predictor would have had in that run, not in an idealised one.
  preds    prediction instances = distinct (prediction point, resource).
           THIS IS THE PRECISION DENOMINATOR.
  specs    raw ResourceSpec objects emitted before dedup — the true volume
  exact%   % OF PREDICTIONS where the resource was realized at exactly the step
           the predictor claimed (step + consumer_step_offset)
  horiz%   % OF PREDICTIONS where the resource was realized anywhere inside the
           claimed horizon (step, step+offset]
  wasted%  % OF PREDICTIONS whose resource was never realized again in the
           remainder of the trace — unambiguously wasted speculation
  p/need   preds / needs.  A variant that raises cov% while raising p/need is
           buying coverage with speculation, not with skill.  Read them together.
  †        this cell's denominator is < {small} — treat as indicative, not a result
""".format(small=SMALL_N)


# ---------------------------------------------------------------------------
# Audit: cross-check the realized-need definition against resource_consumed
# ---------------------------------------------------------------------------

def audit_resource_consumed(traces: list[Trace], registry: ResourceRegistry) -> None:
    import hashlib as _h
    names = set()
    for t in registry.all_tools():
        for s in registry.get(t):
            names.add(s.name)
    byid = {_h.md5(n.encode()).hexdigest()[:12]: n for n in names}
    resolved = defaultdict(int)
    unresolved = defaultdict(int)
    n_traces_with = 0
    for tr in traces:
        seen = False
        for ev in tr.events:
            if ev.get("event_type") != "resource_consumed":
                continue
            seen = True
            rid = (ev.get("payload") or {}).get("resource_id")
            if rid in byid:
                resolved[byid[rid]] += 1
            else:
                unresolved[rid] += 1
        if seen:
            n_traces_with += 1
    tot = sum(resolved.values()) + sum(unresolved.values())
    print("\n--- AUDIT: resource_consumed as an alternative ground truth ---")
    print(f"resource_consumed events: {tot} in {n_traces_with}/{len(traces)} traces")
    print(f"  resolve to a registry resource name: {sum(resolved.values())}")
    print(f"  do NOT resolve:                      {sum(unresolved.values())}")
    for k, v in sorted(resolved.items(), key=lambda x: -x[1]):
        print(f"     resolved   {v:6d}  {k}")
    for k, v in sorted(unresolved.items(), key=lambda x: -x[1])[:10]:
        print(f"     unresolved {v:6d}  {k}")
    print("This is why resource_consumed is NOT the denominator: it is sparse,")
    print("only emitted by instrumented arms, and mostly unresolvable.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_predictor(name: str, spec: dict, registry: ResourceRegistry
                    ) -> tuple[LearnedPredictor | None, str]:
    """-> (predictor, unsupported_reason).  Never silently drops a kwarg: a
    variant whose knob does not exist yet must not be reported under a label
    that implies it was applied."""
    accepted = set(inspect.signature(LearnedPredictor.__init__).parameters)
    missing = [k for k in spec if k not in accepted]
    if missing:
        return None, (f"LearnedPredictor.__init__ has no {missing!r} parameter "
                      f"in this checkout; not run (would be identical to "
                      f"'{spec.get('signals', 'full')}' and mislabelled)")
    return LearnedPredictor(registry=registry, **spec), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--traces-dir", default=str(TRACE_DIR))
    ap.add_argument("--workload", default=None, help="restrict to one workload")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--context-events", type=int, default=10,
                    help="events fed to the predictor; runtime default is 10 "
                         "(runtime/config.py predictor_context_events)")
    ap.add_argument("--limit", type=int, default=None, help="first N traces only")
    ap.add_argument("--min-steps", type=int, default=1,
                    help="skip traces with fewer than this many tool_calls")
    ap.add_argument("--audit", action="store_true",
                    help="also cross-check against resource_consumed events")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    variant_order = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variant_order if v not in VARIANTS]
    if bad:
        print(f"unknown variant(s): {bad}; known: {list(VARIANTS)}")
        return 2

    tdir = Path(args.traces_dir)
    if not tdir.exists():
        print(f"MISSING trace dir: {tdir}")
        return 2

    registry = ResourceRegistry.merged(
        ResourceRegistry.from_json(),
        ResourceRegistry.from_mock_predictor(),
    )

    by_hash, by_run, n_meta = build_facet_index()

    files = sorted(tdir.glob("*.jsonl"))
    n_files = len(files)
    traces: list[Trace] = []
    skip_no_steps = 0
    skip_few_steps = 0
    skip_workload = 0
    n_bad_lines = 0
    for p in files:
        tr = load_trace(p, by_hash, by_run)
        n_bad_lines += tr.n_bad_lines
        if not tr.steps:
            skip_no_steps += 1
            continue
        if len(tr.steps) < args.min_steps:
            skip_few_steps += 1
            continue
        if args.workload and tr.facet.workload != args.workload:
            skip_workload += 1
            continue
        traces.append(tr)
        if args.limit and len(traces) >= args.limit:
            break

    src_counts = defaultdict(int)
    for tr in traces:
        src_counts[tr.facet_source] += 1

    print("=" * 100)
    print("replay_predictor.py — offline predictor ablation over recorded traces")
    print("=" * 100)
    print(f"trace dir                     {tdir}")
    print(f"trace files found             {n_files}")
    print(f"  skipped: no tool_call event {skip_no_steps}")
    if args.min_steps > 1:
        print(f"  skipped: < {args.min_steps} tool_calls        {skip_few_steps}")
    if args.workload:
        print(f"  skipped: other workload     {skip_workload}")
    print(f"  malformed JSON lines        {n_bad_lines}")
    print(f"REPLAYED                      {len(traces)}")
    print(f"facet labels from {n_meta} meta.json under {RUNS_DIR.relative_to(REPO)}")
    print(f"  labelled by trace content hash  {src_counts['hash']}")
    print(f"  labelled by run_id              {src_counts['run_id']}")
    print(f"  UNLABELED (own facet, never pooled into a named workload)  "
          f"{src_counts['none']}")

    n_steps_all = sum(len(t.steps) for t in traces)
    n_mapped = sum(1 for t in traces for s in t.steps if registry.get(s.tool))
    tools_seen = {s.tool for t in traces for s in t.steps}
    print(f"tool_call events              {n_steps_all}")
    print(f"  registry-mapped (contribute needs)   {n_mapped} "
          f"({100.0 * n_mapped / n_steps_all:.1f}% of tool_calls)")
    print(f"  distinct tools in corpus {len(tools_seen)}; "
          f"distinct tools in registry {len(registry.all_tools())}")
    print(f"  tools with NO registry entry (invisible to coverage): "
          f"{sorted(tools_seen - set(registry.all_tools()))}")
    n_plan = sum(1 for t in traces if t.plan_points)
    print(f"traces carrying a plan_extracted event  {n_plan}/{len(traces)} "
          f"— plan_only can only ever score on these")
    print(f"predictor context window      {args.context_events} events "
          f"(runtime default)")
    max_off = table_max_offset()
    if max_off is not None:
        print(f"learned transition table max offset  {max_off}  — the transition "
              f"signal CANNOT")
        print(f"  produce candidates beyond offset {max_off}, so --lookahead arms above "
              f"{max_off + 1} can only")
        print(f"  differ via the plan signal.  A flat lookahead curve above there is a "
              f"property")
        print(f"  of the TABLE, not a finding about the workload.")

    if not traces:
        print("\nno traces to replay")
        return 1

    # ---- run ------------------------------------------------------------
    unsupported: dict[str, str] = {}
    scores: dict[tuple[str, Facet], list[TraceScore]] = defaultdict(list)
    for v in variant_order:
        pred, reason = build_predictor(v, VARIANTS[v], registry)
        if pred is None:
            unsupported[v] = reason
            print(f"\n!! variant {v!r} NOT RUN: {reason}")
            continue
        for tr in traces:
            scores[(v, tr.facet)].append(
                replay_trace(tr, pred, registry, args.context_events))

    rows = aggregate(scores)
    print_tables(rows, variant_order, unsupported)
    print(LEGEND)

    if unsupported:
        print("VARIANTS NOT RUN")
        for v, why in unsupported.items():
            print(f"  {v}: {why}")
        print()

    print("SCOPE AND LIMITS — state these wherever the numbers are used")
    print("  * This measures PREDICTION QUALITY ONLY.  It says nothing about")
    print("    whether a correct prediction was acted on in time, whether a slot")
    print("    was free, or what wall-clock time it saved.  Use replay_capacity.py")
    print("    for the capacity-honest wall-clock bound.")
    print("  * Ground truth is the ResourceRegistry map (see module docstring).")
    print("    A registry error moves every variant in the same direction, so the")
    print("    RANKING is robust and the ABSOLUTE LEVELS are not.")
    print("  * Facets are never pooled: no L40S+Blackwell row, no cross-workload")
    print("    row, and UNLABELED traces stay in their own facet.")
    print("  * Traces were recorded under many different runtime configurations")
    print("    (baseline, full_system, ablations).  The predictor is replayed")
    print("    identically over all of them, but the TOOL SEQUENCE in a trace was")
    print("    produced by whatever arm recorded it.  This is a real confound for")
    print("    absolute coverage and is NOT corrected for here.")

    if args.audit:
        audit_resource_consumed(traces, registry)

    # ---- artifacts ------------------------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "_raw"}
        clean.append(d)
    clean.sort(key=lambda d: (d["workload"] == "UNLABELED", d["workload"],
                              d["gpu"], d["variant"]))

    meta_out = {
        "generated_by": "scripts/replay_predictor.py",
        "trace_dir": str(tdir),
        "n_trace_files": n_files,
        "n_replayed": len(traces),
        "n_skipped_no_tool_call": skip_no_steps,
        "n_skipped_few_steps": skip_few_steps,
        "n_skipped_other_workload": skip_workload,
        "n_malformed_json_lines": n_bad_lines,
        "n_tool_call_events": n_steps_all,
        "n_tool_call_registry_mapped": n_mapped,
        "tools_without_registry_entry": sorted(tools_seen - set(registry.all_tools())),
        "n_traces_with_plan_extracted": n_plan,
        "context_events": args.context_events,
        "learned_table_max_offset": max_off,
        "small_n_threshold": SMALL_N,
        "variants_requested": variant_order,
        "variants_not_run": unsupported,
        "facet_label_sources": dict(src_counts),
        "realized_need_definition":
            "(step k, resource R) where step k is the k-th tool_call event and R "
            "is a resource ResourceRegistry maps the tool at step k to.",
        "denominators": {
            "coverage": "realized need instances (step, resource)",
            "precision": "prediction instances (prediction point, resource)",
            "lead": "covered need instances",
        },
    }
    jpath = out / "replay_predictor.json"
    jpath.write_text(json.dumps({"meta": meta_out, "rows": clean}, indent=2))
    cpath = out / "replay_predictor.csv"
    if clean:
        with cpath.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(clean[0]))
            w.writeheader()
            w.writerows(clean)
    print(f"\nwrote {_rel(jpath)}")
    print(f"wrote {_rel(cpath)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
