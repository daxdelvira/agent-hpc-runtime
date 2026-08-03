#!/usr/bin/env python3
"""
replay_divergence.py — recover real hit/miss/divergence counts by driving the
FIXED DivergenceDetector over recorded trace event streams.

WHY THIS EXISTS
---------------
`runtime/guard/detector.py` carried inverted hit/miss semantics from e68d52b
(2026-06-02) until 96f5f28 (2026-08-03).  In that window:

  * a tool call that NO pending checkpoint predicted — the definition of a
    divergence — fell through to `return True, CONTINUE, None`.  The adapter
    branches on `ckpt is None` and emitted nothing, so the miss never reached
    AccuracyTracker and never reached the trace.
  * a legitimate prefix match ("computation_task" predicted, the concrete
    "computation_task_screw_dislocation" observed) passed the prefilter and then
    failed the `==` test, so a MATCH was logged as a divergence.

Every AtomAgents eval trial ran with the broken detector.  The recorded
aggregate over all AtomAgents campaign summaries is

    prediction_count=219  hit_count=91  miss_count=0  unvalidated_count=128
    divergence_count=0    prefetch_cancelled=0        wasted_prefetch=0

and `accuracy` is 1.0000 in all 34 summaries that record it.  Those numbers are
not accuracy measurements; they are the count of predictions that happened to
match, over a denominator from which every mismatch was silently removed.  From
the summaries alone the true accuracy is bracketed by [91/219, 91/91] =
[0.4155, 1.0000].

CORRECTION TO THAT BRACKET.  Its lower bound holds the numerator fixed at the 91
recorded hits and treats only the 128 unvalidated predictions as unknown.  The
fixed detector does not preserve the numerator: it CONSUMES a checkpoint on a
miss, so a checkpoint the broken detector kept alive until a later matching tool
can instead be retired as a miss first, and the later tool then finds nothing
pending.  In this corpus 26 of the 91 recorded hits are scored as misses by the
fixed detector while 24 never-scored predictions become hits, so the recovered
numerator is 89, not 91, and 91/219 is NOT a guaranteed floor.  The reconciliation
block below prints this breakdown.

The suppressed misses were never emitted, so `divergence_detected` events cannot
be recounted.  But the INPUTS survive: the traces contain the complete
`prediction_result` and `tool_call` streams.  This script replays those streams
through the fixed detector and recovers the outcome the fixed system would have
scored for each recorded prediction.

WHAT IS REPLAYED, AND WHAT IS NOT
---------------------------------
Replayed: the SCORING of the prediction stream that was actually recorded.

NOT replayed: the prediction stream the fixed system would have produced.  A
miss puts the live adapter into conservative mode for `conservative_mode_steps`,
which suppresses subsequent predictions.  Under the broken detector no AtomAgents
run ever entered conservative mode (0 `conservative_mode` events in 59 AtomAgents
traces).  So the fixed system, run live, would have made FEWER predictions than
the 219 scored here.  This is a counterfactual limit, not a bug: the question
answered is "of the predictions that were made, how many were right?", which is
the accuracy/precision number the paper reports.  It is NOT "what would the
fixed runtime have done end to end".  Do not use these counts for wall-clock or
prefetch-waste claims.

ASSUMPTIONS  (every one of these is a place the harness could be wrong)
----------------------------------------------------------------------
A1. FILE ORDER IS CALL ORDER.  Both adapters emit to the same JSONL bus from the
    callback thread, so the order of `prediction_result` / `tool_call` lines in a
    trace is the order of the corresponding `on_prediction` /
    `on_tool_about_to_execute` calls.  Verified structurally:
      - atomagents.py:277-285 emits prediction_result then checkpoint_created,
        then :306 emits tool_call, then :308 calls the detector.
      - chemgraph.py:877 (parent) emits tool_call, :957 runs the phase-1 check,
        :1035-1051 emits prediction_result + checkpoint_created.
    So replaying actions in file order reproduces "check pending, then create
    new checkpoint" for chemgraph and "create checkpoint, then check tool" for
    atomagents, which is what each adapter does.

A2. A PREDICTION ENTERS THE ACCOUNTING IFF IT IS IMMEDIATELY FOLLOWED BY
    `checkpoint_created`.  chemgraph emits `prediction_result` from four sites;
    only chemgraph.py:1035 (followed by :1049 checkpoint_created) sets
    `_pending_checkpoint`.  The other three (pool preboot :298, swap cache stage
    :760, plan extraction :836) create a CheckpointRecord for the scheduler but
    never enter hit/miss scoring.  Empirically over the corpus, a
    `prediction_result` is followed by exactly one of {checkpoint_created (456),
    prefetch_decision (326), chain_start (12)} for chemgraph, and by
    checkpoint_created 219/219 times for atomagents.  So "next line is
    checkpoint_created" is an exact discriminator, not a heuristic.

A3. STEP NUMBERS.  AtomAgents stamps a top-level `step` on every event including
    `tool_call` (adapter emits with step=step), so its steps are read directly.
    ChemGraph's `tool_call` is emitted by the ChemGraph WorkflowTracker
    (chemgraph.py:877 `super().on_tool_start`) BEFORE `self._step += 1` at :884,
    so it carries no step.  The step of a chemgraph tool call is therefore
    reconstructed as the first non-null `step` on a following event before the
    next `tool_call` (the adapter calls `bus.set_step(self._step)` at :886, so
    everything emitted in that tool block carries the new step).  When no such
    event exists — a tool call that produced no runtime events at all — the step
    falls back to previous+1 and the trace is counted in `steps_inferred`.
    This reconstruction is validated two ways, both printed: (i) whenever the
    tool block contains a `checkpoint_created`, its `step` must equal the
    reconstructed step; (ii) the chemgraph_inline control below.

A4. max_horizon = 2 for every trial, so the detector's checkpoint window is
    max(2*8, 60) = 60 steps.  Checked: no meta.json command in the corpus
    (0 of 358) passes `--horizon`, and the argparse default is 2 in all three
    experiment drivers.  Override with --max-horizon.

A5. UNRESOLVED OUTCOMES ARE NOT MISSES.  A checkpoint that the fixed detector
    never scored — because the workflow ended, or because it aged out of the
    60-step window without ever being due at a tool call — has NO determinable
    outcome from the trace.  It is reported in its own column and never folded
    into hits or misses.  Any accuracy computed here is therefore over RESOLVED
    predictions, and the honest whole-corpus statement stays a range.

WHAT IS REUSED VS REIMPLEMENTED
-------------------------------
The `detector` scorer IMPORTS and DRIVES the shipped
`runtime.guard.DivergenceDetector` — no scoring logic is reimplemented.  Two
small things are derived rather than imported:

  * the split of unresolved checkpoints into aged-out vs pending-at-end uses the
    age formula from detector.py:140 (`max(max_horizon*8, 60)`), because the
    detector exposes no API for "why was this never scored".  It does not affect
    hit/miss counts, only how the leftovers are labelled.
  * the `chemgraph_inline` scorer DELIBERATELY reimplements chemgraph.py:957-1003
    (single pending slot, exact `==` match, `expected_at_step or 0` gate).  It
    exists ONLY as the harness control — see below — and its numbers are never
    presented as detector numbers.

THE CONTROL, AND A CORRECTION TO THE BRIEF
------------------------------------------
The task brief states that ChemGraph "is unaffected by the bug, so your replay
should approximately reproduce its recorded hit_count=166, miss_count=104,
divergence_count=104; if it does not, your replay is wrong."

The first half is right and the inference is not.  ChemGraph is unaffected
because its adapter never calls DivergenceDetector at all: chemgraph.py:957-1003
is a separate, inlined check with materially different semantics from the fixed
detector —

    | question              | chemgraph inline      | fixed DivergenceDetector  |
    |-----------------------|-----------------------|---------------------------|
    | pending checkpoints   | ONE slot, overwritten | queue, all kept           |
    | tool-name match       | exact `==` only       | exact or `predicted + "_"` |
    | unset expected_at_step| `or 0` -> always due  | falls back to step+offset |
    | ageing                | none (slot overwrite) | 60-step window            |

So driving the DivergenceDetector over ChemGraph traces is NOT expected to
reproduce ChemGraph's recorded counts, and a mismatch there would be evidence
about the two implementations, not about this harness.  The harness control is
therefore run with the `chemgraph_inline` scorer, which must reproduce 166/104
EXACTLY.  That check validates the part shared with the AtomAgents replay — trace
parsing, action extraction, A2's checkpoint discriminator, A3's step
reconstruction, ordering — while holding scoring semantics fixed.  Both scorers
are then reported over ChemGraph so the semantic delta is visible rather than
hidden.  `--control` runs exactly this check and exits non-zero on failure.

FACETING
--------
Rows are (workload, config, gpu, trial_status).  L40S and Blackwell are never
pooled — note that `atomagents_exp3_aligned` alone spans both, so it MUST be
split.  GPU comes from meta.json `gpus` (AtomAgents summaries carry no
`gpu_name` field at all), falling back to summary.json `gpu_name`.

`trial_status` is meta.json `status` and is part of the key because 17 of the 59
AtomAgents traces and 57 of the 233 ChemGraph traces come from trials that did
not complete.  Several `failed` AtomAgents trials aborted after ~2 s of wall
clock having already emitted predictions and tool calls, so their predictions
are real trace events but are not eval data.  Pooling them with completed trials
would let launch failures move the accuracy number; they get their own rows.

`n` is reported as both trials and predictions; the prediction count is the unit
of the accuracy estimate and rows with n_pred < 30 are marked INDICATIVE.

USAGE
    python scripts/replay_divergence.py                    # full report
    python scripts/replay_divergence.py --control          # control check only
    python scripts/replay_divergence.py --workload atomagents --per-trial
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from runtime.config import RuntimeConfig          # noqa: E402
from runtime.events import PredictionResult       # noqa: E402
from runtime.guard.detector import DivergenceDetector  # noqa: E402

RUNS_DIR = REPO / "results" / "eval_q1_q4" / "runs"
DEFAULT_OUT = REPO / "results" / "replay_divergence"

# Recorded ChemGraph aggregate the control must reproduce.  Source:
#   python -c "...sum over results/eval_q1_q4/runs/chemgraph*/*/*/summary.json..."
# (reproduce with --control, which recomputes it from the summaries on disk).
CONTROL_EXPECT = {"hits": 166, "misses": 104}


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """One replayable call, in trace file order."""
    kind: str                 # "predict" | "tool"
    step: int
    tool: str = ""
    prediction: PredictionResult | None = None
    step_inferred: bool = False
    line_no: int = 0
    checkpoint_id: str = ""   # the REAL id from the following checkpoint_created


@dataclass
class ParsedTrace:
    path: Path
    actions: list[Action] = field(default_factory=list)
    n_prediction_events: int = 0     # all prediction_result lines
    n_scored_predictions: int = 0    # those followed by checkpoint_created (A2)
    n_tool_calls: int = 0
    n_steps_inferred: int = 0        # chemgraph tool calls with no stepped event
    n_step_checks: int = 0           # tool blocks carrying a checkpoint_created
    n_step_check_fail: int = 0       # ... where reconstruction disagreed
    n_bad_lines: int = 0
    # recorded outcomes, for the control comparison
    rec_validated: int = 0
    rec_diverged: int = 0
    rec_validated_ids: set = field(default_factory=set)
    rec_diverged_ids: set = field(default_factory=set)


def parse_trace(path: Path) -> ParsedTrace:
    """Extract the ordered (predict, tool) action stream from one trace."""
    pt = ParsedTrace(path=path)
    events: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pt.n_bad_lines += 1

    cur_step = 0
    for i, ev in enumerate(events):
        et = ev.get("event_type")

        if et == "prediction_validated":
            pt.rec_validated += 1
            cid = (ev.get("payload") or {}).get("checkpoint_id")
            if cid:
                pt.rec_validated_ids.add(cid)
            continue
        if et == "divergence_detected":
            pt.rec_diverged += 1
            cid = (ev.get("payload") or {}).get("checkpoint_id")
            if cid:
                pt.rec_diverged_ids.add(cid)
            continue

        if et == "prediction_result":
            pt.n_prediction_events += 1
            # A2: only a prediction immediately followed by checkpoint_created
            # became a pending checkpoint and can be scored.
            nxt = events[i + 1] if i + 1 < len(events) else None
            if not nxt or nxt.get("event_type") != "checkpoint_created":
                continue
            payload = ev.get("payload") or {}
            try:
                pred = PredictionResult.from_dict(payload)
            except Exception:
                pt.n_bad_lines += 1
                continue
            if not pred.resources:
                continue
            step = ev.get("step")
            if step is None:
                step = payload.get("step", cur_step)
            cur_step = max(cur_step, int(step))
            pt.n_scored_predictions += 1
            pt.actions.append(Action(
                "predict", int(step), prediction=pred, line_no=i,
                checkpoint_id=(nxt.get("payload") or {}).get("checkpoint_id", ""),
            ))
            continue

        if et == "tool_call":
            pt.n_tool_calls += 1
            tool = (ev.get("payload") or {}).get("tool") or ""
            step = ev.get("step")
            inferred = False
            if step is None:
                # A3: reconstruct the chemgraph step from the tool block.
                step = None
                for j in range(i + 1, len(events)):
                    if events[j].get("event_type") == "tool_call":
                        break
                    s = events[j].get("step")
                    if s is not None:
                        step = int(s)
                        break
                if step is None:
                    step = cur_step + 1
                    inferred = True
                    pt.n_steps_inferred += 1
                else:
                    # Validation (i): if this tool block created a checkpoint,
                    # its recorded step must equal the reconstructed step.
                    for j in range(i + 1, len(events)):
                        if events[j].get("event_type") == "tool_call":
                            break
                        if events[j].get("event_type") == "checkpoint_created":
                            pt.n_step_checks += 1
                            rec = (events[j].get("payload") or {}).get("step")
                            if rec is not None and int(rec) != step:
                                pt.n_step_check_fail += 1
                            break
            cur_step = max(cur_step, int(step))
            pt.actions.append(
                Action("tool", int(step), tool=tool,
                       step_inferred=inferred, line_no=i)
            )
            continue

        s = ev.get("step")
        if s is not None:
            cur_step = max(cur_step, int(s))

    return pt


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    predictions: int = 0
    hits: int = 0
    misses: int = 0
    unresolved_aged_out: int = 0
    unresolved_pending_at_end: int = 0
    unresolved_superseded: int = 0
    tool_calls: int = 0
    no_opinion_tool_calls: int = 0   # tool calls that consumed no checkpoint
    # Sub-split of the unresolved population by whether the trace even contains
    # the evidence that would decide it.  See _classify_unresolved.
    unresolved_target_never_reached: int = 0
    unresolved_target_passed: int = 0

    _FIELDS = (
        "predictions", "hits", "misses", "unresolved_aged_out",
        "unresolved_pending_at_end", "unresolved_superseded",
        "tool_calls", "no_opinion_tool_calls",
        "unresolved_target_never_reached", "unresolved_target_passed",
    )

    @property
    def unresolved(self) -> int:
        return (self.unresolved_aged_out
                + self.unresolved_pending_at_end
                + self.unresolved_superseded)

    def add(self, other: "ScoreResult") -> None:
        for f in self._FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))


def score_detector(actions: list[Action], config: RuntimeConfig) -> ScoreResult:
    """Drive the SHIPPED DivergenceDetector over the recorded action stream.

    No scoring logic lives here: hit/miss is entirely
    DivergenceDetector.on_tool_about_to_execute's verdict.
    """
    det = DivergenceDetector(scheduler=None, config=config, bus=None)
    res = ScoreResult()
    created: dict[str, int] = {}      # checkpoint_id -> creating step
    resolved: set[str] = set()
    last_step = 0

    for act in actions:
        last_step = max(last_step, act.step)
        if act.kind == "predict":
            ckpt = det.on_prediction(act.prediction, step=act.step)
            created[ckpt.checkpoint_id] = act.step
            res.predictions += 1
        else:
            res.tool_calls += 1
            hit, _action, ckpt = det.on_tool_about_to_execute(act.tool, step=act.step)
            if ckpt is None:
                res.no_opinion_tool_calls += 1
                continue
            resolved.add(ckpt.checkpoint_id)
            if hit:
                res.hits += 1
            else:
                res.misses += 1

    # A5: leftovers have no determinable outcome.  Label them using the same
    # age formula the detector applies at detector.py:140.
    max_age = max(config.max_horizon * 8, 60)
    for ckpt_id, step in created.items():
        if ckpt_id in resolved:
            continue
        if (last_step - step) > max_age:
            res.unresolved_aged_out += 1
        else:
            res.unresolved_pending_at_end += 1
    _classify_unresolved(actions, created, resolved, res)
    return res


def _classify_unresolved(actions: list[Action], created: dict[str, int],
                         resolved: set[str], res: ScoreResult) -> None:
    """Split the unresolved population by whether the trace could decide it.

    A prediction declares a target step (`expected_at_step`, falling back to
    `step + max(consumer_step_offset, 1)` — the same formula as detector.py:193-197).

      * target_never_reached — no tool call ever fired at or after that step.
        The workflow ended, or stayed inside a sub-conversation, before the
        prediction was answerable.  GENUINELY undeterminable: nothing in the
        trace bears on whether it was right.

      * target_passed — a tool call DID fire at or after the target step, but
        each such call was absorbed by an older checkpoint (the detector retires
        at most one checkpoint per tool call).  The detector declines to score
        it and this harness reports that faithfully, but the evidence leans
        miss: the tool it named did not run when it said it would.  This is the
        population that would narrow the reported range if the paper wanted a
        tighter bound, and it must be argued for explicitly rather than assumed.
    """
    order = 0
    ckpt_target: dict[int, int] = {}
    for act in actions:
        if act.kind != "predict":
            continue
        r = act.prediction.resources[0]
        ckpt_target[order] = r.expected_at_step or (
            act.step + max(r.consumer_step_offset, 1))
        order += 1

    tool_steps = sorted(a.step for a in actions if a.kind == "tool")
    unresolved_order = [i for i, cid in enumerate(created)
                        if cid not in resolved]
    for i in unresolved_order:
        target = ckpt_target.get(i)
        if target is not None and any(s >= target for s in tool_steps):
            res.unresolved_target_passed += 1
        else:
            res.unresolved_target_never_reached += 1


def score_chemgraph_inline(actions: list[Action], config: RuntimeConfig) -> ScoreResult:
    """HARNESS CONTROL ONLY — reimplements chemgraph.py:957-1003.

    The ChemGraph adapter does not use DivergenceDetector.  This mirrors its
    inlined check so the control isolates trace parsing from scoring semantics:

        single `_pending_checkpoint` slot, overwritten by each new prediction;
        gate `step >= (expected_at_step or 0)`;
        hit iff `tool_name == predicted_tool` (exact, no prefix);
        on either outcome the slot is cleared.

    These numbers describe the ChemGraph adapter, never the detector.
    """
    res = ScoreResult()
    pending: tuple[int, PredictionResult] | None = None

    for act in actions:
        if act.kind == "predict":
            if pending is not None:
                res.unresolved_superseded += 1
            pending = (act.step, act.prediction)
            res.predictions += 1
        else:
            res.tool_calls += 1
            if pending is None:
                res.no_opinion_tool_calls += 1
                continue
            _pstep, pred = pending
            r = pred.resources[0]
            expected_at_step = r.expected_at_step or 0
            if act.step >= expected_at_step:
                if act.tool == r.consumer_tool:
                    res.hits += 1
                else:
                    res.misses += 1
                pending = None
            else:
                res.no_opinion_tool_calls += 1

    if pending is not None:
        res.unresolved_pending_at_end += 1
    return res


SCORERS = {"detector": score_detector, "chemgraph_inline": score_chemgraph_inline}


def reconcile(pt: ParsedTrace, config: RuntimeConfig) -> Counter:
    """Per-checkpoint comparison of the RECORDED outcome to the replayed one.

    Each `predict` action carries the real checkpoint_id emitted by the live run,
    so a recorded `prediction_validated` / `divergence_detected` can be matched to
    the exact checkpoint the fixed detector scores.

    This is a diagnostic, not a validation: the fixed detector consumes a
    checkpoint on a miss, so it legitimately REASSIGNS which checkpoint absorbs
    a later match.  A recorded hit turning into a replayed miss is expected
    wherever the broken detector let an already-wrong checkpoint survive to be
    matched by a later tool.  Only the family totals are the recovered numbers.
    """
    det = DivergenceDetector(scheduler=None, config=config, bus=None)
    replay_id: dict[str, str] = {}   # replay ckpt id -> real ckpt id
    outcome: dict[str, str] = {}     # real ckpt id -> "hit" | "miss"
    for act in pt.actions:
        if act.kind == "predict":
            c = det.on_prediction(act.prediction, step=act.step)
            replay_id[c.checkpoint_id] = act.checkpoint_id
        else:
            hit, _a, c = det.on_tool_about_to_execute(act.tool, step=act.step)
            if c is None:
                continue
            outcome[replay_id.get(c.checkpoint_id, "")] = "hit" if hit else "miss"

    out = Counter()
    for act in pt.actions:
        if act.kind != "predict":
            continue
        cid = act.checkpoint_id
        rec = ("hit" if cid in pt.rec_validated_ids
               else "miss" if cid in pt.rec_diverged_ids else "unscored")
        rep = outcome.get(cid, "unresolved")
        out[f"{rec}->{rep}"] += 1
    return out


# ---------------------------------------------------------------------------
# Trial discovery / faceting
# ---------------------------------------------------------------------------

GPU_SHORT = {
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "blackwell",
    "NVIDIA L40S": "l40s",
}


@dataclass
class Trial:
    workload: str
    config: str
    trial: str
    trace: Path
    gpu: str
    gpu_raw: str
    family: str
    status: str


def _meta_for(trial_dir: Path) -> tuple[str, str, str]:
    """(gpu_short, gpu_raw, status) for a trial.

    GPU: meta.json `gpus` first, summary.json `gpu_name` next.  AtomAgents
    summaries carry no `gpu_name` at all (0 of 62), so meta.json is the only
    source for that family and the fallback exists for ChemGraph.

    Status: meta.json `status`.  This is NOT cosmetic — 17 of the 59 AtomAgents
    traces come from trials meta.json marks `failed`, several of which aborted
    after ~2 s having already emitted predictions and tool calls.  Their
    predictions are real events but they are not eval data, so status is part of
    the facet key and never pooled with `completed`.
    """
    gpu, gpu_raw, status = "unknown_gpu", "unknown", "unknown"
    meta = trial_dir / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            status = str(m.get("status") or "unknown")
            names = sorted({str(g).split(",")[0].strip() for g in (m.get("gpus") or [])})
            if names:
                gpu_raw = "+".join(names)
                gpu = GPU_SHORT.get(gpu_raw, gpu_raw)
        except Exception:
            pass
    if gpu == "unknown_gpu":
        summ = trial_dir / "summary.json"
        if summ.exists():
            try:
                s = json.loads(summ.read_text())
                raw = s.get("gpu_name")
                if raw:
                    gpu_raw = raw
                    gpu = GPU_SHORT.get(raw, raw)
            except Exception:
                pass
    return gpu, gpu_raw, status


def discover(runs_dir: Path, workload_filter: str | None) -> list[Trial]:
    trials: list[Trial] = []
    for trace in sorted(runs_dir.glob("*/*/*/trace.jsonl")):
        trial_dir = trace.parent
        config_dir = trial_dir.parent
        workload = config_dir.parent.name
        if workload_filter and workload_filter not in workload:
            continue
        gpu, gpu_raw, status = _meta_for(trial_dir)
        trials.append(Trial(
            workload=workload,
            config=config_dir.name,
            trial=trial_dir.name,
            trace=trace,
            gpu=gpu,
            gpu_raw=gpu_raw,
            family="atomagents" if workload.startswith("atomagents") else "chemgraph",
            status=status,
        ))
    return trials


def recorded_totals(runs_dir: Path, family_prefix: str) -> dict:
    """Recompute the recorded aggregate straight from the summaries on disk."""
    tot = Counter()
    n = 0
    for summ in sorted(runs_dir.glob(f"{family_prefix}*/*/*/summary.json")):
        try:
            s = json.loads(summ.read_text())
        except Exception:
            continue
        n += 1
        for k in ("prediction_count", "hit_count", "miss_count",
                  "unvalidated_count", "divergence_count", "prefetch_cancelled"):
            tot[k] += s.get(k) or 0
    return {"n_summaries": n, **dict(tot)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _fmt_range(hits: int, misses: int, unresolved: int) -> str:
    denom_hi = hits + misses
    denom_lo = hits + misses + unresolved
    if denom_lo == 0:
        return "n/a"
    lo = hits / denom_lo
    hi = hits / denom_hi if denom_hi else 1.0
    if unresolved == 0:
        return f"{lo:.4f}"
    return f"[{lo:.4f}, {hi:.4f}]"


def build_rows(trials: list[Trial], parsed: dict[Path, ParsedTrace],
               scorer: str, config: RuntimeConfig) -> list[dict]:
    fn = SCORERS[scorer]
    groups: dict[tuple, list[Trial]] = defaultdict(list)
    for t in trials:
        groups[(t.workload, t.config, t.gpu, t.status)].append(t)

    rows = []
    for (workload, cfg, gpu, status), ts in sorted(groups.items()):
        agg = ScoreResult()
        rec_val = rec_div = 0
        steps_inferred = step_fail = step_checks = 0
        for t in ts:
            pt = parsed[t.trace]
            agg.add(fn(pt.actions, config))
            rec_val += pt.rec_validated
            rec_div += pt.rec_diverged
            steps_inferred += pt.n_steps_inferred
            step_fail += pt.n_step_check_fail
            step_checks += pt.n_step_checks
        resolved = agg.hits + agg.misses
        rows.append({
            "scorer": scorer,
            "workload": workload,
            "config": cfg,
            "gpu": gpu,
            "trial_status": status,
            "n_trials": len(ts),
            "n_predictions": agg.predictions,
            "n_resolved": resolved,
            "hits": agg.hits,
            "misses": agg.misses,
            "unresolved": agg.unresolved,
            "unresolved_pending_at_end": agg.unresolved_pending_at_end,
            "unresolved_aged_out": agg.unresolved_aged_out,
            "unresolved_superseded": agg.unresolved_superseded,
            "unresolved_target_never_reached": agg.unresolved_target_never_reached,
            "unresolved_target_passed": agg.unresolved_target_passed,
            "accuracy_resolved": round(agg.hits / resolved, 6) if resolved else None,
            "accuracy_lo": round(agg.hits / agg.predictions, 6) if agg.predictions else None,
            "accuracy_hi": round(agg.hits / resolved, 6) if resolved else None,
            "accuracy_bracket": _fmt_range(agg.hits, agg.misses, agg.unresolved),
            "tool_calls": agg.tool_calls,
            "no_opinion_tool_calls": agg.no_opinion_tool_calls,
            "recorded_hit_count": rec_val,
            "recorded_divergence_count": rec_div,
            "steps_inferred": steps_inferred,
            "step_reconstruction_checks": step_checks,
            "step_reconstruction_failures": step_fail,
            "indicative_n_lt_30": agg.predictions < 30,
        })
    return rows


TABLE_COLS = [
    ("workload", 24), ("config", 20), ("gpu", 10), ("trial_status", 11),
    ("n_trials", 8), ("n_predictions", 8), ("hits", 6), ("misses", 7),
    ("unresolved", 11), ("accuracy_bracket", 20), ("recorded_hit_count", 9),
    ("recorded_divergence_count", 9),
]


def print_table(rows: list[dict], title: str) -> None:
    print()
    print(title)
    print("-" * 152)
    hdr = "".join(f"{name[:w - 1]:<{w}}" for name, w in TABLE_COLS)
    print(hdr + "  flag")
    print("-" * 152)
    for r in rows:
        if r["n_predictions"] == 0 and r["n_trials"] == 0:
            continue
        line = "".join(f"{str(r[name])[:w - 1]:<{w}}" for name, w in TABLE_COLS)
        flag = ""
        if r["n_predictions"] == 0:
            flag = "no predictions"
        elif r["indicative_n_lt_30"]:
            flag = "INDICATIVE (n<30)"
        if r["step_reconstruction_failures"]:
            flag += f"  STEP-RECON-FAIL={r['step_reconstruction_failures']}"
        print(line + "  " + flag)
    print("-" * 152)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Control check
# ---------------------------------------------------------------------------

def run_control(trials: list[Trial], parsed: dict[Path, ParsedTrace],
                config: RuntimeConfig) -> tuple[bool, dict]:
    """chemgraph_inline over every ChemGraph trace must reproduce the recorded
    hit/miss counts EXACTLY.  See module docstring for why the control uses this
    scorer and not the detector."""
    cg = [t for t in trials if t.family == "chemgraph"]
    agg = ScoreResult()
    rec_val = rec_div = 0
    step_fail = 0
    for t in cg:
        pt = parsed[t.trace]
        agg.add(score_chemgraph_inline(pt.actions, config))
        rec_val += pt.rec_validated
        rec_div += pt.rec_diverged
        step_fail += pt.n_step_check_fail

    det = ScoreResult()
    for t in cg:
        det.add(score_detector(parsed[t.trace].actions, config))

    ok = (agg.hits == rec_val and agg.misses == rec_div
          and rec_val == CONTROL_EXPECT["hits"] and rec_div == CONTROL_EXPECT["misses"]
          and step_fail == 0)
    return ok, {
        "n_chemgraph_traces": len(cg),
        "recorded_hit_count": rec_val,
        "recorded_divergence_count": rec_div,
        "recorded_hit_count_expected": CONTROL_EXPECT["hits"],
        "recorded_divergence_count_expected": CONTROL_EXPECT["misses"],
        "replay_chemgraph_inline_hits": agg.hits,
        "replay_chemgraph_inline_misses": agg.misses,
        "replay_chemgraph_inline_predictions": agg.predictions,
        "replay_chemgraph_inline_unresolved": agg.unresolved,
        "replay_detector_hits": det.hits,
        "replay_detector_misses": det.misses,
        "replay_detector_predictions": det.predictions,
        "replay_detector_unresolved": det.unresolved,
        "step_reconstruction_failures": step_fail,
        "pass": ok,
    }


def print_control(info: dict) -> None:
    print()
    print("=" * 100)
    print("CONTROL — ChemGraph replay vs recorded  (must match EXACTLY)")
    print("=" * 100)
    print(f"chemgraph traces replayed                {info['n_chemgraph_traces']}")
    print()
    print("  recorded (from trace prediction_validated / divergence_detected events,")
    print("  which equal the campaign summary hit_count / divergence_count):")
    print(f"    hit_count                            {info['recorded_hit_count']}"
          f"   (expected {info['recorded_hit_count_expected']})")
    print(f"    divergence_count                     {info['recorded_divergence_count']}"
          f"   (expected {info['recorded_divergence_count_expected']})")
    print()
    print("  replayed with scorer=chemgraph_inline (the ChemGraph adapter's own")
    print("  semantics — this is the harness control):")
    print(f"    predictions scored                   {info['replay_chemgraph_inline_predictions']}")
    print(f"    hits                                 {info['replay_chemgraph_inline_hits']}")
    print(f"    misses                               {info['replay_chemgraph_inline_misses']}")
    print(f"    unresolved                           {info['replay_chemgraph_inline_unresolved']}")
    print()
    print("  replayed with scorer=detector (the FIXED DivergenceDetector, which the")
    print("  ChemGraph adapter never used — shown to expose the semantic delta, NOT")
    print("  as a control):")
    print(f"    predictions scored                   {info['replay_detector_predictions']}")
    print(f"    hits                                 {info['replay_detector_hits']}")
    print(f"    misses                               {info['replay_detector_misses']}")
    print(f"    unresolved                           {info['replay_detector_unresolved']}")
    print()
    print(f"  step reconstruction failures           {info['step_reconstruction_failures']}"
          "  (must be 0)")
    print()
    print(f"  CONTROL: {'PASS' if info['pass'] else 'FAIL'}")
    if not info["pass"]:
        print("  -> the harness does not reproduce its own control.  Do NOT publish")
        print("     any AtomAgents number produced by it.")
    print("=" * 100)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--workload", default=None,
                    help="substring filter on workload dir name")
    ap.add_argument("--scorer", default="detector", choices=sorted(SCORERS),
                    help="scorer for the main per-facet table (default: detector)")
    ap.add_argument("--max-horizon", type=int, default=2,
                    help="RuntimeConfig.max_horizon (assumption A4; default 2)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--per-trial", action="store_true",
                    help="also emit one row per trial")
    ap.add_argument("--control", action="store_true",
                    help="run the control check only; exit 1 on failure")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    config = RuntimeConfig(run_id="replay", max_horizon=args.max_horizon)

    all_trials = discover(runs_dir, None)
    if not all_trials:
        print(f"no traces under {runs_dir}")
        return 1
    parsed = {t.trace: parse_trace(t.trace) for t in all_trials}

    ok, control = run_control(all_trials, parsed, config)

    trials = [t for t in all_trials
              if not args.workload or args.workload in t.workload]
    if not trials:
        print(f"no traces match --workload {args.workload!r}")
        return 1

    if args.control:
        print_control(control)
        return 0 if ok else 1

    print("=" * 100)
    print("replay_divergence.py — recovering hit/miss by driving the FIXED detector")
    print("=" * 100)
    print(f"repo commit                    {_git_commit()}")
    print(f"runs dir                       {_rel(runs_dir)}")
    print(f"trials with a trace            {len(trials)}")
    print(f"max_horizon (A4)               {config.max_horizon}"
          f"   -> checkpoint window {max(config.max_horizon * 8, 60)} steps")
    sel = [parsed[t.trace] for t in trials]
    tot_pred_events = sum(p.n_prediction_events for p in sel)
    tot_scored = sum(p.n_scored_predictions for p in sel)
    tot_tools = sum(p.n_tool_calls for p in sel)
    tot_bad = sum(p.n_bad_lines for p in sel)
    tot_inf = sum(p.n_steps_inferred for p in sel)
    tot_chk = sum(p.n_step_checks for p in sel)
    tot_fail = sum(p.n_step_check_fail for p in sel)
    print(f"prediction_result events       {tot_pred_events}")
    print(f"  became a pending checkpoint  {tot_scored}   (A2: next line is checkpoint_created)")
    print(f"  staging/preboot, never scored{tot_pred_events - tot_scored}")
    print(f"tool_call events               {tot_tools}")
    print(f"  step reconstructed & checked {tot_chk}, disagreements {tot_fail}  (A3)")
    print(f"  step inferred as prev+1      {tot_inf}")
    print(f"malformed JSON lines           {tot_bad}")

    print_control(control)
    if not ok:
        print("\nCONTROL FAILED — refusing to print recovered AtomAgents numbers.")
        return 1

    rows = build_rows(trials, parsed, args.scorer, config)
    aa = [r for r in rows if r["workload"].startswith("atomagents")]
    cg = [r for r in rows if not r["workload"].startswith("atomagents")]

    print_table(aa, f"ATOMAGENTS — recovered by the FIXED detector (scorer={args.scorer})")
    print_table(cg, f"CHEMGRAPH — same harness, same scorer={args.scorer} "
                    "(NOT the adapter's own semantics; see --control)")

    # Family totals, per GPU (never pooled across GPU).
    print()
    print("FAMILY TOTALS — faceted by GPU and trial status, never pooled")
    print("-" * 120)
    fam_gpu: dict[tuple, ScoreResult] = defaultdict(ScoreResult)
    for r in rows:
        fam = "atomagents" if r["workload"].startswith("atomagents") else "chemgraph"
        s = ScoreResult(
            predictions=r["n_predictions"], hits=r["hits"], misses=r["misses"],
            unresolved_aged_out=r["unresolved_aged_out"],
            unresolved_pending_at_end=r["unresolved_pending_at_end"],
            unresolved_superseded=r["unresolved_superseded"],
            unresolved_target_never_reached=r["unresolved_target_never_reached"],
            unresolved_target_passed=r["unresolved_target_passed"],
        )
        fam_gpu[(fam, r["gpu"], r["trial_status"])].add(s)
    total_rows = []
    for (fam, gpu, status), s in sorted(fam_gpu.items()):
        if s.predictions == 0:
            continue
        rng = _fmt_range(s.hits, s.misses, s.unresolved)
        print(f"  {fam:<12} {gpu:<11} {status:<12} n_pred={s.predictions:<5} "
              f"hits={s.hits:<5} misses={s.misses:<5} unresolved={s.unresolved:<5} "
              f"accuracy={rng}"
              + ("   INDICATIVE (n<30)" if s.predictions < 30 else ""))
        print(f"  {'':<12} {'':<11} {'':<12} of the unresolved: "
              f"{s.unresolved_target_never_reached} never reached their target "
              f"step (undeterminable), "
              f"{s.unresolved_target_passed} did (evidence leans miss)")
        total_rows.append({
            "scorer": args.scorer, "family": fam, "gpu": gpu,
            "trial_status": status,
            "n_predictions": s.predictions, "hits": s.hits, "misses": s.misses,
            "unresolved": s.unresolved,
            "unresolved_pending_at_end": s.unresolved_pending_at_end,
            "unresolved_aged_out": s.unresolved_aged_out,
            "unresolved_superseded": s.unresolved_superseded,
            "unresolved_target_never_reached": s.unresolved_target_never_reached,
            "unresolved_target_passed": s.unresolved_target_passed,
            # NOT a bound.  The value you get if the target_passed leftovers are
            # argued to be misses and the target_never_reached ones are dropped
            # from the denominator as unmeasured.  Sits INSIDE [lo, hi].
            "accuracy_if_only_target_passed_are_misses": (
                round(s.hits / (s.hits + s.misses + s.unresolved_target_passed), 6)
                if (s.hits + s.misses + s.unresolved_target_passed) else None),
            "accuracy_lo": round(s.hits / s.predictions, 6) if s.predictions else None,
            "accuracy_hi": round(s.hits / (s.hits + s.misses), 6) if (s.hits + s.misses) else None,
            "accuracy_bracket": rng,
            "indicative_n_lt_30": s.predictions < 30,
        })
    print("-" * 120)

    # Checkpoint-level reconciliation: what the live (broken) run recorded for
    # each checkpoint vs what the fixed detector scores for that same checkpoint.
    print()
    print("RECONCILIATION — recorded outcome -> replayed outcome, per checkpoint")
    print("  (diagnostic only; a miss consumes a checkpoint, so the fixed detector")
    print("   legitimately reassigns which checkpoint absorbs a later match)")
    print("-" * 110)
    recon: dict[str, Counter] = defaultdict(Counter)
    for t in trials:
        recon[f"{t.family} / {t.status}"].update(reconcile(parsed[t.trace], config))
    recon_out = {}
    for key, c in sorted(recon.items()):
        if not sum(c.values()):
            continue
        print(f"  {key}:")
        for k in sorted(c):
            print(f"      {k:<24} {c[k]}")
        recon_out[key] = dict(c)
    print("-" * 110)

    per_trial_rows = []
    if args.per_trial:
        fn = SCORERS[args.scorer]
        for t in trials:
            s = fn(parsed[t.trace].actions, config)
            per_trial_rows.append({
                "scorer": args.scorer, "workload": t.workload, "config": t.config,
                "gpu": t.gpu, "trial_status": t.status,
                "trial": t.trial, "trace": _rel(t.trace),
                "n_predictions": s.predictions, "hits": s.hits, "misses": s.misses,
                "unresolved": s.unresolved, "tool_calls": s.tool_calls,
                "recorded_hit_count": parsed[t.trace].rec_validated,
                "recorded_divergence_count": parsed[t.trace].rec_diverged,
            })

    print()
    print("SCOPE AND LIMITS — state these wherever these numbers are used")
    print("  * This scores the prediction stream that WAS recorded, under fixed")
    print("    semantics.  It is NOT what the fixed runtime would have produced")
    print("    live: a miss triggers conservative mode, which would have")
    print("    suppressed later predictions.  Prediction-quality claims only.")
    print("  * 'unresolved' predictions have NO determinable outcome in the trace")
    print("    and are never folded into hits or misses.  Where unresolved > 0 the")
    print("    honest statement is the bracket, not a point estimate.")
    print("  * ChemGraph rows under scorer=detector are counterfactual: that")
    print("    adapter never called DivergenceDetector.  Its real recorded numbers")
    print("    are in the recorded_* columns and reproduced by --control.")
    print("  * Facets are never pooled across GPU or trial status;")
    print("    atomagents_exp3_aligned spans both L40S and Blackwell, and 17 of")
    print("    the 59 AtomAgents traces come from trials meta.json marks failed.")
    print("  * The recovered numerator is NOT the recorded one.  A miss consumes")
    print("    a checkpoint, so 91/219 is not a guaranteed floor — see the")
    print("    reconciliation block and the CORRECTION in the module docstring.")

    if not args.no_write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "generated_by": "scripts/replay_divergence.py",
            "git_commit": _git_commit(),
            "runs_dir": _rel(runs_dir),
            "scorer": args.scorer,
            "max_horizon": config.max_horizon,
            "checkpoint_window_steps": max(config.max_horizon * 8, 60),
            "n_trials": len(trials),
            "prediction_result_events": tot_pred_events,
            "scored_predictions": tot_scored,
            "tool_call_events": tot_tools,
            "step_reconstruction_checks": tot_chk,
            "step_reconstruction_failures": tot_fail,
            "steps_inferred": tot_inf,
            "malformed_json_lines": tot_bad,
            "control": control,
            "reconciliation": recon_out,
        }
        jpath = out / f"replay_divergence_{args.scorer}.json"
        jpath.write_text(json.dumps(
            {"meta": meta, "facets": rows, "family_totals": total_rows,
             "per_trial": per_trial_rows}, indent=2))
        cpath = out / f"replay_divergence_{args.scorer}.csv"
        with cpath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {_rel(jpath)}")
        print(f"wrote {_rel(cpath)}")
        if per_trial_rows:
            tpath = out / f"replay_divergence_{args.scorer}_per_trial.csv"
            with tpath.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_trial_rows[0]))
                w.writeheader()
                w.writerows(per_trial_rows)
            print(f"wrote {_rel(tpath)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
