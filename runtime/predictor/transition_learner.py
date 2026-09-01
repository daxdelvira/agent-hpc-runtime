"""
predictor/transition_learner.py — Learn tool/model co-occurrence statistics from traces.

Reads JSONL workflow traces and computes the empirical probability that tool B
(or model B) follows tool A within k steps, where "k steps" counts steps OF THE
SAME KIND -- see learn_from_traces for the offset basis and the 2026-08-30 fix.

Usage (CLI)
-----------
    python3 -m runtime.predictor.transition_learner \\
        logs/workflow_traces/*.jsonl \\
        --output runtime/predictor/data/learned_transitions.json \\
        --max-offset 3 \\
        --min-count 2

    GLOB ALL OF THEM.  The corpus holds three prefixes -- runtime_trace_*
    (AtomAgents), chemgraph_trace_* and workflow_trace_* -- and the tool
    vocabularies are DISJOINT: no AtomAgents trace contains run_ase, no
    ChemGraph trace contains plan_task.  Narrowing the glob to one prefix does
    not shrink the table, it DELETES the other workload's rows entirely.

    Replay-harness traces are EXCLUDED BY DEFAULT (--no-filter-synthetic keeps
    them).  Either way the choice, the rule and the exact drop counts are
    written into the output under `synthetic_filter` -- see
    SYNTHETIC_FILTER_RULE below for the rule and how its thresholds were
    calibrated.

Usage (library)
---------------
    from runtime.predictor.transition_learner import learn_from_traces
    table = learn_from_traces(jsonl_paths, max_offset=3, min_count=2)
    table.save("runtime/predictor/data/learned_transitions.json")
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransitionEntry:
    target: str
    target_type: str       # "tool" | "model"
    probability: float
    count: int
    offset: int            # steps ahead (1-based)


# Written into every table this module produces.  Bumped from "1" on
# 2026-08-30 when the offset basis was fixed (see OFFSET_BASIS below) and from
# "2" on 2026-09-01 when the synthetic-trace filter changed which traces are
# counted at all (see SYNTHETIC_FILTER_RULE).  A version bump here always means
# "a number in this file is not comparable with the same number in the previous
# version", never merely "regenerated with more traces".
TABLE_VERSION = "3"

# What an `offset` in this file COUNTS.  Recorded in the JSON so a reader can
# never mistake one basis for the other, and so a pre-fix table is detectable
# rather than silently misread.
OFFSET_BASIS = {
    "tool_transitions": "tool_call_subsequence",
    "model_transitions": "llm_call_subsequence",
}
LEGACY_OFFSET_BASIS = {
    "tool_transitions": "mixed_event_sequence",
    "model_transitions": "mixed_event_sequence",
}


# ---------------------------------------------------------------------------
# Synthetic-trace filter (2026-09-01)
# ---------------------------------------------------------------------------
# The AtomAgents half of logs/workflow_traces/ mixes real cluster runs with
# replay-harness runs, and nothing used to separate them: 5365 of the 7236
# tool_call events in the unfiltered corpus (74%) came from harness traces, so
# every AtomAgents transition probability was mostly a statement about the
# harness.  There is no provenance field to filter on -- a runtime_trace_*
# payload carries only `tool` / `model` / `step`, and the `runtime_schema`
# marker at adapters/atomagents.py:207 is emitted ONLY in execution mode, of
# which this corpus contains zero events.  So the filter has to be behavioural.
#
# WHAT THE RULE MEASURES: whether the LLM turns in a trace took long enough to
# have actually happened.  A trace is a sequence of agent turns; in a real run
# each turn is a Qwen2.5-VL-72B completion over the full AtomAgents context,
# followed by a tool that usually shells out to LAMMPS.  Timing one turn is the
# closest thing to a provenance check the payloads allow.
#
# THE ONE CONFOUND, and why the threshold is a BURST length rather than a
# median.  Every real AtomAgents trace contains sub-second inter-llm_call gaps,
# because in observation mode one agent step emits its llm_call TWICE ~0.37 s
# apart (see the TOOL_CALL_EMISSION_* note at adapters/atomagents.py:80 and the
# "What a tool_call event means" section in learn_from_traces below).  A real
# trace therefore looks like
#
#     +0.38 | +185.50 | +0.40 | +0.38 | +240.48 | +0.37 | +521.04 | +914.46
#     (runtime_trace_20260709_191258_eval_atomagents_exp3_full_system_t01)
#
# -- short runs of fast gaps separated by real work.  A harness trace looks
# like
#
#     +0.32 +0.29 +0.27 +0.31 +0.40 +0.30 +0.30 +0.48 +0.31 +0.26 +0.24 +0.26
#     (runtime_trace_20260602_135508_f89eba12-582, 29 tool calls in 58 s)
#
# -- an unbroken chain.  A MEDIAN cannot tell these apart: a real trial with
# seven llm_calls, four of them duplicate observations, has a sub-second median
# while running for 1822 s with 911 s LAMMPS gaps.  Filtering on the median
# discards 21 of the paper's own eval trials.  The LENGTH OF THE LONGEST CHAIN
# does tell them apart, because the duplicate-observation artifact is bounded
# and a harness chain is not.
#
# CLAUSE 1 -- fast-turn burst.  Exclude when more than
# MAX_CONSECUTIVE_FAST_TURNS consecutive inter-llm_call gaps are below
# MIN_LLM_TURN_SECONDS.  Calibration on this corpus: traces with unambiguous
# real work (a >= 100 s gap and a LAMMPS-scale runtime) top out at a burst of
# 6; the harness population starts at 8.  NOTHING IN THE CORPUS SITS AT 7, so
# the verdict is identical for any threshold in [7, 8) -- the exact value is
# not doing the work, the gap in the distribution is.  7 is chosen to leave a
# turn of headroom above the largest burst any real trace has shown.
#
# CLAUSE 2 -- tool-call rate.  Exclude when the whole trace averages less than
# MIN_SECONDS_PER_TOOL_CALL of wall clock per tool_call event.  This catches
# short harness traces (and 0.6 s aborted trials) whose gap chains are too
# short to trip clause 1.  Calibration: the fastest trace with unambiguous real
# work averages 69 s per tool_call; the excluded population is all <= 2.5 s.
# 3.0 s is ~23x below the slowest plausible real trace, i.e. the threshold is
# deliberately set to err toward KEEPING data.  Again the distribution has a
# hole (2.46 s -> 4.96 s) and any threshold in [2.5, 4.9) gives the same
# verdicts.
#
# SCOPE.  Both clauses read llm_call timestamps, so the rule applies only to
# traces with at least two of them.  ChemGraph traces contain none (0 llm_call
# events across all 338) and are always kept, UNJUDGED -- inventing a rule for
# them from tool timestamps alone would be a different, uncalibrated filter
# wearing this one's name.  They are counted separately in the header as
# `out_of_scope_traces` so "kept" never has to mean two different things.
#
# LIMITS, stated because the header cannot state them.  This rule tests whether
# the LLM turns were slow enough to be real; it does NOT test whether the tool
# sequence was agent-chosen.  Repeated identical tool sequences are NOT used as
# evidence here, and should not be: on this workload the real trials repeat one
# 7-call sequence 15 times and one 8-call sequence 13 times, because it is a
# fixed benchmark task.  Sequence repetition separates nothing.
SYNTHETIC_FILTER_RULE = "llm_turn_plausibility_v1"

# Floor on a genuine agent turn.  Two llm_call events closer together than this
# cannot both be independent 72B completions; in this corpus the fastest
# interval between DISTINCT turns is 185 s and every sub-second interval is the
# double-observation artifact (0.24-0.48 s).
MIN_LLM_TURN_SECONDS = 1.0

# Longest run of consecutive sub-second turns a real trace may show.  See
# CLAUSE 1.
MAX_CONSECUTIVE_FAST_TURNS = 7

# Floor on whole-trace wall clock per tool_call event.  See CLAUSE 2.
MIN_SECONDS_PER_TOOL_CALL = 3.0

# Verdicts returned by classify_trace.
VERDICT_KEPT = "kept"
VERDICT_OUT_OF_SCOPE = "out_of_scope"
VERDICT_EXCLUDED_BURST = "excluded:fast_turn_burst"
VERDICT_EXCLUDED_RATE = "excluded:tool_call_rate"

# What `synthetic_filter` looks like in a table written before 2026-09-01: the
# key is absent, and its absence means "unfiltered, and nobody said so".
# Reported rather than defaulted, exactly as LEGACY_OFFSET_BASIS is.
LEGACY_SYNTHETIC_FILTER = {
    "applied": False,
    "rule": None,
    "note": ("no synthetic_filter in this table -- it predates the 2026-09-01 "
             "filter and counts replay-harness traces alongside real ones"),
}


def classify_trace(events: list[dict]) -> str:
    """Return one of the VERDICT_* constants for one parsed trace.

    Pure function of the event list so the rule can be tested, and audited,
    without re-reading the corpus.  See the block comment above for what each
    clause means and how its threshold was calibrated.
    """
    llm_t = [e.get("epoch_time") for e in events
             if e.get("event_type") == "llm_call"]
    llm_t = [t for t in llm_t if isinstance(t, (int, float))]
    if len(llm_t) < 2:
        # No measurable turn: the rule has nothing to say.  Keep, and say so.
        return VERDICT_OUT_OF_SCOPE

    # Clause 1: longest run of consecutive sub-second turns.
    longest = current = 0
    for a, b in zip(llm_t, llm_t[1:]):
        if b - a < MIN_LLM_TURN_SECONDS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if longest > MAX_CONSECUTIVE_FAST_TURNS:
        return VERDICT_EXCLUDED_BURST

    # Clause 2: whole-trace wall clock per tool_call event.
    n_tool = sum(1 for e in events if e.get("event_type") == "tool_call")
    if n_tool:
        stamps = [e.get("epoch_time") for e in events]
        stamps = [t for t in stamps if isinstance(t, (int, float))]
        span = (max(stamps) - min(stamps)) if stamps else 0.0
        if span / n_tool < MIN_SECONDS_PER_TOOL_CALL:
            return VERDICT_EXCLUDED_RATE

    return VERDICT_KEPT


@dataclass
class TransitionTable:
    """
    Learned co-occurrence table from JSONL traces.

    tool_transitions[source_tool][offset] = [TransitionEntry, ...]  sorted by prob desc
    model_transitions[source_model][offset] = [TransitionEntry, ...] sorted by prob desc

    `offset` is a position in the SUBSEQUENCE OF THE SAME KIND, never a
    position in the raw event stream — see learn_from_traces.
    """
    version: str = TABLE_VERSION
    n_traces: int = 0
    n_tool_events: int = 0
    generated_at: str = ""
    offset_basis: dict[str, str] = field(default_factory=lambda: dict(OFFSET_BASIS))
    # Which traces were counted, and which were dropped and why.  Always
    # written, including when the filter was OFF -- a reader must be able to
    # tell a filtered table from an unfiltered one without rerunning anything.
    synthetic_filter: dict = field(default_factory=dict)
    tool_transitions: dict[str, dict[int, list[TransitionEntry]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    model_transitions: dict[str, dict[int, list[TransitionEntry]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def top_tools(self, source: str, offset: int, n: int = 3) -> list[TransitionEntry]:
        """Top-N most likely tools to follow `source` in `offset` steps."""
        return self.tool_transitions.get(source, {}).get(offset, [])[:n]

    def top_models(self, source: str, offset: int, n: int = 3) -> list[TransitionEntry]:
        return self.model_transitions.get(source, {}).get(offset, [])[:n]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(transitions: dict) -> dict:
            out = {}
            for source, offset_map in transitions.items():
                out[source] = {}
                for offset, entries in offset_map.items():
                    out[source][str(offset)] = [asdict(e) for e in entries]
            return out

        data = {
            "version": self.version,
            "n_traces": self.n_traces,
            "n_tool_events": self.n_tool_events,
            "generated_at": self.generated_at,
            "offset_basis": self.offset_basis,
            "synthetic_filter": self.synthetic_filter,
            "tool_transitions": _serialise(self.tool_transitions),
            "model_transitions": _serialise(self.model_transitions),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TransitionTable":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)

        def _deserialise(raw: dict) -> dict:
            result: dict = defaultdict(lambda: defaultdict(list))
            for source, offset_map in raw.items():
                for offset_str, entries in offset_map.items():
                    offset = int(offset_str)
                    result[source][offset] = [TransitionEntry(**e) for e in entries]
            return result

        # A table with no `offset_basis` predates the 2026-08-30 fix and its
        # offsets count RAW EVENTS, not steps of the same kind.  Label it
        # rather than defaulting it into the current basis, so a stale file is
        # detectable by any consumer that cares.
        t = cls(
            version=data.get("version", "1"),
            n_traces=data.get("n_traces", 0),
            n_tool_events=data.get("n_tool_events", 0),
            generated_at=data.get("generated_at", ""),
            offset_basis=data.get("offset_basis") or dict(LEGACY_OFFSET_BASIS),
            # Absent key means "written before the filter existed".  Label it,
            # do not default it into looking deliberately unfiltered.
            synthetic_filter=(data.get("synthetic_filter")
                              or dict(LEGACY_SYNTHETIC_FILTER)),
        )
        t.tool_transitions = _deserialise(data.get("tool_transitions", {}))
        t.model_transitions = _deserialise(data.get("model_transitions", {}))
        return t


# ---------------------------------------------------------------------------
# Learning algorithm
# ---------------------------------------------------------------------------

def learn_from_traces(
    jsonl_paths: list[str],
    max_offset: int = 3,
    min_count: int = 2,
    exclude_synthetic: bool = True,
) -> TransitionTable:
    """
    Scan JSONL files and compute co-occurrence probabilities.

    Offset basis
    ------------
    Offsets are counted **within the subsequence of the same kind**:
    tool->tool over the tool_call subsequence, model->model over the llm_call
    subsequence.  `offset = 1` therefore means "the next TOOL", not "the next
    event".

    Until 2026-08-30 both kinds were interleaved into ONE list and the offset
    was that list's index delta, so a tool offset counted EVENTS.  ChemGraph
    traces carry no llm_call events, so their rows were unaffected — which is
    why they reproduced cleanly and hid the defect.  AtomAgents traces
    interleave at least one llm_call between every tool pair, so the true
    successor was pushed past max_offset whenever two or more LLM calls
    intervened, and only same-tool pairs separated by exactly one llm_call
    survived at (mixed) offset 2.  On the workload the paper evaluates, that
    left the table asserting `plan_task +2 -> plan_task, p=1.0, n=45` while
    the real successor `code_task` (78 of 143 occurrences) was invisible.

    Model->model rows were NOT already correct, contrary to first appearances:
    both ends being llm_call fixes the PAIRING, not the OFFSET, because the
    mixed index delta also counted intervening tool_calls.  On the current
    corpus `72B +1 -> 72B` went n=5488 -> n=11311 under this fix.

    What a `tool_call` event means (read before trusting a self-loop)
    ----------------------------------------------------------------
    In AtomAgents traces a `tool_call` event records that the admin agent
    OBSERVED A MESSAGE REQUESTING a tool (runtime/adapters/atomagents.py:387,
    emitted from the position=1 reply handler), not that the tool ran.  The two
    diverge, and per-tool:

      * `plan_task` — 21 of 41 completed non-baseline eval trials emit TWO
        tool_call events ~0.37 s apart and execute ONCE.  Verified against
        AtomAgents' independent metrics.csv: in all 21, BOTH events precede the
        single `agent:plan_task` execution's start.  Of the 45
        `plan_task -> plan_task` self-loops in this corpus, 42 are this
        artifact; collapsing them leaves n=3 and lifts `code_task` to 0.7723.
      * `analyze_screw_core`, `computation_task_screw_dislocation` — the
        repeats are REAL.  Each tool_call has its own `agent:<phase>` row with
        its own duration, and the two runs use different potentials
        (W_screw_Zhou04 then W_screw_w_eam4).

    So a blanket "collapse same-tool events within 1 s" rule is NOT safe here:
    the genuine analyze_screw_core repeats are 0.946-0.999 s apart and would be
    merged.  Time cannot separate the two cases; only metrics.csv can.  Any
    consumer of these traces must decide explicitly, and say which rule it
    used.  The real fix is at the emission site, which should emit tool_call
    once per EXECUTION.

    Corpus hygiene
    --------------
    The AtomAgents half of logs/workflow_traces/ is mostly replay harness.  Of
    its 5977 tool_call events, 5365 (90%) come from 51 traces whose LLM turns
    are too fast to have happened -- see SYNTHETIC_FILTER_RULE above for the
    rule and its calibration.  The single largest bucket in the unfiltered
    table, `create_potential_file +2 -> create_potential_file` (n=4628), is
    4432 events from just two run_ids (observe2-20260526, observe2-20260527)
    that hold the SAME 2265-call sequence, in which create_potential_file
    repeats 82-97 times in a row at 0.27 s intervals.

    `exclude_synthetic` is ON BY DEFAULT.  The contaminated table is never the
    one a caller wants without saying so, and this project has twice shipped a
    corpus that silently changed shape.  Passing False is supported and is
    RECORDED: `synthetic_filter.applied` is written either way, so an
    unfiltered table is labelled rather than merely indistinguishable.

    A trace the rule cannot judge -- fewer than two llm_call events, which is
    every ChemGraph trace -- is kept and counted separately as
    `out_of_scope_traces`.  ChemGraph rows are therefore byte-identical with
    the filter on or off.
    """
    # counts[(source, offset, target)] = int
    tool_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    model_counts: dict[tuple[str, int, str], int] = defaultdict(int)

    n_traces = 0
    n_tool_events = 0
    verdicts: dict[str, int] = defaultdict(int)
    dropped_tool_events: dict[str, int] = defaultdict(int)
    dropped_llm_events: dict[str, int] = defaultdict(int)

    for path in jsonl_paths:
        try:
            with open(path) as f:
                raw = [l.strip() for l in f if l.strip()]
        except OSError:
            continue

        events: list[dict] = []
        for line in raw:
            try:
                events.append(json.loads(line))
            except Exception:
                pass

        if not events:
            continue

        verdict = classify_trace(events) if exclude_synthetic else VERDICT_KEPT
        verdicts[verdict] += 1
        if verdict.startswith("excluded"):
            # Count what is being dropped BEFORE dropping it; a filter whose
            # cost cannot be read off the output is a silent corpus change.
            dropped_tool_events[verdict] += sum(
                1 for e in events
                if e.get("event_type") == "tool_call" and (e.get("payload") or {}).get("tool"))
            dropped_llm_events[verdict] += sum(
                1 for e in events
                if e.get("event_type") == "llm_call" and (e.get("payload") or {}).get("model"))
            continue

        n_traces += 1

        # TWO independent subsequences.  Keeping them separate is the whole
        # point: an offset must mean "n steps of the same kind ahead".
        tool_seq: list[str] = []
        model_seq: list[str] = []
        for ev in events:
            et = ev.get("event_type", "")
            p = ev.get("payload") or {}
            if et == "tool_call":
                tool = p.get("tool", "")
                if tool:
                    tool_seq.append(tool)
                    n_tool_events += 1
            elif et == "llm_call":
                model = p.get("model", "")
                if model:
                    model_seq.append(model)

        for seq, counts in ((tool_seq, tool_counts), (model_seq, model_counts)):
            for i, src_name in enumerate(seq):
                for j in range(i + 1, min(i + 1 + max_offset, len(seq))):
                    counts[(src_name, j - i, seq[j])] += 1
        # cross-type (tool->model) is still not tracked: it has no single
        # basis, and no consumer asks for it.

    # Normalise counts → probabilities
    n_excluded = (verdicts[VERDICT_EXCLUDED_BURST]
                  + verdicts[VERDICT_EXCLUDED_RATE])
    filter_header = {
        "applied": bool(exclude_synthetic),
        "rule": SYNTHETIC_FILTER_RULE if exclude_synthetic else None,
        "summary": (
            "excluded traces whose consecutive llm_call intervals are too "
            "short for the completions to have happened, or whose whole-trace "
            "wall clock per tool_call is too small for the tools to have run"
        ),
        "thresholds": {
            "min_llm_turn_seconds": MIN_LLM_TURN_SECONDS,
            "max_consecutive_fast_turns": MAX_CONSECUTIVE_FAST_TURNS,
            "min_seconds_per_tool_call": MIN_SECONDS_PER_TOOL_CALL,
        },
        "scope": "traces with >= 2 llm_call events",
        "scanned_traces": sum(verdicts.values()),
        "kept_traces": verdicts[VERDICT_KEPT],
        # Kept, but the rule had no llm timing to judge them by.  Every
        # ChemGraph trace lands here, which is why ChemGraph rows do not move.
        "out_of_scope_traces": verdicts[VERDICT_OUT_OF_SCOPE],
        "excluded_traces": n_excluded,
        "excluded_tool_events": sum(dropped_tool_events.values()),
        "excluded_llm_events": sum(dropped_llm_events.values()),
        "excluded_by_clause": {
            "fast_turn_burst": {
                "traces": verdicts[VERDICT_EXCLUDED_BURST],
                "tool_events": dropped_tool_events[VERDICT_EXCLUDED_BURST],
                "llm_events": dropped_llm_events[VERDICT_EXCLUDED_BURST],
            },
            "tool_call_rate": {
                "traces": verdicts[VERDICT_EXCLUDED_RATE],
                "tool_events": dropped_tool_events[VERDICT_EXCLUDED_RATE],
                "llm_events": dropped_llm_events[VERDICT_EXCLUDED_RATE],
            },
        },
    }

    table = TransitionTable(
        n_traces=n_traces,
        n_tool_events=n_tool_events,
        generated_at=datetime.now(timezone.utc).isoformat(),
        offset_basis=dict(OFFSET_BASIS),
        synthetic_filter=filter_header,
    )

    def _build(raw_counts: dict, target_type: str) -> dict:
        # Sum denominators per (source, offset)
        denom: dict[tuple[str, int], int] = defaultdict(int)
        for (src, offset, _tgt), cnt in raw_counts.items():
            denom[(src, offset)] += cnt

        result: dict[str, dict[int, list[TransitionEntry]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (src, offset, tgt), cnt in raw_counts.items():
            if cnt < min_count:
                continue
            prob = cnt / denom[(src, offset)]
            result[src][offset].append(TransitionEntry(
                target=tgt,
                target_type=target_type,
                probability=round(prob, 4),
                count=cnt,
                offset=offset,
            ))

        # Sort each bucket by probability descending
        for src in result:
            for offset in result[src]:
                result[src][offset].sort(key=lambda e: e.probability, reverse=True)
        return result

    table.tool_transitions = _build(tool_counts, "tool")
    table.model_transitions = _build(model_counts, "model")
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn tool/model transition probabilities from JSONL traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "traces",
        nargs="+",
        help="JSONL trace files (glob patterns accepted via shell expansion)",
    )
    parser.add_argument(
        "--output", "-o",
        default="runtime/predictor/data/learned_transitions.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=3,
        help="Maximum step lookahead for co-occurrence",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum observation count to include a transition",
    )
    parser.add_argument(
        "--no-filter-synthetic",
        dest="exclude_synthetic",
        action="store_false",
        default=True,
        help=("Count replay-harness traces too (default: exclude them; see "
              "SYNTHETIC_FILTER_RULE).  The choice is recorded in the output "
              "under `synthetic_filter.applied` either way."),
    )
    args = parser.parse_args()

    print(f"[transition_learner] Reading {len(args.traces)} trace file(s)...")
    table = learn_from_traces(args.traces, max_offset=args.max_offset,
                              min_count=args.min_count,
                              exclude_synthetic=args.exclude_synthetic)

    sf = table.synthetic_filter
    print(f"[transition_learner] synthetic filter: "
          f"{'ON (' + str(sf.get('rule')) + ')' if sf.get('applied') else 'OFF'}")
    print(f"[transition_learner]   scanned {sf.get('scanned_traces')} traces: "
          f"{sf.get('kept_traces')} counted, "
          f"{sf.get('out_of_scope_traces')} out of scope (no llm timing), "
          f"{sf.get('excluded_traces')} excluded "
          f"({sf.get('excluded_tool_events')} tool events, "
          f"{sf.get('excluded_llm_events')} llm events)")
    for clause, c in (sf.get("excluded_by_clause") or {}).items():
        print(f"[transition_learner]     {clause}: {c['traces']} traces, "
              f"{c['tool_events']} tool events, {c['llm_events']} llm events")
    print(f"[transition_learner] {table.n_traces} traces, {table.n_tool_events} tool events")
    print(f"[transition_learner] Tool sources: {len(table.tool_transitions)}, "
          f"Model sources: {len(table.model_transitions)}")

    table.save(args.output)
    print(f"[transition_learner] Saved → {args.output}")

    # Pretty-print top transitions
    print("\nTop tool→tool transitions:")
    for src, offset_map in sorted(table.tool_transitions.items()):
        for offset in sorted(offset_map):
            for entry in offset_map[offset][:2]:
                print(f"  {src:45s} +{offset} → {entry.target:40s}  p={entry.probability:.2f}  n={entry.count}")

    print("\nTop model→model transitions:")
    for src, offset_map in sorted(table.model_transitions.items()):
        for offset in sorted(offset_map):
            for entry in offset_map[offset][:2]:
                short_src = src.split("/")[-1]
                short_tgt = entry.target.split("/")[-1]
                print(f"  {short_src:40s} +{offset} → {short_tgt:40s}  p={entry.probability:.2f}  n={entry.count}")


if __name__ == "__main__":
    _main()
