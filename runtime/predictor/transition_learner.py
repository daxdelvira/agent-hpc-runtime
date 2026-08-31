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
# 2026-08-30 when the offset basis was fixed; see OFFSET_BASIS below.
TABLE_VERSION = "2"

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
    89% of the tool_call events under logs/workflow_traces/runtime_trace_*.jsonl
    (5310 of 5977) come from traces whose median inter-llm_call gap is under a
    second, i.e. mock/simulated harness runs rather than real ones.  The single
    largest bucket in the shipped table, `create_potential_file +2 ->
    create_potential_file`, is 4568/4628 mock, and 4432 of those come from just
    two run_ids (observe2-20260526, observe2-20260527).  Nothing here filters
    them out.  Weigh AtomAgents tool rows accordingly.
    """
    # counts[(source, offset, target)] = int
    tool_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    model_counts: dict[tuple[str, int, str], int] = defaultdict(int)

    n_traces = 0
    n_tool_events = 0

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
    table = TransitionTable(
        n_traces=n_traces,
        n_tool_events=n_tool_events,
        generated_at=datetime.now(timezone.utc).isoformat(),
        offset_basis=dict(OFFSET_BASIS),
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
    args = parser.parse_args()

    print(f"[transition_learner] Reading {len(args.traces)} trace file(s)...")
    table = learn_from_traces(args.traces, max_offset=args.max_offset, min_count=args.min_count)

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
