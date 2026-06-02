"""
predictor/transition_learner.py — Learn tool/model co-occurrence statistics from traces.

Reads JSONL workflow traces and computes the empirical probability that tool B
(or model B) follows tool A within k steps.  The output JSON is consumed by
LearnedPredictor to replace hand-coded confidence values with data-driven ones.

Usage (CLI)
-----------
    python -m runtime.predictor.transition_learner \\
        runtime/logs/workflow_traces/runtime_trace_*.jsonl \\
        --output runtime/predictor/data/learned_transitions.json \\
        --max-offset 3 \\
        --min-count 2

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


@dataclass
class TransitionTable:
    """
    Learned co-occurrence table from JSONL traces.

    tool_transitions[source_tool][offset] = [TransitionEntry, ...]  sorted by prob desc
    model_transitions[source_model][offset] = [TransitionEntry, ...] sorted by prob desc
    """
    version: str = "1"
    n_traces: int = 0
    n_tool_events: int = 0
    generated_at: str = ""
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

        t = cls(
            version=data.get("version", "1"),
            n_traces=data.get("n_traces", 0),
            n_tool_events=data.get("n_tool_events", 0),
            generated_at=data.get("generated_at", ""),
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

    For every tool_call (or llm_call/model) event at position i, look forward
    up to max_offset positions and count all (source, offset, target) triples.
    Then normalise counts to probabilities within each (source, offset) bucket.
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

        # Extract (index, kind, name) for tool_call and llm_call events
        sequence: list[tuple[int, str, str]] = []
        for idx, ev in enumerate(events):
            et = ev.get("event_type", "")
            p = ev.get("payload") or {}
            if et == "tool_call":
                tool = p.get("tool", "")
                if tool:
                    sequence.append((idx, "tool", tool))
                    n_tool_events += 1
            elif et == "llm_call":
                model = p.get("model", "")
                if model:
                    sequence.append((idx, "model", model))

        # Count co-occurrences within max_offset steps in the sequence list
        for i, (_, src_kind, src_name) in enumerate(sequence):
            for j in range(i + 1, min(i + 1 + max_offset, len(sequence))):
                _, tgt_kind, tgt_name = sequence[j]
                offset = j - i
                if src_kind == "tool":
                    if tgt_kind == "tool":
                        tool_counts[(src_name, offset, tgt_name)] += 1
                    # cross-type (tool→model) not currently tracked
                elif src_kind == "model":
                    if tgt_kind == "model":
                        model_counts[(src_name, offset, tgt_name)] += 1

    # Normalise counts → probabilities
    table = TransitionTable(
        n_traces=n_traces,
        n_tool_events=n_tool_events,
        generated_at=datetime.now(timezone.utc).isoformat(),
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
