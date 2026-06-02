"""
predictor/resource_registry.py — Dynamic mapping from consumer tool names to ResourceSpecs.

Instead of hard-coding resource specs inside each predictor, this registry is a
shared, JSON-backed lookup table.  New workflows add entries to
runtime/predictor/data/tool_resources.json without touching Python.

Three construction methods, usable individually or merged:
  from_json()           — load from the JSON file (default path auto-resolved)
  from_mock_predictor() — bootstrap from mock_predictor's existing hand-coded tables
  infer_from_traces()   — heuristic: observe which model was active near each tool_call
                          in existing JSONL traces and infer resource requirements
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from runtime.events import ResourceSpec


_DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "tool_resources.json"


def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _spec_from_dict(d: dict) -> ResourceSpec:
    name = d["name"]
    return ResourceSpec(
        resource_id=_hash(name),
        resource_type=d["resource_type"],
        name=name,
        path=d.get("path"),
        model_endpoint=d.get("model_endpoint"),
        estimated_size_bytes=d.get("estimated_size_bytes"),
        estimated_load_s=d.get("estimated_load_s"),
        confidence=0.0,
        cancellation_safe=d.get("cancellation_safe", True),
        consumer_tool=d.get("consumer_tool", ""),
        consumer_step_offset=d.get("consumer_step_offset", 1),
    )


class ResourceRegistry:
    """
    Maps consumer tool names → list[ResourceSpec].

    Thread-safe for reads (CPython GIL protects dict lookups).
    Populate before the workflow starts; don't mutate during runs.
    """

    def __init__(self) -> None:
        self._map: dict[str, list[ResourceSpec]] = defaultdict(list)

    def register(self, tool_name: str, *specs: ResourceSpec) -> None:
        for spec in specs:
            existing = {s.resource_id for s in self._map[tool_name]}
            if spec.resource_id not in existing:
                self._map[tool_name].append(spec)

    def get(self, tool_name: str) -> list[ResourceSpec]:
        return list(self._map.get(tool_name, []))

    def all_tools(self) -> list[str]:
        return list(self._map.keys())

    def to_dict(self) -> dict[str, list[dict]]:
        return {tool: [asdict(s) for s in specs] for tool, specs in self._map.items()}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "ResourceRegistry":
        """
        Load from a JSON array of tool-resource entries.
        Each entry needs at least: consumer_tool, resource_type, name.
        Entries with any key starting with '_' are treated as comments and skipped.
        """
        path = Path(path) if path else _DEFAULT_DATA_PATH
        registry = cls()
        if not path.exists():
            return registry
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            if not isinstance(entry, dict) or any(k.startswith("_") for k in entry):
                continue
            consumer_tool = entry.get("consumer_tool")
            if not consumer_tool:
                continue
            try:
                registry.register(consumer_tool, _spec_from_dict(entry))
            except (KeyError, TypeError):
                pass
        return registry

    @classmethod
    def from_mock_predictor(cls) -> "ResourceRegistry":
        """
        Bootstrap from mock_predictor's existing hand-coded transition tables.
        Useful as a fallback when no JSON data file exists yet.
        """
        from runtime.predictor.mock_predictor import (
            _ATOMAGENTS_TRANSITIONS,
            _CHEMGRAPH_TRANSITIONS,
        )
        registry = cls()
        for table in (_CHEMGRAPH_TRANSITIONS, _ATOMAGENTS_TRANSITIONS):
            for _trigger_tool, entries in table.items():
                for spec_template, _conf in entries:
                    consumer = spec_template.consumer_tool
                    if consumer:
                        registry.register(consumer, spec_template)
        return registry

    @classmethod
    def infer_from_traces(
        cls,
        jsonl_paths: list[str],
        window: int = 3,
        min_count: int = 2,
    ) -> "ResourceRegistry":
        """
        Heuristic inference: for each tool_call event, look at the preceding
        `window` events for an llm_call with a model name.  If the same model
        is seen near the same tool >= min_count times, record the pairing.

        This supplements the JSON file with empirically observed tool-model
        co-occurrences from real run traces — no manual annotation required.
        """
        _MODEL_SHORT: dict[str, str] = {
            "Qwen/Qwen2.5-VL-72B-Instruct": "qwen_72b",
            "Qwen/Qwen2.5-VL-32B-Instruct": "qwen_32b",
            "Qwen/Qwen2.5-72B-Instruct": "qwen_72b",
            "Qwen/Qwen2.5-32B-Instruct": "qwen_32b",
        }
        _MODEL_ENDPOINT: dict[str, str] = {
            "qwen_72b": "http://localhost:8001",
            "qwen_32b": "http://localhost:8002",
        }
        _MODEL_LOAD_S: dict[str, float] = {
            "qwen_72b": 2700.0,
            "qwen_32b": 1200.0,
        }

        tool_model_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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

            for i, ev in enumerate(events):
                if ev.get("event_type") != "tool_call":
                    continue
                tool = (ev.get("payload") or {}).get("tool", "")
                if not tool:
                    continue
                for prev in reversed(events[max(0, i - window): i]):
                    if prev.get("event_type") == "llm_call":
                        raw_model = (prev.get("payload") or {}).get("model", "")
                        short = _MODEL_SHORT.get(raw_model, "")
                        if short:
                            tool_model_counts[tool][short] += 1
                        break

        registry = cls()
        for tool, model_counts in tool_model_counts.items():
            for model_name, count in model_counts.items():
                if count < min_count:
                    continue
                spec = ResourceSpec(
                    resource_id=_hash(model_name),
                    resource_type="vllm_model",
                    name=model_name,
                    model_endpoint=_MODEL_ENDPOINT.get(model_name),
                    estimated_load_s=_MODEL_LOAD_S.get(model_name),
                    confidence=0.0,
                    cancellation_safe=False,
                    consumer_tool=tool,
                    consumer_step_offset=1,
                )
                registry.register(tool, spec)
        return registry

    @classmethod
    def merged(cls, *registries: "ResourceRegistry") -> "ResourceRegistry":
        """Combine multiple registries; deduplication is by resource_id per tool."""
        result = cls()
        for reg in registries:
            for tool, specs in reg._map.items():
                for spec in specs:
                    result.register(tool, spec)
        return result
