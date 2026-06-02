"""
predictor/plan_extractor.py — Extract an ordered tool sequence from LLM plan text.

When a planner agent writes a numbered/bulleted list of steps, this module
scans the text for known tool names (by word-boundary regex) and returns them
in order of appearance.  The result is passed to predictors as `plan_context`
so they can make lookahead predictions beyond the immediate next step.

Usage
-----
    from runtime.predictor.plan_extractor import extract_plan, KNOWN_TOOLS

    ctx = extract_plan(llm_response_content, KNOWN_TOOLS)
    if ctx:
        print(ctx.tool_sequence)   # e.g. ['computation_task_screw_dislocation', ...]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# All tool names the system may encounter across workflows.
# Add new tools here as workflows expand.
# ---------------------------------------------------------------------------

KNOWN_TOOLS: frozenset[str] = frozenset({
    # AtomAgents outer tools
    "plan_task",
    "computation_task",
    "computation_task_screw_dislocation",
    "computation_task_surface_energy",
    "computation_task_NEB",
    "computation_task_elastic",
    "computation_task_stacking_fault",
    "analyze_screw_core",
    "suggest_orientation",
    # AtomAgents inner tools
    "create_working_folder",
    "create_potential_file",
    "create_crystal",
    "create_screw_dislocation",
    "lattice_constant_simulation",
    "surface_energy_simulation",
    "elastic_constant_simulation",
    "stacking_fault_simulation",
    "NEB_screw_simulation",
    "run_simulation",
    "compute_dislocation_distribution_map",
    "get_DD_map_path",
    # ChemGraph tools
    "molecule_name_to_smiles",
    "smiles_to_coordinate_file",
    "smiles_to_atomsdata",
    "file_to_atomsdata",
    "run_ase",
    "extract_output_json",
    "get_molecule_info",
    "compute_properties",
})


@dataclass
class PlanContext:
    """
    An ordered sequence of tool names extracted from an early LLM response.

    tool_sequence: tools in order of their first mention in the plan text.
    The same tool may appear multiple times (e.g. two computation_task calls);
    adjacent duplicates are collapsed, non-adjacent are preserved.
    """
    tool_sequence: list[str] = field(default_factory=list)
    source: str = "llm_response"           # "llm_response" | "task_description"
    extracted_at_step: int = 0
    n_mentions: int = 0                    # total regex matches before dedup

    def tool_at_offset(self, current_index: int, offset: int) -> str | None:
        """
        Given the current position in the plan, return the tool name expected
        at current_index + offset, or None if out of range.
        """
        target = current_index + offset
        if 0 <= target < len(self.tool_sequence):
            return self.tool_sequence[target]
        return None

    def find_tool(self, tool_name: str) -> int:
        """Return the first index of tool_name in the sequence, or -1."""
        try:
            return self.tool_sequence.index(tool_name)
        except ValueError:
            return -1


def extract_plan(
    content: str,
    known_tools: frozenset[str] = KNOWN_TOOLS,
    min_tools: int = 2,
    step: int = 0,
    source: str = "llm_response",
) -> PlanContext | None:
    """
    Scan LLM text for known tool names in order of character position.

    Returns a PlanContext if at least `min_tools` distinct tool names are
    found; returns None otherwise.

    Strategy:
    1. For each known tool name, find all occurrences via word-boundary regex.
    2. Sort all matches by start position → gives the textual order.
    3. Collapse adjacent duplicates (two consecutive mentions = one step).
    4. Non-adjacent duplicates are preserved (same tool called twice in plan).
    """
    if not content:
        return None

    mentions: list[tuple[int, str]] = []
    for tool in known_tools:
        pattern = r"\b" + re.escape(tool) + r"\b"
        for m in re.finditer(pattern, content):
            mentions.append((m.start(), tool))

    if not mentions:
        return None

    mentions.sort(key=lambda x: x[0])
    n_mentions = len(mentions)

    # Collapse adjacent identical tools
    sequence: list[str] = [mentions[0][1]]
    for _, tool in mentions[1:]:
        if tool != sequence[-1]:
            sequence.append(tool)

    if len(set(sequence)) < min_tools:
        return None

    return PlanContext(
        tool_sequence=sequence,
        source=source,
        extracted_at_step=step,
        n_mentions=n_mentions,
    )
