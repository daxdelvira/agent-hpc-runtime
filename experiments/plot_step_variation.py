"""
plot_step_variation.py — Workflow consistency / step-variation plot.

For each step index i, computes the fraction of runs that used the most-common
tool at that step. A value of 100% means every run took the same action at step i;
lower values reveal unpredictability.

Produces a single figure with one line per workflow, saved as both PDF and PNG.

Usage
-----
    python experiments/plot_step_variation.py [options]

Options
-------
    --trace-dir   DIR   directory containing JSONL trace files
                        (default: logs/workflow_traces relative to repo root)
    --outdir      DIR   output directory for figures  (default: figures/)
    --min-runs    N     minimum runs required to plot a step  (default: 3)
    --max-offset  N     maximum step index to show  (default: 20)
    --dark-bg         use dark Gruvbox background instead of white
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the experiments/ directory importable when run directly
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_utils import GRV, FS, apply_gruvbox_rc, save_figure, load_traces, extract_tool_sequence

REPO_ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_consistency(
    tool_sequences: list[list[str]],
    min_runs: int = 3,
    max_step: int | None = None,
) -> dict[int, tuple[float, str, int]]:
    """
    Compute per-step consistency across runs.

    Returns:
        {step_index: (consistency_pct, most_common_tool, n_runs_at_step)}
    """
    step_tools: dict[int, list[str]] = defaultdict(list)
    for seq in tool_sequences:
        for i, tool in enumerate(seq):
            if max_step is None or i <= max_step:
                step_tools[i].append(tool)

    result: dict[int, tuple[float, str, int]] = {}
    for i, tools in step_tools.items():
        if len(tools) < min_runs:
            continue
        ctr = Counter(tools)
        mode_tool, mode_count = ctr.most_common(1)[0]
        result[i] = (mode_count / len(tools) * 100, mode_tool, len(tools))
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

WORKFLOW_STYLE: dict[str, dict] = {
    "chemgraph": {
        "label":    "ChemGraph",
        "color":    GRV["purple"],
        "pattern":  "chemgraph_trace_*.jsonl",
        "linestyle": "-",
        "marker":   "o",
    },
    "atomagents": {
        "label":    "AtomAgents",
        "color":    GRV["orange"],
        "pattern":  "runtime_trace_*.jsonl",
        "linestyle": "--",
        "marker":   "s",
    },
}


def plot_consistency(
    workflows: list[str],
    trace_dir: Path,
    outdir: Path,
    min_runs: int = 3,
    max_offset: int = 20,
    dark_bg: bool = False,
) -> None:
    apply_gruvbox_rc(dark_bg=dark_bg)

    fig, ax = plt.subplots(figsize=(7, 4))

    plotted_any = False
    for wf in workflows:
        style = WORKFLOW_STYLE[wf]
        traces = load_traces(trace_dir, style["pattern"])
        seqs = [extract_tool_sequence(events) for events in traces]
        seqs = [s for s in seqs if s]  # drop empty

        if not seqs:
            print(f"[{wf}] No tool sequences found in {trace_dir}")
            continue

        consistency = compute_consistency(seqs, min_runs=min_runs, max_step=max_offset)
        if not consistency:
            print(f"[{wf}] No steps meet min_runs={min_runs} threshold")
            continue

        steps = sorted(consistency)
        pcts = [consistency[i][0] for i in steps]
        n_runs = [consistency[i][2] for i in steps]

        ax.plot(
            steps,
            pcts,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.8,
            label=f"{style['label']} (n≤{max(n_runs)} runs)",
            zorder=3,
        )
        plotted_any = True

        print(f"[{wf}] {len(seqs)} runs, {len(steps)} plotted steps")
        print(f"       mean consistency: {np.mean(pcts):.1f}%  "
              f"min: {min(pcts):.1f}%  max: {max(pcts):.1f}%")
        for i in steps[:5]:
            pct, tool, n = consistency[i]
            print(f"       step {i:2d}: {pct:5.1f}%  modal='{tool}'  n={n}")

    if not plotted_any:
        print("Nothing to plot.")
        return

    # ── Reference line ────────────────────────────────────────────────────────
    ax.axhline(100, color=GRV["gray"], linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel("Step index", fontsize=FS["label"])
    ax.set_ylabel("% runs sharing most-common tool", fontsize=FS["label"])
    ax.set_title("Workflow Consistency Across Runs", fontsize=FS["title"],
                 fontweight="bold", loc="left")

    ax.set_ylim(0, 108)
    ax.set_xlim(left=-0.3)
    ax.yaxis.set_major_locator(plt.MultipleLocator(25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(12.5))

    grid_color = GRV["bg2"] if dark_bg else "#e8e8e8"
    ax.yaxis.grid(True, color=grid_color, linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(
        fontsize=FS["annot"],
        framealpha=0.0,
        loc="lower left",
    )
    for text in leg.get_texts():
        text.set_color(GRV["fg"] if dark_bg else "#1d2021")

    fig.tight_layout()
    save_figure(fig, outdir, "step_variation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workflow consistency / step-variation plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trace-dir",
        default=str(REPO_ROOT / "logs" / "workflow_traces"),
        help="Directory containing JSONL trace files",
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=list(WORKFLOW_STYLE),
        default=["chemgraph", "atomagents"],
        help="Which workflows to plot",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum number of runs required at a step to include it",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=20,
        help="Maximum step index to show",
    )
    parser.add_argument(
        "--dark-bg",
        action="store_true",
        help="Use dark Gruvbox background",
    )
    args = parser.parse_args()

    plot_consistency(
        workflows=args.workflows,
        trace_dir=Path(args.trace_dir),
        outdir=Path(args.outdir),
        min_runs=args.min_runs,
        max_offset=args.max_offset,
        dark_bg=args.dark_bg,
    )


if __name__ == "__main__":
    main()
