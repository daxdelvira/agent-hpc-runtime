"""
plot_tool_transitions.py — Tool transition / correlation heatmaps.

For each offset i in [1..max_offset], computes the probability matrix
P(tool_B at step j+i | tool_A at step j) across all runs of a workflow,
then saves a heatmap.  Also produces a combined multi-panel figure.

Usage
-----
    python experiments/plot_tool_transitions.py --workflow chemgraph
    python experiments/plot_tool_transitions.py --workflow atomagents --max-offset 5
    python experiments/plot_tool_transitions.py --workflow chemgraph --workflow atomagents

Options
-------
    --workflow      chemgraph | atomagents  (repeatable; default: both)
    --max-offset    N   maximum lag to compute  (default: 5)
    --min-count     N   minimum source-tool count to include a row  (default: 2)
    --trace-dir     DIR  (default: logs/workflow_traces)
    --outdir        DIR  (default: figures/)
    --dark-bg           use dark Gruvbox background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import seaborn as sns
    _HAVE_SNS = True
except ImportError:
    _HAVE_SNS = False

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_utils import (
    GRV, FS, apply_gruvbox_rc, save_figure,
    load_traces, extract_tool_sequence,
)

REPO_ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# Tool label shortening (keeps heatmap readable)
# ---------------------------------------------------------------------------

_SHORTEN: dict[str, str] = {
    "molecule_name_to_smiles":         "name→smiles",
    "smiles_to_coordinate_file":       "smiles→coords",
    "run_ase":                         "run_ase",
    "extract_output_json":             "extract_json",
    "file_to_atomsdata":               "file→atoms",
    "calculator":                      "calculator",
    "plan_task":                       "plan_task",
    "computation_task":                "comp_task",
    "computation_task_screw_dislocation": "comp_screw",
    "computation_task_surface_energy": "comp_surf_e",
    "computation_task_NEB":            "comp_NEB",
    "analyze_screw_core":              "analyze_core",
    "create_working_folder":           "create_folder",
    "create_potential_file":           "create_pot",
    "create_crystal":                  "create_crystal",
    "create_screw_dislocation":        "create_screw",
    "suggest_orientation":             "suggest_orient",
    "run_simulation":                  "run_sim",
    "NEB_screw_simulation":            "NEB_sim",
    "compute_dislocation_distribution_map": "DD_map",
    "get_DD_map_path":                 "get_DD_path",
    "analyze_plot":                    "analyze_plot",
    "save_csv_data":                   "save_csv",
    "save_image_data":                 "save_img",
    "generate_visualizations":         "gen_viz",
    "get_computation_results":         "get_results",
    "retrieve_atomic_positions":       "get_positions",
    "code_task":                       "code_task",
    "execute_task":                    "exec_task",
    "print":                           "print",
}


def shorten(name: str) -> str:
    return _SHORTEN.get(name, name[:16])


# ---------------------------------------------------------------------------
# Transition-matrix computation
# ---------------------------------------------------------------------------

def compute_transition_matrix(
    tool_sequences: list[list[str]],
    offset: int = 1,
    min_count: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute row-normalised transition matrix at *offset* steps.

    Returns:
        matrix  — shape (n_tools, n_tools), rows sum to 1 (or 0 if no data)
        labels  — list of tool names corresponding to rows/cols
    """
    from collections import Counter

    pair_counts: Counter = Counter()
    source_counts: Counter = Counter()

    for seq in tool_sequences:
        for j in range(len(seq) - offset):
            src, dst = seq[j], seq[j + offset]
            pair_counts[(src, dst)] += 1
            source_counts[src] += 1

    # Filter to tools that appear as source at least min_count times
    tools = sorted(t for t, c in source_counts.items() if c >= min_count)
    # Also include any destination tool that appears enough as source
    dst_tools = sorted({d for (s, d) in pair_counts if source_counts.get(d, 0) >= min_count})
    # Union: keep tools that appear in either role with enough data
    all_tools = sorted(set(tools) | set(dst_tools))

    n = len(all_tools)
    idx = {t: i for i, t in enumerate(all_tools)}
    M = np.zeros((n, n), dtype=float)

    for (src, dst), cnt in pair_counts.items():
        if src in idx and dst in idx:
            M[idx[src], idx[dst]] += cnt

    # Row-normalise (safe: avoid RuntimeWarning on zero rows)
    row_sums = M.sum(axis=1, keepdims=True)
    M = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)

    return M, all_tools


# ---------------------------------------------------------------------------
# Single heatmap
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax: plt.Axes,
    M: np.ndarray,
    labels: list[str],
    title: str,
    dark_bg: bool,
    annot: bool = True,
    fmt: str = ".2f",
    title_fs: float | None = None,
) -> None:
    short_labels = [shorten(t) for t in labels]

    if _HAVE_SNS:
        cmap = "YlOrRd"
        sns.heatmap(
            M,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            annot=annot and len(labels) <= 12,
            fmt=fmt,
            annot_kws={"size": max(10, FS["small"] - len(labels) * 0.2)},
            xticklabels=short_labels,
            yticklabels=short_labels,
            linewidths=0.3,
            linecolor=GRV["bg2"] if dark_bg else "#dddddd",
            cbar_kws={"shrink": 0.8},
        )
    else:
        im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right",
                           fontsize=max(11, FS["small"] - len(labels) * 0.15))
        ax.set_yticklabels(short_labels,
                           fontsize=max(11, FS["small"] - len(labels) * 0.15))
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_title(title, fontsize=title_fs if title_fs is not None else FS["annot"],
                 fontweight="bold", pad=6)
    ax.set_xlabel("Destination tool (step j+k)", fontsize=FS["small"])
    ax.set_ylabel("Source tool (step j)", fontsize=FS["small"])
    ax.tick_params(axis="x", rotation=45, labelsize=max(11, FS["small"] - len(labels) * 0.15))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(11, FS["small"] - len(labels) * 0.15))


# ---------------------------------------------------------------------------
# Per-workflow plot routine
# ---------------------------------------------------------------------------

WORKFLOW_PATTERN = {
    "chemgraph":  "chemgraph_trace_*.jsonl",
    "atomagents": "runtime_trace_*.jsonl",
}


def plot_transitions(
    workflow: str,
    trace_dir: Path,
    outdir: Path,
    max_offset: int = 5,
    min_count: int = 2,
    dark_bg: bool = False,
) -> None:
    pattern = WORKFLOW_PATTERN[workflow]
    traces = load_traces(trace_dir, pattern)
    seqs = [extract_tool_sequence(events) for events in traces]
    seqs = [s for s in seqs if len(s) >= 2]

    if not seqs:
        print(f"[{workflow}] No usable tool sequences found.")
        return

    print(f"[{workflow}] {len(seqs)} runs, "
          f"mean seq length {sum(len(s) for s in seqs)/len(seqs):.1f}")

    apply_gruvbox_rc(dark_bg=dark_bg)

    # ── Individual heatmaps (one PDF per offset) ───────────────────────────
    for i in range(1, max_offset + 1):
        M, labels = compute_transition_matrix(seqs, offset=i, min_count=min_count)
        if M.size == 0:
            print(f"  offset {i}: no data")
            continue

        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7 + 1),
                                        max(4, len(labels) * 0.6 + 1)))
        _draw_heatmap(ax, M, labels,
                      title=f"Tool Transitions — {workflow.capitalize()} (offset k={i})",
                      dark_bg=dark_bg,
                      title_fs=FS["title"])
        fig.tight_layout()
        save_figure(fig, outdir, f"transitions_{workflow}_i{i:02d}")
        plt.close(fig)
        print(f"  offset {i}: {len(labels)}×{len(labels)} matrix")

    # ── Combined multi-panel figure ─────────────────────────────────────────
    ncols = min(max_offset, 3)
    nrows = (max_offset + ncols - 1) // ncols

    # Pre-compute to find common label set for panel sizing
    matrices = []
    for i in range(1, max_offset + 1):
        M, labels = compute_transition_matrix(seqs, offset=i, min_count=min_count)
        matrices.append((i, M, labels))

    max_n = max((len(lbl) for _, _, lbl in matrices), default=1)
    cell_w = max(4.2, max_n * 0.55)
    cell_h = max(3.6, max_n * 0.5)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_w, nrows * cell_h),
        squeeze=False,
    )
    fig.suptitle(
        f"Tool Transition Matrices — {workflow.capitalize()}",
        fontsize=FS["title"],
        fontweight="bold",
        y=1.01,
    )

    # Offset label a little larger than the rest of the panel text.
    offset_fs = FS["title"] + 4

    ax_flat = axes.flatten()
    for idx, (i, M, labels) in enumerate(matrices):
        _draw_heatmap(
            ax_flat[idx], M, labels,
            title=f"offset k={i}",
            dark_bg=dark_bg,
            annot=len(labels) <= 10,
            title_fs=offset_fs,
        )

    # Hide unused axes
    for idx in range(len(matrices), len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.tight_layout()
    save_figure(fig, outdir, f"transitions_{workflow}_combined")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool transition / correlation heatmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workflow",
        action="append",
        dest="workflows",
        choices=list(WORKFLOW_PATTERN),
        default=None,
        help="Workflow(s) to process (can be specified multiple times)",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=5,
        help="Maximum lag offset to compute",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum source-tool occurrence count to include a row",
    )
    parser.add_argument(
        "--trace-dir",
        default=str(REPO_ROOT / "logs" / "workflow_traces"),
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "figures"),
    )
    parser.add_argument("--dark-bg", action="store_true")
    args = parser.parse_args()

    workflows = args.workflows or list(WORKFLOW_PATTERN)
    for wf in workflows:
        plot_transitions(
            workflow=wf,
            trace_dir=Path(args.trace_dir),
            outdir=Path(args.outdir),
            max_offset=args.max_offset,
            min_count=args.min_count,
            dark_bg=args.dark_bg,
        )


if __name__ == "__main__":
    main()
