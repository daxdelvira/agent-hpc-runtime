"""
plot_plan_accuracy.py — Plan compliance visualization (AtomAgents and ChemGraph).

Two-panel figure:
  Panel A — AtomAgents plan compliance heatmap.
    Rows = runs, columns = steps in the plan extracted by the runtime.
    Green  = planned tool appeared in the actual execution (ordered subsequence).
    Gray   = planned tool not reached (run ended early or tool skipped).
    N/A    = this run's plan had fewer steps than the common maximum.
    Data source: plan_extracted + tool_call events from runtime_trace_*.jsonl.

  Panel B — ChemGraph plan compliance heatmap.
    Same style; plan is the canonical 3-step sequence from the task prompt.
    (ChemGraph Experiment 2 with an LLM planner is in progress — this panel
    will be updated once that data is available.)

Usage
-----
    python experiments/plot_plan_accuracy.py [--trace-dir DIR] [--outdir DIR]

Options
-------
    --trace-dir  DIR  (default: logs/workflow_traces)
    --outdir     DIR  (default: figures/)
    --dark-bg        use dark Gruvbox background
    --min-runs   N   minimum runs required per step to include it (default: 3)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_utils import (
    GRV, FS, apply_gruvbox_rc, save_figure,
    load_traces, extract_tool_sequence,
)

REPO_ROOT = _HERE.parent

# Short labels for ChemGraph tool names
CG_TOOL_SHORT_LABELS = {
    "molecule_name_to_smiles":  "name→smiles",
    "smiles_to_coordinate_file": "smiles→coords",
    "smiles_to_atomsdata":      "smiles→atoms",
    "file_to_atomsdata":        "file→atoms",
    "run_ase":                  "run_ase",
    "extract_output_json":      "extract_json",
}

# Short labels for the common AtomAgents 6-step plan
AA_PLAN_SHORT_LABELS = {
    "create_working_folder":              "mk_folder",
    "create_potential_file":              "mk_potential",
    "suggest_orientation":                "orient",
    "create_screw_dislocation":           "mk_dislo",
    "run_simulation":                     "run_sim",
    "compute_dislocation_distribution_map": "DD_map",
}


# ---------------------------------------------------------------------------
# AtomAgents: plan compliance from plan_extracted + tool_call events
# ---------------------------------------------------------------------------

def _ordered_subsequence_match(plan: list[str], actual: list[str]) -> list[bool]:
    """True for each plan step where the tool appears in actual in order.

    Lenient: each planned tool is matched against the first occurrence in
    actual starting from the previous match. Retries (repeated calls to the
    same tool) do not consume extra plan slots.
    """
    matched = []
    pos = -1
    for pt in plan:
        found = False
        for i in range(pos + 1, len(actual)):
            if actual[i] == pt:
                pos = i
                found = True
                break
        matched.append(found)
    return matched


def _strict_match(plan: list[str], actual: list[str]) -> list[bool]:
    """Strict plan compliance: retries consume plan slots.

    Filter actual to tools that appear in the plan, then match positionally.
    A retry (repeated call to the same planned tool) occupies the next plan
    slot, penalizing subsequent steps. Tools not in the plan are ignored so
    that preamble / cleanup steps don't unfairly break compliance.
    """
    plan_set = set(plan)
    actual_filtered = [t for t in actual if t in plan_set]
    matched = []
    for i, pt in enumerate(plan):
        matched.append(i < len(actual_filtered) and actual_filtered[i] == pt)
    return matched


def compute_aa_compliance(
    traces: list[list[dict]],
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    """
    Build a plan compliance matrix from traces that have plan_extracted events.

    Returns:
        matrix   — float array (n_runs, max_plan_len)
                   1.0 = planned tool found in actual execution (ordered)
                   0.0 = planned tool not reached / run ended early
                  -1.0 = N/A (this run's plan shorter than max_plan_len)
        plans    — extracted plan per run (list of tool name lists)
        col_labels — short label for each plan column (from the most common plan)
    """
    run_data: list[tuple[list[str], list[str]]] = []  # (plan, actual)

    for events in traces:
        plan = None
        actual = []
        for e in events:
            et = e.get("event_type", "")
            if et == "plan_extracted":
                plan = e["payload"].get("tool_sequence", [])
            elif et == "tool_call":
                actual.append(e["payload"].get("tool", ""))
        if plan and actual:
            run_data.append((plan, actual))

    if not run_data:
        return np.zeros((0, 0)), [], []

    max_len = max(len(p) for p, _ in run_data)

    # Build column labels from the longest plan (most informative)
    from collections import Counter
    common_plan = Counter(tuple(p) for p, _ in run_data).most_common(1)[0][0]
    longest_plan = max((list(p) for p, _ in run_data), key=len)
    # common_plan drives early columns; longest_plan fills extras
    label_src = list(common_plan) + longest_plan[len(common_plan):]
    lmap = label_map if label_map is not None else AA_PLAN_SHORT_LABELS
    col_labels = [lmap.get(t, t[:12]) for t in label_src[:max_len]]
    while len(col_labels) < max_len:
        col_labels.append(f"step{len(col_labels)}")

    match_fn = _strict_match if strict else _ordered_subsequence_match
    M = np.full((len(run_data), max_len), -1.0)  # -1 = N/A
    for r, (plan, actual) in enumerate(run_data):
        matched = match_fn(plan, actual)
        for p, hit in enumerate(matched):
            M[r, p] = 1.0 if hit else 0.0

    return M, [p for p, _ in run_data], col_labels




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_plan_accuracy(
    trace_dir: Path,
    outdir: Path,
    strict: bool = False,
    dark_bg: bool = False,
    min_runs: int = 3,
) -> None:
    apply_gruvbox_rc(dark_bg=dark_bg)

    aa_traces = load_traces(trace_dir, "runtime_trace_*.jsonl")
    cg_traces_all = load_traces(trace_dir, "chemgraph_trace_*.jsonl")
    # Panel B: multi-agent traces with plan_extracted events (Exp 2)
    cg_traces = [
        t for t in cg_traces_all
        if any(e.get("event_type") == "plan_extracted" for e in t)
    ]

    aa_matrix, aa_plans, aa_col_labels = compute_aa_compliance(aa_traces, strict=strict)
    cg_matrix, cg_plans, cg_col_labels = compute_aa_compliance(
        cg_traces, label_map=CG_TOOL_SHORT_LABELS, strict=strict
    )

    text_color = GRV["fg"] if dark_bg else "#1d2021"
    grid_color = GRV["bg1"] if dark_bg else "#e0e0e0"

    aa_n_runs = aa_matrix.shape[0] if aa_matrix.ndim == 2 else 0
    cg_n_runs = cg_matrix.shape[0] if cg_matrix.ndim == 2 else 0

    fig = plt.figure(figsize=(11, 4.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.38)
    ax_aa = fig.add_subplot(gs[0])
    ax_cg = fig.add_subplot(gs[1])

    def _step_pcts(M):
        """Per-step (n_match, n_valid, pct) tuples."""
        result = []
        for p in range(M.shape[1]):
            col = M[:, p]
            n_v = int((col >= 0).sum())
            n_m = int((col == 1.0).sum())
            result.append((n_m, n_v, n_m / n_v * 100 if n_v > 0 else 0.0))
        return result

    def _draw_bar_panel(ax, M, col_labels, bar_color, title, n_runs):
        if M.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=text_color)
            ax.set_title(title, fontsize=FS["title"], fontweight="bold", loc="left")
            return

        stats = _step_pcts(M)
        xs = np.arange(len(stats))
        pcts = [s[2] for s in stats]

        bars = ax.bar(xs, pcts, color=bar_color, width=0.6, zorder=3,
                      linewidth=0, alpha=0.88)

        # "N/M" count label above each bar
        for i, (n_m, n_v, pct) in enumerate(stats):
            label_y = pct + 1.5
            ax.text(i, min(label_y, 97), f"{n_m}/{n_v}",
                    ha="center", va="bottom",
                    fontsize=FS["small"], color=text_color)

        ax.axhline(100, color=grid_color, linewidth=0.8, linestyle="--", zorder=2)
        ax.set_ylim(0, 108)
        ax.set_xticks(xs)
        ax.set_xticklabels(col_labels, fontsize=FS["tick"], rotation=25, ha="right")
        ax.set_ylabel("% runs compliant", fontsize=FS["label"])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(axis="both", length=2)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=grid_color, linewidth=0.6)

        # Overall summary in top-left
        n_valid_all = int((M >= 0).sum())
        n_match_all = int((M == 1.0).sum())
        overall = n_match_all / n_valid_all * 100 if n_valid_all > 0 else 0
        n_full = sum(
            all(M[r, p] == 1.0 for p in range(M.shape[1]) if M[r, p] >= 0)
            for r in range(n_runs)
        )
        ax.text(0.02, 0.97,
                f"Overall {overall:.0f}%  •  {n_full}/{n_runs} fully compliant",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FS["annot"], color=text_color)

        ax.set_title(title, fontsize=FS["title"], fontweight="bold", loc="left")

    mode_tag = " (strict — retries penalized)" if strict else ""
    _draw_bar_panel(ax_aa, aa_matrix, aa_col_labels,
                    bar_color=GRV["blue"],
                    title=f"AtomAgents: Plan Compliance{mode_tag}",
                    n_runs=aa_n_runs)

    _draw_bar_panel(ax_cg, cg_matrix, cg_col_labels,
                    bar_color=GRV["purple"],
                    title=f"ChemGraph: Plan Compliance{mode_tag}\n(Exp 2 — LLM planner)",
                    n_runs=cg_n_runs)

    fig.tight_layout()
    stem = "plan_accuracy_strict" if strict else "plan_accuracy"
    save_figure(fig, outdir, stem)
    plt.close(fig)

    # ── Print summary ─────────────────────────────────────────────────────────
    def _print_summary(label, M, col_labels, n_runs):
        if M.size == 0:
            return
        n_v_all = int((M >= 0).sum())
        n_m_all = int((M == 1.0).sum())
        overall = n_m_all / n_v_all * 100 if n_v_all > 0 else 0
        n_full = sum(
            all(M[r, p] == 1.0 for p in range(M.shape[1]) if M[r, p] >= 0)
            for r in range(n_runs)
        )
        print(f"{label}: {n_runs} runs, {n_full} fully compliant, overall {overall:.1f}%")
        for p, lbl in enumerate(col_labels):
            col = M[:, p]
            n_v = int((col >= 0).sum())
            n_m = int((col == 1.0).sum())
            print(f"  step {p} ({lbl}): {n_m}/{n_v} = {n_m/n_v*100:.0f}%")

    _print_summary("AtomAgents", aa_matrix, aa_col_labels, aa_n_runs)
    _print_summary("ChemGraph (Exp 2 LLM planner)", cg_matrix, cg_col_labels, cg_n_runs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan accuracy visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument(
        "--strict", action="store_true",
        help="Penalize retries: each actual call consumes a plan slot one-to-one",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="(unused, kept for backward compat)",
    )
    args = parser.parse_args()

    plot_plan_accuracy(
        trace_dir=Path(args.trace_dir),
        outdir=Path(args.outdir),
        dark_bg=args.dark_bg,
        strict=args.strict,
        min_runs=args.min_runs,
    )


if __name__ == "__main__":
    main()
