"""
plot_walltime_comparison.py — End-to-end walltime comparison (Plot 2).

Shows 3 grouped bars per workflow:
  1. Non-agentic / no-AI baseline  (normalized to 1.0×)
  2. Agentic mean  ± std  (N runs)
  3. Best-case agentic  (min of N runs, annotated)

All values are normalized to their workflow's non-agentic baseline.

Data sources (all optional — script warns if missing and skips workflow):
  AtomAgents:
    • results/summary_*.json  with mode=observe_only|baseline  and workflow≠chemgraph_mace
    • mega_mmap_integration/results/stats_dict.csv  (noai-nomega row for non-agentic)
  ChemGraph:
    • results/summary_*.json  with workflow=chemgraph_mace
    • results/noai_chemgraph_*.json  (from chemgraph_noai_runner.py)
  DeepDriveMD:
    • deepdrivemd/all_phase_performance.csv  — sum(duration_s) per pipeline run

Usage
-----
    python experiments/plot_walltime_comparison.py [--outdir DIR] [--dark-bg]

Options
-------
    --results-dir  DIR  summary JSON directory  (default: results/)
    --ddmd-csv     PATH deepdrivemd/all_phase_performance.csv
    --stats-csv    PATH mega_mmap_integration/results/stats_dict.csv
    --outdir       DIR  output figure directory  (default: figures/)
    --min-wall     N    minimum wall_time_s to include a run  (default: 10)
    --dark-bg           use dark Gruvbox background
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_utils import GRV, FS, apply_gruvbox_rc, save_figure

REPO_ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aa_times(
    results_dir: Path,
    stats_csv: Path | None,
    min_wall: float = 10.0,
    max_wall: float = 500.0,
) -> dict[str, list[float]]:
    """
    Returns {'noai': [t,...], 'agentic': [t,...]} for AtomAgents.

    noai: prefers noai_atomagents_*.json (mode=noai_scripted — true no-LLM runs
          from atomagents_noai_runner.py). Falls back to summary_*.json with
          mode=baseline only if no scripted files exist (those still use the LLM).
    agentic: summary_*.json with mode in [real, observe_only] and t < max_wall
             (excludes LAMMPS_SLOWDOWN_S runs which inflate wall time to 4000-6000s)
    """
    noai_times: list[float] = []
    agentic_times: list[float] = []

    # True no-LLM baseline: noai_atomagents_*.json written by atomagents_noai_runner.py
    scripted_noai: list[float] = []
    for f in glob.glob(str(results_dir / "noai_atomagents_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        t = d.get("wall_time_s", 0.0)
        if t >= min_wall:
            scripted_noai.append(t)

    # Agentic times + fallback LLM-baseline from summary_*.json
    llm_baseline_times: list[float] = []
    for f in glob.glob(str(results_dir / "summary_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        wf = d.get("workflow", "")
        if wf == "chemgraph_mace":
            continue
        mode = d.get("mode", "")
        t = d.get("wall_time_s", 0.0)
        if t < min_wall:
            continue
        if mode == "baseline":
            llm_baseline_times.append(t)
        elif mode in ("observe_only", "real", "simulated") and t <= max_wall:
            agentic_times.append(t)

    # Prefer true no-LLM scripted runs; fall back to LLM-baseline if none available
    noai_times = scripted_noai if scripted_noai else llm_baseline_times

    return {"noai": noai_times, "agentic": agentic_times}


def load_cg_times(
    results_dir: Path,
    min_wall: float = 5.0,
) -> dict[str, list[float]]:
    """
    Returns {'noai': [t,...], 'agentic': [t,...]} for ChemGraph.

    Only loads summaries with workflow="chemgraph_mace" or noai_chemgraph_*.json.
    """
    noai_times: list[float] = []
    agentic_times: list[float] = []

    # ChemGraph summary_*.json — only workflow="chemgraph_mace"
    for f in glob.glob(str(results_dir / "summary_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("workflow", "") != "chemgraph_mace":
            continue
        mode = d.get("mode", "")
        t = d.get("wall_time_s", 0.0)
        if t < min_wall:
            continue
        if mode == "baseline":
            noai_times.append(t)
        elif mode in ("observe_only", "simulated", "real"):
            agentic_times.append(t)

    return {"noai": noai_times, "agentic": agentic_times}


def load_ddmd_times(ddmd_csv: Path | None) -> dict[str, list[float]]:
    """
    Returns {'noai': [t,...]} for DeepDriveMD (no agentic version).

    Each complete pipeline cycle = one md_simulation + preceding training phases.
    We use md_simulation durations as a proxy for per-cycle run time (it dominates),
    and add the mean cvae_train and cvae_inference per cycle.
    """
    if not ddmd_csv or not ddmd_csv.exists():
        return {"noai": [], "agentic": []}

    import csv

    phase_times: dict[str, list[float]] = defaultdict(list)
    with open(ddmd_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            phase = row.get("phase", "")
            try:
                phase_times[phase].append(float(row["duration_s"]))
            except (ValueError, KeyError):
                pass

    if not phase_times:
        return {"noai": [], "agentic": []}

    # One pipeline cycle: md_simulation (bottleneck) + fraction of train/inference
    md_times = phase_times.get("md_simulation", [])
    train_times = phase_times.get("cvae_train", [])
    inf_times = phase_times.get("cvae_inference", [])

    if not md_times:
        # Fall back: sum all phase times and divide by number of inferred cycles
        all_t = [t for times in phase_times.values() for t in times]
        total = sum(all_t)
        n_cycles = max(1, len(md_times))
        return {"noai": [total / n_cycles], "agentic": []}

    mean_train = float(np.mean(train_times)) if train_times else 0.0
    mean_inf   = float(np.mean(inf_times))   if inf_times   else 0.0

    # Per-cycle wall time: one md_simulation + amortized training/inference
    cycle_times = [t + mean_train + mean_inf for t in md_times]
    return {"noai": cycle_times, "agentic": []}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

WORKFLOW_ORDER = ["AtomAgents", "ChemGraph", "DeepDriveMD"]

# Colors for the 3 bars
COLOR_NOAI       = GRV["blue"]      # non-agentic reference bar
COLOR_AGENTIC    = GRV["orange"]    # agentic mean bar
COLOR_BEST       = GRV["green"]     # best-case agentic bar


def _bar_group(
    ax: plt.Axes,
    x_center: float,
    noai_t: float,
    agentic_times: list[float],
    bar_width: float,
    label_prefix: str,
    text_color: str,
    dark_bg: bool,
) -> None:
    """Draw one workflow group of 3 bars."""
    edge = GRV["bg2"] if dark_bg else "#aaaaaa"
    gap = bar_width * 0.15

    if not agentic_times or noai_t <= 0:
        return

    agentic_norm   = [t / noai_t for t in agentic_times]
    agentic_mean   = float(np.mean(agentic_norm))
    agentic_std    = float(np.std(agentic_norm))
    best_case_norm = float(min(agentic_norm))
    n_runs         = len(agentic_norm)

    xs = [
        x_center - bar_width - gap,   # non-agentic
        x_center,                       # agentic mean
        x_center + bar_width + gap,     # best-case
    ]
    heights = [1.0, agentic_mean, best_case_norm]
    colors  = [COLOR_NOAI, COLOR_AGENTIC, COLOR_BEST]

    bars = ax.bar(xs, heights, width=bar_width,
                  color=colors, edgecolor=edge, linewidth=0.8, zorder=3)

    # Error bar on agentic mean (clamped so lower cap never goes below 0)
    yerr_lo = min(agentic_std, agentic_mean)  # don't go below zero
    ax.errorbar(
        xs[1], agentic_mean,
        yerr=[[yerr_lo], [agentic_std]],
        fmt="none", color=text_color,
        linewidth=1.2, capsize=4, capthick=1.2,
        zorder=4,
    )

    # Annotation: best-case bar shows "best of N"
    offset = max(heights) * 0.03
    ax.text(xs[2], best_case_norm + offset,
            f"best of\n{n_runs}",
            ha="center", va="bottom",
            fontsize=FS["small"] - 0.5,
            color=text_color, alpha=0.85)

    # Value labels on top of non-agentic and agentic bars
    for xi, hi, lbl in [(xs[0], heights[0], "1.0×"),
                         (xs[1], heights[1], f"{heights[1]:.2f}×")]:
        ax.text(xi, hi + offset * 0.6, lbl,
                ha="center", va="bottom",
                fontsize=FS["small"] - 0.5,
                color=text_color, alpha=0.85)


def plot_walltime_comparison(
    results_dir: Path,
    ddmd_csv: Path | None,
    stats_csv: Path | None,
    outdir: Path,
    min_wall: float = 10.0,
    dark_bg: bool = False,
) -> None:
    apply_gruvbox_rc(dark_bg=dark_bg)

    aa_data   = load_aa_times(results_dir, stats_csv, min_wall)
    cg_data   = load_cg_times(results_dir, min_wall)
    ddmd_data = load_ddmd_times(ddmd_csv)

    print("AtomAgents: noai=", aa_data["noai"], " agentic=", aa_data["agentic"])
    print("ChemGraph:  noai=", cg_data["noai"], " agentic=", cg_data["agentic"])
    print("DeepDriveMD: noai=", ddmd_data["noai"])

    text_color = GRV["fg"] if dark_bg else "#1d2021"
    grid_color = GRV["bg2"] if dark_bg else "#e8e8e8"

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_width = 0.22
    x_positions = {
        "AtomAgents":  0.0,
        "ChemGraph":   1.2,
        "DeepDriveMD": 2.4,
    }

    # ── AtomAgents ──────────────────────────────────────────────────────────
    aa_noai = np.mean(aa_data["noai"]) if aa_data["noai"] else None
    if aa_noai and aa_data["agentic"]:
        _bar_group(ax, x_positions["AtomAgents"], aa_noai,
                   aa_data["agentic"], bar_width, "AA", text_color, dark_bg)
    elif aa_noai:
        print("AtomAgents: no agentic data, showing noai only")

    # ── ChemGraph ───────────────────────────────────────────────────────────
    cg_noai = np.mean(cg_data["noai"]) if cg_data["noai"] else None
    if cg_noai and cg_data["agentic"]:
        _bar_group(ax, x_positions["ChemGraph"], cg_noai,
                   cg_data["agentic"], bar_width, "CG", text_color, dark_bg)
    elif cg_noai:
        # Draw single noai bar as placeholder
        ax.bar(x_positions["ChemGraph"], 1.0, width=bar_width,
               color=COLOR_NOAI,
               edgecolor=GRV["bg2"] if dark_bg else "#aaaaaa",
               linewidth=0.8, zorder=3)
        ax.text(x_positions["ChemGraph"], 1.05, "noai only",
                ha="center", va="bottom", fontsize=FS["small"],
                color=text_color, alpha=0.6, style="italic")

    # ── DeepDriveMD ─────────────────────────────────────────────────────────
    ddmd_noai_times = ddmd_data["noai"]
    if ddmd_noai_times:
        ddmd_noai_mean = float(np.mean(ddmd_noai_times))
        ax.bar(x_positions["DeepDriveMD"] - bar_width / 2, 1.0, width=bar_width,
               color=COLOR_NOAI,
               edgecolor=GRV["bg2"] if dark_bg else "#aaaaaa",
               linewidth=0.8, zorder=3, label="_nolegend_")
        ax.text(x_positions["DeepDriveMD"] - bar_width / 2, 1.05, "1.0×",
                ha="center", va="bottom", fontsize=FS["small"] - 0.5,
                color=text_color, alpha=0.85)
        ax.text(x_positions["DeepDriveMD"] - bar_width / 2, -0.08,
                f"({ddmd_noai_mean:.0f}s)",
                ha="center", va="top", fontsize=FS["small"] - 1,
                color=text_color, alpha=0.6,
                transform=ax.get_xaxis_transform())
        ax.text(x_positions["DeepDriveMD"] + bar_width * 0.5, 0.5,
                "no agentic\nversion",
                ha="center", va="center", fontsize=FS["small"] - 1,
                color=text_color, alpha=0.5, style="italic",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc=GRV["bg1"] if dark_bg else "#f5f5f5",
                          ec="none", alpha=0.7))

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.axhline(1.0, color=GRV["gray"], linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=1, label="Non-agentic baseline")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), fontsize=FS["label"])
    ax.set_ylabel("Normalized wall time (×)", fontsize=FS["label"])
    ax.set_title(
        "Agentic vs. Baseline Wall Time Overhead",
        fontsize=FS["title"], fontweight="bold", loc="left",
    )
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=grid_color, linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 3.0)

    # ── Legend ──────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=COLOR_NOAI,    label="Non-agentic (no LLM)"),
        mpatches.Patch(color=COLOR_AGENTIC, label="Agentic (mean ± std)"),
        mpatches.Patch(color=COLOR_BEST,    label="Agentic best-case"),
        plt.Line2D([0], [0], color=GRV["gray"], linewidth=1.2,
                   linestyle="--", label="1.0× reference"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        fontsize=FS["annot"],
        framealpha=0.0,
        loc="upper right",
    )
    for t in leg.get_texts():
        t.set_color(text_color)

    fig.tight_layout()
    save_figure(fig, outdir, "walltime_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end walltime comparison plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir",
                        default=str(REPO_ROOT / "results"))
    parser.add_argument("--ddmd-csv",
                        default=str(REPO_ROOT.parent / "deepdrivemd" /
                                    "all_phase_performance.csv"))
    parser.add_argument("--stats-csv",
                        default=str(REPO_ROOT.parent / "mega_mmap_integration" /
                                    "results" / "stats_dict.csv"))
    parser.add_argument("--outdir",
                        default=str(REPO_ROOT / "figures"))
    parser.add_argument("--min-wall", type=float, default=10.0,
                        help="Minimum wall_time_s to include a run")
    parser.add_argument("--dark-bg", action="store_true")
    args = parser.parse_args()

    plot_walltime_comparison(
        results_dir=Path(args.results_dir),
        ddmd_csv=Path(args.ddmd_csv),
        stats_csv=Path(args.stats_csv),
        outdir=Path(args.outdir),
        min_wall=args.min_wall,
        dark_bg=args.dark_bg,
    )


if __name__ == "__main__":
    main()
