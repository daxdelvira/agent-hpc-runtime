"""
runtime/analysis/plot_prefetch_timeline.py
------------------------------------------
Generate a timeline figure showing prefetch overlap as a percentage of
total workflow duration.  Two-panel layout:

  Top   : Blackwell (local SSD) experimental run — this session's data
  Bottom: PACE NFS projection — scales load/compute to NFS timings

Usage
-----
    # From hardcoded values (works without a trace file):
    python runtime/analysis/plot_prefetch_timeline.py

    # From an actual JSONL trace (auto-extracts epoch timestamps):
    python runtime/analysis/plot_prefetch_timeline.py \
        --trace logs/workflow_traces/runtime_trace_*.jsonl \
        --output results/prefetch_timeline.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ──────────────────────────────────────────────────────────────────────────────
# Data extraction from JSONL trace
# ──────────────────────────────────────────────────────────────────────────────

def _extract_from_trace(trace_path: str) -> dict:
    """
    Parse a runtime JSONL trace and return key timing data.
    Falls back to None values for any missing events.
    """
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    if not events:
        return {}

    t0 = events[0].get("epoch_time", 0)
    total_wall_s = None
    prefetch_start: dict[str, float] = {}   # resource_id → epoch offset
    prefetch_end: dict[str, float] = {}
    tool_calls: list[dict] = []             # {step, tool, t}

    # Pull run_id → wall_time from summary payload if present
    for ev in events:
        et = ev.get("event_type", "")
        p = ev.get("payload", {})
        t = ev.get("epoch_time", 0) - t0

        if et == "prefetch_started":
            rid = p.get("resource_id", p.get("task_id", ""))
            prefetch_start[rid] = t

        elif et == "prefetch_completed":
            tid = p.get("task_id", "")
            prefetch_end[tid] = t

        elif et == "tool_call":
            tool_calls.append({"step": ev.get("step", 0), "tool": p.get("tool", ""), "t": t})

    # Estimate total wall time from last event
    total_wall_s = events[-1].get("epoch_time", 0) - t0

    return {
        "total_wall_s": total_wall_s,
        "t0": t0,
        "prefetch_start": prefetch_start,
        "prefetch_end": prefetch_end,
        "tool_calls": tool_calls,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────────────────

def _blackwell_scenario(trace_data: dict | None) -> dict:
    """
    Blackwell local-SSD experimental run.
    Uses real timing from trace if available; falls back to known values.
    """
    total = 1560.7

    if trace_data and trace_data.get("total_wall_s"):
        total = trace_data["total_wall_s"]

    # Approximate step-level timing from the known workflow structure:
    # Each computation_task takes ~half the workflow.
    # qwen_32b load started at step 1 (~t=15s) and took 155.1s.
    ct1_start = 5 / total
    ct1_end   = 780 / total
    ct2_start = ct1_end
    ct2_end   = 1550 / total

    qwen_start = 15 / total
    qwen_end   = (15 + 155.1) / total     # load took 155.1s

    eam_start  = 20 / total
    eam_end    = (20 + 2.0) / total

    return {
        "label": f"Blackwell (local SSD) — {total:.0f}s total",
        "total_s": total,
        "phases": [
            {"name": "computation_task #1\n(LAMMPS × 2 relaxations)",
             "start": ct1_start, "end": ct1_end, "color": "#4A90D9"},
            {"name": "computation_task #2\n(LAMMPS × 2 relaxations)",
             "start": ct2_start, "end": ct2_end, "color": "#4A90D9"},
            {"name": "analysis / LLM",
             "start": ct2_end, "end": 1.0,  "color": "#7BB3E0"},
        ],
        "prefetches": [
            {"name": "qwen_32b load\n(155s measured)",
             "start": qwen_start, "end": qwen_end,
             "color": "#E8A838", "hatch": ""},
            {"name": "EAM files\n(~2s staged)",
             "start": eam_start, "end": eam_end,
             "color": "#B0C4DE", "hatch": ""},
        ],
        # Overlap = qwen_32b fully hidden inside computation_task #1
        "overlap": {"start": qwen_start, "end": qwen_end,
                    "label": "155s overlap\n(10% of workflow)"},
        "saving_pct": 155.1 / total * 100,
    }


def _pace_nfs_scenario() -> dict:
    """
    Projected PACE NFS run.
    32B load: ~1200s (20 min).  Each computation_task: ~1800s (30 min).
    Total: 2 × 1800s + LLM overhead ≈ 3700s.
    """
    total = 3700.0

    ct1_start = 60 / total
    ct1_end   = 1860 / total
    ct2_start = ct1_end
    ct2_end   = 3660 / total

    qwen_start = 70 / total
    qwen_end   = (70 + 1200) / total   # 1200s NFS load

    eam_start  = 80 / total
    eam_end    = (80 + 30) / total     # ~30s to warm NFS page cache

    return {
        "label": f"PACE NFS (projected) — {total/3600:.1f}h total",
        "total_s": total,
        "phases": [
            {"name": "computation_task #1\n(LAMMPS ~30 min)",
             "start": ct1_start, "end": ct1_end, "color": "#4A90D9"},
            {"name": "computation_task #2\n(LAMMPS ~30 min)",
             "start": ct2_start, "end": ct2_end, "color": "#4A90D9"},
            {"name": "analysis / LLM",
             "start": ct2_end, "end": 1.0, "color": "#7BB3E0"},
        ],
        "prefetches": [
            {"name": "qwen_32b load\n(~1200s, NFS cold)",
             "start": qwen_start, "end": qwen_end,
             "color": "#E8A838", "hatch": ""},
            {"name": "EAM files\n(~30s staged)",
             "start": eam_start, "end": eam_end,
             "color": "#B0C4DE", "hatch": ""},
        ],
        "overlap": {"start": qwen_start, "end": qwen_end,
                    "label": "~1200s overlap\n(32% of workflow)"},
        "saving_pct": 1200 / total * 100,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

def _draw_scenario(ax, scenario: dict, show_xlabel: bool = False) -> None:
    row_h = 0.35
    gap   = 0.15

    # Row y centres
    y_workflow  = 1.0
    y_prefetch  = y_workflow - row_h - gap

    # ── workflow phases ─────────────────────────────────────────────────────
    for ph in scenario["phases"]:
        ax.barh(y_workflow, ph["end"] - ph["start"], left=ph["start"],
                height=row_h, color=ph["color"], edgecolor="white", linewidth=0.5)

    # ── prefetch bars ───────────────────────────────────────────────────────
    for i, pf in enumerate(scenario["prefetches"]):
        y = y_prefetch - i * (row_h + gap * 0.5)
        ax.barh(y, pf["end"] - pf["start"], left=pf["start"],
                height=row_h, color=pf["color"],
                edgecolor="#555", linewidth=0.8,
                hatch=pf.get("hatch", ""))
        # Label inside bar (if wide enough)
        w = pf["end"] - pf["start"]
        if w > 0.04:
            ax.text(pf["start"] + w / 2, y, pf["name"],
                    ha="center", va="center", fontsize=6.5,
                    color="black", fontweight="bold")

    # ── overlap shading ─────────────────────────────────────────────────────
    ov = scenario["overlap"]
    ov_width = ov["end"] - ov["start"]
    ax.axvspan(ov["start"], ov["end"],
               ymin=0.02, ymax=0.98,
               color="#F5A623", alpha=0.22, zorder=0)

    # Vertical dashed boundary lines for overlap window
    for x in (ov["start"], ov["end"]):
        ax.axvline(x, color="#F5A623", lw=0.9, ls="--", alpha=0.7, zorder=1)

    # Annotation arrow from overlap centre pointing down to x-axis
    mid_x = (ov["start"] + ov["end"]) / 2
    ax.annotate(
        ov["label"],
        xy=(mid_x, y_prefetch - row_h * 0.7),
        xytext=(mid_x + max(ov_width, 0.12), y_prefetch - row_h * 1.5),
        ha="left" if mid_x < 0.5 else "right",
        va="top", fontsize=7.5, color="#F5C842",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#F5C842", lw=1.2),
    )

    # ── row labels ──────────────────────────────────────────────────────────
    ax.text(-0.01, y_workflow, "Workflow", ha="right", va="center",
            fontsize=8, color="white",
            transform=ax.get_yaxis_transform())
    ax.text(-0.01, y_prefetch, "Prefetch", ha="right", va="center",
            fontsize=8, color="white",
            transform=ax.get_yaxis_transform())

    # ── 0% / 100% ticks ─────────────────────────────────────────────────────
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_yticks([])
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.set_facecolor("#1A1A2E")

    if show_xlabel:
        ax.set_xlabel("Workflow progress (%)", fontsize=9, color="#CCC")

    # ── scenario title + saving ──────────────────────────────────────────────
    pct = scenario["saving_pct"]
    ax.set_title(
        f"{scenario['label']}   •   "
        f"Estimated saving: {pct:.0f}% of workflow",
        fontsize=9, color="white", loc="left", pad=6,
    )


def make_figure(trace_data: dict | None, output_path: str) -> None:
    scenarios = [
        _blackwell_scenario(trace_data),
        _pace_nfs_scenario(),
    ]

    fig, axes = plt.subplots(
        nrows=len(scenarios), ncols=1,
        figsize=(12, 5.5),
        facecolor="#12122A",
    )

    for i, (ax, sc) in enumerate(zip(axes, scenarios)):
        _draw_scenario(ax, sc, show_xlabel=(i == len(scenarios) - 1))

    # ── shared legend ────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#4A90D9", label="LAMMPS computation (blocking)"),
        mpatches.Patch(color="#7BB3E0", label="LLM reasoning / analysis"),
        mpatches.Patch(color="#E8A838", label="Model prefetch (background)"),
        mpatches.Patch(color="#B0C4DE", label="EAM file staging (background)"),
        mpatches.Patch(color="#F5A623", alpha=0.5, label="Overlap window"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.25,
        facecolor="#2A2A4A",
        edgecolor="#555",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Speculative Model Prefetch: Overlap with LAMMPS Computation",
        fontsize=12, color="white", y=1.01, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _HAVE_MPL:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Plot prefetch overlap timeline as % of workflow"
    )
    parser.add_argument("--trace", default=None,
                        help="JSONL trace file path (optional; uses known values if omitted)")
    parser.add_argument("--output", default="results/prefetch_timeline.png",
                        help="Output image path (default: results/prefetch_timeline.png)")
    args = parser.parse_args()

    trace_data = None
    if args.trace:
        for p in sorted(Path(".").glob(args.trace)) if "*" in args.trace else [Path(args.trace)]:
            if p.exists():
                trace_data = _extract_from_trace(str(p))
                print(f"[plot] Loaded trace: {p}  ({trace_data.get('total_wall_s', '?'):.1f}s)")
                break

    os.makedirs(Path(args.output).parent, exist_ok=True)
    make_figure(trace_data, args.output)


if __name__ == "__main__":
    main()
