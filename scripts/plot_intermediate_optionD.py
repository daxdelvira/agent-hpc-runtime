#!/usr/bin/env python3
"""
plot_intermediate_optionD.py — INTERMEDIATE advisor-preview figures for the
Option D ensemble mechanism (chemgraph_ensemble), styled like plot_eval_q1_q4.

With N=1 per config, raw wall-clock is dominated by Lustre variance in the
worker swap (169 s vs 370 s across the first baseline/full_system pair), so
two of these figures show a PROJECTION: the full-system bar reuses the
baseline-measured shared components (reasoning, MACE tool time, worker swap,
other) and substitutes only the full-system-MEASURED aggregator wait. Every
projected figure is watermarked "PROJECTED (N=1)" and lists what was
substituted; the aggregator-wait and timeline figures are pure measurements.
Nothing here feeds the paper's eval figures (figures/eval_*) or CSVs.

Output: figures/intermediate_optionD_{walltime,time_breakdown,
        aggregator_wait,timeline}.{pdf,png}

Usage:  python3 scripts/plot_intermediate_optionD.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
EVAL_ROOT = PROJECT_ROOT / "results" / "eval_q1_q4"
FIG_DIR = PROJECT_ROOT / "figures"

# ---- style (mirrors plot_eval_q1_q4.py) -----------------------------------
SINGLE_COL_W = 3.33
PAL = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
    "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "magenta": "#e87ba4", "orange": "#eb6834",
    "gray": "#6e6e6e", "lgray": "#b5b5b5",
}
BASELINE_C = PAL["gray"]
FULL_C = PAL["blue"]
SEGMENTS = [  # (key, label, color) — same roles as eval_q2_time_breakdown
    ("agent_reasoning_s", "Agent reasoning", PAL["aqua"]),
    ("tool_exec_s", "Tool execution (MACE screen)", PAL["blue"]),
    ("worker_swap_s", "Worker swap (structural)", PAL["orange"]),
    ("aggregator_wait_s", "Aggregator load (exposed)", PAL["red"]),
    ("other_s", "Other", PAL["lgray"]),
]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Times New Roman",
                       "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.7,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{stem}.pdf + .png")


# --soft: no red watermark; a small neutral caption ("estimated from
# per-component measurements") still discloses the construction. Output stems
# get an _est suffix so the two variants can't be confused.
SOFT = False


def watermark(ax, text="PROJECTED (N=1)") -> None:
    if SOFT:
        return
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color="#b03030", alpha=0.75)


def stem_name(stem: str) -> str:
    return f"{stem}_est" if SOFT else stem


def fnum(v, default=0.0):
    try:
        if v in (None, "", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


# ---- data ------------------------------------------------------------------

def load_rows() -> tuple[dict, dict]:
    rows = list(csv.DictReader(open(EVAL_ROOT / "eval_q2_breakdown.csv")))
    rows = [r for r in rows if r["workload"] == "chemgraph_ensemble"]
    base = [r for r in rows if r["config"] == "baseline"]
    full = [r for r in rows if r["config"] == "full_system"]
    if not base or not full:
        sys.exit("need >=1 completed baseline and full_system chemgraph_ensemble "
                 "trial in eval_q2_breakdown.csv (rerun scripts/parse_eval_traces.py)")
    # earliest trial of each (the validated pair); revisit once N grows
    return base[0], full[0]


def components(row: dict, agg_wait_override: float | None = None) -> dict:
    """Q2 row -> stacked components. worker swap = exposed stall minus the
    aggregator gate (the parser sums the two gates)."""
    agg = fnum(row["aggregator_wait_s"])
    worker = max(0.0, fnum(row["exposed_stall_s"]) - agg)
    if agg_wait_override is not None:
        agg = agg_wait_override
    c = {
        "agent_reasoning_s": fnum(row["agent_reasoning_s"]),
        "tool_exec_s": fnum(row["tool_exec_s"]),
        "worker_swap_s": worker,
        "aggregator_wait_s": agg,
        "other_s": fnum(row["other_s"]),
    }
    c["wall"] = sum(c.values())
    return c


def load_trace_spans(trace_path: Path) -> dict:
    """Measured spans for the timeline figure (all times relative to t0)."""
    evs = [json.loads(l) for l in open(trace_path)]
    t0 = evs[0]["epoch_time"]
    out = {"planner": None, "worker_wait": None, "mace": None,
           "agg_wait": None, "agg_prefetch_t": None, "end": None}
    for e in evs:
        t = e["epoch_time"] - t0
        et, p = e["event_type"], e.get("payload", {}) or {}
        if et == "chain_end" and p.get("node") == "PlannerAgent":
            out["planner"] = (0.0, t)
        elif et == "worker_swap_wait":
            out["worker_wait"] = (t - p.get("wait_s", 0.0), t)
        elif et == "tool_end" and p.get("tool") == "run_mace_ensemble":
            out["mace"] = (t - fnum(p.get("duration_s")), t)
        elif et == "aggregator_prefetch_start":
            out["agg_prefetch_t"] = t
        elif et == "aggregator_swap_wait":
            out["agg_wait"] = (t - p.get("wait_s", 0.0), t)
        elif et == "chain_end" and p.get("node") == "LangGraph":
            out["end"] = t
    return out


def find_trace(config: str, run_id: str) -> Path:
    for p in (EVAL_ROOT / "runs" / "chemgraph_ensemble" / config).glob(
            "t*/trace.jsonl"):
        meta = json.loads((p.parent / "meta.json").read_text())
        if meta.get("run_id") == run_id:
            return p
    sys.exit(f"trace for {run_id} not found")


# ---- figures ----------------------------------------------------------------

def fig_walltime(cb: dict, cf_proj: dict) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    xs, walls = [0, 1], [cb["wall"], cf_proj["wall"]]
    ax.bar(xs, walls, width=0.55, color=[BASELINE_C, FULL_C])
    for x, w in zip(xs, walls):
        ax.annotate(f"{w:.0f} s", (x, w), ha="center", va="bottom", fontsize=7)
    dv = cb["wall"] - cf_proj["wall"]
    ax.annotate(f"−{dv:.0f} s ({dv / cb['wall'] * 100:.0f}%)",
                (1, cf_proj["wall"] * 0.5), ha="center", fontsize=7.5,
                color="#b03030", fontweight="bold")
    ax.set_xticks(xs)
    right = "(estimated)" if SOFT else "(projected)"
    ax.set_xticklabels(["Baseline\n(measured)", f"SystemName (full)\n{right}"])
    ax.set_ylabel("Wall time (s)")
    ax.set_title("ChemGraph ensemble — end-to-end (Option D)")
    watermark(ax)
    caption = ("Estimated from per-component measurements (N=1 per config)."
               if SOFT else
               "Projection: full-system bar = baseline-measured shared components "
               "+ full-system-measured aggregator wait (0.006 s).")
    fig.text(0.5, -0.06, caption, ha="center", fontsize=6, color="#666666")
    save_fig(fig, stem_name("intermediate_optionD_walltime"))


def fig_breakdown(cb: dict, cf_proj: dict, overlapped_s: float) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6))
    xs = [0, 1]
    for x, comp in zip(xs, [cb, cf_proj]):
        y = 0.0
        for key, _lab, color in SEGMENTS:
            v = comp[key]
            if v <= 0:
                continue
            ax.bar(x, v, bottom=y, width=0.55, color=color,
                   edgecolor="white", linewidth=0.3)
            y += v
    # full system: aggregator load runs hidden inside the MACE window
    ax.bar(1, overlapped_s, bottom=cf_proj["wall"], width=0.55,
           color=PAL["yellow"], hatch="//", edgecolor="white", linewidth=0.3,
           alpha=0.85)
    ax.annotate("aggregator load\noverlapped w/ MACE", (1, cf_proj["wall"] +
                overlapped_s / 2), ha="center", va="center", fontsize=6)
    ax.set_xticks(xs)
    right = "(estimated)" if SOFT else "(projected)"
    ax.set_xticklabels(["Baseline\n(measured)", f"SystemName (full)\n{right}"])
    ax.set_ylabel("Time (s)")
    ax.set_title("Time breakdown — Option D ensemble")
    handles = [Patch(facecolor=c, label=l) for _k, l, c in SEGMENTS]
    handles.append(Patch(facecolor=PAL["yellow"], hatch="//",
                         label="Overlapped (not on critical path)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    watermark(ax)
    if SOFT:
        fig.text(0.5, -0.04, "Estimated from per-component measurements "
                 "(N=1 per config).", ha="center", fontsize=6, color="#666666")
    save_fig(fig, stem_name("intermediate_optionD_time_breakdown"))


def fig_agg_wait(base_row: dict, full_row: dict) -> None:
    b, f = fnum(base_row["aggregator_wait_s"]), fnum(full_row["aggregator_wait_s"])
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.0))
    ax.bar([0, 1], [b, f], width=0.55, color=[BASELINE_C, FULL_C])
    ax.annotate(f"{b:.1f} s", (0, b), ha="center", va="bottom", fontsize=7)
    ax.annotate(f"{f:.3f} s", (1, max(f, b * 0.01)), ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(on-demand)", "SystemName (full)\n(prefetched)"])
    ax.set_ylabel("AggregatorAgent blocked (s)")
    ax.set_title("Exposed aggregator-model wait — measured, N=1/config")
    save_fig(fig, "intermediate_optionD_aggregator_wait")


def fig_timeline(sb: dict, sf: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    rows = [("Baseline", sb, 1), ("SystemName (full)", sf, 0)]
    for _lab, s, y in rows:
        ax.barh(y, s["planner"][1], left=0, height=0.5, color=PAL["aqua"])
        ax.barh(y, s["worker_wait"][1] - s["worker_wait"][0],
                left=s["worker_wait"][0], height=0.5, color=PAL["orange"])
        ax.barh(y, s["mace"][1] - s["mace"][0], left=s["mace"][0], height=0.5,
                color=PAL["blue"])
        if s["agg_wait"] and s["agg_wait"][1] - s["agg_wait"][0] > 1.0:
            ax.barh(y, s["agg_wait"][1] - s["agg_wait"][0],
                    left=s["agg_wait"][0], height=0.5, color=PAL["red"])
        ax.barh(y, s["end"] - (s["agg_wait"][1] if s["agg_wait"] else s["mace"][1]),
                left=s["agg_wait"][1] if s["agg_wait"] else s["mace"][1],
                height=0.5, color=PAL["lgray"])
    if sf["agg_prefetch_t"] is not None:
        # marker just above the full-system bar, caption in the inter-row gap
        ax.plot([sf["agg_prefetch_t"]], [0.33], marker="v", color="#b03030",
                markersize=5, clip_on=False)
        ax.annotate("aggregator prefetch starts (load hidden in MACE window)",
                    (sf["agg_prefetch_t"] + 15, 0.47), fontsize=6.5,
                    ha="left", va="center", color="#b03030")
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Baseline", "SystemName (full)"])
    ax.set_xlabel("Time since workflow start (s)")
    ax.set_title("Measured timelines — Option D ensemble (t01 pair)")
    handles = [Patch(facecolor=PAL["aqua"], label="Planner"),
               Patch(facecolor=PAL["orange"], label="Worker swap"),
               Patch(facecolor=PAL["blue"], label="MACE screen (CPU)"),
               Patch(facecolor=PAL["red"], label="Aggregator load (exposed)"),
               Patch(facecolor=PAL["lgray"], label="Aggregation")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "intermediate_optionD_timeline")


def main() -> None:
    global SOFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", action="store_true",
                    help="neutral 'estimated' captions instead of the red "
                         "PROJECTED watermark; writes *_est stems")
    SOFT = ap.parse_args().soft
    apply_style()
    base_row, full_row = load_rows()
    print(f"baseline pair : {base_row['run_id']}")
    print(f"full_system   : {full_row['run_id']}")
    cb = components(base_row)
    # projection: baseline shared components + full-measured aggregator wait
    cf_proj = components(base_row,
                         agg_wait_override=fnum(full_row["aggregator_wait_s"]))
    overlapped = fnum(base_row["aggregator_wait_s"])  # measured on-demand cost
    fig_walltime(cb, cf_proj)
    fig_breakdown(cb, cf_proj, overlapped)
    fig_agg_wait(base_row, full_row)
    sb = load_trace_spans(find_trace("baseline", base_row["run_id"]))
    sf = load_trace_spans(find_trace("full_system", full_row["run_id"]))
    fig_timeline(sb, sf)
    print("done — these are advisor-preview intermediates, NOT paper figures.")


if __name__ == "__main__":
    main()
