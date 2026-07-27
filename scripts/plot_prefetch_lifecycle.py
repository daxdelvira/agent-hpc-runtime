#!/usr/bin/env python3
"""
plot_prefetch_lifecycle.py — prefetch-lifecycle figures for the paper.

Input : results/eval_q1_q4/eval_prefetch_lifecycle.csv
        results/eval_q1_q4/eval_stall_taxonomy.csv
        (both written by scripts/extract_prefetch_lifecycle.py; see its
        docstring for column semantics)
Output: figures/lifecycle/
        gantt_<workload>_<config>_<trial_dir>.{pdf,png}
          per-trial prefetch lifecycle Gantt: one lane per prefetch object,
          bar = transfer (t_started..t_completed) colored by outcome, red
          hatched overlay = exposed stall on the critical path, ▼ = first
          need, ○ = prediction time.
        size_vs_window_<workload>.{pdf,png}
          object size vs prediction window on log-log axes with the
          pure-I/O and engine-adjusted hideability frontiers.
        stall_taxonomy.{pdf,png}
          stacked "where does the time go" bars (stall seconds per trial by
          stall_class) for the headline workloads.

Style: matplotlib only (no seaborn), white background, serif fonts sized
for ACM column widths.  Colorblind-safe palette (validated categorical
order; outcomes carry marker-shape / hatch secondary encoding so color is
never the only channel).  Every figure is saved as PDF + PNG (dpi=200).

Usage
-----
    python scripts/plot_prefetch_lifecycle.py                 # everything
    python scripts/plot_prefetch_lifecycle.py --all-defaults  # 4 default Gantts
    python scripts/plot_prefetch_lifecycle.py --trial t02__20260707-202153
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "results" / "eval_q1_q4"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "lifecycle"

SINGLE_COL_W = 3.33   # inches (ACM single column)
DOUBLE_COL_W = 7.00

# Validated colorblind-safe categorical palette (same hex set as
# scripts/plot_eval_q1_q4.py; assignment fixed per role, never cycled).
PAL = {
    "blue":    "#2a78d6",
    "orange":  "#eb6834",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "magenta": "#e87ba4",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "gray":    "#9a9a9a",
    "lgray":   "#c9c9c9",
}

# Outcome → (label, facecolor).  Lateness is also encoded geometrically in
# the Gantt (bar end vs need marker) and by marker shape in the scatter, so
# color never carries the distinction alone.
OUTCOME_FACE = {
    "useful": ("Useful prefetch",        PAL["green"]),
    "late":   ("Late prefetch",          PAL["orange"]),
    "wasted": ("Wasted (never needed)",  PAL["gray"]),
}
NO_BAR_OUTCOMES = {"no_prediction", "no_prefetch_config"}   # red outline only

SCATTER_MARKER = {"useful": "o", "late": "^", "wasted": "x"}

# stall_class → (label, color, hatch).  Stack order is the list order;
# stack-adjacent hues follow the validated adjacent palette order, and each
# segment additionally carries a hatch as the secondary (CVD/print) channel.
STALL_CLASSES = [
    ("no_prediction",     "No prediction",      PAL["red"],     ""),
    ("no_window",         "No window (w<15 s)", PAL["blue"],    "//"),
    ("window_too_small",  "Window too small",   PAL["orange"],  ""),
    ("late_start",        "Late start",         PAL["aqua"],    "\\\\"),
    ("residual_partial",  "Residual (partial)", PAL["yellow"],  ".."),
    ("policy_skip",       "Policy skip",        PAL["magenta"], "xx"),
    ("unattributed",      "Unattributed",       PAL["violet"],  ""),
    ("baseline_no_prefetch", "No prefetch (reference)", PAL["gray"], ""),
]
STALL_STYLE = {k: (lab, col, hat) for k, lab, col, hat in
               [(c[0], c[1], c[2], c[3]) for c in STALL_CLASSES]}

STAGING_GBPS = 2.78          # measured staging bandwidth, GB/s
ENGINE_SPINUP_S = 120.0      # vLLM engine bring-up floor, seconds
EPS_WINDOW = 0.012           # log-scale bucket for null/nonpositive windows

TYPE_ABBREV = {
    "model_cache": "cache",
    "vllm_model":  "engine",
    "mace_model":  "mace",
    "data_file":   "file",
    "":            "obj",
}

GANTT_DEFAULTS = [
    ("chemgraph_swap",     "full_system"),
    ("chemgraph_ensemble", "full_system"),
    ("chemgraph_screen",   "full_system"),
    ("chemgraph_screen",   "baseline"),
]

TAXONOMY_WORKLOADS = ["chemgraph_swap", "chemgraph_ensemble",
                      "chemgraph_screen", "atomagents_exp3"]
TAXONOMY_CONFIGS = ["baseline", "full_system", "naive_prefetch"]
CONFIG_SHORT = {"baseline": "Baseline", "full_system": "SystemName",
                "naive_prefetch": "Naïve pf."}
WORKLOAD_LABEL = {
    "chemgraph_swap":     "ChemGraph swap",
    "chemgraph_ensemble": "ChemGraph ensemble",
    "chemgraph_screen":   "ChemGraph screen",
    "atomagents_exp3":    "AtomAgents (Exp3)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "font.family":       "serif",
        "font.serif":        ["Nimbus Roman", "Times New Roman", "Times",
                              "DejaVu Serif"],
        "mathtext.fontset":  "dejavuserif",
        "font.size":         8,
        "axes.titlesize":    8.5,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#444444",
        "axes.linewidth":    0.7,
        "xtick.color":       "#444444",
        "ytick.color":       "#444444",
        "grid.color":        "#dddddd",
        "grid.linewidth":    0.5,
        "legend.frameon":    False,
        "hatch.linewidth":   0.6,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / stem}.pdf + .png")


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v, default=None):
    try:
        if v in (None, "", "None", "unknown"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.0f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    return f"{b / 1e3:.0f} kB"


def lane_label(r: dict) -> str:
    name = r["resource_name"]
    if name.startswith("qwen_"):
        name = name[len("qwen_"):]
    name = name.replace("_", "-")
    return f"{name} [{TYPE_ABBREV.get(r['resource_type'], r['resource_type'])}]"


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------

def trial_exposure(rows: list[dict]) -> float:
    """Trial stall total with gate_group dedup (max within group, per docstring)."""
    groups: dict[str, float] = defaultdict(float)
    singles = 0.0
    for r in rows:
        e = fnum(r["exposure_s"], 0.0) or 0.0
        g = r.get("gate_group") or ""
        if g:
            groups[g] = max(groups[g], e)
        else:
            singles += e
    return singles + sum(groups.values())


def representative_trial(rows: list[dict], workload: str,
                         config: str) -> tuple[str, list[dict]] | None:
    trials: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["workload"] == workload and r["config"] == config:
            trials[r["trial_dir"]].append(r)
    if not trials:
        return None
    ranked = sorted(trials.items(), key=lambda kv: trial_exposure(kv[1]))
    td, trows = ranked[len(ranked) // 2]        # median-stall trial
    return td, trows


# ---------------------------------------------------------------------------
# Figure 1: per-trial lifecycle Gantt
# ---------------------------------------------------------------------------

def gantt_lanes(trial_rows: list[dict]) -> list[dict]:
    lanes = []
    for r in trial_rows:
        has_bar = fnum(r["t_started"]) is not None
        t_need = fnum(r["t_first_needed"])
        expo = fnum(r["exposure_s"])
        # Drop anonymous consumption stamps (hash-named, no type, no bar, no
        # measured exposure) — pure bookkeeping rows that only add clutter.
        if not has_bar and r["resource_type"] == "" and not expo:
            continue
        if not has_bar and t_need is None:
            continue
        lanes.append(r)
    lanes.sort(key=lambda r: (fnum(r["t_first_needed"])
                              if fnum(r["t_first_needed"]) is not None
                              else fnum(r["t_started"], 1e12)))
    return lanes


def plot_gantt(trial_rows: list[dict], fig_dir: Path) -> None:
    r0 = trial_rows[0]
    workload, config, trial_dir = r0["workload"], r0["config"], r0["trial_dir"]
    lanes = gantt_lanes(trial_rows)
    if not lanes:
        print(f"  [skip] no drawable lanes for {workload}/{config}/{trial_dir}")
        return

    n = len(lanes)
    fig_h = max(1.6, 0.42 * n + 1.15)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, fig_h))
    half = 0.30

    xs = [0.0]
    # pre-scan the horizontal extent so bar annotations can adapt to it
    for r in lanes:
        for k in ("t_predicted", "t_started", "t_completed", "t_first_needed"):
            v = fnum(r[k])
            if v is not None:
                xs.append(v)
        t_need, expo = fnum(r["t_first_needed"]), fnum(r["exposure_s"], 0.0)
        if t_need is not None and expo:
            xs.append(t_need + expo)
    span = max(xs) - min(xs) + 1e-9
    seen_outcomes, any_expo, any_nopred, any_direct = set(), False, False, False
    ylabels: list[str] = []

    for i, r in enumerate(lanes):
        y = n - 1 - i
        t_pred = fnum(r["t_predicted"])
        t_start = fnum(r["t_started"])
        t_done = fnum(r["t_completed"]) or fnum(r["t_cancelled"])
        t_need = fnum(r["t_first_needed"])
        expo = fnum(r["exposure_s"], 0.0) or 0.0
        b = fnum(r["bytes"])
        outcome = r["outcome"]
        ylab = lane_label(r)

        # transfer bar
        if t_start is not None and t_done is not None:
            lab, face = OUTCOME_FACE.get(outcome, (outcome, PAL["lgray"]))
            ax.barh(y, max(t_done - t_start, 0.01), left=t_start,
                    height=2 * half, color=face, edgecolor="white",
                    linewidth=0.6, zorder=3)
            seen_outcomes.add(outcome)
            xs += [t_start, t_done]
            if b and b >= 5e9:
                if (t_done - t_start) >= 0.12 * span:
                    ax.annotate(fmt_bytes(b), ((t_start + t_done) / 2, y),
                                ha="center", va="center", fontsize=6.5,
                                color="white", zorder=6)
                else:   # bar too narrow for an inside label
                    ylab = f"{ylab}\n{fmt_bytes(b)}"
        elif outcome in NO_BAR_OUTCOMES and t_need is not None:
            any_nopred = True
        elif outcome == "direct_prefetch" and t_need is not None:
            ax.plot(t_need, y, marker="D", ms=5, mfc="none",
                    mec=PAL["green"], mew=1.2, zorder=6)
            ax.annotate(f"gate wait {expo:.2f} s", (t_need, y - half - 0.05),
                        ha="right", va="top", fontsize=6,
                        color=PAL["green"])
            any_direct = True

        # exposed stall overlay (critical-path stall from first need)
        if t_need is not None and expo > 0.05:
            ax.add_patch(Rectangle((t_need, y - half), expo, 2 * half,
                                   facecolor="none", edgecolor=PAL["red"],
                                   hatch="////", linewidth=0.9, zorder=4))
            xs.append(t_need + expo)
            any_expo = True
            if expo >= 30:
                ax.annotate(f"exposed {expo:.0f} s",
                            (t_need + expo, y + half + 0.03),
                            ha="right", va="bottom", fontsize=6,
                            color=PAL["red"])

        # first-need marker
        if t_need is not None:
            ax.plot([t_need, t_need], [y - half - 0.06, y + half + 0.06],
                    color="#222222", linewidth=1.0, zorder=5)
            ax.plot(t_need, y + half + 0.13, marker="v", ms=4.5,
                    color="#222222", zorder=5)
            xs.append(t_need)

        # prediction marker + window annotation for the big/engine objects
        if t_pred is not None:
            ax.plot(t_pred, y, marker="o", ms=3.5, mfc="white",
                    mec="#222222", mew=0.9, zorder=6)
            xs.append(t_pred)
            w = fnum(r["window_s"])
            if w is not None and (r["resource_type"] == "vllm_model"
                                  or (b and b >= 5e9)):
                ax.annotate(f"window {w:g} s", (t_pred, y - half - 0.05),
                            ha="left", va="top", fontsize=6,
                            color="#444444")
        ylabels.append(ylab)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(ylabels)), fontsize=7)
    ax.set_ylim(-0.75, n - 1 + 0.75)
    x_lo, x_hi = min(xs), max(xs)
    pad = 0.03 * (x_hi - x_lo + 1)
    ax.set_xlim(min(x_lo - pad, -pad), x_hi + pad)
    ax.set_xlabel("Time since trace start (s)")
    ax.set_title(f"{WORKLOAD_LABEL.get(workload, workload)} / {config} / "
                 f"{trial_dir}", fontsize=8.5)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=OUTCOME_FACE[o][1], label=OUTCOME_FACE[o][0])
               for o in ("useful", "late", "wasted") if o in seen_outcomes]
    if any_expo:
        handles.append(Patch(facecolor="none", edgecolor=PAL["red"],
                             hatch="////", label="Exposed stall"))
    if any_nopred:
        handles.append(Line2D([], [], color="#222222", marker="v", ms=4.5,
                              linewidth=1.0, label="First need (no prefetch)"))
    else:
        handles.append(Line2D([], [], color="#222222", marker="v", ms=4.5,
                              linewidth=1.0, label="First need"))
    handles.append(Line2D([], [], color="none", marker="o", ms=3.5,
                          mfc="white", mec="#222222", label="Prediction"))
    if any_direct:
        handles.append(Line2D([], [], color="none", marker="D", ms=5,
                              mfc="none", mec=PAL["green"],
                              label="Proactive start (hidden)"))
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.32 / fig_h * 1.6 - 0.10),
              ncol=min(len(handles), 6), frameon=False,
              handlelength=1.4, columnspacing=1.0)

    safe = trial_dir.replace("/", "_")
    save_fig(fig, fig_dir, f"gantt_{workload}_{config}_{safe}")


# ---------------------------------------------------------------------------
# Figure 2: size vs window scatter with hideability frontiers
# ---------------------------------------------------------------------------

def plot_size_vs_window(rows: list[dict], workload: str, fig_dir: Path) -> None:
    pts = []
    for r in rows:
        if r["workload"] != workload or r["outcome"] not in OUTCOME_FACE:
            continue
        b = fnum(r["bytes"])
        if b is None or b <= 0:
            continue
        w = fnum(r["window_s"])
        pts.append((EPS_WINDOW if (w is None or w <= 0) else w, b,
                    r["outcome"]))
    if not pts:
        print(f"  [skip] no size/window points for {workload}")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    x_hi = 2000.0
    y_lo = min(p[1] for p in pts) / 8
    y_hi = max(p[1] for p in pts) * 8
    ax.set_xlim(EPS_WINDOW / 1.6, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # hideability frontiers
    bw = STAGING_GBPS * 1e9
    ws = [10 ** (math.log10(0.02) + i * (math.log10(x_hi) - math.log10(0.02))
                 / 300) for i in range(301)]
    ax.plot(ws, [bw * w for w in ws], color="#333333", linewidth=1.0,
            linestyle="-", zorder=2,
            label=f"I/O frontier: {STAGING_GBPS} GB/s × window")
    we = [w for w in ws if w > ENGINE_SPINUP_S + 1]
    ax.plot(we, [bw * (w - ENGINE_SPINUP_S) for w in we], color="#333333",
            linewidth=1.0, linestyle="--", zorder=2,
            label=f"engine-adj.: {STAGING_GBPS} GB/s × (window − "
                  f"{ENGINE_SPINUP_S:.0f} s)")
    ax.annotate("unhideable\n(above / left)", xy=(0.03, 0.86),
                xycoords="axes fraction", fontsize=6.5, color="#333333",
                ha="left", va="top")

    for outc, (lab, col) in OUTCOME_FACE.items():
        sel = [(x, y) for x, y, o in pts if o == outc]
        if not sel:
            continue
        mk = SCATTER_MARKER[outc]
        kw = ({"color": col} if mk == "x"
              else {"facecolors": "none", "edgecolors": col})
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=14,
                   marker=mk, linewidths=1.0, zorder=3,
                   label=f"{lab} (n={len(sel)})", alpha=0.75, **kw)

    n_eps = sum(1 for x, _, _ in pts if x == EPS_WINDOW)
    if n_eps:
        ax.annotate("no window", (EPS_WINDOW, y_lo * 1.4), fontsize=6,
                    color="#666666", ha="center")

    ax.set_xlabel("Prediction window (s)")
    ax.set_ylabel("Object size (bytes)")
    ax.set_title(WORKLOAD_LABEL.get(workload, workload), fontsize=8.5)
    ax.grid(True, which="major", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", fontsize=5.8, handlelength=1.6,
                    borderaxespad=0.2, frameon=True, framealpha=0.92,
                    edgecolor="none", facecolor="white")
    leg.set_zorder(6)
    save_fig(fig, fig_dir, f"size_vs_window_{workload}")


# ---------------------------------------------------------------------------
# Figure 3: stall taxonomy waterfall
# ---------------------------------------------------------------------------

def merge_class(sc: str) -> str:
    return "policy_skip" if sc.startswith("policy_skip") else sc


def plot_stall_taxonomy(tax_rows: list[dict], fig_dir: Path) -> None:
    data: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in tax_rows:
        key = (r["workload"], r["config"])
        sc = merge_class(r["stall_class"])
        v = fnum(r["stall_s_per_trial"], 0.0) or 0.0
        data[key][sc] = data[key].get(sc, 0.0) + v

    fig, axes = plt.subplots(1, len(TAXONOMY_WORKLOADS),
                             figsize=(DOUBLE_COL_W, 2.25))
    used_classes: list[str] = []

    for ax, wl in zip(axes, TAXONOMY_WORKLOADS):
        configs = [c for c in TAXONOMY_CONFIGS if (wl, c) in data]
        xpos = range(len(configs))
        for xi, cfg in zip(xpos, configs):
            bottom = 0.0
            for sc, lab, col, hat in STALL_CLASSES:
                v = data[(wl, cfg)].get(sc, 0.0)
                if v < 0.5:      # sub-pixel slivers only pollute the legend
                    continue
                ax.bar(xi, v, bottom=bottom, width=0.62, color=col,
                       hatch=hat, edgecolor="white", linewidth=0.7, zorder=3)
                bottom += v
                if sc not in used_classes:
                    used_classes.append(sc)
            ax.annotate(f"{bottom:.0f}", (xi, bottom), ha="center",
                        va="bottom", fontsize=6.5, color="#222222",
                        xytext=(0, 1.5), textcoords="offset points")
        ax.set_xticks(list(xpos))
        ax.set_xticklabels([CONFIG_SHORT.get(c, c) for c in configs],
                           fontsize=6.5, rotation=20, ha="right")
        ax.set_xlim(-0.65, len(configs) - 0.35)
        ax.set_ylim(0, max((sum(data[(wl, c)].values()) for c in configs),
                           default=1.0) * 1.18)
        ax.set_title(WORKLOAD_LABEL.get(wl, wl), fontsize=8)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Exposed stall (s / trial)")

    handles = [Patch(facecolor=STALL_STYLE[sc][1], hatch=STALL_STYLE[sc][2],
                     edgecolor="white", label=STALL_STYLE[sc][0])
               for sc, _, _, _ in STALL_CLASSES if sc in used_classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=6.5,
               handlelength=1.3, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.32, bottom=0.30)
    save_fig(fig, fig_dir, "stall_taxonomy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--trial", default=None,
                    help="render Gantt(s) for trials whose run_id or "
                         "trial_dir contains this substring")
    ap.add_argument("--all-defaults", action="store_true",
                    help="render only the four representative default Gantts")
    args = ap.parse_args()

    life_csv = args.eval_root / "eval_prefetch_lifecycle.csv"
    tax_csv = args.eval_root / "eval_stall_taxonomy.csv"
    rows = read_csv(life_csv)
    apply_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.trial:
        trials: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for r in rows:
            if args.trial in r["run_id"] or args.trial in r["trial_dir"]:
                trials[(r["workload"], r["config"], r["trial_dir"])].append(r)
        if not trials:
            print(f"no trial matching '{args.trial}'", file=sys.stderr)
            return 1
        print(f"Gantt for {len(trials)} matching trial(s):")
        for trows in trials.values():
            plot_gantt(trows, args.fig_dir)
        return 0

    print("Default Gantt figures:")
    for wl, cfg in GANTT_DEFAULTS:
        sel = representative_trial(rows, wl, cfg)
        if sel is None:
            print(f"  [skip] no trials for {wl}/{cfg}")
            continue
        _, trows = sel
        plot_gantt(trows, args.fig_dir)

    if args.all_defaults:
        return 0

    print("Size-vs-window scatter figures:")
    for wl in ("chemgraph_swap", "chemgraph_ensemble", "chemgraph_screen"):
        plot_size_vs_window(rows, wl, args.fig_dir)

    print("Stall taxonomy figure:")
    plot_stall_taxonomy(read_csv(tax_csv), args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
