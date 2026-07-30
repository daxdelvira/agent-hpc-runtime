#!/usr/bin/env python
"""
plot_mechanism_diagrams.py — paper mechanism timelines (Gruvbox Dark, Times-style serif).

Figure 1  figures/mechanism_ensemble_timeline.{pdf,png}
    chemgraph_ensemble (Option D): baseline pays an on-demand aggregator model
    load at AggregatorAgent; the runtime launches the same load at
    run_mace_ensemble tool_start, hiding it inside the GPU-idle MACE window.

Figure 2  figures/mechanism_swap_timeline.{pdf,png}
    chemgraph_swap: mirrored layout showing why prediction cannot help — the
    planner window (~12 s) is far smaller than the 72B worker bring-up
    (200-700 s), so nearly all of the load stays exposed even with a perfect
    (oracle) predictor.

Segment widths are schematic; every annotated duration is measured, from:
  ensemble baseline t01/t02/t03 (aggregator_wait 150.2/160.2/500.6 s),
  ensemble full_system t01 (aggregator_wait 0.01 s, MACE window ~840-900 s),
  swap baseline t01-t03 (swap_wait 200-271 s), swap full_system t02/t04/t05
  (overlapped_io 11.9-20.0 s, residual load 185-255 s).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

# ---------------------------------------------------------------- Gruvbox Dark
BG      = "#282828"   # bg0
BG_SOFT = "#32302f"   # bg0_s   (lane backing)
GRID    = "#3c3836"   # bg1
NEUTRAL = "#504945"   # bg2     (structural segments, identical in both lanes)
MUTED   = "#a89984"   # gray-light ink
FG      = "#ebdbb2"   # fg1 ink

C_REASON  = "#fabd2f"  # agent / LLM turns
C_COMPUTE = "#458588"  # tool compute (MACE)
C_STALL   = "#fb4934"  # model load, exposed (workflow blocked)
C_OVERLAP = "#8ec07c"  # model load, overlapped (hidden by the runtime); hatched

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": FG,
    "font.size": 9,
    "hatch.linewidth": 0.7,
})

BAR_H = 0.34


def seg(ax, y, x0, x1, color, hatch=None, label=None, label_color=FG,
        fontsize=8.0, label_y=None, zorder=3):
    """One timeline segment: flat bar with a 2px-equivalent surface gap."""
    ax.add_patch(Rectangle((x0, y - BAR_H / 2), x1 - x0, BAR_H,
                           facecolor=color, edgecolor=BG, linewidth=0.9,
                           hatch=hatch, zorder=zorder))
    if label:
        ax.text((x0 + x1) / 2, label_y if label_y is not None else y,
                label, ha="center", va="center", color=label_color,
                fontsize=fontsize, zorder=5)


def lane(ax, y, x1, name):
    ax.add_patch(Rectangle((0, y - BAR_H / 2), x1, BAR_H, facecolor=BG_SOFT,
                           edgecolor="none", zorder=1))
    ax.text(-14, y, name, ha="right", va="center", color=FG, fontsize=9.5)


def legend(ax, extra=(), compute=True):
    handles = [
        Patch(facecolor=C_REASON,  edgecolor=BG, label="agent / LLM turns"),
        *([Patch(facecolor=C_COMPUTE, edgecolor=BG,
                 label="tool compute (GPU-idle for LLMs)")] if compute else []),
        Patch(facecolor=C_STALL,   edgecolor=BG, label="model load — exposed (blocked)"),
        Patch(facecolor=C_OVERLAP, edgecolor=BG, hatch="///",
              label="model load — overlapped (hidden)"),
        *extra,
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.16),
              ncol=3, frameon=False, fontsize=8, labelcolor=FG,
              handlelength=1.4, handleheight=1.1, columnspacing=1.4)


def base_axes(ax, xmax, xlabel, name_margin=0.12):
    ax.set_xlim(-name_margin * xmax, xmax)
    ax.set_ylim(-0.62, 1.78)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0, xmax)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= xmax])
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    for s in ax.get_xticks():
        ax.axvline(s, color=GRID, lw=0.6, zorder=0)


# ============================================================ Figure 1: ensemble
def fig_ensemble(outstem):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    yB, yF = 1.25, 0.45  # baseline lane, SystemName lane

    R0, R1 = 0, 45            # agent turns
    W0, W1 = 45, 215          # worker 72B bring-up (structural, both)
    I0, I1 = 215, 265         # worker inference
    M0, M1 = 265, 1105        # MACE ensemble window (~840 s)
    AGG = 150                 # measured aggregator load, t01
    TAIL = 55                 # aggregation

    for y, name in [(yB, "Baseline"), (yF, "SystemName")]:
        lane(ax, y, (M1 + AGG + TAIL) if y == yB else (M1 + TAIL), name)
        seg(ax, y, R0, R1, C_REASON)
        seg(ax, y, W0, W1, NEUTRAL, label="worker 72B load\n(both lanes)",
            label_color=MUTED, fontsize=7.0)
        seg(ax, y, I0, I1, C_REASON)
        seg(ax, y, M0, M1, C_COMPUTE)

    ax.text((M0 + M1) / 2, yB, "run_mace_ensemble  —  ~840–900 s, GPUs idle for LLMs",
            ha="center", va="center", color=FG, fontsize=8.0, zorder=5)
    # keep the SystemName window label clear of the hatched sub-bar
    ax.text((M0 + 25 + AGG + M1) / 2, yF, "run_mace_ensemble",
            ha="center", va="center", color=FG, fontsize=8.0, zorder=5)

    # Baseline: exposed aggregator load after MACE
    seg(ax, yB, M1, M1 + AGG, C_STALL)
    ax.annotate("aggregator load on demand:\n150.2 / 160.2 / 500.6 s exposed",
                xy=(M1 + 25, yB + BAR_H / 2), xytext=(M1 - 320, yB + 0.42),
                ha="center", color=C_STALL, fontsize=8.0,
                arrowprops=dict(arrowstyle="-", color=C_STALL, lw=0.8))
    seg(ax, yB, M1 + AGG, M1 + AGG + TAIL, C_REASON)

    # SystemName: same load hidden inside the MACE window
    seg(ax, yF, M0 + 25, M0 + 25 + AGG, C_OVERLAP, hatch="///", zorder=4)
    ax.annotate("prefetch launched at tool start;\nload hidden inside the window",
                xy=(M0 + 25 + AGG / 2, yF - BAR_H / 2),
                xytext=(M0 + 340, yF - 0.44), ha="center",
                color=C_OVERLAP, fontsize=8.0,
                arrowprops=dict(arrowstyle="-", color=C_OVERLAP, lw=0.8))
    seg(ax, yF, M1, M1 + TAIL, C_REASON)
    ax.annotate("wait 0.01 s", xy=(M1, yF + BAR_H / 2),
                xytext=(M1 + 60, yF + 0.38), ha="left",
                color=C_OVERLAP, fontsize=8.0,
                arrowprops=dict(arrowstyle="->", color=C_OVERLAP, lw=0.9))

    # wall-time delta bracket
    ax.annotate("", xy=(M1 + AGG + TAIL + 5, 1.60), xytext=(M1 + TAIL, 1.60),
                arrowprops=dict(arrowstyle="<->", color=FG, lw=0.9))
    ax.text(M1 + TAIL + (AGG + TAIL) / 2, 1.66, "gate eliminated (150–500 s)",
            ha="center", va="bottom", color=FG, fontsize=7.6)

    base_axes(ax, M1 + AGG + TAIL + 10, "time (s) — schematic; annotated durations measured")
    ax.set_title("ChemGraph ensemble: aggregator model load overlapped with MACE compute",
                 color=FG, fontsize=10.5, pad=10)
    legend(ax, extra=[Patch(facecolor=NEUTRAL, edgecolor=BG,
                            label="structural (identical in both)")])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================ Figure 2: swap
def fig_swap(outstem):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    yB, yF = 1.25, 0.45

    P0, P1 = 0, 6             # planner turn (~6 s)
    X1 = 12                   # + plan-extraction LLM call (~5.5 s)
    LOAD = 210                # worker 72B bring-up (measured 200-271 s; up to ~700 s)
    OVL = 16                  # overlapped head, full_system (11.9-20.0 s measured)
    INF = 55                  # worker inference

    # Baseline
    lane(ax, yB, P1 + LOAD + INF, "Baseline")
    seg(ax, yB, P0, P1, C_REASON)
    seg(ax, yB, P1, P1 + LOAD, C_STALL,
        label="worker 72B load — 200–700 s exposed", fontsize=8.0)
    seg(ax, yB, P1 + LOAD, P1 + LOAD + INF, C_REASON)

    # SystemName (planner turn + plan-extraction call = the whole window)
    lane(ax, yF, X1 + LOAD + INF, "SystemName")
    seg(ax, yF, P0, X1, C_REASON)
    seg(ax, yF, X1, X1 + OVL, C_OVERLAP, hatch="///", zorder=4)
    seg(ax, yF, X1 + OVL, X1 + LOAD, C_STALL,
        label="still exposed: 185–255 s", fontsize=8.0)
    seg(ax, yF, X1 + LOAD, X1 + LOAD + INF, C_REASON)

    ax.annotate("plan extracted, prefetch starts (~12 s)",
                xy=(X1, yF - BAR_H / 2), xytext=(X1 + 12, yF - 0.44),
                ha="left", color=C_OVERLAP, fontsize=8.0,
                arrowprops=dict(arrowstyle="->", color=C_OVERLAP, lw=0.9))
    ax.annotate("overlapped: only 12–20 s",
                xy=(X1 + OVL / 2, yF + BAR_H / 2), xytext=(64, yF + 0.42),
                ha="center", color=C_OVERLAP, fontsize=8.0,
                arrowprops=dict(arrowstyle="-", color=C_OVERLAP, lw=0.8))

    # the mechanism: window << load
    ax.axvline(X1, color=MUTED, lw=0.8, ls=(0, (4, 3)), ymin=0.14, ymax=0.86, zorder=2)
    ax.text(0, 1.70, "overlap window ≈ 12 s  ≪  load 200–700 s   →   ceiling ≈ 5 %;"
            " even an oracle predictor gains ≈ 0",
            ha="left", va="center", color=FG, fontsize=8.3)

    base_axes(ax, X1 + LOAD + INF + 6,
              "time (s) — schematic; annotated durations measured", name_margin=0.24)
    ax.set_title("ChemGraph swap: the planner window cannot hide a 72B bring-up"
                 " — prediction is not the bottleneck",
                 color=FG, fontsize=10.5, pad=10)
    legend(ax, compute=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import pathlib
    figdir = pathlib.Path(__file__).resolve().parent.parent / "figures"
    figdir.mkdir(exist_ok=True)
    fig_ensemble(str(figdir / "mechanism_ensemble_timeline"))
    fig_swap(str(figdir / "mechanism_swap_timeline"))
    print("wrote", figdir / "mechanism_{ensemble,swap}_timeline.{pdf,png}")
