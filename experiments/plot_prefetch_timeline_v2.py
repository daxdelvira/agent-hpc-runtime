"""
Figure v2: full_system vs no_model_prefetch — Gruvbox Dark color scheme,
extra spacing between panels B and C.
Outputs: results/figure_prefetch_timeline_v2.pdf and .png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_CSV  = os.path.join(REPO_ROOT, "results", "system_profile_679fb244-44c.csv")
NMP_CSV = os.path.join(REPO_ROOT, "results", "system_profile_312ad04e-7a8.csv")

FS_LAMMPS  = [(1183, 2083), (2880, 3780)]
NMP_LAMMPS = [(943,  1843), (3090, 3990)]
NMP_STALLS = [(1843, 2043), (3990, 4176)]
FS_SWAP_READY_S = 1365
FS_LAMMPS2_MID  = (2880 + 3780) / 2

# ── Gruvbox Dark palette ──────────────────────────────────────────────────────
GRV = dict(
    bg      = "#282828",
    bg1     = "#3c3836",
    bg2     = "#504945",
    fg      = "#ebdbb2",
    fg2     = "#d5c4a1",
    gray    = "#a89984",
    red     = "#fb4934",
    green   = "#b8bb26",
    yellow  = "#fabd2f",
    blue    = "#83a598",    # aqua-blue
    purple  = "#d3869b",
    orange  = "#fe8019",
)

# Font sizes (v2 baseline)
FS = dict(title=11, label=10, tick=9, annot=8.5, small=7.5)


def apply_gruvbox_rc(dark_bg=True):
    base = {
        "xtick.labelsize":   FS["tick"],
        "ytick.labelsize":   FS["tick"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
    }
    if dark_bg:
        base.update({
            "figure.facecolor":  GRV["bg"],
            "axes.facecolor":    GRV["bg"],
            "axes.edgecolor":    GRV["bg2"],
            "axes.labelcolor":   GRV["fg"],
            "text.color":        GRV["fg"],
            "xtick.color":       GRV["gray"],
            "ytick.color":       GRV["gray"],
            "savefig.facecolor": GRV["bg"],
            "savefig.edgecolor": GRV["bg"],
        })
    else:
        base.update({
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
            "axes.edgecolor":    "#cccccc",
            "axes.labelcolor":   "#1d2021",
            "text.color":        "#1d2021",
            "xtick.color":       "#504945",
            "ytick.color":       "#504945",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        })
    plt.rcParams.update(base)


def load(path):
    df = pd.read_csv(path)
    df["mem_gb"] = (df["gpu0_mem_used_mb"] + df["gpu1_mem_used_mb"] +
                    df["gpu2_mem_used_mb"] + df["gpu3_mem_used_mb"]) / 4000
    df["t_min"]  = df["t_rel_s"] / 60
    return df


def shade_spans(ax, spans, color, alpha, zorder=0):
    for t0, t1 in spans:
        ax.axvspan(t0 / 60, t1 / 60, color=color, alpha=alpha, zorder=zorder, linewidth=0)


def add_lammps_labels(ax, windows, ylim_bottom):
    for i, (t0, t1) in enumerate(windows):
        mid = (t0 + t1) / 2 / 60
        ax.text(mid, ylim_bottom + 0.5, f"LAMMPS {i+1}\n(900 s)",
                ha="center", va="bottom", fontsize=FS["small"],
                color=GRV["green"], style="italic", fontweight="bold")


def annotate(ax, text, xy, xytext, color, box_fc=None):
    fc = box_fc if box_fc is not None else GRV["bg1"]
    ax.annotate(
        text, xy=xy, xytext=xytext,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        fontsize=FS["annot"], ha="left", color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=color, alpha=0.92),
    )


def build_figure(font_sizes, figsize=(14, 9.5), dark_bg=True):
    global FS
    FS = font_sizes

    fs  = load(FS_CSV)
    nmp = load(NMP_CSV)

    apply_gruvbox_rc(dark_bg=dark_bg)

    # On white bg, swap annotation box fill to white and tweak shading alpha
    annot_fc  = GRV["bg1"] if dark_bg else "white"
    lammps_fg = GRV["fg"]  if dark_bg else "#1d2021"
    leg_alpha  = 0         if dark_bg else 0.85
    shade_alpha_lammps = 0.18 if dark_bg else 0.22
    shade_alpha_stall  = 0.18 if dark_bg else 0.22

    YLIM   = (-10, 100)
    YTICKS = [0, 36, 93]

    # 4-row gridspec: panels A, B, spacer, C — gives extra gap before C
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(4, 1, figure=fig,
                   height_ratios=[3.5, 3.5, 0.55, 1.5],
                   hspace=0.08)

    ax_fs  = fig.add_subplot(gs[0])
    ax_nmp = fig.add_subplot(gs[1], sharex=ax_fs)
    ax_bar = fig.add_subplot(gs[3])

    # ── Panel A ───────────────────────────────────────────────────────────────
    shade_spans(ax_fs, FS_LAMMPS, GRV["green"], alpha=shade_alpha_lammps)
    ax_fs.plot(fs["t_min"], fs["mem_gb"], color=GRV["blue"], lw=1.6, zorder=3)

    add_lammps_labels(ax_fs, FS_LAMMPS, YLIM[0])

    annotate(ax_fs,
             "Proactive swap:\nqwen_72b hot in 185 s\n→ 0 s stall",
             xy=(FS_SWAP_READY_S / 60, 93),
             xytext=((FS_SWAP_READY_S + 240) / 60, 68),
             color=GRV["blue"], box_fc=annot_fc)

    annotate(ax_fs,
             "Scheduler bug:\nno swap → 128 s stall",
             xy=(FS_LAMMPS2_MID / 60, 2),
             xytext=((FS_LAMMPS2_MID + 110) / 60, 28),
             color=GRV["red"], box_fc=annot_fc)

    ax_fs.set_ylim(*YLIM)
    ax_fs.set_yticks(YTICKS)
    ax_fs.set_ylabel("Avg GPU\nMemory (GB)", fontsize=FS["label"], color=GRV["fg"])
    ax_fs.set_title("(A) full_system  —  wall time: 3979 s",
                    fontweight="bold", loc="left", fontsize=FS["title"], color=GRV["fg"])
    ax_fs.tick_params(axis="x", labelbottom=False)

    # ── Panel B ───────────────────────────────────────────────────────────────
    shade_spans(ax_nmp, NMP_LAMMPS, GRV["green"], alpha=shade_alpha_lammps)
    shade_spans(ax_nmp, NMP_STALLS,  GRV["red"],   alpha=shade_alpha_stall)
    ax_nmp.plot(nmp["t_min"], nmp["mem_gb"], color=GRV["orange"], lw=1.6, zorder=3)

    add_lammps_labels(ax_nmp, NMP_LAMMPS, YLIM[0])

    for t0, t1 in NMP_STALLS:
        mid = (t0 + t1) / 2 / 60
        ax_nmp.text(mid, 50, "~200 s\nstall", ha="center", va="center",
                    fontsize=FS["annot"], color=GRV["red"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=annot_fc, ec=GRV["red"], alpha=0.92))

    nmp_lammps1_mid = (NMP_LAMMPS[0][0] + NMP_LAMMPS[0][1]) / 2 / 60
    ax_nmp.text(nmp_lammps1_mid, 46,
                "GPU idle — no prefetch\n(model loaded on demand after)",
                ha="center", va="center", fontsize=FS["annot"], color=GRV["green"],
                bbox=dict(boxstyle="round,pad=0.3", fc=annot_fc, ec=GRV["green"], alpha=0.92))

    ax_nmp.set_ylim(*YLIM)
    ax_nmp.set_yticks(YTICKS)
    ax_nmp.set_ylabel("Avg GPU\nMemory (GB)", fontsize=FS["label"], color=GRV["fg"])
    ax_nmp.set_xlabel("Time (minutes)", fontsize=FS["label"], color=GRV["fg"])
    ax_nmp.set_title("(B) no_model_prefetch  —  wall time: 4476 s",
                     fontweight="bold", loc="left", fontsize=FS["title"], color=GRV["fg"])

    x_max = max(fs["t_min"].max(), nmp["t_min"].max()) * 1.01
    ax_nmp.set_xlim(0, x_max)

    lammps_patch = mpatches.Patch(color=GRV["green"], alpha=0.5,
                                  label="LAMMPS compute window (900 s)")
    stall_patch  = mpatches.Patch(color=GRV["red"],   alpha=0.5,
                                  label="On-demand model load stall")
    ax_fs.legend(handles=[lammps_patch], loc="upper right",
                 fontsize=FS["small"], framealpha=leg_alpha, labelcolor=lammps_fg)
    ax_nmp.legend(handles=[lammps_patch, stall_patch], loc="upper right",
                  fontsize=FS["small"], framealpha=leg_alpha, labelcolor=lammps_fg)

    # ── Panel C ───────────────────────────────────────────────────────────────
    labels = ["Observed\n(1 swap fired)", "Max possible\n(2 swaps)"]
    values = [497, 930]
    colors = [GRV["blue"], GRV["yellow"]]

    bars = ax_bar.barh(labels, values, color=colors, edgecolor=GRV["bg2"],
                       linewidth=1.2, height=0.45)

    for bar, val, col in zip(bars, values, colors):
        ax_bar.text(val + 12, bar.get_y() + bar.get_height() / 2,
                    f"{val} s", va="center", fontsize=FS["label"],
                    fontweight="bold", color=col)

    ax_bar.axvline(497, color=GRV["blue"], linestyle="--", lw=1.2, alpha=0.55)
    ax_bar.set_xlim(0, 1050)
    ax_bar.set_xlabel("Wall-time savings vs. no_model_prefetch (s)",
                      fontsize=FS["label"], color=GRV["fg"])
    ax_bar.set_title("(C) Proactive-swap savings range",
                     fontweight="bold", loc="left", fontsize=FS["title"], color=GRV["fg"])
    ax_bar.tick_params(axis="y", labelsize=FS["annot"])

    return fig


def main(suffix="v2", font_sizes=None, figsize=(14, 9.5), dark_bg=True):
    if font_sizes is None:
        font_sizes = dict(title=11, label=10, tick=9, annot=8.5, small=7.5)
    fig = build_figure(font_sizes, figsize=figsize, dark_bg=dark_bg)
    out = os.path.join(REPO_ROOT, "results", f"figure_prefetch_timeline_{suffix}")
    fig.savefig(out + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}.pdf and .png")
    plt.close(fig)


if __name__ == "__main__":
    main("v2")
