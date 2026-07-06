"""
Figure: full_system vs no_model_prefetch GPU memory timeline + savings bar.
Outputs: results/figure_prefetch_timeline.pdf and .png
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

# LAMMPS window boundaries (seconds) — confirmed from log + 3 s-interval GPU profile
FS_LAMMPS  = [(1183, 2083), (2880, 3780)]   # full_system
NMP_LAMMPS = [(943,  1843), (3090, 3990)]   # no_model_prefetch

# Post-LAMMPS on-demand stall periods for no_model_prefetch
NMP_STALLS = [(1843, 2043), (3990, 4176)]

# Annotation anchors (seconds) — proactive swap result in full_system LAMMPS 1
FS_SWAP_READY_S = 1365   # qwen_72b hot (185 s after kill at 1183 s)
FS_LAMMPS2_MID  = (2880 + 3780) / 2  # midpoint of LAMMPS 2 window


def load(path):
    df = pd.read_csv(path)
    df["mem_gb"] = (df["gpu0_mem_used_mb"] + df["gpu1_mem_used_mb"] +
                    df["gpu2_mem_used_mb"] + df["gpu3_mem_used_mb"]) / 4000
    df["t_min"]  = df["t_rel_s"] / 60
    return df


def shade_spans(ax, spans, color, alpha, zorder=0):
    for t0, t1 in spans:
        ax.axvspan(t0 / 60, t1 / 60, color=color, alpha=alpha, zorder=zorder, linewidth=0)


def add_lammps_labels(ax, lammps_windows, ylim_bottom):
    for i, (t0, t1) in enumerate(lammps_windows):
        mid = (t0 + t1) / 2 / 60
        ax.text(mid, ylim_bottom + 0.5, f"LAMMPS {i+1}\n(900 s)",
                ha="center", va="bottom", fontsize=7.5, color="#1b5e20",
                style="italic", fontweight="bold")


def main():
    fs  = load(FS_CSV)
    nmp = load(NMP_CSV)

    LAMMPS_COLOR  = "#c8e6c9"  # light green
    STALL_COLOR   = "#ffcdd2"  # light red
    YLIM          = (-10, 100)
    YTICKS        = [0, 36, 93]

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3.5, 3.5, 1.5], hspace=0.12)

    ax_fs  = fig.add_subplot(gs[0])
    ax_nmp = fig.add_subplot(gs[1], sharex=ax_fs)
    ax_bar = fig.add_subplot(gs[2])

    # ── Panel A: full_system ──────────────────────────────────────────────────
    shade_spans(ax_fs, FS_LAMMPS, LAMMPS_COLOR, alpha=0.70)
    ax_fs.plot(fs["t_min"], fs["mem_gb"], color="#1565C0", lw=1.4, zorder=3)

    # LAMMPS window labels (inside the green band, near bottom)
    add_lammps_labels(ax_fs, FS_LAMMPS, YLIM[0])

    # Annotation — LAMMPS 1: proactive swap worked
    ax_fs.annotate(
        "Proactive swap:\nqwen_72b hot in 185 s\n→ 0 s stall",
        xy=(FS_SWAP_READY_S / 60, 93),
        xytext=((FS_SWAP_READY_S + 250) / 60, 72),
        arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.2),
        fontsize=8.5, ha="left", color="#0d47a1",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#0d47a1", alpha=0.85),
    )

    # Annotation — LAMMPS 2: no proactive swap (bug)
    ax_fs.annotate(
        "Scheduler bug:\nno swap → 128 s stall",
        xy=(FS_LAMMPS2_MID / 60, 2),
        xytext=((FS_LAMMPS2_MID + 120) / 60, 28),
        arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2),
        fontsize=8.5, ha="left", color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b71c1c", alpha=0.85),
    )

    ax_fs.set_ylim(*YLIM)
    ax_fs.set_yticks(YTICKS)
    ax_fs.set_ylabel("Avg GPU\nMemory (GB)", fontsize=10)
    ax_fs.set_title("(A) full_system  —  wall time: 3979 s",
                    fontweight="bold", loc="left", fontsize=11)
    ax_fs.tick_params(axis="x", labelbottom=False)
    ax_fs.spines["top"].set_visible(False)
    ax_fs.spines["right"].set_visible(False)

    # ── Panel B: no_model_prefetch ────────────────────────────────────────────
    shade_spans(ax_nmp, NMP_LAMMPS, LAMMPS_COLOR, alpha=0.70)
    shade_spans(ax_nmp, NMP_STALLS,  STALL_COLOR,  alpha=0.60)
    ax_nmp.plot(nmp["t_min"], nmp["mem_gb"], color="#E65100", lw=1.4, zorder=3)

    add_lammps_labels(ax_nmp, NMP_LAMMPS, YLIM[0])

    # Stall labels
    for t0, t1 in NMP_STALLS:
        mid = (t0 + t1) / 2 / 60
        ax_nmp.text(mid, 50, "~200 s\nstall", ha="center", va="center",
                    fontsize=8, color="#b71c1c", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b71c1c", alpha=0.85))

    # Annotation — GPU idle during LAMMPS
    nmp_lammps1_mid = (NMP_LAMMPS[0][0] + NMP_LAMMPS[0][1]) / 2 / 60
    ax_nmp.text(nmp_lammps1_mid, 46,
                "GPU idle — no prefetch\n(model loaded on demand after)",
                ha="center", va="center", fontsize=8.5, color="#1b5e20",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1b5e20", alpha=0.85))

    ax_nmp.set_ylim(*YLIM)
    ax_nmp.set_yticks(YTICKS)
    ax_nmp.set_ylabel("Avg GPU\nMemory (GB)", fontsize=10)
    ax_nmp.set_xlabel("Time (minutes)", fontsize=10)
    ax_nmp.set_title("(B) no_model_prefetch  —  wall time: 4476 s",
                     fontweight="bold", loc="left", fontsize=11)
    ax_nmp.spines["top"].set_visible(False)
    ax_nmp.spines["right"].set_visible(False)

    # x-axis range: cover both runs
    x_max = max(fs["t_min"].max(), nmp["t_min"].max()) * 1.01
    ax_nmp.set_xlim(0, x_max)

    # Shared legend patches
    lammps_patch = mpatches.Patch(color=LAMMPS_COLOR, label="LAMMPS compute window (900 s)")
    stall_patch  = mpatches.Patch(color=STALL_COLOR,  label="On-demand model load stall")
    ax_fs.legend(handles=[lammps_patch], loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_nmp.legend(handles=[lammps_patch, stall_patch], loc="upper right", fontsize=8.5, framealpha=0.9)

    # ── Panel C: savings bar ──────────────────────────────────────────────────
    labels   = ["Observed\n(1 swap fired)", "Max possible\n(2 swaps)"]
    values   = [497, 930]
    colors   = ["#2196F3", "#BBDEFB"]
    edgecols = ["#1565C0", "#1565C0"]

    bars = ax_bar.barh(labels, values, color=colors, edgecolor=edgecols,
                       linewidth=1.3, height=0.45)

    for bar, val in zip(bars, values):
        ax_bar.text(val + 12, bar.get_y() + bar.get_height() / 2,
                    f"{val} s", va="center", fontsize=10, fontweight="bold", color="#0d47a1")

    ax_bar.axvline(497, color="#1565C0", linestyle="--", lw=1.2, alpha=0.6)
    ax_bar.set_xlim(0, 1050)
    ax_bar.set_xlabel("Wall-time savings vs. no_model_prefetch (s)", fontsize=10)
    ax_bar.set_title("(C) Proactive-swap savings range",
                     fontweight="bold", loc="left", fontsize=11)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_base = os.path.join(REPO_ROOT, "results", "figure_prefetch_timeline")
    fig.savefig(out_base + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_base}.pdf and .png")


if __name__ == "__main__":
    main()
