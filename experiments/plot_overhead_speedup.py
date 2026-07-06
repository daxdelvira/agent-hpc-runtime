"""
Three companion figures comparing MegaMmap overhead vs. proactive-swap speedup.

  figure_overhead.pdf/png    — +20.2 % MegaMmap wall-time overhead
  figure_speedup.pdf/png     — −22 % proactive-swap speedup (best-case, bug fixed)
  figure_comparison.pdf/png  — both side-by-side on a shared axis

Font: Times New Roman (falls back to Nimbus Roman / serif).
Background: white.  Data colours: Gruvbox Dark palette.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "dejavuserif",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "savefig.edgecolor":  "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Gruvbox Dark colours (on white bg) ───────────────────────────────────────
GRV_RED    = "#cc241d"   # overhead
GRV_GREEN  = "#98971a"   # speedup
GRV_EDGE   = "#504945"
GRV_GRID   = "#e8e8e8"
GRV_TEXT   = "#1d2021"
GRV_GRAY   = "#a89984"   # neutral / baseline bar

# ── Data ──────────────────────────────────────────────────────────────────────
OVERHEAD_PCT = 20.2   # MegaMmap: agentic-nomega 69.1 s → agentic-mega 83.0 s
SPEEDUP_PCT  = 22.0   # Proactive swap best-case (divergence-guard bug fixed)
Y_MAX        = 27.0   # shared y-axis ceiling for all three plots


# ── Helpers ───────────────────────────────────────────────────────────────────
def _style(ax, ylabel="", title="", fs=12):
    ax.yaxis.grid(True, color=GRV_GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=fs - 1)
    ax.set_ylabel(ylabel, fontsize=fs, color=GRV_TEXT, labelpad=6)
    if title:
        ax.set_title(title, fontsize=fs + 1, fontweight="bold",
                     color=GRV_TEXT, pad=10, loc="left")
    ax.set_ylim(0, Y_MAX)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))


def _bar_label(ax, x, pct, color, sign, fs=14):
    """Bold percentage annotation just above the bar."""
    ax.text(x, pct + 0.45, f"{sign}{pct:.1f}%",
            ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color=color, fontfamily="serif")


def _source_note(ax, note, fs=8.5):
    ax.text(0.01, -0.13, note, transform=ax.transAxes,
            fontsize=fs, color=GRV_GRAY, va="top", style="italic")


def save(fig, name):
    out = os.path.join(REPO_ROOT, "results", name)
    fig.savefig(out + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}.pdf / .png")
    plt.close(fig)


# ── Plot 1: MegaMmap overhead ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 5.0))

ax.bar([0], [OVERHEAD_PCT], color=GRV_RED, edgecolor=GRV_EDGE,
       linewidth=1.2, width=0.50, zorder=3)
_bar_label(ax, 0, OVERHEAD_PCT, GRV_RED, "+")
_style(ax,
       ylabel="Wall-time overhead (%)",
       title="MegaMmap overhead\non AtomAgents",
       fs=12)
ax.set_xticks([0])
ax.set_xticklabels(["MegaMmap\n(Hermes hook)"], fontsize=11)
ax.set_xlim(-0.65, 0.65)
_source_note(ax, "agentic-nomega 69.1 s → agentic-mega 83.0 s")
plt.tight_layout()
save(fig, "figure_overhead")


# ── Plot 2: Proactive-swap speedup ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 5.0))

ax.bar([0], [SPEEDUP_PCT], color=GRV_GREEN, edgecolor=GRV_EDGE,
       linewidth=1.2, width=0.50, zorder=3)
_bar_label(ax, 0, SPEEDUP_PCT, GRV_GREEN, "−")   # minus sign
_style(ax,
       ylabel="Wall-time reduction (%)",
       title="Proactive-swap speedup\nvs. on-demand loading",
       fs=12)
ax.set_xticks([0])
ax.set_xticklabels(["Proactive swap\n(full system)", ], fontsize=11)
ax.set_xlim(-0.65, 0.65)
_source_note(ax, "no_model_prefetch 4,476 s → full_system (est.) ≈3,490 s")
plt.tight_layout()
save(fig, "figure_speedup")


# ── Plot 3: Diverging comparison (overhead up, speedup down) ──────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.8))

# overhead bar: 0 → +OVERHEAD_PCT  (points up)
# speedup bar:  0 → -SPEEDUP_PCT   (points down)
ax.bar([0],  OVERHEAD_PCT,  color=GRV_RED,   edgecolor=GRV_EDGE, linewidth=1.2,
       width=0.50, zorder=3)
ax.bar([1], -SPEEDUP_PCT,   color=GRV_GREEN, edgecolor=GRV_EDGE, linewidth=1.2,
       width=0.50, zorder=3)

# Annotations: overhead label sits above the bar, speedup label sits below
ax.text(0,  OVERHEAD_PCT + 0.5, f"+{OVERHEAD_PCT:.1f}%",
        ha="center", va="bottom", fontsize=15, fontweight="bold",
        color=GRV_RED, fontfamily="serif")
ax.text(1, -SPEEDUP_PCT - 0.5, f"−{SPEEDUP_PCT:.1f}%",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color=GRV_GREEN, fontfamily="serif")

# Diverging axis
span = 26
ax.set_ylim(-span, span)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))

# Bold zero line
ax.axhline(0, color="#555555", lw=1.2, zorder=4)

# Grid only (no top/right spines)
ax.yaxis.grid(True, color=GRV_GRID, lw=0.9, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", bottom=False, labelsize=11.5, colors="#555555")
ax.tick_params(axis="y", labelsize=11, colors="#555555")

ax.set_xticks([0, 1])
ax.set_xticklabels(["MegaMmap\noverhead", "Proactive swap\nspeedup"], fontsize=11.5)
ax.set_xlim(-0.65, 1.65)

# Direction labels on the y-axis
ax.text(-0.68, span * 0.55, "slower →", rotation=90, va="center", ha="center",
        fontsize=9.5, color=GRV_RED, fontfamily="serif", style="italic",
        transform=ax.get_yaxis_transform())
ax.text(-0.68, -span * 0.55, "← faster", rotation=90, va="center", ha="center",
        fontsize=9.5, color=GRV_GREEN, fontfamily="serif", style="italic",
        transform=ax.get_yaxis_transform())
ax.set_ylabel("Wall-time change (%)", fontsize=12, color=GRV_TEXT, labelpad=6)

ax.set_title("MegaMmap overhead vs. proactive-swap speedup",
             fontsize=13, fontweight="bold", color=GRV_TEXT, pad=10, loc="left")

_source_note(ax,
    "Overhead: agentic-nomega 69.1 s → agentic-mega 83.0 s  |  "
    "Speedup: best-case estimate, divergence-guard bug fixed")
plt.tight_layout()
save(fig, "figure_comparison")
