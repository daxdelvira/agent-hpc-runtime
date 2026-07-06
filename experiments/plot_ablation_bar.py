"""
Ablation bar chart: wall time per condition across four exp3 / lammps=900 runs.
Conditions (real data, all four runs completed on 4× B200 Blackwell):
  full_system       679fb244  3978.9 s
  no_diverg_guard   e45ebe71  4260.7 s
  no_model_prefetch 312ad04e  4475.9 s
  no_plan_extraction 65046dd8  4799.9 s
Outputs: results/figure_ablation_bar.pdf and .png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data ──────────────────────────────────────────────────────────────────────
CONDITIONS = [
    ("full_system",          3978.9,  "#98971a"),   # gruvbox green
    ("no_diverg_guard",      4260.7,  "#d79921"),   # gruvbox yellow
    ("no_model_prefetch",    4475.9,  "#d65d0e"),   # gruvbox orange
    ("no_plan_extraction",   4799.9,  "#cc241d"),   # gruvbox red
]

LABELS = [
    "full\nsystem",
    "w/o divergence\nguard",
    "no model\nprefetch",
    "w/o plan\nextraction",
]

EDGE_COLOR  = "#504945"
GRID_COLOR  = "#e0e0e0"
TEXT_DARK   = "#1d2021"
TEXT_LIGHT  = "#fffdf0"

# ── Layout ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

names, times, colors = zip(*CONDITIONS)
x = np.arange(len(names))
base = 3700  # truncated y-axis baseline

bar_heights = [t - base for t in times]
bars = ax.bar(x, bar_heights, bottom=base, color=colors, edgecolor=EDGE_COLOR,
              linewidth=1.1, width=0.52, zorder=3)

# ── Axes ──────────────────────────────────────────────────────────────────────
y_max = 5050
ax.set_ylim(base, y_max)
ax.set_xlim(-0.55, len(x) - 0.45)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10.5)
ax.set_ylabel("Wall time (s)", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Broken-axis indicator at y=base
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#bbbbbb")
ax.spines["bottom"].set_color("#bbbbbb")

# Small zigzag break marks on left spine at base
zy = base
zx = -0.55
dz = 18
for i in range(3):
    ax.annotate("", xy=(zx, zy + (i + 1) * dz), xytext=(zx, zy + i * dz),
                xycoords="data", textcoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=1))
# Draw the actual break ticks (two small diagonal lines)
kw = dict(transform=ax.get_yaxis_transform(), color="#999999", lw=1.5,
          clip_on=False, zorder=5)
ax.plot([-0.012, 0.012], [base - 12, base + 12], **kw)
ax.plot([-0.012, 0.012], [base - 24, base     ], **kw)

# ── Annotations ───────────────────────────────────────────────────────────────
ref_time = times[0]  # full_system

for i, (bar, t, col) in enumerate(zip(bars, times, colors)):
    cx = bar.get_x() + bar.get_width() / 2

    # Wall time label above bar
    ax.text(cx, t + 18, f"{t:,.0f} s", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=TEXT_DARK)

    # Delta vs full_system (inside bar, near top)
    if i > 0:
        delta = t - ref_time
        label_y = t - (t - base) * 0.13
        ax.text(cx, label_y, f"+{delta:,.0f} s", ha="center", va="top",
                fontsize=8.5, color=TEXT_LIGHT, fontweight="bold")

# ── Title & labels ────────────────────────────────────────────────────────────
ax.set_title(
    "Ablation: proactive model swap benefit\n"
    "AtomAgents screw-dislocation task · 4× NVIDIA B200 · LAMMPS window = 900 s",
    fontsize=11, fontweight="bold", loc="left", pad=10, color=TEXT_DARK,
)

# ── Legend: colour → meaning ──────────────────────────────────────────────────
import matplotlib.patches as mpatches
legend_items = [
    mpatches.Patch(color="#98971a", ec=EDGE_COLOR, label="Full system (proactive swap + all components)"),
    mpatches.Patch(color="#d79921", ec=EDGE_COLOR, label="Divergence guard disabled"),
    mpatches.Patch(color="#d65d0e", ec=EDGE_COLOR, label="Model prefetch disabled (on-demand only)"),
    mpatches.Patch(color="#cc241d", ec=EDGE_COLOR, label="Plan extraction disabled"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=8.5,
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()

out = os.path.join(REPO_ROOT, "results", "figure_ablation_bar")
fig.savefig(out + ".pdf", dpi=150, bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {out}.pdf and .png")
