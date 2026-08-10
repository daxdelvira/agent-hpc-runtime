"""Shared plot theme for the workshop paper figures.

TYPEFACE. The requested face is Times New Roman, which is not installed here.
The chain below falls back to Nimbus Roman and Liberation Serif, both of which
are metric-compatible Times clones -- same widths, same line breaks -- so a
machine that does have the real face will pick it up first and nothing else
shifts.

COLOUR. Data marks use Gruvbox hues, biased to the less-saturated faded/neutral
families. Two of the seeds could not be used as published: Gruvbox's blues and
aquas sit at chroma 0.066-0.082, below the 0.10 floor at which a hue stops
reading as a hue and starts reading as gray. The blue is therefore snapped to
the nearest in-gamut step at the SAME Gruvbox hue angle (215.8 deg), and orange
is dropped entirely -- orange against green is the classic protan/deutan
collision and measured dE 2.4, which is invisible to a red-green colourblind
reader. The surviving order is validated at every prefix length by
scripts/validate_palette.py:

    n=2,3 (every pair)     worst CVD dE 20.9 / 17.4    normal 32.3 / 22.1
    n=4,5 (adjacent pairs) worst CVD dE 18.2 / 14.3    normal 21.7 / 21.5

against a target of 8 and a normal-vision floor of 15. Assign slots IN ORDER
and never cycle; a sixth series folds into "other" or becomes a facet.

Everything that is not data -- text, axes, ticks, frame -- is black on white,
as requested.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots, in fixed assignment order. Do not reorder or cycle.
BLUE, RED, YELLOW, PURPLE, GREEN = "#008da5", "#9c0006", "#b67717", "#8f3f71", "#79740e"
CATEGORICAL = [BLUE, RED, YELLOW, PURPLE, GREEN]

# Sequential ramp: ONE hue, light -> dark, for magnitude only (never identity).
SEQUENTIAL = ["#cfe9ee", "#9ed3dd", "#6bbccb", "#2fa5b8", "#008da5", "#006d80"]

INK = "#000000"          # text, axes, ticks, frame -- black, as requested
GRID = "#d9d9d9"         # hairline, solid, one shade off the surface
MUTED = "#6f6f6f"        # de-emphasised marks and annotation rules
SURFACE = "#ffffff"

SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,

        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",          # never dashed

        "axes.spines.top": False,
        "axes.spines.right": False,

        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,

        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# Single-column and double-column widths for a two-column article, inches.
COL, WIDE = 3.35, 7.0

# Marker shapes, used as SECONDARY encoding so identity never rests on hue
# alone. Same fixed order as CATEGORICAL.
MARKERS = ["o", "s", "^", "D", "v"]


def finish(fig, path, **kw):
    fig.savefig(path, **kw)
    plt.close(fig)
    return path
