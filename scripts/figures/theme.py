"""Shared plot theme for the workshop paper figures.

TYPEFACE. The requested face is Times New Roman, which is not installed here.
The chain below falls back to Nimbus Roman and Liberation Serif, both of which
are metric-compatible Times clones -- same widths, same line breaks -- so a
machine that does have the real face will pick it up first and nothing else
shifts.

COLOUR. Stock Gruvbox, unmodified hex, biased to the less-saturated "faded"
family because the surface is white. An earlier revision substituted two of
these for accessibility reasons; per request the published values are now used
verbatim, and the two costs that substitution was paying for are stated here
rather than silently absorbed:

  * Gruvbox blue #076678 has OKLCH chroma 0.066 and aqua #427b58 has 0.082,
    both under the ~0.10 floor at which a hue stops reading as a hue. They
    still separate fine from the other slots; they just read desaturated
    rather than blue/green as such.
  * Orange #af3a03 against green #79740e is dE 2.4 under protan/deutan
    simulation -- indistinguishable to a red-green colourblind reader. Slots
    5 and 7 must therefore not be used TOGETHER in one figure. Nothing here
    does; keep it that way, or facet instead.

Marker shape is assigned alongside hue in the same fixed order, so series
identity never rests on colour alone and both costs above stay cosmetic.
Assign slots IN ORDER and never cycle.

Everything that is not data -- text, axes, ticks, frame -- is black on white,
as requested.

LEGENDS never sit on top of the data. Bar charts use legend_above(), which
puts the key in the margin over the axes; line charts get explicit headroom.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots, in fixed assignment order. Do not reorder or cycle.
# Stock Gruvbox "faded" accents -- the published hex, unmodified.
BLUE, RED, YELLOW = "#076678", "#9d0006", "#b57614"
PURPLE, GREEN = "#8f3f71", "#79740e"
AQUA, ORANGE = "#427b58", "#af3a03"
# ORANGE is slot 7 and collides with GREEN (slot 5) for red-green colourblind
# readers; do not put both in one figure.
CATEGORICAL = [BLUE, RED, YELLOW, PURPLE, GREEN, AQUA, ORANGE]

# Sequential ramp for MAGNITUDE only, never identity. Anchored on the Gruvbox
# blues (#458588 mid, #076678 dark); the pale end is those hues tinted toward
# the surface, since Gruvbox ships no light-background ramp of its own.
SEQUENTIAL = ["#dbe7e9", "#b3ccd0", "#8ab2b7", "#458588", "#1d6f7d", "#076678"]

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
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def legend_above(ax, ncol=None, pad=0.02, **kw):
    """Put the key in the margin ABOVE the axes, never over the data.

    This is the correct default for bar charts: a bar's meaning is its length
    from the baseline, so a legend floating inside the frame both hides marks
    and invites the reader to measure a bar against the legend box. Line
    charts have slack that matplotlib's `loc="best"` can usually find; bars
    frequently do not.
    """
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("columnspacing", 1.0)
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.0 + pad, 1.0, 0.12),
              mode="expand", borderaxespad=0.0, ncol=ncol or len(h), **kw)


def headroom(ax, frac=0.22):
    """Grow the top of the y range so an inside legend clears the marks."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * (1.0 + frac))


def finish(fig, path, **kw):
    fig.savefig(path, **kw)
    plt.close(fig)
    return path
