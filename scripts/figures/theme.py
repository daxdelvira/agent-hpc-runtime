"""Shared plot theme for the workshop paper figures.

TYPEFACE. The requested face is Times New Roman, which is not installed here.
The chain below falls back to Nimbus Roman and Liberation Serif, both of which
are metric-compatible Times clones -- same widths, same line breaks -- so a
machine that does have the real face will pick it up first and nothing else
shifts.

COLOUR. Stock Gruvbox REGULAR (neutral) accents, unmodified hex -- the standard
set, not the darker "faded" variants an earlier revision used. Published values
are used verbatim, so the costs are stated here rather than absorbed silently:

  * Blue #458588 has OKLCH chroma 0.066, under the ~0.10 floor at which a hue
    stops reading as a hue. It separates from the other slots fine; it just
    reads desaturated rather than blue as such.
  * Yellow #d79921 has only 2.48:1 contrast against white, below the 3:1
    threshold. Filled bars are unaffected -- there is enough area. Thin lines
    and small markers are, which is why lines.linewidth and markersize below
    are set slightly heavier than they would otherwise be.
  * Two pairs collide under protan/deutan simulation: blue vs purple (slots
    1 and 4) and orange vs green (7 and 5). Neither pair may share a frame.
    Purple currently appears only in scale-sweep's lower panel, which is a
    separate axes with no key; green, aqua and orange are unused.

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
# Stock Gruvbox REGULAR (neutral) accents -- the published hex, unmodified.
# This is the standard set, not the darker "faded" variants used previously.
BLUE, RED, YELLOW = "#458588", "#cc241d", "#d79921"
PURPLE, GREEN = "#b16286", "#98971a"
AQUA, ORANGE = "#689d6a", "#d65d0e"
# ORANGE (7) collides with GREEN (5), and BLUE (1) with PURPLE (4), for
# red-green colourblind readers. Do not put either pair in one frame.
CATEGORICAL = [BLUE, RED, YELLOW, PURPLE, GREEN, AQUA, ORANGE]

# Sequential ramp for MAGNITUDE only, never identity. Anchored on the regular
# Gruvbox blue; the pale end is that hue tinted toward the surface, since
# Gruvbox ships no light-background ramp of its own.
SEQUENTIAL = ["#dfe9ea", "#bcd2d4", "#98babd", "#6f9fa3", "#458588", "#356a6d"]

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

        "lines.linewidth": 1.9,
        "lines.markersize": 5.0,

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


def on(bg: str) -> str:
    """Label colour for text sitting ON a filled mark of colour `bg`.

    Needed because the regular Gruvbox yellow is light enough that white text
    on it is unreadable, while the blue wants white. Picking by hand gets it
    wrong the moment a slot is reassigned.

    Not a pure max-contrast rule. The strict WCAG crossover is at luminance
    0.179, and Gruvbox blue sits just above it at 0.198 -- so max-contrast
    would put black on a mid-dark teal, where both choices clear 4:1 anyway
    (4.97 vs 4.23) and the dark one reads badly. The threshold is raised to
    0.35 so a saturated dark fill keeps light text, which is the convention
    and stays well inside the accessible range either way.
    """
    r, g, b = (int(bg.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    return INK if lum > 0.35 else "#ffffff"


def headroom(ax, frac=0.22):
    """Grow the top of the y range so an inside legend clears the marks."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * (1.0 + frac))


def finish(fig, path, **kw):
    fig.savefig(path, **kw)
    plt.close(fig)
    return path
