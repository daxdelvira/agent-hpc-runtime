"""
plot_utils.py — shared plotting utilities for agent-hpc-runtime experiment figures.

Provides:
  GRV            — Gruvbox Dark color palette dict
  FS             — font-size dict (title, label, tick, annot, small)
  apply_gruvbox_rc(dark_bg=True) — set matplotlib rcParams
  save_figure(fig, outdir, stem, dpi=150) — save PDF + PNG
  load_traces(trace_dir, pattern) -> list[list[dict]]
  extract_tool_sequence(events) -> list[str]
  group_runs_by_run_id(events) -> dict[str, list[dict]]
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Gruvbox Dark palette
# ---------------------------------------------------------------------------

GRV: dict[str, str] = {
    "bg":     "#282828",
    "bg1":    "#3c3836",
    "bg2":    "#504945",
    "fg":     "#ebdbb2",
    "fg2":    "#d5c4a1",
    "gray":   "#a89984",
    "red":    "#fb4934",
    "green":  "#b8bb26",
    "yellow": "#fabd2f",
    "blue":   "#83a598",
    "purple": "#d3869b",
    "orange": "#fe8019",
    "darkred":    "#cc241d",
    "darkgreen":  "#98971a",
    "darkyellow": "#d79921",
    "darkorange": "#d65d0e",
}

FS: dict[str, int | float] = {
    "title": 11,
    "label": 10,
    "tick":   9,
    "annot":  8.5,
    "small":  7.5,
}


# ---------------------------------------------------------------------------
# RC / style helpers
# ---------------------------------------------------------------------------

def apply_gruvbox_rc(dark_bg: bool = True) -> None:
    """Apply Gruvbox-themed rcParams. Call once before creating any figure."""
    base: dict = {
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":  "dejavuserif",
        "xtick.labelsize":   FS["tick"],
        "ytick.labelsize":   FS["tick"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
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


# ---------------------------------------------------------------------------
# Figure I/O
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    outdir: str | Path,
    stem: str,
    dpi: int = 150,
) -> None:
    """Save fig as both PDF and PNG to outdir/stem.{pdf,png}."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    base = str(Path(outdir) / stem)
    fig.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    print(f"Saved {base}.pdf / .png")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces(
    trace_dir: str | Path,
    pattern: str = "*.jsonl",
) -> list[list[dict]]:
    """
    Load all JSONL traces matching *pattern* under *trace_dir*.

    Returns one list-of-events per file.  Empty files are excluded.
    Events that fail JSON parsing are skipped silently.
    """
    paths = sorted(glob.glob(str(Path(trace_dir) / pattern)))
    traces: list[list[dict]] = []
    for path in paths:
        events: list[dict] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if events:
            traces.append(events)
    return traces


def group_runs_by_run_id(events: list[dict]) -> dict[str, list[dict]]:
    """
    Split a flat event list into per-run_id sublists.

    Useful when multiple runs are interleaved in one trace file.
    """
    runs: dict[str, list[dict]] = {}
    for e in events:
        rid = e.get("run_id", "unknown")
        runs.setdefault(rid, []).append(e)
    return runs


# ---------------------------------------------------------------------------
# Tool-sequence extraction
# ---------------------------------------------------------------------------

def extract_tool_sequence(events: list[dict]) -> list[str]:
    """
    Return the ordered list of tool names from *tool_call* events.

    Events are used in the order they appear in the list; callers should
    sort by ``epoch_time`` or ``step`` first if needed.
    """
    return [
        e["payload"]["tool"]
        for e in events
        if e.get("event_type") == "tool_call" and "tool" in e.get("payload", {})
    ]


def extract_llm_predictions(events: list[dict]) -> list[Optional[str]]:
    """
    Return the list of single-step tool predictions from *llm_call* events.

    Each entry is the first element of payload.tool_calls, or None if the
    LLM made no tool prediction at that step.
    """
    preds: list[Optional[str]] = []
    for e in events:
        if e.get("event_type") == "llm_call":
            calls = e.get("payload", {}).get("tool_calls", [])
            preds.append(calls[0] if calls else None)
    return preds
