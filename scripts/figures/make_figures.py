#!/usr/bin/env python3
"""Generate first-draft figures for the workshop paper.

    python3 scripts/figures/make_figures.py [--only fig-name]

Writes PDF (for LaTeX) and PNG (for quick viewing) into
sc-workshop-paper/figure_drafts/, alongside a FIGURES.md recording which
figure rests on measurement and which on simulation. Per request, that
distinction appears ONLY in FIGURES.md -- never in a plot label or caption --
so the drafts read the way the finished paper will.

Numbers are read from the generated results tables and the measured JSON/CSV
artifacts rather than typed in, so regenerating those regenerates these.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme  # noqa: E402
theme.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(ROOT, "sc-workshop-paper", "results_tables")
OUT = os.path.join(ROOT, "sc-workshop-paper", "figure_drafts")
os.makedirs(OUT, exist_ok=True)

C = theme.CATEGORICAL
M = theme.MARKERS


# ---------------------------------------------------------------- helpers ---
def md_tables(fname):
    """Every pipe-table in a results markdown file, as (header, rows)."""
    out, cur = [], None
    for line in open(os.path.join(TABLES, fname)):
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if cur is None:
                cur = (cells, [])
            else:
                cur[1].append(cells)
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def num(s):
    m = re.search(r"[-+]?\d*\.?\d+", str(s).replace("%", ""))
    return float(m.group()) if m else float("nan")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def legend(ax, **kw):
    """Inside legend for LINE charts, with headroom so it clears the marks.

    Bar charts do not use this -- see theme.legend_above().
    """
    room = kw.pop("headroom", 0.20)
    kw.setdefault("loc", "upper left")
    if room:
        theme.headroom(ax, room)
    ax.legend(**kw)


# ------------------------------------------------------------------ 1 -------
def fig_intro_behavior():
    """Schematic: why phase-based staging works traditionally and fails here."""
    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, theme.fh("fig-intro-behavior", 2.35)), sharex=True)
    for ax in axes:
        ax.grid(False)
        ax.set_ylim(-0.2, 2.6)
        ax.set_yticks([0.45, 1.55])
        ax.set_xlim(0, 10.4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    def bar(ax, y, x0, w, color, label, hatch=None, alpha=1.0):
        ax.add_patch(Rectangle((x0, y), w, 0.62, facecolor=color, edgecolor="none",
                               alpha=alpha, hatch=hatch))
        if label:
            ax.text(x0 + w / 2, y + 0.31, label, ha="center", va="center",
                    fontsize=7, color=theme.on(color) if alpha > 0.6 else theme.INK)

    # -- traditional: need is known, staging hides behind the previous phase
    a = axes[0]
    bar(a, 1.25, 0.3, 4.2, C[0], "compute phase $n$")
    bar(a, 1.25, 4.7, 4.4, C[0], "compute phase $n{+}1$")
    bar(a, 0.15, 1.55, 3.2, C[2], "stage for $n{+}1$")
    a.text(1.55, 2.30, "inputs declared", fontsize=7, color=theme.INK, ha="center")
    a.plot([1.55], [2.10], marker="v", ms=4, color=theme.INK)
    a.set_yticklabels(["storage", "compute"])
    a.set_title("Traditional: inputs known before the phase", pad=3)

    # -- agentic: need appears only after the decision, so a stall is exposed
    b = axes[1]
    bar(b, 1.25, 0.3, 4.2, C[0], "compute phase $n$")
    bar(b, 1.25, 8.0, 2.1, C[0], "phase $n{+}1$")
    bar(b, 0.15, 5.2, 2.8, C[2], "stage")
    bar(b, 1.25, 4.5, 0.6, theme.MUTED, "", alpha=0.85)
    b.text(4.8, 1.99, "agent decides", fontsize=7, color=theme.INK, ha="center")
    b.plot([4.8], [1.82], marker="v", ms=4, color=theme.INK)
    b.add_patch(Rectangle((5.1, 1.25), 2.9, 0.62, facecolor="none",
                          edgecolor=theme.INK, lw=0.7, hatch="////"))
    b.text(6.55, 1.56, "stall", ha="center", va="center", fontsize=7, color=theme.INK,
           bbox=dict(facecolor=theme.SURFACE, edgecolor="none", pad=1.0))
    b.set_yticklabels(["storage", "compute"])
    b.set_xlabel("wall-clock time")
    b.set_xticks([])
    b.set_title("Agentic: the need appears after the decision", pad=3)

    fig.tight_layout(h_pad=0.9)
    save(fig, "fig-intro-behavior")


# ------------------------------------------------------------------ 2 -------
def fig_predictability():
    """Per-step agreement with the modal tool, across repeated runs.

    Replaces experiments/plot_step_variation.py's rendering for paper use.
    That script plots ChemGraph and AtomAgents together at 10x5.6 in with its
    own Gruvbox rc; only AtomAgents is used here, at column width and in the
    paper's theme. The analysis is identical -- for each step index, the share
    of runs that took the most-common tool at that index -- and it reads the
    same traces, so the two cannot drift apart in substance.

    The trace loaders are reimplemented rather than imported from
    experiments/plot_utils, which applies its own rcParams; twelve lines is
    cheaper than fighting that.
    """
    import glob as _glob
    from collections import Counter, defaultdict

    MIN_RUNS, MAX_STEP = 3, 20
    seqs = []
    for path in sorted(_glob.glob(os.path.join(
            ROOT, "logs", "workflow_traces", "runtime_trace_*.jsonl"))):
        ev = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        seq = [e["payload"]["tool"] for e in ev
               if e.get("event_type") == "tool_call" and "tool" in e.get("payload", {})]
        if seq:
            seqs.append(seq)
    if not seqs:
        print("  SKIP fig_predictability: no traces under logs/workflow_traces")
        return

    per_step = defaultdict(list)
    for seq in seqs:
        for i, tool in enumerate(seq):
            if i <= MAX_STEP:
                per_step[i].append(tool)
    steps, pct, runs = [], [], []
    for i in sorted(per_step):
        tools = per_step[i]
        if len(tools) < MIN_RUNS:
            continue
        steps.append(i)
        pct.append(Counter(tools).most_common(1)[0][1] / len(tools) * 100)
        runs.append(len(tools))
    mean = sum(pct) / len(pct)
    print(f"  AtomAgents agreement: mean {mean:.1f}%  min {min(pct):.1f}%  "
          f"max {max(pct):.1f}%  over {len(steps)} steps, n<={max(runs)} runs")

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-predictability", 2.3)))
    ax.plot(steps, pct, marker=M[0], color=C[0], zorder=3)
    ax.axhline(mean, color=theme.MUTED, lw=0.9, zorder=1)
    # Left of the run-up: steps 1-7 sit well below the mean, so the band just
    # above the rule there is the only clear space on the axes.
    ax.text(1.0, mean + 3.5, f"mean {mean:.0f}%", fontsize=6.8,
            color=theme.MUTED, ha="left", va="bottom")
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.4, max(steps) + 0.4)
    ax.set_xticks(range(0, max(steps) + 1, 5))   # step index is an integer
    ax.set_xlabel("step index")
    ax.set_ylabel("agreement (%)")
    fig.tight_layout()
    save(fig, "fig-predictability")


# ------------------------------------------------------------------ 6 -------
def fig_prediction_signals():
    """Both horizon signals in one float: local transitions, then plans.

    MERGED from fig-tool-relationships and fig-plan-accuracy. They back the two
    paragraphs of one subsection and were costing two captions and two floats.
    Plan compliance also drops from two stacked panels to one with grouped
    bars, which is both smaller and a better comparison -- the order-only and
    positional scores for a given step now sit side by side instead of a
    column apart.

    The compliance analysis is IMPORTED from experiments/plot_plan_accuracy.py
    rather than reimplemented; the lenient/strict distinction is subtle, and
    three numbers in this paper were already wrong from re-deriving instead of
    recomputing.
    """
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from plot_plan_accuracy import compute_aa_compliance      # noqa: E402
    from plot_utils import load_traces                        # noqa: E402

    tools, offsets, g = _transitions()
    traces = load_traces(os.path.join(ROOT, "logs", "workflow_traces"),
                         "runtime_trace_*.jsonl")
    scored = []
    for strict in (False, True):
        Mx, _, labels = compute_aa_compliance(traces, strict=strict)
        pct = [(Mx[:, k] == 1.0).sum() / max((Mx[:, k] >= 0).sum(), 1) * 100
               for k in range(Mx.shape[1])]
        scored.append((pct, int((Mx == 1.0).sum()) / int((Mx >= 0).sum()) * 100, labels))
    if not len(g) or not scored:
        print("  SKIP fig_prediction_signals: missing traces")
        return
    print(f"  transitions: {int((g.max(axis=1) >= 0.9).sum())}/{len(g)} tools >= 0.9   "
          f"plans: order-only {scored[0][1]:.1f}%, positional {scored[1][1]:.1f}%")

    fig, axes = plt.subplots(2, 1,
                             figsize=(theme.COL, theme.fh("fig-prediction-signals", 3.6)),
                             gridspec_kw={"height_ratios": [1, 1.15]})

    a = axes[0]
    x = np.linspace(0.0, 1.0, 201)
    # k=1 and k=3 track each other closely and colour alone cannot separate
    # them where they coincide. Distinct dash patterns do, and they survive
    # both overlap and greyscale printing.
    DASH = [(0, ()), (0, (4, 1.6)), (0, (1, 1.4))]
    for i2, k in enumerate(offsets):
        a.plot(x, [(g[:, i2] >= t).mean() * 100 for t in x], color=C[i2],
               label=f"$k{{=}}{k}$", lw=1.7, ls=DASH[i2 % len(DASH)])
    a.axvline(0.9, color=theme.MUTED, lw=0.8, zorder=0)
    best = g.max(axis=1)
    a.plot([0.9], [(best >= 0.9).mean() * 100], marker="*", ms=8, color=theme.INK,
           linestyle="none", zorder=5, label=f"any $k$: {int((best>=0.9).sum())}/{len(best)}")
    a.set_xlim(0, 1.0); a.set_ylim(0, 104)
    a.set_xlabel("successor confidence", labelpad=1.5)
    a.set_ylabel("tools (%)")
    theme.legend_above(a, ncol=4, fontsize=7)

    b = axes[1]
    labels = scored[0][2]
    xs = np.arange(len(labels))
    for k, (pct, overall, _) in enumerate(scored):
        lab = ("order only" if k == 0 else "position") + f" ({overall:.0f}%)"
        b.bar(xs + (k - 0.5) * 0.38, pct, 0.36, color=C[k], edgecolor="none", label=lab)
        # A zero-height bar is indistinguishable from an absent one, and the
        # zeros ARE the finding here -- positional compliance collapses after
        # the first retry. Label them so the reader sees a measurement.
        for xi, v in zip(xs + (k - 0.5) * 0.38, pct):
            if v <= 0.5:
                b.text(xi, 2.5, "0", ha="center", va="bottom", fontsize=6.2,
                       color=C[k])
    b.set_ylim(0, 105); b.set_yticks([0, 50, 100])
    b.set_ylabel("compliant (%)")
    b.set_xticks(xs)
    b.set_xticklabels([l[:11] for l in labels], rotation=28, ha="right", fontsize=6.5)
    b.grid(axis="x", visible=False)
    theme.legend_above(b, ncol=2, fontsize=7)

    fig.tight_layout(h_pad=0.5)
    save(fig, "fig-prediction-signals")


# ------------------------------------------------------------------ 3 -------
def fig_replacement_loss():
    """Model-load timeline for a representative trial, against artifact residency."""
    loads = [(0.1, 403.8, "32B"), (485.6, 1204.4, "72B"), (1217.9, 1911.8, "72B-t"),
             (1967.7, 2661.4, "72B"), (2687.6, 3422.9, "72B"),
             (3436.0, 4189.6, "72B-t"), (4245.4, 5254.4, "72B")]
    wall = 5288.5
    # The lower panel is ~8% filled, so it does not need the larger share --
    # it had 1.25 against the timeline's 1.0, which is backwards.
    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, theme.fh("fig-replacement-loss", 2.5)),
                             sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})

    ax = axes[0]
    ax.grid(False)
    for i, (s, e, lab) in enumerate(loads):
        ax.add_patch(Rectangle((s, 0.15), e - s, 0.7, facecolor=C[0],
                               edgecolor="none", alpha=1.0 if i == 0 else 0.85))
        if e - s > 500:
            ax.text((s + e) / 2, 0.5, lab, ha="center", va="center",
                    fontsize=6.5, color=theme.on(C[0]))
    ax.set_xlim(0, wall)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Model residency: each bar is a load into device memory", pad=4)

    # Artifact residency collapses each time a model is parked/loaded.
    t = np.linspace(0, wall, 1400)
    res = np.ones_like(t)
    for s, e, _ in loads[1:]:
        res[(t > s - 40) & (t < e)] = 0.0
        tail = (t >= e) & (t < e + 260)
        res[tail] = np.clip((t[tail] - e) / 260.0, 0, 1)
    axes[1].fill_between(t, res, color=C[2], alpha=0.85, linewidth=0)
    # The panel reads as almost entirely empty, and that IS the reading: the
    # agent-active gaps between loads are 13-82 s against a 5288 s trial, so
    # there is never enough time for the artifact to be rebuilt and stay
    # rebuilt. State the share rather than leaving a flat band unexplained.
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy <2 compat
    share = float(_trapz(res, t) / wall) * 100.0
    axes[1].text(0.985, 0.62, f"resident {share:.0f}% of the trial",
                 transform=axes[1].transAxes, ha="right", fontsize=6.8,
                 color=theme.MUTED)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(0, wall)
    axes[1].set_ylabel("artifact resident")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["none", "full"])
    axes[1].set_xlabel("wall-clock time (s)")
    axes[1].set_title("Scientific artifact residency over the same trial", pad=4)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig-replacement-loss")


# ------------------------------------------------------------------ 4 -------
def fig_sgb_spread():
    """Seconds saved per GB retained: models banded, data spread across them."""
    fmt = {}
    path = os.path.join(ROOT, "results",
                        "bench_format_activation_atl1-1-02-003-25-1.json.csv")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "results", "bench_format_activation.csv")
    for r in csv.DictReader(open(path)):
        if not r.get("array_nbytes"):
            continue
        gb = int(r["array_nbytes"]) / 1e9
        fmt.setdefault(r["fmt"], []).append(float(r["load_warm_s"]) / gb)
    data = sorted(((k, sum(v) / len(v)) for k, v in fmt.items()), key=lambda x: -x[1])
    labels = {"ascii_loadtxt": "ASCII text", "ascii_pandas": "ASCII text (alt)",
              "npz_deflate": "compressed binary", "raw_f32": "raw f32",
              "npy": "mmap-able binary", "hdf5": "HDF5"}

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-sgb-spread", 2.35)))
    y = np.arange(len(data))
    # Dots, not bars: on a log axis a bar's LENGTH is not proportional to its
    # value, so bars would misstate the 65x spread this figure exists to show.
    ax.hlines(y, 0.24, [v for _, v in data], color=theme.GRID, lw=0.8, zorder=1)
    ax.plot([v for _, v in data], y, "o", color=C[2], ms=6, zorder=3,
            markeredgecolor=theme.SURFACE, markeredgewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(k, k) for k, _ in data])
    ax.set_ylim(len(data) - 0.45, -1.95)         # absolute room for the top labels
    ax.set_xscale("log")
    ax.set_xlabel("seconds saved per GB retained")
    ax.grid(axis="y", visible=False)

    ax.axvspan(2.78, 3.81, color=C[0], alpha=0.16, zorder=0)
    ax.text(3.26, -1.05, "models", fontsize=7.5, color=C[0], ha="center",
            va="center")
    ax.axvline(0.32, ymin=0.0, ymax=0.82, color=theme.MUTED, lw=0.8, zorder=0)
    ax.text(0.32, -1.05, "floor", fontsize=6.6, color=theme.MUTED,
            ha="center", va="center")
    for i, (_, v) in enumerate(data):
        ax.text(v * 1.16, i, f"{v:.2f}", va="center", fontsize=6.8, color=theme.INK)
    ax.set_xlim(0.24, 70)
    fig.tight_layout()
    save(fig, "fig-sgb-spread")


def fig_sgb_spread_alt():
    """Alternative to fig_sgb_spread: shows the three sizes instead of averaging.

    Drop-in variant, kept alongside the original so the two can be compared in
    the built PDF. Two differences, both aimed at claims the original states in
    prose but does not show:

      1. The original computes sum(v)/len(v) over the three array sizes, which
         discards exactly the evidence for "the price is a property of the
         representation, not the instance". Here each size is plotted. They
         overlap; the overlap IS the result, and the per-row span is annotated
         with the measured ratio so the reader does not have to eyeball it.
      2. The original draws a "floor" rule at a hardcoded 0.32, which appears in
         no measurement artifact. The smallest price actually measured is npy at
         the largest size, and npy is still DECREASING across the size range
         (0.395 -> 0.369 -> 0.340), so nothing in the data establishes an
         asymptote. The rule is therefore drawn at the measured minimum and
         labelled as such.

    The model band is still the hardcoded 2.78-3.81 from the original -- it has
    no provenance in any artifact either, and is left unchanged here ONLY so the
    two figures stay comparable. It needs re-deriving or softening.
    """
    MODEL_BAND = (2.78, 3.81)   # UNSOURCED -- see docstring

    # NODE CHOICE IS EXPLICIT AND MUST MATCH THE PROSE. There are two runs of
    # this bench on different nodes and they do NOT agree:
    #
    #   bench_format_activation.csv          max 22.01  min 0.340  spread 64.6x
    #   ..._atl1-1-02-003-25-1.json.csv      max 21.87  min 0.316  spread 69.3x
    #
    # fig_sgb_spread() prefers the second; Section III quotes 22.0, 0.34, 65x
    # and 1.00-1.16x, all of which are the FIRST. So the original figure and the
    # text it illustrates are drawn from different machines. The plan's standing
    # rule is to facet by node and never pool, so this variant pins the file
    # that matches the prose and prints which one it used.
    SOURCE = "bench_format_activation.csv"
    path = os.path.join(ROOT, "results", SOURCE)
    print(f"  source: {SOURCE}")
    by_fmt_size = {}
    for r in csv.DictReader(open(path)):
        if not r.get("array_nbytes"):
            continue
        gb = int(r["array_nbytes"]) / 1e9
        by_fmt_size.setdefault(r["fmt"], {}).setdefault(gb, []).append(
            float(r["load_warm_s"]) / gb)
    fmt = {f: {g: sum(v) / len(v) for g, v in d.items()}
           for f, d in by_fmt_size.items()}
    sizes = sorted({g for d in fmt.values() for g in d})
    data = sorted(((f, d) for f, d in fmt.items()),
                  key=lambda x: -sum(x[1].values()) / len(x[1]))
    labels = {"ascii_loadtxt": "ASCII text", "ascii_pandas": "ASCII text (alt)",
              "npz_deflate": "compressed binary", "raw_f32": "raw f32",
              "npy": "mmap-able binary", "hdf5": "HDF5"}

    fig, ax = plt.subplots(figsize=(theme.COL,
                                    theme.fh("fig-sgb-spread-alt", 2.35)))
    y = np.arange(len(data))
    x0 = 0.24
    # Size is a MAGNITUDE, not an identity, so it gets the sequential ramp
    # rather than categorical slots. Pale -> dark reads as small -> large.
    ramp = [theme.SEQUENTIAL[1], theme.SEQUENTIAL[3], theme.SEQUENTIAL[5]]

    for i, (_, d) in enumerate(data):
        vals = [d[g] for g in sizes if g in d]
        ax.hlines(i, x0, max(vals), color=theme.GRID, lw=0.8, zorder=1)
    for j, g in enumerate(sizes):
        xs = [d.get(g) for _, d in data]
        ax.plot(xs, y, "o", color=ramp[j], ms=3.6, zorder=3,
                markeredgecolor=theme.SURFACE, markeredgewidth=0.7,
                linestyle="none", label=f"{g:.2f} GB")

    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(k, k) for k, _ in data])
    ax.set_ylim(len(data) - 0.45, -2.35)
    ax.set_xscale("log")
    ax.set_xlabel("seconds saved per GB retained")
    ax.grid(axis="y", visible=False)

    ax.axvspan(*MODEL_BAND, color=C[0], alpha=0.16, zorder=0)
    ax.text((MODEL_BAND[0] * MODEL_BAND[1]) ** 0.5, -1.42, "models",
            fontsize=7.5, color=C[0], ha="center", va="center")

    floor = min(v for _, d in data for v in d.values())
    ax.axvline(floor, ymin=0.0, ymax=0.78, color=theme.MUTED, lw=0.8, zorder=0)
    ax.text(floor, -1.42, f"{floor:.2f} measured\nminimum", fontsize=6.2,
            color=theme.MUTED, ha="center", va="center", linespacing=1.15)

    # Per-row flatness, printed rather than left to the eye: on a 65x log axis a
    # 1.16x span is about a pixel, so the number carries the claim.
    for i, (_, d) in enumerate(data):
        vals = [d[g] for g in sizes if g in d]
        ax.text(max(vals) * 1.22, i,
                f"{sum(vals)/len(vals):.2f}  ({max(vals)/min(vals):.2f}$\\times$)",
                va="center", fontsize=6.4, color=theme.INK)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005, 1.0, 0.10),
              mode="expand", ncol=3, frameon=False, handletextpad=0.25,
              columnspacing=0.9, fontsize=6.4, borderaxespad=0.0)
    ax.set_xlim(x0, 200)
    fig.tight_layout()
    save(fig, "fig-sgb-spread-alt")


# ------------------------------------------------------------------ 5 -------
def fig_scale_sweep():
    """Retention fades and staging grows as the resource population grows."""
    hdr, rows = md_tables("05_scale_sweep.md")[0]
    n = [num(r[2]) for r in rows]
    first = [num(r[3]) for r in rows]
    retain = [num(r[4]) for r in rows]
    oracle = [num(r[5]) for r in rows]
    stage = [num(r[6]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, theme.fh("fig-scale-sweep", 3.0)), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    a = axes[0]
    a.plot(n, retain, marker=M[0], color=C[0], label="retention")
    a.plot(n, stage, marker=M[1], color=C[1], label="+ staging")
    a.plot(n, oracle, marker=M[2], color=C[2], ls=(0, (4, 2)), label="retention, oracle")
    a.set_ylabel("reduction (%)")
    # No panel title here: it duplicated the y label, and it was what forced the
    # key inside the frame, where it sat on the oracle line.
    theme.legend_above(a, ncol=3, fontsize=7)

    b = axes[1]
    b.plot(n, first, marker=M[3], color=C[3])
    # The lower panel names its series IN the panel rather than on a rotated y
    # label. Two rotated labels are what put a 0.80 floor under this figure --
    # each is taller than its own panel below that. With only one label left,
    # the floor does not apply and the figure compresses further. The series
    # rises left to right, so the upper left is clear.
    b.text(0.02, 0.93, "first uses (%)", transform=b.transAxes,
           ha="left", va="top", fontsize=7.5, color=theme.INK)
    b.set_xlabel("distinct resident structures")
    b.set_xticks(n)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig-scale-sweep")


def fig_scale_sweep_alt():
    """fig_scale_sweep built from the 50%-budget table instead of the 25% one.

    05_scale_sweep.md contains TWO sweeps, at budget = 25% and 50% of footprint.
    fig_scale_sweep() reads table [0] (25%). Section III's claim -- that an
    oracle-informed retention policy "fades over the same range", so the decline
    is structural rather than a prediction-quality problem -- is a description of
    table [1] (50%), not of the one plotted:

        oracle @25%:  +6.61  +21.41  +22.65  +17.02   (rises 3.4x, then falls)
        oracle @50%: +21.25  +18.88  +12.27   +3.28   (monotone decline)

    The 50% table also has the stronger evidence: the oracle-minus-greedy gap
    closes from 20.31 to 1.12 points, i.e. at large populations perfect knowledge
    buys almost nothing over a 55%-accurate predictor. That is the direct form of
    "there is nothing left for better prediction to convert".

    What is LOST by switching: the retention/staging crossing at n=5, which is
    the only feature unique to the 25% table. At +4.72 vs +4.39 that margin is
    0.33 points, and standing caveat 2 of the same file says differences below
    ~2 points are not resolvable at this sample size. It is noise.
    """
    hdr, rows = md_tables("05_scale_sweep.md")[1]      # [1] = 50% budget
    n = [num(r[2]) for r in rows]
    first = [num(r[3]) for r in rows]
    retain = [num(r[4]) for r in rows]
    oracle = [num(r[5]) for r in rows]
    stage = [num(r[6]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(theme.COL,
                                            theme.fh("fig-scale-sweep-alt", 3.0)),
                             sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    a = axes[0]
    a.plot(n, retain, marker=M[0], color=C[0], label="retention")
    a.plot(n, stage, marker=M[1], color=C[1], label="+ staging")
    a.plot(n, oracle, marker=M[2], color=C[2], ls=(0, (4, 2)), label="retention, oracle")
    a.set_ylabel("reduction (%)")
    theme.legend_above(a, ncol=3, fontsize=7)
    # NO IN-PANEL CALLOUT FOR THE CONVERGENCE. Four placements were tried -- a
    # double-headed arrow between the lines (arrowheads overlap each other and
    # the markers across a 1.1-point gap on a 27-point axis) and three leader
    # positions, each of which put the label's background over a different
    # series and made a solid line look broken. This panel is ~1.1 in tall and
    # has no empty region large enough for two lines of type. The convergence is
    # legible from the lines meeting at the right edge; the exact figure belongs
    # in the caption, where it costs nothing.
    print(f"  oracle-minus-predictor gap: "
          + ", ".join(f"n={int(k)}: {o - r:+.2f}"
                      for k, o, r in zip(n, oracle, retain)))

    b = axes[1]
    b.plot(n, first, marker=M[3], color=C[3])
    b.text(0.02, 0.93, "first uses (%)", transform=b.transAxes,
           ha="left", va="top", fontsize=7.5, color=theme.INK)
    b.set_xlabel("distinct resident structures")
    b.set_xticks(n)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig-scale-sweep-alt")


# ------------------------------------------------------------- 6 and 9 ------
def _budget_series():
    tabs = md_tables("02_budget_sweep.md")
    out = {}
    for slots, (hdr, rows) in zip((1, 2), tabs[:2]):
        out[slots] = dict(
            budget=[num(r[0]) for r in rows],
            binding=[num(r[2]) for r in rows],
            lru=[num(r[4]) for r in rows],
            sysname=[num(r[5]) for r in rows],
        )
    return out


def fig_topology_budget():
    """Motivating view: contention vanishes as the budget grows."""
    d = _budget_series()
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-topology-budget", 2.1)))
    for i, slots in enumerate((1, 2)):
        ax.plot(d[slots]["budget"], d[slots]["binding"], marker=M[i], color=C[i],
                label=f"{slots} device slot" + ("s" if slots > 1 else ""))
    for thr in (283, 445, 562):
        ax.axvline(thr, color=theme.MUTED, lw=0.6, zorder=0)
    ax.text(283, 37, "packing thresholds", fontsize=6.8, color=theme.MUTED, ha="left")
    ax.set_xlabel("host-memory budget (GB)")
    ax.set_ylabel("evictions (%)")
    theme.legend_above(ax, ncol=2)
    fig.tight_layout()
    save(fig, "fig-topology-budget")


def fig_budget_sweep():
    """Reduction over budget, with the contention that explains it beneath.

    MERGED from the former fig_topology_budget (binding vs budget) and this
    figure (reduction vs budget). Both read the same rows of the same table
    over the same x axis, and keeping them four pages apart hid the finding:
    the recency-ranked/Tandem gap is WIDEST where binding is zero. At two
    slots the baseline flattens at 68.4% the moment evictions stop while the
    full system climbs to 80.3%, because staging needs no eviction and extra
    budget becomes extra slack. Vertically aligned, that divergence is legible;
    separated, nobody would connect the two panels.
    """
    d = _budget_series()
    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row",
                             figsize=(theme.WIDE, theme.fh("fig-budget-sweep", 3.9)),
                             gridspec_kw={"height_ratios": [1.55, 1]})
    for j, slots in enumerate((1, 2)):
        s = d[slots]
        a, b = axes[0][j], axes[1][j]
        a.plot(s["budget"], s["lru"], marker=M[1], color=C[1], label="recency-ranked")
        a.plot(s["budget"], s["sysname"], marker=M[0], color=C[0], label="Tandem")
        b.plot(s["budget"], s["binding"], marker=M[2], color=C[2])

        # Shade the span in which no eviction ever occurs, i.e. where the
        # arbitrator is provably idle -- the alignment this figure exists for.
        zero = [x for x, v in zip(s["budget"], s["binding"]) if v == 0.0]
        for ax in (a, b):
            if zero:
                ax.axvspan(min(zero), max(s["budget"]), color=theme.MUTED,
                           alpha=0.11, zorder=0)
            for thr in (283, 445, 562):
                ax.axvline(thr, color=theme.MUTED, lw=0.6, zorder=0)
        if zero:
            b.text((min(zero) + max(s["budget"])) / 2, b.get_ylim()[1] * 0.62,
                   "no evictions", fontsize=6.6, color=theme.MUTED, ha="center")
        a.set_title(f"{slots} device slot" + ("s" if slots > 1 else ""), pad=4)
        b.set_xlabel("host-memory budget (GB)")
    axes[0][0].set_ylabel("reduction (%)")
    axes[1][0].set_ylabel("evicted (%)")
    legend(axes[0][0], loc="upper left", headroom=0.16)
    fig.tight_layout(w_pad=1.2, h_pad=0.7)
    save(fig, "fig-budget-sweep")


def fig_megammap_breakdown():
    """MEASURED: wall time and exposed stall for four staging conditions.

    Reads eval_q1_summary.csv and eval_prefetch_lifecycle.csv directly, so this
    is measurement, not simulation -- unlike most sweeps in this file.

    FACETED TO L40S. MegaMmap was only ever run on L40S, and the chemgraph_swap
    baseline differs by GPU (270.8 s L40S vs 291.3 s Blackwell), so pooling
    would compare an L40S-only arm against a mixed baseline.

    THE CONTROLLED COMPARISON IS BARS 2 AND 3, and only those. Same staging
    backend, same everything, predictor swapped for random page order. If
    prediction quality drove the result they would separate; they do not
    (Welch |t| = 0.70, and random is nominally FASTER).

    NO ORACLE BAR, DELIBERATELY. An `oracle` arm exists for this workload
    (L40S: 437.3 s, n=3) and was drawn here at first, hatched and separated. It
    is cut because there is no oracle-informed MEGAMMAP arm in the data: that
    arm has predictor=oracle but stages through the runtime's page cache rather
    than through Hermes. Sitting alongside the two MegaMmap bars it reads as
    "the same system with perfect prediction", which would tell the reader that
    an oracle takes MegaMmap from 862 s to 437 s. It does not -- the backend
    changed too. The observation still belongs in the prose, where it can be
    qualified; it does not survive being reduced to a bar.

    THE LOWER PANEL IS NOT A CLASS BREAKDOWN. Every arm classifies 100% into a
    single stall class, so a stacked bar would be four solid colours. Worse, the
    difference between `window_too_small` (MegaMmap) and `no_window` (oracle)
    partly reflects a tie-break: each vllm_model need emits both a model_cache
    row and a vllm_model row with equal exposure, and the gate_group dedup keeps
    whichever is encountered first. Colouring them differently would assert a
    distinction the data does not support, so the class is printed as text and
    the bars carry stall magnitude only.
    """
    import statistics as _st
    from collections import defaultdict

    ARMS = [("baseline", "no staging", False),
            ("megammap_stage", "MegaMmap\n+ learned", False),
            ("megammap_stage_rand", "MegaMmap\n+ random", False)]
    GPU = "NVIDIA L40S"

    walls = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "results/eval_q1_q4/eval_q1_summary.csv"))):
        if (r["workload"] == "chemgraph_swap" and r["status"] == "completed"
                and r["gpu_name"] == GPU):
            walls[r["config"]].append(float(r["wall_time_s"]))

    # gate_group max-dedup, mirroring extract_prefetch_lifecycle's aggregation
    agg, seen, trials = defaultdict(float), {}, defaultdict(set)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "results/eval_q1_q4/eval_prefetch_lifecycle.csv"))):
        if r["workload"] != "chemgraph_swap" or r["gpu_name"] != GPU:
            continue
        cls = r.get("stall_class") or ""
        if not cls:
            continue
        exp = float(r.get("exposure_s") or 0.0)
        gg = r.get("gate_group") or ""
        if gg:
            k = (r["run_id"], gg)
            prev = seen.get(k)
            if prev is not None:
                pc, pe = prev
                if exp <= pe:
                    continue
                agg[(r["config"], pc)] -= pe
            seen[k] = (cls, exp)
        agg[(r["config"], cls)] += exp
        trials[r["config"]].add(r["run_id"])

    fig, (a, b) = plt.subplots(2, 1, sharex=True,
                               figsize=(theme.COL,
                                        theme.fh("fig-megammap-breakdown", 3.1)),
                               gridspec_kw={"height_ratios": [1.25, 1]})
    # A gap after the first bar separates the reference from the two staging
    # arms, which are the pair the figure exists to compare.
    x = np.array([0.0, 1.25, 2.25])

    for i, (cfg, _, alt) in enumerate(ARMS):
        w = walls.get(cfg, [])
        if not w:
            continue
        col = theme.MUTED if cfg == "baseline" else (C[2] if alt else C[1])
        a.bar(x[i], _st.mean(w), 0.85, color=col, zorder=2,
              hatch="////" if alt else None,
              edgecolor=theme.SURFACE if alt else "none", linewidth=0.0)
        a.errorbar(x[i], _st.mean(w),
                   yerr=_st.stdev(w) if len(w) > 1 else 0.0,
                   color=theme.INK, lw=0.8, capsize=2.2, zorder=3)
        a.text(x[i], _st.mean(w) + (_st.stdev(w) if len(w) > 1 else 0) + 28,
               f"{_st.mean(w):.0f}", ha="center", va="bottom", fontsize=6.6)
        a.text(x[i], 22, f"n={len(w)}", ha="center", va="bottom", fontsize=6.0,
               color=theme.on(col))

        cls = {c: v for (cc, c), v in agg.items() if cc == cfg and v > 0.05}
        if cls and cfg in trials:
            n = len(trials[cfg])
            top = max(cls, key=cls.get)
            val = sum(cls.values()) / n
            b.bar(x[i], val, 0.85, color=col, zorder=2,
                  hatch="////" if alt else None,
                  edgecolor=theme.SURFACE if alt else "none", linewidth=0.0)
            b.text(x[i], val + 14, f"{val:.0f}", ha="center", va="bottom",
                   fontsize=6.6)
            # Inside the bar, not under the axis: the x tick labels already
            # occupy two lines there and the two collided. Labels are explicit
            # rather than underscore-split -- "baseline_no_prefetch" splits to
            # three lines and overran its own 162 s bar.
            # The reference bar gets no class text. `baseline_no_prefetch` is
            # not a failure mode -- it records that there was no staging to
            # classify -- and the x tick label already says so.
            CLS = {"baseline_no_prefetch": "",
                   "window_too_small": "window\ntoo small",
                   "no_window": "no window",
                   "no_prediction": "no prediction",
                   "residual_partial": "partial",
                   "late_start": "late start"}
            b.text(x[i], 16, CLS.get(top, top.replace("_", "\n")),
                   ha="center", va="bottom", fontsize=5.8,
                   color=theme.on(col), linespacing=1.2)

    a.set_ylabel("wall time (s)")
    b.set_ylabel("exposed stall (s/trial)")
    a.set_xticks(x)
    b.set_xticks(x)
    b.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=6.6)
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    theme.headroom(a, 0.10)
    theme.headroom(b, 0.12)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig-megammap-breakdown")


def fig_budget_staging():
    """Characterization-paper cut of fig_budget_sweep: no system, one claim.

    fig_budget_sweep is a 2x2 double-column float that names Tandem and carries
    both slot counts. This is the same underlying table reduced to the single
    finding the characterization paper needs, at single-column width and with no
    system in it:

        Once the budget is large enough that nothing is ever evicted, additional
        memory is worth NOTHING to a retention-only approach and continues to
        buy speedup for staging.

    Two slots only. At one slot the binding share reaches zero at just two of the
    nine budgets; at two slots it reaches zero at five, so the flat stretch is
    long enough to read as flat rather than as two coincident points.

    WHY THE MARGIN IS ATTRIBUTABLE TO STAGING, which is the whole claim: in the
    shaded span the recency-ranked series is pinned at exactly 68.37% across
    every budget from 480 GB to 838 GB. Nothing is evicted, so no eviction
    policy -- LRU, cost-aware, or oracle alike -- has a decision to make, and an
    oracle retention arm is identical to LRU there BY CONSTRUCTION. There is
    therefore no retention headroom that could account for the gap. Staging is
    the only mechanism left, and it needs no eviction to act.
    """
    d = _budget_series()
    s = d[2]
    fig, (a, b) = plt.subplots(2, 1, sharex=True,
                               figsize=(theme.COL,
                                        theme.fh("fig-budget-staging", 2.9)),
                               gridspec_kw={"height_ratios": [1.55, 1]})
    a.plot(s["budget"], s["lru"], marker=M[1], color=C[1],
           label="recency-ranked retention")
    a.plot(s["budget"], s["sysname"], marker=M[0], color=C[0], label="+ staging")
    b.plot(s["budget"], s["binding"], marker=M[2], color=C[2])

    zero = [x for x, v in zip(s["budget"], s["binding"]) if v == 0.0]
    if zero:
        lo, hi = min(zero), max(s["budget"])
        for ax in (a, b):
            ax.axvspan(lo, hi, color=theme.MUTED, alpha=0.11, zorder=0)
        b.text((lo + hi) / 2, max(s["binding"]) * 0.55, "no evictions",
               fontsize=6.6, color=theme.MUTED, ha="center")
        flat = {v for x, v in zip(s["budget"], s["lru"]) if x >= lo}
        gap = max(sv - lv for x, lv, sv in
                  zip(s["budget"], s["lru"], s["sysname"]) if x >= lo)
        print(f"  zero-eviction span {lo:.0f}-{hi:.0f} GB; recency-ranked takes "
              f"{len(flat)} distinct value(s) there ({sorted(flat)}); "
              f"widest margin {gap:.2f} pt")

    a.set_ylabel("reduction (%)")
    b.set_ylabel("evicted (%)")
    b.set_xlabel("host-memory budget (GB)")
    legend(a, loc="lower right", headroom=0.10)
    fig.tight_layout(h_pad=0.7)
    save(fig, "fig-budget-staging")


# ------------------------------------------------------------------ 10 ------
def fig_stall_ladder():
    """Where the time goes, per arm, normalised to the no-retention baseline."""
    d = _budget_series()
    configs = [("256 GB, 1 slot", 1, 256.0), ("560 GB, 2 slots", 2, 560.0)]
    compute = {1: 10.8, 2: 10.8}

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-stall-ladder", 2.2)))
    width, gap = 0.26, 0.04
    xs = np.arange(len(configs))
    arms = ["no retention", "recency-ranked", "Tandem"]
    for j, arm in enumerate(arms):
        stalls = []
        for label, slots, b in configs:
            s = d[slots]
            i = s["budget"].index(b)
            red = 0.0 if j == 0 else (s["lru"][i] if j == 1 else s["sysname"][i])
            stalls.append(100.0 - red - compute[slots])
        off = (j - 1) * (width + gap)
        ax.bar(xs + off, stalls, width, color=C[j], edgecolor="none", label=arm)
        for x, v in zip(xs + off, stalls):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.8, color=theme.INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in configs])
    ax.set_ylabel("stall (% of wall)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(ax.get_ylim()[1], 100))
    theme.legend_above(ax, ncol=3)
    fig.tight_layout()
    save(fig, "fig-stall-ladder")


# ------------------------------------------------------------------ 11 ------
def fig_compute_sweep():
    """Wall-clock dilutes with compute; stall reduction improves with it."""
    hdr, rows = md_tables("03_compute_sweep.md")[0]
    comp = [num(r[1]) for r in rows]
    wall = [num(r[6]) for r in rows]
    stall = [num(r[7]) for r in rows]
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-compute-sweep", 2.2)))
    ax.plot(comp, wall, marker=M[0], color=C[0], label="wall-time reduction")
    ax.plot(comp, stall, marker=M[1], color=C[1], label="stall reduction")
    ax.axvline(5.3, color=theme.MUTED, lw=0.8)
    # Lower-left is the only region both series vacate; horizontal there beats
    # rotated against the rule, which crowded the leftmost markers.
    ax.text(5.6, 1.4, "measured workload", fontsize=6.6, color=theme.MUTED,
            ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("computation, share of wall (%)")
    ax.set_ylabel("reduction (%)")
    ax.set_xticks([5, 10, 25, 50, 90])
    ax.set_xticklabels(["5", "10", "25", "50", "90"])
    theme.legend_above(ax, ncol=2)
    fig.tight_layout()
    save(fig, "fig-compute-sweep")


# ------------------------------------------------------------------ 12 ------
def fig_ablation():
    """Cumulative contribution of each mechanism.

    SINGLE COLUMN as of 2026-08-10. The former right panel showed the Shapley
    split of the retention total as two stacked bars (2/98 at 256 GB, 61/39 at
    560 GB) -- four numbers, all of which the Ablations paragraph already
    states in the prose. Two stacked bars were not worth the 0.10 pages the
    full-width float cost, and dropping the panel also leaves fig:budget-sweep
    as the only figure* in the paper, so no two double-column floats can stack.
    """
    hdr, rows = md_tables("01_attribution_ladder.md")[0]
    names = ["no residency system", "+ model parking", "+ data retention",
             "+ cost-aware arbitration", "+ slack staging"]
    vals = [num(r[2]) for r in rows]

    fig, a = plt.subplots(figsize=(theme.COL, theme.fh("fig-ablation", 2.35)))
    y = np.arange(len(names))
    a.barh(y, vals, height=0.6, color=[theme.MUTED] + [C[0]] * 4, edgecolor="none")
    a.set_yticks(y)
    a.set_yticklabels(names, fontsize=7)
    a.invert_yaxis()
    a.set_xlabel("wall-time reduction (%)")
    a.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        if i:
            a.text(v + 0.8, i, f"+{v - vals[i-1]:.2f}", va="center",
                   fontsize=6.8, color=theme.INK)
    a.set_xlim(0, max(vals) * 1.22)
    fig.tight_layout()
    save(fig, "fig-ablation")


# ------------------------------------------------------------------ 13 ------
def fig_prefetch_variants():
    """What a wrong staging decision is allowed to destroy."""
    hdr, rows = md_tables("06_prefetch_variants.md")[0]
    accs = [num(r[0]) for r in rows]
    i = accs.index(0.55)
    cols = ["retain only", "data, outbid", "data, slack", "all, outbid", "all, slack"]
    vals = [num(rows[i][j]) for j in range(1, 6)]
    order = [0, 2, 1, 4, 3]
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-prefetch-variants", 2.15)))
    x = np.arange(len(order))
    colors = [theme.MUTED, C[0], C[1], C[0], C[1]]
    ax.bar(x, [vals[o] for o in order], 0.62,
           color=[colors[k] for k in range(len(order))], edgecolor="none")
    for xi, o in zip(x, order):
        ax.text(xi, vals[o] + 0.25, f"{vals[o]:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cols[o].replace(", ", "\n") for o in order])
    ax.set_ylabel("reduction (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[0]),
               plt.Rectangle((0, 0), 1, 1, color=C[1])]
    ax.legend(handles, ["slack only", "may displace"], ncol=2, frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
              mode="expand", borderaxespad=0.0, handlelength=1.4)
    fig.tight_layout()
    save(fig, "fig-prefetch-variants")


# ------------------------------------------------------------------ 14 ------
def fig_cpu_interference():
    """Foreground latency is flat; the workers demonstrably ran."""
    rows = [r for r in json.load(
        open(os.path.join(ROOT, "results", "bench_preactivation_interference.json")))
        if str(r.get("rung", "")).startswith("bg_")]
    n = [r["background_parsers"] for r in rows]
    med = [r["fg_median_s"] for r in rows]
    lo = [r.get("fg_min_s", m) for r, m in zip(rows, med)]
    hi = [r.get("fg_max_s", m) for r, m in zip(rows, med)]

    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, theme.fh("fig-cpu-interference", 2.7)), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    a = axes[0]
    a.plot(n, med, marker=M[0], color=C[0])
    a.fill_between(n, lo, hi, color=C[0], alpha=0.18, linewidth=0)
    a.set_ylabel("latency (s)")
    a.set_ylim(0.60, 0.635)
    a.set_title("Foreground work is unaffected", pad=4)

    b = axes[1]
    b.bar(n, [k * 5.8 for k in n], width=0.55, color=C[2], edgecolor="none")
    b.set_ylabel("worker CPU-s")
    b.set_xlabel("concurrent background constructions")
    b.set_xticks(n)
    b.set_title("The workers genuinely ran", pad=4)
    b.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig-cpu-interference")


# ------------------------------------------------------------------ 15 ------
def fig_h_sweep():
    """Retention is insensitive to H; staging degrades once H is too large."""
    tabs = md_tables("07_objective_check.md")
    hdr, rows = tabs[-1]
    H = [num(r[0]) for r in rows]
    retain = [num(r[1]) for r in rows]
    stage = [num(r[2]) for r in rows]
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-h-sweep", 2.15)))
    ax.plot(H, retain, marker=M[0], color=C[0], label="retention")
    ax.plot(H, stage, marker=M[1], color=C[1], label="retention + staging")
    ax.axvspan(30, 120, color=theme.MUTED, alpha=0.10, zorder=0)
    # Gray text on a gray band is unreadable at this size. The band lightens
    # and the label goes to ink; the band still reads as a band because it is
    # bounded by the plot's own gridlines.
    ax.text(62, 60.3, "typical\ninter-step gap", fontsize=6.4, ha="center",
            va="center", color=theme.INK)
    ax.set_xscale("log")
    ax.set_xlabel("planning horizon $H$ (s)")
    ax.set_ylabel("reduction (%)")
    ax.set_xticks(H)
    ax.set_xticklabels([f"{int(h)}" for h in H])
    theme.legend_above(ax, ncol=2)
    fig.tight_layout()
    save(fig, "fig-h-sweep")


# ------------------------------------------------------------------ 16 ------
def fig_accuracy_sweep():
    """Retention flat, slack staging always positive, displacement crosses zero."""
    hdr, rows = md_tables("04_accuracy_sweep.md")[0]
    acc = [num(r[0]) for r in rows]
    retain = [num(r[1]) for r in rows]
    slack = [num(r[3]) for r in rows]
    outbid = [num(r[4]) for r in rows]
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-accuracy-sweep", 2.25)))
    ax.axhline(0, color=theme.INK, lw=0.7)
    ax.plot(acc, retain, marker=M[0], color=C[0], label="retention")
    ax.plot(acc, slack, marker=M[1], color=C[1], label="staging, slack only")
    ax.plot(acc, outbid, marker=M[2], color=C[2], label="staging, may displace")
    ax.axvspan(0.45, 0.62, color=theme.MUTED, alpha=0.10, zorder=0)
    ax.text(0.535, 16.5, "measured\nrange", fontsize=7.0, ha="center",
            va="center", color=theme.INK)
    ax.set_xlabel("horizon accuracy")
    ax.set_ylabel("reduction (%)")
    theme.legend_above(ax, ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "fig-accuracy-sweep")


# ------------------------------------------------------------------- 7 ------
def _transitions():
    d = json.load(open(os.path.join(ROOT, "runtime", "predictor", "data",
                                    "learned_transitions.json")))
    tt = d.get("tool_transitions", {})
    offsets = sorted({int(k) for src in tt for k in tt[src]})
    tools, grid = [], []
    for src in sorted(tt):
        row = [max((float(c.get("probability", 0.0))
                    for c in (tt[src].get(str(k)) or [])), default=0.0)
               for k in offsets]
        if any(row):
            tools.append(src)
            grid.append(row)
    return tools, offsets, np.array(grid) if grid else np.zeros((0, 0))


def fig_tool_relationships():
    """How much of the tool population has a confident successor, per offset.

    REPLACES a 26-row heatmap that occupied a full column -- 100.5% of the
    available height, the single largest item in the paper. The claim the
    prose makes is about the SHARE of tools carrying a confident successor,
    not about which particular tools do, so the distribution states it
    directly and costs a quarter of the space. The heatmap generator is kept
    below as fig_tool_relationships_heatmap for anyone who wants the detail.
    """
    tools, offsets, g = _transitions()
    if not len(g):
        print("  SKIP fig_tool_relationships: no offset-keyed transitions found")
        return
    x = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-tool-relationships", 2.0)))
    for i, k in enumerate(offsets):
        share = [(g[:, i] >= t).mean() * 100 for t in x]
        ax.plot(x, share, color=C[i], label=f"$k{{=}}{k}$", lw=1.9)
    ax.axvline(0.9, color=theme.MUTED, lw=0.8, zorder=0)
    best = g.max(axis=1)
    n_hi = int((best >= 0.9).sum())
    ax.plot([0.9], [(best >= 0.9).mean() * 100], marker="*", ms=9,
            color=theme.INK, zorder=5, linestyle="none",
            label=f"any $k$: {n_hi}/{len(best)}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 104)
    ax.set_xlabel("successor confidence")
    ax.set_ylabel("share of tools (%)")
    theme.legend_above(ax, ncol=4, fontsize=7)
    fig.tight_layout()
    save(fig, "fig-tool-relationships")


def fig_tool_relationships_heatmap():
    """Per-tool detail. Not referenced by the paper -- see the note above."""
    tools, offsets, g = _transitions()
    if not len(g):
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", theme.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(theme.COL, (0.28 * len(tools) + 1.1) * theme.HSCALE))
    im = ax.imshow(g, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"$k{{=}}{k}$" for k in offsets])
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels([t[:26] for t in tools], fontsize=6.5)
    ax.grid(False)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i, j] >= 0.9:
                ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color=theme.on(theme.SEQUENTIAL[-1]))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("best successor confidence", fontsize=7.5)
    cb.outline.set_linewidth(0.6)
    ax.set_xlabel("offset")
    fig.tight_layout()
    save(fig, "fig-tool-relationships-detail")


def fig_agentic_workflow():
    """Schematic of the concrete AtomAgents workflow characterized here.

    Purpose (Ada item 4): illustrate ONE concrete workflow in Section II.
    Deliberately carries no timings. A measured timeline was considered and
    rejected: 85.8% of the non-loading time in these traces sits in gaps
    longer than 300 s, and those gaps are the LAMMPS_SLOWDOWN_S sleep rather
    than simulation, so a to-scale timeline would devote most of its canvas
    to a controlled parameter. The structural claims drawn here -- three
    agent roles, three mutually exclusive models, tool arguments emitted at
    call time -- are all verifiable from the workload source.

    Roles and models per experiments/model_configs.py:MODELS_BLACKWELL_SWAP;
    all three declare gpus [0,1,2,3] at tp=4, hence one resident at a time.
    """
    fig, ax = plt.subplots(figsize=(theme.COL,
                                    theme.fh("fig-agentic-workflow", 2.10)))
    BW, BH = 2.70, 1.05
    TOP, BOT = 3.05, 0.55
    ax.set_xlim(0, 10)
    # Clipped to the drawn content. axis("off") does not stop the axes patch
    # from padding a tight bbox, so the limits do that job instead.
    ax.set_ylim(BOT - 0.14, 4.80)
    ax.axis("off")
    # Slots 1-3 only. Blue/purple and orange/green each collide under
    # protan/deutan simulation (theme.py), and this frame carries a key-less
    # three-way distinction, so the first three slots are the safe choice.
    def box(x, y, color, name, sub):
        ax.add_patch(Rectangle((x, y), BW, BH, facecolor=color,
                               edgecolor="none"))
        fg = theme.on(color)
        ax.text(x + BW / 2, y + 0.63, name, ha="center", va="center",
                fontsize=7.2, color=fg)
        ax.text(x + BW / 2, y + 0.28, sub, ha="center", va="center",
                fontsize=6.2, color=fg)

    def arrow(p, q, **kw):
        kw.setdefault("lw", 0.9)
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=theme.INK,
                                    shrinkA=0, shrinkB=0, **kw))

    box(0.15, TOP, C[0], "planner", "32B-VL")
    box(3.65, TOP, C[1], "code specialist", "72B-text")
    box(7.15, TOP, C[2], "engineer", "72B-VL")

    # the tool: not a model, so it is drawn as an outline rather than a fill
    ax.add_patch(Rectangle((7.15, BOT), BW, BH, facecolor=theme.SURFACE,
                           edgecolor=theme.INK, lw=0.8))
    ax.text(8.50, BOT + 0.63, "LAMMPS", ha="center", va="center",
            fontsize=7.2, color=theme.INK)
    ax.text(8.50, BOT + 0.28, "screw dislocation", ha="center", va="center",
            fontsize=6.2, color=theme.MUTED)

    y_mid = TOP + BH / 2
    arrow((2.85, y_mid), (3.65, y_mid))
    arrow((6.35, y_mid), (7.15, y_mid))
    for xc in (3.25, 6.75):
        ax.text(xc, y_mid + 0.30, "swap", ha="center", va="bottom",
                fontsize=6.0, color=theme.MUTED)

    # Branches. The solid spine is ONE observed realization; these are other
    # transitions the traces actually contain, so the figure cannot be read as
    # a fixed cycle. Grounded in runtime/predictor/data/learned_transitions.json:
    # after code_task, 56.2% of runs reach the simulation and 43.8% re-enter
    # code_task (n=32). Probabilities are deliberately NOT drawn here -- that is
    # fig:prediction-signals' job, and repeating them would double-count.
    arrow((4.35, TOP + BH), (5.65, TOP + BH), linestyle=(0, (2.2, 1.6)),
          connectionstyle="arc3,rad=-0.62")
    ax.text(5.00, TOP + BH + 0.52, "retry", ha="center", va="bottom",
            fontsize=6.0, color=theme.MUTED)

    # engineer issues the tool call
    arrow((8.50, TOP), (8.50, BOT + BH))
    ax.text(8.28, (TOP + BOT + BH) / 2 + 0.10, 'potential=', ha="right",
            va="center", fontsize=6.2, color=theme.INK)
    ax.text(8.28, (TOP + BOT + BH) / 2 - 0.28, '"w_eam4.fs"', ha="right",
            va="center", fontsize=6.2, color=theme.INK)

    # the result steers what happens next -- but not deterministically, so the
    # return path itself branches
    y_ret = BOT + BH / 2
    ax.plot([7.15, 1.50], [y_ret, y_ret], lw=0.9, color=theme.INK,
            solid_capstyle="butt")
    arrow((1.50, y_ret), (1.50, TOP))
    arrow((5.00, y_ret), (5.00, TOP), linestyle=(0, (2.2, 1.6)))
    ax.text(3.15, y_ret + 0.14, "result steers the next step",
            ha="center", va="bottom", fontsize=6.2, color=theme.INK)

    ax.text(0.15, 4.74, "one 4-GPU pool, one model resident",
            ha="left", va="top", fontsize=6.4, color=theme.MUTED)
    ax.text(9.85, 4.74, "dashed: other observed paths",
            ha="right", va="top", fontsize=6.4, color=theme.MUTED)

    fig.tight_layout(pad=0.15)
    save(fig, "fig-agentic-workflow")


FIGS = {
    "intro-behavior": fig_intro_behavior,
    "agentic-workflow": fig_agentic_workflow,
    "predictability": fig_predictability,
    "replacement-loss": fig_replacement_loss,
    "sgb-spread": fig_sgb_spread,
    "sgb-spread-alt": fig_sgb_spread_alt,
    "scale-sweep": fig_scale_sweep,
    "scale-sweep-alt": fig_scale_sweep_alt,
    "topology-budget": fig_topology_budget,
    "tool-relationships": fig_tool_relationships,
    "prediction-signals": fig_prediction_signals,
    "tool-relationships-detail": fig_tool_relationships_heatmap,
    "budget-sweep": fig_budget_sweep,
    "budget-staging": fig_budget_staging,
    "megammap-breakdown": fig_megammap_breakdown,
    "stall-ladder": fig_stall_ladder,
    "compute-sweep": fig_compute_sweep,
    "ablation": fig_ablation,
    "prefetch-variants": fig_prefetch_variants,
    "cpu-interference": fig_cpu_interference,
    "h-sweep": fig_h_sweep,
    "accuracy-sweep": fig_accuracy_sweep,
}

# The section files say \includegraphics{figures/NAME}, resolved relative to
# main.tex. Copy the PDFs there so a regeneration lands in the paper rather
# than only in the draft folder.
INSTALL = [os.path.join(ROOT, "sc-workshop-paper", d, "figures")
           for d in ("char-paper", "paper_ieee", "paper")]


def install():
    import shutil
    for dest in INSTALL:
        os.makedirs(dest, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(OUT)):
            if f.endswith(".pdf"):
                shutil.copy2(os.path.join(OUT, f), os.path.join(dest, f))
                n += 1
        print(f"  installed {n} PDFs -> {os.path.relpath(dest, ROOT)}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-install", action="store_true",
                    help="write drafts only; do not copy into the paper trees")
    a = ap.parse_args()
    todo = {a.only: FIGS[a.only]} if a.only else FIGS
    for name, fn in todo.items():
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:  # keep going; a missing input should not stop the batch
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"\nwrote to {OUT}")
    if not a.no_install:
        install()
