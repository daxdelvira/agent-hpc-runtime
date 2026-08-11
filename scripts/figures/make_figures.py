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
    for i2, k in enumerate(offsets):
        a.plot(x, [(g[:, i2] >= t).mean() * 100 for t in x], color=C[i2],
               label=f"$k{{=}}{k}$", lw=1.7)
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
    b.set_xlabel("distinct resources")
    b.set_xticks(n)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig-scale-sweep")


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
    ax.axvspan(30, 120, color=theme.MUTED, alpha=0.14, zorder=0)
    ax.text(62, 56.5, "typical\ninter-step gap", fontsize=6.6, ha="center",
            color=theme.MUTED)
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
    ax.axvspan(0.45, 0.62, color=theme.MUTED, alpha=0.14, zorder=0)
    ax.text(0.535, -4.4, "measured\nrange", fontsize=6.6, ha="center", color=theme.MUTED)
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


FIGS = {
    "intro-behavior": fig_intro_behavior,
    "predictability": fig_predictability,
    "replacement-loss": fig_replacement_loss,
    "sgb-spread": fig_sgb_spread,
    "scale-sweep": fig_scale_sweep,
    "topology-budget": fig_topology_budget,
    "tool-relationships": fig_tool_relationships,
    "prediction-signals": fig_prediction_signals,
    "tool-relationships-detail": fig_tool_relationships_heatmap,
    "budget-sweep": fig_budget_sweep,
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
           for d in ("paper_ieee", "paper")]


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
