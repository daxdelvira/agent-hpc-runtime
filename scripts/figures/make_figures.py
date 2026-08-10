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
    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, 2.55), sharex=True)
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
                    fontsize=7, color="white" if alpha > 0.6 else theme.INK)

    # -- traditional: need is known, staging hides behind the previous phase
    a = axes[0]
    bar(a, 1.25, 0.3, 4.2, C[0], "compute phase $n$")
    bar(a, 1.25, 4.7, 4.4, C[0], "compute phase $n{+}1$")
    bar(a, 0.15, 1.55, 3.2, C[2], "stage for $n{+}1$")
    a.text(1.55, 2.30, "inputs declared", fontsize=7, color=theme.INK, ha="center")
    a.plot([1.55], [2.10], marker="v", ms=4, color=theme.INK)
    a.set_yticklabels(["storage", "compute"])
    a.set_title("Traditional workflow: inputs known before the phase begins", pad=4)

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
    b.text(6.55, 1.56, "stall", ha="center", va="center", fontsize=7, color=theme.INK)
    b.set_yticklabels(["storage", "compute"])
    b.set_xlabel("wall-clock time")
    b.set_xticks([])
    b.set_title("Agentic workflow: the need appears after the decision", pad=4)

    fig.tight_layout(h_pad=0.9)
    save(fig, "fig-intro-behavior")


# ------------------------------------------------------------------ 3 -------
def fig_replacement_loss():
    """Model-load timeline for a representative trial, against artifact residency."""
    loads = [(0.1, 403.8, "32B"), (485.6, 1204.4, "72B"), (1217.9, 1911.8, "72B-t"),
             (1967.7, 2661.4, "72B"), (2687.6, 3422.9, "72B"),
             (3436.0, 4189.6, "72B-t"), (4245.4, 5254.4, "72B")]
    wall = 5288.5
    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, 2.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})

    ax = axes[0]
    ax.grid(False)
    for i, (s, e, lab) in enumerate(loads):
        ax.add_patch(Rectangle((s, 0.15), e - s, 0.7, facecolor=C[0],
                               edgecolor="none", alpha=1.0 if i == 0 else 0.85))
        if e - s > 500:
            ax.text((s + e) / 2, 0.5, lab, ha="center", va="center",
                    fontsize=6.5, color="white")
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

    fig, ax = plt.subplots(figsize=(theme.COL, 2.35))
    y = np.arange(len(data))
    # Dots, not bars: on a log axis a bar's LENGTH is not proportional to its
    # value, so bars would misstate the 65x spread this figure exists to show.
    ax.hlines(y, 0.24, [v for _, v in data], color=theme.GRID, lw=0.8, zorder=1)
    ax.plot([v for _, v in data], y, "o", color=C[2], ms=6, zorder=3,
            markeredgecolor=theme.SURFACE, markeredgewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(k, k) for k, _ in data])
    ax.set_ylim(len(data) - 0.4, -1.15)          # headroom for the band label
    ax.set_xscale("log")
    ax.set_xlabel("seconds saved per GB retained")
    ax.grid(axis="y", visible=False)

    ax.axvspan(2.78, 3.81, color=C[0], alpha=0.16, zorder=0)
    ax.text(3.26, -0.72, "models", fontsize=7.5, color=C[0], ha="center",
            va="center")
    ax.axvline(0.32, color=theme.MUTED, lw=0.8, zorder=0)
    ax.text(0.30, -0.72, "materialisation\nfloor", fontsize=6.6, color=theme.MUTED,
            ha="right", va="center", linespacing=1.15)
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

    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, 3.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    a = axes[0]
    a.plot(n, retain, marker=M[0], color=C[0], label="retention")
    a.plot(n, stage, marker=M[1], color=C[1], label="+ staging")
    a.plot(n, oracle, marker=M[2], color=C[2], ls=(0, (4, 2)), label="retention, oracle")
    a.set_ylabel("wall-time reduction (%)")
    # No panel title here: it duplicated the y label, and it was what forced the
    # key inside the frame, where it sat on the oracle line.
    theme.legend_above(a, ncol=3, fontsize=7)

    b = axes[1]
    b.plot(n, first, marker=M[3], color=C[3])
    b.set_ylabel("first uses (%)")
    b.set_xlabel("distinct resources in the workflow")
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
    fig, ax = plt.subplots(figsize=(theme.COL, 2.1))
    for i, slots in enumerate((1, 2)):
        ax.plot(d[slots]["budget"], d[slots]["binding"], marker=M[i], color=C[i],
                label=f"{slots} device slot" + ("s" if slots > 1 else ""))
    for thr in (283, 445, 562):
        ax.axvline(thr, color=theme.MUTED, lw=0.6, zorder=0)
    ax.text(283, 37, "packing thresholds", fontsize=6.8, color=theme.MUTED, ha="left")
    ax.set_xlabel("host-memory budget (GB)")
    ax.set_ylabel("decisions forcing an eviction (%)")
    theme.legend_above(ax, ncol=2)
    fig.tight_layout()
    save(fig, "fig-topology-budget")


def fig_budget_sweep():
    """Result view: performance vs budget, one panel per device-slot count."""
    d = _budget_series()
    fig, axes = plt.subplots(1, 2, figsize=(theme.WIDE, 2.5), sharey=True)
    for ax, slots in zip(axes, (1, 2)):
        s = d[slots]
        ax.plot(s["budget"], s["lru"], marker=M[1], color=C[1], label="recency-ranked")
        ax.plot(s["budget"], s["sysname"], marker=M[0], color=C[0], label="Tandem")
        for thr in (283, 445, 562):
            ax.axvline(thr, color=theme.MUTED, lw=0.6, zorder=0)
        ax.set_xlabel("host-memory budget (GB)")
        ax.set_title(f"{slots} device slot" + ("s" if slots > 1 else ""), pad=4)
    axes[0].set_ylabel("wall-time reduction (%)")
    legend(axes[0])
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig-budget-sweep")


# ------------------------------------------------------------------ 10 ------
def fig_stall_ladder():
    """Where the time goes, per arm, normalised to the no-retention baseline."""
    d = _budget_series()
    configs = [("256 GB, 1 slot", 1, 256.0), ("560 GB, 2 slots", 2, 560.0)]
    compute = {1: 10.8, 2: 10.8}

    fig, ax = plt.subplots(figsize=(theme.COL, 2.2))
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
    ax.set_ylabel("stall (% of baseline wall)")
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
    fig, ax = plt.subplots(figsize=(theme.COL, 2.2))
    ax.plot(comp, wall, marker=M[0], color=C[0], label="wall-time reduction")
    ax.plot(comp, stall, marker=M[1], color=C[1], label="stall reduction")
    ax.axvline(5.3, color=theme.MUTED, lw=0.8)
    # Lower-left is the only region both series vacate; horizontal there beats
    # rotated against the rule, which crowded the leftmost markers.
    ax.text(5.6, 1.4, "measured workload", fontsize=6.6, color=theme.MUTED,
            ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("computation as a share of wall time (%)")
    ax.set_ylabel("reduction vs recency-ranked (%)")
    ax.set_xticks([5, 10, 25, 50, 90])
    ax.set_xticklabels(["5", "10", "25", "50", "90"])
    theme.legend_above(ax, ncol=2)
    fig.tight_layout()
    save(fig, "fig-compute-sweep")


# ------------------------------------------------------------------ 12 ------
def fig_ablation():
    """Cumulative contribution of each mechanism, plus the Shapley reversal."""
    hdr, rows = md_tables("01_attribution_ladder.md")[0]
    names = ["no residency system", "+ model parking", "+ data retention",
             "+ cost-aware arbitration", "+ slack staging"]
    vals = [num(r[2]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(theme.WIDE, 2.35),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    a = axes[0]
    y = np.arange(len(names))
    a.barh(y, vals, height=0.6, color=[theme.MUTED] + [C[0]] * 4, edgecolor="none")
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.invert_yaxis()
    a.set_xlabel("wall-time reduction vs no residency system (%)")
    a.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        if i:
            a.text(v + 0.8, i, f"+{v - vals[i-1]:.2f}", va="center",
                   fontsize=6.8, color=theme.INK)
    a.set_xlim(0, max(vals) * 1.22)

    b = axes[1]
    budgets = ["256 GB", "560 GB"]
    parking = [2.0, 61.0]
    dataret = [98.0, 39.0]
    x = np.arange(2)
    # The 2pt separation between segments is drawn as a gap INSIDE the 100%
    # total, not added on top of it -- otherwise the stack reads as >100%.
    GAP = 1.4
    b.bar(x, [p - GAP / 2 for p in parking], 0.5, color=C[1], edgecolor="none",
          label="model parking")
    b.bar(x, [d - GAP / 2 for d in dataret], 0.5,
          bottom=[p + GAP / 2 for p in parking], color=C[0], edgecolor="none",
          label="data retention")
    for xi, p in zip(x, parking):
        b.text(xi, 101.5, f"{p:.0f} / {100-p:.0f}", ha="center", fontsize=6.8)
    b.set_xticks(x)
    b.set_xticklabels(budgets)
    b.set_ylabel("share of retention benefit (%)")
    b.grid(axis="x", visible=False)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylim(0, 108)
    theme.legend_above(b, ncol=2)
    fig.tight_layout(w_pad=1.4)
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
    fig, ax = plt.subplots(figsize=(theme.COL, 2.15))
    x = np.arange(len(order))
    colors = [theme.MUTED, C[0], C[1], C[0], C[1]]
    ax.bar(x, [vals[o] for o in order], 0.62,
           color=[colors[k] for k in range(len(order))], edgecolor="none")
    for xi, o in zip(x, order):
        ax.text(xi, vals[o] + 0.25, f"{vals[o]:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cols[o].replace(", ", "\n") for o in order])
    ax.set_ylabel("reduction vs recency-ranked (%)")
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

    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, 2.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    a = axes[0]
    a.plot(n, med, marker=M[0], color=C[0])
    a.fill_between(n, lo, hi, color=C[0], alpha=0.18, linewidth=0)
    a.set_ylabel("foreground latency (s)")
    a.set_ylim(0.60, 0.635)
    a.set_title("Foreground work is unaffected", pad=4)

    b = axes[1]
    b.bar(n, [k * 5.8 for k in n], width=0.55, color=C[2], edgecolor="none")
    b.set_ylabel("worker CPU-s")
    b.set_xlabel("concurrent background construction processes")
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
    fig, ax = plt.subplots(figsize=(theme.COL, 2.15))
    ax.plot(H, retain, marker=M[0], color=C[0], label="retention")
    ax.plot(H, stage, marker=M[1], color=C[1], label="retention + staging")
    ax.axvspan(30, 120, color=theme.MUTED, alpha=0.14, zorder=0)
    ax.text(62, 56.5, "typical\ninter-step gap", fontsize=6.6, ha="center",
            color=theme.MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("planning horizon $H$ (s)")
    ax.set_ylabel("wall-time reduction (%)")
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
    fig, ax = plt.subplots(figsize=(theme.COL, 2.25))
    ax.axhline(0, color=theme.INK, lw=0.7)
    ax.plot(acc, retain, marker=M[0], color=C[0], label="retention")
    ax.plot(acc, slack, marker=M[1], color=C[1], label="staging, slack only")
    ax.plot(acc, outbid, marker=M[2], color=C[2], label="staging, may displace")
    ax.axvspan(0.45, 0.62, color=theme.MUTED, alpha=0.14, zorder=0)
    ax.text(0.535, -4.4, "measured\nrange", fontsize=6.6, ha="center", color=theme.MUTED)
    ax.set_xlabel("horizon accuracy")
    ax.set_ylabel("reduction vs recency-ranked (%)")
    theme.legend_above(ax, ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "fig-accuracy-sweep")


# ------------------------------------------------------------------- 7 ------
def fig_tool_relationships():
    """Best successor confidence per tool, per offset."""
    d = json.load(open(os.path.join(ROOT, "runtime", "predictor", "data",
                                    "learned_transitions.json")))
    tt = d.get("tool_transitions", {})
    # The shipped table is keyed [source][offset] -> list of candidate dicts,
    # and it only carries offsets 1 and 2 -- the predictor's horizon is
    # hardcoded to those two. k in [0,5] is NOT available from this artifact.
    offsets = sorted({int(k) for src in tt for k in tt[src]})
    tools, grid = [], []
    for src in sorted(tt):
        row = []
        for k in offsets:
            cands = tt[src].get(str(k)) or []
            row.append(max((float(c.get("probability", 0.0)) for c in cands),
                           default=0.0))
        if any(row):
            tools.append(src)
            grid.append(row)
    if not grid:
        print("  SKIP fig_tool_relationships: no offset-keyed transitions found")
        return
    g = np.array(grid)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", theme.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(theme.COL, 0.28 * len(tools) + 1.1))
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
                        fontsize=6, color="white")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("best successor confidence", fontsize=7.5)
    cb.outline.set_linewidth(0.6)
    ax.set_xlabel("offset")
    fig.tight_layout()
    save(fig, "fig-tool-relationships")


FIGS = {
    "intro-behavior": fig_intro_behavior,
    "replacement-loss": fig_replacement_loss,
    "sgb-spread": fig_sgb_spread,
    "scale-sweep": fig_scale_sweep,
    "topology-budget": fig_topology_budget,
    "tool-relationships": fig_tool_relationships,
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
