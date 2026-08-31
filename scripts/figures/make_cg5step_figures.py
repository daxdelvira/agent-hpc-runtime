#!/usr/bin/env python3
"""ChemGraph 5-step task only: the three trace-derived figures, paper styling.

    python3 scripts/figures/make_cg5step_figures.py

Writes fig-*-cg5step.{pdf,png} into sc-workshop-paper/figure_drafts/.

WHY THIS SUBSET. ChemGraph's traces contain three distinct tasks, each with
exactly one plan text, and pooling them produces a compliance number that is a
property of the collection mix rather than of ChemGraph:

    plan len   runs   order   positional   fully compliant
       3        114   100.0%     100.0%      114/114
       5         37    62.2%      47.0%        0/37
      18         27    52.1%      52.1%         1/27

The 5-step task is the only one of the three with any separation between the
two scorings -- all 28 cells of ChemGraph's pooled order-vs-positional gap are
in it. That makes it the only ChemGraph population whose plan-compliance figure
has the same SHAPE as the AtomAgents one.

STYLING MATCHES THE PAPER FIGURES DELIBERATELY (theme colours C[0]/C[1]/C[2],
markers M[*], the same axis labels and the same zero-labelling rule in the
compliance panel) so these can be read directly against fig-predictability and
fig-prediction-signals.

READ THE TRANSITION FIGURE WITH CARE. The paper's version covers 26 AtomAgents
tools drawn from the shipped learned_transitions.json. That table is global and
cannot be filtered to one task, so the table here is RECOMPUTED from these 37
runs alone -- which contain 124 tool events and just 3 distinct source tools,
every one of which has a successor at confidence 1.000. The resulting curve is
a 3-step staircase pinned at 100%. That is a real property of this task, not a
plotting artifact, but it is not a distribution and should not be read as one.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme  # noqa: E402
theme.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "sc-workshop-paper", "figure_drafts")
sys.path.insert(0, os.path.join(ROOT, "experiments"))

C, M = theme.CATEGORICAL, theme.MARKERS
PLAN_LEN = 5
OFFSETS = (1, 2, 3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load_5step():
    """(raw events, tool sequence) for every ChemGraph run whose plan has 5 steps."""
    runs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "logs/workflow_traces",
                                           "chemgraph_trace_*.jsonl"))):
        ev = []
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        plan = next((e["payload"].get("tool_sequence", []) for e in ev
                     if e.get("event_type") == "plan_extracted"), None)
        if not plan or len(plan) != PLAN_LEN:
            continue
        seq = [e["payload"].get("tool", "") for e in ev
               if e.get("event_type") == "tool_call"]
        if seq:
            runs.append((ev, seq))
    return runs


RUNS = load_5step()
print(f"  ChemGraph {PLAN_LEN}-step task: {len(RUNS)} runs, "
      f"{sum(len(s) for _, s in RUNS)} tool events")


# ------------------------------------------------------------ 1 agreement ---
def fig_predictability():
    per = defaultdict(list)
    for _, s in RUNS:
        for i, tool in enumerate(s):
            per[i].append(tool)
    steps, pct = [], []
    for i in sorted(per):
        if len(per[i]) < 3:          # same MIN_RUNS floor as the paper figure
            continue
        steps.append(i)
        pct.append(Counter(per[i]).most_common(1)[0][1] / len(per[i]) * 100)
    mean = sum(pct) / len(pct)
    print(f"    agreement: mean {mean:.1f}%  min {min(pct):.1f}%  "
          f"max {max(pct):.1f}%  over {len(steps)} steps")

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-predictability", 2.3)))
    ax.plot(steps, pct, marker=M[0], color=C[0], zorder=3)
    ax.axhline(mean, color=theme.MUTED, lw=0.9, zorder=1)
    ax.text(max(steps) - 0.1, mean - 4.5, f"mean {mean:.0f}%", fontsize=6.8,
            color=theme.MUTED, ha="right", va="top")
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.4, max(steps) + 0.4)
    ax.set_xticks(range(0, max(steps) + 1))
    ax.set_xlabel("step index")
    ax.set_ylabel("agreement (%)")
    fig.tight_layout()
    save(fig, "fig-predictability-cg5step")


# ----------------------------------------------------------- 2 transitions --
def transitions():
    """Best successor confidence per (source tool, offset), from these runs only."""
    pairs = defaultdict(Counter)
    for _, s in RUNS:
        for k in OFFSETS:
            for i in range(len(s) - k):
                pairs[(s[i], k)][s[i + k]] += 1
    tools = sorted({src for src, _ in pairs})
    g = np.zeros((len(tools), len(OFFSETS)))
    support = np.zeros_like(g)
    for r, src in enumerate(tools):
        for c, k in enumerate(OFFSETS):
            cnt = pairs.get((src, k))
            if cnt:
                n = sum(cnt.values())
                g[r, c] = cnt.most_common(1)[0][1] / n
                support[r, c] = n
    return tools, g, support


def fig_transitions():
    tools, g, support = transitions()
    best = g.max(axis=1)
    print(f"    transitions: {len(tools)} source tools; "
          f"{int((best >= 0.9).sum())}/{len(tools)} have a successor >= 0.9; "
          f"support per cell {int(support.min())}-{int(support.max())}")

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-tool-relationships", 2.3)))
    x = np.linspace(0.0, 1.0, 201)
    DASH = [(0, ()), (0, (4, 1.6)), (0, (1, 1.4))]
    for i, k in enumerate(OFFSETS):
        ax.plot(x, [(g[:, i] >= t).mean() * 100 for t in x], color=C[i],
                label=f"$k{{=}}{k}$", lw=1.7, ls=DASH[i % len(DASH)])
    ax.axvline(0.9, color=theme.MUTED, lw=0.8, zorder=0)
    ax.plot([0.9], [(best >= 0.9).mean() * 100], marker="*", ms=8, color=theme.INK,
            linestyle="none", zorder=5,
            label=f"any $k$: {int((best>=0.9).sum())}/{len(tools)}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 104)
    ax.set_xlabel("successor confidence", labelpad=1.5)
    ax.set_ylabel("tools (%)")
    theme.legend_above(ax, ncol=4, fontsize=7)
    fig.tight_layout()
    save(fig, "fig-tool-relationships-cg5step")


# ------------------------------------------------------------ 3 compliance --
def fig_compliance():
    from plot_plan_accuracy import compute_aa_compliance
    try:
        from plot_plan_accuracy import CG_TOOL_SHORT_LABELS as LMAP
    except ImportError:
        LMAP = None
    events = [ev for ev, _ in RUNS]
    scored = []
    for strict in (False, True):
        Mx, _, labels = compute_aa_compliance(events, label_map=LMAP, strict=strict)
        pct = [(Mx[:, k] == 1.0).sum() / max((Mx[:, k] >= 0).sum(), 1) * 100
               for k in range(Mx.shape[1])]
        overall = int((Mx == 1.0).sum()) / int((Mx >= 0).sum()) * 100
        scored.append((pct, overall, labels))
    print(f"    compliance over {len(events)} runs: "
          f"order-only {scored[0][1]:.1f}%, positional {scored[1][1]:.1f}%")

    fig, ax = plt.subplots(figsize=(theme.COL, theme.fh("fig-prediction-signals", 2.6)))
    labels = scored[0][2]
    xs = np.arange(len(labels))
    for k, (pct, overall, _) in enumerate(scored):
        lab = ("order only" if k == 0 else "position") + f" ({overall:.0f}%)"
        ax.bar(xs + (k - 0.5) * 0.38, pct, 0.36, color=C[k], edgecolor="none",
               label=lab)
        # Same rule as the paper panel: a zero-height bar reads as missing data,
        # and the zeros are a measurement.
        for xi, v in zip(xs + (k - 0.5) * 0.38, pct):
            if v <= 0.5:
                ax.text(xi, 2.5, "0", ha="center", va="bottom", fontsize=6.2,
                        color=C[k])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("compliant (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([l[:11] for l in labels], rotation=28, ha="right",
                       fontsize=6.5)
    ax.grid(axis="x", visible=False)
    theme.legend_above(ax, ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig-plan-compliance-cg5step")


if __name__ == "__main__":
    fig_predictability()
    fig_transitions()
    fig_compliance()
