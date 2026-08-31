#!/usr/bin/env python3
"""Per-workload cuts of the trace-derived figures: ChemGraph alone, and both.

    python3 scripts/figures/make_workload_variants.py

Writes into
    sc-workshop-paper/figure_drafts/chemgraph/   ChemGraph only
    sc-workshop-paper/figure_drafts/compare/     AtomAgents and ChemGraph together

WHY ONLY THESE THREE FIGURES. Most figures in the paper are workload-agnostic:
the budget/scale/compute sweeps are simulation over a measured catalogue, and
fig-intro-behavior and fig-agentic-workflow are schematics. Only three rest on
per-workload traces and therefore have a meaningful ChemGraph counterpart:

    predictability     agreement with the modal action at each step index
    transitions        share of tools with a high-confidence successor
    plan-compliance    per-step plan compliance, order-only and positional

fig-replacement-loss is excluded on purpose: its lower panel is constructed
rather than measured (a known open item), so a ChemGraph "equivalent" would be
a second drawing, not a second measurement.

THE TOOL PARTITION IS DERIVED, NOT HAND-WRITTEN. learned_transitions.json mixes
both frameworks' tools in one table. Rather than maintain a list by hand, the
tool set for each framework is taken from that framework's own traces, so the
partition cannot drift from the data.
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
TRACES = os.path.join(ROOT, "logs", "workflow_traces")
DRAFTS = os.path.join(ROOT, "sc-workshop-paper", "figure_drafts")
OUT_CG = os.path.join(DRAFTS, "chemgraph")
OUT_CMP = os.path.join(DRAFTS, "compare")
for d in (OUT_CG, OUT_CMP):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "experiments"))

WORKLOADS = {
    "AtomAgents": dict(pattern="runtime_trace_*.jsonl", color=theme.CATEGORICAL[0],
                       marker=theme.MARKERS[0]),
    "ChemGraph":  dict(pattern="chemgraph_trace_*.jsonl", color=theme.CATEGORICAL[1],
                       marker=theme.MARKERS[1]),
}
MIN_RUNS, MAX_STEP = 3, 20

# These are exploratory cuts, not page-budgeted paper floats, so they do NOT go
# through theme.fh()'s per-figure compression -- at TIGHT the y label is taller
# than its own panel and bbox="tight" trims it. Explicit, roomier heights.
H1, H2 = 2.05, 3.6


def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(TRACES, pattern))):
        ev = []
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if ev:
            out.append(ev)
    return out


def tool_seqs(traces):
    seqs = []
    for t in traces:
        s = [e["payload"]["tool"] for e in t
             if e.get("event_type") == "tool_call" and "tool" in e.get("payload", {})]
        if s:
            seqs.append(s)
    return seqs


def agreement(seqs):
    per = defaultdict(list)
    for s in seqs:
        for i, tool in enumerate(s):
            if i <= MAX_STEP:
                per[i].append(tool)
    steps, pct, runs = [], [], []
    for i in sorted(per):
        v = per[i]
        if len(v) < MIN_RUNS:
            continue
        steps.append(i)
        pct.append(Counter(v).most_common(1)[0][1] / len(v) * 100)
        runs.append(len(v))
    return steps, pct, runs


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"    {os.path.relpath(outdir, ROOT)}/{name}.pdf")


# ------------------------------------------------------------------ data ----
DATA = {}
for wl, cfg in WORKLOADS.items():
    tr = load(cfg["pattern"])
    seqs = tool_seqs(tr)
    DATA[wl] = dict(traces=tr, seqs=seqs, agreement=agreement(seqs),
                    tools=set(x for s in seqs for x in s), **cfg)
    st, pc, rn = DATA[wl]["agreement"]
    print(f"  {wl}: {len(tr)} traces, {len(seqs)} with tool calls, "
          f"{len(DATA[wl]['tools'])} distinct tools, "
          f"agreement mean {sum(pc)/len(pc):.1f}% over {len(st)} steps")


# ---------------------------------------------------------- predictability --
def _pred_axes(ax, wl, annotate_mean=True):
    st, pc, _ = DATA[wl]["agreement"]
    m = sum(pc) / len(pc)
    ax.plot(st, pc, marker=DATA[wl]["marker"], color=DATA[wl]["color"],
            label=f"{wl} (mean {m:.1f}%)", zorder=3)
    if annotate_mean:
        ax.axhline(m, color=DATA[wl]["color"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    return m


def fig_predictability_cg():
    fig, ax = plt.subplots(figsize=(theme.COL, H1))
    m = _pred_axes(ax, "ChemGraph")
    ax.set_xlabel("step index")
    # Short label deliberately: "agreement with modal action (%)" is taller
    # than the panel at this scale and bbox="tight" clipped both ends of it.
    ax.set_ylabel("modal agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, MAX_STEP + 1, 2))
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, OUT_CG, "fig-predictability-chemgraph")


def fig_predictability_both():
    fig, ax = plt.subplots(figsize=(theme.COL, H1))
    for wl in WORKLOADS:
        _pred_axes(ax, wl)
    ax.set_xlabel("step index")
    # Short label deliberately: "agreement with modal action (%)" is taller
    # than the panel at this scale and bbox="tight" clipped both ends of it.
    ax.set_ylabel("modal agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, MAX_STEP + 1, 2))
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, OUT_CMP, "fig-predictability-both")


# ------------------------------------------------------------- transitions --
def transition_counts(tools):
    """(n_tools_with_successor_at_conf, n_tools) over the shipped table."""
    path = os.path.join(ROOT, "runtime/predictor/data/learned_transitions.json")
    tt = json.load(open(path))["tool_transitions"]
    keys = [k for k in tt if k in tools]
    confs = np.arange(0.5, 1.001, 0.05)
    best = {}
    for k in keys:
        b = 0.0
        for off in tt[k]:
            for e in tt[k][off]:
                b = max(b, float(e["probability"]))
        best[k] = b
    return confs, [sum(1 for v in best.values() if v >= c) for c in confs], len(keys)


def _trans_axes(ax, wl):
    confs, counts, n = transition_counts(DATA[wl]["tools"])
    ax.plot(confs, counts, marker=DATA[wl]["marker"], color=DATA[wl]["color"],
            label=f"{wl} ({n} tools)", zorder=3)
    return n


def fig_transitions_cg():
    fig, ax = plt.subplots(figsize=(theme.COL, H1))
    n = _trans_axes(ax, "ChemGraph")
    ax.set_xlabel("confidence of best successor")
    ax.set_ylabel("tools (count)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, OUT_CG, "fig-transitions-chemgraph")


def fig_transitions_both():
    fig, ax = plt.subplots(figsize=(theme.COL, H1))
    for wl in WORKLOADS:
        _trans_axes(ax, wl)
    ax.set_xlabel("confidence of best successor")
    ax.set_ylabel("tools (count)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, OUT_CMP, "fig-transitions-both")


# --------------------------------------------------------- plan compliance --
def compliance(wl):
    """Per-step compliance, order-only and positional, via the shared analysis.

    Imported from experiments/plot_plan_accuracy.py rather than reimplemented,
    so this cannot drift from the number the paper quotes.
    """
    from plot_plan_accuracy import compute_aa_compliance
    tr = [t for t in DATA[wl]["traces"]
          if any(e.get("event_type") == "plan_extracted" for e in t)]
    out = {}
    for strict in (False, True):
        M, plans, labels = compute_aa_compliance(tr, strict=strict)
        if M.ndim != 2 or M.shape[0] == 0:
            return None
        pct = []
        for p in range(M.shape[1]):
            col = M[:, p]
            nv = int((col >= 0).sum())
            nm = int((col == 1.0).sum())
            pct.append(nm / nv * 100 if nv else 0.0)
        out["positional" if strict else "order"] = pct
        # HEADLINE AGGREGATE IS CELL-WEIGHTED, matching fig_prediction_signals
        # in make_figures.py: total matched cells / total valid cells. The mean
        # of the per-step percentages is NOT the same number -- later planned
        # steps are scored over far fewer runs, so an unweighted mean over steps
        # reports AtomAgents as 80.0/24.2 where the paper says 76.3/40.8. Using
        # the paper's convention here keeps these variants comparable to it.
        key = "positional_overall" if strict else "order_overall"
        out[key] = int((M == 1.0).sum()) / max(int((M >= 0).sum()), 1) * 100
        out["n"] = M.shape[0]
    return out


def _comp_axes(ax, wl, c):
    d = compliance(wl)
    if d is None:
        ax.text(0.5, 0.5, f"no plans in {wl} traces", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=theme.MUTED)
        return None
    x = np.arange(len(d["order"]))
    ax.bar(x - 0.2, d["order"], 0.4, color=c, label="order only")
    ax.bar(x + 0.2, d["positional"], 0.4, color=theme.MUTED, label="positional")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=6.4)
    print(f"    {wl} plan compliance over {d['n']} runs: "
          f"order {d['order_overall']:.1f}%, positional {d['positional_overall']:.1f}%")
    return d


def fig_compliance_cg():
    fig, ax = plt.subplots(figsize=(theme.COL, H1))
    d = _comp_axes(ax, "ChemGraph", DATA["ChemGraph"]["color"])
    ax.set_xlabel("planned step")
    ax.set_ylabel("compliance (%)")
    ax.set_ylim(0, 105)
    theme.legend_above(ax, ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, OUT_CG, "fig-plan-compliance-chemgraph")


def fig_compliance_both():
    fig, axes = plt.subplots(2, 1, figsize=(theme.COL, H2))
    for ax, wl in zip(axes, WORKLOADS):
        _comp_axes(ax, wl, DATA[wl]["color"])
        ax.set_ylabel("compliance (%)")
        ax.set_ylim(0, 105)
        ax.set_title(wl, fontsize=8, pad=3)
    axes[-1].set_xlabel("planned step")
    theme.legend_above(axes[0], ncol=2, fontsize=7)
    fig.tight_layout(h_pad=0.8)
    save(fig, OUT_CMP, "fig-plan-compliance-both")


if __name__ == "__main__":
    print("\n  ChemGraph only ->")
    fig_predictability_cg()
    fig_transitions_cg()
    fig_compliance_cg()
    print("\n  Both workloads ->")
    fig_predictability_both()
    fig_transitions_both()
    fig_compliance_both()
    print()
