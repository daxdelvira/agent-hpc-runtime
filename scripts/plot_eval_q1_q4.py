#!/usr/bin/env python3
"""
plot_eval_q1_q4.py — publication figures for the paper's evaluation (Q1–Q4).

Input : normalized CSVs written by scripts/parse_eval_traces.py under
        results/eval_q1_q4/
Output: figures/eval_q1_walltime.{pdf,png}
        figures/eval_q1_gpu_idle.{pdf,png}
        figures/eval_q2_time_breakdown.{pdf,png}
        figures/eval_q2_overlap_recovered.{pdf,png}
        figures/eval_q2_representative_timeline.{pdf,png}
        figures/eval_q3_prediction_outcomes.{pdf,png}
        figures/eval_q3_precision_recall.{pdf,png}
        figures/eval_q3_confidence_calibration.{pdf,png}
        figures/eval_q3_lead_time_cdf.{pdf,png}
        figures/eval_q4_speculation_bytes.{pdf,png}
        figures/eval_q4_divergence_guard_ablation.{pdf,png}
        figures/eval_q4_memory_overhead.{pdf,png}

Style: matplotlib only (no seaborn), white background, serif fonts sized for
ACM column widths (single column ≈ 3.33 in, double ≈ 7 in).  Colorblind-safe
categorical palette (fixed assignment per configuration, never cycled).
Figures annotate per-bar trial counts; configs with N < 10 are marked "*".

Usage
-----
    python scripts/plot_eval_q1_q4.py
    python scripts/plot_eval_q1_q4.py --eval-root results/eval_q1_q4 --fig-dir figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

DEFAULT_EVAL_ROOT = PROJECT_ROOT / "results" / "eval_q1_q4"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Style — white background, ACM-column serif, colorblind-safe fixed palette
# ---------------------------------------------------------------------------

SINGLE_COL_W = 3.33   # inches (ACM single column)
DOUBLE_COL_W = 7.00

# Categorical palette (validated colorblind-safe order; do not re-order).
PAL = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "magenta": "#e87ba4",
    "orange":  "#eb6834",
    "gray":    "#6e6e6e",
    "lgray":   "#b5b5b5",
}

# Fixed color per configuration (color follows the entity, never its rank).
CONFIG_STYLE = {
    "baseline":            ("Baseline",             PAL["gray"]),
    "observe_only":        ("Observe-only",         PAL["lgray"]),
    "simulated":           ("Simulated",            PAL["magenta"]),
    "full_system":         ("SystemName (full)",    PAL["blue"]),
    "no_cache_stage":      ("No cache staging",     PAL["aqua"]),
    "naive_prefetch":      ("Naïve prefetch",       PAL["yellow"]),
    "no_divergence_guard": ("No divergence guard",  PAL["red"]),
    "no_plan":             ("No plan extraction",   PAL["violet"]),
    "plan_only":           ("Plan-only predictor",  PAL["magenta"]),
    "transition_only":     ("Transition-only",      PAL["orange"]),
    "oracle":              ("Oracle",               PAL["green"]),
    # External-system comparison (MegaMmap/Hermes staging tier): browns, so
    # they can't be confused with any SystemName ablation hue in shared axes.
    "megammap_stage":      ("MegaMmap staging",     "#8a5a2b"),
    "megammap_stage_rand": ("MegaMmap (rand)",      "#c49a6c"),
}
CONFIG_ORDER = list(CONFIG_STYLE.keys())

WORKLOAD_LABEL = {
    "chemgraph_swap": "ChemGraph",
    "atomagents_exp2": "AtomAgents (Exp2)",
    "atomagents_exp3": "AtomAgents (Exp3)",
    "deepdrivemd": "DeepDriveMD",
}

# Q2 stacked-segment styling (status-like roles, distinct from config palette).
SEGMENTS = [
    ("agent_reasoning_s", "Agent reasoning",         PAL["aqua"],   None),
    ("tool_exec_s",       "Tool execution",          PAL["blue"],   None),
    ("exposed_stall_s",   "Exposed staging (stall)", PAL["red"],    None),
    ("other_s",           "Other",                   PAL["lgray"],  None),
]
OVERLAP_SEGMENT = ("overlapped_io_s", "Background staging (overlapped)",
                   PAL["yellow"], "//")

# Q3 outcome stack (status semantics).
OUTCOME_STYLE = [
    ("useful",               "Useful",           PAL["green"]),
    ("late",                 "Late hit",         PAL["yellow"]),
    ("wasted",               "Wrong / wasted",   PAL["red"]),
    ("cancelled",            "Cancelled",        PAL["orange"]),
    ("expired_unvalidated",  "Expired",          PAL["lgray"]),
    ("missed_opportunities", "Missed",           PAL["violet"]),
]

N_TARGET = 10   # trials target; fewer gets an asterisk


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "font.family":      "serif",
        "font.serif":       ["Linux Libertine O", "Times New Roman",
                             "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size":        8,
        "axes.titlesize":   8.5,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  6.5,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.edgecolor":   "#444444",
        "axes.linewidth":   0.7,
        "xtick.color":      "#444444",
        "ytick.color":      "#444444",
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "grid.color":       "#dddddd",
        "grid.linewidth":   0.5,
        "legend.frameon":   False,
        "pdf.fonttype":     42,   # embed TrueType (camera-ready requirement)
        "ps.fonttype":      42,
    })


def save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{stem}.pdf + .png")


# ---------------------------------------------------------------------------
# Data loading / aggregation
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v, default=None):
    try:
        if v in (None, "", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def agg(rows: list[dict], value_key: str) -> dict[tuple[str, str], dict]:
    """(workload, config) → mean/std/n of value_key over successful runs."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        v = fnum(r.get(value_key))
        if v is not None:
            groups[(r["workload"], r["config"])].append(v)
    out = {}
    for k, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "n": n, "values": vals}
    return out


def ordered_configs(present: set[str]) -> list[str]:
    return [c for c in CONFIG_ORDER if c in present] + \
           sorted(c for c in present if c not in CONFIG_ORDER)


def bar_annotation(n: int) -> str:
    return f"n={n}" + ("*" if n < N_TARGET else "")


def gpu_short(name: str) -> str:
    if "L40S" in name:
        return "L40S"
    if "Blackwell" in name or "RTX PRO" in name:
        return "Blackwell"
    if "V100" in name:
        return "V100"
    return (name or "unknown").split()[-1]


def facet_mixed_gpu(*row_sets: list[dict]) -> dict[str, tuple[str, str | None]]:
    """Rekey `workload` to "wl [GPU]" for workloads that mix GPU types.

    Aggregates must never blend GPU types. Single-GPU workloads keep their
    name. Returns {workload key after rekey: (original workload, gpu or None)}.
    """
    gpus_seen: dict[str, set] = defaultdict(set)
    for rows in row_sets:
        for r in rows:
            gpus_seen[r["workload"]].add(gpu_short(r.get("gpu_name", "")))
    facets: dict[str, tuple[str, str | None]] = {}
    for rows in row_sets:
        for r in rows:
            wl = r["workload"]
            if len(gpus_seen[wl]) > 1:
                g = gpu_short(r.get("gpu_name", ""))
                key = f"{wl} [{g}]"
                if wl in WORKLOAD_LABEL:
                    WORKLOAD_LABEL.setdefault(key, f"{WORKLOAD_LABEL[wl]} [{g}]")
                r["workload"] = key
                facets[key] = (wl, g)
            else:
                facets[wl] = (wl, None)
    return facets


# ---------------------------------------------------------------------------
# Q1 — end-to-end wall time (normalized) & GPU idle
# ---------------------------------------------------------------------------

def plot_q1_walltime(q1: list[dict], fig_dir: Path) -> None:
    stats = agg(q1, "wall_time_s")
    if not stats:
        print("  [skip] eval_q1_walltime: no data")
        return
    workloads = sorted({wl for wl, _ in stats})
    configs = ordered_configs({c for _, c in stats})

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W if len(configs) > 5 else SINGLE_COL_W * 1.4, 2.2))
    group_w = 0.82
    bw = group_w / len(configs)
    for wi, wl in enumerate(workloads):
        base = stats.get((wl, "baseline"))
        base_mean = base["mean"] if base else None
        for ci, cfg in enumerate(configs):
            st = stats.get((wl, cfg))
            if not st:
                continue
            x = wi - group_w / 2 + bw * (ci + 0.5)
            if base_mean:
                norm = st["mean"] / base_mean
                err = st["std"] / base_mean
            else:
                norm, err = st["mean"], st["std"]   # absolute fallback
            label, color = CONFIG_STYLE.get(cfg, (cfg, PAL["lgray"]))
            ax.bar(x, norm, width=bw * 0.92, color=color,
                   yerr=err, error_kw={"elinewidth": 0.7, "capsize": 1.5,
                                       "ecolor": "#333333"})
            ax.annotate(f"{norm:.2f}", (x, norm), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=5.5,
                        color="#222222")
            ax.annotate(bar_annotation(st["n"]), (x, 0.015), ha="center",
                        va="bottom", fontsize=5, color="white", rotation=90)
    if any(stats.get((wl, "baseline")) for wl in workloads):
        ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_ylabel("Normalized wall time\n(baseline = 1.0)")
    else:
        ax.set_ylabel("Wall time (s)")
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels([WORKLOAD_LABEL.get(w, w) for w in workloads])
    handles = [Patch(facecolor=CONFIG_STYLE[c][1], label=CONFIG_STYLE[c][0])
               for c in configs if c in CONFIG_STYLE]
    ax.legend(handles=handles, ncol=min(3, len(handles)), loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, fig_dir, "eval_q1_walltime")


def plot_q1_gpu_idle(q1: list[dict], fig_dir: Path) -> None:
    stats = agg(q1, "gpu_all_idle_frac")
    stats = {k: v for k, v in stats.items() if v["n"] > 0}
    if not stats:
        print("  [skip] eval_q1_gpu_idle: no GPU profile data")
        return
    workloads = sorted({wl for wl, _ in stats})
    configs = ordered_configs({c for _, c in stats})
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.4, 2.0))
    bw = 0.82 / len(configs)
    for wi, wl in enumerate(workloads):
        for ci, cfg in enumerate(configs):
            st = stats.get((wl, cfg))
            if not st:
                continue
            x = wi - 0.41 + bw * (ci + 0.5)
            label, color = CONFIG_STYLE.get(cfg, (cfg, PAL["lgray"]))
            ax.bar(x, st["mean"] * 100, width=bw * 0.92, color=color,
                   yerr=st["std"] * 100,
                   error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#333"})
            ax.annotate(bar_annotation(st["n"]), (x, 1), ha="center",
                        va="bottom", fontsize=5, color="white", rotation=90)
    ax.set_ylabel("All-GPU idle time (%)")
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels([WORKLOAD_LABEL.get(w, w) for w in workloads])
    handles = [Patch(facecolor=CONFIG_STYLE[c][1], label=CONFIG_STYLE[c][0])
               for c in configs if c in CONFIG_STYLE]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, fig_dir, "eval_q1_gpu_idle")


# ---------------------------------------------------------------------------
# Q2 — time breakdown & overlap recovered
# ---------------------------------------------------------------------------

def plot_q2_breakdown(q2: list[dict], fig_dir: Path) -> None:
    if not q2:
        print("  [skip] eval_q2_time_breakdown: no data")
        return
    keys = [s[0] for s in SEGMENTS]
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in q2:
        for k in keys + [OVERLAP_SEGMENT[0]]:
            v = fnum(r.get(k))
            if v is not None:
                groups[(r["workload"], r["config"])][k].append(v)
    if not groups:
        print("  [skip] eval_q2_time_breakdown: no data")
        return
    workloads = sorted({wl for wl, _ in groups})
    configs = ordered_configs({c for _, c in groups})
    n_bars = sum(1 for wl in workloads for c in configs if (wl, c) in groups)
    fig, ax = plt.subplots(figsize=(max(SINGLE_COL_W * 1.4, 0.55 * n_bars + 1.2), 2.4))

    xpos, xlabels = [], []
    x = 0.0
    for wl in workloads:
        for cfg in configs:
            g = groups.get((wl, cfg))
            if not g:
                continue
            n = max(len(v) for v in g.values())
            bottom = 0.0
            for key, label, color, hatch in SEGMENTS:
                vals = g.get(key, [])
                mean = sum(vals) / len(vals) if vals else 0.0
                ax.bar(x, mean, width=0.7, bottom=bottom, color=color,
                       hatch=hatch, edgecolor="white", linewidth=0.4)
                bottom += mean
            # Overlapped staging drawn as a hatched sidebar segment above the
            # stack base — it is NOT wall time (it runs in the background), so
            # it must not extend the stack.
            ov = g.get(OVERLAP_SEGMENT[0], [])
            ov_mean = sum(ov) / len(ov) if ov else 0.0
            if ov_mean > 0:
                ax.bar(x + 0.28, ov_mean, width=0.14, bottom=0,
                       color="none", edgecolor=OVERLAP_SEGMENT[2],
                       hatch="////", linewidth=0.6)
            ax.annotate(bar_annotation(n), (x, bottom * 0.02), ha="center",
                        va="bottom", fontsize=5, color="white", rotation=90)
            xpos.append(x)
            xlabels.append(CONFIG_STYLE.get(cfg, (cfg,))[0])
            x += 1.0
        x += 0.6
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_ylabel("Mean workflow time (s)")
    handles = [Patch(facecolor=c, label=l) for _, l, c, _ in SEGMENTS]
    handles.append(Patch(facecolor="none", edgecolor=OVERLAP_SEGMENT[2],
                         hatch="////", label=OVERLAP_SEGMENT[1]))
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    if len(workloads) > 1:
        ax.set_title(" / ".join(WORKLOAD_LABEL.get(w, w) for w in workloads))
    save_fig(fig, fig_dir, "eval_q2_time_breakdown")


def plot_q2_overlap(q2: list[dict], fig_dir: Path) -> None:
    stats = agg(q2, "overlap_recovered")
    stats = {k: v for k, v in stats.items()
             if k[1] not in ("baseline", "observe_only")}
    if not stats:
        print("  [skip] eval_q2_overlap_recovered: no data")
        return
    workloads = sorted({wl for wl, _ in stats})
    configs = ordered_configs({c for _, c in stats})
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.4, 2.0))
    bw = 0.82 / len(configs)
    for wi, wl in enumerate(workloads):
        for ci, cfg in enumerate(configs):
            st = stats.get((wl, cfg))
            if not st:
                continue
            x = wi - 0.41 + bw * (ci + 0.5)
            _, color = CONFIG_STYLE.get(cfg, (cfg, PAL["lgray"]))
            ax.bar(x, st["mean"] * 100, width=bw * 0.92, color=color,
                   yerr=st["std"] * 100,
                   error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#333"})
            ax.annotate(bar_annotation(st["n"]), (x, 2), ha="center",
                        va="bottom", fontsize=5, color="white", rotation=90)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Overlap recovered (%)")
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels([WORKLOAD_LABEL.get(w, w) for w in workloads])
    handles = [Patch(facecolor=CONFIG_STYLE[c][1], label=CONFIG_STYLE[c][0])
               for c in configs if c in CONFIG_STYLE]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, fig_dir, "eval_q2_overlap_recovered")


def plot_q2_timeline(eval_root: Path, fig_dir: Path,
                     workload: str | None = None, gpu: str | None = None) -> None:
    """Gantt-style timeline for one representative full_system run."""
    # pick the completed full_system trial with median wall time
    cands = []
    for meta_path in sorted(eval_root.glob(f"runs/{workload or '*'}/full_system/t*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if gpu and gpu_short((meta.get("gpus") or [""])[0]) != gpu:
            continue
        trace = meta_path.parent / "trace.jsonl"
        if meta.get("status") == "completed" and trace.exists():
            cands.append((meta.get("wall_time_s") or 0, trace, meta))
    if not cands:
        print("  [skip] eval_q2_representative_timeline: no completed full_system run")
        return
    cands.sort(key=lambda c: c[0])
    wall, trace_path, meta = cands[len(cands) // 2]

    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    t0 = min(e["epoch_time"] for e in events if e.get("epoch_time"))

    llm_spans, tool_spans, prefetch_spans, wait_spans, diverg_ts = [], [], [], [], []
    open_chains: dict[str, tuple[str, float]] = {}
    open_tools: list[tuple[str, float]] = []
    task_start: dict[str, tuple[float, str]] = {}
    resource_names: dict[str, str] = {}
    task_resource: dict[str, str] = {}
    for e in events:
        et = e.get("event_type")
        p = e.get("payload", {}) or {}
        t = e.get("epoch_time", 0.0) - t0
        if et == "prediction_result":
            for r in p.get("resources", []):
                resource_names[r.get("resource_id", "")] = r.get("name", "")
        if et == "chain_start":
            open_chains[str(p.get("run_id"))] = (p.get("node", ""), t)
        elif et == "chain_end":
            key = str(p.get("run_id"))
            if key in open_chains:
                node, ts = open_chains.pop(key)
                if node in ("PlannerAgent", "WorkerAgent", "AggregatorAgent"):
                    llm_spans.append((node, ts, t))
        elif et == "tool_call":
            open_tools.append((p.get("tool", ""), t))
        elif et == "tool_end":
            dur = fnum(p.get("duration_s"), 0.0)
            name = p.get("tool", "")
            for i in range(len(open_tools) - 1, -1, -1):
                if open_tools[i][0] == name:
                    tool_spans.append((name, t - dur, t))
                    open_tools.pop(i)
                    break
        elif et == "prefetch_started":
            task_start[p.get("task_id", "")] = (t, p.get("resource_id", ""))
            task_resource[p.get("task_id", "")] = p.get("resource_id", "")
        elif et == "prefetch_completed":
            tid = p.get("task_id", "")
            if tid in task_start:
                ts, rid = task_start.pop(tid)
                # status=failed means nothing was staged; drawing it as a
                # prefetch span puts a bar on the timeline for work that never
                # happened (scheduler.py emits the event for failures too).
                if str(p.get("status") or "").lower() == "failed":
                    continue
                el = fnum(p.get("elapsed_s"), 0.0)
                end = ts + el if el and el > 0 else t
                prefetch_spans.append((resource_names.get(rid, rid[:10]), ts, end))
        elif et == "worker_swap_wait":
            w = fnum(p.get("wait_s"), 0.0)
            if w and w > 0.5:
                wait_spans.append(("swap wait", t - w, t))
        elif et == "divergence_detected":
            diverg_ts.append(t)

    # The WorkerAgent chain span includes the swap wait (the readiness barrier
    # runs inside on_chain_start) — cut the wait intervals out of the agent
    # lane so blocked time is not double-drawn as reasoning.
    def subtract_intervals(spans, cuts):
        out = []
        for name, s, e in spans:
            pieces = [(s, e)]
            for _, cs, ce in cuts:
                nxt = []
                for ps, pe in pieces:
                    if ce <= ps or cs >= pe:
                        nxt.append((ps, pe))
                    else:
                        if ps < cs:
                            nxt.append((ps, cs))
                        if ce < pe:
                            nxt.append((ce, pe))
                pieces = nxt
            out.extend((name, ps, pe) for ps, pe in pieces if pe - ps > 0.2)
        return out

    llm_spans = subtract_intervals(llm_spans, wait_spans)
    lanes = [
        ("Agent LLM", llm_spans, PAL["aqua"]),
        ("Tools", tool_spans, PAL["blue"]),
        ("Prefetch / staging", prefetch_spans, PAL["yellow"]),
        ("Exposed blocking", wait_spans, PAL["red"]),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 1.9))
    yticks, ylabels = [], []
    for li, (lane, spans, color) in enumerate(lanes):
        y = len(lanes) - 1 - li
        yticks.append(y)
        ylabels.append(lane)
        for name, s, e2 in spans:
            ax.barh(y, max(e2 - s, 0.4), left=s, height=0.55, color=color,
                    edgecolor="white", linewidth=0.3)
            if e2 - s > 0.06 * wall:
                ax.annotate(name, ((s + e2) / 2, y), ha="center", va="center",
                            fontsize=5, color="#222222")
    for dt in diverg_ts:
        ax.axvline(dt, color=PAL["violet"], linewidth=0.8, linestyle=":")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time since workflow start (s)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Representative run: {meta.get('workload')}/full_system "
                 f"({meta.get('run_id', '')[:40]})", fontsize=7)
    save_fig(fig, fig_dir, "eval_q2_representative_timeline")


# ---------------------------------------------------------------------------
# Q3 — prediction quality
# ---------------------------------------------------------------------------

def plot_q3_outcomes(q3: list[dict], fig_dir: Path) -> None:
    if not q3:
        print("  [skip] eval_q3_prediction_outcomes: no data")
        return
    keys = [k for k, _, _ in OUTCOME_STYLE]
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in q3:
        wlcfg = (r["workload"], r["config"])
        counts[wlcfg] += 1
        for k in keys:
            groups[wlcfg][k] += fnum(r.get(k), 0.0) or 0.0
    # skip configs with no prefetch activity at all
    groups = {k: v for k, v in groups.items() if sum(v.values()) > 0}
    if not groups:
        print("  [skip] eval_q3_prediction_outcomes: all zero")
        return
    workloads = sorted({wl for wl, _ in groups})
    configs = ordered_configs({c for _, c in groups})
    fig, ax = plt.subplots(figsize=(max(SINGLE_COL_W * 1.4, 0.5 * len(groups) + 1.4), 2.2))
    xpos, xlabels = [], []
    x = 0.0
    for wl in workloads:
        for cfg in configs:
            g = groups.get((wl, cfg))
            if not g:
                continue
            n = counts[(wl, cfg)]
            bottom = 0.0
            for k, label, color in OUTCOME_STYLE:
                v = g.get(k, 0.0) / n   # mean per run
                ax.bar(x, v, width=0.7, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.4)
                bottom += v
            ax.annotate(bar_annotation(n), (x, bottom + 0.05), ha="center",
                        va="bottom", fontsize=5, color="#333333")
            xpos.append(x)
            xlabels.append(CONFIG_STYLE.get(cfg, (cfg,))[0])
            x += 1.0
        x += 0.6
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.margins(y=0.12)
    ax.set_ylabel("Prediction outcomes\n(mean per run)")
    handles = [Patch(facecolor=c, label=l) for _, l, c in OUTCOME_STYLE]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save_fig(fig, fig_dir, "eval_q3_prediction_outcomes")


def plot_q3_precision_recall(q3: list[dict], fig_dir: Path) -> None:
    # One confusion matrix (per admitted task / per needed resource) plus a
    # separate timeliness rate; no "accuracy" — open-world speculation has no
    # countable TN, so accuracy is undefined here:
    #   precision    — admitted prefetches whose resource was consumed (FP=wasted)
    #   recall       — needed resources served by a prefetch (FN=missed)
    #   on-time rate — of served resources, share ready before the consumer
    acc = agg(q3, "on_time_rate")
    prec = agg(q3, "precision")
    rec = agg(q3, "recall")
    keys = sorted(set(acc) | set(prec) | set(rec))
    keys = [k for k in keys if prec.get(k, {}).get("n", 0) or rec.get(k, {}).get("n", 0)
            or acc.get(k, {}).get("n", 0)]
    if not keys:
        print("  [skip] eval_q3_precision_recall: no data")
        return
    fig, ax = plt.subplots(figsize=(max(SINGLE_COL_W * 1.4, 0.75 * len(keys) + 1.2), 2.0))
    xs = range(len(keys))
    series = [(prec, -0.26, PAL["blue"]), (rec, 0.0, PAL["aqua"]),
              (acc, 0.26, PAL["yellow"])]
    for i, k in enumerate(keys):
        for data, off, color in series:
            d = data.get(k)
            if d:
                ax.bar(i + off, d["mean"], width=0.24, color=color,
                       yerr=d["std"], error_kw={"elinewidth": 0.7, "capsize": 1.5})
                ax.annotate(f"{d['mean']:.2f}", (i + off, d["mean"]), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{WORKLOAD_LABEL.get(w, w)}\n{CONFIG_STYLE.get(c, (c,))[0]}"
                        for w, c in keys], fontsize=6)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(handles=[Patch(facecolor=PAL["blue"], label="Precision"),
                       Patch(facecolor=PAL["aqua"], label="Recall"),
                       Patch(facecolor=PAL["yellow"], label="On-time rate")],
              loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    save_fig(fig, fig_dir, "eval_q3_precision_recall")


def plot_q3_calibration(q3l: list[dict], fig_dir: Path) -> None:
    rows = [r for r in q3l if r.get("validated_hit") not in (None, "", "None")]
    if not rows:
        print("  [skip] eval_q3_confidence_calibration: no validated predictions")
        return
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    by_wl: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        conf = fnum(r.get("confidence"), 0.0) or 0.0
        hit = str(r.get("validated_hit")).lower() == "true"
        for bi, (lo, hi) in enumerate(buckets):
            if lo <= conf < hi:
                by_wl[r["workload"]][bi].append(hit)
                break
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7, linestyle="--",
            zorder=0, label="Perfect calibration")
    markers = ["o", "s", "^"]
    for wi, (wl, bmap) in enumerate(sorted(by_wl.items())):
        xs, ys, ns = [], [], []
        for bi, (lo, hi) in enumerate(buckets):
            obs = bmap.get(bi, [])
            if not obs:
                continue
            xs.append((lo + min(hi, 1.0)) / 2)
            ys.append(sum(obs) / len(obs))
            ns.append(len(obs))
        color = [PAL["blue"], PAL["orange"], PAL["green"]][wi % 3]
        ax.plot(xs, ys, marker=markers[wi % 3], markersize=4, linewidth=1.2,
                color=color, label=WORKLOAD_LABEL.get(wl, wl))
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), xytext=(3, -7),
                        textcoords="offset points", fontsize=5, color="#555555")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed hit rate")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    save_fig(fig, fig_dir, "eval_q3_confidence_calibration")


def plot_q3_leadtime(q3l: list[dict], fig_dir: Path) -> None:
    by_wl: dict[str, list[float]] = defaultdict(list)
    for r in q3l:
        if r.get("outcome") in ("useful", "late"):
            lt = fnum(r.get("lead_time_s"))
            if lt is not None:
                by_wl[r["workload"]].append(lt)
    if not any(by_wl.values()):
        print("  [skip] eval_q3_lead_time_cdf: no lead-time data")
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.0))
    for wi, (wl, vals) in enumerate(sorted(by_wl.items())):
        vals = sorted(vals)
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        color = [PAL["blue"], PAL["orange"], PAL["green"]][wi % 3]
        ax.step(vals, ys, where="post", color=color, linewidth=1.3,
                label=f"{WORKLOAD_LABEL.get(wl, wl)} (n={len(vals)})")
    ax.set_xlabel("Prefetch lead time before consumption (s)")
    ax.set_ylabel("Fraction of useful prefetches")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save_fig(fig, fig_dir, "eval_q3_lead_time_cdf")


# ---------------------------------------------------------------------------
# Q4 — speculation cost
# ---------------------------------------------------------------------------

def plot_q4_bytes(q4: list[dict], fig_dir: Path) -> None:
    if not q4:
        print("  [skip] eval_q4_speculation_bytes: no data")
        return
    seg = [("useful_bytes", "Useful", PAL["green"]),
           ("wasted_bytes", "Wasted", PAL["red"]),
           ("cancelled_bytes", "Cancelled (partial reads)", PAL["orange"])]
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in q4:
        for k, _, _ in seg:
            v = fnum(r.get(k))
            if v is not None:
                groups[(r["workload"], r["config"])][k].append(v)
    groups = {k: v for k, v in groups.items()
              if sum(sum(vals) for vals in v.values()) > 0}
    if not groups:
        print("  [skip] eval_q4_speculation_bytes: all zero (no byte data)")
        return
    workloads = sorted({wl for wl, _ in groups})
    configs = ordered_configs({c for _, c in groups})
    fig, ax = plt.subplots(figsize=(max(SINGLE_COL_W * 1.4, 0.5 * len(groups) + 1.4), 2.2))
    xpos, xlabels = [], []
    x = 0.0
    GB = 1e9
    for wl in workloads:
        for cfg in configs:
            g = groups.get((wl, cfg))
            if not g:
                continue
            n = max(len(v) for v in g.values())
            bottom = 0.0
            for k, label, color in seg:
                vals = g.get(k, [])
                mean = (sum(vals) / len(vals) / GB) if vals else 0.0
                ax.bar(x, mean, width=0.7, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.4)
                bottom += mean
            ax.annotate(bar_annotation(n), (x, bottom + 0.5), ha="center",
                        va="bottom", fontsize=5, color="#333333")
            xpos.append(x)
            xlabels.append(CONFIG_STYLE.get(cfg, (cfg,))[0])
            x += 1.0
        x += 0.6
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_ylabel("Speculative bytes moved per run (GB)")
    ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in seg],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save_fig(fig, fig_dir, "eval_q4_speculation_bytes")


def plot_q4_guard(q4: list[dict], fig_dir: Path) -> None:
    """full_system vs no_divergence_guard (and naive_prefetch for context)."""
    focus = ["full_system", "no_divergence_guard", "naive_prefetch"]
    metrics = [("wasted_bytes", "Wasted GB", 1e9),
               ("n_wasted", "# wrong prefetches", 1),
               ("n_wrong_after_divergence", "# wrong after divergence", 1),
               ("speculative_read_s", "Speculative I/O (s)", 1)]
    rows = [r for r in q4 if r["config"] in focus]
    if not rows:
        print("  [skip] eval_q4_divergence_guard_ablation: no data")
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(DOUBLE_COL_W, 1.8))
    for mi, (key, label, scale) in enumerate(metrics):
        ax = axes[mi]
        stats = agg(rows, key)
        configs = [c for c in focus if any(k[1] == c for k in stats)]
        for ci, cfg in enumerate(configs):
            vals = []
            for (wl, c2), st in stats.items():
                if c2 == cfg:
                    vals.extend(st["values"])
            if not vals:
                continue
            mean = sum(vals) / len(vals) / scale
            std = (math.sqrt(sum((v / scale - mean) ** 2 for v in vals) / (len(vals) - 1))
                   if len(vals) > 1 else 0.0)
            _, color = CONFIG_STYLE.get(cfg, (cfg, PAL["lgray"]))
            ax.bar(ci, mean, width=0.6, color=color, yerr=std,
                   error_kw={"elinewidth": 0.7, "capsize": 1.5})
            ax.annotate(f"n={len(vals)}", (ci, mean), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=5)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([CONFIG_STYLE.get(c, (c,))[0].replace(" ", "\n")
                            for c in configs], fontsize=5.5)
        ax.set_title(label, fontsize=7)
    save_fig(fig, fig_dir, "eval_q4_divergence_guard_ablation")


def plot_q4_memory(q4: list[dict], fig_dir: Path) -> None:
    stats = agg(q4, "gpu_peak_mem_mb")
    stats = {k: v for k, v in stats.items() if v["n"] > 0}
    if not stats:
        print("  [skip] eval_q4_memory_overhead: no GPU profile data")
        return
    workloads = sorted({wl for wl, _ in stats})
    configs = ordered_configs({c for _, c in stats})
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.4, 2.0))
    bw = 0.82 / len(configs)
    for wi, wl in enumerate(workloads):
        base = stats.get((wl, "baseline"))
        for ci, cfg in enumerate(configs):
            st = stats.get((wl, cfg))
            if not st:
                continue
            x = wi - 0.41 + bw * (ci + 0.5)
            _, color = CONFIG_STYLE.get(cfg, (cfg, PAL["lgray"]))
            val = st["mean"] / 1024.0
            ax.bar(x, val, width=bw * 0.92, color=color, yerr=st["std"] / 1024.0,
                   error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#333"})
            if base and cfg != "baseline":
                delta = (st["mean"] - base["mean"]) / 1024.0
                ax.annotate(f"{delta:+.1f}", (x, val), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=5)
            ax.annotate(bar_annotation(st["n"]), (x, 1), ha="center",
                        va="bottom", fontsize=5, color="white", rotation=90)
    ax.set_ylabel("Peak total GPU memory (GiB)\n(Δ vs baseline labeled)")
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels([WORKLOAD_LABEL.get(w, w) for w in workloads])
    handles = [Patch(facecolor=CONFIG_STYLE[c][1], label=CONFIG_STYLE[c][0])
               for c in configs if c in CONFIG_STYLE]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_fig(fig, fig_dir, "eval_q4_memory_overhead")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Q1–Q4 publication figures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    ap.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    args = ap.parse_args()
    eval_root = Path(args.eval_root)
    fig_dir = Path(args.fig_dir)

    apply_style()
    q1 = read_csv(eval_root / "eval_q1_summary.csv")
    q2 = read_csv(eval_root / "eval_q2_breakdown.csv")
    q3 = read_csv(eval_root / "eval_q3_prediction_quality.csv")
    q3l = read_csv(eval_root / "eval_q3_predictions_long.csv")
    q4 = read_csv(eval_root / "eval_q4_speculation_cost.csv")

    # Never blend GPU types in one aggregate: mixed-GPU workloads become
    # separate "wl [GPU]" keys in every figure, combined and per-workload.
    facets = facet_mixed_gpu(q1, q2, q3, q3l, q4)

    print("Q1 figures:")
    plot_q1_walltime(q1, fig_dir)
    plot_q1_gpu_idle(q1, fig_dir)
    print("Q2 figures:")
    plot_q2_breakdown(q2, fig_dir)
    plot_q2_overlap(q2, fig_dir)
    plot_q2_timeline(eval_root, fig_dir)
    print("Q3 figures:")
    plot_q3_outcomes(q3, fig_dir)
    plot_q3_precision_recall(q3, fig_dir)
    plot_q3_calibration(q3l, fig_dir)
    plot_q3_leadtime(q3l, fig_dir)
    print("Q4 figures:")
    plot_q4_bytes(q4, fig_dir)
    plot_q4_guard(q4, fig_dir)
    plot_q4_memory(q4, fig_dir)

    for wl in sorted(facets):
        slug = wl.replace(" [", "_").replace("]", "").replace(" ", "_")
        sub = fig_dir / "by_workload" / slug
        print(f"Per-workload figures ({wl}):")
        f1 = [r for r in q1 if r["workload"] == wl]
        f2 = [r for r in q2 if r["workload"] == wl]
        f3 = [r for r in q3 if r["workload"] == wl]
        f3l = [r for r in q3l if r["workload"] == wl]
        f4 = [r for r in q4 if r["workload"] == wl]
        if f1:
            plot_q1_walltime(f1, sub)
            plot_q1_gpu_idle(f1, sub)
        if f2:
            plot_q2_breakdown(f2, sub)
            plot_q2_overlap(f2, sub)
        orig_wl, gpu = facets[wl]
        plot_q2_timeline(eval_root, sub, workload=orig_wl, gpu=gpu)
        if f3:
            plot_q3_outcomes(f3, sub)
            plot_q3_precision_recall(f3, sub)
        if f3l:
            plot_q3_calibration(f3l, sub)
            plot_q3_leadtime(f3l, sub)
        if f4:
            plot_q4_bytes(f4, sub)
            plot_q4_guard(f4, sub)
            plot_q4_memory(f4, sub)
    print("Done.")


if __name__ == "__main__":
    main()
