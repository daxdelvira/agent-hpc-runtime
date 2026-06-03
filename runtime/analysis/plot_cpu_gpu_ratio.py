"""
runtime/analysis/plot_cpu_gpu_ratio.py
----------------------------------------
Plot CPU vs GPU utilization throughout an AtomAgents run from the
system_profile_*.csv produced by SystemProfiler.

Three panels:
  1. CPU utilization by role — vLLM 72B tree, vLLM 32B tree, orchestration,
     and system-wide, all as % of one logical core (so 800% = 8 cores busy)
  2. GPU utilization % per card — shows which GPUs are active during inference
     vs idle during LAMMPS
  3. CPU:GPU ratio over time — aggregate CPU cores in use / GPU utilization,
     directly answering "how CPU-heavy is this agentic workload vs a regular
     LLM serving load?"

Usage
-----
    python runtime/analysis/plot_cpu_gpu_ratio.py \
        results/system_profile_<run_id>.csv \
        --output results/cpu_gpu_ratio_<run_id>.png

    # Overlay the phase boundaries from the metrics CSV for context:
    python runtime/analysis/plot_cpu_gpu_ratio.py \
        results/system_profile_<run_id>.csv \
        --metrics results/atomagents_metrics_<run_id>.csv \
        --output results/cpu_gpu_ratio_<run_id>.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

BG   = "#12122A"
GRID = "#2A2A4A"
TEXT = "#E0E0E0"

C = {
    "sys":      "#BDC3C7",
    "72b":      "#3498DB",
    "32b":      "#E74C3C",
    "orch":     "#2ECC71",
    "lammps":   "#4A90D9",
    "gpu0":     "#F39C12",
    "gpu1":     "#E67E22",
    "gpu2":     "#9B59B6",
    "gpu3":     "#1ABC9C",
    "ratio":    "#F5A623",
}


# ── loaders ────────────────────────────────────────────────────────────────────

def load_profile(path: str) -> dict[str, list]:
    """Load system_profile CSV into column-arrays."""
    cols: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def _f(series: list[str]) -> list[float]:
    """Parse a string series to floats, substituting -1 → NaN."""
    out = []
    for v in series:
        try:
            x = float(v)
            out.append(float("nan") if x < 0 else x)
        except (ValueError, TypeError):
            out.append(float("nan"))
    return out


def load_phases(metrics_csv: str) -> list[dict]:
    """Load phase boundaries from atomagents_metrics CSV for overlay."""
    from datetime import datetime
    phases = []
    t0 = None
    with open(metrics_csv, newline="") as f:
        for r in csv.DictReader(f):
            ts = datetime.fromisoformat(r["timestamp"]).timestamp()
            dur = float(r["duration_s"])
            start = ts - dur
            if t0 is None:
                t0 = start
            phases.append({
                "kind":  r["phase"].split(":")[0],
                "start": start - t0,
                "end":   ts - t0,
            })
    return phases


# ── drawing ────────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.7)
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.5)


def _shade_phases(ax, phases, ymax):
    """Draw light LAMMPS phase bands behind the plot."""
    for ph in phases:
        if ph["kind"] == "lammps":
            ax.axvspan(ph["start"], ph["end"], alpha=0.07,
                       color=C["lammps"], zorder=0)


def draw_cpu_panel(ax, t, cols, phases):
    """Panel 1: CPU utilization by role."""
    sys_cpu  = _f(cols.get("sys_cpu_pct", []))
    v72b_cpu = _f(cols.get("vllm_72b_cpu_pct", []))
    v32b_cpu = _f(cols.get("vllm_32b_cpu_pct", []))
    orch_cpu = _f(cols.get("orch_cpu_pct", []))

    # Smooth with a small window for readability
    def _smooth(y, w=3):
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w)/w, mode="same").tolist()

    if phases:
        _shade_phases(ax, phases, 800)

    ax.plot(t, _smooth(sys_cpu),  color=C["sys"],  lw=1.0, ls="--", alpha=0.6, label="System (all procs)")
    ax.fill_between(t, _smooth(v72b_cpu), alpha=0.35, color=C["72b"])
    ax.plot(t, _smooth(v72b_cpu), color=C["72b"],  lw=1.5, label="vLLM 72B tree")
    ax.fill_between(t, _smooth(v32b_cpu), alpha=0.35, color=C["32b"])
    ax.plot(t, _smooth(v32b_cpu), color=C["32b"],  lw=1.5, label="vLLM 32B tree")
    ax.fill_between(t, _smooth(orch_cpu), alpha=0.35, color=C["orch"])
    ax.plot(t, _smooth(orch_cpu), color=C["orch"], lw=1.2, label="Orchestration (AtomAgents)")

    # Annotation: typical regular-LLM CPU baseline
    ax.axhline(800, color="#888", lw=0.8, ls=":", alpha=0.5)
    ax.text(max(t)*0.98, 810, "~8 cores (typical vLLM baseline)",
            ha="right", va="bottom", fontsize=7, color="#888")

    n_cores = int(cols["sys_cpu_cores"][0]) if "sys_cpu_cores" in cols else 32
    ax.set_ylabel(f"CPU % (100% = 1 core)\nn_cores node = {n_cores}", fontsize=8, color=TEXT)
    ax.set_title("CPU Utilization by Role", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor="#1A1A3A", edgecolor="#444",
              labelcolor=TEXT, loc="upper right")
    _style(ax)


def draw_gpu_panel(ax, t, cols, phases):
    """Panel 2: GPU utilization % per card."""
    gpu_cols = sorted({k.rsplit("_", 1)[0]
                       for k in cols if k.startswith("gpu") and "_util_pct" in k})

    if phases:
        _shade_phases(ax, phases, 100)

    for gc in gpu_cols:
        idx   = gc.replace("gpu", "")
        util  = _f(cols.get(f"{gc}_util_pct", []))
        color = C.get(gc, "#AAA")
        # Assign label based on which model uses which GPU
        if int(idx) in (2, 3):
            label = f"GPU {idx} (qwen_72b)"
        elif int(idx) in (0, 1):
            label = f"GPU {idx} (qwen_32b)"
        else:
            label = f"GPU {idx}"
        ax.fill_between(t, util, alpha=0.25, color=color)
        ax.plot(t, util, color=color, lw=1.5, label=label)

    ax.set_ylim(0, 105)
    ax.set_ylabel("GPU utilization (%)", fontsize=8, color=TEXT)
    ax.set_title("GPU Utilization by Card", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor="#1A1A3A", edgecolor="#444",
              labelcolor=TEXT, loc="upper right")
    _style(ax)


def draw_ratio_panel(ax, t, cols, phases):
    """
    Panel 3: CPU-cores-in-use / GPU-utilization ratio.

    Ratio > baseline (e.g. > 2) indicates CPU-heavy agentic overhead.
    Ratio ≈ baseline indicates pure inference (GPU-bound).

    Computed as:
        cpu_cores_active = (vllm_72b_cpu + vllm_32b_cpu + orch_cpu) / 100
        avg_gpu_util     = mean(GPU 0..3 utilization %)
        ratio            = cpu_cores_active / (avg_gpu_util / 100 + 0.01)
    """
    v72b = _f(cols.get("vllm_72b_cpu_pct", []))
    v32b = _f(cols.get("vllm_32b_cpu_pct", []))
    orch = _f(cols.get("orch_cpu_pct", []))

    gpu_cols = sorted({k.rsplit("_", 1)[0]
                       for k in cols if k.startswith("gpu") and "_util_pct" in k})
    gpu_series = [_f(cols.get(f"{gc}_util_pct", [])) for gc in gpu_cols]

    n = len(t)
    ratio = []
    for i in range(n):
        cpu_cores = ((v72b[i] if not np.isnan(v72b[i]) else 0) +
                     (v32b[i] if not np.isnan(v32b[i]) else 0) +
                     (orch[i] if not np.isnan(orch[i]) else 0)) / 100.0
        gpu_utils = [s[i] for s in gpu_series if i < len(s) and not np.isnan(s[i])]
        avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        ratio.append(cpu_cores / (avg_gpu / 100.0 + 0.01))

    if phases:
        _shade_phases(ax, phases, max(r for r in ratio if not np.isnan(r)) * 1.1 if ratio else 50)

    import numpy as _np
    ratio_arr = _np.array(ratio)
    ax.fill_between(t, ratio_arr, alpha=0.3, color=C["ratio"])
    ax.plot(t, ratio_arr, color=C["ratio"], lw=1.5, label="CPU cores active / GPU util")

    # Annotate baseline for regular LLM serving (~8 CPU cores / 4 GPUs at ~80% each = ~2.5)
    baseline = (8 / 4) / 0.8   # ≈ 2.5
    ax.axhline(baseline, color="#888", lw=1.0, ls="--", alpha=0.7)
    ax.text(max(t)*0.98, baseline + 0.2,
            f"Regular LLM baseline ≈ {baseline:.1f}",
            ha="right", va="bottom", fontsize=7.5, color="#AAA")

    # Shaded "GPU-bound" region
    ax.axhspan(0, baseline * 1.5, alpha=0.04, color="#3498DB", zorder=0)
    ax.text(2, baseline * 0.4, "GPU-bound\n(inference)", fontsize=6.5,
            color="#3498DB", alpha=0.8, style="italic")

    ax.set_ylabel("CPU cores active / GPU fraction\n(higher = more CPU-heavy)", fontsize=8, color=TEXT)
    ax.set_xlabel("Wall clock (seconds from experiment start)", fontsize=8, color=TEXT)
    ax.set_title("CPU : GPU Ratio  — agentic overhead vs regular LLM serving",
                 fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, facecolor="#1A1A3A", edgecolor="#444", labelcolor=TEXT)
    _style(ax)


# ── main ───────────────────────────────────────────────────────────────────────

def make_figure(profile_csv: str, output_path: str,
                metrics_csv: str | None = None) -> None:
    cols = load_profile(profile_csv)
    t = _f(cols.get("t_rel_s", []))

    phases = []
    if metrics_csv:
        try:
            phases = load_phases(metrics_csv)
        except Exception as e:
            print(f"[plot] Could not load phases from {metrics_csv}: {e}", file=sys.stderr)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 11),
                             facecolor=BG,
                             gridspec_kw={"hspace": 0.42})

    draw_cpu_panel(axes[0], t, cols, phases)
    draw_gpu_panel(axes[1], t, cols, phases)
    draw_ratio_panel(axes[2], t, cols, phases)

    for ax in axes:
        ax.set_xlim(0, max(t) * 1.01 if t else 1)

    run_id = Path(profile_csv).stem.replace("system_profile_", "")
    fig.suptitle(
        f"CPU vs GPU Utilization — Run {run_id}\n"
        "Blue shading = LAMMPS phases  |  Ratio > baseline = agentic CPU overhead",
        fontsize=11, color=TEXT, fontweight="bold", y=1.005,
    )
    plt.savefig(output_path, dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close()


def main() -> None:
    if not _HAVE_MPL:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="system_profile_*.csv from SystemProfiler")
    parser.add_argument("--metrics", default=None,
                        help="atomagents_metrics_*.csv for phase boundary overlay")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out = args.output or args.profile.replace(".csv", ".png").replace(
        "system_profile_", "cpu_gpu_ratio_")
    make_figure(args.profile, out, args.metrics)


if __name__ == "__main__":
    main()
