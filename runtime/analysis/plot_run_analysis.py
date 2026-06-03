"""
runtime/analysis/plot_run_analysis.py
--------------------------------------
Generate a 4-panel analysis figure from an atomagents_metrics_*.csv file:

  Panel 1  Timeline / Gantt — real wall-clock phases + prefetch overlay
  Panel 2  GPU memory (GPU 0) over time — shows qwen_32b load spike
  Panel 3  CPU % (main process) over time — tool vs LAMMPS vs LLM
  Panel 4  PCIe throughput over time — host↔GPU data transfer

Usage
-----
    python runtime/analysis/plot_run_analysis.py \
        results/atomagents_metrics_d3e412f6-ec0.csv \
        --output results/run_analysis_d3e412f6.png

    # Specify measured 32B load time if known (from orchestrator log):
    python runtime/analysis/plot_run_analysis.py \
        results/atomagents_metrics_d3e412f6-ec0.csv \
        --model-load-s 155.1
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

# ─── colour palette ────────────────────────────────────────────────────────────
C = {
    "lammps":    "#4A90D9",
    "tool":      "#7ED321",
    "llm":       "#9B59B6",
    "agent":     "#2C3E50",
    "prefetch":  "#F5A623",
    "overlap":   "#F5A623",
    "bg":        "#12122A",
    "grid":      "#2A2A4A",
    "text":      "#E0E0E0",
    "gpu_mem":   "#E74C3C",
    "pcie_rx":   "#1ABC9C",
    "pcie_tx":   "#3498DB",
    "cpu":       "#F39C12",
    "rss":       "#BDC3C7",
}


# ─── data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_phases(rows: list[dict]) -> list[dict]:
    """
    Parse rows into phase records with absolute (relative-to-t0) start/end times.
    Excludes SUMMARY and agent: wrapper rows for leaf-level detail.
    """
    phases = []
    t0 = None

    for r in rows:
        ts = datetime.fromisoformat(r["timestamp"]).timestamp()
        dur = float(r["duration_s"])
        start = ts - dur
        if t0 is None:
            t0 = start

        phase_name = r["phase"]
        if phase_name in ("SUMMARY",):
            continue

        gpu  = int(r["gpu_mem_bytes"])
        rx   = int(r["pcie_rx_bytes_s"])
        tx   = int(r["pcie_tx_bytes_s"])
        cpu  = float(r["cpu_percent"])
        rss  = int(r["mem_rss_bytes"])

        # Determine category
        if phase_name.startswith("lammps:"):
            kind = "lammps"
        elif phase_name.startswith("tool:"):
            kind = "tool"
        elif phase_name.startswith("llm:"):
            kind = "llm"
        elif phase_name.startswith("agent:"):
            kind = "agent"
        else:
            kind = "other"

        phases.append({
            "name":  phase_name,
            "kind":  kind,
            "start": start - t0,
            "end":   ts - t0,
            "dur":   dur,
            "gpu":   gpu,
            "rx":    rx,
            "tx":    tx,
            "cpu":   cpu,
            "rss":   rss,
        })

    return phases


# ─── individual panels ─────────────────────────────────────────────────────────

def _fmt_s(x, _):
    """Axis formatter: seconds → 'Nm' or 'Ns'."""
    if x >= 60:
        return f"{x/60:.0f}m"
    return f"{x:.0f}s"


def draw_timeline(ax, phases: list[dict], model_load_s: float, total_s: float) -> None:
    """Horizontal Gantt showing leaf-level phases + prefetch overlay."""

    ROWS = {
        "lammps_zhou": 3,
        "lammps_eam":  2,
        "tool":        1,
        "prefetch":    0,
    }
    ROW_LABELS = {
        3: "LAMMPS\n(W_Zhou04)",
        2: "LAMMPS\n(w_eam4)",
        1: "Tool setup /\nLLM calls",
        0: "qwen_32b\nprefetch",
    }
    H = 0.55

    lammps_zhou_count = 0
    lammps_eam_count  = 0

    for ph in phases:
        if ph["kind"] == "agent":
            continue

        if ph["kind"] == "lammps":
            if "Zhou04" in ph["name"]:
                row = ROWS["lammps_zhou"]
                lammps_zhou_count += 1
                color = C["lammps"]
                label = f"{ph['dur']:.0f}s\n({375:.0f}s slow)"
            else:
                row = ROWS["lammps_eam"]
                lammps_eam_count += 1
                color = "#2471A3"
                label = f"{ph['dur']:.0f}s"
        elif ph["kind"] == "tool":
            row = ROWS["tool"]
            color = C["tool"]
            label = ""
        elif ph["kind"] == "llm":
            row = ROWS["tool"]
            color = C["llm"]
            label = ""
        else:
            continue

        w = ph["end"] - ph["start"]
        ax.barh(row, w, left=ph["start"], height=H,
                color=color, edgecolor="#1A1A3A", linewidth=0.4, zorder=2)
        if w > 30 and ph["kind"] == "lammps":
            ax.text(ph["start"] + w/2, row, label,
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)

    # Infer prefetch start: first LAMMPS starts just after first few LLM calls (~4s)
    lammps_starts = sorted(ph["start"] for ph in phases if ph["kind"] == "lammps")
    pf_start = max(0, lammps_starts[0] - 3) if lammps_starts else 4.0
    pf_end   = pf_start + model_load_s
    pf_row   = ROWS["prefetch"]

    ax.barh(pf_row, pf_end - pf_start, left=pf_start, height=H,
            color=C["prefetch"], edgecolor="#AA6800", linewidth=0.8, zorder=2)
    ax.text(pf_start + (pf_end - pf_start)/2, pf_row,
            f"{model_load_s:.0f}s measured", ha="center", va="center",
            fontsize=7.5, color="black", fontweight="bold", zorder=3)

    # Overlap shading (prefetch runs entirely inside first LAMMPS call)
    if lammps_starts:
        ov_start = lammps_starts[0]
        ov_end   = pf_end
        ax.axvspan(ov_start, ov_end, ymin=0.02, ymax=0.98,
                   color=C["overlap"], alpha=0.12, zorder=0)
        ax.axvline(pf_end, color=C["prefetch"], lw=1.2, ls="--", alpha=0.8, zorder=3)
        ax.annotate("32B ready\n(fully hidden)", xy=(pf_end, 3.3),
                    xytext=(pf_end + 30, 3.5), fontsize=7,
                    color=C["prefetch"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["prefetch"], lw=1))

    # Y-axis labels
    ax.set_yticks(list(ROW_LABELS.keys()))
    ax.set_yticklabels(list(ROW_LABELS.values()), fontsize=8, color=C["text"])
    ax.set_xlim(0, total_s * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_s))
    ax.set_xlabel("Wall clock (from experiment start)", fontsize=8, color=C["text"])
    ax.set_title("Workflow Timeline", color=C["text"], fontsize=10, loc="left")
    ax.tick_params(colors=C["text"])

    # Legend patches
    patches = [
        mpatches.Patch(color=C["lammps"],   label="LAMMPS W_Zhou04"),
        mpatches.Patch(color="#2471A3",      label="LAMMPS w_eam4"),
        mpatches.Patch(color=C["tool"],      label="Tool setup (Python)"),
        mpatches.Patch(color=C["llm"],       label="LLM call overhead"),
        mpatches.Patch(color=C["prefetch"],  label="qwen_32b load (background)"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              facecolor="#1A1A3A", edgecolor="#444", labelcolor=C["text"],
              ncol=2, framealpha=0.8)


def draw_gpu_memory(ax, phases: list[dict], model_load_s: float) -> None:
    """GPU 0 memory used over time (snapshot at phase end)."""
    pts = [(ph["end"], ph["gpu"] / 1024**3)
           for ph in phases if ph["gpu"] > 0 and ph["kind"] != "agent"]
    if not pts:
        ax.text(0.5, 0.5, "No GPU data", transform=ax.transAxes,
                ha="center", color=C["text"])
        return

    xs, ys = zip(*pts)
    # Step-plot from phase start
    step_x = [phases[0]["start"]]
    step_y = [ys[0]]
    for x, y in zip(xs, ys):
        step_x += [x, x]
        step_y += [step_y[-1], y]
    step_x.append(max(xs) * 1.02)
    step_y.append(step_y[-1])

    ax.fill_between(step_x, step_y, alpha=0.25, color=C["gpu_mem"], step=None)
    ax.plot(step_x, step_y, color=C["gpu_mem"], lw=1.5, label="GPU 0 used")
    ax.scatter(list(xs), list(ys), color=C["gpu_mem"], s=18, zorder=5)

    # Annotate the jump
    jump_idx = next((i for i,y in enumerate(ys) if y > 50), None)
    if jump_idx is not None:
        ax.annotate(f"qwen_32b loaded\n({ys[jump_idx]:.0f} GB)",
                    xy=(xs[jump_idx], ys[jump_idx]),
                    xytext=(xs[jump_idx] + 60, ys[jump_idx] - 20),
                    fontsize=7.5, color=C["gpu_mem"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["gpu_mem"], lw=1))

    # Horizontal reference: ~80% of 96GB
    ax.axhline(0.82 * 96, color=C["gpu_mem"], lw=0.6, ls=":", alpha=0.5)
    ax.text(max(xs) * 0.98, 0.82 * 96 + 1, "82% × 96 GB\n(gpu_mem_utilization)",
            ha="right", va="bottom", fontsize=6.5, color=C["gpu_mem"], alpha=0.7)

    ax.set_ylabel("GPU 0 used (GB)", fontsize=8, color=C["text"])
    ax.set_title("GPU 0 Memory (qwen_32b loads during LAMMPS)", color=C["text"],
                 fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_s))
    ax.tick_params(colors=C["text"])
    ax.legend(fontsize=7, facecolor="#1A1A3A", edgecolor="#444", labelcolor=C["text"])


def draw_cpu(ax, phases: list[dict]) -> None:
    """CPU % (main process) and RSS memory over time."""
    cpu_pts = [(ph["end"], ph["cpu"]) for ph in phases
               if ph["cpu"] > 0 and ph["kind"] != "agent"]
    rss_pts = [(ph["end"], ph["rss"] / 1024**2) for ph in phases
               if ph["rss"] > 0 and ph["kind"] != "agent"]

    ax2 = ax.twinx()

    if cpu_pts:
        xs, ys = zip(*cpu_pts)
        ax.scatter(list(xs), list(ys), color=C["cpu"], s=30, zorder=5, label="CPU %")
        for x, y in zip(xs, ys):
            ax.vlines(x, 0, y, color=C["cpu"], lw=1.5, alpha=0.6)
        ax.set_ylabel("Main-process CPU %\n(LAMMPS runs in subprocess)", fontsize=8, color=C["cpu"])

        # Annotate create_crystal spikes
        spikes = [(x,y) for x,y in zip(xs,ys) if y > 100]
        for sx, sy in spikes[:1]:
            ax.annotate("create_crystal\n(numpy/atomman)", xy=(sx, sy),
                        xytext=(sx + 40, sy + 5),
                        fontsize=7, color=C["cpu"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C["cpu"], lw=0.8))

    if rss_pts:
        xs2, ys2 = zip(*rss_pts)
        ax2.plot(list(xs2), list(ys2), color=C["rss"], lw=1.5, ls="--",
                 marker="o", markersize=4, label="RSS MB")
        ax2.set_ylabel("Main-process RSS (MB)", fontsize=8, color=C["rss"])
        ax2.tick_params(colors=C["rss"])
        ax2.set_ylim(bottom=0)

    ax.set_title("CPU % and Memory (main process)", color=C["text"], fontsize=10, loc="left")
    ax.set_ylim(0, 160)    # cap at 160% so spikes don't dominate
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_s))
    ax.tick_params(colors=C["cpu"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
              facecolor="#1A1A3A", edgecolor="#444", labelcolor=C["text"])

    # Shade LAMMPS phases to highlight "CPU idles while LAMMPS runs in subprocess"
    lammps_cpu = [(ph["start"], ph["end"]) for ph in phases if ph["kind"] == "lammps"]
    for s, e in lammps_cpu:
        ax.axvspan(s, e, alpha=0.06, color=C["lammps"], zorder=0)


def draw_pcie(ax, phases: list[dict]) -> None:
    """PCIe RX and TX throughput over time."""
    rx_pts = [(ph["end"], ph["rx"] / 1024) for ph in phases
              if ph["rx"] > 0 and ph["kind"] != "agent"]
    tx_pts = [(ph["end"], ph["tx"] / 1024) for ph in phases
              if ph["tx"] > 0 and ph["kind"] != "agent"]

    if rx_pts:
        xs, ys = zip(*rx_pts)
        ax.fill_between(list(xs), list(ys), alpha=0.3, color=C["pcie_rx"])
        ax.plot(list(xs), list(ys), color=C["pcie_rx"], lw=1.5,
                marker="o", markersize=4, label="PCIe RX (host→GPU)")

    if tx_pts:
        xs, ys = zip(*tx_pts)
        ax.fill_between(list(xs), list(ys), alpha=0.3, color=C["pcie_tx"])
        ax.plot(list(xs), list(ys), color=C["pcie_tx"], lw=1.5,
                marker="s", markersize=4, label="PCIe TX (GPU→host)")

    ax.set_ylabel("PCIe throughput (KB/s)", fontsize=8, color=C["text"])
    ax.set_xlabel("Wall clock (from experiment start)", fontsize=8, color=C["text"])
    ax.set_title("PCIe Throughput (GPU 0) — sampled at phase boundaries",
                 color=C["text"], fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_s))
    ax.tick_params(colors=C["text"])
    ax.legend(fontsize=7, facecolor="#1A1A3A", edgecolor="#444", labelcolor=C["text"])

    # Note: sampled at phase-end only
    ax.text(0.99, 0.95, "⚠ Sampled at phase end only\n(not continuous)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#AAA", style="italic")


# ─── main figure assembly ──────────────────────────────────────────────────────

def make_figure(csv_path: str, output_path: str, model_load_s: float = 155.1) -> None:
    rows   = load_csv(csv_path)
    phases = build_phases(rows)
    total_s = max(ph["end"] for ph in phases)

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(13, 14),
        facecolor=C["bg"],
        gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5], "hspace": 0.45},
    )

    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.tick_params(colors=C["text"])
        ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(axis="x", color=C["grid"], lw=0.5, alpha=0.7)

    draw_timeline(axes[0], phases, model_load_s, total_s)
    draw_gpu_memory(axes[1], phases, model_load_s)
    draw_cpu(axes[2], phases)
    draw_pcie(axes[3], phases)

    # Shared x-limits
    for ax in axes:
        ax.set_xlim(0, total_s * 1.01)

    fig.suptitle(
        "AtomAgents Exp2 — Run d3e412f6  •  Blackwell (375s LAMMPS slowdown)",
        fontsize=12, color=C["text"], fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {output_path}")
    plt.close()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _HAVE_MPL:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?",
                        default="results/atomagents_metrics_d3e412f6-ec0.csv")
    parser.add_argument("--output", default="results/run_analysis_d3e412f6.png")
    parser.add_argument("--model-load-s", type=float, default=155.1,
                        help="Measured 32B load time in seconds (from orchestrator log)")
    args = parser.parse_args()

    make_figure(args.csv, args.output, args.model_load_s)


if __name__ == "__main__":
    main()
