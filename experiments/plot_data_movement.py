"""
plot_data_movement.py — Data movement / I/O comparison plot (Plot 1).

Shows grouped horizontal bars comparing data movement with and without MegaMMAP
for each workflow. Metrics:
  - Storage→DRAM (rchar_mb — all read() syscall bytes incl. page cache and GPFS)
    Falls back to bytes_read_mb (local block device only) for old-schema CSVs.
  - Writes (wchar_mb — all write() syscall bytes)
  - DRAM→GPU VRAM (PCIe transfers, pcie_rx delta where available)

rchar_mb is the preferred metric because it captures GPFS I/O (which POSIX
local disk counters miss) and any file the workflow consumed via read() calls,
regardless of storage tier.  bytes_read_mb > 0 specifically when Hermes NVMe
buffer is active (data flows through local block device instead of GPFS network).

Data sources
------------
AtomAgents:  mega_mmap_integration/results/stats_dict.csv
             variants: noai-nomega, noai-mega, agentic-nomega, agentic-mega
             columns:  rchar_mb, wchar_mb (new schema) or bytes_read_mb (legacy)
ChemGraph:   mega_mmap_integration/results/stats_dict.csv  (ChemGraph rows)
             OR ChemGraph/data_movement_*.csv  (legacy, disk_read + pcie_rx)
DeepDriveMD: deepdrivemd/frame_block_io_latency_*.csv  (per-frame bytes)
             deepdrivemd/frame_app_latency_*.csv        (app-level per-frame latency)

Usage
-----
    python experiments/plot_data_movement.py [--outdir DIR] [--dark-bg]

Options
-------
    --stats-csv   PATH  AtomAgents stats_dict.csv
    --cg-dir      DIR   ChemGraph data_movement CSV directory
    --ddmd-dir    DIR   DeepDriveMD CSV directory
    --outdir      DIR   output figure directory  (default: figures/)
    --dark-bg         use dark Gruvbox background
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from plot_utils import GRV, FS, apply_gruvbox_rc, save_figure

REPO_ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aa_io(stats_csv: Path) -> dict[str, dict[str, float]]:
    """
    Load AtomAgents I/O stats from stats_dict.csv.

    Returns:
        {variant_key: {"storage_dram_mb": x, "dram_gpu_mb": y, "runtime_s": z}}

    Variant keys: "noai_nomega", "noai_mega", "agentic_nomega", "agentic_mega"
    """
    if not stats_csv.exists():
        return {}

    # Read all rows; average duplicate variants
    MIN_RUNTIME_S = 5.0  # filter out clearly-failed runs (import errors → <2s)

    row_accumulator: dict[str, list[dict]] = defaultdict(list)
    with open(stats_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("app") != "AtomAgents":
                continue  # load_cg_io handles other apps separately
            v = row.get("variant", "")
            try:
                if float(row.get("runtime_s") or 0) < MIN_RUNTIME_S:
                    continue  # skip failed/trivial runs
            except ValueError:
                pass
            row_accumulator[v].append(row)

    result: dict[str, dict[str, float]] = {}

    def _mean(rows, field, fallback=0.0):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(field) or 0))
            except ValueError:
                pass
        return float(np.mean(vals)) if vals else fallback

    # Map variant names → unified keys (use I/O-stress variants for richer I/O data)
    VARIANT_MAP = {
        # Standard variants
        "noai-nomega":    "noai_nomega",
        "agentic-nomega": "agentic_nomega",
        "noai-mega":      "noai_mega",
        "agentic-mega":   "agentic_mega",
        # I/O-stress variants: explicitly write/read LAMMPS dump files
        "noai-nomega-io": "noai_nomega_io",
        "noai-mega-io":   "noai_mega_io",
    }

    for raw_key, rows in row_accumulator.items():
        mapped = VARIANT_MAP.get(raw_key)
        if mapped is None:
            continue

        # Prefer rchar_mb (all read() syscall bytes; captures GPFS + page cache).
        # Fall back to bytes_read_mb (local block device only; GPFS reads show ~0).
        rchar = _mean(rows, "rchar_mb")
        bytes_read = _mean(rows, "bytes_read_mb")
        storage_mb = rchar if rchar > 0.1 else bytes_read

        result[mapped] = {
            # Total data consumed via read(): regardless of storage tier
            "storage_dram_mb": storage_mb,
            # Local block-device reads (non-zero when Hermes NVMe tier is active)
            "local_read_mb":   bytes_read,
            # Total data produced via write(): includes GPFS dump files
            "written_mb":      _mean(rows, "wchar_mb") or _mean(rows, "bytes_written_mb"),
            # Local block-device writes (non-zero when Hermes NVMe tier is active)
            "local_write_mb":  _mean(rows, "bytes_written_mb"),
            "runtime_s":       _mean(rows, "runtime_s"),
        }

    return result


def _aa_io_pairs(aa_io: dict) -> list[tuple[str, float, float, bool]]:
    """
    Build (label, reads_mb, writes_mb, is_mega) tuples for AtomAgents bars.

    Shows all available variant pairs in order:
      1. Standard no-AI (read-heavy: Python+LAMMPS imports)
      2. I/O stress no-AI (write-heavy: LAMMPS dump files)
      3. Agentic (if available)
    Each pair is (no-MegaMMAP, + MegaMMAP) interleaved.
    """
    pairs = []

    def _entry(key: str, label: str, is_mega: bool):
        if key not in aa_io:
            return
        d = aa_io[key]
        pairs.append((label,
                      d.get("storage_dram_mb", 0.0),
                      d.get("written_mb", 0.0),
                      is_mega))

    # Standard: read-heavy (Python modules + LAMMPS potential/output files)
    _entry("noai_nomega", "  no-AI, no MegaMMAP", False)
    _entry("noai_mega",   "  no-AI + MegaMMAP", True)

    # I/O stress: write-heavy (2000 NVT steps, 499 MB dump files)
    _entry("noai_nomega_io", "  no-AI, no MegaMMAP (I/O stress)", False)
    _entry("noai_mega_io",   "  no-AI + MegaMMAP (I/O stress)", True)

    # Agentic (if available)
    _entry("agentic_nomega", "  agentic, no MegaMMAP", False)
    _entry("agentic_mega",   "  agentic + MegaMMAP", True)
    return pairs


def load_cg_io(cg_dir: Path, stats_csv: Path | None = None) -> dict[str, dict[str, float]]:
    """
    Load ChemGraph I/O stats, preferring stats_dict.csv (rchar_mb) over legacy CSV.

    Priority:
      1. stats_dict.csv rows where app=="ChemGraph" — prefers MACE-MP runs (rchar≈318 MB)
      2. ChemGraph/data_movement_*.csv — legacy source (disk_read ≈ 0 for GPFS)

    Returns:
        {"baseline": {"storage_dram_mb": x, "written_mb": y, ...},
         "mega":     {...}}    ← if MegaMMAP data is present
    """
    result: dict[str, dict[str, float]] = {}

    # --- Primary: stats_dict.csv ---
    if stats_csv is not None and stats_csv.exists():
        MIN_RUNTIME_S = 5.0
        cg_rows: dict[str, list[dict]] = defaultdict(list)
        with open(stats_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("app") != "ChemGraph":
                    continue
                try:
                    if float(row.get("runtime_s") or 0) < MIN_RUNTIME_S:
                        continue
                except ValueError:
                    pass
                cg_rows[row.get("variant", "")].append(row)

        def _mean(rows, field):
            vals = [float(r.get(field) or 0) for r in rows
                    if r.get(field) not in (None, "")]
            return float(np.mean(vals)) if vals else 0.0

        def _to_entry(rows):
            rchar = _mean(rows, "rchar_mb")
            bytes_read = _mean(rows, "bytes_read_mb")
            return {
                "storage_dram_mb": rchar if rchar > 0.1 else bytes_read,
                "local_read_mb":   bytes_read,
                "written_mb":      _mean(rows, "wchar_mb") or _mean(rows, "bytes_written_mb"),
                "local_write_mb":  _mean(rows, "bytes_written_mb"),
                "runtime_s":       _mean(rows, "runtime_s"),
                "n_runs":          len(rows),
            }

        # Prefer MACE-MP over EMT (MACE has higher rchar, more representative)
        # Nomega: prefer rows whose rchar_mb > 200 (MACE) over < 100 (EMT)
        nomega_mace = [r for r in cg_rows.get("noai-nomega", [])
                       if float(r.get("rchar_mb") or 0) > 200]
        nomega_any  = cg_rows.get("noai-nomega", [])
        nomega_rows = nomega_mace if nomega_mace else nomega_any
        if nomega_rows:
            result["baseline"] = _to_entry(nomega_rows)

        # Mega variants: prefer warm (page-cache saturated) over cold
        for vkey, rkey in [("noai-mega-warm", "mega_warm"),
                            ("noai-mega-cold", "mega_cold"),
                            ("noai-mega",      "mega")]:
            rows = cg_rows.get(vkey, [])
            if rows:
                result[rkey] = _to_entry(rows)

        if result:
            return result

    # --- Fallback: ChemGraph/data_movement_*.csv (legacy, disk_read≈0 for GPFS) ---
    if not cg_dir.exists():
        return {}

    csv_files = sorted(cg_dir.glob("data_movement_*.csv"))
    if not csv_files:
        return {}

    dram_reads: list[float] = []
    pcie_transfers: list[float] = []

    for f in csv_files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        begin_row = next(
            (r for r in rows if "begin" in r.get("phase", "")), None)
        end_row   = next(
            (r for r in rows if "end"   in r.get("phase", "")), None)

        if begin_row is None or end_row is None:
            continue

        def _delta(field: str) -> float:
            try:
                return (float(end_row.get(field) or 0) -
                        float(begin_row.get(field) or 0))
            except ValueError:
                return 0.0

        disk_read_bytes = max(0.0, _delta("disk_read"))
        pcie_rx_bytes   = max(0.0, _delta("pcie_rx"))

        if disk_read_bytes > 0 or pcie_rx_bytes > 0:
            dram_reads.append(disk_read_bytes / 1e6)
            pcie_transfers.append(pcie_rx_bytes / 1e6)

    if not dram_reads:
        return {}

    return {
        "baseline": {
            "storage_dram_mb": float(np.median(dram_reads)),
            "written_mb":      0.0,
            "dram_gpu_mb":     float(np.median(pcie_transfers)),
            "n_runs":          len(dram_reads),
        }
    }


def load_ddmd_io(ddmd_dir: Path) -> dict[str, dict[str, float]]:
    """
    Parse DeepDriveMD frame_block_io_latency_*.csv for I/O volume.

    Returns per-run I/O volume in MB.
    """
    if not ddmd_dir.exists():
        return {}

    csv_files = sorted(ddmd_dir.glob("frame_block_io_latency_*.csv"))
    if not csv_files:
        return {}

    run_bytes: list[float] = []

    for f in csv_files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        has_bytes = "bytes" in rows[0]
        total = 0.0
        n_frames = 0
        for row in rows:
            if has_bytes:
                try:
                    b = float(row.get("bytes") or 0)
                    # Sanity check: skip corrupt rows (huge values)
                    if b < 1e12:
                        total += b
                        n_frames += 1
                except (ValueError, TypeError):
                    pass
            else:
                n_frames += 1
                total += 6048.0  # 6 KB per frame (contact map)

        if n_frames > 0 and total > 0:
            run_bytes.append(total / 1e6)

    if not run_bytes:
        return {}

    return {
        "baseline": {
            "storage_dram_mb": float(np.median(run_bytes)),
            "dram_gpu_mb":     0.0,  # contact maps don't go to GPU directly
            "n_runs":          len(run_bytes),
        }
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _stacked_hbar(
    ax: plt.Axes,
    y: float,
    storage_mb: float,
    gpu_mb: float,
    bar_height: float,
    edge: str,
    color_storage: str,
    color_gpu: str,
    label_prefix: str | None = None,
    text_color: str = "#1d2021",
) -> None:
    """Draw one stacked horizontal bar: [storage→DRAM | DRAM→GPU]."""
    x_offset = 0.0

    if storage_mb > 0:
        ax.barh(y, storage_mb, height=bar_height, left=x_offset,
                color=color_storage, edgecolor=edge, linewidth=0.6)
        x_offset += storage_mb

    if gpu_mb > 0:
        ax.barh(y, gpu_mb, height=bar_height, left=x_offset,
                color=color_gpu, edgecolor=edge, linewidth=0.6)
        x_offset += gpu_mb

    total = storage_mb + gpu_mb
    if total > 0 and label_prefix:
        ax.text(total + total * 0.03, y,
                f"{total:.1f} MB",
                ha="left", va="center",
                fontsize=FS["small"] - 0.5,
                color=text_color, alpha=0.8)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_data_movement(
    stats_csv: Path,
    cg_dir: Path,
    ddmd_dir: Path,
    outdir: Path,
    dark_bg: bool = False,
) -> None:
    apply_gruvbox_rc(dark_bg=dark_bg)

    aa_io   = load_aa_io(stats_csv)
    cg_io   = load_cg_io(cg_dir, stats_csv=stats_csv)
    ddmd_io = load_ddmd_io(ddmd_dir)

    print("AtomAgents I/O:", {k: {f: f'{v:.3f}' for f, v in d.items()}
                               for k, d in aa_io.items()})
    print("ChemGraph I/O:", cg_io)
    print("DeepDriveMD I/O:", ddmd_io)

    text_color = GRV["fg"] if dark_bg else "#1d2021"
    grid_color = GRV["bg2"] if dark_bg else "#e8e8e8"
    edge_color = GRV["bg2"] if dark_bg else "#aaaaaa"

    COLOR_READS   = GRV["blue"]    # Read volume (rchar_mb: all read() syscalls)
    COLOR_WRITES  = GRV["orange"]  # Write volume (wchar_mb: all write() syscalls)
    COLOR_MEGA    = GRV["green"]   # MegaMMAP read volume (Hermes buffer active)
    COLOR_MEGA_W  = GRV["yellow"]  # MegaMMAP write volume
    COLOR_MISSING = GRV["gray"]    # Placeholder for missing MegaMMAP data
    # Legacy aliases for GPU/PCIe segment (kept for ChemGraph / DeepDriveMD paths)
    COLOR_STORAGE = COLOR_READS
    COLOR_GPU     = COLOR_WRITES

    bar_h  = 0.40
    group_gap = 0.55   # gap between workflow groups
    bar_gap   = 0.07   # gap between bars within a group

    fig, ax = plt.subplots(figsize=(9.5, 7))

    y_cur: float = 0.0
    y_ticks:  list[float] = []
    y_labels: list[str]   = []
    wf_yranges: dict[str, tuple[float, float]] = {}

    # ── AtomAgents ──────────────────────────────────────────────────────────
    aa_pairs = _aa_io_pairs(aa_io)
    y_start = y_cur
    for lbl, reads_mb, writes_mb, is_mega in aa_pairs:
        col_r = COLOR_MEGA   if is_mega else COLOR_READS
        col_w = COLOR_MEGA_W if is_mega else COLOR_WRITES
        _stacked_hbar(ax, y_cur, reads_mb, writes_mb, bar_h,
                      edge=edge_color,
                      color_storage=col_r, color_gpu=col_w,
                      label_prefix=lbl, text_color=text_color)
        # Draw a tiny reference diamond at x=0 for zero-length bars
        if reads_mb + writes_mb < 0.005:
            ax.plot(0.001, y_cur, marker="D", markersize=4,
                    color=col_r, zorder=5)
            ax.text(0.02, y_cur, "< 0.01 MB",
                    ha="left", va="center",
                    fontsize=FS["small"] - 1,
                    color=text_color, alpha=0.6)
        y_ticks.append(y_cur)
        y_labels.append(lbl)
        y_cur -= (bar_h + bar_gap)
    wf_yranges["AtomAgents"] = (y_start, y_cur + bar_h + bar_gap)
    y_cur -= group_gap

    # ── ChemGraph ───────────────────────────────────────────────────────────
    y_start = y_cur
    if "baseline" in cg_io:
        d = cg_io["baseline"]
        st = d.get("storage_dram_mb", 0.0)
        gp = d.get("dram_gpu_mb", 0.0)
        _stacked_hbar(ax, y_cur, st, gp, bar_h,
                      edge=edge_color,
                      color_storage=COLOR_STORAGE, color_gpu=COLOR_GPU,
                      label_prefix="  no MegaMMAP", text_color=text_color)
        y_ticks.append(y_cur);  y_labels.append("  no MegaMMAP")
        y_cur -= (bar_h + bar_gap)

    # Prefer cold (first-access) over warm over the plain mega key
    cg_mega_key = next((k for k in ("mega_cold", "mega_warm", "mega") if k in cg_io), None)
    if cg_mega_key:
        d = cg_io[cg_mega_key]
        st = d.get("storage_dram_mb", 0.0)
        gp = d.get("dram_gpu_mb", 0.0)
        suffix = " (cold)" if cg_mega_key == "mega_cold" else (" (warm)" if cg_mega_key == "mega_warm" else "")
        _stacked_hbar(ax, y_cur, st, gp, bar_h,
                      edge=edge_color,
                      color_storage=COLOR_MEGA, color_gpu=GRV["yellow"],
                      label_prefix=f"  + MegaMMAP{suffix}", text_color=text_color)
    else:
        # Missing MegaMMAP data — draw a hatched placeholder
        max_x = ax.get_xlim()[1] if ax.get_xlim()[1] > 0.1 else 3.5
        ax.barh(y_cur, max_x * 0.5, height=bar_h, left=0,
                color=COLOR_MISSING, edgecolor=edge_color,
                linewidth=0.6, alpha=0.3, hatch="///", zorder=3)
        ax.text(max_x * 0.5 * 0.5, y_cur, "MegaMMAP data pending",
                ha="center", va="center", fontsize=FS["small"] - 1,
                color=text_color, alpha=0.5, style="italic")
    y_ticks.append(y_cur);  y_labels.append("  + MegaMMAP")
    y_cur -= (bar_h + bar_gap)
    wf_yranges["ChemGraph"] = (y_start, y_cur + bar_h + bar_gap)
    y_cur -= group_gap

    # ── DeepDriveMD ─────────────────────────────────────────────────────────
    y_start = y_cur
    if "baseline" in ddmd_io:
        d = ddmd_io["baseline"]
        st = d.get("storage_dram_mb", 0.0)
        gp = d.get("dram_gpu_mb", 0.0)
        _stacked_hbar(ax, y_cur, st, gp, bar_h,
                      edge=edge_color,
                      color_storage=COLOR_STORAGE, color_gpu=COLOR_GPU,
                      label_prefix="  no MegaMMAP", text_color=text_color)
        y_ticks.append(y_cur);  y_labels.append("  no MegaMMAP")
        y_cur -= (bar_h + bar_gap)

    if "mega" in ddmd_io:
        d = ddmd_io["mega"]
        _stacked_hbar(ax, y_cur, d.get("storage_dram_mb", 0.0),
                      d.get("dram_gpu_mb", 0.0), bar_h,
                      edge=edge_color,
                      color_storage=COLOR_MEGA, color_gpu=GRV["yellow"],
                      label_prefix="  + MegaMMAP", text_color=text_color)
    else:
        max_x = ax.get_xlim()[1] if ax.get_xlim()[1] > 0.1 else 3.5
        ax.barh(y_cur, max_x * 0.15, height=bar_h, left=0,
                color=COLOR_MISSING, edgecolor=edge_color,
                linewidth=0.6, alpha=0.3, hatch="///", zorder=3)
        ax.text(max_x * 0.15 * 0.5, y_cur, "pending",
                ha="center", va="center", fontsize=FS["small"] - 1,
                color=text_color, alpha=0.5, style="italic")
    y_ticks.append(y_cur);  y_labels.append("  + MegaMMAP")
    y_cur -= (bar_h + bar_gap)
    wf_yranges["DeepDriveMD"] = (y_start, y_cur + bar_h + bar_gap)

    # ── Workflow group labels (drawn in the left margin) ──────────────────
    for wf_name, (yt, yb) in wf_yranges.items():
        wy = (yt + yb) / 2
        ax.text(
            -0.04, wy, wf_name,
            ha="right", va="center",
            fontsize=FS["annot"],
            fontweight="bold",
            color=text_color,
            transform=ax.get_yaxis_transform(),
            rotation=0,
        )
        # Horizontal separator line between groups
        ax.axhline(yb - group_gap / 2,
                   color=GRV["bg2"] if dark_bg else "#dddddd",
                   linewidth=0.6, linestyle=":",
                   xmin=0.0, xmax=1.0, zorder=0)

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=FS["tick"])
    ax.set_xlabel("I/O volume (MB)  —  rchar/wchar from full process tree", fontsize=FS["label"])
    ax.set_title(
        "Data Movement per Workflow Run (Baseline vs. MegaMMAP)",
        fontsize=FS["title"], fontweight="bold", loc="left",
    )
    ax.xaxis.grid(True, color=grid_color, linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Extra left margin for workflow labels
    ax.set_xlim(left=0)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLOR_READS,  label="Read volume (rchar — all read() syscalls)"),
        mpatches.Patch(color=COLOR_WRITES, label="Write volume (wchar — all write() syscalls)"),
        mpatches.Patch(color=COLOR_MEGA,   label="Read volume + MegaMMAP (Hermes active)"),
        mpatches.Patch(color=COLOR_MEGA_W, label="Write volume + MegaMMAP"),
        mpatches.Patch(color=COLOR_MISSING, alpha=0.4, hatch="///",
                       label="MegaMMAP data pending"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        fontsize=FS["annot"],
        framealpha=0.0,
        loc="lower right",
    )
    for t in leg.get_texts():
        t.set_color(text_color)

    fig.tight_layout()
    fig.subplots_adjust(left=0.25)
    save_figure(fig, outdir, "data_movement")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data movement / I/O comparison plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stats-csv",
        default=str(REPO_ROOT.parent / "mega_mmap_integration" / "results" /
                    "stats_dict.csv"),
    )
    parser.add_argument(
        "--cg-dir",
        default=str(REPO_ROOT.parent / "ChemGraph"),
    )
    parser.add_argument(
        "--ddmd-dir",
        default=str(REPO_ROOT.parent / "deepdrivemd"),
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "figures"),
    )
    parser.add_argument("--dark-bg", action="store_true")
    args = parser.parse_args()

    plot_data_movement(
        stats_csv=Path(args.stats_csv),
        cg_dir=Path(args.cg_dir),
        ddmd_dir=Path(args.ddmd_dir),
        outdir=Path(args.outdir),
        dark_bg=args.dark_bg,
    )


if __name__ == "__main__":
    main()
