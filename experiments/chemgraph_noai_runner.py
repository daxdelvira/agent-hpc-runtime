"""
chemgraph_noai_runner.py — Scripted no-LLM ChemGraph baseline.

Runs the canonical ChemGraph water-molecule geometry optimization directly
(molecule_name_to_smiles → smiles_to_coordinate_file → run_ase) without
any LLM agent, providing the non-agentic wall-time baseline for Plot 2.

Results are written to results/noai_chemgraph_<run_id>.json in the same
schema as summary_*.json so plot_walltime_comparison.py can read them.

Usage
-----
    python experiments/chemgraph_noai_runner.py [--n-runs N] [--outdir DIR]
    # Run 10 times to get best-case distribution:
    python experiments/chemgraph_noai_runner.py --n-runs 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_CG_SRC = _PROJECT_ROOT.parent / "ChemGraph" / "src"
if _CG_SRC.exists():
    sys.path.insert(0, str(_CG_SRC))


def run_scripted_chemgraph(
    molecule_name: str = "water",
    smiles: str | None = None,
    output_dir: Path | None = None,
    mace_device: str = "cpu",
    calculator: str = "mace_mp",
) -> dict:
    """
    Execute the canonical ChemGraph tool sequence directly via .invoke() (no LLM).

    Returns a dict with timing info compatible with summary_*.json.

    Parameters
    ----------
    calculator:
        "mace_mp"  — MACE-MP medium model (requires MACE + CUDA libs on LD_LIBRARY_PATH)
        "emtcalc"  — Effective Medium Theory (fast, no GPU, for smoke-test only)
        "TBLite"   — GFN2-xTB via tblite (accurate, CPU-only, no GPU needed)
    """
    import tempfile

    work_dir = output_dir or Path(tempfile.mkdtemp(prefix="cg_noai_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    from chemgraph.tools.cheminformatics_tools import (
        molecule_name_to_smiles,
        smiles_to_coordinate_file,
    )
    from chemgraph.tools.ase_tools import run_ase

    # Step 1: molecule_name_to_smiles
    t0 = time.perf_counter()
    result1 = molecule_name_to_smiles.invoke({"name": molecule_name})
    smiles = result1.get("smiles") or smiles or "O"
    print(f"  Step1: {molecule_name} → {smiles}")
    timings["molecule_name_to_smiles"] = time.perf_counter() - t0

    # Step 2: smiles_to_coordinate_file
    t0 = time.perf_counter()
    xyz_path = str(work_dir / f"{molecule_name}.xyz")
    smiles_to_coordinate_file.invoke({"smiles": smiles, "output_file": xyz_path})
    print(f"  Step2: XYZ → {xyz_path}")
    timings["smiles_to_coordinate_file"] = time.perf_counter() - t0

    # Step 3: run_ase (geometry optimization)
    t0 = time.perf_counter()
    results_json = str(work_dir / f"{molecule_name}_opt.json")

    if "mace" in calculator.lower():
        calc_cfg: dict = {
            "calculator_type": calculator,  # "mace_mp" or "mace_off"
            "model": "medium",
            "device": mace_device,
        }
    else:
        calc_cfg = {"calculator_type": calculator}

    run_ase.invoke({"params": {
        "input_structure_file": xyz_path,
        "output_results_file":  results_json,
        "driver": "opt",
        "fmax": 0.05,
        "calculator": calc_cfg,
    }})
    print(f"  Step3: ASE opt done")
    timings["run_ase"] = time.perf_counter() - t0

    wall_time_s = time.perf_counter() - t_start

    return {
        "wall_time_s": wall_time_s,
        "workflow": "chemgraph_mace",
        "mode": "noai_scripted",
        "condition": "noai",
        "step_timings": timings,
        "molecule": molecule_name,
        "smiles": smiles,
        "calculator": calculator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted no-LLM ChemGraph baseline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument(
        "--outdir",
        default=str(_PROJECT_ROOT / "results"),
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--mace-device",
        default="cpu",
        help="PyTorch device for MACE (cpu | cuda | cuda:0)",
    )
    parser.add_argument(
        "--calculator",
        default="mace_mp",
        help="Calculator type: mace_mp (default), emtcalc (fast test), TBLite",
    )
    parser.add_argument(
        "--molecule",
        default="water",
        help="Molecule name to pass to molecule_name_to_smiles",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    work_base = _PROJECT_ROOT / "results" / "noai_work"

    all_times: list[float] = []
    for i in range(args.n_runs):
        run_id = f"noai_cg_{uuid.uuid4().hex[:8]}"
        work_dir = work_base / run_id
        print(f"\n[run {i+1}/{args.n_runs}] run_id={run_id}")

        result = run_scripted_chemgraph(
            molecule_name=args.molecule,
            output_dir=work_dir,
            mace_device=args.mace_device,
            calculator=args.calculator,
        )
        result["run_id"] = run_id
        all_times.append(result["wall_time_s"])

        out_path = outdir / f"noai_chemgraph_{run_id}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

        print(f"  wall_time={result['wall_time_s']:.2f}s  "
              f"steps={result['step_timings']}")
        print(f"  Saved: {out_path}")

    if all_times:
        print(f"\nSummary across {len(all_times)} runs:")
        print(f"  mean: {sum(all_times)/len(all_times):.2f}s")
        print(f"  min:  {min(all_times):.2f}s")
        print(f"  max:  {max(all_times):.2f}s")


if __name__ == "__main__":
    main()
