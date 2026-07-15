#!/usr/bin/env python
"""
ensemble_tools.py — a Parsl-free MACE *ensemble* tool for ChemGraph, plus a
standalone timing harness.

The existing ChemGraph ensemble path (mcp/mace_mcp_parsl.py:run_mace_ensemble)
is MCP + Parsl-only, Parsl isn't installed here, and its config is Polaris/Aurora
specific.  This module reimplements the ensemble as a plain function (and a
LangChain @tool) that:
  - loads the MACE calculator ONCE and reuses it across every structure
    (the MCP path reloads the model per structure), and
  - iterates a directory of structure files, timing I/O (ase.io.read) vs
    compute (energy/opt) separately so we can see how much of the wall time is
    out-of-core disk access vs FLOPs.

Standalone:
    python experiments/ensemble_tools.py --dir data/materials_ensemble \
        --driver opt --model medium --device cpu [--limit N]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _load_calc(model: str, device: str):
    t0 = time.perf_counter()
    calc = None
    source = "disk"
    # Same prefetch-cache hook as MaceCalc.get_calculator: without it, a
    # prefetched calculator would never be consumed by the ensemble tool.
    if os.environ.get("RUNTIME_ENABLED"):
        try:
            from runtime.prefetch.mace_prefetch import get_cached_mace_calculator
            calc = get_cached_mace_calculator("mace_mp", model, wait_timeout_s=60.0)
            if calc is not None:
                source = "prefetch_cache"
        except ImportError:
            pass
    if calc is None:
        from mace.calculators import mace_mp
        calc = mace_mp(model=model, device=device)
    try:
        from chemgraph.instrumentation.workflow_tracker import tracker
        tracker.log_event("mace_load", {
            "calculator_type": "mace_mp",
            "model": str(model),
            "device": device,
            "duration_s": round(time.perf_counter() - t0, 3),
            "source": source,
        })
    except Exception:
        pass
    return calc


def run_mace_ensemble_local(
    input_structure_directory: str,
    driver: str = "energy",
    model: str = "medium",
    device: str = "cpu",
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "lbfgs",
    limit: int | None = None,
) -> dict:
    """
    Run MACE over every structure file in a directory with one shared calculator.

    Returns a compact summary plus a timing breakdown (io_s vs compute_s) so the
    caller can reason about out-of-core disk access vs compute.
    """
    from ase.io import read as ase_read
    from ase.optimize import BFGS, LBFGS, FIRE

    d = Path(input_structure_directory)
    if not d.is_dir():
        return {"status": "failure", "message": f"not a directory: {d}"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".cif", ".xyz", ".extxyz", ".poscar") or p.name.upper().startswith("POSCAR"))
    if limit:
        files = files[:limit]
    if not files:
        return {"status": "failure", "message": f"no structure files in {d}"}

    opts = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}
    t_load0 = time.perf_counter()
    calc = _load_calc(model, device)
    load_s = time.perf_counter() - t_load0

    io_s = 0.0
    compute_s = 0.0
    n_ok = 0
    n_atoms_total = 0
    results = []
    t_all0 = time.perf_counter()
    for p in files:
        try:
            t0 = time.perf_counter()
            atoms = ase_read(str(p))
            io_s += time.perf_counter() - t0
            atoms.calc = calc
            n_atoms_total += len(atoms)

            t1 = time.perf_counter()
            if driver == "energy" or len(atoms) <= 1:
                e = float(atoms.get_potential_energy())
                nsteps = 0
            else:
                dyn = opts.get(optimizer, LBFGS)(atoms, logfile=None)
                dyn.run(fmax=fmax, steps=steps)
                e = float(atoms.get_potential_energy())
                nsteps = dyn.get_number_of_steps()
            compute_s += time.perf_counter() - t1
            n_ok += 1
            results.append({"file": p.name, "n_atoms": len(atoms),
                            "energy_eV": e, "opt_steps": nsteps})
        except Exception as exc:
            results.append({"file": p.name, "error": str(exc)[:120]})
    wall_s = time.perf_counter() - t_all0

    return {
        "status": "success",
        "n_structures": len(files),
        "n_ok": n_ok,
        "n_atoms_total": n_atoms_total,
        "model_load_s": load_s,
        "io_s": io_s,
        "compute_s": compute_s,
        "wall_s": wall_s,
        "results": results[:5],  # compact payload for the LLM
        "message": (f"Ran MACE '{driver}' over {n_ok}/{len(files)} structures "
                    f"in {wall_s:.1f}s (io {io_s:.2f}s, compute {compute_s:.1f}s, "
                    f"model load {load_s:.1f}s)."),
    }


def make_ensemble_tool():
    """Return a LangChain @tool wrapping run_mace_ensemble_local."""
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class EnsembleInput(BaseModel):
        input_structure_directory: str = Field(description="Directory of structure files (CIF/XYZ) to run MACE over.")
        driver: str = Field(default="opt", description="'energy' or 'opt'.")
        model: str = Field(default="medium", description="MACE model size.")
        device: str = Field(default="cpu", description="cpu or cuda.")

    @tool("run_mace_ensemble", args_schema=EnsembleInput)
    def run_mace_ensemble(input_structure_directory: str, driver: str = "opt",
                          model: str = "medium", device: str = "cpu") -> dict:
        """Run a MACE energy/geometry calculation over EVERY structure file in a
        directory (an ensemble/batch job over a dataset on disk). Use this when a
        task asks to process, screen, or compute properties for a folder/dataset
        of many structures rather than a single molecule."""
        return run_mace_ensemble_local(input_structure_directory, driver=driver,
                                       model=model, device=device)

    return run_mace_ensemble


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--driver", default="opt")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = run_mace_ensemble_local(a.dir, driver=a.driver, model=a.model,
                                  device=a.device, limit=a.limit)
    for k, v in out.items():
        if k != "results":
            print(f"  {k}: {v}")
