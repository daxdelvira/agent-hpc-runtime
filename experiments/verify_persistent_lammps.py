#!/usr/bin/env python3
"""
verify_persistent_lammps.py — does the retained instance compute the SAME physics?

WHY THIS MUST EXIST
-------------------
`persistent_lammps.py` makes retention work by REWRITING commands: it turns a
script's `create_box` into `delete_atoms` + `change_box` on a live instance, and
it skips a `pair_coeff` whose file is already loaded. Both are semantic changes
to someone else's simulation.

A rewritten run that completes and prints plausible numbers is NOT evidence that
it is right. This project has produced three withdrawn measurements with exactly
that shape: L2 sleep returned fast and generated "!!!!"; a 12.7% I/O share came
from two identically-cached rungs; pyhmmer's 91.8% was a filesystem artifact.
Each looked fine. So the mechanism is checked against ground truth -- the fork
path it replaces -- on physical observables, not on whether it ran.

WHAT IS COMPARED
----------------
The same two-geometry sequence the real workload performs, once through each path:

  FORK PATH        fresh lammps() per geometry, full pair_coeff each time
                   (what execution/runner.py does today)
  RETAINED PATH    one instance via PersistentLammpsPool, geometry changed in
                   place, pair_coeff issued once

Observables: atom count, total potential energy, and per-atom energy for each
geometry. Energies must agree to within tight tolerance -- these are the same
atoms at the same positions under the same potential, so any real difference
means the rewrite changed the simulation.

A PASS here licenses retention for THIS script shape only. Re-run it whenever a
script builder changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "workloads", "AtomAgents"))


def _style_for(p: str) -> str:
    return "eam/fs" if p.endswith(".fs") else "eam/alloy"


GEOM_A = dict(name="small_bcc", lattice="lattice bcc 3.165",
              region="region box block 0 2 0 2 0 2")
GEOM_B = dict(name="oriented_large",
              lattice="lattice bcc 3.165 orient x 1 -1 2 orient y 1 1 0 orient z -1 1 1",
              region="region box block 0 4 0 4 0 3")


def _observe(lmp) -> dict:
    """Physical observables that a wrong rewrite would perturb."""
    n = lmp.get_natoms()
    pe = lmp.get_thermo("pe")
    return {"natoms": int(n),
            "pe_total": float(pe),
            "pe_per_atom": (float(pe) / n) if n else None}


def run_fork(potential: str, logdir: str) -> list[dict]:
    """Ground truth: a fresh instance per geometry, exactly as today."""
    from lammps import lammps
    style = _style_for(potential)
    out = []
    for i, g in enumerate((GEOM_A, GEOM_B)):
        t0 = time.perf_counter()
        lmp = lammps(cmdargs=["-log", os.path.join(logdir, f"fork_{i}.log"),
                              "-screen", "none"])
        for c in (g["lattice"], g["region"], "create_box 1 box",
                  "create_atoms 1 box", "mass 1 183.84",
                  # The oriented cell exceeds the default neighbour-list slot
                  # count ("Neighbor list overflow, boost neigh_modify one").
                  # Issued IDENTICALLY on both paths so the comparison stays a
                  # comparison of the rewrite, not of two different setups.
                  "neigh_modify one 10000 page 1000000",
                  f"pair_style {style}",
                  f"pair_coeff * * {potential} W", "run 0"):
            lmp.command(c)
        obs = _observe(lmp)
        obs.update(geometry=g["name"], elapsed_s=round(time.perf_counter() - t0, 3))
        out.append(obs)
        lmp.close()
    return out


def run_retained(potential: str, logdir: str) -> tuple[list[dict], dict]:
    """The mechanism: one instance, geometry changed in place, one parse."""
    from atomagents.execution.persistent_lammps import PersistentLammpsPool
    style = _style_for(potential)
    pool = PersistentLammpsPool(max_retained=1)
    out = []
    for i, g in enumerate((GEOM_A, GEOM_B)):
        t0 = time.perf_counter()
        inst = pool.acquire(potential, style,
                            os.path.join(logdir, f"retained_{i}.log"))
        # Identical command stream to the fork path. The instance rewrites it.
        for c in (g["lattice"], g["region"], "create_box 1 box",
                  "create_atoms 1 box", "mass 1 183.84",
                  # The oriented cell exceeds the default neighbour-list slot
                  # count ("Neighbor list overflow, boost neigh_modify one").
                  # Issued IDENTICALLY on both paths so the comparison stays a
                  # comparison of the rewrite, not of two different setups.
                  "neigh_modify one 10000 page 1000000",
                  f"pair_style {style}",
                  f"pair_coeff * * {potential} W", "run 0"):
            inst.command(c)
        obs = _observe(inst.lmp)
        obs.update(geometry=g["name"], elapsed_s=round(time.perf_counter() - t0, 3))
        out.append(obs)
    stats = pool.stats()
    pool.release_all()
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential",
                    default="workloads/AtomAgents/potential_repository/w_eam4.fs")
    ap.add_argument("--logdir", default="/tmp")
    ap.add_argument("--out", default="results/verify_persistent_lammps.json")
    ap.add_argument("--rtol", type=float, default=1e-9,
                    help="relative tolerance on energies. These are identical "
                         "atoms under an identical potential, so the bar is "
                         "float noise, not physical agreement.")
    args = ap.parse_args()

    pot = os.path.abspath(args.potential)
    if not os.path.exists(pot):
        print(f"potential not found: {pot}")
        return 2
    print(f"potential {pot} ({os.path.getsize(pot)/1e9:.3f} GB)")

    print("\n--- FORK PATH (ground truth) ---", flush=True)
    fork = run_fork(pot, args.logdir)
    for r in fork:
        print(f"  {r['geometry']:16s} n={r['natoms']:>7}  pe={r['pe_total']:.10g}  "
              f"{r['elapsed_s']}s", flush=True)

    print("\n--- RETAINED PATH ---", flush=True)
    ret, stats = run_retained(pot, args.logdir)
    for r in ret:
        print(f"  {r['geometry']:16s} n={r['natoms']:>7}  pe={r['pe_total']:.10g}  "
              f"{r['elapsed_s']}s", flush=True)
    print(f"  pool: {stats}")

    # --- compare ------------------------------------------------------------
    diffs, ok = [], True
    for f, r in zip(fork, ret):
        same_n = f["natoms"] == r["natoms"]
        if f["pe_total"] == 0:
            rel = abs(r["pe_total"])
        else:
            rel = abs(r["pe_total"] - f["pe_total"]) / abs(f["pe_total"])
        passed = same_n and rel <= args.rtol
        ok &= passed
        diffs.append({"geometry": f["geometry"],
                      "natoms_fork": f["natoms"], "natoms_retained": r["natoms"],
                      "natoms_match": same_n,
                      "pe_fork": f["pe_total"], "pe_retained": r["pe_total"],
                      "pe_rel_diff": rel, "passed": passed,
                      "fork_s": f["elapsed_s"], "retained_s": r["elapsed_s"]})
        print(f"\n  {f['geometry']}: natoms {'OK' if same_n else 'MISMATCH'}"
              f" ({f['natoms']} vs {r['natoms']}), "
              f"pe rel diff {rel:.3e} -> {'PASS' if passed else 'FAIL'}")

    total_fork = sum(d["fork_s"] for d in diffs)
    total_ret = sum(d["retained_s"] for d in diffs)
    verdict = {
        "potential": pot,
        "potential_gb": round(os.path.getsize(pot) / 1e9, 3),
        "comparisons": diffs,
        "pool_stats": stats,
        "fork_total_s": round(total_fork, 3),
        "retained_total_s": round(total_ret, 3),
        "speedup": round(total_fork / total_ret, 3) if total_ret else None,
        "PASS": bool(ok),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n{'PASS' if ok else 'FAIL'} — "
          f"{'retained physics matches the fork path' if ok else 'THE REWRITE CHANGED THE PHYSICS; do not enable retention'}")
    print(f"fork {total_fork:.2f}s vs retained {total_ret:.2f}s "
          f"({verdict['speedup']}x)")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
