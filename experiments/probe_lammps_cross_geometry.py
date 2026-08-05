#!/usr/bin/env python3
"""
probe_lammps_cross_geometry.py — can a live LAMMPS instance serve a DIFFERENT
geometry without re-paying the potential parse?

WHY THIS GATES M2 FOR ATOMAGENTS
--------------------------------
E1 measured retention at 9.0x (42.83 s re-parse -> 4.78 s reuse), but its reuse
rung was `displace_atoms all random` + `run 0` -- the SAME box, same atom count,
perturbed positions. That is a legitimate second force evaluation, and it is not
what the real workload does.

The real trial issues THREE LAMMPS invocations per potential, with genuinely
different geometries (script_builders.py):

    run_lammps_lattice_constant.py   small bcc unit cell, lattice sweep
    run_lammps_screw_initial.py      large oriented dislocation cell
    run_lammps_relax_screw.py        that cell again, relaxed

The plan claims 4 of 6 parses per trial are redundant and recoverable. That claim
assumes a live instance can be re-pointed at a new geometry while KEEPING the
activated `pair_coeff` tables. In LAMMPS the pair tables are bound to atom types
within a simulation box, and the usual way to change geometry is `clear` -- which
destroys everything, including the tables we are trying to retain.

**If cross-geometry retention is impossible, M2 still works but recovers far less
than the plan claims**, because reuse would only be available WITHIN one
invocation, not across the three. That changes the size of the data-side
contribution and must be settled before a persistent worker is built around it.

WHAT THIS MEASURES
------------------
  A  parse_geom1     fresh instance, pair_coeff, small cell        (the baseline cost)
  B  same_geom_reuse displace + run 0, same box                    (E1's rung, control)
  C  cross_geom_keep NEW larger box in the SAME instance,
                     pair_coeff NOT re-issued                      (THE QUESTION)
  D  cross_geom_clear`clear` then rebuild then pair_coeff again    (the fallback cost)

C succeeding at ~B's cost means cross-geometry retention works and the plan's
claim stands. C failing, or costing ~D, means it does not.

Uses the SMALL potential by default: this is a question about LAMMPS semantics,
not about parse time, and the small file makes it a seconds-long probe. Pass
--potential to confirm on the big one.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def _style_for(potential: str) -> str:
    return "eam/fs" if potential.endswith(".fs") else "eam/alloy"


def new_instance(logfile: str):
    from lammps import lammps
    return lammps(cmdargs=["-log", logfile, "-screen", "none"])


def build_small(lmp, a=3.165):
    for c in (f"lattice bcc {a}",
              "region box block 0 2 0 2 0 2",
              "create_box 1 box",
              "create_atoms 1 box",
              "mass 1 183.84"):
        lmp.command(c)


def build_large(lmp, a=3.165):
    """A different box via create_box — the obvious approach, which LAMMPS
    rejects in a live instance ('Cannot create_box after simulation box is
    defined'). Kept as the rung that demonstrates why the next one is needed."""
    for c in (f"lattice bcc {a} orient x 1 -1 2 orient y 1 1 0 orient z -1 1 1",
              "region box2 block 0 4 0 4 0 3",
              "create_box 1 box2",
              "create_atoms 1 box",
              "mass 1 183.84"):
        lmp.command(c)


def rebuild_in_place(lmp, a=3.165):
    """Reach a different geometry WITHOUT create_box and WITHOUT clear.

    This is the only route that could preserve the activated pair tables:
    empty the box, resize it, and refill it. `pair_coeff` is never re-issued, so
    if the tables survive this, a live worker can serve all three of the
    workload's invocations off a single parse.
    """
    for c in ("delete_atoms group all",
              f"lattice bcc {a} orient x 1 -1 2 orient y 1 1 0 orient z -1 1 1",
              "change_box all x final 0 24 y final 0 24 z final 0 18 units box",
              "create_atoms 1 box",
              "mass 1 183.84"):
        lmp.command(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", default="workloads/AtomAgents/potential_repository/w_eam4.fs")
    ap.add_argument("--out", default="results/probe_lammps_cross_geometry.json")
    ap.add_argument("--logdir", default="/tmp")
    args = ap.parse_args()

    pot = os.path.abspath(args.potential)
    if not os.path.exists(pot):
        print(f"potential not found: {pot}")
        return 2
    style = _style_for(pot)
    recs = []

    def rec(**kw):
        kw["t"] = time.time()
        recs.append(kw)
        print("  " + json.dumps({k: v for k, v in kw.items() if k != "t"}), flush=True)

    print(f"potential {pot} ({os.path.getsize(pot)/1e9:.3f} GB), style {style}")

    # --- A: fresh instance, small geometry, pay the parse --------------------
    lmp = new_instance(os.path.join(args.logdir, "xg_a.log"))
    build_small(lmp)
    lmp.command(f"pair_style {style}")
    t0 = time.perf_counter()
    lmp.command(f"pair_coeff * * {pot} W")
    lmp.command("run 0")
    t_parse = time.perf_counter() - t0
    rec(rung="A_parse_geom1", elapsed_s=round(t_parse, 4),
        natoms=lmp.get_natoms())

    # --- B: same box, perturbed — E1's reuse rung, as a control --------------
    t0 = time.perf_counter()
    lmp.command("displace_atoms all random 0.1 0.1 0.1 12345")
    lmp.command("run 0")
    t_same = time.perf_counter() - t0
    rec(rung="B_same_geom_reuse", elapsed_s=round(t_same, 4),
        note="E1's rung: same box, new positions")

    # --- C: THE QUESTION. New, larger, differently-oriented box in the SAME
    # instance, pair_coeff NOT re-issued. Does LAMMPS even allow it?
    t0 = time.perf_counter()
    err = None
    try:
        build_large(lmp)
        lmp.command("run 0")
        t_cross = time.perf_counter() - t0
        ok = True
    except Exception as e:
        t_cross = time.perf_counter() - t0
        ok = False
        err = f"{type(e).__name__}: {str(e)[:200]}"
    rec(rung="C_cross_geom_keep_tables", elapsed_s=round(t_cross, 4),
        succeeded=ok, error=err,
        natoms=(lmp.get_natoms() if ok else None),
        note="new box in the live instance, pair_coeff NOT re-issued")

    # --- C2: the route that avoids create_box entirely. THE REAL QUESTION. ---
    t0 = time.perf_counter()
    err2 = None
    try:
        rebuild_in_place(lmp)
        lmp.command("run 0")
        t_inplace = time.perf_counter() - t0
        ok2 = True
    except Exception as e:
        t_inplace = time.perf_counter() - t0
        ok2 = False
        err2 = f"{type(e).__name__}: {str(e)[:200]}"
    rec(rung="C2_cross_geom_in_place", elapsed_s=round(t_inplace, 4),
        succeeded=ok2, error=err2,
        natoms=(lmp.get_natoms() if ok2 else None),
        note="delete_atoms + change_box + create_atoms; pair_coeff NOT re-issued")

    # --- D: the fallback — clear, rebuild, re-parse. What retention must beat.
    lmp2 = new_instance(os.path.join(args.logdir, "xg_d.log"))
    t0 = time.perf_counter()
    lmp2.command("clear")
    build_small(lmp2)
    rebuild_in_place(lmp2)
    lmp2.command(f"pair_style {style}")
    lmp2.command(f"pair_coeff * * {pot} W")
    lmp2.command("run 0")
    t_clear = time.perf_counter() - t0
    rec(rung="D_cross_geom_clear_reparse", elapsed_s=round(t_clear, 4),
        note="clear + rebuild + full re-parse — what the fork-per-call does today")

    verdict = {
        "rung": "VERDICT",
        "parse_s": round(t_parse, 4),
        "same_geom_reuse_s": round(t_same, 4),
        "cross_geom_keep_s": round(t_cross, 4),
        "cross_geom_succeeded": ok,
        "clear_and_reparse_s": round(t_clear, 4),
    }
    # Judge on the PARSE AVOIDED, not on totals. Both C2 and D pay the same
    # geometry cost (delete/resize/refill + force evaluation); only D also pays
    # the parse. Comparing totals hides the mechanism behind whichever term is
    # larger -- with the 9 MB potential the geometry work is ~2 s against a
    # 0.15 s parse, so totals look identical and the first version of this
    # verdict wrongly concluded "not cheap". The quantity that matters is
    # D - C2, which should approximate the parse cost.
    parse_avoided = t_clear - t_inplace
    verdict["cross_geom_in_place_s"] = round(t_inplace, 4)
    verdict["cross_geom_in_place_succeeded"] = ok2
    verdict["parse_avoided_s"] = round(parse_avoided, 4)
    verdict["parse_avoided_frac_of_call"] = (round(parse_avoided / t_clear, 4)
                                             if t_clear > 0 else None)
    verdict["NOTE"] = ("Judge on parse_avoided_s. A potential whose parse is "
                       "small relative to geometry+force work cannot show a win "
                       "no matter how well retention works -- run with the 3.32 GB "
                       "potential for the number that matters.")
    if ok2 and parse_avoided > 0.5 * t_parse:
        verdict["conclusion"] = (
            "CROSS-GEOMETRY RETENTION WORKS via delete_atoms + change_box + "
            "create_atoms (create_box is rejected in a live instance). A live "
            "worker can serve all three invocations off one parse; the plan's "
            "4-of-6-redundant claim stands.")
    elif ok2:
        verdict["conclusion"] = (
            "Cross-geometry rebuild is ALLOWED and the tables survive, but this "
            "run did not demonstrate a saving — most likely because the parse is "
            "small relative to the geometry and force work at this potential "
            "size. Re-run with the big potential before drawing any conclusion.")
    else:
        verdict["conclusion"] = (
            "CROSS-GEOMETRY RETENTION IS NOT POSSIBLE in one instance. M2 for "
            "AtomAgents can only reuse WITHIN an invocation, so the recoverable "
            "redundancy is much smaller than the plan claims. Do not build the "
            "worker around cross-invocation reuse.")
    rec(**verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\n{verdict['conclusion']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
