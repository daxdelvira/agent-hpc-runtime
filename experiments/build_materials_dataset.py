#!/usr/bin/env python
"""
build_materials_dataset.py — assemble a REAL materials dataset for the ChemGraph
out-of-core ensemble tool by pulling crystal structures from the Crystallography
Open Database (COD).

We sweep the 9,000,000+ COD ID range (the American Mineralogist Crystal Structure
Database — real experimental mineral crystals: inorganic, small unit cells, common
elements all covered by MACE-MP-0).  Each CIF is validated with ASE and kept only
if it parses, has a periodic cell, and its atom count is in a workable range, so
per-structure MACE runs are seconds-to-minutes (not hours).

No API key required.  Output: a directory of .cif files = the ensemble's
`input_structure_directory`.

Usage:
    python experiments/build_materials_dataset.py --out data/materials_ensemble \
        --n 150 --min-atoms 4 --max-atoms 80
"""
import argparse
import os
import sys
import time
import urllib.request

COD_URL = "https://www.crystallography.net/cod/{cid}.cif"


def try_download(cid: int, timeout: float) -> bytes | None:
    try:
        with urllib.request.urlopen(COD_URL.format(cid=cid), timeout=timeout) as r:
            if r.status != 200:
                return None
            data = r.read()
            return data if data and b"_cell_length_a" in data else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--n", type=int, default=150, help="target number of structures")
    ap.add_argument("--start-id", type=int, default=9000000, help="first COD id to try")
    ap.add_argument("--min-atoms", type=int, default=4)
    ap.add_argument("--max-atoms", type=int, default=80)
    ap.add_argument("--max-tries", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=12.0)
    args = ap.parse_args()

    from ase.io import read as ase_read

    os.makedirs(args.out, exist_ok=True)
    kept, tried = 0, 0
    cid = args.start_id
    t0 = time.time()
    manifest = open(os.path.join(args.out, "MANIFEST.tsv"), "w")
    manifest.write("cod_id\tformula\tn_atoms\tcell_volume\n")

    while kept < args.n and tried < args.max_tries:
        tried += 1
        data = try_download(cid, args.timeout)
        cid += 1
        if data is None:
            continue
        tmp = os.path.join(args.out, f"cod_{cid-1}.cif")
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            atoms = ase_read(tmp)
            n = len(atoms)
            vol = float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0
            if not (args.min_atoms <= n <= args.max_atoms) or atoms.cell.rank < 3:
                os.remove(tmp)
                continue
            kept += 1
            manifest.write(f"{cid-1}\t{atoms.get_chemical_formula()}\t{n}\t{vol:.1f}\n")
            manifest.flush()
            if kept % 10 == 0:
                print(f"  kept {kept}/{args.n} (tried {tried}, {time.time()-t0:.0f}s)",
                      flush=True)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            continue

    manifest.close()
    total_bytes = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out) if f.endswith(".cif")
    )
    print(f"\nDone: kept {kept} structures in {args.out} "
          f"({total_bytes/1e6:.1f} MB, tried {tried} ids, {time.time()-t0:.0f}s)")
    if kept == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
