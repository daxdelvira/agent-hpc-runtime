#!/usr/bin/env python3
"""
bench_potential_activation.py — split a tabulated EAM potential's load cost
into BYTE MOVEMENT vs ACTIVATION, and leave an artifact behind.

WHY THIS EXISTS
---------------
The paper's §3.2 claim is that activation, not I/O, dominates the cost of making
a resource usable: "of a 129 s potential load, raw sequential read is 5.45 s, so
>=123.5 s (95.8%) is not byte movement."

That measurement has NO SURVIVING ARTIFACT. An audit on 2026-08-03 searched
every log.lammps* in the tree; the only "Total wall time" values present are
0:00:00, and no log references w_eam4_big.fs at all. The number is asserted in
the plan and echoed in two source comments, each citing the others. It is the
sole evidence for a load-bearing claim, so it gets re-measured and WRITTEN DOWN.

WHAT IS MEASURED, in order, each from a cold page cache:
  1. read_s      — sequential read of the file, nothing else. Byte movement.
  2. lammps_s    — a 0-timestep LAMMPS run that does nothing but load the
                   potential and set up. read + parse + spline construction.
  3. lammps_warm_s — the same run with the cache left hot, to separate "bytes
                   from disk" from "bytes from RAM". The gap between lammps_s
                   and lammps_warm_s is the I/O share; what remains in
                   lammps_warm_s is irreducible activation.

(3) is the rung the original measurement lacked. Without it, "129 - 5.45" quietly
assumes LAMMPS's read runs at the same rate as a bare sequential read, which is
not guaranteed: LAMMPS reads through formatted C++ stream extraction, not read(2).

Cold caches use posix_fadvise(DONTNEED), the same primitive the runtime uses.
NOTE the known limitation: on Lustre, fadvise was measured leaving 56.2% of a
file resident, so a "cold" rung on Lustre understates the true cold cost. The
potential lives on project NFS, where gate (d) measured fadvise working (8.68x
read slowdown), so the eviction is trustworthy HERE -- but the script reports
the residency it achieved so the reader can judge rather than trust.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POSIX_FADV_DONTNEED = 4


def evict(path: str) -> None:
    """Drop `path` from the page cache. Requires no privileges."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    except AttributeError:  # very old Python
        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident_fraction(path: str) -> float:
    """Fraction of the file currently in the page cache, via mincore().

    Reported rather than assumed: an eviction that silently fails turns a "cold"
    number into a warm one, which is exactly how the L2-sleep result was
    misread on 2026-08-02.
    """
    # Call libc mmap directly rather than using Python's mmap object: taking a
    # ctypes pointer from a read-only mmap raises "underlying buffer is not
    # writable", and the file is genuinely read-only.
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_ubyte)]
    PROT_READ, MAP_SHARED, MAP_FAILED = 0x1, 0x01, ctypes.c_void_p(-1).value

    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    addr = None
    try:
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, MAP_FAILED):
            return float("nan")
        pagesize = os.sysconf("SC_PAGE_SIZE")
        npages = (size + pagesize - 1) // pagesize
        vec = (ctypes.c_ubyte * npages)()
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return float("nan")
        return sum(v & 1 for v in vec) / npages
    finally:
        if addr not in (None, MAP_FAILED):
            libc.munmap(ctypes.c_void_p(addr), size)
        os.close(fd)


def timed_read(path: str, bufsize: int = 8 << 20) -> float:
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while f.read(bufsize):
            pass
    return time.perf_counter() - t0


LAMMPS_IN = """\
units metal
boundary p p p
atom_style atomic
lattice bcc 3.165
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 183.84
pair_style {style}
pair_coeff * * {potential} W
run 0
"""


def run_lammps(lmp: str, potential: str, workdir: Path) -> tuple[float, str]:
    style = "eam/fs" if potential.endswith(".fs") else "eam/alloy"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.bench").write_text(
        LAMMPS_IN.format(style=style, potential=potential))
    t0 = time.perf_counter()
    p = subprocess.run([lmp, "-in", "in.bench"], cwd=str(workdir),
                       capture_output=True, text=True, timeout=7200)
    el = time.perf_counter() - t0
    if p.returncode != 0:
        return el, f"FAILED rc={p.returncode}: {(p.stderr or p.stdout)[-500:]}"
    return el, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", required=True,
                    help="absolute path to the .fs / .eam.alloy file")
    ap.add_argument("--lmp", default=os.environ.get("LMP_BIN", "lmp"),
                    help="LAMMPS binary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pot = os.path.abspath(args.potential)
    size = os.path.getsize(pot)
    host = os.uname().nodename
    out = Path(args.out or f"results/bench_potential_activation_{host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def rec(**kw):
        kw["t"] = time.time(); kw["host"] = host; kw["potential"] = pot
        kw["bytes"] = size
        rows.append(kw); out.write_text(json.dumps(rows, indent=2))
        print(f"  -> {json.dumps({k: v for k, v in kw.items() if k != 't'})}",
              flush=True)

    print(f"[bench] {pot}  ({size/1e9:.2f} GB)  lmp={args.lmp}", flush=True)

    # --- 1. raw sequential read, cold ---
    evict(pot)
    frac = resident_fraction(pot)
    rs = timed_read(pot)
    rec(rung="raw_read_cold", elapsed_s=round(rs, 2),
        resident_frac_before=round(frac, 4),
        mb_per_s=round(size / 1e6 / rs, 1))

    # --- 2. LAMMPS, cold ---
    evict(pot)
    frac = resident_fraction(pot)
    lc, note = run_lammps(args.lmp, pot, Path("results/_potbench_cold"))
    rec(rung="lammps_cold", elapsed_s=round(lc, 2),
        resident_frac_before=round(frac, 4), note=note)

    # --- 3. LAMMPS, warm (cache left hot by rung 2) ---
    frac = resident_fraction(pot)
    lw, note = run_lammps(args.lmp, pot, Path("results/_potbench_warm"))
    rec(rung="lammps_warm", elapsed_s=round(lw, 2),
        resident_frac_before=round(frac, 4), note=note)

    print("\n" + "=" * 70)
    print("POTENTIAL ACTIVATION SPLIT")
    print("=" * 70)
    print(f"  file                       {size/1e9:.2f} GB")
    print(f"  raw sequential read (cold) {rs:8.2f} s   <- byte movement")
    print(f"  LAMMPS load (cold cache)   {lc:8.2f} s")
    print(f"  LAMMPS load (warm cache)   {lw:8.2f} s   <- I/O already paid")
    if lc > 0:
        io_share = max(lc - lw, 0.0)
        print(f"\n  I/O share      = cold - warm = {io_share:8.2f} s "
              f"({100*io_share/lc:.1f}% of the cold load)")
        print(f"  activation     = warm        = {lw:8.2f} s "
              f"({100*lw/lc:.1f}% of the cold load)")
        print("\n  The activation share is the part no prefetcher can remove by")
        print("  moving bytes earlier -- it is work in the consumer process.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
