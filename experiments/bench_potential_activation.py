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

WHAT THE 2026-08-09 REVISION ADDS, and why the two-way split was not enough
--------------------------------------------------------------------------
"activation = load_warm" is a RESIDUAL, not a mechanism: it is whatever survives
once the disk leg is paid, so it silently includes byte movement that never
touched the disk -- the kernel materialising the activated structure into the
process address space. bench_format_activation.py, which DOES record getrusage,
shows that residual is not negligible in general: its system time is a
format-INDEPENDENT ~0.35 s per GB of output array, and for npy/hdf5 that copy is
96-99% of the whole "activation" bucket. Those two formats do essentially no
transformation at all, yet the two-way split reports them as 72-77% activation.

So each LAMMPS rung now records the child's own resource usage, splitting the
warm bucket a second time:

    utime  -> transformation proper: lexing ASCII to doubles and building the
              spline coefficients. Work that CHANGES the representation.
    stime  -> kernel time: page faults, zeroing and copying to materialise the
              activated structure. Byte movement that happens to be RAM->RAM.

Only the utime share supports the paper's claim that no byte-oriented tier can
address this cost. The stime share is movement -- a byte mover still cannot
capture it (it is anonymous memory, not a file range), but it is not evidence
that the cost is transformation, and it must not be counted as such.

ru_maxrss additionally gives the activated footprint from the same run, which
independently checks the 5.10x expansion measured by bench_activated_residency.py.

Usage is read via os.wait4() on the LAMMPS child, so the numbers are that
child's alone -- not RUSAGE_CHILDREN, which accumulates over every reaped child
and would fold the cold rung into the warm one.

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


def run_lammps(lmp: str, potential: str, workdir: Path) -> tuple[float, str, dict]:
    """Run the load-only LAMMPS input and return (wall, note, rusage-of-child).

    os.wait4() rather than subprocess.run() so the usage belongs to THIS child
    and nothing else; RUSAGE_CHILDREN accumulates across every child the process
    has reaped, which would make the warm rung read as cold+warm. stdout/stderr
    go to files rather than pipes so there is no reason to drain them before
    waiting, and the LAMMPS log survives for inspection.
    """
    style = "eam/fs" if potential.endswith(".fs") else "eam/alloy"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.bench").write_text(
        LAMMPS_IN.format(style=style, potential=potential))

    t0 = time.perf_counter()
    with open(workdir / "stdout.log", "wb") as so, \
         open(workdir / "stderr.log", "wb") as se:
        p = subprocess.Popen([lmp, "-in", "in.bench"], cwd=str(workdir),
                             stdout=so, stderr=se)
        _, status, ru = os.wait4(p.pid, 0)
    el = time.perf_counter() - t0
    # Tell Popen the child is already reaped so __del__ does not warn or re-wait.
    p.returncode = os.waitstatus_to_exitcode(status)

    usage = {
        "utime_s": round(ru.ru_utime, 3),
        "stime_s": round(ru.ru_stime, 3),
        "cpu_per_wall": round((ru.ru_utime + ru.ru_stime) / el, 4) if el else None,
        "maxrss_kb": ru.ru_maxrss,
        "minflt": ru.ru_minflt,          # page faults served without disk I/O
        "majflt": ru.ru_majflt,          # faults that DID go to disk
        "inblock": ru.ru_inblock,
    }
    if p.returncode != 0:
        tail = (workdir / "stderr.log").read_text(errors="replace")[-500:] \
            or (workdir / "stdout.log").read_text(errors="replace")[-500:]
        return el, f"FAILED rc={p.returncode}: {tail}", usage
    return el, "ok", usage


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
    lc, note, ru_cold = run_lammps(args.lmp, pot, Path("results/_potbench_cold"))
    rec(rung="lammps_cold", elapsed_s=round(lc, 2),
        resident_frac_before=round(frac, 4), note=note, **ru_cold)

    # --- 3. LAMMPS, warm (cache left hot by rung 2) ---
    frac = resident_fraction(pot)
    lw, note, ru_warm = run_lammps(args.lmp, pot, Path("results/_potbench_warm"))
    rec(rung="lammps_warm", elapsed_s=round(lw, 2),
        resident_frac_before=round(frac, 4), note=note, **ru_warm)

    # A failed LAMMPS run still produces a wall time, a utime and a stime, and
    # the summary below would render them as a perfectly plausible split -- the
    # smoke test on a potential with no matching element printed "21.1%
    # transformation" for a run that never parsed a single table. Refuse.
    failures = [r for r in rows if str(r.get("note", "ok")).startswith("FAILED")]
    if failures:
        print("\n" + "!" * 70)
        print("NO SPLIT REPORTED -- a LAMMPS rung failed, so every number below")
        print("would be startup cost wearing the costume of an activation split.")
        for r in failures:
            print(f"  {r['rung']}: {r['note'][:300]}")
        print("!" * 70)
        print(f"\nwrote {out}  (rows kept for diagnosis)")
        return 1

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

        # Split the residual again: transformation vs RAM->RAM movement.
        u, s = ru_warm["utime_s"], ru_warm["stime_s"]
        rss_gb = ru_warm["maxrss_kb"] / 1e6
        print("\n" + "-" * 70)
        print("  THREE-WAY SPLIT of the cold load (the warm bucket decomposed)")
        print("-" * 70)
        print(f"  disk -> RAM        {io_share:8.2f} s  {100*io_share/lc:5.1f}%"
              "   movement a byte tier CAN address")
        print(f"  RAM -> process     {s:8.2f} s  {100*s/lc:5.1f}%"
              "   movement it CANNOT (anonymous memory)")
        print(f"  transformation     {u:8.2f} s  {100*u/lc:5.1f}%"
              "   parse + spline build")
        print(f"  unaccounted        {lc-io_share-s-u:8.2f} s  "
              f"{100*(lc-io_share-s-u)/lc:5.1f}%   wall not on CPU (I/O wait, sched)")
        print(f"\n  activated footprint (ru_maxrss)  {rss_gb:6.2f} GB"
              f"   = {rss_gb/(size/1e9):.2f}x the file")
        print(f"  stime per GB of activated struct {s/max(rss_gb,1e-9):6.2f} s/GB"
              "   (format-independent ~0.35 in bench_format_activation)")
        print(f"  cpu_per_wall (warm)              "
              f"{ru_warm['cpu_per_wall']}   <1 means the parse is I/O-bound or "
              "stalled; >1 means threaded")
        print("\n  Only the TRANSFORMATION row is evidence that the cost is not")
        print("  byte movement. The RAM->process row is movement, and counting")
        print("  it as activation is what overstated npy/hdf5 as 72-77%.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
