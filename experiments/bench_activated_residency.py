#!/usr/bin/env python3
"""
bench_activated_residency.py — E1/E2: what does it cost to KEEP a data resource
at rung R3 (activated, inside a live consumer) instead of re-paying for it?

WHY THIS EXISTS
---------------
experiments/bench_potential_activation.py established the cost SPLIT for
w_eam4_big.fs (3.32 GB): 1.87 s of byte movement (1.9%) against 98.23 s of
parse + spline construction (98.1%). That says a byte-oriented prefetcher can
recover at most 1.9% of the cost, but it does NOT say what the alternative is
worth, because it measured only fresh processes.

The workload runs LAMMPS as `subprocess.run(sys.executable, script)`
(execution/runner.py:26-45) — a fresh process per tool call, building a fresh
lammps() instance. Real trial directories contain three logs per potential
(log.lammps.lattice.constant.alloy, log.lammps_screw_initial,
log.lammps.relax_screw), so the same file is parsed 3x per potential, 6x per
trial. If a live instance can serve the 2nd and 3rd without re-parsing, that is
~4 of 6 parses recoverable with NO prediction of any kind — and it is invisible
to a byte prefetcher, because the page cache is already warm by parse #2.

WHAT IS MEASURED
----------------
  r1_parse_cold      fresh instance, page cache evicted   -> today's parse #1
  r2_parse_warm      fresh instance, page cache warm      -> today's parse #2/#3
  r3_reuse_live      SAME instance, new configuration,
                     pair_coeff NOT re-issued             -> the mechanism
  r4_repeat_coeff    SAME instance, pair_coeff re-issued  -> does LAMMPS cache?
  r5_hold_second     a second potential activated while
                     the first is still held              -> budget arithmetic

E2 (the activated size) falls out of the VmRSS delta across r1: this is an
IN-PROCESS module, so the resident-set delta across pair_coeff IS the size of
the activated structure. The standing estimate is ~9.8 GB of spline doubles
from a 3.32 GB text file; it has never been measured, and the whole
seconds-per-GB arbitration argument rests on it.

Page-cache residency is reported at every rung via mincore(), for the same
reason bench_potential_activation.py reports it: an eviction that silently
fails turns a cold number into a warm one.

CPU-ONLY. No GPU, no allocation needed.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import sys
import time
from pathlib import Path

POSIX_FADV_DONTNEED = 4


# --- page-cache helpers (verbatim from bench_potential_activation.py so the
# --- two benchmarks' "cold" and "warm" mean exactly the same thing) ----------

def evict(path: str) -> None:
    """Drop `path` from the page cache. Requires no privileges."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    except AttributeError:
        libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident_fraction(path: str) -> float:
    """Fraction of `path` currently in the page cache, via mincore()."""
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


def _proc_kb(field: str) -> int:
    """A field from /proc/self/status, in kB. 0 if absent."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def rss_gb() -> float:
    return _proc_kb("VmRSS") / (1024.0 * 1024.0)


def hwm_gb() -> float:
    return _proc_kb("VmHWM") / (1024.0 * 1024.0)


# --- the LAMMPS setup, matching what the workload actually builds -----------
# script_builders.py emits `units metal` + a bcc W lattice + eam/fs; the box is
# small on purpose so the potential parse dominates and we are measuring
# activation rather than MD.

_SETUP = [
    "units metal",
    "boundary p p p",
    "atom_style atomic",
    "lattice bcc 3.165",
    "region box block 0 2 0 2 0 2",
    "create_box 1 box",
    "create_atoms 1 box",
    "mass 1 183.84",
]


def _style_for(potential: str) -> str:
    return "eam/fs" if potential.endswith(".fs") else "eam/alloy"


def new_instance(logfile: str):
    from lammps import lammps
    return lammps(cmdargs=["-log", logfile, "-screen", "none"])


def build_and_time_coeff(lmp, potential: str) -> tuple[float, float, float]:
    """Run the setup, then time pair_style+pair_coeff (the parse) in isolation.

    Returns (coeff_seconds, rss_before_gb, rss_after_gb).
    """
    for cmd in _SETUP:
        lmp.command(cmd)
    lmp.command(f"pair_style {_style_for(potential)}")
    rss_before = rss_gb()
    t0 = time.perf_counter()
    lmp.command(f"pair_coeff * * {potential} W")
    lmp.command("run 0")          # forces the actual setup/interpolate
    el = time.perf_counter() - t0
    return el, rss_before, rss_gb()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", required=True,
                    help="path to the large potential (w_eam4_big.fs)")
    ap.add_argument("--second-potential", default=None,
                    help="a second, smaller potential for the hold-two rung")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pot = os.path.abspath(args.potential)
    if not os.path.exists(pot):
        print(f"ABORT: no such potential: {pot}", file=sys.stderr)
        return 2
    size_b = os.path.getsize(pot)

    # Resolve BEFORE the chdir below, or a relative --second-potential silently
    # fails its existence check and the hold-two rung is skipped without a word.
    second_pot = os.path.abspath(args.second_potential) \
        if args.second_potential else None

    workdir = Path(args.workdir or f"/tmp/activated_residency_{os.getpid()}")
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    host = os.uname().nodename
    out = Path(args.out or
               f"results/bench_activated_residency_{host}.json")
    rows: list[dict] = []

    def rec(**kw):
        kw["t_wall"] = round(time.time(), 3)
        rows.append(kw)
        print(json.dumps(kw), flush=True)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2))
        except OSError:
            pass   # persisting must never abort the measurement

    rec(rung="env", host=host, potential=pot, potential_bytes=size_b,
        python=sys.executable,
        rss_gb_at_start=round(rss_gb(), 3))

    # ---- r1: fresh instance, COLD page cache -------------------------------
    evict(pot)
    frac = resident_fraction(pot)
    lmp1 = new_instance(str(workdir / "log.r1"))
    el, rss0, rss1 = build_and_time_coeff(lmp1, pot)
    rec(rung="r1_parse_cold", elapsed_s=round(el, 3),
        resident_frac_before=round(frac, 4),
        rss_before_gb=round(rss0, 3), rss_after_gb=round(rss1, 3),
        activated_gb=round(rss1 - rss0, 3),
        expansion_vs_file=round((rss1 - rss0) * 1e9 / size_b, 3))

    # ---- r4: SAME instance, re-issue pair_coeff on the SAME file ------------
    # Does LAMMPS itself memoise the tabulation? If it does, the mechanism is
    # already free and no worker is needed. Expected: it does not.
    frac = resident_fraction(pot)
    rss_a = rss_gb()
    t0 = time.perf_counter()
    lmp1.command(f"pair_coeff * * {pot} W")
    lmp1.command("run 0")
    el_repeat = time.perf_counter() - t0
    rec(rung="r4_repeat_coeff", elapsed_s=round(el_repeat, 3),
        resident_frac_before=round(frac, 4),
        rss_before_gb=round(rss_a, 3), rss_after_gb=round(rss_gb(), 3),
        note="same live instance, pair_coeff re-issued on the same file")

    # ---- r3: SAME instance, NEW configuration, pair_coeff NOT re-issued -----
    # This is the mechanism: a live worker serving a second simulation off the
    # already-activated tables. Perturbing every atom forces a genuinely new
    # force evaluation, so this is real work, not a no-op.
    rss_a = rss_gb()
    t0 = time.perf_counter()
    lmp1.command("displace_atoms all random 0.1 0.1 0.1 12345")
    lmp1.command("run 0")
    el_reuse = time.perf_counter() - t0
    rec(rung="r3_reuse_live", elapsed_s=round(el_reuse, 3),
        rss_before_gb=round(rss_a, 3), rss_after_gb=round(rss_gb(), 3),
        note="new configuration served from already-activated tables")

    # ---- r5: hold a SECOND activated potential alongside the first ----------
    # The budget question: what does it cost to keep two resources at R3?
    if second_pot and os.path.exists(second_pot):
        second = second_pot
        rss_a = rss_gb()
        lmp2 = new_instance(str(workdir / "log.r5"))
        el2, s0, s1 = build_and_time_coeff(lmp2, second)
        rec(rung="r5_hold_second", elapsed_s=round(el2, 3),
            second_potential=second,
            second_bytes=os.path.getsize(second),
            rss_before_gb=round(rss_a, 3), rss_after_gb=round(rss_gb(), 3),
            activated_gb=round(s1 - s0, 3),
            note="second instance live while the first is still held")
        lmp2.close()

    # ---- r2: fresh instance, WARM page cache -------------------------------
    # Done LAST so it cannot warm the cache for the cold rung. This is what
    # invocations #2 and #3 cost today, and it is the number the mechanism
    # replaces.
    lmp1.close()
    frac = resident_fraction(pot)
    lmp3 = new_instance(str(workdir / "log.r2"))
    el_warm, w0, w1 = build_and_time_coeff(lmp3, pot)
    rec(rung="r2_parse_warm", elapsed_s=round(el_warm, 3),
        resident_frac_before=round(frac, 4),
        rss_before_gb=round(w0, 3), rss_after_gb=round(w1, 3),
        activated_gb=round(w1 - w0, 3))
    lmp3.close()

    rec(rung="peak", vm_hwm_gb=round(hwm_gb(), 3))

    # ---- the derived numbers the plan actually needs ------------------------
    by = {r["rung"]: r for r in rows}
    warm = by.get("r2_parse_warm", {}).get("elapsed_s")
    reuse = by.get("r3_reuse_live", {}).get("elapsed_s")
    act = by.get("r1_parse_cold", {}).get("activated_gb")
    summary = {"rung": "SUMMARY"}
    if warm is not None and reuse is not None:
        # The workload does 3 invocations per potential, 2 potentials.
        # A live worker replaces invocations 2 and 3 of each.
        saved = 2 * 2 * (warm - reuse)
        summary.update(
            warm_parse_s=warm, live_reuse_s=reuse,
            saved_per_trial_s=round(saved, 1),
            note="4 of 6 parses replaced (3 invocations x 2 potentials)")
        if act:
            summary["seconds_per_gb_held"] = round(2 * (warm - reuse) / act, 2)
    if act:
        summary["activated_gb"] = act
        summary["expansion_vs_file"] = round(act * 1e9 / size_b, 3)
    rec(**summary)

    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
