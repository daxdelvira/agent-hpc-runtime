#!/usr/bin/env python3
"""
bench_format_activation.py — is the ACTIVATION-vs-I/O regime a property of the
on-disk FORMAT, or of the payload?

WHY THIS EXISTS
---------------
bench_potential_activation.py measured exactly one point: a 3.32 GB ASCII
`setfl` EAM potential loads in 98-100 s under LAMMPS, of which only ~1.9 s is
byte movement.  Separately, safetensors model weights load at a storage rate
(0.737 GB/s) and are therefore I/O-bound.  Two points, two payload types, and a
tempting story ("models are I/O-bound, data is activation-bound") that the data
does not actually support, because payload and format are confounded.

The hypothesis this script tests is narrower and falsifiable:

    The regime is set by the DISTANCE between the on-disk representation and
    the in-memory representation the consumer needs.  safetensors is designed
    for near-zero-copy mmap (distance ~0) -> I/O-bound.  ASCII setfl must be
    lexed into doubles and then expanded into interpolation coefficients
    (distance large) -> activation-bound.

To test it we hold the LOGICAL CONTENT FIXED -- literally the same float64
array, bit for bit, verified by md5 of the array buffer -- and vary only the
serialisation.  Any difference in the cold/warm split is then attributable to
representation, not to payload.

RUNGS, per (format, size), each repeated --repeats times:
  read_cold  : evict + mincore-verify + sequential read(2).  Byte movement only.
  load_cold  : evict + mincore-verify + the format's normal consumer, producing
               a usable in-memory ndarray.
  load_warm  : the same consumer with the cache hot (mincore-verified).
  => io_share   = load_cold - load_warm      (bytes that had to come from disk)
     activation = load_warm                  (work in the consumer process)

WHICH RESOURCE DOES THE ACTIVATION
----------------------------------
Every load runs in a fresh fork+exec'd child, so os.wait4() gives that child's
own rusage and nothing else.  The child additionally brackets the loader call
with getrusage(RUSAGE_SELF), which excludes interpreter startup and import cost
from the reported numbers.  We report:
  cpu_per_wall = (utime+stime)/wall   ~1.0 => single-threaded; >1 => it scales
  rss_delta_bytes                      peak RSS attributable to the load
  expansion = rss_delta / file_bytes   does the in-memory form exceed the file?

CAVEATS THIS SCRIPT REFUSES TO HIDE
-----------------------------------
* Cold caches use posix_fadvise(DONTNEED).  On Lustre this was measured leaving
  56.2% of a file resident.  Every rung records the mincore residency it
  actually achieved BEFORE the timed section; a rung whose residency is not
  ~0.0 is not cold and must not be reported as cold.  Run --workdir on
  node-local storage.
* Fixing the logical content means the files are NOT the same size (19 B/value
  ASCII vs 8 B/value float64).  Both bytes/s and values/s are reported; use
  values/s when comparing formats, bytes/s when comparing to a storage rate.
* The generated files and the anchor potential live on different filesystems.
  --anchor-copy re-measures a synthetic ASCII file on the anchor's filesystem
  so the storage rate can be divided out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The mincore/fadvise helpers were debugged on 2026-08-03 and are correct;
# import rather than re-implement so the two benches cannot drift apart.
from bench_potential_activation import (  # noqa: E402
    LAMMPS_IN,
    evict,
    resident_fraction,
    timed_read,
)

# "%.12e" on a value in [1, 100) is always 18 characters (d.dddddddddddde+0d),
# so +newline = 19 bytes/line -- the exact bytes/line of w_eam4_big.fs
# (3320490868 B / 174 762 677 lines = 19.0).
ASCII_FMT = "%.12e\n"
ASCII_BYTES_PER_VALUE = 19

FORMATS = ("ascii_loadtxt", "ascii_pandas", "npy", "raw_f32", "hdf5", "npz_deflate")


# --------------------------------------------------------------------------- #
# loaders: each returns the usable in-memory float64 ndarray
# --------------------------------------------------------------------------- #
def _load_ascii_loadtxt(path):
    import numpy as np
    return np.loadtxt(path, dtype=np.float64)


def _load_ascii_pandas(path):
    import pandas as pd
    return pd.read_csv(path, header=None, dtype="float64",
                       engine="c").values.ravel()


def _load_npy(path):
    import numpy as np
    return np.load(path)


def _load_raw_f32(path):
    import numpy as np
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def _load_hdf5(path):
    import h5py
    with h5py.File(path, "r") as f:
        return f["a"][:]


def _load_npz(path):
    import numpy as np
    with np.load(path) as z:
        return z["a"]


LOADERS = {
    "ascii_loadtxt": _load_ascii_loadtxt,
    "ascii_pandas": _load_ascii_pandas,
    "npy": _load_npy,
    "raw_f32": _load_raw_f32,
    "hdf5": _load_hdf5,
    "npz_deflate": _load_npz,
}


# --------------------------------------------------------------------------- #
# child mode: run one loader, report its own wall/cpu/rss
# --------------------------------------------------------------------------- #
def child_main(fmt: str, path: str, outfile: str) -> int:
    rec = {"fmt": fmt, "path": path}
    if fmt == "noop":
        # baseline: interpreter + numpy import, no file touched
        import numpy  # noqa: F401
        r = resource.getrusage(resource.RUSAGE_SELF)
        rec.update(wall_s=0.0, utime_s=r.ru_utime, stime_s=r.ru_stime,
                   rss_before_kb=r.ru_maxrss, rss_after_kb=r.ru_maxrss,
                   nbytes=0, n=0, md5="")
        Path(outfile).write_text(json.dumps(rec))
        return 0

    loader = LOADERS[fmt]
    # Warm the import machinery BEFORE the bracket so we time the load, not
    # the first `import pandas`.  The file itself is untouched by this.
    if fmt == "ascii_pandas":
        import pandas  # noqa: F401
    elif fmt == "hdf5":
        import h5py  # noqa: F401
    else:
        import numpy  # noqa: F401

    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    arr = loader(path)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)

    rec.update(
        wall_s=wall,
        utime_s=r1.ru_utime - r0.ru_utime,
        stime_s=r1.ru_stime - r0.ru_stime,
        rss_before_kb=r0.ru_maxrss,
        rss_after_kb=r1.ru_maxrss,
        minflt=r1.ru_minflt - r0.ru_minflt,
        majflt=r1.ru_majflt - r0.ru_majflt,
        nbytes=int(arr.nbytes),
        n=int(arr.size),
        dtype=str(arr.dtype),
        md5=hashlib.md5(memoryview(arr).cast("B")).hexdigest(),
    )
    Path(outfile).write_text(json.dumps(rec))
    return 0


def run_child(fmt: str, path: str, tmp: Path) -> dict:
    """fork+exec a fresh interpreter, return its inner metrics plus wait4 rusage.

    exec (not bare fork) so no BLAS/OpenMP thread state is inherited and
    ru_maxrss belongs solely to this load.
    """
    outfile = tmp / f"_child_{os.getpid()}.json"
    if outfile.exists():
        outfile.unlink()
    argv = [sys.executable, os.path.abspath(__file__), "--child", fmt,
            "--child-path", path, "--child-out", str(outfile)]
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        try:
            os.execv(sys.executable, argv)
        finally:
            os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    proc_wall = time.perf_counter() - t0
    if status != 0:
        return {"error": f"child exit status {status}", "proc_wall_s": proc_wall}
    rec = json.loads(outfile.read_text())
    outfile.unlink()
    rec["proc_wall_s"] = proc_wall
    rec["child_utime_s"] = ru.ru_utime
    rec["child_stime_s"] = ru.ru_stime
    rec["child_maxrss_kb"] = ru.ru_maxrss
    return rec


# --------------------------------------------------------------------------- #
# generation: one float64 array, six serialisations, bit-identical content
# --------------------------------------------------------------------------- #
def generate(workdir: Path, n: int, seed: int, formats) -> dict:
    import numpy as np

    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}"
    ascii_path = workdir / f"{tag}.txt"

    rng = np.random.default_rng(seed)
    chunk = 1 << 20
    with open(ascii_path, "w") as fh:
        written = 0
        while written < n:
            k = min(chunk, n - written)
            # uniform in [1, 100): guarantees a 2-digit exponent and no sign,
            # hence exactly 19 bytes per line.
            vals = rng.random(k) * 99.0 + 1.0
            fh.write((ASCII_FMT * k) % tuple(vals))
            written += k
    sz = ascii_path.stat().st_size
    assert sz == n * ASCII_BYTES_PER_VALUE, (sz, n * ASCII_BYTES_PER_VALUE)

    # Round-trip through the ASCII text FIRST, so every other format stores
    # exactly the bits that the ASCII path will produce.  Otherwise the ASCII
    # rung would be decoding slightly different numbers and "same logical
    # content" would be a claim rather than a checked invariant.
    import pandas as pd
    a = pd.read_csv(ascii_path, header=None, dtype="float64",
                    engine="c").values.ravel()
    assert a.size == n

    paths = {"ascii_loadtxt": ascii_path, "ascii_pandas": ascii_path}
    if "npy" in formats:
        p = workdir / f"{tag}.npy"
        np.save(p, a)
        paths["npy"] = p
    if "raw_f32" in formats:
        p = workdir / f"{tag}.f32"
        a.astype(np.float32).tofile(p)
        paths["raw_f32"] = p
    if "hdf5" in formats:
        import h5py
        p = workdir / f"{tag}.h5"
        with h5py.File(p, "w") as f:
            # chunks=None + no filters => contiguous, uncompressed layout
            f.create_dataset("a", data=a, chunks=None, compression=None)
        paths["hdf5"] = p
    if "npz_deflate" in formats:
        p = workdir / f"{tag}.npz"
        np.savez_compressed(p, a=a)
        paths["npz_deflate"] = p

    ref_md5 = hashlib.md5(memoryview(a).cast("B")).hexdigest()
    del a
    return {"paths": {k: str(v) for k, v in paths.items()},
            "ref_md5": ref_md5, "n": n}


# --------------------------------------------------------------------------- #
def measure(fmt: str, path: str, tmp: Path) -> dict:
    size = os.path.getsize(path)

    evict(path)
    frac_r = resident_fraction(path)
    rs = timed_read(path)

    evict(path)
    frac_c = resident_fraction(path)
    cold = run_child(fmt, path, tmp)

    frac_w = resident_fraction(path)
    warm = run_child(fmt, path, tmp)

    row = {
        "fmt": fmt,
        "path": path,
        "file_bytes": size,
        "read_cold_s": rs,
        "read_cold_resident_before": frac_r,
        "read_mb_per_s": size / 1e6 / rs if rs > 0 else float("nan"),
        "load_cold_s": cold.get("wall_s"),
        "load_cold_resident_before": frac_c,
        "load_warm_s": warm.get("wall_s"),
        "load_warm_resident_before": frac_w,
        "warm_utime_s": warm.get("utime_s"),
        "warm_stime_s": warm.get("stime_s"),
        "warm_rss_delta_kb": (warm.get("rss_after_kb", 0)
                              - warm.get("rss_before_kb", 0)),
        "warm_maxrss_kb": warm.get("child_maxrss_kb"),
        "warm_majflt": warm.get("majflt"),
        "array_nbytes": warm.get("nbytes"),
        "n_values": warm.get("n"),
        "md5": warm.get("md5"),
        "error": cold.get("error") or warm.get("error"),
    }
    lc, lw = row["load_cold_s"], row["load_warm_s"]
    if lc and lw is not None:
        # Keep the UNCLAMPED difference as well.  When warm > cold the I/O
        # share is at or below the run-to-run noise floor; clamping to 0 and
        # reporting activation_pct > 100 without saying so is how a noisy
        # rung gets read as a precise one.
        row["io_share_s_raw"] = lc - lw
        row["io_share_s"] = max(lc - lw, 0.0)
        row["io_share_pct"] = 100.0 * max(lc - lw, 0.0) / lc
        row["activation_pct"] = 100.0 * lw / lc
        row["warm_exceeds_cold"] = lw > lc
    cw = (row["warm_utime_s"] or 0) + (row["warm_stime_s"] or 0)
    row["warm_cpu_per_wall"] = cw / lw if lw else float("nan")
    return row


# --------------------------------------------------------------------------- #
LAMMPS_FLOOR_IN = """\
units metal
boundary p p p
atom_style atomic
lattice bcc 3.165
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 183.84
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
run 0
"""


def _lammps_child(lmp: str, workdir: Path) -> dict:
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        try:
            os.chdir(str(workdir))
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.execv(lmp, [lmp, "-in", "in.bench"])
        finally:
            os._exit(127)
    _, status, ru = os.wait4(pid, 0)
    return {"wall_s": time.perf_counter() - t0, "status": status,
            "utime_s": ru.ru_utime, "stime_s": ru.ru_stime,
            "maxrss_kb": ru.ru_maxrss, "majflt": ru.ru_majflt}


def run_lammps_floor(lmp: str, workdir: Path) -> dict:
    """LAMMPS startup with a trivial analytic pair style: the fixed cost that
    is NOT potential loading.  Without it, small-potential rungs are all
    interpreter/MPI startup and the size sweep says nothing."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.bench").write_text(LAMMPS_FLOOR_IN)
    r = _lammps_child(lmp, workdir)
    return {"fmt": "lammps_startup_floor", "path": "lj/cut", "file_bytes": 0,
            "read_cold_s": 0.0, "read_cold_resident_before": float("nan"),
            "read_mb_per_s": float("nan"),
            "load_cold_s": r["wall_s"], "load_cold_resident_before": float("nan"),
            "load_warm_s": r["wall_s"], "load_warm_resident_before": float("nan"),
            "warm_utime_s": r["utime_s"], "warm_stime_s": r["stime_s"],
            "warm_maxrss_kb": r["maxrss_kb"], "warm_rss_delta_kb": r["maxrss_kb"],
            "warm_majflt": r["majflt"], "n_values": 0,
            "warm_cpu_per_wall": (r["utime_s"] + r["stime_s"]) / r["wall_s"],
            "error": None if r["status"] == 0 else f"status {r['status']}"}


def run_lammps_anchor(lmp: str, potential: str, workdir: Path,
                      element: str = "W") -> dict:
    """LAMMPS rungs, but with the child's rusage captured.

    The '~9.8 GB of splines' figure in the plan is arithmetic (3 tables x Nrho
    points x 7 spline coefficients x 8 B), never a measurement.  ru_maxrss of
    the LAMMPS process measures it.
    """
    style = "eam/fs" if potential.endswith(".fs") else "eam/alloy"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.bench").write_text(
        LAMMPS_IN.format(style=style, potential=potential).replace(
            " W\n", f" {element}\n"))

    def one() -> dict:
        return _lammps_child(lmp, workdir)

    size = os.path.getsize(potential)
    evict(potential)
    fr = resident_fraction(potential)
    rs = timed_read(potential)

    evict(potential)
    fc = resident_fraction(potential)
    cold = one()

    fw = resident_fraction(potential)
    warm = one()

    row = {
        "fmt": "lammps_setfl", "path": potential, "file_bytes": size,
        "read_cold_s": rs, "read_cold_resident_before": fr,
        "read_mb_per_s": size / 1e6 / rs,
        "load_cold_s": cold["wall_s"], "load_cold_resident_before": fc,
        "load_warm_s": warm["wall_s"], "load_warm_resident_before": fw,
        "warm_utime_s": warm["utime_s"], "warm_stime_s": warm["stime_s"],
        "warm_maxrss_kb": warm["maxrss_kb"], "warm_rss_delta_kb": warm["maxrss_kb"],
        "warm_majflt": warm["majflt"],
        "error": None if cold["status"] == 0 == warm["status"] else
                 f"lammps status cold={cold['status']} warm={warm['status']}",
    }
    lc, lw = row["load_cold_s"], row["load_warm_s"]
    row["io_share_s"] = max(lc - lw, 0.0)
    row["io_share_pct"] = 100.0 * max(lc - lw, 0.0) / lc
    row["activation_pct"] = 100.0 * lw / lc
    row["warm_cpu_per_wall"] = (warm["utime_s"] + warm["stime_s"]) / lw
    return row


# --------------------------------------------------------------------------- #
CSV_COLS = ["fmt", "n_values", "file_bytes", "rep", "read_cold_s",
            "read_mb_per_s", "load_cold_s", "load_warm_s", "io_share_s",
            "io_share_pct", "activation_pct", "warm_cpu_per_wall",
            "warm_utime_s", "warm_stime_s", "warm_maxrss_kb",
            "warm_rss_delta_kb", "array_nbytes", "read_cold_resident_before",
            "load_cold_resident_before", "load_warm_resident_before",
            "md5", "error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--workdir", default="/tmp/fmtbench",
                    help="node-local scratch for generated files")
    ap.add_argument("--ascii-gb", default="0.5,1,2",
                    help="target ASCII file sizes in GB (10^9 B)")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--anchor-potential", default=None,
                    help="comma list of PATH:ELEMENT, smallest first")
    ap.add_argument("--anchor-copy-dir", default=None,
                    help="also place one ASCII file here (e.g. the anchor's "
                         "filesystem) to separate storage rate from format")
    ap.add_argument("--lmp", default=os.environ.get("LMP_BIN", "lmp"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if args.child:
        return child_main(args.child, args.child_path, args.child_out)

    host = os.uname().nodename
    workdir = Path(args.workdir)
    tmp = workdir / "_ipc"
    tmp.mkdir(parents=True, exist_ok=True)
    formats = [f for f in args.formats.split(",") if f]
    sizes_gb = [float(x) for x in args.ascii_gb.split(",") if x]

    stem = args.out or f"results/bench_format_activation_{host}"
    outj, outc = Path(stem + ".json"), Path(stem + ".csv")
    outj.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "host": host, "started": time.time(), "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "workdir": str(workdir),
        "workdir_fs": _fstype(str(workdir)),
        "argv": sys.argv,
        "page_size": os.sysconf("SC_PAGE_SIZE"),
    }
    rows: list[dict] = []

    def flush2():
        outj.write_text(json.dumps({"env": env, "rows": rows}, indent=2))
        with open(outc, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"[fmtbench] host={host} workdir={workdir} ({env['workdir_fs']}) "
          f"cpus={env['affinity_cpus']} formats={formats} sizes={sizes_gb}GB "
          f"repeats={args.repeats}", flush=True)

    # baseline child cost so RSS deltas and wall times can be judged
    base = run_child("noop", "", tmp)
    env["noop_child_wall_s"] = base.get("proc_wall_s")
    env["noop_child_maxrss_kb"] = base.get("child_maxrss_kb")
    print(f"[fmtbench] bare child (python+numpy): "
          f"{env['noop_child_wall_s']:.2f} s, "
          f"maxrss {env['noop_child_maxrss_kb']/1024:.0f} MB", flush=True)

    for gb in sizes_gb:
        n = int(round(gb * 1e9 / ASCII_BYTES_PER_VALUE))
        print(f"\n[fmtbench] generating n={n} values "
              f"(ascii {n*ASCII_BYTES_PER_VALUE/1e9:.2f} GB) ...", flush=True)
        t0 = time.perf_counter()
        gen = generate(workdir, n, args.seed, formats)
        print(f"[fmtbench]   generated in {time.perf_counter()-t0:.1f} s "
              f"ref_md5={gen['ref_md5'][:12]}", flush=True)

        for fmt in formats:
            for rep in range(args.repeats):
                row = measure(fmt, gen["paths"][fmt], tmp)
                row["rep"] = rep
                row["n_values"] = n
                row["ref_md5"] = gen["ref_md5"]
                row["content_matches_ref"] = (row.get("md5") == gen["ref_md5"])
                rows.append(row)
                flush2()
                print(f"  {fmt:<14} rep{rep} n={n} "
                      f"read {row['read_cold_s']:7.2f}s "
                      f"({row['read_mb_per_s']:6.0f} MB/s)  "
                      f"cold {row['load_cold_s']:7.2f}s  "
                      f"warm {row['load_warm_s']:7.2f}s  "
                      f"act {row.get('activation_pct', float('nan')):5.1f}%  "
                      f"cpu/wall {row['warm_cpu_per_wall']:4.2f}  "
                      f"resid(cold) {row['load_cold_resident_before']:.3f}"
                      + ("  MD5-MISMATCH" if not row["content_matches_ref"]
                         and fmt != "raw_f32" else ""), flush=True)

        if not args.keep:
            for p in set(gen["paths"].values()):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # optional: same ASCII format on a different filesystem, to divide out
    # the storage rate when comparing against the anchor.
    if args.anchor_copy_dir:
        d = Path(args.anchor_copy_dir)
        d.mkdir(parents=True, exist_ok=True)
        gbs = sizes_gb[0]
        n = int(round(gbs * 1e9 / ASCII_BYTES_PER_VALUE))
        print(f"\n[fmtbench] cross-filesystem control in {d} "
              f"({_fstype(str(d))})", flush=True)
        gen = generate(d, n, args.seed, ["ascii_loadtxt"])
        for rep in range(args.repeats):
            row = measure("ascii_loadtxt", gen["paths"]["ascii_loadtxt"], tmp)
            row.update(rep=rep, n_values=n, fmt="ascii_loadtxt@otherfs",
                       ref_md5=gen["ref_md5"],
                       content_matches_ref=row.get("md5") == gen["ref_md5"])
            rows.append(row)
            flush2()
            print(f"  ascii@otherfs  rep{rep} read {row['read_cold_s']:.2f}s "
                  f"({row['read_mb_per_s']:.0f} MB/s) cold {row['load_cold_s']:.2f}s "
                  f"warm {row['load_warm_s']:.2f}s "
                  f"resid {row['load_cold_resident_before']:.3f}", flush=True)
        for p in set(gen["paths"].values()):
            try:
                os.unlink(p)
            except OSError:
                pass

    if args.anchor_potential:
        for rep in range(args.repeats):
            row = run_lammps_floor(args.lmp, workdir / "_lammps")
            row["rep"] = rep
            rows.append(row)
            flush2()
            print(f"\n  lammps floor (lj/cut) rep{rep} "
                  f"{row['load_warm_s']:.2f}s "
                  f"maxrss {row['warm_maxrss_kb']/1048576:.3f} GB "
                  f"err={row['error']}", flush=True)

        for spec in args.anchor_potential.split(","):
            pot, _, elem = spec.partition(":")
            pot = os.path.abspath(pot)
            elem = elem or "W"
            print(f"\n[fmtbench] LAMMPS anchor: {pot} ({elem}, "
                  f"{os.path.getsize(pot)/1e9:.3f} GB, "
                  f"{_fstype(pot)})", flush=True)
            for rep in range(args.repeats):
                row = run_lammps_anchor(args.lmp, pot, workdir / "_lammps",
                                        elem)
                row["rep"] = rep
                row["fmt"] = f"lammps_setfl:{os.path.basename(pot)}"
                row["n_values"] = row["file_bytes"] // ASCII_BYTES_PER_VALUE
                rows.append(row)
                flush2()
                print(f"  {row['fmt']:<28} rep{rep} "
                      f"read {row['read_cold_s']:.2f}s "
                      f"cold {row['load_cold_s']:.2f}s "
                      f"warm {row['load_warm_s']:.2f}s "
                      f"act {row['activation_pct']:.1f}% "
                      f"cpu/wall {row['warm_cpu_per_wall']:.2f} "
                      f"maxrss {row['warm_maxrss_kb']/1048576:.3f} GB "
                      f"resid(cold) {row['load_cold_resident_before']:.3f} "
                      f"err={row['error']}", flush=True)

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(workdir / "_lammps", ignore_errors=True)

    print_table(rows)
    print(f"\nwrote {outj}\nwrote {outc}")
    return 0


def _fstype(path: str) -> str:
    try:
        return subprocess.run(["stat", "-f", "-c", "%T", path],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def print_table(rows) -> None:
    print("\n" + "=" * 118)
    print("FORMAT x SIZE -> I/O vs ACTIVATION   (every value measured; two "
          "reps shown as a/b, never averaged away)")
    print("=" * 118)
    hdr = (f"{'format':<16}{'Nvalues':>12}{'file GB':>9}{'read s':>9}"
           f"{'MB/s':>8}{'cold s':>9}{'warm s':>9}{'io %':>7}{'act %':>7}"
           f"{'cpu/wall':>9}{'maxRSS GB':>10}{'resid':>7}")
    print(hdr)
    print("-" * 118)
    seen = []
    for r in rows:
        key = (r["fmt"], r.get("n_values"))
        if key in seen:
            continue
        seen.append(key)
        grp = [x for x in rows
               if (x["fmt"], x.get("n_values")) == key]

        def col(k, f="{:.2f}"):
            return "/".join(f.format(x[k]) if x.get(k) is not None else "-"
                            for x in grp)
        print(f"{r['fmt']:<16}{r.get('n_values',0):>12}"
              f"{r['file_bytes']/1e9:>9.2f}"
              f"{col('read_cold_s'):>9}{col('read_mb_per_s','{:.0f}'):>8}"
              f"{col('load_cold_s'):>9}{col('load_warm_s'):>9}"
              f"{col('io_share_pct','{:.1f}'):>7}"
              f"{col('activation_pct','{:.1f}'):>7}"
              f"{col('warm_cpu_per_wall'):>9}"
              f"{'/'.join('%.2f' % (x['warm_maxrss_kb']/1048576) for x in grp if x.get('warm_maxrss_kb')):>10}"
              f"{col('load_cold_resident_before','{:.2f}'):>7}")


if __name__ == "__main__":
    raise SystemExit(main())
