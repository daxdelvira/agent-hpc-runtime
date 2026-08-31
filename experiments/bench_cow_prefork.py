#!/usr/bin/env python3
"""Does copy-on-write actually hold for an activated LAMMPS potential?

    $ATOMS_PYTHON experiments/bench_cow_prefork.py --potential <path> [--children 3]

THE DECISION THIS SETTLES. Two ways to keep a parsed artifact resident at R3:

  option 4  one long-lived worker holds ONE live instance; each tool call runs
            its script against it. Needs a command-rewriting layer to undo
            whatever the previous script left behind (this already exists as
            execution/persistent_lammps.py:RetainedInstance, and it already
            produced one real bug -- create_box rewritten into delete_atoms).

  option 6  a TEMPLATE process parses once, then fork()s a child per tool call.
            The child inherits the parsed tables through copy-on-write, at
            page-table cost rather than parse cost, and dies afterwards -- so
            it never inherits a previous script's residue, and a segfaulting
            child cannot take the tables with it.

Option 6 is strictly better IF copy-on-write genuinely holds on a ~17 GB C++
heap. That is an empirical claim about the kernel and about whether LAMMPS
writes to those pages when serving a new configuration. This measures it.

WHAT WOULD MAKE THIS PROBE A LIE. A child that never touches the tables would
show beautiful COW numbers and prove nothing -- the same failure mode as the
L2-sleep result and the Lustre fadvise no-op. Two guards:

  * every child does a GENUINE force evaluation on a perturbed configuration
    (displace_atoms + run 0) and reports the potential energy, which is
    compared against the template's;
  * arm `write_control` deliberately dirties a large buffer inherited from the
    parent. If Private_Dirty does not rise there, the MEASUREMENT is broken and
    no other row in the output means anything.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time


# ---------------------------------------------------------------- memory ---
def smaps() -> dict:
    """Per-process page accounting. Private_* is what this process ALONE holds.

    RSS double-counts pages shared with the parent, which is exactly the
    mistake that would make a failed COW look like a successful one. Pss
    (proportional set size) and Private_Dirty are the honest columns.
    """
    out = {}
    try:
        for line in open("/proc/self/smaps_rollup"):
            k, _, v = line.partition(":")
            if k in ("Rss", "Pss", "Shared_Clean", "Shared_Dirty",
                     "Private_Clean", "Private_Dirty"):
                out[k] = int(v.split()[0]) / 1e6      # kB -> GB
    except FileNotFoundError:
        pass
    return out


def minflt() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def nthreads() -> int:
    """Pattern-D hazard check: fork() clones only the calling thread, so a lock
    held by another thread at fork time is inherited locked and never released."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return -1


# ---------------------------------------------------------------- lammps ---
_SETUP = [                      # matches experiments/bench_activated_residency.py
    "units metal", "boundary p p p", "atom_style atomic",
    "lattice bcc 3.165", "region box block 0 2 0 2 0 2",
    "create_box 1 box", "create_atoms 1 box", "mass 1 183.84",
]


def _style_for(p: str) -> str:
    return "eam/fs" if p.endswith(".fs") else "eam/alloy"


def build_template(potential: str, logfile: str):
    """Parse the potential into a live instance. Returns (lmp, parse_s, gb)."""
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", logfile, "-screen", "none"])
    for c in _SETUP:
        lmp.command(c)
    lmp.command(f"pair_style {_style_for(potential)}")
    before = smaps().get("Rss", 0.0)
    t0 = time.perf_counter()
    lmp.command(f"pair_coeff * * {potential} W")
    lmp.command("run 0")                     # forces interpolate(), not just the read
    return lmp, time.perf_counter() - t0, smaps().get("Rss", 0.0) - before


def force_eval(lmp) -> float:
    """A genuinely new force evaluation off already-activated tables."""
    lmp.command("displace_atoms all random 0.1 0.1 0.1 12345")
    lmp.command("run 0")
    return lmp.get_thermo("pe")


# ------------------------------------------------------------------ arms ---
def run_child(lmp, arm: str, ballast) -> dict:
    """Body of one forked child. Returns a record through a pipe; never returns
    to the caller -- the child always _exit()s so no atexit handler double-runs."""
    t_start = time.perf_counter()
    mf0 = minflt()
    rec = {"arm": arm}

    if arm == "fork_noop":
        # Child does NOTHING. Isolates the fixed cost of fork + interpreter from
        # anything the consumer library does, so the table's contribution to
        # Private_Dirty in the other arms can be read off by subtraction.
        pass
    elif arm == "fork_read_only":
        # Reads a thermo value already computed by the template. Touches the
        # tables for reading but issues no `run`, which is the command suspected
        # of rewriting the spline coefficients.
        rec["pe"] = lmp.get_thermo("pe")
    elif arm == "write_control":
        # NEGATIVE CONTROL. Dirty every page of a buffer inherited from the
        # parent. Private_Dirty MUST rise by roughly the buffer size; if it
        # does not, the measurement below is not measuring anything.
        ballast += 1                          # in-place write over the whole array
        rec["control_checksum"] = float(ballast[0])
    else:
        rec["pe"] = force_eval(lmp)

    rec["work_s"] = round(time.perf_counter() - t_start, 3)
    rec["minflt"] = minflt() - mf0
    rec.update({k: round(v, 3) for k, v in smaps().items()})
    return rec


def fork_arm(lmp, arm: str, ballast) -> dict:
    r_fd, w_fd = os.pipe()
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:                              # ---- child ----
        os.close(r_fd)
        try:
            rec = run_child(lmp, arm, ballast)
        except Exception as e:                # a failed child must not look like a pass
            rec = {"arm": arm, "error": f"{type(e).__name__}: {e}"}
        os.write(w_fd, json.dumps(rec).encode())
        os.close(w_fd)
        os._exit(0)
    os.close(w_fd)                            # ---- parent ----
    t_fork = time.perf_counter() - t0
    buf = b""
    while chunk := os.read(r_fd, 65536):
        buf += chunk
    os.close(r_fd)
    os.waitpid(pid, 0)
    rec = json.loads(buf) if buf else {"arm": arm, "error": "child produced no output"}
    rec["fork_s"] = round(t_fork, 4)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", required=True)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--ballast-gb", type=float, default=2.0,
                    help="size of the negative-control buffer")
    ap.add_argument("--out", default="results/bench_cow_prefork.json")
    ap.add_argument("--no-thp", action="store_true",
                    help="prctl(PR_SET_THP_DISABLE) before allocating. Transparent "
                         "huge pages are [always] on some nodes and [madvise] on "
                         "others; with 2 MB pages a SPARSE write dirties 2 MB, which "
                         "would inflate the measured dirty fraction. If f is "
                         "unchanged with this flag, the writes are dense and the "
                         "result does not depend on the node's THP setting.")
    a = ap.parse_args()

    import ctypes
    if a.no_thp:
        PR_SET_THP_DISABLE = 41
        rc = ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
        if rc != 0:
            print("  WARNING: prctl(PR_SET_THP_DISABLE) failed; THP still active")
            a.no_thp = False
        else:
            print("  THP disabled for this process tree")

    import numpy as np
    host = os.uname().nodename
    records = []

    print(f"host {host}   potential {a.potential}")
    print(f"template: parsing (this is the cost we are trying to avoid paying again)")
    lmp, parse_s, activated_gb = build_template(a.potential, "/tmp/cow_template.log")
    pe_template = force_eval(lmp)
    base = smaps()
    print(f"  parse {parse_s:.2f} s   activated {activated_gb:.2f} GB   pe {pe_template:.6f}")
    print(f"  template Rss {base.get('Rss',0):.2f} GB  Pss {base.get('Pss',0):.2f} GB")

    # Negative-control ballast, allocated and touched BEFORE forking so it is
    # genuinely inherited rather than faulted in fresh by the child.
    ballast = np.ones(int(a.ballast_gb * 1e9 / 8), dtype=np.float64)
    print(f"  ballast {ballast.nbytes/1e9:.2f} GB allocated for the write control")

    nt = nthreads()
    print(f"  threads at fork time: {nt}"
          f"{'   <-- WARNING: fork() clones only the calling thread' if nt > 1 else '   (safe)'}")

    # Python object headers are refcounted, and a refcount bump is a WRITE that
    # dirties a page. freeze() moves existing objects to a permanent generation
    # the collector will not touch, which is the standard prefork mitigation.
    gc.freeze()
    gc.collect()

    for arm in ("fork_noop", "fork_read_only"):
        r = fork_arm(lmp, arm, ballast)
        records.append(r)
        print(f"  {arm:16s}: private_dirty {r.get('Private_Dirty')} GB  "
              f"minflt {r.get('minflt')}  work {r.get('work_s')} s")

    for i in range(a.children):
        rec = fork_arm(lmp, f"prefork_child_{i}", ballast)
        rec["pe_matches_template"] = (
            abs(rec.get("pe", float("nan")) - pe_template) < 1e-9
            if "pe" in rec else None)
        records.append(rec)
        print(f"  child {i}: fork {rec['fork_s']*1000:.1f} ms  work {rec.get('work_s')} s  "
              f"private_dirty {rec.get('Private_Dirty')} GB  "
              f"shared_clean {rec.get('Shared_Clean')} GB  minflt {rec.get('minflt')}")

    ctl = fork_arm(lmp, "write_control", ballast)
    records.append(ctl)
    print(f"  write_control: private_dirty {ctl.get('Private_Dirty')} GB "
          f"(must be >= ~{a.ballast_gb:.1f} GB or this probe is blind)")

    out = {
        "host": host, "potential": a.potential,
        "parse_s": round(parse_s, 3), "activated_gb": round(activated_gb, 3),
        "pe_template": pe_template,
        "template_smaps": {k: round(v, 3) for k, v in base.items()},
        "threads_at_fork": nt, "ballast_gb": a.ballast_gb, "thp_disabled": a.no_thp,
        "records": records,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
