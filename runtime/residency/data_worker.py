#!/usr/bin/env python3
"""data_worker.py — T4b: the resident, evictable data worker (Pattern B).

This is the piece of Tandem that has no 2026 paper: a resource held at
R3_ACTIVATED — a *parsed structure inside a live consumer process* — that the
arbitrator can charge for and genuinely take back.

WHY A SEPARATE PROCESS ("option 4") AND NOT A FORK ("option 6")
---------------------------------------------------------------
Measured, not assumed. `experiments/bench_cow_prefork.py`, written up in
`sc-workshop-paper/cow_prefork_result_20260829.md`: LAMMPS `run` calls
`Pair::init()` -> `init_one()`, which re-runs the EAM spline interpolation and
rewrites every coefficient array IN PLACE. 92% of the table is dirtied, so
copy-on-write buys nothing and a prefork child costs 2x the table. A resident
worker re-interpolates into the SAME memory, so its footprint stays 1x. That
asymmetry — interpolation is the cheap half in TIME and the whole table in
WRITES — is the measured reason option 4 wins.

SHAPE
-----
    tool subprocess                    this worker process
    (LLM-generated script)             (long-lived, one per resource)
    ------------------------           --------------------------------
    import lammps   <-- the shim       holds ONE live lammps instance
        |                              with the potential already parsed
        | AF_UNIX, line-delimited JSON
        +----------------------------> command / file / get_thermo / extract_*

    LammpsDataWorker (in the runtime process) implements ResidencyActor:
    it spawns/kills worker processes and measures them from the outside.

CRASH ISOLATION IS A FEATURE. The generated script still runs in its own
subprocess. A segfault there closes the socket; the worker sees EOF, drops the
connection, and keeps the activated structure. Only a `shutdown` op or a signal
from the actor destroys it.

THE TWO MEASUREMENT TRAPS THIS FILE IS BUILT AROUND
---------------------------------------------------
1. `measure_held_gb()` reads **Private_Dirty** (Pss as a cross-check) from
   `/proc/<pid>/smaps_rollup`, never Rss. Rss double-counts shared pages; in
   the COW benchmark Rss would have made a failed copy-on-write look like a
   success and Private_Dirty did not (invariant I1).
2. Pattern D — thread pools. The COW benchmark measured **193 threads** alive
   by default from OpenMP/MKL pools, dropping to 3 with OMP_NUM_THREADS=1.
   `_THREAD_PINS` is applied to the child environment BEFORE the worker's
   interpreter starts, i.e. before the consumer library is imported, and the
   worker asserts its own thread count at startup and reports it in `ping`.

INVARIANT I2 IS THE POINT OF THE `release()` PATH. A worker that retains but
cannot truly release makes the budget a fiction and every downstream percentage
meaningless. `release()` therefore does not *assume* teardown frees memory: it
measures Private_Dirty immediately before teardown and waits for `/proc/<pid>`
to disappear. Everything is recorded in `last_release_detail` so a caller can
see the measurement rather than a claim. See
`experiments/bench_residency_data_worker.py` for the end-to-end proof: release,
then show the next call pays a cold cost.

AND THIS IS A TEARDOWN ACTOR, WHICH THE LEDGER CANNOT VERIFY ON ITS OWN.
Because release() kills the process, measure_held_gb() afterwards is 0.0 BY
CONSTRUCTION — so the ledger's before/after drop is tautologically the whole
charge and confirms nothing. The ledger must not sniff for "is this a teardown
actor"; that is the class-specific knowledge I4 keeps out of it. So
`release_witness()` (contract, added 2026-08-30) declares an independent
account instead: the job cgroup's `anon` total sampled either side of the
release, which does not depend on the released process still existing. `anon`
and not `memory.current` because the latter counts page cache, and a worker
that had just read a 3.32 GB potential would otherwise get the cache drop
credited to its release. The cached witness is stamped with a monotonic release
counter, so a stale one cannot come back looking like a fresh observation.

NOT IN SCOPE: the ledger and the arbitrator (T1/T2), the horizon estimator
(T3), and Pattern A. `contract.py` signatures are implemented exactly as
written; nothing in this file changes them.
"""
from __future__ import annotations

import errno
import json
import os

import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

if __package__ in (None, ""):                       # `python data_worker.py`
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.residency.contract import (            # noqa: E402
    ReleaseNotHonoured,
    ResourceClass,
    ResourceSpec,
    Rung,
)

# The directory the client shim lives in. Prepending this to a subprocess's
# PYTHONPATH is the ENTIRE integration surface on the framework side.
SHIM_DIR = _HERE / "lammps_shim"

SOCKET_ENV = "TANDEM_LAMMPS_SOCKET"
STRICT_ENV = "TANDEM_LAMMPS_STRICT"      # 1 => never silently fall back
REWRITE_ENV = "TANDEM_LAMMPS_REWRITE"    # 1 => stream+rewrite scripts (see below)

# Pattern D. Applied to the worker's environment before its interpreter starts,
# so every one of these is in place before `import lammps` pulls in OpenMP/MKL.
_THREAD_PINS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_PROC_BIND": "false",
    "KMP_INIT_AT_FORK": "FALSE",
}

# 193 -> 3 was the measured effect of the pins. 8 leaves headroom for a libc
# helper thread without letting an unpinned pool through unnoticed.
DEFAULT_MAX_THREADS = 8


# ===========================================================================
# Measurement primitives (invariant I1)
# ===========================================================================

def read_smaps_rollup(pid: int) -> dict[str, float]:
    """Per-process page accounting for `pid`, in GB.

    Private_Dirty is what this process ALONE holds and had to write. Rss is
    returned for reference only — never budget against it.
    """
    out: dict[str, float] = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("Rss", "Pss", "Shared_Clean", "Shared_Dirty",
                         "Private_Clean", "Private_Dirty", "Swap"):
                    out[k] = int(v.split()[0]) / 1e6      # kB -> GB
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return out


def measure_private_dirty_gb(pid: int) -> float:
    """The honest footprint number. Falls back to Pss, never to Rss."""
    m = read_smaps_rollup(pid)
    if "Private_Dirty" in m:
        return m["Private_Dirty"]
    return m.get("Pss", 0.0)


def thread_count(pid: int = 0) -> int:
    """Pattern-D check. -1 if unreadable."""
    who = "self" if pid in (0, os.getpid()) else str(pid)
    try:
        for line in open(f"/proc/{who}/status"):
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return -1


def _cgroup_path(pid: int = 0) -> Optional[Path]:
    who = "self" if pid in (0, os.getpid()) else str(pid)
    try:
        for line in open(f"/proc/{who}/cgroup"):
            # cgroup v2: "0::/user.slice/user-N.slice/session-M.scope"
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                p = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
                return p if (p / "memory.current").exists() else None
    except (FileNotFoundError, ProcessLookupError, OSError):
        pass
    return None


def cgroup_current_gb(pid: int = 0) -> Optional[float]:
    """`memory.current` of the enclosing cgroup, in GB.

    Includes page cache, so it is the NOISIER of the two readings here — a
    release that also drops 0.5 GB of cached potential file shows up as 0.5 GB
    of extra "freed". Kept for continuity with the numbers already recorded;
    prefer cgroup_anon_gb() for the release witness.
    """
    return _cgroup_field(_cgroup_path(pid), "memory.current")


def cgroup_anon_gb(pid: int = 0) -> Optional[float]:
    """`anon` from the enclosing cgroup's memory.stat, in GB.

    THE RELEASE WITNESS. An activated LAMMPS table is anonymous memory (Rung
    R2/R3 are both anonymous, which is precisely why a byte-oriented tier
    cannot express them), so the job cgroup's anon total is the right column:
    it excludes the page cache that memory.current counts, and — the whole
    point — it does not depend on the released process still existing.
    """
    return _cgroup_field(_cgroup_path(pid), "memory.stat", key="anon")


def _cgroup_field(path: Optional[Path], filename: str,
                  key: Optional[str] = None) -> Optional[float]:
    if path is None:
        return None
    try:
        text = (path / filename).read_text()
    except (OSError, ValueError):
        return None
    try:
        if key is None:
            return int(text.strip()) / 1e9
        for line in text.splitlines():
            k, _, v = line.partition(" ")
            if k == key:
                return int(v) / 1e9
    except ValueError:
        return None
    return None


def cgroup_sample(pid: int = 0, path: Optional[Path] = None) -> dict:
    """Both cgroup readings plus the path they came from.

    The path is returned so a caller can sample the SAME cgroup either side of
    a release. Reading `pid`'s cgroup before teardown and its own afterwards
    would silently compare two different accounts on any node where the worker
    is not in the caller's cgroup.
    """
    p = path if path is not None else _cgroup_path(pid)
    return {"path": str(p) if p else None,
            "anon_gb": _cgroup_field(p, "memory.stat", key="anon"),
            "current_gb": _cgroup_field(p, "memory.current")}


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


# ===========================================================================
# The wire protocol
# ===========================================================================
# Line-delimited JSON over AF_UNIX/SOCK_STREAM. No new dependencies, no pickle
# (the peer is an LLM-generated script's interpreter; it gets to send data, not
# objects). One request per line, one response per line.
#
#   -> {"id": 7, "op": "command", "args": {"cmd": "run 0"}}
#   <- {"id": 7, "ok": true, "result": null}
#   <- {"id": 7, "ok": false, "etype": "RuntimeError", "error": "..."}

_MAX_LINE = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class Connection:
    """Framing on one socket. Used by both ends."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""

    def send(self, obj: dict) -> None:
        self.sock.sendall(json.dumps(obj).encode() + b"\n")

    def recv(self) -> Optional[dict]:
        """Next message, or None on clean EOF (peer closed / crashed)."""
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(1 << 20)
            except (ConnectionResetError, BrokenPipeError):
                return None
            except OSError as e:
                if e.errno in (errno.ECONNRESET, errno.EPIPE):
                    return None
                raise
            if not chunk:
                return None
            self._buf += chunk
            if len(self._buf) > _MAX_LINE:
                raise ProtocolError("line exceeds _MAX_LINE")
        line, _, self._buf = self._buf.partition(b"\n")
        if not line.strip():
            return {}
        return json.loads(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ===========================================================================
# Script rewriting inside the worker
# ===========================================================================
# The saving only exists if a second script can reuse the instance. Three
# things in a real AtomAgents script would otherwise destroy it:
#
#   `quit`       LAMMPS `quit` terminates the PROCESS. Every generated runner
#                ends with `lmp.command('quit')` and the .in scripts end with
#                `quit` too. Un-intercepted, the FIRST tool call kills the
#                worker and takes the 17 GB table with it. This is not a
#                hypothetical: script_builders.py emits it on nine paths.
#   `clear`      destroys the pair tables — the exact thing being retained.
#   `pair_coeff` re-issuing it on the same file costs a FULL re-parse (42.84 s
#                measured, vs 42.83 s for a cold warm-cache parse), so the
#                saving is not available by accident.
#
# `RetainedInstance` in the workload already handles the create_box/pair_coeff
# half and has a passing physics gate (results/verify_persistent_lammps_BIG.json,
# pe rel diff 0.000e+00). It is REUSED rather than reimplemented, because a
# reimplementation would need its own gate. What is added here is the `quit`
# and `clear` handling, which only a resident worker needs.

_STREAM_SKIP_PREFIXES = ("quit",)


# LAMMPS refuses these once a simulation box exists ("Units command after
# simulation box is defined", and the same for dimension/boundary/atom_style).
# Every generated AtomAgents script opens with all four. On a RETAINED instance
# the box already exists, so they have to be compared against what the instance
# was built with — skipped when identical, REFUSED when not. Silently dropping
# a differing `units` would run the next script's numbers in the wrong unit
# system and still return plausible values: measured here at a 70.8% pe
# difference on the first geometry that tried it.
_IMMUTABLE_HEADS = ("units", "dimension", "boundary", "atom_style")


class ScriptRewriter:
    """Wraps a RetainedInstance-like object and adds worker-only rules.

    `soft_clear` is the interesting one. `clear` on a retained instance would
    free the tables, but simply DROPPING it leaves the previous script's fixes,
    computes, dumps and variables installed — which would silently change the
    physics of the next script. So every id issued is tracked and explicitly
    torn down, and the atoms are deleted. That is a semantic change to someone
    else's simulation and is therefore gated by the D4 comparison, not trusted.
    """

    def __init__(self, retained, verbose: bool = False,
                 fixed_settings: Optional[dict] = None):
        self.r = retained
        self.verbose = verbose
        self.fixed = dict(fixed_settings or {})
        self.settings_skipped = 0
        self._fixes: list[str] = []
        self._computes: list[str] = []
        self._dumps: list[str] = []
        self._variables: list[str] = []
        self.clears_softened = 0
        self.quits_suppressed = 0

    # -- id bookkeeping ---------------------------------------------------
    def _track(self, cmd: str) -> None:
        p = cmd.split()
        if not p:
            return
        head = p[0].lower()
        if head == "fix" and len(p) > 1:
            self._fixes.append(p[1])
        elif head == "unfix" and len(p) > 1 and p[1] in self._fixes:
            self._fixes.remove(p[1])
        elif head == "compute" and len(p) > 1:
            self._computes.append(p[1])
        elif head == "uncompute" and len(p) > 1 and p[1] in self._computes:
            self._computes.remove(p[1])
        elif head == "dump" and len(p) > 1:
            self._dumps.append(p[1])
        elif head == "undump" and len(p) > 1 and p[1] in self._dumps:
            self._dumps.remove(p[1])
        elif head == "variable" and len(p) > 2:
            if p[2].lower() == "delete":
                if p[1] in self._variables:
                    self._variables.remove(p[1])
            elif p[1] not in self._variables:
                self._variables.append(p[1])

    def _soft_clear(self) -> None:
        """Undo the previous script WITHOUT dropping the pair tables."""
        for d in reversed(self._dumps):
            self._raw(f"undump {d}")
        for f in reversed(self._fixes):
            self._raw(f"unfix {f}")
        for c in reversed(self._computes):
            self._raw(f"uncompute {c}")
        for v in reversed(self._variables):
            self._raw(f"variable {v} delete")
        self._dumps.clear(); self._fixes.clear()
        self._computes.clear(); self._variables.clear()
        self._raw("delete_atoms group all")
        self._raw("reset_timestep 0")
        self.clears_softened += 1

    def _raw(self, cmd: str) -> None:
        try:
            self.r.lmp.command(cmd)
        except Exception as e:                    # noqa: BLE001
            if self.verbose:
                print(f"[worker] soft_clear: '{cmd}' -> {e}", file=sys.stderr)

    # -- the entry point --------------------------------------------------
    def command(self, cmd: str) -> None:
        s = cmd.strip()
        if not s or s.startswith("#"):
            return
        head = s.split()[0].lower()
        if head in _STREAM_SKIP_PREFIXES:
            # `quit` would kill the worker. Suppressing it is what makes the
            # process survive to serve a second tool call at all.
            self.quits_suppressed += 1
            return
        if head == "clear":
            if getattr(self.r, "_coeff_loaded", False):
                self._soft_clear()
                return
            self.r.lmp.command(s)                 # first script: pass through
            return
        if head in _IMMUTABLE_HEADS:
            want = " ".join(s.split()[1:])
            have = self.fixed.get(head)
            if have is None:
                self.fixed[head] = want
                self.r.lmp.command(s)
                return
            if _norm(want) == _norm(have):
                self.settings_skipped += 1        # already true; re-issuing errors
                return
            raise RuntimeError(
                f"persistent_lammps/worker: this script asks for '{head} {want}' "
                f"but the retained instance was built with '{head} {have}'. "
                f"LAMMPS cannot change that once a box exists, and applying the "
                f"script anyway would silently compute in the wrong "
                f"{head}. Run this script through the fork path.")
        self._track(s)
        self.r.command(s)                         # RetainedInstance's rewrites

    @property
    def stats(self) -> dict:
        return {
            "clears_softened": self.clears_softened,
            "settings_skipped": self.settings_skipped,
            "fixed_settings": dict(self.fixed),
            "quits_suppressed": self.quits_suppressed,
            "parses_avoided": getattr(self.r, "parses_avoided", 0),
            "parses_paid": getattr(self.r, "parses_paid", 0),
            "live_fixes": list(self._fixes),
            "live_computes": list(self._computes),
            "live_dumps": list(self._dumps),
        }


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def flatten_lines(path: str) -> list[str]:
    """One file -> one command per element. Comments and `&` continuations only.

    Deliberately does NOT follow `include`: see stream_file().
    """
    out: list[str] = []
    pending = ""
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.rstrip().endswith("&"):
                pending += line.rstrip()[:-1].rstrip() + " "
                continue
            line = (pending + line.strip()).strip()
            pending = ""
            if not line:
                continue
            head = line.split()[0].lower()
            if head in ("jump", "if", "next", "label"):
                raise RuntimeError(
                    f"flatten_lines: control-flow command '{head}' in {path}; "
                    f"refusing to stream it. Run this script through the native "
                    f"file() path instead of guessing its order.")
            out.append(line)
    if pending.strip():
        out.append(pending.strip())
    return out


_VAR_RE = None


def substitute_vars(text: str, lookup) -> str:
    """Resolve ${name} / $(name) / $x against a caller-supplied lookup."""
    global _VAR_RE
    if _VAR_RE is None:
        import re
        _VAR_RE = re.compile(r"\$\{(\w+)\}|\$\((\w+)\)|\$([A-Za-z])")
    def rep(m):
        name = m.group(1) or m.group(2) or m.group(3)
        v = lookup(name)
        if v is None:
            raise RuntimeError(
                f"cannot resolve LAMMPS variable '{name}' while expanding an "
                f"`include`. The worker has to know the included FILE to reach "
                f"its pair_coeff, and guessing the path would silently load a "
                f"different potential.")
        return str(v)
    return _VAR_RE.sub(rep, text)


def expand_input_script(path: str, _depth: int = 0, resolve=None) -> list[str]:
    """Flatten a LAMMPS .in file, following `include`, to a list of commands.

    STATIC form, used by tests and by the native-mode `quit` scan. The worker
    itself uses stream_file(), because `include ${potential_file}` after a
    `shell cd` -- which is exactly what script_builders.py emits -- can only be
    resolved against the LIVE instance's variables and its current directory.
    """
    if _depth > 8:
        raise RuntimeError("include nesting too deep")
    out: list[str] = []
    base = os.path.dirname(os.path.abspath(path))
    for line in flatten_lines(path):
        if line.split()[0].lower() == "include" and len(line.split()) > 1:
            inc = line.split(maxsplit=1)[1].strip().strip('"').strip("'")
            if resolve is not None:
                inc = substitute_vars(inc, resolve)
            elif "$" in inc:
                raise RuntimeError(
                    f"`include {inc}` needs a LAMMPS variable resolved; call "
                    f"with resolve= or use stream_file()")
            cand = inc if os.path.isabs(inc) else os.path.join(os.getcwd(), inc)
            if not os.path.exists(cand):
                cand = os.path.join(base, inc)
            out.extend(expand_input_script(cand, _depth + 1, resolve))
            continue
        out.append(line)
    return out


# ===========================================================================
# The worker process
# ===========================================================================

class _WorkerState:
    def __init__(self) -> None:
        self.lmp = None
        self.retained = None
        self.rewriter: Optional[ScriptRewriter] = None
        self.potential: Optional[str] = None
        self.pair_style: Optional[str] = None
        self.baseline_gb: float = 0.0
        self.activated_gb: float = 0.0
        self.activate_s: float = 0.0
        self.activations: int = 0
        self.scripts_served: int = 0
        self.threads_at_start: int = -1
        self.rewrite: bool = False
        self.quits_suppressed: int = 0


def _libc_malloc_trim() -> bool:
    """Ask glibc to return free arenas to the OS.

    Without this an in-process `close()` can free the C++ heap while the
    allocator keeps the pages, so Private_Dirty would not fall and a release
    that DID work would look like one that did not. Reported, not assumed.
    """
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except Exception:                             # noqa: BLE001
        return False


class DataWorkerServer:
    """The long-lived process. One activated structure, many client scripts."""

    def __init__(self, socket_path: str, max_threads: int = DEFAULT_MAX_THREADS,
                 atomagents_root: Optional[str] = None, verbose: bool = False):
        self.socket_path = socket_path
        self.max_threads = max_threads
        self.verbose = verbose
        self.st = _WorkerState()
        self._running = True
        if atomagents_root and atomagents_root not in sys.path:
            sys.path.insert(0, atomagents_root)

    # -- lifecycle --------------------------------------------------------
    def _log(self, *a) -> None:
        if self.verbose:
            print("[worker]", *a, file=sys.stderr, flush=True)

    def _check_threads(self, when: str) -> int:
        n = thread_count()
        self._log(f"threads {when}: {n}")
        if n > self.max_threads:
            raise RuntimeError(
                f"Pattern D: {n} threads at {when} (max {self.max_threads}). "
                f"Thread pools were not pinned before the consumer library was "
                f"imported. Measured on this cluster: 193 threads by default, "
                f"3 with OMP_NUM_THREADS=1 (cow_prefork_result_20260829.md).")
        return n

    def serve(self) -> int:
        for k, v in _THREAD_PINS.items():
            os.environ.setdefault(k, v)
        self._check_threads("interpreter start")

        # Import the consumer library only AFTER the pins are in place.
        import lammps as _lammps_mod                      # noqa: F401
        self.st.threads_at_start = self._check_threads("after import lammps")

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        srv.listen(8)
        # Announce readiness on stdout so the actor does not have to poll blind.
        print(json.dumps({"ready": True, "pid": os.getpid(),
                          "socket": self.socket_path,
                          "threads": self.st.threads_at_start}), flush=True)

        try:
            while self._running:
                try:
                    conn_sock, _ = srv.accept()
                except OSError as e:
                    if e.errno == errno.EINTR:
                        continue
                    raise
                conn = Connection(conn_sock)
                try:
                    self._serve_connection(conn)
                except Exception as e:                    # noqa: BLE001
                    # A client that dies mid-request MUST NOT take the retained
                    # structure with it. This is the crash-isolation guarantee.
                    self._log(f"connection ended: {type(e).__name__}: {e}")
                finally:
                    conn.close()
        finally:
            srv.close()
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        return 0

    def _serve_connection(self, conn: Connection) -> None:
        cwd0 = os.getcwd()
        try:
            while True:
                msg = conn.recv()
                if msg is None:                    # EOF: client exited/crashed
                    self._log("client EOF; retained structure kept")
                    return
                if not msg:
                    continue
                rid = msg.get("id")
                op = msg.get("op", "")
                args = msg.get("args") or {}
                try:
                    result = self._dispatch(op, args)
                    conn.send({"id": rid, "ok": True, "result": result})
                except Exception as e:             # noqa: BLE001
                    conn.send({"id": rid, "ok": False,
                               "etype": type(e).__name__, "error": str(e)})
                if op == "shutdown":
                    self._running = False
                    return
        finally:
            # A `shell cd` from one script must not leak into the next one.
            try:
                os.chdir(cwd0)
            except OSError:
                pass

    # -- ops --------------------------------------------------------------
    def _dispatch(self, op: str, a: dict) -> Any:
        fn = getattr(self, f"_op_{op}", None)
        if fn is None:
            raise ProtocolError(f"unknown op: {op!r}")
        return fn(a)

    def _op_ping(self, a: dict) -> dict:
        return {
            "pid": os.getpid(),
            "threads": thread_count(),
            "threads_at_start": self.st.threads_at_start,
            "resident": self.st.lmp is not None,
            "potential": self.st.potential,
            "pair_style": self.st.pair_style,
            "cwd": os.getcwd(),
        }

    def _op_smaps(self, a: dict) -> dict:
        return read_smaps_rollup(os.getpid())

    def _op_stats(self, a: dict) -> dict:
        out = {
            "potential": self.st.potential,
            "pair_style": self.st.pair_style,
            "baseline_gb": round(self.st.baseline_gb, 4),
            "activated_gb": round(self.st.activated_gb, 4),
            "private_dirty_gb": round(measure_private_dirty_gb(os.getpid()), 4),
            "activate_s": round(self.st.activate_s, 3),
            "activations": self.st.activations,
            "scripts_served": self.st.scripts_served,
            "threads": thread_count(),
            "rewrite": self.st.rewrite,
            "quits_suppressed": self.st.quits_suppressed,
        }
        if self.st.rewriter is not None:
            out["rewriter"] = self.st.rewriter.stats
        return out

    def _op_activate(self, a: dict) -> dict:
        """Parse the potential into a live instance. THE cost being retained."""
        potential = os.path.abspath(a["potential"])
        pair_style = a.get("pair_style") or (
            "eam/fs" if potential.endswith(".fs") else "eam/alloy")
        setup = a.get("setup") or [
            "units metal", "boundary p p p", "atom_style atomic",
            "lattice bcc 3.165", "region box block 0 2 0 2 0 2",
            "create_box 1 box", "create_atoms 1 box", "mass 1 183.84",
        ]
        element = a.get("element", "W")
        logfile = a.get("logfile") or "/tmp/tandem_worker.log"

        from lammps import lammps
        self.st.baseline_gb = measure_private_dirty_gb(os.getpid())
        lmp = lammps(cmdargs=["-log", logfile, "-screen", "none"])
        for c in setup:
            lmp.command(c)
        lmp.command(f"pair_style {pair_style}")
        t0 = time.perf_counter()
        lmp.command(f"pair_coeff * * {potential} {element}")
        lmp.command("run 0")           # forces interpolate(), not just the read
        self.st.activate_s = time.perf_counter() - t0

        self.st.lmp = lmp
        self.st.potential = potential
        self.st.pair_style = pair_style
        self.st.activations += 1
        self.st.activated_gb = (measure_private_dirty_gb(os.getpid())
                                - self.st.baseline_gb)
        self.st.rewrite = bool(a.get("rewrite", True))
        if self.st.rewrite:
            # What the instance was BUILT with, so a later script asking for
            # something different is refused rather than silently ignored.
            fixed = {}
            for c in setup:
                p = c.split()
                if p and p[0].lower() in _IMMUTABLE_HEADS:
                    fixed[p[0].lower()] = " ".join(p[1:])
            self.st.retained = _make_retained(lmp, potential, pair_style)
            self.st.rewriter = ScriptRewriter(self.st.retained, self.verbose,
                                              fixed_settings=fixed)
        self._check_threads("after activation")
        return {"activate_s": round(self.st.activate_s, 3),
                "activated_gb": round(self.st.activated_gb, 4),
                "baseline_gb": round(self.st.baseline_gb, 4),
                "private_dirty_gb": round(measure_private_dirty_gb(os.getpid()), 4),
                "threads": thread_count()}

    def _target(self):
        if self.st.lmp is None:
            raise RuntimeError("no activated instance; call activate first")
        return self.st.rewriter if self.st.rewriter is not None else self.st.lmp

    def _one(self, cmd: str) -> None:
        """Every command entering the worker goes through here.

        The `quit` guard is OUTSIDE the rewriter on purpose. LAMMPS `quit`
        terminates the PROCESS, and rewrite=False is a legitimate mode (it saves
        nothing but is a correctness baseline) — so if the guard lived only in
        ScriptRewriter, running the baseline arm would kill the worker on the
        first generated script. Nine of the script builders emit `quit`.
        """
        s = cmd.strip()
        if not s or s.startswith("#"):
            return
        if s.split()[0].lower() == "quit":
            self.st.quits_suppressed += 1
            return
        self._target().command(s)

    def _op_command(self, a: dict) -> None:
        self._one(a["cmd"])

    def _op_commands_string(self, a: dict) -> None:
        for line in str(a["multicmd"]).splitlines():
            self._one(line)

    def _op_commands_list(self, a: dict) -> None:
        for line in a["cmdlist"]:
            self._one(line)

    def _op_file(self, a: dict) -> dict:
        """Run a .in script. `stream` is what makes retention reachable."""
        path = a["path"]
        mode = a.get("mode", "stream" if self.st.rewriter else "native")
        self.st.scripts_served += 1
        if mode == "native":
            # LAMMPS parses the file itself, so nothing here can intercept a
            # `quit` or a `clear` inside it. Correct, but it retains nothing and
            # a `quit` will take the worker down. Refuse rather than surprise.
            try:
                scan = expand_input_script(path, resolve=self._resolve)
            except RuntimeError:
                scan = flatten_lines(path)         # best effort
            for c in scan:
                if c.split()[0].lower() == "quit":
                    raise RuntimeError(
                        f"{path} contains `quit`, which terminates the worker "
                        f"process. Use mode='stream' (the default when the "
                        f"rewriter is active).")
            self.st.lmp.file(path)
            return {"mode": "native", "commands": None}
        n = self._stream_file(path)
        return {"mode": "stream" if self.st.rewriter else "stream_norewrite",
                "commands": n}

    def _resolve(self, name: str):
        """A LAMMPS variable's value, read off the LIVE instance."""
        try:
            return self.st.lmp.extract_variable(name)
        except Exception:                          # noqa: BLE001
            return None

    def _stream_file(self, path: str, depth: int = 0) -> int:
        """Feed a .in file through the rewriting layer, one command at a time.

        LAZY, and it has to be. `shell cd ./wf` changes the directory that a
        later `include potential.inp` resolves against, and
        `include ${potential_file}` needs a variable that only exists once the
        preceding `variable ... string` command has run. Flattening the file up
        front got both wrong -- measured as
        FileNotFoundError: .../wf/${potential_file} on the first real generated
        script this path was pointed at.
        """
        if depth > 8:
            raise RuntimeError("include nesting too deep")
        base = os.path.dirname(os.path.abspath(path))
        n = 0
        for line in flatten_lines(path):
            parts = line.split()
            if parts[0].lower() == "include" and len(parts) > 1:
                inc = substitute_vars(line.split(maxsplit=1)[1].strip()
                                      .strip('"').strip("'"), self._resolve)
                cand = inc if os.path.isabs(inc) else os.path.join(os.getcwd(), inc)
                if not os.path.exists(cand):
                    cand = os.path.join(base, inc)
                n += self._stream_file(cand, depth + 1)
                continue
            self._one(line)
            n += 1
        return n

    def _op_chdir(self, a: dict) -> str:
        os.chdir(a["path"])
        return os.getcwd()

    def _op_get_thermo(self, a: dict) -> float:
        return float(self.st.lmp.get_thermo(a["name"]))

    def _op_get_natoms(self, a: dict) -> int:
        return int(self.st.lmp.get_natoms())

    def _op_version(self, a: dict) -> int:
        return int(self.st.lmp.version())

    # extract_* — scalars pass straight through; arrays go via the numpy
    # wrapper and arrive as lists. POINTER IDENTITY IS NOT PRESERVED: the real
    # API hands back a live ctypes view into LAMMPS memory and a socket cannot.
    # Reads are faithful; in-place writes through the returned object are not
    # supported and raise on the client rather than silently doing nothing.
    def _op_extract_global(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.extract_global(a["name"]))

    def _op_extract_setting(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.extract_setting(a["name"]))

    def _op_extract_box(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.extract_box())

    def _op_extract_variable(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.extract_variable(
            a["name"], a.get("group"), a.get("vartype")))

    def _op_extract_atom(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.numpy.extract_atom(a["name"]))

    def _op_extract_compute(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.numpy.extract_compute(
            a["cid"], int(a["cstyle"]), int(a["ctype"])))

    def _op_extract_fix(self, a: dict) -> Any:
        return _jsonable(self.st.lmp.numpy.extract_fix(
            a["fid"], int(a["fstyle"]), int(a["ftype"]),
            int(a.get("nrow", 0)), int(a.get("ncol", 0))))

    def _op_close(self, a: dict) -> dict:
        """In-process release. MEASURED, because it is allowed to fail.

        glibc may hold freed arenas, in which case Private_Dirty does not fall
        and this returns a small number — which is the honest answer and the
        signal for the actor to fall back to process teardown.
        """
        before = measure_private_dirty_gb(os.getpid())
        if self.st.lmp is not None:
            try:
                self.st.lmp.close()
            except Exception:                      # noqa: BLE001
                pass
        self.st.lmp = None
        self.st.retained = None
        self.st.rewriter = None
        self.st.potential = None
        import gc
        gc.collect()
        trimmed = _libc_malloc_trim()
        after = measure_private_dirty_gb(os.getpid())
        return {"private_dirty_before_gb": round(before, 4),
                "private_dirty_after_gb": round(after, 4),
                "freed_gb": round(before - after, 4),
                "malloc_trim": trimmed}

    def _op_shutdown(self, a: dict) -> dict:
        return {"pid": os.getpid(), "bye": True}


def _make_retained(lmp, potential: str, pair_style: str):
    """Reuse the workload's already-gated rewriting layer if importable.

    `RetainedInstance` is what results/verify_persistent_lammps_BIG.json passed
    (pe rel diff 0.000e+00). Reimplementing its create_box/pair_coeff rules here
    would need a fresh gate for no benefit, so it is imported. The fallback is
    pass-through, which is correct but saves nothing — it never silently
    pretends to have rewritten.
    """
    try:
        from atomagents.execution.persistent_lammps import RetainedInstance
    except ImportError:
        class _PassThrough:                        # explicit, not silent
            def __init__(self, lmp, potential, pair_style):
                self.lmp = lmp; self.potential = potential
                self.pair_style = pair_style
                self.parses_avoided = 0; self.parses_paid = 1
                self._coeff_loaded = True
            def command(self, cmd): self.lmp.command(cmd)
            def __getattr__(self, n): return getattr(self.__dict__["lmp"], n)
        return _PassThrough(lmp, potential, pair_style)
    inst = RetainedInstance(lmp, potential, pair_style)
    # The instance was activated by _op_activate, so the flags describe reality.
    inst._box_defined = True
    inst._style_set = True
    inst._coeff_loaded = True
    return inst


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    raise TypeError(
        f"{type(x).__name__} cannot cross the worker boundary. The real API "
        f"returns a live ctypes view into LAMMPS memory; a socket cannot "
        f"reproduce pointer identity. Refusing rather than returning a copy "
        f"that would silently not write back.")


# ===========================================================================
# The actor (runs in the RUNTIME process) — implements ResidencyActor
# ===========================================================================

def _r4(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 4)


def _delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    """before - after, or None if either sample is missing. NOT clamped: a
    negative delta is a real observation that the enclosing account did not
    shrink, and it must reach the ledger as contrary evidence rather than as
    an absent witness."""
    if before is None or after is None:
        return None
    return round(before - after, 4)


class _Handle:
    __slots__ = ("proc", "socket_path", "spec", "activated_gb", "baseline_gb",
                 "activate_s", "threads")

    def __init__(self, proc, socket_path, spec):
        self.proc = proc
        self.socket_path = socket_path
        self.spec = spec
        self.activated_gb = 0.0
        self.baseline_gb = 0.0
        self.activate_s = 0.0
        self.threads = -1


class LammpsDataWorker:
    """ResidencyActor for ResourceClass.DATA_PATTERN_B.

    One worker PROCESS per resource_id. That is deliberate and it is what makes
    I1 and I2 answerable: a per-process smaps_rollup is an exact per-resource
    footprint, and a released resource can be shown to have left the machine.
    Multiplexing several potentials into one process would make `measure_held_gb
    (resource_id)` an apportionment guess, which is precisely the kind of
    declared-rather-than-measured number I1 forbids.

    `artifacts` maps resource_id -> {"potential": path, "pair_style": ...},
    normally loaded from tool_resources.json via `load_residency_artifacts()`.
    """

    def __init__(
        self,
        artifacts: Optional[dict[str, dict]] = None,
        python: Optional[str] = None,
        socket_dir: str = "/tmp",
        atomagents_root: Optional[str] = None,
        max_threads: int = DEFAULT_MAX_THREADS,
        release_confirm_frac: float = 0.90,
        startup_timeout_s: float = 600.0,
        verbose: bool = False,
    ) -> None:
        self.artifacts = dict(artifacts or {})
        self.python = python or sys.executable
        self.socket_dir = socket_dir
        self.atomagents_root = atomagents_root
        self.max_threads = max_threads
        self.release_confirm_frac = release_confirm_frac
        self.startup_timeout_s = startup_timeout_s
        self.verbose = verbose
        self._held: dict[str, _Handle] = {}
        self.last_release_detail: dict[str, dict] = {}
        # Monotonic, never reset. Stamps each release's cached witness so a
        # stale one cannot be returned as a fresh observation.
        self._release_seq: int = 0

    # -- ResidencyActor ---------------------------------------------------
    @property
    def resource_class(self) -> ResourceClass:
        return ResourceClass.DATA_PATTERN_B

    def stage(self, spec: ResourceSpec) -> Rung:
        """Spawn a worker and activate the potential. Returns the rung REACHED.

        Returns R0_DISK if the worker could not be started and R1_PAGE_CACHE if
        it started but activation failed — a lower rung is not an error, but
        reporting the requested rung when it was not reached is (contract).
        """
        rid = spec.resource_id
        if rid in self._held:
            return Rung.R3_ACTIVATED
        art = self.artifacts.get(rid)
        if art is None:
            raise KeyError(
                f"{rid}: no residency artifact registered. Add it to "
                f"runtime/predictor/data/tool_resources.json (item D3) or pass "
                f"it in `artifacts`.")
        sock = os.path.join(
            self.socket_dir, f"tandem-lmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")

        env = os.environ.copy()
        env.update(_THREAD_PINS)                   # BEFORE the interpreter starts
        env["PYTHONUNBUFFERED"] = "1"
        pypath = [str(_REPO_ROOT)]
        if self.atomagents_root:
            pypath.append(self.atomagents_root)
        if env.get("PYTHONPATH"):
            pypath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pypath)

        cmd = [self.python, "-u", str(_HERE / "data_worker.py"), "--serve",
               "--socket", sock, "--max-threads", str(self.max_threads)]
        if self.atomagents_root:
            cmd += ["--atomagents-root", self.atomagents_root]
        if self.verbose:
            cmd += ["--verbose"]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=None if self.verbose else subprocess.DEVNULL,
                                text=True, start_new_session=True)
        h = _Handle(proc, sock, spec)
        line = proc.stdout.readline()
        if not line:
            proc.kill()
            return Rung.R0_DISK
        hello = json.loads(line)
        h.threads = hello.get("threads", -1)

        try:
            res = self._rpc(h, "activate", {
                "potential": art["potential"],
                "pair_style": art.get("pair_style"),
                "element": art.get("element", "W"),
                "setup": art.get("setup"),
                "logfile": art.get("logfile", f"/tmp/tandem-{rid}.log"),
                "rewrite": art.get("rewrite", True),
            }, timeout_s=self.startup_timeout_s)
        except Exception:                          # noqa: BLE001
            self._kill(h)
            return Rung.R1_PAGE_CACHE
        h.activated_gb = res["activated_gb"]
        h.baseline_gb = res["baseline_gb"]
        h.activate_s = res["activate_s"]
        self._held[rid] = h
        return Rung.R3_ACTIVATED

    def measure_held_gb(self, resource_id: str) -> float:
        """INVARIANT I1: Private_Dirty from the worker's smaps_rollup.

        The WHOLE process is charged, not just the marginal table: the worker
        exists only to hold this resource, so its interpreter and the LAMMPS
        library are part of the price of holding it. `activation_detail()`
        exposes the split for anyone who wants the marginal number.
        """
        h = self._held.get(resource_id)
        if h is None or h.proc.poll() is not None:
            return 0.0
        return measure_private_dirty_gb(h.proc.pid)

    def is_resident(self, resource_id: str) -> bool:
        h = self._held.get(resource_id)
        return h is not None and h.proc.poll() is None

    def release(self, resource_id: str) -> float:
        """INVARIANT I2 — return the GB the OS ACTUALLY gave back.

        Two stages, both measured:
          1. in-process `close()` + malloc_trim, re-read Private_Dirty. This is
             allowed to fail (glibc arena retention) and often will.
          2. process teardown: SIGTERM, then SIGKILL, then WAIT FOR /proc/<pid>
             to disappear. Only then is the address space actually gone.
        The cgroup's anon delta is captured here as the INDEPENDENT WITNESS
        that release_witness() hands to the ledger, because "the process
        exited, therefore the memory came back" is exactly the by-construction
        claim I2 exists to distrust. Everything is kept in
        `last_release_detail[resource_id]`.
        """
        h = self._held.get(resource_id)
        if h is None:
            return 0.0
        held_before = self.measure_held_gb(resource_id)
        # Resolve the cgroup ONCE, from the worker, and read that same account
        # on both sides. Sampling the worker's cgroup before and the actor's
        # after would compare two different accounts on any node where they
        # differ, and the mismatch would look like a witness.
        cg_path = _cgroup_path(h.proc.pid)
        cg_pre = cgroup_sample(path=cg_path)

        in_proc = {}
        if h.proc.poll() is None:
            try:
                in_proc = self._rpc(h, "close", {}, timeout_s=120.0)
            except Exception as e:                 # noqa: BLE001
                in_proc = {"error": f"{type(e).__name__}: {e}"}

        pid = h.proc.pid
        self._kill(h)
        gone_after_s = self._wait_gone(pid, 30.0)
        # Same cgroup path as cg_pre. If it has itself gone (a per-process
        # scope), there is no comparable "after" and therefore no witness --
        # which is a None, not a zero.
        cg_post = cgroup_sample(path=cg_path)

        alive = pid_alive(pid)
        self._release_seq += 1
        detail = {
            "resource_id": resource_id,
            "pid": pid,
            "release_seq": self._release_seq,      # the staleness stamp
            "released_at_s": time.time(),
            "held_before_gb": round(held_before, 4),
            "in_process_close": in_proc,
            "in_process_freed_gb": round(in_proc.get("freed_gb", 0.0), 4)
                                    if isinstance(in_proc, dict) else 0.0,
            "proc_gone": not alive,
            "proc_gone_after_s": round(gone_after_s, 3),
            "cgroup_path": cg_pre["path"],
            "cgroup_anon_before_gb": _r4(cg_pre["anon_gb"]),
            "cgroup_anon_after_gb": _r4(cg_post["anon_gb"]),
            "cgroup_anon_freed_gb": _delta(cg_pre["anon_gb"], cg_post["anon_gb"]),
            "cgroup_before_gb": _r4(cg_pre["current_gb"]),
            "cgroup_after_gb": _r4(cg_post["current_gb"]),
            "cgroup_freed_gb": _delta(cg_pre["current_gb"], cg_post["current_gb"]),
            "measured_by": "Private_Dirty(/proc/<pid>/smaps_rollup)",
            "witness_measured_by": "anon(<job cgroup>/memory.stat)",
        }
        self.last_release_detail[resource_id] = detail
        self._held.pop(resource_id, None)

        if alive:
            raise ReleaseNotHonoured(
                f"{resource_id}: worker pid {pid} still present after SIGKILL; "
                f"{held_before:.3f} GB charged and not demonstrably returned. "
                f"The budget is fiction from this point on.")
        cgf = detail["cgroup_anon_freed_gb"]
        if cgf is None:
            cgf = detail["cgroup_freed_gb"]
        if cgf is not None and held_before > 0.05:
            detail["cgroup_corroborates"] = cgf >= self.release_confirm_frac * held_before
        return held_before

    def release_witness(self, resource_id: str) -> Optional[float]:
        """GB the ENCLOSING allocation gave back, independent of this
        resource's own process. None when there is no such witness.

        WHY THIS ACTOR NEEDS ONE. LammpsDataWorker is a TEARDOWN actor: it
        releases by killing the worker process, so measure_held_gb() afterwards
        is 0.0 BY CONSTRUCTION and the ledger's before/after drop is
        tautologically the whole charge. The witness is the job cgroup's `anon`
        total sampled either side of the release (contract, added 2026-08-30) —
        a reading that does not depend on the released process still existing,
        and that excludes the page cache memory.current would have counted.

        NEGATIVE AND SHORT VALUES ARE RETURNED AS MEASURED. On a shared node
        other processes move the account, so a witness can come back small or
        below zero. Clamping it to None would hide CONTRARY evidence behind
        "no evidence", which is the opposite of the point; the tolerance
        belongs to the ledger.

        THE STALENESS GUARD (raised by A1 about this file specifically). The
        witness is cached in last_release_detail, and that entry outlives the
        release. A later call handing back the same number would look exactly
        like a fresh witness and would vouch for a release it never observed —
        the same tautology this method exists to break. So it is stamped with a
        monotonic release counter and returns None unless:
          * a release for THIS resource is the most recent one this actor
            performed (an intervening release of anything else invalidates it,
            because the cgroup has moved since), and
          * the resource has not been re-staged in the meantime (a resident
            resource has no outstanding release for a witness to describe).
        """
        d = self.last_release_detail.get(resource_id)
        if d is None:
            return None
        if d.get("release_seq") != self._release_seq:
            return None                            # a later release moved the cgroup
        if resource_id in self._held:
            return None                            # re-staged; nothing outstanding
        return d.get("cgroup_anon_freed_gb")

    # -- extras (not part of the protocol) --------------------------------
    def activation_detail(self, resource_id: str) -> dict:
        h = self._held.get(resource_id)
        if h is None:
            return {}
        return {"pid": h.proc.pid, "socket": h.socket_path,
                "activated_gb": h.activated_gb, "baseline_gb": h.baseline_gb,
                "activate_s": h.activate_s, "threads_at_start": h.threads,
                "private_dirty_gb": self.measure_held_gb(resource_id)}

    def socket_for(self, resource_id: str) -> Optional[str]:
        """The value to put in TANDEM_LAMMPS_SOCKET for a tool subprocess."""
        h = self._held.get(resource_id)
        return h.socket_path if h else None

    def client_env(self, resource_id: str, env: Optional[dict] = None) -> dict:
        """Environment for a tool subprocess that should use this worker."""
        e = dict(env or os.environ)
        sock = self.socket_for(resource_id)
        if sock:
            e[SOCKET_ENV] = sock
        return prepend_shim_path(e)

    def call(self, resource_id: str, op: str, args: Optional[dict] = None,
             timeout_s: float = 3600.0) -> Any:
        h = self._held.get(resource_id)
        if h is None:
            raise KeyError(f"{resource_id} not resident")
        return self._rpc(h, op, args or {}, timeout_s)

    def release_all(self) -> None:
        for rid in list(self._held):
            try:
                self.release(rid)
            except Exception:                      # noqa: BLE001
                pass

    # -- plumbing ---------------------------------------------------------
    def _rpc(self, h: _Handle, op: str, args: dict, timeout_s: float) -> Any:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        deadline = time.time() + min(timeout_s, 60.0)
        while True:
            try:
                s.connect(h.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.time() > deadline:
                    raise
                time.sleep(0.02)
        conn = Connection(s)
        try:
            conn.send({"id": 1, "op": op, "args": args})
            reply = conn.recv()
            if reply is None:
                raise RuntimeError(f"worker closed during op {op!r}")
            if not reply.get("ok"):
                raise RuntimeError(
                    f"worker op {op!r} failed: {reply.get('etype')}: "
                    f"{reply.get('error')}")
            return reply.get("result")
        finally:
            conn.close()

    def _kill(self, h: _Handle) -> None:
        if h.proc.poll() is not None:
            return
        try:
            h.proc.terminate()
            h.proc.wait(timeout=15)
        except Exception:                          # noqa: BLE001
            try:
                os.killpg(os.getpgid(h.proc.pid), signal.SIGKILL)
            except Exception:                      # noqa: BLE001
                try:
                    h.proc.kill()
                except Exception:                  # noqa: BLE001
                    pass
            try:
                h.proc.wait(timeout=15)
            except Exception:                      # noqa: BLE001
                pass
        try:
            os.unlink(h.socket_path)
        except OSError:
            pass

    @staticmethod
    def _wait_gone(pid: int, timeout_s: float) -> float:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if not pid_alive(pid):
                return time.perf_counter() - t0
            time.sleep(0.01)
        return time.perf_counter() - t0


# ===========================================================================
# D3 — reading the registration back out of tool_resources.json
# ===========================================================================

_DEFAULT_TOOL_RESOURCES = _REPO_ROOT / "runtime" / "predictor" / "data" / "tool_resources.json"

_CLASS_BY_NAME = {c.value: c for c in ResourceClass}


def load_residency_artifacts(
    path: str | os.PathLike | None = None,
    repo_root: str | os.PathLike | None = None,
) -> tuple[dict[str, ResourceSpec], dict[str, dict]]:
    """Read the `residency_artifact` blocks from tool_resources.json.

    Returns (specs, artifacts): the contract's ResourceSpec for the ledger to
    charge, and the actor-side {potential, pair_style, ...} for staging.

    THESE ENTRIES DELIBERATELY CARRY NO `consumer_tool`. `ResourceRegistry.
    from_json()` skips entries without one, so the legacy byte-oriented prefetch
    registry does not see them and no existing eval trace changes shape. That
    is not a workaround: an R3 activated structure is not a file range, which
    is the whole reason the residency tier exists (contract.py, Rung).
    """
    p = Path(path) if path else _DEFAULT_TOOL_RESOURCES
    root = Path(repo_root) if repo_root else _REPO_ROOT
    specs: dict[str, ResourceSpec] = {}
    arts: dict[str, dict] = {}
    if not p.exists():
        return specs, arts
    for entry in json.loads(p.read_text()):
        if not isinstance(entry, dict):
            continue
        a = entry.get("residency_artifact")
        if not isinstance(a, dict):
            continue
        rid = a["resource_id"]
        pot = a["potential_path"]
        pot = pot if os.path.isabs(pot) else str(root / pot)
        specs[rid] = ResourceSpec(
            resource_id=rid,
            resource_class=_CLASS_BY_NAME[a["resource_class"]],
            held_rung=Rung(int(a["held_rung"])),
            held_gb=float(a["held_gb"]),
            cold_s=float(a["cold_s"]),
            ready_s=float(a["ready_s"]),
        )
        arts[rid] = {"potential": pot,
                     "pair_style": a.get("pair_style"),
                     "element": a.get("element", "W"),
                     "rewrite": a.get("rewrite", True)}
    return specs, arts


# ===========================================================================
# The one thing the framework side needs
# ===========================================================================

def prepend_shim_path(env: dict) -> dict:
    """Put the client shim's directory first on a subprocess's PYTHONPATH.

    This is the ENTIRE integration with the framework runner. It is option 4,
    not option 1: no control flow is patched, the LLM-generated script still
    runs in its own crashable subprocess, and `import lammps` inside it resolves
    to the shim only because the shim's directory sorts ahead of site-packages.
    Unset TANDEM_LAMMPS_SOCKET and the shim imports the real library instead, so
    this is reversible by removing one environment variable.
    """
    parts = [str(SHIM_DIR)]
    old = env.get("PYTHONPATH")
    if old:
        parts.append(old)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve", action="store_true", help="run as the worker")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS)
    ap.add_argument("--atomagents-root", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if not a.serve:
        ap.error("nothing to do; pass --serve (the actor spawns this itself)")
    if not a.socket:
        ap.error("--serve needs --socket")
    return DataWorkerServer(a.socket, a.max_threads, a.atomagents_root,
                            a.verbose).serve()


if __name__ == "__main__":
    raise SystemExit(main())
