"""lammps — the thin client shim for Tandem's resident data worker (T4b).

WHAT THIS IS. A drop-in replacement for the `lammps` Python module, imported by
an LLM-generated tool script because this directory sits first on that
subprocess's PYTHONPATH. `lammps(...)` here does not create a LAMMPS instance:
it connects to a worker process that already has the potential parsed, and
forwards commands over a UNIX socket.

WHY THIS SHAPE. Re-issuing `pair_coeff` on the same file costs a full re-parse
(42.84 s measured, against 42.83 s for a warm cold-start), so the saving is not
available by accident — something has to reach the ALREADY-ACTIVATED instance.
Options 1 (patch the framework's control flow) and 5 (serialize + mmap) were
rejected by the user; option 6 (prefork) was killed by measurement, because
LAMMPS `run` rewrites 92% of the coefficient table in place and copy-on-write
buys nothing (sc-workshop-paper/cow_prefork_result_20260829.md). A resident
worker re-interpolates into the same memory, so its footprint stays 1x.

CRASH ISOLATION IS PRESERVED, AND THAT IS THE POINT. The generated script still
runs in its own subprocess with its own address space. If it segfaults, this
socket closes, the worker sees EOF, and the activated structure survives. The
worker never runs generated code — it only executes LAMMPS commands.

IT FALLS BACK. With TANDEM_LAMMPS_SOCKET unset, or if the connection fails,
`lammps(...)` constructs and returns a REAL lammps instance. So dropping this
directory on PYTHONPATH cannot break a run; set TANDEM_LAMMPS_STRICT=1 to turn
a failed connection into an error instead (use that when measuring, so a silent
fallback cannot be mistaken for a fast worker).

WHAT IS DELIBERATELY NOT SUPPORTED. `extract_atom`/`extract_compute` return
COPIES, not the live ctypes views the real API hands back: a socket cannot
reproduce pointer identity. Reads are faithful; writing through a returned
array does not reach LAMMPS, so anything that would silently do nothing raises
instead. `close()` ends the session and does NOT destroy the retained
structure — only the residency actor may do that (`release()`), because it is
the thing that has to answer to the ledger for the GB.
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import sys

SOCKET_ENV = "TANDEM_LAMMPS_SOCKET"
STRICT_ENV = "TANDEM_LAMMPS_STRICT"
MODE_ENV = "TANDEM_LAMMPS_FILE_MODE"     # stream (default) | native

_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))
_real_module = None
_real_error = None


# --------------------------------------------------------------------------
# Reaching the REAL lammps package from a module that has stolen its name
# --------------------------------------------------------------------------

def _real() -> object:
    """Import the genuine `lammps` package with this shim taken off sys.path.

    sys.modules['lammps'] currently points at THIS module, so it is removed for
    the duration of the import and put back afterwards. Every submodule the
    real package imports resolves normally in between, because import_module
    installs the real package under 'lammps' before executing its body.
    """
    global _real_module, _real_error
    if _real_module is not None or _real_error is not None:
        if _real_error is not None:
            raise _real_error
        return _real_module
    saved_path = list(sys.path)
    saved_mod = sys.modules.get("lammps")
    try:
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or ".") != _SHIM_DIR]
        sys.modules.pop("lammps", None)
        _real_module = importlib.import_module("lammps")
        return _real_module
    except Exception as e:                          # noqa: BLE001
        _real_error = e
        raise
    finally:
        sys.path[:] = saved_path
        if saved_mod is not None:
            sys.modules["lammps"] = saved_mod


# Names that must NOT be forwarded: asking the real package for them during
# its own import would recurse, and Python probes several of them on any
# module. `__version__` is deliberately NOT in this list -- see below.
_NO_FORWARD = frozenset({
    "__file__", "__loader__", "__spec__", "__package__", "__builtins__",
    "__name__", "__doc__", "__dict__", "__cached__",
})
_forwarding = False


def __getattr__(name: str):
    """PEP 562: anything this shim does not define comes from the real module.

    Keeps `from lammps import LMP_STYLE_GLOBAL`, `lammps.formats`, and the rest
    of the module surface working without enumerating it here.

    DUNDERS ARE FORWARDED TOO, AND THAT IS NOT COSMETIC. `lammps/core.py` does
    `import lammps; if lammps.__version__ > 0 ...` inside the constructor, and
    sys.modules['lammps'] is THIS module — so refusing `__version__` made every
    FALLBACK construction raise AttributeError. The fallback is the path that
    guarantees dropping this shim on PYTHONPATH cannot break a run, so a broken
    fallback would have been a very quiet way to break the baseline.
    """
    global _forwarding
    if name in _NO_FORWARD or _forwarding:
        raise AttributeError(name)
    _forwarding = True
    try:
        return getattr(_real(), name)
    except Exception as e:                          # noqa: BLE001
        raise AttributeError(f"{name} (real lammps unavailable: {e})") from e
    finally:
        _forwarding = False


# --------------------------------------------------------------------------
# Wire
# --------------------------------------------------------------------------

class WorkerUnavailable(RuntimeError):
    pass


class _Wire:
    def __init__(self, path: str, timeout_s: float = 86400.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(path)
        self._buf = b""
        self._id = 0

    def call(self, op: str, **args):
        self._id += 1
        self.sock.sendall(
            json.dumps({"id": self._id, "op": op, "args": args}).encode() + b"\n")
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise WorkerUnavailable(
                    f"worker closed the connection during op {op!r}")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        rep = json.loads(line)
        if not rep.get("ok"):
            raise RuntimeError(f"{rep.get('etype', 'Error')}: {rep.get('error')}")
        return rep.get("result")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------
# numpy sub-object
# --------------------------------------------------------------------------

class _NumpyProxy:
    """Mirrors `lmp.numpy`. Returns COPIES — see the module docstring."""

    def __init__(self, owner: "lammps"):
        self._o = owner

    @staticmethod
    def _arr(x):
        if x is None:
            return None
        import numpy as np
        return np.asarray(x)

    def extract_atom(self, name, *a, **kw):
        return self._arr(self._o._w.call("extract_atom", name=name))

    def extract_compute(self, cid, cstyle, ctype, *a, **kw):
        return self._arr(self._o._w.call("extract_compute", cid=cid,
                                         cstyle=int(cstyle), ctype=int(ctype)))

    def extract_fix(self, fid, fstyle, ftype, nrow=0, ncol=0, *a, **kw):
        return self._arr(self._o._w.call("extract_fix", fid=fid,
                                         fstyle=int(fstyle), ftype=int(ftype),
                                         nrow=int(nrow), ncol=int(ncol)))

    def __getattr__(self, name):
        raise NotImplementedError(
            f"lammps.numpy.{name}() is not forwarded by the Tandem shim. Add it "
            f"to runtime/residency/data_worker.py rather than letting it fall "
            f"through to something that would silently differ.")


# --------------------------------------------------------------------------
# The proxy
# --------------------------------------------------------------------------

class lammps:
    """Same constructor signature as the real `lammps.lammps`."""

    def __new__(cls, name="", cmdargs=None, ptr=None, comm=None):
        sock = os.environ.get(SOCKET_ENV)
        strict = os.environ.get(STRICT_ENV) == "1"
        if not sock:
            if strict:
                raise WorkerUnavailable(
                    f"{STRICT_ENV}=1 but {SOCKET_ENV} is unset — refusing to "
                    f"fall back to a fresh parse, which would look like a fast "
                    f"worker in a timing run.")
            return _real().lammps(name=name, cmdargs=cmdargs, ptr=ptr, comm=comm)
        try:
            wire = _Wire(sock)
        except OSError as e:
            if strict:
                raise WorkerUnavailable(f"cannot reach worker at {sock}: {e}") from e
            return _real().lammps(name=name, cmdargs=cmdargs, ptr=ptr, comm=comm)
        self = super().__new__(cls)
        self._w = wire
        return self

    def __init__(self, name="", cmdargs=None, ptr=None, comm=None):
        if not hasattr(self, "_w"):                 # fell back; never reached
            return
        self._closed = False
        self._file_mode = os.environ.get(MODE_ENV, "stream")
        self.numpy = _NumpyProxy(self)
        # The worker is long-lived and shared, so it must adopt THIS script's
        # working directory: generated scripts use relative paths and
        # `shell cd`. The worker restores its own cwd when the session ends.
        self._w.call("chdir", path=os.getcwd())
        args = list(cmdargs or [])
        in_file = None
        for i, tok in enumerate(args):
            if tok == "-log" and i + 1 < len(args):
                self._w.call("command", cmd=f"log {args[i + 1]}")
            elif tok in ("-in", "-i") and i + 1 < len(args):
                in_file = args[i + 1]
        # The real API runs a `-in` file during construction. Four of the
        # AtomAgents runners rely on that, so it has to happen here too.
        if in_file:
            self.file(in_file)

    # -- command surface --------------------------------------------------
    def command(self, cmd):
        self._w.call("command", cmd=cmd)

    def commands_string(self, multicmd):
        self._w.call("commands_string", multicmd=multicmd)

    def commands_list(self, cmdlist):
        self._w.call("commands_list", cmdlist=list(cmdlist))

    def file(self, path):
        self._w.call("file", path=path, mode=self._file_mode)

    # -- readback ---------------------------------------------------------
    def get_thermo(self, name):
        return self._w.call("get_thermo", name=name)

    def get_natoms(self):
        return self._w.call("get_natoms")

    def version(self):
        return self._w.call("version")

    def extract_global(self, name, dtype=None):
        return self._w.call("extract_global", name=name)

    def extract_setting(self, name):
        return self._w.call("extract_setting", name=name)

    def extract_box(self):
        return self._w.call("extract_box")

    def extract_variable(self, name, group=None, vartype=None):
        return self._w.call("extract_variable", name=name, group=group,
                            vartype=vartype)

    def extract_atom(self, name, dtype=None):
        return self._w.call("extract_atom", name=name)

    def extract_compute(self, cid, cstyle, ctype):
        return self._w.call("extract_compute", cid=cid, cstyle=int(cstyle),
                            ctype=int(ctype))

    def extract_fix(self, fid, fstyle, ftype, nrow=0, ncol=0):
        return self._w.call("extract_fix", fid=fid, fstyle=int(fstyle),
                            ftype=int(ftype), nrow=int(nrow), ncol=int(ncol))

    # -- Tandem extras ----------------------------------------------------
    def worker_stats(self):
        return self._w.call("stats")

    def worker_ping(self):
        return self._w.call("ping")

    # -- teardown ---------------------------------------------------------
    def close(self):
        """END THE SESSION. Does NOT drop the retained structure.

        The real `close()` destroys the instance; here that would hand eviction
        to LLM-generated code and take the decision away from the ledger, which
        is the component that has to answer for the GB. Only
        LammpsDataWorker.release() may destroy it.
        """
        if not self._closed:
            self._closed = True
            self._w.close()

    finalize = close

    def __del__(self):
        try:
            self.close()
        except Exception:                           # noqa: BLE001
            pass

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"lammps.{name}() is not forwarded by the Tandem shim. Add the op to "
            f"runtime/residency/data_worker.py. Falling through to a local "
            f"instance here would compute against a DIFFERENT simulation than "
            f"the commands just sent to the worker.")
