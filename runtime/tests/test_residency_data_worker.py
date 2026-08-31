"""test_residency_data_worker.py — T4b, the resident evictable data worker.

Two layers.

OFFLINE (no LAMMPS, no potential, runs anywhere): the measurement primitives,
the wire framing, the script expander, and the command-rewriting rules —
including the two that only a RESIDENT worker needs and that no existing test
covers: `quit`, which terminates the process and appears at the end of nine of
the script builders, and `clear`, which destroys the pair tables that are the
whole point of retaining.

INTEGRATION (skipped unless `lammps` imports): a real worker process, staged
against a tiny generated setfl, exercised through the real client shim in a
real subprocess. This layer asserts the four things that would make T4b a lie
if they were untrue —

    D1  a second call reuses the activated structure (parses_avoided > 0);
    D2  release() frees what it says, the /proc entry is gone, and the next
        use pays a cold cost again;
    D4  pe is bit-identical to the fork path;
    +   a segfaulting client does not take the retained structure with it.

The integration numbers here are NOT the reported measurements — the artifact
is tiny so the suite stays fast. `experiments/bench_residency_data_worker.py`
produces the numbers, on a named node.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.residency.contract import (      # noqa: E402
    ReleaseNotHonoured, ResourceClass, ResidencyActor, ResourceSpec, Rung,
)
from runtime.residency import data_worker as dw    # noqa: E402


try:
    import lammps as _lammps                       # noqa: F401
    HAVE_LAMMPS = True
except Exception:                                  # noqa: BLE001
    HAVE_LAMMPS = False

needs_lammps = pytest.mark.skipif(
    not HAVE_LAMMPS, reason="lammps not importable in this interpreter")


# =========================================================== measurement ===
class TestMeasurement:
    """Invariant I1: held_gb is measured, and measured from the right column."""

    def test_private_dirty_is_the_column_used(self):
        m = dw.read_smaps_rollup(os.getpid())
        assert "Private_Dirty" in m and "Rss" in m
        assert dw.measure_private_dirty_gb(os.getpid()) == m["Private_Dirty"]
        # The trap: Rss double-counts shared pages, so it is >= Private_Dirty
        # and using it would have made a failed copy-on-write look successful.
        assert m["Rss"] >= m["Private_Dirty"]

    def test_dead_pid_measures_zero_not_a_stale_number(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        assert dw.measure_private_dirty_gb(p.pid) == 0.0
        assert not dw.pid_alive(p.pid)

    def test_a_real_allocation_moves_private_dirty(self):
        """Negative control. If this does not move, nothing else here means
        anything — the same role `write_control` plays in the COW benchmark."""
        before = dw.measure_private_dirty_gb(os.getpid())
        blob = bytearray(200 * 1024 * 1024)
        for i in range(0, len(blob), 4096):
            blob[i] = 1                            # dirty every page
        after = dw.measure_private_dirty_gb(os.getpid())
        assert after - before > 0.15, (before, after)
        del blob

    def test_thread_count_readable(self):
        assert dw.thread_count() >= 1

    def test_cgroup_reading_is_optional_not_fatal(self):
        v = dw.cgroup_current_gb()
        assert v is None or v > 0


# ================================================================== wire ===
class TestWire:
    def test_roundtrip_and_clean_eof(self):
        a, b = socket.socketpair()
        ca, cb = dw.Connection(a), dw.Connection(b)
        ca.send({"id": 1, "op": "ping"})
        assert cb.recv() == {"id": 1, "op": "ping"}
        ca.close()
        assert cb.recv() is None                   # EOF, not an exception

    def test_multiple_messages_in_one_packet(self):
        a, b = socket.socketpair()
        ca, cb = dw.Connection(a), dw.Connection(b)
        a.sendall(b'{"id":1}\n{"id":2}\n')
        assert cb.recv()["id"] == 1
        assert cb.recv()["id"] == 2
        ca.close(); cb.close()

    def test_peer_reset_reads_as_eof(self):
        """A segfaulting client resets the socket. The worker must read that as
        end-of-stream, not raise into a path that would tear down the state."""
        a, b = socket.socketpair()
        cb = dw.Connection(b)
        a.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                     __import__("struct").pack("ii", 1, 0))
        a.close()
        assert cb.recv() is None


# ====================================================== script expansion ===
class TestExpand:
    def test_comments_continuations_and_includes(self, tmp_path):
        (tmp_path / "pot.inp").write_text(
            "pair_style eam/fs\npair_coeff * * w.fs W\n")
        (tmp_path / "in.script").write_text(
            "# leading comment\n"
            "units metal   # trailing comment\n"
            "\n"
            "region box &\n"
            "  block 0 1 0 1 0 1\n"
            f"include {tmp_path / 'pot.inp'}\n"
            "run 0\n")
        cmds = dw.expand_input_script(str(tmp_path / "in.script"))
        assert cmds == ["units metal", "region box block 0 1 0 1 0 1",
                        "pair_style eam/fs", "pair_coeff * * w.fs W", "run 0"]

    def test_include_is_expanded_because_pair_coeff_hides_there(self, tmp_path):
        """The AtomAgents scripts put pair_style/pair_coeff in an INCLUDED
        potential.inp. Not expanding it would miss the one command the entire
        saving turns on, and the miss would be silent."""
        (tmp_path / "potential.inp").write_text("pair_coeff * * big.fs W\n")
        (tmp_path / "s.in").write_text("include potential.inp\n")
        cmds = dw.expand_input_script(str(tmp_path / "s.in"))
        assert any(c.startswith("pair_coeff") for c in cmds)

    def test_control_flow_is_refused_not_guessed(self, tmp_path):
        (tmp_path / "s.in").write_text("units metal\njump SELF loop\n")
        with pytest.raises(RuntimeError, match="control-flow"):
            dw.expand_input_script(str(tmp_path / "s.in"))


# ========================================================= rewrite rules ===
class _FakeLmp:
    def __init__(self):
        self.issued = []
    def command(self, c):
        self.issued.append(c)


class _FakeRetained:
    """Stands in for RetainedInstance: records what the rewriter passes down."""
    def __init__(self, lmp, coeff_loaded=True):
        self.lmp = lmp
        self._coeff_loaded = coeff_loaded
        self.parses_avoided = 0
        self.parses_paid = 0
        self.seen = []
    def command(self, c):
        self.seen.append(c)
        self.lmp.command(c)


class TestRewriter:
    def test_quit_is_suppressed(self):
        """`quit` terminates the PROCESS. Un-intercepted, the first tool call
        kills the worker and takes the whole table with it."""
        lmp = _FakeLmp()
        rw = dw.ScriptRewriter(_FakeRetained(lmp))
        rw.command("run 0")
        rw.command("quit")
        assert rw.quits_suppressed == 1
        assert "quit" not in lmp.issued

    def test_clear_on_a_retained_instance_becomes_a_soft_clear(self):
        lmp = _FakeLmp()
        r = _FakeRetained(lmp, coeff_loaded=True)
        rw = dw.ScriptRewriter(r)
        for c in ("compute peratom all pe/atom", "fix 1 all box/relax aniso 0",
                  "dump d1 all custom 1 dump.x id type", "variable p1 equal step"):
            rw.command(c)
        lmp.issued.clear()
        rw.command("clear")
        assert rw.clears_softened == 1
        # every installed id is torn down, so the next script does not inherit
        # the previous one's fixes -- which WOULD change the physics
        assert "undump d1" in lmp.issued
        assert "unfix 1" in lmp.issued
        assert "uncompute peratom" in lmp.issued
        assert "variable p1 delete" in lmp.issued
        assert "delete_atoms group all" in lmp.issued
        # and the pair tables are NOT dropped: no `clear` reaches LAMMPS
        assert "clear" not in lmp.issued

    def test_first_script_clear_passes_through(self):
        """Before anything is loaded the rewrite would be wrong. This is the
        same class of bug persistent_lammps.py already hit once, where the
        first create_box was rewritten into delete_atoms on an empty box."""
        lmp = _FakeLmp()
        rw = dw.ScriptRewriter(_FakeRetained(lmp, coeff_loaded=False))
        rw.command("clear")
        assert lmp.issued == ["clear"]
        assert rw.clears_softened == 0

    def test_identical_units_is_skipped_not_re_issued(self):
        """LAMMPS errors on `units` once a box exists, and every generated
        script opens with one. Skipping an IDENTICAL setting is safe."""
        lmp = _FakeLmp()
        rw = dw.ScriptRewriter(_FakeRetained(lmp),
                               fixed_settings={"units": "metal",
                                               "atom_style": "atomic"})
        rw.command("units metal")
        rw.command("atom_style atomic")
        assert lmp.issued == []
        assert rw.settings_skipped == 2

    def test_a_DIFFERENT_unit_system_is_refused_loudly(self):
        """The failure this rule exists for. Dropping a differing `units` runs
        the next script in the wrong unit system and still returns plausible
        numbers -- measured at a 70.8% pe difference before the rule existed."""
        lmp = _FakeLmp()
        rw = dw.ScriptRewriter(_FakeRetained(lmp), fixed_settings={"units": "metal"})
        with pytest.raises(RuntimeError, match="fork path"):
            rw.command("units real")
        assert lmp.issued == []

    def test_unfix_removes_from_the_teardown_list(self):
        lmp = _FakeLmp()
        rw = dw.ScriptRewriter(_FakeRetained(lmp))
        rw.command("fix 1 all nve")
        rw.command("unfix 1")
        lmp.issued.clear()
        rw.command("clear")
        assert "unfix 1" not in lmp.issued      # would be an error in LAMMPS


# ================================================================== D3 ====
class TestRegistration:
    def test_the_artifact_is_registered_and_priced(self):
        specs, arts = dw.load_residency_artifacts()
        assert "w_eam4_big_activated" in specs, \
            "D3: the activated potential is not registered in tool_resources.json"
        s = specs["w_eam4_big_activated"]
        assert s.resource_class is ResourceClass.DATA_PATTERN_B
        assert s.held_rung is Rung.R3_ACTIVATED
        assert s.cold_s > s.ready_s                # __post_init__ enforces it
        assert arts["w_eam4_big_activated"]["potential"].endswith("w_eam4_big.fs")

    def test_density_is_the_2_25_on_record(self):
        s = dw.load_residency_artifacts()[0]["w_eam4_big_activated"]
        assert abs(s.static_density - 2.25) < 0.01

    def test_it_is_invisible_to_the_byte_oriented_registry(self):
        """The residency block must not change what the prefetch registry sees.
        An R3 activated structure is not a file range; a prefetcher cannot
        stage it, and a stray entry would alter every existing eval trace."""
        from runtime.predictor.resource_registry import ResourceRegistry
        reg = ResourceRegistry.from_json()
        for tool in reg.all_tools():
            for spec in reg.get(tool):
                assert spec.name != "w_eam4_big_activated"


# ============================================================ the actor ====
class TestActorContract:
    def test_it_satisfies_the_protocol(self):
        a = dw.LammpsDataWorker(artifacts={})
        assert isinstance(a, ResidencyActor)
        assert a.resource_class is ResourceClass.DATA_PATTERN_B

    def test_unregistered_resource_is_a_loud_keyerror(self):
        a = dw.LammpsDataWorker(artifacts={})
        spec = ResourceSpec("nope", ResourceClass.DATA_PATTERN_B,
                            Rung.R3_ACTIVATED, 1.0, 10.0, 1.0)
        with pytest.raises(KeyError, match="tool_resources.json"):
            a.stage(spec)

    def test_nothing_held_measures_zero(self):
        a = dw.LammpsDataWorker(artifacts={})
        assert a.measure_held_gb("nope") == 0.0
        assert a.is_resident("nope") is False
        assert a.release("nope") == 0.0

    def test_release_witness_is_required_by_the_protocol(self):
        """The method is structurally mandatory on a @runtime_checkable
        Protocol no matter what its docstring calls it -- describing it as
        optional is what broke every actor's isinstance() check."""
        assert hasattr(dw.LammpsDataWorker, "release_witness")
        assert isinstance(dw.LammpsDataWorker(artifacts={}), ResidencyActor)

    def test_no_release_means_no_witness(self):
        a = dw.LammpsDataWorker(artifacts={})
        assert a.release_witness("nope") is None

    def test_a_later_release_invalidates_an_earlier_witness(self):
        """THE STALENESS GUARD. The cached detail outlives its release; handing
        it back later would vouch for a release it never observed, which is the
        same tautology release_witness() exists to break."""
        a = dw.LammpsDataWorker(artifacts={})
        a._release_seq = 1
        a.last_release_detail["A"] = {"release_seq": 1, "cgroup_anon_freed_gb": 12.0}
        assert a.release_witness("A") == 12.0
        # something else is released: the cgroup has moved since A's samples
        a._release_seq = 2
        a.last_release_detail["B"] = {"release_seq": 2, "cgroup_anon_freed_gb": 3.0}
        assert a.release_witness("A") is None
        assert a.release_witness("B") == 3.0

    def test_restaging_invalidates_the_witness(self):
        """A resident resource has no outstanding release to describe."""
        a = dw.LammpsDataWorker(artifacts={})
        a._release_seq = 1
        a.last_release_detail["A"] = {"release_seq": 1, "cgroup_anon_freed_gb": 12.0}
        a._held["A"] = object()                    # as if re-staged
        assert a.release_witness("A") is None

    def test_an_unreadable_cgroup_is_a_none_not_a_zero(self):
        a = dw.LammpsDataWorker(artifacts={})
        a._release_seq = 1
        a.last_release_detail["A"] = {"release_seq": 1, "cgroup_anon_freed_gb": None}
        assert a.release_witness("A") is None

    def test_a_negative_witness_is_reported_not_clamped(self):
        """Contrary evidence must not be laundered into 'no evidence'."""
        a = dw.LammpsDataWorker(artifacts={})
        a._release_seq = 1
        a.last_release_detail["A"] = {"release_seq": 1, "cgroup_anon_freed_gb": -0.4}
        assert a.release_witness("A") == -0.4

    def test_the_witness_column_is_anon_not_memory_current(self):
        """memory.current counts page cache; releasing a worker that had just
        read a 3.32 GB potential would credit the cache drop to the release."""
        anon, cur = dw.cgroup_anon_gb(), dw.cgroup_current_gb()
        if anon is None or cur is None:
            pytest.skip("no cgroup v2 memory controller here")
        assert anon <= cur

    def test_thread_pins_are_set_before_the_child_interpreter_starts(self):
        """Pattern D. Measured on this cluster: 193 threads by default from
        OpenMP/MKL pools, 3 with OMP_NUM_THREADS=1."""
        assert dw._THREAD_PINS["OMP_NUM_THREADS"] == "1"
        assert set(dw._THREAD_PINS) >= {"OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                        "OPENBLAS_NUM_THREADS"}

    def test_shim_dir_goes_first_on_pythonpath(self):
        env = dw.prepend_shim_path({"PYTHONPATH": "/already/here"})
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(dw.SHIM_DIR)
        assert "/already/here" in env["PYTHONPATH"]
        assert (dw.SHIM_DIR / "lammps.py").exists()
        assert not (dw.SHIM_DIR / "__init__.py").exists(), \
            "the shim directory is a sys.path entry, not a package"


class TestRunnerIntegration:
    """The framework-side change is one env var. Prove it is exactly that."""

    def _runner(self):
        sys.path.insert(0, str(ROOT / "workloads" / "AtomAgents"))
        from atomagents.execution import runner
        return runner

    def test_baseline_env_is_untouched(self, monkeypatch):
        r = self._runner()
        monkeypatch.delenv("TANDEM_LAMMPS_SOCKET", raising=False)
        assert r._tool_env() is None, \
            "with no worker the subprocess must inherit the environment " \
            "unchanged, so the baseline arm is byte-for-byte what it was"

    def test_socket_set_prepends_the_shim(self, monkeypatch):
        r = self._runner()
        monkeypatch.setenv("TANDEM_LAMMPS_SOCKET", "/tmp/does-not-exist.sock")
        env = r._tool_env()
        assert env is not None
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(dw.SHIM_DIR)


# ========================================================= integration ====
def _tiny_setfl(path: Path, n: int = 20000) -> str:
    sys.path.insert(0, str(ROOT / "experiments"))
    from bench_residency_data_worker import make_setfl
    return make_setfl(str(path), n)


@needs_lammps
class TestLiveWorker:
    """A real worker, a real client subprocess, a real socket."""

    @pytest.fixture(scope="class")
    def staged(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("tandem")
        pot = _tiny_setfl(tmp / "tiny.eam.alloy")
        spec = ResourceSpec("tiny", ResourceClass.DATA_PATTERN_B,
                            Rung.R3_ACTIVATED, 0.01, 10.0, 0.1)
        actor = dw.LammpsDataWorker(
            artifacts={"tiny": {"potential": pot, "pair_style": "eam/alloy",
                                "setup": ["units metal", "boundary p p p",
                                          "atom_style atomic",
                                          "lattice bcc 3.165",
                                          "region box block 0 2 0 2 0 2",
                                          "create_box 1 box",
                                          "create_atoms 1 box",
                                          "mass 1 183.84"]}},
            socket_dir=str(tmp),
            atomagents_root=str(ROOT / "workloads" / "AtomAgents"))
        rung = actor.stage(spec)
        yield actor, spec, rung, tmp, pot
        actor.release_all()

    def test_stage_reaches_r3_with_pinned_threads(self, staged):
        actor, spec, rung, tmp, pot = staged
        assert rung is Rung.R3_ACTIVATED
        d = actor.activation_detail("tiny")
        # Pattern D: 193 threads was the measured default. Anything near that
        # means the pins did not land before `import lammps`.
        assert 0 < d["threads_at_start"] <= dw.DEFAULT_MAX_THREADS, d
        assert actor.measure_held_gb("tiny") > 0
        assert actor.is_resident("tiny")

    def test_measure_held_gb_is_private_dirty_not_rss(self, staged):
        actor = staged[0]
        pid = actor.activation_detail("tiny")["pid"]
        m = dw.read_smaps_rollup(pid)
        assert actor.measure_held_gb("tiny") == pytest.approx(m["Private_Dirty"],
                                                             rel=0.2)

    def _client(self, actor, tmp, name, body):
        script = tmp / name
        script.write_text(body)
        env = os.environ.copy()
        env["TANDEM_LAMMPS_SOCKET"] = actor.socket_for("tiny")
        env["TANDEM_LAMMPS_STRICT"] = "1"      # never silently re-parse
        env["PYTHONPATH"] = str(dw.SHIM_DIR)
        return subprocess.run([sys.executable, str(script)], env=env,
                              cwd=str(tmp), capture_output=True, text=True)

    def test_D1_second_call_reuses_the_activated_structure(self, staged):
        actor, spec, rung, tmp, pot = staged
        body = (
            "import json\n"
            "from lammps import lammps\n"
            "l = lammps(cmdargs=['-log','none','-screen','none'])\n"
            "for c in ['clear','units metal','boundary p p p','atom_style atomic',\n"
            "          'lattice bcc 3.165','region box block 0 2 0 2 0 2',\n"
            "          'create_box 1 box','create_atoms 1 box','mass 1 183.84',\n"
            f"          'pair_style eam/alloy','pair_coeff * * {pot} W','run 0','quit']:\n"
            "    l.command(c)\n"
            "print(json.dumps({'pe': l.get_thermo('pe'), 's': l.worker_stats()}))\n")
        r = self._client(actor, tmp, "d1.py", body)
        assert r.returncode == 0, r.stderr[-3000:]
        out = json.loads(r.stdout.strip().splitlines()[-1])
        # THE saving: an identical pair_coeff was skipped rather than re-parsed.
        assert out["s"]["rewriter"]["parses_avoided"] >= 1, out["s"]
        assert out["s"]["quits_suppressed"] >= 1
        assert out["pe"] is not None

    def test_D4_pe_is_bit_identical_to_the_fork_path(self, staged):
        actor, spec, rung, tmp, pot = staged
        # The header matters. Without `units metal` the FORK path defaults to
        # lj units while the retained instance keeps the metal units it was
        # built with -- measured at a 70.8% pe difference, which is what
        # surfaced the missing `units` rule in ScriptRewriter. Both paths get
        # the identical stream, exactly as a generated script emits it.
        cmds = ["units metal", "boundary p p p", "atom_style atomic",
                "lattice bcc 3.165 orient x 1 -1 2 orient y 1 1 0 orient z -1 1 1",
                "region box block 0 3 0 3 0 2", "create_box 1 box",
                "create_atoms 1 box", "mass 1 183.84",
                "neigh_modify one 10000 page 1000000",
                "pair_style eam/alloy", f"pair_coeff * * {pot} W", "run 0"]
        body = ("import json\nfrom lammps import lammps\n"
                "l = lammps(cmdargs=['-log','none','-screen','none'])\n"
                f"for c in {cmds!r}:\n    l.command(c)\n"
                "print(json.dumps({'pe': l.get_thermo('pe'), 'n': l.get_natoms()}))\n")
        r = self._client(actor, tmp, "d4.py", body)
        assert r.returncode == 0, r.stderr[-3000:]
        retained = json.loads(r.stdout.strip().splitlines()[-1])

        fork_src = ("import json\nfrom lammps import lammps\n"
                    "l = lammps(cmdargs=['-log','none','-screen','none'])\n"
                    f"for c in {cmds!r}:\n    l.command(c)\n"
                    "print(json.dumps({'pe': l.get_thermo('pe'), 'n': l.get_natoms()}))\n")
        (tmp / "fork.py").write_text(fork_src)
        env = {k: v for k, v in os.environ.items() if k != "TANDEM_LAMMPS_SOCKET"}
        fr = subprocess.run([sys.executable, str(tmp / "fork.py")], env=env,
                            cwd=str(tmp), capture_output=True, text=True)
        assert fr.returncode == 0, fr.stderr[-3000:]
        fork = json.loads(fr.stdout.strip().splitlines()[-1])

        assert retained["n"] == fork["n"]
        rel = abs(retained["pe"] - fork["pe"]) / abs(fork["pe"])
        assert rel == 0.0, f"retention changed the physics: rel diff {rel:.3e}"

    def test_a_segfaulting_client_does_not_take_the_structure_with_it(self, staged):
        actor, spec, rung, tmp, pot = staged
        before = actor.call("tiny", "get_thermo", {"name": "pe"})
        r = self._client(actor, tmp, "boom.py",
                         "import ctypes\n"
                         "from lammps import lammps\n"
                         "l = lammps(cmdargs=['-log','none','-screen','none'])\n"
                         "l.command('run 0')\n"
                         "ctypes.string_at(1)\n")
        assert r.returncode < 0, "the client was expected to die on a signal"
        time.sleep(0.3)
        assert actor.is_resident("tiny"), "the worker died with its client"
        assert actor.call("tiny", "get_thermo", {"name": "pe"}) == before

    def test_client_falls_back_when_no_socket_is_set(self, staged):
        """Dropping the shim on PYTHONPATH must not be able to break a run."""
        actor, spec, rung, tmp, pot = staged
        (tmp / "fb.py").write_text(
            "from lammps import lammps\n"
            "l = lammps(cmdargs=['-log','none','-screen','none'])\n"
            "l.command('units metal')\n"
            "print('REAL', type(l).__module__)\n")
        env = {k: v for k, v in os.environ.items() if k != "TANDEM_LAMMPS_SOCKET"}
        env["PYTHONPATH"] = str(dw.SHIM_DIR)
        r = subprocess.run([sys.executable, str(tmp / "fb.py")], env=env,
                           cwd=str(tmp), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert "REAL" in r.stdout

    def test_strict_mode_refuses_to_fall_back(self, staged):
        """A silent fallback in a timing run would be a fresh parse reported as
        a fast worker. That is the exact shape of this project's withdrawn
        measurements, so strict mode has to fail instead."""
        actor, spec, rung, tmp, pot = staged
        (tmp / "strict.py").write_text(
            "from lammps import lammps\nlammps()\n")
        env = {k: v for k, v in os.environ.items() if k != "TANDEM_LAMMPS_SOCKET"}
        env["PYTHONPATH"] = str(dw.SHIM_DIR)
        env["TANDEM_LAMMPS_STRICT"] = "1"
        r = subprocess.run([sys.executable, str(tmp / "strict.py")], env=env,
                           cwd=str(tmp), capture_output=True, text=True)
        assert r.returncode != 0
        assert "TANDEM_LAMMPS_SOCKET" in r.stderr


@needs_lammps
def test_D2_release_is_real(tmp_path):
    """The item most likely to fail silently, checked four ways.

    A worker that retains but cannot truly release makes the budget a fiction
    and every downstream percentage meaningless (invariant I2). So: the
    measured free, the vanished /proc entry, the cgroup's independent opinion,
    and — the one that cannot be faked — the next use paying cold again.
    """
    pot = _tiny_setfl(tmp_path / "tiny.eam.alloy")
    spec = ResourceSpec("tiny", ResourceClass.DATA_PATTERN_B,
                        Rung.R3_ACTIVATED, 0.01, 10.0, 0.1)
    actor = dw.LammpsDataWorker(
        artifacts={"tiny": {"potential": pot, "pair_style": "eam/alloy"}},
        socket_dir=str(tmp_path))
    assert actor.stage(spec) is Rung.R3_ACTIVATED
    pid = actor.activation_detail("tiny")["pid"]
    held = actor.measure_held_gb("tiny")
    assert held > 0.01

    reuse_s = _time_call(actor)
    freed = actor.release("tiny")

    # The witness: an account that does not depend on the released process.
    witness = actor.release_witness("tiny")
    d = actor.last_release_detail["tiny"]
    if d["cgroup_anon_freed_gb"] is not None:
        assert witness == d["cgroup_anon_freed_gb"]
        assert witness > 0, (
            "the enclosing cgroup's anon total did not fall across the "
            "release; the teardown freed nothing the job can see")
    else:
        assert witness is None      # honest "no witness", never a fake zero
    assert freed == pytest.approx(held, rel=0.05)
    assert d["proc_gone"] and not dw.pid_alive(pid)
    assert actor.measure_held_gb("tiny") == 0.0
    assert not actor.is_resident("tiny")
    assert d["measured_by"].startswith("Private_Dirty")
    if d["cgroup_freed_gb"] is not None and held > 0.05:
        # Independent witness. Noisy on a shared node, so it corroborates
        # rather than gates -- but it is recorded either way.
        assert d["cgroup_freed_gb"] > 0

    # THE BEHAVIOURAL PROOF. If the release had freed nothing while reporting
    # success, re-staging would be cheap. It must not be.
    t0 = time.perf_counter()
    assert actor.stage(spec) is Rung.R3_ACTIVATED
    restage_s = time.perf_counter() - t0
    assert actor.release_witness("tiny") is None, \
        "a re-staged resource still handed back its old release witness"
    assert restage_s > reuse_s, (
        f"re-staging after release took {restage_s:.3f}s but a live reuse took "
        f"{reuse_s:.3f}s — the release did not actually evict anything")
    actor.release_all()


def _time_call(actor) -> float:
    t0 = time.perf_counter()
    actor.call("tiny", "command", {"cmd": "run 0"})
    return time.perf_counter() - t0


# =================================== the real framework path ==============
_GENERATED_IN = """\
        clear
        units metal
        dimension 3
        boundary p p p
        atom_style atomic

        shell cd ./{folder}

        variable thermo_time equal 10
        variable potential_file string "potential.inp"

        lattice bcc 3.165
        region box block 0 {nx} 0 {nx} 0 {nz}
        create_box 1 box
        create_atoms 1 box
        mass 1 183.84
        neigh_modify one 10000 page 1000000

        include ${{potential_file}}

        compute peratom all pe/atom
        compute pe all pe
        thermo_style custom step pe vol lx ly lz
        thermo ${{thermo_time}}

        fix relax all box/relax aniso 0
        run 0
        unfix relax

        variable p1 equal "pe"
        print "PE ${{p1}}" file {out}
        quit
"""

_GENERATED_RUNNER = """\
from lammps import lammps
lmp = lammps(cmdargs=['-log', './{folder}/log.lammps', '-screen', 'none'])
lmp.file('{folder}/lmp_script.in')
lmp.command('quit')
lmp.close()
"""


@needs_lammps
def test_end_to_end_through_the_real_LammpsRunner(tmp_path):
    """The whole T4b path, driven by the framework's own runner.

    This is the integration the three pieces exist for: LammpsRunner.run_python
    -> subprocess with the shim first on PYTHONPATH -> `from lammps import
    lammps` resolves to the shim -> lmp.file() is STREAMED to the worker, which
    expands the `include`, softens the `clear`, suppresses the `quit`, and skips
    the identical pair_coeff.

    The script here is shaped exactly like script_builders.py output -- the
    same `clear` / `units` / `shell cd` / `include potential.inp` / fix / `quit`
    sequence -- because every one of those is a command that a naive proxy gets
    wrong, and four of them would get it wrong SILENTLY.
    """
    sys.path.insert(0, str(ROOT / "workloads" / "AtomAgents"))
    from atomagents.execution.runner import LammpsRunner

    folder = "wf"
    (tmp_path / folder).mkdir()
    pot = _tiny_setfl(tmp_path / "tiny.eam.alloy")
    (tmp_path / folder / "potential.inp").write_text(
        f"pair_style eam/alloy\npair_coeff * * {pot} W\n")
    (tmp_path / folder / "lmp_script.in").write_text(
        _GENERATED_IN.format(folder=folder, nx=2, nz=2, out="pe.txt"))
    (tmp_path / "run_lammps.py").write_text(_GENERATED_RUNNER.format(folder=folder))

    spec = ResourceSpec("tiny", ResourceClass.DATA_PATTERN_B,
                        Rung.R3_ACTIVATED, 0.01, 10.0, 0.1)
    actor = dw.LammpsDataWorker(
        artifacts={"tiny": {"potential": pot, "pair_style": "eam/alloy",
                            "setup": ["units metal", "dimension 3",
                                      "boundary p p p", "atom_style atomic",
                                      "lattice bcc 3.165",
                                      "region box block 0 2 0 2 0 2",
                                      "create_box 1 box", "create_atoms 1 box",
                                      "mass 1 183.84"]}},
        socket_dir=str(tmp_path),
        atomagents_root=str(ROOT / "workloads" / "AtomAgents"))
    assert actor.stage(spec) is Rung.R3_ACTIVATED
    try:
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # No socket => the runner must leave the environment alone, and the
            # script runs against a real, freshly-parsed instance. Ground truth.
            os.environ.pop("TANDEM_LAMMPS_SOCKET", None)
            r_fork = LammpsRunner(num_procs=1).run_python("run_lammps.py")
            assert r_fork.returncode == 0, r_fork.stderr[-3000:]
            fork_pe = (tmp_path / folder / "pe.txt").read_text()
            (tmp_path / folder / "pe.txt").unlink()

            # Socket set => the SAME runner call goes to the worker.
            os.environ["TANDEM_LAMMPS_SOCKET"] = actor.socket_for("tiny")
            os.environ["TANDEM_LAMMPS_STRICT"] = "1"
            r_ret = LammpsRunner(num_procs=1).run_python("run_lammps.py")
            assert r_ret.returncode == 0, r_ret.stderr[-3000:]
            retained_pe = (tmp_path / folder / "pe.txt").read_text()
        finally:
            os.chdir(cwd)
            os.environ.pop("TANDEM_LAMMPS_SOCKET", None)
            os.environ.pop("TANDEM_LAMMPS_STRICT", None)

        assert retained_pe == fork_pe, (
            f"the generated script computed a different energy through the "
            f"worker: fork {fork_pe!r} vs retained {retained_pe!r}")

        st = actor.call("tiny", "stats")
        assert st["quits_suppressed"] >= 1, "a `quit` reached LAMMPS"
        assert st["rewriter"]["clears_softened"] >= 1, "a `clear` was not softened"
        assert st["rewriter"]["parses_avoided"] >= 1, \
            "the include's pair_coeff was re-parsed; there is no saving"
        assert st["rewriter"]["settings_skipped"] >= 3, \
            "units/dimension/boundary/atom_style were not handled"
        assert actor.is_resident("tiny"), "the worker did not survive the script"
    finally:
        actor.release_all()
