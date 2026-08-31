"""test_residency_model_actor.py — T4a, the model residency actor.

NO GPU IS USED OR REQUIRED. Everything here runs against a fake orchestrator
and a SYNTHETIC CGROUP written into tmp_path, so the real `memory.stat` parser
is exercised on real files. That matters: the two traps this actor exists to
avoid are both about which column is read, and a mock that returned a number
would test nothing about them.

What the fake cgroup lets us prove without hardware:

  * booting an engine reads ~146 GB of weight shards, which lands in `file`
    (page cache) and therefore in `memory.current`. If `measure_held_gb()` read
    `memory.current` it would report ~266 GB for a 120 GB park. The test asserts
    the difference explicitly, in both directions.
  * the park delta is taken against the AWAKE reading, so the engine's own
    baseline allocation is not charged to the park.
  * `measure_held_gb()` re-reads: moving the cgroup underneath it moves the
    answer. A cached number would pass a weaker test and fail this one.

The release proof follows A2's four-way bar (test_residency_data_worker.py
::test_D2_release_is_real): the measured free, the vanished /proc entry, an
independent witness, and the one that cannot be faked — the next use paying
cold again. The engines here are REAL subprocesses (a sleeping interpreter), so
the /proc half is real even though the weights are not.

The GPU-occupancy tests reproduce the actual failure first
("Cannot start qwen_32b: GPUs [0,1,2,3] occupied by qwen_72b") and then show
the actor clearing it — otherwise a passing eviction test proves only that the
fake is agreeable.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.residency.contract import (              # noqa: E402
    ReleaseNotHonoured, ResidencyActor, ResourceClass, ResourceSpec, Rung,
)
from runtime.residency import model_actor as ma       # noqa: E402


GIB = 1024 ** 3


# =========================================================== the fakes =====

class FakeCgroup:
    """A real cgroup v2 memory directory, written to disk.

    `anon` and `file` are separate on purpose: `memory.current` is defined here
    as anon+file, which is what makes the page-cache trap reproducible.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.anon_gib = 0.0
        self.file_gib = 0.0
        self.flush()

    def flush(self) -> None:
        anon = int(self.anon_gib * GIB)
        filed = int(self.file_gib * GIB)
        (self.path / "memory.stat").write_text(
            f"anon {anon}\nfile {filed}\nkernel_stack 1048576\nslab 2097152\n")
        (self.path / "memory.current").write_text(str(anon + filed) + "\n")

    def add_anon(self, gib: float) -> None:
        self.anon_gib = max(0.0, self.anon_gib + gib)
        self.flush()

    def add_file(self, gib: float) -> None:
        self.file_gib = max(0.0, self.file_gib + gib)
        self.flush()


class _Proc:
    """A real child process, so /proc and pid_alive are not simulated."""

    def __init__(self) -> None:
        self.p = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"])

    @property
    def pid(self) -> int:
        return self.p.pid

    def poll(self):
        return self.p.poll()

    def kill(self) -> None:
        self.p.kill()
        self.p.wait(timeout=10)


class FakeOrchestrator:
    """The orchestrator surface the actor uses, and nothing else.

    It reproduces the real refusal at model_orchestrator.py:594-598 verbatim,
    including the rule that a SLEPT engine does not block a start (its VRAM is
    free), so the eviction tests are testing against the real behaviour rather
    than a convenient one.
    """

    ERR = ("[orchestrator] Cannot start {name}: GPUs {conflict} occupied by "
           "{other}. Call stop_model first.")

    def __init__(self, models: dict, cg: FakeCgroup,
                 baseline_gib: float = 8.0,
                 park_gib: dict | None = None,
                 park_file_gib: dict | None = None,
                 shard_file_gib: dict | None = None,
                 backup_survives_wake: bool = False,
                 sleep_fails: set | None = None,
                 stop_really_kills: bool = True,
                 n_gpus: int = 6):
        self.models = models
        self.cg = cg
        self.baseline_gib = baseline_gib
        self.park_gib = park_gib or {}
        # Where the park LANDS. The fp8 run put it in `file`, not `anon`.
        self.park_file_gib = park_file_gib or {}
        self.shard_file_gib = shard_file_gib or {}
        self.backup_survives_wake = backup_survives_wake
        self.sleep_fails = sleep_fails or set()
        self.stop_really_kills = stop_really_kills
        self.n_gpus = n_gpus
        self.processes: dict[str, _Proc] = {}
        self.sleeping: set[str] = set()
        self.calls: list[tuple] = []
        self.boots: dict[str, int] = {}
        self._backed: set[str] = set()      # host-side backup allocated once

    # -- the bug, reproduced -------------------------------------------
    def _refuse_if_occupied(self, name: str) -> None:
        want = set(self.models[name].get("gpus", []))
        for other, proc in list(self.processes.items()):
            if other == name or proc.poll() is not None:
                continue
            if other in self.sleeping:
                continue                     # slept: VRAM already free
            conflict = want & set(self.models.get(other, {}).get("gpus", []))
            if conflict:
                raise RuntimeError(self.ERR.format(
                    name=name, conflict=sorted(conflict), other=other))

    # -- lifecycle ------------------------------------------------------
    def start_model_measured(self, name: str, metrics=None) -> float:
        self.calls.append(("start_model_measured", name))
        self._refuse_if_occupied(name)
        self.boots[name] = self.boots.get(name, 0) + 1
        self.processes[name] = _Proc()
        self.sleeping.discard(name)
        # Reading the weight shards fills the page cache; the engine's own
        # allocations are anonymous. Both move, and only one of them is ours.
        self.cg.add_file(self.shard_file_gib.get(name, 146.0))
        self.cg.add_anon(self.baseline_gib)
        return 1.0

    def stop_model(self, name: str, wait_s: float = 30.0) -> None:
        self.calls.append(("stop_model", name))
        proc = self.processes.get(name)
        if proc is None:
            return
        if name in self.sleeping and name in self._backed:
            self.cg.add_anon(-self.park_gib.get(name, 0.0))
        elif name in self._backed and self.backup_survives_wake:
            self.cg.add_anon(-self.park_gib.get(name, 0.0))
        self.cg.add_anon(-self.baseline_gib)
        self._backed.discard(name)
        self.sleeping.discard(name)
        if self.stop_really_kills:
            proc.kill()
            self.processes.pop(name, None)
        # stop_really_kills=False leaves the pid alive: the I2 failure case.

    def is_sleeping(self, name: str) -> bool:
        p = self.processes.get(name)
        return p is not None and p.poll() is None and name in self.sleeping

    def sleep_model(self, name: str, level: int = 1, timeout: float = 900.0) -> float:
        self.calls.append(("sleep_model", name, level))
        if name in self.sleep_fails:
            raise RuntimeError(f"HTTP 500 from /sleep for {name}")
        if name not in self.processes or name in self.sleeping:
            return 0.0
        first = name not in self._backed
        self._backed.add(name)
        self.sleeping.add(name)
        self.cg.add_anon(self.park_gib.get(name, 0.0))
        self.cg.add_file(self.park_file_gib.get(name, 0.0))
        return ma.FIRST_PARK_S if first else ma.SUBSEQ_PARK_S

    def wake_model(self, name: str, timeout: float = 900.0) -> float:
        self.calls.append(("wake_model", name))
        if name not in self.processes:
            raise RuntimeError(f"Cannot wake {name}: no live server process.")
        self.sleeping.discard(name)
        if not self.backup_survives_wake:
            self.cg.add_anon(-self.park_gib.get(name, 0.0))
            self._backed.discard(name)
        return 2.076

    def wait_until_serving(self, name: str, timeout: int = 60) -> None:
        self.calls.append(("wait_until_serving", name))

    def get_running_model(self):
        return next(iter(self.processes), None)

    def get_gpu_memory(self) -> list[dict]:
        total = 49140
        used = set()
        for name, proc in self.processes.items():
            if proc.poll() is None and name not in self.sleeping:
                used |= set(self.models.get(name, {}).get("gpus", []))
        return [{"total": total, "free": 1000 if g in used else total - 500}
                for g in range(self.n_gpus)]

    def shutdown(self) -> None:
        for n in list(self.processes):
            self.stop_model(n)


def _model(gpus, port, sleep_mode=True, dev="1"):
    args = ["--dtype", "float16", "--served-model-name", "served/x"]
    if sleep_mode:
        args.append("--enable-sleep-mode")
    cfg = {"gpus": list(gpus), "port": port, "model_name": "snapshot",
           "tensor_parallel_size": len(gpus), "extra_args": args}
    if sleep_mode:
        cfg["extra_env"] = {"VLLM_SERVER_DEV_MODE": dev}
    return cfg


# The exp3 topology, exactly: all three models declare gpus [0,1,2,3] at tp=4,
# so M = 1 and staging one REQUIRES evicting the incumbent.
EXP3_MODELS = {
    "qwen_72b": _model([0, 1, 2, 3], 8007),
    "qwen_32b": _model([0, 1, 2, 3], 8012),
    "qwen_72b_text": _model([0, 1, 2, 3], 8003),
}
PARK_GIB = {"qwen_32b": 120.77, "qwen_72b": 259.8, "qwen_72b_text": 257.3}


def _good_probe(name):
    return {"text": " Paris, the capital city of France.", "finish_reason": "length",
            "n_tokens": 8, "degenerate": False, "ok": True}


def _degenerate_probe(name):
    """What level-2 sleep actually returned on this cluster, with a 200."""
    return {"text": "!!!!", "finish_reason": "length", "n_tokens": 4,
            "degenerate": True, "ok": False}


# The literal string the real fp8 job returned for "The capital of France is"
# (24 tokens, finish_reason=length). It passes every composition heuristic.
FP8_TEXT = "\u306f.   1111               "


def _fp8_probe(name):
    """An engine that reports success, terminates cleanly, and is broken.

    Deliberately claims ok=True and degenerate=False, exactly as a probe built
    on composition statistics would: the actor must overrule it.
    """
    return {"text": FP8_TEXT, "finish_reason": "length", "n_tokens": 24,
            "degenerate": False, "ok": True}


@pytest.fixture
def rig(tmp_path):
    cg = FakeCgroup(tmp_path / "cg")
    orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
    actor = ma.VllmModelActor(
        orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
        file_reader=lambda: ma.cgroup_file_gb(root=cg.path),
        probe=_good_probe, teardown_timeout_s=5.0)
    yield cg, orch, actor
    orch.shutdown()


SPECS = ma.model_specs()


# ====================================================== measurement (I1) ===

class TestMeasurement:
    """The two traps, on real files."""

    def test_the_column_read_is_anon_not_current(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        cg.add_anon(10.0)
        cg.add_file(146.0)
        m = ma.cgroup_mem(root=cg.path)
        assert m["anon_gib"] == pytest.approx(10.0, abs=1e-6)
        assert m["file_gib"] == pytest.approx(146.0, abs=1e-6)
        assert m["current_gib"] == pytest.approx(156.0, abs=1e-6)
        # The trap in one assertion: current is 15x anon here, purely from the
        # weight shards sitting in page cache.
        assert m["current_gib"] > 10 * m["anon_gib"]
        assert ma.cgroup_anon_gib(root=cg.path) == pytest.approx(10.0, abs=1e-6)

    def test_gib_to_gb_conversion_is_not_skipped(self, tmp_path):
        """A 279 GB model is 20 GB of budget away from a 279 GiB one."""
        cg = FakeCgroup(tmp_path / "cg")
        cg.add_anon(100.0)
        assert ma.cgroup_anon_gb(root=cg.path) == pytest.approx(107.374, abs=1e-2)

    def test_missing_cgroup_says_so_rather_than_guessing(self, tmp_path):
        assert ma.cgroup_anon_gib(root=tmp_path / "nope") == -1.0
        d, why = ma._cgroup_dir(root=tmp_path / "nope")
        assert d is None and "memory.stat" in why

    def test_page_cache_never_enters_the_park_delta(self, rig):
        """THE trap. Booting reads 146 GiB of shards into page cache; the park
        is 120.77 GiB of anon. A delta on memory.current would report 266."""
        cg, orch, actor = rig
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R2_PROCESS_BYTES
        held = actor.measure_held_gb("qwen_32b")
        assert held == pytest.approx(120.77 * GIB / 1e9, rel=1e-3)
        current_delta_gb = (cg.anon_gib + cg.file_gib) * GIB / 1e9
        assert current_delta_gb > 2 * held, (
            "the page-cache confound must be big enough for this test to mean "
            "something")
        d = actor.last_stage_detail["qwen_32b"]
        assert d["measured_by"].startswith("cgroup memory.stat anon")

    def test_the_delta_is_against_the_awake_reading_not_pre_launch(self, rig):
        """The engine's own 8 GiB baseline is not charged to the park."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        d = actor.park_detail("qwen_32b")
        assert d["anon_gib_awake"] == pytest.approx(
            8.0 * GIB / 1e9, rel=1e-3)          # baseline, measured while awake
        assert d["held_gb"] == pytest.approx(120.77 * GIB / 1e9, rel=1e-3)
        # If the delta had been taken pre-launch it would carry the baseline.
        assert d["held_gb"] < (128.77 - 1) * GIB / 1e9

    def test_a_park_that_lands_in_file_is_refused_not_priced_at_zero(self, tmp_path):
        """The shape of results/bench_h1_fp8_tp1_12561711.json.

        anon 2.77 -> 2.77 GiB, file 1.32 -> 83.36 GiB, VRAM 87251 -> 1267 MiB.
        The park is real; it just did not land in the column this actor
        charges. Reporting held_gb 0.0 would tell the arbitrator that holding a
        72B is free, which would make every retention decision meaningless.
        """
        cg = FakeCgroup(tmp_path / "cg")
        cg.add_anon(2.77); cg.add_file(1.32)
        orch = FakeOrchestrator(
            dict(EXP3_MODELS), cg, baseline_gib=0.0, shard_file_gib={},
            park_gib={"qwen_72b": 0.0},          # nothing into anon...
            park_file_gib={"qwen_72b": 82.04})   # ...82.04 GiB into file
        orch.shard_file_gib = {"qwen_72b": 0.0}
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            file_reader=lambda: ma.cgroup_file_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        assert actor.stage(SPECS["qwen_72b"]) is Rung.R0_DISK
        r = actor.last_stage_detail["qwen_72b"]["reason"]
        # 82.04 GiB is 88.09 GB, and the message reports GB — the unit the
        # budget is in.
        assert "ParkNotMeasurable" in r
        assert "88.09 GB into the cgroup's `file`" in r
        assert "0.00 GB into `anon`" in r
        assert actor.measure_held_gb("qwen_72b") == 0.0
        assert "qwen_72b" not in orch.processes      # the boot was undone
        orch.shutdown()

    def test_charging_file_is_possible_but_must_be_asked_for(self, tmp_path):
        """The escape hatch is explicit, and its cost is stated: `file` also
        carries the weight shards read at boot."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(
            dict(EXP3_MODELS), cg, baseline_gib=0.0, shard_file_gib={},
            park_gib={"qwen_72b": 0.0}, park_file_gib={"qwen_72b": 82.04})
        orch.shard_file_gib = {"qwen_72b": 0.0}
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            file_reader=lambda: ma.cgroup_file_gb(root=cg.path),
            probe=_good_probe, charge_columns=("anon", "file"),
            teardown_timeout_s=5.0)
        assert actor.stage(SPECS["qwen_72b"]) is Rung.R2_PROCESS_BYTES
        assert actor.measure_held_gb("qwen_72b") >= 0.0
        d = actor.park_detail("qwen_72b")
        assert d["park_backing"] == "file"
        assert d["held_gb"] == pytest.approx(82.04 * GIB / 1e9, rel=1e-3)
        with pytest.raises(ValueError, match="anon and/or file"):
            ma.VllmModelActor(orch, charge_columns=("rss",))
        orch.shutdown()

    def test_a_normal_anon_park_is_labelled_as_such(self, rig):
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        assert actor.park_detail("qwen_32b")["park_backing"] == "anon"
        assert actor.charge_columns == ("anon",)

    def test_measure_held_gb_re_reads_rather_than_caching(self, rig):
        """Move the cgroup underneath it; a cached number would not notice."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        before = actor.measure_held_gb("qwen_32b")
        cg.add_anon(-20.0)
        after = actor.measure_held_gb("qwen_32b")
        assert after < before - 20.0 * GIB / 1e9 * 0.99

    def test_nothing_held_measures_zero(self, rig):
        cg, orch, actor = rig
        assert actor.measure_held_gb("qwen_32b") == 0.0
        assert actor.is_resident("qwen_32b") is False
        assert actor.release("qwen_32b") == 0.0

    def test_unknown_id_is_a_loud_keyerror(self, rig):
        cg, orch, actor = rig
        with pytest.raises(KeyError, match="id_to_model"):
            actor.model_for("llama_9000")
        spec = ResourceSpec("llama_9000", ResourceClass.MODEL,
                            Rung.R2_PROCESS_BYTES, 1.0, 10.0, 1.0)
        with pytest.raises(KeyError):
            actor.stage(spec)

    def test_process_tree_witness_is_readable_for_a_real_pid(self):
        p = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])
        try:
            assert ma.pid_alive(p.pid)
            assert p.pid in ma.process_tree(p.pid)
            assert ma.tree_anon_gb(p.pid) >= 0.0
        finally:
            p.kill(); p.wait()

    def test_dead_pid_is_zero_not_stale(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        assert not ma.pid_alive(p.pid)
        assert ma.tree_anon_gb(p.pid) == 0.0


# ============================================================ contract =====

class TestContract:
    def test_it_satisfies_the_protocol(self, rig):
        cg, orch, actor = rig
        assert isinstance(actor, ResidencyActor)
        assert actor.resource_class is ResourceClass.MODEL

    def test_the_catalogue_matches_the_published_densities(self):
        s = ma.model_specs()
        assert s["qwen_32b"].static_density == pytest.approx(3.81, abs=5e-3)
        assert s["qwen_72b"].static_density == pytest.approx(2.86, abs=5e-3)
        assert s["qwen_72b_text"].static_density == pytest.approx(2.78, abs=5e-3)
        assert all(v.held_rung is Rung.R2_PROCESS_BYTES for v in s.values())

    def test_a_declared_footprint_is_never_what_gets_charged(self, rig):
        """I1. The catalogue number is trust-B (host-wide MemTotal-MemAvailable);
        once parked, the measurement supersedes it."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        assert actor.measure_held_gb("qwen_32b") != pytest.approx(129.7)


# =========================================================== coherence =====

class TestCoherence:
    def test_the_h1_rule_would_have_accepted_the_degeneracy(self):
        """Why this actor does not just reuse the bench's `ok`.

        bench_h1_quantized_park.coherence_probe marks a reply ok when the text
        is non-empty and a finish_reason exists. '!!!!' satisfies both.
        """
        text, finish = "!!!!", "length"
        h1_ok = bool(text.strip()) and finish is not None
        assert h1_ok is True
        assert ma._is_degenerate(text) is True

    def test_the_fp8_answer_defeats_both_composition_heuristics(self):
        """The generalisation, as a test.

        The real fp8 run answered "The capital of France is" with
        'は.   1111               '. The original H1 rule (non-empty +
        finish_reason) passes it. The degeneracy repair passes it too: it
        contains alphanumerics and three distinct non-space characters. No
        statistic over character composition can separate this from a terse
        correct answer — only knowing the answer can.
        """
        assert bool(FP8_TEXT.strip()) is True             # rule 1 passes
        assert ma._is_degenerate(FP8_TEXT) is False       # rule 2 passes
        assert len(set(FP8_TEXT.replace(" ", ""))) > 2
        v = ma.judge_probe(FP8_TEXT, "length", ma.PROBE_MUST_CONTAIN)
        assert v["anchored"] is False and v["ok"] is False  # rule 3 catches it

    def test_the_anchor_is_tied_to_the_prompt(self):
        assert "France" in ma.PROBE_PROMPT and ma.PROBE_MUST_CONTAIN == "Paris"
        assert ma.judge_probe(" Paris.", "stop", "Paris")["ok"] is True
        # Case-insensitive, so a lowercased completion is not a false alarm.
        assert ma.judge_probe(" paris", "stop", "Paris")["anchored"] is True
        # No anchor configured => the composition rules alone decide.
        assert ma.judge_probe(FP8_TEXT, "length", None)["ok"] is True

    def test_an_injected_probe_cannot_skip_the_anchor(self, tmp_path):
        """A probe is a measurement instrument, not an authority on whether
        the engine works. _fp8_probe claims ok=True; the actor overrules it."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_fp8_probe, teardown_timeout_s=5.0)
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        r = actor.last_stage_detail["qwen_32b"]["reason"]
        assert "IncoherentWake" in r and "does not contain 'Paris'" in r
        assert actor.measure_held_gb("qwen_32b") == 0.0
        assert "qwen_32b" not in orch.processes
        orch.shutdown()

    def test_degeneracy_detector(self):
        assert ma._is_degenerate("") is True
        assert ma._is_degenerate("   ") is True
        assert ma._is_degenerate("!!!!") is True
        assert ma._is_degenerate("aaaa") is True
        assert ma._is_degenerate(" Paris") is False

    def test_an_incoherent_wake_raises_and_holds_nothing(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_degenerate_probe, teardown_timeout_s=5.0)
        rung = actor.stage(SPECS["qwen_32b"])
        assert rung is Rung.R0_DISK
        assert "IncoherentWake" in actor.last_stage_detail["qwen_32b"]["reason"]
        assert actor.measure_held_gb("qwen_32b") == 0.0
        # And the engine it booted is gone: an engine that cannot be trusted to
        # generate must not be left holding the GPUs.
        assert "qwen_32b" not in orch.processes
        assert actor.coherence_failures and actor.coherence_failures[0]["degenerate"]
        orch.shutdown()

    def test_level_2_is_refused_because_only_l1_was_verified(self, rig):
        cg, orch, actor = rig
        with pytest.raises(ma.ParkRefused, match="not verified"):
            ma.VllmModelActor(orch, park_level=2)
        # ...but a deliberate measurement of the unverified level is allowed.
        a2 = ma.VllmModelActor(orch, park_level=2, allow_unverified_level=True)
        assert a2.park_level == 2

    def test_reference_mismatch_is_recorded_but_not_fatal(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        texts = iter([" Paris.", " Paris.", " Paris, in France."])
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=lambda n: {"text": next(texts), "finish_reason": "length",
                             "degenerate": False, "ok": True},
            teardown_timeout_s=5.0)
        actor.stage(SPECS["qwen_32b"])              # boot probe -> reference
        actor.wake("qwen_32b")                      # matches
        actor.park("qwen_32b")
        actor.wake("qwen_32b")                      # differs
        assert any(f.get("reference_mismatch") for f in actor.coherence_failures)
        orch.shutdown()


# ================================================ the GPU-occupancy path ===

class TestGpuOccupancy:
    """Deliverable 2. The failure is reproduced before it is fixed."""

    def test_the_real_failure_reproduces_without_the_actor(self, rig):
        cg, orch, actor = rig
        orch.start_model_measured("qwen_72b")
        with pytest.raises(RuntimeError) as ei:
            orch.start_model_measured("qwen_32b")
        assert "Cannot start qwen_32b: GPUs [0, 1, 2, 3] occupied by qwen_72b" \
            in str(ei.value)

    def test_the_actor_parks_the_incumbent_and_then_boots(self, rig):
        cg, orch, actor = rig
        orch.start_model_measured("qwen_72b")
        info = actor.activate("qwen_32b")
        assert info["mechanism"] == "cold_boot"
        assert [e["model"] for e in info["evicted"]] == ["qwen_72b"]
        assert info["evicted"][0]["action"] == "park"
        assert orch.is_sleeping("qwen_72b")           # weights kept in host RAM
        assert not orch.is_sleeping("qwen_32b")
        # The incumbent's park was MEASURED, not assumed.
        assert info["evicted"][0]["held_gb"] == pytest.approx(
            259.8 * GIB / 1e9, rel=1e-3)

    def test_a_parked_target_is_woken_not_re_booted(self, rig):
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])                # boot + park
        boots = orch.boots["qwen_32b"]
        info = actor.activate("qwen_32b")
        assert info["mechanism"] == "wake"
        assert orch.boots["qwen_32b"] == boots        # 2.076 s, not 782.27 s

    def test_a_victim_without_sleep_mode_is_stopped_and_the_downgrade_is_named(
            self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        models = dict(EXP3_MODELS)
        models["qwen_72b"] = _model([0, 1, 2, 3], 8007, sleep_mode=False)
        orch = FakeOrchestrator(models, cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        info = actor.activate("qwen_32b")
        ev = info["evicted"][0]
        assert ev["action"] == "stop"
        assert "no --enable-sleep-mode" in ev["downgrade_reason"]
        assert "qwen_72b" not in orch.processes
        orch.shutdown()

    def test_the_budget_can_force_a_stop_instead_of_a_park(self, tmp_path):
        """I4: the budget question belongs to the ledger, not the actor. When
        the answer is no, the victim is stopped and the downgrade is recorded
        as forced rather than preferred."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        seen = []
        def can_park(name, gb):
            seen.append((name, gb))
            return False
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, can_park=can_park, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        info = actor.activate("qwen_32b")
        assert seen == [("qwen_72b", 279.0)]          # priced from the catalogue
        assert info["evicted"][0]["action"] == "stop"
        assert "budget declined" in info["evicted"][0]["downgrade_reason"]
        orch.shutdown()

    def test_a_protected_incumbent_blocks_and_nothing_is_staged(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, evictable=lambda n: False, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        with pytest.raises(ma.GpusNotFreed, match="protected"):
            actor.activate("qwen_32b")
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        # The incumbent is untouched: not slept, not stopped, still serving.
        assert "qwen_72b" in orch.processes and not orch.is_sleeping("qwen_72b")
        assert "qwen_32b" not in orch.processes
        d = actor.last_stage_detail["qwen_32b"]
        assert d["blocked_by"] == "qwen_72b" and d["parked"] is False
        # The failure path records its detail too — a blocked eviction is
        # exactly the case a caller needs to be able to read afterwards.
        ev = actor.last_eviction_detail["qwen_32b"]
        assert ev["ok"] is False and ev["evicted"] == []
        orch.shutdown()

    def test_it_refuses_before_evicting_when_the_target_cannot_park(self, tmp_path):
        """The ordering that matters: a stage that can never reach R2 must not
        cost the incumbent its GPU. Evicting for nothing is worse than
        declining."""
        cg = FakeCgroup(tmp_path / "cg")
        models = dict(EXP3_MODELS)
        models["qwen_32b"] = _model([0, 1, 2, 3], 8012, sleep_mode=False)
        orch = FakeOrchestrator(models, cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        assert "never reach R2" in actor.last_stage_detail["qwen_32b"]["reason"]
        assert not orch.is_sleeping("qwen_72b")       # nothing was evicted
        assert ("sleep_model", "qwen_72b", 1) not in orch.calls
        orch.shutdown()

    def test_vram_that_does_not_come_back_is_caught(self, tmp_path):
        """An eviction is only real if the VRAM returns. /sleep returning 200
        is a claim; nvidia-smi is the evidence."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        orch.get_gpu_memory = lambda: [{"total": 49140, "free": 100}
                                       for _ in range(6)]
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        r = actor.last_stage_detail["qwen_32b"]["reason"]
        assert "VRAM is free" in r and "outside this orchestrator" in r
        assert "qwen_32b" not in orch.processes
        orch.shutdown()

    def test_max_parked_is_bounded_at_the_measured_k(self, tmp_path):
        """k=3 simultaneous L1 sleeps was measured NOT to complete (M6)."""
        cg = FakeCgroup(tmp_path / "cg")
        models = {"a": _model([0], 8100), "b": _model([1], 8101),
                  "c": _model([2], 8102)}
        orch = FakeOrchestrator(models, cg,
                                park_gib={"a": 10.0, "b": 10.0, "c": 10.0})
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        spec = lambda n: ResourceSpec(n, ResourceClass.MODEL,
                                      Rung.R2_PROCESS_BYTES, 10.0, 100.0, 2.0)
        assert actor.stage(spec("a")) is Rung.R2_PROCESS_BYTES
        assert actor.stage(spec("b")) is Rung.R2_PROCESS_BYTES
        assert actor.stage(spec("c")) is Rung.R0_DISK
        assert "max_parked=2" in actor.last_stage_detail["c"]["reason"]
        assert ma.DEFAULT_MAX_PARKED == 2
        orch.shutdown()

    def test_a_failed_boot_after_an_eviction_wakes_the_victim_back(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_72b")
        real_start = orch.start_model_measured

        def boom(name, metrics=None):
            if name == "qwen_32b":
                raise RuntimeError("WorkerProc died during NCCL init")
            return real_start(name, metrics)
        orch.start_model_measured = boom

        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        # The workflow does not lose its planner to a stage that failed.
        assert not orch.is_sleeping("qwen_72b")
        ev = actor.last_eviction_detail["qwen_32b"]
        assert ev["restored"] == ["qwen_72b"]
        orch.shutdown()

    def test_a_park_failure_on_an_engine_we_booted_undoes_the_boot(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB),
                                sleep_fails={"qwen_32b"})
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R0_DISK
        d = actor.last_stage_detail["qwen_32b"]
        assert "HTTP 500" in d["reason"] and "stopped the engine" in d["undone"]
        assert "qwen_32b" not in orch.processes
        orch.shutdown()

    def test_an_already_serving_engine_that_cannot_park_reports_r3_not_r2(
            self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB),
                                sleep_fails={"qwen_32b"})
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        orch.start_model_measured("qwen_32b")         # not ours; leave it alone
        rung = actor.stage(SPECS["qwen_32b"])
        assert rung is Rung.R3_ACTIVATED
        assert actor.last_stage_detail["qwen_32b"]["parked"] is False
        assert "qwen_32b" in orch.processes           # we did not stop it
        orch.shutdown()

    def test_a_slept_engine_does_not_conflict(self, rig):
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_72b"])                # parked: VRAM is free
        assert actor.gpu_conflicts("qwen_32b") == []

    def test_park_amortisation_is_recorded_both_ways(self, rig):
        """A first park is ~9x a later one; a scheduler amortising over a
        workflow needs both numbers, not their average."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        actor.wake("qwen_32b")
        actor.park("qwen_32b")
        d = actor.park_detail("qwen_32b")
        assert d["park_count"] == 2
        assert d["first_park_s"] is not None
        assert len(d["subsequent_park_s"]) == 1
        assert ma.FIRST_PARK_S == 23.692 and ma.SUBSEQ_PARK_S == 2.611

    def test_v2_whether_the_backup_survives_a_wake_is_measured(self, tmp_path):
        """Unverified on hardware. The actor measures it instead of assuming
        either answer; both branches are exercised here."""
        for survives in (False, True):
            cg = FakeCgroup(tmp_path / f"cg{survives}")
            orch = FakeOrchestrator(dict(EXP3_MODELS), cg,
                                    park_gib=dict(PARK_GIB),
                                    backup_survives_wake=survives)
            actor = ma.VllmModelActor(
                orch, anon_reader=lambda c=cg: ma.cgroup_anon_gb(root=c.path),
                probe=_good_probe, teardown_timeout_s=5.0)
            actor.stage(SPECS["qwen_32b"])
            actor.wake("qwen_32b")
            after = actor.park_detail("qwen_32b")["anon_gib_awake_after_wake"]
            held_awake = actor.measure_held_gb("qwen_32b")
            if survives:
                assert held_awake > 100.0             # still charging the budget
            else:
                assert held_awake < 1.0
            assert after is not None
            orch.shutdown()


# ========================================================= release (I2) ====

class TestRelease:
    def test_release_is_real_four_ways(self, rig):
        """A2's bar (test_D2_release_is_real): the measured free, the vanished
        /proc entries, an independent witness, and the next use paying cold."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        pids = ma.process_tree(orch.processes["qwen_32b"].pid)
        held = actor.measure_held_gb("qwen_32b")
        assert held > 100.0

        freed = actor.release("qwen_32b")
        d = actor.last_release_detail["qwen_32b"]

        # 1. the measured give-back, in the same currency as the charge
        assert freed == pytest.approx(held, rel=0.10)
        assert d["measured_by"].startswith("cgroup memory.stat anon delta")
        assert d["cgroup_corroborates"] is True
        # 2. the process tree is gone from /proc
        assert d["proc_gone"] and not any(ma.pid_alive(p) for p in pids)
        # 3. the independent witness was taken
        assert "tree_anon_before_gb" in d and d["witness"].startswith("Anonymous")
        # 4. nothing is held any more
        assert actor.measure_held_gb("qwen_32b") == 0.0
        assert actor.is_resident("qwen_32b") is False

        # THE BEHAVIOURAL PROOF: re-staging pays a cold boot again. If release
        # had freed nothing while reporting success, this would be a wake.
        boots = orch.boots["qwen_32b"]
        assert actor.stage(SPECS["qwen_32b"]) is Rung.R2_PROCESS_BYTES
        assert orch.boots["qwen_32b"] == boots + 1

    def test_a_surviving_process_raises_release_not_honoured(self, tmp_path):
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB),
                                stop_really_kills=False)
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=1.0)
        actor.stage(SPECS["qwen_32b"])
        with pytest.raises(ReleaseNotHonoured, match="budget is fiction"):
            actor.release("qwen_32b")
        d = actor.last_release_detail["qwen_32b"]
        assert d["proc_gone"] is False and d["still_alive_pids"]
        orch.processes["qwen_32b"].kill()
        orch.processes.clear()

    def test_a_short_release_is_returned_short_not_rounded_up(self, tmp_path):
        """The invariant, stated as a test: a partial give-back must reach the
        ledger as a partial give-back. Rounding it to what was charged is
        exactly how a budget becomes fiction."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        actor.stage(SPECS["qwen_32b"])
        held = actor.measure_held_gb("qwen_32b")
        real_stop = orch.stop_model

        def leaky(name, wait_s=30.0):
            """The engine exits but half the anon stays (a leak, or a
            neighbour's allocation landing in the same cgroup)."""
            real_stop(name, wait_s)
            cg.add_anon(60.0)
        orch.stop_model = leaky

        freed = actor.release("qwen_32b")
        assert freed < 0.6 * held
        d = actor.last_release_detail["qwen_32b"]
        assert d["proc_gone"] is True
        assert d["cgroup_corroborates"] is False
        orch.shutdown()

    def test_release_witness_is_the_cgroup_not_the_process(self, rig):
        """The witness the ledger needs from a TEARDOWN actor.

        measure_held_gb() after a teardown is 0.0 by construction, so the
        ledger's before/after drop confirms nothing. This reading is of the
        JOB'S ALLOCATION and does not need the released process to exist.
        """
        cg, orch, actor = rig
        assert actor.release_witness("qwen_32b") is None    # nothing observed
        actor.stage(SPECS["qwen_32b"])
        assert actor.release_witness("qwen_32b") is None    # still nothing

        held = actor.measure_held_gb("qwen_32b")
        freed = actor.release("qwen_32b")
        w = actor.release_witness("qwen_32b")
        assert w is not None
        assert w == pytest.approx(freed, rel=1e-9)
        assert w == pytest.approx(held, rel=0.10)
        # The engine baseline is not credited: that would let the interpreter
        # and the CUDA context vouch for weights that never came back.
        d = actor.last_release_detail["qwen_32b"]
        assert w < d["cgroup_total_freed_gb"]

    def test_a_stale_witness_is_refused_after_a_restage(self, rig):
        """A1's hazard: last_release_detail persists, so an unstamped witness
        would vouch for a release it never observed."""
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        actor.release("qwen_32b")
        assert actor.release_witness("qwen_32b") is not None

        actor.stage(SPECS["qwen_32b"])            # the resource is back
        assert actor.release_witness("qwen_32b") is None, \
            "the previous teardown's delta must not vouch for a live resource"
        # The evidence itself is kept — only its authority to vouch is revoked.
        assert actor.last_release_detail["qwen_32b"]["cgroup_freed_gb"] > 0

        actor.release("qwen_32b")                 # a NEW release re-stamps it
        assert actor.release_witness("qwen_32b") is not None
        assert actor.last_release_detail["qwen_32b"]["release_seq"] == 2

    def test_no_readable_cgroup_means_no_witness_not_a_zero(self, tmp_path):
        """A missing instrument reported as a 0.0 give-back would fail every
        release on a machine without cgroup v2."""
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: -1.0,       # no cgroup anywhere
            probe=_good_probe, teardown_timeout_s=5.0)
        actor.stage(SPECS["qwen_32b"])
        actor.release("qwen_32b")
        d = actor.last_release_detail["qwen_32b"]
        assert d["cgroup_readable"] is False
        assert d["cgroup_freed_gb"] == 0.0
        assert actor.release_witness("qwen_32b") is None
        orch.shutdown()

    def test_an_evicted_victim_gets_its_own_witness(self, rig):
        """The eviction path tears down through the same code, so a stopped
        victim is witnessed exactly as rigorously as a released retention."""
        cg, orch, actor = rig
        orch.start_model_measured("qwen_72b")
        actor._can_park = lambda n, gb: False     # force a stop, not a park
        actor.activate("qwen_32b")
        w = actor.release_witness("qwen_72b")
        assert w is not None and w > 0

    def test_release_of_something_never_held_is_zero(self, rig):
        cg, orch, actor = rig
        assert actor.release("qwen_72b_text") == 0.0


# ==================================================== ledger integration ===

class TestLedgerIntegration:
    def test_the_ledger_charges_the_measured_number(self, rig):
        from runtime.residency.ledger import ResidencyLedger
        cg, orch, actor = rig
        led = ResidencyLedger(budget_gb=512.0)
        led.register_actor(actor)
        rung = actor.stage(SPECS["qwen_32b"])
        e = led.charge(SPECS["qwen_32b"], rung, now_s=0.0)
        assert led.declared_charges == [], \
            "I1: the catalogue's 129.7 GB must not be what gets booked"
        assert e.charged_gb == pytest.approx(120.77 * GIB / 1e9, rel=1e-3)
        assert e.rung is Rung.R2_PROCESS_BYTES
        charged, measured = led.reconcile()["qwen_32b"]
        assert charged == pytest.approx(measured, rel=1e-6)

    def test_a_clean_release_passes_the_ledgers_confirmation(self, rig):
        """The end-to-end I2 path, and the trap inside it.

        A teardown gives back the parked weights AND the engine's own baseline
        (interpreter, CUDA context, allocator arenas). Only the weights were
        charged. Returning the whole delta would make release() claim MORE than
        the ledger measured and trip the confirmation on every clean release —
        so the baseline is subtracted, and the raw total is still recorded.
        """
        from runtime.residency.ledger import ResidencyLedger
        cg, orch, actor = rig
        led = ResidencyLedger(budget_gb=512.0)
        led.register_actor(actor)
        rung = actor.stage(SPECS["qwen_32b"])
        charged = led.charge(SPECS["qwen_32b"], rung, now_s=0.0).charged_gb

        freed = led.release("qwen_32b", now_s=1.0)
        assert freed == pytest.approx(charged, rel=1e-3)
        assert not led.is_held("qwen_32b")
        assert led.release_shortfalls == [] and led.over_releases == []

        d = actor.last_release_detail["qwen_32b"]
        assert d["cgroup_total_freed_gb"] > d["cgroup_freed_gb"]
        assert d["engine_baseline_gb"] == pytest.approx(8.0 * GIB / 1e9, rel=1e-3)
        assert d["baseline_unknown"] is False

        # The release was WITNESSED, by the actor's own cgroup reading, and the
        # ledger recorded which witness spoke.
        assert led.unwitnessed_releases == 0
        log = led.release_log[-1]
        assert log["witness_source"] == "actor-witness"
        assert log["witness_drop_gb"] == pytest.approx(freed, rel=1e-9)

    def test_a_short_release_is_caught_by_the_report_and_the_witness(self, tmp_path):
        from runtime.residency.ledger import ResidencyLedger
        cg = FakeCgroup(tmp_path / "cg")
        orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
        actor = ma.VllmModelActor(
            orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
            probe=_good_probe, teardown_timeout_s=5.0)
        led = ResidencyLedger(budget_gb=512.0)
        led.register_actor(actor)
        rung = actor.stage(SPECS["qwen_32b"])
        led.charge(SPECS["qwen_32b"], rung, now_s=0.0)
        real_stop = orch.stop_model
        orch.stop_model = lambda n, wait_s=30.0: (real_stop(n, wait_s),
                                                  cg.add_anon(100.0))
        charged = led.entry("qwen_32b").charged_gb

        # The gap this test was written for: measure_held_gb() after a teardown
        # is 0.0 BY CONSTRUCTION, so the ledger's before/after drop is
        # tautologically the whole charge and cannot see a cgroup that kept the
        # memory. It is closed twice over now — by the report-below-charge rule
        # and by release_witness() — and both readings agree here.
        with pytest.raises(ReleaseNotHonoured, match="budget is fiction"):
            led.release("qwen_32b", now_s=1.0)
        assert led.is_held("qwen_32b"), \
            "the memory demonstrably did not come back, so the charge stands"

        d = actor.last_release_detail["qwen_32b"]
        assert d["proc_gone"] is True                 # the tautology holds...
        assert actor.measure_held_gb("qwen_32b") == 0.0   # ...and proves nothing
        assert d["cgroup_corroborates"] is False
        assert d["cgroup_freed_gb"] < 0.6 * charged
        # The independent witness reports the same shortfall, from the cgroup
        # rather than from the actor's own account of itself.
        w = actor.release_witness("qwen_32b")
        assert w is not None and w < 0.6 * charged
        orch.shutdown()

    def test_reconcile_detail_publishes_the_attribution_residual(self, rig):
        cg, orch, actor = rig
        actor.stage(SPECS["qwen_32b"])
        r = actor.reconcile_detail()
        assert r["parked"] == ["qwen_32b"]
        # anon moved by the engine baseline + the park; we attribute only the
        # park, so the residual is the baseline — visible, not corrected.
        assert r["residual_gb"] == pytest.approx(8.0 * GIB / 1e9, rel=0.05)


# ======================================== the confidence gate (deliv. 3) ===

class TestConfidenceGate:
    """The gate change, and the ordering that makes it safe."""

    def _resource(self, **kw):
        from runtime.events import ResourceSpec as EventResource
        d = dict(resource_id="r", resource_type="vllm_model", name="qwen_72b",
                 confidence=0.80, cancellation_safe=False,
                 consumer_tool="computation_task_screw_dislocation",
                 consumer_step_offset=1, estimated_load_s=465.0,
                 proactive_swap=True)
        d.update(kw)
        return EventResource(**d)

    def _sched(self, executor):
        from runtime.config import RuntimeConfig, RuntimeMode
        from runtime.prefetch.scheduler import PrefetchScheduler
        return PrefetchScheduler(executor, RuntimeConfig(mode=RuntimeMode.REAL))

    def test_the_arithmetic_that_made_the_gate_unreachable(self):
        """A regression guard on the coupling itself: a plan-only prediction is
        floored at 0.80 and the gate is 0.85, so it can never clear it."""
        from runtime.config import RuntimeConfig
        from runtime.predictor.learned_predictor import _PLAN_CONFIDENCE_DEFAULT
        assert _PLAN_CONFIDENCE_DEFAULT < RuntimeConfig().confidence_threshold

    def test_without_an_eviction_path_the_bypass_does_not_fire(self):
        """Deliverable 3 must not ship alone: without deliverable 2 it turns
        34 silent skips into instant orchestrator failures."""
        class DumbExecutor:
            executor_id = "dumb"
        s = self._sched(DumbExecutor())
        ok, reason = s._should_prefetch(self._resource(), 0.0)
        assert ok is False
        assert reason == "confidence_below_threshold (0.80 < 0.85)"

    def test_with_an_eviction_path_the_proactive_swap_is_admitted(self):
        class EvictingExecutor:
            executor_id = "model_prefetch"
            can_evict_gpu_occupants = True
        s = self._sched(EvictingExecutor())
        ok, reason = s._should_prefetch(self._resource(), 0.0)
        assert ok is True
        # The reason string is UNCHANGED, so no existing trace column gains a
        # new value and A3's parser keeps working.
        assert reason == "proactive_swap_compute_window"

    def test_the_bypass_does_not_jump_the_horizon_check(self):
        """Only the CONFIDENCE gates are bypassed. A proactive swap predicted
        too far ahead still reports horizon_exceeded, under that name."""
        class EvictingExecutor:
            executor_id = "model_prefetch"
            can_evict_gpu_occupants = True
        s = self._sched(EvictingExecutor())
        ok, reason = s._should_prefetch(self._resource(consumer_step_offset=9), 0.0)
        assert ok is False
        # Unchanged: the pre-existing chain checks confidence before horizon,
        # so this still reports the confidence skip and not a new string.
        assert reason == "confidence_below_threshold (0.80 < 0.85)"
        ok, reason = s._should_prefetch(
            self._resource(consumer_step_offset=9, confidence=1.0), 0.0)
        assert ok is False and reason.startswith("horizon_exceeded")

    def test_the_bypass_is_scoped_to_proactive_swap_only(self):
        """Honesty about the size of the fix: of the 34
        confidence_below_threshold skips on L40S, 9 are the proactive_swap
        qwen_72b entry and only those are unblocked. The 21 data_file and 4
        plan_task skips are untouched."""
        class EvictingExecutor:
            executor_id = "model_prefetch"
            can_evict_gpu_occupants = True
        s = self._sched(EvictingExecutor())
        ok, reason = s._should_prefetch(
            self._resource(proactive_swap=False, resource_type="data_file",
                           name="w_eam4_big.fs", cancellation_safe=True), 0.0)
        assert ok is False
        assert reason == "confidence_below_threshold (0.80 < 0.85)"

    def test_the_model_actor_declares_the_capability(self, rig):
        cg, orch, actor = rig
        assert actor.can_evict_gpu_occupants is True

    def test_the_executor_declares_it_only_when_it_can(self, rig):
        from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
        cg, orch, actor = rig
        plain = ModelPrefetchExecutor(orch)
        assert plain.can_evict_gpu_occupants is False
        wired = ModelPrefetchExecutor(orch, residency_actor=actor)
        assert wired.can_evict_gpu_occupants is True
        legacy = ModelPrefetchExecutor(orch, evict_conflicting=True)
        assert legacy.can_evict_gpu_occupants is True
        plain.shutdown(wait=False); wired.shutdown(wait=False)
        legacy.shutdown(wait=False)

    def test_a_composite_asks_the_sub_executor_that_will_run_it(self, rig):
        """A CompositeExecutor holding a GPU-capable model executor must not
        license a data_file admission on that executor's strength."""
        from runtime.prefetch.data_prefetch import CompositeExecutor
        from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
        from runtime.prefetch.simulated import SimulatedPrefetchExecutor
        cg, orch, actor = rig
        mex = ModelPrefetchExecutor(orch, residency_actor=actor)
        comp = CompositeExecutor({"vllm_model": mex,
                                  "data_file": SimulatedPrefetchExecutor()})
        s = self._sched(comp)
        ok, reason = s._should_prefetch(self._resource(), 0.0)
        assert ok is True and reason == "proactive_swap_compute_window"
        ok, reason = s._should_prefetch(
            self._resource(resource_type="data_file", name="w_eam4_big.fs",
                           cancellation_safe=True), 0.0)
        assert ok is False
        assert reason == "confidence_below_threshold (0.80 < 0.85)"
        mex.shutdown(wait=False)

    def test_the_wired_executor_routes_a_load_through_the_actor(self, rig):
        """End to end on the fake: an admitted prefetch for a model whose GPUs
        are occupied now SUCCEEDS instead of failing in 1 ms."""
        from runtime.prefetch.base import PrefetchStatus, PrefetchTask
        from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
        from runtime.events import ResourceSpec as EventResource
        cg, orch, actor = rig
        orch.start_model_measured("qwen_72b")
        ex = ModelPrefetchExecutor(orch, residency_actor=actor)
        task = PrefetchTask(
            resource=EventResource(resource_id="x", resource_type="vllm_model",
                                   name="qwen_32b", confidence=0.8),
            status=PrefetchStatus.PENDING, checkpoint_id="c",
            workflow_step_at_start=0, predicted_at_step=0)
        ex.start(task)
        deadline = time.time() + 60
        while not ex.is_complete(task) and time.time() < deadline:
            time.sleep(0.05)
        res = ex.get_result(task)
        assert res["success"] is True, res
        assert res["mechanism"] == "cold_boot"
        assert res["evicted"] == ["qwen_72b"]
        assert task.status is PrefetchStatus.COMPLETED
        assert orch.is_sleeping("qwen_72b")
        ex.shutdown(wait=False)
