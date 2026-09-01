"""
The proactive swap must never leave the workflow on wedged GPUs.

WHAT BROKE, AND WHERE IT IS WRITTEN DOWN
----------------------------------------
Three end-to-end Tandem trials died the same way on 2026-09-01:

  results/eval_q1_q4/runs/atomagents_exp3_aligned/tandem/
      t01__20260901-144241__f9b64ab/stdout.log      (tp=4, 4x Blackwell)
      t02__20260901-152547__f9b64ab/stdout.log      (tp=4, 4x Blackwell)
  results/eval_q1_q4/runs/atomagents_exp3_aligned_tp2/tandem/
      t01__20260901-151756__f9b64ab/stdout.log      (tp=2, 2x Blackwell)

The chain, read off t01 tp=2 (line numbers are that file):

  275  the ROUTER asks for qwen_32b and starts a cold boot for the current step
  358  the proactive-swap wait loop's 600 s expires -- the incumbent was never
       going to hand over, it was 12/18 shards into a boot the workflow was
       blocked on -- and the legacy fallback calls stop_model() on it
  359  SIGINT, then SIGKILL at 30 s (:360).  The LAUNCHER dies; the EngineCore
       and Worker children do not.  They finish all 18 shards and allocate a
       43.77 GiB KV cache AFTER ":361 qwen_32b stopped." is printed.
  394  "GPU VRAM drained in 242.4s" -- which is stop_model's drain_timeout=240
       ceiling, not its success condition.  The message is printed on BOTH
       exits (model_orchestrator.py:352-368), so a drain that never happened
       reads exactly like one that did.  The same line says 240.2 s and 240.3 s
       in the two tp=4 trials: the ceiling, three times out of three.
  396  the router's qwen_32b times out at 1200 s
  547  the router falls forward to qwen_72b and vLLM refuses to start:
       "Free memory on device cuda:1 (14.28/94.97 GiB)" -- 80.69 GiB still held
       by the orphans.  94.97 - 14.28 = 80.69, and 31.46 GiB of weights +
       43.77 GiB of KV cache + context is exactly that.
  739  rc=1, no serving model, the workflow dies on Connection refused.

TWO THINGS THAT CORRECT THE OBVIOUS READING, both from the same evidence:

  * the residency actor was NOT bypassed at the boot.  Its VRAM confirmation
    fired and REFUSED, exactly as designed --
        trace.jsonl: {"event_type": "prefetch_completed", "success": false,
         "error": "cannot activate qwen_72b: GPUs [0, 1] - after eviction only
          15.9% of VRAM is free ..."}
    The qwen_72b that crashed was the ROUTER's, on the critical path.  What
    bypassed the actor was the EVICTION, and by then the node was wedged.

  * `restore_on_failure` does not cover activate().  `_restore()` is called
    from stage() alone (model_actor.py:1270, :1296).  test_the_actor_alone_
    does_not_restore_on_the_activate_path below pins that, so if the actor
    ever grows the guard itself this file says so instead of going quiet.
"""
from __future__ import annotations

import inspect

import pytest

import runtime.prefetch.model_prefetch as mp
import runtime.residency.model_actor as ma
from runtime.events import ResourceSpec as EventResource
from runtime.prefetch.base import PrefetchStatus, PrefetchTask
from runtime.tests.test_residency_model_actor import (
    EXP3_MODELS,
    PARK_GIB,
    FakeCgroup,
    FakeOrchestrator,
    _good_probe,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Clock:
    """Stands in for the `time` module inside model_prefetch.

    sleep() advances a virtual clock, so a 300 s or 600 s wait costs the suite
    nothing and the timeout branch is reachable at all.  Only model_prefetch is
    patched; the actor keeps the real clock, so its own measurements stay real.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def perf_counter(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(mp, "time", c)
    return c


def _task(name: str = "qwen_72b", proactive: bool = True) -> PrefetchTask:
    return PrefetchTask(
        resource=EventResource(resource_id=name, resource_type="vllm_model",
                               name=name, confidence=0.80,
                               proactive_swap=proactive),
        status=PrefetchStatus.IN_PROGRESS, checkpoint_id="c",
        workflow_step_at_start=0, predicted_at_step=0,
        proactive_swap=proactive)


@pytest.fixture
def rig(tmp_path):
    """The exp3 topology on fakes: three models, all on the same GPUs, so a
    swap REQUIRES taking them off the incumbent."""
    cg = FakeCgroup(tmp_path / "cg")
    orch = FakeOrchestrator(dict(EXP3_MODELS), cg, park_gib=dict(PARK_GIB))
    actor = ma.VllmModelActor(
        orch, anon_reader=lambda: ma.cgroup_anon_gb(root=cg.path),
        file_reader=lambda: ma.cgroup_file_gb(root=cg.path),
        probe=_good_probe, teardown_timeout_s=5.0)
    yield cg, orch, actor
    orch.shutdown()


def _wired(orch, actor, serving: bool | None = True, **kw):
    """An executor with the actor wired, with the HTTP readiness probe stubbed.

    `_incumbent_is_serving` is the one thing in this path that talks to a real
    socket; the fakes have no server behind their ports.
    """
    ex = mp.ModelPrefetchExecutor(orch, probes=None, residency_actor=actor, **kw)
    ex._incumbent_is_serving = lambda name: serving
    return ex


# ======================================================================
# 1. the no-actor path is preserved
# ======================================================================

class TestLegacyPathUnchanged:
    """Arms without --residency must behave EXACTLY as they did.

    The defect the fallback carries predates Tandem; `full_system`,
    `naive_prefetch` and `baseline` numbers are already in the paper and a
    silent behaviour change here would invalidate them.  These tests pin the
    old behaviour deliberately, bug included.
    """

    def _orch_with_incumbent(self):
        orch = mp.FakeModelOrchestrator(
            load_times={"qwen_72b": 0.0, "qwen_32b": 0.0},
            models={k: {"gpus": v["gpus"], "port": v["port"]}
                    for k, v in EXP3_MODELS.items()})
        orch.start_model_measured("qwen_32b")
        orch.calls.clear()
        return orch

    def test_the_timeout_still_stops_the_incumbent_itself(self, clock, capsys):
        orch = self._orch_with_incumbent()
        ex = mp.ModelPrefetchExecutor(orch, probes=None, residency_actor=None)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)

        assert res["success"] is True
        assert ("stop_model", "qwen_32b") in orch.calls
        out = capsys.readouterr().out
        assert "Proactive swap timeout: stopping qwen_32b (fallback)" in out
        assert "Proactive swap: GPUs free — loading qwen_72b." in out
        # The 600 s bound, not the actor path's 300 s.
        assert clock.t == pytest.approx(600.0, abs=5.0)

    def test_a_free_gpu_still_short_circuits_without_stopping_anything(
            self, clock, capsys):
        orch = mp.FakeModelOrchestrator(
            load_times={"qwen_72b": 0.0},
            models={k: {"gpus": v["gpus"], "port": v["port"]}
                    for k, v in EXP3_MODELS.items()})
        ex = mp.ModelPrefetchExecutor(orch, probes=None, residency_actor=None)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)

        assert res["success"] is True
        assert not [c for c in orch.calls if c[0] == "stop_model"]
        assert "Proactive swap: GPUs free — loading qwen_72b." \
            in capsys.readouterr().out
        assert clock.t == 0.0            # no waiting at all

    def test_the_legacy_body_is_the_old_code_verbatim(self):
        """A guard on the move itself.

        `_proactive_swap_legacy` is the block that used to sit inline in
        `_load_model`, dedented and nothing else.  If someone 'fixes' it, this
        fails and they have to say so.
        """
        src = inspect.getsource(mp.ModelPrefetchExecutor._proactive_swap_legacy)
        body = src[src.index("        # Wait for the current model"):]
        assert "deadline = time.perf_counter() + 600.0" in body
        assert "self._orchestrator.stop_model(current)" in body
        assert "Proactive swap timeout: stopping {current} " in body
        assert "Proactive swap: GPUs free — loading {task.resource.name}." in body
        # It must not have grown a residency dependency.
        assert "_residency_actor" not in body

    def test_the_handover_grace_does_not_reach_the_legacy_path(self, clock):
        """Even an explicit override leaves the no-actor arm on 600 s."""
        orch = self._orch_with_incumbent()
        ex = mp.ModelPrefetchExecutor(orch, probes=None, residency_actor=None,
                                      handover_grace_s=5.0)
        ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)
        assert clock.t == pytest.approx(600.0, abs=5.0)


# ======================================================================
# 2. the actor path never stops anything blind
# ======================================================================

class TestActorPathEvictsThroughTheActor:

    def test_the_timeout_delegates_instead_of_calling_stop_model(
            self, rig, clock, capsys):
        """The fix.  The incumbent is serving and will not hand over; the
        eviction goes to the actor, which PARKS it and confirms the VRAM."""
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")
        orch.calls.clear()

        ex = _wired(orch, actor, serving=True)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)

        out = capsys.readouterr().out
        assert "Proactive swap timeout: stopping" not in out
        assert "delegating eviction to the residency actor" in out

        assert res["success"] is True, res
        assert res["evicted"] == ["qwen_32b"]
        # PARKED, not stopped: its weights are still in host RAM and its next
        # use is a wake, not a 782 s cold boot.
        assert orch.is_sleeping("qwen_32b") is True
        assert ("stop_model", "qwen_32b") not in orch.calls
        assert ("sleep_model", "qwen_32b", 1) in orch.calls
        assert clock.t == pytest.approx(mp.HANDOVER_GRACE_S, abs=5.0)

    def test_a_free_gpu_short_circuits_the_grace_period(self, rig, clock,
                                                        capsys):
        cg, orch, actor = rig
        ex = _wired(orch, actor, serving=True)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)
        assert res["success"] is True
        assert clock.t == 0.0
        assert "Proactive swap: GPUs free — loading qwen_72b." \
            in capsys.readouterr().out

    def test_the_grace_period_is_the_orchestrators_own_stop_budget(self):
        """Not a picked constant: stop_model's SIGINT grace (30 s,
        model_orchestrator.py:311) + the SIGKILL reap (10 s, :335) + the VRAM
        drain ceiling (240 s, :352).  The 240 s term is the one the three
        failed trials measured, hitting it every time."""
        assert mp.HANDOVER_GRACE_S >= 30.0 + 10.0 + 240.0
        assert mp.HANDOVER_GRACE_S == 300.0


# ======================================================================
# 3. an incumbent that is still coming up is not evicted at all
# ======================================================================

class TestABootingIncumbentIsLeftAlone:
    """The state all three failed trials were actually in.

    qwen_32b was mid cold boot for the ROUTER when the swap fired.  No
    voluntary handover was ever coming, and taking its GPUs discards a
    581-773 s weight load the current step is blocked on
    ("Model loading took 15.83 GiB memory and 581.144760 seconds",
    atomagents_exp3_aligned/tandem/t01__20260901-144241__f9b64ab/stdout.log).
    A prefetch is worth less than the step, so the swap is declined.
    """

    def test_the_swap_is_declined_and_nothing_is_touched(self, rig, clock,
                                                         capsys):
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")
        orch.calls.clear()

        ex = _wired(orch, actor, serving=False)      # mid bring-up
        task = _task("qwen_72b")
        res = ex._load_model(task)
        ex.shutdown(wait=False)

        assert res["success"] is False
        assert "declining the proactive swap to qwen_72b" in res["error"]
        assert "is NOT serving yet" in res["error"]
        assert task.status is PrefetchStatus.FAILED

        # The incumbent is untouched: not stopped, not parked, still there.
        assert ("stop_model", "qwen_32b") not in orch.calls
        assert not [c for c in orch.calls if c[0] == "sleep_model"]
        assert orch.is_sleeping("qwen_32b") is False
        assert orch.get_running_model() == "qwen_32b"
        assert "Proactive swap timeout: stopping" not in capsys.readouterr().out

    def test_an_unreadable_probe_does_not_decline(self, rig, clock):
        """None means 'could not tell', and a measurement we could not make is
        not evidence for refusing.  Unknown falls through to the actor, which
        has its own refusal."""
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")
        ex = _wired(orch, actor, serving=None)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)
        assert res["success"] is True
        assert res["evicted"] == ["qwen_32b"]

    def test_the_probe_reads_not_serving_when_the_port_refuses(self, rig):
        """The real `_incumbent_is_serving`, unstubbed, against a port with
        nothing behind it.  Connection refused is an answer, not an error."""
        cg, orch, actor = rig
        ex = mp.ModelPrefetchExecutor(orch, probes=None, residency_actor=actor)
        # EXP3_MODELS puts qwen_32b on 8012; nothing is listening in a test.
        assert ex._incumbent_is_serving("qwen_32b") is False
        # No port in the config at all is "could not tell", not "not serving".
        orch.models["portless"] = {"gpus": [0]}
        assert ex._incumbent_is_serving("portless") is None
        ex.shutdown(wait=False)


# ======================================================================
# 4. a target that fails to come up does not strand the workflow
# ======================================================================

class TestNoModelIsNeverLeftBehind:

    @staticmethod
    def _boot_fails(orch, name="qwen_72b"):
        real = orch.start_model_measured

        def boom(n, metrics=None):
            if n == name:
                raise RuntimeError(
                    "WorkerProc failed to start / server process exited rc=1")
            return real(n, metrics=metrics)

        orch.start_model_measured = boom

    def test_a_failed_target_wakes_the_parked_incumbent_back(
            self, rig, clock, capsys):
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")
        self._boot_fails(orch)

        ex = _wired(orch, actor, serving=True)
        task = _task("qwen_72b")
        res = ex._load_model(task)
        ex.shutdown(wait=False)

        assert res["success"] is False
        assert task.status is PrefetchStatus.FAILED
        # THE POINT: something serves again.  Without the guard the incumbent
        # stays parked and the workflow has no model at all.
        assert orch.is_sleeping("qwen_32b") is False
        assert orch.processes["qwen_32b"].poll() is None
        assert ("wake_model", "qwen_32b") in orch.calls
        assert "Restored qwen_32b" in capsys.readouterr().out

    def test_the_actor_alone_does_not_restore_on_the_activate_path(self, rig):
        """The control for the test above, and the answer to 'does
        restore_on_failure cover this path?'.  It does not: `_restore()` is
        reachable from stage() only (model_actor.py:1270, :1296), and
        activate() has no try/except around its wake/boot.  Called directly,
        the same failure leaves the victim parked and nothing serving.
        """
        cg, orch, actor = rig
        assert actor.restore_on_failure is True     # the flag IS on
        orch.start_model_measured("qwen_32b")
        self._boot_fails(orch)

        with pytest.raises(RuntimeError):
            actor.activate("qwen_72b")

        assert orch.is_sleeping("qwen_32b") is True     # still parked
        assert "qwen_72b" not in orch.processes        # and nothing serving

    def test_a_refusal_before_any_eviction_reports_that_it_changed_nothing(
            self, rig, clock, capsys):
        """When the actor refuses up front there is nothing to undo, and the
        restore path must say so rather than inventing a rollback."""
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")
        # Nothing may be taken off the GPUs at all.
        actor._evictable = lambda name: False

        ex = _wired(orch, actor, serving=True)
        res = ex._load_model(_task("qwen_72b"))
        ex.shutdown(wait=False)

        assert res["success"] is False
        assert "cannot activate qwen_72b" in res["error"]
        out = capsys.readouterr().out
        assert "failed before any eviction" in out
        # The incumbent never moved.
        assert orch.is_sleeping("qwen_32b") is False
        assert orch.get_running_model() == "qwen_32b"

    def test_a_stale_eviction_detail_does_not_wake_a_serving_engine(self, rig):
        """`last_eviction_detail` PERSISTS across swaps.  Replaying an old one
        would wake an engine that is already awake — the restore doing damage
        of its own.  Only what is parked NOW is touched."""
        cg, orch, actor = rig
        orch.start_model_measured("qwen_32b")        # awake, serving
        actor.last_eviction_detail["qwen_72b"] = {
            "parked": ["qwen_32b"], "stopped": []}   # stale
        orch.calls.clear()

        ex = _wired(orch, actor, serving=True)
        ex._restore_service("qwen_72b")              # must be a no-op
        ex.shutdown(wait=False)

        assert not [c for c in orch.calls if c[0] == "wake_model"]
        assert orch.is_sleeping("qwen_32b") is False

    def test_the_restore_never_raises(self, rig):
        """It runs inside an `except`; a second exception there would replace
        the real error with a worse one."""
        cg, orch, actor = rig
        ex = _wired(orch, actor, serving=True)
        actor.last_eviction_detail["qwen_72b"] = {"parked": ["ghost"],
                                                  "stopped": ["gone"]}
        actor._engines["ghost"] = ma._Engine(name="ghost", parked=True)
        ex._restore_service("qwen_72b")              # no exception
        ex._restore_service("never-swapped")         # no exception
        ex.shutdown(wait=False)


# ======================================================================
# 5. the executor still routes a non-proactive load unchanged
# ======================================================================

def test_a_plain_load_is_untouched_by_any_of_this(rig, clock):
    """Only `proactive_swap` tasks go near either swap path."""
    cg, orch, actor = rig
    orch.start_model_measured("qwen_32b")
    ex = _wired(orch, actor, serving=False)          # would DECLINE if consulted
    res = ex._load_model(_task("qwen_72b", proactive=False))
    ex.shutdown(wait=False)
    assert res["success"] is True
    assert res["evicted"] == ["qwen_32b"]
    assert clock.t == 0.0
