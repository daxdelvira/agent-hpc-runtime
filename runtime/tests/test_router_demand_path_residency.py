"""The demand swap must consult the residency actor, and must not regress
when there isn't one.

WHY THIS EXISTS.  Tandem trial t03 wired VllmModelActor, ran 3.2 h, performed
six model swaps, and issued ZERO `POST /sleep` and ZERO `POST /wake_up`.  The
actor was only on the PREFETCH path; every swap the workflow actually performs
comes through ModelRouter.ensure_ready().  Because `--residency` leaves
`sleep_level=0` (atomagents_exp3.py:755) the park branch at
model_router.py:150-165 short-circuits and every swap calls stop_model -- while
:654 still injects --enable-sleep-mode, so the engines pay the sleep-mode boot
cost for a park that never happens.  t03 came out 1.57x SLOWER than baseline
(11573.8 s vs a Blackwell mean of 7362.4 s).

These tests pin the three behaviours the fix has to have.  The third is the one
that protects the campaign: with no actor attached, the patched router must be
indistinguishable from the one that produced every trial collected so far.

SKIPS until the demand-path patch lands (the router has no set_residency_actor
until then), so this file is safe to commit ahead of it.
"""
import pathlib
import sys

import pytest

# The workload is a submodule and is not installed; there is no conftest that
# puts it on the path, so do it here rather than depend on how pytest was
# invoked.
_WL = pathlib.Path(__file__).resolve().parents[2] / "workloads" / "AtomAgents"
if _WL.is_dir() and str(_WL) not in sys.path:
    sys.path.insert(0, str(_WL))

ModelRouter = None
try:
    from atomagents.runtime.model_router import ModelRouter
except Exception:                                            # noqa: BLE001
    pass

pytestmark = pytest.mark.skipif(
    ModelRouter is None or not hasattr(ModelRouter, "set_residency_actor"),
    reason="AtomAgents not importable, or the demand-path residency patch is "
           "not applied yet "
           "(sc-workshop-paper/patches/tandem_demand_path_20260901.patch)",
)

MODELS = {"qwen_32b": {"port": 8001}, "qwen_72b": {"port": 8002}}


class GpusNotFreed(Exception):
    pass


class FakeOrch:
    """`processes` is what ensure_ready() consults first.  The target is
    deliberately absent from it, which is what routes control into the demand
    swap -- the block under test."""

    def __init__(self, running=()):
        self.calls: list[tuple[str, str]] = []
        self._running = list(running)
        self.processes = {n: object() for n in running}

    def is_sleeping(self, n):
        return False

    def _sleep_mode_enabled(self, n):
        return True

    def sleep_model(self, n, level=1):
        self.calls.append(("sleep", n))

    def stop_model(self, n):
        self.calls.append(("stop", n))
        if n in self._running:
            self._running.remove(n)

    def start_model_measured(self, n, metrics=None):
        self.calls.append(("start", n))
        self._running.append(n)

    def wait_until_ready(self, n):
        self.calls.append(("wait", n))


class FakeActor:
    def __init__(self, outcome):
        self.outcome = outcome
        self.seen: list[str] = []

    def activate(self, name):
        self.seen.append(name)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeMetrics:
    def __init__(self):
        self.rows: list[tuple[str, str]] = []

    def record(self, phase, duration_s, notes=None):
        self.rows.append((phase, notes))


def _router(orch, actor, metrics):
    r = ModelRouter(orch, MODELS, sleep_level=0)
    r.set_metrics(metrics)
    if actor is not None:
        r.set_residency_actor(actor)
    r._active_managed_models = lambda: [n for n in MODELS if n in orch._running]
    return r


def test_actor_handles_the_swap_and_the_orchestrator_is_never_touched():
    orch, m = FakeOrch(running=["qwen_32b"]), FakeMetrics()
    actor = FakeActor({"mechanism": "wake", "elapsed_s": 2.1,
                       "evicted": [{"model": "qwen_32b", "action": "park"}]})
    _router(orch, actor, m).ensure_ready("http://localhost:8002")

    assert actor.seen == ["qwen_72b"]
    # The whole point: no stop_model, so no cold boot to pay for.
    assert orch.calls == []
    # The mechanism reaches metrics.csv, so wake and cold_boot stay separable
    # in the trial data rather than both landing in one blocking_swap bucket.
    assert m.rows == [("model_swap_wait:qwen_72b", "residency_wake")]


def test_a_refusal_falls_back_and_still_completes_the_swap():
    """A refusal is EXPECTED at M=1/tp=4: qwen_72b parks at ~279 GB against a
    256 GB cgroup, so _can_park should decline.  The workflow cannot proceed
    without the model, so the fallback must still swap it in."""
    orch, m = FakeOrch(running=["qwen_32b"]), FakeMetrics()
    actor = FakeActor(GpusNotFreed("budget declined 279.0 GB for qwen_32b"))
    _router(orch, actor, m).ensure_ready("http://localhost:8002")

    assert orch.calls == [("stop", "qwen_32b"), ("start", "qwen_72b")]
    assert m.rows == [("model_swap_wait:qwen_72b", "blocking_swap")]


def test_without_an_actor_nothing_changes():
    """THE REGRESSION GUARD.  Every trial in the corpus was collected with no
    actor on this path; if that path moves, the baselines stop being baselines."""
    orch, m = FakeOrch(running=["qwen_32b"]), FakeMetrics()
    _router(orch, None, m).ensure_ready("http://localhost:8002")

    assert orch.calls == [("stop", "qwen_32b"), ("start", "qwen_72b")]
    assert m.rows == [("model_swap_wait:qwen_72b", "blocking_swap")]


# ---------------------------------------------------------------------------
# The actor has to be FOUND before it can be attached
# ---------------------------------------------------------------------------

def test_the_actor_is_found_on_the_composite_not_on_its_surface():
    """_build_executor returns a CompositeExecutor, and the actor is on its
    `vllm_model` CHILD, not on the composite itself.

    The first version of the wiring read `getattr(executor, "_residency_actor",
    None)` straight off the returned object.  That is always None, so
    --residency wired the actor and then quietly failed to give it to the
    router -- reproducing exactly the t03/t04 defect the patch was written to
    fix.  Caught 26 minutes into trial t05 of job 12720100 by the explicit
    warning rather than by another 3.2 h trial and a log dig.

    Pinned here because the failure is silent at the type level: both the
    composite and its children are PrefetchExecutors, so nothing complains.
    """
    from runtime.prefetch.data_prefetch import CompositeExecutor

    class _Bare:
        pass

    class _WithActor:
        def __init__(self, actor):
            self._residency_actor = actor

    def find(ex):
        a = getattr(ex, "_residency_actor", None)
        if a is not None:
            return a
        for child in (getattr(ex, "_executors", None) or {}).values():
            a = getattr(child, "_residency_actor", None)
            if a is not None:
                return a
        return None

    sentinel = object()
    comp = CompositeExecutor(
        executors={"vllm_model": _WithActor(sentinel), "data_file": _Bare()},
        default=_Bare())

    # The bug, pinned so it reads as intent rather than accident.
    assert getattr(comp, "_residency_actor", None) is None
    assert find(comp) is sentinel

    # And a run with no actor must still resolve to None, or every non-residency
    # arm would start attaching something.
    assert find(CompositeExecutor(executors={"data_file": _Bare()},
                                  default=_Bare())) is None
