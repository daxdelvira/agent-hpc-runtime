#!/usr/bin/env python3
"""Ten-second check that the Tandem residency path is wired, before a hold is spent.

A tandem trial costs ~2.4 h. A wiring regression that fails on trial 1 would
burn the hold discovering something a ten-second import can tell us. Exits
non-zero and loudly if anything is off, so the job script can abort before
starting a trial rather than after.
"""
import sys

sys.path.insert(0, ".")
fail = []


def check(label, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"raised {exc!r}"
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}", flush=True)
    if not ok:
        fail.append(label)


def _actor():
    from runtime.residency.model_actor import VllmModelActor
    from runtime.residency.contract import ResidencyActor, ResourceClass

    class FakeOrch:
        def list_models(self):
            return {}

    a = VllmModelActor(FakeOrch())
    return (isinstance(a, ResidencyActor)
            and a.resource_class is ResourceClass.MODEL,
            f"constructs, satisfies protocol, class={a.resource_class}")


def _executor_off():
    from runtime.prefetch.model_prefetch import ModelPrefetchExecutor

    class FakeOrch:
        def list_models(self):
            return {}

    e = ModelPrefetchExecutor(FakeOrch(), probes=None, residency_actor=None)
    v = getattr(e, "can_evict_gpu_occupants", None)
    return v is False, f"can_evict_gpu_occupants={v} (must be False without an actor)"


def _executor_on():
    from runtime.prefetch.model_prefetch import ModelPrefetchExecutor
    from runtime.residency.model_actor import VllmModelActor

    class FakeOrch:
        def list_models(self):
            return {}

    e = ModelPrefetchExecutor(FakeOrch(), probes=None,
                              residency_actor=VllmModelActor(FakeOrch()))
    v = getattr(e, "can_evict_gpu_occupants", None)
    return v is True, f"can_evict_gpu_occupants={v} (must be True with an actor)"


def _flag():
    import subprocess
    out = subprocess.run([sys.executable, "experiments/atomagents_exp3.py", "--help"],
                         capture_output=True, text=True, timeout=120).stdout
    return "--residency" in out, "--residency present in exp3 CLI"


def _cgroup():
    from runtime.residency.model_actor import cgroup_mem
    m = cgroup_mem()
    ok = m.get("anon_gib", -1) >= 0
    return ok, (f"anon={m.get('anon_gib'):.2f} file={m.get('file_gib'):.2f} GiB "
                f"@ {m.get('path')}" if ok else f"UNREADABLE: {m}")


print("[preflight] Tandem residency path", flush=True)
check("VllmModelActor", _actor)
check("executor without actor", _executor_off)
check("executor with actor", _executor_on)
check("--residency flag", _flag)
check("cgroup readable (held_gb depends on it)", _cgroup)

if fail:
    print(f"\n[preflight] FAILED: {', '.join(fail)}", file=sys.stderr, flush=True)
    print("[preflight] refusing to spend the hold on a broken path", file=sys.stderr)
    raise SystemExit(1)
print("\n[preflight] all checks passed", flush=True)
