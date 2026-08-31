#!/usr/bin/env python3
"""model_actor.py — T4a: the model residency actor, and the GPU-occupancy path.

`VllmModelActor` is the `ResidencyActor` for `ResourceClass.MODEL`. Parking is
vLLM **L1 sleep** (`POST /sleep?level=1`): the engine process stays alive, the
CUDA context and allocator stay alive, and the weights move OUT of VRAM into
**anonymous host memory**. That is rung R2_PROCESS_BYTES. Waking is a host-to-
device copy — 782.27 s cold boot against a 2.076 s wake, measured
(`results/bench_wake_L1_coherence_32b.json`, register M1).

WHY THIS FILE EXISTS AT ALL — IT IS THE CAUSE OF THE NEGATIVE END-TO-END RESULT
-------------------------------------------------------------------------------
On the L40S rows of `atomagents_exp3_aligned`, **every model prefetch that was
admitted failed** with

    Cannot start qwen_32b: GPUs [0,1,2,3] occupied by qwen_72b. Call stop_model first.

(`workloads/AtomAgents/atomagents/runtime/model_orchestrator.py:594-598`.)
All three exp3 models declare `gpus: [0,1,2,3]` at tp=4, so **M = 1**: reaching
R2 for one model REQUIRES first taking the GPU away from the incumbent. The
precursor predicted correctly and had nowhere to put the result. So the actor
is not "sleep/wake plus a bit of plumbing" — the plumbing IS the contribution:
`_free_gpus_for()` parks or stops the occupant, confirms the VRAM actually came
back, and when it cannot, says so and stages nothing.

Re-derived from the shipped data rather than quoted from the plan
(`results/eval_q1_q4/eval_prefetch_lifecycle.csv`, filtered to
`workload=atomagents_exp3_aligned` and `gpu_name` containing `L40S`, 143 rows):
16 of the 20 admissions are `vllm_model` and ALL 16 have
`completion_status=failed`; the other 4 are data_file (3 completed, 1 pending).
Ten of the sixteen are the `confidence_above_threshold` admissions and they die
in <= 0.001 s. The other six are `naive_prefetch`, and THOSE DO NOT FAIL FAST:
the proactive-swap entries sit in the executor's 600 s "wait for the GPUs to
free" loop and fail after 600.02 s, 600.02 s, 600.03 s and 918.04 s. So the
cost of admitting a swap with no eviction path is not only an instant error --
it is also a ten-minute wait for something nobody was ever going to do.

(The plan's "10 of 13 admitted prefetches" does not reproduce on this facet:
the L40S admissions number 20, of which 16 are models and all 16 failed. The
direction of the finding is unchanged and the mechanism is identical.)

THE TWO MEASUREMENT TRAPS THIS FILE IS BUILT AROUND (invariant I1)
-----------------------------------------------------------------
1. **Not `/proc/meminfo`.** `MemTotal - MemAvailable` is a HOST-WIDE reading on
   a SHARED node, so a neighbour's allocation lands in our number. The park
   ratio on record (1.90x, 120.77 GiB for a 68.28 GB model) was taken that way
   and is trust-**B** in `sc-workshop-paper/measurement_provenance.md` (M2) for
   exactly this reason. `host_ram_used_gib()` is kept here only as the same
   kind of cross-check the H1 bench keeps it as.
2. **Not `memory.current`.** It includes PAGE CACHE, and getting a model parked
   means first reading ~146 GB of weight shards — a delta on `memory.current`
   would count the whole file cache as parked weights. L1 parks into ANONYMOUS
   memory, which is why a measured wake is independent of page-cache state
   (evicting 18 shards moved wake by 0.015 s, register M3).

So `measure_held_gb()` reads **`anon` from the cgroup's `memory.stat`**. The
helper is copied from `experiments/bench_h1_quantized_park.py` — the instrument
that produced the H1 numbers — rather than rewritten, so the actor and the bench
read the same column. The only addition is an optional `root=` for tests.

And the park delta is taken against the **awake** reading, never a pre-launch
one, so the engine's own baseline allocations (CUDA context, allocator arenas,
the Python interpreter) are not charged to the park.

WHAT A CGROUP COUNTER CANNOT DO, STATED RATHER THAN PAPERED OVER
----------------------------------------------------------------
`memory.stat`'s `anon` is a property of the CGROUP, not of one engine. With a
single engine — the M=1 case, which is the real case here — `measure_held_gb()`
is a genuine live re-measurement: `anon_now - anon_awake(this engine)`. With two
engines parked simultaneously it is not: the number returned is then the delta
measured across THAT resource's own park transition, i.e. additive attribution,
and `reconcile_detail()` reports the residual between the sum of attributions
and the cgroup's own movement. It reports; it never corrects. A per-process
witness (`Anonymous:` from `/proc/<pid>/smaps_rollup`, summed over the engine's
process tree) is recorded alongside every transition, because it IS per-resource
and it is the thing that would settle the attribution question — see the GPU
validation note at the bottom of this docstring.

COHERENCE IS NOT OPTIONAL, AND A 200 IS NOT A WAKE
---------------------------------------------------
Level-2 sleep produced verbatim `"!!!!"` degeneracy on this cluster while
returning HTTP success. A footprint number from an engine that wakes incoherent
is worthless. So:

  * `park_level` is **1**. Level 2 is refused unless a caller explicitly passes
    `allow_unverified_level=True`, because "sleep works" is a per-level claim
    and only L1 has been verified.
  * Every boot and every wake runs `coherence_probe()`, which asserts generated
    TEXT and a `finish_reason` — never just a 200.
  * The bench's `ok` rule (non-empty text + finish_reason) would have ACCEPTED
    `"!!!!"`: it is non-empty and it terminates.
  * A COMPOSITION HEURISTIC IS NOT ENOUGH EITHER, and this is measured, not
    argued. The real fp8 job answered "The capital of France is" with
    `"\u306f.   1111               "` (finish_reason=length, 24 tokens). It has
    alphanumerics and three distinct non-space characters, so both the original
    rule and the degeneracy repair pass it. No statistic over character
    composition can separate broken output from terse-but-correct output --
    only knowing the answer can.
  * So the verdict is three tests, cheapest first, and only the last is
    load-bearing: text-and-finish_reason, then `_is_degenerate()`, then
    `PROBE_MUST_CONTAIN = "Paris"`, tied to `PROBE_PROMPT` and changed with it.
    A reference comparison against the text the SAME engine produced at boot is
    recorded on top (M1's evidence for L1 is "identical text") but is not
    fatal, because greedy decoding on a batching engine can flip a token.
  * An INJECTED probe is judged too. A probe is a measurement instrument, not
    an authority on whether the engine works, and an instrument built on
    composition statistics is exactly the thing the fp8 run defeated.

WHAT `stage()` RETURNS, AND WHY IT SOMETIMES UNDOES ITSELF
-----------------------------------------------------------
`stage()` returns the rung ACTUALLY reached (contract):

    R2_PROCESS_BYTES  parked at L1, anon delta measured.  The goal.
    R3_ACTIVATED      the engine was ALREADY live when we were called and the
                      park failed, so it is still serving on the GPU: more of
                      the chain is paid than R2, but the requested residency
                      was not reached. detail["parked"] is False.
    R0_DISK           nothing is held.  Either the GPUs could not be freed, or
                      the boot failed, or we booted it ourselves and could not
                      park it — in which case the engine we created is STOPPED
                      again, because a caller who asked for a parked model did
                      not ask for an engine sitting on the GPU.

Two refusals happen BEFORE any eviction, deliberately:
  * if the target's config has no `--enable-sleep-mode`, it can never reach R2,
    so evicting the incumbent for it would spend a real cold boot to hold
    nothing;
  * if `max_parked` is already reached. k=3 simultaneous L1 sleeps was measured
    NOT to complete (register M6: timeout/HTTP 500, no OOM observed), so the
    default is 2 and it is a constructor argument rather than a hidden rule.

If a stage fails AFTER victims were parked, `restore_on_failure` wakes them back
(2.076 s) rather than leaving the workflow with a slept planner it never asked
to lose. Victims that had to be STOPPED cannot be restored cheaply, and the
detail says so instead of pretending otherwise.

GPU VALIDATION THIS FILE WANTS (it has not had any; everything below is fakes)
------------------------------------------------------------------------------
  V0  WHICH COLUMN A COHERENT L1 PARK LANDS IN. Now the first question, not a
      detail: the only cgroup-instrumented park on record moved 82.04 GiB into
      `file` and 0.00 into `anon` (see PARK_BACKING_L1). If that holds for a
      coherent fp16 engine, this actor charges the wrong column and refuses
      every park; if it does not, the fp8 run is an artefact of that engine and
      `anon` is right. Either answer is cheap and decisive, and nothing else
      here means much until it is answered.
  V1  cgroup `anon` park delta vs the process-tree `Anonymous` sum for the same
      engine. If they agree, per-resource attribution is exact and the M2
      trust-B park ratio is upgraded to a cgroup measurement.
  V2  does the host-side backup SURVIVE a wake? The first park costs 23.692 s
      and later ones 2.611 s because the backup is allocated once and reused,
      which implies an awake engine still holds it — i.e. the budget is charged
      while serving. `measure_held_gb()` measures this rather than assuming it
      (`park_detail()["anon_gib_awake_after_wake"]`), but nothing has run it.
  V3  the four-way release proof (below) against a real engine.

INVARIANT I2. `release()` stops the engine and returns the GB the OS ACTUALLY
gave back, measured as a cgroup `anon` delta across the teardown — not the
number we charged, and not a by-construction "the process exited, therefore the
memory came back". It waits for the whole process TREE to leave `/proc`, and
records the per-process witness taken immediately before teardown so a
disagreement between the two is visible rather than rounded away.

`release_witness()` is the other half of that, added to the contract on
2026-08-30 after this actor showed the hole: for a TEARDOWN actor,
`measure_held_gb()` after a release is 0.0 BY CONSTRUCTION, so the ledger's
before/after drop is tautologically the whole charge and confirms nothing. The
witness is a reading of the ENCLOSING ALLOCATION that does not require the
released process to still exist. It is stamped with the release it belongs to
and revoked when the model is staged again, because `last_release_detail`
persists as evidence and a replayed delta would vouch for a release it never
observed — the same tautology, one level up.

NOT IN SCOPE: the ledger and arbitrator (T1/T2, A1), the horizon (T3, A4), the
data worker (T4b, A2). `contract.py` signatures are implemented exactly as
written; nothing here changes them.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

if __package__ in (None, ""):                        # `python model_actor.py`
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.residency.contract import (             # noqa: E402
    ReleaseNotHonoured,
    ResourceClass,
    ResourceSpec,
    Rung,
)

__all__ = [
    "VllmModelActor",
    "IncoherentWake",
    "GpusNotFreed",
    "ParkRefused",
    "ParkNotMeasurable",
    "cgroup_mem",
    "cgroup_anon_gib",
    "cgroup_anon_gb",
    "cgroup_file_gb",
    "coherence_probe",
    "judge_probe",
    "PROBE_PROMPT",
    "PROBE_MUST_CONTAIN",
    "PARK_BACKING_L1",
    "PARK_FLOOR_GB",
    "model_specs",
    "MODEL_CATALOGUE",
    "PARK_RATIO",
    "FIRST_PARK_S",
    "SUBSEQ_PARK_S",
    "DEFAULT_MAX_PARKED",
]


# ===========================================================================
# Measured constants — every one of these names its source
# ===========================================================================

# E4 / register M2. TRUST B: taken with MemTotal-MemAvailable, which is
# host-wide on a shared node. It is here for projection arithmetic only; the
# actor never uses it to decide what it holds, because it measures instead.
PARK_RATIO = 1.90                # 120.77 GiB parked for 68.28 GB of weights

# E4, via scripts/replay_retention_policy.py:54-55. A first park allocates the
# host-side backup; later parks reuse it. ~9x. A scheduler amortising over a
# workflow needs both, so both are recorded per engine.
FIRST_PARK_S = 23.692
SUBSEQ_PARK_S = 2.611

# Register M6: k=3 simultaneous L1 sleeps did NOT complete on this cluster
# (timeout / HTTP 500; no OOM was ever observed). k=2 did. So 2 is the default
# and it is an argument, not a buried rule.
DEFAULT_MAX_PARKED = 2

# The catalogue, from sc-workshop-paper/results_tables/00_README.md:49-51 —
# the same rows A1's ledger and arbitrator tests are written against.
#   name             held_gb  cold_s  ready_s  -> s/GB
MODEL_CATALOGUE: dict[str, tuple[float, float, float]] = {
    "qwen_32b":      (129.7, 495.2, 1.03),      # 3.81 s/GB
    "qwen_72b":      (279.0, 800.5, 2.21),      # 2.86 s/GB
    "qwen_72b_text": (276.3, 770.3, 2.19),      # 2.78 s/GB
}

PROBE_PROMPT = "The capital of France is"
# THE SEMANTIC ANCHOR, and the reason it has to exist.
#
# The H1 bench's original rule was "non-empty text plus a finish_reason", which
# accepts "!!!!".  The obvious repair is a composition statistic -- no
# alphanumeric character, or too few distinct characters.  The real fp8 job then
# produced, for this exact prompt:
#
#     "\u306f.   1111               "   finish_reason=length, 24 tokens
#
# which passes BOTH rules: it contains alphanumerics ("1111") and three distinct
# non-space characters.  So the generalisation is that NO statistic over
# character composition can separate broken output from terse-but-correct
# output -- only knowing the answer can.  The heuristic stays as a cheap first
# filter; this is what actually decides.
#
# It is tied to PROBE_PROMPT and must be changed with it.
PROBE_MUST_CONTAIN = "Paris"


def model_specs(catalogue: Optional[dict] = None) -> dict[str, ResourceSpec]:
    """The MODEL-class ResourceSpecs for the ledger to charge.

    held_gb here is DECLARED, and it is declared from a trust-B instrument (M2,
    host-wide MemTotal-MemAvailable). It is the fallback `ResidencyLedger.charge`
    uses when an actor cannot measure yet, and every such charge is recorded in
    `declared_charges`. Once an engine is actually parked, `measure_held_gb()`
    supersedes it — that is the whole point of I1.
    """
    cat = catalogue or MODEL_CATALOGUE
    out: dict[str, ResourceSpec] = {}
    for name, (gb, cold_s, ready_s) in cat.items():
        out[name] = ResourceSpec(
            resource_id=name,
            resource_class=ResourceClass.MODEL,
            held_rung=Rung.R2_PROCESS_BYTES,
            held_gb=float(gb),
            cold_s=float(cold_s),
            ready_s=float(ready_s),
        )
    return out


# ===========================================================================
# Measurement primitives (invariant I1)
# ===========================================================================
# _cgroup_dir() and cgroup_mem() are COPIED from
# experiments/bench_h1_quantized_park.py rather than rewritten, so the actor
# and the H1 bench read the same column of the same file. The only change is
# the optional `root=` argument, which exists so the tests can point the reader
# at a synthetic memory.stat and prove which column is being used.


def _cgroup_dir(root: Optional[Path] = None) -> tuple[Optional[Path], str]:
    """This process's cgroup v2 directory, walking up to one with memory.stat.

    Works under a SLURM step cgroup on a compute node and under user.slice on a
    login node. Returns (None, reason) rather than guessing -- a silently wrong
    budget reading is how the L2 sleep result got misread.
    """
    if root is not None:
        p = Path(root)
        return (p, str(p)) if (p / "memory.stat").exists() else (
            None, f"no memory.stat under {p}")
    try:
        rel = ""
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2]
                    break
        if not rel:
            return None, "no cgroup v2 entry in /proc/self/cgroup"
        node = Path("/sys/fs/cgroup") / rel.lstrip("/")
        while True:
            if (node / "memory.stat").exists():
                return node, str(node)
            if node == Path("/sys/fs/cgroup"):
                return None, "no memory.stat found walking to cgroup root"
            node = node.parent
    except Exception as exc:                          # noqa: BLE001
        return None, f"cgroup read failed: {exc!r}"


def cgroup_mem(root: Optional[Path] = None) -> dict:
    """anon / file / current for this cgroup, in GiB.

    **anon is the number that matters, and this distinction is the whole
    measurement.** memory.current includes PAGE CACHE, and getting a model
    parked means first READING ~146 GB of weight shards -- so a delta taken on
    memory.current would count the entire file cache as though it were parked
    weights. An L1 sleep holds the weights in ANONYMOUS memory (which is
    precisely why wake is independent of page-cache state: evicting 18 shards
    changed a measured wake by 0.015 s). So anon is what a park costs the
    budget, and file is recorded alongside only to make the confound visible
    rather than invisible.
    """
    d, path = _cgroup_dir(root)
    if d is None:
        return {"anon_gib": -1.0, "file_gib": -1.0, "current_gib": -1.0,
                "path": path}
    out: dict = {"path": str(d)}
    try:
        stat = {}
        for line in (d / "memory.stat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in ("anon", "file"):
                stat[k] = int(v)
        out["anon_gib"] = stat.get("anon", 0) / 1024 ** 3
        out["file_gib"] = stat.get("file", 0) / 1024 ** 3
    except Exception as exc:                          # noqa: BLE001
        out["anon_gib"] = out["file_gib"] = -1.0
        out["error"] = repr(exc)
    try:
        out["current_gib"] = int(
            (d / "memory.current").read_text().strip()) / 1024 ** 3
    except Exception:                                 # noqa: BLE001
        out["current_gib"] = -1.0
    return out


def cgroup_anon_gib(root: Optional[Path] = None) -> float:
    return cgroup_mem(root)["anon_gib"]


def cgroup_file_gb(root: Optional[Path] = None) -> float:
    """`file` from memory.stat, in GB. NOT a budget number -- it is page cache
    plus every SHARED/tmpfs mapping, and shard reads land here. It is measured
    across the park transition for one reason: to catch a park that went
    somewhere other than anonymous memory. See PARK_BACKING_L1 below."""
    g = cgroup_mem(root)["file_gib"]
    return -1.0 if g < 0 else g * 1024 ** 3 / 1e9


def cgroup_anon_gb(root: Optional[Path] = None) -> float:
    """The same reading in GB, which is the unit the contract budgets in.

    GiB vs GB is a 7.4% difference on a 279 GB model — 20 GB of budget. The
    ledger, ResourceSpec.held_gb and Eq. 1 are all in GB, so the conversion
    happens once, here, rather than in five call sites.
    """
    g = cgroup_anon_gib(root)
    return -1.0 if g < 0 else g * 1024 ** 3 / 1e9


def host_ram_used_gib() -> float:
    """Kept ONLY as a cross-check against the cgroup reading, never as the
    primary number. Host-wide on a shared node; see the module docstring."""
    try:
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024
    except Exception:                                 # noqa: BLE001
        return -1.0


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _child_pids(pid: int) -> list[int]:
    """Direct children of `pid`, from /proc/<pid>/task/*/children."""
    out: list[int] = []
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            try:
                out.extend(int(x) for x in
                           (task / "children").read_text().split())
            except (OSError, ValueError):
                continue
    except (OSError, ValueError):
        pass
    return out


def process_tree(pid: int, _depth: int = 0) -> list[int]:
    """`pid` and every descendant. vLLM at tp=4 is one api_server plus a
    handful of worker processes, and the parked weights live in the WORKERS —
    so a per-process footprint that stopped at the api_server would read ~0
    and make a real park look like nothing happened."""
    if _depth > 8 or not pid_alive(pid):
        return []
    out = [pid]
    for c in _child_pids(pid):
        out.extend(process_tree(c, _depth + 1))
    return out


def read_smaps_rollup(pid: int) -> dict[str, float]:
    """Per-process page accounting for `pid`, in GB.

    `Anonymous` is the per-process analogue of the cgroup's `anon` — the same
    column, one level down. Rss is returned for reference only; it
    double-counts shared pages and, in the COW benchmark, would have made a
    failed copy-on-write look like a success.
    """
    out: dict[str, float] = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("Rss", "Pss", "Anonymous", "Private_Dirty",
                         "Private_Clean", "Shared_Clean", "Swap"):
                    out[k] = int(v.split()[0]) / 1e6       # kB -> GB
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return out


def tree_anon_gb(pid: int) -> float:
    """Anonymous memory held by `pid` and its descendants, in GB.

    This is the PER-RESOURCE witness. It is not the primary number (the
    instruction for this actor is the cgroup's `anon`, and the cgroup is what
    the budget is an allocation of), but it is the only reading that can
    attribute anon to one engine when two are parked, so it is recorded on
    every transition and reported next to the cgroup delta.
    """
    total = 0.0
    for p in process_tree(pid):
        m = read_smaps_rollup(p)
        total += m.get("Anonymous", m.get("Private_Dirty", 0.0))
    return total


# ===========================================================================
# Coherence (the "!!!!" trap)
# ===========================================================================

# WHICH COLUMN AN L1 PARK ACTUALLY LANDS IN -- AN OPEN QUESTION, NOT A FACT
# ---------------------------------------------------------------------------
# The build plan says L1 parks weights into ANONYMOUS host memory, and this
# actor was written to that. The only cgroup-instrumented park on record says
# otherwise. `results/bench_h1_fp8_tp1_12561711.json` (job 12561711, node
# atl1-1-02-014-23-0, Qwen2.5-72B, --quantization fp8, tp=1), read with
#   python3 -c "import json; d=json.load(open('results/bench_h1_fp8_tp1_12561711.json')); print(d['rows'])"
#
#     awake   anon 2.77 GiB   file  1.32 GiB   VRAM 87251 MiB
#     slept   anon 2.77 GiB   file 83.36 GiB   VRAM  1267 MiB
#     ------------------------------------------------------------------
#     anon delta 0.00 GiB     file delta +82.04 GiB   VRAM freed 85.9 GiB
#
# The park is REAL -- 85.9 GiB of VRAM came back and a second park took 1.44 s
# against the first's 44.30 s, which is the backup-reuse signature -- but it
# landed in `file`, not `anon`, and the bench therefore reported held_gb 0.0.
# A 0.0 charge is not a small error: it tells the arbitrator that holding a 72B
# is free, which would make every retention decision in the paper meaningless.
#
# The 1.90x park ratio (register M2) cannot settle this: it was taken with the
# host-wide MemTotal-MemAvailable instrument, which does not distinguish the
# two columns. So there is currently NO measurement showing an L1 park in anon.
# The most likely explanation is that vLLM's CuMemAllocator host backup is a
# SHARED mapping (tmpfs / /dev/shm), which cgroup v2 accounts under `file`.
#
# Until a coherent engine settles it, this actor CHARGES `anon` and REFUSES to
# report a park it could not measure there -- see ParkNotMeasurable. It does not
# quietly switch columns: `file` also holds the 146 GB of weight shards read
# during boot, so charging it by default would resurrect the page-cache trap
# this whole file is built to avoid.
PARK_BACKING_L1 = "unsettled: anon on the plan, file in the only cgroup-instrumented run"

# Below this, a delta is noise rather than a park.
PARK_FLOOR_GB = 1.0


class ParkNotMeasurable(RuntimeError):
    """The park happened but not in the column this actor charges.

    Raised when VRAM came back and `anon` did not move while `file` did. The
    alternative is to charge 0.0 GB for a parked 72B, which is the loudest
    possible I1 violation dressed up as a success.
    """


class IncoherentWake(RuntimeError):
    """An engine returned HTTP success and did not generate usable text.

    Level-2 sleep produced verbatim "!!!!" on this cluster while returning 200.
    A footprint or a wake time from such an engine is worthless, so this is an
    error and not a warning.
    """


class GpusNotFreed(RuntimeError):
    """The GPUs a model needs are held by an engine that could not be evicted.

    Carries WHO holds them and WHY it could not be moved, because the whole
    point of T4a is that the previous failure mode
    ("Cannot start X: GPUs [...] occupied by Y") named the symptom and stopped.
    """


class ParkRefused(RuntimeError):
    """A park was asked for that this actor will not perform.

    Level 2 without an explicit override (unverified on this cluster), or a
    park beyond `max_parked` (k=3 was measured not to complete).
    """


_ALNUM = re.compile(r"[A-Za-z0-9]")


def _is_degenerate(text: str) -> bool:
    """True for the failure mode L2 actually produced: `"!!!!"`.

    The H1 bench's rule -- non-empty text plus a finish_reason -- would ACCEPT
    "!!!!", because it is non-empty and it terminates. That gap is real and
    this is what closes it. Two independent signs, either of which is enough:
    no alphanumeric character at all, or a single character repeated.
    """
    s = text.strip()
    if not s:
        return True
    if not _ALNUM.search(s):
        return True
    return len(set(s.replace(" ", ""))) <= 1


def _post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def judge_probe(text: str, finish_reason, must_contain: Optional[str]) -> dict:
    """The verdict, in one place: composition first, then the ANSWER.

    Three tests, cheapest first, and only the last one is load-bearing:
      1. text and a finish_reason at all -- a 200 is not a wake;
      2. not degenerate by composition -- catches "!!!!";
      3. it contains the answer to the prompt -- catches
         "\u306f.   1111               ", which the first two do not.
    """
    ok = bool(str(text).strip()) and finish_reason is not None
    degenerate = _is_degenerate(text)
    anchored = None
    if must_contain:
        anchored = must_contain.lower() in str(text).lower()
    return {
        "text": text,
        "finish_reason": finish_reason,
        "degenerate": degenerate,
        "anchor": must_contain,
        "anchored": anchored,
        "ok": ok and not degenerate and (anchored is not False),
    }


def coherence_probe(port: int, served_model: str, max_tokens: int = 24,
                    prompt: str = PROBE_PROMPT, timeout: float = 300.0,
                    must_contain: Optional[str] = PROBE_MUST_CONTAIN) -> dict:
    """Assert generated TEXT, a finish_reason, and THE ANSWER -- never just a 200.

    This is the check that separated the real L1 result from the fake L2 one.
    Temperature 0 and a fixed seed so the text is comparable against the same
    engine's earlier answer -- M1's evidence for L1 is "identical text", so a
    reference comparison is the honest bar for a wake.
    """
    d = _post_json(f"http://127.0.0.1:{port}/v1/completions",
                   {"model": served_model, "prompt": prompt,
                    "max_tokens": max_tokens, "temperature": 0.0, "seed": 0},
                   timeout)
    ch = d["choices"][0]
    out = judge_probe(ch.get("text", ""), ch.get("finish_reason"), must_contain)
    out["n_tokens"] = d.get("usage", {}).get("completion_tokens")
    return out


# ===========================================================================
# Per-engine bookkeeping
# ===========================================================================

@dataclass
class _Engine:
    """Everything the actor measured about one model, kept next to the model.

    `anon_awake_gb` is the reading the park delta is taken AGAINST. Taking it
    against a pre-launch reading would charge the engine's own baseline
    allocations to the park.
    """
    name: str
    spec: Optional[ResourceSpec] = None
    pid: Optional[int] = None
    parked: bool = False
    anon_pre_boot_gb: Optional[float] = None   # cgroup anon before the engine existed
    anon_awake_gb: float = 0.0        # cgroup anon while awake, before a park
    file_awake_gb: float = 0.0        # cgroup file while awake (diagnostic only)
    file_parked_gb: float = 0.0       # cgroup file while parked (diagnostic only)
    park_backing: Optional[str] = None   # which column the park actually moved
    anon_parked_gb: float = 0.0       # cgroup anon while parked
    held_gb: float = 0.0              # measured park delta (the charge)
    tree_anon_awake_gb: float = 0.0   # per-process witness, awake
    tree_anon_parked_gb: float = 0.0  # per-process witness, parked
    park_count: int = 0
    first_park_s: Optional[float] = None
    subsequent_park_s: list[float] = field(default_factory=list)
    wake_s: list[float] = field(default_factory=list)
    boot_s: Optional[float] = None
    probe_reference: Optional[str] = None
    last_probe: dict = field(default_factory=dict)
    anon_awake_after_wake_gb: Optional[float] = None   # V2: does the backup survive?

    def park_cost_summary(self) -> dict:
        return {
            "park_count": self.park_count,
            "first_park_s": self.first_park_s,
            "subsequent_park_s": list(self.subsequent_park_s),
            "mean_subsequent_park_s": (
                sum(self.subsequent_park_s) / len(self.subsequent_park_s)
                if self.subsequent_park_s else None),
            "amortisation_note": (
                "a first park allocates the host-side backup and later parks "
                "reuse it; measured 23.692 s then 2.611 s (~9x) on a 32B "
                "(scripts/replay_retention_policy.py:54-55)"),
        }


# ===========================================================================
# The actor
# ===========================================================================

class VllmModelActor:
    """ResidencyActor for ResourceClass.MODEL. L1 sleep plus the GPU path.

    The orchestrator is injected, and only these members of it are used:
    `models`, `processes`, `is_sleeping`, `sleep_model`, `wake_model`,
    `stop_model`, `start_model_measured`, `wait_until_serving` and (optionally)
    `get_gpu_memory`. That is deliberately the same surface
    `FakeModelOrchestrator` already implements, so the whole of this file is
    testable with no GPU.

    NOTHING HERE MODIFIES THE ORCHESTRATOR. The refusal at
    `model_orchestrator.py:594-598` is left exactly where it is; it stops firing
    because the actor frees the GPUs before it calls through. That file belongs
    to the workload, and a residency policy does not belong inside it.

    Callbacks, all optional, all defaulting to the conservative answer:
      can_park(name, est_gb) -> bool
          the BUDGET question, which is the ledger's and the arbitrator's, not
          the actor's (invariant I4). Default: allowed. Answering False makes
          the actor STOP a victim instead of parking it, and record that the
          downgrade was forced by the budget rather than preferred.
      evictable(name) -> bool
          may this engine be taken off the GPU at all? Default: yes, except the
          target. An engine serving the workflow's current request is the
          caller's to protect; the actor cannot see requests.
    """

    def __init__(
        self,
        orchestrator,
        id_to_model: Optional[dict[str, str]] = None,
        park_level: int = 1,
        allow_unverified_level: bool = False,
        max_parked: int = DEFAULT_MAX_PARKED,
        can_park: Optional[Callable[[str, float], bool]] = None,
        evictable: Optional[Callable[[str], bool]] = None,
        probe: Optional[Callable[[str], dict]] = None,
        must_contain: Optional[str] = PROBE_MUST_CONTAIN,
        anon_reader: Optional[Callable[[], float]] = None,
        file_reader: Optional[Callable[[], float]] = None,
        charge_columns: tuple = ("anon",),
        restore_on_failure: bool = True,
        release_confirm_frac: float = 0.90,
        teardown_timeout_s: float = 60.0,
        strict_coherence: bool = True,
        verbose: bool = False,
    ) -> None:
        if park_level != 1 and not allow_unverified_level:
            raise ParkRefused(
                f"park_level={park_level} is not verified on this cluster. "
                f"Level-2 sleep produced verbatim '!!!!' degeneracy here while "
                f"returning HTTP success; only L1 has been shown to wake "
                f"coherently (register M1). Pass allow_unverified_level=True "
                f"if you are deliberately measuring the unverified level.")
        self.orch = orchestrator
        self.id_to_model = dict(id_to_model or {})
        self.park_level = park_level
        self.max_parked = max_parked
        self._can_park = can_park
        self._evictable = evictable
        self._probe = probe
        # The semantic anchor is applied to EVERY probe result that carries
        # text, including an injected one. An injected probe is a measurement
        # instrument, not an authority on whether the engine works, and the fp8
        # run is the reason: two composition heuristics passed an engine that
        # answered this prompt with "\u306f.   1111".
        self.must_contain = must_contain
        self._anon = anon_reader or cgroup_anon_gb
        # `file` is read for ONE purpose: to notice a park that did not land in
        # `anon`. It is never added to the charge unless a caller deliberately
        # asks, because `file` also carries the 146 GB of weight shards read at
        # boot. See PARK_BACKING_L1.
        self._file = file_reader or cgroup_file_gb
        if not set(charge_columns) <= {"anon", "file"}:
            raise ValueError(f"charge_columns must be anon and/or file, "
                             f"got {charge_columns!r}")
        self.charge_columns = tuple(charge_columns)
        self.restore_on_failure = restore_on_failure
        self.release_confirm_frac = release_confirm_frac
        # How long to wait for the whole engine process TREE to leave /proc.
        # After a SIGKILL of a mid-loading vLLM, CUDA driver cleanup has been
        # observed to take 2-3 minutes (model_orchestrator.py:345-347), so this
        # is generous by default and short in tests.
        self.teardown_timeout_s = teardown_timeout_s
        self.strict_coherence = strict_coherence
        self.verbose = verbose

        self._engines: dict[str, _Engine] = {}
        self.last_stage_detail: dict[str, dict] = {}
        self.last_release_detail: dict[str, dict] = {}
        # Staleness control for release_witness(). `last_release_detail`
        # PERSISTS, so without a stamp a later call would hand back an old
        # cgroup delta that looks exactly like a fresh witness -- vouching for
        # a release it never observed, which is the same tautology the witness
        # exists to break. Each teardown gets a sequence number; a token is
        # valid only while it names the newest release of that resource, and
        # re-staging the model revokes it.
        self._release_seq = 0
        self._witness_token: dict[str, int] = {}
        self.last_eviction_detail: dict[str, dict] = {}
        self.coherence_failures: list[dict] = []
        # The cgroup reading when this actor was constructed. reconcile_detail()
        # measures the sum of what we attributed against the cgroup's own
        # movement from here; a gap is the budget and reality parting company.
        self.anon_at_start_gb = self._anon()

    # -- small helpers ----------------------------------------------------

    def _log(self, *a) -> None:
        if self.verbose:
            print("[model_actor]", *a, flush=True)

    def _models(self) -> dict:
        return getattr(self.orch, "models", {}) or {}

    def _procs(self) -> dict:
        return getattr(self.orch, "processes", {}) or {}

    def model_for(self, resource_id: str) -> str:
        """resource_id -> orchestrator model key. Loud when it cannot.

        Accepts the model name itself, an explicit map, or the `_parked` /
        `_r2` suffixes a catalogue might use for the held form.
        """
        if resource_id in self.id_to_model:
            return self.id_to_model[resource_id]
        models = self._models()
        if resource_id in models:
            return resource_id
        for suffix in ("_parked", "_r2", "_activated"):
            if resource_id.endswith(suffix) and resource_id[:-len(suffix)] in models:
                return resource_id[:-len(suffix)]
        raise KeyError(
            f"{resource_id}: no model of that name in the orchestrator "
            f"(known: {sorted(models)}). Pass id_to_model={{'{resource_id}': "
            f"'<model>'}} rather than letting the actor guess — a guess here "
            f"would park the wrong engine.")

    def _alive(self, name: str) -> bool:
        p = self._procs().get(name)
        if p is None:
            return False
        poll = getattr(p, "poll", None)
        return poll() is None if callable(poll) else True

    def _pid(self, name: str) -> Optional[int]:
        p = self._procs().get(name)
        return getattr(p, "pid", None)

    def _asleep(self, name: str) -> bool:
        try:
            return bool(self.orch.is_sleeping(name))
        except Exception:                             # noqa: BLE001
            return False

    def _sleep_mode_enabled(self, name: str) -> bool:
        """A model can only reach R2 if its engine was LAUNCHED with sleep mode.

        vLLM 0.17.x also requires VLLM_SERVER_DEV_MODE=1 for /sleep, /wake_up
        and /is_sleeping to exist at all, so both are checked. MODELS_L40S in
        experiments/model_configs.py sets NEITHER today — which is why this is
        a preflight refusal and not an assertion.
        """
        cfg = self._models().get(name, {}) or {}
        args = cfg.get("extra_args") or []
        if "--enable-sleep-mode" not in args:
            return False
        # The orchestrator's own check (model_orchestrator.py:391-393) stops at
        # extra_args. Dev mode is what makes the endpoints exist, so an explicit
        # "0" is treated as a refusal; absent is treated as the orchestrator
        # treats it, since chemgraph_exp.py --sleep-wake injects both together.
        dev = (cfg.get("extra_env") or {}).get("VLLM_SERVER_DEV_MODE")
        return dev != "0"

    def _gpus(self, name: str) -> set:
        return set((self._models().get(name, {}) or {}).get("gpus", []) or [])

    def _port(self, name: str) -> Optional[int]:
        return (self._models().get(name, {}) or {}).get("port")

    def _served_name(self, name: str) -> str:
        cfg = self._models().get(name, {}) or {}
        extra = cfg.get("extra_args") or []
        if "--served-model-name" in extra:
            return extra[extra.index("--served-model-name") + 1]
        return cfg.get("model_name", name)

    def parked_models(self) -> list[str]:
        return [n for n, e in self._engines.items()
                if e.parked and self._alive(n)]

    # -- coherence --------------------------------------------------------

    def probe(self, name: str) -> dict:
        """One coherence probe, judged here so an injected probe cannot skip
        the semantic anchor.

        Injectable so the tests can drive every outcome -- a good wake, the
        '!!!!' one, and the fp8 one that both composition heuristics passed --
        with no GPU.
        """
        if self._probe is not None:
            p = dict(self._probe(name))
            if "text" in p:
                verdict = judge_probe(p.get("text"), p.get("finish_reason"),
                                      self.must_contain)
                # An injected probe may only make the verdict STRICTER.
                verdict["ok"] = bool(verdict["ok"]) and bool(p.get("ok", True))
                p.update(verdict)
            return p
        port = self._port(name)
        if port is None:
            return {"ok": False, "text": "", "finish_reason": None,
                    "degenerate": True, "error": f"no port for {name}"}
        try:
            return coherence_probe(port, self._served_name(name),
                                   must_contain=self.must_contain)
        except Exception as exc:                      # noqa: BLE001
            return {"ok": False, "text": "", "finish_reason": None,
                    "degenerate": True, "error": repr(exc)}

    def _assert_coherent(self, name: str, when: str) -> dict:
        p = self.probe(name)
        eng = self._engines.get(name)
        ref = eng.probe_reference if eng else None
        p["when"] = when
        if ref is not None:
            p["matches_reference"] = (p.get("text") == ref)
        if eng is not None:
            eng.last_probe = p
        if not p.get("ok"):
            self.coherence_failures.append({"model": name, **p})
            if self.strict_coherence:
                text = str(p.get("text", ""))
                if not text.strip():
                    why = "generated nothing"
                elif p.get("degenerate"):
                    why = f"generated degenerate text {text!r:.80}"
                elif p.get("anchored") is False:
                    why = (f"generated {text!r:.80}, which does not contain "
                           f"{self.must_contain!r} -- the answer to the probe "
                           f"prompt. Composition heuristics pass this; the "
                           f"real fp8 engine returned "
                           f"'\u306f.   1111' here")
                else:
                    why = f"failed the probe: {p!r:.120}"
                raise IncoherentWake(
                    f"{name}: {when} returned HTTP success but {why}. "
                    f"Level-2 sleep did exactly this on this cluster while "
                    f"reporting success; a footprint or a wake time from this "
                    f"engine would be worthless.")
        elif ref is not None and not p["matches_reference"]:
            # Not fatal: greedy decoding on a batching engine can flip a token.
            # Recorded, because M1's L1 evidence is "identical text" and a
            # systematic drift here is the first sign the wake is not clean.
            self.coherence_failures.append({"model": name,
                                            "reference_mismatch": True, **p})
        return p

    # =====================================================================
    # The GPU-occupancy path — deliverable 2
    # =====================================================================

    def gpu_conflicts(self, name: str) -> list[str]:
        """Live engines that hold GPUs this model needs.

        A SLEPT engine's process is alive but its VRAM is freed, so it does not
        conflict — the same rule the orchestrator itself applies at
        model_orchestrator.py:589-592. That rule is why parking a victim is a
        real eviction and not a rename.
        """
        want = self._gpus(name)
        out = []
        for other in list(self._procs()):
            if other == name or not self._alive(other):
                continue
            if self._asleep(other):
                continue
            if not want or not self._gpus(other) or (want & self._gpus(other)):
                out.append(other)
        return out

    def _vram_free_frac(self, gpus: set) -> Optional[float]:
        """Smallest free fraction across `gpus`, or None if unreadable.

        This is what makes an eviction CONFIRMED rather than assumed: /sleep
        returning 200 is a claim, VRAM coming back is the evidence.
        """
        getter = getattr(self.orch, "get_gpu_memory", None)
        if not callable(getter) or not gpus:
            return None
        try:
            mem = getter()
        except Exception:                             # noqa: BLE001
            return None
        fracs = [mem[g]["free"] / mem[g]["total"]
                 for g in gpus
                 if g < len(mem) and mem[g].get("total")]
        return min(fracs) if fracs else None

    def _free_gpus_for(self, name: str, vram_free_target: float = 0.5) -> dict:
        """Take the GPUs away from whoever holds them, or explain why not.

        Preference order, and the reason for it:
          PARK (L1 sleep) — the victim's weights stay in host RAM, so its next
              use is a 2.076 s wake instead of a 782.27 s cold boot. Costs
              budget: ~1.90x its weight file.
          STOP — the victim's weights are gone; its next use is a full cold
              boot. Costs no budget. Chosen when the victim cannot be parked
              (no sleep mode in its config, max_parked reached) or when
              `can_park` says the budget will not carry it.

        Returns a detail dict. `ok=False` means nothing was staged and the
        caller must not proceed; `evicted` lists what was already done, because
        a half-finished eviction that is not reported is worse than a failure.
        """
        detail: dict = {"target": name, "ok": True, "evicted": [],
                        "parked": [], "stopped": [], "blocked_by": None,
                        "reason": None,
                        "gpus": sorted(self._gpus(name))}
        # Recorded NOW, not on the way out: the early returns below are the
        # failure paths, and those are the ones a caller needs the detail for.
        # A half-finished eviction that is not reported is worse than a failure.
        self.last_eviction_detail[name] = detail
        for other in self.gpu_conflicts(name):
            if self._evictable is not None and not self._evictable(other):
                detail.update(ok=False, blocked_by=other,
                              reason=f"{other} is protected by the caller's "
                                     f"evictable() and holds GPUs "
                                     f"{sorted(self._gpus(other) & self._gpus(name))}")
                return detail
            est_gb = self._estimate_park_gb(other)
            want_park = (
                self._sleep_mode_enabled(other)
                and len(self.parked_models()) < self.max_parked
                and (self._can_park is None or self._can_park(other, est_gb))
            )
            why_not_park = (
                None if want_park else
                "no --enable-sleep-mode in its config" if not self._sleep_mode_enabled(other)
                else f"max_parked={self.max_parked} reached (k=3 L1 sleeps was "
                     f"measured not to complete)" if len(self.parked_models()) >= self.max_parked
                else f"the budget declined {est_gb:.1f} GB for it")
            done = False
            if want_park:
                try:
                    self.park(other)
                    detail["parked"].append(other)
                    detail["evicted"].append({"model": other, "action": "park",
                                              "held_gb": self._engines[other].held_gb})
                    done = True
                except Exception as exc:              # noqa: BLE001
                    detail.setdefault("park_failures", []).append(
                        {"model": other, "error": repr(exc)})
                    why_not_park = f"park raised {type(exc).__name__}: {exc}"
            if not done:
                try:
                    freed = self._stop_and_measure(other)
                except Exception as exc:              # noqa: BLE001
                    detail.update(ok=False, blocked_by=other,
                                  reason=f"{other} could neither be parked "
                                         f"({why_not_park}) nor stopped "
                                         f"({type(exc).__name__}: {exc})")
                    return detail
                detail["stopped"].append(other)
                detail["evicted"].append({"model": other, "action": "stop",
                                          "freed_gb": freed,
                                          "downgrade_reason": why_not_park})
        # CONFIRM. The eviction is only real if the VRAM came back.
        frac = self._vram_free_frac(self._gpus(name))
        detail["vram_free_frac"] = frac
        if frac is not None and frac < vram_free_target:
            still = self.gpu_conflicts(name)
            detail.update(
                ok=False, blocked_by=still[0] if still else "unknown",
                reason=f"after eviction only {frac*100:.1f}% of VRAM is free on "
                       f"GPUs {sorted(self._gpus(name))}; something outside this "
                       f"orchestrator holds them. Staging anyway would repeat "
                       f"the failure this actor exists to fix.")
        return detail

    def _estimate_park_gb(self, name: str) -> float:
        """What parking `name` is expected to cost, for the budget callback.

        Measured if we have parked it before, catalogue otherwise, and the
        catalogue value is DECLARED (trust B, M2). Never used as a charge.
        """
        e = self._engines.get(name)
        if e is not None and e.held_gb > 0:
            return e.held_gb
        return MODEL_CATALOGUE.get(name, (0.0, 0.0, 0.0))[0]

    # =====================================================================
    # Park / wake / boot — the primitives, each one measured
    # =====================================================================

    def park(self, name: str) -> float:
        """L1-sleep a live, awake engine. Returns the measured held GB.

        The delta is taken against the AWAKE reading recorded immediately
        before the transition, so the engine's own baseline allocations are not
        charged to the park.
        """
        if not self._alive(name):
            raise ParkRefused(f"{name}: no live engine to park")
        if not self._sleep_mode_enabled(name):
            raise ParkRefused(
                f"{name}: its config has no --enable-sleep-mode, so its engine "
                f"cannot reach R2 at all. Add --enable-sleep-mode to extra_args "
                f"and VLLM_SERVER_DEV_MODE=1 to extra_env (vLLM 0.17.x makes "
                f"/sleep dev-mode-only).")
        e = self._engines.setdefault(name, _Engine(name=name))
        if self._asleep(name):
            e.parked = True
            return e.held_gb
        if len(self.parked_models()) >= self.max_parked:
            raise ParkRefused(
                f"{name}: {len(self.parked_models())} engines are already "
                f"parked and max_parked={self.max_parked}. k=3 simultaneous L1 "
                f"sleeps was measured NOT to complete on this cluster "
                f"(timeout/HTTP 500, no OOM observed; register M6).")

        pid = self._pid(name)
        e.pid = pid
        e.anon_awake_gb = self._anon()
        e.file_awake_gb = self._file()
        e.tree_anon_awake_gb = tree_anon_gb(pid) if pid else 0.0
        vram_before = self._vram_free_frac(self._gpus(name))
        t0 = time.perf_counter()
        self.orch.sleep_model(name, level=self.park_level)
        elapsed = time.perf_counter() - t0
        if not self._asleep(name):
            raise ParkRefused(
                f"{name}: /sleep returned but /is_sleeping is still false. The "
                f"engine still holds its VRAM; treating this as a park would "
                f"make the next start fail with the occupancy error.")
        e.anon_parked_gb = self._anon()
        e.file_parked_gb = self._file()
        e.tree_anon_parked_gb = tree_anon_gb(pid) if pid else 0.0
        anon_delta = max(0.0, e.anon_parked_gb - e.anon_awake_gb)
        file_delta = max(0.0, e.file_parked_gb - e.file_awake_gb)
        e.park_backing = ("anon" if anon_delta > PARK_FLOOR_GB
                          else "file" if file_delta > PARK_FLOOR_GB
                          else "none")

        # THE REFUSAL. VRAM came back, so weights left the GPU and are being
        # held somewhere -- but not where we charge. Reporting a parked 72B at
        # 0.0 GB would tell the arbitrator that models are free to hold.
        if (e.park_backing == "file" and "file" not in self.charge_columns):
            vram_after = self._vram_free_frac(self._gpus(name))
            e.parked = True          # it IS parked; we just cannot price it
            raise ParkNotMeasurable(
                f"{name}: L1 sleep moved {file_delta:.2f} GB into the cgroup's "
                f"`file` column and {anon_delta:.2f} GB into `anon` "
                f"(VRAM free frac {vram_before} -> {vram_after}). This actor "
                f"charges `anon`, so it cannot price this park, and charging "
                f"0.0 GB for a parked model is the loudest possible I1 "
                f"violation dressed as a success. Same shape as "
                f"results/bench_h1_fp8_tp1_12561711.json (anon 2.77 -> 2.77, "
                f"file 1.32 -> 83.36, VRAM 87251 -> 1267 MiB). If a coherent "
                f"engine confirms the host backup is a shared mapping, pass "
                f"charge_columns=('anon', 'file') deliberately -- and know that "
                f"`file` also carries the weight shards read at boot.")

        e.held_gb = anon_delta + (file_delta if "file" in self.charge_columns
                                  else 0.0)
        e.parked = True
        e.park_count += 1
        if e.park_count == 1:
            e.first_park_s = elapsed
        else:
            e.subsequent_park_s.append(elapsed)
        self._log(f"parked {name} in {elapsed:.3f}s, held {e.held_gb:.2f} GB "
                  f"(cgroup anon {e.anon_awake_gb:.2f} -> {e.anon_parked_gb:.2f})")
        return e.held_gb

    def wake(self, name: str) -> float:
        """Wake a parked engine and PROVE it came back coherent.

        Returns elapsed seconds. Raises IncoherentWake if the engine serves
        nothing usable — a 200 is not a wake.
        """
        e = self._engines.setdefault(name, _Engine(name=name))
        if not self._alive(name):
            raise RuntimeError(f"Cannot wake {name}: no live server process.")
        t0 = time.perf_counter()
        self.orch.wake_model(name)
        elapsed = time.perf_counter() - t0
        e.parked = False
        e.wake_s.append(elapsed)
        self._assert_coherent(name, "wake")
        # V2: does the host-side backup survive the wake? The ~9x gap between a
        # first park (23.692 s) and a later one (2.611 s) says the backup is
        # allocated once and reused, which would mean an awake engine STILL
        # charges the host budget. Measured here rather than assumed.
        e.anon_awake_after_wake_gb = self._anon()
        self._log(f"woke {name} in {elapsed:.3f}s; cgroup anon now "
                  f"{e.anon_awake_after_wake_gb:.2f} GB")
        return elapsed

    def boot(self, name: str) -> float:
        """Cold-boot an engine and probe it. Records the reference text."""
        e = self._engines.setdefault(name, _Engine(name=name))
        # Read BEFORE the engine exists. The difference between this and the
        # awake reading is the engine's own baseline — interpreter, CUDA
        # context, allocator arenas. It is deliberately NOT charged to the park
        # (that is what "take the delta against the awake reading" means), and
        # for exactly the same reason it must not be CREDITED back at teardown:
        # returning it would hand the budget GB it never booked and hide a
        # shortfall in the part that was charged.
        e.anon_pre_boot_gb = self._anon()
        self._revoke_witness(name)     # the resource is back; the old release
                                       # can no longer vouch for anything
        t0 = time.perf_counter()
        self.orch.start_model_measured(name, metrics=None)
        e.boot_s = time.perf_counter() - t0
        e.pid = self._pid(name)
        e.parked = False
        p = self._assert_coherent(name, "cold_boot")
        if e.probe_reference is None and p.get("ok"):
            e.probe_reference = p.get("text")
        return e.boot_s

    # =====================================================================
    # ResidencyActor protocol
    # =====================================================================

    @property
    def resource_class(self) -> ResourceClass:
        return ResourceClass.MODEL

    def stage(self, spec: ResourceSpec) -> Rung:
        """Make the model resident at R2. Returns the rung ACTUALLY reached.

        The order below is the whole deliverable, so it is spelled out:
          1. refuse, BEFORE evicting anyone, if this model can never be parked
             (no sleep mode) or if parking it would exceed max_parked. Evicting
             an incumbent to stage something we cannot hold is strictly worse
             than declining;
          2. free the GPUs (park the occupant if the budget allows, else stop
             it), and CONFIRM the VRAM came back;
          3. boot, probe for coherence, park, measure the anon delta;
          4. if the park fails on an engine WE booted, stop it again and return
             R0_DISK — the caller asked for a parked model, not for a live
             engine sitting on the GPU it did not ask to occupy;
          5. if anything failed after victims were parked, wake them back.
        """
        rid = spec.resource_id
        name = self.model_for(rid)
        detail: dict = {"resource_id": rid, "model": name,
                        "requested_rung": int(spec.held_rung),
                        "parked": False, "evicted": [], "reason": None}
        self.last_stage_detail[rid] = detail

        e = self._engines.setdefault(name, _Engine(name=name, spec=spec))
        e.spec = spec

        if self._alive(name) and self._asleep(name):
            e.parked = True
            detail.update(parked=True, reason="already parked",
                          reached_rung=int(Rung.R2_PROCESS_BYTES))
            return Rung.R2_PROCESS_BYTES

        # 1. preflight refusals, before any eviction
        if not self._sleep_mode_enabled(name):
            detail.update(reason=(
                f"{name} is launched without --enable-sleep-mode, so it can "
                f"never reach R2. Refusing BEFORE evicting anyone: an eviction "
                f"here would pay a real cold boot to hold nothing."),
                reached_rung=int(Rung.R0_DISK))
            return Rung.R0_DISK
        if not self._alive(name) and len(self.parked_models()) >= self.max_parked:
            detail.update(reason=(
                f"{len(self.parked_models())} engines already parked and "
                f"max_parked={self.max_parked} (k=3 L1 sleeps measured not to "
                f"complete, register M6)"), reached_rung=int(Rung.R0_DISK))
            return Rung.R0_DISK

        booted_here = False
        ev: dict = {}
        try:
            if not self._alive(name):
                # 2. the GPU-occupancy path
                ev = self._free_gpus_for(name)
                detail["evicted"] = ev.get("evicted", [])
                detail["vram_free_frac"] = ev.get("vram_free_frac")
                if not ev.get("ok"):
                    detail.update(reason=ev.get("reason"),
                                  blocked_by=ev.get("blocked_by"),
                                  reached_rung=int(Rung.R0_DISK))
                    self._restore(ev)
                    return Rung.R0_DISK
                # 3. boot. The flag is set BEFORE the call, not after: a boot
                # that starts an engine and then fails its coherence probe has
                # still created an engine, and that engine is ours to undo.
                booted_here = True
                self.boot(name)
                detail["boot_s"] = e.boot_s

            held = self.park(name)
            detail.update(parked=True, held_gb=held,
                          anon_awake_gb=e.anon_awake_gb,
                          anon_parked_gb=e.anon_parked_gb,
                          tree_anon_awake_gb=e.tree_anon_awake_gb,
                          tree_anon_parked_gb=e.tree_anon_parked_gb,
                          park_backing=e.park_backing,
                          file_gb_awake=e.file_awake_gb,
                          file_gb_parked=e.file_parked_gb,
                          park_costs=e.park_cost_summary(),
                          measured_by="cgroup memory.stat anon delta "
                                      "(awake -> parked)",
                          reached_rung=int(Rung.R2_PROCESS_BYTES))
            return Rung.R2_PROCESS_BYTES

        except Exception as exc:                      # noqa: BLE001
            detail["reason"] = f"{type(exc).__name__}: {exc}"
            self._restore(ev)
            if booted_here:
                # 4. undo our own side effect.
                try:
                    if self._alive(name):
                        self._stop_and_measure(name)
                    detail["undone"] = (
                        f"stopped the engine this stage booted: the caller "
                        f"asked for a parked model and got none, so leaving it "
                        f"on the GPU would be a side effect it did not ask for")
                except Exception as exc2:             # noqa: BLE001
                    detail["undo_failed"] = repr(exc2)
                detail["reached_rung"] = int(Rung.R0_DISK)
                return Rung.R0_DISK
            if self._alive(name) and not self._asleep(name):
                # The engine was already live and serving when we were called;
                # it still is. More of the chain is paid than R2, but the
                # requested residency was NOT reached, and detail says so.
                detail["reached_rung"] = int(Rung.R3_ACTIVATED)
                return Rung.R3_ACTIVATED
            detail["reached_rung"] = int(Rung.R0_DISK)
            return Rung.R0_DISK

    def _restore(self, ev: dict) -> None:
        """Wake back victims we parked for a stage that then failed.

        A stopped victim cannot be restored cheaply (it needs a full cold
        boot); the detail says which ones those are rather than implying the
        state was rolled back.
        """
        if not self.restore_on_failure or not ev:
            return
        for victim in ev.get("parked", []):
            try:
                self.wake(victim)
                ev.setdefault("restored", []).append(victim)
            except Exception as exc:                  # noqa: BLE001
                ev.setdefault("restore_failed", []).append(
                    {"model": victim, "error": repr(exc)})

    def measure_held_gb(self, resource_id: str) -> float:
        """INVARIANT I1 — the cgroup's `anon`, measured now. Never declared.

        Live re-measurement when this actor holds exactly ONE engine, which is
        the M=1 case this workload is actually in: `anon_now - anon_awake` for
        that engine. That number moves if the park drains, which is the point.

        With more than one engine tracked, a cgroup-wide counter cannot
        attribute anon per resource, so the value returned is the delta
        measured across THAT resource's own park transition and
        `reconcile_detail()` publishes the residual. Reported, not corrected.
        """
        try:
            name = self.model_for(resource_id)
        except KeyError:
            return 0.0
        e = self._engines.get(name)
        if e is None or not self._alive(name):
            return 0.0
        live = [n for n in self._engines if self._alive(n)]
        if len(live) == 1 and e.anon_awake_gb > 0:
            now = self._anon()
            if now >= 0:
                return max(0.0, now - e.anon_awake_gb)
        return max(0.0, e.held_gb)

    def is_resident(self, resource_id: str) -> bool:
        """Held at R2: the engine process is alive AND its engine is asleep.

        An AWAKE engine is not counted as resident even if V2 turns out to show
        it still holds the host-side backup. Claiming residency for an engine
        that is occupying the GPU would let the ledger charge host RAM for a
        model that is, in fact, serving — and the budget is over host RAM.
        """
        try:
            name = self.model_for(resource_id)
        except KeyError:
            return False
        return self._alive(name) and self._asleep(name)

    def release(self, resource_id: str) -> float:
        """INVARIANT I2 — the GB the OS ACTUALLY gave back, four ways.

        1. the cgroup `anon` delta across the teardown, MINUS the engine's own
           baseline (which was never charged)              <- what is RETURNED
        2. the whole process TREE leaves /proc (SIGINT, then SIGKILL, then
           wait) — vLLM at tp=4 is several processes and the weights are in
           the workers, so waiting only on the api_server would prove nothing
        3. the per-process `Anonymous` witness taken immediately before
           teardown, so a disagreement with (1) is visible
        4. the VRAM comes back (the orchestrator's stop_model drains it)

        Returning the delta rather than what we charged is the whole invariant.
        A neighbour allocating inside the same cgroup during the teardown can
        make (1) come in short while (2) and (3) say the memory really did go;
        that combination is recorded explicitly as
        `cgroup_short_of_process_witness` and the SHORT number is still what is
        returned, because rounding it up to what we charged is precisely how a
        budget becomes fiction.
        """
        try:
            name = self.model_for(resource_id)
        except KeyError:
            return 0.0
        e = self._engines.get(name)
        if e is None or not self._alive(name):
            return 0.0
        held_before = self.measure_held_gb(resource_id)
        freed = self._stop_and_measure(name, resource_id=resource_id,
                                       held_before=held_before)
        return freed

    def _revoke_witness(self, name: str) -> None:
        """Drop every witness token belonging to this model.

        Called when the model is (re-)staged. After that the resource exists
        again, so the delta from its previous teardown describes a state the
        machine has left, and handing it to a later release would be exactly
        the stale vouching the contract warns about.
        """
        for rid, d in list(self.last_release_detail.items()):
            if d.get("model") == name:
                self._witness_token.pop(rid, None)

    def release_witness(self, resource_id: str) -> Optional[float]:
        """GB the ENCLOSING allocation gave back, measured independently of
        this resource's own process. None when there is no such reading.

        REQUIRED BY THE PROTOCOL, optional in its return value: ResidencyActor
        is @runtime_checkable, so this method's mere presence is what makes
        isinstance() true.

        WHAT IS RETURNED. The cgroup `anon` delta across the teardown, minus
        the engine's own baseline -- `cgroup_freed_gb` from the release detail.
        It is independent in the sense that matters: it is a reading of the
        JOB'S ALLOCATION, and it does not require the released process to still
        exist. That is the whole point for a teardown actor, whose
        measure_held_gb() after a release is 0.0 by construction and therefore
        confirms nothing.

        WHY THE BASELINE-SUBTRACTED NUMBER AND NOT THE RAW DELTA. A teardown
        gives back the parked weights AND the engine's own footprint, but only
        the weights were ever charged. Crediting the baseline would let the
        interpreter and the CUDA context vouch for weights that did not come
        back. When the baseline is unknown (we did not boot this engine) the
        whole delta is returned instead -- an over-count in the permissive
        direction, which can only fail to raise, never raise falsely.

        NONE IS RETURNED, HONESTLY, IN THREE CASES:
          * no teardown of this resource has been observed at all;
          * the cgroup was unreadable at teardown, so there is no reading --
            a missing instrument must not be reported as a zero give-back,
            which would fail every release on a machine without cgroup v2;
          * the recorded teardown is STALE: the model has been staged again
            since, or a newer release supersedes it. `last_release_detail`
            persists by design (it is evidence), so the stamp is what stops it
            from being replayed.
        """
        d = self.last_release_detail.get(resource_id)
        if d is None or not d.get("cgroup_readable"):
            return None
        if self._witness_token.get(resource_id) != d.get("release_seq"):
            return None
        v = d.get("cgroup_freed_gb")
        return None if v is None else float(v)

    def _stop_and_measure(self, name: str, resource_id: Optional[str] = None,
                          held_before: Optional[float] = None) -> float:
        """Stop an engine and measure what came back. Shared by release() and
        the eviction path, so a stopped victim is measured exactly as
        rigorously as a released retention."""
        rid = resource_id or name
        e = self._engines.get(name)
        pid = self._pid(name) or (e.pid if e else None)
        pids = process_tree(pid) if pid else []
        tree_before = tree_anon_gb(pid) if pid else 0.0
        anon_before = self._anon()
        if held_before is None:
            held_before = (self.measure_held_gb(rid) if e is not None else 0.0)

        self.orch.stop_model(name)
        gone_after_s = self._wait_tree_gone(pids, self.teardown_timeout_s)
        anon_after = self._anon()
        alive_pids = [p for p in pids if pid_alive(p)]
        # BOTH readings must be real. `cgroup_anon_gb` returns -1.0 when there
        # is no readable cgroup, and `before - (-1)` would manufacture a
        # positive give-back out of a missing instrument -- which would then be
        # handed to the ledger as an independent witness.
        readable = anon_before >= 0 and anon_after >= 0
        total_freed = max(0.0, anon_before - anon_after) if readable else 0.0

        # Split the give-back. A teardown returns the parked weights AND the
        # engine's own baseline, but only the weights were ever charged, so
        # only the weights are what release() is answerable for. When we did
        # not boot the engine ourselves the baseline is unknown and the whole
        # delta is returned with `baseline_unknown` set, rather than a guess.
        baseline = None
        if e is not None and e.anon_pre_boot_gb is not None and e.anon_awake_gb > 0:
            baseline = max(0.0, e.anon_awake_gb - e.anon_pre_boot_gb)
        freed = max(0.0, total_freed - baseline) if baseline is not None else total_freed

        self._release_seq += 1
        detail = {
            "resource_id": rid,
            "model": name,
            "release_seq": self._release_seq,
            "cgroup_readable": readable,
            "pids": pids,
            "held_before_gb": round(held_before, 4),
            "cgroup_anon_before_gb": round(anon_before, 4),
            "cgroup_anon_after_gb": round(anon_after, 4),
            "cgroup_total_freed_gb": round(total_freed, 4),
            "engine_baseline_gb": None if baseline is None else round(baseline, 4),
            "baseline_unknown": baseline is None,
            "cgroup_freed_gb": round(freed, 4),
            "tree_anon_before_gb": round(tree_before, 4),
            "proc_gone": not alive_pids,
            "proc_gone_after_s": round(gone_after_s, 3),
            "still_alive_pids": alive_pids,
            "vram_free_frac_after": self._vram_free_frac(self._gpus(name)),
            "measured_by": "cgroup memory.stat anon delta across teardown",
            "witness": "Anonymous(/proc/<pid>/smaps_rollup) over the process tree",
        }
        if held_before > 0.05:
            detail["cgroup_corroborates"] = (
                freed >= self.release_confirm_frac * held_before)
            detail["cgroup_short_of_process_witness"] = bool(
                not alive_pids and tree_before > 0
                and freed < self.release_confirm_frac * tree_before)
        self.last_release_detail[rid] = detail
        # This teardown, and only this one, is what release_witness() may vouch
        # for. A witness with no readable cgroup behind it is not a witness.
        if readable:
            self._witness_token[rid] = self._release_seq
        else:
            self._witness_token.pop(rid, None)
        # The record is KEPT, with everything reset to "holds nothing": its
        # measured park costs and its coherence reference are evidence about
        # this engine, and dropping them would lose the amortisation history a
        # scheduler needs (first park 23.692 s vs 2.611 s later).
        if e is not None:
            e.parked = False
            e.held_gb = 0.0
            e.pid = None
            e.anon_awake_gb = 0.0
            e.anon_parked_gb = 0.0

        if alive_pids:
            raise ReleaseNotHonoured(
                f"{name}: pids {alive_pids} still present after stop_model; "
                f"{held_before:.3f} GB charged and not demonstrably returned. "
                f"The budget is fiction from this point on.")
        return freed

    @staticmethod
    def _wait_tree_gone(pids: list[int], timeout_s: float) -> float:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if not any(pid_alive(p) for p in pids):
                return time.perf_counter() - t0
            time.sleep(0.05)
        return time.perf_counter() - t0

    # =====================================================================
    # The serving path — what the prefetcher actually wants
    # =====================================================================

    def activate(self, resource_id: str, allow_cold_boot: bool = True) -> dict:
        """Make this model SERVE, evicting whoever holds the GPUs.

        This is the call that replaces
            orchestrator.start_model_measured(name)
        on the prefetch path. That call raises
            "Cannot start X: GPUs [...] occupied by Y. Call stop_model first."
        which is how 10 of 13 admitted prefetches died in under 15 ms.

        Returns {"mechanism": "wake"|"cold_boot"|"already_serving",
                 "elapsed_s": float, "evicted": [...], "probe": {...}}.
        Raises GpusNotFreed — naming the occupant and the reason — when the
        GPUs cannot be taken. A loud, specific failure is the point; the old
        one named the symptom and stopped.
        """
        name = self.model_for(resource_id)
        t0 = time.perf_counter()
        if self._alive(name) and not self._asleep(name):
            p = self._assert_coherent(name, "already_serving")
            return {"mechanism": "already_serving",
                    "elapsed_s": time.perf_counter() - t0,
                    "evicted": [], "probe": p}

        ev = self._free_gpus_for(name)
        if not ev.get("ok"):
            raise GpusNotFreed(
                f"cannot activate {name}: GPUs {ev.get('gpus')} — "
                f"{ev.get('reason')}. Held by {ev.get('blocked_by')}. "
                f"(This is the failure the old orchestrator reported as "
                f"'Call stop_model first'; the difference is that the actor "
                f"tried, and this message says what it tried.)")

        if self._alive(name) and self._asleep(name):
            elapsed = self.wake(name)
            mech = "wake"
        else:
            if not allow_cold_boot:
                raise GpusNotFreed(
                    f"{name} has no live engine and allow_cold_boot=False")
            elapsed = self.boot(name)
            mech = "cold_boot"
        return {"mechanism": mech, "elapsed_s": elapsed,
                "evicted": ev.get("evicted", []),
                "probe": dict(self._engines[name].last_probe),
                "gpu_path": ev}

    # -- reporting --------------------------------------------------------

    def park_detail(self, resource_id: str) -> dict:
        try:
            name = self.model_for(resource_id)
        except KeyError:
            return {}
        e = self._engines.get(name)
        if e is None:
            return {}
        return {
            "model": name, "pid": e.pid, "parked": e.parked,
            "held_gb": e.held_gb,
            "anon_gib_awake": e.anon_awake_gb,
            "anon_gib_parked": e.anon_parked_gb,
            "file_gb_awake": e.file_awake_gb,
            "file_gb_parked": e.file_parked_gb,
            "park_backing": e.park_backing,
            "charge_columns": self.charge_columns,
            "anon_gib_awake_after_wake": e.anon_awake_after_wake_gb,
            "tree_anon_awake_gb": e.tree_anon_awake_gb,
            "tree_anon_parked_gb": e.tree_anon_parked_gb,
            "boot_s": e.boot_s,
            "wake_s": list(e.wake_s),
            **e.park_cost_summary(),
        }

    def reconcile_detail(self) -> dict:
        """What we attributed against what the cgroup actually moved.

        The honest answer to "a cgroup counter is not per-resource". If
        `residual_gb` is large, the per-resource attributions and the machine
        have parted company, and the fix is a measurement, not a correction.
        """
        attributed = sum(e.held_gb for e in self._engines.values() if e.parked)
        now = self._anon()
        return {
            "attributed_gb": round(attributed, 4),
            "cgroup_anon_now_gb": round(now, 4),
            "cgroup_anon_at_start_gb": round(self.anon_at_start_gb, 4),
            "cgroup_moved_gb": round(now - self.anon_at_start_gb, 4),
            "residual_gb": round((now - self.anon_at_start_gb) - attributed, 4),
            "parked": self.parked_models(),
            "note": ("anon is a CGROUP property; with one engine held the "
                     "measure is exact, with several it is additive "
                     "attribution and this residual is the check on it"),
        }

    def release_all(self) -> None:
        for name in list(self._engines):
            try:
                self._stop_and_measure(name)
            except Exception:                         # noqa: BLE001
                pass

    # The scheduler asks the executor this, and the executor asks us: the
    # confidence-gate bypass is only safe when something can actually evict.
    can_evict_gpu_occupants = True
