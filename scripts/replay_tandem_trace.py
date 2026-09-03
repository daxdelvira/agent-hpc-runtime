#!/usr/bin/env python3
"""Replay recorded trials through the REAL Tandem policy code. Zero GPU.

WHY THIS IS NOT ANOTHER SIMULATOR
---------------------------------
This project already has four: replay_retention_policy.py, replay_capacity.py,
sim_residency_v2.py and search_ceiling_regime.py. Every one of them
REIMPLEMENTS the policy it evaluates -- they have their own value function,
their own eviction loop, their own budget bookkeeping. That is how
`action_rate` came to misrank retain against prefetch by 280x, how `_best_set`
came to be provably time-blind, and how "exact optimisation amplifies
prediction error" survived as a finding until it turned out to be an artifact
of a dead/alive binary in the simulator rather than a property of anything.

This script imports and drives the shipped classes:

    runtime.residency.contract    ResourceSpec, value(), Rung, ResourceClass
    runtime.residency.arbitrator  GreedyArbitrator  (the real admit() + chain)
    runtime.residency.ledger      ResidencyLedger   (the real budget + I1-I5)

So a defect in Eq. 1, in the eviction chain, in the density tie-break or in the
budget accounting shows up here as a wrong number, which is the entire point.
If this replay and the end-to-end trials disagree, one of them is wrong and
both are ours.

WHAT IT DOES NOT TEST
---------------------
The actor is a bookkeeping stub (`ReplayActor`). It reports exactly what it was
asked to hold, so invariant I2 -- "release confirmed by INDEPENDENT
measurement" -- passes trivially and is not exercised. Nor is the vLLM sleep
endpoint, the cgroup read, GPU eviction, or engine coherence after a wake.

    This replay tests the POLICY. The end-to-end trials test the MECHANISM.
    A number from here is a prediction about what the mechanism would achieve
    if it works, never evidence that it does.

THE HORIZON IS AN ORACLE BY DEFAULT
-----------------------------------
`--horizon oracle` reads next-use straight from the realized sequence, so its
results are a CEILING on cost-aware retention, not a deployable figure. The
project's own measured predictor accuracy is 45-62%; `--horizon shuffle` and
`--horizon lastuse` exist to bracket that. Report which arm produced a number,
per the standing reporting rule.

CALIBRATION IS A GATE, NOT A FOOTNOTE
-------------------------------------
`--cost-source trial` (the default) takes each model's cold cost from the
MEDIAN measured swap-wait for that model in the trials being replayed, faceted
by node. The catalogue's cold_s figures were measured elsewhere and are far
lower than what these nodes actually deliver -- qwen_72b is 800.5 s in
MODEL_CATALOGUE against a measured 1025 s median on 020-6-0 and 1466 s on
020-2-0. Replaying with catalogue costs would make every arm look faster than
any trial can be, which is exactly the class of error the retracted 85.4%
belonged to. `--cost-source catalogue` is available for comparison and prints
a warning.

The `never` arm exists to check the plumbing: with retention off, predicted
wall must land within a few percent of the trial's measured wall, or the
extraction is wrong and no other arm's number means anything.

USAGE
    python3 scripts/replay_tandem_trace.py                      # sweep, default arms
    python3 scripts/replay_tandem_trace.py --calibrate-only
    python3 scripts/replay_tandem_trace.py --budgets 256,400,700,1000
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.residency.arbitrator import GreedyArbitrator          # noqa: E402
from runtime.residency.contract import (                           # noqa: E402
    ResourceClass, ResourceSpec, Rung,
)
from runtime.residency.ledger import ResidencyLedger               # noqa: E402
from runtime.residency.model_actor import MODEL_CATALOGUE          # noqa: E402

RUNS = ROOT / "results" / "eval_q1_q4" / "runs"
WORKLOAD = "atomagents_exp3_aligned"


# ---------------------------------------------------------------------------
# Need sequence extraction
# ---------------------------------------------------------------------------

@dataclass
class Need:
    """One realized demand for a model, with the gap that followed it."""
    model: str
    measured_s: float      # what the trial actually paid to satisfy it
    start_s: float         # seconds since the trial's first need began
    end_s: float


@dataclass
class Trial:
    arm: str
    name: str
    node: str
    budget_mb: Optional[int]
    wall_s: float
    needs: List[Need]

    @property
    def gap_total_s(self) -> float:
        """Wall time NOT blocked on a model -- agent turns and LAMMPS.

        Policy cannot change this, so every arm inherits it unchanged. Taken as
        the residual rather than by summing `agent` and `lammps` phases,
        because those OVERLAP the swap waits (naively summing them gives a
        negative unaccounted term of ~3500 s, which is how that overlap was
        discovered).
        """
        return max(0.0, self.wall_s - sum(n.measured_s for n in self.needs))


def load_trials(arms: Sequence[str]) -> List[Trial]:
    out: List[Trial] = []
    for arm in arms:
        for d in sorted((RUNS / WORKLOAD / arm).glob("t*__*")):
            meta_p, summ_p, met_p = d / "meta.json", d / "summary.json", d / "metrics.csv"
            if not (meta_p.exists() and summ_p.exists() and met_p.exists()):
                continue
            meta = json.loads(meta_p.read_text())
            gpus = meta.get("gpus") or [""]
            # Never pool L40S with Blackwell: identical work differs by up to
            # 4.0x across node types, and summary.json has no gpu_name.
            if "Blackwell" not in (gpus[0] or ""):
                continue
            wall = json.loads(summ_p.read_text()).get("wall_time_s")
            if not wall:
                continue
            rows = []
            for r in csv.DictReader(met_p.open()):
                ph = r.get("phase") or ""
                # model_swap_wait is the EXPOSED stall -- the agent is blocked
                # for its whole duration. model_load times the bring-up whether
                # or not anyone waits, and stopped being emitted entirely once
                # demand swaps went through actor.activate(), whose internal
                # boot carries no metrics logger.
                if not ph.startswith("model_swap_wait:"):
                    continue
                try:
                    rows.append((r["timestamp"], ph.split(":", 1)[1],
                                 float(r["duration_s"])))
                except (KeyError, ValueError):
                    continue
            if len(rows) < 5:          # truncated: no trial does real work in <5 swaps
                continue
            rows.sort(key=lambda t: t[0])
            needs, clock = [], 0.0
            from datetime import datetime
            t0 = datetime.fromisoformat(rows[0][0]).timestamp() - rows[0][2]
            for ts, model, dur in rows:
                end = datetime.fromisoformat(ts).timestamp() - t0
                needs.append(Need(model, dur, end - dur, end))
                clock = end
            out.append(Trial(arm, d.name, meta.get("node", "?").split(".")[0],
                             meta.get("slurm_mem_mb"), wall, needs))
    return out


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def _specs_from(measured: Dict[str, List[float]], source: str
                ) -> Dict[str, ResourceSpec]:
    specs: Dict[str, ResourceSpec] = {}
    for model, (held_gb, cat_cold, ready_s) in MODEL_CATALOGUE.items():
        if source == "trial" and measured.get(model):
            cold = statistics.median(measured[model])
        else:
            cold = cat_cold
        specs[model] = ResourceSpec(
            resource_id=model,
            resource_class=ResourceClass.MODEL,
            held_rung=Rung.R2_PROCESS_BYTES,   # vLLM L1 sleep: weights in host RAM
            held_gb=held_gb,
            cold_s=cold,
            ready_s=ready_s,
        )
    return specs


def build_specs_by_node(trials: Sequence[Trial], source: str
                        ) -> Dict[str, Dict[str, ResourceSpec]]:
    """Cold costs PER NODE, plus a pooled fallback.

    Costs are a property of the node, not of the model, and the difference is
    not a rounding term. Median measured swap-wait for qwen_72b is 1050.2 s on
    atl1-1-03-020-6-0 and 1495.8 s on atl1-1-03-020-2-0 -- 1.42x apart for the
    same weights on the same GPU model.

    Pooling them wrecks the calibration in both directions at once, and the
    `never` arm makes that visible: with pooled costs the three 020-6-0
    baselines came out +3.6% to +5.1% (cost overstated) while all three
    020-2-0 tandem trials came out -21% to -32% (cost understated). Faceted,
    the same trials land at -2.5%/-3.9%/-2.5% and +1.6%/-0.8%.

    So every trial is replayed against ITS OWN node's costs. A node with no
    measurements of its own falls back to the pool, and that is flagged.
    """
    by_node: Dict[str, Dict[str, List[float]]] = {}
    pooled: Dict[str, List[float]] = {}
    for t in trials:
        m = by_node.setdefault(t.node, {})
        for n in t.needs:
            m.setdefault(n.model, []).append(n.measured_s)
            pooled.setdefault(n.model, []).append(n.measured_s)
    out = {node: _specs_from(m, source) for node, m in by_node.items()}
    out["__pooled__"] = _specs_from(pooled, source)
    return out


# ---------------------------------------------------------------------------
# The stub actor and the horizon estimators
# ---------------------------------------------------------------------------

class ReplayActor:
    """Satisfies the ResidencyActor Protocol with exact bookkeeping.

    Because it always reports what it was told to hold, the ledger's I2 check
    (release confirmed by independent measurement) can never fail here. That is
    a limitation of the replay, not a property of the system -- said plainly so
    nobody reads a green replay as evidence the release path works.
    """

    def __init__(self, specs: Dict[str, ResourceSpec]) -> None:
        self._specs = specs
        self._held: Dict[str, float] = {}

    @property
    def resource_class(self) -> ResourceClass:
        return ResourceClass.MODEL

    def stage(self, spec: ResourceSpec) -> Rung:
        self._held[spec.resource_id] = spec.held_gb
        return Rung.R2_PROCESS_BYTES

    def measure_held_gb(self, resource_id: str) -> float:
        return self._held.get(resource_id, 0.0)

    def is_resident(self, resource_id: str) -> bool:
        return resource_id in self._held

    def release(self, resource_id: str) -> float:
        return self._held.pop(resource_id, 0.0)

    def release_witness(self, resource_id: str) -> Optional[float]:
        return None            # no independent witness in a replay; see I2


class OracleHorizon:
    """next_use straight from the realized sequence. A CEILING."""

    def __init__(self, needs: Sequence[Need], lookahead_s: float) -> None:
        self._by_model: Dict[str, List[float]] = {}
        for n in needs:
            self._by_model.setdefault(n.model, []).append(n.start_s)
        self._L = lookahead_s

    @property
    def horizon_s(self) -> float:
        return self._L

    def next_use_s(self, resource_id: str, now_s: float) -> Optional[float]:
        for t in self._by_model.get(resource_id, ()):
            if t > now_s + 1e-9:
                dt = t - now_s
                # I3: never say "never again" -- say "not within L".
                return dt if dt <= self._L else None
        return None


class LastUseHorizon:
    """Recency as a stand-in for a forecast: dt = time since last use.

    Deliberately crude. It uses only the past, so it is a floor on what any
    real estimator can do, and it brackets the oracle from below.
    """

    def __init__(self, lookahead_s: float) -> None:
        self._last: Dict[str, float] = {}
        self._L = lookahead_s

    @property
    def horizon_s(self) -> float:
        return self._L

    def observe(self, model: str, now_s: float) -> None:
        self._last[model] = now_s

    def next_use_s(self, resource_id: str, now_s: float) -> Optional[float]:
        seen = self._last.get(resource_id)
        if seen is None:
            return None
        dt = max(1.0, now_s - seen)
        return dt if dt <= self._L else None


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

def run_never(trial: Trial, specs: Dict[str, ResourceSpec]) -> dict:
    """No retention. Every need pays cold. This is the plumbing check."""
    stall = sum(specs[n.model].cold_s for n in trial.needs if n.model in specs)
    return {"arm": "never", "stall_s": stall, "wakes": 0, "parks": 0,
            "wall_s": stall + trial.gap_total_s}


def run_lru(trial: Trial, specs: Dict[str, ResourceSpec],
            budget_gb: float) -> dict:
    """Size-blind LRU under the same budget. The line any policy must clear.

    ServerlessLLM uses LRU, so this -- not "no retention" -- is the baseline a
    policy claim has to beat.
    """
    held: List[str] = []          # least-recently-used first
    stall = wakes = parks = 0
    stall = 0.0
    for n in trial.needs:
        spec = specs.get(n.model)
        if spec is None:
            continue
        if n.model in held:
            stall += spec.ready_s
            wakes += 1
            held.remove(n.model)
        else:
            stall += spec.cold_s
        gb = sum(specs[m].held_gb for m in held) + spec.held_gb
        while held and gb > budget_gb:
            gb -= specs[held.pop(0)].held_gb
        if gb <= budget_gb:
            held.append(n.model)
            parks += 1
    return {"arm": f"lru@{budget_gb:.0f}", "stall_s": stall, "wakes": wakes,
            "parks": parks, "wall_s": stall + trial.gap_total_s}


def run_tandem(trial: Trial, specs: Dict[str, ResourceSpec], budget_gb: float,
               horizon_kind: str, decay_s: float, lookahead_s: float) -> dict:
    """The REAL GreedyArbitrator + ResidencyLedger, driven over the sequence."""
    actor = ReplayActor(specs)
    ledger = ResidencyLedger(budget_gb=budget_gb)
    ledger.register_actor(actor)
    arb = GreedyArbitrator(decay_s=decay_s)
    horizon = (OracleHorizon(trial.needs, lookahead_s) if horizon_kind == "oracle"
               else LastUseHorizon(lookahead_s))

    stall = 0.0
    wakes = parks = declines = evictions = 0
    for n in trial.needs:
        spec = specs.get(n.model)
        if spec is None:
            continue
        now = n.start_s
        resident = any(e.spec.resource_id == n.model for e in ledger.entries())
        if resident:
            stall += spec.ready_s
            wakes += 1
            ledger.release(n.model, now)      # woken: no longer parked
        else:
            stall += spec.cold_s
        if isinstance(horizon, LastUseHorizon):
            horizon.observe(n.model, now)

        # The model has just been used and is about to be displaced. Ask the
        # real arbitrator whether holding it earns its budget.
        plan = arb.admit(spec, ledger, horizon, now)
        if plan.admit is None:
            declines += 1
            continue
        for victim in plan.evict:
            ledger.release(victim, now)
            evictions += 1
        try:
            actor.stage(spec)
            ledger.charge(spec, Rung.R2_PROCESS_BYTES, now)
            parks += 1
        except Exception:
            # The ledger refusing a charge is a real outcome, not an error to
            # swallow silently -- count it as a decline so the totals still add.
            actor.release(spec.resource_id)
            declines += 1
    return {"arm": f"tandem@{budget_gb:.0f}", "stall_s": stall, "wakes": wakes,
            "parks": parks, "declines": declines, "evictions": evictions,
            "wall_s": stall + trial.gap_total_s}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="256,400,700,1000",
                    help="host-RAM budgets in GB")
    ap.add_argument("--arms", default="baseline,tandem",
                    help="which recorded arms to take need sequences from")
    ap.add_argument("--cost-source", default="trial", choices=("trial", "catalogue"))
    ap.add_argument("--horizon", default="oracle", choices=("oracle", "lastuse"))
    ap.add_argument("--decay-s", type=float, default=60.0, help="Eq. 1's D")
    ap.add_argument("--lookahead-s", type=float, default=1800.0, help="estimator L")
    ap.add_argument("--node", default=None, help="facet costs to one node")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--json-out", default="results/replay_tandem_trace.json")
    args = ap.parse_args()

    trials = load_trials(args.arms.split(","))
    if not trials:
        print("no complete Blackwell trials found", file=sys.stderr)
        return 2
    specs_by_node = build_specs_by_node(trials, args.cost_source)

    def specs_for(t: Trial) -> Dict[str, ResourceSpec]:
        return specs_by_node.get(t.node) or specs_by_node["__pooled__"]

    if args.node:
        trials = [t for t in trials if t.node == args.node]
        if not trials:
            print(f"no trials on {args.node}", file=sys.stderr)
            return 2
    if args.cost_source == "catalogue":
        print("!! cost-source=catalogue: cold_s comes from MODEL_CATALOGUE, which\n"
              "   was measured elsewhere and is well below what these nodes\n"
              "   deliver. Every arm will look faster than any trial can be.\n")

    print(f"{len(trials)} complete Blackwell trials "
          f"({sum(len(t.needs) for t in trials)} needs)\n")
    print(f"cold_s PER NODE (source: {args.cost_source}) -- costs are a property"
          f" of the node,\n   and the two Blackwell nodes differ by 1.42x on the"
          f" same weights:")
    for nd in sorted(k for k in specs_by_node if k != "__pooled__"):
        n_tr = sum(1 for t in trials if t.node == nd)
        print(f"   {nd}   ({n_tr} trial{'s' if n_tr != 1 else ''})")
        for m, sp in sorted(specs_by_node[nd].items()):
            print(f"      {m:14s} held {sp.held_gb:6.1f} GB   cold {sp.cold_s:7.1f} s"
                  f"   ready {sp.ready_s:5.2f} s   (catalogue {MODEL_CATALOGUE[m][1]:6.1f})")

    # -- CALIBRATION GATE ---------------------------------------------------
    print("\n=== calibration: `never` arm vs the trial's own measured wall ===")
    print("    (retention off, so predicted wall should match measured; a big\n"
          "     miss means the extraction or the cost model is wrong)")
    errs = []
    for t in sorted(trials, key=lambda x: (x.arm, x.name)):
        r = run_never(t, specs_for(t))
        err = (r["wall_s"] - t.wall_s) / t.wall_s
        errs.append(abs(err))
        mem = f"{int(t.budget_mb)/1024:.0f}G" if t.budget_mb else "?"
        print(f"   {t.arm:9s} {t.name[:26]:26s} {t.node:18s} mem={mem:>5s} "
              f"needs={len(t.needs):2d}  measured {t.wall_s:8.1f}  "
              f"never {r['wall_s']:8.1f}  {err:+6.1%}")
    print(f"\n   median |error| = {statistics.median(errs):.1%}")
    if args.calibrate_only:
        return 0

    # -- SWEEP --------------------------------------------------------------
    budgets = [float(b) for b in args.budgets.split(",")]
    print(f"\n=== budget sweep · horizon={args.horizon} "
          f"(D={args.decay_s:.0f} s, L={args.lookahead_s:.0f} s) ===")
    if args.horizon == "oracle":
        print("    ORACLE horizon: these are CEILINGS, not deployable numbers.")
    print(f"\n{'budget':>8s} {'arm':>8s} {'wall':>9s} {'vs never':>9s} "
          f"{'vs lru':>8s} {'parks':>6s} {'wakes':>6s} {'decl':>5s}")
    records = []
    never_wall = statistics.mean([run_never(t, specs_for(t))["wall_s"] for t in trials])
    for b in budgets:
        lru = [run_lru(t, specs_for(t), b) for t in trials]
        tan = [run_tandem(t, specs_for(t), b, args.horizon, args.decay_s,
                          args.lookahead_s) for t in trials]
        lw = statistics.mean([r["wall_s"] for r in lru])
        tw = statistics.mean([r["wall_s"] for r in tan])
        for label, w, rs in (("lru", lw, lru), ("tandem", tw, tan)):
            print(f"{b:8.0f} {label:>8s} {w:9.1f} {1 - w/never_wall:+8.1%} "
                  f"{'' if label=='lru' else f'{1 - tw/lw:+7.1%}':>8s} "
                  f"{statistics.mean([r['parks'] for r in rs]):6.1f} "
                  f"{statistics.mean([r['wakes'] for r in rs]):6.1f} "
                  f"{statistics.mean([r.get('declines',0) for r in rs]):5.1f}")
        records.append({"budget_gb": b, "lru_wall_s": lw, "tandem_wall_s": tw,
                        "never_wall_s": never_wall,
                        "tandem_vs_lru": 1 - tw/lw,
                        "tandem_wakes": statistics.mean([r["wakes"] for r in tan]),
                        "tandem_parks": statistics.mean([r["parks"] for r in tan])})
        print()

    print("REPORTING RULE: a tandem-vs-never figure is mostly plain retention,\n"
          "which a vLLM flag plus a loop already gets you. The column that is a\n"
          "claim about THIS system is `vs lru`.")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(),
        "trials": [{"arm": t.arm, "name": t.name, "node": t.node,
                    "slurm_mem_mb": t.budget_mb, "wall_s": t.wall_s,
                    "needs": [n.model for n in t.needs]} for t in trials],
        "cost_source": args.cost_source, "horizon": args.horizon,
        "decay_s": args.decay_s, "lookahead_s": args.lookahead_s,
        "specs_by_node": {nd: {m: {"held_gb": sp.held_gb, "cold_s": sp.cold_s,
                                   "ready_s": sp.ready_s}
                               for m, sp in d.items()}
                          for nd, d in specs_by_node.items()},
        "calibration_median_abs_err": statistics.median(errs),
        "sweep": records,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
