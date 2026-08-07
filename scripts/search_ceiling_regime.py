#!/usr/bin/env python3
"""
search_ceiling_regime.py — where does an EXACT solve with PERFECT next-use
knowledge recover >=20% of WALL CLOCK?

THE QUESTION (user, 2026-08-07). Greedy-plus-predictor may simply be the wrong
solution. Before tuning a policy, establish whether the CEILING is ever worth
chasing: find the set of conditions where an exact subset solve with oracle
next-use knowledge beats the realistic baseline by >=20% of end-to-end time. Then
work backwards to what information could approach it.

WHAT THE BASELINE IS. Not "no system." The baseline already has both mechanisms
switched on and is what a competent engineer builds from a vLLM flag plus our
persistent worker:

    - models parked at L1 (weights in host RAM) rather than discarded,
    - activated data held in a live consumer process,
    - LRU deciding what to drop when the host-RAM budget is exceeded.

So every number here is the marginal value of DECIDING WELL, over a system that
already retains. Retention itself is not on the table -- see the retraction at the
top of the plan file.

WHAT THIS FIXES relative to bench_arbitration_harness.py
--------------------------------------------------------
1. GPU SLOTS ARE REAL HERE. In that harness `gpu_slots` only gated prefetching;
   it did nothing for retention, so "3 models, GPUs for 2" was inexpressible. The
   residency model is now genuinely two-level, which is what the hardware does:

       ON GPU (M slots)  live, serves at ~0 cost
       IN HOST RAM       parked at L1 (models) or activated (data); costs held_gb
                         against the budget; returns at ready_s
       ON DISK           costs cold_s

   A model on the GPU does NOT consume the host-RAM budget -- its weights are in
   VRAM. This is the same correction that killed the earlier false 14% two-class
   result, where the running model was wrongly charged against host RAM.

   M >= 2 is qualitatively different from M = 1: alternating between two models
   that both hold slots costs NOTHING, so the pressure moves to the third and
   later models. exp3 is M = 1 only because all three models declare gpus=[0,1,2,3]
   at tp=4.

2. THE FIRST SPIN-UP IS FREE. The first model any flow needs must be loaded no
   matter what policy runs, so charging it dilutes every percentage with a cost
   nothing can address. Here the flow STARTS with its first model already on the
   GPU, in both arms, and that load appears in neither the numerator nor the wall
   denominator.

3. PERCENTAGES ARE OF WALL, NOT STALL. wall = stall + compute. Reporting a share
   of stall overstates by 1/(stall share of wall) -- on one cell 23.47% of stall
   was 9.42% of wall. Stall share is reported alongside so the two never get
   confused again.

WHAT IS MEASURED AND WHAT IS NOT
--------------------------------
MEASURED: the real resources (qwen_32b/72b/72b_text, uniref50/90) and their
held_gb, cold_s, ready_s. See the catalogue in bench_arbitration_harness.py.
SYNTHETIC: catalogue entries beyond the real ones, every schedule, and the clock.
Nothing here runs on a GPU. This answers "what would have to be true," which is
exactly what the user asked for -- it is not a speedup measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "harness", os.path.join(HERE, "bench_arbitration_harness.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

REAL_MODELS = [("qwen_32b", 129.7, 495.2, 1.03),
               ("qwen_72b", 279.0, 800.5, 2.21),
               ("qwen_72b_text", 276.3, 770.3, 2.19)]
REAL_DATA = [("uniref90", 117.20, 372.6, 0.0),
             ("uniref50", 36.08, 107.1, 0.0)]
# Request shares from the exp3 traces; used to order Zipf popularity so the
# generator does not hand the most requests to whichever name sorts first.
TRACE_POP = {"qwen_72b": 99, "uniref90": 71, "qwen_72b_text": 47, "qwen_32b": 27}


def build_catalogue(n_models: int, n_data: int) -> dict:
    """Real resources first; synthetic ones keep the measured relationships."""
    cat = {}
    for k in range(n_models):
        if k < len(REAL_MODELS):
            n, gb, cold, ready = REAL_MODELS[k]
        else:
            j = k - len(REAL_MODELS) + 1
            gb = 150.0 + 60.0 * j
            cold = 500.0 + 150.0 * j          # movement-bound: scales with size
            ready = gb / (16.6 * 4)           # measured wake bandwidth
            n = f"model_syn{k}"
        cat[n] = dict(cls="model", held_gb=gb, cold_s=cold, ready_s=ready)
    for k in range(n_data):
        if k < len(REAL_DATA):
            n, gb, cold, ready = REAL_DATA[k]
        else:
            j = k - len(REAL_DATA) + 1
            gb = 40.0 + 45.0 * j
            cold = gb * 3.1                   # s/GB near the measured 2.97-3.18
            ready = 0.0
            n = f"data_syn{k}"
        cat[n] = dict(cls="data", held_gb=gb, cold_s=cold, ready_s=ready)
    return cat


def popularity_order(cat):
    return sorted(cat, key=lambda n: (-TRACE_POP.get(n, 0), n))


# ---------------------------------------------------------------------------
# The two-level simulator.
# ---------------------------------------------------------------------------
def simulate(cat, sched, budget, slots, policy, accuracy=1.0, seed=0, log=None):
    """Return (wall, stall) with the first model spin-up free in every arm.

    policy: 'lru'    drop least-recently-used from host RAM (the baseline)
            'greedy' evict the single lowest value-density item, repeat until it
                     fits -- the DEPLOYABLE selector
            'exact'  exact subset solve -- the CEILING, since 2^n enumeration is
                     not what a runtime would do online at scale
    `accuracy` < 1 replaces oracle next-use with a predictor that is right that
    often and otherwise returns another resource's horizon, which is what a
    confusion between resources actually looks like. accuracy=1.0 is the oracle.

    GPU displacement is LRU in EVERY arm, so the only thing that differs is the
    host-RAM decision -- which is the claim being tested.
    """
    import random as _random
    rng = _random.Random(seed)
    names = [v for k, v in sched if k == "need"]
    first_model = next((n for n in names if cat[n]["cls"] == "model"), None)

    gpu = [first_model] if first_model else []      # MRU last; free head start
    ram: dict[str, int] = {}                        # name -> last-use step
    clock = 0.0
    stall = 0.0
    compute = 0.0

    def next_use(name, i):
        for j in range(i + 1, len(sched)):
            if sched[j][0] == "need" and sched[j][1] == name:
                return j
        return None

    def true_hold(name, i):
        """Wall-seconds from i until name is next needed; inf if never."""
        j = next_use(name, i)
        if j is None:
            return float("inf")
        t = 0.0
        for k in range(i + 1, j):
            kind, val = sched[k]
            t += val if kind == "compute" else cat[val]["cold_s"]
        return t

    def hold_seconds(name, i):
        if accuracy >= 1.0 or rng.random() < accuracy:
            return true_hold(name, i)
        others = [true_hold(o, i) for o in sorted(cat) if o != name]
        others = [o for o in others if o != float("inf")]
        return rng.choice(others) if others else true_hold(name, i)

    def density(x, i):
        """Stall-seconds avoided per GB-second of budget consumed."""
        dt = hold_seconds(x, i)
        if dt == float("inf"):
            return float("-inf")
        return ((cat[x]["cold_s"] - cat[x]["ready_s"])
                / max(cat[x]["held_gb"] * dt, 1e-9))

    def choose_ram(cands, i):
        """cands: dict name -> last-use step. Return the set to keep."""
        keep = _choose_ram(cands, i)
        if log is not None and cands:
            log.append(dict(kind="ram_decision", step=i, policy=policy,
                            budget=budget,
                            candidates={x: dict(gb=cat[x]["held_gb"],
                                                last_use=cands[x],
                                                dt=hold_seconds(x, i),
                                                density=density(x, i))
                                        for x in sorted(cands)},
                            kept=sorted(keep),
                            dropped=sorted(set(cands) - set(keep)),
                            gb_kept=round(sum(cat[x]["held_gb"] for x in keep), 1)))
        return keep

    def _choose_ram(cands, i):
        fits = lambda s: sum(cat[x]["held_gb"] for x in s) <= budget
        if policy == "lru":
            keep = set(cands)
            while not fits(keep):
                # oldest last-use goes first; name breaks ties deterministically
                victim = min(sorted(keep), key=lambda x: (cands[x], x))
                keep.discard(victim)
            return keep
        if policy == "greedy":
            keep = set(cands)
            while not fits(keep):
                victim = min(sorted(keep), key=lambda x: (density(x, i), x))
                keep.discard(victim)
            return keep
        # exact: maximise stall-seconds-avoided per second of occupancy.
        best, best_v = set(), -1.0
        pool = sorted(cands)
        for r in range(len(pool), -1, -1):
            for combo in itertools.combinations(pool, r):
                if not fits(combo):
                    continue
                v = 0.0
                for x in combo:
                    dt = hold_seconds(x, i)
                    if dt == float("inf"):
                        continue                  # never needed again: worthless
                    v += (cat[x]["cold_s"] - cat[x]["ready_s"]) / max(dt, 1e-9)
                if v > best_v:
                    best, best_v = set(combo), v
        return best

    for i, (kind, val) in enumerate(sched):
        if kind == "compute":
            clock += val
            compute += val
            continue

        name = val
        r = cat[name]

        if r["cls"] == "model":
            if name in gpu:
                cost = 0.0                        # live: nothing to pay
                where = "GPU"
                gpu.remove(name); gpu.append(name)
            else:
                displaced = None
                if len(gpu) >= slots:             # displace LRU from the GPU
                    out = gpu.pop(0)
                    displaced = out
                    ram[out] = i                  # becomes a park candidate
                    # CALL choose_ram EXACTLY ONCE. Putting it in the
                    # comprehension's condition ran it once PER KEY, and with
                    # accuracy<1 every call re-draws the predictor's randomness,
                    # so different keys were filtered against different decisions.
                    _keep = choose_ram(dict(ram), i)
                    ram = {k: v for k, v in ram.items() if k in _keep}
                where = "RAM" if name in ram else "disk"
                cost = r["ready_s"] if name in ram else r["cold_s"]
                ram.pop(name, None)               # leaves host RAM for the GPU
                gpu.append(name)
                if log is not None and displaced:
                    log.append(dict(kind="gpu_displace", step=i,
                                    displaced=displaced,
                                    parked=displaced in ram))
        else:
            where = "RAM" if name in ram else "disk"
            cost = r["ready_s"] if name in ram else r["cold_s"]
            ram.pop(name, None)
        if log is not None:
            log.append(dict(kind="need", step=i, resource=name, cls=r["cls"],
                            where=where, cost=round(cost, 1)))

        stall += cost
        clock += cost

        if r["cls"] == "data":                    # data re-enters host RAM
            cand = dict(ram); cand[name] = i
            _keep = choose_ram(cand, i)           # once, not once per key
            ram = {k: v for k, v in cand.items() if k in _keep}

    return clock, stall


def cell(cat, scheds, budget, slots):
    wl = sl = we = se = 0.0
    for s in scheds:
        w, st = simulate(cat, s, budget, slots, "lru");   wl += w; sl += st
        w, st = simulate(cat, s, budget, slots, "exact"); we += w; se += st
    return dict(lru_wall=wl, lru_stall=sl, exact_wall=we, exact_stall=se,
                wall_gain_pct=100 * (wl - we) / wl if wl else 0.0,
                stall_gain_pct=100 * (sl - se) / sl if sl else 0.0,
                stall_share_pct=100 * sl / wl if wl else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sched", type=int, default=10)
    ap.add_argument("--out", default="results/search_ceiling_regime.json")
    args = ap.parse_args()

    MODELS = (3, 4, 6)
    SLOTS = (1, 2, 3)
    DATA = (1, 2, 4)
    FRACS = (0.20, 0.35, 0.50)
    NEEDS = (24, 48)
    WINDOWS = (0.25, 1.0)        # 0.25x ~ the REAL exp3 regime: the agent does
                                 # 280 s of work in a 5289 s trial, i.e. stall is
                                 # 94.7% of wall. 1.0x is the generator default.

    rows = []
    hdr = (f"  {'M':>2} {'slots':>5} {'D':>2} {'bud%':>5} {'needs':>5} {'win':>5} "
           f"{'budGB':>7} {'stall%wall':>10} {'gain%stall':>10} {'GAIN%WALL':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, slots, nd, frac, needs, win in itertools.product(
            MODELS, SLOTS, DATA, FRACS, NEEDS, WINDOWS):
        if slots >= nm:
            continue                       # no pressure: every model holds a slot
        cat = build_catalogue(nm, nd)
        budget = frac * sum(r["held_gb"] for r in cat.values())
        old = H.CATALOGUE; H.CATALOGUE = cat
        try:
            scheds = [H.synthetic_schedule(n_needs=needs, window_scale=win,
                                           seed=9000 + k,
                                           resources=popularity_order(cat))
                      for k in range(args.n_sched)]
        finally:
            H.CATALOGUE = old
        r = cell(cat, scheds, budget, slots)
        rows.append(dict(n_models=nm, gpu_slots=slots, n_data=nd,
                         budget_frac=frac, n_needs=needs, window_scale=win,
                         budget_gb=round(budget, 1),
                         **{k: round(v, 2) for k, v in r.items()}))
        flag = "  <== >=20%" if r["wall_gain_pct"] >= 20 else ""
        print(f"  {nm:>2} {slots:>5} {nd:>2} {frac:>5.2f} {needs:>5} {win:>5.2f} "
              f"{budget:>7.0f} {r['stall_share_pct']:>9.1f}% "
              f"{r['stall_gain_pct']:>9.2f}% {r['wall_gain_pct']:>9.2f}%{flag}")

    hits = [r for r in rows if r["wall_gain_pct"] >= 20]
    print(f"\n  {len(hits)} of {len(rows)} cells reach >=20% of WALL")
    if hits:
        print("  the >=20% region:")
        for axis in ("n_models", "gpu_slots", "n_data", "budget_frac",
                     "n_needs", "window_scale"):
            vals = sorted({r[axis] for r in hits})
            allv = sorted({r[axis] for r in rows})
            print(f"    {axis:<14} {vals}   (swept {allv})")
        best = max(hits, key=lambda r: r["wall_gain_pct"])
        print(f"  best: {best['wall_gain_pct']:.2f}% of wall at "
              f"M={best['n_models']} slots={best['gpu_slots']} D={best['n_data']} "
              f"bud={best['budget_frac']} needs={best['n_needs']} "
              f"win={best['window_scale']}")
    else:
        print("  NO cell reaches 20% of wall. Report that, do not widen the sweep")
        print("  until the axes already swept are shown to be the wrong ones.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "n_sched": args.n_sched}, f, indent=2)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
