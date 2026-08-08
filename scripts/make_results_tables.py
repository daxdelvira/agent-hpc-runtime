#!/usr/bin/env python3
"""
make_results_tables.py — regenerate every results table in sc-workshop-paper/results_tables/.

WHY THIS EXISTS. By 2026-08-07 the residency results depended on six independent
factors (budget, GPU slots, catalogue scale, compute share, predictor accuracy,
prefetch mode) and could be framed three different ways, and the numbers had been
delivered piecemeal across a long conversation. Piecemeal delivery is how the
57.7% retention headline survived for a day: it was quoted from the budget that
maximised it, with no neighbouring cell visible to show the reader it was a
corner. Every table here therefore shows a full sweep, never a single cell.

Run:  python3 scripts/make_results_tables.py [--quick]

Determinism: results must be identical under any PYTHONHASHSEED. Set-iteration
order caused a 23% run-to-run swing in an earlier harness; every selection in
sim_residency_v2.py is sorted and every predictor draw is a hash of
(seed, resource, step).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "sc-workshop-paper", "results_tables")

_v = importlib.util.spec_from_file_location(
    "V2", os.path.join(HERE, "sim_residency_v2.py"))
V2 = importlib.util.module_from_spec(_v); _v.loader.exec_module(V2)
_h = importlib.util.spec_from_file_location(
    "H", os.path.join(HERE, "bench_arbitration_harness.py"))
H = importlib.util.module_from_spec(_h); _h.loader.exec_module(H)

POPSEEDS = (0, 1, 2)
SCHEDSEEDS = (9000, 20000)
NSCHED = 10
NEEDS = 48
WIN = 0.10
HORIZON_H = 60.0
ACC = 0.55


def schedules(cat, popseed, schedseed, win=WIN, needs=NEEDS, n=NSCHED):
    old = H.CATALOGUE
    H.CATALOGUE = cat
    try:
        return [H.synthetic_schedule(n_needs=needs, window_scale=win,
                                     seed=schedseed + k,
                                     resources=V2.popularity_order(cat, popseed))
                for k in range(n)]
    finally:
        H.CATALOGUE = old


def measure(cat, budget, slots, win=WIN, needs=NEEDS, **kw):
    """Total wall/stall/compute for one arm, summed over all seed pairs."""
    W = S = C = 0.0
    for ps in POPSEEDS:
        for ss in SCHEDSEEDS:
            for k, sc in enumerate(schedules(cat, ps, ss, win, needs)):
                o = V2.Sim(cat, sc, budget, slots, seed=k,
                           objective="horizon", H=HORIZON_H, **kw).run()
                W += o["wall"]; S += o["stall"]; C += o["compute"]
    return W, S, C


ARM_NEVER = dict(policy="never", retain="none", accuracy=1.0, prefetch=False)


class Cell:
    """One configuration measured against the no-system floor and against LRU.

    THE THREE FRAMINGS, and why all three are reported. They can differ by 5x on
    the same seconds, and each answers a different question:

      wall_vs_never   (never_wall - wall) / never_wall
                      Share of end-to-end time removed relative to running with
                      no residency system at all. Good for an attribution ladder
                      because contributions sum on a common scale. BAD as a
                      headline: it is diluted by compute, so it shrinks toward
                      zero on compute-heavy workloads even though the mechanism
                      is unchanged.

      wall_vs_lru     (lru_wall - wall) / lru_wall
                      Speedup over the realistic baseline -- LRU eviction with
                      vLLM L1 parking and a persistent data worker, i.e. what a
                      competent engineer builds without this project. This is
                      what a reader assumes "X% faster" means.

      stall_vs_lru    (lru_stall - stall) / lru_stall
                      Share of the baseline's remaining STALL removed. The only
                      framing that is not diluted by compute, and the one that
                      improves rather than degrades as workloads get more
                      realistic. The honest invariant.
    """

    def __init__(self, cat, budget, slots, win=WIN, needs=NEEDS, **kw):
        self.nw, self.ns, self.nc = measure(cat, budget, slots, win, needs, **ARM_NEVER)
        self.lw, self.ls, _ = measure(cat, budget, slots, win, needs,
                                      policy="lru", retain="all",
                                      accuracy=1.0, prefetch=False)
        self.cat, self.budget, self.slots, self.win, self.needs = cat, budget, slots, win, needs

    def arm(self, **kw):
        w, s, c = measure(self.cat, self.budget, self.slots, self.win, self.needs, **kw)
        return dict(
            wall=w, stall=s, compute=c,
            compute_share=100 * c / w if w else 0.0,
            wall_vs_never=100 * (self.nw - w) / self.nw if self.nw else 0.0,
            wall_vs_lru=100 * (self.lw - w) / self.lw if self.lw else 0.0,
            stall_vs_lru=100 * (self.ls - s) / self.ls if self.ls else 0.0,
        )


def w(path, text):
    with open(os.path.join(OUT, path), "w") as f:
        f.write(text)
    print(f"  wrote {path}")


PREAMBLE = f"""
> Generated by `scripts/make_results_tables.py`. **Simulation, not measurement**:
> resource constants are measured on real hardware (see
> `sc-workshop-paper/measurement_provenance.md`), but schedules and the clock are
> synthetic and nothing here ran on a GPU.
> Sample: {len(POPSEEDS)} popularity orderings x {len(SCHEDSEEDS)} schedule
> populations x {NSCHED} schedules x {NEEDS} needs unless stated.
"""

FRAMINGS = """
## The three framings

| column | formula | what it answers |
|---|---|---|
| **vs never (wall)** | `(never_wall - wall) / never_wall` | share of end-to-end time removed vs no residency system at all |
| **vs LRU (wall)** | `(lru_wall - wall) / lru_wall` | speedup over the *realistic* baseline -- what a reader assumes "X% faster" means |
| **vs LRU (stall)** | `(lru_stall - stall) / lru_stall` | share of the baseline's remaining stall removed -- **the only framing not diluted by compute** |

`wall = compute + stall`, and compute is byte-identical across every arm on a
given schedule set, so all differences between arms are stall. The three columns
can differ by 5x on the *same seconds*: at 560 GB / slots=2 one saving is
simultaneously +10.10 points of the never-wall, +33.52% of LRU's wall, and
+52.73% of LRU's stall.

**The baseline matters more than the framing.** `never` means no L1 parking, no
persistent data worker, no prefetch -- vLLM's own default, since exp3 runs sleep
level 2 which *discards* weights. `LRU` already includes both retention
mechanisms and is what a competent engineer builds without this project.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        global POPSEEDS, SCHEDSEEDS, NSCHED
        POPSEEDS, SCHEDSEEDS, NSCHED = (0,), (9000,), 6
    os.makedirs(OUT, exist_ok=True)
    CAT = V2.build_catalogue(3, 2)
    FOOT = sum(r["held_gb"] for r in CAT.values())

    # ---------------------------------------------------------------- README
    rows = "\n".join(
        f"| `{n}` | {r['cls']} | {r['held_gb']:.1f} | {r['cold_s']:.1f} | "
        f"{r['ready_s']:.2f} | {(r['cold_s']-r['ready_s'])/r['held_gb']:.2f} |"
        for n, r in sorted(CAT.items(), key=lambda x: -x[1]["held_gb"]))
    w("00_README.md", f"""# Residency results tables
{PREAMBLE}
## Index

| file | varies | holds fixed |
|---|---|---|
| [01_attribution_ladder.md](01_attribution_ladder.md) | who contributes what | budget, slots |
| [02_budget_sweep.md](02_budget_sweep.md) | host-RAM budget | scale, compute, accuracy |
| [03_compute_sweep.md](03_compute_sweep.md) | compute share of wall | budget, slots |
| [04_accuracy_sweep.md](04_accuracy_sweep.md) | predictor accuracy | budget, slots |
| [05_scale_sweep.md](05_scale_sweep.md) | number of resources | budget *fraction* |
| [06_prefetch_variants.md](06_prefetch_variants.md) | prefetch scope + safety rule | budget, slots |
| [07_objective_check.md](07_objective_check.md) | the value function | budget, slots |
{FRAMINGS}
## The measured catalogue

Every constant below is measured; provenance in `bench_arbitration_harness.py`.
`held_gb` for a model is the **parked** footprint (weights x 1.90, the measured
L1 park ratio), not the file size.

| resource | class | held_gb | cold_s | ready_s | s/GB |
|---|---|---|---|---|---|
{rows}

Total footprint **{FOOT:.1f} GB**. Production allocation is **256 GB = {100*256/FOOT:.0f}%** of it.

## Standing caveats

1. **Every number is simulated.** Constants are measured; the need sequences are
   generated (Zipf popularity + 35% repeat bias, bimodal lognormal gaps fitted to
   the real traces). No end-to-end speedup has been measured on hardware.
2. **Spread across popularity orderings is 0.7-4.8 points.** Differences smaller
   than ~2 points are not resolvable at this sample size. Do not read two
   decimals as significant.
3. **Wall-clock percentages are mostly a statement about the stall share.** See
   `03_compute_sweep.md`: the same mechanism reads 36.96% or 3.17% of wall
   depending only on how much compute the workload does.
4. **exp3's own trials are ~5% compute** (280.3 s of agent work in a 5288.5 s
   trial), which is the far-left, most flattering column of that sweep.
5. **A 279 GB `qwen_72b` cannot be parked at all at 256 GB.** Much of the
   production-allocation behaviour is that capacity limit, not policy.
6. **Retention policy needs a binding budget; prefetch needs slack.** The two
   mechanisms prefer opposite regimes; see 02 and 06.
7. **Predictor accuracy 45-62% is this project's measured *identity* accuracy.**
   Nothing has measured how well *time-to-next-use* can be predicted, and that
   is the quantity every policy here actually consumes.
""")

    # ------------------------------------------------- 01 attribution ladder
    LADDER = [
        ("L0  no residency system", dict(**ARM_NEVER), "floor"),
        ("L1  + vLLM L1 sleep (models), LRU", dict(policy="lru", retain="models", accuracy=1.0, prefetch=False), "vLLM"),
        ("L2  + persistent data worker, LRU", dict(policy="lru", retain="all", accuracy=1.0, prefetch=False), "ours"),
        ("L3  + cost-aware retention", dict(policy="greedy", retain="all", accuracy=ACC, prefetch=False), "ours"),
        ("L4  + slack prefetch", dict(policy="greedy", retain="all", accuracy=ACC, prefetch="slack"), "ours"),
    ]
    body = [f"# Attribution ladder — who contributes what\n{PREAMBLE}",
            "Each rung adds **exactly one** mechanism to the rung above.\n",
            "`vLLM` = obtainable from `--enable-sleep-mode` plus a loop.  ",
            "`ours` = requires this project.\n"]
    for budget in (256.0, 560.0):
        for slots in (1, 2):
            c = Cell(CAT, budget, slots)
            body.append(f"\n## Budget {budget:.0f} GB, GPU slots {slots}\n")
            body.append("| configuration | owner | vs never (wall) | marginal | vs LRU (wall) | vs LRU (stall) |")
            body.append("|---|---|---|---|---|---|")
            prev = 0.0
            for lbl, kw, own in LADDER:
                a = c.arm(**kw)
                body.append(f"| {lbl} | {own} | {a['wall_vs_never']:.2f}% | "
                            f"{a['wall_vs_never']-prev:+.2f} | {a['wall_vs_lru']:+.2f}% | "
                            f"{a['stall_vs_lru']:+.2f}% |")
                prev = a["wall_vs_never"]
            mo = c.arm(policy="lru", retain="models", accuracy=1.0, prefetch=False)["wall_vs_never"]
            do = c.arm(policy="lru", retain="data", accuracy=1.0, prefetch=False)["wall_vs_never"]
            bo = c.arm(policy="lru", retain="all", accuracy=1.0, prefetch=False)["wall_vs_never"]
            sm, sd = 0.5*mo + 0.5*(bo-do), 0.5*do + 0.5*(bo-mo)
            body.append(f"\n**Shapley split of the {bo:.2f}% retention total** (order-independent; "
                        f"marginal credit otherwise depends on which mechanism is added first): "
                        f"vLLM sleep **{sm:.2f} ({100*sm/bo:.0f}%)**, persistent worker "
                        f"**{sd:.2f} ({100*sd/bo:.0f}%)**.  ")
            body.append(f"Measured alone: models-only {mo:.2f}%, data-only {do:.2f}%, both {bo:.2f}%.")
    w("01_attribution_ladder.md", "\n".join(body))

    # ------------------------------------------------------- 02 budget sweep
    body = [f"# Budget sweep — where the problem is real\n{PREAMBLE}",
            "`binding %` is the share of host-RAM decisions where something was "
            "actually evicted. **At 0% there is no retention problem left** — the "
            "correct policy is 'park everything', which is a vLLM flag.\n"]
    for slots in (1, 2):
        body.append(f"\n## GPU slots {slots}\n")
        body.append("| budget | % of footprint | binding % | largest set that fits | LRU (vs never) | full system (vs never) | system vs LRU (wall) | system vs LRU (stall) |")
        body.append("|---|---|---|---|---|---|---|---|")
        for budget in (200.0, 256.0, 320.0, 400.0, 480.0, 560.0, 640.0, 720.0, 838.0):
            c = Cell(CAT, budget, slots)
            sysm = c.arm(policy="greedy", retain="all", accuracy=ACC, prefetch="slack")
            lru = c.arm(policy="lru", retain="all", accuracy=1.0, prefetch=False)
            # binding fraction and packing
            nd = drop = 0
            for ps in POPSEEDS[:1]:
                for k, sc in enumerate(schedules(CAT, ps, SCHEDSEEDS[0])):
                    lg = []
                    V2.Sim(CAT, sc, budget, slots, policy="lru", retain="all",
                           seed=k, objective="horizon", H=HORIZON_H, log=lg).run()
                    for e in lg:
                        if e["kind"] == "ram":
                            nd += 1
                            if e["dropped"]:
                                drop += 1
            best = ()
            for r in range(len(CAT), 0, -1):
                got = [cb for cb in itertools.combinations(sorted(CAT), r)
                       if sum(CAT[x]["held_gb"] for x in cb) <= budget]
                if got:
                    best = got[0]; break
            body.append(f"| {budget:.0f} GB | {100*budget/FOOT:.0f}% | "
                        f"{100*drop/max(nd,1):.1f}% | {len(best)} of 5 | "
                        f"{lru['wall_vs_never']:.2f}% | {sysm['wall_vs_never']:.2f}% | "
                        f"{sysm['wall_vs_lru']:+.2f}% | {sysm['stall_vs_lru']:+.2f}% |")
    body.append("""
## Packing thresholds — why budget value is lumpy, not linear

Budget only buys something when it crosses a threshold that admits one more
resource. Between thresholds the extra GB is unusable, which is why 320 GB and
400 GB perform almost identically.

| threshold | what it newly admits |
|---|---|
| 283 GB | `qwen_32b` + **both** UniRef |
| 445 GB | a **72B becomes parkable** alongside the 32B |
| 562 GB | 4 of 5 resources |
| 838 GB | everything — **problem degenerates** |
""")
    w("02_budget_sweep.md", "\n".join(body))

    # ------------------------------------------------------ 03 compute sweep
    body = [f"# Compute sweep — every wall-clock claim is a claim about stall share\n{PREAMBLE}",
            "`window_scale` multiplies every inter-need gap. **Nothing about the "
            "mechanism changes across a row group** — only how much real work the "
            "workload does between tool calls.\n"]
    for budget, slots in ((256.0, 1), (560.0, 2)):
        body.append(f"\n## Budget {budget:.0f} GB, GPU slots {slots}\n")
        body.append("| window scale | compute % of wall | LRU (vs never) | retain only (vs never) | +prefetch (vs never) | retain vs LRU (wall) | +PF vs LRU (wall) | **+PF vs LRU (stall)** |")
        body.append("|---|---|---|---|---|---|---|---|")
        for win in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            c = Cell(CAT, budget, slots, win=win)
            lru = c.arm(policy="lru", retain="all", accuracy=1.0, prefetch=False)
            ret = c.arm(policy="greedy", retain="all", accuracy=ACC, prefetch=False)
            pf = c.arm(policy="greedy", retain="all", accuracy=ACC, prefetch="slack")
            body.append(f"| {win:g} | {lru['compute_share']:.1f}% | "
                        f"{lru['wall_vs_never']:.2f}% | {ret['wall_vs_never']:.2f}% | "
                        f"{pf['wall_vs_never']:.2f}% | {ret['wall_vs_lru']:+.2f}% | "
                        f"{pf['wall_vs_lru']:+.2f}% | **{pf['stall_vs_lru']:+.2f}%** |")
    body.append("""
## Reading this

- **Wall-clock columns collapse toward zero as compute rises** — purely
  denominator dilution. The mechanism is unchanged.
- **The stall column moves the other way**, improving with compute, because a
  larger window hides more of a prefetch. This is the honest invariant.
- **exp3's real trials sit at ~5% compute** (280.3 s of agent work in a 5288.5 s
  trial) — the top row, where wall-clock numbers look best. Quoting from there
  without the compute share attached repeats the error that produced the
  retracted 57.7% retention headline.
""")
    w("03_compute_sweep.md", "\n".join(body))

    # ----------------------------------------------------- 04 accuracy sweep
    body = [f"# Predictor accuracy sweep\n{PREAMBLE}",
            "**LRU needs no predictor at all**, so its row is the line any "
            "policy claim must clear. This project's measured *identity* accuracy "
            "is 45-62%; nothing has measured *time-to-next-use* accuracy, which is "
            "what these policies actually consume.\n"]
    for budget, slots in ((256.0, 1), (560.0, 2)):
        body.append(f"\n## Budget {budget:.0f} GB, GPU slots {slots} — all vs LRU (wall)\n")
        body.append("| accuracy | greedy retain | exact retain | greedy + slack PF | greedy + outbid PF |")
        body.append("|---|---|---|---|---|")
        c = Cell(CAT, budget, slots)
        for acc in (0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
            g = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch=False)
            e = c.arm(policy="exact", retain="all", accuracy=acc, prefetch=False)
            sp = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch="slack")
            op = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch=True)
            star = " **<- measured range**" if acc == 0.55 else ""
            body.append(f"| {acc:.2f}{star} | {g['wall_vs_lru']:+.2f}% | {e['wall_vs_lru']:+.2f}% | "
                        f"{sp['wall_vs_lru']:+.2f}% | {op['wall_vs_lru']:+.2f}% |")
    body.append("""
## Reading this

- **Slack prefetch is nearly accuracy-independent.** A prefetch that never
  displaces anything wastes only space nothing else wanted, so a wrong
  prediction costs almost nothing. The *safety rule substitutes for accuracy* —
  a cheaper thing to engineer.
- **Outbidding prefetch is violently accuracy-sensitive**, because a wrong
  prediction evicts something valuable *and* wastes the load.
- **Exact subset selection degrades faster than greedy** under noise: it commits
  to a global optimum, so one wrong horizon restructures the whole retained set,
  while greedy evicts one item at a time and re-evaluates.
""")
    w("04_accuracy_sweep.md", "\n".join(body))

    # -------------------------------------------------------- 05 scale sweep
    body = [f"# Scale sweep — more resources shift value from retention to prefetch\n{PREAMBLE}",
            "**Caveat on budget:** holding budget at a fixed *fraction of total "
            "footprint* does not hold pressure constant as scale grows — a "
            f"{NEEDS}-need schedule touches only ~16 of 28 resources, so the "
            "working set becomes a shrinking share of the budget. Both a fixed "
            "fraction and a tighter one are shown.\n"]
    for frac in (0.25, 0.50):
        body.append(f"\n## Budget = {frac:.0%} of footprint, GPU slots 1 — all vs LRU (wall)\n")
        body.append("| models | data | n | first-use share | greedy retain | greedy oracle | + slack prefetch |")
        body.append("|---|---|---|---|---|---|---|")
        for M, D in ((3, 2), (6, 4), (10, 8), (16, 12)):
            cat = V2.build_catalogue(M, D)
            bud = frac * sum(r["held_gb"] for r in cat.values())
            c = Cell(cat, bud, 1)
            g = c.arm(policy="greedy", retain="all", accuracy=ACC, prefetch=False)
            o = c.arm(policy="greedy", retain="all", accuracy=1.0, prefetch=False)
            p = c.arm(policy="greedy", retain="all", accuracy=ACC, prefetch="slack")
            tot = first = 0
            for k, sc in enumerate(schedules(cat, 0, SCHEDSEEDS[0])):
                seen = set()
                for kind, val in sc:
                    if kind != "need":
                        continue
                    tot += 1
                    if val not in seen:
                        first += 1
                    seen.add(val)
            body.append(f"| {M} | {D} | {len(cat)} | {100*first/tot:.1f}% | "
                        f"{g['wall_vs_lru']:+.2f}% | {o['wall_vs_lru']:+.2f}% | "
                        f"{p['wall_vs_lru']:+.2f}% |")
    body.append("""
## Reading this

**The two mechanisms serve disjoint populations of misses.**

- *Retention* can only help a **reuse**. It peaks at moderate scale and fades,
  and the **oracle fades with it** — so this is a structural limit, not a
  prediction-quality problem.
- *Prefetch* is the only thing that can help a **first use**, and the first-use
  share rises from ~10% to ~33% as resources are added at fixed schedule length.

Their relative worth is therefore set by **reuse density** (needs per resource),
which is the quantity to report when describing a workload.
""")
    w("05_scale_sweep.md", "\n".join(body))

    # -------------------------------------------------- 06 prefetch variants
    body = [f"# Prefetch variants — scope and safety rule\n{PREAMBLE}",
            "Two independent restrictions on what a **wrong** prefetch may destroy:\n\n"
            "- **scope**: all classes, or data only. A model prefetch must displace a "
            "live GPU occupant (vLLM's L1 park state only exists for an engine already "
            "on a GPU), so a wrong one destroys a resident model *and* wastes a "
            "several-hundred-second load. A wrong data prefetch wastes only RAM and "
            "bandwidth (E3: 8 concurrent background parses slowed the foreground by "
            "1.000x).\n"
            "- **safety**: *outbid* lets a prefetch evict a retained resource; *slack* "
            "restricts it to leftover budget.\n"]
    for budget, slots in ((256.0, 1), (560.0, 2)):
        body.append(f"\n## Budget {budget:.0f} GB, GPU slots {slots} — all vs LRU (wall)\n")
        body.append("| accuracy | retain only | data + outbid | data + slack | all + outbid | all + slack |")
        body.append("|---|---|---|---|---|---|")
        c = Cell(CAT, budget, slots)
        for acc in (0.40, 0.55, 0.70, 0.85, 1.00):
            r0 = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch=False)
            do = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch="data")
            ds = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch="slack_data")
            ao = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch=True)
            asl = c.arm(policy="greedy", retain="all", accuracy=acc, prefetch="slack")
            body.append(f"| {acc:.2f} | {r0['wall_vs_lru']:+.2f}% | {do['wall_vs_lru']:+.2f}% | "
                        f"{ds['wall_vs_lru']:+.2f}% | {ao['wall_vs_lru']:+.2f}% | "
                        f"{asl['wall_vs_lru']:+.2f}% |")
    body.append("""
## Reading this

The **safety rule matters more than the scope restriction**, and neither is a
better predictor — both simply bound what a wrong prediction can destroy.

A slack-only prefetch cannot make the system worse by construction, so any
measured harm under that rule is a bug signal. That is exactly how the
gratuitous-eviction defect of 2026-08-07 was found: a predicted-dead retention
was being dropped with 4000 GB of free budget.
""")
    w("06_prefetch_variants.md", "\n".join(body))

    # -------------------------------------------------- 07 objective check
    body = [f"# The value function — and the invariant that filters it\n{PREAMBLE}",
            """## The oracle-monotonicity invariant

Adding prefetch as an **option** under perfect next-use knowledge must never
lower the result: a correct scorer could simply decline to prefetch. Necessary
but not sufficient — every arm here is a myopic one-step solve, so a small
violation is possible even with a right objective. A large one is a broken
objective.

## The three candidates

| objective | value of a candidate | verdict |
|---|---|---|
| `rate` | `benefit / dt` | **FAILS.** Prefetch benefit is capped at `min(load, dt)`, so as `dt -> 0` the rate tends to a flat `1/GB`. Verified: uniref50 scored 0.02772 = exactly 1/36.08 GB. A 5-second prefetch outranks and evicts a 798-second retention. |
| `total` | `benefit` | Passes the prefetch filter, **breaks retention**: two retentions of equal size and cost score identically whether needed in 10 s or 10,000 s. No anti-hoarding pressure. |
| `horizon` | `benefit * H / max(dt, H)` | **Only variant good at both.** Absolute seconds inside a planning horizon `H` so the two actions are commensurable in the near field; decays like `1/dt` beyond it so hoarding is still penalised. |
"""]
    body.append("\n## Invariant check — delta is (retain+prefetch) minus (retain only), oracle\n")
    body.append("| objective | policy | slots | retain only | + prefetch | delta | verdict |")
    body.append("|---|---|---|---|---|---|---|")
    for obj in ("rate", "total", "horizon"):
        for pol in ("greedy", "exact"):
            for slots in (1, 2):
                nw, ns, _ = measure(CAT, 419.0, slots, **ARM_NEVER)
                # `measure` hardcodes objective="horizon", so the objective under
                # test has to be applied here rather than through it.
                def m(pf):
                    W = 0.0
                    for ps in POPSEEDS:
                        for ss in SCHEDSEEDS:
                            for k, sc in enumerate(schedules(CAT, ps, ss)):
                                W += V2.Sim(CAT, sc, 419.0, slots, policy=pol,
                                            retain="all", accuracy=1.0, prefetch=pf,
                                            seed=k, objective=obj, H=HORIZON_H).run()["wall"]
                    return W
                a, b = m(False), m(True)
                ra = 100 * (nw - a) / nw
                rb = 100 * (nw - b) / nw
                d = rb - ra
                v = "PASS" if d >= -0.5 else ("marginal" if d >= -2 else "**FAIL**")
                body.append(f"| `{obj}` | {pol} | {slots} | {ra:.2f}% | {rb:.2f}% | {d:+.2f} | {v} |")
    body.append("""
## Horizon `H` sweep (`horizon` objective, oracle, slots 1, budget 419 GB)

Smaller `H` is better because the pathology lives at **small** `dt`; any `H`
above the typical inter-need gap (median 16.3 s in the real traces) fixes it.
`H -> inf` degenerates to `total`; `H -> 0` degenerates to `rate`.
""")
    body.append("\n| H (s) | greedy retain | greedy + prefetch | exact retain | exact + prefetch |")
    body.append("|---|---|---|---|---|")
    for Hh in (60.0, 150.0, 300.0, 600.0, 1200.0, 3000.0):
        vals = []
        for pol in ("greedy", "exact"):
            for pf in (False, True):
                W = 0.0
                for ps in POPSEEDS:
                    for ss in SCHEDSEEDS:
                        for k, sc in enumerate(schedules(CAT, ps, ss)):
                            W += V2.Sim(CAT, sc, 419.0, 1, policy=pol, retain="all",
                                        accuracy=1.0, prefetch=pf, seed=k,
                                        objective="horizon", H=Hh).run()["wall"]
                vals.append(W)
        nw, _, _ = measure(CAT, 419.0, 1, **ARM_NEVER)
        body.append(f"| {Hh:.0f} | " + " | ".join(f"{100*(nw-x)/nw:.2f}%" for x in vals) + " |")
    w("07_objective_check.md", "\n".join(body))

    print(f"\nAll tables written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
