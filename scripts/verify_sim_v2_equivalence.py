#!/usr/bin/env python3
"""
verify_sim_v2_equivalence.py — the wired simulator must produce the SAME
numbers as the pre-wiring one, bit for bit.

WHY THIS IS IN THE TREE RATHER THAN IN SOMEBODY'S SCRATCH DIRECTORY. On
2026-08-30 `sim_residency_v2.py` stopped carrying its own copy of Eq. 1, its
own density and its own eviction loops and started importing
`runtime.residency`. The claim that this changed no number is only worth
something if the grid behind it can be inspected and re-run: a sweep that
happens to miss `exact + prefetch + horizon` at sub-oracle accuracy will
report "0 differ" while a real divergence sits in that arm. V found exactly
that gap in an earlier, unshipped version of this script. So the grid is
explicit, the arm that matters is asserted to be IN it, and the whole thing
runs from the repo.

A BEHAVIOURAL SWEEP IS A WEAK DETECTOR FOR THIS DEFECT CLASS. It only sees a
last-bit difference when the run happens to hit an exact tie in `_knapsack`,
which needs a particular (catalogue, schedule, budget, slots, H) corner. Both
defects below were invisible to 21,600 configurations of the generic grid in
check (b) and to all 25 checks in verify_sim_v2.py, while check (a) flagged
each one instantly. That is why check (a) gates the exit code, why check (c)
carries the specific corner that is known to be tie-sensitive, and why check
(d) is a NEGATIVE CONTROL: a grid that cannot detect the defect it exists to
detect is worth nothing, and it must prove it can before its silence means
anything. (V's own sweep printed "control failed" on its negative control; the
sensitive arm below is reconstructed from V's grid and does fire.)

THE FAILURE MODE THIS GUARDS IS ONE ULP, NOT ONE PERCENT. `_knapsack` breaks
ties on exact float equality (`v == best_v and gb < best_gb`), so a last-bit
difference in the value function selects a different subset and moves wall
time by double-digit percentages. Two such defects have already been found:

  * `contract.value` computed `benefit * decay_s / max(dt, decay_s)`, making
    the saturated case `(benefit*60.0)/60.0` rather than `benefit`. Found by V;
    fixed in contract.py by grouping the ratio.
  * `_capped_spec` computed the prefetch benefit as `(ready_s + m) - ready_s`.
    Found here, in A1's own code, by check (a) below: 33 of 224 (resource, dt)
    pairs off by up to 3.55e-15. Fixed by anchoring at ready_s=0.0.

Measured consequences of each, on the sensitive corner (check (c)), with the
defect re-introduced: the ungrouped value function moves 10 of 11,520 configs,
worst 15.38% of wall (exact, prefetch, accuracy 0.55, slots 1, H=15, budget
419 GB: 14898.81 -> 12607.07 s); the old `_capped_spec` moves 14 of 11,520,
worst 5.28%. They are INDEPENDENT defects that happened to land in the same
arm. Both are fixed; check (c) is 0/11,520 today.

Usage:  python3 scripts/verify_sim_v2_equivalence.py [--quick]
Exit 0 iff every configuration matches. No GPU, no SLURM, writes nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _head_simulator(ref: str = "HEAD"):
    """The pre-wiring simulator, taken from git rather than kept as a copy so
    it cannot silently rot out of step with what was actually shipped."""
    blob = subprocess.run(
        ["git", "-C", ROOT, "show", f"{ref}:scripts/sim_residency_v2.py"],
        capture_output=True, text=True, check=True).stdout
    tmp = os.path.join(tempfile.mkdtemp(prefix="sim_head_"), "sim_head.py")
    with open(tmp, "w") as f:
        f.write(blob)
    return _load("SIM_HEAD", tmp)


def schedules(cat, n, length, seed):
    """Deterministic mixed schedules. Compute windows span the range that
    matters: seconds (where the prefetch cap bites) to minutes (where a whole
    load hides)."""
    names = sorted(cat)
    out = []
    for k in range(n):
        rng = random.Random(1000 + seed + k)
        sc = []
        for _ in range(length):
            sc.append(("need", rng.choice(names)))
            sc.append(("compute", rng.choice([2.0, 5.0, 30.0, 200.0, 700.0])))
        out.append(sc)
    return out


def check_value_function_is_bit_exact(NEW, verbose=True) -> int:
    """(a) Every value the wired simulator can compute must be bit-identical to
    the arithmetic HEAD used. This is the check that catches a ULP before the
    knapsack turns it into a wall-time difference."""
    from runtime.residency.contract import value as eq1

    cat = NEW.build_catalogue(16, 12)
    bad_retain = bad_prefetch = n_r = n_p = 0
    worst = 0.0
    for name, r in cat.items():
        spec = NEW.spec_for(cat, name)
        full = r["cold_s"] - r["ready_s"]
        for D in (60.0, 150.0, 300.0, 600.0, 1200.0, 3000.0):
            for dt in (0.0, 1.0, D / 2, D, D * 2, 5000.0):
                n_r += 1
                if full * (D / max(dt, D)) != eq1(spec, dt, D, None):
                    bad_retain += 1
        for dt in (0.5, 1.0, 3.3, 7.7, 30.0, 200.0, 700.0, 1234.5):
            n_p += 1
            want = min(full, dt)
            got = NEW._capped_spec(spec, dt).benefit_s
            if want != got:
                bad_prefetch += 1
                worst = max(worst, abs(want - got))
    if verbose:
        print(f"(a) retain values bit-exact vs HEAD arithmetic: "
              f"{n_r - bad_retain}/{n_r}")
        print(f"(a) prefetch capped benefits bit-exact:          "
              f"{n_p - bad_prefetch}/{n_p}"
              + (f"  worst {worst:.3e}" if bad_prefetch else ""))
    return bad_retain + bad_prefetch


def _harness():
    """`bench_arbitration_harness` supplies the Zipf-weighted schedule
    generator the paper's tables use. The sensitive corner needs it: uniform
    random schedules do not produce the ties."""
    return _load("EQUIV_HARN",
                 os.path.join(HERE, "bench_arbitration_harness.py"))


def harness_schedules(H, mod, cat, popseed, schedseed, n, needs=48, win=0.10):
    old_cat = H.CATALOGUE
    H.CATALOGUE = cat
    try:
        return [H.synthetic_schedule(n_needs=needs, window_scale=win,
                                     seed=schedseed + k,
                                     resources=mod.popularity_order(cat, popseed))
                for k in range(n)]
    finally:
        H.CATALOGUE = old_cat


def sensitive_grid(OLD, NEW, nsched=10, verbose=True):
    """(c) THE CORNER THAT IS ACTUALLY TIE-SENSITIVE, reconstructed from V's
    grid. Everything here matters and none of it is in the generic grid:

        catalogue (4 models, 3 data)   -- not (3,2)/(6,4)
        harness Zipf schedules, 48 needs, window_scale 0.10
        slots 1 AND 2                  -- the generic grid fixes slots=1
        H = 15 s as well as 60 s       -- every worst case was at H=15
        budgets 256 / 419 / 1024 GB
        policy exact, prefetch on, accuracy 0.55 -- where all 10 of V's
                                        divergences landed

    Returns (n_configs, [differences]).
    """
    H = _harness()
    cat_o, cat_n = OLD.build_catalogue(4, 3), NEW.build_catalogue(4, 3)
    assert cat_o == cat_n, "catalogues differ"
    n = 0
    diffs = []
    for popseed in (0, 1, 2):
        for schedseed in (9000, 20000):
            scs = harness_schedules(H, OLD, cat_o, popseed, schedseed, nsched)
            assert scs == harness_schedules(H, NEW, cat_n, popseed, schedseed,
                                            nsched), "schedules differ"
            for k, sc in enumerate(scs):
                for pol in ("greedy", "exact"):
                    for pf in (False, True):
                        for slots in (1, 2):
                            for Hh in (15.0, 60.0, 240.0, 600.0):
                                for budget in (256.0, 419.0, 1024.0):
                                    for acc in (1.0, 0.55):
                                        kw = dict(policy=pol, retain="all",
                                                  accuracy=acc, prefetch=pf,
                                                  seed=k, objective="horizon",
                                                  H=Hh)
                                        a = OLD.Sim(cat_o, sc, budget, slots,
                                                    **kw).run()
                                        b = NEW.Sim(cat_n, sc, budget, slots,
                                                    **kw).run()
                                        n += 1
                                        for key in ("wall", "stall", "compute",
                                                    "hidden"):
                                            if abs(a[key] - b[key]) > 1e-9:
                                                diffs.append(
                                                    (pol, pf, acc, slots, Hh,
                                                     budget, popseed, schedseed,
                                                     k, key, a[key], b[key]))
    if verbose:
        print(f"\n(c) sensitive corner: {n} configurations compared, "
              f"{len(diffs)} field-difference(s)")
        for d in diffs[:10]:
            print("    DIFF", d)
    return n, diffs


def _ungrouped_value(spec, dt_s, decay_s, lookahead_s=None):
    """contract.value as it was BEFORE the grouping fix, for the control."""
    if dt_s is None:
        dt = decay_s if lookahead_s is None else lookahead_s
    else:
        dt = max(dt_s, 0.0)
    return spec.benefit_s * decay_s / max(dt, decay_s)


def check_grid_is_sensitive(OLD, nsched=10) -> bool:
    """(d) NEGATIVE CONTROL. Re-introduce the known ULP defect in a private
    copy of the simulator and require the sensitive grid to SEE it. Without
    this, "0 differ" is indistinguishable from "the grid cannot look"."""
    probe = _load("SIM_PROBE", os.path.join(HERE, "sim_residency_v2.py"))
    probe.eq1_value = _ungrouped_value
    probe.eq1_density = (lambda sp, dt, D, L=None:
                         _ungrouped_value(sp, dt, D, L) / sp.held_gb)
    n, diffs = sensitive_grid(OLD, probe, nsched=nsched, verbose=False)
    ok = len(diffs) > 0
    print(f"(d) negative control: the ungrouped value function is detected in "
          f"{len(diffs)} field(s) of {n} configurations -> "
          f"{'grid IS sensitive' if ok else 'GRID IS BLIND -- (c) proves nothing'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one catalogue and fewer budgets; smoke test only")
    ap.add_argument("--ref", default="HEAD", help="git ref for the old simulator")
    args = ap.parse_args()

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    NEW = _load("SIM_NEW", os.path.join(HERE, "sim_residency_v2.py"))
    OLD = _head_simulator(args.ref)

    print(f"comparing scripts/sim_residency_v2.py against {args.ref}\n")
    bad_ulp = check_value_function_is_bit_exact(NEW)

    catalogues = [(3, 2)] if args.quick else [(3, 2), (6, 4)]
    budgets = ([256.0, 419.0] if args.quick
               else [130.0, 256.0, 400.0, 419.0, 900.0])   # 256 = production,
    policies = ("never", "lru", "greedy", "exact")         # 419 = the tables'
    objectives = ("rate", "total", "horizon")
    prefetches = (False, True, "data", "slack", "slack_data")
    accuracies = (1.0, 0.55)          # sub-oracle is where the exact arm bites
    caps = (None, 3000.0)
    retains = ("all", "models", "none")

    # The arm the earlier sweep could have missed. Assert it is present rather
    # than trusting the reader to check the loop bounds.
    assert "exact" in policies and "horizon" in objectives
    assert True in prefetches and 0.55 in accuracies

    n = bad = 0
    diffs = []
    for nm, nd in catalogues:
        cat_o, cat_n = OLD.build_catalogue(nm, nd), NEW.build_catalogue(nm, nd)
        if cat_o != cat_n:
            print("CATALOGUE DIFFERS — nothing else is meaningful")
            return 1
        scs = schedules(cat_o, 3, 24, seed=0)
        for pol in policies:
            for obj in objectives:
                for pf in prefetches:
                    for acc in accuracies:
                        for budget in budgets:
                            for cap in caps:
                                for ret in retains:
                                    for si, sc in enumerate(scs):
                                        kw = dict(accuracy=acc, prefetch=pf,
                                                  seed=si, horizon_cap=cap,
                                                  objective=obj, H=60.0,
                                                  retain=ret)
                                        a = OLD.Sim(cat_o, sc, budget, 1, pol,
                                                    **kw).run()
                                        b = NEW.Sim(cat_n, sc, budget, 1, pol,
                                                    **kw).run()
                                        n += 1
                                        if a != b:
                                            bad += 1
                                            if len(diffs) < 10:
                                                diffs.append(
                                                    (nm, nd, pol, obj, pf, acc,
                                                     budget, cap, ret, si, a, b))
    print(f"\n(b) {n} configurations compared, {bad} differ")
    for d in diffs:
        print("    DIFF", d)

    nsched = 3 if args.quick else 10
    n_c, diffs_c = sensitive_grid(OLD, NEW, nsched=nsched)
    sensitive = check_grid_is_sensitive(OLD, nsched=nsched)

    if bad_ulp:
        print(f"(a) FAILED: {bad_ulp} value(s) not bit-exact")
    if not sensitive:
        print("(d) FAILED: the sensitive grid cannot detect a known defect, so "
              "(c)'s result carries no information")
    return 1 if (bad or bad_ulp or diffs_c or not sensitive) else 0


if __name__ == "__main__":
    raise SystemExit(main())
