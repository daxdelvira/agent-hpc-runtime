#!/usr/bin/env python3
"""
verify_sim_v2.py — one executable check per List-A fix in sim_residency_v2.py.

WHY THIS EXISTS. Every simulator defect found this week was found by INSPECTING
OUTPUT, not by reasoning: the units bug, the LRU that stamped every entry with
the current step, the value function whose denominator cancelled, the per-key
re-randomisation. Each survived because the number it produced looked plausible.
So each fix gets an assertion that fails loudly if it regresses.
"""
from __future__ import annotations

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "V2", os.path.join(HERE, "sim_residency_v2.py"))
V2 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(V2)

CAT = V2.build_catalogue(3, 2)
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


# --- A9: exactly one need is free, whatever its class ----------------------
print("\nA9  only the very first need is free, whatever its class")
sd = [("need", "uniref90"), ("compute", 10.0), ("need", "qwen_72b")]
o = V2.Sim(CAT, sd, 400.0, 1, "lru").run()
check("data-first: data free, later model charged",
      abs(o["stall"] - 800.5) < 1e-6, f"stall={o['stall']:.1f} (expect 800.5)")

sd = [("need", "qwen_72b"), ("compute", 10.0), ("need", "uniref90")]
o = V2.Sim(CAT, sd, 400.0, 1, "lru").run()
check("model-first: model free, later data charged",
      abs(o["stall"] - 372.6) < 1e-6, f"stall={o['stall']:.1f} (expect 372.6)")

sd = [("need", "uniref50"), ("compute", 5.0), ("need", "uniref50")]
o = V2.Sim(CAT, sd, 400.0, 1, "lru").run()
check("free need is genuinely resident afterwards (reuse costs 0)",
      abs(o["stall"]) < 1e-6, f"stall={o['stall']:.1f}")

# --- A2/A4: same (resource, step) -> same prediction, always ---------------
print("\nA2/A4  the predictor is a deterministic function of (seed, resource, step)")
sd = [("need", "qwen_72b"), ("compute", 50.0), ("need", "uniref90"),
      ("compute", 50.0), ("need", "qwen_32b"), ("compute", 50.0),
      ("need", "qwen_72b")]
a = V2.Sim(CAT, sd, 400.0, 1, "greedy", accuracy=0.55, seed=3)
reps = [a._predict_hold("qwen_72b", 2) for _ in range(20)]
check("20 consultations at the same step give one answer", len(set(reps)) == 1,
      f"{len(set(reps))} distinct")

b = V2.Sim(CAT, sd, 400.0, 1, "exact", accuracy=0.55, seed=3)
same = all(a._predict_hold(n, 2) == b._predict_hold(n, 2) for n in CAT)
check("greedy and exact see identical predictions at the same step", same)

c = V2.Sim(CAT, sd, 400.0, 1, "greedy", accuracy=0.55, seed=4)
diff = any(a._predict_hold(n, 2) != c._predict_hold(n, 2) for n in CAT)
check("a different seed does give different noise", diff)

# --- A3: the predictor can be wrong in BOTH directions ---------------------
print("\nA3  errors can wrongly declare a live resource dead, and vice versa")
live_called_dead = dead_called_live = 0
for seed in range(400):
    s = V2.Sim(CAT, sd, 400.0, 1, "greedy", accuracy=0.55, seed=seed)
    for n in CAT:
        t, p = s._true_hold(n, 0), s._predict_hold(n, 0)
        if t != float("inf") and p == float("inf"):
            live_called_dead += 1
        if t == float("inf") and p != float("inf"):
            dead_called_live += 1
check("live wrongly predicted dead occurs", live_called_dead > 0,
      f"{live_called_dead} occurrences over 400 seeds")
check("dead wrongly predicted live occurs", dead_called_live > 0,
      f"{dead_called_live} occurrences over 400 seeds")

# --- A1: one horizon dict per decision, shared by every policy -------------
print("\nA1  predictions are computed once per decision, not once per subset")
calls = {"n": 0}
s = V2.Sim(CAT, sd, 400.0, 1, "exact", accuracy=0.55, seed=1)
_orig = s._predict_hold
s._predict_hold = lambda n, i: (calls.__setitem__("n", calls["n"] + 1),
                                _orig(n, i))[1]
h = s._horizons(list(CAT), 0)
check("_horizons consults the predictor exactly once per resource",
      calls["n"] == len(CAT), f"{calls['n']} calls for {len(CAT)} resources")

# --- A5: horizons priced against current residency, not always-cold --------
print("\nA5  intervening needs priced by residency, not unconditionally cold")
sd5 = [("need", "uniref90"), ("compute", 10.0), ("need", "qwen_72b"),
       ("compute", 10.0), ("need", "uniref90")]
s = V2.Sim(CAT, sd5, 400.0, 1, "greedy")
s.gpu = ["qwen_72b"]                     # 72b already live: costs 0, not 800.5
h_live = s._true_hold("uniref90", 0)
s.gpu = []
h_cold = s._true_hold("uniref90", 0)
check("a GPU-resident intervening need is not charged cold",
      abs(h_live - 20.0) < 1e-6 and abs(h_cold - 820.5) < 1e-6,
      f"live={h_live:.1f} (expect 20.0), cold={h_cold:.1f} (expect 820.5)")

# --- A6: exact does not hoard resources it believes are dead ---------------
print("\nA6  exact excludes resources it predicts will never be used again")
sd6 = [("need", "qwen_32b"), ("compute", 10.0), ("need", "uniref50"),
       ("compute", 10.0), ("need", "uniref90"), ("compute", 500.0),
       ("need", "uniref90")]
s = V2.Sim(CAT, sd6, 400.0, 1, "exact")
hor = {"uniref50": float("inf"), "uniref90": 500.0}
keep = s._choose_ram({"uniref50": 2, "uniref90": 4}, hor)
check("a dead resource is dropped even though it fits",
      keep == {"uniref90"}, f"kept {sorted(keep)} (400 GB budget fits both)")

# --- A8: GPU displacement is policy-driven ---------------------------------
print("\nA8  the GPU victim differs by policy")
sd8 = [("need", "qwen_32b"), ("compute", 5.0), ("need", "qwen_72b"),
       ("compute", 5.0), ("need", "qwen_32b"), ("compute", 5.0),
       ("need", "qwen_72b_text")]
s = V2.Sim(CAT, sd8, 1000.0, 2, "lru")
s.gpu = ["qwen_32b", "qwen_72b"]
v_lru = s._gpu_victim(2)
s2 = V2.Sim(CAT, sd8, 1000.0, 2, "greedy")
s2.gpu = ["qwen_32b", "qwen_72b"]
v_greedy = s2._gpu_victim(2)
check("LRU evicts oldest; cost-aware evicts furthest-next-use",
      v_lru == "qwen_32b" and v_greedy == "qwen_72b",
      f"lru->{v_lru}, greedy->{v_greedy} (32b needed again at step 4)")

# --- A10: prefetch hides only what the window allows ------------------------
print("\nA10  a prefetch hides min(load, window), and occupies the budget")
sd10 = [("need", "qwen_32b"), ("compute", 5.0), ("need", "uniref90")]
o_no = V2.Sim(CAT, sd10, 400.0, 1, "greedy", prefetch=False).run()
o_pf = V2.Sim(CAT, sd10, 400.0, 1, "greedy", prefetch=True).run()
check("a 5 s window hides ~5 s of a 372.6 s load, not 372.6",
      abs(o_no["stall"] - 372.6) < 1e-6 and abs(o_pf["stall"] - 367.6) < 1e-3,
      f"no-pf={o_no['stall']:.1f}, pf={o_pf['stall']:.1f} (expect 367.6)")

sd10b = [("need", "qwen_32b"), ("compute", 500.0), ("need", "uniref90")]
o_pf2 = V2.Sim(CAT, sd10b, 400.0, 1, "greedy", prefetch=True).run()
check("a 500 s window hides the whole 372.6 s load",
      abs(o_pf2["stall"]) < 1e-6, f"stall={o_pf2['stall']:.1f}")

# 100 GB < uniref90's 117.2 GB, so it cannot be staged at all. (An earlier
# version of this check used 130 GB, which uniref90 FITS -- it only passed
# because the prefetch was broken, and would have gone green over a real bug.)
o_tight = V2.Sim(CAT, sd10b, 100.0, 1, "greedy", prefetch=True).run()
check("a budget too small to hold it forbids the prefetch",
      abs(o_tight["stall"] - 372.6) < 1e-6, f"stall={o_tight['stall']:.1f}")

o_fits = V2.Sim(CAT, sd10b, 130.0, 1, "greedy", prefetch=True).run()
check("a budget that does fit it allows the prefetch",
      abs(o_fits["stall"]) < 1e-6, f"stall={o_fits['stall']:.1f}")

# --- A10b: prefetch and retention genuinely compete for one budget ---------
print("\nA10b  a prefetch can OUTBID a retained resource (not just use slack)")
# uniref50 (36.1 GB) is parked and never needed again; uniref90 (117.2 GB) is
# needed right after a long window. Budget 130 GB holds only one of them.
sd11 = [("need", "uniref50"), ("compute", 5.0), ("need", "qwen_32b"),
        ("compute", 600.0), ("need", "uniref90")]
lg = []
V2.Sim(CAT, sd11, 130.0, 1, "greedy", prefetch=True, log=lg).run()
evicted = [e["resource"] for e in lg if e["kind"] == "evict_for_prefetch"]
issued = [e["resource"] for e in lg if e["kind"] == "prefetch"]
check("a retained resource is evicted to make room for a prefetch",
      "uniref50" in evicted and issued,
      f"evicted={evicted}, prefetched={issued}")

# The budget holds only ONE of these at a time, so the two prefetches must be
# issued at DIFFERENT windows: qwen_32b into the 5 s window (hiding 5 s of a
# 495.2 s load -- all the window allows), then uniref90 into the 600 s window
# (hiding all 372.6 s). Residual stall is therefore 495.2 - 5 = 490.2.
# An earlier version of this check asserted uniref90 was starved; that was
# wrong -- it read only the FIRST arbitration and assumed the loser was
# permanently blocked, when it is simply re-evaluated at the next window.
o_slack = V2.Sim(CAT, sd11, 130.0, 1, "greedy", prefetch=True).run()
check("both prefetches are issued, each at the window that fits it",
      issued == ["qwen_32b", "uniref90"] and abs(o_slack["stall"] - 490.2) < 1e-3,
      f"issued={issued}, stall={o_slack['stall']:.1f} (expect 490.2)")

# --- A10c: a model prefetch may take an occupied GPU slot ------------------
print("\nA10c  a model prefetch may displace a GPU occupant (proactive swap)")
# slots=1: qwen_32b is resident and never needed again; qwen_72b is needed after
# a long window. Only a displacing prefetch can hide that load.
sd12 = [("need", "qwen_32b"), ("compute", 900.0), ("need", "qwen_72b")]
o_nopf = V2.Sim(CAT, sd12, 400.0, 1, "greedy", prefetch=False).run()
o_pf = V2.Sim(CAT, sd12, 400.0, 1, "greedy", prefetch=True).run()
check("at slots=1 a model prefetch still happens by taking the slot",
      abs(o_nopf["stall"] - 800.5) < 1e-6 and o_pf["stall"] < 1.0,
      f"no-pf={o_nopf['stall']:.1f}, pf={o_pf['stall']:.1f}")

# --- horizon cap: nothing is ever labelled "never again" -------------------
print("\nhorizon cap  removes the dead/alive distinction symmetrically")
s_cap = V2.Sim(CAT, sd, 400.0, 1, "greedy", accuracy=0.55, seed=1,
               horizon_cap=3000.0)
never = [n for n in CAT if s_cap._true_hold(n, 0) == float("inf")]
check("with a cap, no resource reports an infinite horizon", not never,
      f"{len(never)} still infinite")
s_nocap = V2.Sim(CAT, sd, 400.0, 1, "greedy", accuracy=0.55, seed=1)
check("without a cap, some resource does report infinite",
      any(s_nocap._true_hold(n, 0) == float("inf") for n in CAT))

# --- controls ---------------------------------------------------------------
print("\ncontrols")
sd_c = [("need", "qwen_72b"), ("compute", 10.0), ("need", "qwen_72b"),
        ("compute", 10.0), ("need", "qwen_72b")]
res = {p: V2.Sim(CAT, sd_c, 400.0, 1, p).run()["stall"] for p in
       ("lru", "greedy", "exact")}
check("no decision to make -> all policies identical",
      len(set(res.values())) == 1, str(res))

sd_d = [("need", "qwen_32b"), ("compute", 5.0), ("need", "qwen_72b"),
        ("compute", 5.0), ("need", "qwen_32b")]
o1 = V2.Sim(CAT, sd_d, 300.0, 1, "greedy", accuracy=0.55, seed=7).run()
o2 = V2.Sim(CAT, sd_d, 300.0, 1, "greedy", accuracy=0.55, seed=7).run()
check("identical inputs -> identical output (no hash-order dependence)",
      o1 == o2, f"{o1['stall']:.3f} vs {o2['stall']:.3f}")

o_or = V2.Sim(CAT, sd_d, 300.0, 1, "greedy", accuracy=1.0, seed=0).run()
check("oracle consults no randomness (accuracy=1.0 path)",
      o_or["stall"] >= 0)

print(f"\n{len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
raise SystemExit(1 if FAIL else 0)
