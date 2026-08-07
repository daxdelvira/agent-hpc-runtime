#!/usr/bin/env python3
"""
sim_residency_v2.py — residency simulator with the List-A fairness fixes.

SUPERSEDES `search_ceiling_regime.py`, which is kept only for provenance of the
numbers already reported. Ten defects in that simulator made the policy arms
face different problems rather than solve the same one differently. Each fix
below is numbered against the List-A audit of 2026-08-07.

  A1  PREDICT ONCE PER (resource, decision). The old exact arm called the
      predictor inside its combination loop -- once per (combo, resource) pair --
      so the SAME resource was re-randomised across combinations within a single
      decision. Measured: 63 RNG draws for exact vs 15 for greedy on one 24-need
      schedule. That is not "exact optimisation over noisy estimates," it is
      exact optimisation over MUTUALLY INCONSISTENT estimates, and it is the
      likeliest cause of the reported "exact is fragile" result.
      Here every decision builds ONE horizon dict and every policy reads it.

  A2  ALL ARMS FACE THE SAME NOISE. The old model drew from a sequential RNG, so
      policies that consult it a different number of times saw different noise
      realisations -- the comparison confounded algorithm with luck. Here the
      error overlay is a deterministic function of (seed, resource, step) via a
      hash, so it does not depend on call order, call count, or which policy is
      asking.

      NOTE ON THE ONE THING THAT LEGITIMATELY DIFFERS. The TRUTH is
      state-dependent (see A5), and arms hold different things, so their true
      horizons genuinely differ. What must not differ is the NOISE: whether the
      predictor errs at (resource, step), and which resource it confuses it
      with, are arm-independent. Arms then differ only where their states really
      differ.

  A3  THE PREDICTOR CAN NOW SAY "NEVER AGAIN", AND CAN BE WRONG BOTH WAYS. The
      old error branch filtered `inf` out of the candidate answers, so a wrong
      prediction always returned a finite horizon. Errors could only resurrect
      dead resources, never kill live ones -- a free gift, and `dt=inf` is the
      strongest eviction signal in the value function. Confusion with another
      resource now inherits that resource's horizon INCLUDING inf.

  A4  ONE ANSWER PER (resource, step). The old model re-drew on every call, so
      the same resource could be predicted correctly at one consultation and
      wrongly at the next within one step. A predictor is a function of observed
      state; it does not change its mind between two questions at the same step.

  A5  HORIZONS ARE PRICED AGAINST CURRENT RESIDENCY, NOT ALWAYS-COLD. The old
      `true_hold` charged every intervening need its COLD cost even when that
      need would be served free off the GPU. Observed in a walkthrough: uniref90
      reported "next use in 4824 s" when the intervening model needs were mostly
      GPU-resident and free. This inflated dt, deflated density, and biased ONLY
      the cost-aware arms, since LRU never reads dt. Intervening needs are now
      priced at what the CURRENT state says they would cost -- 0 if on GPU,
      ready_s if parked, cold_s otherwise -- which is information a real runtime
      genuinely has at decision time.

  A6  EXACT NO LONGER HOARDS DEAD RESOURCES. Combos were enumerated from the
      largest subset down with a strict `>`, and dt=inf items contributed 0 to
      the score rather than being excluded -- so a set containing a dead resource
      TIED the set without it and, being enumerated first, won. It burned budget
      on resources it believed would never be used again, and only exact did it.
      Dead resources are now excluded from the pool, and ties break toward the
      SMALLER footprint.

  A8  GPU DISPLACEMENT IS NOW POLICY-DRIVEN. It was hardcoded LRU in every arm,
      which isolated the host-RAM decision but left the larger cost (which model
      loses its slot) permanently un-optimised. Slots are uniform in size, so the
      right rule there is furthest-predicted-next-use; LRU keeps oldest-last-use.

  A9  ONLY THE VERY FIRST NEED IS FREE, WHATEVER ITS CLASS. The old code freed
      the first MODEL even if it appeared at step 10 behind several data needs.
      Per the project owner: a leading data activation is also free because no
      computation has happened yet and the flow has not started -- but once ANY
      need has been served, work has occurred, and a competent predictor had
      something to act on. So exactly one need is free: index 0.

  A10 PREFETCH EXISTS, AND SHARES THE ONE BUDGET. Absent entirely before, though
      it is half the claim. During a compute window the policy may start loading
      a resource it expects to need. An in-flight prefetch OCCUPIES the budget
      from issue until the need, exactly as a retained resource does, so retain
      and prefetch genuinely compete. A prefetch issued at t completes at
      t + cold_s; if the need arrives first, only the elapsed portion is hidden
      and the remainder is still paid -- so a 372.6 s UniRef90 load into a 5 s
      window hides 5 s, not 372.6.
      PHYSICAL CONSTRAINT: prefetching a MODEL requires a free GPU slot. vLLM's
      L1 park state only exists for an engine that has already been started on a
      GPU, so "stage the weights to host RAM without touching the GPU" is not an
      available action. Data has no such constraint (E3: eight concurrent
      background parses cost the foreground nothing measurable).

  A11 ERROR IS MODELLED ON THE QUANTITY ACTUALLY CONSUMED. Every resource gets a
      predicted horizon at every decision, so identity is implicit -- "which
      resource is next" is just "whose horizon is smallest." One error model on
      one quantity, rather than an identity-accuracy number borrowed and applied
      to a time estimate.

DEFERRED by explicit decision: A7 (a Belady arm), A12 (charging the policy's own
runtime -- we want ideal-case savings, and nobody would deploy a 2^n search),
A13 (larger schedule populations, once iteration speed is no longer the binding
constraint).

STILL SIMULATION. Resource constants are measured; schedules and the clock are
not. Nothing here runs on a GPU.
"""
from __future__ import annotations

import hashlib
import itertools
import struct

# --- measured resources; see bench_arbitration_harness.py for provenance ------
REAL_MODELS = [("qwen_32b", 129.7, 495.2, 1.03),
               ("qwen_72b", 279.0, 800.5, 2.21),
               ("qwen_72b_text", 276.3, 770.3, 2.19)]
REAL_DATA = [("uniref90", 117.20, 372.6, 0.0),
             ("uniref50", 36.08, 107.1, 0.0)]
TRACE_POP = {"qwen_72b": 99, "uniref90": 71, "qwen_72b_text": 47, "qwen_32b": 27}


def build_catalogue(n_models: int, n_data: int) -> dict:
    cat = {}
    for k in range(n_models):
        if k < len(REAL_MODELS):
            n, gb, cold, ready = REAL_MODELS[k]
        else:
            j = k - len(REAL_MODELS) + 1
            gb = 150.0 + 60.0 * j
            cold = 500.0 + 150.0 * j
            ready = gb / (16.6 * 4)
            n = f"model_syn{k}"
        cat[n] = dict(cls="model", held_gb=gb, cold_s=cold, ready_s=ready)
    for k in range(n_data):
        if k < len(REAL_DATA):
            n, gb, cold, ready = REAL_DATA[k]
        else:
            j = k - len(REAL_DATA) + 1
            gb = 40.0 + 45.0 * j
            cold, ready, n = gb * 3.1, 0.0, f"data_syn{k}"
        cat[n] = dict(cls="data", held_gb=gb, cold_s=cold, ready_s=ready)
    return cat


def popularity_order(cat):
    return sorted(cat, key=lambda n: (-TRACE_POP.get(n, 0), n))


def _u01(*parts) -> float:
    """Deterministic uniform in [0,1) from the parts. A2/A4: independent of call
    order and call count, so every arm sees the same draw for the same key."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return struct.unpack("<Q", h)[0] / 2.0 ** 64


class Sim:
    """One schedule, one policy, one budget. Call .run()."""

    def __init__(self, cat, sched, budget, slots, policy,
                 accuracy=1.0, prefetch=False, seed=0, log=None):
        self.cat, self.sched = cat, sched
        self.budget, self.slots = budget, slots
        self.policy, self.accuracy = policy, accuracy
        self.prefetch, self.seed, self.log = prefetch, seed, log
        self.gpu: list[str] = []              # models resident on GPU, MRU last
        self.ram: dict[str, int] = {}         # parked models + activated data
        self.inflight: dict[str, float] = {}  # name -> completion clock (A10)
        self.clock = 0.0
        self.stall = 0.0
        self.compute = 0.0
        self.hidden = 0.0                     # seconds a prefetch actually saved
        self.wasted_gb_s = 0.0

    # -- budget ------------------------------------------------------------
    def _occupied(self, extra=()):
        held = set(self.ram) | set(self.inflight) | set(extra)
        return sum(self.cat[x]["held_gb"] for x in held)

    # -- truth, priced against CURRENT residency (A5) -----------------------
    def _cost_if_needed_now(self, name):
        if name in self.gpu:
            return 0.0
        if name in self.ram:
            return self.cat[name]["ready_s"]
        return self.cat[name]["cold_s"]

    def _true_hold(self, name, i):
        """Wall-seconds from decision i until `name` is next needed, inf if never.

        A5: intervening needs are priced at what the CURRENT state says they
        cost, not unconditionally cold. A runtime genuinely knows its own
        residency; assuming every intervening need is a cold load inflated every
        horizon and penalised only the arms that read horizons.
        """
        j = None
        for k in range(i + 1, len(self.sched)):
            if self.sched[k][0] == "need" and self.sched[k][1] == name:
                j = k
                break
        if j is None:
            return float("inf")
        t = 0.0
        for k in range(i + 1, j):
            kind, val = self.sched[k]
            t += val if kind == "compute" else self._cost_if_needed_now(val)
        return t

    # -- the predictor (A1-A4, A11) ----------------------------------------
    def _predict_hold(self, name, i):
        """Predicted seconds until `name` is next needed.

        A11: every resource gets a horizon, so identity prediction is implicit --
        "what is needed next" is just "whose horizon is smallest."
        A2/A4: the ERROR is a deterministic function of (seed, name, i). Whether
        we err, and which resource we confuse `name` with, do not depend on who
        asks or how often.
        A3: a confusion inherits the other resource's horizon INCLUDING inf, so
        the predictor can wrongly declare a live resource dead as well as the
        reverse.
        """
        if self.accuracy >= 1.0:
            return self._true_hold(name, i)
        if _u01("hit", self.seed, name, i) < self.accuracy:
            return self._true_hold(name, i)
        others = [o for o in sorted(self.cat) if o != name]
        if not others:
            return self._true_hold(name, i)
        pick = others[int(_u01("confuse", self.seed, name, i) * len(others))]
        return self._true_hold(pick, i)       # may be inf -- that is the point

    def _horizons(self, names, i):
        """A1: ONE dict per decision. Every policy reads this same dict."""
        return {n: self._predict_hold(n, i) for n in sorted(names)}

    @staticmethod
    def _density(cat, name, dt):
        if dt == float("inf"):
            return float("-inf")
        r = cat[name]
        return (r["cold_s"] - r["ready_s"]) / max(r["held_gb"] * dt, 1e-9)

    # -- host-RAM selection -------------------------------------------------
    def _choose_ram(self, cands: dict, hor: dict) -> set:
        cat, budget = self.cat, self.budget
        fits = lambda s: sum(cat[x]["held_gb"] for x in s) <= budget

        if self.policy == "lru":
            keep = set(cands)
            while not fits(keep):
                keep.discard(min(sorted(keep), key=lambda x: (cands[x], x)))
            return keep

        if self.policy == "greedy":
            keep = set(cands)
            while not fits(keep):
                keep.discard(min(sorted(keep),
                                 key=lambda x: (self._density(cat, x, hor[x]), x)))
            return keep

        # exact. A6: dead resources are EXCLUDED, not scored as zero -- keeping
        # one used to tie and win by being enumerated first, burning budget.
        pool = sorted(x for x in cands if hor[x] != float("inf"))
        best, best_v, best_gb = set(), -1.0, float("inf")
        for r in range(len(pool), -1, -1):
            for combo in itertools.combinations(pool, r):
                gb = sum(cat[x]["held_gb"] for x in combo)
                if gb > budget:
                    continue
                v = sum((cat[x]["cold_s"] - cat[x]["ready_s"]) / max(hor[x], 1e-9)
                        for x in combo)
                # A6: ties break toward the SMALLER footprint, so equal value
                # never costs extra budget.
                if v > best_v or (v == best_v and gb < best_gb):
                    best, best_v, best_gb = set(combo), v, gb
        return best

    def _settle_ram(self, cands: dict, i: int):
        hor = self._horizons(cands, i) if self.policy != "lru" else {}
        keep = self._choose_ram(cands, hor)
        if self.log is not None and cands:
            self.log.append(dict(kind="ram", step=i, kept=sorted(keep),
                                 dropped=sorted(set(cands) - set(keep)),
                                 horizons={k: round(v, 1) if v != float("inf")
                                           else "never" for k, v in hor.items()}))
        return {k: v for k, v in cands.items() if k in keep}

    # -- GPU displacement, now policy-driven (A8) ---------------------------
    def _gpu_victim(self, i):
        """Slots are uniform in size, so furthest-next-use is the right rule for
        the cost-aware arms; LRU keeps its own past-only rule."""
        if self.policy == "lru":
            return self.gpu[0]                       # oldest slot occupant
        hor = self._horizons(self.gpu, i)
        return max(sorted(self.gpu), key=lambda x: (hor[x], x))

    # -- prefetch (A10) -----------------------------------------------------
    def _maybe_prefetch(self, i, window):
        if not self.prefetch or self.policy == "lru":
            return
        hor = self._horizons(self.cat, i)
        cands = []
        for n, dt in hor.items():
            if dt == float("inf") or n in self.gpu or n in self.ram \
               or n in self.inflight:
                continue
            # THE CURRENT COMPUTE STEP IS PART OF THE HORIZON. `_true_hold(n, i)`
            # sums steps AFTER i, which is right for a retention decision taken
            # just after need i -- but during compute step i, that step IS the
            # window we are prefetching into. Omitting it made every horizon 0
            # for a resource needed immediately after the window, so
            # benefit = min(load, 0) = 0 and no prefetch was ever issued.
            dt = dt + window
            r = self.cat[n]
            # A model prefetch needs a free GPU slot: L1 park state only exists
            # for an engine already started on a GPU.
            if r["cls"] == "model" and len(self.gpu) >= self.slots:
                continue
            benefit = min(r["cold_s"] - r["ready_s"], dt)
            if benefit <= 0:
                continue
            cands.append((benefit / max(r["held_gb"] * dt, 1e-9), n))
        for _rate, n in sorted(cands, reverse=True):
            if self._occupied(extra=(n,)) > self.budget:
                continue
            self.inflight[n] = self.clock + self.cat[n]["cold_s"]
            if self.cat[n]["cls"] == "model":
                self.gpu.append(n)               # holds the slot while loading
            if self.log is not None:
                self.log.append(dict(kind="prefetch", step=i, resource=n,
                                     done_at=round(self.inflight[n], 1)))

    # -- main loop ----------------------------------------------------------
    def run(self):
        first_need = next((k for k, (t, _) in enumerate(self.sched)
                           if t == "need"), None)
        for i, (kind, val) in enumerate(self.sched):
            if kind == "compute":
                self._maybe_prefetch(i, val)
                self.clock += val
                self.compute += val
                continue

            name = val
            r = self.cat[name]

            # --- what does this need cost, given where the resource is? ------
            # ORDER MATTERS. `inflight` is checked BEFORE `gpu` because a model
            # prefetch holds its GPU slot WHILE LOADING -- testing `gpu` first
            # would charge 0 for a load that has not finished.
            if i == first_need:
                # A9: exactly one need is free -- the very first, whatever its
                # class. Before it, no computation has happened and the flow has
                # not started; after it, a predictor had something to act on.
                cost, where = 0.0, "FREE-FIRST"
            elif name in self.inflight:
                remaining = max(0.0, self.inflight[name] - self.clock)
                cost = remaining
                where = "PREFETCH-HIT" if remaining <= 0 else "PREFETCH-PARTIAL"
                self.hidden += r["cold_s"] - remaining
            elif name in self.gpu:
                cost, where = 0.0, "GPU"
            elif name in self.ram:
                cost, where = r["ready_s"], "PARK"
            else:
                cost, where = r["cold_s"], "COLD"

            # --- the resource is now live; it leaves every staging area ------
            self.inflight.pop(name, None)
            self.ram.pop(name, None)      # BEFORE any settle: it no longer
                                          # competes for RAM it is vacating

            if r["cls"] == "model":
                if name in self.gpu:
                    self.gpu.remove(name)
                elif len(self.gpu) >= self.slots:
                    out = self._gpu_victim(i)
                    self.gpu.remove(out)
                    self.inflight.pop(out, None)   # an unfinished prefetch dies
                    cand = dict(self.ram); cand[out] = i
                    self.ram = self._settle_ram(cand, i)
                self.gpu.append(name)              # MRU last

            self.stall += cost
            self.clock += cost
            if self.log is not None:
                self.log.append(dict(kind="need", step=i, resource=name,
                                     where=where, cost=round(cost, 1)))

            if r["cls"] == "data":
                cand = dict(self.ram); cand[name] = i
                self.ram = self._settle_ram(cand, i)

            # Drop prefetches that no longer fit alongside what we now hold.
            while self._occupied() > self.budget and self.inflight:
                drop = max(sorted(self.inflight), key=lambda x: self.inflight[x])
                self.inflight.pop(drop)
                if drop in self.gpu:
                    self.gpu.remove(drop)

        return dict(wall=self.clock, stall=self.stall, compute=self.compute,
                    hidden=self.hidden)


def run_arm(cat, scheds, budget, slots, policy, accuracy=1.0, prefetch=False):
    w = s = 0.0
    for k, sc in enumerate(scheds):
        out = Sim(cat, sc, budget, slots, policy, accuracy, prefetch, seed=k).run()
        w += out["wall"]; s += out["stall"]
    return w, s
