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
      PHYSICAL CONSTRAINT: a MODEL prefetch needs a GPU slot. vLLM's L1 park
      state only exists for an engine already started on a GPU, so "stage the
      weights to host RAM without touching the GPU" is not an available action.
      Data has no such constraint (E3: eight concurrent background parses cost
      the foreground nothing measurable).
      A candidate may TAKE the slot of the occupant whose predicted next use is
      furthest away, and only if the candidate is needed sooner. Requiring a
      *free* slot made model prefetch impossible at slots=1 -- yet that is
      exactly the action `tool_resources.json` already configures as
      `proactive_swap`: during a LAMMPS window the resident model is idle, so
      stopping it and loading the next costs no inference time.

      A10-COMPLETION (2026-08-07). The first version admitted a prefetch only if
      it fit ALONGSIDE everything held, and its cleanup only dropped in-flight
      items -- so a retained resource could never be outbid and prefetch ran
      purely on slack. That is retention with a leftovers policy, not
      arbitration, and it cannot express the trade the system is about. Retain
      and prefetch candidates now enter ONE knapsack in ONE currency.

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


def build_catalogue(n_models: int, n_data: int, proportional_data=False) -> dict:
    """Catalogue of n_models + n_data resources.

    The first 3 models and first 2 data entries are MEASURED (see
    bench_arbitration_harness.py for provenance). Beyond those it extrapolates,
    and the extrapolation is where three defects lived until 2026-08-07:

    (a) ready_s WAS gb / (16.6 * 4), dividing the PARKED FOOTPRINT by wake
        bandwidth. But a wake moves the WEIGHTS, and held_gb is weights x 1.90
        (the measured L1 park ratio), so every synthetic model got a wake 1.9x
        too large. Check against measurement: qwen_72b is 279.0 GB parked with a
        measured 2.21 s wake; 279.0/66.4 = 4.20 s, while (279.0/1.90)/66.4 =
        2.21 s exactly. Now divides the weights.

    (b) MODELS GREW WITHOUT BOUND -- 930 GB parked at n_models=16, roughly 245B
        parameters at fp16, larger than anything this project can obtain. Sizes
        now cycle a realistic band anchored on the three measured models.

    (c) SYNTHETIC DATA HAD cold = gb * 3.1 EXACTLY, so cost was strictly
        proportional to size -- provably the regime where value density reduces
        to Belady's ranking, i.e. the one configuration guaranteed to show no
        benefit from cost-awareness. The MEASURED spread across real formats is
        65x (results/bench_format_activation.csv: ascii 22.0, npz 6.2, EAM
        2.5-5.8, raw_f32 0.45, npy/hdf5 0.34), so synthetic data now spans that
        measured range. Pass proportional_data=True to restore the degenerate
        behaviour as a control.

    MODELS KEEP A NARROW s/GB BAND ON PURPOSE, and that is physics rather than a
    modelling shortcut: model load is dominated by moving weights at roughly
    fixed bandwidth (74-81% of a boot is movement), so seconds-per-GB is
    near-constant across models. The measured three span only 2.78-3.81. Data is
    transformation-bound, so its s/GB is set by the format's decoder and varies
    enormously. That asymmetry is Insight A, and the generator should express it.
    """
    cat = {}
    # parked footprints for synthetic models: a realistic band, cycled rather
    # than grown, anchored on the measured 129.7 / 276.3 / 279.0 GB.
    MODEL_GB = [160.0, 210.0, 340.0, 400.0, 470.0, 560.0]
    # s/GB for synthetic data, from the measured format sweep. Ordered so the
    # early entries stay near the real UniRef values (2.97-3.18).
    DATA_SPG = [3.10, 2.50, 6.20, 0.45, 22.0, 0.34, 5.80, 12.0, 0.37, 8.4]
    DATA_GB = [85.0, 130.0, 60.0, 175.0, 45.0, 220.0, 100.0, 265.0, 70.0, 150.0]

    for k in range(n_models):
        if k < len(REAL_MODELS):
            n, gb, cold, ready = REAL_MODELS[k]
        else:
            j = k - len(REAL_MODELS)
            gb = MODEL_GB[j % len(MODEL_GB)] * (1.0 + 0.15 * (j // len(MODEL_GB)))
            gb = round(gb, 1)
            weights = gb / 1.90                  # measured L1 park ratio
            ready = round(weights / (16.6 * 4), 2)   # measured wake bandwidth
            cold = round(125.0 + 2.5 * gb, 1)    # affine: fixed init + movement
            n = f"model_syn{k:02d}"
        cat[n] = dict(cls="model", held_gb=gb, cold_s=cold, ready_s=ready)

    for k in range(n_data):
        if k < len(REAL_DATA):
            n, gb, cold, ready = REAL_DATA[k]
        else:
            j = k - len(REAL_DATA)
            gb = DATA_GB[j % len(DATA_GB)] * (1.0 + 0.20 * (j // len(DATA_GB)))
            gb = round(gb, 1)
            spg = 3.1 if proportional_data else DATA_SPG[j % len(DATA_SPG)]
            cold = round(gb * spg, 1)
            ready = 0.0
            n = f"data_syn{k:02d}"
        cat[n] = dict(cls="data", held_gb=gb, cold_s=cold, ready_s=ready)
    return cat


def popularity_order(cat, seed: int = 0) -> list:
    """Rank resources most-requested first; synthetic_schedule Zipf-weights by
    POSITION in this list, so this function decides which resources dominate.

    THE MEASURED FOUR keep the order the exp3 traces actually show (qwen_72b 99
    requests, uniref90 71, qwen_72b_text 47, qwen_32b 27).

    EVERYTHING ELSE IS SHUFFLED DETERMINISTICALLY BY `seed`, and that is the fix
    for a defect that survived two rounds. The fallback used to be alphabetical,
    which at n_models=16/n_data=12 produced:

        - uniref50, a MEASURED resource, ranked LAST of 28 (it is absent from
          TRACE_POP, so it fell into the fallback);
        - every data_syn* ranked above every model_syn*, because "d" < "m";
        - syn10/syn11 above syn2..syn9, because string sort, so the two LARGEST
          data artifacts drew the highest synthetic popularity.

    Rank was therefore correlated with class and with size for no reason but
    string collation -- the same defect already fixed once for the 5-resource
    catalogue, reintroduced at scale by repairing only the lookup table and not
    the fallback. We do not know the popularity of hypothetical resources, so the
    honest treatment is to average results over several seeds rather than to
    pick one arbitrary assignment.
    """
    measured = [n for n in sorted(cat, key=lambda x: -TRACE_POP.get(x, 0))
                if n in TRACE_POP]
    rest = sorted(n for n in cat if n not in TRACE_POP)
    rest.sort(key=lambda n: _u01("pop", seed, n))     # deterministic shuffle
    return measured + rest


def _u01(*parts) -> float:
    """Deterministic uniform in [0,1) from the parts. A2/A4: independent of call
    order and call count, so every arm sees the same draw for the same key."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return struct.unpack("<Q", h)[0] / 2.0 ** 64


class Sim:
    """One schedule, one policy, one budget. Call .run()."""

    def __init__(self, cat, sched, budget, slots, policy,
                 accuracy=1.0, prefetch=False, seed=0, log=None,
                 horizon_cap=None, objective="rate", H=600.0,
                 retain="all"):
        self.cat, self.sched = cat, sched
        self.budget, self.slots = budget, slots
        self.policy, self.accuracy = policy, accuracy
        self.prefetch, self.seed, self.log = prefetch, seed, log
        # HORIZON CAP. Without it, `inf` means "never appears again in this
        # schedule" -- which is an artifact of simulating a FINITE trace, and it
        # hands both the oracle and the predictor a "this is definitively dead"
        # signal no bounded-lookahead predictor could ever emit. With a cap, a
        # resource not needed within `cap` seconds simply reports `cap`, so the
        # dead/alive distinction disappears for everyone symmetrically and the
        # remaining ranking is purely relative. Set it to model a predictor that
        # can see H seconds ahead and says nothing beyond that.
        self.horizon_cap = horizon_cap
        # WHICH OBJECTIVE RANKS RETAIN AGAINST PREFETCH. See _value().
        self.objective = objective
        self.H = H
        # WHICH CLASSES MAY BE HELD IN HOST RAM. This exists to separate what a
        # vLLM flag gives from what this project adds. "models" = L1 sleep only,
        # which is what --enable-sleep-mode plus a loop provides; activated data
        # is rebuilt on every use because vLLM has no notion of it. "all" adds
        # the persistent consumer worker. "none" is the true floor.
        self.retain = retain
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
            return float("inf") if self.horizon_cap is None else self.horizon_cap
        t = 0.0
        for k in range(i + 1, j):
            kind, val = self.sched[k]
            t += val if kind == "compute" else self._cost_if_needed_now(val)
        return t if self.horizon_cap is None else min(t, self.horizon_cap)

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

    # -- THE OBJECTIVE: how a candidate's worth is scored --------------------
    def _value(self, name, dt, kind):
        """Absolute worth of holding/staging `name` given predicted horizon `dt`.

        Weight is always held_gb (the budget is an instantaneous GB ceiling), so
        the knapsack maximises sum(_value) and greedy ranks by _value/held_gb.

        THE PROBLEM THESE VARIANTS EXIST TO SOLVE. Retain and prefetch have
        benefit functions of DIFFERENT SHAPE in dt:

            retain    benefit = cold - ready            constant in dt
            prefetch  benefit = min(cold - ready, dt)   grows, then saturates

        Both occupy GB x dt. So dividing by dt decays retain's score as 1/dt
        while leaving prefetch's at a FLAT 1/GB whenever the window is smaller
        than the load -- which is every interesting case here, gaps being seconds
        against loads of hundreds of seconds. Verified: uniref50's prefetch rate
        came out 0.02772 = exactly 1/36.08 GB. The curves therefore cross at a dt
        set by the algebra rather than by the workload, and past that crossing a
        5-second prefetch outranks and EVICTS a 798-second retention.

          "rate"      value = benefit / dt        the original. Broken as above.
          "total"     value = benefit             absolute stall avoided. Ranks
                      the two actions correctly, but two retentions of equal size
                      and cost score identically whether they are needed in 10 s
                      or 10,000 s -- the exact defect fixed in _best_set on
                      2026-08-07, reintroduced. No anti-hoarding pressure.
          "horizon"   value = benefit * H / max(dt, H)
                      Absolute benefit inside a planning horizon H, decaying like
                      1/dt beyond it. Near field: prefetch and retain are directly
                      comparable in seconds, so 798 beats 5 as it must. Far field:
                      a resource needed in 10,000 s at H=600 scores 6% of one
                      needed within the horizon, so hoarding is still penalised.
                      H is a real knob and must be swept, not picked.
        """
        r = self.cat[name]
        full = r["cold_s"] - r["ready_s"]
        if dt == float("inf"):
            return 0.0                      # believed dead: worth nothing
        benefit = full if kind == "retain" else min(full, dt)
        if self.objective == "rate":
            return benefit / max(dt, 1e-9)
        if self.objective == "total":
            return benefit
        if self.objective == "horizon":
            return benefit * (self.H / max(dt, self.H))
        raise ValueError(self.objective)

    def _density(self, name, dt, kind="retain"):
        if dt == float("inf"):
            return float("-inf")            # evicted before anything finite
        return self._value(name, dt, kind) / max(self.cat[name]["held_gb"], 1e-9)

    # -- host-RAM selection -------------------------------------------------
    def _retainable(self, name) -> bool:
        if self.retain == "all":
            return True
        if self.retain == "none":
            return False
        return self.cat[name]["cls"] == self.retain.rstrip("s")

    def _choose_ram(self, cands: dict, hor: dict) -> set:
        cat, budget = self.cat, self.budget
        cands = {k: v for k, v in cands.items() if self._retainable(k)}
        fits = lambda s: sum(cat[x]["held_gb"] for x in s) <= budget

        # THE TRUE FLOOR: no L1 parking, no persistent data worker, no prefetch.
        # A displaced model falls all the way to disk and a data artifact is
        # re-activated on every use. This is vLLM's default (exp3 runs sleep
        # level 2, which DISCARDS weights) plus fork-per-call data loading, and
        # it is the honest denominator -- LRU is already a system, so quoting a
        # policy gain against LRU hides how much of the win is just retention.
        if self.policy == "never":
            return set()

        if self.policy == "lru":
            keep = set(cands)
            while not fits(keep):
                keep.discard(min(sorted(keep), key=lambda x: (cands[x], x)))
            return keep

        if self.policy == "greedy":
            keep = set(cands)
            while not fits(keep):
                keep.discard(min(sorted(keep),
                                 key=lambda x: (self._density(x, hor[x]), x)))
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
                v = sum(self._value(x, hor[x], "retain") for x in combo)
                # A6: ties break toward the SMALLER footprint, so equal value
                # never costs extra budget.
                if v > best_v or (v == best_v and gb < best_gb):
                    best, best_v, best_gb = set(combo), v, gb
        return best

    def _settle_ram(self, cands: dict, i: int):
        hor = ({} if self.policy in ("lru", "never")
               else self._horizons(cands, i))
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
        if self.policy in ("lru", "never"):
            return self.gpu[0]                       # oldest slot occupant
        hor = self._horizons(self.gpu, i)
        return max(sorted(self.gpu), key=lambda x: (hor[x], x))

    # -- retain-vs-prefetch arbitration against ONE budget (A10, completed) ---
    def _arbitrate(self, i, window):
        """Choose what occupies host RAM over this compute window.

        WHAT THIS REPLACES, AND WHY IT WAS WRONG. The first version admitted a
        prefetch only if it fit ALONGSIDE everything already held, and its
        cleanup only ever dropped in-flight items. So a retained resource could
        never be outbid, and a prefetch ran purely on slack -- "held in the
        aether," in the project owner's words. That is not arbitration; it is
        retention with a leftovers policy, and it cannot express the trade the
        whole system is about.

        NOW: retained items and prefetch candidates are scored in ONE currency
        (stall-seconds avoided per GB-second) and compete for one budget. A
        prefetch that is worth more than something currently parked evicts it.

            retain x    benefit = cold(x) - ready(x)     the whole rebuild avoided
            prefetch y  benefit = min(cold(y) - ready(y), dt)
                                                          can only hide as much as
                                                          the window actually gives

        MODEL PREFETCH MAY DISPLACE A GPU OCCUPANT. Requiring a free slot meant
        no model prefetch was ever possible at slots=1 -- yet that is precisely
        the action `tool_resources.json` already configures as `proactive_swap`:
        during a LAMMPS window the resident model is idle, so stopping it and
        loading the next one is free of any inference cost. A candidate may take
        the slot of the occupant whose predicted next use is furthest away, and
        only if the candidate is needed sooner. The displaced model is then a
        park candidate in this same arbitration.
        """
        if not self.prefetch or self.policy in ("lru", "never"):
            return
        # PREFETCH SCOPE. True/"all" stages both classes; "data" stages only
        # activated data. The asymmetry is physical, not a tuning choice: a model
        # prefetch must DISPLACE a live GPU occupant (vLLM's L1 park state only
        # exists for an engine already on a GPU), so a wrong one destroys a
        # resident model AND wastes a several-hundred-second load. A wrong data
        # prefetch only wastes RAM and bandwidth -- and E3 measured background
        # activation as free (8 concurrent parses, foreground slowdown 1.000).
        data_only = self.prefetch in ("data", "slack_data")
        # SLACK-ONLY: a prefetch may use leftover budget but may never outbid a
        # retained resource. Costs a wrong prediction only space nothing else
        # wanted, which is why it is near-insensitive to predictor accuracy.
        slack_only = self.prefetch in ("slack", "slack_data")
        hor = self._horizons(self.cat, i)

        # -- who may take a GPU slot? -------------------------------------
        gpu_free = self.slots - len(self.gpu)
        displaceable = None
        if gpu_free <= 0 and self.gpu:
            worst = max(sorted(self.gpu), key=lambda x: (hor[x], x))
            displaceable = worst

        items = {}          # name -> (value, gb, kind)
        for n in self.ram:                                   # retain candidates
            # EVERY retained resource enters the knapsack, INCLUDING one believed
            # dead (dt=inf -> _value returns 0.0).
            #
            # BUG FIXED 2026-08-07. This used to `continue` past dt=inf, so a
            # predicted-dead retention was absent from `items`, therefore absent
            # from `keep`, and the eviction loop `set(self.ram) - keep` DROPPED IT
            # -- with no regard for whether the budget was under any pressure at
            # all. Demonstrated with a 4000 GB budget against an 838 GB total
            # footprint: uniref50, parked and genuinely reused, was evicted on 3
            # of 12 seeds at 0.55 accuracy, turning a 0.0 s park hit into a
            # 107.1 s cold load. Nothing was competing for the space.
            #
            # This is the A6 defect in its opposite form. In _choose_ram,
            # excluding dead items from the pool is right -- the pool IS the
            # keep-set. Here `items` is scored against an eviction loop, so
            # excluding an item silently evicts it. A value of 0.0 gives the
            # intended semantics: kept when there is room, dropped first when
            # there is not.
            r = self.cat[n]
            items[n] = (self._value(n, hor[n], "retain"), r["held_gb"], "retain")

        for n, dt in hor.items():                            # prefetch candidates
            if dt == float("inf") or n in self.ram or n in self.inflight \
               or n in self.gpu:
                continue
            # The current compute step IS the window being prefetched into;
            # `_true_hold` sums steps strictly after i, so add it back.
            dt = dt + window
            r = self.cat[n]
            if data_only and r["cls"] == "model":
                continue
            if r["cls"] == "model":
                if gpu_free <= 0 and (displaceable is None
                                      or hor[n] >= hor[displaceable]):
                    continue
            if min(r["cold_s"] - r["ready_s"], dt) <= 0:
                continue
            items[n] = (self._value(n, dt, "prefetch"), r["held_gb"], "prefetch")

        if not items:
            return
        keep = self._knapsack(items)
        if slack_only:
            held = {n for n, (_v, _g, k) in items.items() if k == "retain"}
            keep |= held                      # retentions are untouchable
            used = sum(items[n][1] for n in keep)
            pf = sorted((n for n in keep if items[n][2] == "prefetch"),
                        key=lambda x: items[x][0] / max(items[x][1], 1e-9))
            while used > self.budget and pf:      # shed prefetches, worst first
                n = pf.pop(0); keep.discard(n); used -= items[n][1]

        for n in sorted(set(self.ram) - keep):               # outbid -> evicted
            self.ram.pop(n)
            if self.log is not None:
                self.log.append(dict(kind="evict_for_prefetch", step=i, resource=n))
        for n in sorted(keep):
            if items[n][2] != "prefetch":
                continue
            if self.cat[n]["cls"] == "model":
                if len(self.gpu) >= self.slots and displaceable is not None:
                    self.gpu.remove(displaceable)
                    self.inflight.pop(displaceable, None)
                    if self._occupied(extra=(displaceable,)) <= self.budget:
                        self.ram[displaceable] = i          # park the displaced
                    displaceable = None
                elif len(self.gpu) >= self.slots:
                    continue
                self.gpu.append(n)                          # holds slot while loading
            self.inflight[n] = self.clock + self.cat[n]["cold_s"]
            if self.log is not None:
                self.log.append(dict(kind="prefetch", step=i, resource=n,
                                     done_at=round(self.inflight[n], 1)))

    def _knapsack(self, items: dict) -> set:
        """Highest total value that fits, over retain AND prefetch candidates."""
        if self.policy == "greedy":
            keep, used = set(), 0.0
            # rank by DENSITY (value per GB); the budget is a GB ceiling
            for n in sorted(items, key=lambda x: (-items[x][0] / max(items[x][1], 1e-9), x)):
                if used + items[n][1] <= self.budget:
                    keep.add(n); used += items[n][1]
            return keep
        pool = sorted(items)
        best, best_v, best_gb = set(), -1.0, float("inf")
        for r in range(len(pool), -1, -1):
            for combo in itertools.combinations(pool, r):
                gb = sum(items[x][1] for x in combo)
                if gb > self.budget:
                    continue
                v = sum(items[x][0] for x in combo)
                if v > best_v or (v == best_v and gb < best_gb):
                    best, best_v, best_gb = set(combo), v, gb
        return best

    # -- main loop ----------------------------------------------------------
    def run(self):
        first_need = next((k for k, (t, _) in enumerate(self.sched)
                           if t == "need"), None)
        for i, (kind, val) in enumerate(self.sched):
            if kind == "compute":
                self._arbitrate(i, val)
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
