# chemgraph_screen — workload design (2026-07-19)

Goal: a *realistic* ChemGraph workload whose structure gives every runtime
component a genuine job, sized from measured cluster numbers so the benefit is
physically attainable — and honestly reported as designed-for-benefit.

## Why the existing workloads can't show the win (measured 7/19)

| Requirement | swap | ensemble | exp3 |
|---|---|---|---|
| Big cold transitions (>60 s load) | yes (~200-245 s stall) | no (page-cache warm) | yes |
| Compute window opens BEFORE need | **no (~20 s)** | no | yes but trigger fired late |
| Branching that defeats naive-prefetch | no (always 72B next) | no | no |
| Trial < 30 min for stats power | yes | yes | **no (1-2 h, ±13%)** |

Measured constants: staging 3.8-4.8 GB/s from Lustre; 72B-Instruct ≈ 145 GB
(~200-245 s cold swap incl. vLLM load); 32B ≈ 65 GB (~90-120 s est.); 256 GB
cgroup ⇒ at most one ~145 GB specialist warm at a time.

## Workload: heterogeneous screening batch with specialist routing

High-throughput screening loop, the standard shape of computational-chemistry
campaigns: N molecules per trial, each needing (a) geometry optimization and
(b) a property analysis performed by a *specialist* LLM+tool combo chosen per
molecule from intermediate results.

Per-trial structure (N ≈ 6 molecules, mixed classes):

```
planner (32B-VL, resident):
  batch plan — for each molecule: predicted class → specialist + tool chain
loop over molecules i = 1..N:
  geometry opt (MACE on GPUs 0-3, molecule sized for ~90-180 s)   <- window
  class check: opt results (energy/geometry descriptors) confirm or FLIP class
  specialist analysis for molecule i (72B-Instruct OR 32B route)  <- transition
```

- **Plan analysis**: the batch plan names each molecule's predicted specialist
  → the predictor stages specialist(i+1) during molecule i's MACE window.
- **Transition tables**: tool chains are stable within a class (opt → class
  check → analyze → next-opt); across trials the learned transitions sharpen
  prefetch timing beyond what any single plan states.
- **Divergence guard**: the class check flips ~20-25 % of molecules (mixed
  borderline cases in the batch on purpose — realistic: pre-screen labels are
  cheap heuristics). A flip invalidates the staged specialist → guard cancels,
  re-stages the correct one; without the guard the wrong 145 GB model occupies
  the cgroup and the right one loads cold + evicts it (visible cost).
- **Naive-prefetch ablation is honestly disadvantaged**: two specialists can't
  both stay staged (145 + 65 GB vs 256 GB cgroup incl. vLLM residency);
  prefetch-everything thrashes page cache, plan-blind immediate prefetch picks
  specialists by static order and eats the ~25 % flip rate with no cancel.

## Sizing

- Window: MACE-MP medium geometry opt tuned via molecule size (~30-80 atoms)
  and fmax to land 90-180 s ≥ specialist load (65-145 GB @ ~4 GB/s + vLLM
  spin-up). Verify empirically in smoke trial; tune fmax not molecule identity.
- Trial walltime target: 6 molecules × (opt + analysis) ≈ 20-25 min.
- Expected mechanism numbers if trigger fires at plan/window start:
  overlap ≈ load_time (60-200 s per transition), benefit ≈ 5 transitions ×
  (load − residual) ⇒ a 15-25 % end-to-end delta vs baseline, versus oracle
  ceiling ~30 %. Failure to reach baseline+10 % ⇒ design revisit, not more n.

## Configs (mirrors chemgraph_swap matrix)

baseline / observe_only / full_system / naive_prefetch / no_divergence_guard /
plan_only / transition_only / no_cache_stage / oracle.

## Flags (new, all toggleable)

- `--pin-calculator` (default ON for screen): constrain run_ase to mace_mp;
  OFF keeps agent freedom — used to study guard behavior on TBLite failures.
- `--early-stage-on-plan` (runtime): stage at plan_extracted, resource choice
  conditioned on plan contents; OFF = legacy post-planning trigger (ablation).

## Validity

- New generation tag (workload name distinct; no pooling with swap trials).
- Same facet rules (never blend GPU types); first_in_allocation flagged.
- Molecule batch fixed per trial index (deterministic task list, sampled agent
  behavior) so arms see identical work.

---

# v2 (2026-07-20): chemgraph_screen_pool — disjoint specialist GPU pools

## Why v1 failed its criterion

21 valid v1 trials: no full_system-vs-baseline win on exposed_swap_s (the
only sizable gap, −811 s on Blackwell, rests on baseline n=1; L40S matched
git is +40 s to −215 s, inside noise).  Decisive mechanism finding:
**oracle ≈ full_system** (1723/1497 vs 1693/1426 same git/facet) — perfect
prediction buys nothing.  On a SHARED pool only 1 of 6 swaps converts to
prefetched, and even that one's residual engine spin-up (130–450 s) stays on
the critical path.  overlap_s (182–577 s) is absorbed, not recovered; no_plan
collapses overlap 182→35 s with wall unchanged.  The bottleneck is vLLM
engine spin-up, which no shared-pool trigger timing can hide.

## v2 mechanism

Specialists on disjoint GPU pools: advanced=72B on GPUs 0-3 (:8001),
standard=32B on GPUs 4-5 (:8005), SpecialistProxy (:8006) as the single
worker-client endpoint (client is built once; adapter flips the target after
wait_until_ready).  The NEXT task's engine boots on its own pool while the
current task is being served — spin-up itself overlaps the serving window.

Residency policy, identical in every arm: idle engines are evicted
(background, off the critical path).  What differs is boot timing only:

- baseline: on-demand boot at task start (spin-up exposed per alternation)
- full_system: plan-conditioned per transition — the other engine is KEPT
  resident when the plan names it next, or pre-booted (cache-stage + engine
  boot on its own pool) during the current task
- blind_stage: pre-boot the other specialist every task (no plan) — wasted
  boots whenever consecutive tasks share a specialist
- naive_prefetch: keep-all-resident (time-optimal, residency-maximal bound;
  feasible only because 2 specialists fit 6 GPUs — the residency ledger is
  the honest comparison axis)
- guard: cancels/evicts wrongly kept or pre-booted engines (resource
  hygiene; with 2 specialists the time cost of misprediction is small)

Molecule order is non-alternating (aspirin, caffeine, water, ibuprofen,
methane, ammonia → adv,adv,std,adv,std,std: transitions keep,swap,swap,
swap,keep) so plan-conditioning is falsifiable against blind alternation.

## Benefit envelope

Baseline pays ~4 exposed boots (first + 3 swaps) ≈ 480–600 s; full_system
should expose only the first boot's residual + any window shortfall
(per-task windows ~30–200 s vs ~100–140 s spin-up) ≈ 150–350 s.  Expected
win 20–40 % of wall on L40S-d26cc46-era walls — comfortably above the
carried-over pre-registered criterion (>10 % on wall AND exposed_swap_s vs
baseline, same facet, matched git), or the design gets revisited again.

## Validity

- L40S 6-GPU facet ONLY (Blackwell nodes have 4 GPUs).
- Distinct workload dir (chemgraph_screen_pool); never pooled with v1.
- Valid from commit 5b85ed7.
