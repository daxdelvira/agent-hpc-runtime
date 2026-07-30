#!/bin/bash
# run_eval_blackwell_residency_20260730.sh — Stage-0 gates for the
# constrained-residency regime, then fall through to sleep_wake collection.
#
# WHY THIS REPLACES run_eval_blackwell_sleepwake.sh AS THE BLACKWELL TARGET:
# the 7/30 pivot moved the research question from "hide the model boot" to
# "N backing models, room for only M < N".  That regime rests on two untested
# hardware assumptions — a 32B-class model serving at tp=1 on ONE 96 GB GPU,
# and N engines sleeping at level 1 simultaneously under the 256 GB cgroup.
# Both are hard gates: if they fail, the whole residency plan needs a different
# topology, and every hour spent building Stages 1-7 first is wasted.  So the
# gates run before anything else on the first Blackwell hold we get.
#
# The sleep_wake collection is NOT dropped — it stays as the fall-through, and
# its numbers remain load-bearing (the residency state ladder is built on
# them).  It just no longer goes first.
#
# GENERATION: sleep_wake valid from commit 94fa2b2 (gates: bench 2026-07-29 +
# supervised smoke on job 11518018).  The arm is explicit-only in the driver.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
nvidia-smi -L | grep -qiE "rtx pro 6000|blackwell" \
  || { log "ABORT: not a Blackwell node — facet mismatch."; exit 1; }
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

PY=python3
HOST=$(hostname)

# --- Stage 0: residency preflight gates -------------------------------------
# n=4 at tp=1: one 32B-class engine per 96 GB GPU, which is exactly the target
# topology (4 backing models, sweepable resident count).  Skipped once a result
# exists for this host — the campaign relaunches verbatim after preemption and
# must never re-pay the boot.  Time-capped and non-fatal: a gate FAILURE is a
# real finding we want recorded, not a reason to waste the rest of the hold.
GATEFILE="results/bench_residency_preflight_${HOST}.json"
if [ ! -e "$GATEFILE" ] && [ ! -e "results/bench_residency_preflight_${HOST}.attempted" ]; then
  touch "results/bench_residency_preflight_${HOST}.attempted"
  log "Stage 0: residency preflight (n=4, tp=1, gates a,b,c,d; cap 90 min)"
  timeout 5400 $PY experiments/bench_residency_preflight.py \
      --model qwen_32b_vl --n 4 --tp 1 --gates a,b,c,d
  rc=$?
  case $rc in
    0)   log "Stage 0: ALL BLOCKING GATES PASS — N=4 tp=1 topology is achievable" ;;
    1)   log "Stage 0: BLOCKING GATE FAILED — see $GATEFILE; topology must be re-derived" ;;
    124) log "Stage 0: timed out at 90 min (partial results in $GATEFILE)" ;;
    *)   log "Stage 0: exited rc=$rc (partial results in $GATEFILE)" ;;
  esac
  # The preflight leaves 4 engines behind only if it died hard; reap either way
  # so the fall-through collection starts from a clean GPU.
  pkill -u "$USER" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -u "$USER" -f "VLLM::" 2>/dev/null || true
  sleep 15
else
  log "Stage 0: residency preflight already attempted on $HOST — skipping"
fi

# --- Fall-through: sleep_wake collection ------------------------------------
# Baseline comparators: existing Blackwell-facet chemgraph_swap baselines
# (N=11+); cross-night Lustre variance applies — compare exposed_stall_s and
# swap_mechanism-faceted gates, not raw wall.
log "Phase 1: sleep_wake x1 (first driver-launched trial of the arm)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 1

log "Phase 2: sleep_wake to n=3"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 3

log "Phase 3 (stretch): sleep_wake to n=5"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_swap \
    --configs sleep_wake --trials 5

log "Campaign complete."
