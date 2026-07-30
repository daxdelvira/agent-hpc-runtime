#!/bin/bash
# run_eval_l40s_pool_n3_20260730.sh — bounded Option-D collection, then repoint
# the L40S facet at the residency regime.
#
# WHY BOUNDED: the 7/30 pivot retired Option D (disjoint GPU pools) as the
# research direction — it assumes enough GPUs to hold every model at once,
# which is the uninteresting case.  But the pool arm is nearly collected and is
# worth exactly one honest number: the "what if you DO have enough GPUs"
# comparison point for the paper.  That needs n=3 baseline/full_system pairs
# and nothing more.  The predecessor script (run_eval_l40s_pool_20260720.sh)
# chased n=6 plus blind_stage and naive_prefetch arms; those phases are dropped
# here rather than left to consume holds the new regime needs.
#
# After the pair completes, the remaining hold goes to the Stage-0 residency
# gates at the L40S topology (n=3 at tp=2 — 6 GPUs, 32B fp16 needs 2x46 GB).
# This is the "then repoint" half: the same gates the Blackwell chain runs at
# n=4/tp=1, measured on the other facet, so a topology decision is not made
# from a single node type.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 6 ] && { log "ABORT: only $ngpu GPUs visible (pool mode needs 6)"; exit 1; }
nvidia-smi -L | grep -qi "l40s" || { log "ABORT: not an L40S node — facet mismatch."; exit 1; }
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

PY=python3
HOST=$(hostname)

# --- Phase 1: the one number Option D still owes us -------------------------
# Pre-registered criterion from chemgraph_screen_DESIGN.md carries over:
# full_system must beat baseline by >10% on wall AND exposed_swap_s.  The
# driver is idempotent — it tops up to the requested trial count and runs
# nothing if already satisfied.
log "Phase 1: chemgraph_screen_pool baseline/full_system pairs to n=3 (BOUNDED)"
$PY experiments/run_eval_q1_q4.py --workload chemgraph_screen_pool \
    --configs baseline,full_system --trials 3

log "Phase 1 done. Option-D status:"
$PY experiments/run_eval_q1_q4.py --list 2>/dev/null | grep -E "workload|screen_pool" || true

# --- Phase 2: repoint — Stage-0 residency gates at the L40S topology --------
# n=3 at tp=2 (32B fp16 ~64 GB needs 2x46 GB).  Same gates as the Blackwell
# chain runs at n=4/tp=1.  Skipped once a result exists for this host so a
# post-preemption relaunch never re-pays the fleet boot.  Non-fatal: a gate
# FAILURE is a finding worth recording, not a reason to abort.
GATEFILE="results/bench_residency_preflight_${HOST}.json"
if [ ! -e "$GATEFILE" ] && [ ! -e "results/bench_residency_preflight_${HOST}.attempted" ]; then
  touch "results/bench_residency_preflight_${HOST}.attempted"
  log "Phase 2: residency preflight (n=3, tp=2, gates a,b,c,d; cap 90 min)"
  timeout 5400 $PY experiments/bench_residency_preflight.py \
      --model qwen_32b_vl --n 3 --tp 2 --gates a,b,c,d
  rc=$?
  case $rc in
    0)   log "Phase 2: ALL BLOCKING GATES PASS — n=3 tp=2 topology achievable on L40S" ;;
    1)   log "Phase 2: BLOCKING GATE FAILED — see $GATEFILE" ;;
    124) log "Phase 2: timed out at 90 min (partial results in $GATEFILE)" ;;
    *)   log "Phase 2: exited rc=$rc (partial results in $GATEFILE)" ;;
  esac
  pkill -u "$USER" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -u "$USER" -f "VLLM::" 2>/dev/null || true
  sleep 15
else
  log "Phase 2: residency preflight already attempted on $HOST — skipping"
fi

log "Campaign complete."
