#!/bin/bash
# run_eval_blackwell_aligned_20260802.sh — the workshop-paper collection.
#
# ORDER IS DELIBERATE. Phase 1 is a ~45 min measurement that produces figure
# material directly; Phase 2 is the headline pair. If a hold is preempted early
# we still keep Phase 1, which is the motivation section.
#
# PHASE 1 — activation ladder (cold / warm / L1 / L2).
#   Three rungs are measured, one is a hole: L2 sleep has NEVER been measured
#   (Stage-0 gate (c) never ran — the n=4 fleet died at k=3 first). L2 is the
#   rung that decides the residency policy: the process stays alive (skipping
#   CUDA init, profiling, graph capture) while weights are discarded, so it
#   should be far faster than a warm boot at far lower standing RAM than L1
#   (gate (b): 108-128 GiB per slept 32B, ~1.9x its 68.3 GB of weights). If
#   that holds, page-cache warming is never the right middle tier.
#   Tried first on idle RTX 6000s to avoid spending a Blackwell hold, but every
#   uncontended partition here is Turing or older (SM 7.5/7.0) and this vLLM
#   needs SM 8.0+, so it rides along on a real hold instead. 45 min of an 8 h
#   allocation, and unlike a 7B proxy these numbers are directly citable.
#
# PHASE 2 — atomagents_exp3_aligned baseline + full_system.
#   A separate workload key from atomagents_exp3 (results must not pool): the
#   planner/executor scope alignment, the 3.32 GB potential, and real compute
#   instead of time.sleep(900). This is the X and the X-z% of the narrative.
#
# Skip markers are per-host so a relaunch after preemption never re-pays a
# completed phase. The driver is idempotent and tops up to the trial count.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

unset $(env | grep -oE '^(PMIX_[A-Z0-9_]*|PMI_[A-Z0-9_]*|SLURM_PMI[A-Z0-9_]*)') 2>/dev/null || true
log(){ echo "[$(date +'%F %T')] === $*"; }

if [ -n "$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null)" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

PY=python3
HOST=$(hostname)

# --- Phase 1: activation ladder -------------------------------------------
LADDER="results/bench_activation_ladder_${HOST}.json"
# "Done" means a sleep rung was actually recorded — NOT merely that the file
# exists. The bench persists after every record (it runs under preemptible
# embers), so a run that died during cold boot still leaves a JSON containing
# only the "env" row. Treating that as complete permanently skipped Phase 1 on
# this host after the first OOM.
ladder_done(){
  [ -e "$LADDER" ] || return 1
  $PY - "$LADDER" <<'EOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if any(str(r.get("rung", "")).startswith("sleep_l") for r in rows) else 1)
EOF
}
if ! ladder_done; then
  # Import it — the constant is built with os.path.join across several lines,
  # so scraping the source with a regex silently yields "" and skips Phase 1.
  SNAP=$($PY - <<'EOF'
import sys
sys.path.insert(0, ".")
try:
    from experiments.model_configs import _SNAPSHOT_32B_VL as p
    print(p)
except Exception as e:
    print("", file=sys.stderr)
EOF
)
  if [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
    # tp must be chosen from the ACTUAL per-GPU memory, not hardcoded. The 32B
    # is 68.28 GB: it fits one 96 GB Blackwell at tp=1, but a 46 GB L40S needs
    # tp=2 or the engine OOMs before a single rung is measured. (32B has 64
    # attention heads, so tp in {1,2,4} all divide evenly.)
    GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${GPUMEM:-0}" -ge 90000 ]; then LTP=1; LGPUS=0; else LTP=2; LGPUS=0,1; fi
    log "Phase 1: activation ladder on 32B (cold/warm/L1/L2; ${GPUMEM} MiB/GPU -> tp=$LTP; cap 45 min)"
    timeout 2700 $PY experiments/bench_activation_ladder.py \
        --model-path "$SNAP" --gpus "$LGPUS" --tp "$LTP" --repeat 2
    case $? in
      0)   log "Phase 1: ladder complete — L2 rung finally measured" ;;
      124) log "Phase 1: timed out at 45 min (partial ladder in $LADDER)" ;;
      *)   log "Phase 1: exited nonzero (partial ladder in $LADDER)" ;;
    esac
  else
    log "Phase 1: SKIP — 32B snapshot not found at '$SNAP'"
  fi
  pkill -u "$USER" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -u "$USER" -f "VLLM::" 2>/dev/null || true
  sleep 15
else
  log "Phase 1: ladder already attempted on $HOST — skipping"
fi

# --- Phase 2: the headline pair -------------------------------------------
# baseline first: it is the X every other number is quoted against, and it is
# the arm we can least afford to be missing if the hold is cut short.
log "Phase 2a: atomagents_exp3_aligned baseline+full_system to n=2 (first light)"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs baseline,full_system --trials 2

log "Phase 2a done. Status:"
$PY experiments/run_eval_q1_q4.py --list 2>/dev/null | grep -E "workload|exp3_aligned" || true

log "Phase 2b: top up to n=5"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs baseline,full_system --trials 5

# --- Phase 3: the two-insight ablation ------------------------------------
# Only meaningful once Phase 2 shows a gap worth decomposing, but harmless to
# attempt: the driver runs nothing if the targets are already satisfied.
log "Phase 3 (stretch): plan_only / no_plan ablation to n=3"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs no_plan,naive_prefetch --trials 3

log "Campaign complete."
