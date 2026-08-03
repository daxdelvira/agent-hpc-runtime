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

# Clean-tree guard: a trial must be attributable to a commit, so refuse to run
# with local edits. EXCLUDE .nfs* — those are NFS silly-rename artifacts, made
# when a file is unlinked while another process still holds it open. They are
# untracked, they are not anyone's edit, and they appear and vanish on their
# own. One of them aborted job 11652948 twice on 2026-08-03 and cost the hold:
# the watcher's two attempts both hit it, then blacklisted a node that was
# otherwise healthy. Anything else still aborts.
_dirty=$(git status --porcelain --ignore-submodules=untracked -- experiments workloads runtime 2>/dev/null \
         | grep -v '/\.nfs[0-9a-f]\{16,\}' || true)
if [ -n "$_dirty" ]; then
  log "ABORT: uncommitted changes in experiments/workloads/runtime — commit first."
  log "$_dirty"; exit 1
fi
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$ngpu" -lt 4 ] && { log "ABORT: only $ngpu GPUs visible (need 4)"; exit 1; }
bash experiments/node_preflight.sh || { log "ABORT: node preflight failed"; exit 1; }

PY=python3
HOST=$(hostname)

# --- AutoGen disk cache must NOT live on NFS ---------------------------------
# autogen's transforms.py evaluates `Cache.disk()` at IMPORT time, which opens a
# diskcache SQLite DB at ./.cache/<seed> relative to CWD. exp3 chdirs into
# workloads/AtomAgents, which is on project NFS, and SQLite locking over NFS is
# broken: with two campaigns live on different nodes every trial died with
#   sqlite3.OperationalError: locking protocol
# 12 trials, 0 completed, before this was caught. It only appears under
# concurrency, so a single-campaign run looks fine and hides the bug.
# Point .cache at node-local /tmp. Every node writes the SAME symlink target, so
# concurrent setup is idempotent, and each node then has its own private DB.
# mkdir FIRST — a dangling symlink makes diskcache's makedirs fail instead.
mkdir -p /tmp/autogen_cache
if [ ! -L workloads/AtomAgents/.cache ]; then
  rm -rf workloads/AtomAgents/.cache
  ln -sfn /tmp/autogen_cache workloads/AtomAgents/.cache
  log "redirected AutoGen cache to node-local /tmp/autogen_cache (was NFS)"
fi

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

# --- Phase 2: the headline arms -------------------------------------------
# ARM ROLES, so nobody re-derives them from flag names later:
#   baseline            kill + cold boot, runtime OFF  -> X in the narrative
#   sleep_wake          full runtime + /sleep level 2  -> the claim, X - z%
#   sleep_wake_baseline /sleep level 2, runtime OFF    -> how much of the gain
#                       is the swap MECHANISM rather than the prediction.
#                       Without it, sleep_wake differs from baseline in two
#                       ways at once and neither is attributable.
#
# ORDER. sleep_wake first, not baseline. Two reasons, and they override the
# earlier "baseline first because it is X" rule:
#   1. It is the cheapest arm per trial (a 72B tp=4 cold boot is 990-1315 s and
#      sleep_wake pays it once per model instead of once per swap), so it is
#      the arm most likely to actually FINISH inside a preemptible hold.
#   2. It is the only arm that can invalidate the whole plan. If parking does
#      not cut the swap stall, we need to know today, while there is still time
#      to fall back to smaller models — not after collecting a full baseline.
# baseline is already at N=6 on the pre-aligned workload, so it is the arm we
# can most afford to be short on for another few hours.
log "Phase 2a: sleep_wake (the claim) to n=2 — first light on parked engines"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs sleep_wake --trials 2

log "Phase 2a done. Status:"
$PY experiments/run_eval_q1_q4.py --list 2>/dev/null | grep -E "workload|exp3_aligned" || true

log "Phase 2b: baseline (X) to n=2"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs baseline --trials 2

log "Phase 2c: sleep_wake_baseline — separates mechanism from prediction"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs sleep_wake_baseline --trials 2

log "Phase 2d: top up the headline pair to n=5"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs sleep_wake,baseline --trials 5

# --- Phase 3: the two-insight ablation ------------------------------------
# Only meaningful once Phase 2 shows a gap worth decomposing, but harmless to
# attempt: the driver runs nothing if the targets are already satisfied.
log "Phase 3 (stretch): no_plan / naive_prefetch ablation to n=3"
$PY experiments/run_eval_q1_q4.py --workload atomagents_exp3_aligned \
    --configs no_plan,naive_prefetch --trials 3

log "Campaign complete."
