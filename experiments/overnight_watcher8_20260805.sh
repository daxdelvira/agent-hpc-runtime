#!/bin/bash
# overnight_watcher8_20260805.sh — supervisor for the workshop-paper holds.
#
# MUST BE STARTED DETACHED:
#     setsid nohup bash experiments/overnight_watcher8_20260805.sh >/dev/null 2>&1 &
#
# WHAT CHANGED FROM watcher7, AND WHY. watcher7's own header documents the same
# failure twice (7/30 and 8/2) and it then happened a third time on 8/4:
#   - 11653557/8 ran 00:36-04:31 on 8/5 with no watcher alive: ~7.6 node-hours
#     of Blackwell burned holding a `sleep infinity`.
# The pattern is always one of two bugs, both fixed here.
#
#   FIX 1 — DISCOVER JOBS, DO NOT HARDCODE THEM.  watcher7 carried a literal
#   list of eight job IDs. Any hold submitted after the watcher started was
#   invisible to it, so "submit some more holds" silently produced idle nodes.
#   watcher8 asks squeue for jobs named wp_bw_* / wp_l40s_* on every pass, so a
#   hold submitted an hour from now is picked up on the next 120 s tick with no
#   edit and no restart. experiments/submit_holds.sh emits exactly those names.
#
#   FIX 2 — NEVER EXIT WHEN THE QUEUE IS EMPTY.  watcher7 exited as soon as all
#   tracked jobs were gone. Combined with fix 1's absence that is lethal: the
#   watcher dies during the gap between one batch finishing and the next being
#   submitted, which is precisely when a human is least likely to notice. This
#   loop idles instead, so it survives arbitrarily long gaps.
#
# Everything else is watcher7's logic verbatim, including the attempt cap, the
# no-op floor, the in-flight-srun check and the clean-tree guard. Those were
# each written against a specific observed failure; do not simplify them away.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"
WLOG=$REPO/logs/overnight_watcher8_20260805.log
LOCK=$REPO/logs/.watcher8.lock
mkdir -p "$REPO/logs"

SCRIPT="run_eval_aligned_20260803.sh"
NOOP_FLOOR_S=180     # a step shorter than this did no real work

# The GPU-type gate must match on AllocTRES, which names the type actually
# ALLOCATED and is only populated once the job is RUNNING. Never match on
# Partition= — watcher7's header records that being a substring of the
# requested partition list and matching even when the job landed elsewhere.
# GRES type names: gpu-rtxpro-blackwell -> rtx_pro_6000_blackwell, gpu-l40s -> l40s.
BW_GRES="gpu:4";   BW_TYPED="gres/gpu:rtx_pro_6000_blackwell=4"
L40S_GRES="gpu:6"; L40S_TYPED="gres/gpu:l40s=6"

# ---- single-instance lock --------------------------------------------------
# Two watchers racing the same hold double-launch the campaign and burn attempts
# in pairs. Cheap to prevent, expensive to diagnose.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another watcher8 already holds $LOCK — exiting" >&2
  exit 0
fi

log(){ echo "[$(date +'%F %T')] $*" >> "$WLOG"; }
job_state(){ squeue -j "$1" -h -o "%T" 2>/dev/null | head -1; }
has_step(){ squeue -s -j "$1" -h -o "%i" 2>/dev/null | grep -qE "^$1\.[0-9]+$"; }
gpu_ok(){ scontrol show job "$1" 2>/dev/null | grep -q "$2"; }

# FIX 1. Name-prefix discovery, evaluated fresh every pass.
discover(){ # $1 = name prefix -> job ids, one per line
  squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
    | awk -v p="$1" '$2 ~ "^"p {print $1}'
}

declare -A attempts blacklist launch_ts step_seen known

usable(){ # running, right GPU type, not blacklisted
  local j=$1 typed=$2
  [ "${blacklist[$j]:-0}" = 1 ] && return 1
  [ "$(job_state "$j")" = "RUNNING" ] || return 1
  if ! gpu_ok "$j" "$typed"; then
    log "job $j is on the wrong GPU type — blacklisting"
    blacklist[$j]=1; return 1
  fi
  return 0
}

# An srun launched but with no step yet is WAITING, not dead. On 2026-08-03 srun
# took 9 min to get a step on job 11652948 because the node was busy with
# foreign compute; has_step stayed false and the watcher spent both attempts
# inside that window on a hold that then ran fine for hours. Anchored on
# "^srun " deliberately: a bare `pgrep -f -- --jobid=$1` also matches any shell
# whose command line merely CONTAINS the job id, which reports every job as
# alive and stops the watcher launching anything at all.
srun_alive(){ ps -u "$USER" -o cmd= 2>/dev/null | grep -q "^srun .*--jobid=$1 "; }

# The campaign's clean-tree guard aborts in ~1 s on a dirty tree. That is a
# correct refusal — a trial that cannot be tied to a commit is worthless — but
# the tree goes dirty for MINUTES AT A TIME while other agents edit. Launching
# into that burns an attempt, logs SUSPECT NO-OP, and after 4 blacklists a
# perfectly good hold. Wait it out instead of spending attempts on it.
# Mirrors the campaign's guard exactly, .nfs* exclusion included.
tree_clean(){
  local d
  d=$(git -C "$REPO" status --porcelain --ignore-submodules=untracked \
        -- experiments workloads runtime 2>/dev/null \
      | grep -v '/\.nfs[0-9a-f]\{16,\}') || true
  [ -z "$d" ]
}

launch(){ # $1=jobid $2=gres
  local n=$(( ${attempts[$1]:-0} + 1 ))
  attempts[$1]=$n
  if [ "$n" -gt 4 ]; then
    log "job $1: attempt limit reached — blacklisting (check campaign log)"
    blacklist[$1]=1; return
  fi
  local logf="campaign_aligned_job$1.log"
  log "launching $SCRIPT into job $1 (attempt $n, log: $logf)"
  launch_ts[$1]=$(date +%s); step_seen[$1]=0
  (unset $(env | grep -o '^SLURM_[A-Z_]*') 2>/dev/null
   nohup srun --jobid="$1" --overlap --gres="$2" \
     --output="$REPO/$logf" bash "$REPO/experiments/$SCRIPT" >/dev/null 2>&1 &)
  sleep 45   # let the step register before the next has_step poll
}

# Flags the exits-immediately failure mode that silently burned five holds on
# 7/29: a campaign whose targets are already satisfied returns 0 in ~10 s, which
# is indistinguishable from success unless the step is timed.
note_step_end(){
  local j=$1 t0=${launch_ts[$j]:-0}
  [ "$t0" = 0 ] && return
  local dur=$(( $(date +%s) - t0 ))
  if [ "$dur" -lt "$NOOP_FLOOR_S" ]; then
    log "job $j: step ended after ${dur}s — SUSPECT NO-OP (check campaign_aligned_job$j.log before trusting it)"
  else
    log "job $j: step ended after ${dur}s"
  fi
  launch_ts[$j]=0
}

log "watcher8 started (pid $$): discovering wp_bw_* / wp_l40s_* every pass -> $SCRIPT"

idle_noted=0
while true; do
  bw_jobs=$(discover wp_bw_)
  l40s_jobs=$(discover wp_l40s_)
  all_jobs="$bw_jobs $l40s_jobs"

  # Announce holds the moment they appear, so an idle node is visible in the
  # log rather than inferred afterwards from sacct.
  for j in $all_jobs; do
    if [ "${known[$j]:-0}" = 0 ]; then
      known[$j]=1
      log "discovered job $j ($(squeue -j "$j" -h -o '%j %T' 2>/dev/null))"
    fi
  done

  for j in $all_jobs; do
    if has_step "$j"; then step_seen[$j]=1
    elif [ "${step_seen[$j]:-0}" = 1 ]; then
      step_seen[$j]=0; note_step_end "$j"
    fi
  done

  if ! tree_clean; then
    [ "${dirty_noted:-0}" = 1 ] || log "tree dirty (another agent is mid-edit) — holding launches; attempts NOT spent"
    dirty_noted=1
    sleep 120; continue
  fi
  if [ "${dirty_noted:-0}" = 1 ]; then log "tree clean again — resuming launches"; dirty_noted=0; fi

  # One campaign per JOB, not per facet: separate allocations are separate
  # NODES, so two campaigns on the same node type do not contend for GPUs.
  # Duplicate work is possible (two nodes may both decide "0/5 completed") but
  # the driver writes timestamped run dirs so nothing collides, and extra
  # trials are worth more than tidy counts against a deadline.
  for facet in BW L40S; do
    if [ "$facet" = BW ]; then jobs="$bw_jobs"; gres="$BW_GRES"; typed="$BW_TYPED"
    else jobs="$l40s_jobs"; gres="$L40S_GRES"; typed="$L40S_TYPED"; fi
    for j in $jobs; do
      usable "$j" "$typed" || continue
      has_step "$j" && continue
      srun_alive "$j" && continue   # launch in flight, waiting for a step
      launch "$j" "$gres"
    done
  done

  # FIX 2. Idle, never exit. The gap between batches is exactly when a watcher
  # that exits costs a night.
  if [ -z "$(echo $all_jobs | tr -d ' ')" ]; then
    [ "$idle_noted" = 1 ] || log "no wp_* holds in the queue — idling (submit more with experiments/submit_holds.sh)"
    idle_noted=1
  else
    idle_noted=0
  fi
  sleep 120
done
