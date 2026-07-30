#!/bin/bash
# overnight_watcher3_20260720.sh — pool-campaign supervisor (2026-07-20).
# Successor to overnight_watcher2_20260719.sh: the remaining L40S chain jobs
# are repointed from the (concluded) shared-pool screen campaign to the
# chemgraph_screen_pool Option-D campaign.  Same policy: resumable driver,
# typed-gres guard against embers gpu-v100 fallback, 2-attempt blacklist.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"
WLOG=$REPO/logs/overnight_watcher3_20260720.log
mkdir -p "$REPO/logs"

# ---- assignments -----------------------------------------------------------
L40S_JOBS="11267691 11267692"
L40S_SCRIPT="run_eval_l40s_pool_20260720.sh"
L40S_GRES="gpu:6";  L40S_TYPED="gres/gpu:l40s=6"
# ---------------------------------------------------------------------------

log(){ echo "[$(date +'%F %T')] $*" >> "$WLOG"; }

job_state(){ squeue -j "$1" -h -o "%T" 2>/dev/null | head -1; }
has_step(){ squeue -s -j "$1" -h -o "%i" 2>/dev/null | grep -qE "^$1\.[0-9]+$"; }
gpu_ok(){ scontrol show job "$1" 2>/dev/null | grep -q "$2"; }

declare -A attempts blacklist

usable(){ # running, right GPU type, not blacklisted
  local j=$1 typed=$2
  [ "${blacklist[$j]:-0}" = 1 ] && return 1
  [ "$(job_state "$j")" = "RUNNING" ] || return 1
  if ! gpu_ok "$j" "$typed"; then
    log "job $j is on the wrong GPU type (embers fallback?) — blacklisting"
    blacklist[$j]=1
    return 1
  fi
  return 0
}

launch(){ # $1=jobid $2=script $3=gres
  local n=$(( ${attempts[$1]:-0} + 1 ))
  attempts[$1]=$n
  if [ "$n" -gt 2 ]; then
    log "job $1: attempt limit reached — blacklisting (check campaign log)"
    blacklist[$1]=1
    return
  fi
  local logf="campaign_$(basename "$2" .sh)_job$1.log"
  log "launching $2 into job $1 (attempt $n, log: $logf)"
  (unset $(env | grep -o '^SLURM_[A-Z_]*') 2>/dev/null
   nohup srun --jobid="$1" --overlap --gres="$3" \
     --output="$REPO/$logf" bash "$REPO/experiments/$2" >/dev/null 2>&1 &)
  sleep 45   # let the step register before the next has_step poll
}

log "watcher3 started: L40S=[$L40S_JOBS -> $L40S_SCRIPT]"

while true; do
  primary_running=0
  for j in $L40S_JOBS; do
    [ "$(job_state "$j")" = "RUNNING" ] && has_step "$j" && primary_running=1
  done
  for j in $L40S_JOBS; do
    usable "$j" "$L40S_TYPED" || continue
    has_step "$j" && continue
    if [ "$primary_running" = 0 ]; then
      launch "$j" "$L40S_SCRIPT" "$L40S_GRES"; primary_running=1
    fi
  done

  alive=0
  for j in $L40S_JOBS; do
    [ -n "$(job_state "$j")" ] && alive=1 && break
  done
  if [ "$alive" = 0 ]; then
    log "all tracked jobs gone; watcher3 exiting"
    exit 0
  fi
  sleep 120
done
