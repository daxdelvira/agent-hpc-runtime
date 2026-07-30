#!/bin/bash
# overnight_watcher5_20260730.sh — residency-regime supervisor (2026-07-30).
#
# Successor to overnight_watcher4_20260727.sh, which died when its parent shell
# was killed on 7/30 (it had been launched from a session shell, not detached).
# THIS SCRIPT MUST BE STARTED DETACHED:
#     setsid nohup bash experiments/overnight_watcher5_20260730.sh >/dev/null 2>&1 &
#
# Repointed for the 7/30 pivot:
#   Blackwell -> run_eval_blackwell_residency_20260730.sh
#       Stage-0 residency gates (n=4, tp=1) first, then sleep_wake collection.
#   L40S      -> run_eval_l40s_pool_n3_20260730.sh
#       bounded Option-D pairs to n=3, then Stage-0 gates at n=3/tp=2.
#
# NEW IN v5 — no-op detection.  Jobs 11518016-020 each burned a Blackwell hold
# running run_eval_blackwell_20260728.sh, whose targets were already satisfied:
# the step exited in ~10 s, twice, then hit the attempt limit and blacklisted.
# A campaign that exits 0 immediately is indistinguishable from one that did
# its work, so v4 could not tell.  v5 times every step and logs SUSPECT NO-OP
# below the floor, so five wasted allocations show up in the log as five
# warnings instead of looking like ordinary completions.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"
WLOG=$REPO/logs/overnight_watcher5_20260730.log
mkdir -p "$REPO/logs"

# ---- assignments -----------------------------------------------------------
L40S_JOBS="11518013 11518014 11518015"
L40S_SCRIPT="run_eval_l40s_pool_n3_20260730.sh"
L40S_GRES="gpu:6";  L40S_TYPED="gres/gpu:l40s=6"
BW_JOBS="11571891 11571892 11571893 11571894 11571895"
BW_SCRIPT="run_eval_blackwell_residency_20260730.sh"
BW_GRES="gpu:4";    BW_TYPED="Partition=gpu-rtxpro"
NOOP_FLOOR_S=180    # a step shorter than this did no real work
# ---------------------------------------------------------------------------

log(){ echo "[$(date +'%F %T')] $*" >> "$WLOG"; }

job_state(){ squeue -j "$1" -h -o "%T" 2>/dev/null | head -1; }
has_step(){ squeue -s -j "$1" -h -o "%i" 2>/dev/null | grep -qE "^$1\.[0-9]+$"; }
gpu_ok(){ scontrol show job "$1" 2>/dev/null | grep -q "$2"; }

declare -A attempts blacklist launch_ts step_seen

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
  launch_ts[$1]=$(date +%s)
  step_seen[$1]=0
  (unset $(env | grep -o '^SLURM_[A-Z_]*') 2>/dev/null
   nohup srun --jobid="$1" --overlap --gres="$3" \
     --output="$REPO/$logf" bash "$REPO/experiments/$2" >/dev/null 2>&1 &)
  sleep 45   # let the step register before the next has_step poll
}

# Called when a step we launched has gone away.  Times it and flags the
# exits-immediately failure mode that silently burned five holds.
note_step_end(){ # $1=jobid $2=script
  local j=$1 t0=${launch_ts[$j]:-0}
  [ "$t0" = 0 ] && return
  local dur=$(( $(date +%s) - t0 ))
  if [ "$dur" -lt "$NOOP_FLOOR_S" ]; then
    log "job $j: step ended after ${dur}s — SUSPECT NO-OP (campaign exited without doing work; check campaign_$(basename "$2" .sh)_job$j.log before trusting the '<10 s completion')"
  else
    log "job $j: step ended after ${dur}s"
  fi
  launch_ts[$j]=0
}

log "watcher5 started (pid $$): L40S=[$L40S_JOBS -> $L40S_SCRIPT] BW=[$BW_JOBS -> $BW_SCRIPT]"

while true; do
  # --- step-lifecycle bookkeeping (for no-op detection) ---------------------
  for j in $L40S_JOBS; do
    if has_step "$j"; then step_seen[$j]=1
    elif [ "${step_seen[$j]:-0}" = 1 ]; then
      step_seen[$j]=0; note_step_end "$j" "$L40S_SCRIPT"
    fi
  done
  for j in $BW_JOBS; do
    if has_step "$j"; then step_seen[$j]=1
    elif [ "${step_seen[$j]:-0}" = 1 ]; then
      step_seen[$j]=0; note_step_end "$j" "$BW_SCRIPT"
    fi
  done

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

  bw_running=0
  for j in $BW_JOBS; do
    [ "$(job_state "$j")" = "RUNNING" ] && has_step "$j" && bw_running=1
  done
  for j in $BW_JOBS; do
    usable "$j" "$BW_TYPED" || continue
    has_step "$j" && continue
    if [ "$bw_running" = 0 ]; then
      launch "$j" "$BW_SCRIPT" "$BW_GRES"; bw_running=1
    fi
  done

  alive=0
  for j in $L40S_JOBS $BW_JOBS; do
    [ -n "$(job_state "$j")" ] && alive=1 && break
  done
  if [ "$alive" = 0 ]; then
    log "all tracked jobs gone; watcher5 exiting"
    exit 0
  fi
  sleep 120
done
