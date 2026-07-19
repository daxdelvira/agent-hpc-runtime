#!/bin/bash
# overnight_watcher_20260719.sh — overnight campaign supervisor (2026-07-19).
# Runs detached on a login node (setsid nohup); survives the interactive session.
#
# Tonight's fleet (all embers QOS — preemption possible, drivers are resumable):
#   L40S hold   11237299  RUNNING  ends ~06:09   gpu:6 (l40s)
#   L40S chain  11267688 -> 89 -> 90 -> 91 -> 92 (afterany), 688 pending on
#               Resources — MAY start while 11237299 still runs.
#   BW hold     11267674  RUNNING  ends ~09:26   gpu:4 (rtx_pro_6000_blackwell)
#   BW chain    11267675 -> 76 -> 77 -> 78 (afterany on 674)
#
# Policy:
#   * One campaign step per node; a running job with no campaign step gets
#     (re)launched. Drivers are resumable + deadline-aware, so relaunch after
#     preemption / mid-trial death is always safe.
#   * GPU-TYPE GUARD: chain jobs carry a gpu-v100 fallback partition. A job
#     whose AllocTRES doesn't match the expected typed gres is blacklisted
#     (these workloads can't run on V100: bf16/sm70, wrong facet).
#   * Max 2 launch attempts per job — a campaign that dies twice quickly is
#     a real failure, not a preemption; don't crashloop it.
#   * Only one PRIMARY L40S campaign at a time (drivers must not share config
#     dirs). A second concurrent L40S node gets the single-shot SECONDARY
#     (megammap top-up, disjoint configs).
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"
WLOG=$REPO/logs/overnight_watcher_20260719.log
mkdir -p "$REPO/logs"

# ---- assignments -----------------------------------------------------------
L40S_JOBS="11237299 11267688 11267689 11267690 11267691 11267692"
BW_JOBS="11267674 11267675 11267676 11267677 11267678"
L40S_SCRIPT="run_eval_l40s_night_20260719.sh"
L40S_SECONDARY="run_eval_l40s_megammap_topup_20260719.sh"
BW_SCRIPT="run_eval_bw_night_20260719.sh"
L40S_GRES="gpu:6";  L40S_TYPED="gres/gpu:l40s=6"
BW_GRES="gpu:4";    BW_TYPED="gres/gpu:rtx_pro_6000_blackwell=4"
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

log "watcher started: L40S=[$L40S_JOBS -> $L40S_SCRIPT / 2nd: $L40S_SECONDARY]"
log "                 BW=[$BW_JOBS -> $BW_SCRIPT]"

while true; do
  # ---- Blackwell: keep exactly one campaign on whichever BW job runs -------
  bw_active=0
  for j in $BW_JOBS; do
    usable "$j" "$BW_TYPED" || continue
    if has_step "$j"; then bw_active=1; continue; fi
    if [ "$bw_active" = 0 ]; then
      launch "$j" "$BW_SCRIPT" "$BW_GRES"; bw_active=1
    fi
  done

  # ---- L40S: primary on first stepless running job; optional secondary -----
  primary_running=0
  for j in $L40S_JOBS; do
    [ "$(job_state "$j")" = "RUNNING" ] && has_step "$j" && primary_running=1
  done
  for j in $L40S_JOBS; do
    usable "$j" "$L40S_TYPED" || continue
    has_step "$j" && continue
    if [ "$primary_running" = 0 ]; then
      launch "$j" "$L40S_SCRIPT" "$L40S_GRES"; primary_running=1
    elif [ -n "$L40S_SECONDARY" ]; then
      launch "$j" "$L40S_SECONDARY" "$L40S_GRES"
      L40S_SECONDARY=""   # single-shot
    fi
  done

  # ---- exit when every tracked job is finished -----------------------------
  alive=0
  for j in $L40S_JOBS $BW_JOBS; do
    [ -n "$(job_state "$j")" ] && alive=1 && break
  done
  if [ "$alive" = 0 ]; then
    log "all tracked jobs gone; watcher exiting"
    exit 0
  fi
  sleep 120
done
