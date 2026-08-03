#!/bin/bash
# overnight_watcher7_20260803.sh — supervisor for the workshop-paper holds.
#
# MUST BE STARTED DETACHED:
#     setsid nohup bash experiments/overnight_watcher7_20260803.sh >/dev/null 2>&1 &
# watcher4 died with its parent shell on 7/30; watcher5 died some time on 8/2
# (its log ends mid-run with all tracked jobs blacklisted). Both cost a night.
#
# Repointed for the 8/2 state:
#   - watcher5's job IDs are all gone (11518013-015 finished/cancelled,
#     11571891-895 long dead). It tracked nothing that still exists.
#   - The 9 holds submitted 8/2 (11629977-985) had NOTHING pointed at them,
#     so any that landed overnight would have sat idle for 8 h.
#
# Both facets run the same campaign: the L40S profile carries the same 3 models
# on [0,1,2,3] tp=4, so exp3 runs there too — only the per-node GPU count check
# differs. Keeping one script means a fix lands on both facets at once.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"
WLOG=$REPO/logs/overnight_watcher7_20260803.log
mkdir -p "$REPO/logs"

# ---- assignments -----------------------------------------------------------
BW_JOBS="11629978 11629979 11629980 11629981 11652952 11652953"
L40S_JOBS="11652948 11652949 11652950 11652951"
SCRIPT="run_eval_blackwell_aligned_20260802.sh"
BW_GRES="gpu:4";  BW_TYPED="Partition=gpu-rtxpro-blackwell"
L40S_GRES="gpu:6"; L40S_TYPED="gres/gpu:l40s=6"
NOOP_FLOOR_S=180     # a step shorter than this did no real work
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
    log "job $j is on the wrong GPU type — blacklisting"
    blacklist[$j]=1; return 1
  fi
  return 0
}

# An srun that has been launched but has no step yet is WAITING, not dead.
# On 2026-08-03 srun took 9 min to get a step on job 11652948 (the node was
# busy with foreign compute). has_step stayed false, so the watcher spent both
# attempts inside that window and blacklisted a node whose campaign then ran
# fine for hours — the hold was only saved by a manual check. Treat a live srun
# process as "launch in flight" and never count a second attempt against it.
# Anchored on "^srun " deliberately: a bare `pgrep -f -- --jobid=$1` also matches
# any shell whose command line merely CONTAINS the job id (including the shell
# a human runs the check from), which reports every job as alive and stops the
# watcher launching anything at all.
srun_alive(){ ps -u "$USER" -o cmd= 2>/dev/null | grep -q "^srun .*--jobid=$1 "; }

# The campaign's clean-tree guard aborts in ~1 s on a dirty tree. That is a
# correct refusal — a trial that cannot be tied to a commit is worthless — but
# the tree goes dirty for MINUTES AT A TIME while other agents edit
# runtime/predictor. Launching into that burns an attempt, logs SUSPECT NO-OP,
# and after 4 of them blacklists a perfectly good hold. The dirty tree is
# transient and external, so wait it out instead of spending attempts on it.
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

log "watcher7 started (pid $$): BW=[$BW_JOBS] L40S=[$L40S_JOBS] -> $SCRIPT"

while true; do
  for j in $BW_JOBS $L40S_JOBS; do
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

  # One campaign per JOB, not per facet. The earlier per-facet rule assumed two
  # campaigns on the same node type would contend for GPUs — but separate
  # allocations are separate NODES (11629982 on 002-4-0, 11629983 on 002-6-0),
  # so there is no contention and the rule just left a landed 7h42m hold idle.
  # Duplicate work is possible (two nodes may both decide "0/5 completed") but
  # the driver writes timestamped run dirs so nothing collides, and with 9 holds
  # against a Saturday deadline, extra trials are worth more than tidy counts.
  for facet in BW L40S; do
    if [ "$facet" = BW ]; then jobs="$BW_JOBS"; gres="$BW_GRES"; typed="$BW_TYPED"
    else jobs="$L40S_JOBS"; gres="$L40S_GRES"; typed="$L40S_TYPED"; fi
    for j in $jobs; do
      usable "$j" "$typed" || continue
      has_step "$j" && continue
      srun_alive "$j" && continue   # launch in flight, waiting for a step
      launch "$j" "$gres"
    done
  done

  alive=0
  for j in $BW_JOBS $L40S_JOBS; do
    [ -n "$(job_state "$j")" ] && alive=1 && break
  done
  if [ "$alive" = 0 ]; then
    log "all tracked jobs gone; watcher6 exiting"
    exit 0
  fi
  sleep 120
done
