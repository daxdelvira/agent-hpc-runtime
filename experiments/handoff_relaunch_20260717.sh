#!/bin/bash
# handoff_relaunch_20260717.sh — run in tmux on a LOGIN node.
#
# Tonight's two Blackwell campaigns run as srun steps whose CLIENTS live on
# L40S job 11237295's node; when that job ends (~00:46, session dies too),
# the steps die mid-trial and the Blackwell holds (alive until ~05:01) idle.
# This script, running somewhere that survives, relaunches each campaign
# into its hold as soon as its step disappears, and also launches the L40S
# night campaign into the next L40S chain job when it starts.
#
# Campaign scripts are resumable and count only completed trials, so a
# relaunch after a mid-trial kill is always safe; the killed trial's dir is
# excluded automatically (no summary.json) and should be annotated later.
#
# Usage:  ssh <login node>; tmux new -s handoff
#         bash /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime/experiments/handoff_relaunch_20260717.sh
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
cd "$REPO"

BW1=11190655   # postfix campaign (exp3/exp2)
BW2=11237288   # swap top-up campaign
L40S_CUR=11237295
L40S_CHAIN="11237296 11237297 11237298 11237299"

log(){ echo "[$(date +'%F %T')] $*"; }

job_state(){ squeue -j "$1" -h -o "%T" 2>/dev/null; }
has_step(){ # running non-batch/extern step on the job?
  squeue -s -j "$1" -h -o "%i" 2>/dev/null | grep -qE "^$1\.[0-9]+"
}

launch(){ # $1=jobid $2=script $3=log $4=gres
  log "launching $2 into job $1"
  (unset $(env | grep -o '^SLURM_[A-Z_]*') 2>/dev/null
   nohup srun --jobid="$1" --overlap --gres="$4" \
     --output="$REPO/$3" bash "$REPO/experiments/$2" >/dev/null 2>&1 &)
}

declare -A relaunched
l40s_launched=0

log "handoff watcher started (BW1=$BW1 BW2=$BW2, L40S chain: $L40S_CHAIN)"
while true; do
  # ---- Blackwell campaigns: relaunch when their step is gone -------------
  for spec in \
    "$BW1|run_eval_blackwell_postfix.sh|campaign_blackwell_postfix_20260717.log|gpu:4" \
    "$BW2|run_eval_blackwell_swap_topup.sh|campaign_blackwell_swap_topup_20260717.log|gpu:4"
  do
    IFS='|' read -r jid script logf gres <<< "$spec"
    [ "${relaunched[$jid]:-0}" = 1 ] && continue
    st=$(job_state "$jid")
    if [ "$st" != "RUNNING" ]; then
      [ -n "$st" ] || { log "job $jid gone; nothing to relaunch"; relaunched[$jid]=1; }
      continue
    fi
    if ! has_step "$jid"; then
      # Guard: only relaunch after the L40S launcher node is actually gone,
      # so we never run two drivers on one campaign dir.
      l40s_state=$(job_state "$L40S_CUR")
      if [ "$l40s_state" != "RUNNING" ]; then
        launch "$jid" "$script" "$logf" "$gres"
        relaunched[$jid]=1
      fi
    fi
  done

  # ---- Next L40S chain job: launch ensemble night campaign once ----------
  if [ "$l40s_launched" = 0 ]; then
    for jid in $L40S_CHAIN; do
      if [ "$(job_state "$jid")" = "RUNNING" ]; then
        launch "$jid" "run_eval_l40s_night.sh" "campaign_l40s_$(date +%Y%m%d)_chain${jid}.log" "gpu:6"
        l40s_launched=1
        break
      fi
    done
  fi

  # Exit when nothing is left to do.
  if [ "${relaunched[$BW1]:-0}" = 1 ] && [ "${relaunched[$BW2]:-0}" = 1 ] && [ "$l40s_launched" = 1 ]; then
    log "all relaunches done; watcher exiting (campaigns log to $REPO/campaign_*.log)"
    exit 0
  fi
  sleep 60
done
