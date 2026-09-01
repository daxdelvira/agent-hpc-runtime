#!/bin/bash
# Event stream for the Tandem collection sprint.
#
# COVERAGE IS THE POINT: silence must not be able to mean "crashed". This emits
# on job state transitions (including disappearance, which is how preemption
# and OOM-kill look), on the wiring/preflight signals that say the tandem path
# is live, on every trial START/END, and on the failure signatures we already
# know this workload produces -- notably "Cannot start ... occupied by", the
# exact error the tandem arm exists to eliminate.
set -uo pipefail
cd /storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime

declare -A SEEN_STATE   # jobid -> last state
declare -A OFFSET       # logfile -> bytes already emitted
KNOWN=""

PAT='=== VERDICT:|MECHANISM (WORKS|FAILED)|INCONCLUSIVE|\[control\]|\[control2\]|\[tandem\]|TANDEM: VllmModelActor wired|preflight\] (all checks passed|FAILED)|START atomagents_exp3_aligned/tandem|END   atomagents_exp3_aligned/tandem|Cannot start .* occupied by|tandem arm exit=|Traceback|CUDA out of memory|Killed|OOM|slurmstepd: error|DUE TO (PREEMPTION|TIME LIMIT)'

while true; do
  # --- job state transitions -------------------------------------------
  cur=$(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | grep -E 'tandem|gpu_eviction' || true)
  now_ids=""
  while read -r jid jname jstate; do
    [ -z "${jid:-}" ] && continue
    now_ids="$now_ids $jid"
    if [ "${SEEN_STATE[$jid]:-}" != "$jstate" ]; then
      echo "[job] $jname ($jid) -> $jstate"
      SEEN_STATE[$jid]=$jstate
    fi
  done <<< "$cur"
  # a job that vanished from the queue: report its terminal state
  for jid in $KNOWN; do
    case " $now_ids " in
      *" $jid "*) ;;
      *) st=$(sacct -j "$jid" -o State -n 2>/dev/null | head -1 | tr -d ' ')
         echo "[job] $jid left the queue -> ${st:-UNKNOWN}"
         unset 'SEEN_STATE[$jid]' ;;
    esac
  done
  KNOWN="$now_ids"

  # --- new log lines worth acting on ------------------------------------
  for f in tandem_inferno_*.log tandem_only_*.log tandem_first_*.log tandem_tp2_*.log gpu_eviction_*.log; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    off=${OFFSET[$f]:-0}
    if [ "$sz" -gt "$off" ]; then
      tail -c "+$((off+1))" "$f" 2>/dev/null \
        | grep -E --line-buffered "$PAT" \
        | sed "s|^|[$f] |" | cut -c1-220
      OFFSET[$f]=$sz
    fi
  done

  sleep 60
done
