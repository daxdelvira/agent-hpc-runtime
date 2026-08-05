#!/bin/bash
# submit_holds.sh — submit the workshop-paper GPU holds.
#
#     bash experiments/submit_holds.sh            # submit the standard batch
#     bash experiments/submit_holds.sh --dry-run  # print the sbatch lines only
#
# WHY THIS FILE EXISTS. Every hold up to 2026-08-05 was submitted by hand, so
# there was no record of what was requested and no way to reproduce a batch.
# Worse, the watcher hardcodes job IDs, so a hand-submitted hold that nobody
# remembered to register lands IDLE. That has now cost three nights:
#   8/02  9 holds (11629977-985) had nothing pointed at them
#   8/04  11653557/8 ran 00:36-04:31 with a dead watcher -> ~7.6 node-h lost
# Submitting and registering must be ONE action. watcher8 discovers jobs by
# NAME, so anything submitted here is picked up automatically -- but only if
# the name keeps the wp_bw_ / wp_l40s_ prefix. Do not rename without changing
# the watcher's discovery pattern.
#
# NO gpu-v100 FALLBACK, deliberately. V100 is SM 7.0 and this vLLM needs 8.0+,
# so a hold that lands there cannot run the campaign; watcher7's header records
# it burning all 4 attempts and blacklisting the hold. A partition list that
# "helps it schedule sooner" schedules it somewhere useless.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
ACCOUNT=gts-ag117
QOS=embers          # free but PREEMPTIBLE: 5 of the last 8 jobs were preempted.
                    # gts-ag117 also carries `inferno` (non-preemptible) -- that
                    # spends real allocation, so switching is dax's call, not a
                    # default.
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# name                partition               gpus  mem    hours  cpus
#
# The 1000G Blackwell holds are the load-bearing new ones. E5 put the retention
# feasibility threshold at 280 GB -- exactly one 72B's park footprint -- and
# every hold so far requested mem=256G, i.e. just on the WRONG SIDE of it. On a
# 2,063,000 MB node that 256G was never a hardware limit, only a request. These
# also settle gate (b)'s open question (plan O5/E6): whether the k=3 sleep
# failure was host-memory exhaustion or cgroup pressure against the allocation.
# Kept alongside 256G holds because a 1000G request is harder to schedule and
# the existing campaign must not stall waiting for one.
HOLDS="
wp_bw_bigmem_a   gpu-rtxpro-blackwell  4  1000G  7:59  12
wp_bw_bigmem_b   gpu-rtxpro-blackwell  4  1000G  7:59  12
wp_bw_std_a      gpu-rtxpro-blackwell  4  256G   7:59  12
wp_bw_std_b      gpu-rtxpro-blackwell  4  256G   7:59  12
wp_bw_short_a    gpu-rtxpro-blackwell  4  256G   3:59  12
wp_l40s_a        gpu-l40s              6  400G   7:59  12
wp_l40s_b        gpu-l40s              6  400G   7:59  12
wp_l40s_short_a  gpu-l40s              6  256G   3:59  12
"

submitted=()
while read -r name part gpus mem hours cpus; do
  [ -z "${name:-}" ] && continue
  cmd=(sbatch --parsable
       --job-name="$name"
       --account="$ACCOUNT"
       --qos="$QOS"
       --partition="$part"
       --nodes=1
       --ntasks=1
       --cpus-per-task="$cpus"
       --gres=gpu:"$gpus"
       --mem="$mem"
       --time="$hours:00"
       --output="$REPO/logs/hold_%x_%j.out"
       --wrap="sleep infinity")
  if [ "$DRY" = 1 ]; then
    printf '%s\n' "${cmd[*]}"
    continue
  fi
  jid=$("${cmd[@]}" 2>&1)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "submitted $name -> $jid  ($part, ${gpus}gpu, $mem, ${hours}h)"
    submitted+=("$jid")
  else
    echo "FAILED   $name: $jid" >&2
  fi
done <<< "$HOLDS"

[ "$DRY" = 1 ] && exit 0
echo
echo "${#submitted[@]} holds submitted. watcher8 discovers these by name -- no"
echo "registration step, and nothing to edit. Confirm it is running:"
echo "    pgrep -fa overnight_watcher8 || setsid nohup bash $REPO/experiments/overnight_watcher8_20260805.sh >/dev/null 2>&1 &"
