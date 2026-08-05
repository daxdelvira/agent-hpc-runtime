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
# NO gpu-v100, AND --partition ALONE DOES NOT ACHIEVE THAT. The site appends
# gpu-v100 to the partition list regardless of what is requested (watcher7's
# header records the same thing: "11629978: Partition=gpu-rtxpro-blackwell,
# gpu-v100"). V100 is SM 7.0 and this vLLM needs 8.0+, so a hold that lands
# there cannot run the campaign and burns all 4 watcher attempts.
#
# Worse than useless — on 2026-08-05 it made two holds UNSCHEDULABLE. The
# mem=1000G Blackwell holds went to Reason=BadConstraints even though
# `sbatch --test-only` confirmed the same request schedules fine on Blackwell
# alone. One impossible partition in the list poisons the whole job, and
# BadConstraints does not clear on its own the way Resources does.
#
# THE MEMORY CEILING IS 772000 MB, and it is not obvious. gpu-v100 is
# heterogeneous — `sinfo -p gpu-v100 -h -N -o %m | sort -n | uniq -c` gives
# 9 nodes at 191000, 26 at 385000, 5 at 772000. So a request is schedulable
# exactly while it fits the LARGEST v100 node:
#     256G = 262144 MB   -> fits            -> PENDING(Resources)   fine
#    1000G = 1024000 MB  -> fits nothing    -> BadConstraints       stuck
# Hence 750G below: comfortably under 772000, and still 2.7x the 280 GB
# retention threshold — enough to park two 72B engines (2 x 279 GB) at once,
# which is the configuration E5 says we actually need.
#
# --constraint on a node feature was tried first and does NOT fix it: the
# appended partition survives the feature filter (verified on 11692938,
# Partition=gpu-rtxpro-blackwell,gpu-v100 with Features=RTX-Pro-Blackwell set).
# The feature is kept anyway because it still prevents a hold from LANDING on
# a V100, which is the failure watcher7's header documents.
set -u
REPO=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
ACCOUNT=gts-ag117
QOS=embers          # free but PREEMPTIBLE: 5 of the last 8 jobs were preempted.
                    # gts-ag117 also carries `inferno` (non-preemptible) -- that
                    # spends real allocation, so switching is dax's call, not a
                    # default.
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# name                partition               gpus  mem    hours  cpus  feature
#
# The 750G Blackwell holds are the load-bearing new ones. E5 put the retention
# feasibility threshold at 280 GB -- exactly one 72B's park footprint -- and
# every hold so far requested mem=256G, i.e. just on the WRONG SIDE of it. On a
# 2,063,000 MB node that 256G was never a hardware limit, only a request. These
# also settle gate (b)'s open question (plan O5/E6): whether the k=3 sleep
# failure was host-memory exhaustion or cgroup pressure against the allocation.
# Kept alongside 256G holds because a large request is slower to schedule and
# the existing campaign must not stall waiting for one.
HOLDS="
wp_bw_bigmem_a   gpu-rtxpro-blackwell  4  750G   7:59  12  RTX-Pro-Blackwell
wp_bw_bigmem_b   gpu-rtxpro-blackwell  4  750G   7:59  12  RTX-Pro-Blackwell
wp_bw_std_a      gpu-rtxpro-blackwell  4  256G   7:59  12  RTX-Pro-Blackwell
wp_bw_std_b      gpu-rtxpro-blackwell  4  256G   7:59  12  RTX-Pro-Blackwell
wp_bw_short_a    gpu-rtxpro-blackwell  4  256G   3:59  12  RTX-Pro-Blackwell
wp_l40s_a        gpu-l40s              6  400G   7:59  12  L40S
wp_l40s_b        gpu-l40s              6  400G   7:59  12  L40S
wp_l40s_short_a  gpu-l40s              6  256G   3:59  12  L40S
"

submitted=()
while read -r name part gpus mem hours cpus feature; do
  [ -z "${name:-}" ] && continue
  cmd=(sbatch --parsable
       --job-name="$name"
       --account="$ACCOUNT"
       --qos="$QOS"
       --partition="$part"
       --constraint="$feature"
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
