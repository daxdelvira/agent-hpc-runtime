#!/usr/bin/env bash
# check_hermes_blackwell.sh — does the Hermes/MegaMmap stack actually SERVE on a
# Blackwell node?
#
# Every megammap_stage trial collected so far ran on L40S; atomagents_exp3 is
# Blackwell-only, and megammap_stage.py carries a libstdc++ ABI workaround, so
# "it builds" was not evidence that it runs here. This stages one real model
# snapshot through mm_model_preload against a live daemon and reports whether
# bytes actually landed in the Hermes tier.
#
# Run INSIDE an allocation (the daemon and the client must share one step -- a
# daemon launched from a step that exits is killed with that step's process
# group):
#     srun --jobid=<JOB> --overlap -n1 bash scripts/check_hermes_blackwell.sh
#
# Writes a self-contained report to /tmp/hermes_check.log on the node.
set -u
LOG=/tmp/hermes_check.log
exec >"$LOG" 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node        : $(hostname)"
say "gpus        : $(nvidia-smi -L 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

source ~/scratch/mega_stack/mega_env.sh >/dev/null 2>&1
export HERMES_CONF=~/scratch/mega_src/hermes/config/hermes_agentic.yaml
say "HERMES_CONF : $HERMES_CONF"
mkdir -p /tmp/hermes_nvme_agentic

BIN=/storage/project/r-ag117-0/shared/agent_hpc/mega_mmap_integration/megammap_tests/build/bin/mm_model_preload
SNAP=$(ls -d ~/scratch/hf_home/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/*/ 2>/dev/null | head -1)
say "snapshot    : $SNAP"
say "shards      : $(ls "$SNAP"*.safetensors 2>/dev/null | wc -l)"
say "shard bytes : $(du -sh "$SNAP" 2>/dev/null | cut -f1)"

say "--- starting daemon ---"
nohup hrun_start_runtime >/tmp/hermes_daemon.log 2>&1 &
DPID=$!
sleep 30
if ! kill -0 "$DPID" 2>/dev/null; then
  say "RESULT: FAIL — daemon exited within 30 s"
  say "--- daemon log ---"; tail -40 /tmp/hermes_daemon.log
  exit 1
fi
say "daemon alive (pid $DPID)"

say "--- staging via mm_model_preload (seq, 4g window) ---"
T0=$(date +%s)
timeout 900 mpirun -n 1 "$BIN" --shard-dir "$SNAP" --tx-type seq --window 4g
RC=$?
T1=$(date +%s)
say "mm_model_preload exit=$RC elapsed=$((T1-T0))s"

say "nvme tier   : $(du -sh /tmp/hermes_nvme_agentic 2>/dev/null | cut -f1)"
say "--- daemon log tail ---"
tail -25 /tmp/hermes_daemon.log

if [ "$RC" -eq 0 ]; then
  say "RESULT: PASS — daemon served a real staging request on Blackwell"
else
  say "RESULT: FAIL — staging returned $RC (124 = timed out)"
fi

hrun_stop_runtime >/dev/null 2>&1
pkill -f hrun_start_runtime 2>/dev/null
say "daemon stopped"
