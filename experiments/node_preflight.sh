#!/bin/bash
# node_preflight.sh — shared-node health canary. Added 2026-07-28 after inferno
# job 11523454 burned a full hold on a node where vLLM boots crawled from ~3 min
# (same node, 07-17) to >3600 s: near-zero disk reads with a crawling engine
# init, i.e. a host-side (PCIe / memory-bandwidth) pathology that no Lustre
# check would catch, while foreign seekr2 MD ran on the node's other GPUs.
#
# Exit 0 = node looks usable; exit 1 = abort the campaign (watcher will retry,
# and after its attempt limit move on / blacklist).  ~60-90 s total.
#
# Checks:
#   1. Foreign compute (non-$USER processes >=50%% CPU): WARN only — nodes are
#      shared by policy; the go/no-go signals are the canaries below.
#   2. Lustre read canary: O_DIRECT read of 2 GiB from the 72B snapshot's
#      first shard.  Floor 50 MB/s: Lustre legitimately swings 40-300 MB/s
#      (model_configs.py planner note) and a slow swap is VALID DATA — the
#      floor only screens the pathological regime where trials time out.
#   3. CUDA init + H2D canary (the check that would have caught 07-28):
#      time torch CUDA init and a 1 GiB pinned host->GPU0 copy in the vLLM
#      env.  Floors: whole step <= 180 s, H2D >= 3 GB/s (L40S/Blackwell PCIe
#      healthy is >20 GB/s; 07-28-style pathology lands far below).
set -uo pipefail
cd "$(dirname "$0")/.."
plog(){ echo "[preflight $(date +'%T')] $*"; }

# --- 1. foreign compute (warn only) ----------------------------------------
FOREIGN=$(ps -eo user,pcpu,cmd --sort=-pcpu | awk -v me="$USER" \
  '$1!=me && $1!="root" && $2>=50 {print; n++} n>=5{exit}' \
  | grep -vE "dcgm|node_exporter|nv-hostengine|prometheus" || true)
if [ -n "$FOREIGN" ]; then
  plog "WARN: foreign compute on node (recorded, not fatal):"
  echo "$FOREIGN" | sed 's/^/[preflight]   /'
else
  plog "no significant foreign compute visible"
fi

# --- 2. Lustre read canary ---------------------------------------------------
SNAP=$(python3 -c "
import sys; sys.path.insert(0, 'experiments')
from model_configs import MODELS_CHEMGRAPH_SWAP as M
print(M['qwen_72b_instruct']['model_name'])" 2>/dev/null)
SHARD=$(ls "$SNAP"/*.safetensors 2>/dev/null | head -1)
if [ -z "$SHARD" ]; then
  plog "WARN: no snapshot shard found under '$SNAP' — skipping read canary"
else
  t0=$(date +%s.%N)
  dd if="$SHARD" of=/dev/null bs=64M count=32 iflag=direct 2>/dev/null
  t1=$(date +%s.%N)
  MBPS=$(python3 -c "print(round(2048/($t1-$t0)))")
  plog "Lustre O_DIRECT read: ${MBPS} MB/s (floor 50)"
  if [ "$MBPS" -lt 50 ]; then
    plog "ABORT: Lustre read canary below floor — node/filesystem unhealthy"
    exit 1
  fi
fi

# --- 3. CUDA init + H2D canary ----------------------------------------------
VLLM_PY=$(python3 -c "
import sys; sys.path.insert(0, 'experiments')
from model_configs import VLLM_PYTHON
print(VLLM_PYTHON)" 2>/dev/null)
if [ -z "$VLLM_PY" ] || [ ! -x "$VLLM_PY" ]; then
  plog "ABORT: vLLM python not found ('$VLLM_PY')"
  exit 1
fi
# NB: fed on stdin via printf (a shell builtin), NOT a here-document.  A
# here-doc needs a writable $TMPDIR, and inside some job containers TMPDIR
# points at a path that does not exist — bash then fails with "cannot create
# temp file for here-document" and this canary reports ABORT on a perfectly
# healthy node (observed job 11518012, 2026-07-29, which burned an attempt).
CUDA_CANARY_PY='
import time, torch
torch.cuda.init()
x = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
x.to("cuda:0")
torch.cuda.synchronize()
print(round(1.0 / (time.perf_counter() - t0), 2))
'
t0=$(date +%s.%N)
H2D=$(printf '%s\n' "$CUDA_CANARY_PY" | timeout 180 "$VLLM_PY" -) \
  || { plog "ABORT: CUDA canary failed or exceeded 180 s — host/GPU path unhealthy"; exit 1; }
ELAPSED=$(python3 -c "print(round($(date +%s.%N)-$t0,1))")
plog "CUDA init+H2D canary: ${H2D} GB/s, step ${ELAPSED}s (floors: 3 GB/s, 180 s)"
if python3 -c "exit(0 if float('$H2D') < 3.0 else 1)"; then
  plog "ABORT: H2D bandwidth below floor — PCIe/host-memory contention"
  exit 1
fi
plog "node canaries PASS"
