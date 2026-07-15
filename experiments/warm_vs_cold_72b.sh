#!/bin/bash
# Decisive test: is the 72B vLLM load I/O-bound (warm cache helps) or GPU-bound
# (warm cache does nothing)?  Loads the 72B twice via the orchestrator:
#   COLD  — evict shards, then load
#   WARM  — mmap+mlock all shards (pinned resident, defeats Lustre eviction), load
# Prints start_model_measured() elapsed for each.  If WARM << COLD, page-cache
# staging is worthwhile (Option A, rescued with mlock).  If WARM ~= COLD, the
# load is GPU-bound and we pivot (Option B co-resident / Option D).
set -uo pipefail
PROJ=/storage/project/r-ag117-0/shared/agent_hpc/agent-hpc-runtime
CG_PYTHON=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/bin/python
AA_NVIDIA=/storage/project/r-ag117-0/shared/agent_hpc/envs/atomagents/lib/python3.11/site-packages/nvidia
TORCH_LIB=/storage/project/r-ag117-0/shared/agent_hpc/envs/chemgraph/lib/python3.10/site-packages/torch/lib
export LD_LIBRARY_PATH=$AA_NVIDIA/cudnn/lib:$AA_NVIDIA/cusparselt/lib:$AA_NVIDIA/nccl/lib:$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJ/workloads/AtomAgents:$PROJ/ChemGraph/src:${PYTHONPATH:-}
export HF_HOME=$HOME/scratch/hf_home
cd $PROJ

# Redirect temp off a broken job-private /tmp and disable XALT so vLLM's Triton
# JIT compile at engine-core init succeeds (see setup/fix_tmp.sh).  The spawned
# vLLM subprocess inherits this env.
source $PROJ/setup/fix_tmp.sh

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 15

$CG_PYTHON - <<'PY'
import time, mmap, os, ctypes
from atomagents.runtime.model_orchestrator import ModelOrchestrator
from experiments.model_configs import MODELS_CHEMGRAPH_SWAP
from runtime.prefetch.model_cache_prefetch import evict_model_cache, list_model_shards

WORKER = "qwen_72b_instruct"
SNAP = MODELS_CHEMGRAPH_SWAP[WORKER]["model_name"]
orch = ModelOrchestrator(MODELS_CHEMGRAPH_SWAP)
libc = ctypes.CDLL("libc.so.6", use_errno=True)

def load_time(tag):
    t0 = time.perf_counter()
    orch.start_model_measured(WORKER, metrics=None)
    dt = time.perf_counter() - t0
    print(f"[{tag}] 72B start_model_measured = {dt:.1f}s", flush=True)
    orch.stop_model(WORKER)
    time.sleep(20)
    return dt

# ---------- COLD ----------
n, nb = evict_model_cache(SNAP)
print(f"[cold] evicted {n} shards / {nb/1e9:.1f} GB", flush=True)
cold = load_time("cold")

# ---------- WARM (mmap + mlock => pages pinned resident) ----------
n, nb = evict_model_cache(SNAP)
shards = list_model_shards(SNAP)
maps = []
t0 = time.perf_counter()
pinned = 0
for p in shards:
    fd = os.open(str(p), os.O_RDONLY)
    sz = os.fstat(fd).st_size
    # MAP_PRIVATE + writable prot so ctypes.from_buffer can take the address;
    # reads fault pages into the shared page cache (COW only triggers on write).
    m = mmap.mmap(fd, sz, flags=mmap.MAP_PRIVATE,
                  prot=mmap.PROT_READ | mmap.PROT_WRITE)
    buf = (ctypes.c_char * sz).from_buffer(m)
    addr = ctypes.addressof(buf)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(sz)) != 0:
        err = ctypes.get_errno()
        print(f"[warm] mlock failed on {os.path.basename(str(p))}: errno {err}", flush=True)
    maps.append((fd, m, buf))
    pinned += sz
warm_read = time.perf_counter() - t0
print(f"[warm] pinned {pinned/1e9:.1f} GB in {warm_read:.1f}s", flush=True)
warm = load_time("warm")
for fd, m, buf in maps:
    try:
        libc.munlock(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(len(m)))
    except Exception:
        pass
    del buf
    m.close(); os.close(fd)

print(f"\n=== RESULT: cold={cold:.1f}s  warm={warm:.1f}s  "
      f"benefit={cold-warm:.1f}s ({100*(cold-warm)/cold:.0f}%) ===", flush=True)
PY
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
