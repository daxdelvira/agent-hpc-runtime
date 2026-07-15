"""
End-to-end Option A confirmation using the REAL staging method (plain buffered
read into page cache -- ModelCacheStagingExecutor._read_into_cache), NOT the
mlock+PROT_WRITE method warm_vs_cold_72b.sh used.

Sequence:
  COLD : evict all 37 shards -> vLLM start_model_measured
  WARM : evict all -> plain-read all shards into cache (multi-thread) -> load
Prints both times. If WARM << COLD, page-cache staging via plain read is viable
and the earlier mlock result was an artifact.
"""
import time, os
from concurrent.futures import ThreadPoolExecutor
from atomagents.runtime.model_orchestrator import ModelOrchestrator
from experiments.model_configs import MODELS_CHEMGRAPH_SWAP
from runtime.prefetch.model_cache_prefetch import evict_model_cache, list_model_shards

WORKER = "qwen_72b_instruct"
SNAP = MODELS_CHEMGRAPH_SWAP[WORKER]["model_name"]
orch = ModelOrchestrator(MODELS_CHEMGRAPH_SWAP)

def plain_read(path, bufsize=8 << 20):
    n = 0
    fd = os.open(str(path), os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            pass
        while True:
            b = os.read(fd, bufsize)
            if not b:
                break
            n += len(b)
    finally:
        os.close(fd)
    return n

def warm_all(shards, workers=8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        tot = sum(ex.map(plain_read, shards))
    dt = time.perf_counter() - t0
    print(f"[warm] plain-read {tot/1e9:.1f} GB in {dt:.1f}s "
          f"({tot/1e9/dt:.2f} GB/s, {workers} threads)", flush=True)
    return dt

def load_time(tag):
    t0 = time.perf_counter()
    orch.start_model_measured(WORKER, metrics=None)
    dt = time.perf_counter() - t0
    print(f"[{tag}] 72B start_model_measured = {dt:.1f}s", flush=True)
    orch.stop_model(WORKER)
    time.sleep(20)
    return dt

def main():
    shards = list_model_shards(SNAP)
    print(f"snapshot={SNAP}\nn_shards={len(shards)}", flush=True)

    # ---------- COLD ----------
    n, nb = evict_model_cache(SNAP)
    print(f"[cold] evicted {n} shards / {nb/1e9:.1f} GB", flush=True)
    cold = load_time("cold")

    # ---------- WARM (plain read) ----------
    n, nb = evict_model_cache(SNAP)
    print(f"[warm] evicted {n} shards / {nb/1e9:.1f} GB", flush=True)
    warm_stage = warm_all(shards)
    warm = load_time("warm")

    print("\n==== Option A (plain-read staging) result ====")
    print(f"cold load        = {cold:.1f}s")
    print(f"warm staging     = {warm_stage:.1f}s (overlappable with planning)")
    print(f"warm load        = {warm:.1f}s")
    print(f"critical-path benefit = cold - warm = {cold - warm:.1f}s")

if __name__ == "__main__":
    main()
