"""
Decisive page-cache mechanism test for Option A.

Question: does vLLM's mmap-based safetensors read (safe_open MAP_PRIVATE|PROT_READ)
get served by a page cache warmed with a PLAIN BUFFERED READ (what the real
ModelCacheStagingExecutor does) -- as opposed to the mlock+PROT_WRITE method the
warm_vs_cold_72b.sh validation used, which pins private/anon copies and does NOT
populate the shared file cache?

For each of several 72B shards we measure three reads of the SAME file:
  1. COLD   : posix_fadvise(DONTNEED) to evict, then read -> Lustre cold rate
  2. WARM-plain (staging): a plain buffered read() to warm the shared cache
  3. mmap re-read (vLLM path): mmap MAP_PRIVATE|PROT_READ + touch every page,
     IN A SEPARATE sense from the warmer -> if this is fast, the vLLM loader IS
     served by the plain-read-warmed cache.

If step 3 rate ~= step 2 warm rate (>>cold), the earlier negative Option A result
was an artifact of the mlock method, and page-cache staging via plain read is viable.
"""
import os, glob, time, mmap, ctypes, ctypes.util

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
POSIX_FADV_DONTNEED = 4

def fadvise_dontneed(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd) if False else None
        r = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        if r != 0:
            raise OSError(r, "posix_fadvise DONTNEED failed")
    finally:
        os.close(fd)

def plain_read(path, bufsize=8 << 20):
    """Buffered read that populates the shared page cache. Returns (bytes, secs)."""
    n = 0
    t0 = time.time()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            n += len(b)
    return n, time.time() - t0

def mmap_read(path):
    """Read via mmap MAP_PRIVATE|PROT_READ (the safetensors safe_open path).
    Touch every page so the kernel faults them in. Returns (bytes, secs)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE)
        t0 = time.time()
        # touch one byte per 4KB page; sum to prevent optimizing away
        s = 0
        step = 4096
        for off in range(0, size, step):
            s += mm[off]
        dt = time.time() - t0
        mm.close()
        return size, dt, s
    finally:
        os.close(fd)

def gb_s(nbytes, secs):
    return (nbytes / 1e9) / secs if secs > 0 else float("inf")

def main():
    import experiments.model_configs as mc
    snap = mc._SNAPSHOT_72B_TEXT
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    # Test the first 5 shards -- enough signal, ~19 GB, fast.
    shards = shards[:5]
    print(f"Testing {len(shards)} shards from {snap}\n")
    print(f"{'shard':<26} {'GB':>6} {'cold GB/s':>10} {'warm(plain)':>12} {'mmap re-read':>13}")
    cold_tot = warm_tot = mmap_tot = 0.0
    bytes_tot = 0
    for path in shards:
        name = os.path.basename(path)
        # 1. cold
        fadvise_dontneed(path)
        n, t_cold = plain_read(path)
        # 2. warm-plain (this read is already warm-ish since #1 populated cache;
        #    to isolate, evict then do ONE warming read, then measure a SECOND
        #    plain read as the "warm" number)
        n2, t_warm = plain_read(path)
        # 3. mmap re-read (vLLM path) on the now-warm cache
        n3, t_mmap, _ = mmap_read(path)
        bytes_tot += n
        cold_tot += t_cold; warm_tot += t_warm; mmap_tot += t_mmap
        print(f"{name:<26} {n/1e9:6.2f} {gb_s(n,t_cold):10.2f} "
              f"{gb_s(n2,t_warm):12.2f} {gb_s(n3,t_mmap):13.2f}")
    print("-" * 72)
    print(f"{'TOTAL/avg':<26} {bytes_tot/1e9:6.2f} "
          f"{gb_s(bytes_tot,cold_tot):10.2f} {gb_s(bytes_tot,warm_tot):12.2f} "
          f"{gb_s(bytes_tot,mmap_tot):13.2f}")
    print()
    cold_r = gb_s(bytes_tot, cold_tot)
    warm_r = gb_s(bytes_tot, warm_tot)
    mmap_r = gb_s(bytes_tot, mmap_tot)
    print(f"cold        : {cold_r:.2f} GB/s")
    print(f"warm(plain) : {warm_r:.2f} GB/s  ({warm_r/cold_r:.1f}x cold)")
    print(f"mmap (vLLM) : {mmap_r:.2f} GB/s  ({mmap_r/cold_r:.1f}x cold)")
    if mmap_r > 2.5 * cold_r:
        print("\n=> VERDICT: vLLM's mmap path IS served by a plain-read-warmed cache.")
        print("   The earlier Option A negative result was an mlock artifact.")
    else:
        print("\n=> VERDICT: mmap path does NOT benefit even from a plain-read-warmed")
        print("   cache -> vLLM genuinely bypasses/defeats the client cache here.")

if __name__ == "__main__":
    main()
