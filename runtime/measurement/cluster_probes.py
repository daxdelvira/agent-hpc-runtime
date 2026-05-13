"""
measurement/cluster_probes.py — PACE-safe hardware probes for storage hierarchy detection.

Answers the question: "when a model loaded in Xs, where did the data actually come from?"

Three independent probe sources, each failing silently if unavailable:

  /proc/self/io  — per-process byte counters.
                   `rchar`      = all bytes passed to read() (includes page cache hits)
                   `read_bytes` = bytes that actually reached storage (cache misses only)
                   delta(read_bytes) ≈ 0        → served from page cache
                   delta(read_bytes) ≈ file size → cold NFS / disk read

  psutil.net_io_counters()  — host-level network traffic.
                   On PACE, NFS-over-TCP appears as RX bytes on the main NIC.
                   Large net_recv delta during a model load corroborates NFS.
                   (Lustre over RDMA won't appear here; see note below.)

  nvidia-smi     — GPU VRAM occupancy.
                   delta(memory.used) after load confirms weights landed in GPU memory
                   vs. the vLLM process starting but deferring the actual CUDA load.

  vmtouch -v     — page-cache residency of a file (optional, may not be installed).
                   Reports fraction of file pages currently in kernel page cache.
                   Run BEFORE the load to detect warm-cache hits before they happen.

Storage hierarchy inference
---------------------------
Given a load of a known-size file (optional):

  cache_hit_ratio = 1.0 - (proc_read_bytes_delta / proc_rchar_delta)
  ≈ 1.0  →  "page_cache"  (OS served from RAM, no storage I/O)
  ≈ 0.0  →  "nfs"         (all bytes from NFS / disk)
  middle →  "mixed"        (partial cache hit, e.g. multi-GPU sharded load
                             where some shards were cached from a prior run)

PACE-specific notes
-------------------
- NFS over TCP: appears in psutil.net_io_counters() as normal RX bytes.
- Lustre (GPFS): NOT visible via net_io_counters(); but /proc/self/io still works
  because Lustre presents a POSIX filesystem — read_bytes tracks Lustre reads.
- RDMA NFS: also NOT in net_io_counters(); /proc/self/io is the reliable fallback.
- For all three storage backends, (read_bytes delta ≈ 0, rchar delta ≈ file_size)
  unambiguously means page cache. This is the most reliable single indicator.

Usage
-----
    probes = ClusterProbes()
    avail = probes.check_availability()

    before = probes.snapshot()
    ... run model load ...
    after = probes.snapshot()

    delta = probes.delta(before, after)
    print(delta.likely_source)   # "nfs" | "page_cache" | "mixed" | "unknown"
    print(delta.to_dict())       # embed in JSONL payload

    # Optional: check page cache BEFORE the load
    residency = probes.page_cache_residency("/path/to/model/weights")
    # 1.0 = fully cached, 0.0 = fully cold

Integration with ModelPrefetchExecutor
---------------------------------------
Pass ClusterProbes to ModelPrefetchExecutor(probes=probes). It will snapshot
before/after start_model_measured() and embed the delta in the prefetch_completed
JSONL event payload under the key "probe_delta".
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ProbeSnapshot:
    """Point-in-time snapshot of all measurable storage/memory metrics."""
    timestamp: float = field(default_factory=time.time)

    # /proc/self/io
    proc_rchar: int | None = None           # total bytes passed to read() (incl. cache)
    proc_read_bytes: int | None = None      # bytes actually fetched from storage

    # psutil network counters (host-wide, all interfaces summed)
    net_bytes_recv: int | None = None
    net_bytes_sent: int | None = None

    # nvidia-smi: {gpu_index: memory_used_mib}
    gpu_memory_mib: dict[int, int] | None = None


@dataclass
class ProbeDelta:
    """
    Difference between two ProbeSnapshots, with derived storage-hierarchy inference.

    All delta fields are None when the corresponding probe was unavailable.
    """
    elapsed_s: float = 0.0

    # Process I/O deltas
    proc_rchar_delta: int | None = None         # total bytes read (cache + storage)
    proc_read_bytes_delta: int | None = None    # bytes from actual storage (cache-miss bytes)

    # Derived cache metric (requires both proc fields)
    cache_hit_ratio: float | None = None        # 1.0 = all cache, 0.0 = all storage

    # Network I/O deltas (NFS-over-TCP proxy)
    net_recv_bytes_delta: int | None = None
    net_sent_bytes_delta: int | None = None

    # GPU VRAM deltas: {gpu_index: mib_increase}
    gpu_vram_delta_mib: dict[int, int] | None = None
    gpu_vram_total_delta_mib: int | None = None    # sum across all GPUs

    # High-level inference
    likely_source: str = "unknown"   # "nfs" | "page_cache" | "mixed" | "unknown"
    inference_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def summary_line(self) -> str:
        """One-line summary for logging."""
        parts = [f"source={self.likely_source}", f"elapsed={self.elapsed_s:.1f}s"]
        if self.proc_read_bytes_delta is not None:
            parts.append(f"storage_read={self.proc_read_bytes_delta / 1e9:.2f}GB")
        if self.cache_hit_ratio is not None:
            parts.append(f"cache_hit={self.cache_hit_ratio:.0%}")
        if self.gpu_vram_total_delta_mib is not None:
            parts.append(f"gpu_vram_delta={self.gpu_vram_total_delta_mib}MiB")
        if self.net_recv_bytes_delta is not None:
            parts.append(f"net_recv={self.net_recv_bytes_delta / 1e9:.2f}GB")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main probe class
# ---------------------------------------------------------------------------

class ClusterProbes:
    """
    PACE-safe hardware probes. All methods fail silently and return None
    for unavailable measurements rather than raising exceptions.
    """

    def check_availability(self) -> dict[str, bool]:
        """Report which probe sources are available on this node."""
        return {
            "proc_io":   _proc_io_available(),
            "psutil":    _psutil_available(),
            "nvidia_smi": _nvidia_smi_available(),
            "vmtouch":   _vmtouch_available(),
        }

    def snapshot(self) -> ProbeSnapshot:
        """Capture a point-in-time snapshot of all available metrics."""
        s = ProbeSnapshot(timestamp=time.time())

        # /proc/self/io
        proc = _read_proc_io()
        if proc:
            s.proc_rchar = proc.get("rchar")
            s.proc_read_bytes = proc.get("read_bytes")

        # psutil network
        net = _read_net_io()
        if net:
            s.net_bytes_recv = net.get("bytes_recv")
            s.net_bytes_sent = net.get("bytes_sent")

        # nvidia-smi
        s.gpu_memory_mib = _read_gpu_memory()

        return s

    def delta(self, before: ProbeSnapshot, after: ProbeSnapshot) -> ProbeDelta:
        """Compute the delta between two snapshots and infer storage source."""
        d = ProbeDelta(elapsed_s=after.timestamp - before.timestamp)

        # Process I/O
        if before.proc_rchar is not None and after.proc_rchar is not None:
            d.proc_rchar_delta = after.proc_rchar - before.proc_rchar
        if before.proc_read_bytes is not None and after.proc_read_bytes is not None:
            d.proc_read_bytes_delta = after.proc_read_bytes - before.proc_read_bytes

        # Cache hit ratio
        if (
            d.proc_rchar_delta is not None
            and d.proc_read_bytes_delta is not None
            and d.proc_rchar_delta > 0
        ):
            d.cache_hit_ratio = max(
                0.0,
                1.0 - d.proc_read_bytes_delta / d.proc_rchar_delta,
            )

        # Network
        if before.net_bytes_recv is not None and after.net_bytes_recv is not None:
            d.net_recv_bytes_delta = after.net_bytes_recv - before.net_bytes_recv
        if before.net_bytes_sent is not None and after.net_bytes_sent is not None:
            d.net_sent_bytes_delta = after.net_bytes_sent - before.net_bytes_sent

        # GPU VRAM
        if before.gpu_memory_mib is not None and after.gpu_memory_mib is not None:
            vram_delta: dict[int, int] = {}
            for gpu_idx, after_mib in after.gpu_memory_mib.items():
                before_mib = before.gpu_memory_mib.get(gpu_idx, 0)
                vram_delta[gpu_idx] = after_mib - before_mib
            d.gpu_vram_delta_mib = vram_delta
            d.gpu_vram_total_delta_mib = sum(vram_delta.values())

        d.likely_source, d.inference_notes = _infer_source(d)
        return d

    def page_cache_residency(self, path: str) -> float | None:
        """
        Return fraction of file pages currently in page cache (0.0–1.0),
        or None if vmtouch is unavailable or the path doesn't exist.

        Run BEFORE a load to detect whether this will be a cache hit.
        """
        if not Path(path).exists():
            return None
        return _run_vmtouch(path)

    def snapshot_gpu_memory(self) -> dict[int, int] | None:
        """Return current {gpu_index: memory_used_mib} or None."""
        return _read_gpu_memory()


# ---------------------------------------------------------------------------
# Low-level readers — each returns None on any failure
# ---------------------------------------------------------------------------

def _proc_io_available() -> bool:
    return Path("/proc/self/io").exists()


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _vmtouch_available() -> bool:
    return shutil.which("vmtouch") is not None


def _read_proc_io(pid: int | None = None) -> dict[str, int] | None:
    """
    Parse /proc/<pid>/io. Uses the current process if pid is None.

    Fields returned (subset):
      rchar       — bytes passed to read() [includes page cache]
      read_bytes  — bytes actually fetched from block device / NFS [cache misses]
    """
    path = Path(f"/proc/{pid}/io") if pid else Path("/proc/self/io")
    try:
        text = path.read_text()
    except (OSError, PermissionError):
        return None
    result: dict[str, int] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            try:
                result[key.strip()] = int(val.strip())
            except ValueError:
                pass
    return result or None


def _read_net_io() -> dict[str, int] | None:
    """
    Read host-wide network I/O counters (all interfaces summed, excluding loopback).
    Returns {"bytes_recv": N, "bytes_sent": N} or None.
    """
    try:
        import psutil
        counters = psutil.net_io_counters(pernic=True)
        total_recv = 0
        total_sent = 0
        for nic, stats in counters.items():
            if nic == "lo":
                continue
            total_recv += stats.bytes_recv
            total_sent += stats.bytes_sent
        return {"bytes_recv": total_recv, "bytes_sent": total_sent}
    except Exception:
        return None


def _read_gpu_memory() -> dict[int, int] | None:
    """
    Run nvidia-smi and parse per-GPU memory usage.
    Returns {gpu_index: memory_used_mib} or None if nvidia-smi unavailable.
    """
    if not _nvidia_smi_available():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        memory: dict[int, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    memory[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
        return memory or None
    except Exception:
        return None


def _run_vmtouch(path: str) -> float | None:
    """
    Run vmtouch -v on path and parse page-cache residency fraction.
    Returns 0.0–1.0 or None.
    """
    try:
        result = subprocess.run(
            ["vmtouch", "-v", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # vmtouch output includes a line like:
        #   [OOO                                      ] 3/768 = 0.39%
        # and a summary line like:
        #   Resident Pages: 3/768  384K/98M  0.39%
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if "Resident Pages:" in line or ("/" in line and "%" in line):
                # Try to extract "N/M" fraction
                import re
                m = re.search(r"(\d+)/(\d+)", line)
                if m:
                    num, denom = int(m.group(1)), int(m.group(2))
                    if denom > 0:
                        return num / denom
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source inference
# ---------------------------------------------------------------------------

def _infer_source(d: ProbeDelta) -> tuple[str, list[str]]:
    """
    Infer where the data came from based on available deltas.

    Returns (source_label, [reasoning notes]).
    """
    notes: list[str] = []
    votes_cache = 0
    votes_nfs = 0
    votes_unknown = 0

    # Primary signal: cache_hit_ratio from /proc/self/io
    if d.cache_hit_ratio is not None:
        if d.cache_hit_ratio >= 0.90:
            votes_cache += 3
            notes.append(
                f"proc/io: cache_hit_ratio={d.cache_hit_ratio:.0%} → data served from page cache"
            )
        elif d.cache_hit_ratio <= 0.15:
            votes_nfs += 3
            notes.append(
                f"proc/io: cache_hit_ratio={d.cache_hit_ratio:.0%} → data fetched from storage (NFS/disk)"
            )
        else:
            votes_cache += 1
            votes_nfs += 1
            notes.append(
                f"proc/io: cache_hit_ratio={d.cache_hit_ratio:.0%} → partial cache hit (mixed)"
            )
    else:
        votes_unknown += 1
        notes.append("proc/io: unavailable (not Linux or permission denied)")

    # Secondary signal: network I/O (corroborates NFS-over-TCP)
    if d.net_recv_bytes_delta is not None:
        gb = d.net_recv_bytes_delta / 1e9
        if gb > 1.0:
            votes_nfs += 2
            notes.append(f"network: received {gb:.1f}GB → consistent with NFS-over-TCP transfer")
        elif gb < 0.01 and d.elapsed_s > 10:
            # Loaded large amount but little network → page cache or Lustre/RDMA
            votes_cache += 1
            notes.append(
                f"network: only {gb:.3f}GB received — likely page cache, Lustre, or RDMA NFS"
            )
        else:
            notes.append(f"network: {gb:.3f}GB received (inconclusive)")
    else:
        notes.append("network: psutil unavailable")

    # GPU VRAM: confirms the load actually populated GPU memory
    if d.gpu_vram_total_delta_mib is not None:
        if d.gpu_vram_total_delta_mib > 1000:
            notes.append(
                f"gpu: VRAM increased by {d.gpu_vram_total_delta_mib}MiB "
                f"→ model weights landed in GPU memory ✓"
            )
        elif d.gpu_vram_total_delta_mib < 100:
            notes.append(
                f"gpu: VRAM delta only {d.gpu_vram_total_delta_mib}MiB "
                f"→ GPU weights not yet loaded (deferred by vLLM?)"
            )
    else:
        notes.append("gpu: nvidia-smi unavailable (expected on non-GPU node)")

    # Verdict
    if votes_cache > votes_nfs and votes_cache > votes_unknown:
        source = "page_cache"
    elif votes_nfs > votes_cache and votes_nfs > votes_unknown:
        source = "nfs"
    elif votes_nfs > 0 and votes_cache > 0:
        source = "mixed"
    else:
        source = "unknown"

    return source, notes


# ---------------------------------------------------------------------------
# Convenience: context manager for bracketing a load
# ---------------------------------------------------------------------------

class LoadProbeContext:
    """
    Context manager that snapshots before/after a code block.

    Usage:
        probes = ClusterProbes()
        with LoadProbeContext(probes) as ctx:
            model.load()
        print(ctx.delta.summary_line())
        print(ctx.delta.likely_source)
    """

    def __init__(self, probes: ClusterProbes | None = None) -> None:
        self._probes = probes or ClusterProbes()
        self.before: ProbeSnapshot | None = None
        self.after: ProbeSnapshot | None = None
        self.delta: ProbeDelta | None = None

    def __enter__(self) -> "LoadProbeContext":
        self.before = self._probes.snapshot()
        return self

    def __exit__(self, *_) -> None:
        self.after = self._probes.snapshot()
        if self.before and self.after:
            self.delta = self._probes.delta(self.before, self.after)
