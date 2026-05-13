"""
measurement/timings.py — Overlap computation and per-prefetch timing records.

PrefetchTimingRecord captures the four key timestamps needed to answer
the research questions:
  - prefetch_start_t   : when speculative I/O began
  - prefetch_end_t     : when speculative I/O completed
  - resource_needed_t  : when the workflow step that consumes this resource started
  - baseline_load_s    : how long loading took without prefetch (from baseline run)

Derived metrics:
  overlap_s   : compute time during which prefetch was running
  benefit_s   : time saved (positive = prefetch finished before it was needed)
  waste_s     : load latency that still blocked the workflow (prefetch arrived late)
  wasted      : prefetch completed but divergence made it unused
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from typing import Iterator


@dataclass
class PrefetchTimingRecord:
    run_id: str = ""
    resource_id: str = ""
    resource_name: str = ""
    resource_type: str = ""
    predictor_id: str = ""
    checkpoint_id: str = ""
    predicted_at_step: int = 0
    consumed_at_step: int = 0

    prefetch_start_t: float | None = None
    prefetch_end_t: float | None = None
    resource_needed_t: float | None = None   # when consuming step started
    baseline_load_s: float | None = None     # from baseline run (if available)

    cancelled: bool = False
    wasted: bool = False     # completed but divergence made it unnecessary
    hit: bool = False        # prediction was correct

    @property
    def overlap_s(self) -> float:
        """Seconds of load time hidden behind concurrent compute."""
        if self.prefetch_start_t is None or self.prefetch_end_t is None:
            return 0.0
        if self.cancelled:
            return 0.0
        needed = self.resource_needed_t or self.prefetch_end_t
        overlap_end = min(self.prefetch_end_t, needed)
        return max(0.0, overlap_end - self.prefetch_start_t)

    @property
    def benefit_s(self) -> float:
        """Seconds saved: positive if prefetch completed before resource was needed."""
        if self.prefetch_end_t is None or self.resource_needed_t is None:
            return 0.0
        if self.cancelled or self.wasted:
            return 0.0
        return max(0.0, self.resource_needed_t - self.prefetch_end_t)

    @property
    def waste_s(self) -> float:
        """Load time that still blocked workflow (prefetch arrived too late)."""
        if self.prefetch_end_t is None or self.resource_needed_t is None:
            return 0.0
        if self.cancelled or self.wasted:
            return 0.0
        return max(0.0, self.prefetch_end_t - self.resource_needed_t)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overlap_s"] = self.overlap_s
        d["benefit_s"] = self.benefit_s
        d["waste_s"] = self.waste_s
        return d


# ---------------------------------------------------------------------------
# CSV writer for per-run timing records
# ---------------------------------------------------------------------------

_COLUMNS = [
    "run_id", "resource_id", "resource_name", "resource_type", "predictor_id",
    "checkpoint_id", "predicted_at_step", "consumed_at_step",
    "prefetch_start_t", "prefetch_end_t", "resource_needed_t", "baseline_load_s",
    "overlap_s", "benefit_s", "waste_s",
    "cancelled", "wasted", "hit",
]


def write_timing_csv(records: list[PrefetchTimingRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in records:
            w.writerow(r.to_dict())


def read_timing_csv(path: str) -> Iterator[PrefetchTimingRecord]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield PrefetchTimingRecord(
                run_id=row["run_id"],
                resource_id=row["resource_id"],
                resource_name=row["resource_name"],
                resource_type=row["resource_type"],
                predictor_id=row["predictor_id"],
                checkpoint_id=row["checkpoint_id"],
                predicted_at_step=int(row["predicted_at_step"]),
                consumed_at_step=int(row["consumed_at_step"]),
                prefetch_start_t=float(row["prefetch_start_t"]) if row["prefetch_start_t"] else None,
                prefetch_end_t=float(row["prefetch_end_t"]) if row["prefetch_end_t"] else None,
                resource_needed_t=float(row["resource_needed_t"]) if row["resource_needed_t"] else None,
                baseline_load_s=float(row["baseline_load_s"]) if row["baseline_load_s"] else None,
                cancelled=row["cancelled"].lower() == "true",
                wasted=row["wasted"].lower() == "true",
                hit=row["hit"].lower() == "true",
            )
