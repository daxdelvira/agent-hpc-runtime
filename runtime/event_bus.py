"""
event_bus.py — Runtime event bus, wraps WorkflowTracker to add typed runtime events.

The EventBus writes all runtime events to the same JSONL file as the
underlying WorkflowTracker so that prediction, prefetch, and divergence
events appear interleaved chronologically with agent/tool/LLM events.
This makes the trace self-contained for later analysis.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

from runtime.events import HpcEvent


class EventBus:
    """
    Append-only JSONL event bus for runtime events.

    Can either:
    - share a file with an existing WorkflowTracker (preferred: same JSONL),
    - or write to its own file (standalone mode).

    Thread-safe.
    """

    def __init__(
        self,
        run_id: str,
        log_path: str | None = None,       # if None, creates its own file
        shared_file=None,                  # pass an open file object to share
    ) -> None:
        self.run_id = run_id
        self._lock = threading.Lock()
        self._step = 0

        if shared_file is not None:
            self._file = shared_file
            self._owns_file = False
            self._log_path = getattr(shared_file, "name", "<shared>")
        else:
            if log_path is None:
                os.makedirs("logs/workflow_traces", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = f"logs/workflow_traces/runtime_trace_{ts}_{run_id}.jsonl"
            self._log_path = log_path
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._file = open(log_path, "a", buffering=1)
            self._owns_file = True

    @property
    def log_path(self) -> str:
        return self._log_path

    def set_step(self, step: int) -> None:
        with self._lock:
            self._step = step

    def emit(self, event_type: str, payload: dict, step: int | None = None) -> None:
        """Write one runtime event to the JSONL file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch_time": time.time(),
            "run_id": self.run_id,
            "step": step if step is not None else self._step,
            "event_type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass

    def emit_event(self, event: HpcEvent) -> None:
        """Write a pre-built HpcEvent."""
        self.emit(event.event_type, event.payload, step=event.step)

    def current_log_position(self) -> int:
        """Return the current byte offset in the log file (for checkpoint WAL records)."""
        try:
            return self._file.tell()
        except Exception:
            return 0

    def close(self) -> None:
        if self._owns_file and self._file:
            self._file.close()
