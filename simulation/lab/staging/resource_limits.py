"""Unified runtime, frame-count and evidence-byte budget enforcement."""
from __future__ import annotations

import time
import threading
from .config import ResourceLimits


class ResourceLimitExceeded(RuntimeError):
    pass


class ResourceBudget:
    """Single accounting authority for one staging run.

    Scenario execution should call `consume_frame` for every generated frame
    and `consume_evidence` for additional persisted evidence. The counters are
    cumulative and protected by a lock for concurrent producers.
    """
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.started = time.monotonic()
        self.frames = 0
        self.total_bytes = 0
        self._lock = threading.Lock()

    def check_time(self) -> None:
        if time.monotonic() - self.started > self.limits.max_runtime_seconds:
            raise ResourceLimitExceeded("staging runtime budget exceeded")

    def _validate_size(self, payload_size: int) -> None:
        if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size < 0:
            raise ValueError("payload_size must be a non-negative integer")
        if payload_size > self.limits.max_payload_bytes:
            raise ResourceLimitExceeded("per-payload byte budget exceeded")

    def consume_evidence(self, payload_size: int) -> None:
        """Account for non-frame evidence bytes atomically."""
        with self._lock:
            self.check_time()
            self._validate_size(payload_size)
            if self.total_bytes + payload_size > self.limits.max_total_bytes:
                raise ResourceLimitExceeded("total staging byte budget exceeded")
            self.total_bytes += payload_size

    def consume_bytes(self, payload_size: int) -> None:
        """Backward-compatible alias for additional evidence accounting."""
        self.consume_evidence(payload_size)

    def refund_evidence(self, payload_size: int) -> None:
        """Roll back a failed evidence write without racing other producers."""
        with self._lock:
            self.total_bytes = max(0, self.total_bytes - payload_size)

    def refund_frame(self, payload_size: int) -> None:
        """Roll back a failed frame write without racing other producers."""
        with self._lock:
            self.frames = max(0, self.frames - 1)
            self.total_bytes = max(0, self.total_bytes - payload_size)

    def consume_frame(self, payload_size: int) -> None:
        """Atomically account for one generated frame and its payload bytes."""
        with self._lock:
            self.check_time()
            self._validate_size(payload_size)
            if self.frames + 1 > self.limits.max_frames:
                raise ResourceLimitExceeded("frame-count budget exceeded")
            if self.total_bytes + payload_size > self.limits.max_total_bytes:
                raise ResourceLimitExceeded("total staging byte budget exceeded")
            self.frames += 1
            self.total_bytes += payload_size

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            self.check_time()
            return {
                "frames": self.frames,
                "total_bytes": self.total_bytes,
                "elapsed_seconds": time.monotonic() - self.started,
            }
