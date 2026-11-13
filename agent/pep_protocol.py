"""Correlated PEP request/ack helper used by synchronous policy decisions."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PepAcknowledgement:
    decision_id: str
    device_id: str
    requested_action: str
    applied_action: str
    result: str
    timestamp_ns: int
    gateway_instance_id: str

    @property
    def confirmed(self) -> bool:
        return self.result == "CONFIRMED" and self.applied_action == self.requested_action


class PepAckRegistry:
    """Tracks outstanding PEP commands, rejects unknown/mismatched/duplicate ACKs."""

    def __init__(self):
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, Any]] = {}

    def register(self, decision_id: str, device_id: str, requested_action: str) -> None:
        if not all(isinstance(v, str) and v for v in (decision_id, device_id, requested_action)):
            raise ValueError("decision_id, device_id and requested_action are required")
        with self._lock:
            if decision_id in self._pending:
                raise ValueError("duplicate outstanding decision_id")
            self._pending[decision_id] = {
                "device_id": device_id,
                "requested_action": requested_action,
                "event": threading.Event(),
                "ack": None,
            }

    def accept(self, ack: PepAcknowledgement) -> bool:
        with self._lock:
            item = self._pending.get(ack.decision_id)
            if item is None or item["ack"] is not None:
                return False
            if item["device_id"] != ack.device_id or item["requested_action"] != ack.requested_action:
                return False
            if ack.result not in {"CONFIRMED", "FAILED"} or ack.timestamp_ns <= 0 or not ack.gateway_instance_id:
                return False
            item["ack"] = ack
            item["event"].set()
            return True

    def wait(self, decision_id: str, timeout_sec: float) -> PepAcknowledgement | None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        with self._lock:
            item = self._pending.get(decision_id)
            if item is None:
                raise KeyError(decision_id)
            event = item["event"]
        event.wait(timeout_sec)
        with self._lock:
            item = self._pending.pop(decision_id, None)
            return None if item is None else item["ack"]

    def cancel(self, decision_id: str) -> None:
        with self._lock:
            self._pending.pop(decision_id, None)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
