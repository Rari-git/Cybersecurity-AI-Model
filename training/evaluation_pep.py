"""Deterministic, non-always-successful PEP simulator for offline benchmarks."""
from __future__ import annotations

import time


class SimulatedPep:
    def __init__(self, *, fail_decisions: set[str] | None = None, timeout_decisions: set[str] | None = None,
                 simulated_latency_ms: float = 5.0):
        self.fail_decisions = set(fail_decisions or ())
        self.timeout_decisions = set(timeout_decisions or ())
        self.simulated_latency_ms = max(0.0, float(simulated_latency_ms))
        self.history: list[tuple[str, bool, float]] = []

    def apply(self, device_id: str, enable: bool, decision_id: str | None = None) -> bool:
        decision_id = decision_id or f"untracked:{device_id}:{len(self.history)}"
        started = time.perf_counter()
        if decision_id in self.timeout_decisions:
            # Simulated timeout is an observable failure, not successful publication.
            self.history.append((decision_id, False, (time.perf_counter() - started) * 1000.0))
            return False
        if self.simulated_latency_ms:
            time.sleep(self.simulated_latency_ms / 1000.0)
        ok = decision_id not in self.fail_decisions
        self.history.append((decision_id, ok, (time.perf_counter() - started) * 1000.0))
        return ok
