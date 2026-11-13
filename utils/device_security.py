"""Independent per-device token buckets and TTL quarantine; never blocks shared topics."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from utils.device_registry import DeviceRegistry, DeviceSpec
from utils.rate_limit import TokenBucketRateLimiter
from utils.ttl_blocklist import TTLBlocklist


@dataclass(frozen=True, slots=True)
class DeviceDecision:
    allowed: bool
    reason: str
    device_id: str
    attack_type: str = ""
    strike_count: int = 0
    block_issued: bool = False
    remaining_ttl: float = 0.0


class DeviceSecurityManager:
    def __init__(self, registry: DeviceRegistry, *, block_ttl_sec: Optional[float] = None):
        self.registry = registry
        self._specs = {spec.device_id: spec for spec in registry.enabled_devices()}
        if not self._specs:
            raise ValueError("no enabled devices configured")
        configured_ttls = [spec.block_ttl_sec for spec in self._specs.values()]
        ttl = max(configured_ttls) if block_ttl_sec is None else float(block_ttl_sec)
        if not math.isfinite(ttl) or not 1.0 <= ttl <= 3600.0:
            raise ValueError("block_ttl_sec must be between 1 and 3600 seconds")
        self.block_ttl_sec = ttl
        self.blocklist = TTLBlocklist(default_ttl=ttl, max_items=max(64, len(self._specs) * 4))
        self._limiters = {
            device_id: TokenBucketRateLimiter(rate=spec.rate_pps, burst=spec.burst, max_keys=1)
            for device_id, spec in self._specs.items()
        }
        self._strikes: dict[str, tuple[int, float]] = {}
        self._last_strike_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self.rate_drops = 0
        self.blocklist_drops = 0
        self.blocks_issued = 0
        self.unknown_devices = 0

    @staticmethod
    def _block_key(device_id: str) -> str:
        return f"device:{device_id}"

    def _known(self, device_id: str) -> DeviceSpec:
        spec = self._specs.get(device_id)
        if spec is None:
            with self._lock:
                self.unknown_devices += 1
            raise KeyError(f"unknown or disabled device: {device_id!r}")
        return spec

    def is_blocked(self, device_id: str) -> bool:
        if device_id not in self._specs:
            return False
        return self.blocklist.is_blocked(self._block_key(device_id))

    def check_rate(self, device_id: str, now: Optional[float] = None) -> DeviceDecision:
        spec = self._known(device_id)
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("now must be finite monotonic time")
        key = self._block_key(device_id)
        if self.blocklist.is_blocked(key):
            with self._lock:
                self.blocklist_drops += 1
            snap = self.blocklist.snapshot().get(key, {})
            return DeviceDecision(False, "device_blocked", device_id,
                                  remaining_ttl=float(snap.get("remaining_sec", 0.0)))
        if self._limiters[device_id].allow(device_id, current):
            return DeviceDecision(True, "allowed", device_id)
        with self._lock:
            self.rate_drops += 1
        decision = self.register_violation(device_id, "DoS / Topic Flooding", now=current)
        return DeviceDecision(False, "device_rate_limited", device_id,
                              attack_type="DoS / Topic Flooding",
                              strike_count=decision.strike_count,
                              block_issued=decision.block_issued,
                              remaining_ttl=decision.remaining_ttl)

    def register_violation(self, device_id: str, attack_type: str,
                           now: Optional[float] = None) -> DeviceDecision:
        spec = self._known(device_id)
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("now must be finite monotonic time")
        block_key = self._block_key(device_id)
        if self.blocklist.is_blocked(block_key):
            snap = self.blocklist.snapshot().get(block_key, {})
            with self._lock:
                self.blocklist_drops += 1
                count = self._strikes.get(device_id, (0, current))[0]
            return DeviceDecision(False, "device_blocked", device_id, attack_type,
                                  strike_count=count, remaining_ttl=float(snap.get("remaining_sec", 0.0)))

        with self._lock:
            # Recheck under the manager lock to close the race with another worker.
            if self.blocklist.is_blocked(block_key):
                snap = self.blocklist.snapshot().get(block_key, {})
                self.blocklist_drops += 1
                count = self._strikes.get(device_id, (0, current))[0]
                return DeviceDecision(False, "device_blocked", device_id, attack_type,
                                      strike_count=count, remaining_ttl=float(snap.get("remaining_sec", 0.0)))
            previous_time = self._last_strike_at.get(device_id)
            if previous_time is not None and current - previous_time < spec.strike_cooldown_sec:
                count = self._strikes.get(device_id, (0, current))[0]
                return DeviceDecision(False, "device_violation_deduplicated", device_id,
                                      attack_type, strike_count=count)
            count, start = self._strikes.get(device_id, (0, current))
            if current - start > spec.strike_window_sec:
                count, start = 0, current
            count += 1
            self._strikes[device_id] = (count, start)
            self._last_strike_at[device_id] = current
            if count < spec.strike_threshold:
                return DeviceDecision(False, "device_violation", device_id,
                                      attack_type, strike_count=count)
            remaining, is_new = self.blocklist.block_ex(
                block_key, ttl=spec.block_ttl_sec,
                meta={"device_id": device_id, "cert_cn": spec.cert_cn,
                      "attack": attack_type[:60], "source": "Inline Gateway"},
            )
            if is_new:
                self.blocks_issued += 1
            return DeviceDecision(False, "device_block_issued", device_id,
                                  attack_type, strike_count=count,
                                  block_issued=is_new, remaining_ttl=remaining)

    def active_blocks(self) -> dict:
        return self.blocklist.snapshot()

    def active_block_count(self) -> int:
        return len(self.blocklist)

    def clear_device(self, device_id: str) -> bool:
        if device_id not in self._specs:
            return False
        with self._lock:
            self._strikes.pop(device_id, None)
            self._last_strike_at.pop(device_id, None)
            self.blocklist.unblock(self._block_key(device_id))
        return True

    def clear_all(self) -> int:
        with self._lock:
            self._strikes.clear()
            self._last_strike_at.clear()
            return self.blocklist.clear()
