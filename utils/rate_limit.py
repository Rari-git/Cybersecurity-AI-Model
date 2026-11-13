"""Thread-safe token-bucket rate limiter with a strict memory bound."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Hashable


class TokenBucketRateLimiter:
    """O(1) token bucket per key, with LRU eviction and bounded memory.

    Pass ``time.monotonic()`` as ``now``. If a caller accidentally supplies a
    timestamp older than the previous timestamp for that bucket, time is
    clamped to the previous value so tokens cannot be gained by clock rollback.

    ``rate=0`` is supported intentionally: the bucket receives its initial
    burst but never replenishes. This is useful for tests and for configuring
    a one-shot alert budget. Negative rates are invalid.
    """

    def __init__(self, rate: float = 30.0, burst: float = 60.0, max_keys: int = 4096):
        try:
            rate = float(rate)
            burst = float(burst)
            max_keys_int = int(max_keys)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rate, burst and max_keys must be numeric") from exc

        if not math.isfinite(rate) or rate < 0:
            raise ValueError("rate must be a finite number greater than or equal to zero")
        if not math.isfinite(burst) or burst < 1:
            raise ValueError("burst must be a finite number greater than or equal to one")
        if isinstance(max_keys, bool) or max_keys_int != max_keys or max_keys_int < 1:
            raise ValueError("max_keys must be a positive integer")

        self.rate = rate
        self.burst = burst
        self.max_keys = max_keys_int
        self.buckets: OrderedDict[Hashable, list[float]] = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, key: Hashable, now: float) -> bool:
        """Consume one token and return True, or return False when depleted."""
        try:
            current = float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("now must be a finite monotonic timestamp") from exc
        if not math.isfinite(current):
            raise ValueError("now must be a finite monotonic timestamp")

        with self.lock:
            bucket = self.buckets.get(key)
            if bucket is None:
                if len(self.buckets) >= self.max_keys:
                    self.buckets.popitem(last=False)
                bucket = [self.burst, current]
                self.buckets[key] = bucket
            else:
                self.buckets.move_to_end(key)

            tokens, last_time = bucket
            effective_now = max(current, last_time)
            elapsed = effective_now - last_time
            tokens = min(self.burst, tokens + elapsed * self.rate)
            bucket[1] = effective_now

            if tokens >= 1.0:
                bucket[0] = tokens - 1.0
                return True

            bucket[0] = tokens
            return False
