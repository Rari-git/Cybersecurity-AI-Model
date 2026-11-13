"""Device-aware ingress policy composing registry, device, topic, and global limits."""
from __future__ import annotations

from dataclasses import dataclass

from utils.device_registry import DeviceRegistry
from utils.device_security import DeviceSecurityManager
from utils.rate_limit import TokenBucketRateLimiter


@dataclass(frozen=True)
class IngressDecision:
    accepted: bool
    reason: str
    device_id: str | None = None
    target_topic: str | None = None
    block_issued: bool = False
    strike_count: int = 0
    remaining_ttl: float = 0.0
    attack_type: str = "Device Rate Abuse"


class DeviceIngressPolicy:
    """Only valid per-device namespaces reach the existing home sensor topics.

    Apply device limits before shared topic/global limits. Shared limits only
    drop traffic; they do not automatically quarantine every device because an
    aggregate overload alone does not prove which device is malicious.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        security: DeviceSecurityManager,
        topic_limiter: TokenBucketRateLimiter,
        global_limiter: TokenBucketRateLimiter,
    ):
        self.registry = registry
        self.security = security
        self.topic_limiter = topic_limiter
        self.global_limiter = global_limiter

    def evaluate(self, key_expr: str, now: float) -> IngressDecision:
        route = self.registry.resolve_ingress(key_expr)
        if route is None:
            return IngressDecision(False, "unknown_or_unauthorized_device_topic")
        spec, target_topic = route

        device_decision = self.security.check_rate(spec.device_id, now)
        if not device_decision.allowed:
            return IngressDecision(
                False,
                device_decision.reason,
                device_id=spec.device_id,
                target_topic=target_topic,
                block_issued=device_decision.block_issued,
                strike_count=device_decision.strike_count,
                remaining_ttl=device_decision.remaining_ttl,
                attack_type=device_decision.attack_type,
            )
        if not self.topic_limiter.allow(target_topic, now):
            return IngressDecision(False, "topic_rate_limited", spec.device_id, target_topic)
        if not self.global_limiter.allow("all-sensors", now):
            return IngressDecision(False, "global_rate_limited", spec.device_id, target_topic)
        return IngressDecision(True, "accepted", spec.device_id, target_topic)
