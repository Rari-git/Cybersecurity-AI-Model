"""Sensor publication compatibility path for the current Lumen Zenoh runtime.

The project's direct-session diagnostic demonstrated that Session.put() is
visible to both exact and wildcard gateway subscribers, while Publisher.put()
was not visible in the same TLS/ACL setup. Until the installed Zenoh version's
Publisher matching/reliability behavior is independently resolved, sensor
producers use the demonstrated direct Session.put() API.
"""
from __future__ import annotations

from typing import Any


ALLOWED_SENSOR_TOPICS = frozenset({
    "ingest/sensors/door/state",
    "ingest/sensors/gas/ppm",
    "ingest/sensors/vibration/shock",
    "ingest/sensors/pir/motion",
    "home/sensors/door/state",
    "home/sensors/gas/ppm",
    "home/sensors/vibration/shock",
    "home/sensors/pir/motion",
})


def publish_sensor_sample(session: Any, topic: str, payload: str | bytes) -> None:
    """Publish one validated sensor sample using Session.put().

    This function intentionally does not report delivery success: Session.put()
    returning means the local Zenoh API accepted the operation, not that a
    remote gateway callback processed it. Gateway JSONL metrics remain the
    authority for end-to-end delivery observation.
    """
    if topic not in ALLOWED_SENSOR_TOPICS:
        raise ValueError(f"sensor topic is not allow-listed: {topic!r}")
    if isinstance(payload, str):
        raw = payload.encode("utf-8", errors="strict")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise TypeError("sensor payload must be str or bytes")
    if not raw or len(raw) > 512:
        raise ValueError("sensor payload must contain 1..512 bytes")
    session.put(topic, raw)
