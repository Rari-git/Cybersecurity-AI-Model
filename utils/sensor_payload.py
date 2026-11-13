"""Canonical, shared validation for scalar IoT sensor payloads.

AdmissionGate and IDS intentionally import this single parser so accepted
telemetry cannot be interpreted differently downstream. The wire payloads are
plain scalar UTF-8 values, not arbitrary JSON documents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_SENSOR_PAYLOAD_BYTES = 512
TOPIC_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "home/sensors/gas/ppm": (0.0, 2000.0),
    "home/sensors/vibration/shock": (0.0, 5.0),
}
DISCRETE_TOPIC_VALUES: dict[str, frozenset[str]] = {
    "home/sensors/door/state": frozenset({"OPEN", "CLOSED"}),
    "home/sensors/pir/motion": frozenset({"MOTION_DETECTED", "NO_MOTION", "IDLE"}),
}
DISCRETE_NUMERIC_MAP = {
    "CLOSED": 0.0,
    "NO_MOTION": 0.0,
    "IDLE": 0.0,
    "OPEN": 1.0,
    "MOTION_DETECTED": 1.0,
}


@dataclass(frozen=True, slots=True)
class SensorPayloadResult:
    valid: bool
    canonical_text: str
    numeric_value: float | None
    reason: str


def parse_sensor_payload(
    topic: str,
    raw_payload: bytes | str,
    *,
    max_bytes: int = MAX_SENSOR_PAYLOAD_BYTES,
) -> SensorPayloadResult:
    """Validate and normalize one sensor scalar using the shared V5 schema."""
    if isinstance(raw_payload, bytes):
        if not raw_payload or len(raw_payload) > max_bytes:
            return SensorPayloadResult(False, "", None, "INVALID_PAYLOAD_SIZE")
        try:
            text = raw_payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return SensorPayloadResult(False, "", None, "INVALID_UTF8_PAYLOAD")
    elif isinstance(raw_payload, str):
        try:
            encoded = raw_payload.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return SensorPayloadResult(False, "", None, "INVALID_UTF8_PAYLOAD")
        if not encoded or len(encoded) > max_bytes:
            return SensorPayloadResult(False, "", None, "INVALID_PAYLOAD_SIZE")
        text = raw_payload
    else:
        return SensorPayloadResult(False, "", None, "UNSUPPORTED_PAYLOAD_TYPE")

    clean = text.strip()
    if not clean:
        return SensorPayloadResult(False, "", None, "EMPTY_PAYLOAD")

    allowed_discrete = DISCRETE_TOPIC_VALUES.get(topic)
    if allowed_discrete is not None:
        canonical = clean.upper()
        if canonical not in allowed_discrete:
            return SensorPayloadResult(False, canonical[:32], None, f"INVALID_DISCRETE_STATE:{canonical[:32]}")
        return SensorPayloadResult(True, canonical, DISCRETE_NUMERIC_MAP[canonical], "OK")

    bounds = TOPIC_PHYSICAL_BOUNDS.get(topic)
    if bounds is None:
        return SensorPayloadResult(False, clean[:32], None, f"UNREGISTERED_SENSOR_TOPIC:{topic}")

    try:
        value = float(clean)
    except (ValueError, OverflowError):
        return SensorPayloadResult(False, clean[:32], None, "NON_NUMERIC_PAYLOAD")
    if not math.isfinite(value):
        return SensorPayloadResult(False, clean[:32], None, "NAN_OR_INF_PAYLOAD")
    low, high = bounds
    if not low <= value <= high:
        return SensorPayloadResult(False, clean[:32], value, f"OUT_OF_PHYSICAL_BOUNDS:{value}")
    # Normalize numeric syntax for all downstream consumers ("001.0", "+1e0",
    # and "1" produce the same canonical scalar). Preserve enough precision for
    # a binary64 sensor measurement and normalize negative zero to zero.
    canonical_numeric = "0" if value == 0.0 else format(value, ".15g")
    return SensorPayloadResult(True, canonical_numeric, value, "OK")
