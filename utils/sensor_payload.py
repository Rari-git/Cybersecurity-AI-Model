"""Canonical scalar and V6 framed sensor-payload validation.

V5 scalar payloads remain supported unchanged. V6 wraps that same canonical
scalar in a small protocol frame that supplies device boot identity and a
monotonic sequence number. Sequence ordering is enforced transactionally by
AdmissionGate/storage, never by this stateless parser.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

MAX_SENSOR_PAYLOAD_BYTES = 512
MAX_SENSOR_FRAME_BYTES = 2048
MAX_SEQUENCE = 2**63 - 1
MAX_DEVICE_TIMESTAMP_NS = 2**63 - 1
_BOOT_ID_RE = re.compile(r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$")
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

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


@dataclass(frozen=True, slots=True)
class V6SensorFrame:
    protocol_version: int
    device_id: str
    boot_id: str
    sequence: int
    device_timestamp_ns: int
    canonical_text: str
    numeric_value: float


@dataclass(frozen=True, slots=True)
class SensorFrameResult:
    valid: bool
    frame: V6SensorFrame | None
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
    canonical_numeric = "0" if value == 0.0 else format(value, ".15g")
    return SensorPayloadResult(True, canonical_numeric, value, "OK")


def normalize_boot_id(value: Any) -> str:
    """Return a 128-bit boot identifier as 32 lowercase hexadecimal characters."""
    if not isinstance(value, str) or not _BOOT_ID_RE.fullmatch(value):
        raise ValueError("boot_id must be a 128-bit hexadecimal identifier")
    return value.replace("-", "").lower()


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_sensor_frame(
    topic: str,
    raw_frame: bytes | str,
    *,
    expected_device_id: str | None = None,
    max_bytes: int = MAX_SENSOR_FRAME_BYTES,
) -> SensorFrameResult:
    """Strictly validate a V6 frame; freshness ordering is checked by SQLite.

    Exact keys are required to prevent schema confusion. The payload remains a
    V5 scalar, so both the admission and IDS paths can reuse parse_sensor_payload.
    """
    if isinstance(raw_frame, bytes):
        if not raw_frame or len(raw_frame) > max_bytes:
            return SensorFrameResult(False, None, "INVALID_FRAME_SIZE")
        try:
            text = raw_frame.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return SensorFrameResult(False, None, "INVALID_FRAME_UTF8")
    elif isinstance(raw_frame, str):
        try:
            encoded = raw_frame.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return SensorFrameResult(False, None, "INVALID_FRAME_UTF8")
        if not encoded or len(encoded) > max_bytes:
            return SensorFrameResult(False, None, "INVALID_FRAME_SIZE")
        text = raw_frame
    else:
        return SensorFrameResult(False, None, "UNSUPPORTED_FRAME_TYPE")

    try:
        doc = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
        if not isinstance(doc, dict):
            raise ValueError("V6 frame root must be a JSON object")
        required = {
            "protocol_version", "device_id", "boot_id", "sequence",
            "device_timestamp_ns", "payload",
        }
        if set(doc) != required:
            missing = sorted(required - set(doc))
            extra = sorted(set(doc) - required)
            raise ValueError(f"invalid V6 frame keys (missing={missing}, extra={extra})")
        if (isinstance(doc["protocol_version"], bool)
                or not isinstance(doc["protocol_version"], int)
                or doc["protocol_version"] != 6):
            raise ValueError("unsupported sensor protocol version")
        device_id = doc["device_id"]
        if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
            raise ValueError("invalid frame device_id")
        if expected_device_id is not None and device_id != expected_device_id:
            raise ValueError("frame device_id does not match authenticated certificate identity")
        boot_id = normalize_boot_id(doc["boot_id"])
        sequence = doc["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= MAX_SEQUENCE:
            raise ValueError("sequence must be a positive signed 64-bit integer")
        device_timestamp_ns = doc["device_timestamp_ns"]
        if (isinstance(device_timestamp_ns, bool) or not isinstance(device_timestamp_ns, int)
                or not 1 <= device_timestamp_ns <= MAX_DEVICE_TIMESTAMP_NS):
            raise ValueError("device_timestamp_ns must be a positive signed 64-bit integer")

        payload = doc["payload"]
        if isinstance(payload, bool) or payload is None or isinstance(payload, (dict, list)):
            raise ValueError("sensor payload must be a scalar string or finite number")
        if isinstance(payload, float) and not math.isfinite(payload):
            raise ValueError("sensor payload must be finite")
        if isinstance(payload, (int, float)):
            payload_text = str(payload)
        elif isinstance(payload, str):
            payload_text = payload
        else:
            raise ValueError("unsupported sensor payload type")
        parsed = parse_sensor_payload(topic, payload_text)
        if not parsed.valid:
            raise ValueError(parsed.reason)
        return SensorFrameResult(
            True,
            V6SensorFrame(
                protocol_version=6,
                device_id=device_id,
                boot_id=boot_id,
                sequence=sequence,
                device_timestamp_ns=device_timestamp_ns,
                canonical_text=parsed.canonical_text,
                numeric_value=float(parsed.numeric_value),
            ),
            "OK",
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return SensorFrameResult(False, None, str(exc)[:160])



def encode_sensor_frame(
    *,
    device_id: str,
    boot_id: str,
    sequence: int,
    device_timestamp_ns: int,
    topic: str,
    payload: bytes | str | int | float,
) -> bytes:
    """Encode a canonical V6 sensor frame using the existing V5 schema."""

    if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
        raise ValueError("invalid frame device_id")

    canonical_boot_id = normalize_boot_id(boot_id)

    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 1 <= sequence <= MAX_SEQUENCE
    ):
        raise ValueError("sequence must be a positive signed 64-bit integer")

    if (
        isinstance(device_timestamp_ns, bool)
        or not isinstance(device_timestamp_ns, int)
        or not 1 <= device_timestamp_ns <= MAX_DEVICE_TIMESTAMP_NS
    ):
        raise ValueError("device_timestamp_ns must be a positive signed 64-bit integer")

    if isinstance(payload, bytes):
        try:
            payload_text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("INVALID_UTF8_PAYLOAD") from exc
    elif isinstance(payload, bool) or not isinstance(payload, (str, int, float)):
        raise ValueError("unsupported sensor payload type")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError("sensor payload must be finite")
    else:
        payload_text = str(payload)

    parsed = parse_sensor_payload(topic, payload_text)
    if not parsed.valid:
        raise ValueError(parsed.reason)

    document = {
        "protocol_version": 6,
        "device_id": device_id,
        "boot_id": canonical_boot_id,
        "sequence": sequence,
        "device_timestamp_ns": device_timestamp_ns,
        "payload": parsed.canonical_text,
    }

    encoded = json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    if len(encoded) > MAX_SENSOR_FRAME_BYTES:
        raise ValueError("INVALID_FRAME_SIZE")

    return encoded
