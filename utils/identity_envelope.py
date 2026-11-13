"""Authenticated Gateway-to-IDS telemetry identity envelope.

The Gateway constructs this envelope only after the router ACL has authorized a
per-device ingress key. Payload identity fields are never trusted as authority.
"""
from __future__ import annotations

import base64
import hmac
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass

_DEVICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOPIC = re.compile(r"^[A-Za-z0-9_./-]{1,256}$")


@dataclass(frozen=True, slots=True)
class AuthenticatedTelemetryEnvelope:
    device_id: str
    generation: int
    certificate_common_name: str
    certificate_fingerprint_sha256: str
    original_topic: str
    payload_b64: str
    event_timestamp_ns: int
    message_id: str
    gateway_instance_id: str
    identity_source: str = "gateway_mtls_acl"
    version: int = 1
    admission_status: str = "ADMITTED"
    admission_reason: str = "LEGACY_UNVERIFIED"
    admission_numeric_value: float | None = None

    @classmethod
    def create(
        cls,
        *,
        device_id: str,
        generation: int,
        certificate_common_name: str,
        certificate_fingerprint: bytes,
        original_topic: str,
        payload: bytes,
        gateway_instance_id: str,
        event_timestamp_ns: int | None = None,
        message_id: str | None = None,
        admission_status: str = "ADMITTED",
        admission_reason: str = "LEGACY_UNVERIFIED",
        admission_numeric_value: float | None = None,
    ) -> "AuthenticatedTelemetryEnvelope":
        if not _DEVICE_ID.fullmatch(device_id):
            raise ValueError("invalid authenticated device_id")
        if isinstance(generation, bool) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not _DEVICE_ID.fullmatch(certificate_common_name):
            raise ValueError("invalid certificate common name")
        if len(certificate_fingerprint) != 32:
            raise ValueError("certificate fingerprint must be the real SHA-256 digest bytes")
        if not _TOPIC.fullmatch(original_topic) or ".." in original_topic.split("/"):
            raise ValueError("invalid original topic")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes")
        if not _DEVICE_ID.fullmatch(gateway_instance_id):
            raise ValueError("invalid gateway instance id")
        ts = time.time_ns() if event_timestamp_ns is None else int(event_timestamp_ns)
        if ts <= 0:
            raise ValueError("event timestamp must be positive")
        return cls(
            device_id=device_id,
            generation=int(generation),
            certificate_common_name=certificate_common_name,
            certificate_fingerprint_sha256=certificate_fingerprint.hex(),
            original_topic=original_topic,
            payload_b64=base64.b64encode(payload).decode("ascii"),
            event_timestamp_ns=ts,
            message_id=message_id or uuid.uuid4().hex,
            gateway_instance_id=gateway_instance_id,
            admission_status=admission_status,
            admission_reason=admission_reason,
            admission_numeric_value=admission_numeric_value,
        )

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> tuple["AuthenticatedTelemetryEnvelope", bytes]:
        doc = json.loads(raw.decode("utf-8"))
        if not isinstance(doc, dict) or doc.get("version") != 1 or doc.get("identity_source") != "gateway_mtls_acl":
            raise ValueError("unsupported or untrusted telemetry envelope")
        doc.setdefault("admission_status", "ADMITTED")
        doc.setdefault("admission_reason", "LEGACY_UNVERIFIED")
        doc.setdefault("admission_numeric_value", None)
        env = cls(**doc)
        if env.admission_status not in {"ADMITTED", "REJECTED"}:
            raise ValueError("invalid admission status in identity envelope")
        if not isinstance(env.admission_reason, str) or len(env.admission_reason) > 256:
            raise ValueError("invalid admission reason in identity envelope")
        if env.admission_numeric_value is not None and (
            isinstance(env.admission_numeric_value, bool)
            or not isinstance(env.admission_numeric_value, (int, float))
            or not math.isfinite(float(env.admission_numeric_value))
        ):
            raise ValueError("invalid admission numeric value in identity envelope")
        if not _DEVICE_ID.fullmatch(env.device_id) or not _DEVICE_ID.fullmatch(env.certificate_common_name):
            raise ValueError("invalid envelope identity")
        if isinstance(env.generation, bool) or env.generation < 1:
            raise ValueError("invalid envelope generation")
        if not re.fullmatch(r"[0-9a-f]{64}", env.certificate_fingerprint_sha256):
            raise ValueError("invalid certificate fingerprint")
        if not _TOPIC.fullmatch(env.original_topic) or ".." in env.original_topic.split("/"):
            raise ValueError("invalid envelope topic")
        if (
            isinstance(env.event_timestamp_ns, bool)
            or not isinstance(env.event_timestamp_ns, int)
            or env.event_timestamp_ns <= 0
            or not isinstance(env.message_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", env.message_id)
            or not isinstance(env.gateway_instance_id, str)
            or not _DEVICE_ID.fullmatch(env.gateway_instance_id)
        ):
            raise ValueError("invalid envelope timestamp/message/gateway identity")
        payload = base64.b64decode(env.payload_b64, validate=True)
        if not payload:
            raise ValueError("empty envelope payload")
        return env, payload


def validate_active_identity(
    envelope: AuthenticatedTelemetryEnvelope, active_identity,
) -> None:
    """Raise when an envelope does not exactly match the IDS-side active PKI identity."""
    if (
        envelope.device_id != active_identity.device_id
        or envelope.generation != active_identity.generation
        or envelope.certificate_common_name != active_identity.common_name
        or not hmac.compare_digest(
            bytes.fromhex(envelope.certificate_fingerprint_sha256),
            active_identity.certificate_fingerprint,
        )
    ):
        raise ValueError("forged or stale active certificate identity envelope")


def validate_envelope_route(
    envelope: AuthenticatedTelemetryEnvelope, key: str, allowed_topics_by_device: dict[str, str],
) -> None:
    """Ensure the internal key and original sensor topic are bound to the same registry ID."""
    if key != f"internal/ids/telemetry/{envelope.device_id}":
        raise ValueError("internal envelope key/device_id mismatch")
    if allowed_topics_by_device.get(envelope.device_id) != envelope.original_topic:
        raise ValueError("envelope device/topic is not registered")

# Friendly public name for callers that refer to the authenticated envelope
# generically; keep the original class name as a backward-compatible alias.
IdentityEnvelope = AuthenticatedTelemetryEnvelope
