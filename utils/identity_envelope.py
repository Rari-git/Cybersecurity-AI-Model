"""Authenticated Gateway-to-IDS telemetry identity envelope.

The Gateway constructs this envelope only after the router ACL has authorized a
per-device ingress key. Payload identity fields are never trusted as authority.
"""
from __future__ import annotations

import base64
import hashlib
import json
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
        )

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> tuple["AuthenticatedTelemetryEnvelope", bytes]:
        doc = json.loads(raw.decode("utf-8"))
        if not isinstance(doc, dict) or doc.get("version") != 1 or doc.get("identity_source") != "gateway_mtls_acl":
            raise ValueError("unsupported or untrusted telemetry envelope")
        env = cls(**doc)
        if not _DEVICE_ID.fullmatch(env.device_id) or not _DEVICE_ID.fullmatch(env.certificate_common_name):
            raise ValueError("invalid envelope identity")
        if isinstance(env.generation, bool) or env.generation < 1:
            raise ValueError("invalid envelope generation")
        if not re.fullmatch(r"[0-9a-f]{64}", env.certificate_fingerprint_sha256):
            raise ValueError("invalid certificate fingerprint")
        if not _TOPIC.fullmatch(env.original_topic) or ".." in env.original_topic.split("/"):
            raise ValueError("invalid envelope topic")
        if env.event_timestamp_ns <= 0 or not env.message_id:
            raise ValueError("invalid envelope timestamp/message id")
        payload = base64.b64decode(env.payload_b64, validate=True)
        if not payload:
            raise ValueError("empty envelope payload")
        return env, payload
