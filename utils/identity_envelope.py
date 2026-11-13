"""Authenticated Gateway-to-IDS telemetry identity envelope.

V5 envelopes remain wire-compatible. V6 envelopes additionally carry a device
boot epoch and monotonic sequence, and an RSA-PSS signature over the complete canonical
message bound to the currently active Gateway certificate identity/generation.
The existing mTLS + Gateway-only Zenoh ACL remains the transport trust boundary;
the V6 signature provides message-level integrity and explicit certificate binding.
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
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from utils.sensor_payload import normalize_boot_id, MAX_SEQUENCE, MAX_DEVICE_TIMESTAMP_NS

_DEVICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOPIC = re.compile(r"^[A-Za-z0-9_./-]{1,256}$")
_V6_SIGNATURE_DOMAIN = b"LUMEN-IDENTITY-ENVELOPE-V6\x00"


def _validate_gateway_signer(identity: Any, private_key: Any) -> None:
    if getattr(identity, "device_id", None) != "gateway":
        raise ValueError("V6 message signer must be the active Gateway identity")
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError("V6 Gateway signer must be an RSA private key from the active mTLS certificate")
    certificate_public_key = getattr(identity, "certificate_public_key", None)
    if not isinstance(certificate_public_key, rsa.RSAPublicKey):
        raise ValueError("active Gateway certificate RSA public key is unavailable")
    private_public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    certificate_public_der = certificate_public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if not hmac.compare_digest(private_public_der, certificate_public_der):
        raise ValueError("V6 Gateway signing key does not match active Gateway certificate")


def _sign_v6(private_key: Any, message: bytes) -> str:
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ValueError("V6 Gateway signer must be an RSA private key from the active mTLS certificate")
    signature = private_key.sign(
        _V6_SIGNATURE_DOMAIN + message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return signature.hex()


def _gateway_claims(identity: Any) -> tuple[str, int, str]:
    try:
        gateway_cn = str(identity.common_name)
        gateway_generation = identity.generation
        gateway_fingerprint = bytes(identity.certificate_fingerprint)
    except Exception as exc:
        raise ValueError("active Gateway certificate identity is required for V6 envelope") from exc
    if not _DEVICE_ID.fullmatch(gateway_cn):
        raise ValueError("invalid active Gateway certificate common name")
    if isinstance(gateway_generation, bool) or not isinstance(gateway_generation, int) or gateway_generation < 1:
        raise ValueError("invalid active Gateway certificate generation")
    if len(gateway_fingerprint) != 32:
        raise ValueError("active Gateway certificate fingerprint must be SHA-256")
    return gateway_cn, gateway_generation, gateway_fingerprint.hex()


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
    # V6 sensor protocol metadata. None denotes the backward-compatible V5 path.
    protocol_version: int = 5
    boot_id: str | None = None
    sequence: int | None = None
    device_timestamp_ns: int | None = None
    # Gateway certificate binding and message integrity tag (mandatory for V6).
    gateway_certificate_common_name: str | None = None
    gateway_certificate_generation: int | None = None
    gateway_certificate_fingerprint_sha256: str | None = None
    gateway_auth_algorithm: str | None = None
    gateway_signature: str | None = None

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
        protocol_version: int = 5,
        boot_id: str | None = None,
        sequence: int | None = None,
        device_timestamp_ns: int | None = None,
        gateway_identity: Any | None = None,
        gateway_signing_key: Any | None = None,
    ) -> "AuthenticatedTelemetryEnvelope":
        if not _DEVICE_ID.fullmatch(device_id):
            raise ValueError("invalid authenticated device_id")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not _DEVICE_ID.fullmatch(certificate_common_name):
            raise ValueError("invalid certificate common name")
        if not isinstance(certificate_fingerprint, bytes) or len(certificate_fingerprint) != 32:
            raise ValueError("certificate fingerprint must be the real SHA-256 digest bytes")
        if not _TOPIC.fullmatch(original_topic) or ".." in original_topic.split("/"):
            raise ValueError("invalid original topic")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes")
        if not _DEVICE_ID.fullmatch(gateway_instance_id):
            raise ValueError("invalid gateway instance id")
        if isinstance(protocol_version, bool) or not isinstance(protocol_version, int) or protocol_version not in (5, 6):
            raise ValueError("unsupported sensor protocol version")
        ts = time.time_ns() if event_timestamp_ns is None else event_timestamp_ns
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
            raise ValueError("event timestamp must be a positive integer")
        if admission_status not in {"ADMITTED", "REJECTED"}:
            raise ValueError("invalid admission status")
        if not isinstance(admission_reason, str) or len(admission_reason) > 256:
            raise ValueError("invalid admission reason")
        if admission_numeric_value is not None and (
            isinstance(admission_numeric_value, bool)
            or not isinstance(admission_numeric_value, (int, float))
            or not math.isfinite(float(admission_numeric_value))
        ):
            raise ValueError("invalid admission numeric value")

        v6_fields = (boot_id, sequence, device_timestamp_ns)
        gateway_fields = (gateway_identity, gateway_signing_key)
        gateway_claim_values = (None, None, None)
        if protocol_version == 6:
            # A malformed V6 frame can still be forwarded as a signed rejection
            # event so IDS receives structured protocol evidence. Only rejected
            # envelopes may omit all freshness fields; partial metadata is never
            # accepted and an admitted V6 message always requires all fields.
            if any(value is None for value in v6_fields):
                if admission_status != "REJECTED" or any(value is not None for value in v6_fields):
                    raise ValueError("V6 envelope requires boot_id, sequence and device_timestamp_ns")
            else:
                boot_id = normalize_boot_id(boot_id)
                if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= MAX_SEQUENCE:
                    raise ValueError("sequence must be a positive signed 64-bit integer")
                if (isinstance(device_timestamp_ns, bool) or not isinstance(device_timestamp_ns, int)
                        or not 1 <= device_timestamp_ns <= MAX_DEVICE_TIMESTAMP_NS):
                    raise ValueError("device_timestamp_ns must be a positive signed 64-bit integer")
            if any(value is None for value in gateway_fields):
                raise ValueError("V6 envelope requires active Gateway identity and signing key")
            gateway_claim_values = _gateway_claims(gateway_identity)
            _validate_gateway_signer(gateway_identity, gateway_signing_key)
        else:
            if any(value is not None for value in v6_fields) or any(value is not None for value in gateway_fields):
                raise ValueError("V5 envelope cannot carry V6 freshness/authentication fields")

        env = cls(
            device_id=device_id,
            generation=generation,
            certificate_common_name=certificate_common_name,
            certificate_fingerprint_sha256=certificate_fingerprint.hex(),
            original_topic=original_topic,
            payload_b64=base64.b64encode(payload).decode("ascii"),
            event_timestamp_ns=ts,
            message_id=message_id or uuid.uuid4().hex,
            gateway_instance_id=gateway_instance_id,
            admission_status=admission_status,
            admission_reason=admission_reason,
            admission_numeric_value=(float(admission_numeric_value) if admission_numeric_value is not None else None),
            protocol_version=protocol_version,
            boot_id=boot_id,
            sequence=sequence,
            device_timestamp_ns=device_timestamp_ns,
            gateway_certificate_common_name=gateway_claim_values[0],
            gateway_certificate_generation=gateway_claim_values[1],
            gateway_certificate_fingerprint_sha256=gateway_claim_values[2],
            gateway_auth_algorithm=("rsa-pss-sha256" if protocol_version == 6 else None),
        )
        if protocol_version == 6:
            signature = _sign_v6(gateway_signing_key, env._canonical_auth_bytes())
            object.__setattr__(env, "gateway_signature", signature)
        return env

    def _canonical_auth_bytes(self) -> bytes:
        """Canonical bytes covering identity, gateway cert, freshness and payload claims."""
        document = asdict(self)
        document.pop("gateway_signature", None)
        return json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> tuple["AuthenticatedTelemetryEnvelope", bytes]:
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid identity envelope JSON") from exc
        if not isinstance(doc, dict) or doc.get("version") != 1 or doc.get("identity_source") != "gateway_mtls_acl":
            raise ValueError("unsupported or untrusted telemetry envelope")
        # Defaults keep existing V5 serialized envelopes wire-compatible.
        doc.setdefault("admission_status", "ADMITTED")
        doc.setdefault("admission_reason", "LEGACY_UNVERIFIED")
        doc.setdefault("admission_numeric_value", None)
        doc.setdefault("protocol_version", 5)
        doc.setdefault("boot_id", None)
        doc.setdefault("sequence", None)
        doc.setdefault("device_timestamp_ns", None)
        doc.setdefault("gateway_certificate_common_name", None)
        doc.setdefault("gateway_certificate_generation", None)
        doc.setdefault("gateway_certificate_fingerprint_sha256", None)
        doc.setdefault("gateway_auth_algorithm", None)
        doc.setdefault("gateway_signature", None)
        try:
            env = cls(**doc)
        except TypeError as exc:
            raise ValueError("invalid identity envelope fields") from exc
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
        if isinstance(env.generation, bool) or not isinstance(env.generation, int) or env.generation < 1:
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
        try:
            payload = base64.b64decode(env.payload_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid envelope payload encoding") from exc
        if not payload:
            raise ValueError("empty envelope payload")
        if isinstance(env.protocol_version, bool) or not isinstance(env.protocol_version, int):
            raise ValueError("invalid envelope protocol version")
        if env.protocol_version == 6:
            v6_values = (env.boot_id, env.sequence, env.device_timestamp_ns)
            if any(value is None for value in v6_values):
                if env.admission_status != "REJECTED" or any(value is not None for value in v6_values):
                    raise ValueError("incomplete V6 freshness metadata")
            else:
                try:
                    normalize_boot_id(env.boot_id)
                except ValueError as exc:
                    raise ValueError("invalid V6 boot ID") from exc
                if isinstance(env.sequence, bool) or not isinstance(env.sequence, int) or not 1 <= env.sequence <= MAX_SEQUENCE:
                    raise ValueError("invalid V6 sequence")
                if (isinstance(env.device_timestamp_ns, bool) or not isinstance(env.device_timestamp_ns, int)
                        or not 1 <= env.device_timestamp_ns <= MAX_DEVICE_TIMESTAMP_NS):
                    raise ValueError("invalid V6 device timestamp")
            if (
                not isinstance(env.gateway_certificate_common_name, str)
                or not _DEVICE_ID.fullmatch(env.gateway_certificate_common_name)
                or isinstance(env.gateway_certificate_generation, bool)
                or not isinstance(env.gateway_certificate_generation, int)
                or env.gateway_certificate_generation < 1
                or not isinstance(env.gateway_certificate_fingerprint_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", env.gateway_certificate_fingerprint_sha256)
                or env.gateway_auth_algorithm != "rsa-pss-sha256"
                or not isinstance(env.gateway_signature, str)
                or not re.fullmatch(r"[0-9a-f]{512}", env.gateway_signature)
            ):
                raise ValueError("incomplete V6 Gateway authentication binding")
        elif env.protocol_version != 5:
            raise ValueError("unsupported sensor protocol version")
        elif any((env.boot_id, env.sequence, env.device_timestamp_ns,
                  env.gateway_certificate_common_name, env.gateway_certificate_generation,
                  env.gateway_certificate_fingerprint_sha256, env.gateway_auth_algorithm,
                  env.gateway_signature)):
            raise ValueError("V5 envelope cannot contain V6 authentication fields")
        return env, payload


def validate_active_identity(envelope: AuthenticatedTelemetryEnvelope, active_identity) -> None:
    """Raise when sensor claims do not match IDS-side active PKI identity."""
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


def validate_gateway_binding(
    envelope: AuthenticatedTelemetryEnvelope,
    active_gateway_identity: Any,
) -> None:
    """Verify the V6 signature with the active Gateway certificate public key."""
    if envelope.protocol_version != 6:
        raise ValueError("Gateway cryptographic binding is required only for V6 envelopes")
    expected_cn, expected_generation, expected_fingerprint = _gateway_claims(active_gateway_identity)
    if getattr(active_gateway_identity, "device_id", None) != "gateway":
        raise ValueError("V6 envelope validation requires the active Gateway identity")
    if (
        envelope.gateway_certificate_common_name != expected_cn
        or envelope.gateway_certificate_generation != expected_generation
        or envelope.gateway_certificate_fingerprint_sha256 != expected_fingerprint
    ):
        raise ValueError("V6 envelope Gateway certificate identity is stale or forged")
    public_key = getattr(active_gateway_identity, "certificate_public_key", None)
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("active Gateway RSA certificate public key is unavailable")
    try:
        public_key.verify(
            bytes.fromhex(envelope.gateway_signature or ""),
            _V6_SIGNATURE_DOMAIN + envelope._canonical_auth_bytes(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("V6 envelope Gateway signature verification failed") from exc


def validate_envelope_route(
    envelope: AuthenticatedTelemetryEnvelope, key: str, allowed_topics_by_device: dict[str, str],
) -> None:
    """Ensure the internal key and original sensor topic are bound to the registry ID."""
    if key != f"internal/ids/telemetry/{envelope.device_id}":
        raise ValueError("internal envelope key/device_id mismatch")
    if allowed_topics_by_device.get(envelope.device_id) != envelope.original_topic:
        raise ValueError("envelope device/topic is not registered")


IdentityEnvelope = AuthenticatedTelemetryEnvelope
