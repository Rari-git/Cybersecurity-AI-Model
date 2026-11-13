"""Authenticated, single-use V6 sensor boot-registration challenge protocol.

The boot ID is not trusted merely because it appears in a telemetry frame.
A device must prove possession of its active leaf private key over a fresh,
Gateway-signed challenge before AdmissionGate is allowed to activate a boot epoch.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from utils.cert_lifecycle import CertRevocationLedger
from utils.device_identity import (
    ActiveDeviceIdentity,
    load_active_device_identity,
    load_active_device_signing_key,
)
from utils.sensor_payload import normalize_boot_id
from storage.admission_gate import ProvenanceContext

_CHALLENGE_DOMAIN = b"LUMEN-V6-BOOT-CHALLENGE\x00"
_PROOF_DOMAIN = b"LUMEN-V6-BOOT-REGISTRATION\x00"
_HEX_128 = re.compile(r"^[0-9a-f]{32}$")
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

BOOT_REQUEST_PREFIX = "control/boot/request/"
BOOT_CHALLENGE_PREFIX = "control/boot/challenge/"
BOOT_PROOF_PREFIX = "control/boot/proof/"
BOOT_RESULT_PREFIX = "control/boot/result/"

def boot_topic(kind: str, device_id: str) -> str:
    """Return the canonical V6 boot-registration Zenoh key expression."""
    prefixes = {
        "request": BOOT_REQUEST_PREFIX,
        "challenge": BOOT_CHALLENGE_PREFIX,
        "proof": BOOT_PROOF_PREFIX,
        "result": BOOT_RESULT_PREFIX,
    }
    if kind not in prefixes or not _ID.fullmatch(str(device_id)):
        raise ValueError("invalid V6 boot topic component")
    return prefixes[kind] + device_id



def _cn_matches_identity(common_name: str, device_id: str) -> bool:
    """Accept the stable CN or the certificate lifecycle generation-bound CN."""
    return bool(re.fullmatch(re.escape(device_id) + r"(?:-g[1-9][0-9]*-[A-Za-z0-9_]+)?", common_name))
_CHALLENGE_KEYS = {
    "version", "challenge_id", "nonce", "device_id", "device_generation",
    "device_common_name", "device_certificate_fingerprint_sha256", "boot_id",
    "issued_at_ns", "expires_at_ns", "gateway_common_name",
    "gateway_generation", "gateway_certificate_fingerprint_sha256",
    "gateway_signature",
}


class BootRegistrationError(ValueError):
    """A boot challenge/response was malformed, stale, replayed, or unauthenticated."""

    def __init__(self, reason_code: str, detail: str = ""):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{': ' + detail if detail else ''}")


def canonical_json(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")



def _device_identity_claims(identity: ActiveDeviceIdentity) -> dict[str, Any]:
    if not _ID.fullmatch(identity.device_id):
        raise BootRegistrationError("INVALID_DEVICE_ID")
    if isinstance(identity.generation, bool) or not isinstance(identity.generation, int) or identity.generation < 1:
        raise BootRegistrationError("INVALID_CERTIFICATE_GENERATION")
    if not _HEX_256.fullmatch(identity.certificate_fingerprint.hex()):
        raise BootRegistrationError("INVALID_CERTIFICATE_FINGERPRINT")
    return {
        "device_id": identity.device_id,
        "device_generation": identity.generation,
        "device_common_name": identity.common_name,
        "device_certificate_fingerprint_sha256": identity.certificate_fingerprint.hex(),
    }


def _sign(private_key: Any, domain: bytes, payload: bytes) -> str:
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise BootRegistrationError("UNSUPPORTED_PRIVATE_KEY")
    return private_key.sign(
        domain + payload,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    ).hex()


def _verify(public_key: Any, signature_hex: str, domain: bytes, payload: bytes) -> None:
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise BootRegistrationError("UNSUPPORTED_CERTIFICATE_PUBLIC_KEY")
    if not isinstance(signature_hex, str) or not re.fullmatch(r"[0-9a-f]{512}", signature_hex):
        raise BootRegistrationError("INVALID_SIGNATURE_ENCODING")
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            domain + payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise BootRegistrationError("INVALID_SIGNATURE") from exc


def verify_gateway_challenge(
    challenge: dict[str, Any],
    device_identity: ActiveDeviceIdentity,
    active_gateway_identity: ActiveDeviceIdentity,
    *,
    now_ns: int | None = None,
    max_ttl_sec: float = 60.0,
) -> None:
    """Verify Gateway signature, active certificate bindings, boot ID and expiry."""
    if not isinstance(challenge, dict) or set(challenge) != _CHALLENGE_KEYS:
        raise BootRegistrationError("MALFORMED_CHALLENGE")
    if challenge.get("version") != 1:
        raise BootRegistrationError("UNSUPPORTED_CHALLENGE_VERSION")
    try:
        challenge_id = str(challenge["challenge_id"])
        nonce = str(challenge["nonce"])
        boot_id = normalize_boot_id(challenge["boot_id"])
        issued = challenge["issued_at_ns"]
        expires = challenge["expires_at_ns"]
        device_gen = challenge["device_generation"]
        gateway_gen = challenge["gateway_generation"]
    except (KeyError, TypeError, ValueError) as exc:
        raise BootRegistrationError("MALFORMED_CHALLENGE") from exc
    if not _HEX_128.fullmatch(challenge_id) or not _HEX_256.fullmatch(nonce):
        raise BootRegistrationError("MALFORMED_CHALLENGE")
    if (isinstance(issued, bool) or not isinstance(issued, int)
            or isinstance(expires, bool) or not isinstance(expires, int)
            or expires <= issued):
        raise BootRegistrationError("INVALID_CHALLENGE_TIME")
    now = time.time_ns() if now_ns is None else now_ns
    max_ttl_ns = int(max_ttl_sec * 1_000_000_000)
    if issued > now + 5_000_000_000 or now >= expires or expires - issued > max_ttl_ns:
        raise BootRegistrationError("CHALLENGE_EXPIRED_OR_INVALID_TTL")
    claims = _device_identity_claims(device_identity)
    expected_gateway = _device_identity_claims(active_gateway_identity)
    if device_identity.device_id != challenge["device_id"] or boot_id != challenge["boot_id"]:
        raise BootRegistrationError("CHALLENGE_DEVICE_OR_BOOT_MISMATCH")
    if (device_gen != claims["device_generation"]
            or challenge["device_common_name"] != claims["device_common_name"]
            or challenge["device_certificate_fingerprint_sha256"] != claims["device_certificate_fingerprint_sha256"]):
        raise BootRegistrationError("STALE_DEVICE_CERTIFICATE_BINDING")
    if (active_gateway_identity.device_id != "gateway"
            or gateway_gen != expected_gateway["device_generation"]
            or challenge["gateway_common_name"] != expected_gateway["device_common_name"]
            or challenge["gateway_certificate_fingerprint_sha256"] != expected_gateway["device_certificate_fingerprint_sha256"]):
        raise BootRegistrationError("STALE_GATEWAY_CERTIFICATE_BINDING")
    signed = dict(challenge)
    signature = signed.pop("gateway_signature")
    _verify(
        active_gateway_identity.certificate_public_key,
        signature,
        _CHALLENGE_DOMAIN,
        canonical_json(signed),
    )


def create_boot_registration_proof(
    challenge: dict[str, Any],
    device_identity: ActiveDeviceIdentity,
    device_private_key: Any,
) -> dict[str, Any]:
    """Create a device-key proof over the exact Gateway-signed challenge."""
    if not isinstance(challenge, dict) or set(challenge) != _CHALLENGE_KEYS:
        raise BootRegistrationError("MALFORMED_CHALLENGE")
    claims = _device_identity_claims(device_identity)
    if (challenge.get("device_id") != device_identity.device_id
            or challenge.get("device_generation") != device_identity.generation
            or challenge.get("device_common_name") != device_identity.common_name
            or challenge.get("device_certificate_fingerprint_sha256") != claims["device_certificate_fingerprint_sha256"]):
        raise BootRegistrationError("CHALLENGE_DEVICE_OR_BOOT_MISMATCH")
    return {
        "version": 1,
        "challenge": challenge,
        "device_signature": _sign(device_private_key, _PROOF_DOMAIN, canonical_json(challenge)),
    }


@dataclass
class _PendingChallenge:
    document: dict[str, Any]
    expires_monotonic: float


class BootRegistrationAuthority:
    """Gateway-side one-shot challenge registry and admission registration adapter.

    Pending challenges are process-local and expire rapidly. A Gateway restart
    invalidates every outstanding challenge, which is fail-closed. A proof is
    consumed before cryptographic verification so even a bad proof cannot be
    retried against the same nonce.
    """

    def __init__(
        self,
        admission_gate,
        *,
        cert_pointer: str,
        revocation_ledger: str,
        session_id: str,
        ttl_sec: float = 20.0,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        identity_loader: Callable = load_active_device_identity,
        signing_key_loader: Callable = load_active_device_signing_key,
    ):
        if ttl_sec <= 0 or ttl_sec > 60:
            raise ValueError("boot challenge TTL must be within (0, 60] seconds")
        self.admission_gate = admission_gate
        self.cert_pointer = str(cert_pointer)
        self.revocation_ledger = str(revocation_ledger)
        self.session_id = str(session_id)
        self.ttl_sec = float(ttl_sec)
        self.clock_ns = clock_ns
        self.monotonic = monotonic
        self.identity_loader = identity_loader
        self.signing_key_loader = signing_key_loader
        self._pending: dict[str, _PendingChallenge] = {}
        # Repeated requests for the same device/boot epoch receive the same
        # outstanding challenge. This makes the sensor's retry-with-same-boot_id
        # behavior idempotent and avoids flooding the Gateway with orphaned
        # one-shot challenges while subscriber discovery converges.
        self._pending_boots: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    def _identity_not_revoked(self, identity: ActiveDeviceIdentity) -> None:
        try:
            revoked = CertRevocationLedger(self.revocation_ledger).is_revoked(
                identity.device_id, identity.serial_number_hex
            )
        except Exception as exc:
            raise BootRegistrationError("REVOCATION_LEDGER_UNAVAILABLE") from exc
        if revoked:
            raise BootRegistrationError("CERTIFICATE_REVOKED")

    def issue_challenge(self, device_id: str, boot_id: str) -> dict[str, Any]:
        if not isinstance(device_id, str) or not _ID.fullmatch(device_id):
            raise BootRegistrationError("INVALID_DEVICE_ID")
        try:
            normalized_boot_id = normalize_boot_id(boot_id)
        except ValueError as exc:
            raise BootRegistrationError("INVALID_BOOT_ID") from exc

        device_identity = self.identity_loader(self.cert_pointer, device_id)
        self._identity_not_revoked(device_identity)
        if device_identity.device_id != device_id or not _cn_matches_identity(device_identity.common_name, device_id):
            raise BootRegistrationError("DEVICE_CERTIFICATE_CN_MISMATCH")
        gateway_identity, gateway_key = self.signing_key_loader(self.cert_pointer, "gateway")
        self._identity_not_revoked(gateway_identity)
        if (gateway_identity.device_id != "gateway"
                or not _cn_matches_identity(gateway_identity.common_name, "gateway")):
            raise BootRegistrationError("INVALID_GATEWAY_IDENTITY")

        now_ns = int(self.clock_ns())
        expires_at_ns = now_ns + int(self.ttl_sec * 1_000_000_000)
        challenge_id = secrets.token_hex(16)
        unsigned = {
            "version": 1,
            "challenge_id": challenge_id,
            "nonce": secrets.token_hex(32),
            **_device_identity_claims(device_identity),
            "boot_id": normalized_boot_id,
            "issued_at_ns": now_ns,
            "expires_at_ns": expires_at_ns,
            "gateway_common_name": gateway_identity.common_name,
            "gateway_generation": gateway_identity.generation,
            "gateway_certificate_fingerprint_sha256": gateway_identity.certificate_fingerprint.hex(),
        }
        challenge = dict(unsigned)
        challenge["gateway_signature"] = _sign(
            gateway_key, _CHALLENGE_DOMAIN, canonical_json(unsigned)
        )
        with self._lock:
            # Bound memory usage; expired entries are discarded on every issue.
            now_mono = self.monotonic()
            expired_ids = {
                key for key, value in self._pending.items()
                if value.expires_monotonic <= now_mono
            }
            for expired_id in expired_ids:
                self._pending.pop(expired_id, None)
            self._pending_boots = {
                boot_key: pending_id for boot_key, pending_id in self._pending_boots.items()
                if pending_id in self._pending
            }
            boot_key = (device_id, normalized_boot_id)
            existing_id = self._pending_boots.get(boot_key)
            existing = self._pending.get(existing_id) if existing_id else None
            if existing is not None and existing.expires_monotonic > now_mono:
                # A retry is the same logical request. Return the exact challenge
                # so the device only needs to prove one nonce and one boot epoch.
                return dict(existing.document)
            if len(self._pending) >= 4096:
                raise BootRegistrationError("CHALLENGE_CAPACITY_EXCEEDED")
            self._pending[challenge_id] = _PendingChallenge(
                document=challenge,
                expires_monotonic=now_mono + self.ttl_sec,
            )
            self._pending_boots[boot_key] = challenge_id
        return challenge

    def complete_registration(
        self,
        proof: dict[str, Any],
        *,
        expected_device_id: str,
        source_type: str = "live_validated",
    ) -> dict[str, Any]:
        if not isinstance(proof, dict) or set(proof) != {"version", "challenge", "device_signature"}:
            raise BootRegistrationError("MALFORMED_REGISTRATION_PROOF")
        if proof.get("version") != 1 or not isinstance(proof.get("challenge"), dict):
            raise BootRegistrationError("MALFORMED_REGISTRATION_PROOF")
        challenge = proof["challenge"]
        if set(challenge) != _CHALLENGE_KEYS:
            raise BootRegistrationError("MALFORMED_CHALLENGE")
        challenge_id = challenge.get("challenge_id")
        if not isinstance(challenge_id, str) or not _HEX_128.fullmatch(challenge_id):
            raise BootRegistrationError("MALFORMED_CHALLENGE")

        # Consume exactly once before any verification. Unknown/used challenges
        # cannot be replayed, including after a previous successful registration.
        with self._lock:
            pending = self._pending.pop(challenge_id, None)
            if pending is not None:
                boot_key = (str(pending.document.get("device_id", "")), str(pending.document.get("boot_id", "")))
                if self._pending_boots.get(boot_key) == challenge_id:
                    self._pending_boots.pop(boot_key, None)
        if pending is None:
            raise BootRegistrationError("CHALLENGE_UNKNOWN_OR_REPLAYED")
        if (self.monotonic() >= pending.expires_monotonic
                or int(self.clock_ns()) >= int(pending.document["expires_at_ns"])):
            raise BootRegistrationError("CHALLENGE_EXPIRED")
        if canonical_json(challenge) != canonical_json(pending.document):
            raise BootRegistrationError("CHALLENGE_TAMPERED")
        if challenge.get("device_id") != expected_device_id:
            raise BootRegistrationError("CHALLENGE_DEVICE_ROUTE_MISMATCH")

        device_id = str(challenge["device_id"])
        boot_id = normalize_boot_id(challenge["boot_id"])
        device_identity = self.identity_loader(self.cert_pointer, device_id)
        self._identity_not_revoked(device_identity)
        if device_identity.device_id != device_id or not _cn_matches_identity(device_identity.common_name, device_id):
            raise BootRegistrationError("DEVICE_CERTIFICATE_CN_MISMATCH")
        gateway_identity = self.identity_loader(self.cert_pointer, "gateway")
        self._identity_not_revoked(gateway_identity)
        if (gateway_identity.device_id != "gateway"
                or not _cn_matches_identity(gateway_identity.common_name, "gateway")):
            raise BootRegistrationError("INVALID_GATEWAY_IDENTITY")
        claims = _device_identity_claims(device_identity)
        gateway_claims = _device_identity_claims(gateway_identity)
        if (challenge["device_generation"] != claims["device_generation"]
                or challenge["device_common_name"] != claims["device_common_name"]
                or challenge["device_certificate_fingerprint_sha256"] != claims["device_certificate_fingerprint_sha256"]):
            raise BootRegistrationError("STALE_DEVICE_CERTIFICATE_BINDING")
        if (challenge["gateway_generation"] != gateway_claims["device_generation"]
                or challenge["gateway_common_name"] != gateway_claims["device_common_name"]
                or challenge["gateway_certificate_fingerprint_sha256"] != gateway_claims["device_certificate_fingerprint_sha256"]):
            raise BootRegistrationError("STALE_GATEWAY_CERTIFICATE_BINDING")
        _verify(
            device_identity.certificate_public_key,
            proof.get("device_signature", ""),
            _PROOF_DOMAIN,
            canonical_json(challenge),
        )

        context = ProvenanceContext(
            session_id=self.session_id,
            device_cn=device_identity.common_name,
            cert_fingerprint=device_identity.certificate_fingerprint,
            source_type=source_type,
            device_id=device_id,
            generation=device_identity.generation,
            admission_policy_version="v6.0-boot-registration",
        )
        self.admission_gate.register_verified_device_boot(context, boot_id)
        return {
            "version": 1,
            "ok": True,
            "device_id": device_id,
            "generation": device_identity.generation,
            "boot_id": boot_id,
            "sequence_start": 0,
            "challenge_id": challenge_id,
            "registered_at_ns": int(self.clock_ns()),
        }


def perform_boot_registration(
    session,
    *,
    device_id: str,
    cert_pointer: str,
    revocation_ledger: str | None = None,
    timeout_sec: float = 10.0,
    retry_interval_sec: float = 0.5,
    boot_id: str | None = None,
) -> dict[str, Any]:
    """Sensor-side Zenoh challenge/response; returns a verified boot session.

    Subscribers are declared before publishing the request to avoid missing a
    fast Gateway response. The sensor verifies the Gateway's certificate-bound
    challenge signature, then proves possession of its own active leaf key.
    """
    from queue import Empty, Queue
    from utils.cert_lifecycle import CertRevocationLedger
    from utils.device_identity import load_active_device_identity, load_active_device_signing_key
    import os
    from pathlib import Path

    if not _ID.fullmatch(device_id):
        raise ValueError("invalid device_id")
    if timeout_sec <= 0 or retry_interval_sec <= 0:
        raise ValueError("boot registration timeout and retry interval must be positive")
    normalized_boot_id = normalize_boot_id(boot_id or secrets.token_hex(16))
    device_identity, device_key = load_active_device_signing_key(cert_pointer, device_id)
    gateway_identity = load_active_device_identity(cert_pointer, "gateway")
    ledger_path = (str(revocation_ledger).strip() if revocation_ledger else
                   os.environ.get("LUMEN_CERT_REVOCATIONS", "").strip())
    if not ledger_path or not Path(ledger_path).is_file():
        raise BootRegistrationError("REVOCATION_LEDGER_UNAVAILABLE")
    ledger = CertRevocationLedger(ledger_path)
    for identity in (device_identity, gateway_identity):
        try:
            revoked = ledger.is_revoked(identity.device_id, identity.serial_number_hex)
        except Exception as exc:
            raise BootRegistrationError("REVOCATION_LEDGER_UNAVAILABLE") from exc
        if revoked:
            raise BootRegistrationError("CERTIFICATE_REVOKED")
    challenge_queue: Queue = Queue(maxsize=4)
    result_queue: Queue = Queue(maxsize=4)
    challenge_topic = boot_topic("challenge", device_id)
    result_topic = boot_topic("result", device_id)

    def _read(sample, target: Queue) -> None:
        try:
            raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            if len(raw) > 8192:
                return
            target.put_nowait(json.loads(raw.decode("utf-8")))
        except Exception:
            return

    challenge_sub = session.declare_subscriber(
        challenge_topic, lambda sample: _read(sample, challenge_queue)
    )
    result_sub = session.declare_subscriber(
        result_topic, lambda sample: _read(sample, result_queue)
    )
    try:
        request = {
            "version": 1,
            "device_id": device_id,
            "boot_id": normalized_boot_id,
        }
        request_topic = boot_topic("request", device_id)
        request_payload = canonical_json(request)
        deadline = time.monotonic() + timeout_sec
        next_request_at = 0.0
        challenge = None
        # Gateway and sensors are separate child processes. Retry the exact same
        # boot request if the Gateway has not yet declared its subscribers; the
        # boot ID remains constant across retries and the first valid challenge
        # ends the request loop.
        while time.monotonic() < deadline and challenge is None:
            now = time.monotonic()
            if now >= next_request_at:
                session.put(request_topic, request_payload)
                next_request_at = now + retry_interval_sec
            try:
                challenge = challenge_queue.get(
                    timeout=min(0.1, max(0.01, deadline - time.monotonic()))
                )
            except Empty:
                try:
                    early_result = result_queue.get_nowait()
                except Empty:
                    continue
                if isinstance(early_result, dict) and early_result.get("ok") is False:
                    raise BootRegistrationError(str(early_result.get("reason", "BOOT_CHALLENGE_REJECTED")))
        if challenge is None:
            raise TimeoutError("timed out waiting for Gateway boot challenge")

        verify_gateway_challenge(
            challenge, device_identity, gateway_identity, max_ttl_sec=60.0
        )
        proof = create_boot_registration_proof(challenge, device_identity, device_key)
        session.put(boot_topic("proof", device_id), canonical_json(proof))
        try:
            result = result_queue.get(timeout=timeout_sec)
        except Empty as exc:
            raise TimeoutError("timed out waiting for Gateway boot registration result") from exc
        if (not isinstance(result, dict) or result.get("ok") is not True
                or result.get("device_id") != device_id
                or result.get("boot_id") != normalized_boot_id
                or result.get("generation") != device_identity.generation
                or result.get("sequence_start") != 0):
            reason = result.get("reason", "BOOT_REGISTRATION_REJECTED") if isinstance(result, dict) else "BOOT_REGISTRATION_REJECTED"
            raise BootRegistrationError(str(reason))
        return {
            "device_id": device_id,
            "generation": device_identity.generation,
            "boot_id": normalized_boot_id,
            "sequence": 0,
            "registered_at_ns": result.get("registered_at_ns"),
        }
    finally:
        challenge_sub.undeclare()
        result_sub.undeclare()
