"""Lumen Edge Security v5.0 — Admission Gate & Provenance Tracking.

Regulă arhitecturală:
- Niciun mesaj nu intră în telemetry_normal.db doar pentru că are certificat mTLS valid.
- AdmissionGate verifică:
  1. Identitatea stabilă (device_id) extrasă strict din CN-ul verificat + fingerprint SHA-256.
  2. Starea curentă a dispozitivului (dispozitivele în carantină, revocate sau în incident
     sunt respinse imediat din normal_db și direcționate către telemetry_incidents.db).
  3. Autorizarea pe topic, dimensiunea maximă, validitatea UTF-8 și plauzibilitatea fizică.
  4. Proveniența explicită a sesiunii ('baseline_lab' sau 'live_validated').
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional

from storage.telemetry_db import LumenStorageManager, FreshnessRejected
from storage.persistence import persist_normal_event, persist_incident_capture
from utils.sensor_payload import (
    MAX_SENSOR_PAYLOAD_BYTES, TOPIC_PHYSICAL_BOUNDS, DISCRETE_TOPIC_VALUES,
    parse_sensor_payload, parse_sensor_frame, normalize_boot_id, MAX_SENSOR_FRAME_BYTES,
)

MAX_RAW_PAYLOAD_BYTES = MAX_SENSOR_PAYLOAD_BYTES
CN_PATTERN = re.compile(r"^([a-zA-Z0-9_]+)(?:-g(\d+)-[a-zA-Z0-9_]+)?$")


@dataclass(frozen=True)
class ProvenanceContext:
    session_id: str
    device_cn: str
    cert_fingerprint: bytes
    source_type: str  # 'baseline_lab' | 'live_validated'
    admission_policy_version: str = "v5.0-gate-1"
    device_id: Optional[str] = None
    generation: Optional[int] = None


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    device_id: str
    generation: int
    reason: str
    numeric_value: Optional[float] = None
    canonical_text: Optional[str] = None
    protocol_version: int = 5
    boot_id: Optional[str] = None
    sequence: Optional[int] = None
    device_timestamp_ns: Optional[int] = None


def parse_device_identity(device_cn: str) -> tuple[str, int]:
    """Extrage determinist (device_id, generation) din CN-ul certificatului verificat."""
    match = CN_PATTERN.match(device_cn.strip())
    if not match:
        raise ValueError(f"Invalid certificate CN format: {device_cn!r}")
    device_id = match.group(1)
    generation = int(match.group(2)) if match.group(2) is not None else 1
    return device_id, generation


def extract_numeric_value(topic: str, payload_text: str) -> tuple[bool, Optional[float], str]:
    """Compatibility wrapper over the single parser shared with IDS."""
    result = parse_sensor_payload(topic, payload_text)
    return result.valid, result.numeric_value, result.reason


class AdmissionGate:
    """Poartă de admitere între ingestia Gateway-ului și telemetry_normal.db."""

    def __init__(
        self,
        storage: LumenStorageManager,
        allowed_topics_by_device: dict[str, set[str]],
    ):
        self.storage = storage
        self.allowed_topics = allowed_topics_by_device
        self._blocked_devices: set[str] = set()

    def set_device_blocked(self, device_id: str, blocked: bool) -> None:
        if blocked:
            self._blocked_devices.add(device_id)
        else:
            self._blocked_devices.discard(device_id)

    def evaluate_and_ingest(
        self,
        ctx: ProvenanceContext,
        topic: str,
        raw_bytes: bytes,
        device_ts_ns: Optional[int] = None,
    ) -> AdmissionResult:
        gateway_ts_ns = time.time_ns()
        payload_sha = hashlib.sha256(raw_bytes).digest()

        try:
            parsed_device_id, parsed_generation = parse_device_identity(ctx.device_cn)
            device_id = ctx.device_id or parsed_device_id
            generation = ctx.generation if ctx.generation is not None else parsed_generation
            if device_id != parsed_device_id:
                raise ValueError("certificate CN does not match stable device_id")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise ValueError("invalid certificate generation")
        except ValueError as exc:
            return AdmissionResult(False, "unknown", 0, str(exc))

        if len(ctx.cert_fingerprint) != 32:
            return AdmissionResult(False, device_id, generation, "INVALID_CERT_FINGERPRINT")

        if ctx.source_type not in ("baseline_lab", "live_validated"):
            return AdmissionResult(False, device_id, generation, f"UNTRUSTED_SOURCE:{ctx.source_type}")

        # Dispozitivele aflate în incident/carantină nu pot intra în normal_db
        if device_id in self._blocked_devices:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes, payload_sha, "DEVICE_QUARANTINED_OR_INCIDENT",
            )
            return AdmissionResult(False, device_id, generation, "DEVICE_QUARANTINED_OR_INCIDENT")

        allowed = self.allowed_topics.get(device_id, set())
        if topic not in allowed:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes, payload_sha, "UNAUTHORIZED_TOPIC_FOR_DEVICE",
            )
            return AdmissionResult(False, device_id, generation, "UNAUTHORIZED_TOPIC_FOR_DEVICE")

        if len(raw_bytes) == 0 or len(raw_bytes) > MAX_RAW_PAYLOAD_BYTES:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes[:MAX_RAW_PAYLOAD_BYTES], payload_sha, "INVALID_PAYLOAD_SIZE",
            )
            return AdmissionResult(False, device_id, generation, "INVALID_PAYLOAD_SIZE")

        parsed = parse_sensor_payload(topic, raw_bytes, max_bytes=MAX_RAW_PAYLOAD_BYTES)
        ok, num_val, reason = parsed.valid, parsed.numeric_value, parsed.reason
        if not ok:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes, payload_sha, reason,
            )
            return AdmissionResult(False, device_id, generation, reason, numeric_value=num_val)

        # Admis în telemetry_normal.db
        persist_normal_event(
            self.storage,
            (
                ctx.session_id,
                device_id,
                ctx.device_cn,
                generation,
                ctx.cert_fingerprint,
                topic,
                gateway_ts_ns,
                device_ts_ns,
                len(raw_bytes),
                payload_sha,
                num_val,
                parsed.canonical_text,
                ctx.source_type,
                ctx.admission_policy_version,
            )
        )
        return AdmissionResult(
            True, device_id, generation, "ADMITTED",
            numeric_value=num_val, canonical_text=parsed.canonical_text,
        )

    def register_verified_device_boot(self, ctx: ProvenanceContext, boot_id: str) -> None:
        """Persist a new V6 boot epoch after a separate challenge proof succeeds.

        The authenticated caller must verify a fresh, single-use Gateway
        challenge with the active device certificate before calling this method.
        A frame's self-asserted boot_id is never sufficient to reset sequence state.
        """
        parsed_device_id, parsed_generation = parse_device_identity(ctx.device_cn)
        device_id = ctx.device_id or parsed_device_id
        generation = ctx.generation if ctx.generation is not None else parsed_generation
        if device_id != parsed_device_id or isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("boot registration identity does not match authenticated certificate CN")
        if len(ctx.cert_fingerprint) != 32:
            raise ValueError("boot registration requires a SHA-256 certificate fingerprint")
        if ctx.source_type not in ("baseline_lab", "live_validated"):
            raise ValueError("boot registration source is not trusted")
        self.storage.register_verified_device_boot(device_id, generation, normalize_boot_id(boot_id))

    def evaluate_v6_and_ingest(
        self,
        ctx: ProvenanceContext,
        topic: str,
        raw_frame: bytes | str,
    ) -> AdmissionResult:
        """Validate a V6 frame and atomically persist it with its sequence update."""
        frame_bytes = raw_frame if isinstance(raw_frame, bytes) else str(raw_frame).encode("utf-8", errors="strict")
        gateway_ts_ns = time.time_ns()
        frame_sha = hashlib.sha256(frame_bytes).digest()
        try:
            parsed_device_id, parsed_generation = parse_device_identity(ctx.device_cn)
            device_id = ctx.device_id or parsed_device_id
            generation = ctx.generation if ctx.generation is not None else parsed_generation
            if device_id != parsed_device_id:
                raise ValueError("certificate CN does not match stable device_id")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
                raise ValueError("invalid certificate generation")
        except ValueError as exc:
            return AdmissionResult(False, "unknown", 0, str(exc), protocol_version=6)

        if len(ctx.cert_fingerprint) != 32:
            return AdmissionResult(False, device_id, generation, "INVALID_CERT_FINGERPRINT", protocol_version=6)
        if ctx.source_type not in ("baseline_lab", "live_validated"):
            return AdmissionResult(False, device_id, generation, f"UNTRUSTED_SOURCE:{ctx.source_type}", protocol_version=6)

        parsed_frame = parse_sensor_frame(topic, raw_frame, expected_device_id=device_id)
        if not parsed_frame.valid or parsed_frame.frame is None:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, None,
                frame_bytes[:MAX_SENSOR_FRAME_BYTES], frame_sha, f"INVALID_V6_FRAME:{parsed_frame.reason}",
            )
            return AdmissionResult(False, device_id, generation, f"INVALID_V6_FRAME:{parsed_frame.reason}", protocol_version=6)
        frame = parsed_frame.frame

        if device_id in self._blocked_devices:
            reason = "DEVICE_QUARANTINED_OR_INCIDENT"
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, frame.device_timestamp_ns,
                frame_bytes, frame_sha, reason,
            )
            return AdmissionResult(False, device_id, generation, reason, protocol_version=6,
                                  boot_id=frame.boot_id, sequence=frame.sequence,
                                  device_timestamp_ns=frame.device_timestamp_ns)
        if topic not in self.allowed_topics.get(device_id, set()):
            reason = "UNAUTHORIZED_TOPIC_FOR_DEVICE"
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, frame.device_timestamp_ns,
                frame_bytes, frame_sha, reason,
            )
            return AdmissionResult(False, device_id, generation, reason, protocol_version=6,
                                  boot_id=frame.boot_id, sequence=frame.sequence,
                                  device_timestamp_ns=frame.device_timestamp_ns)

        canonical_bytes = frame.canonical_text.encode("utf-8")
        canonical_sha = hashlib.sha256(canonical_bytes).digest()
        row = (
            ctx.session_id, device_id, ctx.device_cn, generation, ctx.cert_fingerprint,
            topic, gateway_ts_ns, frame.device_timestamp_ns, len(canonical_bytes),
            canonical_sha, frame.numeric_value, frame.canonical_text, ctx.source_type,
            ctx.admission_policy_version, 6, frame.boot_id, frame.sequence,
        )
        try:
            self.storage.persist_v6_normal_event(
                row,
                device_id=device_id,
                certificate_generation=generation,
                boot_id=frame.boot_id,
                sequence=frame.sequence,
                device_timestamp_ns=frame.device_timestamp_ns,
            )
        except FreshnessRejected as exc:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, frame.device_timestamp_ns,
                frame_bytes, frame_sha, exc.reason_code,
            )
            return AdmissionResult(False, device_id, generation, exc.reason_code,
                                  numeric_value=frame.numeric_value, canonical_text=frame.canonical_text,
                                  protocol_version=6, boot_id=frame.boot_id, sequence=frame.sequence,
                                  device_timestamp_ns=frame.device_timestamp_ns)

        return AdmissionResult(
            True, device_id, generation, "ADMITTED", numeric_value=frame.numeric_value,
            canonical_text=frame.canonical_text, protocol_version=6, boot_id=frame.boot_id,
            sequence=frame.sequence, device_timestamp_ns=frame.device_timestamp_ns,
        )

    def _route_rejected_to_incidents(
        self,
        ctx: ProvenanceContext,
        device_id: str,
        generation: int,
        topic: str,
        gateway_ts_ns: int,
        device_ts_ns: Optional[int],
        raw_bytes: bytes,
        payload_sha: bytes,
        reason: str,
    ) -> None:
        retention_ns = gateway_ts_ns + 7 * 24 * 3600 * 1_000_000_000
        persist_incident_capture(
            self.storage,
            (
                ctx.session_id,
                device_id,
                ctx.device_cn,
                generation,
                ctx.cert_fingerprint,
                topic,
                f"shadow/sandbox/{device_id}/rejected",
                gateway_ts_ns,
                device_ts_ns,
                "ids_alert",
                reason,
                None,
                None,
                len(raw_bytes),
                payload_sha,
                raw_bytes,
                1,
                None,
                None,
                retention_ns,
            )
        )