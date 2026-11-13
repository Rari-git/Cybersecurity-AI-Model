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

from storage.telemetry_db import LumenStorageManager

MAX_RAW_PAYLOAD_BYTES = 512
CN_PATTERN = re.compile(r"^([a-zA-Z0-9_]+)(?:-g(\d+)-[a-zA-Z0-9_]+)?$")

# Limite fizice plauzibile per topic de senzori
TOPIC_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "home/sensors/gas/ppm": (0.0, 2000.0),
    "home/sensors/vibration/shock": (0.0, 5.0),
}
DISCRETE_TOPIC_VALUES: dict[str, frozenset[str]] = {
    "home/sensors/door/state": frozenset({"OPEN", "CLOSED"}),
    "home/sensors/pir/motion": frozenset({"MOTION_DETECTED", "NO_MOTION", "IDLE"}),
}


@dataclass(frozen=True)
class ProvenanceContext:
    session_id: str
    device_cn: str
    cert_fingerprint: bytes
    source_type: str  # 'baseline_lab' | 'live_validated'
    admission_policy_version: str = "v5.0-gate-1"


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    device_id: str
    generation: int
    reason: str
    numeric_value: Optional[float] = None


def parse_device_identity(device_cn: str) -> tuple[str, int]:
    """Extrage determinist (device_id, generation) din CN-ul certificatului verificat."""
    match = CN_PATTERN.match(device_cn.strip())
    if not match:
        raise ValueError(f"Invalid certificate CN format: {device_cn!r}")
    device_id = match.group(1)
    generation = int(match.group(2)) if match.group(2) is not None else 1
    return device_id, generation


def extract_numeric_value(topic: str, payload_text: str) -> tuple[bool, Optional[float], str]:
    """Validează schema și limitele fizice ale payload-ului înainte de scrierea în DB."""
    clean = payload_text.strip()
    if topic in DISCRETE_TOPIC_VALUES:
        if clean not in DISCRETE_TOPIC_VALUES[topic]:
            return False, None, f"INVALID_DISCRETE_STATE:{clean[:32]}"
        mapping = {"CLOSED": 0.0, "NO_MOTION": 0.0, "IDLE": 0.0, "OPEN": 1.0, "MOTION_DETECTED": 1.0}
        return True, mapping[clean], "OK"

    if topic in TOPIC_PHYSICAL_BOUNDS:
        try:
            val = float(clean)
        except ValueError:
            return False, None, "NON_NUMERIC_PAYLOAD"
        if val != val or val in (float("inf"), float("-inf")):
            return False, None, "NAN_OR_INF_PAYLOAD"
        low, high = TOPIC_PHYSICAL_BOUNDS[topic]
        if not (low <= val <= high):
            return False, val, f"OUT_OF_PHYSICAL_BOUNDS:{val}"
        return True, val, "OK"

    return False, None, f"UNREGISTERED_SENSOR_TOPIC:{topic}"


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
            device_id, generation = parse_device_identity(ctx.device_cn)
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

        try:
            payload_text = raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes, payload_sha, "INVALID_UTF8_PAYLOAD",
            )
            return AdmissionResult(False, device_id, generation, "INVALID_UTF8_PAYLOAD")

        ok, num_val, reason = extract_numeric_value(topic, payload_text)
        if not ok:
            self._route_rejected_to_incidents(
                ctx, device_id, generation, topic, gateway_ts_ns, device_ts_ns,
                raw_bytes, payload_sha, reason,
            )
            return AdmissionResult(False, device_id, generation, reason, numeric_value=num_val)

        # Admis în telemetry_normal.db
        self.storage.enqueue_normal_raw_event(
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
                payload_text.strip(),
                ctx.source_type,
                ctx.admission_policy_version,
            )
        )
        return AdmissionResult(True, device_id, generation, "ADMITTED", numeric_value=num_val)

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
        self.storage.enqueue_incident_capture(
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