"""Etapa E6: Forensic Quarantine Sink cu protecție anti-DoS (Token Bucket + Storage Quotas).

Primește mesajele redirecționate de PEP pe shadow/sandbox/<device_id>/data
și le salvează controlat în telemetry_incidents.db.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from storage.telemetry_db import LumenStorageManager
from storage.persistence import persist_incident_capture
from utils.rate_limit import TokenBucketRateLimiter


@dataclass(frozen=True)
class SinkIngestResult:
    accepted: bool
    reason: str


class ForensicQuarantineSink:
    """Collector izolat cu cote de rată și spațiu pentru traficul aflat în SHADOW_QUARANTINE."""

    def __init__(
        self,
        storage: LumenStorageManager,
        max_payload_bytes: int = 512,
        rate_per_sec: float = 5.0,
        burst: float = 10.0,
        max_bytes_per_device: int = 256 * 1024,
    ):
        self.storage = storage
        self.max_payload_bytes = max_payload_bytes
        self.max_bytes_per_device = max_bytes_per_device
        self.limiter = TokenBucketRateLimiter(rate=rate_per_sec, burst=burst, max_keys=64)
        self._bytes_used: dict[str, int] = {}
        self.dropped_rate = 0
        self.dropped_quota = 0
        self.stored_count = 0

    def ingest_shadow_envelope(
        self,
        session_id: str,
        device_id: str,
        device_cn: str,
        generation: int,
        cert_fingerprint: bytes,
        original_topic: str,
        raw_payload: bytes,
        risk_score: float,
        reason_code: str,
        xai_json: str = "{}",
        now_mono: float | None = None,
    ) -> SinkIngestResult:
        now = time.monotonic() if now_mono is None else now_mono

        if len(raw_payload) > self.max_payload_bytes:
            self.dropped_quota += 1
            return SinkIngestResult(False, "PAYLOAD_OVERSIZE_DROPPED")

        if not self.limiter.allow(device_id, now):
            self.dropped_rate += 1
            return SinkIngestResult(False, "SINK_RATE_LIMIT_DROPPED")

        used = self._bytes_used.get(device_id, 0)
        if used + len(raw_payload) > self.max_bytes_per_device:
            self.dropped_quota += 1
            return SinkIngestResult(False, "SINK_STORAGE_QUOTA_EXCEEDED")

        now_ns = time.time_ns()
        retention_ns = now_ns + 14 * 24 * 3600 * 1_000_000_000

        persist_incident_capture(
            self.storage,
            (
                session_id,
                device_id,
                device_cn,
                generation,
                cert_fingerprint,
                original_topic,
                f"shadow/sandbox/{device_id}/data",
                now_ns,
                None,
                "shadow_sink",
                reason_code,
                float(risk_score),
                xai_json,
                len(raw_payload),
                hashlib.sha256(raw_payload).digest(),
                raw_payload,
                1,
                None,
                None,
                retention_ns,
            )
        )
        # Advance in-memory quotas only after the synchronous forensic commit.
        self._bytes_used[device_id] = used + len(raw_payload)
        self.stored_count += 1
        return SinkIngestResult(True, "STORED_IN_FORENSIC_SINK")