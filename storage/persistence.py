"""Synchronous persistence adapters for critical normal/forensic admission."""
from __future__ import annotations

from typing import Any


def persist_normal_event(storage: Any, row: tuple[Any, ...]) -> None:
    """Return only after the SQLite writer committed, or raise on queue/SQL failure."""
    writer = getattr(storage, "_writers", {}).get("normal")
    if writer is None:
        raise RuntimeError("normal telemetry writer is unavailable")
    def operation(conn):
        conn.execute(
            """INSERT INTO raw_events (
                session_id, device_id, device_cn, generation, cert_fingerprint, topic,
                gateway_ts_ns, device_ts_ns, payload_len, payload_sha256, numeric_value,
                raw_payload, source_type, admission_policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
    writer.submit_sync(operation, timeout=10.0)


def persist_incident_capture(storage: Any, row: tuple[Any, ...]) -> None:
    """Return only after forensic evidence commits to telemetry_incidents.db."""
    writer = getattr(storage, "_writers", {}).get("incidents")
    if writer is None:
        raise RuntimeError("incident telemetry writer is unavailable")
    def operation(conn):
        conn.execute(
            """INSERT INTO incident_captures (
                session_id, device_id, device_cn, generation, cert_fingerprint,
                original_topic, shadow_topic, gateway_ts_ns, device_ts_ns,
                capture_source, reason_code, risk_score, xai_contrib_json,
                payload_len, payload_sha256, raw_payload, policy_version,
                ground_truth_label, scenario_id, retention_until_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
    writer.submit_sync(operation, timeout=10.0)
