"""V6 Stage B: deterministic clean-baseline curation and poison-prevention gates.

This module is intentionally local-only. It reads the three SQLite stores and
writes a curation decision through LumenStorageManager's single-writer API.
Forensic/incident captures are indexed in telemetry_incidents.db and are never
copied into telemetry_normal.db or exposed as clean training sessions.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from storage.telemetry_db import LumenStorageManager

_SECURITY_BAD_STATES = {
    "SUSPECT", "QUARANTINED", "SHADOW_QUARANTINE", "REVOKED",
    "ENFORCEMENT_PENDING", "ENFORCEMENT_FAILED", "RECOVERY_PENDING",
}
_OPERATIONAL_BAD_STATES = {"FAULT"}


@dataclass(frozen=True)
class CurationDecision:
    session_id: str
    approved: bool
    status: str
    reasons: tuple[str, ...]
    duration_sec: float
    sensor_coverage: float
    event_count: int
    dataset_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "approved": self.approved,
            "status": self.status,
            "reasons": list(self.reasons),
            "duration_sec": round(self.duration_sec, 6),
            "sensor_coverage": round(self.sensor_coverage, 6),
            "event_count": self.event_count,
            "dataset_sha256": self.dataset_sha256,
        }


def _expected_streams(config: Mapping[str, Any], override) -> set[tuple[str, str]]:
    """Normalize expected streams from ``device -> topics`` or pair lists."""
    source = override if override is not None else config.get("expected_sensor_topics", ())
    result: set[tuple[str, str]] = set()
    if isinstance(source, Mapping):
        for device, topics in source.items():
            if isinstance(topics, str):
                topic_values = (topics,)
            else:
                try:
                    topic_values = tuple(topics)
                except TypeError:
                    topic_values = ()
            for topic in topic_values:
                if str(device).strip() and str(topic).strip():
                    result.add((str(device), str(topic)))
        return result
    for item in source or ():
        if isinstance(item, (tuple, list)) and len(item) == 2:
            device, topic = item
            if str(device).strip() and str(topic).strip():
                result.add((str(device), str(topic)))
        elif isinstance(item, str) and "::" in item:
            device, topic = item.split("::", 1)
            if device.strip() and topic.strip():
                result.add((device.strip(), topic.strip()))
    return result


def _dataset_digest(rows: Iterable[sqlite3.Row]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        numeric = row["numeric_value"]
        fields = (
            int(row["event_id"]), str(row["device_id"]), int(row["generation"]),
            str(row["topic"]), int(row["gateway_ts_ns"]), bytes(row["payload_sha256"]).hex(),
            None if numeric is None else float(numeric), str(row["raw_payload"]),
        )
        digest.update(json.dumps(fields, separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _query_transition_reasons(
    control: sqlite3.Connection,
    devices: set[tuple[str, int]],
    started_at_ns: int,
    ended_at_ns: int,
) -> list[str]:
    if not devices:
        return []
    device_ids = sorted({device for device, _generation in devices})
    placeholders = ",".join("?" for _ in device_ids)
    reasons: list[str] = []
    try:
        rows = control.execute(
            f"""SELECT device_id, generation, dimension, to_state, reason_code
                FROM policy_dimension_transitions
                WHERE ts_ns BETWEEN ? AND ? AND device_id IN ({placeholders})""",
            (started_at_ns, ended_at_ns, *device_ids),
        ).fetchall()
        for row in rows:
            row_identity = (str(row["device_id"]), int(row["generation"]))
            if row_identity not in devices and (row_identity[0], -1) not in devices:
                continue
            state = str(row["to_state"]).upper()
            dimension = str(row["dimension"]).lower()
            if dimension == "security" and state in _SECURITY_BAD_STATES:
                reasons.append(f"SECURITY_TRANSITION:{row['device_id']}:{state}")
            if dimension == "operational" and state in _OPERATIONAL_BAD_STATES:
                reasons.append(f"HARDWARE_FAULT_TRANSITION:{row['device_id']}:{state}")
    except sqlite3.OperationalError as exc:
        # An older control DB may not yet contain V6 policy dimensions. The
        # legacy hash-chained quarantine log below remains mandatory.
        if "no such table" not in str(exc).lower():
            raise

    rows = control.execute(
        f"""SELECT device_id, generation, to_state FROM quarantine_log
            WHERE ts_ns BETWEEN ? AND ? AND device_id IN ({placeholders})""",
        (started_at_ns, ended_at_ns, *device_ids),
    ).fetchall()
    for row in rows:
        row_identity = (str(row["device_id"]), int(row["generation"]))
        if row_identity not in devices and (row_identity[0], -1) not in devices:
            continue
        state = str(row["to_state"]).upper()
        if state in _SECURITY_BAD_STATES:
            reasons.append(f"SECURITY_TRANSITION:{row['device_id']}:{state}")
        if state in _OPERATIONAL_BAD_STATES:
            reasons.append(f"HARDWARE_FAULT_TRANSITION:{row['device_id']}:{state}")

    rows = control.execute(
        f"""SELECT device_id, generation, target_state, reason_code
            FROM enforcement_audit
            WHERE decided_at_ns BETWEEN ? AND ? AND device_id IN ({placeholders})""",
        (started_at_ns, ended_at_ns, *device_ids),
    ).fetchall()
    for row in rows:
        row_identity = (str(row["device_id"]), int(row["generation"]))
        if row_identity in devices or (row_identity[0], -1) in devices:
            reasons.append(
                f"SECURITY_ENFORCEMENT:{row['device_id']}:{row['target_state']}:{row['reason_code']}"
            )
    return reasons


def curate_clean_sessions(
    storage: LumenStorageManager,
    *,
    min_duration_sec: float = 300.0,
    min_sensor_coverage: float = 1.0,
    expected_sensor_topics: Mapping[str, Iterable[str]] | Iterable[tuple[str, str]] | None = None,
    session_ids: Iterable[str] | None = None,
) -> list[CurationDecision]:
    """Approve completed, clean ``baseline_lab`` sessions deterministically.

    Criteria: approved source, completion, minimum effective duration, expected
    sensor/topic coverage, no incidents/protocol/freshness captures, no security
    transition/enforcement, and no operational FAULT transition. ``live_validated``
    is always rejected and the SQLite trigger independently prevents approval.

    For deterministic compressed lab captures, ``effective_duration_sec`` in the
    session config is used. Live data never receives this synthetic duration path.
    """
    if min_duration_sec < 0:
        raise ValueError("min_duration_sec cannot be negative")
    if not 0.0 < min_sensor_coverage <= 1.0:
        raise ValueError("min_sensor_coverage must be within (0, 1]")

    storage.flush_all()
    normal = storage.connect_readonly("normal")
    incidents = storage.connect_readonly("incidents")
    control = storage.connect_readonly("control")
    try:
        params: tuple[Any, ...] = ()
        sql = "SELECT * FROM baseline_sessions"
        requested = tuple(sorted(set(session_ids))) if session_ids is not None else ()
        if session_ids is not None:
            if not requested:
                return []
            sql += f" WHERE session_id IN ({','.join('?' for _ in requested)})"
            params = requested
        sql += " ORDER BY started_at_ns ASC, session_id ASC"
        sessions = normal.execute(sql, params).fetchall()
        results: list[CurationDecision] = []

        for session in sessions:
            sid = str(session["session_id"])
            reasons: list[str] = []
            if str(session["source_type"]) != "baseline_lab":
                reasons.append("SOURCE_NOT_APPROVED_FOR_BASELINE")
            if session["ended_at_ns"] is None:
                reasons.append("SESSION_NOT_COMPLETED")
            started = int(session["started_at_ns"])
            ended = int(session["ended_at_ns"] or time.time_ns())
            if ended < started:
                reasons.append("INVALID_SESSION_TIME_RANGE")

            try:
                config = json.loads(str(session["config_json"]))
                if not isinstance(config, dict):
                    raise ValueError("session config must be an object")
            except (TypeError, ValueError):
                config = {}
                reasons.append("INVALID_SESSION_CONFIG")

            expected_events = config.get("events_per_sensor")
            if expected_events is not None:
                capture_run_id = config.get("capture_run_id")
                if (
                    not isinstance(capture_run_id, str)
                    or re.fullmatch(r"[0-9a-f]{32}", capture_run_id) is None
                    or not sid.startswith(f"baseline-lab-{capture_run_id}-")
                ):
                    reasons.append("CAPTURE_RUN_ID_MISSING_OR_INVALID")
                if (
                    isinstance(expected_events, bool)
                    or not isinstance(expected_events, int)
                    or expected_events < 1
                ):
                    reasons.append("INVALID_EVENTS_PER_SENSOR")
                    expected_events = None

            rows = normal.execute(
                """SELECT event_id, device_id, generation, topic, gateway_ts_ns,
                          payload_sha256, raw_payload, numeric_value
                   FROM raw_events WHERE session_id=? ORDER BY event_id ASC""",
                (sid,),
            ).fetchall()
            event_count = len(rows)
            observed_streams = {
                (str(row["device_id"]), str(row["topic"])) for row in rows
            }
            devices = {(str(row["device_id"]), int(row["generation"])) for row in rows}
            expected = _expected_streams(config, expected_sensor_topics)

            unexpected_streams = sorted(observed_streams - expected)
            if unexpected_streams:
                rendered = ",".join(f"{device}::{topic}" for device, topic in unexpected_streams)
                reasons.append(f"UNEXPECTED_SENSOR_STREAMS:{rendered}")

            if expected_events is not None:
                counts: dict[tuple[str, str], int] = {}
                for row in rows:
                    stream = (str(row["device_id"]), str(row["topic"]))
                    counts[stream] = counts.get(stream, 0) + 1
                for device, topic in sorted(expected):
                    actual_count = counts.get((device, topic), 0)
                    if actual_count != expected_events:
                        reasons.append(
                            f"EVENT_COUNT_MISMATCH:{device}:{topic}:{actual_count}:{expected_events}"
                        )

            expected_devices = {device for device, _topic in expected}
            # A configured sensor that emitted no normal row must still be
            # checked for a fault/security transition during this session.
            for expected_device in expected_devices:
                if not any(device == expected_device for device, _generation in devices):
                    devices.add((expected_device, -1))
            if not expected:
                reasons.append("EXPECTED_SENSOR_COVERAGE_NOT_CONFIGURED")
                coverage = 0.0
            else:
                coverage = len(observed_streams & expected) / len(expected)
                if coverage + 1e-12 < min_sensor_coverage:
                    reasons.append("INSUFFICIENT_SENSOR_COVERAGE")
            if event_count == 0:
                reasons.append("NO_ADMITTED_TELEMETRY")

            duration = max(0.0, (ended - started) / 1_000_000_000.0)
            if str(session["source_type"]) == "baseline_lab" and config.get("compressed_time"):
                synthetic_duration = config.get("effective_duration_sec")
                if synthetic_duration is None:
                    count = config.get("events_per_sensor")
                    rate = config.get("rate_hz", 1.0)
                    try:
                        synthetic_duration = max(0.0, float(count) - 1.0) / float(rate)
                    except (TypeError, ValueError, ZeroDivisionError):
                        synthetic_duration = 0.0
                if isinstance(synthetic_duration, (int, float)) and synthetic_duration >= 0:
                    duration = float(synthetic_duration)
            if duration < min_duration_sec:
                reasons.append("SESSION_BELOW_MINIMUM_DURATION")

            incident = incidents.execute(
                "SELECT capture_source, reason_code, ground_truth_label FROM incident_captures WHERE session_id=? ORDER BY capture_id",
                (sid,),
            ).fetchall()
            if incident:
                reasons.append("INCIDENT_PROTOCOL_OR_FORENSIC_CAPTURE_PRESENT")
                for capture in incident:
                    source = str(capture["capture_source"])
                    reason = str(capture["reason_code"])
                    if source == "fault_lab" or "fault" in str(capture["ground_truth_label"] or "").lower():
                        reasons.append("HARDWARE_FAULT_CAPTURE_PRESENT")
                    if any(token in reason.lower() for token in ("replay", "sequence", "freshness", "boot", "protocol")):
                        reasons.append("PROTOCOL_FRESHNESS_REJECTION_PRESENT")

            if devices:
                reasons.extend(_query_transition_reasons(control, devices, started, ended))
            digest = _dataset_digest(rows) if rows else None
            # Deterministic reason ordering makes repeated curation produce the
            # same decision document for unchanged inputs.
            reasons = sorted(set(reasons))
            approved = not reasons
            decision = CurationDecision(
                session_id=sid,
                approved=approved,
                status="approved_clean_baseline" if approved else "rejected",
                reasons=tuple(reasons),
                duration_sec=duration,
                sensor_coverage=coverage,
                event_count=event_count,
                dataset_sha256=digest,
            )
            storage.mark_session_curated(
                sid, decision.status, decision.to_dict(), dataset_sha256=digest
            )
            results.append(decision)
        return results
    finally:
        normal.close()
        incidents.close()
        control.close()
