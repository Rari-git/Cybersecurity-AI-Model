from __future__ import annotations

import hashlib
import sqlite3
import time

import pytest

from storage.persistence import persist_incident_capture, persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.dataset_governance import curate_clean_sessions
from training.train_local_lstm import _extract_windows_by_session

TOPIC = "home/sensors/gas/ppm"
DEVICE = "gas_sensor_01"
FP = hashlib.sha256(b"governance-test-cert").digest()


def _event_row(session_id: str, index: int, *, value: float = 350.0):
    ts = time.time_ns() + index * 1_000_000_000
    payload = str(value + index / 10.0)
    raw = payload.encode()
    return (
        session_id, DEVICE, DEVICE, 1, FP, TOPIC, ts, ts, len(raw),
        hashlib.sha256(raw).digest(), float(payload), payload,
        "baseline_lab", "v6-clean-gate-test",
    )


def _incident_row(session_id: str, reason: str, *, source="attack_lab", label="attack"):
    raw = b"replayed-payload"
    return (
        session_id, DEVICE, DEVICE, 1, FP, TOPIC, TOPIC, time.time_ns(), None,
        source, reason, 99.0, None, len(raw), hashlib.sha256(raw).digest(), raw,
        1, label, "scenario-replay-1", time.time_ns() + 86_400_000_000_000,
    )


def _create_session(storage, session_id: str, *, source="baseline_lab", count=20):
    storage.register_baseline_session(
        session_id,
        source,
        {
            "expected_sensor_topics": [f"{DEVICE}::{TOPIC}"],
            "compressed_time": False,
            "test_fixture": True,
        },
    )
    if source == "baseline_lab":
        for index in range(count):
            persist_normal_event(storage, _event_row(session_id, index))
    storage.finalize_baseline_session(session_id)


def test_only_completed_clean_baseline_sessions_are_curated(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        _create_session(storage, "clean-one")
        decisions = curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["clean-one"]
        )
        assert len(decisions) == 1 and decisions[0].approved
        conn = storage.connect_readonly("normal")
        try:
            row = conn.execute(
                "SELECT approved_for_training FROM baseline_sessions WHERE session_id='clean-one'"
            ).fetchone()
            curation = conn.execute(
                "SELECT status, dataset_sha256 FROM session_curation WHERE session_id='clean-one'"
            ).fetchone()
            assert row[0] == 1
            assert curation[0] == "approved_clean_baseline"
            assert len(curation[1]) == 64
        finally:
            conn.close()
    finally:
        storage.close()


def test_live_validated_session_cannot_be_approved_in_python_or_sqlite(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        storage.register_baseline_session("live-one", "live_validated", {"expected_sensor_topics": [f"{DEVICE}::{TOPIC}"]})
        storage.finalize_baseline_session("live-one")
        decision = curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["live-one"]
        )[0]
        assert not decision.approved
        assert "SOURCE_NOT_APPROVED_FOR_BASELINE" in decision.reasons
        with pytest.raises(sqlite3.IntegrityError, match="approved_clean_baseline"):
            storage._writers["normal"].submit_sync(
                lambda conn: conn.execute(
                    "UPDATE baseline_sessions SET approved_for_training=1 WHERE session_id='live-one'"
                )
            )
    finally:
        storage.close()


def test_attack_or_protocol_incident_session_is_rejected_and_indexed_separately(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        _create_session(storage, "poisoned-one")
        persist_incident_capture(storage, _incident_row("poisoned-one", "REPLAY_SEQUENCE_REJECTED"))
        decision = curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["poisoned-one"]
        )[0]
        assert not decision.approved
        assert "INCIDENT_PROTOCOL_OR_FORENSIC_CAPTURE_PRESENT" in decision.reasons
        corpus = storage.load_adversarial_regression_corpus(("protocol_violation",))
        assert len(corpus) == 1
        assert corpus[0]["reason_code"] == "REPLAY_SEQUENCE_REJECTED"
        conn = storage.connect_readonly("normal")
        try:
            status = conn.execute(
                "SELECT approved_for_training FROM baseline_sessions WHERE session_id='poisoned-one'"
            ).fetchone()[0]
            assert status == 0
        finally:
            conn.close()
    finally:
        storage.close()


def test_hardware_fault_transition_rejects_session_even_if_device_emitted_no_normal_row(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        storage.register_baseline_session(
            "faulty-one", "baseline_lab", {"expected_sensor_topics": [f"{DEVICE}::{TOPIC}"]}
        )
        # Transition must be considered for configured sensor even if it had
        # no admitted normal samples during the session.
        storage.append_policy_dimension_transition(
            DEVICE, 1, "operational", "HEALTHY", "FAULT", "FROZEN_SENSOR"
        )
        storage.finalize_baseline_session("faulty-one")
        decision = curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["faulty-one"]
        )[0]
        assert not decision.approved
        assert "HARDWARE_FAULT_TRANSITION:gas_sensor_01:FAULT" in decision.reasons
    finally:
        storage.close()


def test_training_rechecks_curation_digest_before_reading_approved_session(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        _create_session(storage, "integrity-one", count=20)
        _create_session(storage, "integrity-two", count=20)
        curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["integrity-one", "integrity-two"]
        )
        storage._writers["normal"].submit_sync(
            lambda conn: conn.execute(
                "UPDATE raw_events SET raw_payload='999.0', numeric_value=999.0 WHERE session_id='integrity-one' AND event_id=(SELECT min(event_id) FROM raw_events WHERE session_id='integrity-one')"
            )
        )
        with pytest.raises(ValueError, match="dataset digest mismatch"):
            _extract_windows_by_session(tmp_path / "db", seq_len=4)
    finally:
        storage.close()


def test_missing_sensor_coverage_prevents_clean_approval(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        storage.register_baseline_session(
            "coverage-one", "baseline_lab",
            {"expected_sensor_topics": [f"{DEVICE}::{TOPIC}", "pir_sensor_01::home/sensors/pir/motion"]},
        )
        for index in range(20):
            persist_normal_event(storage, _event_row("coverage-one", index))
        storage.finalize_baseline_session("coverage-one")
        decision = curate_clean_sessions(
            storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=["coverage-one"]
        )[0]
        assert not decision.approved
        assert "INSUFFICIENT_SENSOR_COVERAGE" in decision.reasons
    finally:
        storage.close()
