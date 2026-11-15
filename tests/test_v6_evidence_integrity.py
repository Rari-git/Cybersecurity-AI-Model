from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid

import pytest

from storage.persistence import persist_incident_capture, persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.dataset_governance import curate_clean_sessions
from training.session_integrity import load_verified_clean_session_rows

DEVICE = "gas_sensor_01"
TOPIC = "home/sensors/gas/ppm"
FP = hashlib.sha256(b"b5.5.6-evidence-test-cert").digest()


def _storage(tmp_path):
    return LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)


def _seed_clean(storage, requested_name):
    run_id = uuid.uuid4().hex
    session_id = f"baseline-lab-{run_id}-{requested_name}"
    storage.register_baseline_session(
        session_id, "baseline_lab",
        {"capture_run_id": run_id, "session_index": 1, "events_per_sensor": 18,
         "expected_sensor_topics": [f"{DEVICE}::{TOPIC}"], "compressed_time": False},
    )
    base = time.time_ns()
    for index in range(18):
        value = 350.0 + index * 0.01
        payload = f"{value:.3f}"
        raw = payload.encode("utf-8")
        persist_normal_event(storage, (
            session_id, DEVICE, DEVICE, 1, FP, TOPIC,
            base + index * 1_000_000_000, base + index * 1_000_000_000,
            len(raw), hashlib.sha256(raw).digest(), value, payload,
            "baseline_lab", "b5.5.6-test",
        ))
    storage.finalize_baseline_session(session_id)
    decision = curate_clean_sessions(
        storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=[session_id]
    )[0]
    assert decision.approved, decision.reasons
    conn = storage.connect_readonly("normal")
    try:
        digest = conn.execute(
            "SELECT dataset_sha256 FROM session_curation WHERE session_id=?", (session_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    return session_id, str(digest)


def _seed_incident(storage, session_id="incident-session"):
    storage.register_baseline_session(
        session_id, "live_validated", {"b5_5_6_test": True}, approved_for_training=False
    )
    scenario_id = f"b556-{uuid.uuid4().hex}"
    storage.register_regression_scenario(
        scenario_id, "attack", runner_version="b5.5.6-test",
        manifest={"scenario": "payload-integrity-test", "family": "test-family"},
    )
    raw = b"known-regression-payload"
    now = time.time_ns()
    persist_incident_capture(storage, (
        session_id, DEVICE, f"{DEVICE}-g2-test", 2, FP, TOPIC,
        f"shadow/sandbox/{DEVICE}/data", now, None,
        "shadow_sink", "SHADOW_QUARANTINE_ACTIVE", 0.95, None,
        len(raw), hashlib.sha256(raw).digest(), raw, 1,
        "confirmed_attack", scenario_id, now + 86_400_000_000_000,
    ))
    assert storage.complete_regression_scenario(scenario_id) == 1
    return scenario_id


def test_heldout_clean_session_matches_candidate_curation_digest(tmp_path):
    storage = _storage(tmp_path)
    try:
        session_id, digest = _seed_clean(storage, "integrity-test")
        rows = load_verified_clean_session_rows(storage, [session_id], {session_id: digest})
        assert len(rows[session_id]) == 18
    finally:
        storage.close()


def test_heldout_clean_payload_tampering_is_detected(tmp_path):
    storage = _storage(tmp_path)
    try:
        session_id, digest = _seed_clean(storage, "integrity-tamper")
        with pytest.raises(sqlite3.IntegrityError, match="approved clean-baseline telemetry is immutable"):
            storage._writers["normal"].submit_sync(lambda conn: conn.execute(
                "UPDATE raw_events SET raw_payload='999.999' WHERE session_id=?", (session_id,)))
        # Simulate an out-of-band DB-file writer that bypasses SQL triggers.
        storage._writers["normal"].submit_sync(
            lambda conn: conn.execute("DROP TRIGGER trg_curated_raw_events_immutable_update"))
        storage._writers["normal"].submit_sync(lambda conn: conn.execute(
            "UPDATE raw_events SET raw_payload='999.999' WHERE session_id=?", (session_id,)))
        with pytest.raises(RuntimeError, match="payload hash mismatch|dataset digest mismatch"):
            load_verified_clean_session_rows(storage, [session_id], {session_id: digest})
    finally:
        storage.close()


def test_candidate_manifest_digest_mismatch_is_rejected(tmp_path):
    storage = _storage(tmp_path)
    try:
        session_id, _ = _seed_clean(storage, "wrong-candidate-digest")
        with pytest.raises(RuntimeError, match="dataset digest mismatch"):
            load_verified_clean_session_rows(storage, [session_id], {session_id: "0" * 64})
    finally:
        storage.close()


def test_missing_heldout_curation_record_is_rejected(tmp_path):
    storage = _storage(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="missing curation records"):
            load_verified_clean_session_rows(storage, ["missing-session"], {"missing-session": "0" * 64})
    finally:
        storage.close()


def test_regression_capture_payload_hash_is_verified_on_load(tmp_path):
    storage = _storage(tmp_path)
    try:
        scenario_id = _seed_incident(storage)
        storage._writers["incidents"].submit_sync(
            lambda conn: conn.execute("DROP TRIGGER trg_incident_captures_immutable_update"))
        storage._writers["incidents"].submit_sync(lambda conn: conn.execute(
            "UPDATE incident_captures SET raw_payload=zeroblob(length(raw_payload)) "
            "WHERE scenario_id=?", (scenario_id,)))
        with pytest.raises(RuntimeError, match="payload hash mismatch"):
            storage.load_adversarial_regression_corpus(("adversarial",))
    finally:
        storage.close()


def test_incident_capture_rows_cannot_be_updated_or_deleted(tmp_path):
    storage = _storage(tmp_path)
    try:
        scenario_id = _seed_incident(storage)
        with pytest.raises(sqlite3.IntegrityError, match="immutable forensic evidence"):
            storage._writers["incidents"].submit_sync(lambda conn: conn.execute(
                "UPDATE incident_captures SET reason_code='CHANGED' WHERE scenario_id=?", (scenario_id,)))
        with pytest.raises(sqlite3.IntegrityError, match="immutable forensic evidence"):
            storage._writers["incidents"].submit_sync(lambda conn: conn.execute(
                "DELETE FROM incident_captures WHERE scenario_id=?", (scenario_id,)))
    finally:
        storage.close()


def test_completed_scenario_provenance_cannot_be_changed_or_deleted(tmp_path):
    storage = _storage(tmp_path)
    try:
        scenario_id = _seed_incident(storage)
        with pytest.raises(sqlite3.IntegrityError, match="provenance is immutable"):
            storage._writers["incidents"].submit_sync(lambda conn: conn.execute(
                "UPDATE regression_scenario_runs SET manifest_json='{}' WHERE scenario_id=?", (scenario_id,)))
        with pytest.raises(sqlite3.IntegrityError, match="provenance is immutable"):
            storage._writers["incidents"].submit_sync(lambda conn: conn.execute(
                "DELETE FROM regression_scenario_runs WHERE scenario_id=?", (scenario_id,)))
    finally:
        storage.close()


def test_closed_scenario_rejects_late_capture_insertion(tmp_path):
    storage = _storage(tmp_path)
    try:
        scenario_id = _seed_incident(storage)
        raw = b"late-capture"
        now = time.time_ns()
        with pytest.raises(sqlite3.IntegrityError, match="closed regression scenario"):
            persist_incident_capture(storage, (
                "incident-session", DEVICE, f"{DEVICE}-g2-test", 2, FP, TOPIC,
                f"shadow/sandbox/{DEVICE}/data", now, None,
                "shadow_sink", "SHADOW_QUARANTINE_ACTIVE", 0.95, None,
                len(raw), hashlib.sha256(raw).digest(), raw, 1,
                "confirmed_attack", scenario_id, now + 86_400_000_000_000,
            ))
    finally:
        storage.close()


def test_approved_curation_metadata_cannot_be_rewritten_or_deleted(tmp_path):
    storage = _storage(tmp_path)
    try:
        session_id, _ = _seed_clean(storage, "curation-lock")
        with pytest.raises(sqlite3.IntegrityError, match="approved clean-baseline curation is immutable"):
            storage._writers["normal"].submit_sync(lambda conn: conn.execute(
                "UPDATE session_curation SET dataset_sha256=? WHERE session_id=?", ("0" * 64, session_id)))
        with pytest.raises(sqlite3.IntegrityError, match="approved clean-baseline curation is immutable"):
            storage._writers["normal"].submit_sync(lambda conn: conn.execute(
                "DELETE FROM session_curation WHERE session_id=?", (session_id,)))
    finally:
        storage.close()
