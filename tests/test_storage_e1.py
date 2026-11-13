"""Etapa E1: Teste automate pentru cele 3 baze de date SQLite izolate,
AdmissionGate, proveniență, jurnalul hash-chained și enforcement_audit.
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.telemetry_db import LumenStorageManager, DBPaths
from storage.admission_gate import AdmissionGate, ProvenanceContext, parse_device_identity


@pytest.fixture()
def storage_env():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e1-"))
    paths = DBPaths.from_dir(tmp_dir)
    mgr = LumenStorageManager(paths=paths, enable_writers=True)
    mgr.register_baseline_session(
        session_id="sess-lab-001",
        source_type="baseline_lab",
        config={"rate_hz": 1.0},
        approved_for_training=False,
    )
    gate = AdmissionGate(
        storage=mgr,
        allowed_topics_by_device={
            "gas_sensor_01": {"home/sensors/gas/ppm"},
            "door_sensor_01": {"home/sensors/door/state"},
        },
    )
    try:
        yield mgr, gate, paths
    finally:
        mgr.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_e1_1_cn_parsing_preserves_stable_device_id():
    assert parse_device_identity("gas_sensor_01-g3-a9f81c2b") == ("gas_sensor_01", 3)
    assert parse_device_identity("door_sensor_01") == ("door_sensor_01", 1)
    with pytest.raises(ValueError):
        parse_device_identity("gas_sensor_01/../admin")


def test_e1_2_admission_gate_isolates_normal_vs_poisoned_data(storage_env):
    mgr, gate, _ = storage_env
    fp = hashlib.sha256(b"cert-gas-01-g2").digest()
    ctx = ProvenanceContext(
        session_id="sess-lab-001",
        device_cn="gas_sensor_01-g2-deadbeef",
        cert_fingerprint=fp,
        source_type="baseline_lab",
    )

    # 1. Eveniment normal valid -> admis în telemetry_normal.db
    res_ok = gate.evaluate_and_ingest(ctx, "home/sensors/gas/ppm", b"345.5")
    assert res_ok.admitted is True
    assert res_ok.device_id == "gas_sensor_01"
    assert res_ok.generation == 2

    # 2. Valoare fizic imposibilă (spoofing 9500 ppm) -> respins din normal, salvat în incidents
    res_spoof = gate.evaluate_and_ingest(ctx, "home/sensors/gas/ppm", b"9500.0")
    assert res_spoof.admitted is False
    assert "OUT_OF_PHYSICAL_BOUNDS" in res_spoof.reason

    # 3. Cross-topic injection -> respins din normal
    res_cross = gate.evaluate_and_ingest(ctx, "home/sensors/door/state", b"OPEN")
    assert res_cross.admitted is False
    assert res_cross.reason == "UNAUTHORIZED_TOPIC_FOR_DEVICE"

    # 4. Dispozitiv în carantină -> respins chiar dacă valoarea pare normală
    gate.set_device_blocked("gas_sensor_01", True)
    res_quar = gate.evaluate_and_ingest(ctx, "home/sensors/gas/ppm", b"350.0")
    assert res_quar.admitted is False
    assert res_quar.reason == "DEVICE_QUARANTINED_OR_INCIDENT"

    mgr.flush_all()

    # Verificăm izolarea fizică folosind conexiune read-only (?mode=ro)
    ro_normal = mgr.connect_readonly("normal")
    ro_incidents = mgr.connect_readonly("incidents")
    try:
        normal_rows = ro_normal.execute("SELECT * FROM raw_events").fetchall()
        incident_rows = ro_incidents.execute("SELECT * FROM incident_captures").fetchall()

        assert len(normal_rows) == 1
        assert normal_rows[0]["device_id"] == "gas_sensor_01"
        assert normal_rows[0]["numeric_value"] == pytest.approx(345.5)

        assert len(incident_rows) == 3
        # Confirmăm că mode=ro blochează orice tentativă de scriere din training
        with pytest.raises(sqlite3.OperationalError):
            ro_normal.execute("DELETE FROM raw_events")
    finally:
        ro_normal.close()
        ro_incidents.close()


def test_e1_3_security_control_nonces_hash_chain_and_pep_audit(storage_env):
    mgr, _, _ = storage_env
    fp = hashlib.sha256(b"cert-gas-01-g1").digest()
    mgr.upsert_device("gas_sensor_01", 1, "gas_sensor_01-g1-11223344", fp, "sensor", criticality="safety_critical")

    # 1. Anti-replay nonce atomic
    assert mgr.consume_nonce_atomic("gas_sensor_01", 1, "nonce-001") is True
    assert mgr.consume_nonce_atomic("gas_sensor_01", 1, "nonce-001") is False

    # 2. Tranziții de stare hash-chained + audit confirmare PEP
    mgr.record_enforcement_decision(
        decision_id="dec-001",
        device_id="gas_sensor_01",
        generation=1,
        old_state="ACTIVE",
        target_state="SHADOW_QUARANTINE",
        reason_code="HYBRID_ANOMALY_CONFIRMED",
        policy_version=1,
    )
    mgr.append_quarantine_transition(
        device_id="gas_sensor_01",
        generation=1,
        from_state="ACTIVE",
        to_state="ENFORCEMENT_PENDING",
        reason_code="HYBRID_ANOMALY_CONFIRMED",
        risk_score=0.92,
        policy_version=1,
    )
    latency_ms = mgr.confirm_enforcement_pep("dec-001", confirmed=True, detail="gateway_pep_active")
    assert latency_ms is not None and latency_ms >= 0.0

    mgr.append_quarantine_transition(
        device_id="gas_sensor_01",
        generation=1,
        from_state="ENFORCEMENT_PENDING",
        to_state="SHADOW_QUARANTINE",
        reason_code="PEP_CONFIRMED",
        risk_score=0.92,
        policy_version=1,
    )

    chain_ok, count = mgr.verify_quarantine_chain()
    assert chain_ok is True
    assert count == 2