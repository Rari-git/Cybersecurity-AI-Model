from __future__ import annotations

import hashlib

import pytest

from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
from storage.telemetry_db import DBPaths, LumenStorageManager


def _evidence(device_id: str = "gas_sensor_01", **overrides) -> DetectionEvidence:
    values = dict(
        device_id=device_id,
        generation=3,
        topic="home/sensors/gas/ppm",
        lstm_score=0.2,
        lstm_threshold=1.0,
        lstm_anomalous=False,
        cusum_score=0.1,
        cusum_triggered=False,
        operational_status="HEALTHY",
        fault_type="NONE",
    )
    values.update(overrides)
    return DetectionEvidence(**values)


@pytest.fixture()
def policy_storage(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    storage.upsert_device(
        "gas_sensor_01", 3, "gas_sensor_01-g3-cafe1234",
        hashlib.sha256(b"gas-cert").digest(), "sensor",
    )
    try:
        yield storage
    finally:
        storage.close()


def test_operational_fault_does_not_clear_existing_security_quarantine(policy_storage):
    pep_events = []
    engine = RiskPolicyEngine(
        storage=policy_storage,
        pep_apply_fn=lambda dev, enable: pep_events.append((dev, enable)) or True,
    )
    quarantined = engine.evaluate_evidence(_evidence(
        protocol_version=6, boot_id="0123456789abcdef0123456789abcdef",
        sequence=9, device_timestamp_ns=1_700_000_000_000_000_009,
        protocol_violation=True, admission_status="REJECTED",
        deterministic_violation=True, deterministic_reason="ADMISSION_REJECTED:REPLAY_OR_OUT_OF_ORDER_SEQUENCE",
    ))
    assert quarantined.security_status == "QUARANTINED"
    assert quarantined.new_state == "SHADOW_QUARANTINE"
    assert pep_events == [("gas_sensor_01", True)]

    faulted = engine.evaluate_evidence(_evidence(
        operational_status="FAULT", fault_type="FROZEN_SENSOR",
        lstm_anomalous=True, cusum_triggered=True, lstm_score=5.0, cusum_score=9.0,
    ))
    assert faulted.operational_status == "FAULT"
    assert faulted.security_status == "QUARANTINED"
    assert faulted.new_state == "SHADOW_QUARANTINE"
    assert engine.get_operational_status("gas_sensor_01") == "FAULT"
    assert engine.get_security_status("gas_sensor_01") == "QUARANTINED"
    assert pep_events == [("gas_sensor_01", True)]  # No implicit release on hardware fault.
    assert policy_storage.load_device_states()["gas_sensor_01"] == "SHADOW_QUARANTINE"


def test_protocol_violation_during_fault_still_triggers_quarantine(policy_storage):
    pep_events = []
    engine = RiskPolicyEngine(
        storage=policy_storage,
        pep_apply_fn=lambda dev, enable: pep_events.append((dev, enable)) or True,
    )
    result = engine.evaluate_evidence(_evidence(
        operational_status="FAULT", fault_type="PACKET_DROP",
        protocol_version=6, boot_id="0123456789abcdef0123456789abcdef",
        sequence=4, device_timestamp_ns=1_700_000_000_000_000_004,
        protocol_violation=True, admission_status="REJECTED",
        deterministic_violation=True, deterministic_reason="ADMISSION_REJECTED:STALE_BOOT_ID",
    ))
    assert result.action == "QUARANTINE_SHADOW"
    assert result.operational_status == "FAULT"
    assert result.security_status == "QUARANTINED"
    assert result.new_state == "SHADOW_QUARANTINE"
    assert pep_events == [("gas_sensor_01", True)]


def test_operational_and_security_histories_are_separate_and_hash_chained(policy_storage):
    engine = RiskPolicyEngine(storage=policy_storage, pep_apply_fn=lambda _d, _e: True)
    # One behavioral signal establishes SUSPECT.
    suspect = engine.evaluate_evidence(_evidence(lstm_anomalous=True, lstm_score=2.0))
    assert suspect.security_status == "SUSPECT"

    # A hardware fault changes only operational status, preserving SUSPECT.
    fault = engine.evaluate_evidence(_evidence(operational_status="FAULT", fault_type="FROZEN_SENSOR"))
    assert fault.operational_status == "FAULT"
    assert fault.security_status == "SUSPECT"

    security_log = policy_storage.load_policy_dimension_transitions("gas_sensor_01", "security")
    operational_log = policy_storage.load_policy_dimension_transitions("gas_sensor_01", "operational")
    assert [(row["from_state"], row["to_state"]) for row in security_log] == [("ACTIVE", "MONITORING")]
    assert [(row["from_state"], row["to_state"]) for row in operational_log] == [("HEALTHY", "FAULT")]
    assert security_log[0]["prev_hash"] == "0" * 64
    assert operational_log[0]["prev_hash"] == "0" * 64


def test_orthogonal_dimensions_survive_policy_engine_restart(policy_storage):
    engine = RiskPolicyEngine(storage=policy_storage, pep_apply_fn=lambda _d, _e: True)
    engine.evaluate_evidence(_evidence(
        operational_status="FAULT", fault_type="FROZEN_SENSOR",
        protocol_violation=True, admission_status="REJECTED",
        deterministic_violation=True, deterministic_reason="REPLAY",
    ))
    policy_storage.flush_all()

    restored = RiskPolicyEngine(storage=policy_storage, pep_apply_fn=lambda _d, _e: True)
    assert restored.get_operational_status("gas_sensor_01") == "FAULT"
    assert restored.get_security_status("gas_sensor_01") == "QUARANTINED"
    assert restored.get_security_phase("gas_sensor_01") == "SHADOW_QUARANTINE"


def test_degraded_health_is_independent_of_security_status(policy_storage):
    engine = RiskPolicyEngine(storage=policy_storage, pep_apply_fn=lambda _d, _e: True)
    degraded = engine.evaluate_evidence(_evidence(
        operational_status="DEGRADED", fault_type="PACKET_LOSS_WARNING",
    ))
    assert degraded.operational_status == "DEGRADED"
    assert degraded.security_status == "ACTIVE"
    assert engine.get_operational_status("gas_sensor_01") == "DEGRADED"
    assert engine.get_security_status("gas_sensor_01") == "ACTIVE"


def test_protocol_replay_during_recovery_pending_reasserts_quarantine(policy_storage):
    pep_events = []
    engine = RiskPolicyEngine(
        storage=policy_storage,
        pep_apply_fn=lambda dev, enable: pep_events.append((dev, enable)) or True,
        recovery_clean_windows=1,
    )
    initial = engine.evaluate_evidence(_evidence(
        protocol_version=6, protocol_violation=True, admission_status="REJECTED",
        deterministic_violation=True, deterministic_reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE",
    ))
    assert initial.security_status == "QUARANTINED"
    assert engine.get_security_phase("gas_sensor_01") == "SHADOW_QUARANTINE"

    recovery = engine.evaluate_evidence(_evidence())
    assert recovery.security_status == "QUARANTINED"
    assert engine.get_security_phase("gas_sensor_01") == "RECOVERY_PENDING"

    replay_during_recovery = engine.evaluate_evidence(_evidence(
        protocol_version=6, protocol_violation=True, admission_status="REJECTED",
        deterministic_violation=True, deterministic_reason="REPLAY_DURING_RELEASE",
    ))
    assert replay_during_recovery.security_status == "QUARANTINED"
    assert engine.get_security_phase("gas_sensor_01") == "SHADOW_QUARANTINE"
    assert pep_events == [
        ("gas_sensor_01", True),
        ("gas_sensor_01", True),
    ]
