"""Runtime regression coverage for E1 authenticated envelopes, E2 policy state and E3 admission."""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agent.pep_protocol import PepAckRegistry, PepAcknowledgement
from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
from agent.forensic_sink import ForensicQuarantineSink
from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.telemetry_db import DBPaths, LumenStorageManager
from utils.device_identity import ActiveDeviceIdentity
from utils.identity_envelope import (
    AuthenticatedTelemetryEnvelope,
    validate_active_identity, validate_envelope_route,
)

ROOT = Path(__file__).resolve().parents[1]
SHARED_GAS_TOPIC = "home/sensors/gas/ppm"


def _identity(device_id: str, generation: int = 3, marker: bytes | None = None):
    fp = hashlib.sha256(marker or f"cert:{device_id}:{generation}".encode()).digest()
    return ActiveDeviceIdentity(
        device_id=device_id,
        generation=generation,
        common_name=f"{device_id}-g{generation}-cafe1234",
        certificate_fingerprint=fp,
        serial_number_hex=f"{generation + 100:x}",
    )


def _ev(device="gas_sensor_01", **changes):
    values = dict(
        device_id=device,
        generation=1,
        topic=SHARED_GAS_TOPIC,
        lstm_score=2.0,
        lstm_threshold=1.0,
        lstm_anomalous=True,
        cusum_score=8.0,
        cusum_triggered=True,
        operational_status="HEALTHY",
        fault_type="NONE",
    )
    values.update(changes)
    return DetectionEvidence(**values)


@pytest.fixture
def storage(tmp_path):
    manager = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        yield manager
    finally:
        manager.close()


def test_e1_dual_gas_devices_keep_independent_identity_on_shared_topic():
    id1 = _identity("gas_sensor_01")
    id2 = _identity("gas_sensor_02")
    env1 = AuthenticatedTelemetryEnvelope.create(
        device_id=id1.device_id, generation=id1.generation,
        certificate_common_name=id1.common_name,
        certificate_fingerprint=id1.certificate_fingerprint,
        original_topic=SHARED_GAS_TOPIC, payload=b"351", gateway_instance_id="gateway-run-1",
    )
    env2 = AuthenticatedTelemetryEnvelope.create(
        device_id=id2.device_id, generation=id2.generation,
        certificate_common_name=id2.common_name,
        certificate_fingerprint=id2.certificate_fingerprint,
        original_topic=SHARED_GAS_TOPIC, payload=b"347", gateway_instance_id="gateway-run-1",
    )
    read1, payload1 = AuthenticatedTelemetryEnvelope.decode(env1.encode())
    read2, payload2 = AuthenticatedTelemetryEnvelope.decode(env2.encode())
    validate_active_identity(read1, id1)
    validate_active_identity(read2, id2)
    validate_envelope_route(read1, "internal/ids/telemetry/gas_sensor_01", {"gas_sensor_01": SHARED_GAS_TOPIC})
    with pytest.raises(ValueError, match="key/device_id"):
        validate_envelope_route(read1, "internal/ids/telemetry/gas_sensor_02", {"gas_sensor_01": SHARED_GAS_TOPIC})
    assert (read1.device_id, payload1) == ("gas_sensor_01", b"351")
    assert (read2.device_id, payload2) == ("gas_sensor_02", b"347")
    assert read1.original_topic == read2.original_topic == SHARED_GAS_TOPIC
    assert read1.device_id != read2.device_id


def test_e1_forged_and_stale_envelopes_are_rejected():
    active = _identity("gas_sensor_01", generation=4)
    stale = _identity("gas_sensor_01", generation=3)
    old_env = AuthenticatedTelemetryEnvelope.create(
        device_id=stale.device_id, generation=stale.generation,
        certificate_common_name=stale.common_name,
        certificate_fingerprint=stale.certificate_fingerprint,
        original_topic=SHARED_GAS_TOPIC, payload=b"310", gateway_instance_id="gateway-run-1",
    )
    with pytest.raises(ValueError, match="forged or stale"):
        validate_active_identity(old_env, active)

    forged = AuthenticatedTelemetryEnvelope.create(
        device_id=active.device_id, generation=active.generation,
        certificate_common_name=active.common_name,
        certificate_fingerprint=hashlib.sha256(b"attacker cert").digest(),
        original_topic=SHARED_GAS_TOPIC, payload=b"999", gateway_instance_id="gateway-run-1",
    )
    with pytest.raises(ValueError, match="forged or stale"):
        validate_active_identity(forged, active)


def test_e1_internal_telemetry_acl_is_gateway_only_and_ids_subscribes():
    for config_name in ("zenohd_router.json5", "zenoh_acl_example.json5"):
        text = (ROOT / "config" / config_name).read_text(encoding="utf-8")
        assert 'default_permission: "deny"' in text
        # Both internal telemetry and PEP ACK ingress rules have Gateway as their
        # only subject; IDS receives them only through explicit subscriber/egress rules.
        for rule in ("in_put_internal_ids_telemetry", "in_put_pep_ack"):
            assert f'"{rule}"' in text
            assigned = re.findall(r'\{ rules: \[([^]]*\b' + rule + r'\b[^]]*)\], subjects: \[([^]]*)\] \}', text)
            assert assigned, f"missing policy assignment for {rule} in {config_name}"
            assert all('"gateway"' in subjects for _, subjects in assigned)
        assert '"sub_internal_ids_telemetry"' in text
        assert '"sub_pep_ack"' in text


def test_e1_ids_agent_unwraps_envelope_and_rejects_stale_pki(monkeypatch):
    """Exercise the actual IDS ingress validator without requiring Zenoh/ONNX binaries."""
    fake_ort = ModuleType("onnxruntime")
    fake_ort.InferenceSession = object
    fake_zenoh = ModuleType("zenoh")
    fake_zenoh.__path__ = []
    fake_zenoh.Session = object
    fake_zenoh.SampleKind = SimpleNamespace(DELETE=object())
    fake_zenoh.Priority = SimpleNamespace(REAL_TIME=1)
    fake_handlers = ModuleType("zenoh.handlers")
    fake_handlers.FifoChannel = object
    fake_handlers.RingChannel = object
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    monkeypatch.setitem(sys.modules, "zenoh.handlers", fake_handlers)

    from agent import ids_agent

    active = _identity("gas_sensor_01", generation=5)
    monkeypatch.setattr(ids_agent, "load_active_device_identity", lambda _pointer, _device: active)
    agent = ids_agent.ZenohIDSAgent.__new__(ids_agent.ZenohIDSAgent)
    agent.cert_pointer = "/test/active-pki-pointer.json"
    agent._device_to_topic = {"gas_sensor_01": SHARED_GAS_TOPIC}

    envelope = AuthenticatedTelemetryEnvelope.create(
        device_id=active.device_id, generation=active.generation,
        certificate_common_name=active.common_name,
        certificate_fingerprint=active.certificate_fingerprint,
        original_topic=SHARED_GAS_TOPIC, payload=b"352",
        gateway_instance_id="gateway-run-1",
    )
    key = "internal/ids/telemetry/gas_sensor_01"
    decoded, payload = agent._validate_identity_envelope(key, envelope.encode())
    assert decoded.device_id == "gas_sensor_01"
    assert payload == b"352"

    stale = _identity("gas_sensor_01", generation=4)
    stale_envelope = AuthenticatedTelemetryEnvelope.create(
        device_id=stale.device_id, generation=stale.generation,
        certificate_common_name=stale.common_name,
        certificate_fingerprint=stale.certificate_fingerprint,
        original_topic=SHARED_GAS_TOPIC, payload=b"352",
        gateway_instance_id="gateway-run-1",
    )
    with pytest.raises(ValueError, match="forged or stale"):
        agent._validate_identity_envelope(key, stale_envelope.encode())


def test_e2_pep_ack_rejects_wrong_duplicate_and_missing_ack():
    registry = PepAckRegistry()
    registry.register("decision-1", "gas_sensor_01", "SHADOW_QUARANTINE")
    wrong_device = PepAcknowledgement(
        "decision-1", "gas_sensor_02", "SHADOW_QUARANTINE", "SHADOW_QUARANTINE",
        "CONFIRMED", time.time_ns(), "gateway-a",
    )
    wrong_applied_action = PepAcknowledgement(
        "decision-1", "gas_sensor_01", "SHADOW_QUARANTINE", "ACTIVE",
        "CONFIRMED", time.time_ns(), "gateway-a",
    )
    assert registry.accept(wrong_device) is False
    assert registry.accept(wrong_applied_action) is False
    accepted = PepAcknowledgement(
        "decision-1", "gas_sensor_01", "SHADOW_QUARANTINE", "SHADOW_QUARANTINE",
        "CONFIRMED", time.time_ns(), "gateway-a",
    )
    assert registry.accept(accepted) is True
    assert registry.accept(accepted) is False
    assert registry.wait("decision-1", 0.01).confirmed is True

    registry.register("decision-timeout", "door_sensor_01", "ACTIVE")
    assert registry.wait("decision-timeout", 0.001) is None


def test_e2_behavioral_violations_quarantine_but_never_revoke(storage):
    revoked = []
    storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01-g1-aabb", hashlib.sha256(b"x").digest(), "sensor")
    engine = RiskPolicyEngine(
        storage=storage,
        pep_request_fn=lambda dev, enabled, decision_id: True,
        revocation_fn=lambda dev, reason: revoked.append((dev, reason)) or True,
        window_n=3,
        trigger_k=2,
    )
    first = engine.evaluate_evidence(_ev(deterministic_violation=True, deterministic_reason="FLOOD_RATE"))
    second = engine.evaluate_evidence(_ev(deterministic_violation=True, deterministic_reason="FLOOD_RATE"))
    assert first.new_state == "MONITORING"
    assert second.new_state == "SHADOW_QUARANTINE"
    assert revoked == []


def test_e2_revoked_state_requires_verified_revocation_callback(storage):
    fp = hashlib.sha256(b"revocation test").digest()
    storage.upsert_device("door_sensor_01", 1, "door_sensor_01", fp, "sensor")
    failed_engine = RiskPolicyEngine(
        storage=storage, pep_apply_fn=lambda _d, _e: True,
        revocation_fn=lambda _d, _r: False,
    )
    failed = failed_engine.evaluate_evidence(
        _ev("door_sensor_01", revocation_requested=True, deterministic_reason="CERT_COMPROMISE")
    )
    assert failed.new_state == "ENFORCEMENT_FAILED"
    assert failed_engine.get_state("door_sensor_01") != "REVOKED"

    verified_engine = RiskPolicyEngine(
        storage=storage, pep_apply_fn=lambda _d, _e: True,
        revocation_fn=lambda _d, _r: True,
    )
    verified = verified_engine.evaluate_evidence(
        _ev("door_sensor_01", revocation_requested=True, deterministic_reason="CERT_COMPROMISE")
    )
    assert verified.new_state == "REVOKED"
    assert verified.confirmed_by_pep is True


def test_e2_policy_state_and_hysteresis_survive_restart(storage):
    fp = hashlib.sha256(b"restart state").digest()
    storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01", fp, "sensor")
    engine = RiskPolicyEngine(storage=storage, pep_apply_fn=lambda _d, _e: True, window_n=3, trigger_k=3)
    engine.evaluate_evidence(_ev(deterministic_violation=True))
    engine.evaluate_evidence(_ev(deterministic_violation=True))
    assert engine.get_state("gas_sensor_01") == "MONITORING"
    storage.flush_all()

    restored = RiskPolicyEngine(storage=storage, pep_apply_fn=lambda _d, _e: True, window_n=3, trigger_k=3)
    assert restored.get_state("gas_sensor_01") == "MONITORING"
    decision = restored.evaluate_evidence(_ev(deterministic_violation=True))
    assert decision.new_state == "SHADOW_QUARANTINE"
    restarted_again = RiskPolicyEngine(storage=storage, pep_apply_fn=lambda _d, _e: True)
    assert restarted_again.get_state("gas_sensor_01") == "SHADOW_QUARANTINE"


def test_e2_device_upsert_does_not_reset_quarantine_or_revocation(storage):
    fp = hashlib.sha256(b"state retention").digest()
    storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01", fp, "sensor")
    storage.append_quarantine_transition("gas_sensor_01", 1, "ACTIVE", "SHADOW_QUARANTINE", "test", 1.0, 1)
    storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01", fp, "sensor", state="ACTIVE")
    assert storage.load_device_states()["gas_sensor_01"] == "SHADOW_QUARANTINE"


def test_e2_fault_returns_to_active_after_configured_clean_windows(storage):
    fp = hashlib.sha256(b"fault recovery").digest()
    storage.upsert_device("door_sensor_01", 1, "door_sensor_01", fp, "sensor")
    engine = RiskPolicyEngine(
        storage=storage, pep_apply_fn=lambda _d, _e: True,
        fault_recovery_clean_windows=2,
    )
    fault = engine.evaluate_evidence(_ev("door_sensor_01", operational_status="FAULT", fault_type="STUCK"))
    assert fault.new_state == "FAULT"
    clean1 = engine.evaluate_evidence(_ev("door_sensor_01", operational_status="HEALTHY", lstm_anomalous=False, cusum_triggered=False))
    assert clean1.new_state == "FAULT"
    clean2 = engine.evaluate_evidence(_ev("door_sensor_01", operational_status="HEALTHY", lstm_anomalous=False, cusum_triggered=False))
    assert clean2.new_state == "ACTIVE"


def test_e3_live_validated_session_is_never_baseline_training_approved(storage):
    session_id = "live-gateway-run"
    storage.register_baseline_session(
        session_id=session_id,
        source_type="live_validated",
        config={"approved_for_training": False},
        approved_for_training=False,
    )
    fp = hashlib.sha256(b"live cert").digest()
    gate = AdmissionGate(storage, {"gas_sensor_01": {SHARED_GAS_TOPIC}})
    ctx = ProvenanceContext(
        session_id=session_id, device_cn="gas_sensor_01-g2-cafe1234",
        cert_fingerprint=fp, source_type="live_validated",
        device_id="gas_sensor_01", generation=2,
    )
    result = gate.evaluate_and_ingest(ctx, SHARED_GAS_TOPIC, b"350")
    assert result.admitted is True
    storage.finalize_baseline_session(session_id, approved_for_training=True)
    storage.flush_all()
    conn = storage.connect_readonly("normal")
    try:
        session = conn.execute(
            "SELECT source_type, approved_for_training FROM baseline_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        assert session["source_type"] == "live_validated"
        assert session["approved_for_training"] == 0
        assert session["source_type"] == "live_validated"
    finally:
        conn.close()


def test_e3_admission_and_forensic_storage_failures_propagate(monkeypatch, storage):
    import storage.admission_gate as admission_module
    import agent.forensic_sink as sink_module

    session_id = "live-failure-run"
    storage.register_baseline_session(session_id, "live_validated", {}, approved_for_training=False)
    fp = hashlib.sha256(b"live failure cert").digest()
    ctx = ProvenanceContext(
        session_id=session_id, device_cn="gas_sensor_01-g1-cafe1234",
        cert_fingerprint=fp, source_type="live_validated",
        device_id="gas_sensor_01", generation=1,
    )
    gate = AdmissionGate(storage, {"gas_sensor_01": {SHARED_GAS_TOPIC}})
    monkeypatch.setattr(admission_module, "persist_normal_event", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("normal writer unavailable")))
    with pytest.raises(RuntimeError, match="normal writer unavailable"):
        gate.evaluate_and_ingest(ctx, SHARED_GAS_TOPIC, b"350")

    gate.set_device_blocked("gas_sensor_01", True)
    monkeypatch.setattr(admission_module, "persist_incident_capture", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("incident queue full")))
    with pytest.raises(RuntimeError, match="incident queue full"):
        gate.evaluate_and_ingest(ctx, SHARED_GAS_TOPIC, b"350")

    class FailingSinkStorage:
        def _writers(self):
            return None

    sink = ForensicQuarantineSink(storage=storage)
    monkeypatch.setattr(sink_module, "persist_incident_capture", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forensic commit failed")))
    with pytest.raises(RuntimeError, match="forensic commit failed"):
        sink.ingest_shadow_envelope(
            session_id, "gas_sensor_01", "gas_sensor_01-g1-cafe1234", 1, fp,
            SHARED_GAS_TOPIC, b"x", 0.9, "TEST", now_mono=10.0,
        )
    assert sink.stored_count == 0
    assert sink._bytes_used.get("gas_sensor_01", 0) == 0


def test_e2_gateway_applies_policy_before_publishing_correlated_ack(monkeypatch):
    import json
    import sys
    import threading
    from types import ModuleType, SimpleNamespace
    # The callback under test does not need a Zenoh runtime; provide only the
    # module symbol required at Gateway import time on this test-only host.
    monkeypatch.setitem(sys.modules, "zenoh", SimpleNamespace(
        SampleKind=SimpleNamespace(DELETE=object()),
        Priority=SimpleNamespace(REAL_TIME=1),
        Session=object,
    ))
    from agent.gateway_ids import ZenohInlineGateway

    class FakeSession:
        def __init__(self):
            self.published = []
        def put(self, key, payload):
            self.published.append((key, payload))

    class FakeAdmissionGate:
        def __init__(self):
            self.blocked = {}
        def set_device_blocked(self, device_id, blocked):
            self.blocked[device_id] = blocked

    gateway = ZenohInlineGateway.__new__(ZenohInlineGateway)
    gateway._shadow_lock = threading.RLock()
    gateway._stats_lock = threading.RLock()
    gateway._shadow_quarantined = set()
    gateway._release_prepared = {}
    gateway.stats = {}
    gateway.admission_gate = FakeAdmissionGate()
    gateway.run_id = "gateway-test-1"
    gateway.session = FakeSession()
    command = {
        "decision_id": "dec-test-01",
        "device_id": "gas_sensor_01",
        "target_state": "SHADOW_QUARANTINE",
    }
    gateway._on_policy_command(SimpleNamespace(
        key_expr="control/policy/gas_sensor_01",
        payload=json.dumps(command).encode(),
    ))
    assert gateway.is_shadow_quarantined("gas_sensor_01") is True
    assert gateway.admission_gate.blocked["gas_sensor_01"] is True
    assert len(gateway.session.published) == 1
    key, raw_ack = gateway.session.published[0]
    ack = json.loads(raw_ack.decode())
    assert key == "control/pep_ack/dec-test-01"
    assert ack["decision_id"] == command["decision_id"]
    assert ack["device_id"] == command["device_id"]
    assert ack["requested_action"] == "SHADOW_QUARANTINE"
    assert ack["applied_action"] == "SHADOW_QUARANTINE"
    assert ack["result"] == "CONFIRMED"
    assert ack["gateway_instance_id"] == "gateway-test-1"


def test_e3_live_session_registration_rejects_training_approval(storage):
    with pytest.raises(ValueError, match="cannot be approved"):
        storage.register_baseline_session(
            "bad-live-session", "live_validated", {}, approved_for_training=True,
        )
