from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from agent.forensic_sink import ForensicQuarantineSink
from agent.gateway_ids import ZenohInlineGateway
from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.persistence import persist_incident_capture
from storage.telemetry_db import DBPaths, LumenStorageManager

DEVICE = "gas_sensor_01"
TOPIC = "home/sensors/gas/ppm"
FP = hashlib.sha256(b"stage-b-regression-test-cert").digest()
BOOT_ID = "0123456789abcdef0123456789abcdef"


def _capture_row(session_id, *, source, reason, scenario_id=None, label=None):
    raw = b"stage-b-controlled-payload"
    now = time.time_ns()
    return (
        session_id, DEVICE, f"{DEVICE}-g2-test", 2, FP, TOPIC,
        f"shadow/sandbox/{DEVICE}/data", now, None, source, reason, 0.95, None,
        len(raw), hashlib.sha256(raw).digest(), raw, 1, label, scenario_id,
        now + 14 * 24 * 3600 * 1_000_000_000,
    )


def _storage(tmp_path):
    return LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)


def _live_session(storage, session_id):
    storage.register_baseline_session(
        session_id, "live_validated", {"stage_b_regression_test": True}, approved_for_training=False
    )


def test_unqualified_incident_is_preserved_but_never_indexed(tmp_path):
    storage = _storage(tmp_path)
    try:
        _live_session(storage, "unqualified-live")
        persist_incident_capture(
            storage,
            _capture_row("unqualified-live", source="ids_alert",
                         reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE"),
        )
        assert storage.load_adversarial_regression_corpus(
            ("adversarial", "protocol_violation", "hardware_fault")
        ) == []
        conn = storage.connect_readonly("incidents")
        try:
            assert conn.execute("SELECT COUNT(*) FROM incident_captures").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM adversarial_regression_corpus").fetchone()[0] == 0
        finally:
            conn.close()
    finally:
        storage.close()


def test_protocol_capture_is_qualified_only_after_completed_scenario(tmp_path):
    storage = _storage(tmp_path)
    try:
        session_id = "protocol-live"
        scenario_id = "protocol-replay-test-001"
        _live_session(storage, session_id)
        storage.register_regression_scenario(
            scenario_id, "protocol", runner_version="pytest-stage-b1",
            manifest={"scenario": "authenticated-v6-replay", "sequence": 1, "expected": "reject"},
        )
        gate = AdmissionGate(storage, {DEVICE: {TOPIC}})
        context = ProvenanceContext(
            session_id=session_id, device_cn=f"{DEVICE}-g2-test", cert_fingerprint=FP,
            source_type="live_validated", device_id=DEVICE, generation=2,
            scenario_id=scenario_id,
        )
        gate.register_verified_device_boot(context, BOOT_ID)
        frame = (
            '{"protocol_version":6,"device_id":"gas_sensor_01",'
            f'"boot_id":"{BOOT_ID}","sequence":1,'
            '"device_timestamp_ns":1700000000000000001,"payload":"350"}'
        )
        assert gate.evaluate_v6_and_ingest(context, TOPIC, frame).admitted
        rejected = gate.evaluate_v6_and_ingest(context, TOPIC, frame)
        assert not rejected.admitted
        assert rejected.reason == "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
        assert storage.load_adversarial_regression_corpus(("protocol_violation",)) == []
        assert storage.complete_regression_scenario(scenario_id) == 1
        corpus = storage.load_adversarial_regression_corpus(("protocol_violation",))
        assert len(corpus) == 1
        assert corpus[0]["scenario_id"] == scenario_id
        assert corpus[0]["capture_source"] == "ids_alert"
        assert corpus[0]["reason_code"] == "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
        assert len(corpus[0]["provenance_sha256"]) == 64
        assert corpus[0]["scenario_type"] == "protocol"
    finally:
        storage.close()


def test_protocol_scenario_with_unqualified_reason_fails_closed(tmp_path):
    storage = _storage(tmp_path)
    try:
        _live_session(storage, "bad-protocol-live")
        scenario_id = "protocol-invalid-frame-test"
        storage.register_regression_scenario(
            scenario_id, "protocol", runner_version="pytest-stage-b1",
            manifest={"scenario": "malformed-frame", "expected": "protocol regression"},
        )
        persist_incident_capture(
            storage,
            _capture_row("bad-protocol-live", source="ids_alert",
                         reason="INVALID_V6_FRAME:bad-json", scenario_id=scenario_id),
        )
        assert storage.complete_regression_scenario(scenario_id) == 0
        assert storage.load_adversarial_regression_corpus(("protocol_violation",)) == []
        conn = storage.connect_readonly("incidents")
        try:
            status = conn.execute(
                "SELECT status FROM regression_scenario_runs WHERE scenario_id=?", (scenario_id,)
            ).fetchone()[0]
            assert status == "failed"
        finally:
            conn.close()
    finally:
        storage.close()


def test_attack_and_hardware_fault_require_explicit_scenario_and_label(tmp_path):
    storage = _storage(tmp_path)
    try:
        _live_session(storage, "attack-live")
        _live_session(storage, "fault-live")
        storage.register_regression_scenario(
            "attack-scenario-001", "attack", runner_version="pytest-stage-b1",
            manifest={"scenario": "controlled-quarantined-attack", "seed": 43},
        )
        persist_incident_capture(
            storage,
            _capture_row("attack-live", source="shadow_sink", reason="SHADOW_QUARANTINE_ACTIVE",
                         scenario_id="attack-scenario-001", label="confirmed_attack"),
        )
        assert storage.complete_regression_scenario("attack-scenario-001") == 1

        storage.register_regression_scenario(
            "fault-scenario-001", "hardware_fault", runner_version="pytest-stage-b1",
            manifest={"scenario": "controlled-sensor-freeze", "seed": 43},
        )
        persist_incident_capture(
            storage,
            _capture_row("fault-live", source="fault_lab", reason="SENSOR_FROZEN",
                         scenario_id="fault-scenario-001", label="confirmed_hardware_fault"),
        )
        assert storage.complete_regression_scenario("fault-scenario-001") == 1
        rows = storage.load_adversarial_regression_corpus(("adversarial", "hardware_fault"))
        assert {row["corpus_class"] for row in rows} == {"adversarial", "hardware_fault"}
    finally:
        storage.close()


def test_scenario_failure_never_enters_regression_corpus(tmp_path):
    storage = _storage(tmp_path)
    try:
        _live_session(storage, "failed-attack-live")
        storage.register_regression_scenario(
            "failed-attack-scenario", "attack", runner_version="pytest-stage-b1",
            manifest={"scenario": "attack-lab-failed-check"},
        )
        persist_incident_capture(
            storage,
            _capture_row("failed-attack-live", source="attack_lab", reason="DOS_TEST",
                         scenario_id="failed-attack-scenario", label="confirmed_attack"),
        )
        assert storage.complete_regression_scenario("failed-attack-scenario", success=False) == 0
        assert storage.load_adversarial_regression_corpus(("adversarial",)) == []
    finally:
        storage.close()


def test_scenario_manifest_digest_is_verified_when_loading_corpus(tmp_path):
    storage = _storage(tmp_path)
    try:
        _live_session(storage, "digest-live")
        scenario_id = "digest-protocol-scenario"
        storage.register_regression_scenario(
            scenario_id, "protocol", runner_version="pytest-stage-b1",
            manifest={"scenario": "replay", "seed": 9},
        )
        persist_incident_capture(
            storage,
            _capture_row("digest-live", source="ids_alert", reason="STALE_BOOT_ID",
                         scenario_id=scenario_id),
        )
        assert storage.complete_regression_scenario(scenario_id) == 1
        def _simulate_out_of_band_tamper(conn):
            conn.execute("DROP TRIGGER trg_regression_scenario_transition_only")
            conn.execute(
                "UPDATE regression_scenario_runs SET manifest_json='{}' WHERE scenario_id=?",
                (scenario_id,),
            )
        storage._writers["incidents"].submit_sync(_simulate_out_of_band_tamper)
        with pytest.raises(RuntimeError, match="provenance digest mismatch"):
            storage.load_adversarial_regression_corpus(("protocol_violation",))
    finally:
        storage.close()


def test_gateway_shadow_route_requires_quarantine_and_preserves_scenario_provenance(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    try:
        session_id = "gateway-shadow-live"
        scenario_id = "gateway-shadow-attack-001"
        _live_session(storage, session_id)
        storage.register_regression_scenario(
            scenario_id, "attack", runner_version="pytest-stage-b1",
            manifest={"scenario": "gateway-quarantined-traffic", "seed": 17},
        )

        class Identity:
            common_name = f"{DEVICE}-g2-test"
            generation = 2
            certificate_fingerprint = FP

        class LocalSession:
            def __init__(self):
                self.messages = []
            def put(self, topic, payload):
                self.messages.append((topic, payload))

        monkeypatch.setattr("agent.gateway_ids.load_active_device_identity", lambda *_args: Identity())
        gateway = ZenohInlineGateway.__new__(ZenohInlineGateway)
        gateway.cert_pointer = "local-test-cert-pointer"
        gateway.run_id = session_id
        gateway.session = LocalSession()
        gateway._shadow_lock = threading.Lock()
        gateway._shadow_quarantined = set()
        gateway._stats_lock = threading.Lock()
        gateway.stats = {}
        gateway.admission_gate = AdmissionGate(storage, {DEVICE: {TOPIC}})
        gateway.forensic_sink = ForensicQuarantineSink(storage, rate_per_sec=100.0, burst=10.0)

        # Calling the route before the PEP state is active must not persist anything.
        gateway._route_to_shadow_sandbox(
            DEVICE, TOPIC, b"9999", scenario_id=scenario_id, ground_truth_label="confirmed_attack"
        )
        assert storage.load_adversarial_regression_corpus(("adversarial",)) == []

        assert gateway.set_shadow_quarantine(DEVICE, True)
        gateway._route_to_shadow_sandbox(
            DEVICE, TOPIC, b"9999", scenario_id=scenario_id, ground_truth_label="confirmed_attack"
        )
        assert storage.complete_regression_scenario(scenario_id) == 1
        rows = storage.load_adversarial_regression_corpus(("adversarial",))
        assert len(rows) == 1
        assert rows[0]["capture_source"] == "shadow_sink"
        assert rows[0]["scenario_id"] == scenario_id
        assert rows[0]["ground_truth_label"] == "confirmed_attack"
        assert gateway.session.messages
    finally:
        storage.close()


def test_incident_migration_removes_legacy_default_attack_index_without_deleting_captures(tmp_path):
    db_dir = tmp_path / "db"
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        _live_session(storage, "legacy-unknown-live")
        persist_incident_capture(
            storage,
            _capture_row("legacy-unknown-live", source="shadow_sink",
                         reason="SHADOW_QUARANTINE_ACTIVE"),
        )
        storage._writers["incidents"].submit_sync(
            lambda conn: (
                conn.execute(
                    "INSERT INTO adversarial_regression_corpus(capture_id,corpus_class,indexed_at_ns) "
                    "SELECT capture_id,'adversarial',? FROM incident_captures WHERE session_id=?",
                    (time.time_ns(), "legacy-unknown-live"),
                ),
                conn.execute(
                    "CREATE TRIGGER trg_incident_to_regression_corpus "
                    "AFTER INSERT ON incident_captures BEGIN SELECT 1; END"
                ),
                conn.execute(
                    "DELETE FROM schema_migrations WHERE migration_id='regression_corpus_provenance_v1'"
                ),
            )
        )
    finally:
        storage.close()

    upgraded = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        conn = upgraded.connect_readonly("incidents")
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM incident_captures WHERE session_id='legacy-unknown-live'"
            ).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM adversarial_regression_corpus").fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name='trg_incident_to_regression_corpus'"
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name='trg_regression_capture_qualified'"
            ).fetchone()[0] == 1
        finally:
            conn.close()
    finally:
        upgraded.close()
