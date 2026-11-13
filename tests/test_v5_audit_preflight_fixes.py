from __future__ import annotations

import hashlib
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agent.pep_protocol import PepAckRegistry
from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
from models.bundle_loader import sign_model_bundle, verify_and_load_bundle
from storage.admission_gate import AdmissionGate, ProvenanceContext, extract_numeric_value
from storage.telemetry_db import DBPaths, LumenStorageManager, _SingleWriterWorker
from utils.sensor_payload import parse_sensor_payload

ROOT = Path(__file__).resolve().parents[1]
GAS_TOPIC = "home/sensors/gas/ppm"


def _keys(tmp_path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    priv = tmp_path / "signing.key"
    pub = tmp_path / "signing.pub"
    priv.write_bytes(private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    pub.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    return priv, pub


def _make_bundle(path: Path, private: Path):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"onnx-main")
    (path / "model.onnx.data").write_bytes(b"onnx-external-sidecar")
    (path / "feature_schema.json").write_text('{"schema_version":"v5.0-f12","seq_len":16,"feature_dim":12}', encoding="utf-8")
    (path / "scaler.json").write_text('{"medians":[],"iqrs":[]}', encoding="utf-8")
    (path / "thresholds.json").write_text('{"global_threshold":1.0}', encoding="utf-8")
    return sign_model_bundle(path, private, {"model_version": "audit-test"})


def _try_symlink(link: Path, target: Path, *, target_is_directory: bool = False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this host: {exc}")


@pytest.mark.parametrize("wildcard", ["internal/**", "internal/*/**", "**/telemetry/**"])
def test_acl_validator_rejects_extra_internal_publish_wildcards(tmp_path, wildcard):
    json5 = pytest.importorskip("json5")
    from utils.zenoh_router import validate_router_acl_template

    doc = json5.loads((ROOT / "config" / "zenohd_router.json5").read_text(encoding="utf-8"))
    doc["access_control"]["rules"].append({
        "id": "attacker_added_internal_wildcard",
        "permission": "allow", "messages": ["put"], "flows": ["ingress"],
        "key_exprs": [wildcard],
    })
    doc["access_control"]["policies"].append({
        "rules": ["attacker_added_internal_wildcard"], "subjects": ["lab_tester"],
    })
    path = tmp_path / "wildcard.json5"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Exception, match="unauthorized subjects"):
        validate_router_acl_template(path)


@pytest.mark.parametrize(("key_expr", "message", "flow", "role"), [
    ("internal/**", "declare_subscriber", "ingress", "lab_tester"),
    ("**/pep_ack/**", "put", "ingress", "controller"),
])
def test_acl_validator_rejects_non_ids_subscriber_or_non_gateway_ack_wildcards(
    tmp_path, key_expr, message, flow, role,
):
    json5 = pytest.importorskip("json5")
    from utils.zenoh_router import validate_router_acl_template

    doc = json5.loads((ROOT / "config" / "zenohd_router.json5").read_text(encoding="utf-8"))
    doc["access_control"]["rules"].append({
        "id": "attacker_added_pep_wildcard",
        "permission": "allow", "messages": [message], "flows": [flow],
        "key_exprs": [key_expr],
    })
    doc["access_control"]["policies"].append({
        "rules": ["attacker_added_pep_wildcard"], "subjects": [role],
    })
    path = tmp_path / "pep-wildcard.json5"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Exception, match="unauthorized subjects"):
        validate_router_acl_template(path)


def test_bundle_signing_rejects_symlinked_directory(tmp_path):
    private, _public = _keys(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.onnx.data").write_bytes(b"external-data")
    bundle = tmp_path / "bundle"
    _make_bundle(bundle, private)
    _try_symlink(bundle / "external", outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|reparse point"):
        sign_model_bundle(bundle, private, {"model_version": "should-fail"})


def test_bundle_verification_rejects_symlinked_directory(tmp_path):
    private, public = _keys(tmp_path)
    bundle = tmp_path / "bundle"
    _make_bundle(bundle, private)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tensor.bin").write_bytes(b"external-data")
    _try_symlink(bundle / "external", outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|reparse point"):
        verify_and_load_bundle(bundle, public)


def test_bundle_signing_rejects_symlinked_onnx_sidecar(tmp_path):
    private, _public = _keys(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.data"
    outside.write_bytes(b"external-data")
    (bundle / "model.onnx").write_bytes(b"onnx-main")
    _try_symlink(bundle / "model.onnx.data", outside)
    (bundle / "feature_schema.json").write_text('{"schema_version":"v5.0-f12","seq_len":16,"feature_dim":12}', encoding="utf-8")
    (bundle / "scaler.json").write_text('{"medians":[],"iqrs":[]}', encoding="utf-8")
    (bundle / "thresholds.json").write_text('{"global_threshold":1.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="symlink|reparse point"):
        sign_model_bundle(bundle, private, {"model_version": "should-fail"})


def test_bundle_verification_rejects_symlinked_onnx_sidecar(tmp_path):
    private, public = _keys(tmp_path)
    bundle = tmp_path / "bundle"
    _make_bundle(bundle, private)
    sidecar = bundle / "model.onnx.data"
    sidecar.unlink()
    outside = tmp_path / "outside.data"
    outside.write_bytes(b"external-data")
    _try_symlink(sidecar, outside)
    with pytest.raises(ValueError, match="symlink|reparse point"):
        verify_and_load_bundle(bundle, public)


def test_async_batch_failure_is_propagated_to_all_waiting_sync_items(tmp_path):
    worker = object.__new__(_SingleWriterWorker)
    worker._queue = queue.Queue()
    conn = sqlite3.connect(tmp_path / "batch-test.sqlite")
    done = threading.Event()
    holder = []

    def fail(_conn):
        raise sqlite3.OperationalError("simulated async batch failure")

    def succeeding(_conn):
        return "must-not-be-reported-as-committed"

    batch = [(fail, None, None), (succeeding, done, holder)]
    for item in batch:
        worker._queue.put_nowait(item)
    with pytest.raises(sqlite3.OperationalError, match="simulated async batch failure"):
        worker._execute_batch(conn, batch)
    assert done.is_set()
    assert len(holder) == 1 and isinstance(holder[0], sqlite3.OperationalError)
    assert worker._queue.unfinished_tasks == 0
    conn.close()


def test_async_writer_recovers_after_failed_async_batch(tmp_path):
    worker = _SingleWriterWorker(tmp_path / "writer.sqlite", "CREATE TABLE IF NOT EXISTS ok_table (id INTEGER)")
    try:
        assert worker.submit_async(lambda conn: conn.execute("INSERT INTO missing_table VALUES (1)"))
        deadline = time.monotonic() + 2.0
        while worker.last_batch_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert isinstance(worker.last_batch_error, sqlite3.Error)
        assert worker.submit_sync(lambda conn: conn.execute("SELECT 42").fetchone()[0]) == 42
    finally:
        worker.close()


def test_shared_payload_parser_rejects_nonfinite_malformed_and_type_mismatch():
    bad_values = (b"NaN", b"Inf", b"-Infinity", b'{"value": 42}', b"OPEN", b"[]")
    for raw in bad_values:
        parsed = parse_sensor_payload(GAS_TOPIC, raw)
        assert parsed.valid is False
        accepted, _number, _reason = extract_numeric_value(GAS_TOPIC, raw.decode("ascii"))
        assert accepted is False
    assert parse_sensor_payload(GAS_TOPIC, b"NaN").reason == "NAN_OR_INF_PAYLOAD"
    assert parse_sensor_payload("home/sensors/door/state", b"open").canonical_text == "OPEN"
    assert parse_sensor_payload("home/sensors/door/state", b"ON").valid is False
    assert parse_sensor_payload(GAS_TOPIC, b" +001.000 ").canonical_text == "1"
    assert parse_sensor_payload(GAS_TOPIC, b"-0.0").canonical_text == "0"


def test_admission_gate_uses_shared_parser_and_routes_nonfinite_to_incidents(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        storage.register_baseline_session("live", "live_validated", {}, approved_for_training=False)
        gate = AdmissionGate(storage, {"gas_sensor_01": {GAS_TOPIC}})
        ctx = ProvenanceContext(
            session_id="live", device_cn="gas_sensor_01-g1-cafe1234",
            cert_fingerprint=hashlib.sha256(b"cert").digest(), source_type="live_validated",
            device_id="gas_sensor_01", generation=1,
        )
        for payload in (b"NaN", b"Inf", b'{"value":10}', b"OPEN"):
            result = gate.evaluate_and_ingest(ctx, GAS_TOPIC, payload)
            assert result.admitted is False
            assert result.reason in {"NAN_OR_INF_PAYLOAD", "NON_NUMERIC_PAYLOAD"}
        canonical = gate.evaluate_and_ingest(ctx, GAS_TOPIC, b" +0351.000 ")
        assert canonical.admitted is True
        assert canonical.canonical_text == "351"
    finally:
        storage.close()


def test_e5_cusum_baselines_only_use_training_sessions():
    from training.train_local_lstm import compute_cusum_baselines

    stream = ("gas_sensor_01", GAS_TOPIC)
    raw_by_session = {
        "train-a": {stream: [100.0, 110.0, 90.0]},
        "train-b": {stream: [100.0]},
        "validation-only": {stream: [1000000.0, 2000000.0]},
    }
    baseline = compute_cusum_baselines(raw_by_session, ["train-a", "train-b"])
    assert baseline[f"gas_sensor_01::{GAS_TOPIC}"]["mean"] == 100.0
    assert baseline == compute_cusum_baselines(raw_by_session, ["train-a", "train-b"])
    assert baseline[f"gas_sensor_01::{GAS_TOPIC}"]["mean"] != 500000.0
    with pytest.raises(ValueError, match="invalid training values"):
        compute_cusum_baselines({"train": {stream: [float("nan")]}}, ["train"])



def test_training_extracts_session_scoped_values_and_preserves_compressed_cadence(monkeypatch, tmp_path):
    import training.train_local_lstm as trainer

    events = {
        "training": [
            {"device_id": "gas_sensor_01", "topic": GAS_TOPIC, "gateway_ts_ns": 1_000_000_000,
             "raw_payload": "350", "numeric_value": 350.0},
            {"device_id": "gas_sensor_01", "topic": GAS_TOPIC, "gateway_ts_ns": 1_020_000_000,
             "raw_payload": "351", "numeric_value": 351.0},
        ],
        "validation": [
            {"device_id": "gas_sensor_01", "topic": GAS_TOPIC, "gateway_ts_ns": 5_000_000_000,
             "raw_payload": "999", "numeric_value": 999.0},
        ],
    }

    class Cursor:
        def __init__(self, rows):
            self.rows = rows
        def fetchall(self):
            return self.rows

    class ReadOnly:
        def execute(self, sql, params=()):
            if "SELECT session_id" in sql:
                return Cursor([{"session_id": "training"}, {"session_id": "validation"}])
            return Cursor(events[params[0]])
        def close(self):
            pass

    class FakeStorage:
        def __init__(self, **_kwargs):
            pass
        def connect_readonly(self, _name):
            return ReadOnly()

    seen_deltas = []

    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass
        def update(self, **kwargs):
            seen_deltas.append(kwargs["delta_t_sec"])
            return SimpleNamespace(raw_value=float(kwargs["numeric_value"]))
        def get_window(self, *_args):
            return None

    monkeypatch.setattr(trainer, "LumenStorageManager", FakeStorage)
    monkeypatch.setattr(trainer, "DeviceTopicFeatureExtractor", FakeExtractor)
    windows, raw_by_session = trainer._extract_windows_by_session(tmp_path, seq_len=2)
    assert set(windows) == {"training", "validation"}
    assert raw_by_session["training"][("gas_sensor_01", GAS_TOPIC)] == [350.0, 351.0]
    assert raw_by_session["validation"][("gas_sensor_01", GAS_TOPIC)] == [999.0]
    assert 0.02 in seen_deltas
    assert 1.0 not in seen_deltas[1:2]


def test_evaluation_scenario_directories_and_policy_state_are_isolated(tmp_path):
    fake_ort = ModuleType("onnxruntime")
    sys.modules.setdefault("onnxruntime", fake_ort)
    from training.evaluate_v5 import scenario_database_dir

    attack_dir = scenario_database_dir(tmp_path, "atk1_stealth_spoof_heldout")
    fault_dir = scenario_database_dir(tmp_path, "fault1_frozen_sensor")
    assert attack_dir != fault_dir

    first = LumenStorageManager(paths=DBPaths.from_dir(attack_dir), enable_writers=True)
    try:
        fp = hashlib.sha256(b"scenario device").digest()
        first.upsert_device("gas_sensor_01", 1, "gas_sensor_01", fp, "sensor")
        attack_policy = RiskPolicyEngine(
            storage=first, pep_apply_fn=lambda _dev, _enabled: True,
            window_n=3, trigger_k=2,
        )
        evidence = DetectionEvidence(
            device_id="gas_sensor_01", generation=1, topic=GAS_TOPIC,
            lstm_score=9.0, lstm_threshold=1.0, lstm_anomalous=True,
            cusum_score=9.0, cusum_triggered=True,
            operational_status="HEALTHY", fault_type="NONE",
        )
        attack_policy.evaluate_evidence(evidence)
        attack_policy.evaluate_evidence(evidence)
        assert attack_policy.get_state("gas_sensor_01") == "SHADOW_QUARANTINE"
    finally:
        first.close()

    second = LumenStorageManager(paths=DBPaths.from_dir(fault_dir), enable_writers=True)
    try:
        fault_policy = RiskPolicyEngine(storage=second, pep_apply_fn=lambda _dev, _enabled: True)
        assert fault_policy.get_state("gas_sensor_01") == "ACTIVE"
    finally:
        second.close()


@pytest.mark.parametrize("dropped_phase", [None, "PREPARE_RELEASE", "COMMIT_RELEASE"])
def test_release_handshake_requires_all_acks_before_gateway_opens(
    monkeypatch, tmp_path, dropped_phase,
):
    # Import runtime classes with light module stubs; no native Zenoh/ONNX is
    # needed because the test exercises the actual PEP callbacks and registry.
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

    from agent.gateway_ids import ZenohInlineGateway
    from agent.ids_agent import ZenohIDSAgent

    class Gate:
        def __init__(self):
            self.blocked = {"gas_sensor_01": True}
        def set_device_blocked(self, device_id, blocked):
            self.blocked[device_id] = blocked

    class GatewaySession:
        def __init__(self):
            self.published = []
        def put(self, key, payload):
            self.published.append((key, payload))

    gateway = ZenohInlineGateway.__new__(ZenohInlineGateway)
    gateway._shadow_lock = threading.RLock()
    gateway._stats_lock = threading.RLock()
    gateway._shadow_quarantined = {"gas_sensor_01"}
    gateway._release_prepared = {}
    gateway.stats = {}
    gateway.admission_gate = Gate()
    gateway.run_id = "gateway-audit-test"
    gateway.session = GatewaySession()

    class IDSNetwork:
        def __init__(self):
            self.commands = []
        def put(self, key, payload):
            if key.startswith("control/policy/"):
                start = len(gateway.session.published)
                gateway._on_policy_command(SimpleNamespace(key_expr=key, payload=payload))
                cmd = json.loads(payload.decode("utf-8"))
                self.commands.append(cmd)
                # Drop the selected ACK in transit; deliver only ACKs newly
                # published for this command (not a duplicate from an old phase).
                if dropped_phase is None or cmd.get("phase") != dropped_phase:
                    for ack_key, ack_payload in gateway.session.published[start:]:
                        if ack_key.startswith("control/pep_ack/"):
                            agent._on_pep_ack(SimpleNamespace(key_expr=ack_key, payload=ack_payload))

    agent = ZenohIDSAgent.__new__(ZenohIDSAgent)
    agent._on_pep_ack = lambda sample: ZenohIDSAgent._on_pep_ack(agent, sample)
    agent.pep_ack_registry = PepAckRegistry()
    agent.pep_ack_timeout_sec = 0.01
    agent.session = IDSNetwork()

    decision = ZenohIDSAgent._dispatch_pep_policy(agent, "gas_sensor_01", False, "release-test")
    expected_success = dropped_phase is None
    assert decision is expected_success
    assert gateway.is_shadow_quarantined("gas_sensor_01") is (not expected_success)
    assert gateway.admission_gate.blocked["gas_sensor_01"] is (not expected_success)
    assert agent.pep_ack_registry.pending_count() == 0
    # Gateway can only open after both ACK phases were delivered and IDS
    # subsequently published FINALIZE_RELEASE.
    finalize_commands = [
        json.loads(payload.decode("utf-8"))
        for key, payload in gateway.session.published
        if key.startswith("control/pep_ack/") is False and key.startswith("control/policy/")
    ]
    assert any(item.get("phase") == "FINALIZE_RELEASE" for item in agent.session.commands) is expected_success
    if dropped_phase == "PREPARE_RELEASE":
        assert gateway._release_prepared == {}
        # A delayed COMMIT from the timed-out transaction cannot reopen Gate.
        stale_commit = {
            "decision_id": "stale-ready", "device_id": "gas_sensor_01",
            "target_state": "ACTIVE", "phase": "COMMIT_RELEASE",
            "prepare_decision_id": "release-test-prepare",
        }
        gateway._on_policy_command(SimpleNamespace(
            key_expr="control/policy/gas_sensor_01",
            payload=json.dumps(stale_commit).encode("utf-8"),
        ))
        assert gateway.admission_gate.blocked["gas_sensor_01"] is True



def test_ids_sensor_processor_rejects_payloads_admission_gate_rejects(monkeypatch):
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

    from agent.ids_agent import ZenohIDSAgent

    class Policy:
        def __init__(self):
            self.evidence = []
        def evaluate_evidence(self, evidence):
            self.evidence.append(evidence)
            return SimpleNamespace(action="ALERT", new_state="ACTIVE")

    ids = ZenohIDSAgent.__new__(ZenohIDSAgent)
    ids.stats = {"unknown": 0}
    ids._resolve_device_id = lambda _topic, device_id=None: device_id or "gas_sensor_01"
    ids.policy_engine = Policy()
    ids._publish_v5_alert = lambda **_kwargs: None

    for text in ("NaN", "Inf", '{"value":10}', "OPEN"):
        # AdmissionGate rejects these values for a gas scalar; even if an
        # untrusted/buggy envelope claims ADMITTED, IDS independently rejects.
        ids._process_sensor(
            GAS_TOPIC, text, 1.0, 1.0,
            device_id="gas_sensor_01", generation=1,
            admission_status="ADMITTED", admission_reason="ADMITTED",
            admission_numeric_value=None,
        )

    assert ids.stats["unknown"] == 4
    assert len(ids.policy_engine.evidence) == 4
    assert all(item.deterministic_violation for item in ids.policy_engine.evidence)


def test_policy_engine_retains_quarantine_when_release_ack_fails(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        fp = hashlib.sha256(b"release-timeout").digest()
        storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01", fp, "sensor")
        engine = RiskPolicyEngine(
            storage=storage,
            pep_request_fn=lambda _dev, enabled, _decision_id: bool(enabled),
            window_n=3, trigger_k=2, recovery_clean_windows=1,
        )
        attack = DetectionEvidence(
            device_id="gas_sensor_01", generation=1, topic=GAS_TOPIC,
            lstm_score=10.0, lstm_threshold=1.0, lstm_anomalous=True,
            cusum_score=10.0, cusum_triggered=True,
            operational_status="HEALTHY", fault_type="NONE",
        )
        engine.evaluate_evidence(attack)
        quarantined = engine.evaluate_evidence(attack)
        assert quarantined.new_state == "SHADOW_QUARANTINE"

        clean = DetectionEvidence(
            device_id="gas_sensor_01", generation=1, topic=GAS_TOPIC,
            lstm_score=0.0, lstm_threshold=1.0, lstm_anomalous=False,
            cusum_score=0.0, cusum_triggered=False,
            operational_status="HEALTHY", fault_type="NONE",
        )
        engine.evaluate_evidence(clean)
        engine.evaluate_evidence(clean)
        failed_release = engine.evaluate_evidence(clean)
        assert failed_release.action == "RELEASE_FAILED_QUARANTINED"
        assert failed_release.new_state == "SHADOW_QUARANTINE"
        assert engine.get_state("gas_sensor_01") == "SHADOW_QUARANTINE"
    finally:
        storage.close()
