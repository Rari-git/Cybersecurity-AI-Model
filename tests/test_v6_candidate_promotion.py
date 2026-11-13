from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from models.bundle_loader import generate_model_signing_keypair, sign_model_bundle, verify_and_load_bundle
from models.model_lifecycle import resolve_active_bundle
from storage.persistence import persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.adaptive_retrainer import run_adaptive_cycle
from training.promotion_gate import (
    evaluate_and_promote_candidate, rollback_on_rate_spike,
    monitor_post_promotion_rates,
)
from training.dataset_governance import curate_clean_sessions

TOPIC = "home/sensors/gas/ppm"
DEVICE = "gas_sensor_01"
FP = hashlib.sha256(b"candidate-test-cert").digest()


def _metric_pair(*, candidate_recall=0.95, active_recall=0.80,
                 candidate_faults=0, active_faults=0,
                 candidate_latency=5, active_latency=8,
                 candidate_normal_p95=0.7, candidate_exceedance=0.01):
    return {
        "candidate": {
            "attack_total": 10,
            "attack_detected": int(round(candidate_recall * 10)),
            "attack_recall": candidate_recall,
            "attack_latency": {"atk-replay": candidate_latency},
            "fault_total": 2,
            "false_fault_quarantines": candidate_faults,
            "normal_error_p95": candidate_normal_p95,
            "normal_threshold_exceedance_rate": candidate_exceedance,
            "clean_security_alert_rate": 0.0,
            "clean_operational_fault_rate": 0.0,
        },
        "active": {
            "attack_total": 10,
            "attack_detected": int(round(active_recall * 10)),
            "attack_recall": active_recall,
            "attack_latency": {"atk-replay": active_latency},
            "fault_total": 2,
            "false_fault_quarantines": active_faults,
            "normal_error_p95": 0.6,
            "normal_threshold_exceedance_rate": 0.02,
            "clean_security_alert_rate": 0.01,
            "clean_operational_fault_rate": 0.0,
        },
    }


def _keypair(key_dir: Path):
    private = key_dir / "model_signing_ed25519.key"
    public = key_dir / "model_signing_ed25519.pub"
    generate_model_signing_keypair(private, public)
    return private, public


def _make_signed_bundle(path: Path, private: Path, version="active-v5"):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"local-test-onnx-main")
    (path / "feature_schema.json").write_text(json.dumps({
        "schema_version": "v5.0-f12", "seq_len": 4, "feature_dim": 12,
        "feature_names": [f"f{i}" for i in range(12)],
    }), encoding="utf-8")
    (path / "scaler.json").write_text(json.dumps({"medians": [0] * 12, "iqrs": [1] * 12}), encoding="utf-8")
    (path / "thresholds.json").write_text(json.dumps({
        "global_threshold": 1.0, "stream_thresholds": {}, "xai_error_scales": [1] * 12,
        "cusum_baselines": {},
    }), encoding="utf-8")
    return sign_model_bundle(path, private, {"model_version": version})


def _seed_curated_sessions(db_dir: Path, session_count=3, events=18):
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        for session_index in range(session_count):
            sid = f"clean-{session_index:02d}"
            storage.register_baseline_session(
                sid, "baseline_lab",
                {"expected_sensor_topics": [f"{DEVICE}::{TOPIC}"], "compressed_time": False},
            )
            base_ts = time.time_ns() + session_index * 100_000_000_000
            for index in range(events):
                value = 350.0 + session_index * 0.05 + index * 0.01
                payload = f"{value:.3f}"
                raw = payload.encode()
                row = (
                    sid, DEVICE, DEVICE, 1, FP, TOPIC,
                    base_ts + index * 1_000_000_000,
                    base_ts + index * 1_000_000_000,
                    len(raw), hashlib.sha256(raw).digest(), value, payload,
                    "baseline_lab", "v6-adaptive-test",
                )
                persist_normal_event(storage, row)
            storage.finalize_baseline_session(sid)
            decision = curate_clean_sessions(
                storage, min_duration_sec=0, min_sensor_coverage=1.0, session_ids=[sid]
            )[0]
            assert decision.approved, decision.reasons
    finally:
        storage.close()


def _unsigned_candidate(path: Path, *, validation_p95=1.0):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"candidate-onnx-test")
    (path / "feature_schema.json").write_text(json.dumps({
        "schema_version": "v5.0-f12", "seq_len": 4, "feature_dim": 12,
        "feature_names": [f"f{i}" for i in range(12)],
    }), encoding="utf-8")
    (path / "scaler.json").write_text(json.dumps({"medians": [0] * 12, "iqrs": [1] * 12}), encoding="utf-8")
    (path / "thresholds.json").write_text(json.dumps({
        "global_threshold": 1.5, "stream_thresholds": {}, "xai_error_scales": [1] * 12,
        "cusum_baselines": {f"{DEVICE}::{TOPIC}": {"mean": 350, "std": 0.1}},
    }), encoding="utf-8")
    digest = hashlib.sha256(b"curated-test-session").hexdigest()
    meta = {
        "model_version": "v6-candidate-unit-test",
        "dataset_policy": "approved_clean_baseline_only",
        "offline_training": True,
        "train_sessions": ["train-a"],
        "val_sessions": ["val-a"],
        "heldout_sessions": ["holdout-a"],
        "cusum_calibration_sessions": ["train-a"],
        "curation_digests": {"train-a": digest, "val-a": digest, "holdout-a": digest},
        "global_threshold": 1.5,
        "validation_normal_error_p95": validation_p95,
    }
    (path / "manifest.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return meta


def _initialize_active(model_root: Path, private: Path, public: Path):
    active = model_root / "lumen_model_bundle"
    _make_signed_bundle(active, private, "active-v5")
    (model_root / "active_model.json").write_text(json.dumps({
        "schema": 1,
        "active_bundle": "lumen_model_bundle",
        "previous_bundle": None,
        "model_version": "active-v5",
        "promoted_at_unix_ns": time.time_ns(),
    }), encoding="utf-8")
    verify_and_load_bundle(active, public)
    return active


def test_degraded_or_overpermissive_candidate_is_rejected_without_signing(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate = model_root / "candidates" / "too-permissive"
    _unsigned_candidate(candidate, validation_p95=0.5)
    metrics = _metric_pair(
        candidate_recall=0.40,
        active_recall=0.80,
        candidate_faults=2,
        active_faults=0,
        candidate_latency=None,
        candidate_normal_p95=2.0,
        candidate_exceedance=0.20,
    )
    result = evaluate_and_promote_candidate(
        candidate,
        storage=None,
        public_key_path=public,
        private_key_path=private,
        model_root=model_root,
        evaluator=lambda *_args: metrics,
    )
    assert not result.promoted
    assert not all(result.gates.values())
    assert not (candidate / "model_manifest.json").exists()
    pointer = json.loads((model_root / "active_model.json").read_text())
    assert pointer["active_bundle"] == "lumen_model_bundle"


def test_valid_candidate_is_signed_promoted_and_automatically_rolled_back(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate = model_root / "candidates" / "improved"
    _unsigned_candidate(candidate)
    result = evaluate_and_promote_candidate(
        candidate,
        storage=None,
        public_key_path=public,
        private_key_path=private,
        model_root=model_root,
        evaluator=lambda *_args: _metric_pair(),
    )
    assert result.promoted and all(result.gates.values())
    verified = verify_and_load_bundle(candidate, public)
    assert verified.manifest["signed_by_authority"] == "local_offline_ed25519"
    active_pointer = json.loads((model_root / "active_model.json").read_text())
    previous_pointer = json.loads((model_root / "previous_model.json").read_text())
    assert active_pointer["active_bundle"] == "candidates/improved"
    assert previous_pointer["active_bundle"] == "lumen_model_bundle"
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == candidate.resolve()

    rollback = rollback_on_rate_spike(
        {"anomaly_rate": 0.30, "fault_rate": 0.10},
        {"anomaly_rate": 0.05, "fault_rate": 0.0},
        model_root=model_root,
        public_key_path=public,
        anomaly_rate_margin=0.05,
        fault_rate_margin=0.0,
    )
    assert rollback is not None and rollback["rollback"] is True
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == (model_root / "lumen_model_bundle").resolve()


def test_offline_worker_trains_unsigned_candidate_then_signs_and_promotes(tmp_path, monkeypatch):
    import sys
    import torch
    import training.adaptive_retrainer as adaptive

    db_dir = tmp_path / "db"
    model_root = tmp_path / "models"
    key_dir = tmp_path / "authority-keys"
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    _seed_curated_sessions(db_dir, session_count=3, events=18)

    # This host intentionally lacks the ONNX exporter package. Keep the actual
    # PyTorch training path and candidate provenance/signing/promotion real,
    # while making export a deterministic local artifact for unit testing.
    def fake_onnx_export(_model, _input, output_path, **_kwargs):
        Path(output_path).write_bytes(b"test-double-onnx-artifact")

    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
    # Earlier shadow-evaluator tests may have installed a synthetic
    # onnxruntime module without an importlib ModuleSpec. PyTorch's optimizer
    # inspects optional backends through find_spec, so remove that test double
    # before exercising the real local optimizer path.
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")
    result = adaptive.run_adaptive_cycle(
        db_dir=db_dir,
        model_root=model_root,
        key_dir=key_dir,
        public_key_path=public,
        min_duration_sec=0,
        min_sensor_coverage=1.0,
        epochs=1,
        seq_len=4,
        run_id="autonomous-test-cycle",
        evaluator=lambda _candidate, _active, _manifest: _metric_pair(),
    )
    assert len(result.curated_sessions) == 3
    assert result.candidate_dir
    assert result.training_metadata["offline_training"] is True
    assert result.training_metadata["signed"] is False
    assert result.promotion and result.promotion.promoted
    candidate = Path(result.candidate_dir)
    assert (candidate / "model.onnx").is_file()
    assert (candidate / "manifest.json").is_file()
    assert (candidate / "model_manifest.sig").is_file()
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == candidate.resolve()


def test_unsigned_candidate_must_bind_disjoint_curated_sessions(tmp_path):
    from models.bundle_loader import validate_unsigned_candidate_bundle

    candidate = tmp_path / "candidate"
    _unsigned_candidate(candidate)
    doc = json.loads((candidate / "manifest.json").read_text())
    doc["cusum_calibration_sessions"] = ["val-a"]
    (candidate / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CUSUM calibration may use training sessions only"):
        validate_unsigned_candidate_bundle(candidate)


def test_shadow_evaluator_counts_policy_quarantine_not_raw_lstm_flags_on_faults(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import numpy as np
    import training.promotion_gate as gate

    class FakeOrtSession:
        def run(self, _outputs, inputs):
            # Deliberately over-sensitive model: every feature has a large
            # reconstruction residual. The policy must still keep a recognized
            # frozen sensor in operational FAULT without cyber-quarantining it.
            x = inputs["window_input"]
            return [np.zeros_like(x)]

    class FakeOrt(ModuleType):
        @staticmethod
        def InferenceSession(_path, providers=None):
            return FakeOrtSession()

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt("onnxruntime"))
    monkeypatch.setattr(gate, "verify_and_load_bundle", lambda *_args: SimpleNamespace(
        onnx_path=tmp_path / "model.onnx",
        scaler={"medians": [0.0] * 12, "iqrs": [1.0] * 12},
        feature_schema={"seq_len": 8},
        thresholds={
            "xai_error_scales": [0.25] * 12,
            "global_threshold": 0.01,
            "stream_thresholds": {},
            "cusum_baselines": {f"{DEVICE}::{TOPIC}": {"mean": 351.0, "std": 0.05}},
        },
        manifest={"model_version": "fake"},
    ))

    events = tuple({
        "device_id": DEVICE,
        "topic": TOPIC,
        "generation": 1,
        "gateway_ts_ns": (index + 1) * 1_000_000_000,
        "raw_payload": b"351",
        "numeric_value": 351.0,
        "operation": "put",
    } for index in range(40))
    group = gate.EvaluationGroup("fault:frozen", "fault", DEVICE, TOPIC, events)
    result = gate._score_bundle(tmp_path, tmp_path / "public.pem", [group])
    assert result["fault_total"] == 1
    assert result["false_fault_quarantines"] == 0
    detection = result["detections"]["fault:frozen"]
    assert detection["final_operational_status"] == "FAULT"
    assert detection["final_security_status"] == "ACTIVE"


def test_shadow_evaluator_protocol_capture_is_immediate_and_not_scored_by_lstm(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    import training.promotion_gate as gate

    class FakeOrt(ModuleType):
        @staticmethod
        def InferenceSession(*_args, **_kwargs):
            raise AssertionError("protocol-only groups must not run ONNX inference")

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt("onnxruntime"))
    monkeypatch.setattr(gate, "verify_and_load_bundle", lambda *_args: SimpleNamespace(
        onnx_path=tmp_path / "model.onnx",
        scaler={}, feature_schema={"seq_len": 8},
        thresholds={"xai_error_scales": [0.25] * 12, "global_threshold": 1.0,
                    "stream_thresholds": {}, "cusum_baselines": {}},
        manifest={"model_version": "fake"},
    ))
    group = gate.EvaluationGroup("protocol:replay", "protocol", DEVICE, TOPIC, tuple())
    result = gate._score_bundle(tmp_path, tmp_path / "public.pem", [group])
    assert result["attack_total"] == 1
    assert result["attack_detected"] == 1
    assert result["detections"]["protocol:replay"] == {
        "detected": True, "latency": 1, "source": "protocol",
    }


def test_process_offline_guard_blocks_network_and_restores_socket_apis():
    import socket
    from training.offline_guard import OfflineNetworkAccessRejected, offline_network_guard

    with offline_network_guard():
        with pytest.raises(OfflineNetworkAccessRejected, match="network operations are prohibited"):
            socket.create_connection(("example.invalid", 443), timeout=0.01)
        with pytest.raises(OfflineNetworkAccessRejected):
            socket.getaddrinfo("example.invalid", 443)
    # Patches are scoped and do not poison unrelated test/runtime code.
    assert callable(socket.create_connection)


def test_post_promotion_rate_monitor_automatically_rolls_back(tmp_path, monkeypatch):
    import training.promotion_gate as gate

    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate = model_root / "candidates" / "rate-monitored"
    _unsigned_candidate(candidate)
    promoted = evaluate_and_promote_candidate(
        candidate, storage=None, public_key_path=public, private_key_path=private,
        model_root=model_root, evaluator=lambda *_args: _metric_pair(),
    )
    assert promoted.promoted

    monkeypatch.setattr(gate, "read_post_promotion_rates", lambda *_args, **_kwargs: {
        "anomaly_rate": 0.25, "fault_rate": 0.02, "event_count": 100.0,
    })
    result = monitor_post_promotion_rates(
        storage=object(), model_root=model_root, public_key_path=public,
        anomaly_rate_margin=0.01, fault_rate_margin=0.01,
    )
    assert result and result["rollback"] is True
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == (model_root / "lumen_model_bundle").resolve()


def test_autonomous_worker_runs_scheduled_cycles_and_monitors_between_them(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training.adaptive_retrainer as adaptive

    monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")
    clock = [0.0]
    monkeypatch.setattr(adaptive.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adaptive.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    cycles = []
    monitors = []

    def fake_cycle(*, storage, **_kwargs):
        cycles.append(storage)
        return SimpleNamespace(
            run_id=f"run-{len(cycles)}", curated_sessions=("clean",),
            rejected_sessions=(), candidate_dir=None, promotion=None,
        )

    def fake_monitor(storage, **_kwargs):
        monitors.append(storage)
        return {"rollback": False}

    sentinel_storage = object()
    monkeypatch.setattr(adaptive, "run_adaptive_cycle", fake_cycle)
    monkeypatch.setattr(adaptive, "monitor_post_promotion_rates", fake_monitor)
    results = adaptive.run_autonomous_worker(
        storage=sentinel_storage,
        db_dir=tmp_path / "db",
        model_root=tmp_path / "models",
        key_dir=tmp_path / "keys",
        interval_sec=1.0,
        monitor_interval_sec=1.0,
        max_cycles=2,
    )
    assert len(results) == 2
    assert cycles == [sentinel_storage, sentinel_storage]
    assert monitors == [sentinel_storage]


def test_recent_post_promotion_rates_are_read_from_local_sqlite(tmp_path):
    from storage.telemetry_db import DBPaths, LumenStorageManager
    from training.promotion_gate import read_post_promotion_rates

    db_dir = tmp_path / "rate-db"
    _seed_curated_sessions(db_dir, session_count=1, events=24)
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        storage.append_policy_dimension_transition(
            DEVICE, 1, "security", "ACTIVE", "SUSPECT", "SHADOW_ANOMALY",
        )
        storage.append_policy_dimension_transition(
            DEVICE, 1, "operational", "HEALTHY", "FAULT", "FROZEN_SENSOR",
        )
        observed = read_post_promotion_rates(storage, window_sec=300, min_events=20)
        assert observed is not None
        assert observed["event_count"] == 24.0
        assert observed["anomaly_rate"] == pytest.approx(1 / 24)
        assert observed["fault_rate"] == pytest.approx(1 / 24)
    finally:
        storage.close()
