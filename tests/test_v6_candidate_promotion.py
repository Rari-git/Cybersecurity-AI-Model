from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from models.bundle_loader import (
    generate_model_signing_keypair,
    hash_unsigned_candidate_artifacts,
    sign_model_bundle,
    validate_unsigned_candidate_bundle,
    verify_and_load_bundle,
)
from models.model_lifecycle import resolve_active_bundle
from storage.persistence import persist_incident_capture, persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.adaptive_retrainer import run_adaptive_cycle
from training.promotion_gate import (
    evaluate_and_promote_candidate,
    evaluate_candidate_only,
    verify_candidate_evaluation_report,
    rollback_on_rate_spike,
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
            # Each invocation/session below represents a distinct independent
            # acquisition run for the V6 adaptive train/validation/holdout split.
            capture_run_id = uuid.uuid4().hex
            sid = f"baseline-lab-{capture_run_id}-s01-seed{session_index}"
            storage.register_baseline_session(
                sid, "baseline_lab",
                {
                    "capture_run_id": capture_run_id,
                    "session_index": 1,
                    "events_per_sensor": events,
                    "expected_sensor_topics": [f"{DEVICE}::{TOPIC}"],
                    "compressed_time": False,
                },
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


def _seed_regression_corpus(db_dir: Path, *, scenarios_per_class: int = 2):
    # Seed completed, provenance-verified regression scenarios in a test-only DB.
    from storage.persistence import persist_incident_capture

    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    scenario_classes = (
        ("adversarial", "attack", "shadow_sink", "SHADOW_QUARANTINE_ACTIVE", "confirmed_attack"),
        ("protocol_violation", "protocol", "ids_alert", "REPLAY_OR_OUT_OF_ORDER_SEQUENCE", None),
        ("hardware_fault", "hardware_fault", "fault_lab", "SENSOR_FROZEN", "confirmed_hardware_fault"),
    )
    try:
        for corpus_class, scenario_type, capture_source, reason_code, label in scenario_classes:
            for index in range(scenarios_per_class):
                suffix = uuid.uuid4().hex[:16]
                scenario_id = f"pytest-{corpus_class}-{index}-{suffix}"
                session_id = f"pytest-live-{suffix}"
                storage.register_baseline_session(
                    session_id,
                    "live_validated",
                    {"stage_b_test_fixture": True, "scenario_id": scenario_id},
                    approved_for_training=False,
                )
                storage.register_regression_scenario(
                    scenario_id,
                    scenario_type,
                    runner_version="pytest-stage-b3-fixture-v1",
                    manifest={
                        "test_fixture": True,
                        "corpus_class": corpus_class,
                        "scenario_index": index,
                        "family": f"{corpus_class}-fixture-family-{index}",
                    },
                )
                raw = f"{corpus_class}-fixture-{index}".encode("ascii")
                now_ns = time.time_ns()
                capture = (
                    session_id,
                    DEVICE,
                    f"{DEVICE}-g2-pytest",
                    2,
                    FP,
                    TOPIC,
                    f"shadow/sandbox/{DEVICE}/data",
                    now_ns,
                    None,
                    capture_source,
                    reason_code,
                    0.95,
                    None,
                    len(raw),
                    hashlib.sha256(raw).digest(),
                    raw,
                    1,
                    label,
                    scenario_id,
                    now_ns + 14 * 24 * 3600 * 1_000_000_000,
                )
                persist_incident_capture(storage, capture)
                indexed = storage.complete_regression_scenario(scenario_id, success=True)
                if indexed < 1:
                    raise AssertionError(
                        f"test scenario {scenario_id} did not qualify for {corpus_class} corpus"
                    )
    finally:
        storage.flush_all()
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
    train_run, val_run, heldout_run = "a" * 32, "b" * 32, "c" * 32
    train_id = f"baseline-lab-{train_run}-s01-seed1"
    val_id = f"baseline-lab-{val_run}-s01-seed2"
    heldout_id = f"baseline-lab-{heldout_run}-s01-seed3"
    meta = {
        "model_version": "v6-candidate-unit-test",
        "dataset_policy": "approved_clean_baseline_only",
        "offline_training": True,
        "split_unit": "capture_run_id",
        "candidate_files_sha256": hash_unsigned_candidate_artifacts(path),
        "train_sessions": [train_id],
        "val_sessions": [val_id],
        "heldout_sessions": [heldout_id],
        "cusum_calibration_sessions": [train_id],
        "curation_digests": {train_id: digest, val_id: digest, heldout_id: digest},
        "capture_run_ids_by_session": {
            train_id: train_run, val_id: val_run, heldout_id: heldout_run,
        },
        "train_capture_runs": [train_run],
        "val_capture_runs": [val_run],
        "heldout_capture_runs": [heldout_run],
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
    result = evaluate_candidate_only(
        candidate, storage=None, public_key_path=public, model_root=model_root,
        evaluator=lambda *_args: metrics,
    )
    assert result.evaluated
    assert not result.all_gates_passed
    assert not all(result.gates.values())
    assert not (candidate / "model_manifest.json").exists()
    pointer = json.loads((model_root / "active_model.json").read_text(encoding="utf-8"))
    assert pointer["active_bundle"] == "lumen_model_bundle"

def test_valid_candidate_is_evaluated_without_signing_or_deployment(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate = model_root / "candidates" / "improved"
    _unsigned_candidate(candidate)
    result = evaluate_candidate_only(
        candidate, storage=None, public_key_path=public, model_root=model_root,
        evaluator=lambda *_args: _metric_pair(),
    )
    assert result.evaluated and result.all_gates_passed
    assert result.report_path and Path(result.report_path).is_file()
    assert not (candidate / "model_manifest.json").exists()
    active_pointer = json.loads((model_root / "active_model.json").read_text(encoding="utf-8"))
    assert active_pointer["active_bundle"] == "lumen_model_bundle"
    assert not (model_root / "previous_model.json").exists()
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == (model_root / "lumen_model_bundle").resolve()

def _candidate_evaluation_fixture(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    model_root.mkdir()
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate_dir = model_root / "candidates" / "b5-2-candidate"
    candidate_dir.parent.mkdir(parents=True)
    _unsigned_candidate(candidate_dir)
    result = evaluate_candidate_only(
        candidate_dir,
        storage=None,
        public_key_path=public,
        model_root=model_root,
        evaluator=lambda _candidate, _active, _manifest: _metric_pair(),
    )
    return model_root, public, candidate_dir, result


def test_candidate_evaluation_report_binds_candidate_and_active_snapshots(tmp_path):
    model_root, public, candidate_dir, result = _candidate_evaluation_fixture(tmp_path)

    assert result.evaluated is True
    assert result.all_gates_passed is True
    assert result.report_id
    assert result.report_sha256
    assert Path(result.report_path).parent == model_root / "evaluations"

    report = verify_candidate_evaluation_report(
        result.report_path,
        model_root=model_root,
        public_key_path=public,
        candidate_dir=candidate_dir,
    )
    assert report["report_id"] == result.report_id
    assert report["report_sha256"] == result.report_sha256
    assert report["candidate"]["manifest_sha256"] == result.candidate_manifest_sha256
    assert report["candidate"]["artifacts_sha256"]["model.onnx"]
    assert report["active"]["manifest_sha256"] == result.active_manifest_sha256
    assert report["active"]["signature_sha256"]
    assert report["active"]["pointer_sha256"]
    assert report["data_provenance"]["split_unit"] == "capture_run_id"
    assert len(report["data_provenance"]["train_capture_runs"]) == 1
    assert len(report["data_provenance"]["validation_capture_runs"]) == 1
    assert len(report["data_provenance"]["heldout_capture_runs"]) == 1
    assert report["evaluation"]["all_gates_passed"] is True
    assert not (candidate_dir / "model_manifest.sig").exists()


def test_candidate_evaluation_report_rejects_report_tampering(tmp_path):
    model_root, public, candidate_dir, result = _candidate_evaluation_fixture(tmp_path)
    report_path = Path(result.report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evaluation"]["reason"] = "FORGED_APPROVAL"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation report SHA-256 mismatch"):
        verify_candidate_evaluation_report(
            report_path, model_root=model_root, public_key_path=public,
            candidate_dir=candidate_dir,
        )


def test_candidate_evaluation_report_rejects_candidate_changed_after_evaluation(tmp_path):
    model_root, public, candidate_dir, result = _candidate_evaluation_fixture(tmp_path)
    (candidate_dir / "model.onnx").write_bytes(b"changed-after-evaluation")

    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        verify_candidate_evaluation_report(
            result.report_path, model_root=model_root, public_key_path=public,
            candidate_dir=candidate_dir,
        )


def test_candidate_evaluation_report_rejects_active_pointer_changed_after_evaluation(tmp_path):
    model_root, public, _candidate_dir, result = _candidate_evaluation_fixture(tmp_path)
    pointer_path = model_root / "active_model.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["promoted_at_unix_ns"] += 1
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="active model or pointer snapshot"):
        verify_candidate_evaluation_report(
            result.report_path, model_root=model_root, public_key_path=public,
        )


def test_candidate_mutation_during_evaluation_prevents_report_creation(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    model_root.mkdir()
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate_dir = model_root / "candidates" / "mutated-during-evaluation"
    candidate_dir.parent.mkdir(parents=True)
    _unsigned_candidate(candidate_dir)

    def mutating_evaluator(candidate, _active, _manifest):
        (Path(candidate) / "model.onnx").write_bytes(b"mutated-in-evaluator")
        return _metric_pair()

    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        evaluate_candidate_only(
            candidate_dir,
            storage=None,
            public_key_path=public,
            model_root=model_root,
            evaluator=mutating_evaluator,
        )
    assert not (model_root / "evaluations").exists()


def test_active_pointer_mutation_during_evaluation_prevents_report_creation(tmp_path):
    model_root = tmp_path / "models"
    key_dir = tmp_path / "keys"
    model_root.mkdir()
    key_dir.mkdir()
    private, public = _keypair(key_dir)
    _initialize_active(model_root, private, public)
    candidate_dir = model_root / "candidates" / "active-mutated-during-evaluation"
    candidate_dir.parent.mkdir(parents=True)
    _unsigned_candidate(candidate_dir)

    def mutating_evaluator(_candidate, _active, _manifest):
        pointer_path = model_root / "active_model.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["promoted_at_unix_ns"] += 1
        pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
        return _metric_pair()

    with pytest.raises(RuntimeError, match="active model or pointer changed during shadow evaluation"):
        evaluate_candidate_only(
            candidate_dir,
            storage=None,
            public_key_path=public,
            model_root=model_root,
            evaluator=mutating_evaluator,
        )
    assert not (model_root / "evaluations").exists()


def test_offline_worker_trains_and_evaluates_unsigned_candidate_without_deploying(tmp_path, monkeypatch):
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
    _seed_regression_corpus(db_dir)

    # Historical/unreviewed sessions must not be re-curated by an adaptive cycle.
    historical = LumenStorageManager(
        paths=DBPaths.from_dir(db_dir), enable_writers=True
    )
    try:
        historical.register_baseline_session(
            "historical-unreviewed", "baseline_lab",
            {"events_per_sensor": 400},
        )
        historical.finalize_baseline_session("historical-unreviewed")
    finally:
        historical.close()

    # This host intentionally lacks the ONNX exporter package. Keep the actual
    # PyTorch training path and candidate provenance/shadow evaluation real,
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
    assert result.evaluation is not None
    assert result.evaluation.evaluated is True
    assert result.evaluation.all_gates_passed is True
    assert result.evaluation.reason == "ALL_EVALUATION_GATES_PASSED"
    candidate = Path(result.candidate_dir)
    assert (candidate / "model.onnx").is_file()
    assert (candidate / "manifest.json").is_file()
    assert not (candidate / "model_manifest.sig").exists()
    assert resolve_active_bundle(model_root, model_root / "active_model.json") == (
        model_root / "lumen_model_bundle"
    ).resolve()

    # The worker must leave unrelated historical curation rows untouched.
    check = sqlite3.connect(db_dir / "telemetry_normal.db")
    try:
        historical_row = check.execute(
            """SELECT approved_for_training FROM baseline_sessions
               WHERE session_id='historical-unreviewed'"""
        ).fetchone()
        historical_curation = check.execute(
            """SELECT status FROM session_curation
               WHERE session_id='historical-unreviewed'"""
        ).fetchone()
        assert historical_row == (0,)
        assert historical_curation is None
    finally:
        check.close()


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
    assert result["attack_total"] == 0
    assert result["attack_detected"] == 0
    assert result["attack_recall"] == 0.0
    assert result["protocol_total"] == 1
    assert result["protocol_detected"] == 1
    assert result["protocol_recall"] == 1.0
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


def test_legacy_post_promotion_rate_monitor_is_disabled(tmp_path):
    with pytest.raises(PermissionError, match="legacy deployment monitor is disabled"):
        monitor_post_promotion_rates(
            storage=object(), model_root=tmp_path / "models",
            public_key_path=tmp_path / "model_signing_ed25519.pub",
        )

def test_autonomous_worker_runs_evaluation_cycles_without_deployment_monitoring(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import training.adaptive_retrainer as adaptive

    monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")
    clock = [0.0]
    monkeypatch.setattr(
        adaptive.time, "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    cycles = []

    def fake_cycle(*, storage, **_kwargs):
        cycles.append(storage)
        return SimpleNamespace(
            run_id=f"run-{len(cycles)}",
            curated_sessions=("clean",),
            rejected_sessions=(),
            candidate_dir=None,
            evaluation=None,
            preflight_reason="INSUFFICIENT_CURATED_SESSIONS",
            preflight_details=None,
        )

    sentinel_storage = object()
    monkeypatch.setattr(adaptive, "run_adaptive_cycle", fake_cycle)
    assert not hasattr(adaptive, "monitor_post_promotion_rates")
    results = adaptive.run_autonomous_worker(
        storage=sentinel_storage,
        db_dir=tmp_path / "db",
        model_root=tmp_path / "models",
        interval_sec=1.0,
        max_cycles=2,
    )
    assert len(results) == 2
    assert cycles == [sentinel_storage, sentinel_storage]
    assert clock[0] == 1.0


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

def test_unsigned_candidate_artifacts_are_hash_checked_before_shadow_evaluation(tmp_path):
    candidate = tmp_path / "candidate"
    _unsigned_candidate(candidate)
    (candidate / "model.onnx").write_bytes(b"tampered-candidate-model")
    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        validate_unsigned_candidate_bundle(candidate)


def test_shadow_evaluator_loads_hash_checked_unsigned_candidate(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    import training.promotion_gate as gate

    class FakeOrtSession:
        def run(self, _outputs, inputs):
            return [inputs["window_input"].copy()]

    class FakeOrt(ModuleType):
        @staticmethod
        def InferenceSession(_path, providers=None):
            return FakeOrtSession()

    candidate = tmp_path / "candidate"
    _unsigned_candidate(candidate)
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt("onnxruntime"))

    events = tuple({
        "device_id": DEVICE,
        "topic": TOPIC,
        "generation": 1,
        "gateway_ts_ns": (index + 1) * 1_000_000_000,
        "raw_payload": f"{350.0 + index * 0.01:.2f}".encode(),
        "numeric_value": 350.0 + index * 0.01,
        "operation": "put",
    } for index in range(24))
    group = gate.EvaluationGroup("clean:unsigned", "clean", DEVICE, TOPIC, events)
    result = gate._score_bundle(
        candidate, tmp_path / "unused-public-key.pem", [group], unsigned_candidate=True
    )
    assert result["clean_event_count"] == 24
    assert result["normal_error_p95"] == pytest.approx(0.0)
    assert result["normal_threshold_exceedance_rate"] == 0.0


def test_offline_worker_can_train_and_shadow_without_signing_private_key(tmp_path, monkeypatch):
    import shutil
    import sys
    import torch
    import training.adaptive_retrainer as adaptive

    db_dir = tmp_path / "db"
    model_root = tmp_path / "models"
    bootstrap_keys = tmp_path / "bootstrap-signer"
    bootstrap_keys.mkdir()
    private, bootstrap_public = _keypair(bootstrap_keys)
    _initialize_active(model_root, private, bootstrap_public)

    runtime_keys = tmp_path / "runtime-public-only"
    runtime_keys.mkdir()
    public = runtime_keys / "model_signing_ed25519.pub"
    shutil.copyfile(bootstrap_public, public)
    _seed_curated_sessions(db_dir, session_count=3, events=18)
    _seed_regression_corpus(db_dir)

    def fake_onnx_export(_model, _input, output_path, **_kwargs):
        Path(output_path).write_bytes(b"test-double-onnx-artifact")

    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")

    result = adaptive.run_adaptive_cycle(
        db_dir=db_dir,
        model_root=model_root,
        public_key_path=public,
        min_duration_sec=0,
        min_sensor_coverage=1.0,
        epochs=1,
        seq_len=4,
        run_id="candidate-without-signer",
        evaluator=lambda _candidate, _active, _manifest: _metric_pair(),
    )

    assert result.candidate_dir
    assert result.training_metadata["signed"] is False
    assert result.evaluation is not None
    assert result.evaluation.evaluated is True
    assert result.evaluation.all_gates_passed is True
    assert result.evaluation.reason == "ALL_EVALUATION_GATES_PASSED"
    candidate = Path(result.candidate_dir)
    assert (candidate / "manifest.json").is_file()
    assert not (candidate / "model_manifest.sig").exists()
    assert not (candidate / "model_manifest.json").exists()
    pointer = json.loads((model_root / "active_model.json").read_text())
    assert pointer["active_bundle"] == "lumen_model_bundle"

def test_worker_selection_enforces_duration_without_recurating(tmp_path):
    from training.adaptive_retrainer import _approved_session_ids

    db_dir = tmp_path / "db"
    _seed_curated_sessions(db_dir, session_count=3, events=18)
    storage = LumenStorageManager(
        paths=DBPaths.from_dir(db_dir), enable_writers=True
    )
    try:
        eligible_without_duration_gate = _approved_session_ids(
            storage, min_duration_sec=0, min_sensor_coverage=1.0
        )
        eligible_with_worker_default = _approved_session_ids(
            storage, min_duration_sec=300, min_sensor_coverage=1.0
        )
        assert len(eligible_without_duration_gate) == 3
        assert eligible_with_worker_default == []
    finally:
        storage.close()



def test_protocol_detection_regression_is_a_separate_promotion_gate():
    from training.promotion_gate import _promotion_gates

    metrics = {
        "candidate": {
            "attack_recall": 0.9,
            "attack_latency": {},
            "protocol_recall": 0.5,
            "false_fault_quarantines": 0,
            "normal_error_p95": 0.5,
            "normal_threshold_exceedance_rate": 0.01,
        },
        "active": {
            "attack_recall": 0.8,
            "attack_latency": {},
            "protocol_recall": 1.0,
            "false_fault_quarantines": 0,
            "normal_error_p95": 0.6,
            "normal_threshold_exceedance_rate": 0.02,
        },
    }
    gates = _promotion_gates(metrics, {"validation_normal_error_p95": 1.0})
    assert gates["attack_recall_not_worse_than_active"] is True
    assert gates["protocol_detection_not_worse_than_active"] is False
