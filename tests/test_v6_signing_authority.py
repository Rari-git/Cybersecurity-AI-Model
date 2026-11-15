from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from models.bundle_loader import (
    generate_model_signing_keypair,
    hash_unsigned_candidate_artifacts,
    sign_model_bundle,
    verify_and_load_bundle,
)
from training.promotion_gate import evaluate_candidate_only
import training.signing_authority as signing_authority


DEVICE = "sensor-01"
TOPIC = "telemetry/temperature"


def _keypair(key_dir: Path):
    key_dir.mkdir(parents=True, exist_ok=True)
    private = key_dir / "model_signing_ed25519.key"
    public = key_dir / "model_signing_ed25519.pub"
    generate_model_signing_keypair(private, public)
    return private, public


def _signed_active(model_root: Path, private: Path):
    active = model_root / "lumen_model_bundle"
    active.mkdir(parents=True)
    (active / "model.onnx").write_bytes(b"v5-test-model")
    (active / "feature_schema.json").write_text(json.dumps({
        "schema_version": "v5.0-f12",
        "seq_len": 4,
        "feature_dim": 12,
        "feature_names": [f"f{i}" for i in range(12)],
    }), encoding="utf-8")
    (active / "scaler.json").write_text(
        json.dumps({"medians": [0] * 12, "iqrs": [1] * 12}), encoding="utf-8"
    )
    (active / "thresholds.json").write_text(json.dumps({
        "global_threshold": 1.0,
        "stream_thresholds": {},
        "xai_error_scales": [1] * 12,
        "cusum_baselines": {},
    }), encoding="utf-8")
    sign_model_bundle(active, private, {"model_version": "active-v5"})
    model_root.joinpath("active_model.json").write_text(json.dumps({
        "schema": 1,
        "active_bundle": "lumen_model_bundle",
        "previous_bundle": None,
        "model_version": "active-v5",
        "promoted_at_unix_ns": time.time_ns(),
    }), encoding="utf-8")
    return active


def _unsigned_candidate(path: Path):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"candidate-test-model")
    (path / "feature_schema.json").write_text(json.dumps({
        "schema_version": "v5.0-f12",
        "seq_len": 4,
        "feature_dim": 12,
        "feature_names": [f"f{i}" for i in range(12)],
    }), encoding="utf-8")
    (path / "scaler.json").write_text(
        json.dumps({"medians": [0] * 12, "iqrs": [1] * 12}), encoding="utf-8"
    )
    (path / "thresholds.json").write_text(json.dumps({
        "global_threshold": 1.5,
        "stream_thresholds": {},
        "xai_error_scales": [1] * 12,
        "cusum_baselines": {f"{DEVICE}::{TOPIC}": {"mean": 350, "std": 0.1}},
    }), encoding="utf-8")

    digest = hashlib.sha256(b"curated-test-session").hexdigest()
    train_run, val_run, heldout_run = "a" * 32, "b" * 32, "c" * 32
    train_id = f"baseline-lab-{train_run}-s01-seed1"
    val_id = f"baseline-lab-{val_run}-s01-seed2"
    heldout_id = f"baseline-lab-{heldout_run}-s01-seed3"
    metadata = {
        "model_version": "v6-candidate-signing-test",
        "dataset_policy": "approved_clean_baseline_only",
        "offline_training": True,
        "split_unit": "capture_run_id",
        "candidate_files_sha256": hash_unsigned_candidate_artifacts(path),
        "train_sessions": [train_id],
        "val_sessions": [val_id],
        "heldout_sessions": [heldout_id],
        "cusum_calibration_sessions": [train_id],
        "curation_digests": {
            train_id: digest, val_id: digest, heldout_id: digest,
        },
        "capture_run_ids_by_session": {
            train_id: train_run, val_id: val_run, heldout_id: heldout_run,
        },
        "train_capture_runs": [train_run],
        "val_capture_runs": [val_run],
        "heldout_capture_runs": [heldout_run],
        "global_threshold": 1.5,
        "validation_normal_error_p95": 1.0,
    }
    (path / "manifest.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def _metric_pair(candidate_recall=0.95):
    return {
        "candidate": {
            "attack_total": 10,
            "attack_detected": int(round(candidate_recall * 10)),
            "attack_recall": candidate_recall,
            "protocol_total": 1,
            "protocol_detected": 1,
            "protocol_recall": 1.0,
            "attack_latency": {"atk-replay": 5},
            "fault_total": 2,
            "false_fault_quarantines": 0,
            "normal_error_p95": 0.7,
            "normal_threshold_exceedance_rate": 0.01,
        },
        "active": {
            "attack_total": 10,
            "attack_detected": 8,
            "attack_recall": 0.8,
            "protocol_total": 1,
            "protocol_detected": 1,
            "protocol_recall": 1.0,
            "attack_latency": {"atk-replay": 8},
            "fault_total": 2,
            "false_fault_quarantines": 0,
            "normal_error_p95": 0.6,
            "normal_threshold_exceedance_rate": 0.02,
        },
    }


def _prepared_signing_case(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    key_dir = tmp_path / "keys"
    private, public = _keypair(key_dir)
    _signed_active(model_root, private)
    candidate = model_root / "candidates" / "candidate-signing-test"
    candidate.parent.mkdir()
    _unsigned_candidate(candidate)

    evaluated = evaluate_candidate_only(
        candidate,
        storage=None,
        public_key_path=public,
        model_root=model_root,
        evaluator=lambda *_args: _metric_pair(),
    )
    return model_root, candidate, private, public, evaluated


def test_signing_authority_signs_candidate_without_deploying(tmp_path, monkeypatch):
    model_root, candidate, private, public, evaluated = _prepared_signing_case(tmp_path)
    pointer_path = model_root / "active_model.json"
    pointer_before = pointer_path.read_bytes()

    monkeypatch.setenv("LUMEN_MODEL_SIGNING_AUTHORITY", "1")
    monkeypatch.setattr(
        signing_authority,
        "evaluate_promotion_metrics",
        lambda *_args: _metric_pair(),
    )
    result = signing_authority.sign_evaluated_candidate(
        evaluated.report_path,
        model_root=model_root,
        db_dir=tmp_path / "db-readonly",
        private_key_path=private,
        public_key_path=public,
    )

    assert result.signed is True
    assert result.audit_status == "written"
    assert evaluated.audit_status == "written"
    assert result.reason == "SIGNED_AND_VERIFIED_NOT_DEPLOYED"
    assert result.report_id == evaluated.report_id
    verified = verify_and_load_bundle(candidate, public)
    assert verified.manifest["signed_by_authority"] == (
        "local_offline_ed25519_signing_authority"
    )
    attestation = verified.manifest["evaluation_attestation"]
    assert attestation["report_id"] == evaluated.report_id
    assert attestation["report_sha256"] == evaluated.report_sha256
    assert pointer_path.read_bytes() == pointer_before
    assert not (model_root / "previous_model.json").exists()
    audit = (model_root / "signing_authority_history.jsonl").read_text(encoding="utf-8")
    assert '"event":"candidate_signed_without_deployment"' in audit


def test_signing_authority_requires_explicit_opt_in(tmp_path, monkeypatch):
    model_root, candidate, private, public, evaluated = _prepared_signing_case(tmp_path)
    monkeypatch.delenv("LUMEN_MODEL_SIGNING_AUTHORITY", raising=False)

    with pytest.raises(PermissionError, match="signing disabled"):
        signing_authority.sign_evaluated_candidate(
            evaluated.report_path,
            model_root=model_root,
            db_dir=tmp_path / "db",
            private_key_path=private,
            public_key_path=public,
        )
    assert not (candidate / "model_manifest.json").exists()
    assert not (candidate / "model_manifest.sig").exists()


def test_signing_authority_rejects_candidate_tampered_after_evaluation(tmp_path, monkeypatch):
    model_root, candidate, private, public, evaluated = _prepared_signing_case(tmp_path)
    (candidate / "model.onnx").write_bytes(b"tampered-after-evaluation")
    monkeypatch.setenv("LUMEN_MODEL_SIGNING_AUTHORITY", "1")

    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        signing_authority.sign_evaluated_candidate(
            evaluated.report_path,
            model_root=model_root,
            db_dir=tmp_path / "db",
            private_key_path=private,
            public_key_path=public,
        )
    assert not (candidate / "model_manifest.sig").exists()


def test_signing_authority_recomputes_metrics_before_signing(tmp_path, monkeypatch):
    model_root, candidate, private, public, evaluated = _prepared_signing_case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_SIGNING_AUTHORITY", "1")
    monkeypatch.setattr(
        signing_authority,
        "evaluate_promotion_metrics",
        lambda *_args: _metric_pair(candidate_recall=0.50),
    )

    with pytest.raises(RuntimeError, match="metrics or gates differ"):
        signing_authority.sign_evaluated_candidate(
            evaluated.report_path,
            model_root=model_root,
            db_dir=tmp_path / "db",
            private_key_path=private,
            public_key_path=public,
        )
    assert not (candidate / "model_manifest.json").exists()
    assert not (candidate / "model_manifest.sig").exists()


def test_signing_authority_rejects_private_key_not_matching_runtime_trust_anchor(tmp_path, monkeypatch):
    model_root, candidate, _private, public, evaluated = _prepared_signing_case(tmp_path)
    wrong_dir = tmp_path / "wrong-key"
    wrong_private, _wrong_public = _keypair(wrong_dir)
    monkeypatch.setenv("LUMEN_MODEL_SIGNING_AUTHORITY", "1")

    with pytest.raises(ValueError, match="does not match the existing runtime public trust anchor"):
        signing_authority.sign_evaluated_candidate(
            evaluated.report_path,
            model_root=model_root,
            db_dir=tmp_path / "db",
            private_key_path=wrong_private,
            public_key_path=public,
        )
    assert not (candidate / "model_manifest.json").exists()
    assert not (candidate / "model_manifest.sig").exists()



def test_signing_audit_failure_does_not_hide_verified_signature(tmp_path, monkeypatch):
    model_root, candidate, private, public, evaluated = _prepared_signing_case(tmp_path)
    pointer_path = model_root / "active_model.json"
    pointer_before = pointer_path.read_bytes()
    monkeypatch.setenv("LUMEN_MODEL_SIGNING_AUTHORITY", "1")
    monkeypatch.setattr(signing_authority, "evaluate_promotion_metrics", lambda *_args: _metric_pair())
    def fail_audit(*_args, **_kwargs):
        raise OSError("simulated signing audit failure")
    monkeypatch.setattr(signing_authority, "_append_audit", fail_audit)
    result = signing_authority.sign_evaluated_candidate(
        evaluated.report_path, model_root=model_root, db_dir=tmp_path / "db-readonly",
        private_key_path=private, public_key_path=public,
    )
    assert result.signed is True
    assert result.audit_status == "failed"
    assert "simulated signing audit failure" in result.audit_warning
    verified = verify_and_load_bundle(candidate, public)
    assert verified.manifest["signed_by_authority"] == "local_offline_ed25519_signing_authority"
    assert pointer_path.read_bytes() == pointer_before


def test_evaluation_audit_failure_preserves_completed_report(tmp_path, monkeypatch):
    import training.promotion_gate as promotion_gate
    model_root, candidate, _private, public, _first_evaluation = _prepared_signing_case(tmp_path)
    def fail_audit(*_args, **_kwargs):
        raise OSError("simulated evaluation audit failure")
    monkeypatch.setattr(promotion_gate, "_append_audit", fail_audit)
    result = evaluate_candidate_only(
        candidate, storage=None, public_key_path=public, model_root=model_root,
        evaluator=lambda *_args: _metric_pair(),
    )
    assert result.evaluated is True
    assert result.all_gates_passed is True
    assert result.audit_status == "failed"
    assert "simulated evaluation audit failure" in result.audit_warning
    assert Path(result.report_path).is_file()
    assert result.report_sha256
