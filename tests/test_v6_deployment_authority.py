from __future__ import annotations

import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from models.bundle_loader import sign_model_bundle, sha256_file
from models.model_lifecycle import _pointer_digest
from training.deployment_authority import deploy_signed_candidate, rollback_deployed_model


def _keypair(tmp_path):
    key = Ed25519PrivateKey.generate()
    private = tmp_path / "test-signing.key"
    public = tmp_path / "test-signing.pub"
    private.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    return private, public


def _write_model_files(path, label):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.onnx").write_bytes(("model-" + label).encode())
    (path / "feature_schema.json").write_text(
        '{"schema_version":"v5.0-f12","seq_len":16,"feature_dim":12}', encoding="utf-8"
    )
    (path / "scaler.json").write_text('{"medians":[],"iqrs":[]}', encoding="utf-8")
    (path / "thresholds.json").write_text('{"global_threshold":1.0}', encoding="utf-8")
    (path / "manifest.json").write_text('{"provenance":"' + label + '"}', encoding="utf-8")


def _case(tmp_path):
    private, public = _keypair(tmp_path)
    root = tmp_path / "models"
    active = root / "lumen_model_bundle"
    _write_model_files(active, "active-v5")
    sign_model_bundle(active, private, {"model_version": "v5.0.0"})
    pointer = root / "active_model.json"
    pointer.write_text(
        json.dumps(
            {"schema": 1, "active_bundle": "lumen_model_bundle", "previous_bundle": None},
            sort_keys=True, separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    pointer_digest = _pointer_digest(pointer)

    candidate = root / "candidates" / "candidate-b5.4"
    _write_model_files(candidate, "candidate-b5.4")
    metrics = {
        "candidate": {
            "clean_security_alert_rate": 0.0,
            "clean_operational_fault_rate": 0.0,
        }
    }
    attestation = {
        "schema": "lumen-offline-signing-attestation-v1",
        "report_id": "test-report-b5.4",
        "report_sha256": "a" * 64,
        "evaluated_at_unix_ns": 1,
        "signed_at_unix_ns": 2,
        "candidate_manifest_sha256": sha256_file(candidate / "manifest.json"),
        "active_manifest_sha256": sha256_file(active / "model_manifest.json"),
        "active_pointer_sha256": pointer_digest,
        "evaluation_gates": {"quality": True, "regression": True},
        "evaluation_metrics_sha256": hashlib.sha256(
            json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }
    sign_model_bundle(candidate, private, metadata={
        "model_version": "candidate-b5.4",
        "signed_by_authority": "local_offline_ed25519_signing_authority",
        "evaluation_attestation": attestation,
        "promotion_gate": "passed",
        "promotion_gates": attestation["evaluation_gates"],
        "promotion_metrics": metrics,
    })
    return root, candidate, private, public, pointer, pointer_digest


def test_deployment_requires_explicit_opt_in(tmp_path, monkeypatch):
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.delenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", raising=False)
    with pytest.raises(PermissionError, match="deployment authority disabled"):
        deploy_signed_candidate(
            candidate, model_root=root, public_key_path=public,
            pointer_path=pointer, expected_active_pointer_sha256=digest,
        )


def test_deployment_rejects_stale_pointer(tmp_path, monkeypatch):
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    pointer.write_text(pointer.read_text(encoding="utf-8") + " ", encoding="utf-8")
    before = pointer.read_bytes()
    with pytest.raises(RuntimeError, match="active model changed"):
        deploy_signed_candidate(
            candidate, model_root=root, public_key_path=public,
            pointer_path=pointer, expected_active_pointer_sha256=digest,
        )
    assert pointer.read_bytes() == before


def test_deployment_rejects_candidate_without_approved_attestation(tmp_path, monkeypatch):
    root, candidate, private, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    (candidate / "model_manifest.json").unlink()
    (candidate / "model_manifest.sig").unlink()
    sign_model_bundle(candidate, private, {"model_version": "unauthorized-candidate"})
    with pytest.raises(PermissionError, match="approved offline signing authority"):
        deploy_signed_candidate(
            candidate, model_root=root, public_key_path=public,
            pointer_path=pointer, expected_active_pointer_sha256=digest,
        )
    assert _pointer_digest(pointer) == digest


def test_successful_deployment_preserves_previous_pointer(tmp_path, monkeypatch):
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    result = deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    current = json.loads(pointer.read_text(encoding="utf-8"))
    previous = json.loads((root / "previous_model.json").read_text(encoding="utf-8"))
    assert result["operation_status"] == "committed"
    assert result["audit_status"] == "written"
    assert result["active_bundle"] == "candidates/candidate-b5.4"
    assert current["previous_bundle"] == "lumen_model_bundle"
    assert previous["active_bundle"] == "lumen_model_bundle"
    assert (root / "deployment_history.jsonl").is_file()


def test_rollback_requires_current_pointer_digest_and_restores_previous(tmp_path, monkeypatch):
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    deployed_digest = _pointer_digest(pointer)
    result = rollback_deployed_model(
        model_root=root, public_key_path=public, pointer_path=pointer,
        expected_active_pointer_sha256=deployed_digest,
    )
    assert result["operation_status"] == "committed"
    assert result["audit_status"] == "written"
    assert result["active_bundle"] == "lumen_model_bundle"
    assert json.loads(pointer.read_text(encoding="utf-8"))["rollback"] is True
    assert "authorized_rollback" in (root / "deployment_history.jsonl").read_text(encoding="utf-8")


class _SimulatedDeploymentCrash(BaseException):
    pass


@pytest.mark.parametrize(
    ("phase", "expected_status"),
    [
        ("after_prepare", "aborted"),
        ("after_previous_pointer", "aborted"),
        ("after_active_pointer", "committed"),
        ("after_commit", None),
    ],
)
def test_deployment_journal_recovers_each_pointer_transaction_boundary(
    tmp_path, monkeypatch, phase, expected_status
):
    from models import model_lifecycle
    from models.deployment_journal import read_journal

    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")

    def crash_at(name, transaction_id):
        if name == phase:
            raise _SimulatedDeploymentCrash(f"simulated crash at {name}")

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", crash_at)
    with pytest.raises(_SimulatedDeploymentCrash):
        deploy_signed_candidate(
            candidate, model_root=root, public_key_path=public,
            pointer_path=pointer, expected_active_pointer_sha256=digest,
        )

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", lambda *_: None)
    from training.deployment_authority import recover_deployment_transactions
    outcomes = recover_deployment_transactions(
        model_root=root, public_key_path=public, pointer_path=pointer
    )
    if expected_status is None:
        assert outcomes == []
        assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "candidates/candidate-b5.4"
    else:
        assert outcomes[-1]["status"] == expected_status
        if expected_status == "aborted":
            assert _pointer_digest(pointer) == digest
            assert not (root / "previous_model.json").exists()
        else:
            assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "candidates/candidate-b5.4"
            assert json.loads((root / "previous_model.json").read_text(encoding="utf-8"))["active_bundle"] == "lumen_model_bundle"

    records = read_journal(root)
    assert records[0]["event"] == "prepare"
    assert records[-1]["event"] in {"commit", "abort"}


def test_deployment_journal_repairs_torn_tail_before_recovery(tmp_path, monkeypatch):
    from models import model_lifecycle
    from models.deployment_journal import read_journal

    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")

    def crash_at(name, transaction_id):
        if name == "after_active_pointer":
            raise _SimulatedDeploymentCrash("power loss after active pointer replacement")

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", crash_at)
    with pytest.raises(_SimulatedDeploymentCrash):
        deploy_signed_candidate(
            candidate, model_root=root, public_key_path=public,
            pointer_path=pointer, expected_active_pointer_sha256=digest,
        )
    with (root / "deployment_journal.jsonl").open("ab") as handle:
        handle.write(b'{"partial":"torn')

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", lambda *_: None)
    outcome = model_lifecycle.recover_incomplete_deployments(root, public, pointer_path=pointer)
    assert outcome[-1]["status"] == "committed"
    assert read_journal(root)[-1]["event"] == "commit"


def test_rollback_rejects_pointer_pair_not_bound_to_committed_journal(tmp_path, monkeypatch):
    root, candidate, private, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )

    decoy = root / "decoy_bundle"
    _write_model_files(decoy, "decoy")
    sign_model_bundle(decoy, private, {"model_version": "decoy"})
    active_doc = json.loads(pointer.read_text(encoding="utf-8"))
    active_doc["previous_bundle"] = "decoy_bundle"
    pointer.write_text(json.dumps(active_doc, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (root / "previous_model.json").write_text(
        json.dumps({"schema": 1, "active_bundle": "decoy_bundle"}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="latest committed deployment journal entry"):
        rollback_deployed_model(
            model_root=root, public_key_path=public, pointer_path=pointer,
            expected_active_pointer_sha256=_pointer_digest(pointer),
        )


def test_deployment_journal_detects_record_tampering(tmp_path, monkeypatch):
    from models.deployment_journal import DeploymentJournalIntegrityError, read_journal

    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    journal = root / "deployment_journal.jsonl"
    content = journal.read_text(encoding="utf-8")
    journal.write_text(content.replace("candidate-b5.4", "candidate-tampered", 1), encoding="utf-8")
    with pytest.raises(DeploymentJournalIntegrityError, match="hash"):
        read_journal(root)


@pytest.mark.parametrize(
    ("phase", "expected_status"),
    [
        ("after_prepare", "aborted"),
        ("after_previous_pointer", "aborted"),
        ("after_active_pointer", "committed"),
        ("after_commit", None),
    ],
)
def test_rollback_journal_recovers_each_pointer_transaction_boundary(
    tmp_path, monkeypatch, phase, expected_status
):
    from models import model_lifecycle
    from training.deployment_authority import rollback_deployed_model, recover_deployment_transactions

    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    deployed_digest = _pointer_digest(pointer)

    def crash_at(name, transaction_id):
        if name == phase:
            raise _SimulatedDeploymentCrash(f"simulated rollback crash at {name}")

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", crash_at)
    with pytest.raises(_SimulatedDeploymentCrash):
        rollback_deployed_model(
            model_root=root, public_key_path=public, pointer_path=pointer,
            expected_active_pointer_sha256=deployed_digest,
        )

    monkeypatch.setattr(model_lifecycle, "_deployment_failpoint", lambda *_: None)
    outcomes = recover_deployment_transactions(
        model_root=root, public_key_path=public, pointer_path=pointer
    )
    if expected_status is None:
        assert outcomes == []
        assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "lumen_model_bundle"
    else:
        assert outcomes[-1]["status"] == expected_status
        if expected_status == "aborted":
            assert _pointer_digest(pointer) == deployed_digest
            assert json.loads((root / "previous_model.json").read_text(encoding="utf-8"))["active_bundle"] == "lumen_model_bundle"
        else:
            assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "lumen_model_bundle"
            assert json.loads((root / "previous_model.json").read_text(encoding="utf-8"))["active_bundle"] == "candidates/candidate-b5.4"




def test_deployment_audit_failure_is_reported_after_commit(tmp_path, monkeypatch):
    import training.deployment_authority as authority
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    def fail_audit(*_args, **_kwargs):
        raise OSError("simulated audit disk failure")
    monkeypatch.setattr(authority, "_append_audit", fail_audit)
    result = deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    assert result["operation_status"] == "committed"
    assert result["audit_status"] == "failed"
    assert "simulated audit disk failure" in result["audit_warning"]
    assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "candidates/candidate-b5.4"


def test_rollback_audit_failure_is_reported_after_commit(tmp_path, monkeypatch):
    import training.deployment_authority as authority
    root, candidate, _, public, pointer, digest = _case(tmp_path)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "1")
    deploy_signed_candidate(
        candidate, model_root=root, public_key_path=public,
        pointer_path=pointer, expected_active_pointer_sha256=digest,
    )
    deployed_digest = _pointer_digest(pointer)
    def fail_audit(*_args, **_kwargs):
        raise ValueError("simulated audit serialization failure")
    monkeypatch.setattr(authority, "_append_audit", fail_audit)
    result = rollback_deployed_model(
        model_root=root, public_key_path=public, pointer_path=pointer,
        expected_active_pointer_sha256=deployed_digest,
    )
    assert result["operation_status"] == "committed"
    assert result["audit_status"] == "failed"
    assert "simulated audit serialization failure" in result["audit_warning"]
    assert json.loads(pointer.read_text(encoding="utf-8"))["active_bundle"] == "lumen_model_bundle"
