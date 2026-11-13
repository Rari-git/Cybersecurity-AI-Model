from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from models.bundle_loader import sign_model_bundle, verify_and_load_bundle
from models.model_lifecycle import promote_candidate, resolve_active_bundle, rollback_model


def _bundle(path: Path, version: str, private_key_path: Path):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"onnx-model-" + version.encode())
    (path / "model.onnx.data").write_bytes(b"external-tensor-weights-" + version.encode())
    (path / "feature_schema.json").write_text('{"schema_version":"v5.0-f12","seq_len":16,"feature_dim":12}')
    (path / "scaler.json").write_text('{"medians":[],"iqrs":[]}')
    (path / "thresholds.json").write_text('{"global_threshold":1.0}')
    return sign_model_bundle(path, private_key_path, {"model_version": version})


def _keys(tmp_path: Path):
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


def test_external_data_is_signed_and_tampering_is_detected(tmp_path):
    private, public = _keys(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = _bundle(bundle, "candidate-1", private)
    assert "model.onnx.data" in manifest["files_sha256"]
    verify_and_load_bundle(bundle, public)
    (bundle / "model.onnx.data").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_and_load_bundle(bundle, public)


def test_unlisted_extra_artifact_is_rejected(tmp_path):
    private, public = _keys(tmp_path)
    bundle = tmp_path / "bundle"
    _bundle(bundle, "candidate-2", private)
    (bundle / "unexpected.bin").write_bytes(b"not signed")
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_and_load_bundle(bundle, public)


def test_promotion_and_rollback_are_atomic_pointer_updates(tmp_path):
    private, public = _keys(tmp_path)
    model_root = tmp_path / "models"
    c1 = model_root / "candidates" / "candidate-a"
    c2 = model_root / "candidates" / "candidate-b"
    _bundle(c1, "candidate-a", private)
    _bundle(c2, "candidate-b", private)
    pointer = model_root / "active_model.json"
    first = promote_candidate(c1, public, model_root=model_root, pointer_path=pointer)
    assert first["active_bundle"] == "candidates/candidate-a"
    second = promote_candidate(c2, public, model_root=model_root, pointer_path=pointer)
    assert second["previous_bundle"] == "candidates/candidate-a"
    assert resolve_active_bundle(model_root, pointer) == c2.resolve()
    reverted = rollback_model(model_root=model_root, pointer_path=pointer, public_key_path=public)
    assert reverted["active_bundle"] == "candidates/candidate-a"
    assert resolve_active_bundle(model_root, pointer) == c1.resolve()
