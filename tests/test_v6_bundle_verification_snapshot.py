from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from models import bundle_loader as loader


def _make_signed_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.onnx").write_bytes(b"verified-onnx-model-bytes")
    (bundle / "feature_schema.json").write_text('{"features":["a","b"]}', encoding="utf-8")
    (bundle / "scaler.json").write_text('{"mean":[0,0],"scale":[1,1]}', encoding="utf-8")
    (bundle / "thresholds.json").write_text('{"global":0.25}', encoding="utf-8")
    nested = bundle / "external" / "weights.data"
    nested.parent.mkdir()
    nested.write_bytes(b"external-onnx-data")
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    public_path.write_bytes(private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    loader.sign_model_bundle(bundle, private_path, {})
    return bundle, public_path


def test_verified_bundle_uses_private_snapshot_and_preserves_external_data(tmp_path: Path):
    bundle, public = _make_signed_bundle(tmp_path)
    verified = loader.verify_and_load_bundle(bundle, public)
    try:
        assert verified.bundle_dir == bundle.resolve()
        assert verified.onnx_path != bundle / "model.onnx"
        assert verified.onnx_path.is_file()
        assert verified.onnx_path.read_bytes() == b"verified-onnx-model-bytes"
        assert (verified.snapshot_dir / "external" / "weights.data").read_bytes() == b"external-onnx-data"
        assert verified.feature_schema == {"features": ["a", "b"]}
        assert verified.scaler == {"mean": [0, 0], "scale": [1, 1]}
        assert verified.thresholds == {"global": 0.25}
    finally:
        verified.close()


def test_source_bundle_replacement_after_verification_does_not_change_snapshot(tmp_path: Path):
    bundle, public = _make_signed_bundle(tmp_path)
    verified = loader.verify_and_load_bundle(bundle, public)
    snapshot_bytes = verified.onnx_path.read_bytes()
    try:
        (bundle / "model.onnx").write_bytes(b"attacker-replaced-source-model")
        (bundle / "feature_schema.json").write_text('{"features":["attacker"]}', encoding="utf-8")
        assert verified.onnx_path.read_bytes() == snapshot_bytes
        assert verified.onnx_path.read_bytes() == b"verified-onnx-model-bytes"
        assert verified.feature_schema == {"features": ["a", "b"]}
    finally:
        verified.close()


def test_replacement_during_snapshot_copy_fails_hash_check(tmp_path: Path, monkeypatch):
    bundle, public = _make_signed_bundle(tmp_path)
    original_copy = loader._copy_verified_artifact
    changed = {"done": False}

    def replace_before_copy(source, target, expected):
        if Path(source).name == "model.onnx" and not changed["done"]:
            changed["done"] = True
            Path(source).write_bytes(b"changed-between-verification-and-copy")
        return original_copy(source, target, expected)

    monkeypatch.setattr(loader, "_copy_verified_artifact", replace_before_copy)
    with pytest.raises(ValueError, match="SHA-256 mismatch while snapshotting"):
        loader.verify_and_load_bundle(bundle, public)
    assert changed["done"] is True


def test_snapshot_close_removes_private_snapshot(tmp_path: Path):
    bundle, public = _make_signed_bundle(tmp_path)
    verified = loader.verify_and_load_bundle(bundle, public)
    snapshot = verified.snapshot_dir
    assert snapshot.is_dir()
    verified.close()
    assert not snapshot.exists()


def test_invalid_signature_does_not_return_snapshot(tmp_path: Path):
    bundle, public = _make_signed_bundle(tmp_path)
    (bundle / loader.SIGNATURE_NAME).write_text("not-a-valid-signature", encoding="ascii")
    with pytest.raises(ValueError, match="signature verification failed"):
        loader.verify_and_load_bundle(bundle, public)


def test_snapshot_artifact_paths_are_confined_and_regular(tmp_path: Path):
    bundle, public = _make_signed_bundle(tmp_path)
    verified = loader.verify_and_load_bundle(bundle, public)
    try:
        for path in verified.snapshot_dir.rglob("*"):
            assert not path.is_symlink()
            if path.is_file():
                assert path.resolve().is_relative_to(verified.snapshot_dir.resolve())
    finally:
        verified.close()
