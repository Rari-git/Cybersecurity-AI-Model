from __future__ import annotations

import json
from pathlib import Path

import pytest

import models.bundle_loader as loader


def make_candidate(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    artifacts = {
        "model.onnx": b"test-onnx-model",
        "feature_schema.json": b'{"features":["a"]}',
        "scaler.json": b'{"mean":[0.0],"scale":[1.0]}',
        "thresholds.json": b'{"global":0.5}',
    }

    for name, payload in artifacts.items():
        (candidate / name).write_bytes(payload)

    hashes = {
        name: loader.sha256_file(candidate / name)
        for name in artifacts
    }

    (candidate / "manifest.json").write_text(
        json.dumps({"candidate_files_sha256": hashes}),
        encoding="utf-8",
    )

    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"

    loader.generate_model_signing_keypair(private, public)

    return candidate, private, public


def test_transactional_signing_publishes_verified_bundle(tmp_path):
    candidate, private, public = make_candidate(tmp_path)

    original_manifest = (
        candidate / "manifest.json"
    ).read_bytes()

    result = loader.sign_model_bundle_transactional(
        candidate,
        private,
        public,
        metadata={"model_version": "test-transaction"},
    )

    assert result["model_version"] == "test-transaction"

    assert (
        candidate / "manifest.json"
    ).read_bytes() == original_manifest

    assert (candidate / loader.MANIFEST_NAME).is_file()
    assert (candidate / loader.SIGNATURE_NAME).is_file()

    verified = loader.verify_and_load_bundle(
        candidate,
        public,
    )

    assert verified.manifest["model_version"] == "test-transaction"


def test_staging_sign_failure_preserves_unsigned_candidate(
    tmp_path,
    monkeypatch,
):
    candidate, private, public = make_candidate(tmp_path)

    before = {
        path.name: path.read_bytes()
        for path in candidate.iterdir()
        if path.is_file()
    }

    def fail_sign(*args, **kwargs):
        raise RuntimeError("injected staging signing failure")

    monkeypatch.setattr(
        loader,
        "sign_model_bundle",
        fail_sign,
    )

    with pytest.raises(
        RuntimeError,
        match="injected",
    ):
        loader.sign_model_bundle_transactional(
            candidate,
            private,
            public,
            metadata={},
        )

    after = {
        path.name: path.read_bytes()
        for path in candidate.iterdir()
        if path.is_file()
    }

    assert after == before
    assert not (candidate / loader.MANIFEST_NAME).exists()
    assert not (candidate / loader.SIGNATURE_NAME).exists()


def test_publish_failure_restores_unsigned_candidate(
    tmp_path,
    monkeypatch,
):
    candidate, private, public = make_candidate(tmp_path)

    original_manifest = (
        candidate / "manifest.json"
    ).read_bytes()

    real_replace = loader.os.replace

    def fail_signature_publish(source, destination):
        if Path(destination) == candidate / loader.SIGNATURE_NAME:
            raise OSError(
                "injected signature publication failure"
            )

        return real_replace(source, destination)

    monkeypatch.setattr(
        loader.os,
        "replace",
        fail_signature_publish,
    )

    with pytest.raises(
        OSError,
        match="injected",
    ):
        loader.sign_model_bundle_transactional(
            candidate,
            private,
            public,
            metadata={"model_version": "failure-test"},
        )

    assert (
        candidate / "manifest.json"
    ).read_bytes() == original_manifest

    assert not (candidate / loader.MANIFEST_NAME).exists()
    assert not (candidate / loader.SIGNATURE_NAME).exists()

    assert not list(candidate.glob("*.publish"))
