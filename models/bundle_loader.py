"""Hardened Lumen model bundle signing and verification.

Drop-in compatible with the V5 bundle_loader API. Every regular file in a
bundle directory (except the manifest and signature) is covered by the signed
manifest, including ONNX external-data files such as model.onnx.data.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MANIFEST_NAME = "model_manifest.json"
SIGNATURE_NAME = "model_manifest.sig"
REQUIRED_ARTIFACTS = frozenset({"model.onnx", "feature_schema.json", "scaler.json", "thresholds.json"})
RESERVED_METADATA = frozenset({"bundle_format", "files_sha256"})


def _safe_relative_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"invalid bundle artifact path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe bundle artifact path: {name!r}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(bundle_dir: Path) -> dict[str, Path]:
    """Return safe, regular, non-symlink artifact paths relative to bundle_dir."""
    root = Path(bundle_dir).resolve(strict=True)
    artifacts: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.name in (MANIFEST_NAME, SIGNATURE_NAME):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle artifacts must be regular files, not links: {path}")
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative)
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ValueError(f"bundle artifact escapes bundle directory: {relative}")
        artifacts[relative] = path
    missing = REQUIRED_ARTIFACTS.difference(artifacts)
    if missing:
        raise FileNotFoundError(f"missing required model artifacts: {sorted(missing)}")
    return artifacts


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def generate_model_signing_keypair(priv_path: Path, pub_path: Path) -> None:
    """Create a new keypair only at absent paths; never silently rotate trust."""
    priv_path, pub_path = Path(priv_path), Path(pub_path)
    if priv_path.exists() or pub_path.exists():
        raise FileExistsError("refusing to overwrite existing model-signing key material")
    priv_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Exclusive creation prevents accidental replacement in concurrent setup.
    fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(private_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            priv_path.unlink()
        except OSError:
            pass
        raise
    fd = os.open(pub_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    with os.fdopen(fd, "wb") as handle:
        handle.write(public_bytes)
        handle.flush()
        os.fsync(handle.fileno())


def sign_model_bundle(bundle_dir: Path, private_key_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    root = Path(bundle_dir).resolve(strict=True)
    if RESERVED_METADATA.intersection(metadata):
        raise ValueError("metadata cannot override reserved manifest fields")
    artifacts = _artifact_files(root)
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    manifest = {"bundle_format": "lumen-model-bundle-v5", "files_sha256": hashes, **metadata}
    canonical = canonical_manifest_bytes(manifest)
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("Expected Ed25519 private key for model signing")
    signature = base64.b64encode(private_key.sign(canonical)).decode("ascii")

    # Publish signature and manifest as atomic file replacements.
    for name, payload in ((MANIFEST_NAME, canonical), (SIGNATURE_NAME, signature.encode("ascii"))):
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / name)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return manifest


@dataclass(frozen=True)
class VerifiedModelBundle:
    bundle_dir: Path
    onnx_path: Path
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    scaler: dict[str, Any]
    thresholds: dict[str, Any]


def verify_and_load_bundle(bundle_dir: str | Path, public_key_path: str | Path) -> VerifiedModelBundle:
    root = Path(bundle_dir).resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    signature_path = root / SIGNATURE_NAME
    if not manifest_path.is_file() or not signature_path.is_file():
        raise FileNotFoundError(f"unsigned or incomplete model bundle in {root}")
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Expected Ed25519 public key for bundle verification")
    raw_manifest = manifest_path.read_bytes()
    try:
        signature = base64.b64decode(signature_path.read_text(encoding="ascii").strip(), validate=True)
        public_key.verify(signature, raw_manifest)
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("SECURITY ALERT: model manifest Ed25519 signature verification failed") from exc
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid model manifest JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("bundle_format") != "lumen-model-bundle-v5":
        raise ValueError("unsupported model bundle format")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict):
        raise ValueError("manifest files_sha256 must be an object")
    artifacts = _artifact_files(root)
    actual_names = set(artifacts)
    expected_names = set(expected)
    if actual_names != expected_names:
        missing_hashes = sorted(actual_names - expected_names)
        missing_files = sorted(expected_names - actual_names)
        raise ValueError(f"bundle file set mismatch: unhashed={missing_hashes}, missing={missing_files}")
    for name, path in artifacts.items():
        _safe_relative_path(name)
        digest = expected.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 declaration for {name}")
        if not hmac.compare_digest(sha256_file(path), digest.lower()):
            raise ValueError(f"SECURITY ALERT: SHA-256 mismatch for {name}")
    def read_json(name: str) -> dict[str, Any]:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value
    return VerifiedModelBundle(
        bundle_dir=root,
        onnx_path=root / "model.onnx",
        manifest=manifest,
        feature_schema=read_json("feature_schema.json"),
        scaler=read_json("scaler.json"),
        thresholds=read_json("thresholds.json"),
    )
