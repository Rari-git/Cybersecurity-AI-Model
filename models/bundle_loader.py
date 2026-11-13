"""Etapa E5: Semnare și verificare criptografică Ed25519 pentru Lumen Model Bundle.

Structura pachetului:
  lumen_model_bundle/
    model.onnx
    feature_schema.json
    scaler.json
    thresholds.json
    model_manifest.json
    model_manifest.sig
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

BUNDLE_FILES: tuple[str, ...] = (
    "model.onnx",
    "feature_schema.json",
    "scaler.json",
    "thresholds.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_model_signing_keypair(priv_path: Path, pub_path: Path) -> None:
    """Generează o pereche de chei Ed25519 pentru semnarea artifact-urilor ML."""
    priv_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def sign_model_bundle(
    bundle_dir: Path,
    private_key_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Calculează SHA-256 pentru toate fișierele din bundle și semnează manifestul cu Ed25519."""
    hashes: dict[str, str] = {}
    for fname in BUNDLE_FILES:
        fpath = bundle_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing bundle artifact: {fpath}")
        hashes[fname] = sha256_file(fpath)

    manifest = {
        "bundle_format": "lumen-model-bundle-v5",
        "files_sha256": hashes,
        **metadata,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")

    priv_pem = private_key_path.read_bytes()
    priv_key = serialization.load_pem_private_key(priv_pem, password=None)
    if not isinstance(priv_key, Ed25519PrivateKey):
        raise TypeError("Expected Ed25519 private key for model signing")

    sig = priv_key.sign(canonical)
    (bundle_dir / "model_manifest.json").write_bytes(canonical)
    (bundle_dir / "model_manifest.sig").write_text(base64.b64encode(sig).decode("ascii"), encoding="utf-8")
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
    """Verifică semnătura Ed25519 și hash-urile SHA-256 înainte de a permite încărcarea ONNX."""
    bdir = Path(bundle_dir).resolve()
    manifest_path = bdir / "model_manifest.json"
    sig_path = bdir / "model_manifest.sig"

    if not manifest_path.is_file() or not sig_path.is_file():
        raise FileNotFoundError(f"Unsigned or incomplete model bundle in {bdir}")

    pub_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(pub_key, Ed25519PublicKey):
        raise TypeError("Expected Ed25519 public key for bundle verification")

    raw_manifest = manifest_path.read_bytes()
    sig_bytes = base64.b64decode(sig_path.read_text(encoding="utf-8").strip())

    try:
        pub_key.verify(sig_bytes, raw_manifest)
    except InvalidSignature as exc:
        raise ValueError("SECURITY ALERT: Model manifest Ed25519 signature verification FAILED!") from exc

    manifest = json.loads(raw_manifest.decode("utf-8"))
    expected_hashes = manifest.get("files_sha256", {})
    for fname in BUNDLE_FILES:
        fpath = bdir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing artifact declared in manifest: {fname}")
        actual = sha256_file(fpath)
        if actual != expected_hashes.get(fname):
            raise ValueError(f"SECURITY ALERT: SHA-256 mismatch for {fname}!")

    return VerifiedModelBundle(
        bundle_dir=bdir,
        onnx_path=bdir / "model.onnx",
        manifest=manifest,
        feature_schema=json.loads((bdir / "feature_schema.json").read_text(encoding="utf-8")),
        scaler=json.loads((bdir / "scaler.json").read_text(encoding="utf-8")),
        thresholds=json.loads((bdir / "thresholds.json").read_text(encoding="utf-8")),
    )