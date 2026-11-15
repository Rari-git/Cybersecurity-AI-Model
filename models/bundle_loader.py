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
from dataclasses import dataclass, field
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


def _assert_plain_file(path: Path) -> None:
    """Reject symlinks/reparse points and non-regular entries before opening."""
    if _is_symlink_or_reparse(path):
        raise ValueError(f"bundle file must not be a symlink or reparse point: {path}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"bundle file is not a regular file: {path}")


def _safe_bundle_root(bundle_dir: str | os.PathLike[str]) -> Path:
    requested = Path(bundle_dir)
    if _is_symlink_or_reparse(requested):
        raise ValueError(f"model bundle root must not be a symlink or reparse point: {requested}")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"model bundle root is not a directory: {root}")
    return root


def sha256_file(path: Path) -> str:
    """Hash a regular file using a no-follow descriptor where supported."""
    path = Path(path)
    _assert_plain_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"bundle file is not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(fd)
    # Recheck after reading to detect path replacement during hashing.
    _assert_plain_file(path)
    return digest.hexdigest()


def _is_symlink_or_reparse(path: Path) -> bool:
    """Detect symlinks and Windows junction/reparse points without following them."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        info = path.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = getattr(info, "st_file_attributes", 0)
        return bool(attributes & reparse_flag)
    except FileNotFoundError as exc:
        raise ValueError(f"bundle entry disappeared during scan: {path}") from exc


def _artifact_files(bundle_dir: Path) -> dict[str, Path]:
    """Return every regular artifact, rejecting links before directory traversal.

    ``Path.rglob`` plus ``is_dir`` follows directory symlinks on some platforms.
    Use an explicit no-follow scandir walk and reject Windows reparse points too.
    """
    root = _safe_bundle_root(bundle_dir)

    artifacts: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        if _is_symlink_or_reparse(directory):
            raise ValueError(f"bundle directory must not be a symlink or reparse point: {directory}")
        try:
            with os.scandir(directory) as iterator:
                for entry in sorted(iterator, key=lambda item: item.name):
                    path = Path(entry.path)
                    # Must precede is_dir/is_file: both may follow links.
                    if _is_symlink_or_reparse(path):
                        raise ValueError(f"bundle artifacts must not be symlinks or reparse points: {path}")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                        continue
                    if entry.name in (MANIFEST_NAME, SIGNATURE_NAME):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError(f"bundle artifacts must be regular files: {path}")
                    relative = path.relative_to(root).as_posix()
                    _safe_relative_path(relative)
                    resolved = path.resolve(strict=True)
                    if root not in resolved.parents:
                        raise ValueError(f"bundle artifact escapes bundle directory: {relative}")
                    artifacts[relative] = path
        except OSError as exc:
            raise ValueError(f"cannot safely scan model bundle directory {directory}: {exc}") from exc
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
    root = _safe_bundle_root(bundle_dir)
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



def sign_model_bundle_transactional(
    bundle_dir: Path,
    private_key_path: Path,
    public_key_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Stage, verify, then publish a signed candidate.

    The original unsigned manifest.json is preserved.

    Ordinary Python-level publication failures remove partially published
    runtime signature files and restore the candidate to its unsigned state.

    This is not a power-loss transaction. Durable crash recovery is reserved
    for the later signing/deployment journal stage.
    """
    import shutil
    import tempfile

    root = _safe_bundle_root(bundle_dir)

    unsigned_manifest = root / "manifest.json"
    if not unsigned_manifest.is_file():
        raise FileNotFoundError(
            "unsigned candidate manifest.json is missing"
        )

    if (root / MANIFEST_NAME).exists() or (root / SIGNATURE_NAME).exists():
        raise ValueError(
            "transactional signing requires an unsigned candidate"
        )

    # Snapshot every original candidate artifact.
    original_snapshot = {
        name: sha256_file(path)
        for name, path in _artifact_files(root).items()
    }

    if "manifest.json" not in original_snapshot:
        raise ValueError(
            "candidate provenance manifest is not in the artifact set"
        )

    staging_parent = Path(tempfile.mkdtemp(
        prefix=".lumen-signing-stage-",
        dir=root.parent,
    ))

    staged_bundle = staging_parent / "bundle"

    try:
        staged_bundle.mkdir()

        # Copy only regular files discovered by the no-follow scanner.
        for relative, source in _artifact_files(root).items():
            target = staged_bundle.joinpath(
                *PurePosixPath(relative).parts
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        staged_snapshot = {
            name: sha256_file(path)
            for name, path in _artifact_files(staged_bundle).items()
        }

        if staged_snapshot != original_snapshot:
            raise RuntimeError(
                "staged candidate differs from original candidate"
            )

        # Sign only the staging copy.
        signed_manifest = sign_model_bundle(
            staged_bundle,
            private_key_path,
            metadata=metadata,
        )

        # Verify staged output before publishing anything.
        verified = verify_and_load_bundle(
            staged_bundle,
            public_key_path,
        )

        if verified.manifest != signed_manifest:
            raise RuntimeError(
                "staged signed manifest verification mismatch"
            )

        # Confirm original candidate did not change during signing.
        current_snapshot = {
            name: sha256_file(path)
            for name, path in _artifact_files(root).items()
        }

        if current_snapshot != original_snapshot:
            raise RuntimeError(
                "original candidate changed during signing"
            )

        staged_manifest = staged_bundle / MANIFEST_NAME
        staged_signature = staged_bundle / SIGNATURE_NAME

        manifest_bytes = staged_manifest.read_bytes()
        signature_bytes = staged_signature.read_bytes()

        manifest_tmp = staging_parent / f".{MANIFEST_NAME}.publish"
        signature_tmp = staging_parent / f".{SIGNATURE_NAME}.publish"

        for temp_path in (manifest_tmp, signature_tmp):
            if temp_path.exists() or temp_path.is_symlink():
                raise FileExistsError(
                    f"unexpected publication temporary file: {temp_path}"
                )

        try:
            # Prepare both files before replacing either destination.
            for temp_path, payload in (
                (manifest_tmp, manifest_bytes),
                (signature_tmp, signature_bytes),
            ):
                with temp_path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())

            # Last candidate-integrity check before publication.
            final_snapshot = {
                name: sha256_file(path)
                for name, path in _artifact_files(root).items()
            }

            if final_snapshot != original_snapshot:
                raise RuntimeError(
                    "candidate changed immediately before publication"
                )

            # Two atomic file replacements. The pair is not crash-atomic.
            os.replace(
                manifest_tmp,
                root / MANIFEST_NAME,
            )

            os.replace(
                signature_tmp,
                root / SIGNATURE_NAME,
            )

            # Verify the published candidate itself.
            final_verified = verify_and_load_bundle(
                root,
                public_key_path,
            )

            if final_verified.manifest != signed_manifest:
                raise RuntimeError(
                    "published candidate differs from staged signed bundle"
                )

            return signed_manifest

        except BaseException:
            # This operation began with an unsigned candidate.
            # Remove partial runtime signing artifacts on ordinary failure.
            for name in (MANIFEST_NAME, SIGNATURE_NAME):
                target = root / name

                if target.exists() and not target.is_symlink():
                    target.unlink()

            raise

        finally:
            for temp_path in (
                staging_parent / f".{MANIFEST_NAME}.publish",
                staging_parent / f".{SIGNATURE_NAME}.publish",
            ):
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


@dataclass(frozen=True)
class VerifiedModelBundle:
    bundle_dir: Path
    onnx_path: Path
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    scaler: dict[str, Any]
    thresholds: dict[str, Any]
    # Keep the private snapshot alive for as long as this verified object lives.
    _snapshot_owner: Any = field(repr=False, compare=False)

    @property
    def snapshot_dir(self) -> Path:
        return self.onnx_path.parent

    def close(self) -> None:
        # Explicitly release the verified snapshot when its consumer is finished.
        self._snapshot_owner.cleanup()


def _read_plain_file_bytes(path: Path) -> bytes:
    # Read a regular file through an opened descriptor, refusing links.
    path = Path(path)
    _assert_plain_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"bundle file is not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(fd)
    _assert_plain_file(path)
    return payload


def _copy_verified_artifact(source: Path, target: Path, expected_digest: str) -> None:
    # Hash exactly the bytes copied from one opened source descriptor.
    source = Path(source)
    target = Path(target)
    _assert_plain_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(source, flags)
    digest = hashlib.sha256()
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"bundle artifact is not a regular file: {source}")
        with os.fdopen(fd, "rb", closefd=False) as reader:
            with target.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
    finally:
        os.close(fd)

    observed = digest.hexdigest()
    if not hmac.compare_digest(observed, expected_digest):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"SECURITY ALERT: SHA-256 mismatch while snapshotting {source.name}")

    if not hmac.compare_digest(sha256_file(target), expected_digest):
        raise ValueError(f"SECURITY ALERT: verified snapshot changed during creation: {target}")

    # Read-only is defense in depth; private temp-directory ACLs are the
    # primary cross-process boundary. The owning OS account can still override it.
    try:
        target.chmod(stat.S_IRUSR)
    except OSError:
        pass


def verify_and_load_bundle(bundle_dir: str | Path, public_key_path: str | Path) -> VerifiedModelBundle:
    # Verify a signed bundle and return data/model paths from a private snapshot.
    import tempfile

    root = _safe_bundle_root(bundle_dir)
    manifest_path = root / MANIFEST_NAME
    signature_path = root / SIGNATURE_NAME
    if not manifest_path.exists() or not signature_path.exists():
        raise FileNotFoundError(f"unsigned or incomplete model bundle in {root}")

    raw_manifest = _read_plain_file_bytes(manifest_path)
    raw_signature = _read_plain_file_bytes(signature_path)
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("Expected Ed25519 public key for bundle verification")

    try:
        signature = base64.b64decode(raw_signature.decode("ascii").strip(), validate=True)
        public_key.verify(signature, raw_manifest)
    except (InvalidSignature, ValueError, UnicodeDecodeError) as exc:
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
        raise ValueError(
            f"bundle file set mismatch: unhashed={missing_hashes}, missing={missing_files}"
        )

    normalized_hashes: dict[str, str] = {}
    for name, digest in expected.items():
        _safe_relative_path(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            raise ValueError(f"invalid SHA-256 declaration for {name}")
        normalized_hashes[name] = digest.lower()

    # Snapshot outside the mutable source bundle. On Windows the directory
    # inherits the current user's temp-directory ACL; POSIX temp dirs are private.
    owner = tempfile.TemporaryDirectory(prefix="lumen-verified-bundle-")
    snapshot_root = Path(owner.name)
    try:
        for name, source_path in artifacts.items():
            relative = _safe_relative_path(name)
            target = snapshot_root.joinpath(*relative.parts)
            _copy_verified_artifact(source_path, target, normalized_hashes[name])

        (snapshot_root / MANIFEST_NAME).write_bytes(raw_manifest)
        (snapshot_root / SIGNATURE_NAME).write_bytes(raw_signature)

        snapshot_artifacts = _artifact_files(snapshot_root)
        if set(snapshot_artifacts) != expected_names:
            raise ValueError("verified snapshot artifact set changed during creation")
        for name, path in snapshot_artifacts.items():
            if not hmac.compare_digest(sha256_file(path), normalized_hashes[name]):
                raise ValueError(f"SECURITY ALERT: snapshot artifact verification failed: {name}")

        def read_snapshot_json(name: str) -> dict[str, Any]:
            value = json.loads((snapshot_root / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{name} must contain a JSON object")
            return value

        return VerifiedModelBundle(
            bundle_dir=root,
            onnx_path=snapshot_root / "model.onnx",
            manifest=manifest,
            feature_schema=read_snapshot_json("feature_schema.json"),
            scaler=read_snapshot_json("scaler.json"),
            thresholds=read_snapshot_json("thresholds.json"),
            _snapshot_owner=owner,
        )
    except BaseException:
        owner.cleanup()
        raise


def hash_unsigned_candidate_artifacts(bundle_dir: str | os.PathLike[str]) -> dict[str, str]:
    # Hash every candidate artifact except manifest.json, which stores these hashes.
    root = _safe_bundle_root(bundle_dir)
    artifacts = _artifact_files(root)
    return {
        name: sha256_file(path)
        for name, path in sorted(artifacts.items())
        if name != "manifest.json"
    }


@dataclass(frozen=True)
class UnsignedCandidateBundle:
    bundle_dir: Path
    metadata: dict[str, Any]
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluationBundle:
    # Integrity-checked candidate inputs; not a runtime trust object.
    bundle_dir: Path
    onnx_path: Path
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    scaler: dict[str, Any]
    thresholds: dict[str, Any]


def validate_unsigned_candidate_bundle(bundle_dir: str | Path) -> UnsignedCandidateBundle:
    """Validate a local, unsigned, curated-clean candidate before shadow evaluation.

    This function performs no signing and does not load ONNX. It validates the
    candidate's provenance and no-follow artifact set; runtime verification
    remains ``verify_and_load_bundle`` after offline signing.
    """
    root = _safe_bundle_root(bundle_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("candidate manifest.json is missing")
    _assert_plain_file(manifest_path)
    if (root / MANIFEST_NAME).exists() or (root / SIGNATURE_NAME).exists():
        raise ValueError("candidate must remain unsigned until promotion gates pass")
    artifacts = _artifact_files(root)
    if "manifest.json" not in artifacts:
        raise ValueError("candidate manifest must be a regular hashed artifact")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate manifest.json is invalid") from exc
    if not isinstance(metadata, dict):
        raise ValueError("candidate manifest must be a JSON object")

    declared_hashes = metadata.get("candidate_files_sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("candidate artifact SHA-256 declarations are missing")
    candidate_artifacts = {
        name: path for name, path in artifacts.items() if name != "manifest.json"
    }
    if set(declared_hashes) != set(candidate_artifacts):
        raise ValueError("candidate artifact file set mismatch")
    for name, path in candidate_artifacts.items():
        digest = declared_hashes.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            raise ValueError(f"invalid candidate SHA-256 declaration for {name}")
        if not hmac.compare_digest(sha256_file(path), digest.lower()):
            raise ValueError(f"candidate SHA-256 mismatch for {name}")

    if metadata.get("offline_training") is not True:
        raise ValueError("candidate was not produced by the offline training pipeline")
    if metadata.get("dataset_policy") != "approved_clean_baseline_only":
        raise ValueError("candidate dataset policy is not approved_clean_baseline_only")
    train = metadata.get("train_sessions")
    validation = metadata.get("val_sessions")
    heldout = metadata.get("heldout_sessions")
    cusum = metadata.get("cusum_calibration_sessions")
    for label, value in (("train_sessions", train), ("val_sessions", validation),
                         ("heldout_sessions", heldout), ("cusum_calibration_sessions", cusum)):
        if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"candidate {label} must be a non-empty list of session IDs")
    if set(train) & set(validation) or set(train) & set(heldout) or set(validation) & set(heldout):
        raise ValueError("training, validation and held-out sessions must be disjoint")
    if any(len(set(value)) != len(value) for value in (train, validation, heldout, cusum)):
        raise ValueError("candidate session partition lists must not contain duplicates")
    if metadata.get("split_unit") != "capture_run_id":
        raise ValueError("unsigned V6 candidate must be partitioned by capture_run_id")
    capture_map = metadata.get("capture_run_ids_by_session")
    if not isinstance(capture_map, dict) or set(capture_map) != (set(train) | set(validation) | set(heldout)):
        raise ValueError("candidate must map every partition session to its capture_run_id")
    for session_id, capture_run_id in capture_map.items():
        if (
            not isinstance(capture_run_id, str)
            or len(capture_run_id) != 32
            or capture_run_id != capture_run_id.lower()
            or any(ch not in "0123456789abcdef" for ch in capture_run_id)
            or not session_id.startswith(f"baseline-lab-{capture_run_id}-")
        ):
            raise ValueError("candidate contains invalid session capture_run_id provenance")

    run_partitions = (
        ("train_capture_runs", train),
        ("val_capture_runs", validation),
        ("heldout_capture_runs", heldout),
    )
    capture_sets: dict[str, set[str]] = {}
    for field, session_partition in run_partitions:
        values = metadata.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str)
                or len(value) != 32
                or value != value.lower()
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"candidate {field} must be a non-empty unique list of capture_run_id values")
        observed = {capture_map[session_id] for session_id in session_partition}
        if observed != set(values):
            raise ValueError(f"candidate {field} does not match its session capture_run_id provenance")
        capture_sets[field] = set(values)
    if (
        capture_sets["train_capture_runs"] & capture_sets["val_capture_runs"]
        or capture_sets["train_capture_runs"] & capture_sets["heldout_capture_runs"]
        or capture_sets["val_capture_runs"] & capture_sets["heldout_capture_runs"]
    ):
        raise ValueError("candidate capture_run_id partitions must be disjoint")
    if set(capture_map.values()) != (
        capture_sets["train_capture_runs"]
        | capture_sets["val_capture_runs"]
        | capture_sets["heldout_capture_runs"]
    ):
        raise ValueError("candidate capture_run_id partition coverage is incomplete")
    if not set(cusum).issubset(set(train)):
        raise ValueError("CUSUM calibration may use training sessions only")
    curation_digests = metadata.get("curation_digests")
    required_session_ids = set(train) | set(validation) | set(heldout)
    if not isinstance(curation_digests, dict) or set(curation_digests) != required_session_ids:
        raise ValueError("candidate must bind every train/validation/held-out session to its curation digest")
    if any(not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower())
           for value in curation_digests.values()):
        raise ValueError("candidate curation digests must be lowercase SHA-256 hex strings")
    if not isinstance(metadata.get("global_threshold"), (int, float)) or metadata["global_threshold"] <= 0:
        raise ValueError("candidate global_threshold must be positive")
    if not isinstance(metadata.get("validation_normal_error_p95"), (int, float)):
        raise ValueError("candidate validation normal error metric is missing")
    return UnsignedCandidateBundle(root, metadata, tuple(sorted(artifacts)))


def load_unsigned_candidate_for_evaluation(
    bundle_dir: str | os.PathLike[str],
) -> CandidateEvaluationBundle:
    # Load a hash-checked unsigned candidate for isolated offline shadow scoring.
    # This type is deliberately not accepted by the online runtime.
    candidate = validate_unsigned_candidate_bundle(bundle_dir)
    root = candidate.bundle_dir

    def read_json(name: str) -> dict[str, Any]:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value

    return CandidateEvaluationBundle(
        bundle_dir=root,
        onnx_path=root / "model.onnx",
        manifest=candidate.metadata,
        feature_schema=read_json("feature_schema.json"),
        scaler=read_json("scaler.json"),
        thresholds=read_json("thresholds.json"),
    )
