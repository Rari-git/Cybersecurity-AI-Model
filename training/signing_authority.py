from __future__ import annotations

from models.authority_lock import candidate_authority_lock

# Dedicated offline signing authority for evaluated Lumen V6 candidates.
# This module signs an evaluated candidate only. It never changes active_model.json,
# promotes a candidate, or performs rollback. Run it explicitly as a separate
# command under the locally authorized signing identity.

import argparse
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from models.bundle_loader import (
    _is_symlink_or_reparse,
    hash_unsigned_candidate_artifacts,
    sign_model_bundle_transactional,
    validate_unsigned_candidate_bundle,
    verify_and_load_bundle,
)
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.offline_guard import offline_network_guard
from training.promotion_gate import (
    _active_evidence_snapshot,
    _append_audit,
    _promotion_gates,
    evaluate_promotion_metrics,
    verify_candidate_evaluation_report,
)


@dataclass(frozen=True)
class SigningResult:
    signed: bool
    candidate_dir: str
    report_id: str
    report_sha256: str
    model_version: str
    signed_manifest_sha256: str
    reason: str
    audit_status: str = "written"
    audit_warning: str | None = None


def _require_explicit_signing_authority() -> None:
    if os.environ.get("LUMEN_MODEL_SIGNING_AUTHORITY") != "1":
        raise PermissionError(
            "signing disabled: set LUMEN_MODEL_SIGNING_AUTHORITY=1 "
            "in the explicitly authorized offline signing session"
        )


def _load_and_validate_key_pair(
    private_key_path: str | Path,
    public_key_path: str | Path,
    model_root: Path,
) -> tuple[Path, Path]:
    private_path = Path(private_key_path).expanduser()
    public_path = Path(public_key_path).expanduser()

    for label, path in (("private", private_path), ("public", public_path)):
        if not path.is_file() or _is_symlink_or_reparse(path):
            raise FileNotFoundError(
                f"authorized Ed25519 {label} key must be an existing regular file"
            )

    private_path = private_path.resolve(strict=True)
    public_path = public_path.resolve(strict=True)
    root = model_root.resolve()

    for label, path in (("private", private_path), ("public", public_path)):
        if path == root or root in path.parents:
            raise ValueError(f"Ed25519 {label} key must remain outside models/")

    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(), password=None
    )
    public_key = serialization.load_pem_public_key(public_path.read_bytes())

    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("configured signing private key is not Ed25519")
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("configured verification public key is not Ed25519")

    private_public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    configured_public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if not hmac.compare_digest(private_public_bytes, configured_public_bytes):
        raise ValueError(
            "configured private key does not match the existing runtime public trust anchor"
        )

    return private_path, public_path


def sign_evaluated_candidate(
    report_path: str | Path,
    *,
    model_root: str | Path = "models",
    db_dir: str | Path = "logs/db",
    private_key_path: str | Path,
    public_key_path: str | Path,
) -> SigningResult:
    # Revalidate and independently re-evaluate a report, then sign only.
    # The report digest is not authorization; metrics and gates are recomputed.
    _require_explicit_signing_authority()
    root = Path(model_root).expanduser().resolve()
    with candidate_authority_lock(root):
        public_path = Path(public_key_path).expanduser()
        private_path, public_path = _load_and_validate_key_pair(
            private_key_path, public_path, root
        )

        # Verify report integrity and exact candidate/active snapshots before work.
        report = verify_candidate_evaluation_report(
            report_path,
            model_root=root,
            public_key_path=public_path,
        )
        evaluation = report["evaluation"]
        if evaluation.get("all_gates_passed") is not True:
            raise PermissionError("candidate evaluation report did not pass all quality gates")
        expected_gates = evaluation.get("gates")
        expected_metrics = evaluation.get("metrics")
        if not isinstance(expected_gates, dict) or not isinstance(expected_metrics, dict):
            raise ValueError("evaluation report metrics or gates are malformed")

        candidate_record = report["candidate"]
        candidate_parts = PurePosixPath(candidate_record["relative_path"])
        candidate_dir = root.joinpath(*candidate_parts.parts)
        candidate = validate_unsigned_candidate_bundle(candidate_dir)

        # Verify that the private key is still the existing public trust anchor,
        # and that the report remains valid before the expensive scoring operation.
        _load_and_validate_key_pair(private_path, public_path, root)
        manager = LumenStorageManager(
            paths=DBPaths.from_dir(db_dir),
            enable_writers=False,
        )
        try:
            with offline_network_guard():
                active_record = report["active"]
                active_parts = PurePosixPath(active_record["bundle_relative_path"])
                active_dir = root.joinpath(*active_parts.parts)
                current_metrics = evaluate_promotion_metrics(
                    manager,
                    candidate.bundle_dir,
                    active_dir,
                    public_path,
                    candidate.metadata["heldout_sessions"],
                    candidate.metadata.get("curation_digests"),
                )
        finally:
            manager.close()

        current_gates = _promotion_gates(current_metrics, candidate.metadata)
        if current_metrics != expected_metrics or current_gates != expected_gates:
            raise RuntimeError(
                "current shadow metrics or gates differ from the evaluation report; signing refused"
            )
        if not all(current_gates.values()):
            raise PermissionError("independent signing evaluation did not pass every gate")

        # Revalidate immediately before signing. The candidate must still be unsigned,
        # unchanged, and compared against the same active pointer/bundle snapshot.
        report_again = verify_candidate_evaluation_report(
            report_path,
            model_root=root,
            public_key_path=public_path,
            candidate_dir=candidate_dir,
        )
        if report_again.get("report_sha256") != report.get("report_sha256"):
            raise RuntimeError("evaluation report changed during signing preflight")

        candidate = validate_unsigned_candidate_bundle(candidate_dir)
        actual_candidate_artifacts = hash_unsigned_candidate_artifacts(candidate.bundle_dir)
        expected_candidate_artifacts = candidate_record.get("artifacts_sha256")
        if actual_candidate_artifacts != expected_candidate_artifacts:
            raise RuntimeError("candidate artifact hashes changed before signing")

        expected_active = report["active"]
        current_active = _active_evidence_snapshot(
            root,
            root / expected_active["pointer_relative_path"],
            public_path,
        )
        if current_active != expected_active:
            raise RuntimeError("active model or pointer changed before signing")

        evaluated_at_ns = int(report["evaluated_at_unix_ns"])
        signing_at_ns = time.time_ns()
        attestation = {
            "schema": "lumen-offline-signing-attestation-v1",
            "report_id": report["report_id"],
            "report_sha256": report["report_sha256"],
            "evaluated_at_unix_ns": evaluated_at_ns,
            "signed_at_unix_ns": signing_at_ns,
            "candidate_manifest_sha256": candidate_record["manifest_sha256"],
            "active_manifest_sha256": expected_active["manifest_sha256"],
            "active_pointer_sha256": expected_active["pointer_sha256"],
            "evaluation_gates": current_gates,
            "evaluation_metrics_sha256": hashlib.sha256(
                json.dumps(
                    current_metrics,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        signed_metadata = dict(candidate.metadata)
        signed_metadata.update({
            "promotion_gate": "passed",
            "promotion_gates": current_gates,
            "promotion_metrics": current_metrics,
            "signed_by_authority": "local_offline_ed25519_signing_authority",
            "evaluation_attestation": attestation,
        })

        # This operation ends after signature verification; it never changes the active pointer.
        try:
            signed_manifest = sign_model_bundle_transactional(
                candidate.bundle_dir,
                private_path,
                public_path,
                metadata=signed_metadata,
            )

            # Confirm the signed manifest covers precisely the evaluated candidate
            # bytes, plus the original unsigned provenance manifest.
            expected_signed_files = {
                **expected_candidate_artifacts,
                "manifest.json": candidate_record["manifest_sha256"],
            }
            actual_signed_files = signed_manifest.get("files_sha256")
            if actual_signed_files != expected_signed_files:
                raise RuntimeError("candidate files changed during signing")

            verified = verify_and_load_bundle(candidate.bundle_dir, public_path)
            if verified.manifest.get("evaluation_attestation") != attestation:
                raise RuntimeError("signed model does not contain the expected evaluation attestation")

            # Verify deployment state was not changed by this signer.
            active_after = _active_evidence_snapshot(
                root,
                root / expected_active["pointer_relative_path"],
                public_path,
            )
            if active_after != expected_active:
                raise RuntimeError("active model or pointer changed while signing candidate")
        except BaseException:
            # Never leave an invalid, partially verified signature in the candidate.
            for name in ("model_manifest.sig", "model_manifest.json"):
                artifact = candidate.bundle_dir / name
                if artifact.is_file() and not _is_symlink_or_reparse(artifact):
                    artifact.unlink()
            raise

        from models.bundle_loader import sha256_file

        manifest_digest = sha256_file(candidate.bundle_dir / "model_manifest.json")
        audit_warning = None
        try:
            _append_audit(root / "signing_authority_history.jsonl", {
                "at_ns": signing_at_ns,
                "event": "candidate_signed_without_deployment",
                "report_id": report["report_id"],
                "report_sha256": report["report_sha256"],
                "candidate": candidate.metadata.get("model_version"),
                "candidate_dir": candidate_record["relative_path"],
                "signed_manifest_sha256": manifest_digest,
                "active_bundle_unchanged": True,
            })
        except Exception as exc:
            # Signing is already verified; audit failure must not hide success.
            audit_warning = f"{type(exc).__name__}: {exc}"

        return SigningResult(
            signed=True,
            candidate_dir=str(candidate.bundle_dir),
            report_id=report["report_id"],
            report_sha256=report["report_sha256"],
            model_version=str(verified.manifest.get("model_version") or "unknown"),
            signed_manifest_sha256=manifest_digest,
            reason="SIGNED_AND_VERIFIED_NOT_DEPLOYED",
            audit_status="failed" if audit_warning else "written",
            audit_warning=audit_warning,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit Lumen V6 offline candidate signing authority (sign-only)"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--public-key", required=True)
    args = parser.parse_args()

    result = sign_evaluated_candidate(
        args.report,
        model_root=args.model_root,
        db_dir=args.db_dir,
        private_key_path=args.private_key,
        public_key_path=args.public_key,
    )
    print(json.dumps({
        "signed": result.signed,
        "candidate_dir": result.candidate_dir,
        "report_id": result.report_id,
        "report_sha256": result.report_sha256,
        "model_version": result.model_version,
        "signed_manifest_sha256": result.signed_manifest_sha256,
        "reason": result.reason,
        "audit_status": result.audit_status,
        "audit_warning": result.audit_warning,
        "deployed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
