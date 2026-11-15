from __future__ import annotations

from models.authority_lock import candidate_authority_lock

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from models.bundle_loader import verify_and_load_bundle, _is_symlink_or_reparse, sha256_file

POINTER_SCHEMA = 1
DEPLOYMENT_SIGNER = "local_offline_ed25519_signing_authority"
ATTESTATION_SCHEMA = "lumen-offline-signing-attestation-v1"


def _require_deployment_opt_in() -> None:
    if os.environ.get("LUMEN_MODEL_DEPLOYMENT_AUTHORITY") != "1":
        raise PermissionError("deployment authority disabled; set LUMEN_MODEL_DEPLOYMENT_AUTHORITY=1")


def _relative_bundle(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("active bundle path must be a non-empty string")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError("active bundle path must be a safe relative path")
    resolved = (root / Path(*p.parts)).resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise ValueError("active bundle must remain inside model root")
    if not resolved.is_dir():
        raise ValueError("active bundle path must point to a directory")
    return resolved


def _pointer_digest(pointer: Path) -> str:
    if _is_symlink_or_reparse(pointer):
        raise ValueError("active-model pointer must not be a symlink or reparse point")
    if not pointer.is_file():
        raise FileNotFoundError(f"active-model pointer is missing: {pointer}")
    return hashlib.sha256(pointer.read_bytes()).hexdigest()


@contextmanager
def _pointer_lock(pointer: Path) -> Iterator[None]:
    # Serialize deployment and rollback across processes.
    lock_path = pointer.with_name(pointer.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A missing lock file is normal on first use. Inspect it with lstat only
    # when it already exists; this also detects dangling symlinks/reparse points.
    try:
        lock_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if _is_symlink_or_reparse(lock_path):
            raise ValueError("active-model lock must not be a symlink or reparse point")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_pointer_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The destination is normally absent on first deployment. Reject an existing
    # symlink/reparse point, but do not treat FileNotFoundError as an integrity error.
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        if _is_symlink_or_reparse(path):
            raise ValueError("pointer destination must not be a symlink or reparse point")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name == "posix":
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_active_bundle(model_root="models", pointer_path="models/active_model.json",
                          fallback_bundle="models/lumen_model_bundle") -> Path:
    # Read-only resolution is used by evaluation and signing as well as runtime.
    # Deployment authorization belongs only on operations that mutate pointers.
    root = Path(model_root).resolve()
    pointer = Path(pointer_path)
    if not pointer.is_absolute():
        pointer = pointer.resolve()
    if pointer.is_file():
        doc = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != POINTER_SCHEMA:
            raise ValueError("invalid active-model pointer schema")
        return _relative_bundle(root, doc["active_bundle"])
    fallback = Path(fallback_bundle)
    if not fallback.is_absolute():
        fallback = fallback.resolve()
    if not fallback.is_dir():
        raise FileNotFoundError(f"active model pointer and fallback bundle are absent: {pointer}")
    return fallback


def _validate_attestation(verified: Any) -> dict[str, Any]:
    manifest = verified.manifest
    if manifest.get("signed_by_authority") != DEPLOYMENT_SIGNER:
        raise PermissionError("candidate was not signed by the approved offline signing authority")
    att = manifest.get("evaluation_attestation")
    if not isinstance(att, dict) or att.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("signed candidate has no supported evaluation attestation")
    for field in ("report_sha256", "candidate_manifest_sha256", "active_manifest_sha256",
                  "active_pointer_sha256", "evaluation_metrics_sha256"):
        value = att.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"evaluation attestation field is invalid: {field}")
    if not isinstance(att.get("report_id"), str) or not att["report_id"].strip():
        raise ValueError("evaluation attestation report_id is invalid")
    if not isinstance(att.get("evaluated_at_unix_ns"), int) or not isinstance(att.get("signed_at_unix_ns"), int):
        raise ValueError("evaluation attestation timestamps are invalid")
    gates = att.get("evaluation_gates")
    if not isinstance(gates, dict) or not gates or not all(v is True for v in gates.values()):
        raise PermissionError("evaluation attestation does not record every gate as passed")
    if manifest.get("promotion_gate") != "passed" or manifest.get("promotion_gates") != gates:
        raise ValueError("signed promotion metadata does not match the evaluation attestation")
    metrics = manifest.get("promotion_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("signed promotion metrics are missing")
    metrics_digest = hashlib.sha256(
        json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if metrics_digest != att["evaluation_metrics_sha256"]:
        raise ValueError("signed promotion metrics do not match the evaluation attestation")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or files.get("manifest.json") != att["candidate_manifest_sha256"]:
        raise ValueError("candidate provenance manifest digest does not match its signed file list")
    return att



def _pointer_payload_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _optional_regular_pointer_info(path: Path):
    # Missing pointer files are valid before the first deployment. Use lstat
    # directly instead of the bundle-scan helper, which treats missing entries
    # as integrity errors.
    stat_module = __import__("stat")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None

    reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    if stat_module.S_ISLNK(info.st_mode) or (attributes & reparse_flag):
        raise ValueError("model pointer must not be a symlink or reparse point")
    if not stat_module.S_ISREG(info.st_mode):
        raise ValueError("model pointer must be a regular file")
    return info


def _read_regular_pointer_bytes(path: Path) -> bytes | None:
    info = _optional_regular_pointer_info(path)
    if info is None:
        return None
    return path.read_bytes()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # An absent destination is expected when previous_model.json is first created.
    _optional_regular_pointer_info(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        if os.name == "posix":
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _delete_file_durable(path: Path) -> None:
    # Deleting an already-absent optional pointer is an idempotent no-op.
    info = _optional_regular_pointer_info(path)
    if info is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    if os.name == "posix":
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _deployment_failpoint(name: str, transaction_id: str) -> None:
    """No-op hook used by deterministic crash-injection tests."""
    return None


def _prepare_deployment_transaction(
    root: Path,
    pointer: Path,
    previous_pointer: Path,
    current_raw: bytes,
    current_doc: dict[str, Any],
    target_payload: dict[str, Any],
    target_bundle: Path,
    active_bundle: Path,
    operation: str,
    attestation: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    import base64
    import uuid
    from models.deployment_journal import append_event, read_journal, pending_transactions

    records = read_journal(root, repair_torn_tail=True)
    if pending_transactions(records):
        raise RuntimeError("an unresolved deployment transaction remains; recover it before starting another")
    previous_raw = _read_regular_pointer_bytes(previous_pointer)
    txid = uuid.uuid4().hex
    previous_doc = None
    if previous_raw is not None:
        try:
            previous_doc = json.loads(previous_raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("existing previous-model pointer is malformed") from exc
        if not isinstance(previous_doc, dict) or previous_doc.get("schema") != POINTER_SCHEMA:
            raise ValueError("existing previous-model pointer has an unsupported schema")

    # Keep the public active-pointer schema unchanged; transaction identity lives
    # in the journal and is correlated by the exact expected pointer digest.
    after_active_raw = _pointer_payload_bytes(target_payload)
    before_active_doc = current_doc
    payload = {
        "operation": operation,
        "pointer_name": pointer.name,
        "previous_pointer_name": previous_pointer.name,
        "before_active_pointer_sha256": hashlib.sha256(current_raw).hexdigest(),
        "before_active_bundle": before_active_doc["active_bundle"],
        "before_previous_pointer_exists": previous_raw is not None,
        "before_previous_pointer_b64": (
            base64.b64encode(previous_raw).decode("ascii") if previous_raw is not None else None
        ),
        "before_previous_pointer_sha256": (
            hashlib.sha256(previous_raw).hexdigest() if previous_raw is not None else None
        ),
        "after_previous_pointer_sha256": hashlib.sha256(current_raw).hexdigest(),
        "after_active_pointer_sha256": hashlib.sha256(after_active_raw).hexdigest(),
        "target_bundle": target_payload["active_bundle"],
        "before_manifest_sha256": sha256_file(active_bundle / "model_manifest.json"),
        "target_manifest_sha256": sha256_file(target_bundle / "model_manifest.json"),
        "report_id": attestation.get("report_id") if attestation else None,
        "report_sha256": attestation.get("report_sha256") if attestation else None,
    }
    prepared = append_event(root, "prepare", txid, payload)
    return txid, payload, prepared


def _commit_deployment_transaction(
    root: Path, transaction_id: str, tx_payload: dict[str, Any], prepare_record: dict[str, Any],
    *, recovered: bool = False,
) -> dict[str, Any]:
    from models.deployment_journal import append_event

    payload = {
        "operation": tx_payload["operation"],
        "pointer_name": tx_payload["pointer_name"],
        "previous_pointer_name": tx_payload["previous_pointer_name"],
        "before_active_bundle": tx_payload["before_active_bundle"],
        "target_bundle": tx_payload["target_bundle"],
        "before_active_pointer_sha256": tx_payload["before_active_pointer_sha256"],
        "after_active_pointer_sha256": tx_payload["after_active_pointer_sha256"],
        "before_manifest_sha256": tx_payload["before_manifest_sha256"],
        "target_manifest_sha256": tx_payload["target_manifest_sha256"],
        "prepare_record_sha256": prepare_record["record_sha256"],
        "recovered": bool(recovered),
    }
    return append_event(root, "commit", transaction_id, payload)


def _validate_rollback_provenance(
    records: list[dict[str, Any]],
    pointer_name: str,
    previous_pointer_name: str,
    current_doc: dict[str, Any],
    previous_doc: dict[str, Any],
    current_digest: str,
    previous_digest: str,
) -> dict[str, Any]:
    commits = [
        record for record in records
        if record["event"] == "commit" and record["payload"].get("pointer_name") == pointer_name
    ]
    if not commits:
        raise PermissionError("rollback target has no committed deployment provenance")
    latest = commits[-1]["payload"]
    if latest.get("previous_pointer_name") != previous_pointer_name:
        raise PermissionError("rollback pointer path differs from committed deployment journal provenance")
    if (
        current_digest != latest.get("after_active_pointer_sha256")
        or current_doc.get("active_bundle") != latest.get("target_bundle")
    ):
        raise PermissionError("active pointer does not match the latest committed deployment journal entry")
    if (
        previous_digest != latest.get("before_active_pointer_sha256")
        or previous_doc.get("active_bundle") != latest.get("before_active_bundle")
    ):
        raise PermissionError("rollback target is not the prior active model recorded by the deployment journal")
    return latest


def _validate_pending_rollback_provenance(
    records: list[dict[str, Any]], prepare_record: dict[str, Any], tx_payload: dict[str, Any]
) -> None:
    prior_commits = [
        record for record in records
        if record["event"] == "commit"
        and record["seq"] < prepare_record["seq"]
        and record["payload"].get("pointer_name") == tx_payload["pointer_name"]
    ]
    if not prior_commits:
        raise PermissionError("incomplete rollback has no prior committed deployment provenance")
    prior = prior_commits[-1]["payload"]
    if (
        tx_payload.get("before_active_bundle") != prior.get("target_bundle")
        or tx_payload.get("target_bundle") != prior.get("before_active_bundle")
        or tx_payload.get("before_active_pointer_sha256") != prior.get("after_active_pointer_sha256")
        or tx_payload.get("previous_pointer_name") != prior.get("previous_pointer_name")
    ):
        raise PermissionError("incomplete rollback target is not authorized by committed deployment history")


def recover_incomplete_deployments(
    model_root="models",
    public_key_path="config/model_keys/model_signing_ed25519.pub",
    pointer_path=None,
    previous_pointer_path=None,
) -> list[dict[str, Any]]:
    """Reconcile one interrupted pointer transaction; ambiguous states fail closed."""
    from models.authority_lock import candidate_authority_lock
    from models.deployment_journal import (
        append_event, pending_transactions, read_journal,
    )

    _require_deployment_opt_in()
    root = Path(model_root).expanduser().resolve(strict=True)
    pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
    if not pointer.is_absolute():
        pointer = pointer.resolve()
    previous_pointer = (
        Path(previous_pointer_path) if previous_pointer_path is not None
        else root / "previous_model.json"
    )
    if not previous_pointer.is_absolute():
        previous_pointer = previous_pointer.resolve()
    if pointer.resolve().parent != root or previous_pointer.resolve().parent != root:
        raise ValueError("deployment journal pointers must be direct children of model_root")
    if pointer == previous_pointer:
        raise ValueError("active and previous pointers must be distinct")

    outcomes: list[dict[str, Any]] = []
    with candidate_authority_lock(root):
        records = read_journal(root, repair_torn_tail=True)
        pending = pending_transactions(records)
        if not pending:
            return outcomes
        if len(pending) != 1:
            raise RuntimeError("multiple unresolved deployment transactions; manual recovery required")
        prepare = pending[0]
        tx = prepare["payload"]
        if tx.get("pointer_name") != pointer.name or tx.get("previous_pointer_name") != previous_pointer.name:
            raise RuntimeError("pending deployment transaction references different pointer paths")

        with _pointer_lock(pointer):
            active_digest = _pointer_digest(pointer)
            previous_raw = _read_regular_pointer_bytes(previous_pointer)
            previous_digest = hashlib.sha256(previous_raw).hexdigest() if previous_raw is not None else None

            if active_digest == tx.get("after_active_pointer_sha256"):
                if previous_digest != tx.get("after_previous_pointer_sha256"):
                    raise RuntimeError("active pointer is committed but previous pointer does not match transaction; refusing recovery")
                active_doc = json.loads(pointer.read_text(encoding="utf-8"))
                if not isinstance(active_doc, dict) or active_doc.get("active_bundle") != tx.get("target_bundle"):
                    raise RuntimeError("active pointer content does not match pending deployment transaction")
                target = _relative_bundle(root, tx["target_bundle"])
                verified = verify_and_load_bundle(target, public_key_path)
                try:
                    if sha256_file(target / "model_manifest.json") != tx.get("target_manifest_sha256"):
                        raise RuntimeError("deployment target manifest changed during interrupted transaction")
                    prior_active = _relative_bundle(root, tx["before_active_bundle"])
                    prior_verified = verify_and_load_bundle(prior_active, public_key_path)
                    try:
                        if sha256_file(prior_active / "model_manifest.json") != tx.get("before_manifest_sha256"):
                            raise RuntimeError("prior active model changed during interrupted transaction")
                    finally:
                        prior_verified.close()
                    if tx.get("operation") == "deploy":
                        att = _validate_attestation(verified)
                        if (
                            att.get("active_pointer_sha256") != tx.get("before_active_pointer_sha256")
                            or att.get("active_manifest_sha256") != tx.get("before_manifest_sha256")
                            or att.get("report_id") != tx.get("report_id")
                            or att.get("report_sha256") != tx.get("report_sha256")
                        ):
                            raise PermissionError("recovered deployment attestation does not match journal prepare record")
                    elif tx.get("operation") == "rollback":
                        _validate_pending_rollback_provenance(records, prepare, tx)
                    else:
                        raise RuntimeError("unknown pending deployment operation")
                finally:
                    verified.close()
                _commit_deployment_transaction(root, prepare["transaction_id"], tx, prepare, recovered=True)
                outcomes.append({"transaction_id": prepare["transaction_id"], "status": "committed"})
                return outcomes

            if active_digest == tx.get("before_active_pointer_sha256"):
                before_exists = tx.get("before_previous_pointer_exists")
                before_b64 = tx.get("before_previous_pointer_b64")
                before_digest = tx.get("before_previous_pointer_sha256")
                if before_exists is True:
                    if not isinstance(before_b64, str):
                        raise RuntimeError("journal lacks the previous-pointer recovery snapshot")
                    import base64
                    try:
                        before_raw = base64.b64decode(before_b64, validate=True)
                    except Exception as exc:
                        raise RuntimeError("journal previous-pointer snapshot is malformed") from exc
                    if hashlib.sha256(before_raw).hexdigest() != before_digest:
                        raise RuntimeError("journal previous-pointer snapshot digest mismatch")
                elif before_exists is False:
                    before_raw = None
                else:
                    raise RuntimeError("journal previous-pointer existence flag is invalid")

                if previous_digest == tx.get("after_previous_pointer_sha256"):
                    if before_raw is None:
                        _delete_file_durable(previous_pointer)
                    else:
                        _write_bytes_atomic(previous_pointer, before_raw)
                elif previous_digest != before_digest:
                    raise RuntimeError("previous pointer is in an ambiguous state; refusing automatic recovery")

                abort_payload = {
                    "operation": tx["operation"],
                    "pointer_name": pointer.name,
                    "prepare_record_sha256": prepare["record_sha256"],
                    "recovered": True,
                    "reason": "active pointer remained unchanged; previous pointer restored",
                }
                append_event(root, "abort", prepare["transaction_id"], abort_payload)
                outcomes.append({"transaction_id": prepare["transaction_id"], "status": "aborted"})
                return outcomes

            raise RuntimeError("active pointer matches neither side of the pending transaction; refusing automatic recovery")
    return outcomes


def promote_candidate(candidate_dir, public_key_path, *, model_root="models",
                      pointer_path="models/active_model.json", previous_pointer_path=None,
                      expected_active_pointer_sha256: str):
    _require_deployment_opt_in()
    root = Path(model_root).resolve()
    with candidate_authority_lock(root):
        requested = Path(candidate_dir)
        if _is_symlink_or_reparse(requested):
            raise ValueError("candidate bundle root must not be a symlink or reparse point")
        candidate = requested.resolve(strict=True)
        candidates = (root / "candidates").resolve()
        if candidate != candidates and candidates not in candidate.parents:
            raise ValueError("only candidates under models/candidates may be deployed")
        pointer = Path(pointer_path)
        if not pointer.is_absolute():
            pointer = pointer.resolve()
        if pointer.resolve().parent != root:
            raise ValueError("active pointer must be a direct child of model_root")
        previous_pointer = Path(previous_pointer_path) if previous_pointer_path is not None else root / "previous_model.json"
        if not previous_pointer.is_absolute():
            previous_pointer = previous_pointer.resolve()
        if previous_pointer.resolve().parent != root or previous_pointer == pointer:
            raise ValueError("previous pointer must be a distinct direct child of model_root")
        if not isinstance(expected_active_pointer_sha256, str) or len(expected_active_pointer_sha256) != 64:
            raise ValueError("expected active pointer SHA-256 is required")

        # Reconcile any interrupted prior operation before beginning another one.
        recover_incomplete_deployments(
            root, public_key_path, pointer_path=pointer, previous_pointer_path=previous_pointer
        )

        preflight_verified = verify_and_load_bundle(candidate, public_key_path)
        try:
            preflight_attestation = _validate_attestation(preflight_verified)
            if preflight_attestation["active_pointer_sha256"] != expected_active_pointer_sha256:
                raise RuntimeError("candidate attestation is bound to a different active pointer")
        finally:
            preflight_verified.close()

        with _pointer_lock(pointer):
            current_digest = _pointer_digest(pointer)
            if current_digest != expected_active_pointer_sha256:
                raise RuntimeError("active model changed since candidate evaluation; compare-and-swap refused")
            current_raw = pointer.read_bytes()
            verified = verify_and_load_bundle(candidate, public_key_path)
            try:
                att = _validate_attestation(verified)
                if att["active_pointer_sha256"] != expected_active_pointer_sha256:
                    raise RuntimeError("candidate attestation is bound to a different active pointer")
                current = json.loads(current_raw.decode("utf-8"))
                if not isinstance(current, dict) or current.get("schema") != POINTER_SCHEMA:
                    raise ValueError("refusing to replace malformed active-model pointer")
                active = _relative_bundle(root, current["active_bundle"])
                active_verified = verify_and_load_bundle(active, public_key_path)
                try:
                    if sha256_file(active / "model_manifest.json") != att["active_manifest_sha256"]:
                        raise RuntimeError("active bundle differs from the model used during evaluation")
                finally:
                    active_verified.close()
                model_version = str(verified.manifest.get("model_version", candidate.name))
                payload = {
                    "schema": POINTER_SCHEMA,
                    "active_bundle": candidate.relative_to(root).as_posix(),
                    "previous_bundle": current.get("active_bundle"),
                    "model_version": model_version,
                    "promoted_at_unix_ns": time.time_ns(),
                    "deployment_attestation": {
                        "report_id": att["report_id"],
                        "report_sha256": att["report_sha256"],
                    },
                }
            finally:
                verified.close()

            txid, tx_payload, prepare_record = _prepare_deployment_transaction(
                root, pointer, previous_pointer, current_raw, current, payload,
                candidate, active, "deploy", att,
            )
            _deployment_failpoint("after_prepare", txid)
            try:
                _write_bytes_atomic(previous_pointer, current_raw)
                _deployment_failpoint("after_previous_pointer", txid)
                _write_pointer_atomic(pointer, payload)
                _deployment_failpoint("after_active_pointer", txid)
                _commit_deployment_transaction(root, txid, tx_payload, prepare_record)
                _deployment_failpoint("after_commit", txid)
            except Exception as exc:
                raise RuntimeError(
                    f"deployment transaction {txid} is unresolved; run recover_incomplete_deployments before retrying"
                ) from exc
        return payload


def rollback_model(*, model_root="models", pointer_path="models/active_model.json",
                   public_key_path, previous_pointer_path=None,
                   expected_active_pointer_sha256: str):
    _require_deployment_opt_in()
    root = Path(model_root).resolve()
    with candidate_authority_lock(root):
        pointer = Path(pointer_path)
        if not pointer.is_absolute():
            pointer = pointer.resolve()
        previous_pointer = Path(previous_pointer_path) if previous_pointer_path is not None else root / "previous_model.json"
        if not previous_pointer.is_absolute():
            previous_pointer = previous_pointer.resolve()
        if pointer.resolve().parent != root:
            raise ValueError("active pointer must be a direct child of model_root")
        if previous_pointer.resolve().parent != root or previous_pointer == pointer:
            raise ValueError("previous pointer must be a distinct direct child of model_root")
        if _is_symlink_or_reparse(pointer) or _is_symlink_or_reparse(previous_pointer):
            raise ValueError("model pointers must not be symlinks or reparse points")

        recover_incomplete_deployments(
            root, public_key_path, pointer_path=pointer, previous_pointer_path=previous_pointer
        )

        with _pointer_lock(pointer):
            current_digest = _pointer_digest(pointer)
            if current_digest != expected_active_pointer_sha256:
                raise RuntimeError("active model changed before rollback; compare-and-swap refused")
            current_raw = pointer.read_bytes()
            current = json.loads(current_raw.decode("utf-8"))
            if not isinstance(current, dict) or current.get("schema") != POINTER_SCHEMA or not current.get("previous_bundle"):
                raise RuntimeError("no previous approved model is available for rollback")
            previous_raw = _read_regular_pointer_bytes(previous_pointer)
            if previous_raw is None:
                raise RuntimeError("previous_model.json is required for rollback")
            previous_doc = json.loads(previous_raw.decode("utf-8"))
            if (
                not isinstance(previous_doc, dict)
                or previous_doc.get("schema") != POINTER_SCHEMA
                or previous_doc.get("active_bundle") != current.get("previous_bundle")
            ):
                raise ValueError("previous_model.json does not match the active pointer's rollback target")

            from models.deployment_journal import read_journal
            records = read_journal(root, repair_torn_tail=True)
            _validate_rollback_provenance(
                records, pointer.name, previous_pointer.name, current, previous_doc, current_digest,
                hashlib.sha256(previous_raw).hexdigest(),
            )

            previous = _relative_bundle(root, current["previous_bundle"])
            verified = verify_and_load_bundle(previous, public_key_path)
            try:
                model_version = str(verified.manifest.get("model_version", previous.name))
            finally:
                verified.close()
            payload = {
                "schema": POINTER_SCHEMA,
                "active_bundle": previous.relative_to(root).as_posix(),
                "previous_bundle": current.get("active_bundle"),
                "model_version": model_version,
                "promoted_at_unix_ns": time.time_ns(),
                "rollback": True,
            }
            txid, tx_payload, prepare_record = _prepare_deployment_transaction(
                root, pointer, previous_pointer, current_raw, current, payload,
                previous, _relative_bundle(root, current["active_bundle"]), "rollback", None,
            )
            _deployment_failpoint("after_prepare", txid)
            try:
                _write_bytes_atomic(previous_pointer, current_raw)
                _deployment_failpoint("after_previous_pointer", txid)
                _write_pointer_atomic(pointer, payload)
                _deployment_failpoint("after_active_pointer", txid)
                _commit_deployment_transaction(root, txid, tx_payload, prepare_record)
                _deployment_failpoint("after_commit", txid)
            except Exception as exc:
                raise RuntimeError(
                    f"rollback transaction {txid} is unresolved; run recover_incomplete_deployments before retrying"
                ) from exc
        return payload
