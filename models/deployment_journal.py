from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

JOURNAL_NAME = "deployment_journal.jsonl"
JOURNAL_SCHEMA = "lumen-deployment-journal-v1"
GENESIS_HASH = "0" * 64


class DeploymentJournalIntegrityError(RuntimeError):
    """The deployment journal is incomplete, malformed, or hash-chain-invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        if callable(junction) and junction():
            return True
        info = path.lstat()
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(getattr(info, "st_file_attributes", 0) & flag)
    except FileNotFoundError:
        return False


def _journal_path(root: str | os.PathLike[str]) -> Path:
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError("model root must be a directory")
    return base / JOURNAL_NAME


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentJournalIntegrityError(f"duplicate journal JSON key: {key}")
        result[key] = value
    return result


def read_journal(
    root: str | os.PathLike[str], *, repair_torn_tail: bool = False
) -> list[dict[str, Any]]:
    path = _journal_path(root)
    if _is_link_or_reparse(path):
        raise DeploymentJournalIntegrityError("deployment journal must not be a symlink or reparse point")
    if not path.exists():
        return []
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise DeploymentJournalIntegrityError("deployment journal is not a regular file")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not repair_torn_tail:
            raise DeploymentJournalIntegrityError("deployment journal has an incomplete trailing record")
        valid_end = raw.rfind(b"\n") + 1
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise DeploymentJournalIntegrityError("deployment journal changed type during repair")
            os.ftruncate(fd, valid_end)
            os.fsync(fd)
        finally:
            os.close(fd)
        raw = raw[:valid_end]

    records: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    for expected_seq, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise DeploymentJournalIntegrityError("deployment journal contains an empty record")
        try:
            record = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
        except Exception as exc:
            raise DeploymentJournalIntegrityError(
                f"deployment journal record {expected_seq} is not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise DeploymentJournalIntegrityError("deployment journal record must be an object")
        if record.get("schema") != JOURNAL_SCHEMA:
            raise DeploymentJournalIntegrityError("unsupported deployment journal schema")
        if record.get("seq") != expected_seq:
            raise DeploymentJournalIntegrityError("deployment journal sequence is discontinuous")
        if record.get("previous_record_sha256") != previous_hash:
            raise DeploymentJournalIntegrityError("deployment journal hash chain is discontinuous")
        if record.get("event") not in {"prepare", "commit", "abort"}:
            raise DeploymentJournalIntegrityError("unknown deployment journal event")
        txid = record.get("transaction_id")
        if not isinstance(txid, str) or not txid:
            raise DeploymentJournalIntegrityError("deployment journal transaction ID is invalid")
        claimed = record.get("record_sha256")
        if not isinstance(claimed, str) or len(claimed) != 64:
            raise DeploymentJournalIntegrityError("deployment journal record hash is invalid")
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        actual = hashlib.sha256(_canonical(body)).hexdigest()
        if not __import__("hmac").compare_digest(actual, claimed):
            raise DeploymentJournalIntegrityError("deployment journal record hash mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise DeploymentJournalIntegrityError("deployment journal payload must be an object")
        records.append(record)
        previous_hash = claimed
    return records


def append_event(
    root: str | os.PathLike[str],
    event: str,
    transaction_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if event not in {"prepare", "commit", "abort"}:
        raise ValueError("unsupported deployment journal event")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("transaction_id must be non-empty")
    records = read_journal(root)
    pending = pending_transactions(records)
    pending_ids = {record["transaction_id"] for record in pending}
    if event == "prepare" and transaction_id in {
        record["transaction_id"] for record in records
    }:
        raise DeploymentJournalIntegrityError("deployment transaction ID has already been used")
    if event in {"commit", "abort"} and transaction_id not in pending_ids:
        raise DeploymentJournalIntegrityError("terminal event has no unresolved prepare record")
    previous_hash = records[-1]["record_sha256"] if records else GENESIS_HASH
    body = {
        "schema": JOURNAL_SCHEMA,
        "seq": len(records) + 1,
        "transaction_id": transaction_id,
        "event": event,
        "at_unix_ns": time.time_ns(),
        "previous_record_sha256": previous_hash,
        "payload": payload,
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    record = {**body, "record_sha256": digest}
    encoded = _canonical(record) + b"\n"
    path = _journal_path(root)
    existed = path.exists()
    if _is_link_or_reparse(path):
        raise DeploymentJournalIntegrityError("deployment journal must not be a symlink or reparse point")
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DeploymentJournalIntegrityError("deployment journal is not a regular file")
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while appending deployment journal")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if not existed and os.name == "posix":
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    return record


def pending_transactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for record in records:
        txid = record["transaction_id"]
        event = record["event"]
        if event == "prepare":
            if txid in seen:
                raise DeploymentJournalIntegrityError("duplicate deployment transaction prepare")
            seen.add(txid)
            pending[txid] = record
        else:
            if txid not in pending:
                raise DeploymentJournalIntegrityError("terminal journal event has no prepare record")
            prepared = pending[txid]
            if record["payload"].get("prepare_record_sha256") != prepared["record_sha256"]:
                raise DeploymentJournalIntegrityError("terminal journal event is not bound to its prepare record")
            del pending[txid]
    return list(pending.values())
