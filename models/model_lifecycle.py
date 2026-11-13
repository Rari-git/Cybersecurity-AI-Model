"""Atomic active-model pointer with promotion history and rollback.

Candidates remain immutable under models/candidates/<candidate_id>. Runtime
processes resolve one atomically replaced JSON pointer, avoiding in-place writes
into the currently loaded bundle.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from models.bundle_loader import verify_and_load_bundle

POINTER_SCHEMA = 1


def _relative_bundle(root: Path, value: str) -> Path:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError("active bundle path must be a safe relative path")
    resolved = (root / Path(*p.parts)).resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise ValueError("active bundle must remain inside model root")
    if not resolved.is_dir():
        raise ValueError("active bundle path must point to a directory")
    return resolved


def _write_pointer_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        # Best-effort directory fsync on POSIX.
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


def resolve_active_bundle(
    model_root: str | Path = "models",
    pointer_path: str | Path = "models/active_model.json",
    fallback_bundle: str | Path = "models/lumen_model_bundle",
) -> Path:
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


def promote_candidate(
    candidate_dir: str | Path,
    public_key_path: str | Path,
    *,
    model_root: str | Path = "models",
    pointer_path: str | Path = "models/active_model.json",
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    candidate = Path(candidate_dir).resolve(strict=True)
    candidates_root = (root / "candidates").resolve()
    if candidate != candidates_root and candidates_root not in candidate.parents:
        raise ValueError("only verified candidates under models/candidates may be promoted")
    verified = verify_and_load_bundle(candidate, public_key_path)
    relative = candidate.relative_to(root).as_posix()
    pointer = Path(pointer_path)
    if not pointer.is_absolute():
        pointer = pointer.resolve()
    previous = None
    if pointer.is_file():
        current = json.loads(pointer.read_text(encoding="utf-8"))
        if current.get("schema") != POINTER_SCHEMA:
            raise ValueError("refusing to replace malformed active-model pointer")
        previous = current.get("active_bundle")
    payload = {
        "schema": POINTER_SCHEMA,
        "active_bundle": relative,
        "previous_bundle": previous,
        "model_version": str(verified.manifest.get("model_version", candidate.name)),
        "promoted_at_unix_ns": __import__("time").time_ns(),
    }
    _write_pointer_atomic(pointer, payload)
    return payload


def rollback_model(
    *,
    model_root: str | Path = "models",
    pointer_path: str | Path = "models/active_model.json",
    public_key_path: str | Path,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    pointer = Path(pointer_path)
    if not pointer.is_absolute():
        pointer = pointer.resolve()
    current = json.loads(pointer.read_text(encoding="utf-8"))
    if current.get("schema") != POINTER_SCHEMA or not current.get("previous_bundle"):
        raise RuntimeError("no previous approved model is available for rollback")
    previous = _relative_bundle(root, current["previous_bundle"])
    verified = verify_and_load_bundle(previous, public_key_path)
    payload = {
        "schema": POINTER_SCHEMA,
        "active_bundle": previous.relative_to(root).as_posix(),
        "previous_bundle": current.get("active_bundle"),
        "model_version": str(verified.manifest.get("model_version", previous.name)),
        "promoted_at_unix_ns": __import__("time").time_ns(),
        "rollback": True,
    }
    _write_pointer_atomic(pointer, payload)
    return payload
