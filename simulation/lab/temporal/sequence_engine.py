"""C3.1 source-aware temporal normalization for verified C2 evidence.

This module normalizes C2.5-indexed C2.4 generated frames only. It preserves
interval semantics: delta_t_sec is an inter-event interval, never a wall clock.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from simulation.lab.evidence.evidence_index import (
    EvidenceError,
    verify_evidence_index,
)

TEMPORAL_SEQUENCE_SCHEMA = "c3.temporal-sequence.v1"
FRAME_SCHEMA_VERSION = "c2.generated-frame.v1"


class TemporalSequenceError(ValueError):
    """Temporal evidence is invalid, inconsistent, or unsupported."""


def _finite_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalSequenceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TemporalSequenceError(f"{field} must be finite and non-negative")
    return result


def _read_snapshot(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise TemporalSequenceError(f"artifact is not a regular file: {path.name}")
        if before.st_size > max_bytes:
            raise TemporalSequenceError(f"artifact exceeds size limit: {path.name}")
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise TemporalSequenceError(f"artifact is not a regular file: {path.name}")
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise TemporalSequenceError(f"artifact changed while opening: {path.name}")
            chunks = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(fd)
            if len(raw) > max_bytes:
                raise TemporalSequenceError(f"artifact exceeds size limit: {path.name}")
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise TemporalSequenceError(f"artifact changed during read: {path.name}")
            return raw
        finally:
            os.close(fd)
    except TemporalSequenceError:
        raise
    except OSError as exc:
        raise TemporalSequenceError(f"unable to read {path.name}: {exc}") from exc


def _decode_frame(raw: bytes, *, expected_step: int, filename: str) -> dict[str, Any]:
    try:
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                            parse_constant=lambda token: (_ for _ in ()).throw(
                                TemporalSequenceError(f"invalid JSON constant: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalSequenceError(f"invalid frame JSON: {filename}") from exc
    if not isinstance(record, dict):
        raise TemporalSequenceError(f"frame must be a JSON object: {filename}")
    required = {
        "schema_version", "step", "device_id", "topic", "operation",
        "delta_t_sec", "payload_base64", "ground_truth_label", "scenario_id", "dropped",
    }
    if set(record) != required:
        raise TemporalSequenceError(f"frame fields do not match the C2 frame contract: {filename}")
    if record["schema_version"] != FRAME_SCHEMA_VERSION:
        raise TemporalSequenceError(f"unsupported frame schema: {filename}")
    if isinstance(record["step"], bool) or not isinstance(record["step"], int):
        raise TemporalSequenceError(f"frame step must be an integer: {filename}")
    if record["step"] != expected_step:
        raise TemporalSequenceError(f"non-contiguous frame step: {filename}")
    for field in ("device_id", "topic", "operation", "scenario_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise TemporalSequenceError(f"{field} must be a non-empty string: {filename}")
    if record["ground_truth_label"] != "attack":
        raise TemporalSequenceError(f"unexpected ground-truth label: {filename}")
    if not isinstance(record["dropped"], bool):
        raise TemporalSequenceError(f"dropped must be boolean: {filename}")
    if not isinstance(record["payload_base64"], str):
        raise TemporalSequenceError(f"payload_base64 must be a string: {filename}")
    try:
        payload = base64.b64decode(record["payload_base64"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise TemporalSequenceError(f"invalid Base64 payload: {filename}") from exc
    record["_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    record["_payload_len"] = len(payload)
    record["_delta"] = _finite_interval(record["delta_t_sec"], "delta_t_sec")
    return record


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise TemporalSequenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def normalize_verified_generated_run(
    run_directory: Path,
    *,
    max_artifact_bytes: int = 1_048_576,
) -> dict[str, Any]:
    """Verify a C2.5 run and normalize its frames into a relative timeline.

    Each frame is read once by this adapter and its bytes are compared against
    the SHA-256 and size recorded by the verified C2.5 index. No wall-clock
    timestamps are created. The first frame's delta is retained as supplied;
    cumulative_elapsed_sec is the sum of intervals through that frame.
    """
    if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or max_artifact_bytes <= 0:
        raise ValueError("max_artifact_bytes must be a positive integer")
    root = Path(run_directory).expanduser().absolute()
    try:
        verified = verify_evidence_index(root, max_artifact_bytes=max_artifact_bytes)
    except (EvidenceError, OSError, ValueError) as exc:
        raise TemporalSequenceError(f"C2.5 evidence verification failed: {exc}") from exc

    payload = verified.payload
    if payload.get("schema_version") != "c2.evidence-index.v1":
        raise TemporalSequenceError("unsupported evidence index schema")
    source = payload.get("source_scenario")
    variation = payload.get("variation")
    if not isinstance(source, Mapping) or not isinstance(variation, Mapping):
        raise TemporalSequenceError("evidence index is missing scenario or variation provenance")
    if source.get("category") != "attack" or source.get("corpus_class") != "adversarial":
        raise TemporalSequenceError("C3.1 generated temporal adapter accepts adversarial attack runs only")
    if source.get("split") != "dev":
        raise TemporalSequenceError("C3.1 generated temporal adapter accepts development runs only")
    runner = payload.get("runner")
    if not isinstance(runner, Mapping) or runner.get("execution_mode") != "offline_frame_generation_only":
        raise TemporalSequenceError("unsupported evidence execution mode")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise TemporalSequenceError("evidence index artifact list is malformed")
    frame_artifacts = [item for item in artifacts if isinstance(item, Mapping) and item.get("kind") == "frame"]
    frame_artifacts.sort(key=lambda item: item.get("path", ""))
    if len(frame_artifacts) != payload.get("frame_count") or not frame_artifacts:
        raise TemporalSequenceError("frame artifact count is inconsistent")

    events = []
    cumulative = 0.0
    frame_hasher = hashlib.sha256()
    for step, artifact in enumerate(frame_artifacts):
        relative = artifact.get("path")
        if not isinstance(relative, str) or relative != f"frames/frame_{step:06d}.json":
            raise TemporalSequenceError("frame artifact paths are not canonical and contiguous")
        expected_digest = artifact.get("sha256")
        expected_size = artifact.get("size_bytes")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise TemporalSequenceError("frame artifact digest is malformed")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise TemporalSequenceError("frame artifact size is malformed")
        frame_path = root / relative
        raw = _read_snapshot(frame_path, max_bytes=max_artifact_bytes)
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_digest:
            raise TemporalSequenceError(f"frame differs from verified C2.5 snapshot: {relative}")
        frame_hasher.update(len(raw).to_bytes(8, "big"))
        frame_hasher.update(raw)
        record = _decode_frame(raw, expected_step=step, filename=frame_path.name)
        if record["scenario_id"] != variation.get("variation_id"):
            raise TemporalSequenceError("frame variation identity differs from evidence index")
        delta = record["_delta"]
        cumulative += delta
        events.append({
            "sequence_index": step,
            "source_step": record["step"],
            "relative_interval_sec": delta,
            "cumulative_elapsed_sec": round(cumulative, 9),
            "time_basis": "accumulated_delta_t_sec",
            "absolute_timestamp_available": False,
            "device_id": record["device_id"],
            "topic": record["topic"],
            "operation": record["operation"],
            "dropped": record["dropped"],
            "ground_truth_label": record["ground_truth_label"],
            "payload_len": record["_payload_len"],
            "payload_sha256": record["_payload_sha256"],
        })
    if frame_hasher.hexdigest() != payload.get("frames_sha256"):
        raise TemporalSequenceError("normalized frame digest differs from evidence index")

    result = {
        "schema_version": TEMPORAL_SEQUENCE_SCHEMA,
        "source": {
            "source_type": "c2_verified_generated_evidence",
            "run_id": payload.get("run_id"),
            "scenario_id": source.get("scenario_id"),
            "scenario_definition_digest": source.get("definition_digest"),
            "family": source.get("family"),
            "category": source.get("category"),
            "corpus_class": source.get("corpus_class"),
            "split": source.get("split"),
            "ground_truth": source.get("ground_truth"),
            "variation_id": variation.get("variation_id"),
            "variation_digest": variation.get("variation_digest"),
            "evidence_index_sha256": verified.evidence_index_sha256,
            "frames_sha256": payload.get("frames_sha256"),
            "execution_mode": runner.get("execution_mode"),
            "external_execution_performed": False,
        },
        "time_semantics": {
            "clock_type": "relative_interval",
            "interval_field": "delta_t_sec",
            "cumulative_origin": "zero_at_start_of_recorded_sequence",
            "absolute_timestamps_synthesized": False,
            "first_frame_interval_included": True,
        },
        "event_count": len(events),
        "events": events,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    result["sequence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result
