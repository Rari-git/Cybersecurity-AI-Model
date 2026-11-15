"""C2.5 deterministic evidence indexing and independent run verification.

The index binds the C2.4 run manifest, summary, and every frame artifact.
It is an integrity mechanism, not a digital signature or authenticity claim.
All source run files are opened read-only and are never modified.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from simulation.lab.scenario_registry import ScenarioRegistry, build_default_registry
from simulation.lab.staging import ManifestError, verify_manifest

EVIDENCE_INDEX_SCHEMA = "c2.evidence-index.v1"
FRAME_SCHEMA_VERSION = "c2.generated-frame.v1"
SUMMARY_SCHEMA_VERSION = "c2.offline-run-summary.v1"
INDEX_FILENAME = "evidence_index.json"
DEFAULT_MAX_ARTIFACTS = 100_002
DEFAULT_MAX_TOTAL_BYTES = 268_435_456
DEFAULT_MAX_ARTIFACT_BYTES = 1_048_576


class EvidenceError(ValueError):
    """Evidence is missing, malformed, inconsistent, or fails integrity checks."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"non-canonical evidence value: {exc}") from exc


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise EvidenceError(f"evidence artifact is not a regular file: {path.name}")
        if info.st_size > max_bytes:
            raise EvidenceError(f"evidence artifact exceeds size limit: {path.name}")
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"unable to read evidence artifact {path.name}: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(EvidenceError(f"invalid JSON constant: {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON artifact must contain an object: {path.name}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact(path: Path, kind: str, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "kind": kind, "size_bytes": len(raw), "sha256": _sha256(raw)}


def _safe_run_directory(run_directory: Path) -> Path:
    path = Path(run_directory).expanduser().absolute()
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"run directory is unavailable: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise EvidenceError("run directory must be a real directory, not a link")
    return path


def _validate_digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise EvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _build_payload(
    run_directory: Path,
    *,
    registry: ScenarioRegistry,
    max_artifacts: int,
    max_total_bytes: int,
    max_artifact_bytes: int,
) -> dict[str, Any]:
    root = _safe_run_directory(run_directory)
    manifest_path = root / "run_manifest.json"
    summary_path = root / "run_summary.json"
    frames_dir = root / "frames"
    if not frames_dir.is_dir() or frames_dir.is_symlink():
        raise EvidenceError("frames directory is missing or is a link")
    allowed_root_entries = {"frames", "run_summary.json", "run_manifest.json", INDEX_FILENAME}
    unexpected_root_entries = [item.name for item in root.iterdir() if item.name not in allowed_root_entries]
    if unexpected_root_entries:
        raise EvidenceError(f"run directory contains unexpected artifacts: {sorted(unexpected_root_entries)}")
    for required_name in ("run_summary.json", "run_manifest.json"):
        required_path = root / required_name
        if not required_path.exists() or required_path.is_symlink():
            raise EvidenceError(f"required evidence artifact is missing or is a link: {required_name}")

    try:
        _manifest_envelope, manifest_raw = _read_json(manifest_path, max_bytes=max_artifact_bytes)
        manifest = verify_manifest(manifest_path, max_bytes=max_artifact_bytes)
    except (ManifestError, EvidenceError) as exc:
        raise EvidenceError(f"manifest verification failed: {exc}") from exc
    summary, summary_raw = _read_json(summary_path, max_bytes=max_artifact_bytes)

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise EvidenceError("unsupported run summary schema")
    if summary.get("execution_mode") != "offline_frame_generation_only":
        raise EvidenceError("unsupported execution mode")
    if summary.get("external_execution_performed") is not False:
        raise EvidenceError("run summary must explicitly deny external execution")
    if summary.get("os_containment_verified") is not False:
        raise EvidenceError("run summary must not claim OS containment verification")

    required_summary = (
        "run_id", "source_scenario_id", "source_definition_digest", "variation_id",
        "variation_digest", "variation_seed", "frame_count", "frames_sha256", "budget",
    )
    if any(key not in summary for key in required_summary):
        raise EvidenceError("run summary is missing required provenance or accounting fields")

    for summary_key, manifest_key in (
        ("run_id", "run_id"),
        ("source_scenario_id", "scenario_id"),
        ("source_definition_digest", "scenario_definition_digest"),
        ("variation_digest", "variation_digest"),
        ("variation_seed", "seed"),
    ):
        if summary[summary_key] != manifest[manifest_key]:
            raise EvidenceError(f"summary/manifest provenance mismatch: {summary_key}")
    if summary.get("engine_version") != manifest.get("runner_version"):
        raise EvidenceError("summary/manifest runner version mismatch")

    scenario_id = summary["source_scenario_id"]
    try:
        scenario = registry.get(scenario_id)
    except (KeyError, ValueError) as exc:
        raise EvidenceError(f"scenario is not present in the canonical registry: {scenario_id}") from exc
    if scenario.definition_digest != summary["source_definition_digest"]:
        raise EvidenceError("source scenario definition digest does not match the canonical registry")
    if scenario.category != "attack" or scenario.corpus_class != "adversarial":
        raise EvidenceError("C2.4 evidence index currently accepts development attack runs only")
    if scenario.split != "dev":
        raise EvidenceError("held-out or regression scenario evidence cannot be indexed as development output")

    if manifest.get("resource_policy", {}).get("execution_mode") != "offline_frame_generation_only":
        raise EvidenceError("manifest resource policy execution mode mismatch")
    if manifest.get("resource_policy", {}).get("external_execution_performed") is not False:
        raise EvidenceError("manifest must explicitly deny external execution")
    if manifest.get("resource_policy", {}).get("os_containment_verified") is not False:
        raise EvidenceError("manifest must not claim OS containment verification")

    frame_paths = sorted(frames_dir.glob("frame_*.json"), key=lambda p: p.name)
    if not frame_paths:
        raise EvidenceError("run contains no frame artifacts")
    if len(frame_paths) > max_artifacts - 2:
        raise EvidenceError("run contains too many evidence artifacts")
    expected_names = [f"frame_{i:06d}.json" for i in range(len(frame_paths))]
    if [p.name for p in frame_paths] != expected_names:
        raise EvidenceError("frame artifact names must be contiguous and zero-based")
    unexpected = [p for p in frames_dir.iterdir() if p.name not in set(expected_names)]
    if unexpected:
        raise EvidenceError("frames directory contains unexpected artifacts")

    frame_hasher = hashlib.sha256()
    frame_artifacts: list[dict[str, Any]] = []
    frame_total_bytes = 0
    variation_id = summary["variation_id"]
    for expected_step, frame_path in enumerate(frame_paths):
        record, raw = _read_json(frame_path, max_bytes=max_artifact_bytes)
        if record.get("schema_version") != FRAME_SCHEMA_VERSION:
            raise EvidenceError(f"unsupported frame schema: {frame_path.name}")
        if record.get("step") != expected_step:
            raise EvidenceError(f"non-contiguous frame step: {frame_path.name}")
        if record.get("scenario_id") != variation_id:
            raise EvidenceError(f"frame variation ID mismatch: {frame_path.name}")
        if record.get("ground_truth_label") != "attack":
            raise EvidenceError(f"unexpected frame ground-truth label: {frame_path.name}")
        for field in ("device_id", "topic", "operation"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise EvidenceError(f"invalid frame field {field}: {frame_path.name}")
        delta = record.get("delta_t_sec")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
            raise EvidenceError(f"invalid frame timing delta: {frame_path.name}")
        if not isinstance(record.get("dropped"), bool):
            raise EvidenceError(f"invalid dropped flag: {frame_path.name}")
        payload = record.get("payload_base64")
        if not isinstance(payload, str):
            raise EvidenceError(f"invalid encoded payload: {frame_path.name}")
        try:
            base64.b64decode(payload, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise EvidenceError(f"invalid Base64 payload: {frame_path.name}") from exc
        frame_hasher.update(len(raw).to_bytes(8, "big"))
        frame_hasher.update(raw)
        frame_total_bytes += len(raw)
        frame_artifacts.append(_artifact(f"frames/{frame_path.name}", "frame", raw))

    frames_digest = frame_hasher.hexdigest()
    if frames_digest != _validate_digest(summary["frames_sha256"], "frames_sha256"):
        raise EvidenceError("aggregate frame digest does not match run summary")
    if summary["frame_count"] != len(frame_paths):
        raise EvidenceError("frame count does not match persisted frame artifacts")

    budget = summary["budget"]
    if not isinstance(budget, dict) or budget.get("frames") != len(frame_paths):
        raise EvidenceError("frame resource accounting does not match persisted frames")
    if budget.get("total_bytes") != frame_total_bytes:
        raise EvidenceError("frame resource byte accounting does not match persisted frame sizes")
    if summary.get("budget_snapshot_stage") != "after_frames_before_summary_and_manifest":
        raise EvidenceError("unrecognized resource snapshot stage")
    if set(summary.get("budget_snapshot_excludes", [])) != {"run_summary.json", "run_manifest.json"}:
        raise EvidenceError("resource snapshot exclusions are missing or inconsistent")

    policy = manifest["resource_policy"]
    max_total = policy.get("max_total_bytes")
    max_payload = policy.get("max_payload_bytes")
    max_manifest = policy.get("max_manifest_bytes")
    for name, value in (("max_total_bytes", max_total), ("max_payload_bytes", max_payload), ("max_manifest_bytes", max_manifest)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvidenceError(f"manifest resource policy has invalid {name}")
    artifacts = frame_artifacts + [
        _artifact("run_summary.json", "summary", summary_raw),
        _artifact("run_manifest.json", "manifest", manifest_raw),
    ]
    if any(item["size_bytes"] > max_payload for item in artifacts):
        raise EvidenceError("an evidence artifact exceeds the configured per-payload byte budget")
    if len(manifest_raw) > max_manifest:
        raise EvidenceError("manifest exceeds the configured manifest byte budget")
    total_bytes = sum(item["size_bytes"] for item in artifacts)
    if total_bytes > max_total:
        raise EvidenceError("persisted evidence exceeds the configured total byte budget")
    if total_bytes > max_total_bytes:
        raise EvidenceError("persisted evidence exceeds caller max_total_bytes limit")

    return {
        "schema_version": EVIDENCE_INDEX_SCHEMA,
        "run_id": summary["run_id"],
        "source_scenario": {
            "scenario_id": scenario.scenario_id,
            "definition_digest": scenario.definition_digest,
            "family": scenario.family,
            "category": scenario.category,
            "corpus_class": scenario.corpus_class,
            "split": scenario.split,
            "ground_truth": scenario.ground_truth,
        },
        "variation": {
            "variation_id": variation_id,
            "variation_digest": _validate_digest(summary["variation_digest"], "variation_digest"),
            "seed": summary["variation_seed"],
        },
        "runner": {"engine_version": summary["engine_version"], "execution_mode": summary["execution_mode"]},
        "frame_count": len(frame_paths),
        "frames_sha256": frames_digest,
        "resource_accounting": {
            "raw_payload_bytes": sum(len(base64.b64decode(json.loads(p.read_text(encoding="utf-8"))["payload_base64"], validate=True)) for p in frame_paths),
            "serialized_frame_artifact_bytes": frame_total_bytes,
            "total_persisted_evidence_bytes": total_bytes,
            "max_payload_bytes": max_payload,
            "max_total_bytes": max_total,
        },
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class EvidenceIndex:
    """Verified immutable-by-convention index payload and its digest."""
    payload: Mapping[str, Any]
    evidence_index_sha256: str


def _validate_limits(*, max_artifacts: int, max_total_bytes: int, max_artifact_bytes: int) -> None:
    for name, value in (
        ("max_artifacts", max_artifacts),
        ("max_total_bytes", max_total_bytes),
        ("max_artifact_bytes", max_artifact_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def create_evidence_index(
    run_directory: Path,
    *,
    registry: ScenarioRegistry | None = None,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> EvidenceIndex:
    """Validate a C2.4 run and write evidence_index.json without overwriting."""
    selected_registry = registry if registry is not None else build_default_registry()
    if not isinstance(selected_registry, ScenarioRegistry):
        raise TypeError("registry must be a ScenarioRegistry")
    _validate_limits(max_artifacts=max_artifacts, max_total_bytes=max_total_bytes, max_artifact_bytes=max_artifact_bytes)
    root = _safe_run_directory(run_directory)
    payload = _build_payload(root, registry=selected_registry, max_artifacts=max_artifacts,
                             max_total_bytes=max_total_bytes, max_artifact_bytes=max_artifact_bytes)
    digest = _sha256(_canonical(payload))
    envelope = _canonical({"evidence_index": payload, "evidence_index_sha256": digest})
    target = root / INDEX_FILENAME
    if target.exists() or target.is_symlink():
        raise EvidenceError("evidence index already exists; refusing to overwrite")
    if len(envelope) > max_artifact_bytes:
        raise EvidenceError("evidence index exceeds configured artifact size limit")
    try:
        with target.open("xb") as stream:
            stream.write(envelope)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise EvidenceError("evidence index already exists; refusing to overwrite") from exc
    except OSError as exc:
        raise EvidenceError(f"unable to persist evidence index: {exc}") from exc
    return EvidenceIndex(payload=payload, evidence_index_sha256=digest)


def verify_evidence_index(
    run_directory: Path,
    *,
    registry: ScenarioRegistry | None = None,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> EvidenceIndex:
    """Verify index envelope and independently recompute all bound evidence."""
    selected_registry = registry if registry is not None else build_default_registry()
    if not isinstance(selected_registry, ScenarioRegistry):
        raise TypeError("registry must be a ScenarioRegistry")
    _validate_limits(max_artifacts=max_artifacts, max_total_bytes=max_total_bytes, max_artifact_bytes=max_artifact_bytes)
    root = _safe_run_directory(run_directory)
    index_path = root / INDEX_FILENAME
    envelope, _ = _read_json(index_path, max_bytes=max_artifact_bytes)
    if set(envelope) != {"evidence_index", "evidence_index_sha256"}:
        raise EvidenceError("invalid evidence index envelope")
    payload = envelope["evidence_index"]
    expected = _validate_digest(envelope["evidence_index_sha256"], "evidence_index_sha256")
    actual = _sha256(_canonical(payload))
    if actual != expected:
        raise EvidenceError("evidence index SHA-256 verification failed")
    recomputed = _build_payload(root, registry=selected_registry, max_artifacts=max_artifacts,
                                max_total_bytes=max_total_bytes, max_artifact_bytes=max_artifact_bytes)
    if _canonical(payload) != _canonical(recomputed):
        raise EvidenceError("evidence index does not match current run artifacts")
    return EvidenceIndex(payload=payload, evidence_index_sha256=expected)
