"""V6 Stage B: offline adversarial shadow evaluation and model promotion gate.

This module contains no network clients or remote dependencies. The online IDS
is not an authority: it only verifies the Ed25519-signed active bundle. This
module is intended to run under a separately provisioned local offline-authority
account with access to the model-signing private key.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np

from models.bundle_loader import (
    _is_symlink_or_reparse,
    hash_unsigned_candidate_artifacts,
    load_unsigned_candidate_for_evaluation,
    sha256_file,
    sign_model_bundle,
    validate_unsigned_candidate_bundle,
    verify_and_load_bundle,
)
from models.model_lifecycle import promote_candidate, resolve_active_bundle, rollback_model
from storage.telemetry_db import LumenStorageManager


@dataclass(frozen=True)
class EvaluationGroup:
    scenario_id: str
    category: str  # clean | attack | protocol | fault
    device_id: str
    topic: str
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    candidate_dir: str
    gates: dict[str, bool]
    metrics: dict[str, Any]
    reason: str
    active_pointer: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateEvaluationResult:
    # Quality-gate result only; this object grants no signing/deployment authority.
    evaluated: bool
    candidate_dir: str
    active_bundle: str
    gates: dict[str, bool]
    metrics: dict[str, Any]
    all_gates_passed: bool
    reason: str
    report_id: str
    report_path: str
    report_sha256: str
    candidate_manifest_sha256: str
    active_manifest_sha256: str
    audit_status: str = "written"
    audit_warning: str | None = None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"evaluation metric {field} must be a finite number")
    return float(value)


def _event_group_key(row: dict[str, Any], default_category: str) -> tuple[str, str, str]:
    device = str(row.get("device_id") or "unknown")
    topic = str(row.get("topic") or row.get("original_topic") or "unknown")
    scenario = str(row.get("scenario_id") or row.get("reason_code") or row.get("capture_id") or "unknown")
    return scenario, device, topic


def _load_clean_groups(
    storage: LumenStorageManager,
    session_ids: Iterable[str],
    expected_curation_digests: dict[str, str] | None = None,
) -> list[EvaluationGroup]:
    from training.session_integrity import load_verified_clean_session_rows

    verified = load_verified_clean_session_rows(
        storage, session_ids, expected_curation_digests
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for session_id, rows in verified.items():
        for item in rows:
            key = (session_id, str(item["device_id"]), str(item["topic"]))
            grouped.setdefault(key, []).append(item)
    for events in grouped.values():
        events.sort(key=lambda item: (int(item["gateway_ts_ns"]), int(item["event_id"])))
    return [
        EvaluationGroup(
            f"clean:{session_id}:{device}:{topic}", "clean", device, topic, tuple(events)
        )
        for (session_id, device, topic), events in sorted(grouped.items())
    ]

def _load_incident_groups(storage: LumenStorageManager) -> tuple[list[EvaluationGroup], list[EvaluationGroup], list[EvaluationGroup]]:
    captures = storage.load_adversarial_regression_corpus(
        ("adversarial", "protocol_violation", "hardware_fault")
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in captures:
        category = {
            "adversarial": "attack",
            "protocol_violation": "protocol",
            "hardware_fault": "fault",
        }[str(row["corpus_class"])]
        scenario, device, topic = _event_group_key(row, category)
        item = dict(row)
        try:
            lab_metadata = json.loads(str(item.get("xai_contrib_json") or "{}"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("regression capture metadata is invalid JSON") from exc
        operation = lab_metadata.get("operation", "put") if isinstance(lab_metadata, dict) else "put"
        if operation not in {"put", "query", "declare_subscriber"}:
            raise RuntimeError("regression capture contains an unsupported operation label")
        item["operation"] = operation
        grouped.setdefault((category, scenario, device, topic), []).append(item)

    attacks: list[EvaluationGroup] = []
    protocols: list[EvaluationGroup] = []
    faults: list[EvaluationGroup] = []
    for (category, scenario, device, topic), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: (int(item.get("gateway_ts_ns") or 0), int(item.get("capture_id") or 0)))
        group = EvaluationGroup(f"{category}:{scenario}:{device}:{topic}", category, device, topic, tuple(rows))
        if category == "attack":
            attacks.append(group)
        elif category == "protocol":
            protocols.append(group)
        else:
            faults.append(group)
    return attacks, protocols, faults


def _load_fault_scenarios() -> list[EvaluationGroup]:
    """Generate deterministic fault-only groups so absence of lab captures is visible."""
    from attacks.advanced_generators import iter_scenario_frames

    groups: list[EvaluationGroup] = []
    for scenario in ("fault1_frozen_sensor", "fault2_packet_drop_40"):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for frame in iter_scenario_frames(scenario, steps=90):
            if frame.dropped:
                continue
            grouped.setdefault((frame.device_id, frame.topic), []).append({
                "device_id": frame.device_id,
                "topic": frame.topic,
                "gateway_ts_ns": int(frame.step * frame.delta_t_sec * 1_000_000_000),
                "raw_payload": frame.payload.decode("utf-8", errors="strict"),
                "numeric_value": None,
                "operation": frame.operation,
            })
        for (device, topic), events in sorted(grouped.items()):
            groups.append(EvaluationGroup(f"generated:{scenario}:{device}:{topic}", "fault", device, topic, tuple(events)))
    return groups


class _ShadowPolicyStorage:
    """In-memory policy audit double; no writes escape the offline evaluation."""

    def __init__(self):
        self.transitions: list[tuple[Any, ...]] = []
        self.enforcement: list[tuple[Any, ...]] = []

    def load_policy_dimensions(self):
        return {}

    def load_policy_states(self):
        return {}

    def load_device_state_rows(self):
        return {}

    def append_quarantine_transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))

    def append_policy_dimension_transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))

    def record_enforcement_decision(self, *args, **kwargs):
        self.enforcement.append((args, kwargs))

    def confirm_enforcement_pep(self, *args, **kwargs):
        return None

    def save_policy_dimensions(self, **_kwargs):
        return None


def _score_bundle(
    bundle_dir: Path,
    public_key_path: Path,
    groups: list[EvaluationGroup],
    *,
    unsigned_candidate: bool = False,
) -> dict[str, Any]:
    """Run local CPU ONNX inference through the real orthogonal policy engine.

    Protocol-labelled captures are counted as ingress/protocol detections and
    never passed to model features. Other events share the production V5/V6
    detector and policy semantics: ONNX + CUSUM + OperationalFaultClassifier +
    RiskPolicyEngine. PEP succeeds in this shadow environment only; no Gateway
    commands or external side effects are issued.
    """
    from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
    from agent.statistical_detectors import OperationalFaultClassifier, TwoSidedCusumDetector
    from utils.features_v5 import DeviceTopicFeatureExtractor, RobustFeatureScaler

    # Candidate remains unsigned until promotion gates pass. Its local hashes
    # are checked here; the active runtime bundle always requires Ed25519.
    verified = (
        load_unsigned_candidate_for_evaluation(bundle_dir)
        if unsigned_candidate
        else verify_and_load_bundle(bundle_dir, public_key_path)
    )
    needs_onnx = any(group.category != "protocol" for group in groups)
    session = None
    if needs_onnx:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for offline shadow evaluation; no network fallback is permitted") from exc
        session = ort.InferenceSession(str(verified.onnx_path), providers=["CPUExecutionProvider"])
        scaler = RobustFeatureScaler.from_dict(verified.scaler)
        seq_len = int(verified.feature_schema["seq_len"])
        scales = np.maximum(np.asarray(verified.thresholds["xai_error_scales"], dtype=np.float32), 1e-3)
        global_threshold = float(verified.thresholds["global_threshold"])
        stream_thresholds = verified.thresholds.get("stream_thresholds", {})
    else:
        scaler = None
        seq_len = 0
        scales = np.asarray([], dtype=np.float32)
        global_threshold = 0.0
        stream_thresholds = {}

    detections: dict[str, dict[str, Any]] = {}
    normal_scores: list[float] = []
    false_faults = 0
    total_faults = sum(group.category == "fault" for group in groups)
    attack_total = sum(group.category == "attack" for group in groups)
    protocol_total = sum(group.category == "protocol" for group in groups)
    attack_hits = 0
    protocol_hits = 0
    attack_latency: dict[str, int | None] = {}
    protocol_latency: dict[str, int | None] = {}
    clean_event_count = 0
    clean_security_alert_count = 0
    clean_operational_fault_count = 0

    for group in groups:
        # Protocol/freshness captures are explicit, already-verified ingress
        # labels. They are immediate protocol detections, not ML detections.
        if group.category == "protocol":
            detections[group.scenario_id] = {"detected": True, "latency": 1, "source": "protocol"}
            protocol_hits += 1
            protocol_latency[group.scenario_id] = 1
            continue

        extractor = DeviceTopicFeatureExtractor(seq_len=seq_len, stat_window=seq_len)
        fault_classifier = OperationalFaultClassifier(frozen_min_steps=8)
        cusum = TwoSidedCusumDetector(k_allowance=0.5, h_threshold=6.0)
        for stream_key, baseline in verified.thresholds.get("cusum_baselines", {}).items():
            device, topic = stream_key.split("::", 1)
            cusum.calibrate_stream(device, topic, baseline["mean"], baseline["std"])
        policy = RiskPolicyEngine(
            storage=_ShadowPolicyStorage(),
            pep_apply_fn=lambda _device, _enable: True,
            window_n=5,
            trigger_k=3,
        )

        previous_ts: int | None = None
        detected_at: int | None = None
        protocol_security_detected = False
        last_operational_status = "HEALTHY"
        for index, event in enumerate(group.events, start=1):
            raw = event.get("raw_payload", b"")
            raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
            ts_ns = int(event.get("gateway_ts_ns") or 0)
            if previous_ts is None:
                dt = 1.0
            else:
                elapsed = (ts_ns - previous_ts) / 1_000_000_000.0
                dt = elapsed if elapsed > 0 else 1.0
            previous_ts = ts_ns

            value = event.get("numeric_value")
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    # Malformed/NaN forensic records remain in the corpus but
                    # are not silently converted into normal numeric model data.
                    continue
            step = extractor.update(
                group.device_id, group.topic, dt, raw_bytes,
                operation=str(event.get("operation") or "put"), numeric_value=value,
            )
            c_res = cusum.update(group.device_id, group.topic, step.raw_value)
            op_health = fault_classifier.evaluate(
                group.device_id, group.topic, step.raw_value, dt,
                operation=str(event.get("operation") or "put"),
            )
            last_operational_status = op_health.status
            if group.category == "clean":
                clean_event_count += 1
                if op_health.status == "FAULT":
                    clean_operational_fault_count += 1

            lstm_score = 0.0
            threshold = float(stream_thresholds.get(
                f"{group.device_id}::{group.topic}", global_threshold
            ))
            lstm_anomalous = False
            window = extractor.get_window(group.device_id, group.topic)
            if window is not None:
                scaled = scaler.transform(window)[None, :, :]
                reconstructed = session.run(None, {"window_input": scaled})[0][0]
                residual = np.abs(scaled[0, -1, :] - reconstructed[-1]) / scales
                lstm_score = float(np.mean(residual))
                lstm_anomalous = lstm_score > threshold
                if group.category == "clean":
                    normal_scores.append(lstm_score)

            evidence = DetectionEvidence(
                device_id=group.device_id,
                generation=int(event.get("generation") or 1),
                topic=group.topic,
                lstm_score=lstm_score,
                lstm_threshold=threshold,
                lstm_anomalous=lstm_anomalous,
                cusum_score=float(c_res.score),
                cusum_triggered=bool(c_res.triggered),
                operational_status=op_health.status,
                fault_type=op_health.fault_type,
                xai_summary=f"shadow-lstm={lstm_score:.6f};cusum={c_res.score:.4f}",
            )
            decision = policy.evaluate_evidence(evidence)
            if group.category == "clean" and decision.security_status in {"SUSPECT", "QUARANTINED", "REVOKED"}:
                clean_security_alert_count += 1
            if decision.security_status == "QUARANTINED" and detected_at is None:
                detected_at = index

        detected = detected_at is not None
        detections[group.scenario_id] = {
            "detected": detected,
            "latency": detected_at,
            "source": "policy_hybrid",
            "final_operational_status": last_operational_status,
            "final_security_status": policy.get_security_status(group.device_id),
        }
        if group.category == "attack" and detected:
            attack_hits += 1
            attack_latency[group.scenario_id] = detected_at
        if group.category == "fault" and detected:
            # A false cyber-quarantine is measured from the actual policy state,
            # not from a raw LSTM threshold crossing on a fault symptom.
            false_faults += 1

    normal_p95 = float(np.percentile(normal_scores, 95)) if normal_scores else float("inf")
    threshold_exceedance = (
        float(sum(score > global_threshold for score in normal_scores) / len(normal_scores))
        if normal_scores else 1.0
    )
    return {
        "attack_total": attack_total,
        "attack_detected": attack_hits,
        "attack_recall": (attack_hits / attack_total) if attack_total else 0.0,
        "attack_latency": attack_latency,
        "protocol_total": protocol_total,
        "protocol_detected": protocol_hits,
        "protocol_recall": (protocol_hits / protocol_total) if protocol_total else 0.0,
        "protocol_latency": protocol_latency,
        "fault_total": total_faults,
        "false_fault_quarantines": false_faults,
        "normal_error_p95": normal_p95,
        "normal_threshold_exceedance_rate": threshold_exceedance,
        "clean_event_count": clean_event_count,
        "clean_security_alert_rate": (clean_security_alert_count / clean_event_count) if clean_event_count else 0.0,
        "clean_operational_fault_rate": (clean_operational_fault_count / clean_event_count) if clean_event_count else 0.0,
        "detections": detections,
    }


def evaluate_promotion_metrics(
    storage: LumenStorageManager,
    candidate_dir: Path,
    active_dir: Path,
    public_key_path: Path,
    heldout_session_ids: Iterable[str],
    expected_curation_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    clean = _load_clean_groups(storage, heldout_session_ids, expected_curation_digests)
    attacks, protocols, captured_faults = _load_incident_groups(storage)
    faults = captured_faults if captured_faults else _load_fault_scenarios()
    if not clean:
        raise RuntimeError("promotion gate requires held-out clean sessions")
    if not attacks and not protocols:
        raise RuntimeError("promotion gate requires a non-empty adversarial/protocol regression corpus")
    if not faults:
        raise RuntimeError("promotion gate requires fault regression scenarios")
    groups = clean + attacks + protocols + faults
    candidate = _score_bundle(
        candidate_dir, public_key_path, groups, unsigned_candidate=True
    )
    active = _score_bundle(active_dir, public_key_path, groups)
    return {"candidate": candidate, "active": active, "group_count": len(groups)}


def _promotion_gates(metrics: dict[str, Any], candidate_manifest: dict[str, Any]) -> dict[str, bool]:
    candidate = metrics["candidate"]
    active = metrics["active"]
    cand_faults = int(candidate["false_fault_quarantines"])
    active_faults = int(active["false_fault_quarantines"])
    fault_no_regression = cand_faults <= active_faults

    recall_ok = _finite_number(candidate["attack_recall"], "candidate.attack_recall") >= _finite_number(
        active["attack_recall"], "active.attack_recall"
    )
    protocol_metrics_available = (
        "protocol_recall" in candidate and "protocol_recall" in active
    )
    protocol_recall_ok = (
        _finite_number(candidate["protocol_recall"], "candidate.protocol_recall")
        >= _finite_number(active["protocol_recall"], "active.protocol_recall")
        if protocol_metrics_available else True
    )
    candidate_latency = candidate.get("attack_latency", {})
    active_latency = active.get("attack_latency", {})
    latency_ok = True
    for scenario, active_step in active_latency.items():
        if active_step is None:
            continue
        candidate_step = candidate_latency.get(scenario)
        if candidate_step is None or int(candidate_step) > int(active_step):
            latency_ok = False
            break

    p95 = _finite_number(candidate["normal_error_p95"], "candidate.normal_error_p95")
    validation_p95 = _finite_number(candidate_manifest["validation_normal_error_p95"], "validation_normal_error_p95")
    exceedance = _finite_number(
        candidate["normal_threshold_exceedance_rate"], "candidate.normal_threshold_exceedance_rate"
    )
    normal_bound = max(validation_p95 * 1.25, 1e-6)
    normal_ok = p95 <= normal_bound and exceedance <= 0.05
    return {
        "zero_increase_false_quarantines_on_faults": fault_no_regression,
        "attack_recall_not_worse_than_active": recall_ok,
        "protocol_detection_not_worse_than_active": protocol_recall_ok,
        "attack_latency_not_worse_than_active": latency_ok,
        "normal_reconstruction_within_calibrated_bounds": normal_ok,
    }


def _append_audit(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    # Append-only local audit with fsync; promotion remains independent of network.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _canonical_report_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _candidate_report_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_policy": metadata["dataset_policy"],
        "split_unit": metadata["split_unit"],
        "train_sessions": list(metadata["train_sessions"]),
        "validation_sessions": list(metadata["val_sessions"]),
        "heldout_sessions": list(metadata["heldout_sessions"]),
        "train_capture_runs": list(metadata["train_capture_runs"]),
        "validation_capture_runs": list(metadata["val_capture_runs"]),
        "heldout_capture_runs": list(metadata["heldout_capture_runs"]),
        "capture_run_ids_by_session": dict(metadata["capture_run_ids_by_session"]),
        "curation_digests": dict(metadata["curation_digests"]),
    }


def _candidate_evidence_snapshot(
    candidate_dir: str | Path,
    model_root: str | Path,
) -> tuple[dict[str, Any], Any]:
    root = Path(model_root).expanduser().resolve()
    candidate = validate_unsigned_candidate_bundle(candidate_dir)
    try:
        relative = candidate.bundle_dir.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("candidate must reside under the configured model root") from exc
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or len(relative_path.parts) < 2
        or relative_path.parts[0] != "candidates"
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise ValueError("candidate must reside under models/candidates")

    artifacts = hash_unsigned_candidate_artifacts(candidate.bundle_dir)
    declared = candidate.metadata["candidate_files_sha256"]
    if artifacts != declared:
        raise ValueError("candidate artifact hashes changed during evidence snapshot")
    snapshot = {
        "relative_path": relative,
        "model_version": str(candidate.metadata.get("model_version") or "unknown"),
        "manifest_sha256": sha256_file(candidate.bundle_dir / "manifest.json"),
        "artifacts_sha256": artifacts,
    }
    return snapshot, candidate


def _active_evidence_snapshot(
    model_root: str | Path,
    pointer_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    pointer_requested = Path(pointer_path).expanduser()
    if not pointer_requested.is_file() or _is_symlink_or_reparse(pointer_requested):
        raise FileNotFoundError("active-model pointer must be a regular file inside models/")
    pointer = pointer_requested.resolve()
    try:
        pointer_relative = pointer.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("active-model pointer must remain inside the model root") from exc

    pointer_bytes = pointer.read_bytes()
    try:
        pointer_document = json.loads(pointer_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("active-model pointer JSON is invalid") from exc
    if not isinstance(pointer_document, dict) or pointer_document.get("schema") != 1:
        raise ValueError("invalid active-model pointer schema")

    active_dir = resolve_active_bundle(root, pointer, root / "lumen_model_bundle")
    verified = verify_and_load_bundle(active_dir, public_key_path)
    try:
        active_relative = active_dir.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("active bundle must remain inside the model root") from exc

    manifest_hashes = verified.manifest.get("files_sha256")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("active signed manifest has no artifact hash map")
    return {
        "pointer_relative_path": pointer_relative,
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "pointer_document": pointer_document,
        "bundle_relative_path": active_relative,
        "model_version": str(verified.manifest.get("model_version") or "unknown"),
        "manifest_sha256": sha256_file(active_dir / "model_manifest.json"),
        "signature_sha256": sha256_file(active_dir / "model_manifest.sig"),
        "artifacts_sha256": dict(sorted(manifest_hashes.items())),
    }


def _write_candidate_evaluation_report(
    model_root: Path,
    document: dict[str, Any],
) -> tuple[Path, str]:
    report_sha256 = hashlib.sha256(_canonical_report_bytes(document)).hexdigest()
    payload_document = {**document, "report_sha256": report_sha256}
    payload = _canonical_report_bytes(payload_document) + b"\n"

    report_dir = model_root / "evaluations"
    report_dir.mkdir(parents=True, exist_ok=True)
    if _is_symlink_or_reparse(report_dir):
        raise ValueError("evaluation report directory must not be a symlink or reparse point")
    report_path = report_dir / f"{document['report_id']}.json"

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(report_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            directory_fd = os.open(report_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            report_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return report_path, report_sha256


def verify_candidate_evaluation_report(
    report_path: str | Path,
    *,
    model_root: str | Path,
    public_key_path: str | Path,
    candidate_dir: str | Path | None = None,
    pointer_path: str | Path | None = None,
) -> dict[str, Any]:
    # Verify the report digest and revalidate its candidate and active snapshots.
    # This is an integrity binding, not a digital signature or operator authorization.
    root = Path(model_root).expanduser().resolve()
    report_dir_requested = root / "evaluations"
    if not report_dir_requested.is_dir() or _is_symlink_or_reparse(report_dir_requested):
        raise ValueError("evaluation report directory must be a regular directory")
    report_requested = Path(report_path).expanduser()
    if not report_requested.is_file() or _is_symlink_or_reparse(report_requested):
        raise ValueError("evaluation report must be a regular file")
    report_file = report_requested.resolve()
    report_dir = report_dir_requested.resolve()
    if report_file.parent != report_dir:
        raise ValueError("evaluation report must be a direct child of models/evaluations")

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation report JSON is invalid") from exc
    if not isinstance(report, dict) or report.get("schema") != "lumen-candidate-evaluation-report-v1":
        raise ValueError("unsupported candidate evaluation report schema")

    report_id = report.get("report_id")
    if (
        not isinstance(report_id, str)
        or len(report_id) != 32
        or any(ch not in "0123456789abcdef" for ch in report_id)
        or report_file.name != f"{report_id}.json"
    ):
        raise ValueError("evaluation report ID or filename is invalid")

    supplied_digest = report.get("report_sha256")
    if (
        not isinstance(supplied_digest, str)
        or len(supplied_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in supplied_digest)
    ):
        raise ValueError("evaluation report SHA-256 is missing or invalid")
    unsigned_report = {key: value for key, value in report.items() if key != "report_sha256"}
    actual_digest = hashlib.sha256(_canonical_report_bytes(unsigned_report)).hexdigest()
    if not hmac.compare_digest(actual_digest, supplied_digest):
        raise ValueError("evaluation report SHA-256 mismatch")

    candidate_record = report.get("candidate")
    active_record = report.get("active")
    provenance_record = report.get("data_provenance")
    evaluation_record = report.get("evaluation")
    if not all(isinstance(value, dict) for value in (
        candidate_record, active_record, provenance_record, evaluation_record
    )):
        raise ValueError("evaluation report is missing structured evidence sections")

    candidate_relative = candidate_record.get("relative_path")
    if not isinstance(candidate_relative, str):
        raise ValueError("evaluation report candidate path is invalid")
    candidate_parts = PurePosixPath(candidate_relative)
    if (
        candidate_parts.is_absolute()
        or len(candidate_parts.parts) < 2
        or candidate_parts.parts[0] != "candidates"
        or any(part in ("", ".", "..") for part in candidate_parts.parts)
    ):
        raise ValueError("evaluation report candidate path is unsafe")
    current_candidate_dir = root.joinpath(*candidate_parts.parts)
    if candidate_dir is not None and Path(candidate_dir).expanduser().resolve() != current_candidate_dir.resolve():
        raise ValueError("supplied candidate path does not match the evaluation report")

    # Preserve the lexical path so validate_unsigned_candidate_bundle can reject
    # a symlink/reparse-point candidate root before resolving it.
    current_candidate, candidate_bundle = _candidate_evidence_snapshot(
        current_candidate_dir, root
    )
    if current_candidate != candidate_record:
        raise ValueError("candidate snapshot no longer matches evaluation report")
    current_provenance = _candidate_report_provenance(candidate_bundle.metadata)
    if current_provenance != provenance_record:
        raise ValueError("candidate training provenance no longer matches evaluation report")

    pointer_relative = active_record.get("pointer_relative_path")
    if not isinstance(pointer_relative, str):
        raise ValueError("evaluation report active pointer path is invalid")
    pointer_parts = PurePosixPath(pointer_relative)
    if (
        pointer_parts.is_absolute()
        or not pointer_parts.parts
        or any(part in ("", ".", "..") for part in pointer_parts.parts)
    ):
        raise ValueError("evaluation report active pointer path is unsafe")
    stored_pointer = root.joinpath(*pointer_parts.parts)
    if pointer_path is not None and Path(pointer_path).expanduser().resolve() != stored_pointer.resolve():
        raise ValueError("supplied active pointer does not match the evaluation report")
    current_active = _active_evidence_snapshot(
        root, stored_pointer, public_key_path
    )
    if current_active != active_record:
        raise ValueError("active model or pointer snapshot no longer matches evaluation report")

    gates = evaluation_record.get("gates")
    metrics = evaluation_record.get("metrics")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(not isinstance(value, bool) for value in gates.values())
        or not isinstance(metrics, dict)
        or evaluation_record.get("all_gates_passed") is not all(gates.values())
    ):
        raise ValueError("evaluation report gate or metric structure is invalid")
    return report


def evaluate_candidate_only(
    candidate_dir: str | Path,
    *,
    storage: LumenStorageManager | None,
    public_key_path: str | Path,
    model_root: str | Path = "models",
    pointer_path: str | Path | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> CandidateEvaluationResult:
    # Validate and shadow-evaluate an unsigned candidate. No private key is accepted.
    root = Path(model_root).expanduser().resolve()
    pointer = (
        Path(pointer_path).expanduser()
        if pointer_path is not None
        else root / "active_model.json"
    )
    public_key = Path(public_key_path).expanduser().resolve()

    candidate_snapshot_before, candidate = _candidate_evidence_snapshot(candidate_dir, root)
    active_snapshot_before = _active_evidence_snapshot(root, pointer, public_key)
    active_parts = PurePosixPath(active_snapshot_before["bundle_relative_path"])
    active_dir = root.joinpath(*active_parts.parts).resolve()

    if evaluator is None:
        if storage is None:
            raise ValueError("storage is required for production shadow evaluation")
        metrics = evaluate_promotion_metrics(
            storage,
            candidate.bundle_dir,
            active_dir,
            public_key,
            candidate.metadata["heldout_sessions"],
            candidate.metadata.get("curation_digests"),
        )
    else:
        metrics = evaluator(candidate.bundle_dir, active_dir, candidate.metadata)

    gates = _promotion_gates(metrics, candidate.metadata)
    all_passed = all(gates.values())
    reason = (
        "ALL_EVALUATION_GATES_PASSED"
        if all_passed
        else "CANDIDATE_REJECTED_BY_EVALUATION_GATES"
    )

    # Re-hash both sides after inference. Any mutation invalidates this evaluation.
    candidate_snapshot_after, candidate_after = _candidate_evidence_snapshot(candidate_dir, root)
    active_snapshot_after = _active_evidence_snapshot(root, pointer, public_key)
    if candidate_snapshot_after != candidate_snapshot_before:
        raise RuntimeError("candidate changed during shadow evaluation")
    if active_snapshot_after != active_snapshot_before:
        raise RuntimeError("active model or pointer changed during shadow evaluation")
    if _candidate_report_provenance(candidate_after.metadata) != _candidate_report_provenance(candidate.metadata):
        raise RuntimeError("candidate training provenance changed during shadow evaluation")

    report_id = uuid.uuid4().hex
    report_document = {
        "schema": "lumen-candidate-evaluation-report-v1",
        "report_id": report_id,
        "evaluated_at_unix_ns": time.time_ns(),
        "candidate": candidate_snapshot_before,
        "active": active_snapshot_before,
        "data_provenance": _candidate_report_provenance(candidate.metadata),
        "evaluation": {
            "gates": gates,
            "metrics": metrics,
            "all_gates_passed": all_passed,
            "reason": reason,
        },
    }
    report_path, report_sha256 = _write_candidate_evaluation_report(root, report_document)

    audit_warning = None
    try:
        _append_audit(root / "candidate_evaluations.jsonl", {
            "at_ns": report_document["evaluated_at_unix_ns"],
            "event": "candidate_evaluated",
            "report_id": report_id,
            "report_path": report_path.relative_to(root).as_posix(),
            "report_sha256": report_sha256,
            "candidate": candidate.metadata.get("model_version"),
            "active_bundle": active_snapshot_before["bundle_relative_path"],
            "evaluated": True,
            "all_gates_passed": all_passed,
            "gates": gates,
            "reason": reason,
        })
    except Exception as exc:
        # The report is already durable; expose audit failure separately.
        audit_warning = f"{type(exc).__name__}: {exc}"

    return CandidateEvaluationResult(
        evaluated=True,
        candidate_dir=str(candidate.bundle_dir),
        active_bundle=str(active_dir),
        gates=gates,
        metrics=metrics,
        all_gates_passed=all_passed,
        reason=reason,
        report_id=report_id,
        report_path=str(report_path),
        report_sha256=report_sha256,
        candidate_manifest_sha256=candidate_snapshot_before["manifest_sha256"],
        active_manifest_sha256=active_snapshot_before["manifest_sha256"],
        audit_status="failed" if audit_warning else "written",
        audit_warning=audit_warning,
    )



def evaluate_and_promote_candidate(*args, **kwargs):
    raise PermissionError("combined evaluation/signing/deployment path is disabled; use evaluate_candidate_only(), sign_evaluated_candidate(), then deploy_signed_candidate()")



def rollback_on_rate_spike(*args, **kwargs):
    raise PermissionError("legacy rollback path is disabled; use training.deployment_authority")



def read_post_promotion_rates(
    storage: LumenStorageManager, *, window_sec: float = 300.0, min_events: int = 20,
) -> dict[str, float] | None:
    """Calculate local policy/fault event rates over a recent admitted-telemetry window."""
    if window_sec <= 0 or min_events < 1:
        raise ValueError("window_sec and min_events must be positive")
    since_ns = time.time_ns() - int(window_sec * 1_000_000_000)
    normal = storage.connect_readonly("normal")
    try:
        total_events = int(normal.execute(
            "SELECT COUNT(*) FROM raw_events WHERE gateway_ts_ns >= ?", (since_ns,)
        ).fetchone()[0])
    finally:
        normal.close()
    if total_events < min_events:
        return None
    control = storage.connect_readonly("control")
    try:
        security_events = int(control.execute(
            """SELECT COUNT(*) FROM policy_dimension_transitions
               WHERE ts_ns >= ? AND dimension='security'
                 AND to_state IN ('SUSPECT','QUARANTINED','REVOKED')""", (since_ns,)
        ).fetchone()[0])
        operational_faults = int(control.execute(
            """SELECT COUNT(*) FROM policy_dimension_transitions
               WHERE ts_ns >= ? AND dimension='operational' AND to_state='FAULT'""", (since_ns,)
        ).fetchone()[0])
    finally:
        control.close()
    return {
        "anomaly_rate": security_events / total_events,
        "fault_rate": operational_faults / total_events,
        "event_count": float(total_events),
    }


def monitor_post_promotion_rates(*args, **kwargs):
    raise PermissionError("legacy deployment monitor is disabled; use training.deployment_authority")
