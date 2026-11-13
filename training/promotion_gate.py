"""V6 Stage B: offline adversarial shadow evaluation and model promotion gate.

This module contains no network clients or remote dependencies. The online IDS
is not an authority: it only verifies the Ed25519-signed active bundle. This
module is intended to run under a separately provisioned local offline-authority
account with access to the model-signing private key.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from models.bundle_loader import (
    load_unsigned_candidate_for_evaluation,
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


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"evaluation metric {field} must be a finite number")
    return float(value)


def _event_group_key(row: dict[str, Any], default_category: str) -> tuple[str, str, str]:
    device = str(row.get("device_id") or "unknown")
    topic = str(row.get("topic") or row.get("original_topic") or "unknown")
    scenario = str(row.get("scenario_id") or row.get("reason_code") or row.get("capture_id") or "unknown")
    return scenario, device, topic


def _load_clean_groups(storage: LumenStorageManager, session_ids: Iterable[str]) -> list[EvaluationGroup]:
    ids = tuple(dict.fromkeys(str(value) for value in session_ids))
    if not ids:
        return []
    conn = storage.connect_readonly("normal")
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""SELECT session_id, device_id, topic, gateway_ts_ns, raw_payload, numeric_value
                FROM raw_events WHERE session_id IN ({placeholders})
                ORDER BY session_id, device_id, topic, gateway_ts_ns, event_id""",
            ids,
        ).fetchall()
    finally:
        conn.close()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        key = (str(item["session_id"]), str(item["device_id"]), str(item["topic"]))
        grouped.setdefault(key, []).append(item)
    return [
        EvaluationGroup(f"clean:{sid}:{device}:{topic}", "clean", device, topic, tuple(events))
        for (sid, device, topic), events in sorted(grouped.items())
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
        grouped.setdefault((category, scenario, device, topic), []).append(dict(row))

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
    attack_hits = 0
    attack_total = sum(group.category in ("attack", "protocol") for group in groups)
    latency: dict[str, int | None] = {}
    clean_event_count = 0
    clean_security_alert_count = 0
    clean_operational_fault_count = 0

    for group in groups:
        # Protocol/freshness captures are explicit, already-verified ingress
        # labels. They are immediate protocol detections, not ML detections.
        if group.category == "protocol":
            detections[group.scenario_id] = {"detected": True, "latency": 1, "source": "protocol"}
            attack_hits += 1
            latency[group.scenario_id] = 1
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
            latency[group.scenario_id] = detected_at
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
        "attack_latency": latency,
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
) -> dict[str, Any]:
    clean = _load_clean_groups(storage, heldout_session_ids)
    attacks, protocols, captured_faults = _load_incident_groups(storage)
    faults = captured_faults + _load_fault_scenarios()
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


def evaluate_and_promote_candidate(
    candidate_dir: str | Path,
    *,
    storage: LumenStorageManager | None,
    public_key_path: str | Path,
    private_key_path: str | Path | None,
    model_root: str | Path = "models",
    pointer_path: str | Path | None = None,
    previous_pointer_path: str | Path | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> PromotionResult:
    """Evaluate candidate offline; sign and promote only after every gate passes.

    ``evaluator`` is injectable for deterministic unit tests. Production callers
    leave it unset to run real CPU ONNX shadow inference over local SQLite data.
    """
    root = Path(model_root).resolve()
    pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
    if not pointer.is_absolute():
        pointer = pointer.resolve()
    candidate = validate_unsigned_candidate_bundle(candidate_dir)
    candidate_root = (root / "candidates").resolve()
    if candidate.bundle_dir != candidate_root and candidate_root not in candidate.bundle_dir.parents:
        raise ValueError("candidate must reside under models/candidates")
    key_path: Path | None = None
    if private_key_path is not None and Path(private_key_path).is_file():
        key_path = Path(private_key_path).resolve(strict=True)
        if root == key_path or root in key_path.parents:
            raise ValueError("model signing private key must remain outside the runtime model directory")
    if not Path(public_key_path).is_file():
        raise FileNotFoundError("runtime model-verification public key is unavailable")
    if not pointer.is_file():
        raise FileNotFoundError("active_model.json is required; promotion cannot bootstrap trust from a candidate")
    active_dir = resolve_active_bundle(root, pointer, root / "lumen_model_bundle")
    if evaluator is None:
        if storage is None:
            raise ValueError("storage is required for production shadow evaluation")
        metrics = evaluate_promotion_metrics(
            storage, candidate.bundle_dir, active_dir, Path(public_key_path),
            candidate.metadata["heldout_sessions"],
        )
    else:
        metrics = evaluator(candidate.bundle_dir, active_dir, candidate.metadata)
    gates = _promotion_gates(metrics, candidate.metadata)
    if not all(gates.values()):
        result = PromotionResult(
            promoted=False, candidate_dir=str(candidate.bundle_dir), gates=gates,
            metrics=metrics, reason="CANDIDATE_REJECTED_BY_PROMOTION_GATES",
        )
        _append_audit(root / "promotion_history.jsonl", {
            "at_ns": time.time_ns(), "candidate": candidate.metadata.get("model_version"),
            "promoted": False, "gates": gates, "reason": result.reason,
        })
        return result

    # Without the offline private key, keep the candidate unsigned and do not
    # change the active pointer even when every quality gate passes.
    if key_path is None:
        result = PromotionResult(
            promoted=False,
            candidate_dir=str(candidate.bundle_dir),
            gates=gates,
            metrics=metrics,
            reason="OFFLINE_SIGNING_KEY_UNAVAILABLE",
        )
        _append_audit(root / "promotion_history.jsonl", {
            "at_ns": time.time_ns(),
            "candidate": candidate.metadata.get("model_version"),
            "promoted": False,
            "gates": gates,
            "reason": result.reason,
        })
        return result

    metadata = dict(candidate.metadata)
    metadata.update({
        "promotion_gate": "passed",
        "promotion_gates": gates,
        "promotion_metrics": metrics,
        "signed_by_authority": "local_offline_ed25519",
    })
    sign_model_bundle(candidate.bundle_dir, key_path, metadata=metadata)
    # Immediately verify the just-signed artifact set before touching active pointer.
    verify_and_load_bundle(candidate.bundle_dir, public_key_path)
    pointer_doc = promote_candidate(
        candidate.bundle_dir, public_key_path, model_root=root, pointer_path=pointer,
        previous_pointer_path=previous_pointer_path,
    )
    _append_audit(root / "promotion_history.jsonl", {
        "at_ns": time.time_ns(), "candidate": metadata.get("model_version"),
        "promoted": True, "gates": gates, "active_pointer": pointer_doc,
    })
    return PromotionResult(
        promoted=True, candidate_dir=str(candidate.bundle_dir), gates=gates,
        metrics=metrics, reason="ALL_PROMOTION_GATES_PASSED", active_pointer=pointer_doc,
    )


def rollback_on_rate_spike(
    observed: dict[str, float],
    baseline: dict[str, float],
    *,
    model_root: str | Path,
    public_key_path: str | Path,
    pointer_path: str | Path | None = None,
    anomaly_rate_margin: float = 0.05,
    fault_rate_margin: float = 0.0,
) -> dict[str, Any] | None:
    """Automatically roll back when post-promotion anomaly/fault rates spike."""
    if anomaly_rate_margin < 0 or fault_rate_margin < 0:
        raise ValueError("rollback margins must be non-negative")
    anomaly = _finite_number(observed.get("anomaly_rate"), "observed.anomaly_rate")
    fault = _finite_number(observed.get("fault_rate"), "observed.fault_rate")
    baseline_anomaly = _finite_number(baseline.get("anomaly_rate"), "baseline.anomaly_rate")
    baseline_fault = _finite_number(baseline.get("fault_rate"), "baseline.fault_rate")
    if anomaly > baseline_anomaly + anomaly_rate_margin or fault > baseline_fault + fault_rate_margin:
        root = Path(model_root).resolve()
        return rollback_model(
            model_root=root,
            pointer_path=pointer_path or root / "active_model.json",
            public_key_path=public_key_path,
        )
    return None


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


def monitor_post_promotion_rates(
    storage: LumenStorageManager, *, model_root: str | Path, public_key_path: str | Path,
    pointer_path: str | Path | None = None, window_sec: float = 300.0, min_events: int = 20,
    anomaly_rate_margin: float = 0.05, fault_rate_margin: float = 0.01,
) -> dict[str, Any] | None:
    """Monitor recent local rates and automatically restore previous_model.json on regression."""
    root = Path(model_root).resolve()
    pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
    active_dir = resolve_active_bundle(root, pointer, root / "lumen_model_bundle")
    verified = verify_and_load_bundle(active_dir, public_key_path)
    promotion = verified.manifest.get("promotion_metrics")
    candidate_baseline = promotion.get("candidate") if isinstance(promotion, dict) else None
    if not isinstance(candidate_baseline, dict):
        return None
    if "clean_security_alert_rate" not in candidate_baseline or "clean_operational_fault_rate" not in candidate_baseline:
        # Older signed models lack rate baselines: fail safe by not inventing a
        # reference. A future successful promotion writes both values.
        return None
    observed = read_post_promotion_rates(storage, window_sec=window_sec, min_events=min_events)
    if observed is None:
        return None
    baseline = {
        "anomaly_rate": _finite_number(candidate_baseline["clean_security_alert_rate"], "baseline.clean_security_alert_rate"),
        "fault_rate": _finite_number(candidate_baseline["clean_operational_fault_rate"], "baseline.clean_operational_fault_rate"),
    }
    rollback = rollback_on_rate_spike(
        observed, baseline, model_root=root, public_key_path=public_key_path, pointer_path=pointer,
        anomaly_rate_margin=anomaly_rate_margin, fault_rate_margin=fault_rate_margin,
    )
    if rollback is None:
        return {"rollback": False, "observed": observed, "baseline": baseline}
    result = {"rollback": True, "observed": observed, "baseline": baseline, "pointer": rollback}
    _append_audit(root / "promotion_history.jsonl", {
        "at_ns": time.time_ns(), "event": "automatic_post_promotion_rollback",
        "observed_rates": observed, "baseline_rates": baseline, "active_pointer": rollback,
    })
    return result
