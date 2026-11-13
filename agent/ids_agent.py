"""Lumen IDS Agent v5.0:
- Verificare criptografică Ed25519 a pachetului de model (lumen_model_bundle)
- Extractor de 12 trăsături per (device_id, topic) + RobustFeatureScaler
- Detecție hibridă: Temporal LSTM Autoencoder + Two-Sided CUSUM + OperationalFaultClassifier
- Explainable AI (XAI) normalizat cu direcție
- Risk & Policy Engine determinist + publicare comenzi PEP pe control/policy/<device_id>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from importlib import import_module
from pathlib import Path
from queue import Empty, Full, Queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import onnxruntime as ort
import zenoh

_zenoh_handlers = import_module("zenoh.handlers")
FifoChannel = _zenoh_handlers.FifoChannel
RingChannel = _zenoh_handlers.RingChannel

from training.feature_extractor import (
    TOPIC_MAP,
    RateMeter,
    rule_flood,
    truncate_payload,
    MAX_PAYLOAD_BYTES as FEATURE_MAX_PAYLOAD_BYTES,
)
from agent.enforcement import EnforcementEngine, sanitize, ALERT_TOPIC
from agent.statistical_detectors import TwoSidedCusumDetector, OperationalFaultClassifier
from agent.policy_engine import RiskPolicyEngine, DetectionEvidence
from agent.pep_protocol import PepAckRegistry, PepAcknowledgement
from storage.telemetry_db import LumenStorageManager, DBPaths
from models.bundle_loader import verify_and_load_bundle
from utils.features_v5 import (
    DeviceTopicFeatureExtractor,
    RobustFeatureScaler,
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
)
from utils.sensor_payload import parse_sensor_payload
from utils.xai_explainer import NormalizedXaiExplainer
from utils.device_registry import DeviceRegistry
from utils.device_identity import load_active_device_identity
from utils.cert_lifecycle import CertRevocationLedger, revoke_active_certificate
from utils.identity_envelope import (
    IdentityEnvelope, validate_active_identity, validate_envelope_route,
)
from models.model_lifecycle import resolve_active_bundle
from utils.crypto_utils import CommandSigner
from utils.replay_store import ReplayStore
from utils.rate_limit import TokenBucketRateLimiter
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("ids_agent")

MAX_PAYLOAD_BYTES = 512
ACTUATOR_BURST_STREAK = 4
STATS_INTERVAL_SEC = 10.0
RATE_ALERT_INTERVAL_SEC = 1.0
IGNORED_PREFIXES = ("ingest/", "shadow/", "control/policy/", "control/pep_ack/")
INTERNAL_TELEMETRY_PREFIX = "internal/ids/telemetry/"
PEP_ACK_PREFIX = "control/pep_ack/"
FAIL_CLOSED_LOSS = 999.0

LAB_EXTRA_COMMAND_TOPICS = frozenset({"home/actuators/garage/door"})
KNOWN_TOPICS = (frozenset(TOPIC_MAP) | LAB_EXTRA_COMMAND_TOPICS) - {ALERT_TOPIC}

if MAX_PAYLOAD_BYTES != FEATURE_MAX_PAYLOAD_BYTES:
    raise RuntimeError(
        f"[IDS AGENT] Limite inconsistente: MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES} "
        f"!= feature_extractor.MAX_PAYLOAD_BYTES={FEATURE_MAX_PAYLOAD_BYTES}"
    )


class ZenohIDSAgent:
    def __init__(
        self,
        bundle_dir: str = "models/lumen_model_bundle",
        pub_key_path: str = "config/model_keys/model_signing_ed25519.pub",
    ):
        logger.info("[IDS AGENT v5.0] Inițializare detector hibrid (LSTM ONNX + CUSUM + XAI + Policy Engine)...")

        self.registry = DeviceRegistry.load(
            os.environ.get("LUMEN_DEVICE_REGISTRY", "config/devices.json")
        )
        self._device_to_topic: dict[str, str] = {}
        self._device_criticality: dict[str, str] = {}
        for spec in self.registry.enabled_devices():
            self._device_to_topic[spec.device_id] = spec.target_topic
            self._device_criticality[spec.device_id] = (
                "safety_critical" if "gas" in spec.device_id else "normal"
            )
        self.cert_pointer = os.environ.get("LUMEN_CERT_POINTER")
        if not self.cert_pointer:
            raise RuntimeError("LUMEN_CERT_POINTER is required to validate Gateway identity envelopes")

        self.storage = LumenStorageManager(
            paths=DBPaths.from_dir(os.environ.get("LUMEN_DB_DIR", "logs/db")),
            enable_writers=True,
        )
        for spec in self.registry.enabled_devices():
            pointer = os.environ.get("LUMEN_CERT_POINTER")
            if not pointer:
                raise RuntimeError("LUMEN_CERT_POINTER is required to bind devices to real certificates")
            identity = load_active_device_identity(pointer, spec.device_id)
            initial_state = "ACTIVE"
            ledger_path = os.environ.get("LUMEN_CERT_REVOCATIONS")
            if ledger_path and CertRevocationLedger(ledger_path).is_revoked(
                spec.device_id, identity.serial_number_hex
            ):
                initial_state = "REVOKED"
            self.storage.upsert_device(
                device_id=spec.device_id,
                generation=identity.generation,
                device_cn=identity.common_name,
                cert_fingerprint=identity.certificate_fingerprint,
                device_role="sensor",
                criticality=self._device_criticality[spec.device_id],
                state=initial_state,
            )

        self.ml_enabled = False
        self.ort_session = None
        self.input_name = "window_input"
        self.seq_len = 16
        self.global_threshold = 1.5
        self.stream_thresholds: dict[str, float] = {}
        self.scaler: RobustFeatureScaler | None = None
        self.xai: NormalizedXaiExplainer | None = None

        explicit_bundle = os.environ.get("LUMEN_MODEL_BUNDLE_DIR")
        if explicit_bundle:
            bdir = Path(explicit_bundle)
        else:
            bdir = resolve_active_bundle(
                model_root="models",
                pointer_path=os.environ.get("LUMEN_ACTIVE_MODEL_POINTER", "models/active_model.json"),
                fallback_bundle=bundle_dir,
            )
        pkey = Path(os.environ.get("LUMEN_MODEL_PUBKEY", pub_key_path))
        allow_fallback = os.environ.get("LUMEN_ALLOW_UNBUNDLED_FALLBACK", "1") == "1"

        if (bdir / "model_manifest.json").is_file() and pkey.is_file():
            self._load_verified_bundle(bdir, pkey)
        elif allow_fallback:
            logger.warning(
                "[IDS AGENT v5.0] Bundle semnat negăsit în %s; pornire în mod tranzitoriu "
                "(CUSUM + Fault Classifier + Reguli deterministe) până la antrenarea modelului E5.",
                bdir,
            )
        else:
            raise RuntimeError(f"Fail-closed: pachetul de model semnat lipsește din {bdir}")

        self.extractor = DeviceTopicFeatureExtractor(seq_len=self.seq_len, stat_window=self.seq_len)
        self.cusum = TwoSidedCusumDetector(k_allowance=0.5, h_threshold=6.0)
        self.fault_classifier = OperationalFaultClassifier(frozen_min_steps=8)

        if self.ml_enabled and hasattr(self, "_cusum_baselines"):
            for skey, st in self._cusum_baselines.items():
                if "::" in skey:
                    dev, top = skey.split("::", 1)
                    self.cusum.calibrate_stream(dev, top, st["mean"], st["std"])

        self.actuator_queue: Queue = Queue(maxsize=5000)
        self.sensor_queue: Queue = Queue(maxsize=2000)
        self.wake = threading.Event()
        self._streak = 0

        self.rate_limiter = TokenBucketRateLimiter(rate=30.0, burst=60.0, max_keys=max(1, len(self._device_to_topic) + len(KNOWN_TOPICS) + 1))
        self.unknown_limiter = TokenBucketRateLimiter(rate=5.0, burst=10.0, max_keys=1)

        self._meter_lock = threading.Lock()
        self.meters: dict[str, RateMeter] = {}
        self._rate_alert_next: dict[str, float] = {}
        self._feature_lock = threading.Lock()

        CommandSigner.validate_trust_store()
        self.replay_store = ReplayStore.for_component("ids")
        self.stats = {
            "unknown": 0,
            "rate": 0,
            "oversize": 0,
            "queue_full": 0,
            "ml_inferences": 0,
            "faults_isolated": 0,
            "quarantines_issued": 0,
            "nonfinite": 0,
        }
        self._last_stats = dict(self.stats)

        self.running = True
        self.subs = []
        self.pumps = []
        self.worker_thread = None

        self.session = open_mtls_client(os.environ.get("LUMEN_CERT_IDENTITY", "ids"))
        self.enforcement = EnforcementEngine(self.session)
        self.pep_ack_registry = PepAckRegistry()
        self.pep_ack_timeout_sec = float(os.environ.get("LUMEN_PEP_ACK_TIMEOUT_SEC", "2.0"))
        self.policy_engine = RiskPolicyEngine(
            storage=self.storage,
            pep_request_fn=self._dispatch_pep_policy,
            revocation_fn=self._revoke_device_certificate,
            device_criticality=self._device_criticality,
            window_n=5,
            trigger_k=3,
            recovery_clean_windows=8,
            fault_recovery_clean_windows=int(os.environ.get("LUMEN_FAULT_RECOVERY_CLEAN_WINDOWS", "8")),
        )

    def _load_verified_bundle(self, bundle_dir: Path, pub_key_path: Path) -> None:
        verified = verify_and_load_bundle(bundle_dir, pub_key_path)
        schema_ver = verified.feature_schema.get("schema_version")
        if schema_ver != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema model incompatibilă: bundle={schema_ver} != runtime={FEATURE_SCHEMA_VERSION}"
            )
        if int(verified.feature_schema.get("feature_dim", 0)) != FEATURE_DIM:
            raise RuntimeError("Dimensiune vector trăsături incompatibilă în bundle")

        self.seq_len = int(verified.feature_schema["seq_len"])
        self.scaler = RobustFeatureScaler.from_dict(verified.scaler)
        self.global_threshold = float(verified.thresholds["global_threshold"])
        self.stream_thresholds = {
            k: float(v) for k, v in verified.thresholds.get("stream_thresholds", {}).items()
        }
        self.xai = NormalizedXaiExplainer(verified.thresholds["xai_error_scales"])
        self._cusum_baselines = verified.thresholds.get("cusum_baselines", {})

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            str(verified.onnx_path), so, providers=["CPUExecutionProvider"]
        )
        model_input = self.ort_session.get_inputs()[0]
        self.input_name = model_input.name
        in_shape = list(model_input.shape)
        if len(in_shape) != 3 or (isinstance(in_shape[2], int) and in_shape[2] != FEATURE_DIM):
            raise RuntimeError(f"Model ONNX cu intrare invalidă {in_shape}")

        self.ml_enabled = True
        logger.info(
            "[IDS AGENT v5.0] Bundle Ed25519 verificat cu succes: version=%s seq_len=%d thr=%.4f",
            verified.manifest.get("model_version"),
            self.seq_len,
            self.global_threshold,
        )

    def _resolve_device_id(self, topic: str, device_id: str | None = None) -> str:
        """Resolve only an envelope-bound stable ID; never infer from shared-topic suffixes."""
        if not device_id or self._device_to_topic.get(device_id) != topic:
            raise ValueError("sensor identity is missing or not authorized for this topic")
        return device_id

    def _validate_identity_envelope(self, key: str, raw: bytes):
        if len(raw) > 2048:
            raise ValueError("internal telemetry envelope exceeds size limit")
        env, payload = IdentityEnvelope.decode(raw)
        validate_envelope_route(env, key, self._device_to_topic)
        identity = load_active_device_identity(self.cert_pointer, env.device_id)
        validate_active_identity(env, identity)
        return env, payload

    def _on_pep_ack(self, sample) -> None:
        """Accept only a correctly correlated, first-seen Gateway PEP ACK."""
        try:
            key = str(sample.key_expr)
            raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            payload = json.loads(raw.decode("utf-8"))
            ack = PepAcknowledgement(
                decision_id=str(payload["decision_id"]),
                device_id=str(payload["device_id"]),
                requested_action=str(payload["requested_action"]),
                applied_action=str(payload["applied_action"]),
                result=str(payload["result"]),
                timestamp_ns=int(payload["timestamp_ns"]),
                gateway_instance_id=str(payload["gateway_instance_id"]),
            )
            if key != f"{PEP_ACK_PREFIX}{ack.decision_id}":
                return
            if not self.pep_ack_registry.accept(ack):
                logger.warning("[PEP ACK REJECTED] decision=%s", sanitize(ack.decision_id, 48))
        except Exception:
            logger.exception("[PEP ACK] malformed acknowledgment rejected")

    def _publish_pep_request(self, device_id: str, decision_id: str, target_state: str, **extra) -> bool:
        command = {
            "decision_id": decision_id,
            "device_id": device_id,
            "target_state": target_state,
            "issued_at_unix": time.time(),
            "source": "ids_policy_engine_v5",
            **extra,
        }
        try:
            self.session.put(
                f"control/policy/{device_id}",
                json.dumps(command, separators=(",", ":")).encode("utf-8"),
            )
            return True
        except Exception as exc:
            logger.error("[PEP DISPATCH ERROR] device=%s target=%s err=%s",
                         sanitize(device_id, 64), target_state, sanitize(exc, 120))
            return False

    def _wait_for_pep_ack(self, decision_id: str, device_id: str, requested_action: str) -> bool:
        ack = self.pep_ack_registry.wait(decision_id, self.pep_ack_timeout_sec)
        return bool(
            ack is not None and ack.confirmed and ack.device_id == device_id
            and ack.requested_action == requested_action
        )

    def _best_effort_reassert_quarantine(self, device_id: str, source_decision_id: str) -> None:
        """Compensate for an uncertain release; Gateway APPLY is idempotent."""
        decision_id = f"{source_decision_id}-reassert-{uuid.uuid4().hex[:8]}"
        if not self._publish_pep_request(
            device_id, decision_id, "SHADOW_QUARANTINE", phase="APPLY",
        ):
            logger.critical("[PEP FAIL-CLOSED] unable to publish quarantine reassertion for %s",
                            sanitize(device_id, 64))

    def _dispatch_pep_policy(self, device_id: str, enable_quarantine: bool, decision_id: str) -> bool:
        """Correlate quarantine ACKs and use a staged fail-closed release.

        Gateway remains quarantined through PREPARE and READY acknowledgments.
        IDS sends FINALIZE only after both exact ACKs are received; either
        timeout triggers a compensating quarantine assertion.
        """
        if enable_quarantine:
            action = "SHADOW_QUARANTINE"
            self.pep_ack_registry.register(decision_id, device_id, action)
            if not self._publish_pep_request(device_id, decision_id, action, phase="APPLY"):
                self.pep_ack_registry.cancel(decision_id)
                return False
            return self._wait_for_pep_ack(decision_id, device_id, action)

        prepare_id = f"{decision_id}-prepare"
        commit_id = f"{decision_id}-commit"
        self.pep_ack_registry.register(prepare_id, device_id, "RELEASE_PREPARE")
        if not self._publish_pep_request(
            device_id, prepare_id, "ACTIVE", phase="PREPARE_RELEASE",
        ):
            self.pep_ack_registry.cancel(prepare_id)
            self._best_effort_reassert_quarantine(device_id, decision_id)
            return False
        if not self._wait_for_pep_ack(prepare_id, device_id, "RELEASE_PREPARE"):
            self._best_effort_reassert_quarantine(device_id, decision_id)
            return False

        self.pep_ack_registry.register(commit_id, device_id, "RELEASE_READY")
        if not self._publish_pep_request(
            device_id, commit_id, "ACTIVE", phase="COMMIT_RELEASE",
            prepare_decision_id=prepare_id,
        ):
            self.pep_ack_registry.cancel(commit_id)
            self._best_effort_reassert_quarantine(device_id, decision_id)
            return False
        if not self._wait_for_pep_ack(commit_id, device_id, "RELEASE_READY"):
            self._best_effort_reassert_quarantine(device_id, decision_id)
            return False

        # Both Gateway acknowledgments have reached IDS while Gateway remains
        # quarantined. Finalize is one-way: if delivery is lost, Gate remains
        # blocked (availability loss, not an ingress security bypass).
        finalize_id = f"{decision_id}-finalize"
        if not self._publish_pep_request(
            device_id, finalize_id, "ACTIVE", phase="FINALIZE_RELEASE",
            prepare_decision_id=prepare_id, ready_decision_id=commit_id,
        ):
            self._best_effort_reassert_quarantine(device_id, decision_id)
            return False
        return True

    def _revoke_device_certificate(self, device_id: str, reason: str) -> bool:
        """Revoke an active certificate and explicitly verify its serial in the ledger."""
        pointer = self.cert_pointer
        ledger_path = os.environ.get("LUMEN_CERT_REVOCATIONS")
        if not pointer or not ledger_path:
            logger.error("[CERT REVOCATION] pointer/ledger path is not configured")
            return False
        try:
            identity = load_active_device_identity(pointer, device_id)
            revoke_active_certificate(pointer, ledger_path, device_id, reason, quarantine_only=False)
            return CertRevocationLedger(ledger_path).is_revoked(device_id, identity.serial_number_hex)
        except Exception:
            logger.exception("[CERT REVOCATION] failed or ledger verification did not complete for %s",
                             sanitize(device_id, 64))
            return False

    def _publish_v5_alert(
        self,
        device_id: str,
        topic: str,
        attack_label: str,
        action: str,
        score: float,
        detail: str,
        xai_report=None,
        severity: str = "high",
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "topic": topic,
            "entity_type": "device",
            "entity_id": f"device:{device_id}",
            "device_id": device_id,
            "attack": attack_label,
            "action": action,
            "loss": round(float(score), 4),
            "detail": sanitize(detail, 180),
            "severity": severity,
            "repeats": 0,
            "source": "IDS Agent v5.0",
        }
        if xai_report is not None:
            payload["xai_summary"] = xai_report.summary_text
            payload["xai_top_features"] = [
                {
                    "feature": a.feature,
                    "contribution_pct": a.contribution_pct,
                    "direction": a.direction,
                }
                for a in xai_report.top_attributions
            ]
        try:
            self.session.put(ALERT_TOPIC, json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        except Exception as exc:
            logger.error("[IDS ALERT PUBLISH] %s", sanitize(exc, 120))

    def _process_actuator(self, key: str, payload_str: str) -> None:
        ok, reason = CommandSigner.verify_command(
            key, payload_str, replay_store=self.replay_store
        )
        if not ok:
            logger.warning("[CRYPTO REJECTION] '%s': %s", sanitize(key, 80), sanitize(reason, 120))
            if reason.startswith("REPLAY"):
                attack = "Replay Attack"
            elif reason.startswith("KEY_REVOKED"):
                attack = "Revoked Key Use"
            else:
                attack = "Command Tampering"
            self.enforcement.execute_mitigation(attack, key, 99.0, reason, critical=True)
            return
        logger.debug("[CRYPTO OK] '%s'", sanitize(key, 80))

    def _process_sensor(
        self, key: str, payload_str: str, delta_t: float, rate: float,
        device_id: str | None = None, generation: int = 1,
        admission_status: str = "ADMITTED", admission_reason: str = "LEGACY_UNVERIFIED",
        admission_numeric_value: float | None = None,
    ) -> None:
        payload_str = truncate_payload(payload_str)
        raw_bytes = payload_str.encode("utf-8", errors="strict")
        device_id = self._resolve_device_id(key, device_id)
        parsed = parse_sensor_payload(key, raw_bytes)
        disagreement = (
            admission_status == "ADMITTED"
            and (not parsed.valid or (
                admission_numeric_value is not None
                and parsed.numeric_value is not None
                and not math.isclose(float(admission_numeric_value), parsed.numeric_value, rel_tol=1e-9, abs_tol=1e-9)
            ))
        )
        if admission_status != "ADMITTED" or disagreement or not parsed.valid:
            reason = admission_reason if admission_status == "REJECTED" else (
                "GATE_IDS_SCHEMA_DISAGREEMENT" if disagreement else parsed.reason
            )
            self._record_protocol_violation(device_id, generation, key, reason)
            return
        # Both Gateway and IDS operate on the same normalized scalar string.
        payload_str = parsed.canonical_text
        raw_bytes = payload_str.encode("utf-8")

        flood = rule_flood(delta_t, rate)
        if flood:
            self.enforcement.execute_mitigation(
                "DoS / Topic Flooding",
                key,
                99.0,
                f"rate={rate:.1f} dt={delta_t:.3f}",
            )

        with self._feature_lock:
            step_stats = self.extractor.update(
                device_id=device_id,
                topic=key,
                delta_t_sec=delta_t,
                raw_payload=raw_bytes,
                operation="put",
            )
            cusum_res = self.cusum.update(device_id, key, step_stats.raw_value)
            op_health = self.fault_classifier.evaluate(
                device_id=device_id,
                topic=key,
                raw_value=step_stats.raw_value,
                delta_t=delta_t,
                operation="put",
            )
            window = self.extractor.get_window(device_id, key)

        if op_health.status == "FAULT":
            self.stats["faults_isolated"] += 1

        lstm_score = 0.0
        lstm_thr = self.stream_thresholds.get(f"{device_id}::{key}", self.global_threshold)
        lstm_anomalous = False
        xai_report = None

        if self.ml_enabled and window is not None and self.scaler is not None and self.xai is not None:
            scaled = self.scaler.transform(window)
            try:
                recon = self.ort_session.run(None, {self.input_name: scaled[None, :, :]})[0][0]
                self.stats["ml_inferences"] += 1
                xai_report = self.xai.explain(
                    x_last_scaled=scaled[-1],
                    recon_last_scaled=recon[-1],
                    threshold=lstm_thr,
                    top_k=3,
                )
                lstm_score = xai_report.anomaly_score
                lstm_anomalous = xai_report.is_anomalous
            except Exception as exc:
                self.stats["nonfinite"] += 1
                logger.exception("[IDS ML ERROR] inferență eșuată pe '%s': %s", sanitize(key, 80), exc)
                lstm_score = FAIL_CLOSED_LOSS
                lstm_anomalous = True

        if not math.isfinite(lstm_score):
            self.stats["nonfinite"] += 1
            lstm_score = FAIL_CLOSED_LOSS
            lstm_anomalous = True

        evidence = DetectionEvidence(
            device_id=device_id,
            generation=generation,
            topic=key,
            lstm_score=lstm_score,
            lstm_threshold=lstm_thr,
            lstm_anomalous=lstm_anomalous,
            cusum_score=cusum_res.score,
            cusum_triggered=cusum_res.triggered,
            operational_status=op_health.status,
            fault_type=op_health.fault_type,
            deterministic_violation=bool(flood),
            deterministic_reason="FLOOD_RATE_VIOLATION" if flood else "",
            xai_summary=xai_report.summary_text if xai_report else f"CUSUM={cusum_res.score:.2f}",
        )
        decision = self.policy_engine.evaluate_evidence(evidence)

        if decision.action == "ALERT_FAULT":
            self._publish_v5_alert(
                device_id=device_id,
                topic=key,
                attack_label=f"Operational Fault ({op_health.fault_type})",
                action="MAINTENANCE_ALERT",
                score=0.0,
                detail=op_health.detail,
                severity="medium",
            )
        elif decision.action == "QUARANTINE_SHADOW":
            self.stats["quarantines_issued"] += 1
            detail = (
                f"{decision.reason_code} | score={lstm_score:.3f}>{lstm_thr:.3f} "
                f"cusum={cusum_res.score:.2f} | {evidence.xai_summary}"
            )
            self._publish_v5_alert(
                device_id=device_id,
                topic=key,
                attack_label="Behavioral Cyber Anomaly",
                action="SHADOW_QUARANTINE",
                score=max(lstm_score, cusum_res.score),
                detail=detail,
                xai_report=xai_report,
                severity="critical",
            )
            logger.critical(
                "[SHADOW QUARANTINE] device=%s topic=%s reason=%s xai=%s",
                sanitize(device_id, 64),
                sanitize(key, 80),
                decision.reason_code,
                sanitize(evidence.xai_summary, 140),
            )
        elif decision.action == "RELEASE":
            self._publish_v5_alert(
                device_id=device_id,
                topic=key,
                attack_label="Behavioral Recovery Verified",
                action="RELEASE_QUARANTINE",
                score=lstm_score,
                detail="Dispozitiv readus în starea ACTIVE după ferestre consecutive curate",
                severity="low",
            )

    def _record_protocol_violation(self, device_id: str, generation: int, topic: str, reason: str) -> None:
        """Keep rejected ingress out of normal features and feed explicit protocol evidence."""
        self.stats["unknown"] += 1
        evidence = DetectionEvidence(
            device_id=device_id, generation=generation, topic=topic,
            lstm_score=0.0, lstm_threshold=1.0, lstm_anomalous=False,
            cusum_score=0.0, cusum_triggered=False,
            operational_status="HEALTHY", fault_type="NONE",
            deterministic_violation=True,
            deterministic_reason=f"ADMISSION_REJECTED:{reason}"[:160],
            xai_summary="Gateway AdmissionGate rejected the raw sensor payload",
        )
        decision = self.policy_engine.evaluate_evidence(evidence)
        self._publish_v5_alert(
            device_id=device_id, topic=topic,
            attack_label="Sensor Protocol / Admission Violation",
            action="ADMISSION_REJECTED",
            score=1.0,
            detail=f"AdmissionGate/IDS shared-schema rejection: {reason}"[:180],
            severity="high",
        )
        if decision.action == "QUARANTINE_SHADOW":
            self.stats["quarantines_issued"] += 1

    def _process_sample(self, key, payload_bytes, recv_time, delta_t, rate, device_id=None, generation=1,
                        admission_status="ADMITTED", admission_reason="LEGACY_UNVERIFIED",
                        admission_numeric_value=None) -> None:
        payload_str = payload_bytes.decode("utf-8", errors="replace")
        if key.startswith("home/actuators"):
            self._process_actuator(key, payload_str)
        else:
            self._process_sensor(
                key, truncate_payload(payload_str), delta_t, rate, device_id, generation,
                admission_status, admission_reason, admission_numeric_value,
            )

    def _next_item(self):
        self.wake.clear()

        if self._streak < ACTUATOR_BURST_STREAK:
            try:
                item = self.actuator_queue.get_nowait()
                self._streak += 1
                return item
            except Empty:
                pass
        try:
            item = self.sensor_queue.get_nowait()
            self._streak = 0
            return item
        except Empty:
            pass
        try:
            item = self.actuator_queue.get_nowait()
            self._streak = 0
            return item
        except Empty:
            pass

        self.wake.wait(0.05)
        return None

    def _worker_loop(self) -> None:
        while self.running:
            item = self._next_item()
            if item is None:
                continue
            try:
                self._process_sample(*item)
            except Exception as exc:
                logger.exception("[WORKER RECOVERY] Eșec la procesare '%s': %s", sanitize(item[0], 80), exc)

    def _on_sample(self, sample) -> None:
        key = str(sample.key_expr)
        if key == ALERT_TOPIC or key.startswith(PEP_ACK_PREFIX):
            return
        if key.startswith(INTERNAL_TELEMETRY_PREFIX):
            now = time.monotonic()
            try:
                raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
                env, payload = self._validate_identity_envelope(key, raw)
            except Exception as exc:
                self.stats["unknown"] += 1
                logger.warning("[IDENTITY ENVELOPE REJECTED] key=%s reason=%s",
                               sanitize(key, 100), sanitize(exc, 160))
                return
            stream_key = f"{env.device_id}::{env.original_topic}"
            with self._meter_lock:
                meter = self.meters.get(stream_key)
                if meter is None:
                    meter = self.meters[stream_key] = RateMeter(now)
                delta_t, rate = meter.observe(now)
            if not self.rate_limiter.allow(stream_key, now):
                self.stats["rate"] += 1
                return
            if len(payload) == 0 or len(payload) > MAX_PAYLOAD_BYTES:
                self.stats["oversize"] += 1
                return
            try:
                self.sensor_queue.put_nowait((
                    env.original_topic, payload, now, delta_t, rate, env.device_id, env.generation,
                    env.admission_status, env.admission_reason, env.admission_numeric_value,
                ))
                self.wake.set()
            except Full:
                self.stats["queue_full"] += 1
            return

        # Raw home/sensors/** is no longer authoritative for IDS identity. The
        # Gateway-only internal envelope is the single sensor telemetry path.
        if key.startswith("home/sensors/") or key.startswith(IGNORED_PREFIXES):
            return
        if key not in KNOWN_TOPICS:
            if key.startswith("home/actuators/"):
                self.stats["unknown"] += 1
                return
            self.stats["unknown"] += 1
            now = time.monotonic()
            if self.unknown_limiter.allow("unknown", now):
                self.enforcement.execute_mitigation(
                    "Reconnaissance / Scanning", sanitize(key, 80), 99.0, "unknown topic"
                )
            return

        now = time.monotonic()
        with self._meter_lock:
            meter = self.meters.get(key)
            if meter is None:
                meter = self.meters[key] = RateMeter(now)
            delta_t, rate = meter.observe(now)
        if not self.rate_limiter.allow(key, now):
            self.stats["rate"] += 1
            return
        raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
        if len(raw) > MAX_PAYLOAD_BYTES:
            self.stats["oversize"] += 1
            self.enforcement.execute_mitigation("Anomalous Payload", key, 99.0, f"oversize {len(raw)}B")
            return
        try:
            self.actuator_queue.put_nowait((key, raw, now, delta_t, rate))
            self.wake.set()
        except Full:
            self.stats["queue_full"] += 1

    def _pump(self, sub, skip_known: bool = False) -> None:
        try:
            for sample in sub:
                if not self.running:
                    break
                try:
                    if skip_known and str(sample.key_expr) in KNOWN_TOPICS:
                        continue
                    self._on_sample(sample)
                except Exception as exc:
                    logger.exception("[PUMP RECOVERY] %s", exc)
        except Exception:
            pass

    def _log_stats(self) -> None:
        delta = {k: self.stats[k] - self._last_stats[k] for k in self.stats}
        if any(delta.values()):
            logger.warning(
                "[STATS %.0fs] ml_inf=%d faults=%d quarantines=%d unknown=%d rate_limited=%d "
                "oversize=%d queue_full=%d nonfinite=%d",
                STATS_INTERVAL_SEC,
                delta["ml_inferences"],
                delta["faults_isolated"],
                delta["quarantines_issued"],
                delta["unknown"],
                delta["rate"],
                delta["oversize"],
                delta["queue_full"],
                delta["nonfinite"],
            )
        self._last_stats = dict(self.stats)

    def start_listening(self) -> None:
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self.subs = [
            (self.session.declare_subscriber("home/actuators/**", FifoChannel(1024)), False),
            (self.session.declare_subscriber(f"{INTERNAL_TELEMETRY_PREFIX}**", RingChannel(512)), False),
        ]
        self.pep_ack_sub = self.session.declare_subscriber(f"{PEP_ACK_PREFIX}**", self._on_pep_ack)
        self.pumps = []
        for sub, skip in self.subs:
            t = threading.Thread(target=self._pump, args=(sub, skip), daemon=True)
            t.start()
            self.pumps.append(t)

        logger.info(
            "[IDS AGENT v5.0] Client Zenoh mTLS activ (ml_enabled=%s): canale mărginite + Policy Engine...",
            self.ml_enabled,
        )

        try:
            while True:
                time.sleep(STATS_INTERVAL_SEC)
                self._log_stats()
        except KeyboardInterrupt:
            logger.info("[IDS AGENT v5.0] Oprire agent de monitorizare.")
        finally:
            self.running = False
            for sub, _ in self.subs:
                try:
                    sub.undeclare()
                except Exception:
                    pass
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=2.0)
            try:
                self.storage.flush_all()
                self.storage.close()
            except Exception:
                pass
            self.session.close()
            self.replay_store.close()


if __name__ == "__main__":
    agent = ZenohIDSAgent()
    agent.start_listening()