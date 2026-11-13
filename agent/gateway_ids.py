"""Lumen inline gateway with HMAC actuator checks, Patch 03 device isolation,
and v5.0 Policy Enforcement Point (PEP) for Shadow Quarantine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from queue import Empty, Full, Queue
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zenoh

from utils.logger import get_script_logger

logger = get_script_logger("gateway_ids")

try:
    from utils.crypto_utils import CommandSigner, MAX_FRAME_BYTES
except (ValueError, ImportError) as exc:
    logger.critical("[GATEWAY IDS] Crypto initialization failed; refusing to start: %s", exc)
    raise SystemExit(1) from exc

from training.feature_extractor import TOPIC_MAP
from agent.enforcement import EnforcementEngine, sanitize, ALERT_TOPIC
from agent.forensic_sink import ForensicQuarantineSink
from agent.pep_protocol import PepAcknowledgement
from storage.telemetry_db import LumenStorageManager, DBPaths
from storage.admission_gate import AdmissionGate, ProvenanceContext
from utils.rate_limit import TokenBucketRateLimiter
from utils.mtls_hot_reload import open_mtls_client
from utils.cert_lifecycle import revoke_active_certificate
from utils.device_registry import DeviceRegistry
from utils.device_security import DeviceSecurityManager
from utils.device_identity import load_active_device_identity, load_active_device_signing_key
from utils.identity_envelope import IdentityEnvelope
from utils.sensor_payload import parse_sensor_frame, MAX_SENSOR_FRAME_BYTES
from utils.key_lifecycle import KeyHealthMonitor, RevocationLedger

MAX_PAYLOAD_BYTES = 512
INGEST_PREFIX = "ingest/"
POLICY_CONTROL_TOPIC = "control/policy/**"
INTERNAL_TELEMETRY_PREFIX = "internal/ids/telemetry/"
PEP_ACK_PREFIX = "control/pep_ack/"
SAFE_DEVICE_ID_RE = re.compile(r"^[a-zA-Z0-9_]{1,64}$")

ALLOWED_COMMAND_TOPICS = frozenset(
    topic for topic in TOPIC_MAP if topic.startswith("home/actuators/")
)
ALLOWED_SENSOR_TOPICS = frozenset(
    topic for topic in TOPIC_MAP if topic.startswith("home/sensors/")
)
RATE_ALERT_INTERVAL_SEC = 1.0
METRICS_INTERVAL_SEC = 2.0
DEFAULT_METRICS_PATH = "logs/gateway_metrics.jsonl"

if MAX_PAYLOAD_BYTES != MAX_FRAME_BYTES:
    raise RuntimeError(
        f"Payload size mismatch: gateway={MAX_PAYLOAD_BYTES}, crypto={MAX_FRAME_BYTES}"
    )


def _crypto_selfcheck() -> None:
    CommandSigner.validate_trust_store()


class ZenohInlineGateway:
    def __init__(self):
        logger.info("[GATEWAY IDS] Starting HMAC inline gateway with device-aware ingress and PEP...")
        _crypto_selfcheck()

        self.run_id = uuid.uuid4().hex[:12]
        self.running = True
        self._closed = False
        self._shutdown_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._nonce_lock = threading.Lock()
        self._shadow_lock = threading.Lock()
        self._rate_alert_next: dict[str, float] = {}
        self.last_seen_nonces: dict[str, int] = {}
        self._shadow_quarantined: set[str] = set()
        # Release is a fail-closed handshake: PREPARE and READY are ACKed
        # while ingress remains blocked; only IDS FINALIZE can open the Gate.
        self._release_prepared: dict[str, tuple[str, float, str]] = {}

        self.registry = DeviceRegistry.load(
            os.environ.get("LUMEN_DEVICE_REGISTRY", "config/devices.json")
        )
        self.device_security = DeviceSecurityManager(self.registry)
        self.ingest_queue: Queue = Queue(maxsize=int(os.environ.get("LUMEN_INGEST_QUEUE_SIZE", "10000")))
        self.device_alert_queue: Queue = Queue(maxsize=512)

        self.storage = LumenStorageManager(
            paths=DBPaths.from_dir(os.environ.get("LUMEN_DB_DIR", "logs/db")),
            enable_writers=True,
        )
        self.cert_pointer = os.environ.get("LUMEN_CERT_POINTER")
        if not self.cert_pointer:
            raise RuntimeError("LUMEN_CERT_POINTER is required for authenticated Gateway-to-IDS envelopes")
        allowed_topics = {spec.device_id: {spec.target_topic} for spec in self.registry.enabled_devices()}
        self.admission_gate = AdmissionGate(self.storage, allowed_topics)
        self.storage.register_baseline_session(
            session_id=self.run_id,
            source_type="live_validated",
            config={"gateway_instance_id": self.run_id, "approved_for_training": False},
            git_hash=os.environ.get("LUMEN_GIT_HASH", "v5.0-runtime"),
            approved_for_training=False,
        )
        # Restore fail-closed blocks before accepting any new sensor ingress.
        for device_id, state in self.storage.load_device_states().items():
            if state in {"SHADOW_QUARANTINE", "RECOVERY_PENDING", "ENFORCEMENT_PENDING", "ENFORCEMENT_FAILED", "QUARANTINED", "REVOKED"}:
                self._shadow_quarantined.add(device_id)
                self.admission_gate.set_device_blocked(device_id, True)
        self.forensic_sink = ForensicQuarantineSink(storage=self.storage)

        self.topic_rate_limiter = TokenBucketRateLimiter(
            rate=float(os.environ.get("LUMEN_GATEWAY_TOPIC_RATE", "30")),
            burst=float(os.environ.get("LUMEN_GATEWAY_TOPIC_BURST", "60")),
            max_keys=max(1, len(ALLOWED_SENSOR_TOPICS)),
        )
        self.global_rate_limiter = TokenBucketRateLimiter(
            rate=float(os.environ.get("LUMEN_GATEWAY_GLOBAL_RATE", "80")),
            burst=float(os.environ.get("LUMEN_GATEWAY_GLOBAL_BURST", "160")),
            max_keys=1,
        )
        self.unknown_limiter = TokenBucketRateLimiter(rate=5.0, burst=10.0, max_keys=1)

        self.stats = {
            "ingress_received": 0,
            "unknown_dropped": 0,
            "blocklist_dropped": 0,
            "rate_topic_dropped": 0,
            "rate_global_dropped": 0,
            "oversize_dropped": 0,
            "sensor_invalid_dropped": 0,
            "sensor_queue_full_dropped": 0,
            "sensor_queue_stale_dropped": 0,
            "sensor_queued": 0,
            "sensor_forwarded": 0,
            "shadow_forwarded": 0,
            "shadow_quarantine_transitions": 0,
            "actuator_queued": 0,
            "actuator_queue_full_dropped": 0,
            "actuator_verified": 0,
            "actuator_auth_rejected": 0,
            "actuator_replay_rejected": 0,
            "actuator_forwarded": 0,
            "publish_errors": 0,
            "callback_errors": 0,
            "device_rate_dropped": 0,
            "device_blocklist_dropped": 0,
            "device_unknown_dropped": 0,
            "device_blocks_issued": 0,
            "device_alert_queue_full": 0,
        }
        self.device_publish_accepted = {
            spec.device_id: 0 for spec in self.registry.enabled_devices()
        }

        self.key_monitor = None
        self.revocations = None
        if os.environ.get("LUMEN_REKEY_ENABLED", "0") == "1":
            self.key_monitor = KeyHealthMonitor.from_env()
            self.revocations = RevocationLedger(CommandSigner.revocation_path())

        # The proxy transparently reconnects and re-declares subscribers whenever
        # active.json changes. No static certificate paths or fallback plaintext mode.
        self.mtls_client = open_mtls_client("gateway")
        self.session = self.mtls_client
        self.enforcement = EnforcementEngine(self.session, is_inline_gateway=True)
        self.sub = None
        self.policy_sub = None
        self.worker_thread = None
        self.metrics_path = os.environ.get("LUMEN_GATEWAY_METRICS_PATH", DEFAULT_METRICS_PATH)
        Path(self.metrics_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _inc(self, name: str, amount: int = 1) -> None:
        with self._stats_lock:
            self.stats[name] = self.stats.get(name, 0) + amount

    def set_shadow_quarantine(self, device_id: str, enabled: bool) -> bool:
        """Aplică sau ridică SHADOW_QUARANTINE atomic și confirmă starea către Policy Engine."""
        if not SAFE_DEVICE_ID_RE.match(device_id):
            logger.warning("[PEP REJECT] Invalid device_id format: %r", device_id)
            return False
        with self._shadow_lock:
            if enabled:
                self._shadow_quarantined.add(device_id)
            else:
                self._shadow_quarantined.discard(device_id)
            confirmed = (device_id in self._shadow_quarantined) == enabled
        if hasattr(self, "admission_gate"):
            self.admission_gate.set_device_blocked(device_id, enabled)
        self._inc("shadow_quarantine_transitions")
        logger.warning(
            "[PEP STATE] device=%s shadow_quarantine=%s confirmed=%s",
            sanitize(device_id, 64),
            enabled,
            confirmed,
        )
        return confirmed

    def is_shadow_quarantined(self, device_id: str) -> bool:
        with self._shadow_lock:
            return device_id in self._shadow_quarantined

    def _on_policy_command(self, sample) -> None:
        """Apply idempotent PEP commands with a fail-closed staged release.

        PREPARE and READY are acknowledged while the device remains blocked.
        Gateway opens AdmissionGate only after IDS has received both correlated
        ACKs and sends FINALIZE_RELEASE with the matching transaction IDs.
        """
        key = str(sample.key_expr)
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "control" or parts[1] != "policy":
            return
        device_id = parts[2]
        phase = ""
        decision_id = ""
        target_state = ""
        try:
            raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            cmd = json.loads(raw.decode("utf-8"))
            target_state = str(cmd.get("target_state", "")).upper()
            decision_id = str(cmd.get("decision_id", ""))
            phase = str(cmd.get("phase", "APPLY")).upper()
            if cmd.get("device_id") != device_id or not decision_id:
                raise ValueError("PEP command device/decision correlation is missing or mismatched")

            if phase == "PREPARE_RELEASE":
                if target_state != "ACTIVE":
                    raise ValueError("PREPARE_RELEASE must request ACTIVE")
                # A prepare ACK is proof only that Gateway is still quarantined
                # and ready. It never opens the ingress gate.
                if not self.set_shadow_quarantine(device_id, True):
                    raise RuntimeError("could not confirm quarantined state before release prepare")
                ttl = max(1.0, float(os.environ.get("LUMEN_PEP_RELEASE_PREPARE_TTL_SEC", "10")))
                with self._shadow_lock:
                    self._release_prepared[device_id] = (decision_id, time.monotonic() + ttl)
                ack = PepAcknowledgement(
                    decision_id=decision_id, device_id=device_id,
                    requested_action="RELEASE_PREPARE", applied_action="RELEASE_PREPARE",
                    result="CONFIRMED", timestamp_ns=time.time_ns(), gateway_instance_id=self.run_id,
                )
                self._publish_pep_ack(ack)
                return

            if phase == "COMMIT_RELEASE":
                if target_state != "ACTIVE":
                    raise ValueError("COMMIT_RELEASE must request ACTIVE")
                prepared_id = str(cmd.get("prepare_decision_id", ""))
                with self._shadow_lock:
                    prepared = self._release_prepared.get(device_id)
                    valid_prepare = bool(
                        prepared and prepared[0] == prepared_id
                        and prepared[1] >= time.monotonic()
                    )
                if not valid_prepare:
                    raise ValueError("release commit has no matching, unexpired prepare")
                # This is the second release acknowledgment, but it is still
                # fail-closed: Gateway confirms readiness while Gate remains blocked.
                with self._shadow_lock:
                    self._release_prepared[device_id] = (prepared_id, prepared[1], decision_id)
                ack = PepAcknowledgement(
                    decision_id=decision_id, device_id=device_id,
                    requested_action="RELEASE_READY", applied_action="RELEASE_READY",
                    result="CONFIRMED", timestamp_ns=time.time_ns(), gateway_instance_id=self.run_id,
                )
                self._publish_pep_ack(ack)
                return

            if phase == "FINALIZE_RELEASE":
                if target_state != "ACTIVE":
                    raise ValueError("FINALIZE_RELEASE must request ACTIVE")
                prepared_id = str(cmd.get("prepare_decision_id", ""))
                ready_id = str(cmd.get("ready_decision_id", ""))
                with self._shadow_lock:
                    prepared = self._release_prepared.get(device_id)
                    valid_finalize = bool(
                        prepared and prepared[0] == prepared_id
                        and prepared[2] == ready_id
                        and prepared[1] >= time.monotonic()
                    )
                if not valid_finalize:
                    raise ValueError("release finalize has no matching, unexpired ready transaction")
                # IDS has already received the READY ACK before it can send this
                # command. Only now may Gateway open the admission gate. There is
                # deliberately no post-open ACK on which security depends.
                if not self.set_shadow_quarantine(device_id, False):
                    raise RuntimeError("Gateway could not apply the finalized release")
                with self._shadow_lock:
                    self._release_prepared.pop(device_id, None)
                return

            if phase == "CANCEL_RELEASE":
                with self._shadow_lock:
                    self._release_prepared.pop(device_id, None)
                ok = self.set_shadow_quarantine(device_id, True)
                action = "SHADOW_QUARANTINE"
            elif phase == "APPLY" and target_state == "SHADOW_QUARANTINE":
                # An explicit quarantine assertion supersedes any outstanding
                # release prepare token; a stale COMMIT must never reopen ingress.
                with self._shadow_lock:
                    self._release_prepared.pop(device_id, None)
                ok = self.set_shadow_quarantine(device_id, True)
                action = "SHADOW_QUARANTINE"
            else:
                # Bare ACTIVE commands are legacy one-phase releases and are
                # intentionally rejected; they could reopen ingress on a lost ACK.
                if target_state == "ACTIVE":
                    self.set_shadow_quarantine(device_id, True)
                raise ValueError(f"unsupported or unsafe PEP phase/target: {phase}/{target_state}")

            ack = PepAcknowledgement(
                decision_id=decision_id, device_id=device_id,
                requested_action=action,
                applied_action=action if ok else (
                    "SHADOW_QUARANTINE" if self.is_shadow_quarantined(device_id) else "ACTIVE"
                ),
                result="CONFIRMED" if ok else "FAILED",
                timestamp_ns=time.time_ns(), gateway_instance_id=self.run_id,
            )
            self._publish_pep_ack(ack)
        except Exception:
            # A failure anywhere in a release transaction restores the local
            # ingress block before the callback returns. IDS also compensates on
            # a missing ACK, so a dropped ACK converges back to quarantine.
            if target_state == "ACTIVE" or phase in {"PREPARE_RELEASE", "COMMIT_RELEASE", "CANCEL_RELEASE"}:
                try:
                    with self._shadow_lock:
                        self._release_prepared.pop(device_id, None)
                    self.set_shadow_quarantine(device_id, True)
                except Exception:
                    logger.critical("[PEP ROLLBACK] failed to restore quarantine for %s",
                                    sanitize(device_id, 64), exc_info=True)
            self._inc("callback_errors")
            logger.exception("[PEP POLICY CMD] Failed to apply/ack policy command for %s", sanitize(device_id, 64))

    def _publish_pep_ack(self, ack: PepAcknowledgement) -> None:
        self.session.put(
            f"{PEP_ACK_PREFIX}{ack.decision_id}",
            json.dumps({
                "decision_id": ack.decision_id, "device_id": ack.device_id,
                "requested_action": ack.requested_action, "applied_action": ack.applied_action,
                "result": ack.result, "timestamp_ns": ack.timestamp_ns,
                "gateway_instance_id": ack.gateway_instance_id,
            }, separators=(",", ":")).encode("utf-8"),
        )

    def _route_to_shadow_sandbox(self, device_id: str, target_topic: str, raw: bytes) -> None:
        """Persist and optionally publish quarantined telemetry using real PKI metadata."""
        if not SAFE_DEVICE_ID_RE.match(device_id):
            return
        try:
            identity = load_active_device_identity(self.cert_pointer, device_id)
        except Exception:
            logger.error("[SHADOW SINK] no valid active certificate identity for %s", sanitize(device_id, 64))
            return
        envelope = {
            "source_device": device_id, "original_topic": target_topic,
            "shadow_topic": f"shadow/sandbox/{device_id}/data",
            "gateway_received_at_ns": time.time_ns(),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_preview": raw.decode("utf-8", errors="replace")[:256],
        }
        encoded = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        try:
            self.session.put(f"shadow/sandbox/{device_id}/data", encoded)
            self._inc("shadow_forwarded")
        except Exception as exc:
            self._inc("publish_errors")
            logger.error("[SHADOW PUBLISH] device=%s error=%s", sanitize(device_id, 64), sanitize(exc))
        self.forensic_sink.ingest_shadow_envelope(
            session_id=self.run_id, device_id=device_id, device_cn=identity.common_name,
            generation=identity.generation, cert_fingerprint=identity.certificate_fingerprint,
            original_topic=target_topic, raw_payload=raw, risk_score=0.95,
            reason_code="SHADOW_QUARANTINE_ACTIVE",
        )

    def _reject_actuator(self, target: str, reason: str, raw: bytes = None) -> None:
        self._inc("actuator_auth_rejected")
        if reason.startswith("REPLAY"):
            self._inc("actuator_replay_rejected")
            attack = "Replay Attack"
        elif reason.startswith("KEY_REVOKED"):
            attack = "Revoked Key Use"
        else:
            attack = "Command Tampering"
        logger.warning("[CRYPTO REJECTION] topic=%s reason=%s", sanitize(target, 80), sanitize(reason, 120))
        self.enforcement.execute_mitigation(attack, target, 99.0, reason, critical=True)
        self._evaluate_rekey(target, reason, raw)

    def _evaluate_rekey(self, target: str, reason: str, raw) -> None:
        if self.key_monitor is None or raw is None:
            return
        decision = self.key_monitor.on_rejected(target, reason, raw)
        if not decision.revoke:
            return
        CommandSigner.revoke_in_process(decision.sender_id, decision.key_id)
        self._inc("rekey_requested")
        try:
            self.revocations.revoke(decision.sender_id, decision.key_id, decision.reason)
        except Exception:
            self._inc("rekey_errors")
            logger.critical("[REKEY] command-key ledger write failed; key stays revoked in this process only",
                            exc_info=True)
        # A command-key compromise for a role that also owns an mTLS identity
        # revokes that certificate serial and quarantines the role. This process
        # has ledger-write authority only; it never receives the CA signing key.
        cert_pointer = os.environ.get("LUMEN_CERT_POINTER")
        cert_ledger = os.environ.get("LUMEN_CERT_REVOCATIONS")
        if cert_pointer and cert_ledger:
            try:
                revoke_active_certificate(cert_pointer, cert_ledger, decision.sender_id,
                                           f"command-key compromise: {decision.reason}")
                self._inc("mtls_revocations")
                logger.critical("[MTLS REVOCATION] identity=%s quarantined; router ACL reload required",
                                sanitize(decision.sender_id, 64))
            except KeyError:
                # A command signer may not have a same-named TLS identity.
                pass
            except Exception:
                self._inc("mtls_revocation_errors")
                logger.critical("[MTLS REVOCATION] ledger/pointer update failed; lifecycle remains fail-closed",
                                exc_info=True)
        logger.critical("[REKEY] revoked signer=%s key_id=%s why=%s",
                        sanitize(decision.sender_id, 64), sanitize(decision.key_id, 64),
                        sanitize(decision.reason, 100))
        self.enforcement.execute_mitigation(
            "Key Compromise Suspected", target, 99.0,
            f"signer={decision.sender_id} key_id={decision.key_id} {decision.reason}",
            critical=True, action="REKEY")

    def _handle_actuator(self, target: str, raw: bytes) -> None:
        try:
            payload_str = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            self._reject_actuator(target, "MALFORMED_PACKET (utf8)", raw)
            return

        with self._nonce_lock:
            last_nonce = self.last_seen_nonces.get(target, 0)
            try:
                ok, reason = CommandSigner.verify_command(
                    target, payload_str, last_seen_nonce=last_nonce
                )
            except Exception as exc:
                ok, reason = False, f"CRYPTO_FAILURE ({type(exc).__name__})"
            if ok:
                try:
                    self.last_seen_nonces[target] = int(json.loads(payload_str)["nonce"])
                except (ValueError, KeyError, TypeError):
                    ok, reason = False, "MALFORMED_PACKET (nonce)"

        if not ok:
            self._reject_actuator(target, reason, raw)
            return

        if self.key_monitor is not None:
            self.key_monitor.on_accepted(target, raw)
        self._inc("actuator_verified")
        try:
            self.session.put(target, raw)
            self._inc("actuator_forwarded")
        except Exception as exc:
            self._inc("publish_errors")
            logger.error("[GATEWAY PUBLISH] actuator target=%s error=%s", sanitize(target, 80), sanitize(exc))

    def _queue_device_block_alert(self, device_id: str, target_topic: str, decision, detail: str) -> None:
        event = (device_id, target_topic, decision.remaining_ttl,
                 decision.attack_type or "Device Policy Violation", detail)
        try:
            self.device_alert_queue.put_nowait(event)
        except Full:
            self._inc("device_alert_queue_full")
            logger.error("[DEVICE BLOCK] quarantine active but alert queue full for %s", sanitize(device_id, 64))

    def _register_device_violation(self, device_id: str, target_topic: str,
                                   attack_type: str, detail: str) -> None:
        decision = self.device_security.register_violation(
            device_id, attack_type=attack_type, now=time.monotonic()
        )
        logger.warning(
            "[DEVICE VIOLATION] device=%s target=%s attack=%s reason=%s strikes=%d",
            sanitize(device_id, 64), sanitize(target_topic, 80),
            sanitize(attack_type, 60), sanitize(detail, 100), decision.strike_count,
        )
        if decision.block_issued:
            self._inc("device_blocks_issued")
            self._queue_device_block_alert(device_id, target_topic, decision, detail)
            # Repeated device compromise evidence quarantines its mTLS identity.
            # Deliberately do not mint a replacement automatically for a device
            # that may still be under attacker control; an operator can re-enrol it.
            cert_pointer = os.environ.get("LUMEN_CERT_POINTER")
            cert_ledger = os.environ.get("LUMEN_CERT_REVOCATIONS")
            if cert_pointer and cert_ledger and os.environ.get("LUMEN_MTLS_QUARANTINE_ON_DEVICE_BLOCK", "1") == "1":
                try:
                    revoke_active_certificate(cert_pointer, cert_ledger, device_id,
                                              f"device quarantined: {attack_type}: {detail}",
                                              quarantine_only=True)
                    self._inc("mtls_device_quarantines")
                    logger.critical("[MTLS DEVICE QUARANTINE] device=%s router ACL reload required",
                                    sanitize(device_id, 64))
                except KeyError:
                    # Some registry IDs may be local aliases without a dedicated certificate.
                    pass
                except Exception:
                    self._inc("mtls_device_quarantine_errors")
                    logger.critical("[MTLS DEVICE QUARANTINE] failed; session remains fail-closed",
                                    exc_info=True)

    def _publish_device_block_alert(self, event) -> None:
        device_id, target_topic, ttl, attack_type, detail = event
        payload = {
            "timestamp": time.time(),
            "topic": target_topic,
            "entity_type": "device",
            "entity_id": f"device:{device_id}",
            "device_id": device_id,
            "attack": attack_type,
            "action": "BLOCK",
            "loss": 99.0,
            "detail": sanitize(detail, 120),
            "severity": "critical",
            "repeats": 0,
            "block_ttl": round(float(ttl), 1),
            "source": "Inline Gateway",
        }
        self.session.put(ALERT_TOPIC, json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        logger.warning(
            "[DEVICE BLOCKED] device=%s topic=%s ttl=%.1fs attack=%s",
            sanitize(device_id, 64), sanitize(target_topic, 80), float(ttl), sanitize(attack_type, 60),
        )

    def _on_ingest(self, sample) -> None:
        try:
            self._process_ingest(sample)
        except Exception:
            self._inc("callback_errors")
            logger.exception("[GATEWAY CALLBACK] ingest callback failed")

    def _process_ingest(self, sample) -> None:
        try:
            if sample.kind == zenoh.SampleKind.DELETE:
                return
        except AttributeError:
            pass

        key = str(sample.key_expr)
        if not key.startswith(INGEST_PREFIX):
            return
        self._inc("ingress_received")
        now = time.monotonic()

        if key.startswith("ingest/actuators/"):
            target = "home/" + key[len(INGEST_PREFIX):]
            if target not in ALLOWED_COMMAND_TOPICS:
                self._inc("unknown_dropped")
                if self.unknown_limiter.allow("unknown", now):
                    self.enforcement.execute_mitigation(
                        "Reconnaissance / Scanning", sanitize(target, 80),
                        99.0, "INVALID_ACTUATOR_NAMESPACE",
                    )
                return
            payload = sample.payload
            try:
                reported_size = len(payload)
            except (TypeError, AttributeError):
                reported_size = None
            if reported_size is not None and reported_size > MAX_PAYLOAD_BYTES:
                self._inc("oversize_dropped")
                self._reject_actuator(target, f"MALFORMED_PACKET (oversize {reported_size}B)")
                return
            raw = payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)
            if len(raw) > MAX_PAYLOAD_BYTES:
                self._inc("oversize_dropped")
                self._reject_actuator(target, f"MALFORMED_PACKET (oversize {len(raw)}B)")
                return
            self._handle_actuator(target, raw)
            return

        route = self.registry.resolve_ingress(key)
        if route is None:
            self._inc("device_unknown_dropped")
            self._inc("unknown_dropped")
            if self.unknown_limiter.allow("unknown", now):
                self.enforcement.execute_mitigation(
                    "Reconnaissance / Scanning", sanitize(key, 80), 99.0,
                    "UNKNOWN_OR_UNREGISTERED_DEVICE_INGRESS",
                )
            return

        spec, target = route
        if target not in ALLOWED_SENSOR_TOPICS:
            self._inc("device_unknown_dropped")
            return

        decision = self.device_security.check_rate(spec.device_id, now)
        if not decision.allowed:
            if decision.reason == "device_blocked":
                self._inc("device_blocklist_dropped")
            else:
                self._inc("device_rate_dropped")
            if decision.block_issued:
                self._inc("device_blocks_issued")
                self._queue_device_block_alert(
                    spec.device_id, target, decision,
                    "per-device ingress rate repeatedly exceeded",
                )
            return

        payload = sample.payload
        try:
            reported_size = len(payload)
        except (TypeError, AttributeError):
            reported_size = None
        if reported_size is not None and reported_size > MAX_SENSOR_FRAME_BYTES:
            self._inc("oversize_dropped")
            self._register_device_violation(
                spec.device_id, target, "Anomalous Payload", f"oversize {reported_size}B"
            )
            return
        raw = payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)
        if len(raw) > MAX_SENSOR_FRAME_BYTES:
            self._inc("oversize_dropped")
            self._register_device_violation(
                spec.device_id, target, "Anomalous Payload", f"oversize {len(raw)}B"
            )
            return

        if not self.topic_rate_limiter.allow(target, now):
            self._inc("rate_topic_dropped")
            return
        if not self.global_rate_limiter.allow("all-sensors", now):
            self._inc("rate_global_dropped")
            return

        try:
            self.ingest_queue.put_nowait((target, raw, now, spec.device_id))
            self._inc("sensor_queued")
        except Full:
            self._inc("sensor_queue_full_dropped")

    def _process_sensor(self, item) -> None:
        target, raw, received_mono, device_id = item
        age = max(0.0, time.monotonic() - received_mono)
        if age > float(os.environ.get("LUMEN_MAX_INGEST_QUEUE_AGE_SEC", "2.0")):
            self._inc("sensor_queue_stale_dropped")
            return
        if self.device_security.is_blocked(device_id):
            self._inc("device_blocklist_dropped")
            return
        if self.is_shadow_quarantined(device_id):
            self._route_to_shadow_sandbox(device_id, target, raw)
            return

        identity = load_active_device_identity(self.cert_pointer, device_id)
        spec = self.registry.get(device_id)
        if spec is None or identity.common_name != spec.cert_cn and not identity.common_name.startswith(f"{device_id}-g"):
            raise ValueError("active certificate identity is not bound to the registered sensor")
        ctx = ProvenanceContext(
            session_id=self.run_id, device_cn=identity.common_name,
            cert_fingerprint=identity.certificate_fingerprint, source_type="live_validated",
            device_id=device_id, generation=identity.generation,
        )
        # Detect V6 framed telemetry without changing the existing V5 scalar path.
        # Malformed V6 frames are incident-captured by the Gate and are never
        # forwarded as model input.
        v6_frame = None
        is_v6 = False
        try:
            frame_doc = json.loads(raw.decode("utf-8"))
            # Any versioned JSON frame is handled by the strict V6 parser, even
            # when the version field itself is unsupported or mistyped. Do not
            # let malformed protocol envelopes fall back to the V5 scalar path.
            is_v6 = isinstance(frame_doc, dict) and "protocol_version" in frame_doc
        except (UnicodeError, json.JSONDecodeError):
            # The only supported scalar wire payload is not JSON. A malformed
            # JSON object is therefore treated as a malformed protocol frame
            # and incident-captured by the V6 parser rather than downgraded to
            # the legacy V5 scalar path.
            is_v6 = raw.lstrip().startswith(b"{")
        if is_v6:
            parsed_frame = parse_sensor_frame(target, raw, expected_device_id=device_id)
            if not parsed_frame.valid or parsed_frame.frame is None:
                admission = self.admission_gate.evaluate_v6_and_ingest(ctx, target, raw)
                self._inc("sensor_invalid_dropped")
                logger.warning("[V6 FRAME REJECTED] device=%s reason=%s", sanitize(device_id, 64), admission.reason)
                # Forward only a small signed rejection marker. The original
                # untrusted bytes are already persisted in incident storage;
                # IDS needs the structured reason, not another parser input.
                envelope_payload = b"V6_PROTOCOL_FRAME_REJECTED"
            else:
                v6_frame = parsed_frame.frame
                admission = self.admission_gate.evaluate_v6_and_ingest(ctx, target, raw)
                envelope_payload = v6_frame.canonical_text.encode("utf-8")
        else:
            # AdmissionGate persists admitted rows or rejected evidence synchronously.
            # Any queue/SQLite failure propagates to the worker callback and fails closed.
            admission = self.admission_gate.evaluate_and_ingest(ctx, target, raw)
            envelope_payload = raw
        if not admission.admitted:
            self._inc("sensor_invalid_dropped")

        # Internal envelope publication is Gateway-only by ACL. V6 adds an
        # RSA-PSS signature from the active Gateway leaf key, verified by IDS.
        envelope_args = dict(
            device_id=device_id, generation=identity.generation,
            certificate_common_name=identity.common_name,
            certificate_fingerprint=identity.certificate_fingerprint,
            original_topic=target, payload=envelope_payload, gateway_instance_id=self.run_id,
            admission_status="ADMITTED" if admission.admitted else "REJECTED",
            admission_reason=admission.reason,
            admission_numeric_value=admission.numeric_value,
        )
        if is_v6:
            gateway_identity, gateway_signing_key = load_active_device_signing_key(self.cert_pointer, "gateway")
            envelope_args.update(
                protocol_version=6,
                gateway_identity=gateway_identity,
                gateway_signing_key=gateway_signing_key,
            )
            if v6_frame is not None:
                envelope_args.update(
                    boot_id=v6_frame.boot_id,
                    sequence=v6_frame.sequence,
                    device_timestamp_ns=v6_frame.device_timestamp_ns,
                )
        envelope = IdentityEnvelope.create(**envelope_args)
        self.session.put(f"{INTERNAL_TELEMETRY_PREFIX}{device_id}", envelope.encode())

        # Public topics receive only canonical values accepted by AdmissionGate.
        if admission.admitted:
            canonical_raw = str(admission.canonical_text).encode("utf-8")
            self.session.put(target, canonical_raw)
            self._inc("sensor_forwarded")
            with self._stats_lock:
                self.device_publish_accepted[device_id] = self.device_publish_accepted.get(device_id, 0) + 1

    def _worker_loop(self) -> None:
        while self.running or not self.ingest_queue.empty() or not self.device_alert_queue.empty():
            try:
                alert_event = self.device_alert_queue.get_nowait()
            except Empty:
                alert_event = None
            if alert_event is not None:
                try:
                    self._publish_device_block_alert(alert_event)
                except Exception:
                    self._inc("publish_errors")
                    logger.exception("[DEVICE BLOCK ALERT] publish failed")
                finally:
                    self.device_alert_queue.task_done()
                continue
            try:
                item = self.ingest_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._process_sensor(item)
            except Exception:
                self._inc("callback_errors")
                logger.exception("[GATEWAY WORKER] sensor processing failed")
            finally:
                self.ingest_queue.task_done()

    def _metrics_snapshot(self) -> dict:
        with self._stats_lock:
            counters = dict(self.stats)
            counters["device_publish_accepted"] = dict(self.device_publish_accepted)
        with self._shadow_lock:
            shadow_list = sorted(self._shadow_quarantined)
        counters.update({
            "run_id": self.run_id,
            "timestamp": time.time(),
            "sensor_queue_depth": self.ingest_queue.qsize(),
            "actuator_queue_depth": 0,
            "mitigation_queue_depth": self.device_alert_queue.qsize(),
            "blocks_active": len(self.enforcement.get_active_blocks()),
            "blocks_issued": self.enforcement.blocks_issued,
            "alerts_suppressed": self.enforcement.suppressed,
            "critical_alerts_suppressed": self.enforcement.suppressed_critical,
            "device_blocks_active": self.device_security.active_block_count(),
            "device_blocks_issued_total": self.device_security.blocks_issued,
            "device_blocks": self.device_security.active_blocks(),
            "shadow_quarantined_devices": shadow_list,
            "shadow_quarantined_count": len(shadow_list),
            "forensic_sink_stored": self.forensic_sink.stored_count,
            "forensic_sink_dropped_rate": self.forensic_sink.dropped_rate,
            "forensic_sink_dropped_quota": self.forensic_sink.dropped_quota,
        })
        return counters

    def _write_metrics(self) -> None:
        snapshot = self._metrics_snapshot()
        try:
            with open(self.metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + "\n")
                handle.flush()
        except Exception:
            logger.exception("[GATEWAY METRICS] unable to append %s", self.metrics_path)

    def _log_stats(self) -> None:
        snap = self._metrics_snapshot()
        logger.info(
            "[GATEWAY METRICS] run_id=%s ingress=%d forwarded=%d shadow_fwd=%d "
            "shadow_active=%d device_rate_drop=%d device_block_drop=%d queue=%d",
            self.run_id,
            snap["ingress_received"],
            snap["sensor_forwarded"],
            snap["shadow_forwarded"],
            snap["shadow_quarantined_count"],
            snap["device_rate_dropped"],
            snap["device_blocklist_dropped"],
            snap["sensor_queue_depth"],
        )

    def stop(self) -> None:
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
            self.running = False
            if self.sub is not None:
                try:
                    self.sub.undeclare()
                except Exception:
                    logger.exception("[GATEWAY SHUTDOWN] ingest subscriber undeclare failed")
                self.sub = None
            if self.policy_sub is not None:
                try:
                    self.policy_sub.undeclare()
                except Exception:
                    logger.exception("[GATEWAY SHUTDOWN] policy subscriber undeclare failed")
                self.policy_sub = None
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=10.0)
                if self.worker_thread.is_alive():
                    logger.error("[GATEWAY SHUTDOWN] worker did not drain within 10 seconds")
            self._write_metrics()
            try:
                self.storage.flush_all()
                self.storage.close()
            except Exception:
                logger.exception("[GATEWAY SHUTDOWN] storage close failed")
            try:
                self.session.close()
            except Exception:
                logger.exception("[GATEWAY SHUTDOWN] Zenoh session close failed")

    def start(self) -> None:
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="lumen-gateway-worker", daemon=True
        )
        self.worker_thread.start()
        try:
            self.sub = self.session.declare_subscriber("ingest/**", self._on_ingest)
            self.policy_sub = self.session.declare_subscriber(
                POLICY_CONTROL_TOPIC, self._on_policy_command
            )
            logger.info(
                "[GATEWAY READY] run_id=%s listening=ingest/** policy=%s device_aware=true pep=true",
                self.run_id,
                POLICY_CONTROL_TOPIC,
            )
            self._write_metrics()
            while self.running:
                time.sleep(METRICS_INTERVAL_SEC)
                self._write_metrics()
                self._log_stats()
        except KeyboardInterrupt:
            logger.info("[GATEWAY] Keyboard interrupt")
        except Exception:
            logger.exception("[GATEWAY] fatal runtime error")
            raise
        finally:
            self.stop()


if __name__ == "__main__":
    try:
        ZenohInlineGateway().start()
    except Exception as exc:
        logger.critical("[GATEWAY IDS] startup failed; refusing to run: %s", sanitize(exc, 160))
        raise SystemExit(1) from exc