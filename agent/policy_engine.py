"""V6 orthogonal operational/security policy with V5 PEP compatibility.

Operational health is diagnostic/maintenance state. Security state is the
independent trust/enforcement state. A hardware FAULT must never overwrite a
security quarantine, clear replay evidence, or suppress a protocol violation.
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from storage.telemetry_db import LumenStorageManager


_OPERATIONAL_STATES = frozenset({"HEALTHY", "DEGRADED", "FAULT"})
_SECURITY_STATUS_BY_PHASE = {
    "ACTIVE": "ACTIVE",
    "MONITORING": "SUSPECT",
    "ENFORCEMENT_PENDING": "QUARANTINED",
    "SHADOW_QUARANTINE": "QUARANTINED",
    "RECOVERY_PENDING": "QUARANTINED",
    "ENFORCEMENT_FAILED": "QUARANTINED",
    "REVOKED": "REVOKED",
}
_SECURITY_PHASE_RANK = {
    "ACTIVE": 0,
    "MONITORING": 1,
    "ENFORCEMENT_PENDING": 2,
    "ENFORCEMENT_FAILED": 3,
    "SHADOW_QUARANTINE": 4,
    "RECOVERY_PENDING": 5,
    "REVOKED": 6,
}


@dataclass(frozen=True)
class DetectionEvidence:
    device_id: str
    generation: int
    topic: str
    lstm_score: float
    lstm_threshold: float
    lstm_anomalous: bool
    cusum_score: float
    cusum_triggered: bool
    operational_status: str  # 'HEALTHY' | 'DEGRADED' | 'FAULT'
    fault_type: str
    deterministic_violation: bool = False
    deterministic_reason: str = ""
    xai_summary: str = ""
    # Cryptographic revocation is a separate, explicit decision.
    revocation_requested: bool = False
    # V6 protocol evidence is admitted to the policy layer without being fed to
    # feature extraction/ONNX. A freshness violation is an immediate signal.
    protocol_version: int = 5
    boot_id: str | None = None
    sequence: int | None = None
    device_timestamp_ns: int | None = None
    protocol_violation: bool = False
    admission_status: str = "ADMITTED"


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    device_id: str
    old_state: str
    new_state: str
    action: str
    reason_code: str
    confirmed_by_pep: bool
    operational_status: str = "HEALTHY"
    security_status: str = "ACTIVE"


class RiskPolicyEngine:
    """Separate operational health from security enforcement and retain V5 PEP semantics.

    ``get_state`` and PolicyDecision.old_state/new_state remain compatibility
    views for V5 callers. New integrations should use get_operational_status()
    and get_security_status().
    """

    def __init__(
        self,
        storage: LumenStorageManager,
        pep_apply_fn: Optional[Callable[[str, bool], bool]] = None,
        device_criticality: Optional[dict[str, str]] = None,
        window_n: int = 5,
        trigger_k: int = 3,
        recovery_clean_windows: int = 8,
        policy_version: int = 1,
        *,
        pep_request_fn: Optional[Callable[[str, bool, str], bool]] = None,
        revocation_fn: Optional[Callable[[str, str], bool]] = None,
        fault_recovery_clean_windows: int = 8,
    ):
        if window_n < 1 or not 1 <= trigger_k <= window_n:
            raise ValueError("trigger_k must be within 1..window_n")
        if recovery_clean_windows < 1 or fault_recovery_clean_windows < 1:
            raise ValueError("recovery windows must be positive")
        self.storage = storage
        self.pep_apply_fn = pep_apply_fn
        self.pep_request_fn = pep_request_fn
        self.revocation_fn = revocation_fn
        self.criticality = device_criticality or {}
        self.window_n = window_n
        self.trigger_k = trigger_k
        self.recovery_clean_windows = recovery_clean_windows
        self.fault_recovery_clean_windows = fault_recovery_clean_windows
        self.policy_version = policy_version

        # Security phase preserves V5's detailed PEP lifecycle. It maps to the
        # V6 public security states ACTIVE/SUSPECT/QUARANTINED/REVOKED.
        self._security_phases: dict[str, str] = {}
        self._operational_states: dict[str, str] = {}
        self._security_anomaly_hist: dict[str, deque[int]] = {}
        self._security_clean_streak: dict[str, int] = {}
        self._operational_clean_streak: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._security_transition_history: dict[str, list[str]] = {}
        self._operational_transition_history: dict[str, list[str]] = {}

        # Private compatibility aliases used by existing V5 regression helpers.
        self._states = self._security_phases
        self._anomaly_hist = self._security_anomaly_hist
        self._clean_streak = self._security_clean_streak
        self._restore_state()

    @staticmethod
    def _status_for_phase(phase: str) -> str:
        return _SECURITY_STATUS_BY_PHASE.get(phase, "QUARANTINED")

    def _restore_state(self) -> None:
        dimension_loader = getattr(self.storage, "load_policy_dimensions", None)
        dimensions = dimension_loader() if callable(dimension_loader) else {}
        legacy_loader = getattr(self.storage, "load_policy_states", None)
        legacy = legacy_loader() if callable(legacy_loader) else {}
        registry_loader = getattr(self.storage, "load_device_state_rows", None)
        registry = registry_loader() if callable(registry_loader) else {}

        # Restore the V6 dimensions first.
        for device_id, row in dimensions.items():
            try:
                generation = int(row["generation"])
                phase = str(row.get("security_phase", "ACTIVE"))
                operational = str(row.get("operational_status", "HEALTHY"))
                history = [int(v) for v in row.get("security_anomaly_history", [])][-self.window_n:]
                if (generation < 1 or phase not in _SECURITY_STATUS_BY_PHASE
                        or operational not in _OPERATIONAL_STATES or any(v not in (0, 1) for v in history)):
                    continue
                self._generations[device_id] = generation
                self._security_phases[device_id] = phase
                self._operational_states[device_id] = operational
                self._security_anomaly_hist[device_id] = deque(history, maxlen=self.window_n)
                self._security_clean_streak[device_id] = max(0, int(row.get("security_clean_streak", 0)))
                self._operational_clean_streak[device_id] = max(0, int(row.get("operational_clean_streak", 0)))
                self._security_transition_history[device_id] = [
                    str(v) for v in row.get("security_transition_history", [])
                ][-64:]
                self._operational_transition_history[device_id] = [
                    str(v) for v in row.get("operational_transition_history", [])
                ][-64:]
            except (TypeError, ValueError, KeyError):
                continue

        # Upgrade legacy V5 rows. FAULT is migrated to the operational dimension;
        # all other legacy values remain the detailed security phase.
        for device_id, row in legacy.items():
            if device_id in self._security_phases:
                continue
            try:
                generation = int(row["generation"])
                state = str(row["current_state"])
                phase = "ACTIVE" if state == "FAULT" else state
                operational = "FAULT" if state == "FAULT" else "HEALTHY"
                history = [int(v) for v in row.get("anomaly_history", [])][-self.window_n:]
                if generation < 1 or phase not in _SECURITY_STATUS_BY_PHASE or any(v not in (0, 1) for v in history):
                    continue
                self._generations[device_id] = generation
                self._security_phases[device_id] = phase
                self._operational_states[device_id] = operational
                self._security_anomaly_hist[device_id] = deque(history, maxlen=self.window_n)
                self._security_clean_streak[device_id] = max(0, int(row.get("clean_streak", 0)))
                self._operational_clean_streak[device_id] = 0
            except (TypeError, ValueError, KeyError):
                continue

        # device_registry remains the V5 authoritative security phase. It must
        # never override an independent V6 operational status.
        for device_id, row in registry.items():
            try:
                generation = int(row["generation"])
                registry_phase = str(row["current_state"])
                if device_id not in self._security_phases:
                    if registry_phase == "FAULT":
                        phase, operational = "ACTIVE", "FAULT"
                    elif registry_phase in _SECURITY_STATUS_BY_PHASE:
                        phase, operational = registry_phase, "HEALTHY"
                    else:
                        continue
                    self._generations[device_id] = generation
                    self._security_phases[device_id] = phase
                    self._operational_states[device_id] = operational
                    self._security_anomaly_hist[device_id] = deque(maxlen=self.window_n)
                    self._security_clean_streak[device_id] = 0
                    self._operational_clean_streak[device_id] = 0
                    continue
                if self._generations.get(device_id) != generation:
                    continue
                if registry_phase in _SECURITY_STATUS_BY_PHASE:
                    restored_phase = self._security_phases[device_id]
                    if _SECURITY_PHASE_RANK[registry_phase] > _SECURITY_PHASE_RANK[restored_phase]:
                        self._security_phases[device_id] = registry_phase
            except (TypeError, ValueError, KeyError):
                continue

    def get_operational_status(self, device_id: str) -> str:
        return self._operational_states.get(device_id, "HEALTHY")

    def get_security_status(self, device_id: str) -> str:
        return self._status_for_phase(self._security_phases.get(device_id, "ACTIVE"))

    def get_security_phase(self, device_id: str) -> str:
        return self._security_phases.get(device_id, "ACTIVE")

    def get_state(self, device_id: str) -> str:
        """Return the historical V5 state view without conflating stored dimensions."""
        phase = self.get_security_phase(device_id)
        if phase in {"ENFORCEMENT_PENDING", "SHADOW_QUARANTINE", "RECOVERY_PENDING", "ENFORCEMENT_FAILED", "REVOKED"}:
            return phase
        operational = self.get_operational_status(device_id)
        if operational == "FAULT":
            return "FAULT"
        if operational == "DEGRADED":
            return "DEGRADED"
        if phase == "MONITORING":
            return "MONITORING"
        return "ACTIVE"

    def _persist(self, device_id: str, generation: int) -> None:
        phase = self.get_security_phase(device_id)
        operational = self.get_operational_status(device_id)
        security_history = list(self._security_transition_history.get(device_id, ()))
        operational_history = list(self._operational_transition_history.get(device_id, ()))
        saver = getattr(self.storage, "save_policy_dimensions", None)
        if callable(saver):
            saver(
                device_id=device_id,
                generation=generation,
                security_phase=phase,
                security_status=self.get_security_status(device_id),
                operational_status=operational,
                security_anomaly_history=list(self._security_anomaly_hist.get(device_id, ())),
                security_clean_streak=self._security_clean_streak.get(device_id, 0),
                operational_clean_streak=self._operational_clean_streak.get(device_id, 0),
                security_transition_history=security_history,
                operational_transition_history=operational_history,
            )
        else:
            # Lightweight V5 storage doubles still receive the old snapshot.
            legacy_saver = getattr(self.storage, "save_policy_state", None)
            if callable(legacy_saver):
                legacy_saver(
                    device_id=device_id, generation=generation, current_state=phase,
                    anomaly_history=list(self._security_anomaly_hist.get(device_id, ())),
                    clean_streak=self._security_clean_streak.get(device_id, 0),
                )
        self._generations[device_id] = generation

    def _record_dimension_transition(
        self, device_id: str, generation: int, dimension: str,
        old_state: str, new_state: str, reason: str,
    ) -> None:
        if old_state == new_state:
            return
        history_map = (self._security_transition_history if dimension == "security"
                       else self._operational_transition_history)
        history = history_map.setdefault(device_id, [])
        history.append(f"{old_state}->{new_state}:{reason[:96]}")
        del history[:-64]
        recorder = getattr(self.storage, "append_policy_dimension_transition", None)
        if callable(recorder):
            recorder(device_id, generation, dimension, old_state, new_state, reason)

    def _set_operational_status(self, device_id: str, generation: int, new_status: str, reason: str) -> bool:
        if new_status not in _OPERATIONAL_STATES:
            raise ValueError(f"unsupported operational status: {new_status}")
        old = self.get_operational_status(device_id)
        if old == new_status:
            return False
        self._operational_states[device_id] = new_status
        self._record_dimension_transition(device_id, generation, "operational", old, new_status, reason)
        return True

    def _set_security_phase(
        self, device_id: str, generation: int, new_phase: str, reason: str,
        risk_score: float = 0.0,
    ) -> bool:
        if new_phase not in _SECURITY_STATUS_BY_PHASE:
            raise ValueError(f"unsupported security phase: {new_phase}")
        old = self.get_security_phase(device_id)
        if old == new_phase:
            return False
        # Preserve the existing V5 hash-chained quarantine log for enforcement
        # transitions, and add the independent V6 security-dimension chain.
        self.storage.append_quarantine_transition(
            device_id, generation, old, new_phase, reason, risk_score, self.policy_version,
        )
        self._security_phases[device_id] = new_phase
        self._record_dimension_transition(device_id, generation, "security", old, new_phase, reason)
        return True

    def _apply_pep(self, device_id: str, enable: bool, decision_id: str) -> bool:
        if self.pep_request_fn is not None:
            return bool(self.pep_request_fn(device_id, enable, decision_id))
        if self.pep_apply_fn is None:
            return False
        return bool(self.pep_apply_fn(device_id, enable))

    def _decision(
        self, dec_id: str, dev: str, old: str, action: str, reason: str,
        confirmed: bool, generation: int,
    ) -> PolicyDecision:
        self._persist(dev, generation)
        return PolicyDecision(
            dec_id, dev, old, self.get_state(dev), action, reason, confirmed,
            self.get_operational_status(dev), self.get_security_status(dev),
        )

    def _update_operational_dimension(self, ev: DetectionEvidence) -> tuple[bool, str]:
        dev = ev.device_id
        current = self.get_operational_status(dev)
        incoming = ev.operational_status if ev.operational_status in _OPERATIONAL_STATES else current
        changed = False
        reason = ""
        if incoming == "FAULT":
            self._operational_clean_streak[dev] = 0
            changed = self._set_operational_status(dev, ev.generation, "FAULT", ev.fault_type or "OPERATIONAL_FAULT")
            reason = ev.fault_type or "OPERATIONAL_FAULT"
        elif incoming == "DEGRADED":
            self._operational_clean_streak[dev] = 0
            changed = self._set_operational_status(dev, ev.generation, "DEGRADED", "OPERATIONAL_DEGRADED")
            reason = "OPERATIONAL_DEGRADED"
        elif incoming == "HEALTHY" and current != "HEALTHY":
            streak = self._operational_clean_streak.get(dev, 0) + 1
            self._operational_clean_streak[dev] = streak
            required = self.fault_recovery_clean_windows if current == "FAULT" else 2
            if streak >= required:
                changed = self._set_operational_status(dev, ev.generation, "HEALTHY", "OPERATIONAL_CLEAN_WINDOW_RECOVERY")
                self._operational_clean_streak[dev] = 0
                reason = "OPERATIONAL_CLEAN_WINDOW_RECOVERY"
        elif incoming == "HEALTHY":
            self._operational_clean_streak[dev] = 0
        return changed, reason

    def _evaluate_revocation(self, ev: DetectionEvidence, legacy_old: str) -> PolicyDecision:
        dev = ev.device_id
        phase = self.get_security_phase(dev)
        if phase == "REVOKED":
            return self._decision("noop", dev, legacy_old, "NONE", "ALREADY_REVOKED", True, ev.generation)
        decision_id = f"dec-{uuid.uuid4().hex[:10]}"
        reason = ev.deterministic_reason or "EXPLICIT_CERTIFICATE_REVOCATION"
        self.storage.record_enforcement_decision(
            decision_id, dev, ev.generation, phase, "REVOKED", reason, self.policy_version,
        )
        verified = False
        if self.revocation_fn is not None:
            try:
                verified = bool(self.revocation_fn(dev, reason))
            except Exception:
                verified = False
        self.storage.confirm_enforcement_pep(
            decision_id, verified,
            detail="revocation_ledger_verified" if verified else "revocation_not_verified",
        )
        target = "REVOKED" if verified else "ENFORCEMENT_FAILED"
        self._set_security_phase(dev, ev.generation, target, reason, 1.0)
        return self._decision(
            decision_id, dev, legacy_old,
            "REVOKE_DETERMINISTIC" if verified else "REVOCATION_FAILED_QUARANTINED",
            reason, verified, ev.generation,
        )

    def _enforce_quarantine(self, ev: DetectionEvidence, legacy_old: str, reason: str) -> PolicyDecision:
        dev = ev.device_id
        decision_id = f"dec-{uuid.uuid4().hex[:10]}"
        phase = self.get_security_phase(dev)
        self.storage.record_enforcement_decision(
            decision_id, dev, ev.generation, phase, "SHADOW_QUARANTINE", reason, self.policy_version,
        )
        self._set_security_phase(dev, ev.generation, "ENFORCEMENT_PENDING", reason, ev.lstm_score)
        self._persist(dev, ev.generation)
        pep_ok = self._apply_pep(dev, True, decision_id)
        self.storage.confirm_enforcement_pep(
            decision_id, pep_ok,
            detail=ev.xai_summary[:120] if pep_ok else "pep_ack_missing_or_invalid",
        )
        final_phase = "SHADOW_QUARANTINE" if pep_ok else "ENFORCEMENT_FAILED"
        self._set_security_phase(
            dev, ev.generation, final_phase,
            "PEP_ACK_CONFIRMED" if pep_ok else "PEP_ACK_FAILED_OR_TIMEOUT",
            ev.lstm_score,
        )
        return self._decision(
            decision_id, dev, legacy_old, "QUARANTINE_SHADOW", reason, pep_ok, ev.generation,
        )

    def evaluate_evidence(self, ev: DetectionEvidence) -> PolicyDecision:
        dev = ev.device_id
        if ev.generation < 1:
            raise ValueError("evidence generation must be positive")
        legacy_old = self.get_state(dev)
        known_generation = self._generations.get(dev)
        if known_generation is not None and known_generation != ev.generation:
            # Preserve V5 generation-scoped hysteresis reset semantics.
            self._security_phases[dev] = "ACTIVE"
            self._operational_states[dev] = "HEALTHY"
            self._security_anomaly_hist[dev] = deque(maxlen=self.window_n)
            self._security_clean_streak[dev] = 0
            self._operational_clean_streak[dev] = 0
            self._security_transition_history[dev] = []
            self._operational_transition_history[dev] = []
            legacy_old = "ACTIVE"

        op_changed, op_reason = self._update_operational_dimension(ev)
        operational_now = self.get_operational_status(dev)

        if ev.revocation_requested:
            decision = self._evaluate_revocation(ev, legacy_old)
            return decision

        phase = self.get_security_phase(dev)
        if phase == "REVOKED":
            return self._decision("noop", dev, legacy_old, "NONE", "REVOKED_TERMINAL", True, ev.generation)

        # A technical fault is not a security signal. Crucially, it does not
        # clear anomaly history or short-circuit replay/protocol evidence.
        criticality = self.criticality.get(dev, "normal")
        if ev.protocol_violation:
            strong_signal = True
        elif ev.deterministic_violation:
            strong_signal = True
        elif operational_now == "FAULT":
            # Model/statistical evidence from a known-bad physical stream is
            # not promoted on its own. Explicit protocol/freshness violations
            # above remain decisive, and any existing security quarantine is
            # retained unchanged while the device is faulted.
            strong_signal = False
        elif criticality == "safety_critical":
            strong_signal = (ev.lstm_anomalous and ev.cusum_triggered) or (
                ev.lstm_score > ev.lstm_threshold * 1.6
            )
        else:
            strong_signal = ev.lstm_anomalous or ev.cusum_triggered

        hist = self._security_anomaly_hist.setdefault(dev, deque(maxlen=self.window_n))
        if strong_signal:
            hist.append(1)
            self._security_clean_streak[dev] = 0
        elif operational_now == "HEALTHY":
            # Only healthy operational windows count as clean security evidence.
            hist.append(0)
            self._security_clean_streak[dev] = self._security_clean_streak.get(dev, 0) + 1
        # DEGRADED/FAULT with no anomaly does not erase nor dilute security evidence.
        hits = sum(hist)
        security_clean = self._security_clean_streak.get(dev, 0)

        # Protocol/freshness violations are deterministic protocol failures and
        # bypass behavioral K-of-N. They are rejected by AdmissionGate before
        # feature extraction, then handled here as explicit security evidence.
        if ev.protocol_violation and phase in {
            "ACTIVE", "MONITORING", "RECOVERY_PENDING", "ENFORCEMENT_FAILED",
        }:
            # A replay during recovery invalidates the clean-window proof. Reassert
            # the PEP quarantine rather than merely resetting the streak: the
            # Gateway may already have prepared a two-phase release and must be
            # driven back to the confirmed quarantined phase.
            reason = ev.deterministic_reason or "V6_PROTOCOL_OR_FRESHNESS_VIOLATION"
            return self._enforce_quarantine(ev, legacy_old, reason)

        if phase == "ACTIVE" and strong_signal:
            self._set_security_phase(dev, ev.generation, "MONITORING", "BEHAVIORAL_ANOMALY_OBSERVED", ev.lstm_score)
            if operational_now == "FAULT":
                return self._decision("fault-monitor", dev, legacy_old, "ALERT_FAULT", ev.fault_type, True, ev.generation)
            if operational_now == "DEGRADED":
                return self._decision("degraded-monitor", dev, legacy_old, "MONITOR", "BEHAVIORAL_ANOMALY_DURING_DEGRADED_OPERATION", True, ev.generation)
            return self._decision("mon", dev, legacy_old, "MONITOR", "BEHAVIORAL_ANOMALY_OBSERVED", True, ev.generation)

        if phase in {"ACTIVE", "MONITORING"} and hits >= self.trigger_k:
            reason = (
                ev.deterministic_reason or "DETERMINISTIC_BEHAVIORAL_VIOLATION"
                if ev.deterministic_violation else f"HYBRID_ANOMALY_{hits}_OF_{self.window_n}"
            )
            return self._enforce_quarantine(ev, legacy_old, reason)

        # Recovery is gated by clean security windows and by healthy operation.
        if phase == "SHADOW_QUARANTINE" and security_clean >= self.recovery_clean_windows:
            self._set_security_phase(dev, ev.generation, "RECOVERY_PENDING", "CLEAN_WINDOW_STREAK", 0.0)
            return self._decision("rec", dev, legacy_old, "MONITOR", "CLEAN_WINDOW_STREAK", True, ev.generation)

        if phase == "RECOVERY_PENDING" and security_clean >= self.recovery_clean_windows + 2:
            decision_id = f"dec-{uuid.uuid4().hex[:10]}"
            self.storage.record_enforcement_decision(
                decision_id, dev, ev.generation, phase, "ACTIVE", "RECOVERY_VERIFIED", self.policy_version,
            )
            pep_ok = self._apply_pep(dev, False, decision_id)
            self.storage.confirm_enforcement_pep(
                decision_id, pep_ok,
                detail="quarantine_released" if pep_ok else "release_ack_missing_or_invalid",
            )
            final_phase = "ACTIVE" if pep_ok else "SHADOW_QUARANTINE"
            final_reason = "RECOVERY_VERIFIED" if pep_ok else "RELEASE_ACK_FAILED_QUARANTINE_RETAINED"
            self._set_security_phase(dev, ev.generation, final_phase, final_reason, 0.0)
            hist.clear()
            self._security_clean_streak[dev] = 0
            return self._decision(
                decision_id, dev, legacy_old,
                "RELEASE" if pep_ok else "RELEASE_FAILED_QUARANTINED",
                final_reason, pep_ok, ev.generation,
            )

        # Emit operational alerts even when security is already quarantined.
        if operational_now == "FAULT" and (op_changed or phase != "ACTIVE"):
            return self._decision("fault", dev, legacy_old, "ALERT_FAULT", ev.fault_type, True, ev.generation)
        if operational_now == "DEGRADED" and op_changed:
            return self._decision("degraded", dev, legacy_old, "ALERT_DEGRADED", op_reason, True, ev.generation)
        if op_changed and operational_now == "HEALTHY":
            return self._decision("op-recovery", dev, legacy_old, "RECOVER_OPERATIONAL", op_reason, True, ev.generation)

        return self._decision("noop", dev, legacy_old, "NONE", "STABLE", True, ev.generation)
