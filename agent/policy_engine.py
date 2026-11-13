"""Deterministic v5.0 risk policy with correlated PEP and restart-safe hysteresis."""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from storage.telemetry_db import LumenStorageManager


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
    operational_status: str  # 'HEALTHY' | 'FAULT'
    fault_type: str
    deterministic_violation: bool = False
    deterministic_reason: str = ""
    xai_summary: str = ""
    # Cryptographic revocation is a separate, explicit decision. Behavioral
    # violations (including rate/flood rules) never set this flag implicitly.
    revocation_requested: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    device_id: str
    old_state: str
    new_state: str
    action: str
    reason_code: str
    confirmed_by_pep: bool


class RiskPolicyEngine:
    """Separates behavioral quarantine, maintenance FAULT, and PKI revocation."""

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

        self._states: dict[str, str] = {}
        self._anomaly_hist: dict[str, deque[int]] = {}
        self._clean_streak: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._restore_state()

    def _restore_state(self) -> None:
        loader = getattr(self.storage, "load_policy_states", None)
        if not callable(loader):
            return
        restored = loader()
        # The hash-chained device_registry state is committed with every state
        # transition. It is authoritative if a process stopped between that
        # commit and the following hysteresis snapshot write.
        state_loader = getattr(self.storage, "load_device_state_rows", None)
        registry_states = state_loader() if callable(state_loader) else {}
        for device_id, row in restored.items():
            try:
                generation = int(row["generation"])
                state = str(row["current_state"])
                history = [int(v) for v in row.get("anomaly_history", [])][-self.window_n:]
                clean_streak = max(0, int(row.get("clean_streak", 0)))
                registry_row = registry_states.get(device_id)
                if registry_row and int(registry_row["generation"]) == generation:
                    state = str(registry_row["current_state"])
                if generation < 1 or any(v not in (0, 1) for v in history):
                    continue
                self._generations[device_id] = generation
                self._states[device_id] = state
                self._anomaly_hist[device_id] = deque(history, maxlen=self.window_n)
                self._clean_streak[device_id] = clean_streak
            except (TypeError, ValueError, KeyError):
                continue
        # Registry-only entries (for example, a verified certificate revocation
        # during the last instruction before a crash) must also be restored.
        for device_id, row in registry_states.items():
            if device_id not in self._states:
                self._generations[device_id] = int(row["generation"])
                self._states[device_id] = str(row["current_state"])
                self._anomaly_hist[device_id] = deque(maxlen=self.window_n)
                self._clean_streak[device_id] = 0

    def _persist(self, device_id: str, generation: int) -> None:
        saver = getattr(self.storage, "save_policy_state", None)
        if callable(saver):
            hist = list(self._anomaly_hist.get(device_id, ()))
            saver(
                device_id=device_id,
                generation=generation,
                current_state=self.get_state(device_id),
                anomaly_history=hist,
                clean_streak=self._clean_streak.get(device_id, 0),
            )
            self._generations[device_id] = generation

    def get_state(self, device_id: str) -> str:
        return self._states.get(device_id, "ACTIVE")

    def _apply_pep(self, device_id: str, enable: bool, decision_id: str) -> bool:
        if self.pep_request_fn is not None:
            return bool(self.pep_request_fn(device_id, enable, decision_id))
        if self.pep_apply_fn is None:
            return False
        return bool(self.pep_apply_fn(device_id, enable))

    def _decision(self, dec_id, dev, old, new, action, reason, confirmed, ev):
        self._states[dev] = new
        self._persist(dev, ev.generation)
        return PolicyDecision(dec_id, dev, old, new, action, reason, confirmed)

    def evaluate_evidence(self, ev: DetectionEvidence) -> PolicyDecision:
        dev = ev.device_id
        curr = self.get_state(dev)
        # A state from a different active certificate generation is not
        # silently inherited by the new identity generation.
        known_generation = self._generations.get(dev)
        if known_generation is not None and known_generation != ev.generation:
            curr = "ACTIVE"
            self._states[dev] = curr
            self._anomaly_hist[dev] = deque(maxlen=self.window_n)
            self._clean_streak[dev] = 0
        hist = self._anomaly_hist.setdefault(dev, deque(maxlen=self.window_n))

        # Revocation only happens on an explicit cryptographic request and only
        # after the helper returns AND the ledger-verification callback succeeds.
        if ev.revocation_requested:
            if curr == "REVOKED":
                return self._decision("noop", dev, curr, curr, "NONE", "ALREADY_REVOKED", True, ev)
            dec_id = f"dec-{uuid.uuid4().hex[:10]}"
            reason = ev.deterministic_reason or "EXPLICIT_CERTIFICATE_REVOCATION"
            self.storage.record_enforcement_decision(
                dec_id, dev, ev.generation, curr, "REVOKED", reason, self.policy_version,
            )
            verified = False
            if self.revocation_fn is not None:
                try:
                    verified = bool(self.revocation_fn(dev, reason))
                except Exception:
                    verified = False
            self.storage.confirm_enforcement_pep(dec_id, verified, detail="revocation_ledger_verified" if verified else "revocation_not_verified")
            target = "REVOKED" if verified else "ENFORCEMENT_FAILED"
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, target, reason, 1.0, self.policy_version,
            )
            return self._decision(dec_id, dev, curr, target, "REVOKE_DETERMINISTIC", reason, verified, ev)

        if curr == "REVOKED":
            return self._decision("noop", dev, curr, curr, "NONE", "REVOKED_TERMINAL", True, ev)

        # Technical FAULT has its own clean-window recovery. It is not a cyber
        # quarantine, and a healthy stream must be observed before ACTIVE.
        if ev.operational_status == "FAULT":
            hist.clear()
            self._clean_streak[dev] = 0
            if curr != "FAULT":
                self.storage.append_quarantine_transition(
                    dev, ev.generation, curr, "FAULT", ev.fault_type, 0.0, self.policy_version
                )
            return self._decision("fault", dev, curr, "FAULT", "ALERT_FAULT", ev.fault_type, True, ev)

        # Safety-critical devices require corroboration except a deterministic
        # protocol/ingress violation, which counts as a strong behavioral signal.
        crit = self.criticality.get(dev, "normal")
        if ev.deterministic_violation:
            strong_signal = True
        elif crit == "safety_critical":
            strong_signal = (ev.lstm_anomalous and ev.cusum_triggered) or (
                ev.lstm_score > ev.lstm_threshold * 1.6
            )
        else:
            strong_signal = ev.lstm_anomalous or ev.cusum_triggered

        hist.append(1 if strong_signal else 0)
        self._clean_streak[dev] = 0 if strong_signal else self._clean_streak.get(dev, 0) + 1
        hits = sum(hist)

        if curr == "FAULT":
            if self._clean_streak[dev] >= self.fault_recovery_clean_windows:
                self.storage.append_quarantine_transition(
                    dev, ev.generation, "FAULT", "ACTIVE", "FAULT_CLEAN_WINDOW_RECOVERY",
                    0.0, self.policy_version,
                )
                hist.clear()
                self._clean_streak[dev] = 0
                return self._decision("fault-recovery", dev, "FAULT", "ACTIVE", "RECOVER_FAULT", "FAULT_CLEAN_WINDOW_RECOVERY", True, ev)
            if strong_signal:
                self.storage.append_quarantine_transition(
                    dev, ev.generation, "FAULT", "MONITORING", "BEHAVIORAL_ANOMALY_DURING_FAULT",
                    ev.lstm_score, self.policy_version,
                )
                return self._decision("fault-monitor", dev, "FAULT", "MONITORING", "MONITOR", "BEHAVIORAL_ANOMALY_DURING_FAULT", True, ev)
            self._persist(dev, ev.generation)
            return PolicyDecision("noop", dev, "FAULT", "FAULT", "NONE", "FAULT_RECOVERY_OBSERVING_CLEAN_WINDOWS", True)

        if curr in ("ACTIVE", "FAULT") and hits >= 1:
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "MONITORING", "BEHAVIORAL_ANOMALY_OBSERVED",
                ev.lstm_score, self.policy_version,
            )
            return self._decision("mon", dev, curr, "MONITORING", "MONITOR", "BEHAVIORAL_ANOMALY_OBSERVED", True, ev)

        if curr in ("ACTIVE", "MONITORING") and hits >= self.trigger_k:
            dec_id = f"dec-{uuid.uuid4().hex[:10]}"
            reason = (ev.deterministic_reason or "DETERMINISTIC_BEHAVIORAL_VIOLATION") if ev.deterministic_violation else f"HYBRID_ANOMALY_{hits}_OF_{self.window_n}"
            self.storage.record_enforcement_decision(
                dec_id, dev, ev.generation, curr, "SHADOW_QUARANTINE", reason, self.policy_version
            )
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "ENFORCEMENT_PENDING", reason, ev.lstm_score, self.policy_version
            )
            self._states[dev] = "ENFORCEMENT_PENDING"
            self._persist(dev, ev.generation)
            pep_ok = self._apply_pep(dev, True, dec_id)
            self.storage.confirm_enforcement_pep(
                dec_id, pep_ok, detail=ev.xai_summary[:120] if pep_ok else "pep_ack_missing_or_invalid"
            )
            final_state = "SHADOW_QUARANTINE" if pep_ok else "ENFORCEMENT_FAILED"
            self.storage.append_quarantine_transition(
                dev, ev.generation, "ENFORCEMENT_PENDING", final_state,
                "PEP_ACK_CONFIRMED" if pep_ok else "PEP_ACK_FAILED_OR_TIMEOUT", ev.lstm_score, self.policy_version
            )
            return self._decision(dec_id, dev, curr, final_state, "QUARANTINE_SHADOW", reason, pep_ok, ev)

        if curr == "SHADOW_QUARANTINE" and self._clean_streak.get(dev, 0) >= self.recovery_clean_windows:
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "RECOVERY_PENDING", "CLEAN_WINDOW_STREAK", 0.0, self.policy_version
            )
            return self._decision("rec", dev, curr, "RECOVERY_PENDING", "MONITOR", "CLEAN_WINDOW_STREAK", True, ev)

        if curr == "RECOVERY_PENDING" and self._clean_streak.get(dev, 0) >= self.recovery_clean_windows + 2:
            dec_id = f"dec-{uuid.uuid4().hex[:10]}"
            self.storage.record_enforcement_decision(
                dec_id, dev, ev.generation, curr, "ACTIVE", "RECOVERY_VERIFIED", self.policy_version
            )
            pep_ok = self._apply_pep(dev, False, dec_id)
            self.storage.confirm_enforcement_pep(dec_id, pep_ok, detail="quarantine_released" if pep_ok else "release_ack_missing_or_invalid")
            # A release failure is not a neutral enforcement failure: IDS must
            # retain the security quarantine and require a fresh clean window
            # plus a new two-phase PEP handshake before trying again.
            final_state = "ACTIVE" if pep_ok else "SHADOW_QUARANTINE"
            final_reason = "RECOVERY_VERIFIED" if pep_ok else "RELEASE_ACK_FAILED_QUARANTINE_RETAINED"
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, final_state, final_reason, 0.0, self.policy_version
            )
            if pep_ok:
                hist.clear()
            else:
                self._clean_streak[dev] = 0
                hist.clear()
            return self._decision(
                dec_id, dev, curr, final_state,
                "RELEASE" if pep_ok else "RELEASE_FAILED_QUARANTINED",
                final_reason, pep_ok, ev,
            )

        self._persist(dev, ev.generation)
        return PolicyDecision("noop", dev, curr, curr, "NONE", "STABLE", True)
