"""Etapa E6: Risk & Policy Engine Determinist cu Mașină de Stări, Histerezis,
Reguli Safety-Aware și Confirmare Obligatorie PEP.
"""
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


@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    device_id: str
    old_state: str
    new_state: str
    action: str  # 'NONE' | 'ALERT_FAULT' | 'MONITOR' | 'QUARANTINE_SHADOW' | 'REVOKE_DETERMINISTIC' | 'RELEASE'
    reason_code: str
    confirmed_by_pep: bool


class RiskPolicyEngine:
    """Separă detecția probabilistică de decizia deterministă și enforcement-ul PEP."""

    def __init__(
        self,
        storage: LumenStorageManager,
        pep_apply_fn: Callable[[str, bool], bool],
        device_criticality: Optional[dict[str, str]] = None,
        window_n: int = 5,
        trigger_k: int = 3,
        recovery_clean_windows: int = 8,
        policy_version: int = 1,
    ):
        self.storage = storage
        self.pep_apply_fn = pep_apply_fn
        self.criticality = device_criticality or {}
        self.window_n = window_n
        self.trigger_k = trigger_k
        self.recovery_clean_windows = recovery_clean_windows
        self.policy_version = policy_version

        self._states: dict[str, str] = {}
        self._anomaly_hist: dict[str, deque[int]] = {}
        self._clean_streak: dict[str, int] = {}

    def get_state(self, device_id: str) -> str:
        return self._states.get(device_id, "ACTIVE")

    def evaluate_evidence(self, ev: DetectionEvidence) -> PolicyDecision:
        dev = ev.device_id
        curr = self.get_state(dev)
        hist = self._anomaly_hist.setdefault(dev, deque(maxlen=self.window_n))

        # 1. Revocare ireversibilă EXCLUSIV pentru încălcări deterministe (semnătură/replay/ACL)
        if ev.deterministic_violation:
            if curr != "REVOKED":
                dec_id = f"dec-{uuid.uuid4().hex[:10]}"
                self.storage.record_enforcement_decision(
                    dec_id, dev, ev.generation, curr, "REVOKED",
                    ev.deterministic_reason or "DETERMINISTIC_VIOLATION", self.policy_version,
                )
                ok = self.pep_apply_fn(dev, True)
                self.storage.confirm_enforcement_pep(dec_id, ok, detail="deterministic_revocation")
                target = "REVOKED" if ok else "ENFORCEMENT_FAILED"
                self.storage.append_quarantine_transition(
                    dev, ev.generation, curr, target,
                    ev.deterministic_reason or "DETERMINISTIC_VIOLATION",
                    1.0, self.policy_version,
                )
                self._states[dev] = target
                return PolicyDecision(dec_id, dev, curr, target, "REVOKE_DETERMINISTIC", ev.deterministic_reason, ok)
            return PolicyDecision("noop", dev, curr, curr, "NONE", "ALREADY_REVOKED", True)

        if curr == "REVOKED":
            return PolicyDecision("noop", dev, curr, curr, "NONE", "REVOKED_TERMINAL", True)

        # 2. Defecțiune tehnică dovedită (FAULT -> alertă de mentenanță, NICIODATĂ carantină cibernetică)
        if ev.operational_status == "FAULT":
            hist.clear()
            self._clean_streak[dev] = 0
            if curr != "FAULT":
                self.storage.append_quarantine_transition(
                    dev, ev.generation, curr, "FAULT", ev.fault_type, 0.0, self.policy_version
                )
                self._states[dev] = "FAULT"
            return PolicyDecision("fault", dev, curr, "FAULT", "ALERT_FAULT", ev.fault_type, True)

        # 3. Evaluare risc comportamental (LSTM + CUSUM)
        crit = self.criticality.get(dev, "normal")
        if crit == "safety_critical":
            # Regula safety-aware: senzorii critici (ex. gaz) cer confirmare hibridă (LSTM + CUSUM)
            # pentru a nu orbi sistemul la o singură deviație izolată
            strong_signal = (ev.lstm_anomalous and ev.cusum_triggered) or (
                ev.lstm_score > ev.lstm_threshold * 1.6
            )
        else:
            strong_signal = ev.lstm_anomalous or ev.cusum_triggered

        hist.append(1 if strong_signal else 0)
        if strong_signal:
            self._clean_streak[dev] = 0
        else:
            self._clean_streak[dev] = self._clean_streak.get(dev, 0) + 1

        hits = sum(hist)

        # Tranziție ACTIVE/FAULT -> MONITORING la prima anomalie
        if curr in ("ACTIVE", "FAULT") and hits >= 1:
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "MONITORING", "BEHAVIORAL_ANOMALY_OBSERVED",
                ev.lstm_score, self.policy_version,
            )
            self._states[dev] = "MONITORING"
            return PolicyDecision("mon", dev, curr, "MONITORING", "MONITOR", "BEHAVIORAL_ANOMALY_OBSERVED", True)

        # Escaladare MONITORING -> ENFORCEMENT_PENDING -> SHADOW_QUARANTINE la K din N ferestre
        if curr in ("ACTIVE", "MONITORING") and hits >= self.trigger_k:
            dec_id = f"dec-{uuid.uuid4().hex[:10]}"
            reason = f"HYBRID_ANOMALY_{hits}_OF_{self.window_n}"
            self.storage.record_enforcement_decision(
                dec_id, dev, ev.generation, curr, "SHADOW_QUARANTINE", reason, self.policy_version
            )
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "ENFORCEMENT_PENDING", reason, ev.lstm_score, self.policy_version
            )
            pep_ok = self.pep_apply_fn(dev, True)
            self.storage.confirm_enforcement_pep(
                dec_id, pep_ok, detail=ev.xai_summary[:120] if pep_ok else "pep_apply_failed"
            )
            final_state = "SHADOW_QUARANTINE" if pep_ok else "ENFORCEMENT_FAILED"
            self.storage.append_quarantine_transition(
                dev, ev.generation, "ENFORCEMENT_PENDING", final_state,
                "PEP_CONFIRMED" if pep_ok else "PEP_FAILED", ev.lstm_score, self.policy_version
            )
            self._states[dev] = final_state
            return PolicyDecision(dec_id, dev, curr, final_state, "QUARANTINE_SHADOW", reason, pep_ok)

        # Recuperare reversibilă cu histerezis: SHADOW_QUARANTINE -> RECOVERY_PENDING -> ACTIVE
        if curr == "SHADOW_QUARANTINE" and self._clean_streak.get(dev, 0) >= self.recovery_clean_windows:
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, "RECOVERY_PENDING", "CLEAN_WINDOW_STREAK", 0.0, self.policy_version
            )
            self._states[dev] = "RECOVERY_PENDING"
            return PolicyDecision("rec", dev, curr, "RECOVERY_PENDING", "MONITOR", "CLEAN_WINDOW_STREAK", True)

        if curr == "RECOVERY_PENDING" and self._clean_streak.get(dev, 0) >= self.recovery_clean_windows + 2:
            dec_id = f"dec-{uuid.uuid4().hex[:10]}"
            self.storage.record_enforcement_decision(
                dec_id, dev, ev.generation, curr, "ACTIVE", "RECOVERY_VERIFIED", self.policy_version
            )
            pep_ok = self.pep_apply_fn(dev, False)
            self.storage.confirm_enforcement_pep(dec_id, pep_ok, detail="quarantine_released")
            final_state = "ACTIVE" if pep_ok else "ENFORCEMENT_FAILED"
            self.storage.append_quarantine_transition(
                dev, ev.generation, curr, final_state, "RECOVERY_VERIFIED", 0.0, self.policy_version
            )
            self._states[dev] = final_state
            hist.clear()
            return PolicyDecision(dec_id, dev, curr, final_state, "RELEASE", "RECOVERY_VERIFIED", pep_ok)

        return PolicyDecision("noop", dev, curr, curr, "NONE", "STABLE", True)