"""Etapa E4: Detectoare statistice complementare (Two-Sided CUSUM + EWMA)
și clasificator determinist pentru defecțiuni tehnice (FAULT vs ATTACK).
"""
from __future__ import annotations

from dataclasses import dataclass

from collections import deque


@dataclass
class CusumResult:
    score: float
    pos_cusum: float
    neg_cusum: float
    ewma_z: float
    triggered: bool
    direction: str  # 'UPWARD_DRIFT' | 'DOWNWARD_DRIFT' | 'NOMINAL'


class TwoSidedCusumDetector:
    """Detector Page-Hinkley / Two-Sided CUSUM per (device_id, topic).

    Prinde deplasările lente și persistente (Slow-Ramp Value Spoofing)
    pe care un autoencoder singur le poate reconstrui parțial.
    """

    def __init__(self, k_allowance: float = 0.5, h_threshold: float = 6.0, ewma_alpha: float = 0.15):
        self.k = k_allowance
        self.h = h_threshold
        self.alpha = ewma_alpha
        self._baselines: dict[tuple[str, str], tuple[float, float]] = {}
        self._s_pos: dict[tuple[str, str], float] = {}
        self._s_neg: dict[tuple[str, str], float] = {}
        self._ewma: dict[tuple[str, str], float] = {}

    def calibrate_stream(self, device_id: str, topic: str, mean: float, std: float) -> None:
        self._baselines[(device_id, topic)] = (float(mean), max(float(std), 1e-3))

    def update(self, device_id: str, topic: str, value: float) -> CusumResult:
        key = (device_id, topic)
        mean, std = self._baselines.get(key, (float(value), 1.0))
        z = (float(value) - mean) / std

        s_pos = max(0.0, self._s_pos.get(key, 0.0) + z - self.k)
        s_neg = max(0.0, self._s_neg.get(key, 0.0) - z - self.k)
        ewma_prev = self._ewma.get(key, 0.0)
        ewma_curr = self.alpha * z + (1.0 - self.alpha) * ewma_prev

        self._s_pos[key] = s_pos
        self._s_neg[key] = s_neg
        self._ewma[key] = ewma_curr

        peak = max(s_pos, s_neg)
        triggered = peak > self.h
        if s_pos > self.h:
            direction = "UPWARD_DRIFT"
        elif s_neg > self.h:
            direction = "DOWNWARD_DRIFT"
        else:
            direction = "NOMINAL"

        return CusumResult(
            score=round(peak, 4),
            pos_cusum=round(s_pos, 4),
            neg_cusum=round(s_neg, 4),
            ewma_z=round(ewma_curr, 4),
            triggered=triggered,
            direction=direction,
        )


@dataclass(frozen=True)
class OperationalHealthAssessment:
    status: str  # 'HEALTHY' | 'FAULT'
    fault_type: str  # 'NONE' | 'FROZEN_SENSOR' | 'PACKET_DROP_DEGRADATION'
    detail: str


class OperationalFaultClassifier:
    """Separă defecțiunile tehnice (FAULT -> alertă de mentenanță, fără carantină)
    de atacurile cibernetice.
    """

    def __init__(self, frozen_min_steps: int = 8, drop_dt_multiplier: float = 1.8, drop_window: int = 10, drop_min_hits: int = 3):
        self.frozen_min_steps = frozen_min_steps
        self.drop_dt_multiplier = drop_dt_multiplier
        self.drop_window = drop_window
        self.drop_min_hits = drop_min_hits
        self._same_val_streak: dict[tuple[str, str], int] = {}
        self._last_val: dict[tuple[str, str], float] = {}
        self._drop_history: dict[tuple[str, str], deque[int]] = {}

    def evaluate(
        self,
        device_id: str,
        topic: str,
        raw_value: float,
        delta_t: float,
        nominal_dt: float = 1.0,
        operation: str = "put",
        entropy_normal: bool = True,
    ) -> OperationalHealthAssessment:
        if operation != "put" or not entropy_normal:
            return OperationalHealthAssessment("HEALTHY", "NONE", "Non-operational signature")

        key = (device_id, topic)

        # 1. Senzor înghețat (varianță 0 pe senzori analogici continui)
        is_continuous = ("gas" in topic) or ("vibration" in topic)
        if is_continuous:
            prev = self._last_val.get(key)
            if prev is not None and abs(raw_value - prev) < 1e-6:
                self._same_val_streak[key] = self._same_val_streak.get(key, 1) + 1
            else:
                self._same_val_streak[key] = 1
            self._last_val[key] = raw_value

            if self._same_val_streak[key] >= self.frozen_min_steps:
                return OperationalHealthAssessment(
                    status="FAULT",
                    fault_type="FROZEN_SENSOR",
                    detail=f"Zero variance over {self._same_val_streak[key]} consecutive samples (val={raw_value})",
                )

        # 2. Packet Drop / Link Degradation (cel puțin 3 goluri în ultimele 10 recepții)
        hist = self._drop_history.setdefault(key, deque(maxlen=self.drop_window))
        is_gap = 1 if delta_t >= nominal_dt * self.drop_dt_multiplier else 0
        hist.append(is_gap)

        if sum(hist) >= self.drop_min_hits:
            return OperationalHealthAssessment(
                status="FAULT",
                fault_type="PACKET_DROP_DEGRADATION",
                detail=f"Packet loss detected ({sum(hist)}/{len(hist)} gaps, last delta_t={delta_t:.2f}s)",
            )

        return OperationalHealthAssessment("HEALTHY", "NONE", "Nominal")