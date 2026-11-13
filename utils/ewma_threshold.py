import math
import threading

class EwmaThreshold:
    """
    Calculează pragul dinamic bazat pe Exponentially Weighted Moving Average (EWMA).
    Previne poisoning-ul prin înghețarea actualizărilor în timpul anomaliilor.
    """
    def __init__(self, alpha: float = 0.02, k: float = 3.5, warmup: int = 50, max_drift: float = 3.0):
        self.alpha = alpha
        self.k = k
        self.warmup = warmup
        self.max_drift = max_drift
        
        self.mu = 0.0
        self.var = 0.0
        self.n = 0
        self.mu0 = None

    def update_and_check(self, loss_val: float) -> tuple[bool, float]:
        self.n += 1
        
        if self.n == 1:
            self.mu = loss_val
            self.mu0 = loss_val
            self.var = 1e-4
            return False, self.mu + 0.1

        std_dev = math.sqrt(max(self.var, 1e-6))
        dynamic_threshold = self.mu + (self.k * std_dev)

        is_anomalous = (self.n > self.warmup) and (loss_val > dynamic_threshold)

        # Anti-poisoning guardrail: Actualizăm media doar dacă pachetul este normal
        # și dacă media nu a suferit o derivă masivă (drift cap)
        if not is_anomalous and (self.mu <= self.mu0 * self.max_drift):
            delta = loss_val - self.mu
            self.mu += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * (delta ** 2))

        return is_anomalous, dynamic_threshold


class EwmaThresholdManager:
    """
    Gestionează pragurile EWMA independente per topic.
    """
    def __init__(self):
        self._trackers = {}
        self._lock = threading.Lock()

    def check(self, topic: str, loss_val: float, fallback_threshold: float) -> tuple[bool, float]:
        with self._lock:
            if topic not in self._trackers:
                self._trackers[topic] = EwmaThreshold()
            
            is_anom, calculated_thr = self._trackers[topic].update_and_check(loss_val)
            
            # Folosim pragul fallback în perioada de warmup
            effective_thr = max(calculated_thr, fallback_threshold * 0.8) if self._trackers[topic].n > 50 else fallback_threshold
            final_anom = loss_val > effective_thr
            
            return final_anom, effective_thr