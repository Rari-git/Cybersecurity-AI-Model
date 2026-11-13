"""Etapa E4: Feature Engineering v5.0 (12 trăsături per (device_id, topic)).

Organizare pe 4 familii:
- Temporal: log_delta_t, rolling_rate, jitter_mad, dt_autocorr_lag1, burstiness_fano
- Traffic: payload_len, payload_entropy
- Semantic: value_norm, d_value_dt, trend_slope, adc_fractional_residual
- Protocol: protocol_anomaly_flag
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

FEATURE_SCHEMA_VERSION = "v5.0-f12"
FEATURE_NAMES: tuple[str, ...] = (
    "log_delta_t",
    "rolling_rate",
    "jitter_mad",
    "dt_autocorr_lag1",
    "burstiness_fano",
    "payload_len",
    "payload_entropy",
    "value_norm",
    "d_value_dt",
    "trend_slope",
    "adc_fractional_residual",
    "protocol_anomaly_flag",
)
FEATURE_DIM = len(FEATURE_NAMES)


def shannon_entropy(raw: bytes) -> float:
    if not raw:
        return 0.0
    counts: dict[int, int] = {}
    for b in raw:
        counts[b] = counts.get(b, 0) + 1
    n = float(len(raw))
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def parse_sensor_scalar(topic: str, raw: bytes) -> float:
    """Extrage valoarea numerică canonică pentru calculul derivatelor și pantei."""
    text = raw.decode("utf-8", errors="ignore").strip()
    if "|" in text:
        text = text.split("|", 1)[0].strip()
    upper = text.upper()
    if upper in ("CLOSED", "NO_MOTION", "IDLE", "OFF"):
        return 0.0
    if upper in ("OPEN", "MOTION_DETECTED", "ON"):
        return 1.0
    try:
        val = float(text)
        if math.isfinite(val):
            return val
    except ValueError:
        pass
    return 0.0


@dataclass
class StreamStepStats:
    device_id: str
    topic: str
    features: np.ndarray  # shape (12,), float32
    raw_value: float
    delta_t: float
    payload_entropy: float
    operation: str
    value_std_window: float
    dt_ratio_window: float


class DeviceTopicFeatureExtractor:
    """Extractor stateful per (device_id, topic) care menține istoricul necesar
    pentru ferestre glisante, derivată, trend slope, autocorelație și ADC residual.
    """

    def __init__(self, seq_len: int = 16, stat_window: int = 16):
        self.seq_len = seq_len
        self.stat_window = stat_window
        self._dt_hist: dict[tuple[str, str], deque[float]] = {}
        self._val_hist: dict[tuple[str, str], deque[float]] = {}
        self._step_buf: dict[tuple[str, str], deque[np.ndarray]] = {}
        self._seen_topics_by_device: dict[str, set[str]] = {}

    def reset_stream(self, device_id: str, topic: str) -> None:
        key = (device_id, topic)
        self._dt_hist.pop(key, None)
        self._val_hist.pop(key, None)
        self._step_buf.pop(key, None)

    def update(
        self,
        device_id: str,
        topic: str,
        delta_t_sec: float,
        raw_payload: bytes,
        operation: str = "put",
        numeric_value: Optional[float] = None,
    ) -> StreamStepStats:
        key = (device_id, topic)
        dt_q = self._dt_hist.setdefault(key, deque(maxlen=self.stat_window))
        val_q = self._val_hist.setdefault(key, deque(maxlen=self.stat_window))
        win_q = self._step_buf.setdefault(key, deque(maxlen=self.seq_len))
        topics_seen = self._seen_topics_by_device.setdefault(device_id, set())
        topics_seen.add(topic)

        dt = max(0.01, min(60.0, float(delta_t_sec)))
        val = float(numeric_value) if numeric_value is not None else parse_sensor_scalar(topic, raw_payload)
        plen = float(len(raw_payload))
        pent = shannon_entropy(raw_payload)

        prev_val = val_q[-1] if val_q else val
        d_val_dt = (val - prev_val) / max(dt, 0.05)

        dt_q.append(dt)
        val_q.append(val)

        dt_arr = np.asarray(dt_q, dtype=np.float64)
        val_arr = np.asarray(val_q, dtype=np.float64)

        # 1. Temporal family
        log_dt = math.log1p(dt)
        recent_dts = dt_arr[-5:]
        rolling_rate = float(len(recent_dts) / max(float(np.sum(recent_dts)), 0.05))
        med_dt = float(np.median(dt_arr))
        jitter_mad = float(np.median(np.abs(dt_arr - med_dt)))

        if len(dt_arr) >= 4:
            x0 = dt_arr[:-1] - np.mean(dt_arr[:-1])
            x1 = dt_arr[1:] - np.mean(dt_arr[1:])
            denom_ac = float(np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2)))
            dt_autocorr = float(np.sum(x0 * x1) / denom_ac) if denom_ac > 1e-9 else 0.0
        else:
            dt_autocorr = 0.0

        mean_dt = max(float(np.mean(dt_arr)), 1e-4)
        burstiness_fano = float(np.var(dt_arr) / mean_dt)

        # 2. Semantic family (slope + ADC quantization check)
        if len(val_arr) >= 3:
            x = np.arange(len(val_arr), dtype=np.float64)
            x_centered = x - np.mean(x)
            denom = float(np.sum(x_centered ** 2))
            trend_slope = float(np.sum(x_centered * (val_arr - np.mean(val_arr))) / denom) if denom > 0 else 0.0
        else:
            trend_slope = 0.0

        # Senzorii de gaz produc valori cuantizate întregi în producție;
        # zgomotul gaussian sintetic introduce fracțiuni nenule constante.
        if "gas" in topic:
            adc_residual = abs(val - round(val))
        elif "vibration" in topic:
            scaled = val * 1000.0
            adc_residual = abs(scaled - round(scaled))
        else:
            adc_residual = 0.0

        # 3. Protocol family
        proto_flag = 0.0
        if operation != "put":
            proto_flag += 0.6
        if topic.startswith(("home/admin", "home/security/keys", "system/", "shadow/")):
            proto_flag += 0.4
        if len(topics_seen) > 1:
            proto_flag += min(0.5, 0.25 * (len(topics_seen) - 1))

        vec = np.array(
            [
                log_dt,
                rolling_rate,
                jitter_mad,
                dt_autocorr,
                burstiness_fano,
                plen,
                pent,
                val,
                d_val_dt,
                trend_slope,
                adc_residual,
                proto_flag,
            ],
            dtype=np.float32,
        )
        win_q.append(vec)

        val_std = float(np.std(val_arr)) if len(val_arr) >= 6 else 1.0
        dt_ratio = float(dt / max(med_dt, 0.05))

        return StreamStepStats(
            device_id=device_id,
            topic=topic,
            features=vec,
            raw_value=val,
            delta_t=dt,
            payload_entropy=pent,
            operation=operation,
            value_std_window=val_std,
            dt_ratio_window=dt_ratio,
        )

    def get_window(self, device_id: str, topic: str) -> Optional[np.ndarray]:
        """Returnează matricea (seq_len, 12) float32 dacă fereastra este completă."""
        buf = self._step_buf.get((device_id, topic))
        if buf is None or len(buf) < self.seq_len:
            return None
        return np.stack(list(buf), axis=0).astype(np.float32)


class RobustFeatureScaler:
    """Scaler robust (mediană / IQR) calibrat exclusiv pe baseline-ul normal."""

    def __init__(self, medians: np.ndarray, iqrs: np.ndarray):
        self.medians = np.asarray(medians, dtype=np.float32)
        self.iqrs = np.maximum(np.asarray(iqrs, dtype=np.float32), 1e-3)

    @classmethod
    def fit(cls, samples_2d: np.ndarray) -> "RobustFeatureScaler":
        med = np.median(samples_2d, axis=0)
        q75 = np.percentile(samples_2d, 75, axis=0)
        q25 = np.percentile(samples_2d, 25, axis=0)
        raw_iqr = q75 - q25
        # Prag minim per trăsătură pentru a preveni amplificarea artificială a trăsăturilor constante în baseline
        min_scales = np.full(samples_2d.shape[1], 0.25, dtype=np.float32)
        min_scales[10] = 0.10  # adc_fractional_residual
        min_scales[11] = 0.15  # protocol_anomaly_flag
        iqr = np.maximum(raw_iqr, min_scales)
        return cls(medians=med, iqrs=iqr)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.medians) / self.iqrs).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "medians": [round(float(v), 6) for v in self.medians],
            "iqrs": [round(float(v), 6) for v in self.iqrs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobustFeatureScaler":
        return cls(
            medians=np.asarray(data["medians"], dtype=np.float32),
            iqrs=np.asarray(data["iqrs"], dtype=np.float32),
        )