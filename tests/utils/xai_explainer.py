"""Etapa E6: Explainable AI (XAI) calibrat pe reziduuri normalizate cu semn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.features_v5 import FEATURE_NAMES

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "log_delta_t": "Interval inter-sosire (delta_t)",
    "rolling_rate": "Rată locală de transmisie",
    "jitter_mad": "Jitter temporal (MAD)",
    "dt_autocorr_lag1": "Autocorelație temporală lag-1",
    "burstiness_fano": "Burstiness (Factor Fano)",
    "payload_len": "Dimensiune payload",
    "payload_entropy": "Entropie structurală payload",
    "value_norm": "Valoare fizică senzor",
    "d_value_dt": "Viteză de variație (derivată)",
    "trend_slope": "Pantă cumulativă (trend)",
    "adc_fractional_residual": "Reziduu cuantizare ADC",
    "protocol_anomaly_flag": "Comportament control-plane / protocol",
}


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    description: str
    contribution_pct: float
    normalized_sq_err: float
    signed_residual: float
    direction: str  # 'HIGH' | 'LOW'


@dataclass(frozen=True)
class XaiDiagnosticReport:
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    top_attributions: list[FeatureAttribution]
    summary_text: str


class NormalizedXaiExplainer:
    """Calculează contribuția normalizată a fiecărei trăsături la eroarea LSTM."""

    def __init__(self, xai_error_scales: Sequence[float], feature_names: Sequence[str] = FEATURE_NAMES):
        self.scales = np.maximum(np.asarray(xai_error_scales, dtype=np.float32), 1e-3)
        self.feature_names = tuple(feature_names)

    def explain(
        self,
        x_last_scaled: np.ndarray,
        recon_last_scaled: np.ndarray,
        threshold: float,
        top_k: int = 3,
    ) -> XaiDiagnosticReport:
        signed = (x_last_scaled - recon_last_scaled).astype(np.float32)
        norm_err = np.abs(signed) / self.scales
        score = float(np.mean(norm_err))
        sq_weights = norm_err ** 2
        total_err = float(np.sum(sq_weights)) + 1e-9

        pcts = (sq_weights / total_err) * 100.0
        order = np.argsort(norm_err)[::-1]

        attributions: list[FeatureAttribution] = []
        for idx in order[:top_k]:
            fname = self.feature_names[int(idx)]
            s_val = float(signed[int(idx)])
            attributions.append(
                FeatureAttribution(
                    feature=fname,
                    description=FEATURE_DESCRIPTIONS.get(fname, fname),
                    contribution_pct=round(float(pcts[int(idx)]), 2),
                    normalized_sq_err=round(float(norm_err[int(idx)]), 4),
                    signed_residual=round(s_val, 4),
                    direction="HIGH" if s_val >= 0 else "LOW",
                )
            )

        primary = attributions[0]
        secondary = attributions[1] if len(attributions) > 1 else None
        summary = (
            f"Deviație dominantă pe '{primary.feature}' ({primary.contribution_pct:.1f}%, {primary.direction})"
            + (f" urmată de '{secondary.feature}' ({secondary.contribution_pct:.1f}%)" if secondary else "")
        )

        return XaiDiagnosticReport(
            anomaly_score=round(score, 6),
            threshold=round(float(threshold), 6),
            is_anomalous=score > threshold,
            top_attributions=attributions,
            summary_text=summary,
        )