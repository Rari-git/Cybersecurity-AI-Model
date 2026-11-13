"""Etapa E4: Teste automate pentru Feature Engineering v5 (12 trăsături),
detectorul CUSUM pe Slow-Ramp și clasificatorul de defecțiuni FAULT vs ATTACK.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.features_v5 import (
    DeviceTopicFeatureExtractor,
    RobustFeatureScaler,
    FEATURE_DIM,
    FEATURE_NAMES,
)
from agent.statistical_detectors import TwoSidedCusumDetector, OperationalFaultClassifier
from attacks.advanced_generators import iter_scenario_frames


def test_e4_1_feature_extractor_produces_12_features_and_catches_slow_ramp():
    ext = DeviceTopicFeatureExtractor(seq_len=12, stat_window=12)
    cusum = TwoSidedCusumDetector(k_allowance=0.5, h_threshold=6.0)
    cusum.calibrate_stream("gas_sensor_01", "home/sensors/gas/ppm", mean=350.0, std=8.0)

    # 1. Trafic normal (cuantizat întreg, pantă ~ 0)
    for i in range(15):
        st = ext.update("gas_sensor_01", "home/sensors/gas/ppm", 1.0, str(350 + (i % 3) - 1).encode())
        res_c = cusum.update("gas_sensor_01", "home/sensors/gas/ppm", st.raw_value)

    win_norm = ext.get_window("gas_sensor_01", "home/sensors/gas/ppm")
    assert win_norm is not None
    assert win_norm.shape == (12, FEATURE_DIM)
    assert res_c.triggered is False

    # 2. Atac 1: Stealth Slow-Ramp -> panta pozitivă + CUSUM declanșat + reziduu ADC nenul
    cusum_triggered = False
    max_slope = 0.0
    max_adc_res = 0.0
    idx_slope = FEATURE_NAMES.index("trend_slope")
    idx_adc = FEATURE_NAMES.index("adc_fractional_residual")

    for frame in iter_scenario_frames("atk1_stealth_spoof_dev", steps=25):
        st = ext.update(frame.device_id, frame.topic, frame.delta_t_sec, frame.payload, frame.operation)
        c_res = cusum.update(frame.device_id, frame.topic, st.raw_value)
        if c_res.triggered:
            cusum_triggered = True
        max_slope = max(max_slope, float(st.features[idx_slope]))
        max_adc_res = max(max_adc_res, float(st.features[idx_adc]))

    assert cusum_triggered is True
    assert max_slope > 2.0
    assert max_adc_res > 0.01


def test_e4_2_fault_classifier_separates_frozen_and_drop_from_attacks():
    clf = OperationalFaultClassifier(frozen_min_steps=8)

    # Defect 1: Senzor înghețat -> FAULT (nu atac)
    statuses = []
    for frame in iter_scenario_frames("fault1_frozen_sensor", steps=12):
        res = clf.evaluate(frame.device_id, frame.topic, float(frame.payload), frame.delta_t_sec)
        statuses.append(res.status)
    assert statuses[-1] == "FAULT"

    # Defect 2: Packet drop 40% -> FAULT la goluri consecutive
    dt_acc = 0.0
    drop_statuses = []
    for frame in iter_scenario_frames("fault2_packet_drop_40", steps=30):
        dt_acc += frame.delta_t_sec
        if frame.dropped:
            continue
        res = clf.evaluate(frame.device_id, frame.topic, float(frame.payload), dt_acc)
        drop_statuses.append(res.fault_type)
        dt_acc = 0.0
    assert "PACKET_DROP_DEGRADATION" in drop_statuses


def test_e4_3_robust_scaler_fit_and_serialization():
    rng = np.random.default_rng(42)
    X = rng.normal(loc=10.0, scale=2.0, size=(100, FEATURE_DIM)).astype(np.float32)
    scaler = RobustFeatureScaler.fit(X)
    reloaded = RobustFeatureScaler.from_dict(scaler.to_dict())
    np.testing.assert_allclose(scaler.transform(X[:5]), reloaded.transform(X[:5]), atol=1e-5)