"""Etapa E6: Teste automate pentru XAI normalizat, Risk & Policy Engine (histerezis,
safety-aware, confirmare PEP, recuperare reversibilă) și Forensic Sink anti-DoS.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.telemetry_db import LumenStorageManager, DBPaths
from utils.xai_explainer import NormalizedXaiExplainer
from agent.policy_engine import RiskPolicyEngine, DetectionEvidence
from agent.forensic_sink import ForensicQuarantineSink


def test_e6_1_xai_normalized_residuals_identify_dominant_feature():
    scales = [0.2] * 12
    xai = NormalizedXaiExplainer(xai_error_scales=scales)

    x_true = np.zeros(12, dtype=np.float32)
    x_recon = np.zeros(12, dtype=np.float32)
    # Simulăm deviație mare pe value_norm (index 7) și trend_slope (index 9)
    x_true[7] = 1.8
    x_true[9] = 0.9

    report = xai.explain(x_true, x_recon, threshold=1.0, top_k=2)
    assert report.is_anomalous is True
    assert report.top_attributions[0].feature == "value_norm"
    assert report.top_attributions[0].direction == "HIGH"
    assert report.top_attributions[0].contribution_pct > 75.0
    assert report.top_attributions[1].feature == "trend_slope"


def test_e6_2_policy_engine_hysteresis_safety_and_pep_confirmation():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e6-"))
    try:
        storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_dir), enable_writers=True)
        fp = hashlib.sha256(b"gas-01").digest()
        storage.upsert_device("gas_sensor_01", 1, "gas_sensor_01-g1-abc", fp, "sensor", criticality="safety_critical")

        pep_state = {"quarantined": False}

        def fake_pep(dev_id: str, enable: bool) -> bool:
            pep_state["quarantined"] = enable
            return True

        engine = RiskPolicyEngine(
            storage=storage,
            pep_apply_fn=fake_pep,
            device_criticality={"gas_sensor_01": "safety_critical"},
            window_n=5,
            trigger_k=3,
            recovery_clean_windows=3,
        )

        # 1. Senzor safety_critical: anomalie LSTM izolată fără CUSUM -> nu blochează imediat
        d1 = engine.evaluate_evidence(
            DetectionEvidence("gas_sensor_01", 1, "home/sensors/gas/ppm", 1.2, 1.0, True, 1.0, False, "HEALTHY", "NONE")
        )
        assert d1.new_state == "ACTIVE"
        assert pep_state["quarantined"] is False

        # 2. Confirmare hibridă (LSTM + CUSUM) pe 3 ferestre -> ACTIVE -> MONITORING -> SHADOW_QUARANTINE
        for _ in range(3):
            dec = engine.evaluate_evidence(
                DetectionEvidence(
                    "gas_sensor_01", 1, "home/sensors/gas/ppm", 2.5, 1.0, True, 8.2, True,
                    "HEALTHY", "NONE", xai_summary="value_norm high"
                )
            )
        assert dec.new_state == "SHADOW_QUARANTINE"
        assert dec.confirmed_by_pep is True
        assert pep_state["quarantined"] is True

        # 3. Revenire reversibilă după ferestre curate
        for _ in range(6):
            dec_rel = engine.evaluate_evidence(
                DetectionEvidence("gas_sensor_01", 1, "home/sensors/gas/ppm", 0.3, 1.0, False, 0.5, False, "HEALTHY", "NONE")
            )
        assert dec_rel.new_state == "ACTIVE"
        assert pep_state["quarantined"] is False

        storage.flush_all()
        chain_ok, _ = storage.verify_quarantine_chain()
        assert chain_ok is True
        storage.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_e6_3_forensic_sink_enforces_rate_and_storage_quotas():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e6-sink-"))
    try:
        storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_dir), enable_writers=True)
        sink = ForensicQuarantineSink(
            storage=storage,
            max_payload_bytes=64,
            rate_per_sec=2.0,
            burst=2.0,
            max_bytes_per_device=40,
        )
        fp = hashlib.sha256(b"dev").digest()

        r1 = sink.ingest_shadow_envelope("s1", "gas_sensor_01", "gas_sensor_01-g1-a", 1, fp, "home/sensors/gas/ppm", b"x" * 30, 0.9, "TEST", now_mono=10.0)
        assert r1.accepted is True

        # Depășește cota de 40 bytes -> respins (anti-DoS)
        r2 = sink.ingest_shadow_envelope("s1", "gas_sensor_01", "gas_sensor_01-g1-a", 1, fp, "home/sensors/gas/ppm", b"x" * 30, 0.9, "TEST", now_mono=10.1)
        assert r2.accepted is False
        assert r2.reason == "SINK_STORAGE_QUOTA_EXCEEDED"
        storage.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)