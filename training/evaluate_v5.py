"""Etapa E7: Harness reproductibil de evaluare comparativă pentru licență.

Compară 4 configurații pe setul held-out înghețat:
1. Deterministic Only (fără AI — doar reguli statice)
2. Statistical Only (Two-Sided CUSUM)
3. Temporal LSTM Autoencoder Only
4. Lumen v5.0 Hybrid (LSTM + CUSUM + Operational Fault Classifier + Policy Engine)

Calculează: Recall, False Positive Rate/oră, False Quarantine Rate, Detection Latency (steps),
Operational Fault Separation și Top-2 XAI Attribution Accuracy.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bundle_loader import verify_and_load_bundle
from models.model_lifecycle import resolve_active_bundle
from utils.features_v5 import DeviceTopicFeatureExtractor, RobustFeatureScaler
from agent.statistical_detectors import TwoSidedCusumDetector, OperationalFaultClassifier
from agent.policy_engine import RiskPolicyEngine, DetectionEvidence
from storage.telemetry_db import LumenStorageManager, DBPaths
from utils.xai_explainer import NormalizedXaiExplainer
from attacks.advanced_generators import iter_scenario_frames

EXPECTED_XAI_FEATURES: dict[str, set[str]] = {
    "atk1_stealth_spoof_heldout": {"value_norm", "trend_slope", "d_value_dt", "adc_fractional_residual"},
    "atk2_replay_skew_heldout": {"log_delta_t", "rolling_rate", "dt_autocorr_lag1", "protocol_anomaly_flag", "burstiness_fano"},
    "atk3_low_slow_recon_heldout": {"protocol_anomaly_flag", "log_delta_t", "payload_len", "payload_entropy"},
    "atk4_adaptive_burst_heldout": {"rolling_rate", "payload_entropy", "value_norm", "burstiness_fano", "log_delta_t"},
}


def scenario_database_dir(root: str | Path, scenario_id: str) -> Path:
    """Return a unique per-scenario directory so policy persistence cannot leak."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scenario_id)
    if not safe:
        raise ValueError("scenario_id must contain a safe path component")
    base = Path(root).resolve()
    target = (base / safe).resolve()
    if base not in target.parents:
        raise ValueError("scenario database path escapes evaluation root")
    return target


def run_benchmark(
    bundle_dir: str | Path | None = None,
    pub_key_path: str | Path = "config/model_keys/model_signing_ed25519.pub",
    output_json: str | Path = "logs/evaluation_v5_report.json",
) -> dict:
    if bundle_dir is None:
        bundle_dir = resolve_active_bundle(model_root="models", pointer_path="models/active_model.json", fallback_bundle="models/lumen_model_bundle")
    bundle = verify_and_load_bundle(bundle_dir, pub_key_path)
    ort_sess = ort.InferenceSession(str(bundle.onnx_path), providers=["CPUExecutionProvider"])
    scaler = RobustFeatureScaler.from_dict(bundle.scaler)
    seq_len = int(bundle.feature_schema["seq_len"])
    xai = NormalizedXaiExplainer(bundle.thresholds["xai_error_scales"])

    heldout_attacks = [
        "atk1_stealth_spoof_heldout",
        "atk2_replay_skew_heldout",
        "atk3_low_slow_recon_heldout",
        "atk4_adaptive_burst_heldout",
    ]
    fault_scenarios = ["fault1_frozen_sensor", "fault2_packet_drop_40"]

    results: dict[str, dict] = {
        "evaluation_version": "v5-audit-isolated-2",
        "scenario_policy_storage_isolated": True,
        "hybrid_false_quarantine_metric": "final PEP-confirmed SHADOW_QUARANTINE or RECOVERY_PENDING state",
        "deterministic_only": {"detected": 0, "total": len(heldout_attacks), "false_alerts_on_faults": 0},
        "cusum_only": {"detected": 0, "total": len(heldout_attacks), "false_alerts_on_faults": 0},
        "lstm_only": {"detected": 0, "total": len(heldout_attacks), "false_alerts_on_faults": 0},
        "lumen_v5_hybrid": {
            "detected": 0,
            "total": len(heldout_attacks),
            "false_quarantines_on_faults": 0,
            "faults_correctly_isolated_as_fault": 0,
            "xai_top2_hits": 0,
            "scenarios": {},
        },
    }

    tmp_eval_dir = Path(tempfile.mkdtemp(prefix="lumen-eval-v5-"))

    try:
        for sc_id in heldout_attacks + fault_scenarios:
            scenario_dir = scenario_database_dir(tmp_eval_dir, sc_id)
            storage = LumenStorageManager(paths=DBPaths.from_dir(scenario_dir), enable_writers=True)
            try:
                ext = DeviceTopicFeatureExtractor(seq_len=seq_len, stat_window=seq_len)
                cusum = TwoSidedCusumDetector(k_allowance=0.5, h_threshold=6.0)
                fault_clf = OperationalFaultClassifier(frozen_min_steps=8)
                policy = RiskPolicyEngine(
                    storage=storage,
                    pep_apply_fn=lambda _dev, _en: True,
                    window_n=5,
                    trigger_k=3,
                )

                # Calibrăm CUSUM din bundle; bundle CUSUM baseline comes only from
                # train/calibration sessions after the E5 leakage fix.
                for skey, stats in bundle.thresholds.get("cusum_baselines", {}).items():
                    dev, top = skey.split("::", 1)
                    cusum.calibrate_stream(dev, top, stats["mean"], stats["std"])

                det_flag = False
                cusum_flag = False
                lstm_flag = False
                hybrid_flag = False
                fault_flag = False
                scenario_devices: set[str] = set()
                first_detect_step = None
                xai_hit = False
                dt_acc = 0.0

                for frame in iter_scenario_frames(sc_id, steps=90):
                    scenario_devices.add(frame.device_id)
                    dt_acc += frame.delta_t_sec
                    if frame.dropped:
                        continue
                    dt_used = dt_acc
                    dt_acc = 0.0

                    # 1. Control determinist simplu (doar depășire limite fizice brute)
                    try:
                        val_f = float(frame.payload.decode("utf-8", errors="ignore").split("|")[0])
                        if val_f > 2000.0 or val_f < 0.0:
                            det_flag = True
                    except ValueError:
                        pass

                    st = ext.update(frame.device_id, frame.topic, dt_used, frame.payload, frame.operation)
                    c_res = cusum.update(frame.device_id, frame.topic, st.raw_value)
                    op_health = fault_clf.evaluate(
                        frame.device_id, frame.topic, st.raw_value, dt_used, operation=frame.operation
                    )

                    if c_res.triggered:
                        cusum_flag = True
                    if op_health.status == "FAULT":
                        fault_flag = True

                    win = ext.get_window(frame.device_id, frame.topic)
                    if win is not None:
                        x_scaled = scaler.transform(win)[None, :, :]
                        recon = ort_sess.run(None, {"window_input": x_scaled})[0]
                        skey = f"{frame.device_id}::{frame.topic}"
                        thr = float(bundle.thresholds["stream_thresholds"].get(skey, bundle.thresholds["global_threshold"]))
                        report = xai.explain(x_scaled[0, -1, :], recon[0, -1, :], threshold=thr, top_k=2)

                        if report.is_anomalous:
                            lstm_flag = True

                        # Ground truth/scenario identity is evaluation-only. Policy sees
                        # only the operational classifier's observable output.
                        effective_op_status = op_health.status

                        decision = policy.evaluate_evidence(
                            DetectionEvidence(
                                device_id=frame.device_id,
                                generation=1,
                                topic=frame.topic,
                                lstm_score=report.anomaly_score,
                                lstm_threshold=thr,
                                lstm_anomalous=report.is_anomalous,
                                cusum_score=float(getattr(c_res, "score", 0.0)),
                                cusum_triggered=c_res.triggered,
                                operational_status=effective_op_status,
                                fault_type=op_health.fault_type,
                                xai_summary=report.summary_text,
                            )
                        )

                        if decision.action == "QUARANTINE_SHADOW" and decision.confirmed_by_pep:
                            hybrid_flag = True
                            if first_detect_step is None:
                                first_detect_step = frame.step
                            if not xai_hit:
                                top_feats = {a.feature for a in report.top_attributions}
                                if sc_id in EXPECTED_XAI_FEATURES and (top_feats & EXPECTED_XAI_FEATURES[sc_id]):
                                    xai_hit = True

                confirmed_quarantined = any(
                    policy.get_state(device_id) in {"SHADOW_QUARANTINE", "RECOVERY_PENDING"}
                    for device_id in scenario_devices
                )
                hybrid_flag = hybrid_flag or confirmed_quarantined

                if sc_id in heldout_attacks:
                    if det_flag:
                        results["deterministic_only"]["detected"] += 1
                    if cusum_flag:
                        results["cusum_only"]["detected"] += 1
                    if lstm_flag:
                        results["lstm_only"]["detected"] += 1
                    if hybrid_flag:
                        results["lumen_v5_hybrid"]["detected"] += 1
                    if xai_hit:
                        results["lumen_v5_hybrid"]["xai_top2_hits"] += 1
                    results["lumen_v5_hybrid"]["scenarios"][sc_id] = {
                        "detected_hybrid": hybrid_flag,
                        "detection_latency_steps": first_detect_step,
                        "xai_top2_accurate": xai_hit,
                        "final_security_states": {dev: policy.get_state(dev) for dev in sorted(scenario_devices)},
                    }
                else:
                    # Test negativ: defecțiune tehnică
                    if det_flag:
                        results["deterministic_only"]["false_alerts_on_faults"] += 1
                    if cusum_flag:
                        results["cusum_only"]["false_alerts_on_faults"] += 1
                    if lstm_flag:
                        results["lstm_only"]["false_alerts_on_faults"] += 1
                    if hybrid_flag:
                        results["lumen_v5_hybrid"]["false_quarantines_on_faults"] += 1
                    if fault_flag and not hybrid_flag:
                        results["lumen_v5_hybrid"]["faults_correctly_isolated_as_fault"] += 1
                    results["lumen_v5_hybrid"]["scenarios"][sc_id] = {
                        "classified_as_fault": fault_flag,
                        "false_cyber_quarantine": hybrid_flag,
                        "final_security_states": {dev: policy.get_state(dev) for dev in sorted(scenario_devices)},
                    }

            finally:
                storage.close()

    finally:
        import shutil
        shutil.rmtree(tmp_eval_dir, ignore_errors=True)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report, indent=2))