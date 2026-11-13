import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import random
import numpy as np
import onnxruntime as ort

from training.feature_extractor import load_scaler, scale_window, last_step_loss, rule_flood
from training.traffic_sim import legit_events, attack_events, windows_from_events, ATTACK_KINDS
from utils.logger import get_script_logger

logger = get_script_logger("evaluate_model")


def build_eval_stream(seed=4242, duration=1800):
    rng = random.Random(seed)
    ev = legit_events(duration, rng)
    for t0, kind in ((300, "flood"), (600, "value_spoof"), (900, "payload_inject"),
                     (1200, "flapping"), (1500, "flood")):
        ev += attack_events(kind, float(t0), rng)
    return ev


def _confusion(pred, y):
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y)); fn = int(np.sum(~pred & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tp, fp, tn, fn, p, r, f1, fpr


def run_evaluation():
    logger.info("==================================================")
    logger.info(" 🧪 EVALUARE ONNX LSTM - ML pur vs. hibrid (ML + regulă de rată)")
    logger.info("==================================================")

    meta_path, model_path = "models/saved/model_metadata.json", "models/saved/autoencoder.onnx"
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        logger.error("[EROARE] Modelul ONNX sau metadatele nu există. Antrenați mai întâi modelul!")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    mn, inv = load_scaler(meta)
    seq_len = int(meta["seq_len"])
    g_thr = float(meta["threshold"])
    t_thr = meta.get("topic_thresholds", {})

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    raw, topics, last_kind, any_attack = windows_from_events(build_eval_stream(), seq_len)
    y = np.array([k != "legit" for k in last_kind])
    keep = ~((~y) & any_attack)    # exclude ferestrele legitime "de recuperare" (conțin încă pachete de atac)

    ml_pred = np.zeros(len(raw), dtype=bool)
    rule_pred = np.zeros(len(raw), dtype=bool)
    lat = []
    for i in range(len(raw)):
        scaled = scale_window(raw[i], mn, inv)
        t0 = time.perf_counter_ns()
        out = sess.run(None, {in_name: scaled[None]})[0][0]
        lat.append(time.perf_counter_ns() - t0)
        ml_pred[i] = last_step_loss(scaled, out) > t_thr.get(topics[i], g_thr)
        rule_pred[i] = rule_flood(float(raw[i][-1][0]), float(raw[i][-1][1]))
    hybrid = ml_pred | rule_pred

    for name, pred in (("ML pur (fără euristici)", ml_pred), ("Hibrid (ML OR regulă rată)", hybrid)):
        tp, fp, tn, fn, p, r, f1, fpr = _confusion(pred[keep], y[keep])
        logger.info(f" [{name}] ferestre={int(keep.sum())} TP={tp} FP={fp} TN={tn} FN={fn}")
        logger.info(f"    Precision={p*100:.2f}%  Recall={r*100:.2f}%  F1={f1*100:.2f}%  FPR={fpr*100:.2f}%")

    logger.info("--------------------------------------------------")
    logger.info(" Recall per tip de atac (ML pur | hibrid):")
    kinds = np.array(last_kind)
    for k in ATTACK_KINDS:
        m = kinds == k
        if m.any():
            logger.info(f"   {k:16s} n={int(m.sum()):4d}  ML={ml_pred[m].mean()*100:6.2f}%  Hibrid={hybrid[m].mean()*100:6.2f}%")
    logger.info(" NOTĂ: Replay/Tampering/Scan sunt decise deterministic (crypto, allow-list, honeypot) și")
    logger.info("       testate de attacks/attack_2_replay.py și attack_3_scan.py - nu sunt sarcină ML.")
    logger.info(f" ⚡ Latență medie ONNX: {np.mean(lat) / 1e6:.3f} ms / fereastră")
    logger.info("==================================================")


if __name__ == "__main__":
    run_evaluation()