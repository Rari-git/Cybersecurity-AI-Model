"""Etapa E5: Antrenare locală PyTorch pe desktop -> Export ONNX -> Calibrare -> Bundle Semnat.

Caracteristici:
- Citește exclusiv sesiuni aprobate din telemetry_normal.db cu ?mode=ro
- Split determinist pe sesiuni (fără leakage între ferestre glisante suprapuse)
- Calibrează pragurile și scalele reziduurilor XAI pe sesiunea de validare
- Exportă model.onnx compatibil onnxruntime și semnează manifestul cu Ed25519
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.telemetry_db import LumenStorageManager, DBPaths
from utils.features_v5 import (
    DeviceTopicFeatureExtractor,
    RobustFeatureScaler,
    FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_DIM,
)
from models.bundle_loader import generate_model_signing_keypair, sign_model_bundle


class TemporalLSTMAutoencoder(nn.Module):
    """Autoencoder secvențial LSTM pentru detecția anomaliilor comportamentale."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        enc_out, _ = self.encoder(x)
        h_last = torch.tanh(self.to_latent(enc_out[:, -1, :]))
        dec_in = torch.tanh(self.from_latent(h_last)).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output_proj(dec_out)


def _extract_windows_by_session(
    db_dir: str | Path,
    seq_len: int = 16,
) -> tuple[dict[str, list[tuple[str, str, np.ndarray]]], dict[tuple[str, str], list[float]]]:
    """Citește raw_events din telemetry_normal.db (?mode=ro) și construiește ferestrele
    grupat pe session_id pentru split fără data leakage.
    """
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=False)
    ro = storage.connect_readonly("normal")
    try:
        sessions = [
            r["session_id"]
            for r in ro.execute(
                "SELECT session_id FROM baseline_sessions WHERE approved_for_training = 1 ORDER BY started_at_ns ASC"
            ).fetchall()
        ]
        if len(sessions) < 2:
            raise RuntimeError(
                f"Need at least 2 approved baseline sessions for train/val split, found {len(sessions)}. "
                "Run: python simulation/capture_baseline.py --sessions 4 --events 400"
            )

        windows_by_session: dict[str, list[tuple[str, str, np.ndarray]]] = {s: [] for s in sessions}
        raw_values_by_stream: dict[tuple[str, str], list[float]] = {}

        for sid in sessions:
            rows = ro.execute(
                """
                SELECT device_id, topic, gateway_ts_ns, raw_payload, numeric_value
                FROM raw_events
                WHERE session_id = ?
                ORDER BY device_id ASC, topic ASC, event_id ASC
                """,
                (sid,),
            ).fetchall()

            extractor = DeviceTopicFeatureExtractor(seq_len=seq_len, stat_window=seq_len)
            last_ts: dict[tuple[str, str], int] = {}

            for r in rows:
                dev = r["device_id"]
                topic = r["topic"]
                ts_ns = int(r["gateway_ts_ns"])
                key = (dev, topic)

                if key in last_ts:
                    dt = (ts_ns - last_ts[key]) / 1_000_000_000.0
                    # Dacă datele au fost capturate în timp comprimat, folosim nominal 1.0s
                    if dt < 0.05:
                        dt = 1.0
                else:
                    dt = 1.0
                last_ts[key] = ts_ns

                num_val = float(r["numeric_value"]) if r["numeric_value"] is not None else None
                st = extractor.update(
                    device_id=dev,
                    topic=topic,
                    delta_t_sec=dt,
                    raw_payload=r["raw_payload"].encode("utf-8"),
                    operation="put",
                    numeric_value=num_val,
                )
                raw_values_by_stream.setdefault(key, []).append(st.raw_value)
                win = extractor.get_window(dev, topic)
                if win is not None:
                    windows_by_session[sid].append((dev, topic, win))

        return windows_by_session, raw_values_by_stream
    finally:
        ro.close()


def train_and_export_bundle(
    db_dir: str | Path = "logs/db",
    bundle_dir: str | Path = "models/lumen_model_bundle",
    key_dir: str | Path = "config/model_keys",
    seq_len: int = 16,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    windows_by_session, raw_values_by_stream = _extract_windows_by_session(db_dir, seq_len=seq_len)
    session_ids = list(windows_by_session.keys())
    train_sessions = session_ids[:-1]
    val_sessions = [session_ids[-1]]

    train_items = [item for sid in train_sessions for item in windows_by_session[sid]]
    val_items = [item for sid in val_sessions for item in windows_by_session[sid]]

    if not train_items or not val_items:
        raise RuntimeError("Insufficient windows extracted for training/validation.")

    X_train_raw = np.stack([w for _, _, w in train_items], axis=0)  # (N_tr, T, 12)
    X_val_raw = np.stack([w for _, _, w in val_items], axis=0)      # (N_val, T, 12)

    scaler = RobustFeatureScaler.fit(X_train_raw.reshape(-1, FEATURE_DIM))
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    model = TemporalLSTMAutoencoder(input_dim=FEATURE_DIM, hidden_dim=32, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    model.train()
    final_train_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_x.size(0)
        final_train_loss = epoch_loss / len(X_train)

    # Calibrare pe setul de validare (sesiune complet separată de antrenare)
    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val)
        val_recon = model(val_tensor).numpy()

    # Eliminăm primele 3 ferestre de warm-up per stream din validare (unde rolling_rate se stabilizează)
    seen_per_stream: dict[str, int] = {}
    keep_indices: list[int] = []
    for idx, (dev, topic, _) in enumerate(val_items):
        skey = f"{dev}::{topic}"
        cnt = seen_per_stream.get(skey, 0)
        seen_per_stream[skey] = cnt + 1
        if cnt >= 3:
            keep_indices.append(idx)

    X_val_clean = X_val[keep_indices]
    val_recon_clean = val_recon[keep_indices]
    val_items_clean = [val_items[i] for i in keep_indices]

    # Eroare absolută (L1) pe ultimul pas temporal per trăsătură: (N_val_clean, 12)
    last_step_abs_err = np.abs(X_val_clean[:, -1, :] - val_recon_clean[:, -1, :])
    # Scale robuste pentru normalizarea XAI (IQR + marjă minimă)
    q75_err = np.percentile(last_step_abs_err, 75, axis=0)
    q25_err = np.percentile(last_step_abs_err, 25, axis=0)
    xai_scales = np.maximum(q75_err - q25_err, 0.25)

    norm_err = last_step_abs_err / xai_scales
    val_scores = np.mean(norm_err, axis=1)

    global_threshold = float(np.percentile(val_scores, 98.0) * 1.25)
    stream_scores: dict[str, list[float]] = {}
    for idx, (dev, topic, _) in enumerate(val_items_clean):
        stream_key = f"{dev}::{topic}"
        stream_scores.setdefault(stream_key, []).append(float(val_scores[idx]))

    stream_thresholds: dict[str, float] = {}
    for skey, sc_list in stream_scores.items():
        if len(sc_list) >= 20:
            stream_thresholds[skey] = round(float(np.percentile(sc_list, 98.0) * 1.25), 6)
        else:
            stream_thresholds[skey] = round(global_threshold, 6)

    cusum_baselines: dict[str, dict[str, float]] = {}
    for (dev, topic), vals in raw_values_by_stream.items():
        arr = np.asarray(vals, dtype=np.float64)
        cusum_baselines[f"{dev}::{topic}"] = {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(max(float(np.std(arr)), 0.05), 6),
        }

    bdir = Path(bundle_dir).resolve()
    bdir.mkdir(parents=True, exist_ok=True)
    onnx_path = bdir / "model.onnx"

    dummy_in = torch.zeros((1, seq_len, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_in,
        str(onnx_path),
        input_names=["window_input"],
        output_names=["window_recon"],
        dynamic_axes={"window_input": {0: "batch"}, "window_recon": {0: "batch"}},
        opset_version=17,
    )

    (bdir / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "seq_len": seq_len,
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bdir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    (bdir / "thresholds.json").write_text(
        json.dumps(
            {
                "calibration_method": "p98_val_l1_x1.25",
                "global_threshold": round(global_threshold, 6),
                "stream_thresholds": stream_thresholds,
                "xai_error_scales": [round(float(v), 6) for v in xai_scales],
                "cusum_baselines": cusum_baselines,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    kdir = Path(key_dir).resolve()
    priv_key = kdir / "model_signing_ed25519.key"
    pub_key = kdir / "model_signing_ed25519.pub"
    if not priv_key.is_file() or not pub_key.is_file():
        generate_model_signing_keypair(priv_key, pub_key)

    manifest = sign_model_bundle(
        bdir,
        priv_key,
        metadata={
            "model_version": "v5.0.0",
            "architecture": "TemporalLSTMAutoencoder(12->32->16->32->12)",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "trained_at_unix": int(time.time()),
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "train_windows": len(X_train),
            "val_windows": len(X_val_clean),
            "final_train_loss": round(final_train_loss, 6),
            "global_threshold": round(global_threshold, 6),
        },
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train local LSTM Autoencoder and sign ONNX bundle.")
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--bundle-dir", default="models/lumen_model_bundle")
    parser.add_argument("--key-dir", default="config/model_keys")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    res = train_and_export_bundle(
        db_dir=args.db_dir,
        bundle_dir=args.bundle_dir,
        key_dir=args.key_dir,
        seq_len=args.seq_len,
        epochs=args.epochs,
    )
    print(f"[E5 MODEL BUNDLE SIGNED] version={res['model_version']} threshold={res['global_threshold']} train_win={res['train_windows']} val_win={res['val_windows']}")