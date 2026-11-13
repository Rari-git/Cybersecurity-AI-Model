import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder_net import TemporalLSTMAutoencoder
from training.feature_extractor import FEATURE_DIM, FEATURE_NAMES, load_scaler, scale_window
from training.traffic_sim import legit_events, windows_from_events
from utils.logger import get_script_logger

logger = get_script_logger("train_lstm_autoencoder")

SEQ_LEN = 10
TRAIN_SECONDS = 10800
VAL_SECONDS = 3600
EPOCHS = 30
BATCH = 256
THRESHOLD_PCTL = 99.9
THRESHOLD_MARGIN = 1.5
MIN_TOPIC_WINDOWS = 200


def _last_step_losses(model, tensor):
    with torch.no_grad():
        rec = model(tensor)
        return torch.mean((tensor[:, -1, :] - rec[:, -1, :]) ** 2, dim=1).numpy()


def train_lstm_model(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("[TRAINING] Simulare trafic legitim (aceleași trăsături ca la serving)...")
    raw_tr, _, _, _ = windows_from_events(legit_events(TRAIN_SECONDS, random.Random(seed)), SEQ_LEN)
    raw_va, topics_va, _, _ = windows_from_events(legit_events(VAL_SECONDS, random.Random(seed + 1)), SEQ_LEN)
    logger.info(f"[TRAINING] Ferestre train={len(raw_tr)} val={len(raw_va)}")

    flat = raw_tr.reshape(-1, FEATURE_DIM)
    min_vals, max_vals = flat.min(axis=0), flat.max(axis=0)
    meta_scaler = {"min_vals": min_vals.tolist(), "max_vals": max_vals.tolist()}
    mn, inv = load_scaler(meta_scaler)

    train_t = torch.tensor(scale_window(raw_tr, mn, inv))
    val_t = torch.tensor(scale_window(raw_va, mn, inv))

    model = TemporalLSTMAutoencoder(input_dim=FEATURE_DIM, hidden_dim=32, latent_dim=8, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = DataLoader(TensorDataset(train_t), batch_size=BATCH, shuffle=True)

    logger.info(f"[TRAINING] Pornire antrenare LSTM Autoencoder ({EPOCHS} epoci)...")
    model.train()
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        for (xb,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoca [{epoch+1}/{EPOCHS}] - Loss MSE: {total / n:.6f}")

    model.eval()
    val_losses = _last_step_losses(model, val_t)   # aceeași metrică ca în serving (ultimul pas)
    topics_arr = np.array(topics_va)
    threshold = float(max(np.percentile(val_losses, THRESHOLD_PCTL) * THRESHOLD_MARGIN, 1e-4))
    topic_thr = {}
    for tp in sorted(set(topics_va)):
        arr = val_losses[topics_arr == tp]
        if len(arr) >= MIN_TOPIC_WINDOWS:
            topic_thr[tp] = float(max(np.percentile(arr, THRESHOLD_PCTL) * THRESHOLD_MARGIN, 1e-4))
    logger.info(f"[TRAINING] Prag global={threshold:.6f} | per-topic={topic_thr}")

    os.makedirs("models/saved", exist_ok=True)
    meta = {**meta_scaler, "threshold": threshold, "topic_thresholds": topic_thr,
            "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM, "feature_names": FEATURE_NAMES}
    with open("models/saved/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    torch.save(model.state_dict(), "models/saved/autoencoder.pth")
    logger.info("[TRAINING] Model PyTorch LSTM salvat în 'models/saved/autoencoder.pth'")

    dummy = torch.randn(1, SEQ_LEN, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, "models/saved/autoencoder.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    logger.info("[TRAINING] Model ONNX LSTM exportat cu succes în 'models/saved/autoencoder.onnx'")


if __name__ == "__main__":
    train_lstm_model()