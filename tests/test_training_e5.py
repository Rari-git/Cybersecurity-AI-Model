"""Etapa E5: Teste automate pentru antrenarea locală PyTorch, export ONNX,
verificarea semnăturii Ed25519 și detectarea manipulării (tampering) pachetului.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.capture_baseline import run_baseline_capture
from training.train_local_lstm import train_and_export_bundle
from models.bundle_loader import generate_model_signing_keypair
from models.bundle_loader import verify_and_load_bundle


def test_e5_train_export_sign_and_tamper_detection():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e5-"))
    db_dir = tmp_dir / "db"
    bundle_dir = tmp_dir / "bundle"
    key_dir = tmp_dir / "keys"

    try:
        # 1. Generăm 3 sesiuni curate (2 train + 1 val)
        run_baseline_capture(db_dir=db_dir, sessions=3, events_per_sensor=60, seed=42)

        generate_model_signing_keypair(key_dir / "model_signing_ed25519.key", key_dir / "model_signing_ed25519.pub")

        # 2. Antrenam, exportam ONNX si semnam manifestul
        manifest = train_and_export_bundle(
            db_dir=db_dir,
            bundle_dir=bundle_dir,
            key_dir=key_dir,
            seq_len=12,
            epochs=8,
        )
        assert manifest["train_windows"] > 0
        assert manifest["val_windows"] > 0

        # 3. Verificăm semnătura și rulăm o inferență ONNX Runtime
        pub_key = key_dir / "model_signing_ed25519.pub"
        verified = verify_and_load_bundle(bundle_dir, pub_key)
        sess = ort.InferenceSession(str(verified.onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 12, 12), dtype=np.float32)
        out = sess.run(None, {"window_input": dummy})[0]
        assert out.shape == (1, 12, 12)

        # 4. Tamper test: modificăm un singur byte din thresholds.json -> trebuie respins!
        thresh_file = bundle_dir / "thresholds.json"
        thresh_file.write_text(thresh_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify_and_load_bundle(bundle_dir, pub_key)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)