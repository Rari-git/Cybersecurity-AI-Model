"""Etapa E2: Captură de baseline curat în laborator (multi-sesiune).

Generează trafic normal validat pentru toți cei 5 senzori, cu seed determinist
și cuantizare ADC realistă, populând telemetry_normal.db prin AdmissionGate.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.telemetry_db import LumenStorageManager, DBPaths
from storage.admission_gate import AdmissionGate, ProvenanceContext
from training.dataset_governance import curate_clean_sessions

DEFAULT_ALLOWED_TOPICS: dict[str, set[str]] = {
    "gas_sensor_01": {"home/sensors/gas/ppm"},
    "gas_sensor_02": {"home/sensors/gas/ppm"},
    "door_sensor_01": {"home/sensors/door/state"},
    "vibration_sensor_01": {"home/sensors/vibration/shock"},
    "pir_sensor_01": {"home/sensors/pir/motion"},
}


def _simulate_sensor_reading(device_id: str, step: int, rng: random.Random) -> bytes:
    """Generează o citire realistă (cu cuantizare ADC) pentru un senzor curat."""
    if device_id == "gas_sensor_01":
        # Oscilație diurnă lentă + zgomot discretizat pe pași de 1.0 ppm (ADC 10-bit)
        diurnal = 18.0 * math.sin(step * 0.02)
        val = int(round(350.0 + diurnal + rng.gauss(0.0, 6.0)))
        return str(max(250, min(480, val))).encode("utf-8")

    if device_id == "gas_sensor_02":
        diurnal = 15.0 * math.cos(step * 0.02)
        val = int(round(340.0 + diurnal + rng.gauss(0.0, 5.5)))
        return str(max(250, min(470, val))).encode("utf-8")

    if device_id == "vibration_sensor_01":
        # Rezoluție ADC de 0.001 g
        val = round(max(0.0, 0.015 + abs(rng.gauss(0.0, 0.008))), 3)
        return f"{val:.3f}".encode("utf-8")

    if device_id == "door_sensor_01":
        # Stare stabilă cu tranziții rare (probabilitate 3%)
        state = "OPEN" if rng.random() < 0.03 else "CLOSED"
        return state.encode("utf-8")

    if device_id == "pir_sensor_01":
        state = "MOTION_DETECTED" if rng.random() < 0.12 else "NO_MOTION"
        return state.encode("utf-8")

    raise ValueError(f"Unknown device_id: {device_id}")


def run_baseline_capture(
    db_dir: str | Path = "logs/db",
    sessions: int = 3,
    events_per_sensor: int = 300,
    rate_hz: float = 1.0,
    seed: int = 42,
    realtime_sleep: bool = False,
) -> dict[str, int]:
    """Capturează mai multe sesiuni curate în telemetry_normal.db.

    Dacă realtime_sleep=False, folosește timp comprimat determinist (ideal pentru
    generarea rapidă a 24-48h echivalent de trafic pentru calibrarea FPR/oră).
    """
    paths = DBPaths.from_dir(db_dir)
    storage = LumenStorageManager(paths=paths, enable_writers=True)
    gate = AdmissionGate(storage=storage, allowed_topics_by_device=DEFAULT_ALLOWED_TOPICS)

    rng = random.Random(seed)
    capture_run_id = uuid.uuid4().hex
    admitted_total = 0
    rejected_total = 0

    try:
        for s_idx in range(1, sessions + 1):
            session_id = f"baseline-lab-{capture_run_id}-s{s_idx:02d}-seed{seed}"
            storage.register_baseline_session(
                session_id=session_id,
                source_type="baseline_lab",
                config={
                    "capture_run_id": capture_run_id,
                    "session_index": s_idx,
                    "events_per_sensor": events_per_sensor,
                    "rate_hz": rate_hz,
                    "seed": seed,
                    "compressed_time": not realtime_sleep,
                    "effective_duration_sec": (max(0, events_per_sensor - 1) / rate_hz) if not realtime_sleep else None,
                    "expected_sensor_topics": [
                        f"{device_id}::{next(iter(topics))}"
                        for device_id, topics in sorted(DEFAULT_ALLOWED_TOPICS.items())
                    ],
                },
                approved_for_training=False,
            )

            for step in range(events_per_sensor):
                for device_id, topics in DEFAULT_ALLOWED_TOPICS.items():
                    topic = next(iter(topics))
                    cn = f"{device_id}-g{s_idx}-base{seed:04d}"
                    fp = hashlib.sha256(cn.encode("utf-8")).digest()
                    ctx = ProvenanceContext(
                        session_id=session_id,
                        device_cn=cn,
                        cert_fingerprint=fp,
                        source_type="baseline_lab",
                    )

                    payload = _simulate_sensor_reading(device_id, step, rng)
                    res = gate.evaluate_and_ingest(ctx, topic, payload)
                    if res.admitted:
                        admitted_total += 1
                    else:
                        rejected_total += 1

                if realtime_sleep:
                    time.sleep(1.0 / rate_hz)

            storage.finalize_baseline_session(session_id, approved_for_training=False)
            curate_clean_sessions(
                storage,
                min_duration_sec=0.0 if not realtime_sleep else max(0.0, events_per_sensor / rate_hz * 0.8),
                min_sensor_coverage=1.0,
                session_ids=(session_id,),
            )
    finally:
        storage.flush_all()
        storage.close()

    return {
        "sessions": sessions,
        "admitted": admitted_total,
        "rejected": rejected_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture clean baseline telemetry into SQLite.")
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--events", type=int, default=300, help="Events per sensor per session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    stats = run_baseline_capture(
        db_dir=args.db_dir,
        sessions=args.sessions,
        events_per_sensor=args.events,
        seed=args.seed,
        realtime_sleep=args.realtime,
    )
    print(f"[E2 BASELINE COMPLETE] {stats}")