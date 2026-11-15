from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from models.bundle_loader import (
    hash_unsigned_candidate_artifacts,
    validate_unsigned_candidate_bundle,
)
from storage.persistence import persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.adaptive_retrainer import run_adaptive_cycle
from training.dataset_governance import curate_clean_sessions
from training.train_local_lstm import (
    build_capture_run_split,
    train_and_export_bundle,
)

TOPIC = "home/sensors/gas/ppm"
DEVICE = "gas_sensor_01"
FP = hashlib.sha256(b"capture-run-split-test-cert").digest()


def _event_row(session_id: str, index: int):
    ts = time.time_ns() + index * 1_000_000_000
    payload = f"{350.0 + index / 10.0:.1f}"
    raw = payload.encode()
    return (
        session_id, DEVICE, DEVICE, 1, FP, TOPIC, ts, ts, len(raw),
        hashlib.sha256(raw).digest(), float(payload), payload,
        "baseline_lab", "v6-capture-run-split-test",
    )


def _add_approved_session(storage, capture_run_id: str, session_index: int) -> str:
    session_id = (
        f"baseline-lab-{capture_run_id}-s{session_index:02d}-seed{session_index}"
    )
    events = 10
    storage.register_baseline_session(
        session_id,
        "baseline_lab",
        {
            "capture_run_id": capture_run_id,
            "session_index": session_index,
            "events_per_sensor": events,
            "expected_sensor_topics": [f"{DEVICE}::{TOPIC}"],
            "compressed_time": False,
        },
    )
    for index in range(events):
        persist_normal_event(storage, _event_row(session_id, index))
    storage.finalize_baseline_session(session_id)
    decision = curate_clean_sessions(
        storage,
        min_duration_sec=0,
        min_sensor_coverage=1.0,
        session_ids=[session_id],
    )[0]
    assert decision.approved, decision.reasons
    return session_id


def test_capture_run_split_keeps_all_sessions_from_one_run_together(tmp_path):
    db_dir = tmp_path / "db"
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        run_a, run_b, run_c = uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex
        a1 = _add_approved_session(storage, run_a, 1)
        a2 = _add_approved_session(storage, run_a, 2)
        b1 = _add_approved_session(storage, run_b, 1)
        c1 = _add_approved_session(storage, run_c, 1)

        split = build_capture_run_split(db_dir, [a1, a2, b1, c1], storage=storage)
        assert split is not None
        assert set(split["train_session_ids"]) == {a1, a2}
        assert set(split["validation_session_ids"]) == {b1}
        assert set(split["heldout_session_ids"]) == {c1}
        assert set(split["train_capture_run_ids"]) == {run_a}
        assert set(split["validation_capture_run_ids"]) == {run_b}
        assert set(split["heldout_capture_run_ids"]) == {run_c}
    finally:
        storage.close()


def test_three_sessions_from_one_capture_run_are_not_three_independent_runs(tmp_path):
    db_dir = tmp_path / "db"
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        run_id = uuid.uuid4().hex
        session_ids = [
            _add_approved_session(storage, run_id, index)
            for index in (1, 2, 3)
        ]
        assert build_capture_run_split(db_dir, session_ids, storage=storage) is None
    finally:
        storage.close()


def _make_unsigned_candidate(path: Path, *, leak_run: bool = False):
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"capture-run-candidate-onnx")
    (path / "feature_schema.json").write_text("{}", encoding="utf-8")
    (path / "scaler.json").write_text("{}", encoding="utf-8")
    (path / "thresholds.json").write_text("{}", encoding="utf-8")
    run_train = "a" * 32
    run_validation = run_train if leak_run else "b" * 32
    run_heldout = "c" * 32
    train_id = f"baseline-lab-{run_train}-s01-seed1"
    validation_id = f"baseline-lab-{run_validation}-s01-seed2"
    heldout_id = f"baseline-lab-{run_heldout}-s01-seed3"
    digest = hashlib.sha256(b"capture-run-curation").hexdigest()
    metadata = {
        "model_version": "v6-candidate-capture-run-test",
        "dataset_policy": "approved_clean_baseline_only",
        "offline_training": True,
        "split_unit": "capture_run_id",
        "candidate_files_sha256": hash_unsigned_candidate_artifacts(path),
        "train_sessions": [train_id],
        "val_sessions": [validation_id],
        "heldout_sessions": [heldout_id],
        "cusum_calibration_sessions": [train_id],
        "curation_digests": {
            train_id: digest, validation_id: digest, heldout_id: digest,
        },
        "capture_run_ids_by_session": {
            train_id: run_train,
            validation_id: run_validation,
            heldout_id: run_heldout,
        },
        "train_capture_runs": [run_train],
        "val_capture_runs": [run_validation],
        "heldout_capture_runs": [run_heldout],
        "global_threshold": 1.5,
        "validation_normal_error_p95": 1.0,
    }
    (path / "manifest.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return metadata


def test_unsigned_candidate_manifest_requires_disjoint_capture_run_partitions(tmp_path):
    candidate = tmp_path / "candidate"
    _make_unsigned_candidate(candidate)
    validated = validate_unsigned_candidate_bundle(candidate)
    assert validated.metadata["split_unit"] == "capture_run_id"

    leaking = tmp_path / "leaking-candidate"
    _make_unsigned_candidate(leaking, leak_run=True)
    with pytest.raises(ValueError, match="capture_run_id partitions must be disjoint"):
        validate_unsigned_candidate_bundle(leaking)


def test_training_rejects_capture_run_leakage_before_model_fit(tmp_path, monkeypatch):
    import training.train_local_lstm as trainer

    run_a, run_b, run_c = "a" * 32, "b" * 32, "c" * 32
    train_id = f"baseline-lab-{run_a}-s01-seed1"
    validation_id = f"baseline-lab-{run_a}-s02-seed2"
    heldout_id = f"baseline-lab-{run_c}-s01-seed3"
    window = np.zeros((4, 12), dtype=np.float32)
    monkeypatch.setattr(
        trainer,
        "_extract_windows_by_session",
        lambda *_args, **_kwargs: (
            {train_id: [(DEVICE, TOPIC, window)], validation_id: [(DEVICE, TOPIC, window)],
             heldout_id: [(DEVICE, TOPIC, window)]},
            {train_id: {(DEVICE, TOPIC): [350.0]},
             validation_id: {(DEVICE, TOPIC): [350.0]},
             heldout_id: {(DEVICE, TOPIC): [350.0]}},
        ),
    )
    monkeypatch.setattr(
        trainer,
        "load_capture_run_ids",
        lambda *_args, **_kwargs: {
            train_id: run_a, validation_id: run_a, heldout_id: run_c,
        },
    )
    with pytest.raises(ValueError, match="capture_run_id leakage"):
        train_and_export_bundle(
            db_dir=tmp_path / "db",
            bundle_dir=tmp_path / "candidate",
            eligible_session_ids=[train_id, validation_id, heldout_id],
            validation_session_ids=[validation_id],
            heldout_session_ids=[heldout_id],
            require_independent_capture_runs=True,
            sign_bundle=False,
            seq_len=4,
            epochs=1,
        )
    assert not (tmp_path / "candidate").exists()


def test_adaptive_worker_stops_before_candidate_creation_without_three_runs(tmp_path, monkeypatch):
    import training.adaptive_retrainer as adaptive

    db_dir = tmp_path / "db"
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        run_id = uuid.uuid4().hex
        session_ids = [
            _add_approved_session(storage, run_id, index)
            for index in (1, 2, 3)
        ]
        key_dir = tmp_path / "authority-keys"
        key_dir.mkdir()
        public_key = key_dir / "model_signing_ed25519.pub"
        public_key.write_bytes(b"preflight-test-public-key-placeholder")
        model_root = tmp_path / "models"
        monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")

        def forbidden_training(**_kwargs):
            raise AssertionError("training must not start before capture-run preflight passes")

        monkeypatch.setattr(adaptive, "train_and_export_bundle", forbidden_training)
        result = run_adaptive_cycle(
            db_dir=db_dir,
            model_root=model_root,
            public_key_path=public_key,
            min_duration_sec=0,
            min_sensor_coverage=1.0,
            storage=storage,
            run_id="insufficient-run-preflight",
        )
        assert result.curated_sessions == tuple(session_ids)
        assert result.candidate_dir is None
        assert result.preflight_reason == "INSUFFICIENT_INDEPENDENT_CAPTURE_RUNS"
        assert not (model_root / "candidates").exists()
    finally:
        storage.close()
