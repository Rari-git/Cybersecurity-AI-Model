from __future__ import annotations

import copy
import hashlib
import time

import pytest

from storage.persistence import persist_normal_event
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.dataset_governance import curate_clean_sessions
from training.train_local_lstm import build_capture_run_split
from simulation.lab.temporal.transition_profile import (
    CLEAN_TRANSITION_PROFILE_SCHEMA,
    TransitionProfileError,
    build_clean_transition_profile,
    evaluate_clean_transition_profile,
)

DEVICE = "profile_sensor_01"
TOPIC = "home/sensors/profile/value"
FINGERPRINT = hashlib.sha256(b"transition-profile-test-cert").digest()
RUN_TRAIN = "1" * 32
RUN_VALIDATION = "2" * 32
RUN_HELDOUT = "3" * 32


def _create_session(storage, capture_run_id, suffix, events):
    session_id = f"baseline-lab-{capture_run_id}-{suffix}"
    storage.register_baseline_session(
        session_id,
        "baseline_lab",
        {
            "capture_run_id": capture_run_id,
            "events_per_sensor": len(events),
            "expected_sensor_topics": [f"{DEVICE}::{TOPIC}"],
            "compressed_time": False,
        },
    )
    for gateway_ts_ns, payload in events:
        raw = payload.encode("utf-8")
        persist_normal_event(
            storage,
            (
                session_id, DEVICE, DEVICE, 1, FINGERPRINT, TOPIC,
                gateway_ts_ns, gateway_ts_ns, len(raw), hashlib.sha256(raw).digest(),
                1.0, payload, "baseline_lab", "transition-profile-test",
            ),
        )
    storage.finalize_baseline_session(session_id)
    return session_id


def _profile_split(storage, training_sessions):
    base = time.time_ns()
    validation_id = _create_session(
        storage,
        RUN_VALIDATION,
        "validation",
        [(base + 10_000_000_000, "3"),
         (base + 10_050_000_000, "3"),
         (base + 10_250_000_000, "12345678901234567"),
         (base + 10_500_000_000, "12345678901234568")],
    )
    heldout_id = _create_session(
        storage, RUN_HELDOUT, "heldout", [(base + 20_000_000_000, "4")]
    )
    session_ids = list(training_sessions) + [validation_id, heldout_id]
    decisions = curate_clean_sessions(
        storage,
        min_duration_sec=0,
        min_sensor_coverage=1.0,
        session_ids=session_ids,
    )
    assert len(decisions) == len(session_ids)
    assert all(item.approved for item in decisions)
    split = build_capture_run_split(".", session_ids, storage=storage)
    assert split is not None
    all_digests = {item.session_id: item.dataset_sha256 for item in decisions}
    training_digests = {
        session_id: all_digests[session_id] for session_id in split["train_session_ids"]
    }
    validation_digests = {
        session_id: all_digests[session_id] for session_id in split["validation_session_ids"]
    }
    heldout_digests = {
        session_id: all_digests[session_id] for session_id in split["heldout_session_ids"]
    }
    return split, training_digests, validation_digests, heldout_digests


@pytest.fixture
def storage(tmp_path):
    manager = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        yield manager
    finally:
        manager.close()


def test_builds_deterministic_profile_from_train_runs_only(storage):
    base = time.time_ns()
    train_a = _create_session(
        storage,
        RUN_TRAIN,
        "train-a",
        [(base, "1"), (base + 50_000_000, "2"), (base + 250_000_000, "12345678901234567")],
    )
    train_b = _create_session(
        storage,
        RUN_TRAIN,
        "train-b",
        [(base + 2_000_000_000, "12345678901234567"),
         (base + 3_500_000_000, "12345678901234568")],
    )
    split, digests, _validation_digests, _heldout_digests = _profile_split(storage, [train_a, train_b])

    first = build_clean_transition_profile(storage, split, digests)
    second = build_clean_transition_profile(storage, split, digests)
    uppercase_digests = {session_id: digest.upper() for session_id, digest in digests.items()}
    uppercase = build_clean_transition_profile(storage, split, uppercase_digests)

    assert first == second == uppercase
    assert first["schema_version"] == CLEAN_TRANSITION_PROFILE_SCHEMA
    assert first["summary"] == {
        "session_count": 2,
        "stream_segment_count": 2,
        "event_count": 5,
        "unique_state_count": 2,
        "observed_transition_count": 3,
        "unique_transition_count": 3,
    }
    assert {edge["inter_arrival_class"] for edge in first["transitions"]} == {"rapid", "normal", "delayed"}
    assert sum(edge["count"] for edge in first["transitions"]) == 3
    assert all(edge["outgoing_probability"] > 0 for edge in first["transitions"])
    assert first["configuration"]["clock_basis"] == "gateway_ts_ns_differences_within_session_and_stream"
    assert first["source"]["capture_run_split"]["training_session_ids"] == sorted([train_a, train_b])
    assert train_a not in first["source"]["capture_run_split"]["validation_session_ids"]
    assert train_b not in first["source"]["capture_run_split"]["heldout_session_ids"]
    assert first["interpretation"]["anomaly_score_generated"] is False
    assert len(first["profile_sha256"]) == 64


def test_rejects_mismatched_curation_digest(storage):
    base = time.time_ns()
    train_id = _create_session(storage, RUN_TRAIN, "train", [(base, "1"), (base + 1, "2")])
    split, digests, _validation_digests, _heldout_digests = _profile_split(storage, [train_id])
    digests[train_id] = "0" * 64

    with pytest.raises(RuntimeError, match="dataset digest mismatch"):
        build_clean_transition_profile(storage, split, digests)


def test_rejects_gateway_clock_rollback_within_a_session(storage):
    base = time.time_ns()
    train_id = _create_session(storage, RUN_TRAIN, "train", [(base, "1"), (base - 1, "2")])
    split, digests, _validation_digests, _heldout_digests = _profile_split(storage, [train_id])

    with pytest.raises(TransitionProfileError, match="gateway timestamp moved backwards"):
        build_clean_transition_profile(storage, split, digests)


def test_rejects_capture_run_leakage_between_partitions(storage):
    base = time.time_ns()
    train_id = _create_session(storage, RUN_TRAIN, "train", [(base, "1"), (base + 1, "2")])
    split, digests, _validation_digests, _heldout_digests = _profile_split(storage, [train_id])
    invalid = dict(split)
    invalid_run_map = dict(split["capture_run_ids_by_session"])
    validation_id = invalid["validation_session_ids"][0]
    invalid_run_map[validation_id] = invalid_run_map[train_id]
    invalid["capture_run_ids_by_session"] = invalid_run_map

    with pytest.raises(TransitionProfileError, match="capture runs must not cross"):
        build_clean_transition_profile(storage, invalid, digests)


def test_validation_reports_coverage_without_updating_profile(storage):
    base = time.time_ns()
    train_a = _create_session(
        storage,
        RUN_TRAIN,
        "train-a",
        [(base, "1"), (base + 50_000_000, "2"), (base + 250_000_000, "12345678901234567")],
    )
    train_b = _create_session(
        storage,
        RUN_TRAIN,
        "train-b",
        [(base + 2_000_000_000, "12345678901234567"),
         (base + 3_500_000_000, "12345678901234568")],
    )
    split, train_digests, validation_digests, heldout_digests = _profile_split(storage, [train_a, train_b])
    profile = build_clean_transition_profile(storage, split, train_digests)

    result = evaluate_clean_transition_profile(
        profile, storage, split["validation_session_ids"], validation_digests
    )

    assert result["source"]["evaluation_partition"] == "validation"
    assert result["summary"]["validation_session_count"] == 1
    assert result["summary"]["state_coverage_rate"] == 1.0
    assert result["summary"]["transition_count"] == 3
    assert result["summary"]["known_transition_count"] == 2
    assert result["summary"]["transition_coverage_rate"] == pytest.approx(2 / 3)
    assert result["interpretation"]["anomaly_score_generated"] is False
    assert result["interpretation"]["profile_updated"] is False
    assert result["source"]["profile_sha256"] == profile["profile_sha256"]
    assert len(result["validation_sha256"]) == 64

    with pytest.raises(TransitionProfileError, match="complete validation partition only"):
        evaluate_clean_transition_profile(profile, storage, split["heldout_session_ids"], heldout_digests)


def test_validation_rejects_tampered_profile_and_bad_curation_digest(storage):
    base = time.time_ns()
    train_id = _create_session(storage, RUN_TRAIN, "train", [(base, "1"), (base + 1, "2")])
    split, train_digests, validation_digests, _heldout_digests = _profile_split(storage, [train_id])
    profile = build_clean_transition_profile(storage, split, train_digests)
    tampered = copy.deepcopy(profile)
    tampered["configuration"]["rapid_interval_seconds"] = 0.05
    with pytest.raises(TransitionProfileError, match="profile digest"):
        evaluate_clean_transition_profile(tampered, storage, split["validation_session_ids"], validation_digests)

    bad_digests = dict(validation_digests)
    bad_digests[next(iter(bad_digests))] = "f" * 64
    with pytest.raises(RuntimeError, match="dataset digest mismatch"):
        evaluate_clean_transition_profile(profile, storage, split["validation_session_ids"], bad_digests)
