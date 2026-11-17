from __future__ import annotations

import base64
import hashlib
import json

import pytest

from simulation.lab.temporal.sequence_engine import (
    TemporalSequenceError,
    _decode_frame,
    _finite_interval,
    TEMPORAL_SEQUENCE_SCHEMA,
)


def _frame(step=0, delta=0.25):
    payload = b"sample"
    return {
        "schema_version": "c2.generated-frame.v1",
        "step": step,
        "device_id": "sensor-1",
        "topic": "ingest/sensors/temperature",
        "operation": "publish",
        "delta_t_sec": delta,
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "ground_truth_label": "attack",
        "scenario_id": "variation-1",
        "dropped": False,
    }


def test_interval_rejects_bool():
    with pytest.raises(TemporalSequenceError):
        _finite_interval(True, "delta_t_sec")


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_interval_rejects_negative_and_non_finite(value):
    with pytest.raises(TemporalSequenceError):
        _finite_interval(value, "delta_t_sec")


def test_interval_accepts_zero():
    assert _finite_interval(0, "delta_t_sec") == 0.0


def test_decode_frame_preserves_step_and_payload_digest():
    record = _decode_frame(json.dumps(_frame()).encode(), expected_step=0, filename="frame_000000.json")
    assert record["step"] == 0
    assert record["_delta"] == 0.25
    assert record["_payload_len"] == 6
    assert record["_payload_sha256"] == hashlib.sha256(b"sample").hexdigest()


def test_decode_frame_rejects_non_contiguous_step():
    with pytest.raises(TemporalSequenceError, match="non-contiguous"):
        _decode_frame(json.dumps(_frame(step=2)).encode(), expected_step=0, filename="frame_000000.json")


def test_decode_frame_rejects_wrong_label():
    record = _frame()
    record["ground_truth_label"] = "normal"
    with pytest.raises(TemporalSequenceError, match="ground-truth"):
        _decode_frame(json.dumps(record).encode(), expected_step=0, filename="frame_000000.json")


def test_decode_frame_rejects_extra_fields():
    record = _frame()
    record["timestamp"] = 123
    with pytest.raises(TemporalSequenceError, match="fields"):
        _decode_frame(json.dumps(record).encode(), expected_step=0, filename="frame_000000.json")


def test_duplicate_json_keys_rejected():
    raw = b'{"schema_version":"c2.generated-frame.v1","schema_version":"x"}'
    with pytest.raises(TemporalSequenceError, match="duplicate"):
        _decode_frame(raw, expected_step=0, filename="frame_000000.json")

# C3.1 adversarial boundary and real-evidence integration tests.
import copy
from types import SimpleNamespace

from simulation.lab.evidence.evidence_index import create_evidence_index
from simulation.lab.temporal import sequence_engine


@pytest.fixture
def generated_run(tmp_path):
    """Create a real, isolated C2.4 development run for C2.5/C3.1 tests."""
    from simulation.lab.execution_engine import OfflineScenarioEngine
    from simulation.lab.staging import ResourceLimits, StagingConfig

    config = StagingConfig(
        workspace_root=tmp_path / "c3_generated_evidence",
        resource_limits=ResourceLimits(
            max_frames=32,
            max_payload_bytes=16384,
            max_total_bytes=2_000_000,
            max_runtime_seconds=30.0,
            max_manifest_bytes=65536,
        ),
    )
    run = OfflineScenarioEngine(config).run_development_attack(
        "atk1_stealth_spoof_dev",
        variation_seed=731,
        steps=8,
    )
    return run.output_directory


@pytest.fixture
def indexed_generated_run(generated_run):
    create_evidence_index(generated_run)
    return generated_run


def test_real_indexed_run_normalizes_deterministically(indexed_generated_run):
    first = sequence_engine.normalize_verified_generated_run(indexed_generated_run)
    second = sequence_engine.normalize_verified_generated_run(indexed_generated_run)

    assert first["schema_version"] == "c3.temporal-sequence.v1"
    assert first["event_count"] == 8
    assert first["events"] == second["events"]
    assert first["sequence_sha256"] == second["sequence_sha256"]
    assert first["source"]["evidence_index_sha256"]
    assert first["source"]["split"] == "dev"
    assert first["source"]["corpus_class"] == "adversarial"
    assert first["time_semantics"]["absolute_timestamps_synthesized"] is False
    assert all(event["absolute_timestamp_available"] is False for event in first["events"])


def test_cumulative_time_is_sum_of_source_intervals(indexed_generated_run):
    sequence = sequence_engine.normalize_verified_generated_run(indexed_generated_run)
    source_intervals = []
    for frame_path in sorted((indexed_generated_run / "frames").glob("frame_*.json")):
        record = json.loads(frame_path.read_text(encoding="utf-8"))
        source_intervals.append(float(record["delta_t_sec"]))

    assert sequence["events"][0]["cumulative_elapsed_sec"] == pytest.approx(source_intervals[0])
    assert sequence["events"][-1]["cumulative_elapsed_sec"] == pytest.approx(sum(source_intervals))
    assert [event["sequence_index"] for event in sequence["events"]] == list(range(8))


def test_normalizer_rejects_frame_tampering_after_index(indexed_generated_run):
    frame = indexed_generated_run / "frames" / "frame_000000.json"
    record = json.loads(frame.read_text(encoding="utf-8"))
    record["topic"] = "tampered/topic"
    frame.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(TemporalSequenceError, match="verification failed"):
        sequence_engine.normalize_verified_generated_run(indexed_generated_run)


def test_normalizer_rejects_index_digest_tampering(indexed_generated_run):
    index_path = indexed_generated_run / "evidence_index.json"
    envelope = json.loads(index_path.read_text(encoding="utf-8"))
    digest = envelope["evidence_index_sha256"]
    envelope["evidence_index_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    index_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(TemporalSequenceError, match="verification failed"):
        sequence_engine.normalize_verified_generated_run(indexed_generated_run)


@pytest.mark.parametrize("delta", [True, -0.001, float("nan"), float("inf"), float("-inf")])
def test_frame_decoder_rejects_invalid_temporal_intervals(delta):
    record = _frame(delta=delta)
    raw = json.dumps(record, allow_nan=True).encode("utf-8")
    with pytest.raises(TemporalSequenceError):
        _decode_frame(raw, expected_step=0, filename="frame_000000.json")


@pytest.mark.parametrize(
    "source",
    [
        {"category": "attack", "corpus_class": "adversarial", "split": "held_out"},
        {"category": "protocol", "corpus_class": "protocol_violation", "split": "regression"},
        {"category": "fault", "corpus_class": "hardware_fault", "split": "regression"},
        {"category": "clean", "corpus_class": "clean_control", "split": "dev"},
    ],
)
def test_normalizer_rejects_non_development_adversarial_sources(monkeypatch, tmp_path, source):
    fake_payload = {
        "schema_version": "c2.evidence-index.v1",
        "source_scenario": {
            **source,
            "scenario_id": "synthetic-test",
            "definition_digest": "a" * 64,
            "family": "synthetic-test",
            "ground_truth": "test",
        },
        "variation": {"variation_id": "variation-test", "variation_digest": "b" * 64},
        "runner": {"execution_mode": "offline_frame_generation_only"},
        "artifacts": [],
        "frame_count": 0,
        "frames_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        sequence_engine,
        "verify_evidence_index",
        lambda *args, **kwargs: SimpleNamespace(
            payload=fake_payload,
            evidence_index_sha256="d" * 64,
        ),
    )
    with pytest.raises(TemporalSequenceError):
        sequence_engine.normalize_verified_generated_run(tmp_path)


@pytest.mark.parametrize("invalid_step", [True, 0.0, "0"])
def test_decoder_rejects_non_integer_step(invalid_step):
    record = _frame(step=invalid_step)
    with pytest.raises(TemporalSequenceError, match="integer"):
        _decode_frame(json.dumps(record).encode(), expected_step=0, filename="frame_000000.json")
