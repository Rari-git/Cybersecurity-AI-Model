from __future__ import annotations

import hashlib
import json
import pytest

from simulation.lab.temporal.feature_extractor import (
    TEMPORAL_FEATURE_SCHEMA, TemporalFeatureError, extract_temporal_features,
)


def _sequence(intervals=(0.1, 0.05, 0.04, 0.5, 0.05), dropped=None):
    events, elapsed = [], 0.0
    dropped = set(dropped or [])
    for index, interval in enumerate(intervals):
        elapsed += interval
        events.append({"sequence_index": index, "source_step": index,
            "relative_interval_sec": interval, "cumulative_elapsed_sec": round(elapsed, 9),
            "time_basis": "accumulated_delta_t_sec", "absolute_timestamp_available": False,
            "device_id": "sensor-a" if index < 3 else "sensor-b",
            "topic": "telemetry/temp" if index < 3 else "telemetry/pressure",
            "operation": "publish" if index != 4 else "query", "dropped": index in dropped,
            "ground_truth_label": "attack", "payload_len": 10 + index,
            "payload_sha256": hashlib.sha256(f"payload-{index}".encode()).hexdigest()})
    result = {"schema_version": "c3.temporal-sequence.v1",
        "source": {"source_type": "c2_verified_generated_evidence", "run_id": "run-test",
            "scenario_id": "scenario-test", "family": "family-test", "variation_id": "variation-test",
            "split": "dev", "corpus_class": "adversarial", "category": "attack",
            "execution_mode": "offline_frame_generation_only", "external_execution_performed": False},
        "time_semantics": {"clock_type": "relative_interval", "interval_field": "delta_t_sec",
            "cumulative_origin": "zero_at_start_of_recorded_sequence",
            "absolute_timestamps_synthesized": False, "first_frame_interval_included": True},
        "event_count": len(events), "events": events}
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    result["sequence_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def _rehash(sequence):
    sequence.pop("sequence_sha256", None)
    raw = json.dumps(sequence, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    sequence["sequence_sha256"] = hashlib.sha256(raw).hexdigest()
    return sequence


def test_deterministic_versioned_features():
    seq = _sequence()
    a, b = extract_temporal_features(seq), extract_temporal_features(seq)
    assert a == b and a["schema_version"] == TEMPORAL_FEATURE_SCHEMA


def test_inter_arrival_statistics_include_origin_interval():
    stats = extract_temporal_features(_sequence())["inter_arrival"]["interval_seconds_stats"]
    assert stats["count"] == 5 and stats["mean"] == pytest.approx(0.148)
    assert stats["min"] == pytest.approx(0.04) and stats["max"] == pytest.approx(0.5)


def test_window_rates_cover_every_event():
    result = extract_temporal_features(_sequence(), window_seconds=0.25)
    assert sum(result["traffic_rate"]["fixed_windows"]["events_per_window"]) == 5


def test_burst_detection_obeys_configured_threshold():
    result = extract_temporal_features(_sequence(), burst_gap_threshold_seconds=0.1, min_burst_events=3)
    assert result["bursts"]["burst_count"] == 1
    assert result["bursts"]["segments"][0]["start_sequence_index"] == 0
    assert result["bursts"]["segments"][0]["end_sequence_index"] == 2


def test_transition_counts():
    result = extract_temporal_features(_sequence())
    assert result["transitions"]["device_id"]["change_count"] == 1
    assert result["transitions"]["topic"]["change_count"] == 1
    assert result["transitions"]["operation"]["change_count"] == 1


def test_sequence_quality():
    quality = extract_temporal_features(_sequence(dropped={1, 4}))["sequence_quality"]
    assert quality["dropped_event_count"] == 2 and quality["dropped_event_ratio"] == pytest.approx(0.4)


def test_rejects_digest_tampering():
    seq = _sequence(); seq["events"][0]["topic"] = "tampered"
    with pytest.raises(TemporalFeatureError, match="digest verification"):
        extract_temporal_features(seq)


@pytest.mark.parametrize("split", ["held_out", "regression"])
def test_rejects_non_development_split(split):
    seq = _sequence(); seq["source"]["split"] = split; _rehash(seq)
    with pytest.raises(TemporalFeatureError, match="development"):
        extract_temporal_features(seq)


def test_rejects_changed_time_semantics():
    seq = _sequence(); seq["time_semantics"]["absolute_timestamps_synthesized"] = True; _rehash(seq)
    with pytest.raises(TemporalFeatureError, match="time semantics"):
        extract_temporal_features(seq)


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf")])
def test_rejects_invalid_window(bad):
    with pytest.raises(TemporalFeatureError):
        extract_temporal_features(_sequence(), window_seconds=bad)


def test_rejects_non_integer_minimum_burst_size():
    with pytest.raises(TemporalFeatureError, match="integer"):
        extract_temporal_features(_sequence(), min_burst_events=2.0)


def test_rejects_inconsistent_cumulative_time():
    seq = _sequence(); seq["events"][2]["cumulative_elapsed_sec"] += 0.25; _rehash(seq)
    with pytest.raises(TemporalFeatureError, match="cumulative event time"):
        extract_temporal_features(seq)


def test_zero_elapsed_has_null_overall_and_window_rates():
    result = extract_temporal_features(_sequence(intervals=(0.0, 0.0)))
    assert result["traffic_rate"]["overall_events_per_second"] is None
    assert result["traffic_rate"]["rate_undefined_reason"] == "zero_elapsed_duration"
    windows = result["traffic_rate"]["fixed_windows"]
    assert windows["window_count"] == 0
    assert windows["events_per_window"] == []
    assert windows["rate_events_per_second"]["count"] == 0
    assert windows["rate_undefined_reason"] == "zero_elapsed_duration"


def test_real_c2_to_c3_pipeline_feature_extraction(tmp_path):
    """Exercise C2.4 evidence -> C2.5 index -> C3.1 sequence -> C3.2 features."""
    from simulation.lab.execution_engine import OfflineScenarioEngine
    from simulation.lab.evidence.evidence_index import create_evidence_index
    from simulation.lab.staging import ResourceLimits, StagingConfig
    from simulation.lab.temporal.sequence_engine import normalize_verified_generated_run

    config = StagingConfig(
        workspace_root=tmp_path / "c3_2_real_pipeline",
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
        variation_seed=90210,
        steps=8,
    )
    create_evidence_index(run.output_directory)
    sequence = normalize_verified_generated_run(run.output_directory)
    first = extract_temporal_features(sequence)
    second = extract_temporal_features(sequence)

    assert first["schema_version"] == TEMPORAL_FEATURE_SCHEMA
    assert first["source"]["sequence_sha256"] == sequence["sequence_sha256"]
    assert first["source"]["split"] == "dev"
    assert first["source"]["corpus_class"] == "adversarial"
    assert first["source"]["external_execution_performed"] is False
    assert first["sequence_quality"]["event_count"] == 8
    assert first["feature_sha256"] == second["feature_sha256"]
    assert first == second

def test_zero_duration_burst_intensity_is_undefined():
    result = extract_temporal_features(
        _sequence(intervals=(0.0, 0.0, 0.0)),
        burst_gap_threshold_seconds=0.1,
        min_burst_events=3,
    )
    bursts = result["bursts"]
    assert bursts["burst_count"] == 1
    assert bursts["intensity_measurable_burst_count"] == 0
    assert bursts["intensity_undefined_burst_count"] == 1
    assert bursts["intensity_events_per_second_stats"]["count"] == 0
    segment = bursts["segments"][0]
    assert segment["duration_seconds"] == 0.0
    assert segment["intensity_events_per_second"] is None
    assert segment["intensity_undefined_reason"] == "zero_duration"
