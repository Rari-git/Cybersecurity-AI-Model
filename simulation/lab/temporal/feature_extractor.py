"""C3.2 deterministic temporal feature extraction from C3.1 sequences."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping

TEMPORAL_FEATURE_SCHEMA = "c3.temporal-features.v1"
SEQUENCE_SCHEMA = "c3.temporal-sequence.v1"
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_BURST_GAP_SECONDS = 0.1
DEFAULT_MIN_BURST_EVENTS = 3
MAX_TRANSITIONS_REPORTED = 20


class TemporalFeatureError(ValueError):
    """A normalized sequence is invalid or cannot be safely summarized."""


def _digest(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _number(value: Any, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalFeatureError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise TemporalFeatureError(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TemporalFeatureError(f"{field} must be an integer >= {minimum}")
    return value


def _nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev_population": None,
                "min": None, "max": None, "p50_nearest_rank": None,
                "p90_nearest_rank": None, "p95_nearest_rank": None,
                "coefficient_of_variation": None}
    ordered = sorted(values)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    stddev = math.sqrt(variance)
    return {"count": len(values), "mean": mean, "median": median,
            "stddev_population": stddev, "min": ordered[0], "max": ordered[-1],
            "p50_nearest_rank": _nearest_rank(values, 50),
            "p90_nearest_rank": _nearest_rank(values, 90),
            "p95_nearest_rank": _nearest_rank(values, 95),
            "coefficient_of_variation": stddev / mean if mean > 0 else None}


def _validate(sequence: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    if not isinstance(sequence, Mapping) or sequence.get("schema_version") != SEQUENCE_SCHEMA:
        raise TemporalFeatureError("unsupported temporal sequence schema")
    supplied = sequence.get("sequence_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64 or _digest(sequence, "sequence_sha256") != supplied:
        raise TemporalFeatureError("sequence digest verification failed")
    source = sequence.get("source")
    if not isinstance(source, Mapping):
        raise TemporalFeatureError("sequence source provenance is missing")
    if (source.get("source_type") != "c2_verified_generated_evidence" or
        source.get("category") != "attack" or source.get("corpus_class") != "adversarial"):
        raise TemporalFeatureError("C3.2 currently accepts adversarial attack sequences only")
    if source.get("split") != "dev":
        raise TemporalFeatureError("C3.2 currently accepts development sequences only")
    if source.get("execution_mode") != "offline_frame_generation_only" or source.get("external_execution_performed") is not False:
        raise TemporalFeatureError("unsupported source execution provenance")
    semantics = sequence.get("time_semantics")
    expected = {"clock_type": "relative_interval", "interval_field": "delta_t_sec",
                "cumulative_origin": "zero_at_start_of_recorded_sequence",
                "absolute_timestamps_synthesized": False, "first_frame_interval_included": True}
    if not isinstance(semantics, Mapping) or dict(semantics) != expected:
        raise TemporalFeatureError("unsupported or modified time semantics")
    events = sequence.get("events")
    if not isinstance(events, list) or not events or sequence.get("event_count") != len(events):
        raise TemporalFeatureError("sequence event count is invalid")
    required = {"sequence_index", "source_step", "relative_interval_sec", "cumulative_elapsed_sec",
                "time_basis", "absolute_timestamp_available", "device_id", "topic", "operation",
                "dropped", "ground_truth_label", "payload_len", "payload_sha256"}
    result: list[dict[str, Any]] = []
    cumulative = 0.0
    previous = -1.0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != required:
            raise TemporalFeatureError(f"event {index} does not match C3.1 event contract")
        if _integer(event["sequence_index"], "sequence_index") != index or _integer(event["source_step"], "source_step") != index:
            raise TemporalFeatureError("event sequence indices and source steps must be contiguous")
        if event["time_basis"] != "accumulated_delta_t_sec" or event["absolute_timestamp_available"] is not False:
            raise TemporalFeatureError("unsupported event time basis")
        for field in ("device_id", "topic", "operation"):
            if not isinstance(event[field], str) or not event[field].strip():
                raise TemporalFeatureError(f"{field} must be a non-empty string")
        if event["ground_truth_label"] != "attack" or not isinstance(event["dropped"], bool):
            raise TemporalFeatureError("event label or dropped flag is invalid")
        _integer(event["payload_len"], "payload_len")
        if not isinstance(event["payload_sha256"], str) or len(event["payload_sha256"]) != 64:
            raise TemporalFeatureError("payload digest is malformed")
        interval = _number(event["relative_interval_sec"], "relative_interval_sec")
        cumulative += interval
        timestamp = _number(event["cumulative_elapsed_sec"], "cumulative_elapsed_sec")
        if abs(timestamp - round(cumulative, 9)) > 1e-9 or timestamp < previous:
            raise TemporalFeatureError("cumulative event time is inconsistent or unordered")
        previous = timestamp
        result.append(dict(event))
    return result, previous


def _transitions(events: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter((a[field], b[field]) for a, b in zip(events, events[1:]) if a[field] != b[field])
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    pairs = max(0, len(events) - 1)
    return {"change_count": sum(counts.values()),
            "change_rate_per_adjacent_pair": sum(counts.values()) / pairs if pairs else 0.0,
            "unique_transition_count": len(counts),
            "top_transitions": [{"from": a, "to": b, "count": n} for (a, b), n in ordered[:MAX_TRANSITIONS_REPORTED]],
            "top_transitions_truncated": len(ordered) > MAX_TRANSITIONS_REPORTED}


def _windows(events: list[dict[str, Any]], width: float, elapsed: float) -> dict[str, Any]:
    if elapsed == 0:
        return {"window_seconds": width, "window_count": 0, "events_per_window": [],
                "rate_events_per_second": _stats([]),
                "rate_undefined_reason": "zero_elapsed_duration",
                "boundary_rule": "left_closed_right_open; final sequence endpoint included in final window"}
    count = max(1, math.ceil(elapsed / width))
    bins = [0] * count
    for event in events:
        index = min(int(event["cumulative_elapsed_sec"] // width), count - 1)
        bins[index] += 1
    rates = []
    for index, amount in enumerate(bins):
        actual_width = width if index < count - 1 else max(elapsed - index * width, 0.0)
        rates.append(amount / (actual_width if actual_width > 0 else width))
    return {"window_seconds": width, "window_count": count, "events_per_window": bins,
            "rate_events_per_second": _stats(rates),
            "boundary_rule": "left_closed_right_open; final sequence endpoint included in final window"}


def _bursts(events: list[dict[str, Any]], gap: float, minimum: int) -> dict[str, Any]:
    times = [event["cumulative_elapsed_sec"] for event in events]
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > gap:
            if index - start >= minimum:
                segments.append((start, index - 1))
            start = index
    if len(times) - start >= minimum:
        segments.append((start, len(times) - 1))
    durations = [times[end] - times[start] for start, end in segments]
    sizes = [end - start + 1 for start, end in segments]
    intensity_values = [
        (size - 1) / duration
        for size, duration in zip(sizes, durations)
        if duration > 0
    ]
    spacing = [times[next_start] - times[prev_end]
               for (_, prev_end), (next_start, _) in zip(segments, segments[1:])]
    segment_records = []
    for (start_index, end_index), size, duration in zip(segments, sizes, durations):
        measurable = duration > 0
        segment_records.append({
            "start_sequence_index": start_index,
            "end_sequence_index": end_index,
            "event_count": size,
            "duration_seconds": duration,
            "intensity_events_per_second": ((size - 1) / duration) if measurable else None,
            "intensity_undefined_reason": None if measurable else "zero_duration",
        })
    return {"gap_threshold_seconds": gap, "minimum_events": minimum, "burst_count": len(segments),
            "total_burst_events": sum(sizes), "event_count_stats": _stats([float(x) for x in sizes]),
            "duration_seconds_stats": _stats(durations),
            "intensity_events_per_second_stats": _stats(intensity_values),
            "intensity_measurable_burst_count": len(intensity_values),
            "intensity_undefined_burst_count": len(segments) - len(intensity_values),
            "inter_burst_spacing_seconds_stats": _stats(spacing),
            "segments": segment_records,
            "definition": "maximal consecutive event groups joined by gaps <= threshold; groups below minimum_events are not bursts"}


def extract_temporal_features(sequence: Mapping[str, Any], *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        burst_gap_threshold_seconds: float = DEFAULT_BURST_GAP_SECONDS,
        min_burst_events: int = DEFAULT_MIN_BURST_EVENTS) -> dict[str, Any]:
    """Extract deterministic features from one digest-verified C3.1 sequence."""
    window = _number(window_seconds, "window_seconds", positive=True)
    gap = _number(burst_gap_threshold_seconds, "burst_gap_threshold_seconds")
    minimum = _integer(min_burst_events, "min_burst_events", minimum=2)
    events, elapsed = _validate(sequence)
    intervals = [float(e["relative_interval_sec"]) for e in events]
    sizes = [float(e["payload_len"]) for e in events]
    count = len(events)
    dropped = sum(e["dropped"] for e in events)
    result = {
        "schema_version": TEMPORAL_FEATURE_SCHEMA,
        "source": {"source_type": "c3_verified_temporal_sequence",
                   "sequence_sha256": sequence["sequence_sha256"],
                   "run_id": sequence["source"].get("run_id"),
                   "scenario_id": sequence["source"].get("scenario_id"),
                   "family": sequence["source"].get("family"),
                   "variation_id": sequence["source"].get("variation_id"),
                   "split": sequence["source"].get("split"),
                   "corpus_class": sequence["source"].get("corpus_class"),
                   "execution_mode": sequence["source"].get("execution_mode"),
                   "external_execution_performed": False},
        "configuration": {"window_seconds": window, "burst_gap_threshold_seconds": gap,
                          "min_burst_events": minimum, "percentile_method": "nearest_rank",
                          "standard_deviation": "population"},
        "sequence_quality": {"event_count": count, "elapsed_seconds": elapsed,
                             "dropped_event_count": dropped, "dropped_event_ratio": dropped / count,
                             "payload_size_bytes_stats": _stats(sizes),
                             "zero_interval_count": sum(x == 0 for x in intervals),
                             "time_basis": "relative_interval", "absolute_timestamps_available": False},
        "inter_arrival": {"sample_semantics": "C3.1 relative intervals including first frame interval from sequence origin",
                           "interval_seconds_stats": _stats(intervals)},
        "traffic_rate": {"overall_events_per_second": count / elapsed if elapsed > 0 else None,
                         "rate_undefined_reason": "zero_elapsed_duration" if elapsed == 0 else None,
                         "fixed_windows": _windows(events, window, elapsed)},
        "bursts": _bursts(events, gap, minimum),
        "transitions": {"adjacent_pair_count": max(0, count - 1),
                        "device_id": _transitions(events, "device_id"),
                        "topic": _transitions(events, "topic"),
                        "operation": _transitions(events, "operation")}}
    result["feature_sha256"] = _digest(result, "feature_sha256")
    return result
