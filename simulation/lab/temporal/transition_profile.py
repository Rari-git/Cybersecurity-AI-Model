"""Deterministic transition profiles from verified clean baseline sessions."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from training.session_integrity import load_verified_clean_session_rows

CLEAN_TRANSITION_PROFILE_SCHEMA = "c3.clean-transition-profile.v1"
DEFAULT_RAPID_INTERVAL_SECONDS = 0.1
DEFAULT_NORMAL_INTERVAL_SECONDS = 1.0
_SHA256_HEX = frozenset("0123456789abcdef")


class TransitionProfileError(ValueError):
    """A clean transition profile input is invalid or inconsistent."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransitionProfileError("profile contains non-canonical values") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _finite_non_negative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransitionProfileError(f"{field} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise TransitionProfileError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise TransitionProfileError(f"{field} must be finite and non-negative")
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TransitionProfileError(f"{field} must be an integer >= {minimum}")
    return value


def _payload_bucket(size: int) -> str:
    if size == 0:
        return "empty"
    if size <= 16:
        return "tiny"
    if size <= 64:
        return "small"
    if size <= 256:
        return "medium"
    if size <= 1024:
        return "large"
    return "very_large"


def _interval_class(seconds: float, rapid: float, normal: float) -> str:
    if seconds == 0:
        return "simultaneous"
    if seconds <= rapid:
        return "rapid"
    if seconds <= normal:
        return "normal"
    return "delayed"


def _state_id(descriptor: Mapping[str, Any]) -> str:
    # Keep the full digest so profile state identity has no deliberate truncation.
    return "state_" + hashlib.sha256(_canonical(descriptor)).hexdigest()


def _session_streams(rows: Any, session_id: str) -> dict[tuple[str, int, str], list[tuple[int, int, int]]]:
    if not isinstance(rows, list) or not rows:
        raise TransitionProfileError(f"approved session has no telemetry rows: {session_id}")
    streams: dict[tuple[str, int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TransitionProfileError(f"telemetry row is malformed: {session_id}")
        device_id = row.get("device_id")
        topic = row.get("topic")
        if not isinstance(device_id, str) or not device_id.strip():
            raise TransitionProfileError(f"device_id is invalid: {session_id}")
        if not isinstance(topic, str) or not topic.strip():
            raise TransitionProfileError(f"topic is invalid: {session_id}")
        event_id = _integer(row.get("event_id"), "event_id", 1)
        generation = _integer(row.get("generation"), "generation")
        gateway_ts_ns = _integer(row.get("gateway_ts_ns"), "gateway_ts_ns")
        payload_len = _integer(row.get("payload_len"), "payload_len")
        streams[(device_id, generation, topic)].append((event_id, gateway_ts_ns, payload_len))
    return streams


def build_clean_transition_profile(
    storage: Any,
    capture_run_split: Mapping[str, Any],
    curation_digests: Mapping[str, str],
    *,
    rapid_interval_seconds: float = DEFAULT_RAPID_INTERVAL_SECONDS,
    normal_interval_seconds: float = DEFAULT_NORMAL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Build a descriptive profile from the training partition of a clean split.

    ``capture_run_split`` must be the result of
    ``training.train_local_lstm.build_capture_run_split``. The profile uses only
    its training sessions; validation and held-out sessions are recorded as
    provenance and never read as profile data.

    Only fields present in curated ``raw_events`` are modeled: device identity
    generation, topic, payload size, and the gateway inter-arrival category.
    Operation and dropped-event status are intentionally omitted because the
    clean telemetry table does not preserve them. Session and stream boundaries
    reset timing; no transition is formed across either boundary. This function
    returns no anomaly score, attack label, or policy recommendation.
    """
    rapid = _finite_non_negative(rapid_interval_seconds, "rapid_interval_seconds")
    normal = _finite_non_negative(normal_interval_seconds, "normal_interval_seconds")
    if rapid <= 0 or normal <= rapid:
        raise TransitionProfileError(
            "thresholds must satisfy 0 < rapid_interval_seconds < normal_interval_seconds"
        )
    if not isinstance(capture_run_split, Mapping):
        raise TransitionProfileError("capture_run_split must be a capture-run split mapping")

    partitions: dict[str, tuple[str, ...]] = {}
    for key in ("train_session_ids", "validation_session_ids", "heldout_session_ids"):
        values = capture_run_split.get(key)
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or not values:
            raise TransitionProfileError(f"capture-run split {key} must be a non-empty sequence")
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise TransitionProfileError(f"capture-run split {key} contains an invalid session ID")
        if len(set(values)) != len(values):
            raise TransitionProfileError(f"capture-run split {key} contains duplicate session IDs")
        partitions[key] = tuple(values)

    train_ids = set(partitions["train_session_ids"])
    validation_ids = set(partitions["validation_session_ids"])
    heldout_ids = set(partitions["heldout_session_ids"])
    if train_ids & validation_ids or train_ids & heldout_ids or validation_ids & heldout_ids:
        raise TransitionProfileError("training, validation, and held-out sessions must be disjoint")
    all_split_ids = train_ids | validation_ids | heldout_ids

    run_map = capture_run_split.get("capture_run_ids_by_session")
    if not isinstance(run_map, Mapping) or set(run_map) != all_split_ids:
        raise TransitionProfileError("capture-run provenance must exactly cover all split sessions")
    run_sets: dict[str, set[str]] = {}
    for key, session_set in (("train_session_ids", train_ids),
                             ("validation_session_ids", validation_ids),
                             ("heldout_session_ids", heldout_ids)):
        values = set()
        for session_id in session_set:
            run_id = run_map.get(session_id)
            if (not isinstance(run_id, str) or len(run_id) != 32
                    or run_id != run_id.lower()
                    or any(char not in _SHA256_HEX for char in run_id)):
                raise TransitionProfileError(f"capture-run ID is invalid for session {session_id}")
            values.add(run_id)
        run_sets[key] = values
    if (run_sets["train_session_ids"] & run_sets["validation_session_ids"]
            or run_sets["train_session_ids"] & run_sets["heldout_session_ids"]
            or run_sets["validation_session_ids"] & run_sets["heldout_session_ids"]):
        raise TransitionProfileError("capture runs must not cross training/validation/held-out partitions")

    ordered_ids = sorted(train_ids)
    if not isinstance(curation_digests, Mapping):
        raise TransitionProfileError("curation_digests must map training session IDs to SHA-256 digests")
    digests = dict(curation_digests)
    if set(digests) != train_ids:
        raise TransitionProfileError("curation digest IDs must exactly match training session IDs")
    for session_id, digest in digests.items():
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in _SHA256_HEX for char in digest.lower())):
            raise TransitionProfileError(f"curation digest is invalid: {session_id}")
        digests[session_id] = digest.lower()

    # The shared loader verifies approved-clean status, curation decisions,
    # payload hashes, and the exact stored dataset digest in a read transaction.
    verified_rows = load_verified_clean_session_rows(storage, ordered_ids, digests)
    if not isinstance(verified_rows, Mapping) or set(verified_rows) != set(ordered_ids):
        raise TransitionProfileError("clean-session verifier returned an unexpected session set")

    state_descriptors: dict[str, dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    state_sessions: dict[str, set[str]] = defaultdict(set)
    transition_counts: Counter[tuple[str, str, str]] = Counter()
    transition_sessions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    stream_segment_count = 0
    event_count = 0

    for session_id in ordered_ids:
        streams = _session_streams(verified_rows[session_id], session_id)
        for stream_key in sorted(streams):
            device_id, generation, topic = stream_key
            records = sorted(streams[stream_key], key=lambda record: record[0])
            if len({record[0] for record in records}) != len(records):
                raise TransitionProfileError(f"duplicate event IDs in stream: {session_id}")
            stream_segment_count += 1
            previous_state: str | None = None
            previous_timestamp: int | None = None
            for _event_id, timestamp, payload_len in records:
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise TransitionProfileError(
                        f"gateway timestamp moved backwards within stream: {session_id}"
                    )
                descriptor = {
                    "device_id": device_id,
                    "generation": generation,
                    "topic": topic,
                    "payload_size_bucket": _payload_bucket(payload_len),
                }
                current_state = _state_id(descriptor)
                prior_descriptor = state_descriptors.setdefault(current_state, descriptor)
                if prior_descriptor != descriptor:
                    raise TransitionProfileError("state digest collision detected")
                state_counts[current_state] += 1
                state_sessions[current_state].add(session_id)
                event_count += 1

                if previous_state is not None and previous_timestamp is not None:
                    elapsed = (timestamp - previous_timestamp) / 1_000_000_000.0
                    interval_kind = _interval_class(elapsed, rapid, normal)
                    edge = (previous_state, current_state, interval_kind)
                    transition_counts[edge] += 1
                    transition_sessions[edge].add(session_id)
                previous_state = current_state
                previous_timestamp = timestamp

    outgoing_counts: Counter[str] = Counter()
    for (source_state, _target_state, _interval_kind), count in transition_counts.items():
        outgoing_counts[source_state] += count

    states = [
        {
            "state_id": state_id,
            "event_count": state_counts[state_id],
            "event_frequency": state_counts[state_id] / event_count,
            "session_count": len(state_sessions[state_id]),
            "descriptor": state_descriptors[state_id],
        }
        for state_id in sorted(state_descriptors)
    ]
    transitions = [
        {
            "from_state_id": source_state,
            "to_state_id": target_state,
            "inter_arrival_class": interval_kind,
            "count": count,
            "outgoing_probability": count / outgoing_counts[source_state],
            "session_count": len(transition_sessions[(source_state, target_state, interval_kind)]),
        }
        for (source_state, target_state, interval_kind), count in sorted(transition_counts.items())
    ]

    profile: dict[str, Any] = {
        "schema_version": CLEAN_TRANSITION_PROFILE_SCHEMA,
        "source": {
            "source_type": "approved_clean_baseline_sessions",
            "curation_digests": [
                {"session_id": session_id, "dataset_sha256": digests[session_id]}
                for session_id in ordered_ids
            ],
            "capture_run_split": {
                "training_session_ids": ordered_ids,
                "validation_session_ids": sorted(validation_ids),
                "heldout_session_ids": sorted(heldout_ids),
                "training_capture_run_ids": sorted(run_sets["train_session_ids"]),
                "validation_capture_run_ids": sorted(run_sets["validation_session_ids"]),
                "heldout_capture_run_ids": sorted(run_sets["heldout_session_ids"]),
                "independence_unit": "capture_run_id",
            },
        },
        "configuration": {
            "rapid_interval_seconds": rapid,
            "normal_interval_seconds": normal,
            "clock_basis": "gateway_ts_ns_differences_within_session_and_stream",
            "absolute_timestamps_in_profile": False,
            "first_event_interval": "omitted_until_a_prior_event_exists",
            "session_boundary_rule": "never_transition_between_sessions",
            "stream_key": ["device_id", "generation", "topic"],
            "state_dimensions": ["device_id", "generation", "topic", "payload_size_bucket"],
            "transition_dimensions": ["from_state_id", "to_state_id", "inter_arrival_class"],
            "payload_size_boundaries_bytes": [0, 16, 64, 256, 1024],
            "operation_available": False,
            "dropped_event_status_available": False,
        },
        "summary": {
            "session_count": len(ordered_ids),
            "stream_segment_count": stream_segment_count,
            "event_count": event_count,
            "unique_state_count": len(states),
            "observed_transition_count": sum(transition_counts.values()),
            "unique_transition_count": len(transitions),
        },
        "states": states,
        "transitions": transitions,
        "interpretation": {
            "analysis_type": "descriptive_clean_behavior_transition_profile",
            "anomaly_score_generated": False,
            "maliciousness_inferred": False,
            "fault_diagnosis_generated": False,
            "policy_recommendation_generated": False,
        },
    }
    profile["profile_sha256"] = _digest(profile, "profile_sha256")
    return profile


def evaluate_clean_transition_profile(
    profile: Mapping[str, Any],
    storage: Any,
    validation_session_ids: Iterable[str],
    curation_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Measure profile coverage on the complete, disjoint validation partition.

    This reports observed state/transition coverage only. Unseen items are not
    called attacks or faults, and the profile is never modified. Held-out
    sessions are rejected so they remain available for a later frozen check.
    """
    if not isinstance(profile, Mapping) or profile.get("schema_version") != CLEAN_TRANSITION_PROFILE_SCHEMA:
        raise TransitionProfileError("unsupported clean transition profile schema")
    supplied_digest = profile.get("profile_sha256")
    if (not isinstance(supplied_digest, str) or len(supplied_digest) != 64
            or any(char not in _SHA256_HEX for char in supplied_digest.lower())
            or not hmac.compare_digest(_digest(profile, "profile_sha256"), supplied_digest.lower())):
        raise TransitionProfileError("profile digest verification failed")

    source = profile.get("source")
    if not isinstance(source, Mapping):
        raise TransitionProfileError("profile source provenance is missing")
    split = source.get("capture_run_split")
    if not isinstance(split, Mapping):
        raise TransitionProfileError("profile capture-run split provenance is missing")
    permitted_ids = split.get("validation_session_ids")
    if (not isinstance(permitted_ids, list) or not permitted_ids
            or any(not isinstance(item, str) or not item for item in permitted_ids)
            or len(set(permitted_ids)) != len(permitted_ids)):
        raise TransitionProfileError("profile validation session IDs are invalid")
    if isinstance(validation_session_ids, (str, bytes)):
        raise TransitionProfileError("validation_session_ids must be an iterable, not a string")
    requested = list(validation_session_ids)
    if (not requested or any(not isinstance(item, str) or not item for item in requested)
            or len(set(requested)) != len(requested)):
        raise TransitionProfileError("validation_session_ids must be non-empty and unique")
    ordered_ids = sorted(requested)
    if set(ordered_ids) != set(permitted_ids):
        raise TransitionProfileError("evaluation must use the complete validation partition only")
    if not isinstance(curation_digests, Mapping) or set(curation_digests) != set(ordered_ids):
        raise TransitionProfileError("validation curation digests must exactly match the validation partition")
    digests = {}
    for session_id in ordered_ids:
        digest = curation_digests[session_id]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in _SHA256_HEX for char in digest.lower())):
            raise TransitionProfileError(f"validation curation digest is invalid: {session_id}")
        digests[session_id] = digest.lower()

    configuration = profile.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TransitionProfileError("profile configuration is missing")
    rapid = _finite_non_negative(configuration.get("rapid_interval_seconds"), "rapid_interval_seconds")
    normal = _finite_non_negative(configuration.get("normal_interval_seconds"), "normal_interval_seconds")
    if rapid <= 0 or normal <= rapid:
        raise TransitionProfileError("profile interval thresholds are invalid")

    state_rows = profile.get("states")
    transition_rows = profile.get("transitions")
    if not isinstance(state_rows, list) or not isinstance(transition_rows, list):
        raise TransitionProfileError("profile states or transitions are malformed")
    known_states: set[str] = set()
    for item in state_rows:
        if not isinstance(item, Mapping) or not isinstance(item.get("descriptor"), Mapping):
            raise TransitionProfileError("profile state record is malformed")
        state_id = item.get("state_id")
        descriptor = dict(item["descriptor"])
        if not isinstance(state_id, str) or state_id != _state_id(descriptor) or state_id in known_states:
            raise TransitionProfileError("profile state identity is invalid or duplicated")
        known_states.add(state_id)

    known_transitions: dict[tuple[str, str, str], float] = {}
    probability_sums: Counter[str] = Counter()
    for item in transition_rows:
        if not isinstance(item, Mapping):
            raise TransitionProfileError("profile transition record is malformed")
        edge = (item.get("from_state_id"), item.get("to_state_id"), item.get("inter_arrival_class"))
        if (edge[0] not in known_states or edge[1] not in known_states
                or edge[2] not in {"simultaneous", "rapid", "normal", "delayed"}
                or edge in known_transitions):
            raise TransitionProfileError("profile transition identity is invalid or duplicated")
        count = _integer(item.get("count"), "transition count", 1)
        probability = _finite_non_negative(item.get("outgoing_probability"), "outgoing_probability")
        if probability <= 0 or probability > 1:
            raise TransitionProfileError("profile transition probability is outside (0, 1]")
        known_transitions[edge] = probability
        probability_sums[edge[0]] += probability
    if any(not math.isclose(value, 1.0, rel_tol=1e-9, abs_tol=1e-9)
           for value in probability_sums.values()):
        raise TransitionProfileError("profile outgoing transition probabilities do not sum to one")

    # Read only the validation partition and re-check its clean curation evidence.
    rows_by_session = load_verified_clean_session_rows(storage, ordered_ids, digests)
    if not isinstance(rows_by_session, Mapping) or set(rows_by_session) != set(ordered_ids):
        raise TransitionProfileError("clean-session verifier returned an unexpected validation session set")

    session_results = []
    total_events = total_known_states = 0
    total_transitions = total_known_transitions = 0
    known_surprisal_bits = []
    for session_id in ordered_ids:
        streams = _session_streams(rows_by_session[session_id], session_id)
        event_count = known_state_count = transition_count = known_transition_count = 0
        unseen_state_ids: set[str] = set()
        unseen_edges: set[tuple[str, str, str]] = set()
        for stream_key in sorted(streams):
            records = sorted(streams[stream_key], key=lambda record: record[0])
            if len({record[0] for record in records}) != len(records):
                raise TransitionProfileError(f"duplicate event IDs in validation stream: {session_id}")
            previous_state: str | None = None
            previous_timestamp: int | None = None
            for _event_id, timestamp, payload_len in records:
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise TransitionProfileError(
                        f"gateway timestamp moved backwards within validation stream: {session_id}"
                    )
                device_id, generation, topic = stream_key
                descriptor = {
                    "device_id": device_id,
                    "generation": generation,
                    "topic": topic,
                    "payload_size_bucket": _payload_bucket(payload_len),
                }
                current_state = _state_id(descriptor)
                event_count += 1
                if current_state in known_states:
                    known_state_count += 1
                else:
                    unseen_state_ids.add(current_state)
                if previous_state is not None and previous_timestamp is not None:
                    elapsed = (timestamp - previous_timestamp) / 1_000_000_000.0
                    edge = (previous_state, current_state, _interval_class(elapsed, rapid, normal))
                    transition_count += 1
                    probability = known_transitions.get(edge)
                    if probability is None:
                        unseen_edges.add(edge)
                    else:
                        known_transition_count += 1
                        known_surprisal_bits.append(-math.log2(probability))
                previous_state = current_state
                previous_timestamp = timestamp

        total_events += event_count
        total_known_states += known_state_count
        total_transitions += transition_count
        total_known_transitions += known_transition_count
        session_results.append({
            "session_id": session_id,
            "dataset_sha256": digests[session_id],
            "event_count": event_count,
            "known_state_event_count": known_state_count,
            "state_coverage_rate": known_state_count / event_count if event_count else 0.0,
            "unique_unseen_state_count": len(unseen_state_ids),
            "transition_count": transition_count,
            "known_transition_count": known_transition_count,
            "transition_coverage_rate": known_transition_count / transition_count if transition_count else 0.0,
            "unique_unseen_transition_count": len(unseen_edges),
        })

    result: dict[str, Any] = {
        "schema_version": "c3.clean-transition-validation.v1",
        "source": {
            "profile_sha256": supplied_digest,
            "evaluation_partition": "validation",
            "session_curation_digests": [
                {"session_id": session_id, "dataset_sha256": digests[session_id]}
                for session_id in ordered_ids
            ],
        },
        "summary": {
            "validation_session_count": len(ordered_ids),
            "event_count": total_events,
            "known_state_event_count": total_known_states,
            "state_coverage_rate": total_known_states / total_events if total_events else 0.0,
            "transition_count": total_transitions,
            "known_transition_count": total_known_transitions,
            "transition_coverage_rate": total_known_transitions / total_transitions if total_transitions else 0.0,
            "known_transition_mean_surprisal_bits": (
                sum(known_surprisal_bits) / len(known_surprisal_bits) if known_surprisal_bits else None
            ),
        },
        "sessions": session_results,
        "interpretation": {
            "analysis_type": "descriptive_validation_coverage",
            "unseen_behavior_is_not_an_attack_verdict": True,
            "anomaly_score_generated": False,
            "profile_updated": False,
            "policy_changed": False,
        },
    }
    result["validation_sha256"] = _digest(result, "validation_sha256")
    return result
