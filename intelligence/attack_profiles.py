from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping

PROFILE_SCHEMA_VERSION = "c1.attack-profile.v1"
_CLASS_TO_SCENARIO = {
    "adversarial": "attack",
    "protocol_violation": "protocol",
    "hardware_fault": "hardware_fault",
}
_REQUIRED_LABEL = {
    "adversarial": "confirmed_attack",
    "hardware_fault": "confirmed_hardware_fault",
}
_REQUIRED_SOURCE = {
    "adversarial": {"attack_lab", "shadow_sink"},
    "protocol_violation": {"ids_alert"},
    "hardware_fault": {"fault_lab"},
}


class AttackProfileError(ValueError):
    pass


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttackProfileError("profile contains non-canonical values") from exc


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttackProfileError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise AttackProfileError(f"{field} is outside its valid range")
    return number


def _entropy(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = Counter(payload)
    size = float(len(payload))
    return -sum((count / size) * math.log2(count / size) for count in counts.values())


def _metadata(record: Mapping[str, Any], corpus_class: str, source: str) -> dict[str, Any]:
    raw = record.get("xai_contrib_json")
    if raw in (None, ""):
        # AdmissionGate protocol captures and legacy shadow-sink captures do not
        # necessarily carry per-generated-frame metadata. Their verified
        # scenario manifest and observed gateway timestamps remain usable.
        if corpus_class == "protocol_violation" or source == "shadow_sink":
            return {}
        raise AttackProfileError(
            f"capture {record.get('capture_id')} is missing required lab event metadata"
        )
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise AttackProfileError("capture metadata is not valid UTF-8") from exc
    if not isinstance(raw, str):
        raise AttackProfileError("capture metadata must be a JSON string")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AttackProfileError(
            f"capture {record.get('capture_id')} has malformed event metadata"
        ) from exc
    if not isinstance(parsed, dict):
        raise AttackProfileError("capture event metadata must be a JSON object")
    if source in {"attack_lab", "fault_lab"}:
        for key in ("family", "step", "operation", "delta_t_sec"):
            if key not in parsed:
                raise AttackProfileError(
                    f"capture {record.get('capture_id')} metadata is missing {key!r}"
                )
        if not isinstance(parsed["family"], str) or not parsed["family"].strip():
            raise AttackProfileError("event family metadata must be a non-empty string")
        step = parsed["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise AttackProfileError("event step must be a non-negative integer")
        if not isinstance(parsed["operation"], str) or not parsed["operation"].strip():
            raise AttackProfileError("event operation must be a non-empty string")
        _finite_number(parsed["delta_t_sec"], "delta_t_sec", positive=True)
    return parsed


def _prepare_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise AttackProfileError("corpus entries must be mappings")
    capture_id = record.get("capture_id")
    if isinstance(capture_id, bool) or not isinstance(capture_id, int) or capture_id <= 0:
        raise AttackProfileError("capture_id must be a positive integer")
    corpus_class = record.get("corpus_class")
    if corpus_class not in _CLASS_TO_SCENARIO:
        raise AttackProfileError(f"capture {capture_id} has an unsupported corpus class")
    scenario_type = record.get("scenario_type")
    if scenario_type != _CLASS_TO_SCENARIO[corpus_class]:
        raise AttackProfileError(f"capture {capture_id} class/scenario mismatch")
    scenario_id = record.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise AttackProfileError(f"capture {capture_id} has no scenario_id")
    manifest_raw = record.get("scenario_manifest_json")
    try:
        manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AttackProfileError(f"scenario {scenario_id} has malformed manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise AttackProfileError(f"scenario {scenario_id} manifest must be a JSON object")
    family = manifest.get("family")
    if not isinstance(family, str) or not family.strip():
        # Legacy manifests predate the explicit `family` field. Only use their
        # stable scenario identifier for narrowly recognized legacy evidence.
        # The namespace records that this is a scenario-level fallback, not a
        # validated reusable attack-family taxonomy.
        legacy_scenario = manifest.get("scenario")
        if not isinstance(legacy_scenario, str) or not legacy_scenario.strip():
            raise AttackProfileError(f"scenario {scenario_id} has no valid family")
        if corpus_class == "protocol_violation":
            family = f"protocol:{legacy_scenario.strip()}"
        elif corpus_class == "adversarial" and record.get("capture_source") == "shadow_sink":
            family = f"adversarial:legacy-shadow:{legacy_scenario.strip()}"
        else:
            raise AttackProfileError(f"scenario {scenario_id} has no valid family")
    provenance = record.get("provenance_sha256")
    if not isinstance(provenance, str) or len(provenance) != 64:
        raise AttackProfileError(f"scenario {scenario_id} has malformed provenance digest")
    try:
        int(provenance, 16)
    except ValueError as exc:
        raise AttackProfileError(f"scenario {scenario_id} has malformed provenance digest") from exc

    label = record.get("ground_truth_label")
    if corpus_class in _REQUIRED_LABEL and label != _REQUIRED_LABEL[corpus_class]:
        raise AttackProfileError(
            f"capture {capture_id} ground-truth label does not match {corpus_class}"
        )
    # Protocol captures intentionally carry no row-level ground-truth label.
    if corpus_class == "protocol_violation" and label not in (None, ""):
        raise AttackProfileError(
            f"protocol capture {capture_id} must not be relabeled as an attack"
        )
    source = record.get("capture_source")
    if source not in _REQUIRED_SOURCE[corpus_class]:
        raise AttackProfileError(f"capture {capture_id} source does not match its corpus class")
    ts = record.get("gateway_ts_ns")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
        raise AttackProfileError(f"capture {capture_id} has invalid gateway timestamp")
    device_id = record.get("device_id")
    topic = record.get("original_topic")
    if not isinstance(device_id, str) or not device_id:
        raise AttackProfileError(f"capture {capture_id} has invalid device_id")
    if not isinstance(topic, str) or not topic:
        raise AttackProfileError(f"capture {capture_id} has invalid original_topic")

    raw = record.get("raw_payload")
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise AttackProfileError(f"capture {capture_id} has no byte payload")
    payload = bytes(raw)
    declared_len = record.get("payload_len")
    if isinstance(declared_len, bool) or not isinstance(declared_len, int) or declared_len != len(payload):
        raise AttackProfileError(f"capture {capture_id} payload length is inconsistent")
    digest = record.get("payload_sha256")
    if not isinstance(digest, (bytes, bytearray, memoryview)) or len(digest) != 32:
        raise AttackProfileError(f"capture {capture_id} payload digest is malformed")
    if hashlib.sha256(payload).digest() != bytes(digest):
        raise AttackProfileError(f"capture {capture_id} payload digest mismatch")

    metadata = _metadata(record, corpus_class, source)
    if metadata.get("family", family) != family:
        raise AttackProfileError(f"capture {capture_id} family differs from its scenario manifest")
    reason = str(record.get("reason_code") or "")
    if corpus_class == "protocol_violation":
        expected = manifest.get("expected_reasons")
        # Older Stage-B protocol manifests used the singular `expected_reason`.
        # Accept it only as a one-element equivalent; provenance remains checked
        # by LumenStorageManager before this profile builder receives the record.
        if expected is None:
            legacy_expected = manifest.get("expected_reason")
            if isinstance(legacy_expected, str) and legacy_expected.strip():
                expected = [legacy_expected.strip()]
        if not isinstance(expected, list) or reason not in expected:
            raise AttackProfileError(
                f"protocol capture {capture_id} reason is not authorized by its manifest"
            )
    return {
        "capture_id": capture_id, "scenario_id": scenario_id,
        "scenario_type": scenario_type, "corpus_class": corpus_class,
        "family": family, "runner_version": record.get("runner_version"),
        "provenance_sha256": provenance.lower(), "gateway_ts_ns": ts,
        "device_id": device_id, "topic": topic, "capture_source": source,
        "reason_code": reason, "payload": payload,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": metadata,
    }


def _profile_for_scenario(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    scenario_id = first["scenario_id"]
    consistency = (
        "scenario_type", "corpus_class", "family", "runner_version", "provenance_sha256"
    )
    for item in records[1:]:
        if any(item[key] != first[key] for key in consistency):
            raise AttackProfileError(f"scenario {scenario_id} has inconsistent provenance")
    records.sort(key=lambda item: (item["gateway_ts_ns"], item["capture_id"]))
    steps = [item["metadata"]["step"] for item in records if "step" in item["metadata"]]
    if steps and (
        len(steps) != len(records) or any(b <= a for a, b in zip(steps, steps[1:]))
    ):
        raise AttackProfileError(f"scenario {scenario_id} has missing or non-increasing event steps")
    timestamps = [item["gateway_ts_ns"] for item in records]
    inter_arrival = [
        (timestamps[i] - timestamps[i - 1]) / 1_000_000_000
        for i in range(1, len(timestamps))
    ]
    if any(value < 0 for value in inter_arrival):
        raise AttackProfileError(f"scenario {scenario_id} event timestamps are not ordered")

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None, "population_stddev": None}
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {
            "min": round(min(values), 9), "max": round(max(values), 9),
            "mean": round(mean, 9), "population_stddev": round(math.sqrt(variance), 9),
        }

    topics = Counter(item["topic"] for item in records)
    operations = Counter(item["metadata"].get("operation", "unknown") for item in records)
    reasons = Counter(item["reason_code"] for item in records if item["reason_code"])
    lengths = [float(len(item["payload"])) for item in records]
    entropies = [_entropy(item["payload"]) for item in records]
    events = []
    for item in records:
        metadata = item["metadata"]
        events.append({
            "capture_id": item["capture_id"],
            "gateway_ts_ns": item["gateway_ts_ns"],
            "device_id": item["device_id"],
            "topic": item["topic"],
            "operation": metadata.get("operation", "unknown"),
            "step": metadata.get("step"),
            "declared_delta_t_sec": (
                round(float(metadata["delta_t_sec"]), 9)
                if "delta_t_sec" in metadata else None
            ),
            "payload_len": len(item["payload"]),
            "payload_entropy": round(_entropy(item["payload"]), 9),
            "payload_sha256": item["payload_sha256"],
            "reason_code": item["reason_code"],
        })
    profile = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "scenario": {
            "scenario_id": scenario_id, "scenario_type": first["scenario_type"],
            "corpus_class": first["corpus_class"], "family": first["family"],
            "runner_version": first["runner_version"],
            "provenance_sha256": first["provenance_sha256"],
        },
        "evidence": {
            "capture_count": len(records),
            "capture_ids": [item["capture_id"] for item in records],
            "payload_sha256": [item["payload_sha256"] for item in records],
        },
        "behavior": {
            "device_counts": dict(sorted(Counter(item["device_id"] for item in records).items())),
            "topic_counts": dict(sorted(topics.items())),
            "operation_counts": dict(sorted(operations.items())),
            "reason_code_counts": dict(sorted(reasons.items())),
            "payload_length": stats(lengths),
            "payload_entropy_bits": stats(entropies),
            "observed_inter_arrival_sec": stats(inter_arrival),
        },
        "events": events,
    }
    profile["profile_sha256"] = hashlib.sha256(_canonical_json(profile)).hexdigest()
    return profile


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttackProfileError("profile contains non-canonical values") from exc


def build_profiles_from_captures(
    captures: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # Input must come from the existing verified corpus loader.
    prepared = [_prepare_record(record) for record in captures]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        grouped.setdefault(item["scenario_id"], []).append(item)
    return [_profile_for_scenario(grouped[key]) for key in sorted(grouped)]


def build_attack_profiles(
    storage: Any,
    corpus_classes: tuple[str, ...] = (
        "adversarial", "protocol_violation", "hardware_fault"
    ),
) -> list[dict[str, Any]]:
    loader = getattr(storage, "load_adversarial_regression_corpus", None)
    if not callable(loader):
        raise TypeError("storage must expose load_adversarial_regression_corpus()")
    captures = loader(corpus_classes=corpus_classes)
    if not isinstance(captures, list):
        raise AttackProfileError("verified corpus loader must return a list")
    return build_profiles_from_captures(captures)
