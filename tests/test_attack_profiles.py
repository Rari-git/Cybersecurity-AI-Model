from __future__ import annotations
import hashlib
import json
import pytest
from intelligence.attack_profiles import (
    PROFILE_SCHEMA_VERSION, AttackProfileError,
    build_attack_profiles, build_profiles_from_captures,
)


def _capture(
    capture_id=1, *, corpus_class="adversarial", scenario_type="attack",
    family="stealth_value_spoofing", ts=1_000_000_000, step=0,
    payload=b"350", label="confirmed_attack", source="attack_lab",
    reason="LAB_ATTACK_STEALTH_VALUE_SPOOFING", metadata="default",
):
    if metadata == "default":
        metadata = {
            "family": family, "step": step, "operation": "put", "delta_t_sec": 1.0,
        }
    manifest = {
        "family": family,
        "expected_reasons": [reason] if corpus_class == "protocol_violation" else [],
    }
    return {
        "capture_id": capture_id, "scenario_id": "scenario-1",
        "scenario_type": scenario_type, "corpus_class": corpus_class,
        "runner_version": "test-runner-v1",
        "scenario_manifest_json": json.dumps(manifest),
        "provenance_sha256": "a" * 64, "ground_truth_label": label,
        "capture_source": source, "gateway_ts_ns": ts,
        "device_id": "sensor-1", "original_topic": "home/sensors/gas/ppm",
        "reason_code": reason, "raw_payload": payload, "payload_len": len(payload),
        "payload_sha256": hashlib.sha256(payload).digest(),
        "xai_contrib_json": json.dumps(metadata) if metadata is not None else None,
    }


def test_profiles_are_deterministic_and_events_are_ordered():
    later = _capture(2, ts=3_000_000_000, step=1, payload=b"352")
    earlier = _capture(1, ts=2_000_000_000, step=0, payload=b"350")
    first = build_profiles_from_captures([later, earlier])
    second = build_profiles_from_captures([earlier, later])
    assert first == second
    profile = first[0]
    assert profile["profile_schema_version"] == PROFILE_SCHEMA_VERSION
    assert profile["evidence"]["capture_ids"] == [1, 2]
    assert profile["behavior"]["observed_inter_arrival_sec"]["mean"] == 1.0
    assert len(profile["profile_sha256"]) == 64


def test_profile_digest_changes_when_payload_changes():
    first = build_profiles_from_captures([_capture()])[0]
    second = build_profiles_from_captures([_capture(payload=b"351")])[0]
    assert first["profile_sha256"] != second["profile_sha256"]


def test_rejects_payload_tampering():
    record = _capture()
    record["raw_payload"] = b"tampered"
    with pytest.raises(AttackProfileError, match="payload length|payload digest"):
        build_profiles_from_captures([record])


def test_rejects_wrong_attack_label():
    with pytest.raises(AttackProfileError, match="ground-truth"):
        build_profiles_from_captures([_capture(label="normal")])


def test_protocol_uses_corpus_class_not_row_label():
    record = _capture(
        corpus_class="protocol_violation", scenario_type="protocol",
        family="authenticated_replay_duplicate_sequence", label=None,
        source="ids_alert", reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE", metadata=None,
    )
    profile = build_profiles_from_captures([record])[0]
    assert profile["scenario"]["corpus_class"] == "protocol_violation"
    assert profile["events"][0]["operation"] == "unknown"


def test_rejects_protocol_row_relabelled_as_attack():
    record = _capture(
        corpus_class="protocol_violation", scenario_type="protocol",
        family="authenticated_replay_duplicate_sequence", label="confirmed_attack",
        source="ids_alert", reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE", metadata=None,
    )
    with pytest.raises(AttackProfileError, match="must not be relabeled"):
        build_profiles_from_captures([record])


def test_legacy_shadow_sink_capture_can_be_profiled_without_frame_metadata():
    record = _capture(source="shadow_sink", metadata=None)
    profile = build_profiles_from_captures([record])[0]
    assert profile["events"][0]["operation"] == "unknown"
    assert profile["events"][0]["step"] is None


def test_rejects_missing_or_malformed_lab_metadata():
    with pytest.raises(AttackProfileError, match="missing required"):
        build_profiles_from_captures([_capture(metadata=None)])
    record = _capture()
    record["xai_contrib_json"] = "{broken"
    with pytest.raises(AttackProfileError, match="malformed event metadata"):
        build_profiles_from_captures([record])


def test_rejects_non_increasing_steps():
    records = [
        _capture(1, step=2),
        _capture(2, ts=2_000_000_000, step=1),
    ]
    with pytest.raises(AttackProfileError, match="non-increasing"):
        build_profiles_from_captures(records)


def test_storage_adapter_uses_existing_verified_loader():
    class Storage:
        def __init__(self):
            self.calls = []
        def load_adversarial_regression_corpus(self, *, corpus_classes):
            self.calls.append(corpus_classes)
            return [_capture()]
    storage = Storage()
    assert len(build_attack_profiles(storage, ("adversarial",))) == 1
    assert storage.calls == [("adversarial",)]


def test_adapter_rejects_incompatible_storage():
    with pytest.raises(TypeError, match="load_adversarial_regression_corpus"):
        build_attack_profiles(object())


def test_legacy_protocol_manifest_uses_namespaced_scenario_and_singular_reason():
    record = _capture(
        corpus_class="protocol_violation",
        scenario_type="protocol",
        family="authenticated_replay_duplicate_sequence",
        label=None,
        source="ids_alert",
        reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE",
        metadata=None,
    )
    manifest = json.loads(record["scenario_manifest_json"])
    manifest.pop("family")
    manifest.pop("expected_reasons")
    manifest["scenario"] = "authenticated-v6-duplicate-sequence"
    manifest["expected_reason"] = "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
    record["scenario_manifest_json"] = json.dumps(manifest)

    profile = build_profiles_from_captures([record])[0]

    assert profile["scenario"]["family"] == (
        "protocol:authenticated-v6-duplicate-sequence"
    )
    assert profile["scenario"]["corpus_class"] == "protocol_violation"
    assert profile["events"][0]["operation"] == "unknown"


def test_legacy_protocol_manifest_without_scenario_identifier_is_rejected():
    record = _capture(
        corpus_class="protocol_violation",
        scenario_type="protocol",
        family="authenticated_replay_duplicate_sequence",
        label=None,
        source="ids_alert",
        reason="REPLAY_OR_OUT_OF_ORDER_SEQUENCE",
        metadata=None,
    )
    manifest = json.loads(record["scenario_manifest_json"])
    manifest.pop("family")
    manifest.pop("expected_reasons")
    manifest["expected_reason"] = "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
    record["scenario_manifest_json"] = json.dumps(manifest)

    with pytest.raises(AttackProfileError, match="no valid family"):
        build_profiles_from_captures([record])


def test_attack_manifest_without_family_does_not_use_protocol_fallback():
    record = _capture()
    manifest = json.loads(record["scenario_manifest_json"])
    manifest.pop("family")
    manifest["scenario"] = "some-attack-scenario"
    record["scenario_manifest_json"] = json.dumps(manifest)

    with pytest.raises(AttackProfileError, match="no valid family"):
        build_profiles_from_captures([record])


def test_legacy_shadow_sink_manifest_uses_namespaced_scenario_fallback():
    record = _capture(
        corpus_class="adversarial",
        scenario_type="attack",
        label="confirmed_attack",
        source="shadow_sink",
        reason="SHADOW_QUARANTINE_ACTIVE",
        metadata={},
    )
    manifest = json.loads(record["scenario_manifest_json"])
    manifest.pop("family")
    manifest["scenario"] = "controlled-sensor-traffic-after-gateway-quarantine"
    record["scenario_manifest_json"] = json.dumps(manifest)

    profile = build_profiles_from_captures([record])[0]

    assert profile["scenario"]["family"] == (
        "adversarial:legacy-shadow:controlled-sensor-traffic-after-gateway-quarantine"
    )
    assert profile["scenario"]["corpus_class"] == "adversarial"
    assert profile["events"][0]["operation"] == "unknown"
    assert profile["events"][0]["step"] is None


def test_attack_lab_manifest_without_family_is_not_given_legacy_shadow_fallback():
    record = _capture(
        corpus_class="adversarial",
        scenario_type="attack",
        label="confirmed_attack",
        source="attack_lab",
        metadata=None,
    )
    manifest = json.loads(record["scenario_manifest_json"])
    manifest.pop("family")
    manifest["scenario"] = "some-attack-scenario"
    record["scenario_manifest_json"] = json.dumps(manifest)

    with pytest.raises(AttackProfileError, match="no valid family"):
        build_profiles_from_captures([record])
