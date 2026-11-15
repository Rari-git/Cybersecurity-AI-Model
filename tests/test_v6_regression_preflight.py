from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.adaptive_retrainer import (
    evaluate_regression_evidence_preflight,
    run_adaptive_cycle,
)


class _EvidenceStorage:
    def __init__(self, rows):
        self.rows = list(rows)
        self.flush_calls = 0

    def flush_all(self):
        self.flush_calls += 1

    def load_adversarial_regression_corpus(self, _classes):
        return list(self.rows)


def _row(corpus_class: str, scenario_id: str, capture_id: int, *, family: str | None = None):
    scenario_type = {
        "adversarial": "attack",
        "protocol_violation": "protocol",
        "hardware_fault": "hardware_fault",
    }[corpus_class]
    family_value = family or f"{corpus_class}:{scenario_id}"
    return {
        "corpus_class": corpus_class,
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "scenario_manifest_json": json.dumps({"family": family_value}),
        "capture_id": capture_id,
        "provenance_sha256": "a" * 64,
    }


def _sufficient_rows():
    rows = []
    capture_id = 1
    for corpus_class in ("adversarial", "protocol_violation", "hardware_fault"):
        for index in range(2):
            rows.append(_row(corpus_class, f"{corpus_class}-scenario-{index}", capture_id))
            capture_id += 1
    return rows


def test_regression_preflight_requires_two_independent_scenarios_per_class():
    storage = _EvidenceStorage([
        _row("adversarial", "attack-1", 1),
        _row("protocol_violation", "protocol-1", 2),
    ])
    result = evaluate_regression_evidence_preflight(storage)
    assert result["ready"] is False
    assert result["scenario_counts"] == {
        "adversarial": 1,
        "protocol_violation": 1,
        "hardware_fault": 0,
    }
    assert result["deficits"] == {
        "adversarial": 1,
        "protocol_violation": 1,
        "hardware_fault": 2,
    }
    assert storage.flush_calls == 1


def test_regression_preflight_counts_distinct_scenarios_not_capture_rows():
    rows = [
        _row("adversarial", "attack-same", 1),
        _row("adversarial", "attack-same", 2),
        _row("protocol_violation", "protocol-1", 3),
        _row("protocol_violation", "protocol-2", 4),
        _row("hardware_fault", "fault-1", 5),
        _row("hardware_fault", "fault-2", 6),
    ]
    result = evaluate_regression_evidence_preflight(_EvidenceStorage(rows))
    assert result["ready"] is False
    assert result["scenario_counts"]["adversarial"] == 1
    assert result["capture_counts"]["adversarial"] == 2


def test_regression_preflight_accepts_two_provenance_valid_scenarios_per_class():
    result = evaluate_regression_evidence_preflight(_EvidenceStorage(_sufficient_rows()))
    assert result["ready"] is True
    assert result["deficits"] == {}
    assert result["scenario_counts"] == {
        "adversarial": 2,
        "protocol_violation": 2,
        "hardware_fault": 2,
    }


def test_regression_preflight_requires_distinct_families_not_just_new_scenario_ids():
    rows = [
        _row("adversarial", "attack-1", 1, family="stealth_spoofing"),
        _row("adversarial", "attack-2", 2, family="stealth_spoofing"),
        _row("protocol_violation", "protocol-1", 3, family="duplicate_sequence"),
        _row("protocol_violation", "protocol-2", 4, family="unregistered_boot"),
        _row("hardware_fault", "fault-1", 5, family="frozen_sensor"),
        _row("hardware_fault", "fault-2", 6, family="packet_drop"),
    ]
    result = evaluate_regression_evidence_preflight(_EvidenceStorage(rows))
    assert result["scenario_counts"]["adversarial"] == 2
    assert result["family_counts"]["adversarial"] == 1
    assert result["ready"] is False
    assert result["family_deficits"] == {"adversarial": 1}


@pytest.mark.parametrize(
    "bad_row, message",
    [
        (_row("adversarial", "bad-class", 1) | {"scenario_type": "protocol"}, "classification"),
        (_row("hardware_fault", "bad-digest", 2) | {"provenance_sha256": "bad"}, "provenance"),
    ],
)
def test_regression_preflight_fails_closed_on_invalid_row_provenance(bad_row, message):
    with pytest.raises(RuntimeError, match=message):
        evaluate_regression_evidence_preflight(_EvidenceStorage([bad_row]))


def test_worker_does_not_create_candidate_when_regression_evidence_is_insufficient(
    tmp_path, monkeypatch
):
    import training.adaptive_retrainer as adaptive

    class _WorkerStorage(_EvidenceStorage):
        pass

    storage = _WorkerStorage([
        _row("adversarial", "attack-only-one", 1),
        _row("protocol_violation", "protocol-only-one", 2),
    ])
    key_dir = tmp_path / "authority-keys"
    key_dir.mkdir()
    public_key = key_dir / "model_signing_ed25519.pub"
    public_key.write_bytes(b"test-public-key-placeholder")
    model_root = tmp_path / "models"

    monkeypatch.setenv("LUMEN_ADAPTIVE_OFFLINE", "1")
    monkeypatch.setattr(adaptive, "_approved_session_ids", lambda *_a, **_k: ["s1", "s2", "s3"])
    monkeypatch.setattr(adaptive, "build_capture_run_split", lambda *_a, **_k: {
        "train_session_ids": ("s1",),
        "validation_session_ids": ("s2",),
        "heldout_session_ids": ("s3",),
        "train_capture_run_ids": ("a" * 32,),
        "validation_capture_run_ids": ("b" * 32,),
        "heldout_capture_run_ids": ("c" * 32,),
    })
    monkeypatch.setattr(adaptive, "verify_active_bundle_preflight", lambda *_a, **_k: {
        "active_bundle": "verified-active", "model_version": "v5-active", "signature_verified": True,
    })

    def _training_must_not_run(**_kwargs):
        raise AssertionError("training ran despite insufficient regression evidence")

    monkeypatch.setattr(adaptive, "train_and_export_bundle", _training_must_not_run)
    result = run_adaptive_cycle(
        db_dir=tmp_path / "db",
        model_root=model_root,
        public_key_path=public_key,
        min_duration_sec=0,
        min_sensor_coverage=1.0,
        storage=storage,
        run_id="blocked-preflight-test",
    )
    assert result.candidate_dir is None
    assert result.preflight_reason == "INSUFFICIENT_REGRESSION_EVIDENCE"
    assert result.preflight_details["deficits"] == {
        "adversarial": 1,
        "protocol_violation": 1,
        "hardware_fault": 2,
    }
    assert not (model_root / "candidates").exists()
