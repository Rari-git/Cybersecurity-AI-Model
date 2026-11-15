from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.lab.evidence.evidence_index import (
    EvidenceError,
    create_evidence_index,
    verify_evidence_index,
)
from simulation.lab.execution_engine import OfflineScenarioEngine
from simulation.lab.staging import ResourceLimits, StagingConfig


@pytest.fixture
def generated_run(tmp_path):
    config = StagingConfig(
        workspace_root=tmp_path / "staging",
        resource_limits=ResourceLimits(
            max_frames=100,
            max_payload_bytes=16_384,
            max_total_bytes=2_000_000,
            max_runtime_seconds=30.0,
            max_manifest_bytes=65_536,
        ),
    )
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk1_stealth_spoof_dev", variation_seed=991, steps=8
    )
    return result.output_directory


def test_create_and_verify_index_binds_every_run_artifact(generated_run):
    original_frames = {
        p.relative_to(generated_run).as_posix(): p.read_bytes()
        for p in (generated_run / "frames").glob("*.json")
    }
    index = create_evidence_index(generated_run)
    verified = verify_evidence_index(generated_run)
    assert index.evidence_index_sha256 == verified.evidence_index_sha256
    assert index.payload["frame_count"] == 8
    assert len(index.payload["artifacts"]) == 10
    assert index.payload["source_scenario"]["split"] == "dev"
    assert index.payload["source_scenario"]["corpus_class"] == "adversarial"
    assert index.payload["resource_accounting"]["total_persisted_evidence_bytes"] > 0
    assert original_frames == {
        p.relative_to(generated_run).as_posix(): p.read_bytes()
        for p in (generated_run / "frames").glob("*.json")
    }


def test_index_refuses_overwrite(generated_run):
    create_evidence_index(generated_run)
    with pytest.raises(EvidenceError, match="already exists"):
        create_evidence_index(generated_run)


def test_verifier_detects_frame_tampering_after_index(generated_run):
    create_evidence_index(generated_run)
    frame = generated_run / "frames" / "frame_000000.json"
    record = json.loads(frame.read_text(encoding="utf-8"))
    record["topic"] = "tampered/topic"
    frame.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_evidence_index(generated_run)


def test_verifier_detects_summary_tampering(generated_run):
    create_evidence_index(generated_run)
    path = generated_run / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["frame_count"] += 1
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_evidence_index(generated_run)


def test_verifier_rejects_unexpected_frame_file(generated_run):
    create_evidence_index(generated_run)
    (generated_run / "frames" / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(EvidenceError, match="unexpected"):
        verify_evidence_index(generated_run)


def test_verifier_rejects_duplicate_json_keys(generated_run):
    create_evidence_index(generated_run)
    frame = generated_run / "frames" / "frame_000000.json"
    frame.write_text('{"step":0,"step":0}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="duplicate JSON"):
        verify_evidence_index(generated_run)


def test_index_rejects_manifest_provenance_mismatch(generated_run):
    summary_path = generated_run / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_scenario_id"] = "atk1_stealth_spoof_heldout"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(EvidenceError, match="provenance mismatch"):
        create_evidence_index(generated_run)


def test_index_is_not_written_when_source_evidence_fails(generated_run):
    (generated_run / "frames" / "frame_000000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvidenceError):
        create_evidence_index(generated_run)
    assert not (generated_run / "evidence_index.json").exists()


def test_verifier_rejects_unexpected_run_level_artifact(generated_run):
    create_evidence_index(generated_run)
    (generated_run / "untracked.bin").write_bytes(b"unexpected")
    with pytest.raises(EvidenceError, match="unexpected artifacts"):
        verify_evidence_index(generated_run)

def test_create_index_enforces_caller_total_byte_limit(generated_run):
    with pytest.raises(EvidenceError, match="caller max_total_bytes limit"):
        create_evidence_index(generated_run, max_total_bytes=1)
    assert not (generated_run / "evidence_index.json").exists()


def test_verify_index_validates_caller_limits(generated_run):
    create_evidence_index(generated_run)
    with pytest.raises(ValueError, match="max_total_bytes must be a positive integer"):
        verify_evidence_index(generated_run, max_total_bytes=0)
    with pytest.raises(ValueError, match="max_artifacts must be a positive integer"):
        verify_evidence_index(generated_run, max_artifacts=True)

