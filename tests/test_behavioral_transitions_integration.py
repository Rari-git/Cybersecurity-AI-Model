"""C3.3 integration against a verified C3.1 generated sequence."""
from __future__ import annotations

import pytest

from simulation.lab.evidence.evidence_index import create_evidence_index
from simulation.lab.execution_engine import OfflineScenarioEngine
from simulation.lab.staging import ResourceLimits, StagingConfig
from simulation.lab.temporal import analyze_behavioral_transitions
from simulation.lab.temporal.sequence_engine import normalize_verified_generated_run


@pytest.fixture
def indexed_generated_run(tmp_path):
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
    create_evidence_index(run.output_directory)
    return run.output_directory


def test_analyzer_accepts_real_verified_c31_sequence(indexed_generated_run):
    sequence = normalize_verified_generated_run(indexed_generated_run)
    result = analyze_behavioral_transitions(sequence)

    assert result["source"]["sequence_sha256"] == sequence["sequence_sha256"]
    assert result["sequence_summary"]["event_count"] == 8
    assert len(result["event_state_sequence"]) == 8
    assert result["sequence_summary"]["absolute_timestamps_available"] is False
    assert result["interpretation"]["maliciousness_inferred"] is False
    assert result["behavioral_transition_sha256"]
