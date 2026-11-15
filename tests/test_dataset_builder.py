from __future__ import annotations

import pytest

from simulation.lab.evidence.dataset_builder import build_dataset_catalog
from simulation.lab.evidence.evidence_index import create_evidence_index, EvidenceError
from simulation.lab.execution_engine import OfflineScenarioEngine
from simulation.lab.staging import ResourceLimits, StagingConfig


def _run(tmp_path, seed):
    config = StagingConfig(
        workspace_root=tmp_path / f"staging_{seed}",
        resource_limits=ResourceLimits(
            max_frames=100, max_payload_bytes=16_384,
            max_total_bytes=2_000_000, max_runtime_seconds=30.0,
            max_manifest_bytes=65_536,
        ),
    )
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk2_replay_skew_dev", variation_seed=seed, steps=5
    )
    create_evidence_index(result.output_directory)
    return result.output_directory


def test_dataset_catalog_keeps_verified_split_and_classification(tmp_path):
    left = _run(tmp_path, 10)
    right = _run(tmp_path, 11)
    catalog = build_dataset_catalog([right, left])
    assert catalog["catalog_type"] == "verified_evidence_references_only"
    assert catalog["training_data_created"] is False
    assert catalog["held_out_data_merged_into_development"] is False
    assert catalog["counts"] == {"dev": {"adversarial": 2}}
    assert len(catalog["groups"]["dev"]["adversarial"]) == 2
    assert catalog["records"] == sorted(
        catalog["records"], key=lambda item: (item["split"], item["corpus_class"], item["scenario_id"], item["run_id"])
    )


def test_dataset_catalog_rejects_duplicate_run_ids(tmp_path):
    run = _run(tmp_path, 12)
    with pytest.raises(EvidenceError, match="duplicate run_id"):
        build_dataset_catalog([run, run])
