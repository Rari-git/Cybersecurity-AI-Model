from __future__ import annotations

from simulation.lab.evidence.dataset_builder import build_dataset_catalog
from simulation.lab.evidence.evaluation_report import build_evaluation_report
from simulation.lab.evidence.evidence_index import create_evidence_index
from simulation.lab.execution_engine import OfflineScenarioEngine
from simulation.lab.staging import ResourceLimits, StagingConfig


def test_report_describes_integrity_coverage_not_detector_performance(tmp_path):
    config = StagingConfig(
        workspace_root=tmp_path / "staging",
        resource_limits=ResourceLimits(
            max_frames=100, max_payload_bytes=16_384,
            max_total_bytes=2_000_000, max_runtime_seconds=30.0,
            max_manifest_bytes=65_536,
        ),
    )
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk4_adaptive_burst_dev", variation_seed=88, steps=6
    )
    create_evidence_index(result.output_directory)
    report = build_evaluation_report([result.output_directory])
    assert report["verified_run_count"] == 1
    assert report["verified_frame_count"] == 6
    assert report["runs_by_split"] == {"dev": 1}
    assert report["runs_by_corpus_class"] == {"adversarial": 1}
    assert report["detection_metrics_available"] is False
    assert report["training_data_created"] is False
    assert report["active_model_modified"] is False
