"""C2.5 evidence coverage report; deliberately does not claim detector metrics."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .dataset_builder import build_dataset_catalog

EVALUATION_REPORT_SCHEMA = "c2.evidence-evaluation-report.v1"


def build_evaluation_report(run_directories) -> dict[str, Any]:
    """Summarize verified evidence coverage, not IDS detection performance."""
    catalog = build_dataset_catalog(run_directories)
    split_counts = Counter(record["split"] for record in catalog["records"])
    class_counts = Counter(record["corpus_class"] for record in catalog["records"])
    families = sorted({record["scenario_family"] for record in catalog["records"]})
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA,
        "report_type": "evidence_integrity_and_coverage",
        "verified_run_count": len(catalog["records"]),
        "verified_frame_count": sum(record["frame_count"] for record in catalog["records"]),
        "runs_by_split": dict(sorted(split_counts.items())),
        "runs_by_corpus_class": dict(sorted(class_counts.items())),
        "represented_scenario_families": families,
        "detection_metrics_available": False,
        "detection_metrics_note": "C2.4 offline frame evidence contains no detector outcomes; recall, precision, F1 and false-positive rates are not computed.",
        "training_data_created": False,
        "active_model_modified": False,
        "catalog": catalog,
    }
