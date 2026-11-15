"""C2.5 read-only evidence verification and evaluation primitives."""
from .evidence_index import (
    EvidenceError,
    EvidenceIndex,
    create_evidence_index,
    verify_evidence_index,
)
from .dataset_builder import build_dataset_catalog
from .evaluation_report import build_evaluation_report

__all__ = [
    "EvidenceError", "EvidenceIndex", "create_evidence_index",
    "verify_evidence_index", "build_dataset_catalog", "build_evaluation_report",
]
