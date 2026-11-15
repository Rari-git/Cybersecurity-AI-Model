"""C2.5 deterministic catalog builder for independently verified evidence runs.

This module catalogs evidence references only. It does not copy, relabel,
merge, or feed evidence into model training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .evidence_index import EvidenceError, verify_evidence_index

DATASET_CATALOG_SCHEMA = "c2.evidence-dataset-catalog.v1"
_ALLOWED_GROUPS = {
    ("dev", "adversarial"),
    ("held_out", "adversarial"),
    ("regression", "protocol_violation"),
    ("regression", "hardware_fault"),
    ("regression", "clean_control"),
    ("dev", "clean_control"),
    ("held_out", "clean_control"),
}


def build_dataset_catalog(run_directories: Iterable[Path]) -> dict:
    """Build a deterministic, split-preserving catalog from verified run dirs.

    A catalog is descriptive metadata, not a materialized ML dataset. Existing
    C2.4 evidence is DEV attack evidence only; this builder never fabricates
    missing held-out, protocol, fault, or clean-control records.
    """
    records = []
    seen_run_ids: set[str] = set()
    for directory in run_directories:
        verified = verify_evidence_index(Path(directory))
        payload = verified.payload
        run_id = payload["run_id"]
        if run_id in seen_run_ids:
            raise EvidenceError(f"duplicate run_id in dataset catalog: {run_id}")
        seen_run_ids.add(run_id)
        scenario = payload["source_scenario"]
        key = (scenario["split"], scenario["corpus_class"])
        if key not in _ALLOWED_GROUPS:
            raise EvidenceError(f"unsupported split/corpus class combination: {key}")
        records.append({
            "run_id": run_id,
            "scenario_id": scenario["scenario_id"],
            "scenario_family": scenario["family"],
            "scenario_definition_digest": scenario["definition_digest"],
            "split": scenario["split"],
            "corpus_class": scenario["corpus_class"],
            "ground_truth": scenario["ground_truth"],
            "variation_id": payload["variation"]["variation_id"],
            "variation_digest": payload["variation"]["variation_digest"],
            "frame_count": payload["frame_count"],
            "frames_sha256": payload["frames_sha256"],
            "evidence_index_sha256": verified.evidence_index_sha256,
        })
    records.sort(key=lambda item: (item["split"], item["corpus_class"], item["scenario_id"], item["run_id"]))
    groups: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        groups.setdefault(record["split"], {}).setdefault(record["corpus_class"], []).append(record)
    counts = {
        split: {corpus_class: len(items) for corpus_class, items in sorted(classes.items())}
        for split, classes in sorted(groups.items())
    }
    return {
        "schema_version": DATASET_CATALOG_SCHEMA,
        "catalog_type": "verified_evidence_references_only",
        "training_data_created": False,
        "held_out_data_merged_into_development": False,
        "records": records,
        "groups": groups,
        "counts": counts,
    }
