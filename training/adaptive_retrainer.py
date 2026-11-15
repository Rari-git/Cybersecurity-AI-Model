"""V6 Stage B: autonomous local offline-authority retraining worker.

This process is intentionally separate from ``agent.ids_agent``. The IDS
runtime does not import or execute this module; it only verifies an active
Ed25519-signed bundle. The worker uses local SQLite, local PyTorch/ONNX Runtime
and a separately provisioned offline signing key. It contains no network client
or download fallback.

The OS/operator must run this process in an air-gapped environment (or under a
network-denied account). ``LUMEN_ADAPTIVE_OFFLINE=1`` is an explicit guard, not a
substitute for host firewall/physical isolation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from storage.telemetry_db import DBPaths, LumenStorageManager
from training.promotion_gate import CandidateEvaluationResult, evaluate_candidate_only
from training.offline_guard import offline_network_guard
from models.bundle_loader import verify_and_load_bundle
from models.model_lifecycle import resolve_active_bundle
from training.train_local_lstm import build_capture_run_split, train_and_export_bundle


@dataclass(frozen=True)
class AdaptiveCycleResult:
    run_id: str
    curated_sessions: tuple[str, ...]
    rejected_sessions: tuple[str, ...]
    candidate_dir: str | None
    training_metadata: dict[str, Any] | None
    evaluation: CandidateEvaluationResult | None
    preflight_reason: str | None = None
    preflight_details: dict[str, Any] | None = None


def _require_offline_authority() -> None:
    if os.environ.get("LUMEN_ADAPTIVE_OFFLINE") != "1":
        raise RuntimeError(
            "adaptive authority refuses to run unless LUMEN_ADAPTIVE_OFFLINE=1; "
            "run this worker under a network-denied/offline OS account"
        )


def _approved_session_ids(
    storage: LumenStorageManager,
    *,
    min_duration_sec: float = 300.0,
    min_sensor_coverage: float = 1.0,
) -> list[str]:
    """Select only previously curated sessions meeting this worker's policy.

    This is read-only by design. The adaptive worker must not re-curate the
    entire database on every cycle, because that would rewrite historical
    curation decisions. Capture/operator workflows curate sessions explicitly.
    """
    if not math.isfinite(float(min_duration_sec)) or min_duration_sec < 0:
        raise ValueError("min_duration_sec must be finite and non-negative")
    if (
        not math.isfinite(float(min_sensor_coverage))
        or not 0.0 < min_sensor_coverage <= 1.0
    ):
        raise ValueError("min_sensor_coverage must be within (0, 1]")

    storage.flush_all()
    conn = storage.connect_readonly("normal")
    try:
        rows = conn.execute(
            """SELECT s.session_id, c.decision_json, c.dataset_sha256
               FROM baseline_sessions s
               JOIN session_curation c ON c.session_id=s.session_id
               WHERE s.source_type='baseline_lab' AND s.approved_for_training=1
                 AND c.status='approved_clean_baseline'
               ORDER BY s.started_at_ns ASC, s.session_id ASC"""
        ).fetchall()

        approved: list[str] = []
        for row in rows:
            try:
                decision = json.loads(str(row["decision_json"]))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(decision, dict):
                continue
            if (
                decision.get("approved") is not True
                or decision.get("status") != "approved_clean_baseline"
                or decision.get("reasons") not in ([], ())
            ):
                continue

            duration = decision.get("duration_sec")
            coverage = decision.get("sensor_coverage")
            event_count = decision.get("event_count")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) < min_duration_sec
            ):
                continue
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, (int, float))
                or not math.isfinite(float(coverage))
                or float(coverage) + 1e-12 < min_sensor_coverage
            ):
                continue
            if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
                continue

            digest = row["dataset_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest.lower())
                or decision.get("dataset_sha256") != digest
            ):
                continue

            approved.append(str(row["session_id"]))
        return approved
    finally:
        conn.close()


DEFAULT_MIN_REGRESSION_SCENARIOS = {
    "adversarial": 2,
    "protocol_violation": 2,
    "hardware_fault": 2,
}


def evaluate_regression_evidence_preflight(
    storage: LumenStorageManager,
    *,
    min_adversarial_scenarios: int = 2,
    min_protocol_scenarios: int = 2,
    min_hardware_fault_scenarios: int = 2,
    min_adversarial_families: int = 2,
    min_protocol_families: int = 2,
    min_hardware_fault_families: int = 2,
) -> dict[str, Any]:
    """Require independent executions and diversity of attack/fault families.

    Scenario IDs establish separate completed executions. Family IDs prevent
    meeting the preflight by replaying one attack/fault template with new IDs.
    Older captures without a family stay preserved and count as scenarios, but
    do not contribute to family diversity.
    """
    requirements = {
        "adversarial": min_adversarial_scenarios,
        "protocol_violation": min_protocol_scenarios,
        "hardware_fault": min_hardware_fault_scenarios,
    }
    family_requirements = {
        "adversarial": min_adversarial_families,
        "protocol_violation": min_protocol_families,
        "hardware_fault": min_hardware_fault_families,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (*requirements.values(), *family_requirements.values())
    ):
        raise ValueError("minimum regression scenario/family requirements must be positive integers")

    storage.flush_all()
    rows = storage.load_adversarial_regression_corpus(tuple(requirements))
    expected_scenario_types = {
        "adversarial": "attack",
        "protocol_violation": "protocol",
        "hardware_fault": "hardware_fault",
    }
    scenario_ids: dict[str, set[str]] = {key: set() for key in requirements}
    families: dict[str, set[str]] = {key: set() for key in requirements}
    capture_counts: dict[str, int] = {key: 0 for key in requirements}
    scenario_classes: dict[str, str] = {}
    scenario_family_by_id: dict[str, str] = {}

    for row in rows:
        corpus_class = str(row.get("corpus_class") or "")
        scenario_id = str(row.get("scenario_id") or "")
        scenario_type = str(row.get("scenario_type") or "")
        provenance_digest = str(row.get("provenance_sha256") or "")
        if corpus_class not in requirements:
            raise RuntimeError("regression corpus contains an unsupported class")
        if not scenario_id or scenario_type != expected_scenario_types[corpus_class]:
            raise RuntimeError("regression corpus scenario classification is inconsistent")
        if (
            len(provenance_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in provenance_digest.lower())
        ):
            raise RuntimeError("regression corpus entry has missing or invalid scenario provenance")
        previous_class = scenario_classes.setdefault(scenario_id, corpus_class)
        if previous_class != corpus_class:
            raise RuntimeError("one regression scenario is indexed under multiple corpus classes")

        family = None
        manifest_raw = row.get("scenario_manifest_json")
        if isinstance(manifest_raw, str) and manifest_raw:
            try:
                manifest = json.loads(manifest_raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("regression scenario manifest cannot be decoded") from exc
            if not isinstance(manifest, dict):
                raise RuntimeError("regression scenario manifest must be an object")
            candidate_family = manifest.get("family")
            if isinstance(candidate_family, str):
                candidate_family = candidate_family.strip()
                if candidate_family and len(candidate_family) <= 128:
                    family = candidate_family
        if family is not None:
            prior_family = scenario_family_by_id.setdefault(scenario_id, family)
            if prior_family != family:
                raise RuntimeError("one regression scenario is bound to multiple scenario families")
            families[corpus_class].add(family)

        scenario_ids[corpus_class].add(scenario_id)
        capture_counts[corpus_class] += 1

    scenario_counts = {key: len(values) for key, values in scenario_ids.items()}
    family_counts = {key: len(values) for key, values in families.items()}
    deficits = {
        key: max(0, requirements[key] - scenario_counts[key])
        for key in requirements
        if scenario_counts[key] < requirements[key]
    }
    family_deficits = {
        key: max(0, family_requirements[key] - family_counts[key])
        for key in family_requirements
        if family_counts[key] < family_requirements[key]
    }
    return {
        "ready": not deficits and not family_deficits,
        "scenario_counts": scenario_counts,
        "capture_counts": capture_counts,
        "required_scenarios": requirements,
        "deficits": deficits,
        "family_counts": family_counts,
        "required_families": family_requirements,
        "family_deficits": family_deficits,
        "independence_unit": "completed_scenario_id_plus_distinct_family",
    }

def verify_active_bundle_preflight(
    model_root: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    """Verify the currently active signed bundle before spending time on training."""
    root = Path(model_root).expanduser().resolve()
    public_key = Path(public_key_path).expanduser().resolve()
    pointer = root / "active_model.json"
    if not pointer.is_file():
        raise FileNotFoundError("active_model.json is required before adaptive training")
    if not public_key.is_file():
        raise FileNotFoundError("runtime model-verification public key is unavailable")
    active_dir = resolve_active_bundle(root, pointer, root / "lumen_model_bundle")
    verified = verify_and_load_bundle(active_dir, public_key)
    return {
        "active_bundle": str(active_dir),
        "model_version": str(verified.manifest.get("model_version") or "unknown"),
        "signature_verified": True,
    }


def run_adaptive_cycle(
    *,
    db_dir: str | Path = "logs/db",
    model_root: str | Path = "models",
    public_key_path: str | Path | None = None,
    min_duration_sec: float = 300.0,
    min_sensor_coverage: float = 1.0,
    epochs: int = 35,
    seq_len: int = 16,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    storage: LumenStorageManager | None = None,
    run_id: str | None = None,
    min_adversarial_scenarios: int = 2,
    min_protocol_scenarios: int = 2,
    min_hardware_fault_scenarios: int = 2,
) -> AdaptiveCycleResult:
    # Train from existing clean approvals, then evaluate; never sign or deploy here.
    _require_offline_authority()
    run_id = run_id or uuid.uuid4().hex
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("run_id must be a single safe path segment")

    root = Path(model_root).expanduser().resolve()
    public_key = (
        Path(public_key_path).expanduser().resolve()
        if public_key_path
        else Path("config/model_keys/model_signing_ed25519.pub").expanduser().resolve()
    )
    if not public_key.is_file():
        raise FileNotFoundError(
            "runtime model-verification public key is required to evaluate the active bundle"
        )
    if root == public_key or root in public_key.parents:
        raise ValueError("runtime model-verification public key must remain outside models/")

    owns_storage = storage is None
    manager = storage or LumenStorageManager(
        paths=DBPaths.from_dir(db_dir), enable_writers=True
    )
    try:
        with offline_network_guard():
            # Curation remains at the capture/operator boundary.
            approved = _approved_session_ids(
                manager,
                min_duration_sec=min_duration_sec,
                min_sensor_coverage=min_sensor_coverage,
            )
            rejected: tuple[str, ...] = ()
            if len(approved) < 3:
                return AdaptiveCycleResult(
                    run_id, tuple(approved), rejected, None, None, None,
                    "INSUFFICIENT_CURATED_SESSIONS",
                )

            split = build_capture_run_split(db_dir, approved, storage=manager)
            if split is None:
                return AdaptiveCycleResult(
                    run_id, tuple(approved), rejected, None, None, None,
                    "INSUFFICIENT_INDEPENDENT_CAPTURE_RUNS",
                )
            heldout = split["heldout_session_ids"]

            # Regression evidence and active signed-bundle trust are checked
            # before candidate-directory creation or PyTorch training.
            regression_preflight = evaluate_regression_evidence_preflight(
                manager,
                min_adversarial_scenarios=min_adversarial_scenarios,
                min_protocol_scenarios=min_protocol_scenarios,
                min_hardware_fault_scenarios=min_hardware_fault_scenarios,
            )
            if not regression_preflight["ready"]:
                return AdaptiveCycleResult(
                    run_id, tuple(approved), rejected, None, None, None,
                    "INSUFFICIENT_REGRESSION_EVIDENCE", regression_preflight,
                )
            try:
                verify_active_bundle_preflight(root, public_key)
            except Exception as exc:
                return AdaptiveCycleResult(
                    run_id, tuple(approved), rejected, None, None, None,
                    "ACTIVE_BUNDLE_NOT_VERIFIABLE",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )

            candidate_dir = root / "candidates" / run_id
            candidate_dir.parent.mkdir(parents=True, exist_ok=True)
            # Never overwrite or reuse a previous candidate directory.
            candidate_dir.mkdir(parents=False, exist_ok=False)
            metadata = train_and_export_bundle(
                db_dir=db_dir,
                bundle_dir=candidate_dir,
                seq_len=seq_len,
                epochs=epochs,
                run_id=run_id,
                eligible_session_ids=approved,
                validation_session_ids=split["validation_session_ids"],
                heldout_session_ids=heldout,
                require_independent_capture_runs=True,
                sign_bundle=False,
            )
            evaluation = evaluate_candidate_only(
                candidate_dir,
                storage=manager,
                public_key_path=public_key,
                model_root=root,
                evaluator=evaluator,
            )
            return AdaptiveCycleResult(
                run_id, tuple(approved), rejected, str(candidate_dir), metadata, evaluation,
            )
    finally:
        if owns_storage:
            manager.flush_all()
            manager.close()



def run_autonomous_worker(
    *,
    interval_sec: float = 3600.0,
    max_cycles: int | None = None,
    **kwargs: Any,
) -> list[AdaptiveCycleResult]:
    # Scheduled training/evaluation only. Production monitoring/rollback is separate.
    _require_offline_authority()
    if interval_sec < 1.0:
        raise ValueError("cycle interval must be at least one second")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be positive or None")
    forbidden = {"key_dir", "monitor_interval_sec", "monitor_window_sec", "monitor_min_events"}
    supplied = forbidden.intersection(kwargs)
    if supplied:
        raise TypeError(
            "adaptive evaluation worker does not accept signing-key or deployment-monitor "
            f"arguments: {sorted(supplied)}"
        )

    db_dir = kwargs.get("db_dir", "logs/db")
    manager = kwargs.pop("storage", None)
    owns_storage = manager is None
    if manager is None:
        manager = LumenStorageManager(
            paths=DBPaths.from_dir(db_dir), enable_writers=True
        )

    results: list[AdaptiveCycleResult] = []
    cycle = 0
    try:
        while max_cycles is None or cycle < max_cycles:
            try:
                result = run_adaptive_cycle(storage=manager, **kwargs)
                results.append(result)
                evaluation = result.evaluation
                reason = (
                    getattr(result, "preflight_reason", None)
                    or (evaluation.reason if evaluation else "INSUFFICIENT_CURATED_SESSIONS")
                )
                print(json.dumps({
                    "run_id": result.run_id,
                    "curated_sessions": list(result.curated_sessions),
                    "rejected_sessions": list(result.rejected_sessions),
                    "candidate_dir": result.candidate_dir,
                    "evaluated": bool(evaluation and evaluation.evaluated),
                    "all_gates_passed": bool(evaluation and evaluation.all_gates_passed),
                    "evaluation_gates": evaluation.gates if evaluation else None,
                    "reason": reason,
                    "preflight_details": getattr(result, "preflight_details", None),
                    "active_model_changed": False,
                }, sort_keys=True))
            except Exception as exc:
                # Fail closed. This worker has no active-model mutation path.
                print(json.dumps({
                    "cycle_error": type(exc).__name__,
                    "detail": str(exc),
                    "active_model_changed": False,
                }, sort_keys=True))
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                break
            time.sleep(interval_sec)
        return results
    finally:
        if owns_storage:
            manager.flush_all()
            manager.close()



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Lumen's local evaluation-only adaptive training worker"
    )
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--public-key", default=None)
    parser.add_argument("--min-duration-sec", type=float, default=300.0)
    parser.add_argument("--min-sensor-coverage", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--min-adversarial-scenarios", type=int, default=2)
    parser.add_argument("--min-protocol-scenarios", type=int, default=2)
    parser.add_argument("--min-hardware-fault-scenarios", type=int, default=2)
    parser.add_argument("--interval-sec", type=float, default=3600.0)
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help="Optional finite cycles; default runs continuously",
    )
    args = parser.parse_args()
    run_autonomous_worker(
        db_dir=args.db_dir,
        model_root=args.model_root,
        public_key_path=args.public_key,
        min_duration_sec=args.min_duration_sec,
        min_sensor_coverage=args.min_sensor_coverage,
        epochs=args.epochs,
        seq_len=args.seq_len,
        min_adversarial_scenarios=args.min_adversarial_scenarios,
        min_protocol_scenarios=args.min_protocol_scenarios,
        min_hardware_fault_scenarios=args.min_hardware_fault_scenarios,
        interval_sec=args.interval_sec,
        max_cycles=args.max_cycles,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
