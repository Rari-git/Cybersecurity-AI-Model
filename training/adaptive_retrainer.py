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
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from storage.telemetry_db import DBPaths, LumenStorageManager
from training.dataset_governance import CurationDecision, curate_clean_sessions
from training.promotion_gate import PromotionResult, evaluate_and_promote_candidate, monitor_post_promotion_rates
from training.offline_guard import offline_network_guard
from training.train_local_lstm import train_and_export_bundle


@dataclass(frozen=True)
class AdaptiveCycleResult:
    run_id: str
    curated_sessions: tuple[str, ...]
    rejected_sessions: tuple[str, ...]
    candidate_dir: str | None
    training_metadata: dict[str, Any] | None
    promotion: PromotionResult | None


def _require_offline_authority() -> None:
    if os.environ.get("LUMEN_ADAPTIVE_OFFLINE") != "1":
        raise RuntimeError(
            "adaptive authority refuses to run unless LUMEN_ADAPTIVE_OFFLINE=1; "
            "run this worker under a network-denied/offline OS account"
        )


def _approved_session_ids(storage: LumenStorageManager) -> list[str]:
    storage.flush_all()
    conn = storage.connect_readonly("normal")
    try:
        rows = conn.execute(
            """SELECT s.session_id FROM baseline_sessions s
               JOIN session_curation c ON c.session_id=s.session_id
               WHERE s.source_type='baseline_lab' AND s.approved_for_training=1
                 AND c.status='approved_clean_baseline'
               ORDER BY s.started_at_ns ASC, s.session_id ASC"""
        ).fetchall()
        return [str(row["session_id"]) for row in rows]
    finally:
        conn.close()


def run_adaptive_cycle(
    *,
    db_dir: str | Path = "logs/db",
    model_root: str | Path = "models",
    key_dir: str | Path = "config/model_keys",
    public_key_path: str | Path | None = None,
    min_duration_sec: float = 300.0,
    min_sensor_coverage: float = 1.0,
    epochs: int = 35,
    seq_len: int = 16,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    storage: LumenStorageManager | None = None,
    run_id: str | None = None,
) -> AdaptiveCycleResult:
    """Curate, train unsigned candidate, shadow-evaluate, then conditionally promote."""
    _require_offline_authority()
    run_id = run_id or uuid.uuid4().hex
    root = Path(model_root).expanduser().resolve()
    keys = Path(key_dir).expanduser().resolve()
    private_key = keys / "model_signing_ed25519.key"
    public_key = Path(public_key_path).resolve() if public_key_path else keys / "model_signing_ed25519.pub"
    if not private_key.is_file() or not public_key.is_file():
        raise FileNotFoundError(
            "offline Ed25519 model-signing keypair must be provisioned outside the runtime model directory"
        )
    if root == keys or root in keys.parents:
        raise ValueError("offline signing authority keys must be outside models/")

    owns_storage = storage is None
    manager = storage or LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        with offline_network_guard():
                decisions: list[CurationDecision] = curate_clean_sessions(
                    manager,
                    min_duration_sec=min_duration_sec,
                    min_sensor_coverage=min_sensor_coverage,
                )
                rejected = tuple(item.session_id for item in decisions if not item.approved)
                approved = _approved_session_ids(manager)
                # Need at least one training session, one calibration/validation session,
                # and one independent held-out clean session.
                if len(approved) < 3:
                    return AdaptiveCycleResult(
                        run_id, tuple(approved), rejected, None, None, None,
                    )
                heldout = (approved[-1],)
                candidate_dir = root / "candidates" / run_id
                candidate_dir.parent.mkdir(parents=True, exist_ok=True)
                metadata = train_and_export_bundle(
                    db_dir=db_dir,
                    bundle_dir=candidate_dir,
                    key_dir=keys,
                    seq_len=seq_len,
                    epochs=epochs,
                    run_id=run_id,
                    eligible_session_ids=approved,
                    heldout_session_ids=heldout,
                    sign_bundle=False,
                )
                promotion = evaluate_and_promote_candidate(
                    candidate_dir,
                    storage=manager,
                    public_key_path=public_key,
                    private_key_path=private_key,
                    model_root=root,
                    evaluator=evaluator,
                )
                return AdaptiveCycleResult(
                    run_id, tuple(approved), rejected, str(candidate_dir), metadata, promotion,
                )
    finally:
        if owns_storage:
            manager.flush_all()
            manager.close()


def run_autonomous_worker(
    *,
    interval_sec: float = 3600.0,
    max_cycles: int | None = None,
    monitor_interval_sec: float = 10.0,
    monitor_window_sec: float = 300.0,
    monitor_min_events: int = 20,
    **kwargs: Any,
) -> list[AdaptiveCycleResult]:
    """Run local training cycles and continuously monitor post-promotion rates.

    ``max_cycles=None`` is the autonomous service mode. The same local SQLite
    manager is reused by curation and the rollback monitor.
    """
    _require_offline_authority()
    if interval_sec < 1.0 or monitor_interval_sec < 1.0 or monitor_window_sec <= 0:
        raise ValueError("cycle/monitor intervals must be positive (at least one second for polling)")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be positive or None")

    worker_kwargs = dict(kwargs)
    db_dir = worker_kwargs.get("db_dir", "logs/db")
    model_root = worker_kwargs.get("model_root", "models")
    key_dir = worker_kwargs.get("key_dir", os.environ.get("LUMEN_MODEL_AUTHORITY_KEY_DIR", "config/model_keys"))
    public_key_path = worker_kwargs.get("public_key_path") or (Path(key_dir) / "model_signing_ed25519.pub")
    manager = worker_kwargs.pop("storage", None)
    owns_storage = manager is None
    if manager is None:
        manager = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)

    results: list[AdaptiveCycleResult] = []
    cycle = 0
    next_cycle_at = 0.0
    try:
        while max_cycles is None or cycle < max_cycles:
            now = time.monotonic()
            if now >= next_cycle_at:
                try:
                    result = run_adaptive_cycle(storage=manager, **worker_kwargs)
                    results.append(result)
                    print(json.dumps({
                        "run_id": result.run_id,
                        "curated_sessions": list(result.curated_sessions),
                        "rejected_sessions": list(result.rejected_sessions),
                        "candidate_dir": result.candidate_dir,
                        "promoted": result.promotion.promoted if result.promotion else False,
                        "reason": result.promotion.reason if result.promotion else "INSUFFICIENT_CURATED_SESSIONS",
                    }, sort_keys=True))
                except Exception as exc:
                    # Fail closed and leave the currently active model unchanged.
                    print(json.dumps({"cycle_error": type(exc).__name__, "detail": str(exc)}, sort_keys=True))
                cycle += 1
                next_cycle_at = time.monotonic() + interval_sec
                if max_cycles is not None and cycle >= max_cycles:
                    break

            try:
                with offline_network_guard():
                    monitor_result = monitor_post_promotion_rates(
                        manager, model_root=model_root, public_key_path=public_key_path,
                        window_sec=monitor_window_sec, min_events=monitor_min_events,
                        anomaly_rate_margin=float(os.environ.get("LUMEN_ADAPTIVE_ANOMALY_RATE_MARGIN", "0.05")),
                        fault_rate_margin=float(os.environ.get("LUMEN_ADAPTIVE_FAULT_RATE_MARGIN", "0.01")),
                    )
                if monitor_result and monitor_result.get("rollback"):
                    print(json.dumps({"automatic_rollback": True, "result": monitor_result}, sort_keys=True))
            except FileNotFoundError:
                # Before the first signed V6 promotion there may be no usable
                # active bundle or promotion-rate baseline yet. No model changes.
                pass
            except Exception as exc:
                print(json.dumps({"monitor_error": type(exc).__name__, "detail": str(exc)}, sort_keys=True))

            if max_cycles is None or cycle < max_cycles:
                time.sleep(monitor_interval_sec)
        return results
    finally:
        if owns_storage:
            manager.flush_all()
            manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lumen's local air-gapped adaptive authority worker")
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--key-dir", default=os.environ.get("LUMEN_MODEL_AUTHORITY_KEY_DIR", "config/model_keys"))
    parser.add_argument("--public-key", default=None)
    parser.add_argument("--min-duration-sec", type=float, default=300.0)
    parser.add_argument("--min-sensor-coverage", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--interval-sec", type=float, default=3600.0)
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional finite cycles; default runs continuously")
    parser.add_argument("--monitor-interval-sec", type=float, default=10.0)
    parser.add_argument("--monitor-window-sec", type=float, default=300.0)
    args = parser.parse_args()
    run_autonomous_worker(
        db_dir=args.db_dir,
        model_root=args.model_root,
        key_dir=args.key_dir,
        public_key_path=args.public_key,
        min_duration_sec=args.min_duration_sec,
        min_sensor_coverage=args.min_sensor_coverage,
        epochs=args.epochs,
        seq_len=args.seq_len,
        interval_sec=args.interval_sec,
        max_cycles=args.max_cycles,
        monitor_interval_sec=args.monitor_interval_sec,
        monitor_window_sec=args.monitor_window_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
