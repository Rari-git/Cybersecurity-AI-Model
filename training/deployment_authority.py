from __future__ import annotations

from models.authority_lock import candidate_authority_lock

# Independent deployment authority. It never accepts or reads a signing private key.
import json
import os
from pathlib import Path
from typing import Any

from models.bundle_loader import verify_and_load_bundle
from models.model_lifecycle import (
    promote_candidate, rollback_model, _pointer_digest,
    recover_incomplete_deployments as _recover_incomplete_deployments,
)


def _append_audit(root: Path, event: str, payload: dict[str, Any]) -> str | None:
    # Return an audit warning; the pointer transaction has already committed.
    audit = root / "deployment_history.jsonl"
    try:
        record = {"event": event, "at_unix_ns": __import__("time").time_ns(), **payload}
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with audit.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"



def _require_authority():
    if os.environ.get("LUMEN_MODEL_DEPLOYMENT_AUTHORITY") != "1":
        raise PermissionError("deployment authority disabled; set LUMEN_MODEL_DEPLOYMENT_AUTHORITY=1")


def deploy_signed_candidate(candidate_dir, *, model_root="models",
                            public_key_path="config/model_keys/model_signing_ed25519.pub",
                            pointer_path=None, previous_pointer_path=None,
                            expected_active_pointer_sha256: str):
    _require_authority()
    root = Path(model_root).expanduser().resolve()
    with candidate_authority_lock(root):
        pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
        if not pointer.is_absolute():
            pointer = pointer.resolve()
        if _pointer_digest(pointer) != expected_active_pointer_sha256:
            raise RuntimeError("active model changed before deployment request")
        # Lifecycle repeats signature and attestation verification under its pointer lock.
        verify_and_load_bundle(candidate_dir, public_key_path)
        result = promote_candidate(
            candidate_dir, public_key_path, model_root=root, pointer_path=pointer,
            previous_pointer_path=previous_pointer_path,
            expected_active_pointer_sha256=expected_active_pointer_sha256,
        )
        try:
            audit_warning = _append_audit(root, "candidate_deployed", {
                "candidate": result.get("active_bundle"),
                "model_version": result.get("model_version"),
                "report_id": result.get("deployment_attestation", {}).get("report_id"),
                "active_pointer_sha256": _pointer_digest(pointer),
            })
        except Exception as exc:
            # Defensive boundary: even a replaced/injected audit writer must not
            # make an already committed pointer transaction look unsuccessful.
            audit_warning = f"{type(exc).__name__}: {exc}"
        result["operation_status"] = "committed"
        result["audit_status"] = "failed" if audit_warning else "written"
        if audit_warning:
            result["audit_warning"] = audit_warning
        return result


def rollback_deployed_model(*, model_root="models",
                            public_key_path="config/model_keys/model_signing_ed25519.pub",
                            pointer_path=None, previous_pointer_path=None,
                            expected_active_pointer_sha256: str):
    _require_authority()
    root = Path(model_root).expanduser().resolve()
    with candidate_authority_lock(root):
        pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
        if not pointer.is_absolute():
            pointer = pointer.resolve()
        result = rollback_model(
            model_root=root, pointer_path=pointer, public_key_path=public_key_path,
            previous_pointer_path=previous_pointer_path,
            expected_active_pointer_sha256=expected_active_pointer_sha256,
        )
        try:
            audit_warning = _append_audit(root, "authorized_rollback", {
                "active_bundle": result.get("active_bundle"),
                "model_version": result.get("model_version"),
                "active_pointer_sha256": _pointer_digest(pointer),
            })
        except Exception as exc:
            # Keep rollback outcome distinct from audit-log availability.
            audit_warning = f"{type(exc).__name__}: {exc}"
        result["operation_status"] = "committed"
        result["audit_status"] = "failed" if audit_warning else "written"
        if audit_warning:
            result["audit_warning"] = audit_warning
        return result


def monitor_post_deployment_rates(storage: Any, *, model_root, public_key_path,
                                  pointer_path=None, window_sec=300.0, min_events=20,
                                  anomaly_rate_margin=0.05, fault_rate_margin=0.01):
    _require_authority()
    if window_sec <= 0 or min_events < 1 or anomaly_rate_margin < 0 or fault_rate_margin < 0:
        raise ValueError("monitoring window, event count, and margins must be valid")
    from training.promotion_gate import read_post_promotion_rates, _finite_number
    from models.model_lifecycle import resolve_active_bundle
    root = Path(model_root).expanduser().resolve()
    with candidate_authority_lock(root):
        pointer = Path(pointer_path) if pointer_path is not None else root / "active_model.json"
        if not pointer.is_absolute():
            pointer = pointer.resolve()
        expected = _pointer_digest(pointer)
        active = resolve_active_bundle(root, pointer, root / "lumen_model_bundle")
        verified = verify_and_load_bundle(active, public_key_path)
        metrics = verified.manifest.get("promotion_metrics")
        base = metrics.get("candidate") if isinstance(metrics, dict) else None
        if not isinstance(base, dict) or "clean_security_alert_rate" not in base or "clean_operational_fault_rate" not in base:
            return None
        observed = read_post_promotion_rates(storage, window_sec=window_sec, min_events=min_events)
        if observed is None:
            return None
        baseline = {
            "anomaly_rate": _finite_number(base["clean_security_alert_rate"], "baseline.clean_security_alert_rate"),
            "fault_rate": _finite_number(base["clean_operational_fault_rate"], "baseline.clean_operational_fault_rate"),
        }
        anomaly = _finite_number(observed.get("anomaly_rate"), "observed.anomaly_rate")
        fault = _finite_number(observed.get("fault_rate"), "observed.fault_rate")
        if anomaly <= baseline["anomaly_rate"] + anomaly_rate_margin and fault <= baseline["fault_rate"] + fault_rate_margin:
            return {"rollback": False, "observed": observed, "baseline": baseline}
        restored = rollback_deployed_model(
            model_root=root, public_key_path=public_key_path, pointer_path=pointer,
            expected_active_pointer_sha256=expected,
        )
        return {"rollback": True, "observed": observed, "baseline": baseline, "pointer": restored}

def recover_deployment_transactions(*, model_root="models",
                                    public_key_path="config/model_keys/model_signing_ed25519.pub",
                                    pointer_path=None, previous_pointer_path=None):
    _require_authority()
    return _recover_incomplete_deployments(
        model_root=model_root,
        public_key_path=public_key_path,
        pointer_path=pointer_path,
        previous_pointer_path=previous_pointer_path,
    )
