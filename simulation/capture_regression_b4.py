"""Stage B4: capture a diverse, provenance-verified local regression corpus.

Uses the frozen E3 generators for four attack families and two hardware faults,
plus two authenticated V6 protocol scenarios. Attack/fault traces retain
inter-arrival timing and operation type in the forensic metadata. No external
network, cloud API, or live production device is contacted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.advanced_generators import FROZEN_SCENARIOS, iter_scenario_frames
from agent.statistical_detectors import OperationalFaultClassifier
from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.persistence import persist_incident_capture
from storage.telemetry_db import DBPaths, LumenStorageManager
from utils.boot_registration import (
    BootRegistrationAuthority,
    create_boot_registration_proof,
    verify_gateway_challenge,
)
from utils.cert_lifecycle import CertAuthority
from utils.device_identity import load_active_device_identity, load_active_device_signing_key
from utils.sensor_payload import encode_sensor_frame

RUNNER_VERSION = "stage-b4-regression-capture-v1"
ATTACK_SCENARIOS = (
    "atk1_stealth_spoof_dev",
    "atk2_replay_skew_dev",
    "atk3_low_slow_recon_dev",
    "atk4_adaptive_burst_dev",
)
PROTOCOL_SCENARIOS = (
    "authenticated_replay_duplicate_sequence",
    "unregistered_boot_epoch",
)
FAULT_SCENARIOS = (
    "fault1_frozen_sensor",
    "fault2_packet_drop_40",
)


class RegressionCaptureError(RuntimeError):
    """A controlled scenario did not produce the expected verified evidence."""


def _new_scenario_id(family: str, seed: int) -> str:
    safe_family = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in family)[:48]
    return f"stageb4-{safe_family}-{seed}-{uuid.uuid4().hex[:12]}"


def _new_live_session(storage: LumenStorageManager, scenario_id: str) -> str:
    session_id = f"regression-live-{scenario_id}"
    storage.register_baseline_session(
        session_id=session_id,
        source_type="live_validated",
        config={"regression_scenario_id": scenario_id, "stage": "B4", "training_eligible": False},
        approved_for_training=False,
    )
    return session_id


def _finalize_live_session(storage: LumenStorageManager, session_id: str) -> None:
    storage.finalize_baseline_session(session_id, approved_for_training=False)


def _fail_scenario(storage: LumenStorageManager, scenario_id: str) -> None:
    try:
        storage.complete_regression_scenario(scenario_id, success=False)
    except (KeyError, ValueError):
        pass


def _persist_lab_capture(
    storage: LumenStorageManager,
    *,
    session_id: str,
    scenario_id: str,
    identity: Any,
    device_id: str,
    topic: str,
    source: str,
    reason_code: str,
    label: str,
    raw_payload: bytes,
    gateway_ts_ns: int,
    metadata: dict[str, Any],
) -> None:
    if not raw_payload or len(raw_payload) > 512:
        raise RegressionCaptureError("generated forensic payload is empty or exceeds 512 bytes")
    if source not in {"attack_lab", "fault_lab"}:
        raise ValueError("lab capture source must be attack_lab or fault_lab")
    if label not in {"confirmed_attack", "confirmed_hardware_fault"}:
        raise ValueError("invalid controlled-lab ground-truth label")
    metadata_json = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    retention_until_ns = gateway_ts_ns + 14 * 24 * 3600 * 1_000_000_000
    persist_incident_capture(
        storage,
        (
            session_id,
            device_id,
            identity.common_name,
            int(identity.generation),
            identity.certificate_fingerprint,
            topic,
            f"{source}/{scenario_id}",
            gateway_ts_ns,
            None,
            source,
            reason_code,
            0.95,
            metadata_json,
            len(raw_payload),
            hashlib.sha256(raw_payload).digest(),
            raw_payload,
            1,
            label,
            scenario_id,
            retention_until_ns,
        ),
    )


def capture_attack_family(
    storage: LumenStorageManager,
    cert_pointer: str,
    generator_scenario_id: str,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    spec = FROZEN_SCENARIOS[generator_scenario_id]
    if spec.category != "attack" or spec.split != "dev":
        raise ValueError("B4 only captures frozen development attack scenarios")
    if steps < 16:
        raise ValueError("attack_steps must be at least 16")

    family = spec.attack_family
    scenario_id = _new_scenario_id(family, seed)
    session_id = _new_live_session(storage, scenario_id)
    identity = load_active_device_identity(cert_pointer, spec.target_device)
    manifest = {
        "family": family,
        "generator": "attacks.advanced_generators.iter_scenario_frames",
        "generator_scenario_id": generator_scenario_id,
        "generator_split": spec.split,
        "generator_seed": int(spec.seed),
        "runner_seed": int(seed),
        "steps": int(steps),
        "target_device": spec.target_device,
        "target_topic": spec.target_topic,
        "parameters": spec.params,
        "ground_truth": "confirmed_attack",
        "timestamp_policy": "synthetic_monotonic_from_generated_delta_t",
        "operation_policy": "preserve_per_capture_in_xai_metadata",
    }
    storage.register_regression_scenario(
        scenario_id, "attack", runner_version=RUNNER_VERSION, manifest=manifest
    )

    try:
        frames = list(iter_scenario_frames(generator_scenario_id, steps=steps))
        if len(frames) != steps:
            raise RegressionCaptureError(f"{generator_scenario_id}: expected {steps} frames, got {len(frames)}")
        base_ts_ns = time.time_ns()
        elapsed_sec = 0.0
        capture_count = 0
        for frame in frames:
            delta = float(frame.delta_t_sec)
            if not math.isfinite(delta) or delta <= 0:
                raise RegressionCaptureError("attack generator produced invalid inter-arrival time")
            elapsed_sec += delta
            ts_ns = base_ts_ns + int(round(elapsed_sec * 1_000_000_000))
            _persist_lab_capture(
                storage,
                session_id=session_id,
                scenario_id=scenario_id,
                identity=identity,
                device_id=frame.device_id,
                topic=frame.topic,
                source="attack_lab",
                reason_code=f"LAB_ATTACK_{family.upper()}",
                label="confirmed_attack",
                raw_payload=bytes(frame.payload),
                gateway_ts_ns=ts_ns,
                metadata={
                    "family": family,
                    "generator_scenario_id": generator_scenario_id,
                    "step": int(frame.step),
                    "operation": str(frame.operation),
                    "delta_t_sec": delta,
                    "generator_label": str(frame.ground_truth_label),
                    "split": spec.split,
                },
            )
            capture_count += 1

        indexed = storage.complete_regression_scenario(scenario_id)
        if indexed != capture_count or indexed < 1:
            raise RegressionCaptureError(f"{family}: expected {capture_count} indexed captures, got {indexed}")
        _finalize_live_session(storage, session_id)
        return {
            "scenario_id": scenario_id,
            "session_id": session_id,
            "corpus_class": "adversarial",
            "family": family,
            "source_scenario_id": generator_scenario_id,
            "captures_indexed": indexed,
        }
    except BaseException:
        _fail_scenario(storage, scenario_id)
        raise


def _register_authenticated_boot(
    gate: AdmissionGate,
    *,
    cert_pointer: str,
    revocation_ledger: str,
    session_id: str,
    identity: Any,
    sensor_key: Any,
) -> str:
    gateway_identity = load_active_device_identity(cert_pointer, "gateway")
    boot_id = uuid.uuid4().hex
    authority = BootRegistrationAuthority(
        gate,
        cert_pointer=cert_pointer,
        revocation_ledger=revocation_ledger,
        session_id=session_id,
        ttl_sec=10.0,
    )
    challenge = authority.issue_challenge(identity.device_id, boot_id)
    verify_gateway_challenge(challenge, identity, gateway_identity)
    proof = create_boot_registration_proof(challenge, identity, sensor_key)
    registration = authority.complete_registration(proof, expected_device_id=identity.device_id)
    if registration.get("ok") is not True:
        raise RegressionCaptureError("authenticated device boot registration failed")
    return boot_id


def capture_protocol_family(
    storage: LumenStorageManager,
    cert_pointer: str,
    revocation_ledger: str,
    family: str,
    *,
    seed: int,
) -> dict[str, Any]:
    if family not in PROTOCOL_SCENARIOS:
        raise ValueError(f"unsupported protocol family: {family}")

    scenario_id = _new_scenario_id(family, seed)
    session_id = _new_live_session(storage, scenario_id)
    identity = load_active_device_identity(cert_pointer, "gas_sensor_01")
    _key_identity, sensor_key = load_active_device_signing_key(cert_pointer, "gas_sensor_01")
    expected_reasons = (
        ["REPLAY_OR_OUT_OF_ORDER_SEQUENCE"]
        if family == "authenticated_replay_duplicate_sequence"
        else ["BOOT_NOT_REGISTERED", "STALE_BOOT_ID"]
    )
    manifest = {
        "family": family,
        "generator": "AdmissionGate.evaluate_v6_and_ingest",
        "runner_seed": int(seed),
        "device_id": "gas_sensor_01",
        "topic": "home/sensors/gas/ppm",
        "registered_certificate_generation": int(identity.generation),
        "expected_reasons": expected_reasons,
        "ground_truth": "confirmed_protocol_violation",
    }
    storage.register_regression_scenario(
        scenario_id, "protocol", runner_version=RUNNER_VERSION, manifest=manifest
    )

    try:
        gate = AdmissionGate(storage, {"gas_sensor_01": {"home/sensors/gas/ppm"}})
        context = ProvenanceContext(
            session_id=session_id,
            device_cn=identity.common_name,
            cert_fingerprint=identity.certificate_fingerprint,
            source_type="live_validated",
            admission_policy_version=RUNNER_VERSION,
            device_id="gas_sensor_01",
            generation=identity.generation,
            scenario_id=scenario_id,
        )
        registered_boot_id = _register_authenticated_boot(
            gate,
            cert_pointer=cert_pointer,
            revocation_ledger=revocation_ledger,
            session_id=session_id,
            identity=identity,
            sensor_key=sensor_key,
        )
        first_frame = encode_sensor_frame(
            device_id="gas_sensor_01",
            boot_id=registered_boot_id,
            sequence=1,
            device_timestamp_ns=time.time_ns(),
            topic="home/sensors/gas/ppm",
            payload=str(350 + (seed % 11)),
        )
        first_result = gate.evaluate_v6_and_ingest(context, "home/sensors/gas/ppm", first_frame)
        if not first_result.admitted:
            raise RegressionCaptureError(f"protocol setup frame rejected: {first_result.reason}")

        if family == "authenticated_replay_duplicate_sequence":
            invalid_frame = first_frame
        else:
            invalid_frame = encode_sensor_frame(
                device_id="gas_sensor_01",
                boot_id=uuid.uuid4().hex,
                sequence=1,
                device_timestamp_ns=time.time_ns(),
                topic="home/sensors/gas/ppm",
                payload=str(351 + (seed % 11)),
            )

        rejected = gate.evaluate_v6_and_ingest(context, "home/sensors/gas/ppm", invalid_frame)
        if rejected.admitted or rejected.reason not in set(expected_reasons):
            raise RegressionCaptureError(
                f"{family}: expected one of {expected_reasons}, got admitted={rejected.admitted}, "
                f"reason={rejected.reason}"
            )
        indexed = storage.complete_regression_scenario(scenario_id)
        if indexed != 1:
            raise RegressionCaptureError(f"{family}: expected one indexed protocol capture, got {indexed}")
        _finalize_live_session(storage, session_id)
        return {
            "scenario_id": scenario_id,
            "session_id": session_id,
            "corpus_class": "protocol_violation",
            "family": family,
            "reason_code": rejected.reason,
            "captures_indexed": indexed,
        }
    except BaseException:
        _fail_scenario(storage, scenario_id)
        raise


def capture_hardware_fault_family(
    storage: LumenStorageManager,
    cert_pointer: str,
    generator_scenario_id: str,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    spec = FROZEN_SCENARIOS[generator_scenario_id]
    if spec.category != "fault":
        raise ValueError("hardware capture requires a frozen fault scenario")
    if steps < 16:
        raise ValueError("fault_steps must be at least 16")

    family = spec.attack_family
    scenario_id = _new_scenario_id(family, seed)
    session_id = _new_live_session(storage, scenario_id)
    identity = load_active_device_identity(cert_pointer, spec.target_device)
    manifest = {
        "family": family,
        "generator": "attacks.advanced_generators.iter_scenario_frames",
        "generator_scenario_id": generator_scenario_id,
        "generator_split": spec.split,
        "generator_seed": int(spec.seed),
        "runner_seed": int(seed),
        "steps": int(steps),
        "target_device": spec.target_device,
        "target_topic": spec.target_topic,
        "parameters": spec.params,
        "ground_truth": "confirmed_hardware_fault",
        "fault_classifier": "OperationalFaultClassifier",
        "timestamp_policy": "preserve_observed_inter_arrival_gaps",
    }
    storage.register_regression_scenario(
        scenario_id, "hardware_fault", runner_version=RUNNER_VERSION, manifest=manifest
    )

    try:
        classifier = OperationalFaultClassifier(frozen_min_steps=8)
        frames = list(iter_scenario_frames(generator_scenario_id, steps=steps))
        observed: list[tuple[Any, float, Any]] = []
        previous_step: int | None = None
        detected = None
        for frame in frames:
            if frame.dropped:
                continue
            raw = bytes(frame.payload)
            try:
                numeric_value = float(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RegressionCaptureError(f"{generator_scenario_id}: non-numeric fault payload") from exc
            if previous_step is None:
                delta_t = float(frame.delta_t_sec)
            else:
                # Elapsed receive time includes the frames dropped by the generator.
                delta_t = float(frame.step - previous_step) * float(frame.delta_t_sec)
            previous_step = int(frame.step)
            assessment = classifier.evaluate(
                frame.device_id, frame.topic, numeric_value, delta_t,
                operation=str(frame.operation),
            )
            observed.append((frame, delta_t, assessment))
            if assessment.status == "FAULT":
                detected = assessment

        expected_fault = (
            "FROZEN_SENSOR" if generator_scenario_id == "fault1_frozen_sensor"
            else "PACKET_DROP_DEGRADATION"
        )
        if detected is None or detected.fault_type != expected_fault:
            detail = detected.detail if detected is not None else "classifier never entered FAULT"
            raise RegressionCaptureError(
                f"{family}: expected {expected_fault}, got "
                f"{detected.fault_type if detected else 'NONE'} ({detail})"
            )

        base_ts_ns = time.time_ns()
        elapsed_sec = 0.0
        capture_count = 0
        for frame, delta_t, assessment in observed:
            elapsed_sec += delta_t
            ts_ns = base_ts_ns + int(round(elapsed_sec * 1_000_000_000))
            _persist_lab_capture(
                storage,
                session_id=session_id,
                scenario_id=scenario_id,
                identity=identity,
                device_id=frame.device_id,
                topic=frame.topic,
                source="fault_lab",
                reason_code=expected_fault,
                label="confirmed_hardware_fault",
                raw_payload=bytes(frame.payload),
                gateway_ts_ns=ts_ns,
                metadata={
                    "family": family,
                    "generator_scenario_id": generator_scenario_id,
                    "step": int(frame.step),
                    "operation": str(frame.operation),
                    "delta_t_sec": float(delta_t),
                    "classifier_status": str(assessment.status),
                    "classifier_fault_type": str(assessment.fault_type),
                },
            )
            capture_count += 1

        indexed = storage.complete_regression_scenario(scenario_id)
        if indexed != capture_count or indexed < 8:
            raise RegressionCaptureError(f"{family}: expected at least 8 indexed captures, got {indexed}")
        _finalize_live_session(storage, session_id)
        return {
            "scenario_id": scenario_id,
            "session_id": session_id,
            "corpus_class": "hardware_fault",
            "family": family,
            "fault_type": expected_fault,
            "observed_events": len(observed),
            "captures_indexed": indexed,
        }
    except BaseException:
        _fail_scenario(storage, scenario_id)
        raise


def run_capture_suite(
    db_dir: str | Path = "logs/db",
    *,
    seed: int = 43,
    attack_steps: int = 72,
    fault_steps: int = 90,
) -> list[dict[str, Any]]:
    """Capture four attack families, two protocol variants, and two faults."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if attack_steps < 16 or fault_steps < 16:
        raise ValueError("attack_steps and fault_steps must both be at least 16")

    with tempfile.TemporaryDirectory(prefix="lumen-stage-b4-pki-") as temp_dir:
        pki = CertAuthority.create_session(
            Path(temp_dir),
            identities=(
                "router", "gateway", "ids", "honeypot",
                "gas_sensor_01", "gas_sensor_02", "vibration_sensor_01", "pir_sensor_01",
            ),
            leaf_ttl_days=7,
            ca_validity_days=3650,
        )
        storage = LumenStorageManager(
            paths=DBPaths.from_dir(db_dir), enable_writers=True
        )
        try:
            results: list[dict[str, Any]] = []
            for generator_scenario_id in ATTACK_SCENARIOS:
                results.append(capture_attack_family(
                    storage, str(pki.pointer_path), generator_scenario_id,
                    seed=seed, steps=attack_steps,
                ))
            for family in PROTOCOL_SCENARIOS:
                results.append(capture_protocol_family(
                    storage, str(pki.pointer_path), str(pki.revocation_path),
                    family, seed=seed,
                ))
            for generator_scenario_id in FAULT_SCENARIOS:
                results.append(capture_hardware_fault_family(
                    storage, str(pki.pointer_path), generator_scenario_id,
                    seed=seed, steps=fault_steps,
                ))
            storage.flush_all()
            return results
        finally:
            storage.flush_all()
            storage.close()
            pki.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--attack-steps", type=int, default=72)
    parser.add_argument("--fault-steps", type=int, default=90)
    args = parser.parse_args()
    try:
        results = run_capture_suite(
            args.db_dir, seed=args.seed,
            attack_steps=args.attack_steps, fault_steps=args.fault_steps,
        )
        counts = {
            category: sum(item["corpus_class"] == category for item in results)
            for category in ("adversarial", "protocol_violation", "hardware_fault")
        }
        print(json.dumps({
            "status": "ok",
            "results": results,
            "completed_scenarios": len(results),
            "scenario_counts_by_class": counts,
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[STAGE B4 CAPTURE FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
