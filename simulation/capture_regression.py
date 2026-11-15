"""Stage B local regression capture runner.

Creates authentic local V6 freshness evidence and controlled quarantined traffic.
All outputs go to telemetry_incidents.db; live_validated sessions are never
eligible for clean-baseline training. No cloud services or external APIs are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.forensic_sink import ForensicQuarantineSink
from agent.gateway_ids import ZenohInlineGateway
from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.telemetry_db import DBPaths, LumenStorageManager
from utils.boot_registration import (
    BootRegistrationAuthority, create_boot_registration_proof, verify_gateway_challenge,
)
from utils.cert_lifecycle import CertAuthority
from utils.device_identity import load_active_device_identity, load_active_device_signing_key
from utils.sensor_payload import encode_sensor_frame

DEVICE = "gas_sensor_01"
TOPIC = "home/sensors/gas/ppm"
RUNNER_VERSION = "stage-b-capture-v1"


class _LocalShadowSession:
    def __init__(self):
        self.messages = []

    def put(self, topic: str, payload: bytes) -> None:
        self.messages.append((topic, bytes(payload)))


def _new_scenario_id(kind: str, seed: int) -> str:
    return f"stageb-{kind}-{seed}-{uuid.uuid4().hex[:10]}"


def _new_live_session(storage: LumenStorageManager, scenario_id: str) -> str:
    session_id = f"regression-live-{scenario_id}"
    storage.register_baseline_session(
        session_id=session_id,
        source_type="live_validated",
        config={"regression_scenario_id": scenario_id, "training_eligible": False},
        approved_for_training=False,
    )
    return session_id


def capture_authenticated_replay(
    storage: LumenStorageManager, cert_pointer: str, revocation_ledger: str, seed: int
) -> dict:
    scenario_id = _new_scenario_id("protocol-replay", seed)
    session_id = _new_live_session(storage, scenario_id)
    identity = load_active_device_identity(cert_pointer, DEVICE)
    _key_identity, sensor_key = load_active_device_signing_key(cert_pointer, DEVICE)
    gateway_identity = load_active_device_identity(cert_pointer, "gateway")
    manifest = {
        "scenario": "authenticated-v6-duplicate-sequence",
        "seed": seed,
        "device_id": DEVICE,
        "topic": TOPIC,
        "protocol_version": 6,
        "expected_reason": "REPLAY_OR_OUT_OF_ORDER_SEQUENCE",
        "device_certificate_sha256": identity.certificate_fingerprint.hex(),
    }
    storage.register_regression_scenario(
        scenario_id, "protocol", runner_version=RUNNER_VERSION, manifest=manifest
    )
    gate = AdmissionGate(storage, {DEVICE: {TOPIC}})
    context = ProvenanceContext(
        session_id=session_id,
        device_cn=identity.common_name,
        cert_fingerprint=identity.certificate_fingerprint,
        source_type="live_validated",
        admission_policy_version=RUNNER_VERSION,
        device_id=DEVICE,
        generation=identity.generation,
        scenario_id=scenario_id,
    )
    boot_id = uuid.uuid4().hex
    boot_authority = BootRegistrationAuthority(
        gate,
        cert_pointer=cert_pointer,
        revocation_ledger=revocation_ledger,
        session_id=session_id,
        ttl_sec=10.0,
    )
    challenge = boot_authority.issue_challenge(DEVICE, boot_id)
    verify_gateway_challenge(challenge, identity, gateway_identity)
    proof = create_boot_registration_proof(challenge, identity, sensor_key)
    registration = boot_authority.complete_registration(proof, expected_device_id=DEVICE)
    if registration.get("ok") is not True:
        storage.complete_regression_scenario(scenario_id, success=False)
        raise RuntimeError("authenticated boot registration failed")

    frame = encode_sensor_frame(
        device_id=DEVICE,
        boot_id=boot_id,
        sequence=1,
        device_timestamp_ns=time.time_ns(),
        topic=TOPIC,
        payload=str(350 + (seed % 7)),
    )
    first = gate.evaluate_v6_and_ingest(context, TOPIC, frame)
    replay = gate.evaluate_v6_and_ingest(context, TOPIC, frame)
    if not first.admitted or replay.admitted or replay.reason != "REPLAY_OR_OUT_OF_ORDER_SEQUENCE":
        storage.complete_regression_scenario(scenario_id, success=False)
        raise RuntimeError(
            f"replay scenario did not produce expected evidence: first={first.reason}, replay={replay.reason}"
        )
    indexed = storage.complete_regression_scenario(scenario_id)
    if indexed < 1:
        raise RuntimeError("replay capture was persisted but failed corpus qualification")
    storage.finalize_baseline_session(session_id, approved_for_training=False)
    return {"scenario_id": scenario_id, "session_id": session_id, "corpus_class": "protocol_violation", "captures_indexed": indexed}


def capture_quarantined_attack(
    storage: LumenStorageManager, cert_pointer: str, seed: int
) -> dict:
    scenario_id = _new_scenario_id("shadow-attack", seed)
    session_id = _new_live_session(storage, scenario_id)
    identity = load_active_device_identity(cert_pointer, DEVICE)
    manifest = {
        "scenario": "controlled-sensor-traffic-after-gateway-quarantine",
        "seed": seed,
        "device_id": DEVICE,
        "topic": TOPIC,
        "capture_path": "ZenohInlineGateway._route_to_shadow_sandbox -> ForensicQuarantineSink",
        "quarantine_assertion": "Gateway.set_shadow_quarantine returned true",
        "payload_sha256": hashlib.sha256(str(950 + (seed % 25)).encode()).hexdigest(),
    }
    storage.register_regression_scenario(
        scenario_id, "attack", runner_version=RUNNER_VERSION, manifest=manifest
    )

    # Construct a local Gateway facade without opening Zenoh or network sockets.
    # The actual Gateway quarantine and forensic route methods are invoked.
    gateway = ZenohInlineGateway.__new__(ZenohInlineGateway)
    gateway.cert_pointer = cert_pointer
    gateway.run_id = session_id
    gateway.session = _LocalShadowSession()
    gateway._shadow_lock = threading.Lock()
    gateway._shadow_quarantined = set()
    gateway._stats_lock = threading.Lock()
    gateway.stats = {}
    gateway.admission_gate = AdmissionGate(storage, {DEVICE: {TOPIC}})
    gateway.forensic_sink = ForensicQuarantineSink(storage=storage)

    if not gateway.set_shadow_quarantine(DEVICE, True):
        storage.complete_regression_scenario(scenario_id, success=False)
        raise RuntimeError("Gateway did not confirm shadow quarantine")
    attack_payload = str(950 + (seed % 25)).encode("ascii")
    gateway._route_to_shadow_sandbox(
        DEVICE,
        TOPIC,
        attack_payload,
        scenario_id=scenario_id,
        ground_truth_label="confirmed_attack",
    )
    indexed = storage.complete_regression_scenario(scenario_id)
    gateway.set_shadow_quarantine(DEVICE, False)
    if indexed < 1:
        raise RuntimeError("quarantined traffic was captured but failed corpus qualification")
    storage.finalize_baseline_session(session_id, approved_for_training=False)
    return {"scenario_id": scenario_id, "session_id": session_id, "corpus_class": "adversarial", "captures_indexed": indexed}


def run_capture(db_dir: str | Path, *, scenario: str = "all", seed: int = 43) -> list[dict]:
    if scenario not in {"all", "authenticated-replay", "quarantined-attack"}:
        raise ValueError("unsupported regression scenario")
    with tempfile.TemporaryDirectory(prefix="lumen-stage-b-pki-") as temp:
        pki = CertAuthority.create_session(
            Path(temp),
            identities=("router", "gateway", "ids", "honeypot", DEVICE),
            leaf_ttl_days=7,
            ca_validity_days=3650,
        )
        storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
        try:
            results = []
            if scenario in {"all", "authenticated-replay"}:
                results.append(capture_authenticated_replay(
                    storage, str(pki.pointer_path), str(pki.revocation_path), seed
                ))
            if scenario in {"all", "quarantined-attack"}:
                results.append(capture_quarantined_attack(storage, str(pki.pointer_path), seed))
            storage.flush_all()
            return results
        finally:
            storage.close()
            pki.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--scenario", choices=("all", "authenticated-replay", "quarantined-attack"), default="all")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    try:
        results = run_capture(args.db_dir, scenario=args.scenario, seed=args.seed)
        print(json.dumps({"status": "ok", "results": results}, indent=2))
        return 0
    except Exception as exc:
        print(f"[STAGE B CAPTURE FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
