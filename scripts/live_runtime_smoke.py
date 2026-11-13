"""End-to-end smoke test for an already-running V5 Router/Gateway/IDS/Sensor stack.

Run manually with the active run_all session environment, or let run_all execute
it when LUMEN_RUN_LIVE_SMOKE=1. It uses real mTLS Zenoh clients and verifies:
  * both gas sensors on the shared output topic retain distinct envelope IDs;
  * each envelope matches the active PKI generation/CN/fingerprint; and
  * a Gateway quarantine + release each return an exact correlated PEP ACK.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cert_lifecycle import CertAuthority
from utils.device_identity import load_active_device_identity
from utils.identity_envelope import IdentityEnvelope, validate_active_identity, validate_envelope_route
from utils.zenoh_conn import open_client_session

DEVICES = {
    "gas_sensor_01": {
        "ingress": "ingest/devices/gas_sensor_01/sensors/gas/ppm",
        "topic": "home/sensors/gas/ppm",
        "payload": b"351",
    },
    "gas_sensor_02": {
        "ingress": "ingest/devices/gas_sensor_02/sensors/gas/ppm",
        "topic": "home/sensors/gas/ppm",
        "payload": b"359",
    },
}
INTERNAL_PREFIX = "internal/ids/telemetry/"
ACK_PREFIX = "control/pep_ack/"


def _sample_payload(sample) -> bytes:
    value = sample.payload
    return value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)


def _wait_for(queue_obj: queue.Queue, predicate, timeout: float, description: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = queue_obj.get(timeout=max(0.01, min(0.25, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if predicate(value):
            return value
    raise TimeoutError(f"timed out waiting for {description}")


def _pep_transition(ids_session, device_id: str, target_state: str, timeout: float = 8.0) -> dict:
    decision_id = f"smoke-{uuid.uuid4().hex}"
    received: queue.Queue[dict] = queue.Queue()

    def on_ack(sample) -> None:
        try:
            doc = json.loads(_sample_payload(sample).decode("utf-8"))
            if str(sample.key_expr) == f"{ACK_PREFIX}{decision_id}":
                received.put(doc)
        except Exception:
            return

    sub = ids_session.declare_subscriber(f"{ACK_PREFIX}{decision_id}", on_ack)
    try:
        command = {
            "decision_id": decision_id,
            "device_id": device_id,
            "target_state": target_state,
            "issued_at_unix": time.time(),
            "source": "v5_live_runtime_smoke",
        }
        ids_session.put(
            f"control/policy/{device_id}",
            json.dumps(command, separators=(",", ":")).encode("utf-8"),
        )
        ack = _wait_for(received, lambda _item: True, timeout, f"PEP ACK for {target_state}")
        expected = {
            "decision_id": decision_id,
            "device_id": device_id,
            "requested_action": target_state,
            "applied_action": target_state,
            "result": "CONFIRMED",
        }
        mismatches = {key: (ack.get(key), value) for key, value in expected.items() if ack.get(key) != value}
        if mismatches:
            raise RuntimeError(f"PEP ACK mismatch for {target_state}: {mismatches}")
        if not ack.get("gateway_instance_id"):
            raise RuntimeError("PEP ACK missing gateway_instance_id")
        return ack
    finally:
        sub.undeclare()


def run_smoke(timeout: float = 15.0) -> dict:
    pointer = os.environ.get("LUMEN_CERT_POINTER", "").strip()
    ledger = os.environ.get("LUMEN_CERT_REVOCATIONS", "").strip()
    endpoint = os.environ.get("LUMEN_ZENOH_ENDPOINT", "")
    if not pointer or not Path(pointer).is_file():
        raise RuntimeError("LUMEN_CERT_POINTER must point to the active PKI pointer")
    if not ledger or not Path(ledger).is_file():
        raise RuntimeError("LUMEN_CERT_REVOCATIONS must point to the active certificate ledger")
    if not endpoint.startswith("tls/"):
        raise RuntimeError("live smoke requires a tls/ Zenoh endpoint")
    # Fail early if the active PKI doesn't contain both shared-topic identities.
    active = {
        device_id: load_active_device_identity(pointer, device_id)
        for device_id in DEVICES
    }
    allowed_topics = {device_id: spec["topic"] for device_id, spec in DEVICES.items()}

    ids_session = None
    sensor_sessions = []
    envelope_sub = None
    acl_probe_sub = None
    ack_quarantine = None
    ack_release = None
    quarantined = False
    received: queue.Queue[tuple[str, bytes]] = queue.Queue()
    unauthorized_probe_seen = threading.Event()
    unauthorized_probe_payload = f"lumen-unauthorized-acl-probe-{uuid.uuid4().hex}".encode("ascii")
    try:
        ids_session = open_client_session("ids")

        def on_envelope(sample) -> None:
            try:
                received.put((str(sample.key_expr), _sample_payload(sample)))
            except Exception:
                return

        envelope_sub = ids_session.declare_subscriber(f"{INTERNAL_PREFIX}**", on_envelope)

        def on_acl_probe(sample) -> None:
            try:
                if _sample_payload(sample) == unauthorized_probe_payload:
                    unauthorized_probe_seen.set()
            except Exception:
                return

        acl_probe_sub = ids_session.declare_subscriber(
            f"{INTERNAL_PREFIX}gas_sensor_01", on_acl_probe
        )
        # Send a separate authenticated event from each device despite their
        # common downstream home/sensors/gas/ppm topic.
        for device_id, spec in DEVICES.items():
            sensor = open_client_session(device_id)
            sensor_sessions.append(sensor)
            sensor.put(spec["ingress"], spec["payload"])

        envelopes: dict[str, IdentityEnvelope] = {}
        payloads: dict[str, bytes] = {}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and set(envelopes) != set(DEVICES):
            try:
                key, raw = received.get(timeout=max(0.01, min(0.25, deadline - time.monotonic())))
            except queue.Empty:
                continue
            try:
                env, payload = IdentityEnvelope.decode(raw)
                validate_envelope_route(env, key, allowed_topics)
                validate_active_identity(env, active[env.device_id])
            except Exception:
                # Ignore unrelated or invalid messages; only count a fully
                # validated Gateway envelope toward the two-device proof.
                continue
            envelopes[env.device_id] = env
            payloads[env.device_id] = payload

        missing = set(DEVICES) - set(envelopes)
        if missing:
            raise TimeoutError(f"no valid authenticated Gateway envelope for: {sorted(missing)}")
        if envelopes["gas_sensor_01"].message_id == envelopes["gas_sensor_02"].message_id:
            raise RuntimeError("Gateway reused message_id across distinct sensor events")
        if payloads != {key: DEVICES[key]["payload"] for key in DEVICES}:
            raise RuntimeError("Gateway envelope payload mismatch")

        # An authenticated sensor must not be able to impersonate Gateway on
        # the internal envelope topic. ACLs may reject the put synchronously or
        # silently drop it; either way the IDS subscriber must never receive it.
        try:
            sensor_sessions[0].put(
                f"{INTERNAL_PREFIX}gas_sensor_01", unauthorized_probe_payload
            )
        except Exception:
            pass  # Explicit router rejection is an acceptable denial outcome.
        time.sleep(float(os.environ.get("LUMEN_ACL_PROBE_SETTLE_SEC", "0.5")))
        if unauthorized_probe_seen.is_set():
            raise RuntimeError("router ACL allowed a sensor to publish an internal IdentityEnvelope")

        # Mark the device as potentially quarantined before sending the command:
        # if the ACK is lost after Gateway enforcement, finally must still attempt release.
        quarantined = True
        ack_quarantine = _pep_transition(ids_session, "gas_sensor_01", "SHADOW_QUARANTINE")
        ack_release = _pep_transition(ids_session, "gas_sensor_01", "ACTIVE")
        quarantined = False

        return {
            "status": "PASS",
            "device_ids": sorted(envelopes),
            "shared_topic": DEVICES["gas_sensor_01"]["topic"],
            "generations": {key: envelopes[key].generation for key in envelopes},
            "gateway_instance_ids": sorted({env.gateway_instance_id for env in envelopes.values()}),
            "unauthorized_internal_publish_blocked": True,
            "pep_quarantine_ack": ack_quarantine,
            "pep_release_ack": ack_release,
        }
    finally:
        # A failed smoke run must not leave the test device quarantined.
        if quarantined and ids_session is not None:
            try:
                _pep_transition(ids_session, "gas_sensor_01", "ACTIVE", timeout=4.0)
            except Exception:
                print("WARNING: unable to confirm smoke-test quarantine release", file=sys.stderr)
        if acl_probe_sub is not None:
            try:
                acl_probe_sub.undeclare()
            except Exception:
                pass
        if envelope_sub is not None:
            try:
                envelope_sub.undeclare()
            except Exception:
                pass
        for sensor in sensor_sessions:
            try:
                sensor.close()
            except Exception:
                pass
        if ids_session is not None:
            ids_session.close()


def main() -> int:
    try:
        result = run_smoke(float(os.environ.get("LUMEN_LIVE_SMOKE_TIMEOUT_SEC", "15")))
    except Exception as exc:
        print(f"LIVE RUNTIME SMOKE: FAIL: {exc}", file=sys.stderr)
        return 1
    print("LIVE RUNTIME SMOKE: PASS")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
