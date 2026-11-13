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
from utils.identity_envelope import (
    IdentityEnvelope, validate_active_identity, validate_envelope_route,
    validate_gateway_binding,
)
from sensors.v6_publisher import V6SensorPublisher
from utils.boot_registration import (
    BOOT_REQUEST_PREFIX, BOOT_CHALLENGE_PREFIX, BOOT_PROOF_PREFIX,
    BOOT_RESULT_PREFIX,
)
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


def _wait_for_smoke_envelopes(
    q: queue.Queue,
    active: dict,
    allowed: dict[str, str],
    expected,
    timeout: float = 15.0,
):
    """Collect only fully validated Gateway envelopes for the requested devices.

    Background virtual sensors, unrelated Gateway traffic, malformed payloads,
    and stale identities are ignored. The return shape is ``(envelopes,
    payloads)`` to retain the V5 smoke helper contract.
    """
    expected_ids = set(expected)
    if not expected_ids:
        return {}, {}
    envelopes: dict[str, IdentityEnvelope] = {}
    payloads: dict[str, bytes] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and set(envelopes) != expected_ids:
        try:
            key, raw = q.get(timeout=max(0.01, min(0.25, deadline - time.monotonic())))
        except queue.Empty:
            continue
        try:
            env, payload = IdentityEnvelope.decode(raw)
            if env.device_id not in expected_ids:
                continue
            validate_envelope_route(env, key, allowed)
            identity = active.get(env.device_id)
            if identity is None:
                continue
            validate_active_identity(env, identity)
            gateway_identity = active.get("gateway")
            if gateway_identity is not None:
                validate_gateway_binding(env, gateway_identity)
        except Exception:
            continue
        envelopes[env.device_id] = env
        payloads[env.device_id] = payload
    missing = expected_ids - set(envelopes)
    if missing:
        raise TimeoutError(f"no valid authenticated Gateway envelope for: {sorted(missing)}")
    return envelopes, payloads


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
    active_gateway = load_active_device_identity(pointer, "gateway")
    active["gateway"] = active_gateway
    allowed_topics = {device_id: spec["topic"] for device_id, spec in DEVICES.items()}

    ids_session = None
    sensor_sessions = []
    v6_sensor_session = None
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

        # Preserve the V5 smoke helper semantics: background traffic is ignored,
        # only the two explicitly expected device identities count.
        envelopes, payloads = _wait_for_smoke_envelopes(
            received, active, allowed_topics, set(DEVICES), timeout=timeout
        )
        if envelopes["gas_sensor_01"].message_id == envelopes["gas_sensor_02"].message_id:
            raise RuntimeError("Gateway reused message_id across distinct sensor events")
        if payloads != {key: DEVICES[key]["payload"] for key in DEVICES}:
            raise RuntimeError("Gateway envelope payload mismatch")

        # V6 live protocol: register a fresh boot, admit two increasing
        # sequences, then replay the second frame and require an immediate
        # signed REJECTED envelope from Gateway.
        # Use a dedicated mTLS client for the V6 handshake. The V5 clients above
        # have already published test events; reusing one can hide lifecycle and
        # subscriber-readiness races behind its previous subscriptions.
        v6_sensor_session = open_client_session("gas_sensor_01")
        logger_topics = {
            "request": f"{BOOT_REQUEST_PREFIX}gas_sensor_01",
            "challenge": f"{BOOT_CHALLENGE_PREFIX}gas_sensor_01",
            "proof": f"{BOOT_PROOF_PREFIX}gas_sensor_01",
            "result": f"{BOOT_RESULT_PREFIX}gas_sensor_01",
        }
        print("[V6 BOOT] retrying same boot_id until Gateway responds; topics=" + json.dumps(logger_topics, sort_keys=True))
        v6_publisher = V6SensorPublisher(
            v6_sensor_session,
            device_id="gas_sensor_01",
            ingress_key=DEVICES["gas_sensor_01"]["ingress"],
            target_topic=DEVICES["gas_sensor_01"]["topic"],
            cert_pointer=pointer,
            revocation_ledger=ledger,
            timeout_sec=float(os.environ.get("LUMEN_BOOT_REGISTRATION_TIMEOUT_SEC", "60")),
            retry_interval_sec=float(os.environ.get("LUMEN_BOOT_REGISTRATION_RETRY_SEC", "0.25")),
        )
        v6_frames = {
            1: v6_publisher.build_frame("361"),
            2: v6_publisher.build_frame("362"),
        }
        v6_sensor_session.put(DEVICES["gas_sensor_01"]["ingress"], v6_frames[1])
        v6_sensor_session.put(DEVICES["gas_sensor_01"]["ingress"], v6_frames[2])

        def _wait_v6(sequence: int, status: str, wait_sec: float = timeout):
            deadline_v6 = time.monotonic() + wait_sec
            while time.monotonic() < deadline_v6:
                try:
                    key, raw = received.get(
                        timeout=max(0.01, min(0.25, deadline_v6 - time.monotonic()))
                    )
                except queue.Empty:
                    continue
                try:
                    env, payload = IdentityEnvelope.decode(raw)
                    if env.device_id != "gas_sensor_01" or env.protocol_version != 6:
                        continue
                    validate_envelope_route(env, key, allowed_topics)
                    validate_active_identity(env, active["gas_sensor_01"])
                    validate_gateway_binding(env, active_gateway)
                    if env.boot_id != v6_publisher.boot_id or env.sequence != sequence:
                        continue
                    if env.admission_status != status:
                        continue
                    return env, payload
                except Exception:
                    continue
            raise TimeoutError(
                f"no V6 envelope for gas_sensor_01 sequence={sequence} status={status}"
            )

        v6_first, v6_first_payload = _wait_v6(1, "ADMITTED")
        v6_second, v6_second_payload = _wait_v6(2, "ADMITTED")
        # Re-send the exact signed-by-device frame; no fresh sequence is assigned.
        v6_sensor_session.put(DEVICES["gas_sensor_01"]["ingress"], v6_frames[2])
        v6_replay, v6_replay_payload = _wait_v6(2, "REJECTED")
        if not v6_replay.admission_reason.startswith("REPLAY_OR_OUT_OF_ORDER_SEQUENCE"):
            raise RuntimeError(
                f"V6 replay was not rejected by freshness store: {v6_replay.admission_reason}"
            )
        # IDS should now issue a protocol quarantine asynchronously. Mark the
        # smoke device as potentially quarantined before doing any more checks,
        # so the finally block always attempts a confirmed release on failure.
        quarantined = True
        if (v6_first_payload, v6_second_payload, v6_replay_payload) != (b"361", b"362", b"362"):
            raise RuntimeError("V6 canonical payload mismatch in live envelope path")

        # An authenticated sensor must not be able to impersonate Gateway on
        # the internal envelope topic. ACLs may reject the put synchronously or
        # silently drop it; either way the IDS subscriber must never receive it.
        try:
            v6_sensor_session.put(
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
            "v6_boot_registration": {
                "device_id": v6_publisher.device_id,
                "boot_id": v6_publisher.boot_id,
                "generation": v6_publisher.registration["generation"],
            },
            "v6_admitted_sequences": [v6_first.sequence, v6_second.sequence],
            "v6_replay_rejected": {
                "sequence": v6_replay.sequence,
                "reason": v6_replay.admission_reason,
            },
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
        if v6_sensor_session is not None:
            try:
                v6_sensor_session.close()
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
