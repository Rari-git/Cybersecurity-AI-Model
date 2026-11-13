"""End-to-end smoke test for an already-running V5 Router/Gateway/IDS/Sensor stack.

Run manually with the active run_all session environment, or let run_all execute
it when LUMEN_RUN_LIVE_SMOKE=1. It uses real mTLS Zenoh clients and verifies:
  * both gas sensors on the shared output topic retain distinct envelope IDs;
  * each envelope matches the active PKI generation/CN/fingerprint;
  * unauthorized sensor publication to internal envelope topics is blocked; and
  * a Gateway quarantine + staged 3-phase release (PREPARE -> COMMIT -> FINALIZE)
    return exact correlated PEP ACKs and re-open sensor ingress.
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

from utils.device_identity import load_active_device_identity
from utils.identity_envelope import IdentityEnvelope, validate_active_identity, validate_envelope_route
from utils.sensor_payload import parse_sensor_payload
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


def _send_policy_and_wait_ack(
    ids_session,
    device_id: str,
    command: dict,
    expected_action: str,
    timeout: float = 8.0,
) -> dict:
    decision_id = str(command["decision_id"])
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
        ids_session.put(
            f"control/policy/{device_id}",
            json.dumps(command, separators=(",", ":")).encode("utf-8"),
        )
        ack = _wait_for(received, lambda _item: True, timeout, f"PEP ACK for {expected_action}")
        expected = {
            "decision_id": decision_id,
            "device_id": device_id,
            "requested_action": expected_action,
            "applied_action": expected_action,
            "result": "CONFIRMED",
        }
        mismatches = {key: (ack.get(key), value) for key, value in expected.items() if ack.get(key) != value}
        if mismatches:
            raise RuntimeError(f"PEP ACK mismatch for {expected_action}: {mismatches}")
        if not ack.get("gateway_instance_id"):
            raise RuntimeError("PEP ACK missing gateway_instance_id")
        return ack
    finally:
        sub.undeclare()


def _pep_quarantine(ids_session, device_id: str, timeout: float = 8.0) -> dict:
    decision_id = f"smoke-quar-{uuid.uuid4().hex}"
    command = {
        "decision_id": decision_id,
        "device_id": device_id,
        "phase": "APPLY",
        "target_state": "SHADOW_QUARANTINE",
        "issued_at_unix": time.time(),
        "source": "v5_live_runtime_smoke",
    }
    return _send_policy_and_wait_ack(
        ids_session, device_id, command, "SHADOW_QUARANTINE", timeout=timeout
    )


def _pep_staged_release(ids_session, device_id: str, timeout: float = 8.0) -> dict:
    tx_id = f"smoke-rel-{uuid.uuid4().hex}"
    prepare_id = f"{tx_id}-prepare"
    ready_id = f"{tx_id}-ready"
    finalize_id = f"{tx_id}-finalize"

    ack_prepare = _send_policy_and_wait_ack(
        ids_session,
        device_id,
        {
            "decision_id": prepare_id,
            "device_id": device_id,
            "phase": "PREPARE_RELEASE",
            "target_state": "ACTIVE",
            "issued_at_unix": time.time(),
            "source": "v5_live_runtime_smoke",
        },
        "RELEASE_PREPARE",
        timeout=timeout,
    )
    ack_ready = _send_policy_and_wait_ack(
        ids_session,
        device_id,
        {
            "decision_id": ready_id,
            "prepare_decision_id": prepare_id,
            "device_id": device_id,
            "phase": "COMMIT_RELEASE",
            "target_state": "ACTIVE",
            "issued_at_unix": time.time(),
            "source": "v5_live_runtime_smoke",
        },
        "RELEASE_READY",
        timeout=timeout,
    )
    ids_session.put(
        f"control/policy/{device_id}",
        json.dumps(
            {
                "decision_id": finalize_id,
                "prepare_decision_id": prepare_id,
                "ready_decision_id": ready_id,
                "device_id": device_id,
                "phase": "FINALIZE_RELEASE",
                "target_state": "ACTIVE",
                "issued_at_unix": time.time(),
                "source": "v5_live_runtime_smoke",
            },
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    return {
        "prepare_ack": ack_prepare,
        "ready_ack": ack_ready,
        "finalize_decision_id": finalize_id,
    }


def _wait_for_smoke_envelopes(
    received: queue.Queue[tuple[str, bytes]],
    active: dict,
    allowed_topics: dict[str, str],
    expected_payloads: dict[str, bytes],
    timeout: float,
) -> tuple[dict[str, IdentityEnvelope], dict[str, bytes]]:
    envelopes: dict[str, IdentityEnvelope] = {}
    payloads: dict[str, bytes] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and set(envelopes) != set(expected_payloads):
        try:
            key, raw = received.get(timeout=max(0.01, min(0.25, deadline - time.monotonic())))
        except queue.Empty:
            continue
        try:
            env, payload = IdentityEnvelope.decode(raw)
            validate_envelope_route(env, key, allowed_topics)
            validate_active_identity(env, active[env.device_id])
        except Exception:
            continue
        # Ignore background telemetry emitted concurrently by Virtual Sensors;
        # match the specific smoke-test canonical payload for each device.
        if payload != expected_payloads.get(env.device_id):
            continue
        envelopes[env.device_id] = env
        payloads[env.device_id] = payload
    return envelopes, payloads


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

    active = {
        device_id: load_active_device_identity(pointer, device_id)
        for device_id in DEVICES
    }
    allowed_topics = {device_id: spec["topic"] for device_id, spec in DEVICES.items()}
    expected_canonical = {
        device_id: parse_sensor_payload(spec["topic"], spec["payload"]).canonical_text.encode("utf-8")
        for device_id, spec in DEVICES.items()
    }

    ids_session = None
    sensor_sessions: dict[str, object] = {}
    envelope_sub = None
    acl_probe_sub = None
    ack_quarantine = None
    release_result = None
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
            sensor_sessions[device_id] = sensor
            sensor.put(spec["ingress"], spec["payload"])

        envelopes, payloads = _wait_for_smoke_envelopes(
            received, active, allowed_topics, expected_canonical, timeout
        )

        missing = set(DEVICES) - set(envelopes)
        if missing:
            raise TimeoutError(f"no valid authenticated Gateway envelope for: {sorted(missing)}")
        if envelopes["gas_sensor_01"].message_id == envelopes["gas_sensor_02"].message_id:
            raise RuntimeError("Gateway reused message_id across distinct sensor events")
        if payloads != expected_canonical:
            raise RuntimeError(f"Gateway envelope payload mismatch: got={payloads!r} expected={expected_canonical!r}")

        # An authenticated sensor must not be able to impersonate Gateway on
        # the internal envelope topic.
        try:
            sensor_sessions["gas_sensor_01"].put(
                f"{INTERNAL_PREFIX}gas_sensor_01", unauthorized_probe_payload
            )
        except Exception:
            pass
        time.sleep(float(os.environ.get("LUMEN_ACL_PROBE_SETTLE_SEC", "0.5")))
        if unauthorized_probe_seen.is_set():
            raise RuntimeError("router ACL allowed a sensor to publish an internal IdentityEnvelope")

        # Mark the device as potentially quarantined before sending the command:
        # if an ACK is lost after Gateway enforcement, finally attempts release.
        quarantined = True
        ack_quarantine = _pep_quarantine(ids_session, "gas_sensor_01")
        release_result = _pep_staged_release(ids_session, "gas_sensor_01")
        quarantined = False

        # Verify Gateway actually re-opened ingress after FINALIZE_RELEASE.
        post_release_payload = b"367"
        sensor_sessions["gas_sensor_01"].put(DEVICES["gas_sensor_01"]["ingress"], post_release_payload)
        post_envelopes, _ = _wait_for_smoke_envelopes(
            received,
            active,
            allowed_topics,
            {"gas_sensor_01": post_release_payload},
            timeout=5.0,
        )
        if "gas_sensor_01" not in post_envelopes:
            raise RuntimeError("Gateway did not re-open ingress after FINALIZE_RELEASE")

        return {
            "status": "PASS",
            "device_ids": sorted(envelopes),
            "shared_topic": DEVICES["gas_sensor_01"]["topic"],
            "generations": {key: envelopes[key].generation for key in envelopes},
            "gateway_instance_ids": sorted({env.gateway_instance_id for env in envelopes.values()}),
            "unauthorized_internal_publish_blocked": True,
            "pep_quarantine_ack": ack_quarantine,
            "pep_release": release_result,
            "post_release_ingress_verified": True,
        }
    finally:
        if quarantined and ids_session is not None:
            try:
                _pep_staged_release(ids_session, "gas_sensor_01", timeout=4.0)
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
        for sensor in sensor_sessions.values():
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