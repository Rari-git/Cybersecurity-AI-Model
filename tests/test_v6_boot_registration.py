from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

from sensors.v6_publisher import V6SensorPublisher
from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.telemetry_db import DBPaths, LumenStorageManager
from utils.boot_registration import (
    BootRegistrationAuthority,
    BootRegistrationError,
    create_boot_registration_proof,
    verify_gateway_challenge,
)
from utils.cert_lifecycle import CertAuthority
from utils.device_identity import load_active_device_identity, load_active_device_signing_key
from utils.sensor_payload import parse_sensor_frame


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    root = tmp_path_factory.mktemp("v6-boot-pki")
    authority = CertAuthority.create_session(
        root,
        identities=("router", "gateway", "ids", "honeypot", "gas_sensor_01"),
        leaf_ttl_days=7,
        ca_validity_days=3650,
    )
    yield authority
    authority.cleanup()


@pytest.fixture
def environment(pki, tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    session_id = f"boot-test-{time.time_ns()}"
    storage.register_baseline_session(
        session_id=session_id,
        source_type="live_validated",
        config={"test": True},
        approved_for_training=False,
    )
    gate = AdmissionGate(storage, {"gas_sensor_01": {"home/sensors/gas/ppm"}})
    sensor_identity = load_active_device_identity(pki.pointer_path, "gas_sensor_01")
    sensor_key_identity, sensor_key = load_active_device_signing_key(pki.pointer_path, "gas_sensor_01")
    gateway_identity = load_active_device_identity(pki.pointer_path, "gateway")
    context = ProvenanceContext(
        session_id=session_id,
        device_cn=sensor_identity.common_name,
        cert_fingerprint=sensor_identity.certificate_fingerprint,
        source_type="live_validated",
        admission_policy_version="v6.0-boot-registration-test",
        device_id="gas_sensor_01",
        generation=sensor_identity.generation,
    )
    service = BootRegistrationAuthority(
        gate,
        cert_pointer=str(pki.pointer_path),
        revocation_ledger=str(pki.revocation_path),
        session_id=session_id,
        ttl_sec=5.0,
    )
    yield {
        "storage": storage,
        "gate": gate,
        "service": service,
        "context": context,
        "sensor_identity": sensor_identity,
        "sensor_key_identity": sensor_key_identity,
        "sensor_key": sensor_key,
        "gateway_identity": gateway_identity,
        "pointer": str(pki.pointer_path),
    }
    storage.flush_all()
    storage.close()


def _registered_boot(env, boot_id: str):
    challenge = env["service"].issue_challenge("gas_sensor_01", boot_id)
    verify_gateway_challenge(
        challenge,
        env["sensor_identity"],
        env["gateway_identity"],
    )
    proof = create_boot_registration_proof(
        challenge,
        env["sensor_identity"],
        env["sensor_key"],
    )
    result = env["service"].complete_registration(
        proof,
        expected_device_id="gas_sensor_01",
    )
    return challenge, proof, result


def test_valid_boot_registration_is_bound_to_active_device_and_gateway_certificates(environment):
    env = environment
    boot_id = "0123456789abcdef0123456789abcdef"
    challenge, proof, result = _registered_boot(env, boot_id)

    assert result["ok"] is True
    assert result["device_id"] == "gas_sensor_01"
    assert result["generation"] == env["sensor_identity"].generation
    assert result["boot_id"] == boot_id
    assert result["sequence_start"] == 0
    assert challenge["gateway_common_name"] == env["gateway_identity"].common_name
    assert proof["challenge"]["device_certificate_fingerprint_sha256"] == env["sensor_identity"].certificate_fingerprint.hex()

    freshness = env["storage"].load_device_freshness("gas_sensor_01")["gas_sensor_01"]
    assert freshness["active_boot_id"] == boot_id
    assert freshness["last_sequence"] == 0


def test_boot_registration_proof_is_single_use_and_cannot_reset_sequence(environment):
    env = environment
    boot_id = "11111111111111111111111111111111"
    _challenge, proof, _result = _registered_boot(env, boot_id)

    from utils.sensor_payload import encode_sensor_frame

    frame = encode_sensor_frame(
        device_id="gas_sensor_01",
        boot_id=boot_id,
        sequence=7,
        device_timestamp_ns=time.time_ns(),
        topic="home/sensors/gas/ppm",
        payload="351",
    )
    admitted = env["gate"].evaluate_v6_and_ingest(env["context"], "home/sensors/gas/ppm", frame)
    assert admitted.admitted is True

    with pytest.raises(BootRegistrationError, match="CHALLENGE_UNKNOWN_OR_REPLAYED"):
        env["service"].complete_registration(proof, expected_device_id="gas_sensor_01")

    freshness = env["storage"].load_device_freshness("gas_sensor_01")["gas_sensor_01"]
    assert freshness["active_boot_id"] == boot_id
    assert freshness["last_sequence"] == 7


def test_expired_boot_challenge_is_rejected_without_changing_active_epoch(environment):
    env = environment
    active_boot = "22222222222222222222222222222222"
    _registered_boot(env, active_boot)

    monotonic_now = [100.0]
    expired_service = BootRegistrationAuthority(
        env["gate"],
        cert_pointer=env["pointer"],
        revocation_ledger=str(env["service"].revocation_ledger),
        session_id=env["context"].session_id,
        ttl_sec=2.0,
        monotonic=lambda: monotonic_now[0],
    )
    candidate_boot = "33333333333333333333333333333333"
    challenge = expired_service.issue_challenge("gas_sensor_01", candidate_boot)
    proof = create_boot_registration_proof(challenge, env["sensor_identity"], env["sensor_key"])
    monotonic_now[0] += 3.0

    with pytest.raises(BootRegistrationError, match="CHALLENGE_EXPIRED"):
        expired_service.complete_registration(proof, expected_device_id="gas_sensor_01")

    freshness = env["storage"].load_device_freshness("gas_sensor_01")["gas_sensor_01"]
    assert freshness["active_boot_id"] == active_boot
    assert freshness["last_sequence"] == 0


@dataclass
class _Sample:
    key_expr: str
    payload: bytes


class _FakeSubscriber:
    def __init__(self, session, topic, callback):
        self.session = session
        self.topic = topic
        self.callback = callback

    def undeclare(self):
        self.session.subscribers.pop(self.topic, None)


class _FakeZenohSession:
    def __init__(self, authority, *, drop_initial_requests: int = 0):
        self.authority = authority
        self.subscribers = {}
        self.published = []
        self.drop_initial_requests = drop_initial_requests
        self.request_attempts = 0

    def declare_subscriber(self, topic, callback):
        self.subscribers[topic] = callback
        return _FakeSubscriber(self, topic, callback)

    def _deliver(self, topic, document):
        callback = self.subscribers.get(topic)
        if callback is not None:
            callback(_Sample(topic, json.dumps(document, sort_keys=True, separators=(",", ":")).encode()))

    def put(self, topic, payload):
        raw = payload if isinstance(payload, bytes) else bytes(payload)
        if topic.startswith("control/boot/request/"):
            self.request_attempts += 1
            if self.request_attempts <= self.drop_initial_requests:
                return
            request = json.loads(raw.decode())
            challenge = self.authority.issue_challenge(request["device_id"], request["boot_id"])
            self._deliver(f"control/boot/challenge/{request['device_id']}", challenge)
            return
        if topic.startswith("control/boot/proof/"):
            proof = json.loads(raw.decode())
            device_id = topic.rsplit("/", 1)[-1]
            result = self.authority.complete_registration(proof, expected_device_id=device_id)
            self._deliver(f"control/boot/result/{device_id}", result)
            return
        self.published.append((topic, raw))


def test_v6_sensor_publisher_registers_at_startup_and_emits_monotonic_frames(environment):
    env = environment
    session = _FakeZenohSession(env["service"])
    publisher = V6SensorPublisher(
        session,
        device_id="gas_sensor_01",
        ingress_key="ingest/devices/gas_sensor_01/sensors/gas/ppm",
        target_topic="home/sensors/gas/ppm",
        cert_pointer=env["pointer"],
        revocation_ledger=str(env["service"].revocation_ledger),
        boot_id="44444444444444444444444444444444",
    )
    assert publisher.registration["boot_id"] == publisher.boot_id
    assert publisher.publish("351") == 1
    assert publisher.publish("352") == 2

    first = parse_sensor_frame("home/sensors/gas/ppm", session.published[0][1]).frame
    second = parse_sensor_frame("home/sensors/gas/ppm", session.published[1][1]).frame
    assert first is not None and second is not None
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.boot_id == second.boot_id == publisher.boot_id


def test_sensor_retries_boot_request_when_gateway_subscriber_is_not_ready(environment):
    env = environment
    session = _FakeZenohSession(env["service"], drop_initial_requests=1)
    publisher = V6SensorPublisher(
        session,
        device_id="gas_sensor_01",
        ingress_key="ingest/devices/gas_sensor_01/sensors/gas/ppm",
        target_topic="home/sensors/gas/ppm",
        cert_pointer=env["pointer"],
        revocation_ledger=str(env["service"].revocation_ledger),
        timeout_sec=3.0,
        retry_interval_sec=0.05,
        boot_id="55555555555555555555555555555555",
    )
    assert session.request_attempts >= 2
    assert publisher.registration["boot_id"] == publisher.boot_id
    assert publisher.publish("356") == 1


def test_retries_for_same_boot_epoch_return_same_outstanding_challenge(environment):
    service = environment["service"]
    boot_id = "66666666666666666666666666666666"
    first = service.issue_challenge("gas_sensor_01", boot_id)
    second = service.issue_challenge("gas_sensor_01", boot_id)
    assert first == second
    assert first["challenge_id"] == second["challenge_id"]


def test_sensor_and_both_router_templates_share_exact_boot_key_expressions():
    import json5
    from pathlib import Path
    from utils.boot_registration import boot_topic

    root = Path(__file__).resolve().parents[1]
    device_ids = ("door_sensor_01", "gas_sensor_01", "gas_sensor_02", "vibration_sensor_01", "pir_sensor_01")
    for filename in ("zenohd_router.json5", "zenoh_acl_example.json5"):
        document = json5.loads((root / "config" / filename).read_text(encoding="utf-8"))
        rules = {rule["id"]: rule for rule in document["access_control"]["rules"]}
        for kind, rule_id in (
            ("request", "in_put_boot_request_"),
            ("proof", "in_put_boot_proof_"),
            ("challenge", "sub_boot_challenge_"),
            ("result", "sub_boot_result_"),
        ):
            for device_id in device_ids:
                assert rules[rule_id + device_id]["key_exprs"] == [boot_topic(kind, device_id)]
        assert rules["sub_boot_request"]["key_exprs"] == ["control/boot/request/**"]
        assert rules["sub_boot_proof"]["key_exprs"] == ["control/boot/proof/**"]
        assert rules["out_put_boot_challenge"]["key_exprs"] == ["control/boot/challenge/**"]
        assert rules["out_put_boot_result"]["key_exprs"] == ["control/boot/result/**"]
