from __future__ import annotations

import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from utils.device_identity import ActiveDeviceIdentity
from utils.identity_envelope import (
    IdentityEnvelope,
    validate_active_identity,
    validate_gateway_binding,
)
from utils.sensor_payload import (
    MAX_SENSOR_FRAME_BYTES,
    normalize_boot_id,
    parse_sensor_frame,
    parse_sensor_payload,
)

GAS_TOPIC = "home/sensors/gas/ppm"
BOOT_ID = "0123456789abcdef0123456789abcdef"


def _frame(**overrides) -> bytes:
    doc = {
        "protocol_version": 6,
        "device_id": "gas_sensor_01",
        "boot_id": BOOT_ID,
        "sequence": 1,
        "device_timestamp_ns": 1_700_000_000_000_000_000,
        "payload": "351.25",
    }
    doc.update(overrides)
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def _identity(device_id: str, generation: int, marker: bytes, public_key=None) -> ActiveDeviceIdentity:
    return ActiveDeviceIdentity(
        device_id=device_id,
        generation=generation,
        common_name=f"{device_id}-g{generation}-cafe1234",
        certificate_fingerprint=hashlib.sha256(marker).digest(),
        serial_number_hex=f"{generation + 100:x}",
        certificate_public_key=public_key,
    )


def test_v6_frame_parses_canonical_scalar_and_normalizes_boot_id():
    result = parse_sensor_frame(GAS_TOPIC, _frame(boot_id="01234567-89ab-cdef-0123-456789abcdef"),
                                expected_device_id="gas_sensor_01")
    assert result.valid is True
    assert result.reason == "OK"
    assert result.frame is not None
    assert result.frame.protocol_version == 6
    assert result.frame.boot_id == BOOT_ID
    assert result.frame.sequence == 1
    assert result.frame.device_timestamp_ns == 1_700_000_000_000_000_000
    assert result.frame.canonical_text == "351.25"
    assert result.frame.numeric_value == pytest.approx(351.25)


def test_v5_scalar_parser_remains_unchanged():
    good = parse_sensor_payload(GAS_TOPIC, b"+00351.2500")
    bad = parse_sensor_payload(GAS_TOPIC, b"NaN")
    assert good.valid and good.canonical_text == "351.25"
    assert not bad.valid and bad.reason == "NAN_OR_INF_PAYLOAD"


@pytest.mark.parametrize("boot_id", [
    "", "not-a-boot-id", "0123456789abcdef", "g" * 32,
    "01234567-89ab-cdef-0123-456789abcde",
])
def test_v6_frame_rejects_invalid_boot_ids(boot_id):
    result = parse_sensor_frame(GAS_TOPIC, _frame(boot_id=boot_id))
    assert not result.valid
    assert "boot_id" in result.reason


@pytest.mark.parametrize("sequence", [0, -1, True, False, 1.0, "1", 2**63])
def test_v6_frame_rejects_invalid_sequence_types_and_ranges(sequence):
    result = parse_sensor_frame(GAS_TOPIC, _frame(sequence=sequence))
    assert not result.valid
    assert "sequence" in result.reason


@pytest.mark.parametrize("timestamp", [0, -1, True, 1.0, "1700000000000000000", 2**63])
def test_v6_frame_rejects_invalid_device_timestamps(timestamp):
    result = parse_sensor_frame(GAS_TOPIC, _frame(device_timestamp_ns=timestamp))
    assert not result.valid
    assert "device_timestamp_ns" in result.reason


@pytest.mark.parametrize("payload", ["NaN", "Inf", "-Infinity", float("nan"), float("inf"), True, None, {}, []])
def test_v6_frame_rejects_nonfinite_and_non_scalar_payloads(payload):
    result = parse_sensor_frame(GAS_TOPIC, _frame(payload=payload))
    assert not result.valid


def test_v6_frame_requires_integer_protocol_version():
    assert not parse_sensor_frame(GAS_TOPIC, _frame(protocol_version=6.0)).valid
    assert not parse_sensor_frame(GAS_TOPIC, _frame(protocol_version=True)).valid


def test_v6_frame_rejects_wrong_device_unexpected_keys_duplicate_keys_and_oversize():
    assert not parse_sensor_frame(GAS_TOPIC, _frame(device_id="gas_sensor_02"),
                                  expected_device_id="gas_sensor_01").valid
    assert not parse_sensor_frame(GAS_TOPIC, _frame(extra="not-allowed")).valid
    duplicate = (b'{"protocol_version":6,"device_id":"gas_sensor_01",'
                 b'"device_id":"attacker","boot_id":"' + BOOT_ID.encode() +
                 b'","sequence":1,"device_timestamp_ns":1,"payload":"351"}')
    assert not parse_sensor_frame(GAS_TOPIC, duplicate).valid
    assert not parse_sensor_frame(GAS_TOPIC, b" " * (MAX_SENSOR_FRAME_BYTES + 1)).valid


def test_v6_frame_rejects_physical_out_of_range_values():
    result = parse_sensor_frame(GAS_TOPIC, _frame(payload="99999"))
    assert not result.valid
    assert "OUT_OF_PHYSICAL_BOUNDS" in result.reason


def test_v6_envelope_binds_freshness_fields_to_active_gateway_certificate():
    sensor = _identity("gas_sensor_01", 4, b"sensor-cert")
    gateway_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    gateway = _identity("gateway", 7, b"gateway-cert", gateway_private_key.public_key())
    env = IdentityEnvelope.create(
        device_id=sensor.device_id,
        generation=sensor.generation,
        certificate_common_name=sensor.common_name,
        certificate_fingerprint=sensor.certificate_fingerprint,
        original_topic=GAS_TOPIC,
        payload=b"351.25",
        gateway_instance_id="gateway-run-6",
        protocol_version=6,
        boot_id=BOOT_ID,
        sequence=42,
        device_timestamp_ns=1_700_000_000_000_000_000,
        gateway_identity=gateway,
        gateway_signing_key=gateway_private_key,
    )
    decoded, payload = IdentityEnvelope.decode(env.encode())
    validate_active_identity(decoded, sensor)
    validate_gateway_binding(decoded, gateway)
    assert payload == b"351.25"
    assert decoded.protocol_version == 6
    assert decoded.sequence == 42

    stale_gateway = _identity("gateway", 6, b"old-gateway-cert", gateway_private_key.public_key())
    with pytest.raises(ValueError, match="Gateway certificate"):
        validate_gateway_binding(decoded, stale_gateway)


def test_v6_envelope_signature_covers_sequence_boot_timestamp_payload_and_device_generation():
    sensor = _identity("gas_sensor_01", 4, b"sensor-cert")
    gateway_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    gateway = _identity("gateway", 7, b"gateway-cert", gateway_private_key.public_key())
    env = IdentityEnvelope.create(
        device_id=sensor.device_id, generation=sensor.generation,
        certificate_common_name=sensor.common_name,
        certificate_fingerprint=sensor.certificate_fingerprint,
        original_topic=GAS_TOPIC, payload=b"351", gateway_instance_id="gateway-run-6",
        protocol_version=6, boot_id=BOOT_ID, sequence=42,
        device_timestamp_ns=1_700_000_000_000_000_000,
        gateway_identity=gateway, gateway_signing_key=gateway_private_key,
    )
    doc = json.loads(env.encode())
    doc["sequence"] = 43
    tampered = IdentityEnvelope.decode(json.dumps(doc, separators=(",", ":")).encode())[0]
    with pytest.raises(ValueError, match="signature verification"):
        validate_gateway_binding(tampered, gateway)

    doc = json.loads(env.encode())
    doc["boot_id"] = "f" * 32
    tampered_boot = IdentityEnvelope.decode(json.dumps(doc, separators=(",", ":")).encode())[0]
    with pytest.raises(ValueError, match="signature verification"):
        validate_gateway_binding(tampered_boot, gateway)


def test_v6_envelope_requires_all_freshness_and_gateway_binding_fields():
    sensor = _identity("gas_sensor_01", 1, b"sensor")
    with pytest.raises(ValueError, match="requires boot_id"):
        IdentityEnvelope.create(
            device_id=sensor.device_id, generation=1,
            certificate_common_name=sensor.common_name,
            certificate_fingerprint=sensor.certificate_fingerprint,
            original_topic=GAS_TOPIC, payload=b"350", gateway_instance_id="gateway-run",
            protocol_version=6,
        )


def test_v5_envelope_default_is_still_v5_and_wire_compatible():
    sensor = _identity("gas_sensor_01", 1, b"sensor")
    env = IdentityEnvelope.create(
        device_id=sensor.device_id, generation=1,
        certificate_common_name=sensor.common_name,
        certificate_fingerprint=sensor.certificate_fingerprint,
        original_topic=GAS_TOPIC, payload=b"350", gateway_instance_id="gateway-run",
    )
    decoded, payload = IdentityEnvelope.decode(env.encode())
    assert decoded.protocol_version == 5
    assert decoded.boot_id is None and decoded.sequence is None
    assert payload == b"350"


def test_normalize_boot_id_requires_exact_128_bit_identifier():
    assert normalize_boot_id(BOOT_ID) == BOOT_ID
    with pytest.raises(ValueError):
        normalize_boot_id("deadbeef")


def test_v6_envelope_uses_active_gateway_leaf_private_key_and_certificate_generation(tmp_path):
    from utils.cert_lifecycle import CertAuthority, default_identities
    from utils.device_identity import load_active_device_identity, load_active_device_signing_key

    authority = CertAuthority.create_session(tmp_path / "pki", identities=default_identities())
    try:
        gateway_identity, gateway_private_key = load_active_device_signing_key(
            authority.pointer_path, "gateway"
        )
        sensor_identity = load_active_device_identity(authority.pointer_path, "gas_sensor_01")
        env = IdentityEnvelope.create(
            device_id=sensor_identity.device_id,
            generation=sensor_identity.generation,
            certificate_common_name=sensor_identity.common_name,
            certificate_fingerprint=sensor_identity.certificate_fingerprint,
            original_topic=GAS_TOPIC,
            payload=b"351",
            gateway_instance_id="gateway-run-v6",
            protocol_version=6,
            boot_id=BOOT_ID,
            sequence=10,
            device_timestamp_ns=1_700_000_000_000_000_010,
            gateway_identity=gateway_identity,
            gateway_signing_key=gateway_private_key,
        )
        decoded, payload = IdentityEnvelope.decode(env.encode())
        validate_active_identity(decoded, sensor_identity)
        validate_gateway_binding(decoded, gateway_identity)
        assert decoded.gateway_certificate_generation == gateway_identity.generation
        assert decoded.gateway_certificate_fingerprint_sha256 == gateway_identity.certificate_fingerprint.hex()
        assert payload == b"351"
    finally:
        authority.cleanup()


def test_v6_ids_envelope_validator_checks_active_gateway_certificate_signature(monkeypatch):
    """Exercise the actual IDS envelope validation method, with mocked Zenoh/ONNX imports."""
    import sys
    from types import ModuleType, SimpleNamespace

    fake_ort = ModuleType("onnxruntime")
    fake_ort.InferenceSession = object
    fake_zenoh = ModuleType("zenoh")
    fake_zenoh.__path__ = []
    fake_zenoh.Session = object
    fake_zenoh.SampleKind = SimpleNamespace(DELETE=object())
    fake_zenoh.Priority = SimpleNamespace(REAL_TIME=1)
    fake_handlers = ModuleType("zenoh.handlers")
    fake_handlers.FifoChannel = object
    fake_handlers.RingChannel = object
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "zenoh", fake_zenoh)
    monkeypatch.setitem(sys.modules, "zenoh.handlers", fake_handlers)

    from agent import ids_agent

    sensor_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    gateway_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sensor = _identity("gas_sensor_01", 3, b"sensor-cert", sensor_private.public_key())
    gateway = _identity("gateway", 5, b"gateway-cert", gateway_private.public_key())
    identities = {"gas_sensor_01": sensor, "gateway": gateway}
    monkeypatch.setattr(ids_agent, "load_active_device_identity", lambda _pointer, name: identities[name])

    agent = ids_agent.ZenohIDSAgent.__new__(ids_agent.ZenohIDSAgent)
    agent.cert_pointer = "active-pki-pointer.json"
    agent._device_to_topic = {"gas_sensor_01": GAS_TOPIC}
    env = IdentityEnvelope.create(
        device_id=sensor.device_id,
        generation=sensor.generation,
        certificate_common_name=sensor.common_name,
        certificate_fingerprint=sensor.certificate_fingerprint,
        original_topic=GAS_TOPIC,
        payload=b"352",
        gateway_instance_id="gateway-run-6",
        protocol_version=6,
        boot_id=BOOT_ID,
        sequence=77,
        device_timestamp_ns=1_700_000_000_000_000_077,
        gateway_identity=gateway,
        gateway_signing_key=gateway_private,
    )
    key = f"internal/ids/telemetry/{sensor.device_id}"
    decoded, payload = agent._validate_identity_envelope(key, env.encode())
    assert decoded.sequence == 77
    assert payload == b"352"

    tampered = json.loads(env.encode())
    tampered["device_timestamp_ns"] += 1
    with pytest.raises(ValueError, match="signature verification"):
        agent._validate_identity_envelope(key, json.dumps(tampered).encode())
