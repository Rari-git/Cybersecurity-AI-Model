from __future__ import annotations

import hashlib
import json

import pytest

from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.telemetry_db import DBPaths, LumenStorageManager
from utils.sensor_payload import normalize_boot_id


GAS_TOPIC = "home/sensors/gas/ppm"
BOOT_A = "0123456789abcdef0123456789abcdef"
BOOT_B = "fedcba9876543210fedcba9876543210"


def _ctx(session_id: str = "v6-live") -> ProvenanceContext:
    return ProvenanceContext(
        session_id=session_id,
        device_cn="gas_sensor_01-g3-cafe1234",
        cert_fingerprint=hashlib.sha256(b"gas-sensor-cert-g3").digest(),
        source_type="live_validated",
        device_id="gas_sensor_01",
        generation=3,
    )


def _frame(*, boot_id: str = BOOT_A, sequence: int = 1, timestamp: int | None = None) -> bytes:
    doc = {
        "protocol_version": 6,
        "device_id": "gas_sensor_01",
        "boot_id": boot_id,
        "sequence": sequence,
        "device_timestamp_ns": timestamp or (1_700_000_000_000_000_000 + sequence),
        "payload": "351.25",
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


@pytest.fixture()
def gate_storage(tmp_path):
    storage = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    storage.register_baseline_session(
        "v6-live", "live_validated", {"protocol_version": 6}, approved_for_training=False,
    )
    gate = AdmissionGate(storage, {"gas_sensor_01": {GAS_TOPIC}})
    try:
        yield storage, gate
    finally:
        storage.close()


def test_v6_admission_rejects_duplicate_and_out_of_order_sequences_before_normal_insert(gate_storage):
    storage, gate = gate_storage
    ctx = _ctx()
    gate.register_verified_device_boot(ctx, BOOT_A)

    accepted_1 = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(sequence=1))
    duplicate_1 = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(sequence=1))
    accepted_3 = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(sequence=3))
    out_of_order_2 = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(sequence=2))

    assert accepted_1.admitted and accepted_1.sequence == 1
    assert not duplicate_1.admitted
    assert duplicate_1.reason == "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
    assert accepted_3.admitted and accepted_3.sequence == 3
    assert not out_of_order_2.admitted
    assert out_of_order_2.reason == "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"

    normal = storage.connect_readonly("normal")
    incidents = storage.connect_readonly("incidents")
    try:
        assert normal.execute("SELECT COUNT(*) FROM raw_events WHERE protocol_version=6").fetchone()[0] == 2
        reasons = [r[0] for r in incidents.execute(
            "SELECT reason_code FROM incident_captures ORDER BY capture_id"
        ).fetchall()]
        assert reasons == ["REPLAY_OR_OUT_OF_ORDER_SEQUENCE", "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"]
    finally:
        normal.close()
        incidents.close()


def test_v5_normal_event_integrity_metadata_binds_canonical_payload(gate_storage):
    storage, gate = gate_storage
    raw_payload = b"00351.250"

    result = gate.evaluate_and_ingest(_ctx(), GAS_TOPIC, raw_payload)

    assert result.admitted is True
    assert result.canonical_text == "351.25"
    normal = storage.connect_readonly("normal")
    try:
        row = normal.execute(
            "SELECT payload_len,payload_sha256,raw_payload FROM raw_events WHERE session_id=?",
            ("v6-live",),
        ).fetchone()
    finally:
        normal.close()
    canonical = result.canonical_text.encode("utf-8")
    assert row["raw_payload"] == result.canonical_text
    assert row["payload_len"] == len(canonical)
    assert row["payload_sha256"] == hashlib.sha256(canonical).digest()


def test_v6_admission_rejects_stale_and_invalid_boot_ids_as_incidents(gate_storage):
    storage, gate = gate_storage
    ctx = _ctx()
    gate.register_verified_device_boot(ctx, BOOT_A)

    assert gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(sequence=1)).admitted
    gate.register_verified_device_boot(ctx, BOOT_B)

    stale_boot = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(boot_id=BOOT_A, sequence=2))
    invalid_boot = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(boot_id="not-a-boot-id", sequence=1))
    valid_new_boot = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, _frame(boot_id=BOOT_B, sequence=1))

    assert not stale_boot.admitted and stale_boot.reason == "STALE_BOOT_ID"
    assert not invalid_boot.admitted and "INVALID_V6_FRAME" in invalid_boot.reason
    assert valid_new_boot.admitted

    normal = storage.connect_readonly("normal")
    incidents = storage.connect_readonly("incidents")
    try:
        assert normal.execute("SELECT COUNT(*) FROM raw_events WHERE protocol_version=6").fetchone()[0] == 2
        reasons = [r[0] for r in incidents.execute(
            "SELECT reason_code FROM incident_captures ORDER BY capture_id"
        ).fetchall()]
        assert reasons[0] == "STALE_BOOT_ID"
        assert reasons[1].startswith("INVALID_V6_FRAME:")
    finally:
        normal.close()
        incidents.close()


def test_v6_gate_requires_explicit_boot_registration_not_frame_self_assertion(gate_storage):
    _storage, gate = gate_storage
    result = gate.evaluate_v6_and_ingest(_ctx(), GAS_TOPIC, _frame(sequence=1))
    assert not result.admitted
    assert result.reason == "BOOT_NOT_REGISTERED"


def test_v6_gate_boot_registration_normalizes_uuid_format(gate_storage):
    _storage, gate = gate_storage
    ctx = _ctx()
    gate.register_verified_device_boot(ctx, "01234567-89ab-cdef-0123-456789abcdef")
    assert gate.storage.load_device_freshness("gas_sensor_01")["gas_sensor_01"]["active_boot_id"] == normalize_boot_id(
        "01234567-89ab-cdef-0123-456789abcdef"
    )


def test_malformed_v6_frame_can_be_forwarded_as_gateway_signed_rejection_evidence():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from utils.device_identity import ActiveDeviceIdentity
    from utils.identity_envelope import IdentityEnvelope, validate_gateway_binding

    sensor = ActiveDeviceIdentity(
        device_id="gas_sensor_01", generation=3,
        common_name="gas_sensor_01-g3-cafe1234",
        certificate_fingerprint=hashlib.sha256(b"sensor-g3").digest(),
        serial_number_hex="103",
    )
    gateway_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    gateway = ActiveDeviceIdentity(
        device_id="gateway", generation=2, common_name="gateway",
        certificate_fingerprint=hashlib.sha256(b"gateway-g2").digest(),
        serial_number_hex="102", certificate_public_key=gateway_key.public_key(),
    )
    rejection = IdentityEnvelope.create(
        device_id=sensor.device_id,
        generation=sensor.generation,
        certificate_common_name=sensor.common_name,
        certificate_fingerprint=sensor.certificate_fingerprint,
        original_topic=GAS_TOPIC,
        payload=b"V6_PROTOCOL_FRAME_REJECTED",
        gateway_instance_id="gateway-run-2",
        protocol_version=6,
        admission_status="REJECTED",
        admission_reason="INVALID_V6_FRAME:boot_id must be a 128-bit hexadecimal identifier",
        gateway_identity=gateway,
        gateway_signing_key=gateway_key,
    )

    decoded, payload = IdentityEnvelope.decode(rejection.encode())
    validate_gateway_binding(decoded, gateway)
    assert decoded.protocol_version == 6
    assert decoded.admission_status == "REJECTED"
    assert decoded.boot_id is None and decoded.sequence is None
    assert decoded.admission_reason.startswith("INVALID_V6_FRAME")
    assert payload == b"V6_PROTOCOL_FRAME_REJECTED"
