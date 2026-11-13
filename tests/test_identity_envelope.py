import hashlib
import pytest

from utils.identity_envelope import AuthenticatedTelemetryEnvelope


def test_dual_sensor_identity_is_preserved_on_shared_topic():
    shared_topic = "home/sensors/gas/ppm"
    fp1 = hashlib.sha256(b"real-cert-1").digest()
    fp2 = hashlib.sha256(b"real-cert-2").digest()
    env1 = AuthenticatedTelemetryEnvelope.create(
        device_id="gas_sensor_01", generation=1, certificate_common_name="gas_sensor_01",
        certificate_fingerprint=fp1, original_topic=shared_topic, payload=b"350", gateway_instance_id="gateway-run-1")
    env2 = AuthenticatedTelemetryEnvelope.create(
        device_id="gas_sensor_02", generation=1, certificate_common_name="gas_sensor_02",
        certificate_fingerprint=fp2, original_topic=shared_topic, payload=b"340", gateway_instance_id="gateway-run-1")
    read1, payload1 = AuthenticatedTelemetryEnvelope.decode(env1.encode())
    read2, payload2 = AuthenticatedTelemetryEnvelope.decode(env2.encode())
    assert read1.device_id != read2.device_id
    assert read1.original_topic == read2.original_topic == shared_topic
    assert read1.certificate_fingerprint_sha256 != read2.certificate_fingerprint_sha256
    assert (payload1, payload2) == (b"350", b"340")


def test_envelope_rejects_fake_or_missing_fingerprint():
    with pytest.raises(ValueError, match="real SHA-256"):
        AuthenticatedTelemetryEnvelope.create(
            device_id="gas_sensor_01", generation=1, certificate_common_name="gas_sensor_01",
            certificate_fingerprint=b"fake", original_topic="home/sensors/gas/ppm",
            payload=b"350", gateway_instance_id="gateway-run-1")


def test_envelope_rejects_untrusted_identity_source():
    env = AuthenticatedTelemetryEnvelope.create(
        device_id="gas_sensor_01", generation=1, certificate_common_name="gas_sensor_01",
        certificate_fingerprint=hashlib.sha256(b"cert").digest(), original_topic="home/sensors/gas/ppm",
        payload=b"350", gateway_instance_id="gateway-run-1")
    raw = env.encode().replace(b"gateway_mtls_acl", b"payload_claim____")
    with pytest.raises(ValueError, match="untrusted"):
        AuthenticatedTelemetryEnvelope.decode(raw)
