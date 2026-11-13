from __future__ import annotations

import hashlib
import queue
from types import SimpleNamespace

from scripts.live_runtime_smoke import _wait_for_smoke_envelopes
from utils.identity_envelope import IdentityEnvelope


def test_v5_smoke_helper_ignores_background_sensor_traffic_and_returns_expected():
    good_fp = hashlib.sha256(b"gas-sensor-cert").digest()
    active = {
        "gas_sensor_01": SimpleNamespace(
            device_id="gas_sensor_01", generation=1, common_name="gas_sensor_01",
            certificate_fingerprint=good_fp,
        ),
    }
    allowed = {"gas_sensor_01": "home/sensors/gas/ppm"}
    q: queue.Queue = queue.Queue()
    q.put(("internal/ids/telemetry/gas_sensor_02", b"not-an-envelope"))
    q.put(("internal/ids/telemetry/gas_sensor_01", b"malformed"))
    envelope = IdentityEnvelope.create(
        device_id="gas_sensor_01", generation=1,
        certificate_common_name="gas_sensor_01", certificate_fingerprint=good_fp,
        original_topic="home/sensors/gas/ppm", payload=b"351",
        gateway_instance_id="gateway-run-1",
    )
    q.put(("internal/ids/telemetry/gas_sensor_01", envelope.encode()))

    envelopes, payloads = _wait_for_smoke_envelopes(
        q, active, allowed, {"gas_sensor_01"}, timeout=0.5,
    )
    assert set(envelopes) == {"gas_sensor_01"}
    assert payloads == {"gas_sensor_01": b"351"}
    assert envelopes["gas_sensor_01"].message_id == envelope.message_id
