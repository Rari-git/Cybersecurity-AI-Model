import pytest
from utils.sensor_publish import publish_sensor_sample


class FakeSession:
    def __init__(self): self.calls = []
    def put(self, topic, payload): self.calls.append((topic, payload))


def test_sensor_helper_uses_direct_session_put_for_string_payload():
    session = FakeSession()
    publish_sensor_sample(session, "ingest/sensors/gas/ppm", "351")
    assert session.calls == [("ingest/sensors/gas/ppm", b"351")]


def test_sensor_helper_accepts_bytes():
    session = FakeSession()
    publish_sensor_sample(session, "home/sensors/door/state", b"OPEN")
    assert session.calls == [("home/sensors/door/state", b"OPEN")]


@pytest.mark.parametrize("topic", ["ingest/admin/config", "home/actuators/door/lock", "ingest/sensors/unknown"])
def test_sensor_helper_rejects_non_sensor_topic(topic):
    with pytest.raises(ValueError): publish_sensor_sample(FakeSession(), topic, "x")


@pytest.mark.parametrize("payload", [b"", "", "x" * 513, 123])
def test_sensor_helper_rejects_empty_oversized_or_non_text_payload(payload):
    with pytest.raises((TypeError, ValueError)):
        publish_sensor_sample(FakeSession(), "ingest/sensors/gas/ppm", payload)
