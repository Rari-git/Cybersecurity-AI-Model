import threading
import time

from agent.pep_protocol import PepAckRegistry, PepAcknowledgement


def test_pep_ack_requires_matching_decision_device_and_action():
    registry = PepAckRegistry()
    registry.register("decision-1", "gas_sensor_01", "SHADOW_QUARANTINE")
    wrong = PepAcknowledgement("decision-1", "gas_sensor_02", "SHADOW_QUARANTINE", "SHADOW_QUARANTINE", "CONFIRMED", time.time_ns(), "gateway-a")
    assert registry.accept(wrong) is False
    wrong_action = PepAcknowledgement("decision-1", "gas_sensor_01", "REVOKED", "REVOKED", "CONFIRMED", time.time_ns(), "gateway-a")
    assert registry.accept(wrong_action) is False
    good = PepAcknowledgement("decision-1", "gas_sensor_01", "SHADOW_QUARANTINE", "SHADOW_QUARANTINE", "CONFIRMED", time.time_ns(), "gateway-a")
    result = []
    waiter = threading.Thread(target=lambda: result.append(registry.wait("decision-1", 1.0)))
    waiter.start()
    assert registry.accept(good) is True
    waiter.join()
    assert result[0].confirmed is True
    assert registry.accept(good) is False


def test_pep_timeout_is_not_reported_as_success():
    registry = PepAckRegistry()
    registry.register("decision-timeout", "door_sensor_01", "ACTIVE")
    assert registry.wait("decision-timeout", 0.001) is None
    assert registry.pending_count() == 0
