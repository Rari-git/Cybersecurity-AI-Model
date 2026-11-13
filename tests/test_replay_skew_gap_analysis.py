"""Regression analysis for the V5 held-out replay-skew blind spot.

This is diagnostic coverage only: it records why V5 classifies the current
replay generator as an operational freeze and why the benchmark does not model
AdmissionGate/ACL rejection for its cross-topic frames. It does not alter the
frozen V5 scenario or implement a V6 detector.
"""
from __future__ import annotations

from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
from agent.statistical_detectors import OperationalFaultClassifier
from attacks.advanced_generators import iter_scenario_frames


class _MemoryPolicyStorage:
    def __init__(self):
        self.transitions = []

    def load_policy_states(self):
        return {}

    def load_device_state_rows(self):
        return {}

    def append_quarantine_transition(self, *args, **kwargs):
        self.transitions.append((args, kwargs))

    def save_policy_state(self, **kwargs):
        pass

    def record_enforcement_decision(self, *args, **kwargs):
        pass

    def confirm_enforcement_pep(self, *args, **kwargs):
        pass


def test_atk2_heldout_replay_is_classified_as_frozen_fault_before_hybrid_action():
    frames = list(iter_scenario_frames("atk2_replay_skew_heldout", steps=40))
    target_topic = "home/sensors/vibration/shock"
    allowed_topic = target_topic
    target_frames = [frame for frame in frames if frame.topic == target_topic]
    cross_topic_frames = [frame for frame in frames if frame.topic != allowed_topic]

    # The held-out target stream repeats one scalar forever, while every fourth
    # frame is injected under gas/ppm despite carrying vibration_sensor_01.
    assert len({frame.payload for frame in target_frames}) == 1
    assert {frame.payload for frame in target_frames} == {b"0.018"}
    assert cross_topic_frames
    assert all(frame.device_id == "vibration_sensor_01" for frame in cross_topic_frames)
    assert {frame.topic for frame in cross_topic_frames} == {"home/sensors/gas/ppm"}

    classifier = OperationalFaultClassifier(frozen_min_steps=8)
    target_statuses = []
    for frame in frames:
        health = classifier.evaluate(
            frame.device_id, frame.topic, 0.018, frame.delta_t_sec,
            operation=frame.operation,
        )
        if frame.topic == target_topic:
            target_statuses.append((frame.step, health.status, health.fault_type))

    first_fault = next((item for item in target_statuses if item[1] == "FAULT"), None)
    assert first_fault is not None
    assert first_fault[2] == "FROZEN_SENSOR"

    # The policy's explicit FAULT branch takes precedence even when its other
    # evidence fields are anomalous, so the held-out replay never advances into
    # the hybrid quarantine path once the first usable ML window exists.
    policy = RiskPolicyEngine(
        storage=_MemoryPolicyStorage(),
        pep_apply_fn=lambda _device_id, _enable: True,
        window_n=5,
        trigger_k=3,
    )
    decision = policy.evaluate_evidence(DetectionEvidence(
        device_id="vibration_sensor_01",
        generation=1,
        topic=target_topic,
        lstm_score=100.0,
        lstm_threshold=1.0,
        lstm_anomalous=True,
        cusum_score=100.0,
        cusum_triggered=True,
        operational_status="FAULT",
        fault_type="FROZEN_SENSOR",
    ))
    assert decision.action == "ALERT_FAULT"
    assert decision.new_state == "FAULT"
