from __future__ import annotations

import pytest
import threading
from types import SimpleNamespace

from agent.runtime_transitions import RuntimeTransitionObserver
from agent.policy_engine import DetectionEvidence, RiskPolicyEngine
from storage.telemetry_db import DBPaths, LumenStorageManager


def _observe(observer, *, generation=1, payload_len=8, delta_t=0.5, topic="sensors/temp"):
    return observer.observe(
        device_id="sensor-1",
        generation=generation,
        topic=topic,
        payload_len=payload_len,
        delta_t_seconds=delta_t,
    )


def test_observer_emits_deterministic_state_and_transition_ids():
    first = RuntimeTransitionObserver()
    second = RuntimeTransitionObserver()

    first_a = _observe(first)
    second_a = _observe(second)
    first_b = _observe(first, payload_len=20)
    second_b = _observe(second, payload_len=20)

    assert first_a == second_a
    assert first_b == second_b
    assert first_a.previous_state_id is None
    assert first_b.previous_state_id == first_a.state_id
    assert first_b.state_changed is True
    assert first_b.changed_dimensions == ("payload_size_bucket",)
    assert first_b.explanation.startswith("transition=state_change:")


def test_interval_category_is_transition_context_not_a_behavioral_state_dimension():
    observer = RuntimeTransitionObserver()
    _observe(observer)
    changed_interval = _observe(observer, delta_t=2.0)
    repeated = _observe(observer, delta_t=0.5)

    assert changed_interval.changed_dimensions == ()
    assert changed_interval.state_changed is False
    assert changed_interval.interval_class == "delayed"
    assert repeated.state_changed is False
    assert repeated.transition_id is not None
    assert repeated.interval_class == "normal"
    assert repeated.explanation.startswith("transition=self:")
    assert changed_interval.transition_id != repeated.transition_id


def test_generation_and_topic_are_isolated_streams():
    observer = RuntimeTransitionObserver(max_streams=4)
    _observe(observer)
    assert _observe(observer, generation=2).previous_state_id is None
    assert _observe(observer, topic="sensors/humidity").previous_state_id is None


def test_observer_bounds_stream_memory():
    observer = RuntimeTransitionObserver(max_streams=2)
    _observe(observer, topic="sensors/a")
    _observe(observer, topic="sensors/b")
    _observe(observer, topic="sensors/c")

    assert len(observer._last_descriptors) == 2
    assert _observe(observer, topic="sensors/a").previous_state_id is None


def test_transition_context_does_not_change_policy_outcome(tmp_path):
    outcomes = []
    for index, transition_summary in enumerate(("", "state_change:payload_size_bucket")):
        storage = LumenStorageManager(
            paths=DBPaths.from_dir(tmp_path / f"policy-{index}"), enable_writers=True
        )
        try:
            engine = RiskPolicyEngine(storage=storage, pep_apply_fn=lambda _device, _enabled: True)
            evidence = DetectionEvidence(
                device_id="sensor-1",
                generation=1,
                topic="sensors/temp",
                lstm_score=2.0,
                lstm_threshold=1.0,
                lstm_anomalous=True,
                cusum_score=0.0,
                cusum_triggered=False,
                operational_status="HEALTHY",
                fault_type="NONE",
                behavioral_transition_summary=transition_summary,
            )
            decision = engine.evaluate_evidence(evidence)
            outcomes.append((decision.action, decision.reason_code, decision.security_status))
        finally:
            storage.close()

    assert outcomes[0] == outcomes[1]


def test_ids_sensor_path_attaches_transition_context_to_existing_evidence():
    from agent.ids_agent import ZenohIDSAgent

    agent = ZenohIDSAgent.__new__(ZenohIDSAgent)
    agent._resolve_device_id = lambda _topic, device_id: device_id
    agent._feature_lock = threading.Lock()
    agent.extractor = SimpleNamespace(
        update=lambda **_kwargs: SimpleNamespace(raw_value=1.0),
        get_window=lambda *_args: None,
    )
    agent.cusum = SimpleNamespace(
        update=lambda *_args: SimpleNamespace(score=0.25, triggered=False)
    )
    agent.fault_classifier = SimpleNamespace(
        evaluate=lambda **_kwargs: SimpleNamespace(
            status="HEALTHY", fault_type="NONE", detail="Nominal"
        )
    )
    agent.transition_observer = RuntimeTransitionObserver()
    agent.ml_enabled = False
    agent.stream_thresholds = {}
    agent.global_threshold = 1.0
    agent.stats = {"faults_isolated": 0, "nonfinite": 0, "quarantines_issued": 0}
    captured = []
    agent.policy_engine = SimpleNamespace(
        evaluate_evidence=lambda evidence: (
            captured.append(evidence) or SimpleNamespace(
                operational_status="HEALTHY", action="NONE", reason_code="NO_ACTION"
            )
        )
    )

    agent._process_sensor(
        "home/sensors/gas/ppm", "1", 0.5, 1.0, device_id="sensor-1", generation=1
    )
    agent._process_sensor(
        "home/sensors/gas/ppm", "1", 2.0, 1.0, device_id="sensor-1", generation=1
    )

    assert len(captured) == 2
    assert captured[1].lstm_anomalous is False
    assert captured[1].cusum_score == 0.25
    assert captured[1].operational_status == "HEALTHY"
    assert "transition=self:" in captured[1].behavioral_transition_summary
    assert "interval=delayed" in captured[1].xai_summary


@pytest.mark.parametrize("delta_t", [-1.0, float("nan"), float("inf")])
def test_invalid_timing_is_rejected_without_observation(delta_t):
    observer = RuntimeTransitionObserver()
    with pytest.raises(ValueError, match="delta_t_seconds"):
        _observe(observer, delta_t=delta_t)
    assert len(observer._last_descriptors) == 0
