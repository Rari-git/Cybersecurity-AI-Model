import json
from utils.gateway_metrics import read_latest_snapshot, calculate_delta, snapshot_signature


def test_latest_snapshot_ignores_truncated_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"run_id": "a", "ingress_received": 1}) + "\n" +
                    json.dumps({"run_id": "a", "ingress_received": 2}) + "\n" + '{"broken":\n',
                    encoding="utf-8")
    assert read_latest_snapshot(path)["ingress_received"] == 2


def test_delta_rejects_changed_gateway_run_id():
    assert calculate_delta({"run_id": "a"}, {"run_id": "b"}) is None


def test_delta_includes_queue_and_drop_counters():
    before = {"run_id": "a", "ingress_received": 10, "sensor_queued": 10,
              "sensor_forwarded": 10, "rate_topic_dropped": 0,
              "sensor_queue_depth": 0, "actuator_queue_depth": 0,
              "mitigation_queue_depth": 0}
    after = {"run_id": "a", "ingress_received": 110, "sensor_queued": 30,
             "sensor_forwarded": 25, "rate_topic_dropped": 70,
             "blocklist_dropped": 5, "blocks_issued": 1,
             "sensor_queue_depth": 0, "actuator_queue_depth": 0,
             "mitigation_queue_depth": 0, "blocks_active": 1}
    delta = calculate_delta(before, after)
    assert delta["ingress_received"] == 100
    assert delta["sensor_queued"] == 20
    assert delta["sensor_forwarded"] == 15
    assert delta["dropped_total"] == 75
    assert delta["blocks_issued"] == 1 and delta["blocks_active"] == 1


def test_device_block_snapshot_is_part_of_settled_signature():
    a = {"run_id": "a", "device_blocks": {"device:gas_sensor_01": {"remaining_sec": 40}}}
    b = {"run_id": "a", "device_blocks": {"device:gas_sensor_02": {"remaining_sec": 40}}}
    assert snapshot_signature(a) != snapshot_signature(b)


def test_delta_includes_device_drop_and_block_metrics():
    before = {"run_id": "a", "device_rate_dropped": 0, "device_blocks_issued": 0,
              "device_blocks_active": 0, "device_blocks": {}}
    after = {"run_id": "a", "device_rate_dropped": 10, "device_blocks_issued": 1,
             "device_blocks_active": 1,
             "device_blocks": {"device:gas_sensor_01": {"attack": "Device Rate Abuse"}}}
    delta = calculate_delta(before, after)
    assert delta["device_rate_dropped"] == 10
    assert delta["device_blocks_issued"] == 1
    assert delta["device_blocks_active"] == 1
    assert "device:gas_sensor_01" in delta["device_blocks"]


def test_device_forwarding_delta_is_calculated_per_identity():
    before = {"run_id": "a", "device_publish_accepted": {"gas_sensor_01": 5, "gas_sensor_02": 7}}
    after = {"run_id": "a", "device_publish_accepted": {"gas_sensor_01": 5, "gas_sensor_02": 12}}
    delta = calculate_delta(before, after)
    assert delta["device_publish_accepted"] == {"gas_sensor_01": 0, "gas_sensor_02": 5}


def test_snapshot_signature_changes_when_counters_or_queues_change():
    snapshot = {"run_id": "a", "ingress_received": 1, "sensor_queue_depth": 0}
    assert snapshot_signature(snapshot) != snapshot_signature({**snapshot, "ingress_received": 2})
    assert snapshot_signature(snapshot) != snapshot_signature({**snapshot, "sensor_queue_depth": 1})
