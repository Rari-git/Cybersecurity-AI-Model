import json

from utils.gateway_metrics import read_latest_snapshot, calculate_delta


def test_read_latest_snapshot_ignores_truncated_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"run_id": "a", "ingress_received": 1}) + "\n" +
                    json.dumps({"run_id": "a", "ingress_received": 2}) + "\n" +
                    '{"broken":\n', encoding="utf-8")
    assert read_latest_snapshot(path)["ingress_received"] == 2


def test_delta_rejects_different_gateway_processes():
    assert calculate_delta({"run_id": "a"}, {"run_id": "b"}) is None


def test_device_publish_accepted_delta_is_per_device():
    before = {"run_id": "a", "device_publish_accepted": {"gas_sensor_01": 10, "gas_sensor_02": 20}}
    after = {"run_id": "a", "device_publish_accepted": {"gas_sensor_01": 10, "gas_sensor_02": 25}}
    delta = calculate_delta(before, after)
    assert delta["device_publish_accepted"] == {"gas_sensor_01": 0, "gas_sensor_02": 5}


def test_delta_includes_device_drop_counters():
    before = {"run_id": "a", "device_rate_dropped": 0, "device_blocks_issued": 0}
    after = {"run_id": "a", "device_rate_dropped": 10, "device_blocks_issued": 1,
             "device_blocks_active": 1, "device_blocks": {"device:gas_sensor_01": {"remaining_sec": 20}}}
    delta = calculate_delta(before, after)
    assert delta["device_rate_dropped"] == 10
    assert delta["device_blocks_issued"] == 1
    assert delta["device_blocks_active"] == 1
    assert "device:gas_sensor_01" in delta["device_blocks"]
