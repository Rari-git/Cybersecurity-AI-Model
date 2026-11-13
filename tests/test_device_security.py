import json

import pytest

from utils.device_registry import DeviceRegistry
from utils.device_security import DeviceSecurityManager


def _small_registry(tmp_path):
    devices = []
    for device_id in ("gas_sensor_01", "gas_sensor_02"):
        devices.append({
            "device_id": device_id,
            "cert_cn": device_id,
            "ingress_key": f"ingest/devices/{device_id}/sensors/gas/ppm",
            "target_topic": "home/sensors/gas/ppm",
            "rate_pps": 0.1,
            "burst": 1.0,
            "strike_threshold": 3,
            "strike_window_sec": 10.0,
            "strike_cooldown_sec": 1.0,
            "block_ttl_sec": 5.0,
        })
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"version": 1, "devices": devices}), encoding="utf-8")
    return DeviceRegistry.load(path)


def test_per_device_rate_limit_quarantines_only_offending_identity(tmp_path):
    manager = DeviceSecurityManager(_small_registry(tmp_path))
    assert manager.check_rate("gas_sensor_01", now=0.0).allowed
    assert manager.check_rate("gas_sensor_01", now=0.1).reason == "device_rate_limited"
    manager.check_rate("gas_sensor_01", now=0.2)  # strike cooldown: not counted
    manager.check_rate("gas_sensor_01", now=1.2)
    decision = manager.check_rate("gas_sensor_01", now=2.2)
    assert decision.block_issued
    assert manager.is_blocked("gas_sensor_01")
    assert not manager.is_blocked("gas_sensor_02")
    assert manager.check_rate("gas_sensor_02", now=2.2).allowed
    assert manager.active_block_count() == 1
    assert "device:gas_sensor_01" in manager.active_blocks()


def test_bad_device_identifier_is_not_implicitly_trusted(tmp_path):
    manager = DeviceSecurityManager(_small_registry(tmp_path))
    with pytest.raises(KeyError):
        manager.check_rate("gas_sensor_typo", now=1.0)


def test_explicit_violation_escalates_to_block(tmp_path):
    manager = DeviceSecurityManager(_small_registry(tmp_path))
    assert not manager.register_violation("gas_sensor_01", "Anomalous Payload", now=1.0).block_issued
    assert not manager.register_violation("gas_sensor_01", "Anomalous Payload", now=2.1).block_issued
    decision = manager.register_violation("gas_sensor_01", "Anomalous Payload", now=3.2)
    assert decision.block_issued
    block = manager.active_blocks()["device:gas_sensor_01"]
    assert block["attack"] == "Anomalous Payload"
    assert block["cert_cn"] == "gas_sensor_01"


def test_clear_device_does_not_clear_peer(tmp_path):
    manager = DeviceSecurityManager(_small_registry(tmp_path))
    manager.register_violation("gas_sensor_01", "Anomalous Payload", now=1.0)
    manager.register_violation("gas_sensor_01", "Anomalous Payload", now=2.1)
    manager.register_violation("gas_sensor_01", "Anomalous Payload", now=3.2)
    assert manager.clear_device("gas_sensor_01")
    assert not manager.is_blocked("gas_sensor_01")
    assert not manager.is_blocked("gas_sensor_02")
