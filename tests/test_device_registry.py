import json

import pytest

from utils.device_registry import DeviceRegistry, DeviceRegistryError, DeviceSpec


def test_config_has_individual_device_identity_and_exact_route():
    registry = DeviceRegistry.load("config/devices.json")
    route = registry.resolve_ingress("ingest/devices/gas_sensor_01/sensors/gas/ppm")
    assert route is not None
    spec, target = route
    assert spec.device_id == "gas_sensor_01"
    assert spec.cert_cn == "gas_sensor_01"
    assert target == "home/sensors/gas/ppm"
    assert registry.resolve_ingress("ingest/devices/gas_sensor_01/sensors/pir/motion") is None
    assert registry.resolve_ingress("ingest/sensors/gas/ppm") is None


def test_two_gas_devices_share_output_but_not_ingress_identity():
    registry = DeviceRegistry.load("config/devices.json")
    one = registry.resolve_ingress("ingest/devices/gas_sensor_01/sensors/gas/ppm")
    two = registry.resolve_ingress("ingest/devices/gas_sensor_02/sensors/gas/ppm")
    assert one and two
    assert one[1] == two[1] == "home/sensors/gas/ppm"
    assert one[0].device_id != two[0].device_id
    assert one[0].ingress_key != two[0].ingress_key


def test_disabled_device_does_not_resolve(tmp_path):
    data = {
        "version": 1,
        "devices": [{
            "device_id": "sensor_a", "cert_cn": "sensor_a",
            "ingress_key": "ingest/devices/sensor_a/sensors/gas/ppm",
            "target_topic": "home/sensors/gas/ppm", "enabled": False,
        }],
    }
    path = tmp_path / "devices.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    registry = DeviceRegistry.load(path)
    assert registry.resolve_ingress(data["devices"][0]["ingress_key"]) is None
    assert registry.enabled_devices() == ()


def test_device_id_must_match_certificate_cn(tmp_path):
    data = {
        "version": 1,
        "devices": [{
            "device_id": "sensor_a", "cert_cn": "sensor_b",
            "ingress_key": "ingest/devices/sensor_a/sensors/gas/ppm",
            "target_topic": "home/sensors/gas/ppm",
        }],
    }
    path = tmp_path / "devices.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DeviceRegistryError, match="cert_cn"):
        DeviceRegistry.load(path)


def test_duplicate_ingress_key_is_rejected():
    first = DeviceSpec(
        "sensor_a", "sensor_a", "ingest/devices/sensor_a/sensors/gas/ppm",
        "home/sensors/gas/ppm",
    )
    second = DeviceSpec(
        "sensor_b", "sensor_b", "ingest/devices/sensor_a/sensors/gas/ppm",
        "home/sensors/gas/ppm",
    )
    with pytest.raises(DeviceRegistryError, match="duplicate ingress_key"):
        DeviceRegistry([first, second])


def test_ingress_suffix_must_match_canonical_target_and_payload_type(tmp_path):
    data = {
        "version": 1,
        "devices": [{
            "device_id": "gas_sensor_01", "cert_cn": "gas_sensor_01",
            "ingress_key": "ingest/devices/gas_sensor_01/sensors/door/state",
            "target_topic": "home/sensors/gas/ppm", "payload_type": "gas",
        }],
    }
    path = tmp_path / "devices.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DeviceRegistryError, match="suffix"):
        DeviceRegistry.load(path)
