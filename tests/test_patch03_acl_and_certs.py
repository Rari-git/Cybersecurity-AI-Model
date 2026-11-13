from pathlib import Path
import importlib.util

import pytest


def test_router_acls_use_exact_per_device_publish_rules():
    json5 = pytest.importorskip("json5")
    expected = {
        "door_sensor_01": "ingest/devices/door_sensor_01/sensors/door/state",
        "gas_sensor_01": "ingest/devices/gas_sensor_01/sensors/gas/ppm",
        "gas_sensor_02": "ingest/devices/gas_sensor_02/sensors/gas/ppm",
        "vibration_sensor_01": "ingest/devices/vibration_sensor_01/sensors/vibration/shock",
        "pir_sensor_01": "ingest/devices/pir_sensor_01/sensors/pir/motion",
    }
    for filename in ("config/zenohd_router.json5", "config/zenoh_acl_example.json5"):
        doc = json5.loads(Path(filename).read_text(encoding="utf-8"))
        ac = doc["access_control"]
        assert ac["enabled"] is True
        assert ac["default_permission"] == "deny"
        rules = {rule["id"]: rule for rule in ac["rules"]}
        assert "in_put_ingest_sensors" not in rules
        policies = ac["policies"]
        for device_id, ingress_key in expected.items():
            rule_id = f"in_put_device_{device_id}"
            assert rules[rule_id]["key_exprs"] == [ingress_key]
            assert any(p["rules"] == [rule_id] and p["subjects"] == [device_id] for p in policies)


def test_certificate_generator_includes_each_device_identity(tmp_path):
    from utils.cert_lifecycle import CertAuthority, DEFAULT_IDENTITIES

    module_path = Path("config/generate_certs.py")
    spec = importlib.util.spec_from_file_location("lumen_patch03_certgen", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    authority = module.generate_mtls_certificates(str(tmp_path))
    try:
        _, bundle_dir, _ = CertAuthority.inspect_active(authority.pointer_path)
        for cn in DEFAULT_IDENTITIES:
            assert (bundle_dir / f"{cn}.pem").is_file()
            assert (bundle_dir / f"{cn}.key").is_file()
    finally:
        authority.cleanup()