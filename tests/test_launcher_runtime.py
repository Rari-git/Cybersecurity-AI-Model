from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_run_all_passes_active_cert_pointer_and_ledger_to_gateway_and_ids(tmp_path, monkeypatch):
    import run_all

    class FakeAuthority:
        trust_path = tmp_path / "trusted_signers.json"
        revocation_path = tmp_path / "command_revocations.json"
        session_dir = tmp_path / "session"

        @staticmethod
        def env_signer(_signer_id):
            return {"LUMEN_SIGNER_ID": "controller"}

    class FakeCertAuthority:
        pointer_path = tmp_path / "active.json"
        revocation_path = tmp_path / "cert_revocations.json"
        identities = ("gas_sensor_01", "gas_sensor_02", "door_sensor_01")
        leaf_ttl_days = 14
        ca_validity_days = 3650
        max_rotations_per_hour = 6

    model_root = tmp_path / "models"
    pointer = model_root / "active_model.json"
    public_key = tmp_path / "model_signing_ed25519.pub"
    monkeypatch.setattr(run_all, "MODEL_ROOT", model_root)
    monkeypatch.setattr(run_all, "ACTIVE_MODEL_POINTER", pointer)
    monkeypatch.setattr(run_all, "MODEL_PUBLIC_KEY", public_key)

    envs = run_all.build_window_env(FakeAuthority(), FakeCertAuthority(), tmp_path)
    for title, identity in (("Inline Gateway", "gateway"), ("IDS Agent AI", "ids")):
        assert envs[title]["LUMEN_CERT_POINTER"] == str(FakeCertAuthority.pointer_path.resolve())
        assert envs[title]["LUMEN_CERT_REVOCATIONS"] == str(FakeCertAuthority.revocation_path.resolve())
        assert envs[title]["LUMEN_CERT_IDENTITY"] == identity
    ids_env = envs["IDS Agent AI"]
    assert ids_env["LUMEN_ACTIVE_MODEL_POINTER"] == str(pointer)
    assert ids_env["LUMEN_MODEL_PUBKEY"] == str(public_key)
    assert ids_env["LUMEN_ALLOW_UNBUNDLED_FALLBACK"] == "0"
    assert "LUMEN_MODEL_BUNDLE_DIR" not in ids_env

    inherited = {
        "PATH": "test-path",
        "LUMEN_MODEL_BUNDLE_DIR": "models/candidates/unverified-shell-override",
        "LUMEN_ALLOW_EXPLICIT_MODEL_BUNDLE": "1",
        "LUMEN_ALLOW_UNBUNDLED_FALLBACK": "1",
    }
    child_ids = run_all.build_child_env("IDS Agent AI", envs, inherited)
    assert "LUMEN_MODEL_BUNDLE_DIR" not in child_ids
    assert "LUMEN_ALLOW_EXPLICIT_MODEL_BUNDLE" not in child_ids
    assert child_ids["LUMEN_ALLOW_UNBUNDLED_FALLBACK"] == "0"
    assert child_ids["PATH"] == "test-path"


def test_launcher_resolves_and_verifies_signed_active_bundle(tmp_path, monkeypatch):
    import run_all

    model_root = tmp_path / "models"
    active_bundle = model_root / "lumen_model_bundle"
    shutil.copytree(ROOT / "models" / "lumen_model_bundle", active_bundle)
    pointer = model_root / "active_model.json"
    pointer.write_text(json.dumps({"schema": 1, "active_bundle": "lumen_model_bundle"}), encoding="utf-8")
    public_key = tmp_path / "model_signing_ed25519.pub"
    shutil.copy2(ROOT / "config" / "model_keys" / "model_signing_ed25519.pub", public_key)

    monkeypatch.setattr(run_all, "MODEL_ROOT", model_root)
    monkeypatch.setattr(run_all, "ACTIVE_MODEL_POINTER", pointer)
    monkeypatch.setattr(run_all, "MODEL_PUBLIC_KEY", public_key)
    assert run_all.resolve_verified_active_model_bundle() == active_bundle.resolve()

    # Tampering is rejected by manifest signature/hash verification.
    with (active_bundle / "thresholds.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(Exception):
        run_all.resolve_verified_active_model_bundle()


def test_router_templates_retain_default_deny_and_gateway_ids_internal_acl():
    json5 = pytest.importorskip("json5")
    from utils.zenoh_router import validate_router_acl_template
    for name in ("zenohd_router.json5", "zenoh_acl_example.json5"):
        template_path = ROOT / "config" / name
        doc = json5.loads(template_path.read_text(encoding="utf-8"))
        validate_router_acl_template(template_path)
        ac = doc["access_control"]
        assert ac["enabled"] is True
        assert ac["default_permission"] == "deny"
        rules = {rule["id"]: rule for rule in ac["rules"]}
        assert rules["in_put_internal_ids_telemetry"]["key_exprs"] == ["internal/ids/telemetry/**"]
        assert rules["in_put_pep_ack"]["key_exprs"] == ["control/pep_ack/**"]
        assert rules["sub_internal_ids_telemetry"]["key_exprs"] == ["internal/ids/telemetry/**"]
        assert rules["sub_pep_ack"]["key_exprs"] == ["control/pep_ack/**"]
        assert any(
            set(policy["rules"]) >= {"in_put_internal_ids_telemetry", "in_put_pep_ack"}
            and policy["subjects"] == ["gateway"]
            for policy in ac["policies"]
        )
        assert any(
            set(policy["rules"]) >= {"sub_internal_ids_telemetry", "sub_pep_ack"}
            and policy["subjects"] == ["ids"]
            for policy in ac["policies"]
        )
        assert not any(
            policy["subjects"] not in (["gateway"], ["ids"])
            and set(policy["rules"]) & {
                "in_put_internal_ids_telemetry", "in_put_pep_ack",
                "sub_internal_ids_telemetry", "sub_pep_ack",
            }
            for policy in ac["policies"]
        )


def test_rendered_router_config_preserves_internal_acl_and_default_deny(tmp_path):
    json5 = pytest.importorskip("json5")
    from utils.cert_lifecycle import CertAuthority, default_identities
    from utils.zenoh_router import build_router_config

    authority = CertAuthority.create_session(tmp_path, identities=default_identities(), leaf_ttl_days=7)
    try:
        output = tmp_path / "router.active.json5"
        build_router_config(
            ROOT / "config" / "zenohd_router.json5",
            authority.pointer_path,
            authority.revocation_path,
            output,
        )
        rendered = json5.loads(output.read_text(encoding="utf-8"))
        ac = rendered["access_control"]
        assert ac["default_permission"] == "deny"
        rules = {rule["id"]: rule for rule in ac["rules"]}
        assert rules["in_put_internal_ids_telemetry"]["key_exprs"] == ["internal/ids/telemetry/**"]
        assert rules["in_put_pep_ack"]["key_exprs"] == ["control/pep_ack/**"]
        assert any(
            {"in_put_internal_ids_telemetry", "in_put_pep_ack"}.issubset(set(policy["rules"]))
            and policy["subjects"] == ["gateway"]
            for policy in ac["policies"]
        )
        assert any(
            {"sub_internal_ids_telemetry", "sub_pep_ack"}.issubset(set(policy["rules"]))
            and policy["subjects"] == ["ids"]
            for policy in ac["policies"]
        )
    finally:
        authority.cleanup()


def test_legacy_menu_refuses_standalone_ids_without_managed_pki(monkeypatch):
    import builtins
    import main

    for name in ("LUMEN_CERT_POINTER", "LUMEN_CERT_REVOCATIONS", "LUMEN_ZENOH_ENDPOINT", "LUMEN_CERT_IDENTITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "5")
    assert main.main() == 2


def test_legacy_menu_refuses_dashboard_without_managed_pki(monkeypatch):
    import builtins
    import main

    for name in ("LUMEN_CERT_POINTER", "LUMEN_CERT_REVOCATIONS", "LUMEN_ZENOH_ENDPOINT", "LUMEN_CERT_IDENTITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "4")
    assert main.main() == 2


def test_router_acl_validator_rejects_legacy_or_weakened_templates(tmp_path):
    from utils.zenoh_router import validate_router_acl_template

    source = (ROOT / "config" / "zenohd_router.json5").read_text(encoding="utf-8")
    weakened = tmp_path / "weakened-router.json5"
    weakened.write_text(source.replace('default_permission: "deny"', 'default_permission: "allow"', 1), encoding="utf-8")
    with pytest.raises(Exception, match="default_permission: deny"):
        validate_router_acl_template(weakened)

    missing_internal = tmp_path / "missing-internal-router.json5"
    missing_internal.write_text(source.replace('"internal/ids/telemetry/**"', '"internal/telemetry/**"', 1), encoding="utf-8")
    with pytest.raises(Exception, match="in_put_internal_ids_telemetry"):
        validate_router_acl_template(missing_internal)


def test_run_all_preflight_fails_closed_on_weakened_router_override(tmp_path, monkeypatch):
    import run_all

    source = (ROOT / "config" / "zenohd_router.json5").read_text(encoding="utf-8")
    weakened = tmp_path / "operator-override.json5"
    weakened.write_text(source.replace('default_permission: "deny"', 'default_permission: "allow"', 1), encoding="utf-8")
    monkeypatch.setattr(run_all, "ROUTER_TEMPLATE", weakened)
    monkeypatch.setattr(run_all, "resolve_verified_active_model_bundle", lambda: ROOT / "models" / "lumen_model_bundle")
    monkeypatch.setenv("LUMEN_HMAC_KEY", "x" * 48)

    with pytest.raises(SystemExit, match="default_permission: deny"):
        run_all.preflight_or_exit()


def test_live_smoke_reserves_gas_sensor_identity_from_background_simulator(monkeypatch):
    import run_all

    envs = {"Virtual Sensors": {"LUMEN_SIM_SKIP_DEVICES": "pir_sensor_01"}}
    live = run_all.build_child_env(
        "Virtual Sensors",
        envs,
        {"PATH": "test-path", "LUMEN_RUN_LIVE_SMOKE": "1", "LUMEN_SIM_SKIP_DEVICES": "door_sensor_01"},
    )
    assert set(live["LUMEN_SIM_SKIP_DEVICES"].split(",")) == {
        "door_sensor_01", "gas_sensor_01", "pir_sensor_01",
    }

    normal = run_all.build_child_env(
        "Virtual Sensors",
        envs,
        {"PATH": "test-path", "LUMEN_RUN_LIVE_SMOKE": "0"},
    )
    assert normal["LUMEN_SIM_SKIP_DEVICES"] == "pir_sensor_01"


def test_gateway_ready_marker_waits_for_exact_boot_subscriber_contract(tmp_path):
    import run_all

    marker = tmp_path / "gateway_ready.json"
    marker.write_text(json.dumps({
        "status": "READY",
        "run_id": "test-run",
        "boot_topics": {
            "request_subscriber": "control/boot/request/**",
            "proof_subscriber": "control/boot/proof/**",
            "challenge_publisher_prefix": "control/boot/challenge/",
            "result_publisher_prefix": "control/boot/result/",
        },
    }), encoding="utf-8")
    ready = run_all.wait_for_gateway_ready_marker(marker, timeout=0.1)
    assert ready["run_id"] == "test-run"


def test_gateway_ready_marker_rejects_wrong_boot_topic_contract(tmp_path):
    import run_all

    marker = tmp_path / "gateway_ready.json"
    marker.write_text(json.dumps({
        "status": "READY",
        "boot_topics": {
            "request_subscriber": "control/boot/requests/**",
            "proof_subscriber": "control/boot/proof/**",
            "challenge_publisher_prefix": "control/boot/challenge/",
            "result_publisher_prefix": "control/boot/result/",
        },
    }), encoding="utf-8")
    with pytest.raises(TimeoutError, match="did not become ready"):
        run_all.wait_for_gateway_ready_marker(marker, timeout=0.15)
