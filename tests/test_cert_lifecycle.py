"""Patch 04 regression suite: mTLS lifecycle, atomic publication and incidents."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from utils.cert_lifecycle import (
    CertAuthority, CertRevocationLedger, CertRevocationStoreError,
    CertRotationSuspended, CertStoreError, revoke_active_certificate,
)
from utils.key_lifecycle import KeyAuthority
from utils.mtls_hot_reload import HotReloadZenohClient


IDENTITIES = ("router", "gateway", "ids", "honeypot")


def _signer_template(root: Path) -> Path:
    path = root / "signers.json"
    path.write_text(json.dumps({"version": 1, "signers": {
        "controller": {"active_key_id": "old", "keys": {
            "old": {"public_key": "unused.pem", "allowed_topics": ["home/actuators/**"]}
        }}
    }}), encoding="utf-8")
    return path


@pytest.fixture
def stack(tmp_path):
    clock = [1_800_000_000.0]
    key = KeyAuthority.create_session(
        _signer_template(tmp_path), base_dir=tmp_path,
        audit_path=tmp_path / "key-audit.jsonl", clock=lambda: clock[0],
    )
    cert = CertAuthority.create_session(
        key.session_dir, identities=IDENTITIES, leaf_ttl_days=14,
        ca_validity_days=3650, max_rotations_per_hour=6,
        clock=lambda: clock[0], monotonic=lambda: clock[0],
    )
    key.attach_cert_authority(cert)
    yield key, cert, clock
    if key.session_dir.exists():
        key.cleanup()


def test_session_ca_private_key_is_owner_only_and_removed_on_cleanup(stack):
    key, cert, _ = stack
    assert cert.ca_key_path.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(cert.ca_key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(cert.ca_dir.stat().st_mode) == 0o700
        active = CertAuthority.inspect_active(cert.pointer_path)[1]
        assert stat.S_IMODE((active / "ids.key").stat().st_mode) == 0o600
    ca_key = cert.ca_key_path
    session = key.session_dir
    key.cleanup()
    assert not ca_key.exists()
    assert not session.exists()


def test_identity_keypairs_are_distinct_and_ids_is_not_honeypot(stack):
    _, cert, _ = stack
    _pointer, bundle, manifest = CertAuthority.inspect_active(cert.pointer_path)
    certs = {}
    for identity in ("ids", "honeypot"):
        parsed = x509.load_pem_x509_certificate((bundle / f"{identity}.pem").read_bytes())
        key = serialization.load_pem_private_key((bundle / f"{identity}.key").read_bytes(), password=None)
        assert parsed.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ) == key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        certs[identity] = parsed
    assert certs["ids"].public_key().public_numbers() != certs["honeypot"].public_key().public_numbers()
    assert manifest["identities"]["ids"]["common_name"] == "ids"
    assert manifest["identities"]["honeypot"]["common_name"] == "honeypot"


def test_scheduled_rotation_occurs_at_two_thirds_of_ttl(stack):
    key, cert, clock = stack
    before = cert.status()["ids"]["common_name"]
    # 70% of the 14-day leaf TTL is later than the renewal threshold (2/3).
    clock[0] += 14 * 86400 * 0.70
    events = key.poll()
    ids_events = [e for e in events if e.get("component") == "certificate" and e.get("identity") == "ids"]
    assert len(ids_events) == 1
    assert ids_events[0]["reason"] == "scheduled"
    assert cert.status()["ids"]["common_name"] != before
    assert cert.status()["ids"]["generation"] == 2


def test_atomic_publication_rejects_corrupt_or_incomplete_replacement(stack, monkeypatch):
    _, cert, _ = stack
    before = cert.pointer_path.read_bytes()
    original_issue = cert._issue_leaf

    def corrupt_issue(identity, generation, ca_key, ca_cert):
        cert_pem, _key_pem, info = original_issue(identity, generation, ca_key, ca_cert)
        # Return a certificate and a different private key. Stage verification must fail.
        other_cert, other_key, _ = original_issue("gateway", 90, ca_key, ca_cert)
        return cert_pem, other_key, info

    monkeypatch.setattr(cert, "_issue_leaf", corrupt_issue)
    with pytest.raises(CertStoreError):
        cert.rotate("ids", "corrupt-package")
    assert cert.pointer_path.read_bytes() == before
    # Ledger revocation is written before mint/publication, so a failed rotation
    # cannot resurrect the old identity. The active CN is subsequently quarantined
    # by the router's ledger-derived ACL until a valid replacement is installed.
    status = cert.status()["ids"]
    assert status["revoked"] is True


def test_bundle_tampering_is_detected_before_status_or_rotation(stack):
    _, cert, _ = stack
    _pointer, bundle, _manifest = CertAuthority.inspect_active(cert.pointer_path)
    with (bundle / "ids.key").open("ab") as handle:
        handle.write(b"#corrupt")
    with pytest.raises(CertStoreError):
        cert.status()
    with pytest.raises(CertStoreError):
        cert.rotate("ids", "after-corruption")


def test_revocation_ledger_corruption_and_disappearance_fail_closed(stack):
    _, cert, _ = stack
    cert.revocation_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CertRevocationStoreError):
        cert.status()
    with pytest.raises(CertRevocationStoreError):
        cert.poll()
    cert.revocation_path.unlink()
    with pytest.raises(CertRevocationStoreError):
        CertRevocationLedger(cert.revocation_path).revoked_serials()


def test_rotation_rate_limit_suspends_and_quarantines_identity(stack):
    _, cert, _ = stack
    cert.max_rotations_per_hour = 1
    cert.rotate("ids", "first-rotation")
    with pytest.raises(CertRotationSuspended):
        cert.rotate("ids", "second-rotation")
    state = cert.status()["ids"]
    assert state["suspended"] is True
    assert state["quarantined"] is True
    pointer, _bundle, _manifest = CertAuthority.inspect_active(cert.pointer_path)
    assert "ids" in pointer["quarantined"]


def test_incident_revokes_old_serial_rotates_legitimate_node_and_unquarantines(stack):
    key, cert, _ = stack
    old = cert.status()["ids"]
    result = revoke_active_certificate(cert.pointer_path, cert.revocation_path, "ids", "device compromise")
    assert result["serial"] == old["serial"]
    pointer, _, _ = CertAuthority.inspect_active(cert.pointer_path)
    assert "ids" in pointer["quarantined"]
    events = key.poll()
    assert any(e.get("identity") == "ids" and e.get("reason") == "incident" for e in events)
    new = cert.status()["ids"]
    assert new["serial"] != old["serial"]
    assert new["common_name"] != old["common_name"]
    assert new["revoked"] is False
    assert new["quarantined"] is False
    assert CertRevocationLedger(cert.revocation_path).is_revoked("ids", old["serial"])


def test_router_acl_renderer_uses_new_generation_cn_and_denies_quarantined_identity(stack, tmp_path):
    from utils.zenoh_router import build_router_config
    _, cert, _ = stack
    # Build a tiny JSON5 template containing the complete role subject set.
    subjects = ",\n".join(
        f'{{ id: "{identity}", cert_common_names: ["{identity}"] }}'
        for identity in ("gateway", "ids", "honeypot")
    )
    template = tmp_path / "router.json5"
    template.write_text(
        '{ mode: "router", transport: { link: { tls: { '
        'root_ca_certificate: "old-ca", listen_private_key: "old-key", '
        'listen_certificate: "old-cert", enable_mtls: true } } }, '
        f'access_control: {{ subjects: [{subjects}], policies: [] }} }}',
        encoding="utf-8",
    )
    output = tmp_path / "active.json5"
    old_cn = cert.status()["ids"]["common_name"]
    new_cn = cert.rotate("ids", "acl-test")
    build_router_config(template, cert.pointer_path, cert.revocation_path, output)
    rendered = output.read_text(encoding="utf-8")
    assert new_cn in rendered
    assert f'cert_common_names: ["{old_cn}"]' not in rendered
    cert.quarantine("honeypot", "incident-test")
    build_router_config(template, cert.pointer_path, cert.revocation_path, output)
    rendered = output.read_text(encoding="utf-8")
    assert "__lumen_denied__honeypot" in rendered
    assert "__lumen_denied__ids" not in rendered


class FakeSubscription:
    def __init__(self):
        self.undeclared = False
    def undeclare(self):
        self.undeclared = True


class FakeSession:
    def __init__(self):
        self.closed = False
        self.subscribers = []
        self.puts = []
    def declare_subscriber(self, key, callback, **kwargs):
        handle = FakeSubscription()
        self.subscribers.append((key, callback, handle))
        return handle
    def put(self, *args, **kwargs):
        self.puts.append((args, kwargs))
    def close(self):
        self.closed = True


def test_hot_reload_reconnects_and_redeclares_subscribers(stack):
    _, cert, clock = stack
    sessions = []
    def factory(_cert, _key, _ca, _endpoint):
        item = FakeSession()
        sessions.append(item)
        return item
    client = HotReloadZenohClient(
        "ids", endpoint="tls/127.0.0.1:7447", pointer_path=cert.pointer_path,
        ledger_path=cert.revocation_path, session_factory=factory, clock=lambda: clock[0], start_monitor=False,
    )
    try:
        sub = client.declare_subscriber("home/**", lambda _sample: None)
        assert len(sessions) == 1 and len(sessions[0].subscribers) == 1
        cert.rotate("ids", "scheduled-test")
        assert client._refresh() is True
        assert len(sessions) == 2
        assert sessions[0].closed is True
        assert len(sessions[1].subscribers) == 1
        client.put("home/test", b"data")
        assert sessions[1].puts
        sub.undeclare()
    finally:
        client.close()


def test_hot_reload_drops_session_when_identity_is_revoked(stack):
    _, cert, clock = stack
    sessions = []
    def factory(_cert, _key, _ca, _endpoint):
        item = FakeSession()
        sessions.append(item)
        return item
    client = HotReloadZenohClient(
        "ids", endpoint="tls/127.0.0.1:7447", pointer_path=cert.pointer_path,
        ledger_path=cert.revocation_path, session_factory=factory, clock=lambda: clock[0], start_monitor=False,
    )
    try:
        old_session = sessions[0]
        revoke_active_certificate(cert.pointer_path, cert.revocation_path, "ids", "incident")
        assert client._refresh() is False
        assert old_session.closed is True
        with pytest.raises(RuntimeError):
            client.put("home/test", b"blocked")
        cert.poll()  # authority installs a fresh leaf and clears quarantine
        assert client._refresh() is True
        assert len(sessions) == 2
    finally:
        client.close()


def test_restrict_to_owner_helper_is_used_for_ca_and_leaf_private_keys(tmp_path, monkeypatch):
    from utils import cert_lifecycle as module
    calls = []
    real = module.restrict_to_owner
    def spy(path, *, directory=False):
        calls.append((Path(path).name, directory))
        return real(path, directory=directory)
    monkeypatch.setattr(module, "restrict_to_owner", spy)
    key = KeyAuthority.create_session(
        _signer_template(tmp_path), base_dir=tmp_path,
    )
    try:
        ca = CertAuthority.create_session(key.session_dir, identities=IDENTITIES)
        assert ("ca", True) in calls
        assert any(name == "bundles" and is_dir for name, is_dir in calls)
        assert ca.ca_key_path.is_file()
    finally:
        key.cleanup()


def test_operator_quarantine_is_persistent_and_not_auto_reenrolled(stack):
    key, cert, _ = stack
    before = cert.status()["honeypot"]["serial"]
    cert.quarantine("honeypot", "manual-review")
    events = key.poll()
    assert not any(e.get("identity") == "honeypot" for e in events)
    after = cert.status()["honeypot"]
    assert after["serial"] == before
    assert after["quarantined"] is True
    assert after["pending_rotation"] is False


def test_ledger_revocation_is_idempotent_and_monotone(stack):
    _, cert, _ = stack
    info = cert.status()["ids"]
    ledger = CertRevocationLedger(cert.revocation_path)
    assert ledger.revoke("ids", info["serial"], "incident", common_name=info["common_name"])
    assert not ledger.revoke("ids", info["serial"], "second-writer", common_name=info["common_name"])
    assert ledger.is_revoked("ids", info["serial"])
    assert info["serial"] in ledger.revoked_serials("ids")


def test_run_all_required_cert_set_includes_controller_and_actuator():
    import run_all
    assert "controller.pem" in run_all.REQUIRED_CERTS
    assert "controller.key" in run_all.REQUIRED_CERTS
    assert "actuator.pem" in run_all.REQUIRED_CERTS
    assert "actuator.key" in run_all.REQUIRED_CERTS


def test_ledger_write_failure_still_quarantines_pointer_fail_closed(stack, monkeypatch):
    _, cert, _ = stack
    old = cert.status()["ids"]
    monkeypatch.setattr(CertRevocationLedger, "revoke",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(CertRevocationStoreError):
        revoke_active_certificate(cert.pointer_path, cert.revocation_path, "ids", "disk failure")
    pointer, _, _ = CertAuthority.inspect_active(cert.pointer_path)
    assert "ids" in pointer["quarantined"]
    assert "ids" not in pointer["pending_rotation"]
    assert cert.status()["ids"]["serial"] == old["serial"]
    assert cert.status()["ids"]["quarantined"] is True
