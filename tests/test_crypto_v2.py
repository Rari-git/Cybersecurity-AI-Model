import json
import os
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from utils.crypto_utils import CommandSigner
from utils.replay_store import ReplayStore


def setup_signer(tmp_path, monkeypatch):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "controller.private.pem"
    public_path = tmp_path / "controller.public.pem"
    private_path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_path.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    if os.name == "posix":
        private_path.chmod(0o600)
    trust_path = tmp_path / "trusted.json"
    trust_path.write_text(json.dumps({
        "version": 1,
        "signers": {"controller": {
            "active_key_id": "controller-v1",
            "keys": {"controller-v1": {
                "public_key": public_path.name,
                "allowed_topics": ["home/actuators/door/lock", "home/actuators/garage/door"],
            }},
        }},
    }), encoding="utf-8")
    monkeypatch.setenv("LUMEN_TRUSTED_COMMAND_SIGNERS", str(trust_path))
    monkeypatch.setenv("LUMEN_COMMAND_PRIVATE_KEY", str(private_path))
    monkeypatch.setenv("LUMEN_COMMAND_SIGNER_ID", "controller")
    monkeypatch.setenv("LUMEN_SEQUENCE_DB", str(tmp_path / "sequence.sqlite3"))
    return trust_path


def test_valid_frame_is_consumed_and_duplicate_rejected(tmp_path, monkeypatch):
    setup_signer(tmp_path, monkeypatch)
    topic = "home/actuators/door/lock"
    frame = CommandSigner.sign_command(topic, "UNLOCK")
    store = ReplayStore(tmp_path / "gateway.sqlite3")
    assert CommandSigner.verify_command(topic, frame, replay_store=store) == (True, "AUTHENTICATED")
    ok, reason = CommandSigner.verify_command(topic, frame, replay_store=store)
    assert not ok and "REPLAY_DUPLICATE_NONCE" in reason


def test_tampering_and_cross_topic_replay_fail_signature(tmp_path, monkeypatch):
    setup_signer(tmp_path, monkeypatch)
    frame = json.loads(CommandSigner.sign_command("home/actuators/door/lock", "LOCK"))
    frame["action"] = "UNLOCK"
    ok, reason = CommandSigner.verify_command(
        "home/actuators/door/lock", json.dumps(frame), replay_store=ReplayStore(tmp_path / "tamper.sqlite3")
    )
    assert not ok and reason == "INVALID_SIGNATURE"
    valid_for_other = CommandSigner.sign_command("home/actuators/garage/door", "UNLOCK")
    ok, reason = CommandSigner.verify_command(
        "home/actuators/door/lock", valid_for_other,
        replay_store=ReplayStore(tmp_path / "cross.sqlite3"),
    )
    assert not ok and reason == "INVALID_SIGNATURE"


def test_signer_cannot_issue_command_to_unlisted_topic(tmp_path, monkeypatch):
    setup_signer(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        CommandSigner.sign_command("home/actuators/kitchen/oven", "UNLOCK")


def test_verifier_does_not_need_private_key(tmp_path, monkeypatch):
    setup_signer(tmp_path, monkeypatch)
    frame = CommandSigner.sign_command("home/actuators/door/lock", "LOCK")
    monkeypatch.delenv("LUMEN_COMMAND_PRIVATE_KEY")
    store = ReplayStore(tmp_path / "verifier.sqlite3")
    assert CommandSigner.verify_command("home/actuators/door/lock", frame, replay_store=store)[0]
