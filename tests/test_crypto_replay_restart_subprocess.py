"""Durable anti-replay check across genuinely separate Python processes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from utils.crypto_utils import CommandSigner


def _configure_signer(tmp_path, monkeypatch):
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
        "signers": {"controller": {"active_key_id": "controller-v1", "keys": {
            "controller-v1": {"public_key": public_path.name,
                              "allowed_topics": ["home/actuators/door/lock"]}
        }}},
    }), encoding="utf-8")
    db_path = tmp_path / "gateway_replay.sqlite3"
    monkeypatch.setenv("LUMEN_TRUSTED_COMMAND_SIGNERS", str(trust_path))
    monkeypatch.setenv("LUMEN_COMMAND_PRIVATE_KEY", str(private_path))
    monkeypatch.setenv("LUMEN_COMMAND_SIGNER_ID", "controller")
    monkeypatch.setenv("LUMEN_SEQUENCE_DB", str(tmp_path / "sequence.sqlite3"))
    monkeypatch.setenv("LUMEN_GATEWAY_REPLAY_DB", str(db_path))
    return db_path, trust_path


def _verify_in_new_process(frame: str, db_path: Path, cwd: Path, env: dict) -> dict:
    code = r'''
import json, os
from utils.crypto_utils import CommandSigner
from utils.replay_store import ReplayStore
ok, reason = CommandSigner.verify_command(
    "home/actuators/door/lock", os.environ["LUMEN_TEST_FRAME"],
    replay_store=ReplayStore.for_component("gateway", os.environ["LUMEN_GATEWAY_REPLAY_DB"]),
)
print(json.dumps({"ok": ok, "reason": reason}))
'''
    child_env = dict(env)
    child_env["LUMEN_TEST_FRAME"] = frame
    child_env["LUMEN_GATEWAY_REPLAY_DB"] = str(db_path)
    result = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=child_env,
                            text=True, capture_output=True, check=True, timeout=20)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_duplicate_is_rejected_after_process_restart_even_when_timestamp_expired(tmp_path, monkeypatch):
    db_path, trust_path = _configure_signer(tmp_path, monkeypatch)
    frame = CommandSigner.sign_command("home/actuators/door/lock", "UNLOCK")
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["LUMEN_TRUSTED_COMMAND_SIGNERS"] = str(trust_path)
    env["LUMEN_GATEWAY_REPLAY_DB"] = str(db_path)
    first = _verify_in_new_process(frame, db_path, project_root, env)
    assert first == {"ok": True, "reason": "AUTHENTICATED"}
    time.sleep(3.2)
    second = _verify_in_new_process(frame, db_path, project_root, env)
    assert second["ok"] is False
    assert "REPLAY_DUPLICATE_NONCE" in second["reason"]
    assert "REPLAY_EXPIRED_TIMESTAMP" not in second["reason"]


def test_never_consumed_expired_frame_still_fails_timestamp_check(tmp_path, monkeypatch):
    from utils import crypto_utils
    from utils.replay_store import ReplayStore
    db_path, _ = _configure_signer(tmp_path, monkeypatch)
    frame = CommandSigner.sign_command("home/actuators/door/lock", "LOCK")
    packet = json.loads(frame)
    store = ReplayStore(db_path)
    monkeypatch.setattr(crypto_utils.time, "time", lambda: packet["timestamp"] + 10.0)
    ok, reason = CommandSigner.verify_command("home/actuators/door/lock", frame, replay_store=store)
    assert not ok and "REPLAY_EXPIRED_TIMESTAMP" in reason
    assert store.get_last_nonce("controller", "home/actuators/door/lock") == 0


def test_invalid_signature_does_not_advance_persistent_nonce(tmp_path, monkeypatch):
    db_path, _ = _configure_signer(tmp_path, monkeypatch)
    frame = json.loads(CommandSigner.sign_command("home/actuators/door/lock", "LOCK"))
    frame["action"] = "UNLOCK"
    from utils.replay_store import ReplayStore
    store = ReplayStore(db_path)
    ok, reason = CommandSigner.verify_command("home/actuators/door/lock", json.dumps(frame), replay_store=store)
    assert not ok and reason == "INVALID_SIGNATURE"
    assert store.get_last_nonce("controller", "home/actuators/door/lock") == 0
