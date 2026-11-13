"""Ed25519 command authentication with durable, restart-aware anti-replay checks.

Protocol v2: commands are signed by an identified controller. Verifiers receive
public keys only. The signed canonical message includes the destination topic.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from utils.key_lifecycle import RevocationStoreError, load_revocations
from utils.replay_store import ReplayStore, ReplayStoreError, SequenceStore

MAX_TIME_DRIFT_SEC = 3.0
MAX_FRAME_BYTES = 512
MAX_TOPIC_CHARS = 256
MAX_NONCE = 2**63 - 1
ALLOWED_ACTIONS = frozenset({"LOCK", "UNLOCK"})
INGEST_PREFIX = "ingest/"
_FRAME_FIELDS = frozenset({"action", "timestamp", "nonce", "sender_id", "key_id", "signature"})
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SIG_RE = re.compile(r"^[0-9a-f]{128}$")

# Patch 03.1: in-process revocations take effect immediately, even if the shared ledger
# file cannot be written. They are monotone (never removed during the process lifetime).
_PROCESS_REVOKED: set = set()
_PROCESS_REVOKED_LOCK = threading.Lock()

class KeyRotationPending(ValueError):
    """Signing key is mid-rotation. Still a ValueError so existing callers keep working."""
    def __init__(self, message: str, max_wait_sec: float):
        super().__init__(message)
        self.max_wait_sec = max_wait_sec

def canonical_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise ValueError("topic must be a string")
    if topic.startswith(INGEST_PREFIX):
        return "home/" + topic[len(INGEST_PREFIX):]
    return topic

def _validate_topic(topic: str) -> str:
    topic = canonical_topic(topic)
    if (not topic or len(topic) > MAX_TOPIC_CHARS or not topic.isprintable()
            or any(ch in topic for ch in "*$?#")):
        raise ValueError("topic is invalid")
    if not topic.startswith("home/actuators/"):
        raise ValueError("command topic must be under home/actuators/")
    return topic

def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result

def _reject_constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")

def _canonical_bytes(topic: str, *, action: str, timestamp: float, nonce: int,
                     sender_id: str, key_id: str) -> bytes:
    body = {
        "action": action,
        "key_id": key_id,
        "nonce": nonce,
        "sender_id": sender_id,
        "timestamp": timestamp,
        "topic": topic,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def _trust_store_path() -> Path:
    return Path(os.environ.get(
        "LUMEN_TRUSTED_COMMAND_SIGNERS", "config/trusted_command_signers.json"
    )).expanduser().resolve()

@lru_cache(maxsize=16)
def _load_trust_store_cached(path_text: str, mtime_ns: int, size: int) -> dict:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("signers"), dict):
        raise ValueError("trusted signer file must contain version=1 and a signers object")

    trusted = {}
    for signer_id, signer_doc in doc["signers"].items():
        if not isinstance(signer_id, str) or not _ID_RE.fullmatch(signer_id):
            raise ValueError(f"invalid signer id: {signer_id!r}")
        if not isinstance(signer_doc, dict):
            raise ValueError(f"invalid signer entry for {signer_id}")
        active_key_id = signer_doc.get("active_key_id")
        keys = signer_doc.get("keys")
        if not isinstance(active_key_id, str) or not isinstance(keys, dict) or active_key_id not in keys:
            raise ValueError(f"signer {signer_id} has no active key")
        parsed_keys = {}
        for key_id, key_doc in keys.items():
            if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id) or not isinstance(key_doc, dict):
                raise ValueError(f"invalid key entry for signer {signer_id}")
            public_value = key_doc.get("public_key")
            topics = key_doc.get("allowed_topics")
            if not isinstance(public_value, str) or not isinstance(topics, list) or not topics:
                raise ValueError(f"key {key_id} needs public_key and non-empty allowed_topics")
            normalized_topics = []
            for topic in topics:
                normalized = _validate_topic(topic)
                if normalized != topic:
                    raise ValueError(f"allowed topic must use canonical home/ form: {topic!r}")
                normalized_topics.append(normalized)
            public_path = (path.parent / public_value).resolve() if not Path(public_value).is_absolute() else Path(public_value)
            if not public_path.is_file():
                raise FileNotFoundError(f"public key missing for {signer_id}/{key_id}: {public_path}")
            public_key = serialization.load_pem_public_key(public_path.read_bytes())
            if not isinstance(public_key, Ed25519PublicKey):
                raise ValueError(f"public key {public_path} is not Ed25519")
            parsed_keys[key_id] = {
                "public_key": public_key,
                "allowed_topics": frozenset(normalized_topics),
            }
        trusted[signer_id] = {"active_key_id": active_key_id, "keys": parsed_keys}
    if not trusted:
        raise ValueError("trusted signer file has no signers")
    return trusted

def _revocation_path() -> Path:
    configured = os.environ.get("LUMEN_REVOKED_KEYS", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _trust_store_path().with_name("revoked_keys.json")

def _is_revoked(sender_id: str, key_id: str) -> bool:
    with _PROCESS_REVOKED_LOCK:
        if (sender_id, key_id) in _PROCESS_REVOKED:
            return True
    return (sender_id, key_id) in load_revocations(_revocation_path())

def _load_trust_store() -> dict:
    path = _trust_store_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Trusted signer file not found: {path}. Run scripts/generate_command_keys.py first."
        )
    result = path.stat()
    return _load_trust_store_cached(str(path), result.st_mtime_ns, result.st_size)

def _load_private_key(path: Path, passphrase: Optional[bytes]) -> Ed25519PrivateKey:
    if not path.is_file():
        raise FileNotFoundError(f"Command signing private key not found: {path}")
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(f"Private key permissions are too broad ({oct(mode)}); require 0600")
    key = serialization.load_pem_private_key(path.read_bytes(), password=passphrase)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Command signing key must be Ed25519")
    return key

class CommandSigner:
    """Create and verify topic-bound Ed25519 actuator commands."""

    @classmethod
    def validate_trust_store(cls) -> None:
        _load_trust_store()
        load_revocations(_revocation_path())   # corrupt ledger => refuse to start (fail-closed)

    @classmethod
    def revocation_path(cls) -> Path:
        return _revocation_path()

    @classmethod
    def revoke_in_process(cls, sender_id: str, key_id: str) -> None:
        with _PROCESS_REVOKED_LOCK:
            _PROCESS_REVOKED.add((sender_id, key_id))

    @classmethod
    def _reset_process_revocations(cls) -> None:   # test hook
        with _PROCESS_REVOKED_LOCK:
            _PROCESS_REVOKED.clear()

    @classmethod
    def sign_command(cls, topic: str, action: str, sender_id: Optional[str] = None) -> str:
        """Sign, transparently riding out an in-flight key rotation (<= exc.max_wait_sec)."""
        deadline = None
        while True:
            try:
                return cls._sign_once(topic, action, sender_id)
            except KeyRotationPending as exc:
                if deadline is None:
                    deadline = time.monotonic() + exc.max_wait_sec
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    @classmethod
    def _sign_once(cls, topic: str, action: str, sender_id: Optional[str] = None) -> str:
        canonical = _validate_topic(topic)
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"action is not allowed: {action!r}")
        sender = sender_id or os.environ.get("LUMEN_COMMAND_SIGNER_ID", "controller")
        if not isinstance(sender, str) or not _ID_RE.fullmatch(sender):
            raise ValueError("LUMEN_COMMAND_SIGNER_ID is invalid")
        trust = _load_trust_store()
        if sender not in trust:
            raise ValueError(f"signer {sender!r} is not present in trusted signer configuration")
        signer_entry = trust[sender]
        key_id = signer_entry["active_key_id"]
        key_entry = signer_entry["keys"][key_id]
        if canonical not in key_entry["allowed_topics"]:
            raise PermissionError(f"signer {sender!r} is not authorized for {canonical!r}")
        if _is_revoked(sender, key_id):
            raise KeyRotationPending(
                f"active signing key {key_id!r} is revoked; awaiting rotation", max_wait_sec=3.0)

        private_path_text = os.environ.get("LUMEN_COMMAND_PRIVATE_KEY", "")
        if not private_path_text.strip():
            raise ValueError("LUMEN_COMMAND_PRIVATE_KEY is required in the command-signing process")
        passphrase_text = os.environ.get("LUMEN_COMMAND_KEY_PASSPHRASE")
        passphrase = passphrase_text.encode("utf-8") if passphrase_text else None
        private_key = _load_private_key(Path(private_path_text).expanduser().resolve(), passphrase)

        public_key = key_entry["public_key"]
        if private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ) != public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ):
            # Also the transient state between the authority replacing the private key
            # file and committing the new trust store (milliseconds).
            raise KeyRotationPending(
                "private signing key does not match active trusted public key", max_wait_sec=0.25)

        sequence_path = os.environ.get("LUMEN_SEQUENCE_DB")
        nonce = SequenceStore.for_sender(sender, sequence_path).next_nonce(sender)
        timestamp = time.time()
        if not math.isfinite(timestamp):
            raise ValueError("system clock returned a non-finite timestamp")
        message = _canonical_bytes(
            canonical, action=action, timestamp=timestamp, nonce=nonce,
            sender_id=sender, key_id=key_id,
        )
        signature = private_key.sign(message).hex()
        packet = {
            "action": action,
            "timestamp": timestamp,
            "nonce": nonce,
            "sender_id": sender,
            "key_id": key_id,
            "signature": signature,
        }
        encoded = json.dumps(packet, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ValueError("signed command exceeds MAX_FRAME_BYTES")
        return encoded

    @classmethod
    def verify_command(cls, topic: str, packet_str: str, last_seen_nonce: int = 0,
                       replay_store: Optional[ReplayStore] = None) -> tuple[bool, str]:
        """Verify a command and atomically consume a fresh nonce.

        Order when a durable store is supplied:
          1. strict structure and topic authorization;
          2. Ed25519 signature;
          3. read-only duplicate check against persisted state;
          4. timestamp freshness;
          5. atomic nonce consume (closes races between concurrent verifiers).

        This order deliberately reports an already-consumed captured frame as a
        duplicate even if it has become older than the 3-second timestamp window
        during a process restart. A never-consumed stale frame still fails freshness.
        """
        try:
            canonical = _validate_topic(topic)
            if not isinstance(packet_str, str):
                return False, "MALFORMED_PACKET (types)"
            if len(packet_str.encode("utf-8", errors="strict")) > MAX_FRAME_BYTES:
                return False, "MALFORMED_PACKET (oversize)"
            packet = json.loads(packet_str, parse_constant=_reject_constant,
                                object_pairs_hook=_strict_pairs)
            if not isinstance(packet, dict) or set(packet) != _FRAME_FIELDS:
                return False, "MALFORMED_PACKET (fields)"

            action = packet["action"]
            timestamp = packet["timestamp"]
            nonce = packet["nonce"]
            sender_id = packet["sender_id"]
            key_id = packet["key_id"]
            signature = packet["signature"]

            if action not in ALLOWED_ACTIONS:
                return False, "INVALID_ACTION"
            if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                return False, "MALFORMED_PACKET (timestamp)"
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                return False, "MALFORMED_PACKET (timestamp non-finite)"
            if isinstance(nonce, bool) or not isinstance(nonce, int) or not 0 < nonce <= MAX_NONCE:
                return False, "MALFORMED_PACKET (nonce)"
            if not isinstance(sender_id, str) or not _ID_RE.fullmatch(sender_id):
                return False, "MALFORMED_PACKET (sender_id)"
            if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id):
                return False, "MALFORMED_PACKET (key_id)"
            if not isinstance(signature, str) or not _SIG_RE.fullmatch(signature):
                return False, "MALFORMED_PACKET (signature)"

            trust = _load_trust_store()
            signer = trust.get(sender_id)
            if signer is None or key_id not in signer["keys"]:
                return False, "UNTRUSTED_SIGNER"
            if _is_revoked(sender_id, key_id):
                return False, "KEY_REVOKED"
            key_entry = signer["keys"][key_id]
            if canonical not in key_entry["allowed_topics"]:
                return False, "TOPIC_NOT_AUTHORIZED"

            message = _canonical_bytes(
                canonical, action=action, timestamp=timestamp, nonce=nonce,
                sender_id=sender_id, key_id=key_id,
            )
            try:
                key_entry["public_key"].verify(bytes.fromhex(signature), message)
            except InvalidSignature:
                return False, "INVALID_SIGNATURE"

            if replay_store is not None:
                try:
                    previous = replay_store.get_last_nonce(sender_id, canonical)
                except ReplayStoreError:
                    return False, "REPLAY_STORE_ERROR (fail-closed)"
                if nonce <= previous:
                    return False, f"REPLAY_DUPLICATE_NONCE (Received: {nonce}, Last: {previous})"

            now = time.time()
            if abs(now - timestamp) > MAX_TIME_DRIFT_SEC:
                return False, f"REPLAY_EXPIRED_TIMESTAMP (Drift: {now - timestamp:.2f}s)"

            if replay_store is not None:
                try:
                    accepted, previous = replay_store.consume(sender_id, canonical, nonce)
                except ReplayStoreError:
                    return False, "REPLAY_STORE_ERROR (fail-closed)"
                if not accepted:
                    return False, f"REPLAY_DUPLICATE_NONCE (Received: {nonce}, Last: {previous})"
            else:
                if isinstance(last_seen_nonce, bool) or not isinstance(last_seen_nonce, int) or last_seen_nonce < 0:
                    return False, "MALFORMED_PACKET (last_seen_nonce)"
                if nonce <= last_seen_nonce:
                    return False, f"REPLAY_DUPLICATE_NONCE (Received: {nonce}, Expected > {last_seen_nonce})"
            return True, "AUTHENTICATED"
        except RevocationStoreError:
            return False, "REVOCATION_STORE_ERROR (fail-closed)"
        except FileNotFoundError:
            return False, "TRUST_STORE_ERROR (missing file)"
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
            return False, "MALFORMED_PACKET"
        except Exception as exc:
            return False, f"CRYPTO_FAILURE ({type(exc).__name__})"
