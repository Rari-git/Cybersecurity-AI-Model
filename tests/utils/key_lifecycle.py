"""Ephemeral command-key lifecycle (Patch 03.1).

Roles (privilege separation is the whole point of this module):

  KeyAuthority  - the ONLY component that mints keys and writes the trust store.
                  Holds private keys only long enough to hand them to the signer
                  through an owner-only file inside a per-session directory.
  Gateway / IDS - verifiers. They hold public keys only. The gateway may *revoke*
                  (append to the revocation ledger) but can never *trust* a key.
                  Revocation is monotone: it can only shrink what is accepted, so
                  a compromised gateway cannot use it to enrol a rogue signer.
  Signer        - the controller process. Reads its private key path from the
                  environment; never sees the ledger writer or the trust writer.

Rotation == revoke(old) -> mint(new) -> commit trust store (atomic, last).
Nothing in here logs or returns private key material.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

log = logging.getLogger("lumen.key_lifecycle")

SESSION_PREFIX = "lumen-session-"
TRUST_FILE = "trusted_command_signers.json"
REVOKED_FILE = "revoked_keys.json"
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_REPLACE_RETRIES = 8
_REPLACE_DELAY_SEC = 0.02


class RevocationStoreError(Exception):
    """Revocation ledger unreadable/corrupt. Verifiers must fail closed."""


class RotationSuspended(Exception):
    """Too many rotations in the last hour; operator action required."""


# ------------------------------------------------------------------ filesystem helpers
def restrict_to_owner(path: Path, *, directory: bool = False) -> None:
    """Owner-only ACL. POSIX: chmod. Windows: icacls (raises if it cannot be applied)."""
    path = Path(path)
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600)
        return
    user = os.environ.get("USERNAME")
    if not user:
        raise PermissionError("cannot restrict ACL: USERNAME is not set")
    domain = os.environ.get("USERDOMAIN")
    principal = f"{domain}\\{user}" if domain else user
    grant = f"{principal}:(OI)(CI)F" if directory else f"{principal}:F"
    result = subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", grant],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise PermissionError(f"icacls failed for {path}: {result.stderr.strip()[:160]}")


def _replace_with_retry(src: Path, dst: Path) -> None:
    # Windows refuses os.replace while another process has dst open without FILE_SHARE_DELETE.
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_DELAY_SEC)


def atomic_write_bytes(path: Path, data: bytes, *, private: bool = False) -> None:
    """Write via temp file + fsync + os.replace so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if private:
            restrict_to_owner(tmp)
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _FileLock:
    """Tiny cross-process lock (O_EXCL lockfile) for the revocation ledger writers."""

    def __init__(self, path: Path, timeout: float = 5.0, stale: float = 10.0):
        self.path, self.timeout, self.stale = Path(str(path) + ".lock"), timeout, stale
        self._fd: Optional[int] = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        os.unlink(self.path)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"revocation ledger lock timeout: {self.path}")
                time.sleep(0.01)

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
        try:
            os.unlink(self.path)
        except OSError:
            pass
        return False


# ------------------------------------------------------------------ revocation ledger
def _parse_revocations(text: str) -> dict:
    doc = json.loads(text)
    if (not isinstance(doc, dict) or doc.get("version") != 1
            or not isinstance(doc.get("revoked"), dict)):
        raise ValueError("revocation ledger must contain version=1 and a revoked object")
    for sender, keys in doc["revoked"].items():
        if not isinstance(sender, str) or not _ID_RE.fullmatch(sender) or not isinstance(keys, dict):
            raise ValueError("invalid revocation sender entry")
        for key_id, meta in keys.items():
            if not isinstance(key_id, str) or not _ID_RE.fullmatch(key_id) or not isinstance(meta, dict):
                raise ValueError("invalid revocation key entry")
    return doc


def _read_text_retry(path: Path) -> str:
    last: Optional[Exception] = None
    for _ in range(4):
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except OSError as exc:      # transient Windows sharing violation during os.replace
            last = exc
            time.sleep(0.01)
    raise RevocationStoreError(f"cannot read revocation ledger: {last}")


@lru_cache(maxsize=8)
def _load_revocations_cached(path_text: str, mtime_ns: int, size: int) -> frozenset:
    try:
        doc = _parse_revocations(_read_text_retry(Path(path_text)))
    except FileNotFoundError:
        return frozenset()
    except RevocationStoreError:
        raise
    except (ValueError, TypeError) as exc:
        raise RevocationStoreError(f"corrupt revocation ledger: {exc}") from exc
    return frozenset((s, k) for s, keys in doc["revoked"].items() for k in keys)


def load_revocations(path) -> frozenset:
    """{(sender_id, key_id)} that are revoked. Missing file == nothing revoked.
    Unreadable or corrupt file raises RevocationStoreError (verifiers fail closed)."""
    p = Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return frozenset()
    except OSError as exc:
        raise RevocationStoreError(f"cannot stat revocation ledger: {exc}") from exc
    return _load_revocations_cached(str(p), st.st_mtime_ns, st.st_size)


class RevocationLedger:
    """Append-only (monotone) set of revoked (sender_id, key_id). Multi-writer safe."""

    def __init__(self, path):
        self.path = Path(path)
        self._tlock = threading.Lock()

    def _read_doc(self) -> dict:
        try:
            return _parse_revocations(_read_text_retry(self.path))
        except FileNotFoundError:
            return {"version": 1, "revoked": {}}
        except (ValueError, TypeError) as exc:
            # Never overwrite a corrupt ledger: that would silently un-revoke keys.
            raise RevocationStoreError(f"corrupt revocation ledger: {exc}") from exc

    def ensure(self) -> None:
        with self._tlock, _FileLock(self.path):
            if not self.path.exists():
                atomic_write_bytes(self.path, json.dumps(self._read_doc(), indent=2).encode("utf-8"))

    def revoke(self, sender_id: str, key_id: str, reason: str, *, now: Optional[float] = None) -> bool:
        if not _ID_RE.fullmatch(str(sender_id)) or not _ID_RE.fullmatch(str(key_id)):
            raise ValueError("invalid sender_id/key_id for revocation")
        safe_reason = "".join(c if c.isprintable() else "?" for c in str(reason))[:120]
        with self._tlock, _FileLock(self.path):
            doc = self._read_doc()
            entries = doc["revoked"].setdefault(sender_id, {})
            if key_id in entries:
                return False
            entries[key_id] = {"ts": round(now if now is not None else time.time(), 3),
                               "reason": safe_reason}
            atomic_write_bytes(self.path, json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))
            return True


# ------------------------------------------------------------------ incident monitor (gateway side)
@dataclass(frozen=True)
class FrameIdentity:
    sender_id: str
    key_id: str
    nonce: int


def frame_identity(raw) -> Optional[FrameIdentity]:
    """Best-effort, UNAUTHENTICATED parse of the claimed (sender, key, nonce)."""
    try:
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, (bytes, bytearray)) else str(raw)
        doc = json.loads(text)
        sender, key_id, nonce = doc["sender_id"], doc["key_id"], doc["nonce"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if (not isinstance(sender, str) or not _ID_RE.fullmatch(sender)
            or not isinstance(key_id, str) or not _ID_RE.fullmatch(key_id)
            or isinstance(nonce, bool) or not isinstance(nonce, int)):
        return None
    return FrameIdentity(sender, key_id, nonce)


def _digest(raw) -> str:
    data = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


@dataclass(frozen=True)
class RekeyDecision:
    revoke: bool
    sender_id: Optional[str] = None
    key_id: Optional[str] = None
    reason: str = ""


NO_ACTION = RekeyDecision(False)


class KeyHealthMonitor:
    """Turns verifier rejections into 'revoke this key' decisions.

    Evidence quality matters because the trigger itself is an attack surface:
      equivocation  signature VALID + same (target, nonce) as an accepted frame but
                    different bytes -> two holders of the private key. Unforgeable
                    without the key. Immediate revoke.
      replay        signature VALID but nonce/timestamp stale. Proves capture, not key
                    compromise; rotation bounds the capture's lifetime. Threshold.
      tampering     INVALID signature naming a trusted key. Forgeable by anyone with
                    publish rights (they can name any key_id), so threshold + cooldown
                    keep it from becoming a cheap lever to force rotations.
    Reasons that carry no key-bound evidence (KEY_REVOKED, UNTRUSTED_SIGNER, TOPIC_*,
    MALFORMED_*) are ignored on purpose.
    """

    def __init__(self, *, window_sec: float = 30.0, replay_threshold: int = 5,
                 tamper_threshold: int = 10, cooldown_sec: float = 20.0,
                 max_accepted: int = 4096, max_tracked: int = 256,
                 clock: Callable[[], float] = time.monotonic):
        self.window_sec = float(window_sec)
        self.thresholds = {"replay": int(replay_threshold), "tampering": int(tamper_threshold)}
        self.cooldown_sec = float(cooldown_sec)
        self.max_accepted, self.max_tracked = int(max_accepted), int(max_tracked)
        self._clock = clock
        self._lock = threading.Lock()
        self._accepted: "OrderedDict[tuple, str]" = OrderedDict()
        self._events: "OrderedDict[tuple, deque]" = OrderedDict()
        self._cooldown_until: dict = {}

    @classmethod
    def from_env(cls) -> "KeyHealthMonitor":
        env = os.environ.get
        return cls(window_sec=float(env("LUMEN_REKEY_WINDOW_SEC", "30")),
                   replay_threshold=int(env("LUMEN_REKEY_REPLAY_THRESHOLD", "5")),
                   tamper_threshold=int(env("LUMEN_REKEY_TAMPER_THRESHOLD", "10")),
                   cooldown_sec=float(env("LUMEN_REKEY_COOLDOWN_SEC", "20")))

    def on_accepted(self, target: str, raw) -> None:
        ident = frame_identity(raw)
        if ident is None:
            return
        with self._lock:
            self._accepted[(target, ident.nonce)] = _digest(raw)
            self._accepted.move_to_end((target, ident.nonce))
            while len(self._accepted) > self.max_accepted:
                self._accepted.popitem(last=False)

    def on_rejected(self, target: str, reason: str, raw) -> RekeyDecision:
        if reason.startswith(("REPLAY_DUPLICATE_NONCE", "REPLAY_EXPIRED")):
            kind = "replay"      # reaching here means the Ed25519 signature already verified
        elif reason.startswith("INVALID_SIGNATURE"):
            kind = "tampering"
        else:
            return NO_ACTION
        ident = frame_identity(raw)
        if ident is None:
            return NO_ACTION
        with self._lock:
            now = self._clock()
            if kind == "replay":
                seen = self._accepted.get((target, ident.nonce))
                if seen is not None and seen != _digest(raw):
                    kind = "equivocation"
            sender = ident.sender_id
            if kind == "equivocation":
                return self._fire_locked(sender, ident.key_id, now,
                                         f"nonce {ident.nonce} signed twice with different content")
            if now < self._cooldown_until.get(sender, 0.0):
                return NO_ACTION
            ekey = (sender, ident.key_id, kind)
            events = self._events.setdefault(ekey, deque())
            self._events.move_to_end(ekey)
            events.append(now)
            while events and now - events[0] > self.window_sec:
                events.popleft()
            while len(self._events) > self.max_tracked:
                self._events.popitem(last=False)
            if len(events) >= self.thresholds[kind]:
                return self._fire_locked(sender, ident.key_id, now,
                                         f"{len(events)} {kind} events in {self.window_sec:.0f}s")
            return NO_ACTION

    def _fire_locked(self, sender: str, key_id: str, now: float, why: str) -> RekeyDecision:
        for k in [k for k in self._events if k[0] == sender]:
            del self._events[k]
        self._cooldown_until[sender] = now + self.cooldown_sec
        return RekeyDecision(True, sender, key_id, why)


# ------------------------------------------------------------------ key authority
@dataclass
class _SignerState:
    topics: list
    active_key_id: str
    generation: int
    born_wall: float


def sweep_stale_sessions(base_dir: Optional[Path] = None, max_age_sec: float = 6 * 3600.0) -> int:
    """Best-effort removal of session dirs left behind by crashed stacks."""
    base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
    removed = 0
    for entry in base.glob(SESSION_PREFIX + "*"):
        try:
            if entry.is_dir() and time.time() - entry.stat().st_mtime > max_age_sec:
                _wipe_session(entry)
                removed += 1
        except OSError:
            continue
    return removed


def _wipe_session(session_dir: Path) -> None:
    # Patch 04: wipe both Ed25519 signer keys and all mTLS private keys (session CA
    # plus immutable versioned leaf bundles) before removing the ephemeral session.
    private_roots = [session_dir / "signing", session_dir / "certificates"]
    for root in private_roots:
        if not root.is_dir():
            continue
        candidates = root.rglob("*.key") if root.name == "certificates" else root.iterdir()
        for f in candidates:
            try:
                if not f.is_file():
                    continue
                size = f.stat().st_size
                with open(f, "r+b") as handle:  # best effort; SSD wear-levelling may retain copies
                    handle.write(b"\0" * size)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                pass
    shutil.rmtree(session_dir, ignore_errors=True)


class KeyAuthority:
    def __init__(self, session_dir, *, max_rotations_per_hour: int = 6,
                 max_key_age_sec: float = 0.0, audit_path=None,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic):
        self.session_dir = Path(session_dir)
        self.max_rotations_per_hour = int(max_rotations_per_hour)
        self.max_key_age_sec = float(max_key_age_sec)
        self.audit_path = Path(audit_path) if audit_path else self.session_dir / "key_events.jsonl"
        self._clock, self._mono = clock, monotonic
        self._state: dict[str, _SignerState] = {}
        self._rotations: dict[str, deque] = {}
        self._suspended: set = set()
        self.ledger = RevocationLedger(self.revocation_path)
        # Patch 04 attaches the session-scoped X.509 authority in the Key Authority process.
        self.cert_authority = None

    # -- paths
    @property
    def trust_path(self) -> Path: return self.session_dir / TRUST_FILE
    @property
    def revocation_path(self) -> Path: return self.session_dir / REVOKED_FILE
    @property
    def keys_dir(self) -> Path: return self.session_dir / "keys"
    @property
    def signing_dir(self) -> Path: return self.session_dir / "signing"
    def private_key_path(self, signer_id: str) -> Path: return self.signing_dir / f"{signer_id}.key"

    # -- environments handed to child processes (paths only, never key material)
    def env_public(self) -> dict:
        return {"LUMEN_TRUSTED_COMMAND_SIGNERS": str(self.trust_path),
                "LUMEN_REVOKED_KEYS": str(self.revocation_path),
                "LUMEN_REKEY_ENABLED": "1"}

    def env_signer(self, signer_id: str) -> dict:
        return {"LUMEN_COMMAND_PRIVATE_KEY": str(self.private_key_path(signer_id)),
                "LUMEN_COMMAND_SIGNER_ID": signer_id}

    # -- construction
    @classmethod
    def create_session(cls, template_path, *, base_dir=None, signers=None, **kwargs) -> "KeyAuthority":
        """Mint generation-1 keys for each signer in the (public, committable) template.

        The template contributes only signer ids and allowed_topics. Its public keys are
        ignored: every session starts from fresh keys, so nothing static can be replayed."""
        template = json.loads(Path(template_path).read_text(encoding="utf-8"))
        template_signers = template["signers"]
        wanted = list(signers) if signers else list(template_signers)
        base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        session_dir = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX, dir=base))
        authority = cls(session_dir, **kwargs)
        try:
            restrict_to_owner(session_dir, directory=True)
            for sub in (authority.keys_dir, authority.signing_dir):
                sub.mkdir()
                restrict_to_owner(sub, directory=True)
            for signer_id in wanted:
                if not _ID_RE.fullmatch(signer_id):
                    raise ValueError(f"invalid signer id: {signer_id!r}")
                entry = template_signers[signer_id]
                topics = list(entry["keys"][entry["active_key_id"]]["allowed_topics"])
                key_id = authority._mint(signer_id, generation=1)
                authority._state[signer_id] = _SignerState(topics, key_id, 1, authority._clock())
            authority.ledger.ensure()
            authority._commit_trust_store()      # commit point: verifiers see keys only now
            authority._audit("session_provisioned", signers=sorted(authority._state))
        except BaseException:
            _wipe_session(session_dir)
            raise
        return authority

    @classmethod
    def adopt(cls, session_dir, **kwargs) -> "KeyAuthority":
        authority = cls(session_dir, **kwargs)
        doc = json.loads(authority.trust_path.read_text(encoding="utf-8"))
        for signer_id, entry in doc["signers"].items():
            active = entry["active_key_id"]
            authority._state[signer_id] = _SignerState(
                list(entry["keys"][active]["allowed_topics"]), active,
                int(entry.get("generation", 1)), authority._clock())
        return authority

    # -- minting / committing
    def _mint(self, signer_id: str, generation: int) -> str:
        key_id = f"g{generation}-{secrets.token_hex(4)}"
        private = Ed25519PrivateKey.generate()
        atomic_write_bytes(
            self.keys_dir / f"{key_id}.pub.pem",
            private.public_key().public_bytes(serialization.Encoding.PEM,
                                              serialization.PublicFormat.SubjectPublicKeyInfo))
        atomic_write_bytes(
            self.private_key_path(signer_id),
            private.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()),
            private=True)
        return key_id

    def _commit_trust_store(self) -> None:
        doc = {"version": 1, "signers": {
            sid: {"active_key_id": st.active_key_id, "generation": st.generation,
                  "keys": {st.active_key_id: {
                      "public_key": f"keys/{st.active_key_id}.pub.pem",
                      "allowed_topics": list(st.topics)}}}
            for sid, st in self._state.items()}}
        atomic_write_bytes(self.trust_path, json.dumps(doc, indent=2, sort_keys=True).encode("utf-8"))

    # -- rotation
    def rotate(self, signer_id: str, reason: str) -> str:
        st = self._state[signer_id]
        now = self._mono()
        window = self._rotations.setdefault(signer_id, deque())
        while window and now - window[0] > 3600.0:
            window.popleft()
        if len(window) >= self.max_rotations_per_hour:
            self._audit("rotation_suspended", signer=signer_id, reason=reason)
            raise RotationSuspended(f"{signer_id}: {len(window)} rotations in the last hour")
        old = st.active_key_id
        self.ledger.revoke(signer_id, old, reason)            # 1. old key dead everywhere
        generation = st.generation + 1
        new = self._mint(signer_id, generation)                # 2. pub file, then private file
        st.active_key_id, st.generation, st.born_wall = new, generation, self._clock()
        self._commit_trust_store()                             # 3. commit (atomic, last)
        window.append(now)
        try:
            (self.keys_dir / f"{old}.pub.pem").unlink()
        except OSError:
            pass
        self._audit("key_rotated", signer=signer_id, old_key_id=old, new_key_id=new, reason=reason)
        return new

    def attach_cert_authority(self, cert_authority) -> None:
        """Attach the X.509 authority sharing this exact ephemeral session."""
        if Path(cert_authority.session_dir).resolve() != self.session_dir.resolve():
            raise ValueError("CertAuthority must use the KeyAuthority session directory")
        self.cert_authority = cert_authority

    def poll(self) -> list:
        """Poll Ed25519 and mTLS lifecycles; a corrupt store aborts the whole poll."""
        events = []
        revoked = load_revocations(self.revocation_path)
        for signer_id, st in list(self._state.items()):
            reason = None
            if (signer_id, st.active_key_id) in revoked:
                reason = "incident"
            elif self.max_key_age_sec and self._clock() - st.born_wall > self.max_key_age_sec:
                reason = "scheduled"
            if reason is None or signer_id in self._suspended:
                continue
            try:
                new = self.rotate(signer_id, reason)
                events.append({"event": "rotated", "signer": signer_id, "reason": reason, "key_id": new})
            except RotationSuspended as exc:
                self._suspended.add(signer_id)               # stay locked down: fail closed
                events.append({"event": "suspended", "signer": signer_id, "detail": str(exc)})
        if self.cert_authority is not None:
            cert_events = self.cert_authority.poll()
            for event in cert_events:
                event["component"] = "certificate"
            events.extend(cert_events)
        return events

    # -- misc
    def status(self) -> dict:
        revoked = load_revocations(self.revocation_path)
        result = {sid: {"active_key_id": st.active_key_id, "generation": st.generation,
                        "active_revoked": (sid, st.active_key_id) in revoked,
                        "suspended": sid in self._suspended}
                  for sid, st in self._state.items()}
        if self.cert_authority is not None:
            # Reserved namespace keeps existing signer lookups backward-compatible.
            result["certificates"] = self.cert_authority.status()
        return result

    def _audit(self, event: str, **fields) -> None:
        record = {"ts": round(self._clock(), 3), "event": event, **fields}
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            log.exception("key audit log write failed")
        log.warning("[KEY LIFECYCLE] %s %s", event, {k: v for k, v in fields.items()})

    def cleanup(self) -> None:
        if self.cert_authority is not None:
            try:
                self.cert_authority.cleanup()
            finally:
                self.cert_authority = None
        _wipe_session(self.session_dir)
