"""Durable atomic anti-replay counters and outbound sequence allocation."""
from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_COMPONENT_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_MAX_NONCE = 2**63 - 1


class ReplayStoreError(RuntimeError):
    pass


class _SQLiteNonceDB:
    def __init__(self, path: str | os.PathLike[str]):
        if str(path) == ":memory:":
            raise ValueError("replay state must be stored on disk, not in memory")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except Exception as exc:
            raise ReplayStoreError(f"Cannot initialize nonce database {self.path}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS replay_state (
                sender_id TEXT NOT NULL, topic TEXT NOT NULL,
                last_nonce INTEGER NOT NULL CHECK(last_nonce > 0),
                updated_at REAL NOT NULL,
                PRIMARY KEY(sender_id, topic)
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS outbound_sequence (
                sender_id TEXT PRIMARY KEY,
                last_nonce INTEGER NOT NULL CHECK(last_nonce > 0),
                updated_at REAL NOT NULL
            )""")
            result = conn.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise sqlite3.DatabaseError(f"SQLite quick_check failed: {result!r}")
        finally:
            conn.close()
        if os.name == "posix":
            try:
                os.chmod(self.path, 0o600)
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(str(self.path) + suffix)
                    if sidecar.exists():
                        os.chmod(sidecar, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        # Connections are opened/closed per operation. Kept for uniform lifecycle APIs.
        return None

    @staticmethod
    def _validate_identity(sender_id: str, topic: str) -> None:
        if not isinstance(sender_id, str) or not _ID_RE.fullmatch(sender_id):
            raise ValueError("sender_id must be 1-64 safe characters")
        if not isinstance(topic, str) or not topic or len(topic) > 256:
            raise ValueError("topic is invalid")


class ReplayStore(_SQLiteNonceDB):
    """Atomically consumes strictly increasing nonces scoped by signer and topic."""

    @classmethod
    def for_component(cls, component: str, path: Optional[str] = None) -> "ReplayStore":
        if not isinstance(component, str) or not _COMPONENT_RE.fullmatch(component):
            raise ValueError("component must be a simple lowercase identifier")
        env_path = os.environ.get(f"LUMEN_{component.upper()}_REPLAY_DB")
        return cls(path or env_path or f"state/{component}_replay.sqlite3")

    def get_last_nonce(self, sender_id: str, topic: str) -> int:
        self._validate_identity(sender_id, topic)
        conn = None
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT last_nonce FROM replay_state WHERE sender_id=? AND topic=?",
                (sender_id, topic),
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            raise ReplayStoreError(f"Cannot read replay state: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    def consume(self, sender_id: str, topic: str, nonce: int) -> tuple[bool, int]:
        self._validate_identity(sender_id, topic)
        if isinstance(nonce, bool) or not isinstance(nonce, int) or not 0 < nonce <= _MAX_NONCE:
            raise ValueError("nonce must be a positive signed 64-bit integer")
        conn = None
        try:
            conn = self._connect()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT last_nonce FROM replay_state WHERE sender_id=? AND topic=?",
                (sender_id, topic),
            ).fetchone()
            previous = int(row[0]) if row else 0
            if nonce <= previous:
                conn.execute("ROLLBACK")
                return False, previous
            conn.execute(
                "INSERT INTO replay_state(sender_id,topic,last_nonce,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(sender_id,topic) DO UPDATE SET "
                "last_nonce=excluded.last_nonce,updated_at=excluded.updated_at",
                (sender_id, topic, nonce, time.time()),
            )
            conn.execute("COMMIT")
            return True, previous
        except Exception as exc:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise ReplayStoreError(f"Replay state update failed: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()


class SequenceStore(_SQLiteNonceDB):
    """Persistently allocates monotonically increasing nonces for command signers."""

    @classmethod
    def for_sender(cls, sender_id: str, path: Optional[str] = None) -> "SequenceStore":
        if not isinstance(sender_id, str) or not _ID_RE.fullmatch(sender_id):
            raise ValueError("sender_id must be 1-64 safe characters")
        return cls(path or os.environ.get("LUMEN_SEQUENCE_DB") or f"state/{sender_id}_sequence.sqlite3")

    def next_nonce(self, sender_id: str) -> int:
        if not isinstance(sender_id, str) or not _ID_RE.fullmatch(sender_id):
            raise ValueError("sender_id must be 1-64 safe characters")
        conn = None
        try:
            conn = self._connect()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT last_nonce FROM outbound_sequence WHERE sender_id=?", (sender_id,)
            ).fetchone()
            previous = int(row[0]) if row else 0
            nonce = max(previous + 1, time.time_ns() // 1_000_000)
            if nonce > _MAX_NONCE:
                raise OverflowError("outbound nonce exhausted signed 64-bit range")
            conn.execute(
                "INSERT INTO outbound_sequence(sender_id,last_nonce,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(sender_id) DO UPDATE SET "
                "last_nonce=excluded.last_nonce,updated_at=excluded.updated_at",
                (sender_id, nonce, time.time()),
            )
            conn.execute("COMMIT")
            return nonce
        except Exception as exc:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise ReplayStoreError(f"Cannot allocate outbound nonce: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()
