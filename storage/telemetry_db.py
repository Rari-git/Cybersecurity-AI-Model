"""Lumen Edge Security v5.0 — Manager pentru cele 3 baze de date SQLite izolate.

Caracteristici:
- Izolare fizică: telemetry_normal.db, telemetry_incidents.db, security_control.db
- Single-writer thread cu coadă per DB (elimină SQLITE_BUSY în arhitectura multi-proces)
- Deschidere read-only (?mode=ro) pentru training pe telemetry_normal.db
- Jurnal append-only hash-chained în quarantine_log
- Tranzacții atomice pentru seen_nonces și enforcement_audit
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

GENESIS_HASH = "0" * 64
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class DBPaths:
    normal_db: Path
    incidents_db: Path
    control_db: Path

    @classmethod
    def from_dir(cls, base_dir: str | Path = "logs/db") -> "DBPaths":
        root = Path(base_dir).expanduser().resolve()
        return cls(
            normal_db=root / "telemetry_normal.db",
            incidents_db=root / "telemetry_incidents.db",
            control_db=root / "security_control.db",
        )


def _load_schema_sections(schema_file: Path = SCHEMA_PATH) -> dict[str, str]:
    raw = schema_file.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- [DB:") and stripped.endswith("]"):
            current = stripped[len("-- [DB:"):-1].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


class _SingleWriterWorker:
    """Thread unic de scriere cu batching pentru o bază de date SQLite WAL."""

    def __init__(self, db_path: Path, schema_sql: str, max_queue: int = 8192):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_sql = schema_sql
        self._queue: Queue[Optional[tuple[Callable[[sqlite3.Connection], Any], Optional[threading.Event], list[Any]]]] = Queue(
            maxsize=max_queue
        )
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"lumen-db-writer-{db_path.stem}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=10.0)
        if self._init_error is not None:
            raise RuntimeError(f"Failed to initialize DB {db_path}") from self._init_error

    def _run(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0, isolation_level=None)
            conn.executescript(self._schema_sql)
            self._ready.set()

            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break

                batch = [item]
                # Grupăm până la 64 de operații asincrone într-o singură tranzacție
                if item[1] is None:
                    while len(batch) < 64:
                        try:
                            nxt = self._queue.get_nowait()
                        except Empty:
                            break
                        if nxt is None:
                            self._queue.task_done()
                            self._execute_batch(conn, batch)
                            return
                        batch.append(nxt)
                        if nxt[1] is not None:
                            break

                self._execute_batch(conn, batch)
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
        finally:
            if conn is not None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except sqlite3.Error:
                    pass
                conn.close()

    def _execute_batch(self, conn: sqlite3.Connection, batch: list) -> None:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            for fn, done_evt, holder in batch:
                try:
                    res = fn(conn)
                    if holder is not None:
                        holder.append(res)
                except BaseException as exc:
                    if holder is not None:
                        holder.append(exc)
                    else:
                        raise
            conn.execute("COMMIT;")
        except BaseException:
            conn.execute("ROLLBACK;")
            raise
        finally:
            for _, done_evt, _ in batch:
                if done_evt is not None:
                    done_evt.set()
                self._queue.task_done()

    def submit_async(self, fn: Callable[[sqlite3.Connection], Any]) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait((fn, None, None))
            return True
        except Full:
            return False

    def submit_sync(self, fn: Callable[[sqlite3.Connection], Any], timeout: float = 5.0) -> Any:
        if self._closed:
            raise RuntimeError(f"Writer closed for {self.db_path}")
        done = threading.Event()
        holder: list[Any] = []
        self._queue.put((fn, done, holder), timeout=timeout)
        if not done.wait(timeout=timeout):
            raise TimeoutError(f"DB sync write timed out for {self.db_path}")
        if holder and isinstance(holder[0], BaseException):
            raise holder[0]
        return holder[0] if holder else None

    def flush(self, timeout: float = 5.0) -> None:
        self.submit_sync(lambda _c: None, timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)


class LumenStorageManager:
    """Fațadă unificată pentru cele 3 baze de date SQLite izolate."""

    def __init__(self, paths: Optional[DBPaths] = None, enable_writers: bool = True):
        self.paths = paths or DBPaths.from_dir()
        sections = _load_schema_sections()
        self._writers: dict[str, _SingleWriterWorker] = {}
        if enable_writers:
            self._writers["normal"] = _SingleWriterWorker(self.paths.normal_db, sections["telemetry_normal"])
            self._writers["incidents"] = _SingleWriterWorker(self.paths.incidents_db, sections["telemetry_incidents"])
            self._writers["control"] = _SingleWriterWorker(self.paths.control_db, sections["security_control"])

    def connect_readonly(self, db_name: str) -> sqlite3.Connection:
        """Deschide o conexiune strict read-only (?mode=ro) pentru training sau HMI."""
        path_map = {
            "normal": self.paths.normal_db,
            "incidents": self.paths.incidents_db,
            "control": self.paths.control_db,
        }
        if db_name not in path_map:
            raise ValueError(f"Unknown DB name: {db_name}")
        target = path_map[db_name].resolve()
        uri = f"{target.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON;")
        conn.execute("PRAGMA busy_timeout = 3000;")
        return conn

    # ---- Baseline Sessions & Normal Telemetry (telemetry_normal.db) ----

    def register_baseline_session(
        self,
        session_id: str,
        source_type: str,
        config: dict[str, Any],
        git_hash: str = "v5.0-dev",
        approved_for_training: bool = False,
    ) -> None:
        now_ns = time.time_ns()
        cfg_json = json.dumps(config, sort_keys=True)

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO baseline_sessions
                (session_id, started_at_ns, source_type, git_hash, config_json, approved_for_training)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, now_ns, source_type, git_hash, cfg_json, 1 if approved_for_training else 0),
            )

        self._writers["normal"].submit_sync(_op)

    def finalize_baseline_session(self, session_id: str, approved_for_training: bool = True) -> None:
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                UPDATE baseline_sessions
                SET ended_at_ns = ?, approved_for_training = ?
                WHERE session_id = ?
                """,
                (now_ns, 1 if approved_for_training else 0, session_id),
            )

        self._writers["normal"].submit_sync(_op)

    def enqueue_normal_raw_event(self, row: tuple[Any, ...]) -> bool:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO raw_events (
                    session_id, device_id, device_cn, generation, cert_fingerprint,
                    topic, gateway_ts_ns, device_ts_ns, payload_len, payload_sha256,
                    numeric_value, raw_payload, source_type, admission_policy_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

        return self._writers["normal"].submit_async(_op)

    def enqueue_normal_window(self, row: tuple[Any, ...]) -> bool:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO windows (
                    session_id, device_id, device_cn, generation, topic,
                    ts_start_ns, ts_end_ns, seq_len, feature_dim, features_blob,
                    feature_schema_version, source_type, source_manifest_hash, admitted_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

        return self._writers["normal"].submit_async(_op)

    # ---- Incident & Forensic Sink Captures (telemetry_incidents.db) ----

    def enqueue_incident_capture(self, row: tuple[Any, ...]) -> bool:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO incident_captures (
                    session_id, device_id, device_cn, generation, cert_fingerprint,
                    original_topic, shadow_topic, gateway_ts_ns, device_ts_ns,
                    capture_source, reason_code, risk_score, xai_contrib_json,
                    payload_len, payload_sha256, raw_payload, policy_version,
                    ground_truth_label, scenario_id, retention_until_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

        return self._writers["incidents"].submit_async(_op)

    # ---- Security Control, Nonces, Hash-Chained Quarantine & PEP Audit ----

    def upsert_device(
        self,
        device_id: str,
        generation: int,
        device_cn: str,
        cert_fingerprint: bytes,
        device_role: str,
        criticality: str = "normal",
        state: str = "ACTIVE",
    ) -> None:
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO device_registry (
                    device_id, generation, device_cn, cert_fingerprint,
                    device_role, criticality, current_state, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id, generation) DO UPDATE SET
                    device_cn = excluded.device_cn,
                    cert_fingerprint = excluded.cert_fingerprint,
                    device_role = excluded.device_role,
                    criticality = excluded.criticality,
                    current_state = excluded.current_state,
                    updated_at_ns = excluded.updated_at_ns
                """,
                (device_id, generation, device_cn, cert_fingerprint, device_role, criticality, state, now_ns),
            )

        self._writers["control"].submit_sync(_op)

    def consume_nonce_atomic(self, device_id: str, generation: int, nonce: str) -> bool:
        """Consumă atomic un nonce. Returnează False dacă este replay (deja văzut)."""
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO seen_nonces (device_id, generation, nonce, first_seen_ns)
                VALUES (?, ?, ?, ?)
                """,
                (device_id, generation, nonce, now_ns),
            )
            return cur.rowcount == 1

        return bool(self._writers["control"].submit_sync(_op))

    def append_quarantine_transition(
        self,
        device_id: str,
        generation: int,
        from_state: str,
        to_state: str,
        reason_code: str,
        risk_score: Optional[float],
        policy_version: int,
        operator_id: Optional[str] = None,
    ) -> str:
        """Adaugă o tranziție de stare în jurnalul hash-chained și actualizează device_registry."""
        ts_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT entry_hash FROM quarantine_log ORDER BY id DESC LIMIT 1").fetchone()
            prev_hash = row[0] if row else GENESIS_HASH
            canonical = json.dumps(
                {
                    "ts_ns": ts_ns,
                    "device_id": device_id,
                    "generation": generation,
                    "from_state": from_state,
                    "to_state": to_state,
                    "reason_code": reason_code,
                    "risk_score": risk_score,
                    "policy_version": policy_version,
                    "operator_id": operator_id,
                    "prev_hash": prev_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            entry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO quarantine_log (
                    ts_ns, device_id, generation, from_state, to_state,
                    reason_code, risk_score, policy_version, operator_id, prev_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_ns, device_id, generation, from_state, to_state,
                    reason_code, risk_score, policy_version, operator_id, prev_hash, entry_hash,
                ),
            )
            conn.execute(
                """
                UPDATE device_registry
                SET current_state = ?, updated_at_ns = ?
                WHERE device_id = ? AND generation = ?
                """,
                (to_state, ts_ns, device_id, generation),
            )
            return entry_hash

        return str(self._writers["control"].submit_sync(_op))

    def record_enforcement_decision(
        self,
        decision_id: str,
        device_id: str,
        generation: int,
        old_state: str,
        target_state: str,
        reason_code: str,
        policy_version: int,
    ) -> int:
        decided_at_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> int:
            conn.execute(
                """
                INSERT INTO enforcement_audit (
                    decision_id, device_id, generation, old_state, target_state,
                    reason_code, policy_version, decided_at_ns, command_sent_at_ns, pep_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """,
                (
                    decision_id, device_id, generation, old_state, target_state,
                    reason_code, policy_version, decided_at_ns, decided_at_ns,
                ),
            )
            return decided_at_ns

        return int(self._writers["control"].submit_sync(_op))

    def confirm_enforcement_pep(
        self,
        decision_id: str,
        confirmed: bool,
        detail: str = "",
    ) -> Optional[float]:
        """Confirmă aplicarea efectivă de către PEP și calculează latența decizie -> efect."""
        confirmed_at_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> Optional[float]:
            row = conn.execute(
                "SELECT decided_at_ns FROM enforcement_audit WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if not row:
                return None
            latency_ms = max(0.0, (confirmed_at_ns - int(row[0])) / 1_000_000.0)
            status = "CONFIRMED" if confirmed else "FAILED"
            conn.execute(
                """
                UPDATE enforcement_audit
                SET pep_confirmed_at_ns = ?, pep_status = ?, enforcement_latency_ms = ?, detail = ?
                WHERE decision_id = ?
                """,
                (confirmed_at_ns, status, round(latency_ms, 3), detail, decision_id),
            )
            return latency_ms

        return self._writers["control"].submit_sync(_op)

    def verify_quarantine_chain(self) -> tuple[bool, int]:
        """Verifică integritatea criptografică a jurnalului hash-chained."""
        conn = self.connect_readonly("control")
        try:
            rows = conn.execute("SELECT * FROM quarantine_log ORDER BY id ASC").fetchall()
            expected_prev = GENESIS_HASH
            for r in rows:
                if r["prev_hash"] != expected_prev:
                    return False, int(r["id"])
                canonical = json.dumps(
                    {
                        "ts_ns": r["ts_ns"],
                        "device_id": r["device_id"],
                        "generation": r["generation"],
                        "from_state": r["from_state"],
                        "to_state": r["to_state"],
                        "reason_code": r["reason_code"],
                        "risk_score": r["risk_score"],
                        "policy_version": r["policy_version"],
                        "operator_id": r["operator_id"],
                        "prev_hash": r["prev_hash"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                calc = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if calc != r["entry_hash"]:
                    return False, int(r["id"])
                expected_prev = calc
            return True, len(rows)
        finally:
            conn.close()

    def flush_all(self) -> None:
        for w in self._writers.values():
            w.flush()

    def close(self) -> None:
        for w in self._writers.values():
            w.close()