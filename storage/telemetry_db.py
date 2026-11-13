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


class FreshnessRejected(ValueError):
    """A V6 sensor frame failed the durable boot/sequence admission checks."""

    def __init__(self, reason_code: str, detail: str = ""):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{': ' + detail if detail else ''}")


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
        self._fatal_error: Optional[BaseException] = None
        self._last_batch_error: Optional[BaseException] = None
        self._state_lock = threading.Lock()
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
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.executescript(self._schema_sql)
            self._ensure_schema_migrations(conn)
            self._ready.set()

            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break

                batch = [item]
                # Group up to 64 operations. A synchronous item closes the batch
                # so its caller never waits behind more asynchronous submissions.
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

                try:
                    self._execute_batch(conn, batch)
                    with self._state_lock:
                        self._last_batch_error = None
                except BaseException as exc:
                    # A failed async operation aborts the entire transaction and
                    # is delivered to every synchronous caller in that batch.
                    # Keep the worker alive for a later independent batch.
                    with self._state_lock:
                        self._last_batch_error = exc
        except BaseException as exc:
            with self._state_lock:
                self._fatal_error = exc
            self._init_error = self._init_error or exc
            self._ready.set()
            self._fail_queued_callers(exc)
        finally:
            if conn is not None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except sqlite3.Error:
                    pass
                conn.close()

    def _ensure_schema_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply additive migrations for existing V5 databases without data loss."""
        if self.db_path.name == "security_control.db":
            # ``executescript`` already creates these for fresh DBs. Keep the
            # CREATE IF NOT EXISTS calls here as an explicit additive migration
            # for pre-V6 control databases opened with an older schema bundle.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS policy_dimensions (
                    device_id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL CHECK(generation > 0),
                    security_phase TEXT NOT NULL,
                    security_status TEXT NOT NULL CHECK(security_status IN ('ACTIVE','SUSPECT','QUARANTINED','REVOKED')),
                    operational_status TEXT NOT NULL CHECK(operational_status IN ('HEALTHY','DEGRADED','FAULT')),
                    security_anomaly_history_json TEXT NOT NULL DEFAULT '[]',
                    security_clean_streak INTEGER NOT NULL DEFAULT 0 CHECK(security_clean_streak >= 0),
                    operational_clean_streak INTEGER NOT NULL DEFAULT 0 CHECK(operational_clean_streak >= 0),
                    security_transition_history_json TEXT NOT NULL DEFAULT '[]',
                    operational_transition_history_json TEXT NOT NULL DEFAULT '[]',
                    updated_at_ns INTEGER NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS policy_dimension_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ns INTEGER NOT NULL,
                    device_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    dimension TEXT NOT NULL CHECK(dimension IN ('operational','security')),
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_policy_dimension_transitions_device "
                "ON policy_dimension_transitions(device_id, dimension, transition_id)"
            )
            return
        if self.db_path.name == "telemetry_incidents.db":
            # Additive migration for existing incident databases.
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS adversarial_regression_corpus (
                    capture_id INTEGER PRIMARY KEY REFERENCES incident_captures(capture_id) ON DELETE CASCADE,
                    corpus_class TEXT NOT NULL CHECK(corpus_class IN ('adversarial','hardware_fault','protocol_violation')),
                    indexed_at_ns INTEGER NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS trg_incident_to_regression_corpus
                AFTER INSERT ON incident_captures
                BEGIN
                  INSERT OR IGNORE INTO adversarial_regression_corpus(capture_id,corpus_class,indexed_at_ns)
                  VALUES (NEW.capture_id,
                    CASE WHEN NEW.capture_source='fault_lab' OR lower(coalesce(NEW.ground_truth_label,'')) LIKE '%fault%' THEN 'hardware_fault'
                         WHEN lower(NEW.reason_code) LIKE '%replay%' OR lower(NEW.reason_code) LIKE '%sequence%'
                           OR lower(NEW.reason_code) LIKE '%freshness%' OR lower(NEW.reason_code) LIKE '%boot%'
                           OR lower(NEW.reason_code) LIKE '%protocol%' THEN 'protocol_violation'
                         ELSE 'adversarial' END,
                    CAST(strftime('%s','now') AS INTEGER)*1000000000);
                END;
                INSERT OR IGNORE INTO adversarial_regression_corpus(capture_id,corpus_class,indexed_at_ns)
                SELECT capture_id,
                  CASE WHEN capture_source='fault_lab' OR lower(coalesce(ground_truth_label,'')) LIKE '%fault%' THEN 'hardware_fault'
                       WHEN lower(reason_code) LIKE '%replay%' OR lower(reason_code) LIKE '%sequence%'
                         OR lower(reason_code) LIKE '%freshness%' OR lower(reason_code) LIKE '%boot%'
                         OR lower(reason_code) LIKE '%protocol%' THEN 'protocol_violation'
                       ELSE 'adversarial' END,
                  CAST(strftime('%s','now') AS INTEGER)*1000000000
                FROM incident_captures;
                """
            )
            return
        if self.db_path.name != "telemetry_normal.db":
            return
        columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_events)").fetchall()}
        additions = {
            "protocol_version": "INTEGER NOT NULL DEFAULT 5 CHECK(protocol_version IN (5, 6))",
            "boot_id": "TEXT",
            "sequence": "INTEGER",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE raw_events ADD COLUMN {name} {declaration}")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_v6_raw_device_boot_sequence "
            "ON raw_events(device_id, generation, boot_id, sequence) "
            "WHERE boot_id IS NOT NULL AND sequence IS NOT NULL"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS device_freshness (
                device_id TEXT PRIMARY KEY,
                certificate_generation INTEGER NOT NULL CHECK(certificate_generation > 0),
                active_boot_id TEXT NOT NULL CHECK(length(active_boot_id) = 32),
                last_sequence INTEGER NOT NULL DEFAULT 0 CHECK(last_sequence >= 0),
                last_device_timestamp_ns INTEGER,
                updated_at_ns INTEGER NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS session_curation (
                session_id TEXT PRIMARY KEY REFERENCES baseline_sessions(session_id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK(status IN ('pending','approved_clean_baseline','rejected')),
                decision_json TEXT NOT NULL,
                dataset_sha256 TEXT,
                curated_at_ns INTEGER NOT NULL
            )"""
        )
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS trg_baseline_session_insert_approval_guard
            BEFORE INSERT ON baseline_sessions WHEN NEW.approved_for_training = 1
            BEGIN SELECT RAISE(ABORT, 'training approval requires approved_clean_baseline curation'); END;
            CREATE TRIGGER IF NOT EXISTS trg_baseline_session_update_approval_guard
            BEFORE UPDATE OF approved_for_training ON baseline_sessions
            WHEN NEW.approved_for_training = 1 AND (
                NEW.source_type <> 'baseline_lab' OR NOT EXISTS (
                    SELECT 1 FROM session_curation c WHERE c.session_id=NEW.session_id
                    AND c.status='approved_clean_baseline'
                )
            )
            BEGIN SELECT RAISE(ABORT, 'training approval requires approved_clean_baseline curation'); END;
            CREATE TRIGGER IF NOT EXISTS trg_live_session_training_guard
            BEFORE UPDATE OF source_type ON baseline_sessions
            WHEN NEW.source_type='live_validated' AND NEW.approved_for_training <> 0
            BEGIN SELECT RAISE(ABORT, 'live_validated sessions cannot be approved for training'); END;
            """
        )
        # Existing V5 rows were approved by a weaker boolean-only mechanism.
        # Revoke those approvals unless a V6 curation record exists.
        conn.execute(
            """UPDATE baseline_sessions SET approved_for_training=0
               WHERE approved_for_training=1 AND (
                   source_type <> 'baseline_lab' OR NOT EXISTS (
                       SELECT 1 FROM session_curation c WHERE c.session_id=baseline_sessions.session_id
                       AND c.status='approved_clean_baseline'
                   )
               )"""
        )
        return

    def _fail_queued_callers(self, error: BaseException) -> None:
        """Wake every queued synchronous submitter if the worker dies."""
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            try:
                if item is not None:
                    _fn, done_evt, holder = item
                    if holder is not None:
                        holder[:] = [error]
                    if done_evt is not None:
                        done_evt.set()
            finally:
                self._queue.task_done()

    def _execute_batch(self, conn: sqlite3.Connection, batch: list) -> None:
        """Commit a whole batch or fail all synchronous callers in that batch."""
        results: list[tuple[Optional[threading.Event], list[Any], Any]] = []
        failure: BaseException | None = None
        try:
            conn.execute("BEGIN IMMEDIATE;")
            for fn, done_evt, holder in batch:
                result = fn(conn)
                results.append((done_evt, holder, result))
            conn.execute("COMMIT;")
        except BaseException as exc:
            failure = exc
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
            except sqlite3.Error:
                pass
        finally:
            if failure is None:
                for done_evt, holder, result in results:
                    if holder is not None:
                        holder[:] = [result]
            else:
                # Do not report success for a synchronous operation whose
                # earlier/later batch peer caused the transaction to roll back.
                for _fn, done_evt, holder in batch:
                    if holder is not None:
                        holder[:] = [failure]
            for _fn, done_evt, _holder in batch:
                if done_evt is not None:
                    done_evt.set()
                self._queue.task_done()
        if failure is not None:
            raise failure

    def submit_async(self, fn: Callable[[sqlite3.Connection], Any]) -> bool:
        if self._closed or self._fatal_error is not None or not self._thread.is_alive():
            return False
        try:
            self._queue.put_nowait((fn, None, None))
            return True
        except Full:
            return False

    def submit_sync(self, fn: Callable[[sqlite3.Connection], Any], timeout: float = 5.0) -> Any:
        if self._closed:
            raise RuntimeError(f"Writer closed for {self.db_path}")
        if self._fatal_error is not None or not self._thread.is_alive():
            cause = self._fatal_error or RuntimeError("writer thread is not running")
            raise RuntimeError(f"Writer unavailable for {self.db_path}") from cause
        done = threading.Event()
        holder: list[Any] = []
        self._queue.put((fn, done, holder), timeout=timeout)
        if not done.wait(timeout=timeout):
            if self._fatal_error is not None:
                raise RuntimeError(f"Writer failed for {self.db_path}") from self._fatal_error
            raise TimeoutError(f"DB sync write timed out for {self.db_path}")
        if holder and isinstance(holder[0], BaseException):
            raise holder[0]
        return holder[0] if holder else None

    @property
    def last_batch_error(self) -> Optional[BaseException]:
        with self._state_lock:
            return self._last_batch_error

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
        if source_type not in {"baseline_lab", "live_validated"}:
            raise ValueError("unsupported session source_type")
        if approved_for_training:
            if source_type == "live_validated":
                raise ValueError("live_validated sessions cannot be approved for training")
            raise ValueError(
                "sessions must pass training.dataset_governance.curate_clean_sessions before approval"
            )
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

    def finalize_baseline_session(self, session_id: str, approved_for_training: bool = False) -> None:
        """Close a session without granting training approval.

        The legacy flag is accepted for call compatibility but deliberately
        ignored; only the Stage B deterministic curation gate can approve data.
        """
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE baseline_sessions SET ended_at_ns=?, approved_for_training=0 WHERE session_id=?",
                (now_ns, session_id),
            )

        self._writers["normal"].submit_sync(_op)

    def mark_session_curated(
        self,
        session_id: str,
        status: str,
        decision: dict[str, Any],
        dataset_sha256: str | None = None,
    ) -> None:
        """Persist a curation decision and its approval atomically."""
        if status not in {"approved_clean_baseline", "rejected", "pending"}:
            raise ValueError("invalid session curation status")
        decision_json = json.dumps(decision, sort_keys=True, separators=(",", ":"), allow_nan=False)
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT source_type, ended_at_ns FROM baseline_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown baseline session: {session_id}")
            if status == "approved_clean_baseline" and (row[0] != "baseline_lab" or row[1] is None):
                raise ValueError("only completed baseline_lab sessions can be approved")
            conn.execute(
                """INSERT INTO session_curation(session_id,status,decision_json,dataset_sha256,curated_at_ns)
                   VALUES(?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET
                   status=excluded.status, decision_json=excluded.decision_json,
                   dataset_sha256=excluded.dataset_sha256, curated_at_ns=excluded.curated_at_ns""",
                (session_id, status, decision_json, dataset_sha256, now_ns),
            )
            conn.execute(
                "UPDATE baseline_sessions SET approved_for_training=? WHERE session_id=?",
                (1 if status == "approved_clean_baseline" else 0, session_id),
            )

        self._writers["normal"].submit_sync(_op, timeout=10.0)

    def load_adversarial_regression_corpus(
        self, corpus_classes: tuple[str, ...] = ("adversarial", "protocol_violation")
    ) -> list[dict[str, Any]]:
        allowed = {"adversarial", "hardware_fault", "protocol_violation"}
        if not corpus_classes or not set(corpus_classes).issubset(allowed):
            raise ValueError("invalid adversarial corpus classes")
        conn = self.connect_readonly("incidents")
        try:
            placeholders = ",".join("?" for _ in corpus_classes)
            rows = conn.execute(
                f"""SELECT c.*, a.corpus_class FROM adversarial_regression_corpus a
                    JOIN incident_captures c ON c.capture_id=a.capture_id
                    WHERE a.corpus_class IN ({placeholders}) ORDER BY c.capture_id""",
                tuple(corpus_classes),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

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

    def register_verified_device_boot(
        self, device_id: str, certificate_generation: int, boot_id: str,
    ) -> None:
        """Activate a boot epoch after the caller has verified its boot challenge.

        This method is the persistence half of the boot-registration protocol.
        Callers MUST authenticate a fresh, single-use Gateway challenge with the
        active device certificate before invoking it. Re-registering the same
        epoch is idempotent; a new epoch resets its sequence to zero.
        """
        from utils.sensor_payload import normalize_boot_id

        if not isinstance(device_id, str) or not device_id or len(device_id) > 64:
            raise ValueError("invalid device_id")
        if (isinstance(certificate_generation, bool) or not isinstance(certificate_generation, int)
                or certificate_generation < 1):
            raise ValueError("certificate_generation must be a positive integer")
        normalized_boot_id = normalize_boot_id(boot_id)
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            current = conn.execute(
                "SELECT certificate_generation, active_boot_id FROM device_freshness WHERE device_id=?",
                (device_id,),
            ).fetchone()
            if current is not None:
                current_generation = int(current[0])
                if certificate_generation < current_generation:
                    raise FreshnessRejected("STALE_CERTIFICATE_GENERATION")
                if (certificate_generation == current_generation
                        and str(current[1]) == normalized_boot_id):
                    return
            conn.execute(
                """INSERT INTO device_freshness
                   (device_id, certificate_generation, active_boot_id, last_sequence,
                    last_device_timestamp_ns, updated_at_ns)
                   VALUES (?, ?, ?, 0, NULL, ?)
                   ON CONFLICT(device_id) DO UPDATE SET
                     certificate_generation=excluded.certificate_generation,
                     active_boot_id=excluded.active_boot_id,
                     last_sequence=0,
                     last_device_timestamp_ns=NULL,
                     updated_at_ns=excluded.updated_at_ns""",
                (device_id, certificate_generation, normalized_boot_id, now_ns),
            )

        self._writers["normal"].submit_sync(_op, timeout=10.0)

    def persist_v6_normal_event(
        self,
        row: tuple[Any, ...],
        *,
        device_id: str,
        certificate_generation: int,
        boot_id: str,
        sequence: int,
        device_timestamp_ns: int,
    ) -> int:
        """Atomically check freshness, persist a normal event, and advance sequence.

        The writer wraps this callable in ``BEGIN IMMEDIATE``/``COMMIT``. Any
        freshness or INSERT failure rolls the whole transaction back, so the
        counter cannot advance without its matching telemetry row (or vice versa).
        ``row`` contains the existing 14 V5 raw_events values, followed by
        protocol_version, boot_id and sequence.
        """
        from utils.sensor_payload import normalize_boot_id, MAX_SEQUENCE, MAX_DEVICE_TIMESTAMP_NS

        normalized_boot_id = normalize_boot_id(boot_id)
        if isinstance(certificate_generation, bool) or not isinstance(certificate_generation, int) or certificate_generation < 1:
            raise ValueError("invalid certificate generation")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= MAX_SEQUENCE:
            raise ValueError("invalid sequence")
        if isinstance(device_timestamp_ns, bool) or not isinstance(device_timestamp_ns, int) or not 1 <= device_timestamp_ns <= MAX_DEVICE_TIMESTAMP_NS:
            raise ValueError("invalid device timestamp")
        if len(row) != 17:
            raise ValueError("V6 raw event row must contain 17 values")
        if (row[1] != device_id or row[3] != certificate_generation
                or row[7] != device_timestamp_ns or row[14] != 6
                or row[15] != normalized_boot_id or row[16] != sequence):
            raise ValueError("V6 raw event row metadata does not match freshness arguments")
        now_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> int:
            state = conn.execute(
                """SELECT certificate_generation, active_boot_id, last_sequence
                   FROM device_freshness WHERE device_id=?""",
                (device_id,),
            ).fetchone()
            if state is None:
                raise FreshnessRejected("BOOT_NOT_REGISTERED")
            if int(state[0]) != certificate_generation:
                raise FreshnessRejected("CERTIFICATE_GENERATION_MISMATCH")
            if str(state[1]) != normalized_boot_id:
                raise FreshnessRejected("STALE_BOOT_ID")
            previous_sequence = int(state[2])
            if sequence <= previous_sequence:
                raise FreshnessRejected("REPLAY_OR_OUT_OF_ORDER_SEQUENCE")

            conn.execute(
                """INSERT INTO raw_events (
                    session_id, device_id, device_cn, generation, cert_fingerprint,
                    topic, gateway_ts_ns, device_ts_ns, payload_len, payload_sha256,
                    numeric_value, raw_payload, source_type, admission_policy_version,
                    protocol_version, boot_id, sequence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
            conn.execute(
                """UPDATE device_freshness SET last_sequence=?, last_device_timestamp_ns=?,
                   updated_at_ns=? WHERE device_id=?""",
                (sequence, device_timestamp_ns, now_ns, device_id),
            )
            return previous_sequence

        return int(self._writers["normal"].submit_sync(_op, timeout=10.0))

    def load_device_freshness(self, device_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Read durable active boot/sequence checkpoints for diagnostics and recovery."""
        conn = self.connect_readonly("normal")
        try:
            if device_id is None:
                rows = conn.execute(
                    "SELECT device_id, certificate_generation, active_boot_id, last_sequence, "
                    "last_device_timestamp_ns FROM device_freshness"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT device_id, certificate_generation, active_boot_id, last_sequence, "
                    "last_device_timestamp_ns FROM device_freshness WHERE device_id=?",
                    (device_id,),
                ).fetchall()
            return {
                str(row["device_id"]): {
                    "certificate_generation": int(row["certificate_generation"]),
                    "active_boot_id": str(row["active_boot_id"]),
                    "last_sequence": int(row["last_sequence"]),
                    "last_device_timestamp_ns": (int(row["last_device_timestamp_ns"])
                                                   if row["last_device_timestamp_ns"] is not None else None),
                }
                for row in rows
            }
        finally:
            conn.close()

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
                    current_state = CASE
                        WHEN device_registry.current_state <> 'ACTIVE' THEN device_registry.current_state
                        ELSE excluded.current_state
                    END,
                    updated_at_ns = excluded.updated_at_ns
                """,
                (device_id, generation, device_cn, cert_fingerprint, device_role, criticality, state, now_ns),
            )

        self._writers["control"].submit_sync(_op)

    def load_device_state_rows(self) -> dict[str, dict[str, Any]]:
        """Return the latest generation and authoritative state per stable identity."""
        conn = self.connect_readonly("control")
        try:
            rows = conn.execute(
                "SELECT device_id, generation, current_state FROM device_registry ORDER BY generation ASC"
            ).fetchall()
            return {
                str(row["device_id"]): {
                    "generation": int(row["generation"]),
                    "current_state": str(row["current_state"]),
                }
                for row in rows
            }
        finally:
            conn.close()

    def load_device_states(self) -> dict[str, str]:
        """Return the latest persisted registry state for each stable device identity."""
        return {device_id: row["current_state"] for device_id, row in self.load_device_state_rows().items()}

    def save_policy_state(
        self, device_id: str, generation: int, current_state: str,
        anomaly_history: list[int] | tuple[int, ...], clean_streak: int,
    ) -> None:
        """Persist policy state/hysteresis atomically for restart restoration."""
        if generation < 1 or clean_streak < 0:
            raise ValueError("invalid generation or clean_streak")
        history_json = json.dumps([int(v) for v in anomaly_history], separators=(",", ":"))
        updated_at_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO policy_state
                   (device_id, generation, current_state, anomaly_history_json, clean_streak, updated_at_ns)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(device_id) DO UPDATE SET
                     generation=excluded.generation, current_state=excluded.current_state,
                     anomaly_history_json=excluded.anomaly_history_json,
                     clean_streak=excluded.clean_streak, updated_at_ns=excluded.updated_at_ns""",
                (device_id, generation, current_state, history_json, clean_streak, updated_at_ns),
            )

        self._writers["control"].submit_sync(_op)

    def load_policy_states(self) -> dict[str, dict[str, Any]]:
        """Load persisted policy states, including anomaly history and clean streak."""
        conn = self.connect_readonly("control")
        try:
            rows = conn.execute(
                "SELECT device_id, generation, current_state, anomaly_history_json, clean_streak FROM policy_state"
            ).fetchall()
            restored: dict[str, dict[str, Any]] = {}
            for row in rows:
                try:
                    history = json.loads(row["anomaly_history_json"])
                    if not isinstance(history, list) or any(v not in (0, 1) for v in history):
                        continue
                    restored[str(row["device_id"])] = {
                        "generation": int(row["generation"]),
                        "current_state": str(row["current_state"]),
                        "anomaly_history": [int(v) for v in history],
                        "clean_streak": max(0, int(row["clean_streak"])),
                    }
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            return restored
        finally:
            conn.close()

    def append_policy_dimension_transition(
        self,
        device_id: str,
        generation: int,
        dimension: str,
        from_state: str,
        to_state: str,
        reason_code: str,
    ) -> str:
        """Append a per-device, per-dimension hash-chained state transition."""
        if dimension not in {"operational", "security"}:
            raise ValueError("dimension must be 'operational' or 'security'")
        if generation < 1 or not device_id:
            raise ValueError("invalid device identity for policy transition")
        ts_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> str:
            previous = conn.execute(
                """SELECT entry_hash FROM policy_dimension_transitions
                   WHERE device_id=? AND dimension=? ORDER BY transition_id DESC LIMIT 1""",
                (device_id, dimension),
            ).fetchone()
            prev_hash = str(previous[0]) if previous else GENESIS_HASH
            canonical = json.dumps(
                {
                    "ts_ns": ts_ns,
                    "device_id": device_id,
                    "generation": generation,
                    "dimension": dimension,
                    "from_state": from_state,
                    "to_state": to_state,
                    "reason_code": reason_code,
                    "prev_hash": prev_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            entry_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            conn.execute(
                """INSERT INTO policy_dimension_transitions
                   (ts_ns, device_id, generation, dimension, from_state, to_state,
                    reason_code, prev_hash, entry_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts_ns, device_id, generation, dimension, from_state, to_state,
                 reason_code[:256], prev_hash, entry_hash),
            )
            return entry_hash

        return str(self._writers["control"].submit_sync(_op))

    def save_policy_dimensions(
        self,
        *,
        device_id: str,
        generation: int,
        security_phase: str,
        security_status: str,
        operational_status: str,
        security_anomaly_history: list[int] | tuple[int, ...],
        security_clean_streak: int,
        operational_clean_streak: int,
        security_transition_history: list[str] | tuple[str, ...],
        operational_transition_history: list[str] | tuple[str, ...],
    ) -> None:
        """Persist independent V6 state dimensions and V5 compatibility snapshot."""
        valid_security = {"ACTIVE", "SUSPECT", "QUARANTINED", "REVOKED"}
        valid_operational = {"HEALTHY", "DEGRADED", "FAULT"}
        if generation < 1 or security_status not in valid_security or operational_status not in valid_operational:
            raise ValueError("invalid V6 policy dimensions")
        if security_clean_streak < 0 or operational_clean_streak < 0:
            raise ValueError("policy clean streaks cannot be negative")
        if any(value not in (0, 1) for value in security_anomaly_history):
            raise ValueError("security anomaly history must contain only 0/1 values")
        anomaly_json = json.dumps(list(security_anomaly_history), separators=(",", ":"))
        security_history_json = json.dumps(list(security_transition_history)[-64:], separators=(",", ":"))
        operational_history_json = json.dumps(list(operational_transition_history)[-64:], separators=(",", ":"))
        updated_at_ns = time.time_ns()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO policy_dimensions
                   (device_id, generation, security_phase, security_status, operational_status,
                    security_anomaly_history_json, security_clean_streak, operational_clean_streak,
                    security_transition_history_json, operational_transition_history_json, updated_at_ns)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(device_id) DO UPDATE SET
                    generation=excluded.generation,
                    security_phase=excluded.security_phase,
                    security_status=excluded.security_status,
                    operational_status=excluded.operational_status,
                    security_anomaly_history_json=excluded.security_anomaly_history_json,
                    security_clean_streak=excluded.security_clean_streak,
                    operational_clean_streak=excluded.operational_clean_streak,
                    security_transition_history_json=excluded.security_transition_history_json,
                    operational_transition_history_json=excluded.operational_transition_history_json,
                    updated_at_ns=excluded.updated_at_ns""",
                (device_id, generation, security_phase, security_status, operational_status,
                 anomaly_json, security_clean_streak, operational_clean_streak,
                 security_history_json, operational_history_json, updated_at_ns),
            )
            # Keep old readers and V5 diagnostics working, but store the
            # security phase here: operational FAULT never overwrites it.
            conn.execute(
                """INSERT INTO policy_state
                   (device_id, generation, current_state, anomaly_history_json, clean_streak, updated_at_ns)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(device_id) DO UPDATE SET
                    generation=excluded.generation,
                    current_state=excluded.current_state,
                    anomaly_history_json=excluded.anomaly_history_json,
                    clean_streak=excluded.clean_streak,
                    updated_at_ns=excluded.updated_at_ns""",
                (device_id, generation, security_phase, anomaly_json, security_clean_streak, updated_at_ns),
            )

        self._writers["control"].submit_sync(_op)

    def load_policy_dimensions(self) -> dict[str, dict[str, Any]]:
        """Restore orthogonal operational/security policy state after restart."""
        conn = self.connect_readonly("control")
        try:
            rows = conn.execute(
                """SELECT device_id, generation, security_phase, security_status,
                          operational_status, security_anomaly_history_json,
                          security_clean_streak, operational_clean_streak,
                          security_transition_history_json, operational_transition_history_json
                   FROM policy_dimensions"""
            ).fetchall()
            result: dict[str, dict[str, Any]] = {}
            for row in rows:
                try:
                    anomaly = json.loads(row["security_anomaly_history_json"])
                    sec_hist = json.loads(row["security_transition_history_json"])
                    op_hist = json.loads(row["operational_transition_history_json"])
                    if (not isinstance(anomaly, list) or any(v not in (0, 1) for v in anomaly)
                            or not isinstance(sec_hist, list) or not isinstance(op_hist, list)):
                        continue
                    result[str(row["device_id"])] = {
                        "generation": int(row["generation"]),
                        "security_phase": str(row["security_phase"]),
                        "security_status": str(row["security_status"]),
                        "operational_status": str(row["operational_status"]),
                        "security_anomaly_history": [int(v) for v in anomaly],
                        "security_clean_streak": max(0, int(row["security_clean_streak"])),
                        "operational_clean_streak": max(0, int(row["operational_clean_streak"])),
                        "security_transition_history": [str(v) for v in sec_hist][-64:],
                        "operational_transition_history": [str(v) for v in op_hist][-64:],
                    }
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            return result
        finally:
            conn.close()

    def load_policy_dimension_transitions(
        self, device_id: str | None = None, dimension: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read the independently auditable operational/security histories."""
        if dimension is not None and dimension not in {"operational", "security"}:
            raise ValueError("invalid policy transition dimension")
        clauses, params = [], []
        if device_id is not None:
            clauses.append("device_id=?")
            params.append(device_id)
        if dimension is not None:
            clauses.append("dimension=?")
            params.append(dimension)
        sql = "SELECT * FROM policy_dimension_transitions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY transition_id"
        conn = self.connect_readonly("control")
        try:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

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