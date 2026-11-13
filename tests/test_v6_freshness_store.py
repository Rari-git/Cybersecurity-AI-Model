from __future__ import annotations

import hashlib
import sqlite3
import threading

import pytest

from storage.admission_gate import AdmissionGate, ProvenanceContext
from storage.telemetry_db import DBPaths, FreshnessRejected, LumenStorageManager

GAS_TOPIC = "home/sensors/gas/ppm"
BOOT_ID = "0123456789abcdef0123456789abcdef"


def _context(session_id="v6-live"):
    return ProvenanceContext(
        session_id=session_id,
        device_cn="gas_sensor_01-g2-cafe1234",
        cert_fingerprint=hashlib.sha256(b"gas-sensor-certificate").digest(),
        source_type="live_validated",
        device_id="gas_sensor_01",
        generation=2,
    )


def _row(session_id: str, sequence: int, boot_id: str = BOOT_ID, generation: int = 2):
    payload = str(350 + sequence / 10)
    payload_bytes = payload.encode()
    return (
        session_id, "gas_sensor_01", f"gas_sensor_01-g{generation}-cafe1234", generation,
        hashlib.sha256(b"gas-sensor-certificate").digest(), GAS_TOPIC,
        1_700_000_000_000_000_000 + sequence, 1_700_000_000_000_000_000 + sequence,
        len(payload_bytes), hashlib.sha256(payload_bytes).digest(), float(payload),
        payload, "live_validated", "v6-test", 6, boot_id, sequence,
    )


def _open_storage(path):
    mgr = LumenStorageManager(paths=DBPaths.from_dir(path), enable_writers=True)
    mgr.register_baseline_session("v6-live", "live_validated", {"protocol": 6}, approved_for_training=False)
    return mgr


def test_v6_boot_registration_and_sequence_survive_storage_restart(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        assert mgr.persist_v6_normal_event(
            _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
        ) == 0
        assert mgr.persist_v6_normal_event(
            _row("v6-live", 2), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=2, device_timestamp_ns=1_700_000_000_000_000_002,
        ) == 1
    finally:
        mgr.close()

    restarted = _open_storage(tmp_path / "db")
    try:
        state = restarted.load_device_freshness("gas_sensor_01")["gas_sensor_01"]
        assert state["certificate_generation"] == 2
        assert state["active_boot_id"] == BOOT_ID
        assert state["last_sequence"] == 2
        with pytest.raises(FreshnessRejected, match="REPLAY_OR_OUT_OF_ORDER_SEQUENCE"):
            restarted.persist_v6_normal_event(
                _row("v6-live", 2), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=2, device_timestamp_ns=1_700_000_000_000_000_002,
            )
        assert restarted.persist_v6_normal_event(
            _row("v6-live", 3), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=3, device_timestamp_ns=1_700_000_000_000_000_003,
        ) == 2
    finally:
        restarted.close()


def test_v6_rejects_unregistered_or_stale_boot_and_generation(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    try:
        with pytest.raises(FreshnessRejected, match="BOOT_NOT_REGISTERED"):
            mgr.persist_v6_normal_event(
                _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
            )
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        with pytest.raises(FreshnessRejected, match="STALE_BOOT_ID"):
            mgr.persist_v6_normal_event(
                _row("v6-live", 1, "f" * 32), device_id="gas_sensor_01", certificate_generation=2,
                boot_id="f" * 32, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
            )
        with pytest.raises(FreshnessRejected, match="CERTIFICATE_GENERATION_MISMATCH"):
            mgr.persist_v6_normal_event(
                _row("v6-live", 1, generation=1), device_id="gas_sensor_01", certificate_generation=1,
                boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
            )
        with pytest.raises(ValueError, match="invalid sequence"):
            mgr.persist_v6_normal_event(
                _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=0, device_timestamp_ns=1_700_000_000_000_000_001,
            )
    finally:
        mgr.close()


def test_v6_event_insert_failure_does_not_advance_sequence(tmp_path):
    mgr = LumenStorageManager(paths=DBPaths.from_dir(tmp_path / "db"), enable_writers=True)
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        # Unknown session violates the raw_events FK after freshness validation.
        with pytest.raises(sqlite3.IntegrityError):
            mgr.persist_v6_normal_event(
                _row("missing-session", 7), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=7, device_timestamp_ns=1_700_000_000_000_000_007,
            )
        state = mgr.load_device_freshness("gas_sensor_01")["gas_sensor_01"]
        assert state["last_sequence"] == 0
        mgr.register_baseline_session("v6-live", "live_validated", {"protocol": 6}, approved_for_training=False)
        assert mgr.persist_v6_normal_event(
            _row("v6-live", 7), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=7, device_timestamp_ns=1_700_000_000_000_000_007,
        ) == 0
    finally:
        mgr.close()


def test_v6_two_concurrent_writers_cannot_accept_same_sequence(tmp_path):
    db_dir = tmp_path / "db"
    mgr1 = _open_storage(db_dir)
    mgr1.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
    mgr2 = _open_storage(db_dir)
    barrier = threading.Barrier(2)
    outcomes = []
    lock = threading.Lock()

    def submit(mgr):
        barrier.wait(timeout=5)
        try:
            result = mgr.persist_v6_normal_event(
                _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
            )
            outcome = ("accepted", result)
        except FreshnessRejected as exc:
            outcome = ("rejected", exc.reason_code)
        except Exception as exc:  # pragma: no cover - asserts diagnostic if SQLite unexpectedly fails
            outcome = ("error", repr(exc))
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=submit, args=(mgr,)) for mgr in (mgr1, mgr2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert all(not thread.is_alive() for thread in threads)
        assert sum(item[0] == "accepted" for item in outcomes) == 1
        assert sum(item[0] == "rejected" for item in outcomes) == 1
    finally:
        mgr2.close()
        mgr1.close()


def test_v6_admission_gate_uses_atomic_freshness_and_routes_replay_to_incidents(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    gate = AdmissionGate(mgr, {"gas_sensor_01": {GAS_TOPIC}})
    ctx = _context()
    try:
        gate.register_verified_device_boot(ctx, BOOT_ID)
        frame1 = (
            '{"protocol_version":6,"device_id":"gas_sensor_01",'
            f'"boot_id":"{BOOT_ID}","sequence":1,'
            '"device_timestamp_ns":1700000000000000001,"payload":"350"}'
        )
        accepted = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, frame1)
        assert accepted.admitted and accepted.sequence == 1
        replay = gate.evaluate_v6_and_ingest(ctx, GAS_TOPIC, frame1)
        assert not replay.admitted
        assert replay.reason == "REPLAY_OR_OUT_OF_ORDER_SEQUENCE"
        mgr.flush_all()
        normal = mgr.connect_readonly("normal")
        incidents = mgr.connect_readonly("incidents")
        try:
            assert normal.execute("SELECT COUNT(*) FROM raw_events WHERE protocol_version=6").fetchone()[0] == 1
            assert incidents.execute("SELECT COUNT(*) FROM incident_captures WHERE reason_code='REPLAY_OR_OUT_OF_ORDER_SEQUENCE'").fetchone()[0] == 1
        finally:
            normal.close()
            incidents.close()
    finally:
        mgr.close()


def test_v5_schema_upgrade_adds_v6_columns_without_losing_v5_rows(tmp_path):
    # Exercise an old V5 normal DB whose raw_events table lacks V6 columns.
    import sqlite3
    db_dir = tmp_path / "legacy"
    db_dir.mkdir()
    old_db = db_dir / "telemetry_normal.db"
    schema = ( __import__("pathlib").Path(__file__).parents[1] / "storage" / "schema.sql").read_text()
    normal_section = schema.split("-- [DB: telemetry_normal]", 1)[1].split("-- [DB: telemetry_incidents]", 1)[0]
    old_section = normal_section.replace(
        "    protocol_version INTEGER NOT NULL DEFAULT 5 CHECK(protocol_version IN (5, 6)),\n    boot_id TEXT,\n    sequence INTEGER,\n", ""
    ).replace(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_v6_raw_device_boot_sequence\n    ON raw_events(device_id, generation, boot_id, sequence)\n    WHERE boot_id IS NOT NULL AND sequence IS NOT NULL;\n", ""
    ).replace(
        "CREATE TABLE IF NOT EXISTS device_freshness (\n    device_id TEXT PRIMARY KEY,\n    certificate_generation INTEGER NOT NULL CHECK(certificate_generation > 0),\n    active_boot_id TEXT NOT NULL CHECK(length(active_boot_id) = 32),\n    last_sequence INTEGER NOT NULL DEFAULT 0 CHECK(last_sequence >= 0),\n    last_device_timestamp_ns INTEGER,\n    updated_at_ns INTEGER NOT NULL\n);\n", ""
    )
    with sqlite3.connect(old_db) as conn:
        conn.executescript(old_section)
        conn.execute(
            "INSERT INTO baseline_sessions(session_id,started_at_ns,source_type,git_hash,config_json,approved_for_training) "
            "VALUES('legacy-v5',1,'baseline_lab','v5','{}',0)"
        )
        payload = b"350.0"
        conn.execute(
            """INSERT INTO raw_events(
                session_id,device_id,device_cn,generation,cert_fingerprint,topic,
                gateway_ts_ns,device_ts_ns,payload_len,payload_sha256,numeric_value,
                raw_payload,source_type,admission_policy_version
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ('legacy-v5','gas_sensor_01','gas_sensor_01-g2-cafe1234',2,
             hashlib.sha256(b'legacy-cert').digest(),GAS_TOPIC,10,10,len(payload),
             hashlib.sha256(payload).digest(),350.0,'350.0','baseline_lab','v5'),
        )
    mgr = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    ro = None
    try:
        ro = mgr.connect_readonly("normal")
        columns = {row[1] for row in ro.execute("PRAGMA table_info(raw_events)")}
        assert {"protocol_version", "boot_id", "sequence"}.issubset(columns)
        legacy_rows = ro.execute(
            "SELECT device_id, raw_payload, protocol_version, boot_id, sequence FROM raw_events"
        ).fetchall()
        assert len(legacy_rows) == 1
        assert tuple(legacy_rows[0]) == ("gas_sensor_01", "350.0", 5, None, None)
        assert "gas_sensor_01" not in mgr.load_device_freshness()
    finally:
        if ro is not None:
            ro.close()
        mgr.close()


def test_v6_boot_re_registration_is_idempotent_and_new_boot_resets_counter(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    new_boot = "f" * 32
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        mgr.persist_v6_normal_event(
            _row("v6-live", 5), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=5, device_timestamp_ns=1_700_000_000_000_000_005,
        )
        # Re-registering the same epoch is a no-op, not a replay-counter reset.
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        assert mgr.load_device_freshness("gas_sensor_01")["gas_sensor_01"]["last_sequence"] == 5
        # A new boot epoch is activated only through the explicit trusted registration API.
        mgr.register_verified_device_boot("gas_sensor_01", 2, new_boot)
        state = mgr.load_device_freshness("gas_sensor_01")["gas_sensor_01"]
        assert state["active_boot_id"] == new_boot
        assert state["last_sequence"] == 0
        with pytest.raises(FreshnessRejected, match="STALE_BOOT_ID"):
            mgr.persist_v6_normal_event(
                _row("v6-live", 6), device_id="gas_sensor_01", certificate_generation=2,
                boot_id=BOOT_ID, sequence=6, device_timestamp_ns=1_700_000_000_000_000_006,
            )
        assert mgr.persist_v6_normal_event(
            _row("v6-live", 1, new_boot), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=new_boot, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
        ) == 0
    finally:
        mgr.close()


def test_v6_boot_registration_rejects_downgrade_generation(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 3, BOOT_ID)
        with pytest.raises(FreshnessRejected, match="STALE_CERTIFICATE_GENERATION"):
            mgr.register_verified_device_boot("gas_sensor_01", 2, "f" * 32)
        state = mgr.load_device_freshness("gas_sensor_01")["gas_sensor_01"]
        assert state["certificate_generation"] == 3
        assert state["active_boot_id"] == BOOT_ID
    finally:
        mgr.close()


def test_baseline_session_registration_is_idempotent_and_preserves_children(tmp_path):
    mgr = _open_storage(tmp_path / "db")
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        mgr.persist_v6_normal_event(
            _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
        )
        mgr.mark_session_curated(
            "v6-live", "pending", {"reason": "idempotency regression test"}
        )
        mgr.flush_all()

        def snapshot():
            conn = mgr.connect_readonly("normal")
            try:
                session = conn.execute(
                    """SELECT started_at_ns, source_type, git_hash, config_json,
                              approved_for_training
                       FROM baseline_sessions WHERE session_id=?""",
                    ("v6-live",),
                ).fetchone()
                event_count = conn.execute(
                    "SELECT COUNT(*) FROM raw_events WHERE session_id=?", ("v6-live",)
                ).fetchone()[0]
                curation = conn.execute(
                    """SELECT status, decision_json FROM session_curation
                       WHERE session_id=?""",
                    ("v6-live",),
                ).fetchone()
                return tuple(session), event_count, tuple(curation)
            finally:
                conn.close()

        before = snapshot()
        # Same identity and same immutable metadata: this must be a no-op.
        mgr.register_baseline_session(
            "v6-live", "live_validated", {"protocol": 6},
            approved_for_training=False,
        )
        mgr.flush_all()
        after = snapshot()

        assert after == before
        assert after[1] == 1
        assert after[2][0] == "pending"
    finally:
        mgr.close()


@pytest.mark.parametrize(
    ("source_type", "config", "git_hash"),
    [
        ("live_validated", {"protocol": 7}, "v5.0-dev"),
        ("live_validated", {"protocol": 6}, "different-git-hash"),
        ("baseline_lab", {"protocol": 6}, "v5.0-dev"),
    ],
)
def test_baseline_session_registration_rejects_conflicting_metadata_without_mutation(
    tmp_path, source_type, config, git_hash
):
    mgr = _open_storage(tmp_path / "db")
    try:
        mgr.register_verified_device_boot("gas_sensor_01", 2, BOOT_ID)
        mgr.persist_v6_normal_event(
            _row("v6-live", 1), device_id="gas_sensor_01", certificate_generation=2,
            boot_id=BOOT_ID, sequence=1, device_timestamp_ns=1_700_000_000_000_000_001,
        )
        mgr.flush_all()

        conn = mgr.connect_readonly("normal")
        try:
            before_session = tuple(conn.execute(
                """SELECT started_at_ns, source_type, git_hash, config_json,
                          approved_for_training
                   FROM baseline_sessions WHERE session_id=?""",
                ("v6-live",),
            ).fetchone())
            before_events = conn.execute(
                "SELECT COUNT(*) FROM raw_events WHERE session_id=?", ("v6-live",)
            ).fetchone()[0]
        finally:
            conn.close()

        with pytest.raises(ValueError, match="conflicting metadata"):
            mgr.register_baseline_session(
                "v6-live", source_type, config, git_hash=git_hash,
                approved_for_training=False,
            )

        conn = mgr.connect_readonly("normal")
        try:
            after_session = tuple(conn.execute(
                """SELECT started_at_ns, source_type, git_hash, config_json,
                          approved_for_training
                   FROM baseline_sessions WHERE session_id=?""",
                ("v6-live",),
            ).fetchone())
            after_events = conn.execute(
                "SELECT COUNT(*) FROM raw_events WHERE session_id=?", ("v6-live",)
            ).fetchone()[0]
        finally:
            conn.close()

        assert after_session == before_session
        assert after_events == before_events == 1
        assert mgr.load_device_freshness("gas_sensor_01")["gas_sensor_01"]["last_sequence"] == 1
    finally:
        mgr.close()
