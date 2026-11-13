-- Lumen Edge Security v5.0 — Schema SQLite (3 baze de date fizic separate)
-- Fiecare secțiune [DB: ...] este aplicată bazei de date corespunzătoare.

-- ============================================================================
-- [DB: telemetry_normal]
-- ============================================================================
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 3000;

CREATE TABLE IF NOT EXISTS baseline_sessions (
    session_id TEXT PRIMARY KEY,
    started_at_ns INTEGER NOT NULL,
    ended_at_ns INTEGER,
    source_type TEXT NOT NULL CHECK(source_type IN ('baseline_lab', 'live_validated')),
    git_hash TEXT,
    config_json TEXT NOT NULL,
    approved_for_training INTEGER NOT NULL DEFAULT 0 CHECK(approved_for_training IN (0, 1))
);

CREATE TABLE IF NOT EXISTS raw_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    device_cn TEXT NOT NULL,
    generation INTEGER NOT NULL,
    cert_fingerprint BLOB NOT NULL CHECK(length(cert_fingerprint) = 32),
    topic TEXT NOT NULL,
    gateway_ts_ns INTEGER NOT NULL,
    device_ts_ns INTEGER,
    payload_len INTEGER NOT NULL,
    payload_sha256 BLOB NOT NULL CHECK(length(payload_sha256) = 32),
    numeric_value REAL,
    raw_payload TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('baseline_lab', 'live_validated')),
    admission_policy_version TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES baseline_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_normal_raw_dev_topic_ts
    ON raw_events(device_id, topic, gateway_ts_ns);
CREATE INDEX IF NOT EXISTS idx_normal_raw_session
    ON raw_events(session_id);

CREATE TABLE IF NOT EXISTS windows (
    window_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    device_cn TEXT NOT NULL,
    generation INTEGER NOT NULL,
    topic TEXT NOT NULL,
    ts_start_ns INTEGER NOT NULL,
    ts_end_ns INTEGER NOT NULL,
    seq_len INTEGER NOT NULL,
    feature_dim INTEGER NOT NULL,
    features_blob BLOB NOT NULL,
    feature_schema_version TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('baseline_lab', 'live_validated')),
    source_manifest_hash BLOB NOT NULL CHECK(length(source_manifest_hash) = 32),
    admitted_at_ns INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES baseline_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_normal_win_dev_topic
    ON windows(device_id, topic, ts_start_ns);

-- ============================================================================
-- [DB: telemetry_incidents]
-- ============================================================================
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 3000;

CREATE TABLE IF NOT EXISTS incident_captures (
    capture_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    device_cn TEXT NOT NULL,
    generation INTEGER NOT NULL,
    cert_fingerprint BLOB NOT NULL CHECK(length(cert_fingerprint) = 32),
    original_topic TEXT NOT NULL,
    shadow_topic TEXT,
    gateway_ts_ns INTEGER NOT NULL,
    device_ts_ns INTEGER,
    capture_source TEXT NOT NULL CHECK(capture_source IN ('shadow_sink', 'ids_alert', 'attack_lab', 'fault_lab')),
    reason_code TEXT NOT NULL,
    risk_score REAL,
    xai_contrib_json TEXT,
    payload_len INTEGER NOT NULL,
    payload_sha256 BLOB NOT NULL CHECK(length(payload_sha256) = 32),
    raw_payload BLOB NOT NULL,
    policy_version INTEGER NOT NULL DEFAULT 1,
    ground_truth_label TEXT,
    scenario_id TEXT,
    retention_until_ns INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_incidents_dev_ts
    ON incident_captures(device_id, gateway_ts_ns);
CREATE INDEX IF NOT EXISTS idx_incidents_scenario
    ON incident_captures(scenario_id, capture_source);

-- ============================================================================
-- [DB: security_control]
-- ============================================================================
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA busy_timeout = 3000;

CREATE TABLE IF NOT EXISTS device_registry (
    device_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    device_cn TEXT NOT NULL,
    cert_fingerprint BLOB NOT NULL CHECK(length(cert_fingerprint) = 32),
    device_role TEXT NOT NULL,
    criticality TEXT NOT NULL DEFAULT 'normal' CHECK(criticality IN ('low', 'normal', 'safety_critical')),
    current_state TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(current_state IN (
        'ACTIVE', 'MONITORING', 'ENFORCEMENT_PENDING',
        'SHADOW_QUARANTINE', 'ENFORCEMENT_FAILED', 'RECOVERY_PENDING',
        'REVOKED', 'FAULT', 'OFFLINE'
    )),
    updated_at_ns INTEGER NOT NULL,
    PRIMARY KEY (device_id, generation)
);

CREATE TABLE IF NOT EXISTS seen_nonces (
    device_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    nonce TEXT NOT NULL,
    first_seen_ns INTEGER NOT NULL,
    PRIMARY KEY (device_id, generation, nonce)
);

CREATE TABLE IF NOT EXISTS quarantine_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns INTEGER NOT NULL,
    device_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    risk_score REAL,
    policy_version INTEGER NOT NULL,
    operator_id TEXT,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enforcement_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT UNIQUE NOT NULL,
    device_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    old_state TEXT NOT NULL,
    target_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    decided_at_ns INTEGER NOT NULL,
    command_sent_at_ns INTEGER,
    pep_confirmed_at_ns INTEGER,
    pep_status TEXT NOT NULL DEFAULT 'PENDING' CHECK(pep_status IN ('PENDING', 'CONFIRMED', 'FAILED', 'TIMEOUT')),
    enforcement_latency_ms REAL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_enforcement_dev_ts
    ON enforcement_audit(device_id, decided_at_ns);