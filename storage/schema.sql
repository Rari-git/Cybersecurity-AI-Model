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
    approved_for_training INTEGER NOT NULL DEFAULT 0 CHECK(approved_for_training IN (0, 1)),
    CHECK(source_type <> 'live_validated' OR approved_for_training = 0)
);


-- V6 Stage B: a session may enter normal training only after a deterministic
-- curation decision. `live_validated` is never eligible, regardless of caller.
CREATE TABLE IF NOT EXISTS session_curation (
    session_id TEXT PRIMARY KEY REFERENCES baseline_sessions(session_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('pending', 'approved_clean_baseline', 'rejected')),
    decision_json TEXT NOT NULL,
    dataset_sha256 TEXT,
    curated_at_ns INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_baseline_session_insert_approval_guard
BEFORE INSERT ON baseline_sessions
WHEN NEW.approved_for_training = 1
BEGIN
    SELECT RAISE(ABORT, 'training approval requires approved_clean_baseline curation');
END;

CREATE TRIGGER IF NOT EXISTS trg_baseline_session_update_approval_guard
BEFORE UPDATE OF approved_for_training ON baseline_sessions
WHEN NEW.approved_for_training = 1 AND (
    NEW.source_type <> 'baseline_lab'
    OR NOT EXISTS (
        SELECT 1 FROM session_curation c
        WHERE c.session_id = NEW.session_id AND c.status = 'approved_clean_baseline'
    )
)
BEGIN
    SELECT RAISE(ABORT, 'training approval requires approved_clean_baseline curation');
END;

CREATE TRIGGER IF NOT EXISTS trg_live_session_training_guard
BEFORE UPDATE OF source_type ON baseline_sessions
WHEN NEW.source_type = 'live_validated' AND NEW.approved_for_training <> 0
BEGIN
    SELECT RAISE(ABORT, 'live_validated sessions cannot be approved for training');
END;

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
    protocol_version INTEGER NOT NULL DEFAULT 5 CHECK(protocol_version IN (5, 6)),
    boot_id TEXT,
    sequence INTEGER,
    FOREIGN KEY (session_id) REFERENCES baseline_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_normal_raw_dev_topic_ts
    ON raw_events(device_id, topic, gateway_ts_ns);
CREATE INDEX IF NOT EXISTS idx_normal_raw_session
    ON raw_events(session_id);
-- Durable sensor freshness state is kept beside raw_events so freshness
-- advancement and telemetry insertion can share one SQLite transaction.
CREATE TABLE IF NOT EXISTS device_freshness (
    device_id TEXT PRIMARY KEY,
    certificate_generation INTEGER NOT NULL CHECK(certificate_generation > 0),
    active_boot_id TEXT NOT NULL CHECK(length(active_boot_id) = 32),
    last_sequence INTEGER NOT NULL DEFAULT 0 CHECK(last_sequence >= 0),
    last_device_timestamp_ns INTEGER,
    updated_at_ns INTEGER NOT NULL
);

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


-- Incident/forensic material is indexed separately as an adversarial regression
-- corpus. It is never joined into telemetry_normal.raw_events or clean training.
CREATE TABLE IF NOT EXISTS adversarial_regression_corpus (
    capture_id INTEGER PRIMARY KEY REFERENCES incident_captures(capture_id) ON DELETE CASCADE,
    corpus_class TEXT NOT NULL CHECK(corpus_class IN ('adversarial', 'hardware_fault', 'protocol_violation')),
    indexed_at_ns INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_incident_to_regression_corpus
AFTER INSERT ON incident_captures
BEGIN
    INSERT OR IGNORE INTO adversarial_regression_corpus(capture_id, corpus_class, indexed_at_ns)
    VALUES (
        NEW.capture_id,
        CASE
            WHEN NEW.capture_source = 'fault_lab'
                 OR lower(coalesce(NEW.ground_truth_label, '')) LIKE '%fault%'
                THEN 'hardware_fault'
            WHEN lower(NEW.reason_code) LIKE '%replay%'
                 OR lower(NEW.reason_code) LIKE '%sequence%'
                 OR lower(NEW.reason_code) LIKE '%freshness%'
                 OR lower(NEW.reason_code) LIKE '%boot%'
                 OR lower(NEW.reason_code) LIKE '%protocol%'
                THEN 'protocol_violation'
            ELSE 'adversarial'
        END,
        CAST(strftime('%s','now') AS INTEGER) * 1000000000
    );
END;

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
        'SHADOW_QUARANTINE', 'QUARANTINED', 'ENFORCEMENT_FAILED', 'RECOVERY_PENDING',
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

CREATE TABLE IF NOT EXISTS policy_state (
    device_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    current_state TEXT NOT NULL,
    anomaly_history_json TEXT NOT NULL,
    clean_streak INTEGER NOT NULL DEFAULT 0,
    updated_at_ns INTEGER NOT NULL
);

-- V6 keeps operational health and cybersecurity enforcement orthogonal.
-- current_state in the V5 policy_state table remains a compatibility/security
-- phase field; it is never overwritten by a physical FAULT transition.
CREATE TABLE IF NOT EXISTS policy_dimensions (
    device_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL CHECK(generation > 0),
    security_phase TEXT NOT NULL,
    security_status TEXT NOT NULL CHECK(security_status IN ('ACTIVE', 'SUSPECT', 'QUARANTINED', 'REVOKED')),
    operational_status TEXT NOT NULL CHECK(operational_status IN ('HEALTHY', 'DEGRADED', 'FAULT')),
    security_anomaly_history_json TEXT NOT NULL DEFAULT '[]',
    security_clean_streak INTEGER NOT NULL DEFAULT 0 CHECK(security_clean_streak >= 0),
    operational_clean_streak INTEGER NOT NULL DEFAULT 0 CHECK(operational_clean_streak >= 0),
    security_transition_history_json TEXT NOT NULL DEFAULT '[]',
    operational_transition_history_json TEXT NOT NULL DEFAULT '[]',
    updated_at_ns INTEGER NOT NULL
);

-- Separate, independently hash-chained histories for operational and security
-- state transitions. This prevents a hardware fault from replacing or
-- obscuring the security quarantine history.
CREATE TABLE IF NOT EXISTS policy_dimension_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns INTEGER NOT NULL,
    device_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    dimension TEXT NOT NULL CHECK(dimension IN ('operational', 'security')),
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_dimension_transitions_device
    ON policy_dimension_transitions(device_id, dimension, transition_id);
