"""Etapa E5: Antrenare locală PyTorch pe desktop -> Export ONNX -> Calibrare -> Bundle Semnat.

Caracteristici:
- Citește exclusiv sesiuni aprobate din telemetry_normal.db cu ?mode=ro
- Split determinist pe sesiuni (fără leakage între ferestre glisante suprapuse)
- Calibrează pragurile și scalele reziduurilor XAI pe sesiunea de validare
- Exportă model.onnx compatibil onnxruntime și semnează manifestul cu Ed25519
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.telemetry_db import LumenStorageManager, DBPaths
from utils.features_v5 import (
    DeviceTopicFeatureExtractor,
    RobustFeatureScaler,
    FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_DIM,
)
from models.bundle_loader import (
    generate_model_signing_keypair,
    hash_unsigned_candidate_artifacts,
    sign_model_bundle,
)


class TemporalLSTMAutoencoder(nn.Module):
    """Autoencoder secvențial LSTM pentru detecția anomaliilor comportamentale."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        enc_out, _ = self.encoder(x)
        h_last = torch.tanh(self.to_latent(enc_out[:, -1, :]))
        dec_in = torch.tanh(self.from_latent(h_last)).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output_proj(dec_out)


def _extract_windows_by_session(
    db_dir: str | Path,
    seq_len: int = 16,
    eligible_session_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[
    dict[str, list[tuple[str, str, np.ndarray]]],
    dict[str, dict[tuple[str, str], list[float]]],
]:
    """Citește raw_events din telemetry_normal.db (?mode=ro) și construiește ferestrele
    grupat pe session_id pentru split fără data leakage.
    """
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=False)
    ro = storage.connect_readonly("normal")
    try:
        eligible_rows = ro.execute(
            """SELECT s.session_id, c.status, c.decision_json, c.dataset_sha256
               FROM baseline_sessions s
               JOIN session_curation c ON c.session_id=s.session_id
               WHERE s.approved_for_training=1 AND s.source_type='baseline_lab'
                 AND c.status='approved_clean_baseline'
               ORDER BY s.started_at_ns ASC"""
        ).fetchall()
        approved_ids = [str(r["session_id"]) for r in eligible_rows]
        curation_by_id = {str(row["session_id"]): row for row in eligible_rows}
        if eligible_session_ids is not None:
            requested = list(dict.fromkeys(str(value) for value in eligible_session_ids))
            missing = sorted(set(requested) - set(approved_ids))
            if missing:
                raise ValueError(f"requested training sessions are not curated approved_clean_baseline sessions: {missing}")
            requested_set = set(requested)
            sessions = [sid for sid in approved_ids if sid in requested_set]
        else:
            sessions = approved_ids
        if len(sessions) < 2:
            raise RuntimeError(
                f"Need at least 2 approved baseline sessions for train/val split, found {len(sessions)}. "
                "Run: python simulation/capture_baseline.py --sessions 4 --events 400"
            )

        windows_by_session: dict[str, list[tuple[str, str, np.ndarray]]] = {s: [] for s in sessions}
        raw_values_by_session: dict[str, dict[tuple[str, str], list[float]]] = {
            session_id: {} for session_id in sessions
        }

        for sid in sessions:
            rows = ro.execute(
                """
                SELECT event_id, device_id, generation, topic, gateway_ts_ns,
                       payload_sha256, raw_payload, numeric_value
                FROM raw_events
                WHERE session_id = ?
                ORDER BY device_id ASC, topic ASC, event_id ASC
                """,
                (sid,),
            ).fetchall()

            # Do not trust the boolean alone: verify the immutable curation
            # decision and its content digest every time training reads a session.
            curation = curation_by_id[sid]
            decision = json.loads(str(curation["decision_json"]))
            if (decision.get("approved") is not True
                    or decision.get("status") != "approved_clean_baseline"
                    or decision.get("reasons") not in ([], ())):
                raise ValueError(f"session {sid} has an invalid clean-baseline curation decision")
            digest = hashlib.sha256()

            # Curation hashes records in event_id order.
            # Reproduce that exact canonical ordering during training.
            digest_rows = sorted(rows, key=lambda row: int(row["event_id"]))

            for row in digest_rows:
                numeric = row["numeric_value"]
                digest_fields = (
                    int(row["event_id"]), str(row["device_id"]), int(row["generation"]),
                    str(row["topic"]), int(row["gateway_ts_ns"]), bytes(row["payload_sha256"]).hex(),
                    None if numeric is None else float(numeric), str(row["raw_payload"]),
                )
                digest.update(json.dumps(digest_fields, separators=(",", ":"), ensure_ascii=True).encode())
                digest.update(b"\n")
            if not curation["dataset_sha256"] or digest.hexdigest() != str(curation["dataset_sha256"]):
                raise ValueError(f"session {sid} clean-baseline dataset digest mismatch")

            extractor = DeviceTopicFeatureExtractor(seq_len=seq_len, stat_window=seq_len)
            last_ts: dict[tuple[str, str], int] = {}

            for r in rows:
                dev = r["device_id"]
                topic = r["topic"]
                ts_ns = int(r["gateway_ts_ns"])
                key = (dev, topic)

                if key in last_ts:
                    dt = (ts_ns - last_ts[key]) / 1_000_000_000.0
                    # Preserve compressed but positive cadence: replacing short
                    # intervals with 1s would erase replay/burst timing features.
                    if dt <= 0:
                        raise ValueError(
                            f"non-monotonic gateway timestamps for {dev}::{topic} in session {sid}"
                        )
                else:
                    dt = 1.0
                last_ts[key] = ts_ns

                num_val = float(r["numeric_value"]) if r["numeric_value"] is not None else None
                st = extractor.update(
                    device_id=dev,
                    topic=topic,
                    delta_t_sec=dt,
                    raw_payload=r["raw_payload"].encode("utf-8"),
                    operation="put",
                    numeric_value=num_val,
                )
                raw_values_by_session[sid].setdefault(key, []).append(st.raw_value)
                win = extractor.get_window(dev, topic)
                if win is not None:
                    windows_by_session[sid].append((dev, topic, win))

        return windows_by_session, raw_values_by_session
    finally:
        ro.close()


def load_capture_run_ids(
    db_dir: str | Path,
    session_ids: list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Load and validate the capture_run_id provenance for approved sessions.

    A capture run is the independent acquisition unit. A session ID alone is
    not sufficient evidence of independence because one capture invocation can
    create several sessions.
    """
    ids = tuple(dict.fromkeys(str(value) for value in session_ids))
    if not ids or any(not value for value in ids):
        raise ValueError("capture-run provenance requires non-empty session IDs")
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=False)
    conn = storage.connect_readonly("normal")
    try:
        rows = []
        for offset in range(0, len(ids), 500):
            batch = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(conn.execute(
                f"""SELECT s.session_id, s.source_type, s.approved_for_training,
                           s.config_json, c.status
                    FROM baseline_sessions s
                    JOIN session_curation c ON c.session_id=s.session_id
                    WHERE s.session_id IN ({placeholders})""",
                batch,
            ).fetchall())
        if {str(row["session_id"]) for row in rows} != set(ids):
            raise ValueError("one or more sessions are missing approved capture-run provenance")

        result: dict[str, str] = {}
        for row in rows:
            session_id = str(row["session_id"])
            if (
                str(row["source_type"]) != "baseline_lab"
                or int(row["approved_for_training"]) != 1
                or str(row["status"]) != "approved_clean_baseline"
            ):
                raise ValueError(f"session {session_id} is not an approved clean baseline")
            try:
                config = json.loads(str(row["config_json"]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"session {session_id} has invalid capture configuration") from exc
            if not isinstance(config, dict):
                raise ValueError(f"session {session_id} capture configuration must be an object")
            capture_run_id = config.get("capture_run_id")
            if (
                not isinstance(capture_run_id, str)
                or len(capture_run_id) != 32
                or capture_run_id != capture_run_id.lower()
                or any(ch not in "0123456789abcdef" for ch in capture_run_id)
                or not session_id.startswith(f"baseline-lab-{capture_run_id}-")
            ):
                raise ValueError(
                    f"session {session_id} has missing or invalid capture_run_id provenance"
                )
            result[session_id] = capture_run_id
        return result
    finally:
        conn.close()
        storage.close()


def build_capture_run_split(
    db_dir: str | Path,
    session_ids: list[str] | tuple[str, ...],
    *,
    storage: LumenStorageManager | None = None,
) -> dict[str, tuple[str, ...] | dict[str, str]] | None:
    """Build a deterministic train/validation/holdout split by capture run.

    All sessions produced by one capture invocation remain in exactly one
    partition. The oldest run(s) train, the penultimate independent run
    validates/calibrates, and the newest independent run is held out. Returns
    None until at least three distinct, approved capture runs exist.
    """
    ids = tuple(dict.fromkeys(str(value) for value in session_ids))
    if not ids or any(not value for value in ids):
        raise ValueError("capture-run split requires non-empty session IDs")
    owns_storage = storage is None
    manager = storage or LumenStorageManager(
        paths=DBPaths.from_dir(db_dir), enable_writers=False
    )
    try:
        manager.flush_all()
        conn = manager.connect_readonly("normal")
        try:
            rows = []
            for offset in range(0, len(ids), 500):
                batch = ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(conn.execute(
                    f"""SELECT s.session_id, s.started_at_ns, s.source_type,
                               s.approved_for_training, s.config_json, c.status
                        FROM baseline_sessions s
                        JOIN session_curation c ON c.session_id=s.session_id
                        WHERE s.session_id IN ({placeholders})""",
                    batch,
                ).fetchall())
        finally:
            conn.close()
    finally:
        if owns_storage:
            manager.close()

    if {str(row["session_id"]) for row in rows} != set(ids):
        raise ValueError("one or more sessions are missing approved capture-run provenance")

    groups: dict[str, dict[str, object]] = {}
    capture_run_ids_by_session: dict[str, str] = {}
    for row in rows:
        session_id = str(row["session_id"])
        if (
            str(row["source_type"]) != "baseline_lab"
            or int(row["approved_for_training"]) != 1
            or str(row["status"]) != "approved_clean_baseline"
        ):
            raise ValueError(f"session {session_id} is not an approved clean baseline")
        try:
            config = json.loads(str(row["config_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"session {session_id} has invalid capture configuration") from exc
        if not isinstance(config, dict):
            raise ValueError(f"session {session_id} capture configuration must be an object")
        capture_run_id = config.get("capture_run_id")
        if (
            not isinstance(capture_run_id, str)
            or len(capture_run_id) != 32
            or capture_run_id != capture_run_id.lower()
            or any(ch not in "0123456789abcdef" for ch in capture_run_id)
            or not session_id.startswith(f"baseline-lab-{capture_run_id}-")
        ):
            raise ValueError(
                f"session {session_id} has missing or invalid capture_run_id provenance"
            )
        try:
            started_at_ns = int(row["started_at_ns"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"session {session_id} has an invalid start timestamp") from exc
        group = groups.setdefault(
            capture_run_id,
            {"first_started_at_ns": started_at_ns, "sessions": []},
        )
        group["first_started_at_ns"] = min(
            int(group["first_started_at_ns"]), started_at_ns
        )
        group["sessions"].append((started_at_ns, session_id))
        capture_run_ids_by_session[session_id] = capture_run_id

    ordered_runs = sorted(
        groups.items(),
        key=lambda item: (int(item[1]["first_started_at_ns"]), item[0]),
    )
    if len(ordered_runs) < 3:
        return None

    def sessions_for(run_items):
        return tuple(
            session_id
            for _run_id, group in run_items
            for _started_at_ns, session_id in sorted(group["sessions"])
        )

    train_runs = ordered_runs[:-2]
    validation_runs = ordered_runs[-2:-1]
    heldout_runs = ordered_runs[-1:]
    return {
        "train_session_ids": sessions_for(train_runs),
        "validation_session_ids": sessions_for(validation_runs),
        "heldout_session_ids": sessions_for(heldout_runs),
        "train_capture_run_ids": tuple(run_id for run_id, _group in train_runs),
        "validation_capture_run_ids": tuple(run_id for run_id, _group in validation_runs),
        "heldout_capture_run_ids": tuple(run_id for run_id, _group in heldout_runs),
        "capture_run_ids_by_session": capture_run_ids_by_session,
    }


def _load_curation_digests(db_dir: str | Path, session_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return immutable curation digests for the exact sessions in a candidate."""
    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=False)
    ro = storage.connect_readonly("normal")
    try:
        ids = tuple(dict.fromkeys(str(value) for value in session_ids))
        if not ids:
            raise ValueError("no curated sessions provided for candidate provenance")
        placeholders = ",".join("?" for _ in ids)
        rows = ro.execute(
            f"""SELECT session_id, status, dataset_sha256 FROM session_curation
                WHERE session_id IN ({placeholders})""",
            ids,
        ).fetchall()
        result = {str(row["session_id"]): str(row["dataset_sha256"] or "") for row in rows}
        if set(result) != set(ids) or any(not value for value in result.values()):
            raise ValueError("candidate session is missing a clean curation digest")
        if any(str(row["status"]) != "approved_clean_baseline" for row in rows):
            raise ValueError("candidate session is not an approved clean baseline")
        return result
    finally:
        ro.close()


def compute_cusum_baselines(
    raw_values_by_session: dict[str, dict[tuple[str, str], list[float]]],
    training_session_ids: list[str] | tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Compute CUSUM statistics from training/calibration sessions only."""
    aggregated: dict[tuple[str, str], list[float]] = {}
    for session_id in training_session_ids:
        session_values = raw_values_by_session.get(session_id, {})
        for stream, values in session_values.items():
            aggregated.setdefault(stream, []).extend(float(v) for v in values)
    baselines: dict[str, dict[str, float]] = {}
    for (device_id, topic), values in aggregated.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"invalid training values for CUSUM stream {device_id}::{topic}")
        baselines[f"{device_id}::{topic}"] = {
            "mean": round(float(np.mean(array)), 6),
            "std": round(max(float(np.std(array)), 0.05), 6),
        }
    return baselines


def train_and_export_bundle(
    db_dir: str | Path = "logs/db",
    bundle_dir: str | Path | None = None,
    key_dir: str | Path = "config/model_keys",
    seq_len: int = 16,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
    run_id: str | None = None,
    eligible_session_ids: list[str] | tuple[str, ...] | None = None,
    heldout_session_ids: list[str] | tuple[str, ...] = (),
    sign_bundle: bool = True,
    validation_session_ids: list[str] | tuple[str, ...] | None = None,
    require_independent_capture_runs: bool | None = None,
) -> dict:
    run_id = run_id or uuid.uuid4().hex
    if require_independent_capture_runs is None:
        # Unsigned bundles belong to the V6 offline authority and must use
        # independent capture runs. Legacy signed E5 calls retain their API.
        require_independent_capture_runs = not sign_bundle
    if bundle_dir is None:
        bundle_dir = Path("models") / "candidates" / run_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    windows_by_session, raw_values_by_session = _extract_windows_by_session(
        db_dir, seq_len=seq_len, eligible_session_ids=eligible_session_ids
    )
    session_ids = list(windows_by_session.keys())
    heldout_input = [str(value) for value in heldout_session_ids]
    if len(set(heldout_input)) != len(heldout_input):
        raise ValueError("held-out session IDs must be unique")
    heldout_sessions = heldout_input
    missing_holdout = sorted(set(heldout_sessions) - set(session_ids))
    if missing_holdout:
        raise ValueError(f"held-out sessions are not curated approved baseline sessions: {missing_holdout}")

    capture_run_ids_by_session: dict[str, str] = {}
    train_capture_runs: list[str] = []
    val_capture_runs: list[str] = []
    heldout_capture_runs: list[str] = []
    if require_independent_capture_runs:
        if eligible_session_ids is None:
            raise ValueError("V6 capture-run splitting requires explicit eligible_session_ids")
        if validation_session_ids is None:
            raise ValueError("V6 capture-run splitting requires explicit validation_session_ids")
        validation_input = [str(value) for value in validation_session_ids]
        if len(set(validation_input)) != len(validation_input):
            raise ValueError("validation session IDs must be unique")
        if not validation_input or not heldout_sessions:
            raise RuntimeError("V6 candidate requires independent validation and held-out capture runs")
        if set(validation_input) & set(heldout_sessions):
            raise ValueError("validation and held-out sessions must be disjoint")
        missing_validation = sorted(set(validation_input) - set(session_ids))
        if missing_validation:
            raise ValueError(
                f"validation sessions are not curated approved baseline sessions: {missing_validation}"
            )
        excluded = set(validation_input) | set(heldout_sessions)
        train_sessions = [sid for sid in session_ids if sid not in excluded]
        val_sessions = validation_input
        if not train_sessions:
            raise RuntimeError("V6 candidate requires at least one training capture run")
        capture_run_ids_by_session = load_capture_run_ids(db_dir, session_ids)
        train_capture_runs = sorted({capture_run_ids_by_session[sid] for sid in train_sessions})
        val_capture_runs = sorted({capture_run_ids_by_session[sid] for sid in val_sessions})
        heldout_capture_runs = sorted({capture_run_ids_by_session[sid] for sid in heldout_sessions})
        if not train_capture_runs or not val_capture_runs or not heldout_capture_runs:
            raise RuntimeError("training, validation and held-out partitions must each contain a capture run")
        if (
            set(train_capture_runs) & set(val_capture_runs)
            or set(train_capture_runs) & set(heldout_capture_runs)
            or set(val_capture_runs) & set(heldout_capture_runs)
        ):
            raise ValueError(
                "capture_run_id leakage: one acquisition run appears in multiple data partitions"
            )
    else:
        fit_sessions = [sid for sid in session_ids if sid not in set(heldout_sessions)]
        if len(fit_sessions) < 2:
            raise RuntimeError("Need at least 2 non-heldout curated sessions for training/calibration.")
        if validation_session_ids is None:
            train_sessions = fit_sessions[:-1]
            val_sessions = [fit_sessions[-1]]
        else:
            val_sessions = [str(value) for value in validation_session_ids]
            if not val_sessions or len(set(val_sessions)) != len(val_sessions):
                raise ValueError("validation session IDs must be non-empty and unique")
            if set(val_sessions) & set(heldout_sessions):
                raise ValueError("validation and held-out sessions must be disjoint")
            missing_validation = sorted(set(val_sessions) - set(session_ids))
            if missing_validation:
                raise ValueError(
                    f"validation sessions are not curated approved baseline sessions: {missing_validation}"
                )
            train_sessions = [sid for sid in fit_sessions if sid not in set(val_sessions)]
            if not train_sessions:
                raise RuntimeError("Need at least one training session separate from validation/holdout.")

    train_items = [item for sid in train_sessions for item in windows_by_session[sid]]
    val_items = [item for sid in val_sessions for item in windows_by_session[sid]]

    if not train_items or not val_items:
        raise RuntimeError("Insufficient windows extracted for training/validation.")

    X_train_raw = np.stack([w for _, _, w in train_items], axis=0)  # (N_tr, T, 12)
    X_val_raw = np.stack([w for _, _, w in val_items], axis=0)      # (N_val, T, 12)

    scaler = RobustFeatureScaler.fit(X_train_raw.reshape(-1, FEATURE_DIM))
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    model = TemporalLSTMAutoencoder(input_dim=FEATURE_DIM, hidden_dim=32, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    model.train()
    final_train_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_x.size(0)
        final_train_loss = epoch_loss / len(X_train)

    # Calibrare pe setul de validare (sesiune complet separată de antrenare)
    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val)
        val_recon = model(val_tensor).numpy()

    # Eliminăm primele 3 ferestre de warm-up per stream din validare (unde rolling_rate se stabilizează)
    seen_per_stream: dict[str, int] = {}
    keep_indices: list[int] = []
    for idx, (dev, topic, _) in enumerate(val_items):
        skey = f"{dev}::{topic}"
        cnt = seen_per_stream.get(skey, 0)
        seen_per_stream[skey] = cnt + 1
        if cnt >= 3:
            keep_indices.append(idx)

    X_val_clean = X_val[keep_indices]
    val_recon_clean = val_recon[keep_indices]
    val_items_clean = [val_items[i] for i in keep_indices]

    # Eroare absolută (L1) pe ultimul pas temporal per trăsătură: (N_val_clean, 12)
    last_step_abs_err = np.abs(X_val_clean[:, -1, :] - val_recon_clean[:, -1, :])
    # Scale robuste pentru normalizarea XAI (IQR + marjă minimă)
    q75_err = np.percentile(last_step_abs_err, 75, axis=0)
    q25_err = np.percentile(last_step_abs_err, 25, axis=0)
    xai_scales = np.maximum(q75_err - q25_err, 0.25)

    norm_err = last_step_abs_err / xai_scales
    val_scores = np.mean(norm_err, axis=1)
    validation_p95 = float(np.percentile(val_scores, 95))
    validation_p99 = float(np.percentile(val_scores, 99))

    global_threshold = float(np.percentile(val_scores, 98.0) * 1.25)
    stream_scores: dict[str, list[float]] = {}
    for idx, (dev, topic, _) in enumerate(val_items_clean):
        stream_key = f"{dev}::{topic}"
        stream_scores.setdefault(stream_key, []).append(float(val_scores[idx]))

    stream_thresholds: dict[str, float] = {}
    for skey, sc_list in stream_scores.items():
        if len(sc_list) >= 20:
            stream_thresholds[skey] = round(float(np.percentile(sc_list, 98.0) * 1.25), 6)
        else:
            stream_thresholds[skey] = round(global_threshold, 6)

    # CUSUM calibration is deliberately derived from the training split only.
    # The held-out validation session contributes only reconstruction-threshold
    # calibration below; it cannot change the CUSUM baseline statistics.
    cusum_baselines = compute_cusum_baselines(raw_values_by_session, train_sessions)
    candidate_session_ids = train_sessions + val_sessions + heldout_sessions
    curation_digests = _load_curation_digests(db_dir, candidate_session_ids)

    bdir = Path(bundle_dir).resolve()
    bdir.mkdir(parents=True, exist_ok=True)
    onnx_path = bdir / "model.onnx"

    dummy_in = torch.zeros((1, seq_len, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_in,
        str(onnx_path),
        input_names=["window_input"],
        output_names=["window_recon"],
        dynamic_axes={"window_input": {0: "batch"}, "window_recon": {0: "batch"}},
        opset_version=17,
    )

    (bdir / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "seq_len": seq_len,
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bdir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    (bdir / "thresholds.json").write_text(
        json.dumps(
            {
                "calibration_method": "p98_val_l1_x1.25",
                "global_threshold": round(global_threshold, 6),
                "stream_thresholds": stream_thresholds,
                "xai_error_scales": [round(float(v), 6) for v in xai_scales],
                "cusum_baselines": cusum_baselines,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Bind candidate model/schema/scaler/threshold bytes before shadow evaluation.
    candidate_files_sha256 = hash_unsigned_candidate_artifacts(bdir)

    metadata = {
        "model_version": f"v6-candidate-{run_id}" if not sign_bundle else f"v5-candidate-{run_id}",
        "architecture": "TemporalLSTMAutoencoder(12->32->16->32->12)",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at_unix": int(time.time()),
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "heldout_sessions": heldout_sessions,
        "cusum_calibration_sessions": train_sessions,
        "train_windows": len(X_train),
        "val_windows": len(X_val_clean),
        "final_train_loss": round(final_train_loss, 6),
        "global_threshold": round(global_threshold, 6),
        "validation_normal_error_p95": round(validation_p95, 8),
        "validation_normal_error_p99": round(validation_p99, 8),
        "dataset_policy": "approved_clean_baseline_only",
        "offline_training": True,
        "candidate_files_sha256": candidate_files_sha256,
        "curation_digests": curation_digests,
    }
    if require_independent_capture_runs:
        metadata.update({
            "split_unit": "capture_run_id",
            "capture_run_ids_by_session": capture_run_ids_by_session,
            "train_capture_runs": train_capture_runs,
            "val_capture_runs": val_capture_runs,
            "heldout_capture_runs": heldout_capture_runs,
        })
    (bdir / "manifest.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if not sign_bundle:
        # Candidate remains unsigned until offline shadow evaluation passes.
        # The separate authority worker is the only caller allowed to sign it.
        return {**metadata, "candidate_dir": str(bdir), "signed": False}

    kdir = Path(key_dir).resolve()
    priv_key = kdir / "model_signing_ed25519.key"
    pub_key = kdir / "model_signing_ed25519.pub"
    if not priv_key.is_file() or not pub_key.is_file():
        raise RuntimeError(
            "Model signing keypair is not provisioned. Create it offline; "
            "training must never replace the runtime trust anchor automatically."
        )
    return sign_model_bundle(bdir, priv_key, metadata=metadata)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train local LSTM Autoencoder and sign ONNX bundle.")
    parser.add_argument("--db-dir", default="logs/db")
    parser.add_argument("--bundle-dir", default=None,
                        help="Optional explicit output; default is models/candidates/<run_id>")
    parser.add_argument("--key-dir", default="config/model_keys")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--unsigned-candidate", action="store_true",
                        help="Write an unsigned local candidate for the separate promotion authority")
    args = parser.parse_args()

    res = train_and_export_bundle(
        db_dir=args.db_dir,
        bundle_dir=args.bundle_dir,
        key_dir=args.key_dir,
        seq_len=args.seq_len,
        epochs=args.epochs,
        sign_bundle=not args.unsigned_candidate,
    )
    print(json.dumps(res, indent=2, sort_keys=True))