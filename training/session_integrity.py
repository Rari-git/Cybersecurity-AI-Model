"""Verify candidate-bound held-out clean-baseline evidence."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any, Iterable


def load_verified_clean_session_rows(
    storage: Any,
    session_ids: Iterable[str],
    expected_digests: dict[str, str] | None,
) -> dict[str, list[dict[str, Any]]]:
    ids = tuple(dict.fromkeys(str(value) for value in session_ids))
    if not ids or any(not value for value in ids):
        raise ValueError("held-out clean-session verification requires non-empty session IDs")
    if not isinstance(expected_digests, dict):
        raise ValueError("candidate manifest must contain held-out clean-session curation digests")
    if set(ids) - set(expected_digests):
        raise ValueError("candidate manifest is missing one or more held-out curation digests")
    for session_id in ids:
        expected = expected_digests.get(session_id)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(ch not in "0123456789abcdef" for ch in expected.lower())):
            raise ValueError(f"candidate curation digest is invalid for held-out session {session_id}")

    conn = storage.connect_readonly("normal")
    try:
        # One read transaction ensures the verified rows are the rows later scored.
        conn.execute("BEGIN")
        placeholders = ",".join("?" for _ in ids)
        metadata_rows = conn.execute(
            f"""SELECT s.session_id, s.source_type, s.approved_for_training,
                       c.status, c.decision_json, c.dataset_sha256
                FROM baseline_sessions s
                JOIN session_curation c ON c.session_id=s.session_id
                WHERE s.session_id IN ({placeholders})""",
            ids,
        ).fetchall()
        metadata_by_id = {str(row["session_id"]): row for row in metadata_rows}
        if set(metadata_by_id) != set(ids):
            missing = sorted(set(ids) - set(metadata_by_id))
            raise RuntimeError(f"held-out sessions are missing curation records: {missing}")

        verified: dict[str, list[dict[str, Any]]] = {}
        for session_id in ids:
            metadata = metadata_by_id[session_id]
            if (str(metadata["source_type"]) != "baseline_lab"
                    or int(metadata["approved_for_training"]) != 1
                    or str(metadata["status"]) != "approved_clean_baseline"):
                raise RuntimeError(f"held-out session is not an approved clean baseline: {session_id}")
            try:
                decision = json.loads(str(metadata["decision_json"]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"held-out curation decision is malformed: {session_id}") from exc
            if (not isinstance(decision, dict)
                    or decision.get("approved") is not True
                    or decision.get("status") != "approved_clean_baseline"
                    or decision.get("reasons") not in ([], ())):
                raise RuntimeError(f"held-out curation decision is not approved-clean: {session_id}")

            rows = conn.execute(
                """SELECT event_id, device_id, generation, topic, gateway_ts_ns,
                          payload_len, payload_sha256, numeric_value, raw_payload
                   FROM raw_events WHERE session_id=?
                   ORDER BY device_id ASC, topic ASC, event_id ASC""",
                (session_id,),
            ).fetchall()
            if not rows:
                raise RuntimeError(f"held-out clean session contains no telemetry: {session_id}")
            materialized = [dict(row) for row in rows]
            digest = hashlib.sha256()
            # Exact canonical order and tuple shape used by train_local_lstm.py.
            for item in sorted(materialized, key=lambda value: int(value["event_id"])):
                raw_payload = item.get("raw_payload")
                if not isinstance(raw_payload, str):
                    raise RuntimeError(f"held-out raw payload has an invalid type: {session_id}")
                raw_bytes = raw_payload.encode("utf-8")
                payload_len = item.get("payload_len")
                payload_hash = item.get("payload_sha256")
                if (isinstance(payload_len, bool) or not isinstance(payload_len, int)
                        or payload_len != len(raw_bytes)):
                    raise RuntimeError(f"held-out payload length mismatch: {session_id}")
                if not isinstance(payload_hash, (bytes, bytearray, memoryview)) or len(payload_hash) != 32:
                    raise RuntimeError(f"held-out payload hash is malformed: {session_id}")
                if not hmac.compare_digest(hashlib.sha256(raw_bytes).digest(), bytes(payload_hash)):
                    raise RuntimeError(f"held-out payload hash mismatch: {session_id}")
                numeric = item.get("numeric_value")
                if numeric is not None:
                    if (isinstance(numeric, bool) or not isinstance(numeric, (int, float))
                            or not math.isfinite(float(numeric))):
                        raise RuntimeError(f"held-out numeric value is invalid: {session_id}")
                    numeric = float(numeric)
                fields = (
                    int(item["event_id"]), str(item["device_id"]), int(item["generation"]),
                    str(item["topic"]), int(item["gateway_ts_ns"]), bytes(payload_hash).hex(),
                    numeric, raw_payload,
                )
                try:
                    encoded = json.dumps(
                        fields, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                    ).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"held-out telemetry cannot be canonically encoded: {session_id}") from exc
                digest.update(encoded)
                digest.update(b"\n")

            actual = digest.hexdigest()
            stored = str(metadata["dataset_sha256"] or "")
            expected = expected_digests[session_id]
            if (len(stored) != 64
                    or not hmac.compare_digest(actual, stored.lower())
                    or not hmac.compare_digest(actual, expected.lower())):
                raise RuntimeError(f"held-out clean-baseline dataset digest mismatch: {session_id}")
            verified[session_id] = materialized
        return verified
    finally:
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        finally:
            conn.close()
