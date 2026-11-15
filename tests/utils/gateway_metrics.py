"""Read and compare gateway JSONL snapshots without mixing process runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

COUNTERS = (
    "ingress_received", "unknown_dropped", "blocklist_dropped",
    "rate_topic_dropped", "rate_global_dropped", "oversize_dropped",
    "sensor_invalid_dropped", "sensor_queue_full_dropped", "sensor_queue_stale_dropped",
    "sensor_queued", "sensor_forwarded", "actuator_queued", "actuator_queue_full_dropped",
    "actuator_verified", "actuator_auth_rejected", "actuator_replay_rejected",
    "actuator_forwarded", "publish_errors", "callback_errors",
    "mitigation_queue_full", "emergency_blocks", "blocks_issued",
    "device_rate_dropped", "device_blocklist_dropped", "device_unknown_dropped",
    "device_blocks_issued", "device_alert_queue_full",
)
DROP_COUNTERS = (
    "unknown_dropped", "blocklist_dropped", "rate_topic_dropped", "rate_global_dropped",
    "oversize_dropped", "sensor_invalid_dropped", "sensor_queue_full_dropped",
    "sensor_queue_stale_dropped", "actuator_queue_full_dropped",
    "device_rate_dropped", "device_blocklist_dropped", "device_unknown_dropped",
)
QUEUE_FIELDS = ("sensor_queue_depth", "actuator_queue_depth", "mitigation_queue_depth")


def read_latest_snapshot(path: str | Path) -> Optional[dict]:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    latest = None
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("run_id"), str):
                    latest = obj
    except OSError:
        return None
    return latest


def snapshot_signature(snapshot: dict) -> tuple:
    blocks = snapshot.get("device_blocks", {})
    block_ids = tuple(sorted(blocks)) if isinstance(blocks, dict) else ()
    accepted = snapshot.get("device_publish_accepted", {})
    accepted_sig = tuple(sorted((str(k), int(v or 0)) for k, v in accepted.items())) if isinstance(accepted, dict) else ()
    return tuple(snapshot.get(k, 0) for k in COUNTERS + QUEUE_FIELDS) + (
        snapshot.get("blocks_active", 0), snapshot.get("device_blocks_active", 0),
        block_ids, accepted_sig,
    )


def calculate_delta(before: Optional[dict], after: Optional[dict]) -> Optional[dict]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    if before.get("run_id") != after.get("run_id"):
        return None
    result = {
        name: max(0, int(after.get(name, 0) or 0) - int(before.get(name, 0) or 0))
        for name in COUNTERS
    }
    for field in QUEUE_FIELDS:
        result[field] = int(after.get(field, 0) or 0)
    result["run_id"] = after["run_id"]
    result["blocks_active"] = int(after.get("blocks_active", 0) or 0)
    result["device_blocks_active"] = int(after.get("device_blocks_active", 0) or 0)
    result["device_blocks"] = after.get("device_blocks", {})
    before_accepted = before.get("device_publish_accepted", {})
    after_accepted = after.get("device_publish_accepted", {})
    if isinstance(before_accepted, dict) and isinstance(after_accepted, dict):
        keys = set(before_accepted) | set(after_accepted)
        result["device_publish_accepted"] = {
            key: max(0, int(after_accepted.get(key, 0) or 0) - int(before_accepted.get(key, 0) or 0))
            for key in keys
        }
    else:
        result["device_publish_accepted"] = {}
    result["device_blocks_issued_total"] = int(after.get("device_blocks_issued_total", 0) or 0)
    result["dropped_total"] = sum(result[name] for name in DROP_COUNTERS)
    return result
