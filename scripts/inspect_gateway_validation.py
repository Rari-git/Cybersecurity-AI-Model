"""Summarize the latest gateway build marker and metrics snapshot."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from utils.gateway_metrics import read_latest_snapshot

LOG_PATH = Path(os.environ.get("LUMEN_GATEWAY_LOG", "logs/gateway_ids.log"))
METRICS_PATH = Path(os.environ.get("LUMEN_GATEWAY_METRICS_PATH", "logs/gateway_metrics.jsonl"))


def main() -> int:
    if not LOG_PATH.is_file():
        print(f"Missing gateway log: {LOG_PATH}")
        return 2
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    build_line = next((line for line in reversed(lines) if "[GATEWAY BUILD]" in line), None)
    ready_line = next((line for line in reversed(lines) if "[GATEWAY READY]" in line), None)
    if not build_line or "patch02-completion" not in build_line:
        print("FAIL: latest gateway log does not contain the patch02-completion build marker")
        print("Latest build line:", build_line)
        return 1
    build_match = re.search(r"run_id=([A-Za-z0-9_-]+)", build_line)
    ready_match = re.search(r"run_id=([A-Za-z0-9_-]+)", ready_line or "")
    snapshot = read_latest_snapshot(METRICS_PATH)
    if not snapshot:
        print(f"FAIL: no valid gateway metrics snapshot in {METRICS_PATH}")
        return 1
    build_id = build_match.group(1) if build_match else None
    ready_id = ready_match.group(1) if ready_match else None
    metrics_id = snapshot.get("run_id")
    print("Gateway build:", build_line)
    print("Gateway ready:", ready_line)
    print("Metrics run_id:", metrics_id)
    print("Ingress received:", snapshot.get("ingress_received"))
    print("Sensor forwarded:", snapshot.get("sensor_forwarded"))
    print("Topic-rate drops:", snapshot.get("rate_topic_dropped"))
    print("Global-rate drops:", snapshot.get("rate_global_dropped"))
    print("Blocklist drops:", snapshot.get("blocklist_dropped"))
    print("Blocks issued:", snapshot.get("blocks_issued"))
    print("Active blocks:", snapshot.get("blocks_active"))
    if not (build_id and build_id == ready_id == metrics_id):
        print("FAIL: build, ready and latest metrics run IDs do not match")
        return 1
    print("PASS: latest log and metrics all belong to the same patch02-completion gateway process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
