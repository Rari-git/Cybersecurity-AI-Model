"""Print the latest Patch 03 device-enforcement metrics snapshot."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.gateway_metrics import read_latest_snapshot


def main() -> int:
    path = os.environ.get("LUMEN_GATEWAY_METRICS_PATH", "logs/gateway_metrics.jsonl")
    snapshot = read_latest_snapshot(path)
    if snapshot is None:
        print(f"No valid gateway metrics snapshot: {path}")
        return 2
    print("=== LUMEN DEVICE ENFORCEMENT ===")
    print(f"run_id: {snapshot.get('run_id')}")
    for field in (
        "ingress_received", "sensor_queued", "sensor_forwarded",
        "device_rate_dropped", "device_blocklist_dropped", "device_unknown_dropped",
        "device_blocks_issued", "device_blocks_active", "publish_errors", "callback_errors",
    ):
        print(f"{field}: {snapshot.get(field, 'MISSING — gateway integration not applied')}")
    print("device publish accepted (gateway Session.put returned):")
    for device_id, count in sorted(snapshot.get("device_publish_accepted", {}).items()):
        print(f"  {device_id}: {count}")
    print("active device blocks:")
    blocks = snapshot.get("device_blocks", {})
    if not isinstance(blocks, dict) or not blocks:
        print("  none")
    else:
        for key, item in blocks.items():
            item = item if isinstance(item, dict) else {}
            print(f"  {key}: remaining={item.get('remaining_sec')}s attack={item.get('attack')} CN={item.get('cert_cn')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
