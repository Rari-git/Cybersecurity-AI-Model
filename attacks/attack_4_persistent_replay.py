"""Two-phase live Zenoh test for replay rejection after gateway restart.

Phase 1: record and send one valid signed command, then confirm gateway's SQLite
replay DB consumed its nonce. Restart ONLY the gateway without deleting state/.
Phase 2: resend the exact captured frame and require a duplicate-nonce alert.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.crypto_utils import CommandSigner
from utils.replay_store import ReplayStore
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("attack_4_persistent_replay")
TARGET_TOPIC = "home/actuators/door/lock"
INJECT_TOPIC = f"{os.environ.get('LUMEN_ATTACK_PREFIX', 'ingest').strip('/')}/actuators/door/lock"
FRAME_PATH = Path(os.environ.get("LUMEN_REPLAY_TEST_FRAME", "state/restart_replay_frame.json"))
GATEWAY_DB = os.environ.get("LUMEN_GATEWAY_REPLAY_DB", "state/gateway_replay.sqlite3")


def _gateway_store() -> ReplayStore:
    return ReplayStore.for_component("gateway", GATEWAY_DB)


def _load_frame() -> dict:
    if not FRAME_PATH.is_file():
        raise FileNotFoundError(f"Captured frame not found: {FRAME_PATH}; run the 'record' phase first")
    obj = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    if obj.get("topic") != TARGET_TOPIC or not isinstance(obj.get("frame"), str):
        raise ValueError("Captured frame metadata/topic is invalid")
    obj["packet"] = json.loads(obj["frame"])
    return obj


def record_phase() -> bool:
    frame = CommandSigner.sign_command(TARGET_TOPIC, "UNLOCK", sender_id="controller")
    packet = json.loads(frame)
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "topic": TARGET_TOPIC,
        "frame": frame,
        "nonce": packet["nonce"],
        "sender_id": packet["sender_id"],
        "key_id": packet["key_id"],
        "sha256": hashlib.sha256(frame.encode("utf-8")).hexdigest(),
        "created_at": time.time(),
    }
    tmp_path = FRAME_PATH.with_suffix(FRAME_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, FRAME_PATH)

    store = _gateway_store()
    before = store.get_last_nonce(packet["sender_id"], TARGET_TOPIC)
    if packet["nonce"] <= before:
        raise RuntimeError("new command nonce is not greater than gateway's stored nonce")

    session = open_mtls_client("lab_tester")
    try:
        session.put(INJECT_TOPIC, frame.encode("utf-8"))
    finally:
        session.close()

    deadline = time.monotonic() + 8.0
    consumed = False
    while time.monotonic() < deadline:
        current = store.get_last_nonce(packet["sender_id"], TARGET_TOPIC)
        if current >= packet["nonce"]:
            consumed = True
            break
        time.sleep(0.1)

    logger.info("[REPLAY RESTART] frame_path=%s nonce=%s sha256=%s",
                FRAME_PATH, packet["nonce"], payload["sha256"])
    if consumed:
        logger.info("[PASS] Gateway replay DB consumed nonce=%s. Now stop/restart ONLY the gateway; keep state/ intact.",
                    packet["nonce"])
        return True
    logger.error("[FAIL] Gateway DB did not consume nonce. Check gateway logs, trusted signer topic ACL and DB path.")
    return False


def replay_phase() -> bool:
    saved = _load_frame()
    packet = saved["packet"]
    store = _gateway_store()
    current = store.get_last_nonce(saved["sender_id"], TARGET_TOPIC)
    if current < int(saved["nonce"]):
        raise RuntimeError(
            f"Gateway DB has last_nonce={current}, expected >= captured nonce {saved['nonce']}. "
            "Do not run the replay phase until the record phase has been accepted."
        )

    received = []
    received_event = threading.Event()

    def on_alert(sample):
        try:
            alert = json.loads(sample.payload.to_bytes().decode("utf-8"))
            if (alert.get("topic") == TARGET_TOPIC
                    and alert.get("attack") == "Replay Attack"
                    and "REPLAY_DUPLICATE_NONCE" in str(alert.get("detail", ""))):
                received.append(alert)
                received_event.set()
        except Exception:
            logger.exception("[REPLAY RESTART] malformed alert received")

    session = open_mtls_client("lab_tester")
    sub = session.declare_subscriber("home/security/alerts", on_alert)
    try:
        time.sleep(0.5)
        session.put(INJECT_TOPIC, saved["frame"].encode("utf-8"))
        if not received_event.wait(8.0):
            logger.error("[FAIL] No duplicate-nonce alert received after replay. Check fresh gateway log and subscriber ACL.")
            return False
        logger.info("[PASS] Exact captured frame rejected after gateway restart: %s", received[0].get("detail"))
        return True
    finally:
        try:
            sub.undeclare()
        except Exception:
            pass
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("record", "replay"))
    args = parser.parse_args()
    try:
        ok = record_phase() if args.phase == "record" else replay_phase()
        return 0 if ok else 1
    except Exception as exc:
        logger.exception("[REPLAY RESTART] %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())