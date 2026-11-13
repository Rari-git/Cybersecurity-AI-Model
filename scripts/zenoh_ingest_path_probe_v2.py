"""Correlate each sensor put with exact/wildcard subscribers and live gateway metrics.

Run from the project root while run_all.py is active. It uses a valid gas sensor
payload (300-450 ppm) with a unique value for every sample, allowing us to tell
which publishing API delivered each message. It performs only six low-rate puts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from utils.zenoh_conn import open_client_session, DEFAULT_ENDPOINT

# Fallback sau preluare directă pentru CERT_DIR
CERT_DIR = os.environ.get("LUMEN_CERT_DIR", "certs")

TOPIC = "ingest/sensors/gas/ppm"
METRICS_PATH = Path(os.environ.get("LUMEN_GATEWAY_METRICS_PATH", "logs/gateway_metrics.jsonl"))
WAIT_PER_SAMPLE_SEC = 2.0

lock = threading.Lock()
received = {"wildcard": defaultdict(list), "exact": defaultdict(list)}
events = {"wildcard": threading.Event(), "exact": threading.Event()}


def latest_metrics():
    if not METRICS_PATH.is_file():
        return None
    latest = None
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and isinstance(item.get("run_id"), str):
                    latest = item
    except OSError:
        return None
    return latest


def callback(kind):
    def _handle(sample):
        try:
            key = str(sample.key_expr)
            payload = sample.payload.to_bytes().decode("utf-8", errors="strict")
        except Exception as exc:
            key, payload = "<decode-error>", f"{type(exc).__name__}:{exc}"
        with lock:
            received[kind][payload].append({"key": key, "at": time.time()})
            events[kind].set()
    return _handle


def wait_for_pair(payload: str, timeout: float = WAIT_PER_SAMPLE_SEC) -> tuple[bool, bool]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with lock:
            wildcard_seen = bool(received["wildcard"].get(payload))
            exact_seen = bool(received["exact"].get(payload))
        if wildcard_seen and exact_seen:
            return True, True
        time.sleep(0.02)
    with lock:
        return (bool(received["wildcard"].get(payload)),
                bool(received["exact"].get(payload)))


def main() -> int:
    print("=== LUMEN ZENOH INGEST PATH PROBE V2 ===")
    print(f"endpoint={os.environ.get('LUMEN_ZENOH_ENDPOINT', DEFAULT_ENDPOINT)!r}")
    print(f"cert_dir={os.environ.get('LUMEN_CERT_DIR', str(CERT_DIR))!r}")
    print(f"topic={TOPIC!r}; payload values are valid gas ppm values")
    baseline = latest_metrics()
    if baseline:
        print(f"gateway_metrics_baseline_run_id={baseline.get('run_id')} timestamp={baseline.get('timestamp')}")
    else:
        print(f"gateway_metrics_baseline=UNAVAILABLE ({METRICS_PATH})")

    sensor = gateway = None
    wildcard = exact = publisher = None
    try:
        print("[1] Open sensor and gateway mTLS sessions")
        sensor = open_client_session("sensors")
        print("    SENSOR_SESSION=CONNECTED")
        gateway = open_client_session("gateway")
        print("    GATEWAY_SESSION=CONNECTED")

        print("[2] Declare exact and wildcard subscribers")
        wildcard = gateway.declare_subscriber("ingest/**", callback("wildcard"))
        exact = gateway.declare_subscriber(TOPIC, callback("exact"))
        time.sleep(0.75)
        print("    WILDCARD_SUBSCRIBER=DECLARED; EXACT_SUBSCRIBER=DECLARED")

        results = []
        print("[3] Test direct Session.put with individually identifiable valid payloads")
        for value in ("351", "352", "353"):
            status = "RETURNED"
            try:
                sensor.put(TOPIC, value.encode("ascii"))
            except Exception as exc:
                status = f"ERROR:{type(exc).__name__}:{exc}"
            w, e = wait_for_pair(value)
            results.append(("session.put", value, status, w, e))
            print(f"    session.put payload={value} result={status} wildcard={w} exact={e}")

        print("[4] Test declared Publisher.put with individually identifiable valid payloads")
        publisher = sensor.declare_publisher(TOPIC)
        time.sleep(0.75)
        for value in ("361", "362", "363"):
            status = "RETURNED"
            try:
                publisher.put(value)
            except Exception as exc:
                status = f"ERROR:{type(exc).__name__}:{exc}"
            w, e = wait_for_pair(value)
            results.append(("Publisher.put", value, status, w, e))
            print(f"    Publisher.put payload={value} result={status} wildcard={w} exact={e}")

        print("[5] Wait for gateway metrics to flush")
        final_metrics = None
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            candidate = latest_metrics()
            if candidate and baseline and candidate.get("run_id") == baseline.get("run_id"):
                try:
                    if float(candidate.get("timestamp", 0)) > float(baseline.get("timestamp", 0)):
                        final_metrics = candidate
                        break
                except (TypeError, ValueError):
                    pass
            time.sleep(0.25)
        if baseline and final_metrics:
            names = (
                "ingress_received", "sensor_queued", "sensor_forwarded",
                "rate_topic_dropped", "rate_global_dropped", "blocklist_dropped",
                "sensor_queue_full_dropped", "publish_errors", "callback_errors",
            )
            print(f"    gateway_run_id={final_metrics.get('run_id')}")
            for name in names:
                before = int(baseline.get(name, 0) or 0)
                after = int(final_metrics.get(name, 0) or 0)
                print(f"    gateway_delta_{name}={max(0, after-before)}")
        else:
            print("    GATEWAY_METRICS_DELTA=UNAVAILABLE (run_id changed or no post-probe snapshot)")

        all_delivered = all(row[3] and row[4] for row in results)
        direct_delivered = all(row[3] and row[4] for row in results if row[0] == "session.put")
        publisher_delivered = all(row[3] and row[4] for row in results if row[0] == "Publisher.put")
        print("[6] Summary")
        print(f"    direct_session_put_all_delivered={direct_delivered}")
        print(f"    declared_publisher_put_all_delivered={publisher_delivered}")
        print(f"    all_six_delivered_to_both_subscriptions={all_delivered}")
        if all_delivered:
            print("RESULT=PASS_SENSOR_TO_ROUTER_TO_BOTH_GATEWAY_SUBSCRIPTIONS")
            return 0
        if direct_delivered and not publisher_delivered:
            print("RESULT=PUBLISHER_API_DELIVERY_PROBLEM")
            return 2
        if publisher_delivered and not direct_delivered:
            print("RESULT=SESSION_PUT_DELIVERY_PROBLEM")
            return 2
        print("RESULT=PARTIAL_OR_NO_ROUTER_SUBSCRIBER_DELIVERY")
        return 1
    except Exception as exc:
        print(f"RESULT=SETUP_ERROR {type(exc).__name__}: {exc}")
        return 3
    finally:
        for entity in (publisher, wildcard, exact):
            if entity is not None:
                try:
                    entity.undeclare()
                except Exception:
                    pass
        for session in (sensor, gateway):
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
