"""Diagnose sensors -> Zenoh router -> gateway subscriber delivery.

Run from the project root while the router is up. This opens two independent
mTLS sessions in one process: `sensors` publishes and `gateway` subscribes.
It tests both the gateway's actual wildcard subscription (`ingest/**`) and an
exact subscription, and compares direct session.put() with Publisher.put().
This is a diagnostic, not a load test.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter

from utils.zenoh_conn import open_client_session, DEFAULT_ENDPOINT

# Fallback sau preluare directă pentru CERT_DIR
CERT_DIR = os.environ.get("LUMEN_CERT_DIR", "certs")

TOPIC = "ingest/sensors/door/state"
PAYLOAD = "OPEN"
SAMPLES_PER_METHOD = 3
WAIT_SEC = 6.0

lock = threading.Lock()
received = Counter()
first_samples = []
all_received = threading.Event()


def _make_handler(kind: str):
    def _handler(sample):
        try:
            key = str(sample.key_expr)
            raw = sample.payload.to_bytes()
            text = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            key, text = "<decode-error>", f"{type(exc).__name__}: {exc}"
        with lock:
            received[kind] += 1
            if len(first_samples) < 12:
                first_samples.append((kind, key, text, time.time()))
            if received["wildcard"] >= SAMPLES_PER_METHOD * 2 and received["exact"] >= SAMPLES_PER_METHOD * 2:
                all_received.set()
    return _handler


def main() -> int:
    print("=== LUMEN ZENOH INGEST PATH PROBE ===")
    print(f"endpoint_env={os.environ.get('LUMEN_ZENOH_ENDPOINT')!r}")
    print(f"endpoint_default={DEFAULT_ENDPOINT!r}")
    print(f"cert_dir_env={os.environ.get('LUMEN_CERT_DIR')!r}")
    print(f"cert_dir_loaded={CERT_DIR!r}")
    print(f"topic={TOPIC!r} payload={PAYLOAD!r}")

    sensor = gateway = None
    wildcard_sub = exact_sub = publisher = None
    try:
        print("[1] Opening sensors mTLS session...")
        sensor = open_client_session("sensors")
        print("    SENSOR_SESSION=CONNECTED")

        print("[2] Opening gateway mTLS session...")
        gateway = open_client_session("gateway")
        print("    GATEWAY_SESSION=CONNECTED")

        print("[3] Declaring gateway wildcard subscription ingest/**...")
        wildcard_sub = gateway.declare_subscriber("ingest/**", _make_handler("wildcard"))
        print("    WILDCARD_SUBSCRIBER=DECLARED")

        print("[4] Declaring gateway exact subscription...")
        exact_sub = gateway.declare_subscriber(TOPIC, _make_handler("exact"))
        print("    EXACT_SUBSCRIBER=DECLARED")

        time.sleep(0.5)
        print("[5] Sending direct session.put() samples...")
        direct_ok = 0
        for _ in range(SAMPLES_PER_METHOD):
            try:
                sensor.put(TOPIC, PAYLOAD.encode("utf-8"))
                direct_ok += 1
            except Exception as exc:
                print(f"    DIRECT_PUT_ERROR={type(exc).__name__}: {exc}")
            time.sleep(0.1)
        print(f"    DIRECT_PUT_RETURNED={direct_ok}/{SAMPLES_PER_METHOD}")

        print("[6] Declaring publisher and sending Publisher.put() samples...")
        publisher = sensor.declare_publisher(TOPIC)
        pub_ok = 0
        for _ in range(SAMPLES_PER_METHOD):
            try:
                publisher.put(PAYLOAD)
                pub_ok += 1
            except Exception as exc:
                print(f"    PUBLISHER_PUT_ERROR={type(exc).__name__}: {exc}")
            time.sleep(0.1)
        print(f"    PUBLISHER_PUT_RETURNED={pub_ok}/{SAMPLES_PER_METHOD}")

        all_received.wait(WAIT_SEC)
        with lock:
            snapshot = dict(received)
            samples = list(first_samples)

        print("[7] Delivery summary")
        print(f"    direct_put_returned={direct_ok}")
        print(f"    publisher_put_returned={pub_ok}")
        print(f"    wildcard_callbacks={snapshot.get('wildcard', 0)}")
        print(f"    exact_callbacks={snapshot.get('exact', 0)}")
        for kind, key, text, stamp in samples:
            print(f"    SAMPLE kind={kind} key={key!r} payload={text!r} epoch={stamp:.6f}")

        wildcard_count = snapshot.get("wildcard", 0)
        exact_count = snapshot.get("exact", 0)
        expected = SAMPLES_PER_METHOD * 2
        if wildcard_count >= expected and exact_count >= expected:
            print("RESULT=PASS_SENSOR_TO_ROUTER_TO_GATEWAY_SUBSCRIBER")
            print("Interpretation: the sensor identity, topic ACL, router delivery and gateway identity subscription work in this diagnostic.")
            return 0
        if wildcard_count == 0 and exact_count == 0:
            print("RESULT=FAIL_NO_SUBSCRIBER_DELIVERY")
            print("Interpretation: investigate router ACL policy, subject CN, publisher identity, endpoint, or router delivery. Put() return is not a delivery ACK.")
            return 2
        print("RESULT=PARTIAL_DELIVERY")
        print("Interpretation: compare direct put with Publisher.put and exact with wildcard callback counts; inspect ACL and key-expression matching.")
        return 1

    except Exception as exc:
        print(f"RESULT=SETUP_ERROR {type(exc).__name__}: {exc}")
        return 3
    finally:
        for entity in (publisher, wildcard_sub, exact_sub):
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
