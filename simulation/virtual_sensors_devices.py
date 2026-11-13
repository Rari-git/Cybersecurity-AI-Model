"""Run independent virtual sensor devices, each with its own mTLS certificate."""
from __future__ import annotations

import os
import random
import signal
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device_registry import DeviceRegistry
from utils.zenoh_conn import open_client_session
from utils.logger import get_script_logger

logger = get_script_logger("virtual_sensors_devices")


def _payload(payload_type: str) -> str:
    if payload_type == "door":
        return random.choice(("OPEN", "CLOSED"))
    if payload_type == "gas":
        return str(random.randint(300, 420))
    if payload_type == "vibration":
        return f"{random.uniform(0.0, 0.04):.3f}"
    if payload_type == "pir":
        return "MOTION_DETECTED"
    return "OK"


def _interval(payload_type: str) -> float:
    if payload_type == "door":
        return random.uniform(3.0, 6.0)
    if payload_type == "pir":
        return random.uniform(2.0, 4.0)
    return random.uniform(0.8, 1.4)


def run() -> None:
    registry = DeviceRegistry.load(os.environ.get("LUMEN_DEVICE_REGISTRY", "config/devices.json"))
    skip = {item.strip() for item in os.environ.get("LUMEN_SIM_SKIP_DEVICES", "").split(",") if item.strip()}
    selected = [spec for spec in registry.enabled_devices() if spec.device_id not in skip]
    if not selected:
        raise RuntimeError("no enabled virtual devices remain after LUMEN_SIM_SKIP_DEVICES")

    stop_event = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    sessions = []
    sessions_lock = threading.Lock()

    def stop_handler(_signum=None, _frame=None):
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, stop_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, stop_handler)
    except ValueError:
        pass

    def device_loop(spec) -> None:
        session = None
        try:
            session = open_client_session(spec.cert_cn)
            with sessions_lock:
                sessions.append(session)
            logger.info("[DEVICE SIM] connected device_id=%s cert_cn=%s key=%s",
                        spec.device_id, spec.cert_cn, spec.ingress_key)
            while not stop_event.is_set():
                payload = _payload(spec.payload_type)
                session.put(spec.ingress_key, payload.encode("utf-8"))
                logger.info("[SENSOR] device=%s target=%s payload=%s",
                            spec.device_id, spec.target_topic, payload)
                stop_event.wait(_interval(spec.payload_type))
        except Exception as exc:
            logger.exception("[DEVICE SIM] device=%s failed", spec.device_id)
            with errors_lock:
                errors.append(exc)
            stop_event.set()
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    logger.exception("[DEVICE SIM] session close failed for %s", spec.device_id)

    threads = [threading.Thread(target=device_loop, args=(spec,),
                                name=f"device-{spec.device_id}", daemon=True)
               for spec in selected]
    logger.info("[DEVICE SIM] starting %d independently authenticated devices; skipped=%s",
                len(selected), sorted(skip))
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.25)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5.0)
        logger.info("[DEVICE SIM] stopping")
    if errors:
        raise RuntimeError(f"virtual sensor worker failed: {errors[0]}") from errors[0]


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        logger.exception("[DEVICE SIM] startup failed: %s", exc)
        raise SystemExit(1) from exc