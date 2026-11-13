"""Virtual sensor producer using the Session.put path validated by Zenoh probe V2."""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger
from utils.sensor_publish import publish_sensor_sample

logger = get_script_logger("virtual_sensors")
PREFIX = os.environ.get("LUMEN_SENSOR_PREFIX", "home").strip("/")


def run_virtual_sensors() -> None:
    logger.info("[SIMULATOR] Inițializare senzori virtuali (prefix '%s')...", PREFIX)
    session = None
    try:
        session = open_mtls_client(os.environ.get("LUMEN_CERT_IDENTITY", "sensors"))
        logger.info("[SIMULATOR] Conectat cu identitatea mTLS 'sensors'.")
        topics = {
            "door": f"{PREFIX}/sensors/door/state",
            "pir": f"{PREFIX}/sensors/pir/motion",
            "gas": f"{PREFIX}/sensors/gas/ppm",
            "vibration": f"{PREFIX}/sensors/vibration/shock",
        }
        invalid = [topic for topic in topics.values() if topic not in {
            "ingest/sensors/door/state", "ingest/sensors/pir/motion",
            "ingest/sensors/gas/ppm", "ingest/sensors/vibration/shock",
            "home/sensors/door/state", "home/sensors/pir/motion",
            "home/sensors/gas/ppm", "home/sensors/vibration/shock",
        }]
        if invalid:
            raise ValueError(f"LUMEN_SENSOR_PREFIX produces unsupported topics: {invalid}")

        door_states = ("CLOSED", "OPEN")
        logger.info("[SIMULATOR] Publishing by Session.put() (no persistent Publisher handles).")
        while True:
            gas = str(random.randint(300, 420))
            publish_sensor_sample(session, topics["gas"], gas)
            time.sleep(random.uniform(0.4, 0.7))

            vibration = f"{random.uniform(0.0, 0.04):.3f}"
            publish_sensor_sample(session, topics["vibration"], vibration)
            time.sleep(random.uniform(0.4, 0.7))

            if random.random() < 0.15:
                state = random.choice(door_states)
                publish_sensor_sample(session, topics["door"], state)
                logger.info("[EVENT] Senzor Ușă: %s", state)
                time.sleep(0.2)

            if random.random() < 0.25:
                publish_sensor_sample(session, topics["pir"], "MOTION_DETECTED")
                logger.info("[EVENT] Senzor PIR: Mișcare detectată")
                time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("[SIMULATOR] Oprire generator senzori.")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("[SIMULATOR] Session close failed")


if __name__ == "__main__":
    run_virtual_sensors()