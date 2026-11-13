import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import threading
from utils.crypto_utils import CommandSigner
from utils.replay_store import ReplayStore
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("virtual_actuators")


class SmartDoorLockActuator:
    """Zăvor inteligent: validează semnătura Ed25519/HMAC (legată de topic) + nonce + timestamp înainte de a schimba starea."""
    def __init__(self):
        self.target_topic = "home/actuators/door/lock"
        self.state = "LOCKED"
        self._command_lock = threading.Lock()
        CommandSigner.validate_trust_store()
        # Actuatorul are propriul jurnal durabil de replay, separat de gateway și IDS.
        self.replay_store = ReplayStore.for_component("actuator")

        try:
            self.session = open_mtls_client(os.environ.get("LUMEN_CERT_IDENTITY", "actuator"))
            logger.info(f"[ACTUATOR] Nod Zăvor Ușă inițializat (mTLS). Stare curentă: {self.state}")
        except Exception as e:
            logger.error(f"[ACTUATOR ERROR] Conectare eșuată la router: {e}")
            sys.exit(1)

    def _command_listener(self, sample):
        payload_str = sample.payload.to_string()
        logger.info(f"[ACTUATOR I/O] Pachet primit pe '{sample.key_expr}'")

        with self._command_lock:
            is_valid, reason = CommandSigner.verify_command(
                self.target_topic, payload_str, replay_store=self.replay_store
            )
            if is_valid:
                try:
                    data = json.loads(payload_str)
                    self.state = "UNLOCKED" if data["action"] == "UNLOCK" else "LOCKED"
                    logger.info(f"  └─ 🟢 [COMANDĂ ACCEPTATĂ] Stare: '{self.state}' (Nonce: {data['nonce']})")
                except Exception as exc:
                    logger.error(f"  └─ 🔴 [EROARE PARSARE]: {exc}")
            else:
                logger.error(f"  └─ 🔴 [COMANDĂ BLOCATĂ] Respinsă de firmware actuator: {reason}")

    def start(self):
        self.sub = self.session.declare_subscriber(self.target_topic, self._command_listener)
        logger.info(f"[ACTUATOR] Ascultare activă pentru comenzi pe '{self.target_topic}'...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[ACTUATOR] Oprire nod hardware.")
        finally:
            try:
                self.sub.undeclare()
            except Exception:
                pass
            self.session.close()
            self.replay_store.close()


if __name__ == "__main__":
    SmartDoorLockActuator().start()