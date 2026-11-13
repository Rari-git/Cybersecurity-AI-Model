import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
from utils.crypto_utils import CommandSigner, MAX_TIME_DRIFT_SEC
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("attack_2_replay")

TARGET_TOPIC = "home/actuators/door/lock"
INJECT_TOPIC = f"{os.environ.get('LUMEN_ATTACK_PREFIX', 'home').strip('/')}/actuators/door/lock"
OTHER_TOPIC = "home/actuators/garage/door"
SETTLE_SEC = 2.5


class ReplayAttack:
    def __init__(self):
        try:
            self.session = open_mtls_client("lab_tester")
            logger.info(f"[ATAC 2 - REPLAY] Conectat la router (mTLS). Injectare pe '{INJECT_TOPIC}'.")
        except Exception as e:
            logger.error(f"[EROARE ATAC] Conexiune eșuată: {e}")
            sys.exit(1)

        self.alerts = []
        self.sub = self.session.declare_subscriber("home/security/alerts", self._on_alert)
        time.sleep(1.0)

    def _on_alert(self, sample):
        try:
            raw = sample.payload.to_bytes().decode('utf-8', errors='ignore')
            try:
                data = json.loads(raw)
            except ValueError:
                data = {"raw": raw}
            self.alerts.append(data)
            logger.info(f" 📩 [ALERTĂ IDS] {raw}")
        except Exception as e:
            logger.error(f"Eroare la recepția alertei: {e}")

    def _send(self, frame: str):
        self.session.put(INJECT_TOPIC, frame.encode('utf-8'))

    def _alerts_since(self, idx: int):
        return [a for a in self.alerts[idx:] if a.get("topic") == TARGET_TOPIC]

    @staticmethod
    def _has_reason(alerts, needle: str) -> bool:
        return any(needle in str(a.get("detail", "")).upper() for a in alerts)

    def run_attack(self):
        results = {}

        # Pasul -1: Sanity Check pe reactivitatea IDS-ului
        logger.warning("[ATAC 2] Pas -1: verificare că IDS răspunde (cadru invalid trebuie să alerteze)")
        start = len(self.alerts)
        self._send("not-json")
        time.sleep(SETTLE_SEC)
        if not self._has_reason(self._alerts_since(start), "MALFORMED"):
            logger.error(" [STATUS] ❌ IDS nu a reacționat la un cadru invalid. Testul nu e valid "
                         "(agent suprasolicitat sau oprit; nu rula atacul 1 în paralel).")
            self.sub.undeclare()
            self.session.close()
            return False

        logger.warning("[ATAC 2] Pas 0: comandă validă semnată (control, nu trebuie să alerteze)")
        start = len(self.alerts)
        valid_frame = CommandSigner.sign_command(TARGET_TOPIC, "UNLOCK", sender_id="controller")
        self._send(valid_frame)
        time.sleep(SETTLE_SEC)
        results["Comandă validă fără alertă (fără fals pozitiv)"] = len(self._alerts_since(start)) == 0

        logger.warning("[ATAC 2] Pas 1: replay al aceluiași cadru (nonce duplicat)")
        start = len(self.alerts)
        self._send(valid_frame)
        time.sleep(SETTLE_SEC)
        results["Replay nonce duplicat detectat"] = self._has_reason(self._alerts_since(start), "DUPLICATE_NONCE")

        logger.warning(f"[ATAC 2] Pas 2: cadru valid dar expirat (așteptare {MAX_TIME_DRIFT_SEC + 0.5:.1f}s)")
        stale_frame = CommandSigner.sign_command(TARGET_TOPIC, "UNLOCK", sender_id="controller")
        time.sleep(MAX_TIME_DRIFT_SEC + 0.5)
        start = len(self.alerts)
        self._send(stale_frame)
        time.sleep(SETTLE_SEC)
        results["Timestamp expirat detectat"] = self._has_reason(self._alerts_since(start), "EXPIRED")

        logger.warning("[ATAC 2] Pas 3: cadru modificat (semnătură invalidă)")
        tampered = json.loads(CommandSigner.sign_command(TARGET_TOPIC, "LOCK", sender_id="controller"))
        tampered["action"] = "UNLOCK"
        start = len(self.alerts)
        self._send(json.dumps(tampered))
        time.sleep(SETTLE_SEC)
        results["Semnătură Ed25519 invalidă detectată"] = self._has_reason(self._alerts_since(start), "INVALID_SIGNATURE")

        logger.warning("[ATAC 2] Pas 4: cadru semnat valid pentru ALT actuator, reluat pe ușă (cross-topic)")
        cross_frame = CommandSigner.sign_command(OTHER_TOPIC, "UNLOCK", sender_id="controller")
        start = len(self.alerts)
        self._send(cross_frame)
        time.sleep(SETTLE_SEC)
        results["Replay cross-topic respins (topic legat de semnătură)"] = (
            self._has_reason(self._alerts_since(start), "INVALID_SIGNATURE")
            or self._has_reason(self._alerts_since(start), "TOPIC_NOT_AUTHORIZED")
        )

        logger.info("==================================================")
        logger.info(" REZULTAT TEST ATAC 2 (Persistent Replay / Ed25519 / Cross-Topic)")
        all_ok = True
        for name, ok in results.items():
            all_ok &= ok
            logger.info(f" [{'OK  ' if ok else 'FAIL'}] {name}")
        if all_ok:
            logger.info(" [STATUS] 🛡️ TOATE VERIFICĂRILE AU TRECUT.")
        else:
            logger.error(" [STATUS] ❌ UNELE VERIFICĂRI AU EȘUAT (vezi mai sus).")
        logger.info("==================================================")

        self.sub.undeclare()
        self.session.close()
        return all_ok


if __name__ == "__main__":
    sys.exit(0 if ReplayAttack().run_attack() else 1)