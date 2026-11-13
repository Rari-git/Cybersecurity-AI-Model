import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("attack_3_scan")

TARGETS = ["home/admin/config", "home/security/keys", "system/internal/logs"]
SCAN_ATTACK = "Reconnaissance / Scanning"
# lab_tester: are drept de query pe capcane (honeypot_query_in) și de abonare la alerte.
ATTACKER_IDENTITY = "lab_tester"


def run_scan_attack():
    try:
        session = open_mtls_client(ATTACKER_IDENTITY)
    except Exception as e:
        logger.error(f"[EROARE ATAC] Conexiune eșuată: {e}")
        return False
    logger.info(f"[ATAC 3 - RECONNAISSANCE] Conectat la router (mTLS) ca '{ATTACKER_IDENTITY}'.")

    alerts = []

    def alert_listener(sample):
        try:
            payload = json.loads(sample.payload.to_bytes().decode('utf-8'))
            if payload.get("attack") == SCAN_ATTACK:     # ignoră alertele altor atacuri rulate în paralel
                alerts.append(payload)
                logger.info(f"  📩 [ALERTĂ PRIMITĂ DE LA IDS] {payload}")
        except Exception:
            pass

    sub = session.declare_subscriber("home/security/alerts", alert_listener)
    time.sleep(1.0)

    traps_hit = 0
    for target in TARGETS:
        logger.warning(f"[SCAN PROBE] Interogare GET pe momeală: '{target}'")
        try:
            hit = False
            for reply in session.get(target, timeout=3.0):   # API zenoh 1.x: iterator FIFO
                ok = reply.ok
                if ok is not None:
                    hit = True
                    logger.info(f"   └─ Răspuns pentru '{target}' | Payload: {ok.payload.to_bytes().decode('utf-8')[:80]}")
            traps_hit += int(hit)
        except Exception as e:
            logger.error(f"   └─ Eșec la interogare '{target}': {e}")
        time.sleep(0.5)

    time.sleep(2.0)

    success = traps_hit == len(TARGETS) and len(alerts) >= len(TARGETS)
    print("\n==================================================")
    print(" REZULTAT TEST ATAC 3 (Reconnaissance / Scan)")
    print(f" Total ținte sondate: {len(TARGETS)}")
    print(f" Capcane Honeypot care au răspuns: {traps_hit}")
    print(f" Alerte de scanare recepționate: {len(alerts)}")
    if success:
        print(" [STATUS] 🛡️ ATAC NEUTRALIZAT CU SUCCES! Honeypot-ul a capturat scanarea și s-au emis alerte.")
    else:
        print(" [STATUS] ❌ TEST EȘUAT - Scanarea nu a fost capturată complet.")
    print("==================================================\n")

    sub.undeclare()
    session.close()
    return success


if __name__ == "__main__":
    sys.exit(0 if run_scan_attack() else 1)