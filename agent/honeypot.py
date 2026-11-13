import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import zenoh

from agent.enforcement import sanitize
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("honeypot")

HONEYPOT_IDENTITY = os.environ.get("LUMEN_CERT_IDENTITY", "honeypot")


def _open_honeypot_session():
    """Deschide sesiunea mTLS hot-reload pentru identitatea 'honeypot'. Fail-closed la eroare."""
    try:
        session = open_mtls_client(HONEYPOT_IDENTITY)
    except Exception as e:
        logger.critical(f"[HONEYPOT EROARE FATALĂ] Conexiune mTLS eșuată (certificat invalid / respins / "
                        f"router indisponibil): {e}")
        sys.exit(1)
    logger.info(f"[HONEYPOT] Conectat ca client mTLS la routerul Zenoh cu identitatea '{HONEYPOT_IDENTITY}'.")
    return session


class ZenohHoneypot:
    def __init__(self, session=None, enforcement=None):
        self.session = session
        self.enforcement = enforcement
        self._owns_session = False
        self.trap_topics = [
            "home/admin/config",
            "home/security/keys",
            "system/internal/logs"
        ]
        self.queryables = []

    def _make_handler(self, trap_topic: str):
        plausible_payload = json.dumps({
            "status": "error", "code": 403, "message": "Access Denied: Insufficient Privileges"
        }).encode("utf-8")

        def handler(query: zenoh.Query):
            logger.warning(f"[HONEYPOT TRAP ACTIVATED] Probe/Scan detectat pe momeala: '{trap_topic}'")

            try:
                if self.enforcement:
                    try:
                        detail = sanitize(f"honeypot query sel={query.key_expr} params={query.parameters}", 120)
                    except Exception:
                        detail = "honeypot query"
                    self.enforcement.execute_mitigation(
                        "Reconnaissance / Scanning", trap_topic, loss=99.0, payload=detail, critical=True
                    )
            except Exception as e:
                logger.exception(f"[EROARE HONEYPOT] Eșec la emiterea alertei pentru '{trap_topic}': {e}")
            finally:
                try:
                    query.reply(trap_topic, plausible_payload)
                except Exception as e:
                    logger.warning(f"[HONEYPOT] Reply eșuat pe '{trap_topic}' (query expirat?): {e}")

        return handler

    def start(self):
        if not self.session:
            self.session = _open_honeypot_session()
            self._owns_session = True

        if not self.enforcement:
            logger.warning("[HONEYPOT] EnforcementEngine lipsă: sondele vor primi reply dar NU vor genera alerte.")

        logger.info("[HONEYPOT] Se activează capcanele (Queryables)...")
        failed = []
        for target in self.trap_topics:
            try:
                q = self.session.declare_queryable(target, self._make_handler(target))
                self.queryables.append(q)
                logger.info(f"  ├─ Capcană activă pe: '{target}'")
            except Exception as e:
                logger.critical(f"  └─ Eșec la declararea capcanei pe '{target}': {e}")
                failed.append(target)

        if failed:
            logger.critical(f"[HONEYPOT EROARE FATALĂ] Capcane nedeclarate: {failed}. Oprire (fail-closed).")
            self.stop()
            if self._owns_session:
                try:
                    self.session.close()
                except Exception:
                    pass
            sys.exit(1)

    def stop(self):
        for q in self.queryables:
            try:
                q.undeclare()
            except Exception:
                pass
        self.queryables.clear()


HoneypotSink = ZenohHoneypot

if __name__ == "__main__":
    from agent.enforcement import EnforcementEngine

    session = _open_honeypot_session()
    try:
        enforcement = EnforcementEngine(session)
    except Exception as e:
        logger.critical(f"[HONEYPOT EROARE FATALĂ] Inițializare EnforcementEngine eșuată: {e}")
        try:
            session.close()
        except Exception:
            pass
        sys.exit(1)

    sink = ZenohHoneypot(session=session, enforcement=enforcement)
    sink.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[HONEYPOT] Oprire Honeypot Sink...")
    finally:
        sink.stop()
        try:
            session.close()
        except Exception:
            pass