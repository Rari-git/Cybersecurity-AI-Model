import time
import json
import threading
from collections import OrderedDict
import zenoh
from utils.logger import get_script_logger
from utils.rate_limit import TokenBucketRateLimiter
from utils.ttl_blocklist import TTLBlocklist

logger = get_script_logger("enforcement")

ALERT_TOPIC = "home/security/alerts"

# Acțiuni de enforcement (valorile apar în câmpul "action" al alertei).
ACTION_ALERT = "ALERT"   # IDS pasiv: doar alertă (ALERT_ONLY)
ACTION_DROP = "DROP"     # Inline Gateway: pachetul curent este eliminat
ACTION_BLOCK = "BLOCK"   # Entitatea/topic-ul intră în TTL Blocklist
ACTION_REKEY = "REKEY"   # Cheia de semnare a fost revocată; Key Authority emite una nouă (Patch 03.1)
_ACTION_ALIASES = {"ALERT_ONLY": ACTION_ALERT, ACTION_ALERT: ACTION_ALERT,
                   ACTION_DROP: ACTION_DROP, ACTION_BLOCK: ACTION_BLOCK,
                   ACTION_REKEY: ACTION_REKEY}

def sanitize(text, max_len: int = 120) -> str:
    """Elimină caracterele de control (anti log-forging) și trunchiază."""
    s = "".join(c if c.isprintable() else "?" for c in str(text))
    return s[:max_len]

class EnforcementEngine:
    DEDUP_WINDOW_SEC = 2.0
    MAX_HISTORY = 2048

    # --- TTL Blocklist ---
    DEFAULT_BLOCK_TTL_SEC = 45.0          # implicit: în intervalul 30-60 s cerut
    MIN_BLOCK_TTL_SEC = 1.0
    MAX_BLOCK_TTL_SEC = 3600.0
    MAX_BLOCKS = 1024                     # plafon de memorie

    # --- Escaladare automată DROP -> BLOCK (doar pe Inline Gateway) ---
    STRIKE_WINDOW_SEC = 30.0
    STRIKE_THRESHOLDS = {
        "DoS / Topic Flooding": 3,        # DoS persistent
        "Anomalous Payload": 3,
        # Replay / Command Tampering lipsesc intenționat: pe actuatoare decide HMAC-ul, iar un BLOCK
        # pe topic ar putea bloca și comenzile legitime (DoS indus de atacator).
    }
    # Topicuri care NU escaladează automat la BLOCK (blocarea pe topic ar lovi și traficul legitim).
    BLOCK_EXEMPT_PREFIXES = ("home/actuators",)

    def __init__(self, session: zenoh.Session, is_inline_gateway: bool = False,
                 global_rate: float = 5.0, global_burst: float = 10.0,
                 critical_rate: float = 5.0, critical_burst: float = 10.0,
                 block_ttl_sec: float = DEFAULT_BLOCK_TTL_SEC,
                 strike_window_sec: float = STRIKE_WINDOW_SEC):
        self.session = session
        self.is_inline_gateway = is_inline_gateway
        self.block_ttl_sec = self._clamp_ttl(block_ttl_sec)
        self.strike_window_sec = max(1.0, float(strike_window_sec))

        self.blocklist = TTLBlocklist(default_ttl=self.block_ttl_sec, max_items=self.MAX_BLOCKS)
        self._last_alert_times = OrderedDict()
        self._repeats = {}
        self._strikes = OrderedDict()     # (topic, attack) -> (count, window_start_monotonic)
        # Ordinea lock-urilor: self._lock -> blocklist._lock (blocklist nu apelează niciodată înapoi).
        self._lock = threading.RLock()

        # Două benzi independente: flood-ul de "zgomot" nu poate epuiza bugetul alertelor critice.
        self._noise_limiter = TokenBucketRateLimiter(rate=global_rate, burst=global_burst, max_keys=1)
        self._critical_limiter = TokenBucketRateLimiter(rate=critical_rate, burst=critical_burst, max_keys=1)
        self.suppressed = 0
        self.suppressed_critical = 0
        self.blocks_issued = 0

    # ------------------------------------------------------------------ TTL Blocklist
    @classmethod
    def _clamp_ttl(cls, ttl) -> float:
        try:
            ttl = float(ttl)
        except (TypeError, ValueError):
            return cls.DEFAULT_BLOCK_TTL_SEC
        if ttl != ttl or ttl in (float("inf"), float("-inf")):
            return cls.DEFAULT_BLOCK_TTL_SEC
        return min(max(ttl, cls.MIN_BLOCK_TTL_SEC), cls.MAX_BLOCK_TTL_SEC)

    def is_blocked(self, entity_id: str) -> bool:
        """True dacă entitatea e în blocklist și TTL-ul NU a expirat (cele expirate sunt șterse).
        Nu ia lock-ul engine-ului: e apelată pe calea fierbinte a gateway-ului."""
        return self.blocklist.is_blocked(entity_id)

    def get_active_blocks(self) -> dict:
        """{entity_id: {"remaining_sec", "ttl_sec", "attack", "source", "since"}} pentru blocările active."""
        return self.blocklist.snapshot()

    def clear_blocklist(self) -> int:
        """Golește blocklist-ul și contoarele de escaladare (laborator/test). Returnează nr. intrărilor șterse."""
        with self._lock:
            self._strikes.clear()
            return self.blocklist.clear()

    # ------------------------------------------------------------------ Escaladare
    def _register_strike_locked(self, key, attack_type: str, mono: float) -> bool:
        """Numără evenimentul și spune dacă pragul a fost atins. Apelat DOAR sub self._lock.
        Se numără și evenimentele care vor fi deduplicate/limitate la alertare."""
        threshold = self.STRIKE_THRESHOLDS.get(attack_type)
        if threshold is None:
            return False
        count, start = self._strikes.get(key, (0, mono))
        if mono - start > self.strike_window_sec:
            count, start = 0, mono
        count += 1
        self._strikes[key] = (count, start)
        self._strikes.move_to_end(key)
        if len(self._strikes) > self.MAX_HISTORY:
            self._strikes.popitem(last=False)
        return count >= threshold

    # ------------------------------------------------------------------ Mitigare
    def execute_mitigation(self, attack_type: str, topic: str, loss: float = 0.0,
                           payload: str = "", critical: bool = False,
                           action: str = None, ttl: float = None) -> str:
        """`payload` = 'detail' (trunchiat/sanitizat). `critical=True` -> bandă dedicată.

        `action`: None (decis automat), "ALERT"/"ALERT_ONLY", "DROP" sau "BLOCK" (explicit, respectat
        oriunde). Pe Inline Gateway, atacurile din STRIKE_THRESHOLDS escaladează automat DROP -> BLOCK
        după pragul de evenimente din fereastră, cu excepția BLOCK_EXEMPT_PREFIXES.
        `ttl` suprascrie TTL-ul implicit. Returnează acțiunea efectivă."""
        mono = time.monotonic()
        requested = _ACTION_ALIASES.get(str(action).upper()) if action is not None else None
        strike_key = (topic, attack_type)
        source = "Inline Gateway" if self.is_inline_gateway else "Passive IDS Agent"

        with self._lock:
            exempt = str(topic).startswith(self.BLOCK_EXEMPT_PREFIXES)
            escalate = (self.is_inline_gateway and not exempt
                        and self._register_strike_locked(strike_key, attack_type, mono))
            if requested == ACTION_BLOCK or escalate:
                eff_action = ACTION_BLOCK
            elif requested is not None:
                eff_action = requested
            else:
                eff_action = ACTION_DROP if self.is_inline_gateway else ACTION_ALERT

            # Blocarea se aplică imediat, independent de dedup/rate-limit-ul alertelor.
            block_remaining = 0.0
            if eff_action == ACTION_BLOCK:
                block_ttl = self._clamp_ttl(ttl) if ttl is not None else self.block_ttl_sec
                block_remaining, is_new = self.blocklist.block_ex(
                    topic, block_ttl, {"attack": attack_type[:60], "source": source})
                if is_new:
                    self.blocks_issued += 1

            # Cheia de dedup include acțiunea: tranziția DROP -> BLOCK nu e înghițită de fereastra de dedup.
            key = (topic, attack_type, eff_action)
            last = self._last_alert_times.get(key)
            if last is not None and mono - last < self.DEDUP_WINDOW_SEC:
                self._repeats[key] = self._repeats.get(key, 0) + 1
                return eff_action

            use_critical = critical or eff_action in (ACTION_BLOCK, ACTION_REKEY)    # BLOCK/REKEY nu se pierd
            limiter = self._critical_limiter if use_critical else self._noise_limiter
            if not limiter.allow("all", mono):
                if use_critical:
                    self.suppressed_critical += 1
                else:
                    self.suppressed += 1
                return eff_action
            self._last_alert_times[key] = mono
            self._last_alert_times.move_to_end(key)
            repeats = self._repeats.pop(key, 0)
            if len(self._last_alert_times) > self.MAX_HISTORY:
                old, _ = self._last_alert_times.popitem(last=False)
                self._repeats.pop(old, None)

        safe_topic = sanitize(topic, 80)
        safe_detail = sanitize(payload, 120)
        if eff_action == ACTION_BLOCK:
            tag = f"ENFORCEMENT - BLOCKED (TTL {block_remaining:.0f}s)"
        elif eff_action == ACTION_REKEY:
            tag = "ENFORCEMENT - REKEY (key revoked, rotation pending)"
        elif eff_action == ACTION_DROP:
            tag = "ENFORCEMENT - DROPPED"
        else:
            tag = "ENFORCEMENT - ALERT_ONLY"
        logger.warning(f"[{tag}] '{safe_topic}' Atac: {sanitize(attack_type, 60)} | Loss: {loss:.4f} | repeats={repeats}")

        alert_payload = json.dumps({
            "timestamp": time.time(),
            "topic": safe_topic,
            "attack": attack_type,
            "action": eff_action,
            "loss": loss,
            "detail": safe_detail,
            "severity": "critical" if use_critical else "normal",
            "repeats": repeats,
            # Secunde de blocare rămase la emitere (durată relativă => dashboard imun la decalaj de ceas).
            "block_ttl": round(block_remaining, 1) if eff_action == ACTION_BLOCK else 0.0,
            "source": source
        }).encode("utf-8")

        try:
            try:
                self.session.put(ALERT_TOPIC, alert_payload, priority=zenoh.Priority.REAL_TIME)
            except (TypeError, AttributeError):
                self.session.put(ALERT_TOPIC, alert_payload)
            logger.info(f"  └─ Alertă emisă pe '{ALERT_TOPIC}'")
        except Exception as e:
            logger.error(f"  └─ Eșec la publicarea alertei: {e}")
        return eff_action