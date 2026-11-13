import time
import math
import threading
from collections import OrderedDict


class TTLBlocklist:
    """
    Listă neagră thread-safe cu expirare automată (TTL) și limitare strictă de memorie.
    Previne blocarea definitivă a senzorilor legitimi după un atac.

    - o re-blocare PRELUNGEȘTE blocarea activă, niciodată nu o scurtează;
    - plafonul `max_items` este garantat (întâi GC pe expirate, apoi evicțiune FIFO);
    - `snapshot()` exportă starea pentru UI/inspecție (timp rămas, TTL, metadate).
    Timpul de expirare folosește time.monotonic() (imun la schimbări ale ceasului de sistem).
    """

    def __init__(self, default_ttl: float = 60.0, max_items: int = 5000):
        self._d = OrderedDict()   # key -> (expires_mono, ttl, meta, since_epoch)
        self.default_ttl = self._valid_ttl(default_ttl, 60.0)
        self.max_items = max(1, int(max_items))
        self._lock = threading.Lock()

    @staticmethod
    def _valid_ttl(ttl, fallback: float) -> float:
        try:
            ttl = float(ttl)
        except (TypeError, ValueError):
            return fallback
        return ttl if math.isfinite(ttl) and ttl > 0 else fallback

    def block_ex(self, key: str, ttl: float = None, meta: dict = None):
        """Returnează (secunde_rămase, este_blocare_nouă)."""
        ttl = self.default_ttl if ttl is None else self._valid_ttl(ttl, self.default_ttl)
        with self._lock:
            now = time.monotonic()
            expires = now + ttl
            since = time.time()
            is_new = True
            prev = self._d.get(key)
            if prev is not None and prev[0] > now:
                expires = max(expires, prev[0])
                since = prev[3]
                is_new = False
            self._d[key] = (expires, ttl, dict(meta or {}), since)
            self._d.move_to_end(key)
            if len(self._d) > self.max_items:
                self._gc(now)
                while len(self._d) > self.max_items:
                    self._d.popitem(last=False)
            return expires - now, is_new

    def block(self, key: str, ttl: float = None, meta: dict = None) -> float:
        """Compatibil cu API-ul vechi; returnează secundele rămase."""
        return self.block_ex(key, ttl, meta)[0]

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            entry = self._d.get(key)
            if entry is None:
                return False
            if entry[0] <= time.monotonic():
                del self._d[key]          # auto-cleanup
                return False
            return True

    def unblock(self, key: str):
        with self._lock:
            self._d.pop(key, None)

    def clear(self) -> int:
        with self._lock:
            n = len(self._d)
            self._d.clear()
            return n

    def snapshot(self) -> dict:
        """{key: {"remaining_sec", "ttl_sec", "since", **meta}} doar pentru intrările active."""
        now = time.monotonic()
        with self._lock:
            self._gc(now)
            return {
                k: {"remaining_sec": round(max(0.0, e[0] - now), 1),
                    "ttl_sec": e[1], "since": e[3], **e[2]}
                for k, e in self._d.items()
            }

    def __len__(self) -> int:
        with self._lock:
            self._gc(time.monotonic())
            return len(self._d)

    def _gc(self, now: float = None):
        """Apelat DOAR sub self._lock."""
        now = time.monotonic() if now is None else now
        for k in [k for k, e in self._d.items() if e[0] <= now]:
            del self._d[k]