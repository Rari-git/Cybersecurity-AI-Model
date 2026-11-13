import math
from collections import deque
import numpy as np

TOPIC_MAP = {
    "home/sensors/gas/ppm": 1,
    "home/sensors/vibration/shock": 2,
    "home/sensors/door/state": 3,
    "home/sensors/pir/motion": 4,
    "home/actuators/door/lock": 5,
    "home/security/alerts": 6
}

FEATURE_NAMES = ["delta_t", "packet_rate", "payload_len", "topic_id",
                 "payload_entropy", "jitter", "value_norm"]
FEATURE_DIM = len(FEATURE_NAMES)

DT_CAP = 30.0
JITTER_CAP = 10.0
JITTER_WINDOW = 5

# Limita unică de inspecție a payload-ului, în OCTEȚI (identică cu Gateway-ul și cu agentul IDS).
MAX_PAYLOAD_BYTES = 512
# Alias păstrat pentru compatibilitate cu codul existent care importa vechea constantă.
MAX_PAYLOAD_CHARS = MAX_PAYLOAD_BYTES

SCALE_CLIP = (-5.0, 5.0)      # NU se taie la 1.0: valorile OOD trebuie să rămână peste 1
FLOOD_DT = 0.05
FLOOD_RATE = 15.0

# Interval normal -> value_norm in [0,1]. Peste/sub interval => OOD.
VALUE_RANGES = {
    "home/sensors/gas/ppm": (300.0, 450.0),
    "home/sensors/vibration/shock": (0.0, 0.05),
}
DISCRETE_TOPICS = frozenset({"home/sensors/door/state", "home/sensors/pir/motion"})
STATE_VALUES = {"CLOSED": 0.0, "OPEN": 1.0, "MOTION_DETECTED": 1.0}
OOD_VALUE = 5.0


def truncate_payload(payload: str) -> str:
    """Trunchiază payload-ul la cel mult MAX_PAYLOAD_BYTES octeți UTF-8, fără a lăsa
    un caracter multi-octet tăiat la mijloc (octeții parțiali de la coadă sunt eliminați).
    Funcția este idempotentă: un payload deja scurt este returnat neschimbat."""
    raw = payload.encode("utf-8", errors="replace")
    if len(raw) <= MAX_PAYLOAD_BYTES:
        return payload
    return raw[:MAX_PAYLOAD_BYTES].decode("utf-8", errors="ignore")


def payload_length(payload: str) -> int:
    """Lungimea payload-ului în octeți UTF-8, plafonată la MAX_PAYLOAD_BYTES."""
    return min(len(payload.encode("utf-8", errors="replace")), MAX_PAYLOAD_BYTES)


def calculate_shannon_entropy(data_str: str) -> float:
    if not data_str:
        return 0.0
    length = len(data_str)
    counts = {}
    for ch in data_str:
        counts[ch] = counts.get(ch, 0) + 1
    return float(-sum((c / length) * math.log2(c / length) for c in counts.values()))


def payload_value(topic: str, payload: str) -> float:
    p = truncate_payload(payload).strip()
    rng = VALUE_RANGES.get(topic)
    if rng is not None:
        try:
            v = float(p)
        except ValueError:
            return OOD_VALUE
        if not math.isfinite(v):
            return OOD_VALUE
        lo, hi = rng
        return max(-OOD_VALUE, min(OOD_VALUE, (v - lo) / (hi - lo)))
    if topic in DISCRETE_TOPICS:
        return STATE_VALUES.get(p, OOD_VALUE)
    return 0.0


def payload_plausible(topic: str, payload: str) -> bool:
    """Validare deterministă (folosită în gateway): valoare în/aproape de intervalul normal."""
    return -1.0 <= payload_value(topic, payload) <= 2.0


def rule_flood(delta_t: float, rate: float) -> bool:
    return delta_t < FLOOD_DT or rate > FLOOD_RATE


class RateMeter:
    """Δt și rata calculate pe TRAFICUL OFERIT (inclusiv pachetele respinse de rate limiter).
    Aceeași clasă este folosită la antrenare, evaluare și în agent => fără train-serve skew."""
    __slots__ = ("last", "count", "win_start", "rate")

    def __init__(self, now: float):
        self.last = None
        self.count = 0
        self.win_start = now
        self.rate = 1.0

    def observe(self, now: float):
        dt = 1.0 if self.last is None else max(0.001, now - self.last)
        self.last = now
        self.count += 1
        elapsed = now - self.win_start
        if elapsed >= 1.0:
            self.rate = self.count / elapsed
            self.count = 0
            self.win_start = now
        return dt, self.rate


def extract_features(topic: str, payload: str, delta_t: float, window_rate: float, recent_dts) -> list:
    """Pură (fără stare globală). `recent_dts` = ultimele Δt ale ACESTUI topic, incluzând delta_t curent.
    Payload-ul este trunchiat la MAX_PAYLOAD_BYTES o singură dată, iar toate feature-urile
    (lungime, entropie, valoare) sunt calculate pe aceeași versiune trunchiată."""
    payload = truncate_payload(payload)
    jitter = float(np.std(recent_dts)) if len(recent_dts) > 1 else 0.0
    return [
        float(min(delta_t, DT_CAP)),
        float(window_rate),
        float(payload_length(payload)),
        float(TOPIC_MAP.get(topic, 99)),
        calculate_shannon_entropy(payload),
        float(min(jitter, JITTER_CAP)),
        payload_value(topic, payload),
    ]


class FeatureWindow:
    """Fereastră glisantă per topic (un singur thread consumator).

    Contaminare: în timpul unui flood/DoS fereastra NU este reinițializată (ar forța un
    cold-start de `seq_len` eșantioane în care topicul rămâne neinspectat = window starvation).
    În schimb este marcată `is_contaminated`, iar consumatorul ignoră scorul ML cât timp
    flag-ul este activ. Flag-ul se resetează abia după ce fereastra a fost umplută integral
    (`seq_len` eșantioane consecutive) cu eșantioane curate, adică fără semnătură de flood."""
    __slots__ = ("seq_len", "feats", "dts", "is_contaminated", "_clean_count")

    def __init__(self, seq_len: int):
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 1:
            raise ValueError(f"seq_len invalid pentru FeatureWindow: {seq_len!r}")
        self.seq_len = seq_len
        self.feats = deque(maxlen=seq_len)
        self.dts = deque(maxlen=JITTER_WINDOW)
        self.is_contaminated = False
        self._clean_count = 0

    def mark_contaminated(self):
        """Marchează fereastra ca fiind contaminată de trafic de tip flood/DoS.
        NU atinge `feats`, `dts` sau `seq_len`: acumularea continuă fără cold-start."""
        self.is_contaminated = True
        self._clean_count = 0

    def add_sample(self, topic, payload, delta_t, rate):
        """Adaugă un eșantion și returnează fereastra [seq_len, FEATURE_DIM] când este plină
        (altfel None). Gestionează ieșirea din starea de contaminare."""
        self.dts.append(delta_t)
        self.feats.append(extract_features(topic, payload, delta_t, rate, self.dts))

        if self.is_contaminated:
            if rule_flood(delta_t, rate):
                # Eșantion încă ofensiv: numărătoarea de eșantioane curate se reia de la zero.
                self._clean_count = 0
            else:
                self._clean_count += 1
                if self._clean_count >= self.seq_len and len(self.feats) >= self.seq_len:
                    self.is_contaminated = False
                    self._clean_count = 0

        if len(self.feats) < self.seq_len:
            return None
        return np.asarray(self.feats, dtype=np.float32)

    def push(self, topic, payload, delta_t, rate):
        """Alias retrocompatibil pentru `add_sample` (folosit de agent și de simulator)."""
        return self.add_sample(topic, payload, delta_t, rate)


def load_scaler(meta: dict):
    min_vals = np.array(meta["min_vals"], dtype=np.float32)
    max_vals = np.array(meta["max_vals"], dtype=np.float32)
    return min_vals, (1.0 / (max_vals - min_vals + 1e-6)).astype(np.float32)


def scale_window(raw: np.ndarray, min_vals: np.ndarray, inv_range: np.ndarray) -> np.ndarray:
    return np.clip((raw - min_vals) * inv_range, SCALE_CLIP[0], SCALE_CLIP[1]).astype(np.float32)


def last_step_loss(scaled: np.ndarray, recon: np.ndarray) -> float:
    return float(np.mean((scaled[-1] - recon[-1]) ** 2))