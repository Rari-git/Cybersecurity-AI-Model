import random
from collections import deque
import numpy as np

from training.feature_extractor import FEATURE_DIM, FeatureWindow, RateMeter

GAS = "home/sensors/gas/ppm"
VIB = "home/sensors/vibration/shock"
DOOR = "home/sensors/door/state"
PIR = "home/sensors/pir/motion"

ATTACK_KINDS = ("flood", "value_spoof", "payload_inject", "flapping")


def legit_events(duration_s: float, rng: random.Random, t0: float = 0.0):
    """Replică temporizarea și formatul payload-urilor din simulation/virtual_sensors.py."""
    ev = []

    def periodic(topic, fn, lo, hi):
        t = t0 + rng.uniform(0, hi)
        while t < t0 + duration_s:
            ev.append((t, topic, fn(), "legit"))
            t += rng.uniform(lo, hi)

    def sporadic(topic, fn, mean):
        t = t0 + 0.5 + rng.expovariate(1.0 / mean)
        while t < t0 + duration_s:
            ev.append((t, topic, fn(), "legit"))
            t += 0.5 + rng.expovariate(1.0 / mean)

    periodic(GAS, lambda: str(rng.randint(300, 420)), 0.8, 1.9)
    periodic(VIB, lambda: f"{rng.uniform(0.0, 0.04):.3f}", 0.8, 1.9)
    sporadic(DOOR, lambda: rng.choice(["OPEN", "CLOSED"]), 6.5)
    sporadic(PIR, lambda: "MOTION_DETECTED", 4.0)
    return ev


def attack_events(kind: str, t0: float, rng: random.Random):
    ev = []
    if kind == "flood":              # payload valid, doar rata anormală
        ev = [(t0 + i * 0.003, DOOR, "OPEN", kind) for i in range(400)]
    elif kind == "value_spoof":      # ritm normal, valoare imposibilă
        t = t0
        for _ in range(30):
            ev.append((t, GAS, str(rng.randint(2000, 9000)), kind))
            t += rng.uniform(0.8, 1.9)
    elif kind == "payload_inject":   # payload cu entropie mare / lungime anormală
        t = t0
        for _ in range(20):
            ev.append((t, PIR, f"{rng.getrandbits(256):064x}", kind))
            t += rng.uniform(3.0, 5.0)
    elif kind == "flapping":         # stare ușă comutată rapid (sub rata de flood)
        t = t0
        for i in range(40):
            ev.append((t, DOOR, "OPEN" if i % 2 == 0 else "CLOSED", kind))
            t += 0.25
    return ev


def windows_from_events(events, seq_len: int):
    """Pipeline identic cu agentul: RateMeter + FeatureWindow per topic.
    Returnează (raw[N,seq,D], topics, last_kind, any_attack_in_window)."""
    meters, wins, kinds = {}, {}, {}
    raws, topics, last_kind, any_attack = [], [], [], []
    for t, topic, payload, kind in sorted(events, key=lambda e: e[0]):
        m = meters.get(topic)
        if m is None:
            m = meters[topic] = RateMeter(t)
            wins[topic] = FeatureWindow(seq_len)
            kinds[topic] = deque(maxlen=seq_len)
        dt, rate = m.observe(t)
        kinds[topic].append(kind)
        raw = wins[topic].push(topic, payload, dt, rate)
        if raw is None:
            continue
        raws.append(raw)
        topics.append(topic)
        last_kind.append(kind)
        any_attack.append(any(k != "legit" for k in kinds[topic]))
    if not raws:
        return np.zeros((0, seq_len, FEATURE_DIM), np.float32), [], [], np.zeros(0, bool)
    return np.stack(raws), topics, last_kind, np.array(any_attack, dtype=bool)