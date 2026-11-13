import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import zenoh  # noqa: F401
except ImportError:
    stub = types.ModuleType("zenoh")
    stub.Session = object
    stub.Priority = types.SimpleNamespace(REAL_TIME=1)
    sys.modules["zenoh"] = stub

import time
import pytest
from agent.enforcement import EnforcementEngine
from utils.ttl_blocklist import TTLBlocklist


class FakeSession:
    def __init__(self):
        self.sent = []
        self._lock = threading.Lock()
    def put(self, topic, payload, **kw):
        with self._lock:
            self.sent.append((topic, json.loads(payload)))


@pytest.fixture
def clock(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    return t


def make(inline=True, **kw):
    session = FakeSession()
    return EnforcementEngine(session, is_inline_gateway=inline, **kw), session


def test_ttl_expiry_and_autocleanup(clock):
    bl = TTLBlocklist(default_ttl=10)
    bl.block("a")
    assert bl.is_blocked("a")
    clock[0] += 9.9
    assert bl.is_blocked("a")
    clock[0] += 0.2
    assert not bl.is_blocked("a")
    assert len(bl._d) == 0


def test_reblock_never_shortens(clock):
    bl = TTLBlocklist(default_ttl=60)
    bl.block("a", ttl=60)
    clock[0] += 10
    assert bl.block("a", ttl=5) == pytest.approx(50.0)


def test_max_items_is_a_hard_cap(clock):
    bl = TTLBlocklist(default_ttl=60, max_items=5)
    for i in range(50): bl.block(f"k{i}")
    assert len(bl._d) == 5 and bl.is_blocked("k49")


def test_invalid_ttl_falls_back_to_default(clock):
    bl = TTLBlocklist(default_ttl=7)
    for bad in (float("nan"), float("inf"), -3, "x", 0):
        assert bl.block("a", ttl=bad) == pytest.approx(7.0)


def test_snapshot_and_clear(clock):
    bl = TTLBlocklist(default_ttl=30)
    bl.block("a", meta={"attack": "X"})
    clock[0] += 10
    snap = bl.snapshot()
    assert snap["a"]["remaining_sec"] == pytest.approx(20.0)
    assert snap["a"]["attack"] == "X"
    assert bl.clear() == 1 and bl.snapshot() == {}


def test_passive_ids_never_escalates(clock):
    eng, _ = make(inline=False)
    for _ in range(10):
        assert eng.execute_mitigation("DoS / Topic Flooding", "home/sensors/t") == "ALERT"
        clock[0] += 3
    assert not eng.is_blocked("home/sensors/t")


def test_inline_dos_escalates_to_block(clock):
    eng, session = make()
    topic = "home/sensors/temp"
    assert eng.execute_mitigation("DoS / Topic Flooding", topic) == "DROP"
    assert eng.execute_mitigation("DoS / Topic Flooding", topic) == "DROP"
    assert eng.execute_mitigation("DoS / Topic Flooding", topic) == "BLOCK"
    assert eng.is_blocked(topic)
    blocks = [a for _, a in session.sent if a["action"] == "BLOCK"]
    assert len(blocks) == 1 and blocks[0]["severity"] == "critical"
    assert blocks[0]["block_ttl"] == pytest.approx(45.0)
    assert eng.get_active_blocks()[topic]["attack"] == "DoS / Topic Flooding"


def test_block_expires(clock):
    eng, _ = make(block_ttl_sec=30)
    eng.execute_mitigation("Anomalous Payload", "home/sensors/x", action="BLOCK")
    assert eng.is_blocked("home/sensors/x")
    clock[0] += 31
    assert not eng.is_blocked("home/sensors/x")
    assert eng.get_active_blocks() == {}


def test_strikes_outside_window_do_not_escalate(clock):
    eng, _ = make(strike_window_sec=30)
    for _ in range(5):
        assert eng.execute_mitigation("DoS / Topic Flooding", "home/sensors/y") == "DROP"
        clock[0] += 31
    assert not eng.is_blocked("home/sensors/y")


def test_actuators_exempt_from_auto_block(clock):
    eng, _ = make()
    for _ in range(10):
        eng.execute_mitigation("Replay Attack", "home/actuators/lock", 99.0, critical=True)
        eng.execute_mitigation("Anomalous Payload", "home/actuators/lock")
        clock[0] += 0.1
    assert not eng.is_blocked("home/actuators/lock")


def test_explicit_block_with_ttl_clamped(clock):
    eng, _ = make()
    eng.execute_mitigation("X", "home/sensors/z", action="BLOCK", ttl=10**9)
    assert eng.get_active_blocks()["home/sensors/z"]["ttl_sec"] == EnforcementEngine.MAX_BLOCK_TTL_SEC


def test_clear_resets_blocks_and_strikes(clock):
    eng, _ = make()
    for _ in range(3): eng.execute_mitigation("DoS / Topic Flooding", "home/sensors/t")
    assert eng.clear_blocklist() == 1
    assert not eng.is_blocked("home/sensors/t")
    assert eng.execute_mitigation("DoS / Topic Flooding", "home/sensors/t") == "DROP"


def test_block_applied_even_if_alert_rate_limited(clock):
    eng, _ = make(critical_rate=0.0, critical_burst=1.0)
    eng.execute_mitigation("A", "home/sensors/a", action="BLOCK")
    eng.execute_mitigation("A", "home/sensors/b", action="BLOCK")
    assert eng.is_blocked("home/sensors/b") and eng.suppressed_critical >= 1


def test_blocklist_bounded_under_topic_spraying(clock):
    eng, _ = make()
    for i in range(EnforcementEngine.MAX_BLOCKS + 500):
        eng.execute_mitigation("X", f"home/sensors/{i}", action="BLOCK")
    assert len(eng.get_active_blocks()) <= EnforcementEngine.MAX_BLOCKS


def test_concurrent_access_is_safe():
    eng, _ = make()
    errors = []
    def worker(n):
        try:
            for i in range(300):
                topic = f"home/sensors/{(n * 7 + i) % 40}"
                eng.execute_mitigation("DoS / Topic Flooding", topic)
                eng.is_blocked(topic)
                if i % 25 == 0: eng.get_active_blocks()
                if i % 97 == 0: eng.clear_blocklist()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(eng.get_active_blocks()) <= EnforcementEngine.MAX_BLOCKS
