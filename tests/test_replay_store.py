from concurrent.futures import ThreadPoolExecutor
from utils.replay_store import ReplayStore, SequenceStore


def test_replay_state_survives_store_recreation(tmp_path):
    path = tmp_path / "gateway.sqlite3"
    first = ReplayStore(path)
    assert first.consume("controller", "home/actuators/door/lock", 100)[0] is True
    first.close()
    reopened = ReplayStore(path)
    assert reopened.get_last_nonce("controller", "home/actuators/door/lock") == 100
    assert reopened.consume("controller", "home/actuators/door/lock", 100) == (False, 100)
    assert reopened.consume("controller", "home/actuators/door/lock", 99) == (False, 100)
    assert reopened.consume("controller", "home/actuators/door/lock", 101) == (True, 100)


def test_replay_state_is_scoped_by_sender_and_topic(tmp_path):
    store = ReplayStore(tmp_path / "replay.sqlite3")
    assert store.consume("controller-a", "home/actuators/door/lock", 100)[0]
    assert store.consume("controller-b", "home/actuators/door/lock", 1)[0]
    assert store.consume("controller-a", "home/actuators/garage/door", 1)[0]


def test_concurrent_duplicate_is_consumed_exactly_once(tmp_path):
    store = ReplayStore(tmp_path / "concurrent.sqlite3")
    def attempt(_):
        return store.consume("controller", "home/actuators/door/lock", 500)[0]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(attempt, range(24)))
    assert sum(results) == 1


def test_outbound_sequence_survives_restart_and_is_monotonic(tmp_path):
    path = tmp_path / "sender.sqlite3"
    first = SequenceStore(path)
    n1, n2 = first.next_nonce("controller"), first.next_nonce("controller")
    second = SequenceStore(path)
    n3 = second.next_nonce("controller")
    assert 0 < n1 < n2 < n3
