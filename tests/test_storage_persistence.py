import pytest

from storage.persistence import persist_normal_event, persist_incident_capture


class Writer:
    def __init__(self, error=None):
        self.error = error
        self.committed = False
    def submit_sync(self, fn, timeout=5.0):
        if self.error:
            raise self.error
        class Conn:
            def execute(self, *_args):
                return None
        fn(Conn())
        self.committed = True


class Storage:
    def __init__(self, normal=None, incidents=None):
        self._writers = {"normal": normal or Writer(), "incidents": incidents or Writer()}


def test_normal_and_forensic_persistence_wait_for_commit():
    storage = Storage()
    persist_normal_event(storage, tuple(range(14)))
    persist_incident_capture(storage, tuple(range(20)))
    assert storage._writers["normal"].committed
    assert storage._writers["incidents"].committed


def test_storage_failure_propagates_to_admission_caller():
    storage = Storage(normal=Writer(RuntimeError("queue unavailable")))
    with pytest.raises(RuntimeError, match="queue unavailable"):
        persist_normal_event(storage, tuple(range(14)))
