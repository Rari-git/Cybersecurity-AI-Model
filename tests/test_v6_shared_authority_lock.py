from __future__ import annotations
import multiprocessing
import time
from pathlib import Path
from models.authority_lock import candidate_authority_lock

def _lock_worker(root: str, active, maximum) -> None:
    with candidate_authority_lock(root):
        with active.get_lock():
            active.value += 1
            maximum.value = max(maximum.value, active.value)
        time.sleep(0.20)
        with active.get_lock():
            active.value -= 1

def test_shared_authority_lock_is_reentrant(tmp_path: Path):
    with candidate_authority_lock(tmp_path):
        with candidate_authority_lock(tmp_path):
            assert (tmp_path / ".candidate_authority.lock").is_file()

def test_shared_authority_lock_serializes_processes(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    active, maximum = ctx.Value("i", 0), ctx.Value("i", 0)
    workers = [ctx.Process(target=_lock_worker, args=(str(tmp_path), active, maximum)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(15)
    for worker in workers:
        assert worker.exitcode == 0
    assert maximum.value == 1
    assert active.value == 0
