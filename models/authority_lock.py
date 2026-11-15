# Shared cross-process lock for Lumen model-authority operations.
from __future__ import annotations
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_PROCESS_LOCK = threading.RLock()
_THREAD_STATE = threading.local()

def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        if callable(junction) and junction():
            return True
        info = path.lstat()
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(getattr(info, "st_file_attributes", 0) & flag)
    except FileNotFoundError:
        return False

def _lock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

@contextmanager
def candidate_authority_lock(model_root: str | os.PathLike[str]) -> Iterator[None]:
    # Cooperative lock: OS permissions must protect model_root from untrusted writers.
    root = Path(model_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"model root is not a directory: {root}")
    lock_path = root / ".candidate_authority.lock"
    key = os.fspath(lock_path)

    # The process-local RLock serializes threads and makes nested calls reentrant.
    with _PROCESS_LOCK:
        depths = getattr(_THREAD_STATE, "depths", None)
        if depths is None:
            depths = {}
            _THREAD_STATE.depths = depths
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return

        if _is_link_or_reparse(lock_path):
            raise ValueError("model-authority lock must not be a symlink or reparse point")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        handle = None
        acquired = False
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("model-authority lock is not a regular file")
            if _is_link_or_reparse(lock_path):
                raise ValueError("model-authority lock path changed during open")
            handle = os.fdopen(fd, "r+b", closefd=True)
            fd = -1
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
                os.fsync(handle.fileno())
            _lock_file(handle)
            acquired = True
            depths[key] = 1
            try:
                yield
            finally:
                depths.pop(key, None)
        finally:
            if acquired and handle is not None:
                _unlock_file(handle)
            if handle is not None:
                handle.close()
            elif fd >= 0:
                os.close(fd)
