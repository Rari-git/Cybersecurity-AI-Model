"""Safe, bounded workspace creation and cleanup.

Security note: path checks reduce accidental escapes and reject links/reparse
points, but Python path APIs cannot eliminate TOCTOU races against a concurrent
local process with write access to the workspace parent. Use an OS-enforced
sandbox/VM for hostile concurrent actors or untrusted executable code.
"""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
import os
import shutil
import stat
import tempfile
import threading
import uuid

from .config import StagingConfig
from .resource_limits import ResourceBudget, ResourceLimitExceeded


class WorkspaceError(RuntimeError):
    pass


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _has_reparse_attribute(path: Path) -> bool:
    try:
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & _REPARSE_POINT)
    except FileNotFoundError:
        return False


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or _has_reparse_attribute(path)


def _reject_link_components(path: Path) -> None:
    absolute = Path(os.path.abspath(str(path)))
    for part in absolute.parents[::-1] + (absolute,):
        if _is_link_or_reparse(part):
            raise WorkspaceError(f"symlink/reparse-point path component rejected: {part}")


def _reject_nested_links(root: Path) -> None:
    """Walk without following links; reject links anywhere before recursive deletion."""
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirs) + list(files):
            candidate = current_path / name
            if _is_link_or_reparse(candidate):
                raise WorkspaceError(f"nested symlink/reparse point rejected: {candidate}")
        dirs[:] = [
            name for name in dirs
            if not _is_link_or_reparse(current_path / name)
        ]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


class StagingWorkspace:
    def __init__(self, config: StagingConfig):
        self.config = config
        self.root = Path(config.workspace_root)
        self.run_id = uuid.uuid4().hex
        self.path = self.root / self.run_id
        self._created = False
        self._budget = ResourceBudget(config.resource_limits)
        self._write_lock = threading.Lock()

    @property
    def budget(self) -> ResourceBudget:
        """The per-run budget used by the future scenario execution engine."""
        return self._budget

    def _validate_root(self) -> None:
        raw = Path(os.path.abspath(str(self.root)))
        _reject_link_components(raw)
        resolved = raw.resolve(strict=False)
        for forbidden in self.config.forbidden_roots:
            forbidden_resolved = Path(os.path.abspath(str(forbidden))).resolve(strict=False)
            if _is_within(resolved, forbidden_resolved) or _is_within(forbidden_resolved, resolved):
                raise WorkspaceError(f"staging root overlaps a forbidden path: {forbidden_resolved}")

    def create(self) -> "StagingWorkspace":
        self._validate_root()
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_components(self.root)
        root_resolved = self.root.resolve(strict=True)
        if self.path.exists() or self.path.is_symlink():
            raise WorkspaceError("run workspace already exists")
        self.path.mkdir(mode=0o700)
        _reject_link_components(self.path)
        if self.path.resolve(strict=True).parent != root_resolved:
            raise WorkspaceError("workspace escaped its configured root")
        self._created = True
        return self

    def resolve_output(self, relative_path: str) -> Path:
        if not self._created:
            raise WorkspaceError("workspace has not been created")
        rel = Path(relative_path)
        win_rel = PureWindowsPath(str(relative_path))
        if (
            rel.is_absolute() or win_rel.is_absolute() or win_rel.drive
            or not rel.parts or any(part in ("..", "") for part in rel.parts)
        ):
            raise WorkspaceError("output path must be a safe relative path")
        target = self.path.joinpath(rel)
        _reject_link_components(target.parent)
        if not _is_within(target.parent.resolve(strict=False), self.path.resolve(strict=True)):
            raise WorkspaceError("output path escapes staging workspace")
        if _is_link_or_reparse(target):
            raise WorkspaceError("refusing to write through a link/reparse point")
        if target.exists():
            raise WorkspaceError("refusing to overwrite an existing staging file")
        return target

    def _write_bytes_unaccounted(self, relative_path: str, payload: bytes) -> Path:
        """Write after the caller has reserved/accounted for the bytes."""
        target = self.resolve_output(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_link_components(target.parent)
        if _is_link_or_reparse(target) or target.exists():
            raise WorkspaceError("refusing to write through or overwrite an existing path")
        # Write privately, flush, then atomically publish without overwriting.
        # os.link fails if another writer created the destination in the meantime.
        fd, temporary_name = tempfile.mkstemp(
            prefix=".lumen-staging-", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(temporary_name)
        published = False
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
            published = True
            return target
        except Exception:
            # Never remove the destination here: os.link may have failed
            # because another writer created it. Preserve that writer's file.
            raise
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A completed published target is preferable to reporting a
                # failed write merely because temporary cleanup was interrupted.
                pass

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        with self._write_lock:
            try:
                self._budget.consume_evidence(len(payload))
            except ResourceLimitExceeded as exc:
                raise WorkspaceError(str(exc)) from exc
            try:
                return self._write_bytes_unaccounted(relative_path, payload)
            except Exception:
                # Roll back accounting if persistence failed.
                self._budget.refund_evidence(len(payload))
                raise

    def write_frame(self, relative_path: str, payload: bytes) -> Path:
        """Persist a generated frame and account for its frame and payload once."""
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        with self._write_lock:
            try:
                self._budget.consume_frame(len(payload))
            except ResourceLimitExceeded as exc:
                raise WorkspaceError(str(exc)) from exc
            try:
                return self._write_bytes_unaccounted(relative_path, payload)
            except Exception:
                self._budget.refund_frame(len(payload))
                raise

    def write_manifest(self, relative_path: str, payload: bytes) -> Path:
        """Persist a manifest only through the validated workspace boundary."""
        if not isinstance(payload, bytes):
            raise TypeError("manifest payload must be bytes")
        if len(payload) > self.config.resource_limits.max_manifest_bytes:
            raise WorkspaceError("manifest exceeds configured size limit")
        return self.write_bytes(relative_path, payload)

    def cleanup(self) -> None:
        if not self._created:
            return
        self._validate_root()
        root_resolved = self.root.resolve(strict=True)
        info = self.path.lstat()
        if stat.S_ISLNK(info.st_mode) or _has_reparse_attribute(self.path):
            raise WorkspaceError("refusing to clean a linked/reparse-point workspace")
        if self.path.resolve(strict=True).parent != root_resolved:
            raise WorkspaceError("refusing cleanup outside staging root")
        _reject_nested_links(self.path)
        shutil.rmtree(self.path)
        self._created = False

    def __enter__(self) -> "StagingWorkspace":
        return self.create()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
