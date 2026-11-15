from pathlib import Path

import pytest

from models.model_lifecycle import (
    _delete_file_durable,
    _read_regular_pointer_bytes,
    _write_bytes_atomic,
)


def test_missing_optional_previous_pointer_is_valid(tmp_path):
    pointer = tmp_path / "previous_model.json"
    assert _read_regular_pointer_bytes(pointer) is None
    _delete_file_durable(pointer)
    assert not pointer.exists()


def test_atomic_write_can_create_first_previous_pointer(tmp_path):
    pointer = tmp_path / "previous_model.json"
    payload = b'{"schema":1,"active_bundle":"lumen_model_bundle"}\n'
    _write_bytes_atomic(pointer, payload)
    assert pointer.read_bytes() == payload
    assert _read_regular_pointer_bytes(pointer) == payload


def test_durable_delete_is_idempotent_for_missing_pointer(tmp_path):
    pointer = tmp_path / "previous_model.json"
    _delete_file_durable(pointer)
    _delete_file_durable(pointer)
    assert not pointer.exists()


def test_optional_pointer_helpers_reject_symlinks(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    pointer = tmp_path / "previous_model.json"
    try:
        pointer.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this Windows environment")
    with pytest.raises(ValueError, match="symlink or reparse point"):
        _read_regular_pointer_bytes(pointer)
    with pytest.raises(ValueError, match="symlink or reparse point"):
        _write_bytes_atomic(pointer, b"replacement")
    with pytest.raises(ValueError, match="symlink or reparse point"):
        _delete_file_durable(pointer)
