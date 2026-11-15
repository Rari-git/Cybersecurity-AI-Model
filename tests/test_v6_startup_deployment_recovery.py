from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run_all


def test_startup_recovers_journal_before_preflight(monkeypatch):
    calls = []

    def stop_at_preflight():
        calls.append("preflight")
        raise SystemExit("stop before provisioning")

    monkeypatch.setattr(
        run_all, "recover_deployment_journal_or_exit",
        lambda: calls.append("recovery") or [],
    )
    monkeypatch.setattr(run_all, "preflight_or_exit", stop_at_preflight)

    with pytest.raises(SystemExit, match="stop before provisioning"):
        run_all.launch_and_arrange_grid()

    assert calls == ["recovery", "preflight"]


def test_recovery_temporarily_enables_authority_and_restores_environment(monkeypatch):
    observed = []
    fake = SimpleNamespace(
        recover_deployment_transactions=lambda **kwargs: (
            observed.append((run_all.os.environ.get("LUMEN_MODEL_DEPLOYMENT_AUTHORITY"), kwargs))
            or [{"transaction_id": "test-tx", "status": "committed"}]
        )
    )
    monkeypatch.setitem(sys.modules, "training.deployment_authority", fake)
    monkeypatch.delenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", raising=False)

    outcomes = run_all.recover_deployment_journal_or_exit()

    assert outcomes == [{"transaction_id": "test-tx", "status": "committed"}]
    assert observed[0][0] == "1"
    assert observed[0][1]["model_root"] == run_all.MODEL_ROOT
    assert observed[0][1]["public_key_path"] == run_all.MODEL_PUBLIC_KEY
    assert observed[0][1]["pointer_path"] == run_all.ACTIVE_MODEL_POINTER
    assert "LUMEN_MODEL_DEPLOYMENT_AUTHORITY" not in run_all.os.environ


def test_recovery_restores_preexisting_authority_value(monkeypatch):
    fake = SimpleNamespace(recover_deployment_transactions=lambda **kwargs: [])
    monkeypatch.setitem(sys.modules, "training.deployment_authority", fake)
    monkeypatch.setenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", "0")

    assert run_all.recover_deployment_journal_or_exit() == []
    assert run_all.os.environ["LUMEN_MODEL_DEPLOYMENT_AUTHORITY"] == "0"


def test_recovery_failure_cancels_startup_and_restores_environment(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("journal hash-chain failure")

    fake = SimpleNamespace(recover_deployment_transactions=fail)
    monkeypatch.setitem(sys.modules, "training.deployment_authority", fake)
    monkeypatch.delenv("LUMEN_MODEL_DEPLOYMENT_AUTHORITY", raising=False)

    with pytest.raises(SystemExit, match="deployment journal recovery failed"):
        run_all.recover_deployment_journal_or_exit()

    assert "LUMEN_MODEL_DEPLOYMENT_AUTHORITY" not in run_all.os.environ
