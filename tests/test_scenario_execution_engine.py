from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from simulation.lab.execution_engine import (
    ENGINE_VERSION,
    OfflineScenarioEngine,
    ScenarioExecutionError,
)
from simulation.lab.scenario_registry import build_default_registry
from simulation.lab.staging import (
    ResourceLimits,
    StagingConfig,
    verify_manifest,
)


@pytest.fixture
def config(tmp_path):
    return StagingConfig(
        workspace_root=tmp_path / "offline-lab",
        resource_limits=ResourceLimits(
            max_frames=100,
            max_payload_bytes=16_384,
            max_total_bytes=2_000_000,
            max_runtime_seconds=30.0,
            max_manifest_bytes=65_536,
        ),
    )


def test_engine_generates_and_persists_development_attack_frames(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk1_stealth_spoof_dev", variation_seed=17, steps=12
    )
    assert result.frame_count == 12
    assert result.execution_mode == "offline_frame_generation_only"
    assert result.external_execution_performed is False
    assert result.os_containment_verified is False
    assert (result.output_directory / "run_summary.json").is_file()
    assert (result.output_directory / "run_manifest.json").is_file()
    frame_files = sorted((result.output_directory / "frames").glob("frame_*.json"))
    assert len(frame_files) == 12
    first = json.loads(frame_files[0].read_text(encoding="utf-8"))
    assert first["step"] == 0
    assert first["ground_truth_label"] == "attack"
    assert base64.b64decode(first["payload_base64"])


def test_engine_manifest_binds_canonical_source_and_variation(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk2_replay_skew_dev", variation_seed=42, steps=8
    )
    manifest = verify_manifest(result.output_directory / "run_manifest.json")
    assert manifest["scenario_id"] == "atk2_replay_skew_dev"
    assert manifest["variation_digest"] == result.variation_digest
    assert manifest["runner_version"] == ENGINE_VERSION
    assert manifest["resource_policy"]["external_execution_performed"] is False
    assert manifest["resource_policy"]["os_containment_verified"] is False


def test_engine_is_deterministic_for_same_seed_and_inputs(config):
    engine = OfflineScenarioEngine(config)
    left = engine.run_development_attack(
        "atk3_low_slow_recon_dev", variation_seed=123, steps=20
    )
    right = engine.run_development_attack(
        "atk3_low_slow_recon_dev", variation_seed=123, steps=20
    )
    assert left.variation_id == right.variation_id
    assert left.frames_sha256 == right.frames_sha256
    assert left.frame_count == right.frame_count
    # Run IDs and timestamps are intentionally unique; frame evidence is deterministic.


def test_engine_rejects_held_out_attack(config):
    with pytest.raises(ScenarioExecutionError, match="preflight rejected"):
        OfflineScenarioEngine(config).run_development_attack(
            "atk1_stealth_spoof_heldout", variation_seed=1, steps=5
        )


@pytest.mark.parametrize("scenario_id", [
    "fault1_frozen_sensor",
    "fault2_packet_drop_40",
    "authenticated_replay_duplicate_sequence",
    "unregistered_boot_epoch",
])
def test_engine_rejects_fault_and_protocol_regression_scenarios(config, scenario_id):
    with pytest.raises(ScenarioExecutionError, match="preflight rejected"):
        OfflineScenarioEngine(config).run_development_attack(
            scenario_id, variation_seed=1, steps=5
        )


@pytest.mark.parametrize("steps", [0, -1, 101, True, 1.5, 10_001])
def test_engine_validates_step_count_before_workspace_creation(config, steps):
    with pytest.raises(ScenarioExecutionError, match="steps"):
        OfflineScenarioEngine(config).run_development_attack(
            "atk1_stealth_spoof_dev", variation_seed=1, steps=steps
        )
    assert not config.workspace_root.exists()


def test_engine_rejects_steps_above_frame_budget(config):
    limited = StagingConfig(
        workspace_root=config.workspace_root,
        resource_limits=ResourceLimits(
            max_frames=4,
            max_payload_bytes=16_384,
            max_total_bytes=100_000,
            max_runtime_seconds=30.0,
            max_manifest_bytes=65_536,
        ),
    )
    with pytest.raises(ScenarioExecutionError, match="frame budget"):
        OfflineScenarioEngine(limited).run_development_attack(
            "atk1_stealth_spoof_dev", variation_seed=1, steps=5
        )
    assert not limited.workspace_root.exists()


def test_engine_parameter_override_changes_variation(config):
    engine = OfflineScenarioEngine(config)
    base = engine.run_development_attack(
        "atk1_stealth_spoof_dev", variation_seed=9, steps=10
    )
    changed = engine.run_development_attack(
        "atk1_stealth_spoof_dev",
        variation_seed=9,
        steps=10,
        parameter_overrides={"ramp_per_step": 4.0},
    )
    assert base.variation_digest != changed.variation_digest
    assert base.frames_sha256 != changed.frames_sha256


def test_engine_summary_digest_matches_persisted_frame_records(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk4_adaptive_burst_dev", variation_seed=31, steps=15
    )
    hasher = hashlib.sha256()
    files = sorted((result.output_directory / "frames").glob("frame_*.json"))
    for path in files:
        encoded = path.read_bytes()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    summary = json.loads((result.output_directory / "run_summary.json").read_text())
    assert hasher.hexdigest() == result.frames_sha256 == summary["frames_sha256"]


def test_engine_keeps_timing_as_metadata_and_does_not_replay_in_real_time(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk2_replay_skew_dev", variation_seed=7, steps=6
    )
    records = [
        json.loads(path.read_text())
        for path in sorted((result.output_directory / "frames").glob("frame_*.json"))
    ]
    assert any(record["delta_t_sec"] < 0.2 for record in records)
    assert result.execution_mode == "offline_frame_generation_only"
    assert result.external_execution_performed is False


def test_engine_uses_canonical_default_registry(config):
    engine = OfflineScenarioEngine(config)
    assert engine.registry.report()["scenario_count"] == 12
    assert engine.registry is not None


def test_engine_result_reports_resource_accounting(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk1_stealth_spoof_dev", variation_seed=5, steps=10
    )
    assert result.total_bytes > 0
    assert result.elapsed_seconds >= 0
    assert result.output_directory.is_dir()


def _assert_workspace_root_empty(root: Path) -> None:
    assert not root.exists() or list(root.iterdir()) == []


def test_engine_final_resource_accounting_covers_every_artifact(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk1_stealth_spoof_dev", variation_seed=51, steps=9
    )
    actual_bytes = sum(p.stat().st_size for p in result.output_directory.rglob("*") if p.is_file())
    summary = json.loads((result.output_directory / "run_summary.json").read_text(encoding="utf-8"))
    assert result.total_bytes == actual_bytes
    assert summary["budget_snapshot_stage"] == "after_frames_before_summary_and_manifest"
    assert set(summary["budget_snapshot_excludes"]) == {"run_summary.json", "run_manifest.json"}
    frame_bytes = sum(p.stat().st_size for p in (result.output_directory / "frames").glob("frame_*.json"))
    assert summary["budget"]["total_bytes"] == frame_bytes
    assert summary["budget"]["frames"] == result.frame_count


def test_engine_payload_budget_exhaustion_cleans_workspace(config):
    limited = StagingConfig(workspace_root=config.workspace_root, resource_limits=ResourceLimits(
        max_frames=100, max_payload_bytes=32, max_total_bytes=100_000,
        max_runtime_seconds=30.0, max_manifest_bytes=65_536,
    ))
    with pytest.raises(ScenarioExecutionError, match="payload byte budget"):
        OfflineScenarioEngine(limited).run_development_attack("atk1_stealth_spoof_dev", variation_seed=1, steps=3)
    _assert_workspace_root_empty(limited.workspace_root)


def test_engine_total_byte_budget_exhaustion_cleans_workspace(config):
    limited = StagingConfig(workspace_root=config.workspace_root, resource_limits=ResourceLimits(
        max_frames=100, max_payload_bytes=2_048, max_total_bytes=2_048,
        max_runtime_seconds=30.0, max_manifest_bytes=65_536,
    ))
    with pytest.raises(ScenarioExecutionError, match="total staging byte budget"):
        OfflineScenarioEngine(limited).run_development_attack("atk1_stealth_spoof_dev", variation_seed=2, steps=10)
    _assert_workspace_root_empty(limited.workspace_root)


def test_engine_runtime_budget_exhaustion_cleans_workspace(config, monkeypatch):
    import simulation.lab.staging.resource_limits as resource_limits
    ticks = iter([0.0, 2.0, 2.0, 2.0, 2.0])
    monkeypatch.setattr(resource_limits.time, "monotonic", lambda: next(ticks, 2.0))
    limited = StagingConfig(workspace_root=config.workspace_root, resource_limits=ResourceLimits(
        max_frames=100, max_payload_bytes=16_384, max_total_bytes=2_000_000,
        max_runtime_seconds=1.0, max_manifest_bytes=65_536,
    ))
    with pytest.raises(ScenarioExecutionError, match="runtime budget"):
        OfflineScenarioEngine(limited).run_development_attack("atk1_stealth_spoof_dev", variation_seed=3, steps=3)
    _assert_workspace_root_empty(limited.workspace_root)


def test_engine_generator_failure_cleans_partial_workspace(config, monkeypatch):
    from types import SimpleNamespace
    import simulation.lab.execution_engine as execution_engine
    original = execution_engine.build_parameterized_scenario
    def failing_builder(*args, **kwargs):
        variation = original(*args, **kwargs)
        def broken_frames(steps):
            iterator = iter(variation.iter_frames(steps))
            yield next(iterator)
            raise RuntimeError("injected generator failure")
        return SimpleNamespace(variation_id=variation.variation_id,
            variation_digest=variation.variation_digest, variation_seed=variation.variation_seed,
            iter_frames=broken_frames)
    monkeypatch.setattr(execution_engine, "build_parameterized_scenario", failing_builder)
    with pytest.raises(ScenarioExecutionError, match="injected generator failure"):
        OfflineScenarioEngine(config).run_development_attack("atk1_stealth_spoof_dev", variation_seed=4, steps=3)
    _assert_workspace_root_empty(config.workspace_root)


def test_engine_manifest_write_failure_cleans_partial_workspace(config, monkeypatch):
    from simulation.lab.staging import ManifestError, RunManifest
    def fail_manifest(self, workspace, relative_path="manifest.json"):
        raise ManifestError("injected manifest write failure")
    monkeypatch.setattr(RunManifest, "write", fail_manifest)
    with pytest.raises(ScenarioExecutionError, match="injected manifest write failure"):
        OfflineScenarioEngine(config).run_development_attack("atk1_stealth_spoof_dev", variation_seed=5, steps=3)
    _assert_workspace_root_empty(config.workspace_root)


def test_engine_manifest_and_summary_share_run_provenance(config):
    result = OfflineScenarioEngine(config).run_development_attack(
        "atk2_replay_skew_dev", variation_seed=66, steps=7
    )
    summary = json.loads((result.output_directory / "run_summary.json").read_text(encoding="utf-8"))
    manifest = verify_manifest(result.output_directory / "run_manifest.json")
    assert summary["run_id"] == manifest["run_id"] == result.run_id
    assert summary["source_scenario_id"] == manifest["scenario_id"] == result.scenario_id
    assert summary["source_definition_digest"] == manifest["scenario_definition_digest"]
    assert summary["variation_digest"] == manifest["variation_digest"] == result.variation_digest
    assert summary["variation_seed"] == manifest["seed"]
    assert summary["frames_sha256"] == result.frames_sha256

