from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from simulation.lab.staging import (
    IsolationPolicy, ManifestError, ResourceBudget, ResourceLimitExceeded,
    ResourceLimits, StagingConfig, StagingWorkspace, WorkspaceError,
    assess_isolation, verify_manifest,
)
from simulation.lab.staging.manifest import RunManifest


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_config_denies_network_subprocess_and_production_access(tmp_path):
    for kwargs in ({"allow_network": True}, {"allow_subprocess": True}, {"allow_production_access": True}):
        with pytest.raises(ValueError):
            StagingConfig(tmp_path / "lab", **kwargs)


def test_resource_limits_validate_values():
    with pytest.raises(ValueError):
        ResourceLimits(max_frames=0)
    with pytest.raises(ValueError):
        ResourceLimits(max_payload_bytes=20, max_total_bytes=10)


def test_isolation_fails_closed_without_os_attestation():
    report = assess_isolation(IsolationPolicy(require_os_containment=True))
    assert report.application_restrictions_configured
    assert not report.application_restrictions_enforced
    assert not report.application_restrictions_verified
    assert not report.os_containment_verified
    assert report.staging_ready
    assert not report.execution_ready


def test_isolation_cannot_be_claimed_by_caller_supplied_values():
    report = assess_isolation(IsolationPolicy(require_os_containment=False))
    assert report.staging_ready
    assert report.application_restrictions_configured
    assert not report.application_restrictions_enforced
    assert not report.application_restrictions_verified
    assert not report.os_containment_verified
    assert not report.execution_ready


def test_workspace_creates_and_writes_only_inside_root(tmp_path):
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        target = workspace.write_bytes("captures/frame.bin", b"abc")
        assert target.read_bytes() == b"abc"
        assert target.resolve().is_relative_to(workspace.path.resolve())


@pytest.mark.parametrize("relative", [
    "../escape.txt", "a/../../escape.txt", "C:/outside",
    r"C:\\outside", r"\\server\\share\\file",
])
def test_workspace_rejects_traversal_and_absolute_paths(tmp_path, relative):
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        with pytest.raises(WorkspaceError):
            workspace.resolve_output(relative)


def test_workspace_rejects_forbidden_root_overlap(tmp_path):
    production = tmp_path / "production"
    with pytest.raises(WorkspaceError):
        StagingWorkspace(StagingConfig(
            production / "staging", forbidden_roots=(production,)
        )).create()


def test_workspace_rejects_symlink_component_when_supported(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(WorkspaceError):
        StagingWorkspace(StagingConfig(link / "staging")).create()


def test_workspace_payload_limit(tmp_path):
    limits = ResourceLimits(max_payload_bytes=2, max_total_bytes=10)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        with pytest.raises(WorkspaceError):
            workspace.write_bytes("too-big.bin", b"abc")


def test_workspace_cleanup_removes_only_run_directory(tmp_path):
    root = tmp_path / "lab"
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_text("keep")
    workspace = StagingWorkspace(StagingConfig(root)).create()
    workspace.write_bytes("data.bin", b"x")
    workspace.cleanup()
    assert keep.read_text() == "keep" and not workspace.path.exists()


def test_resource_budget_enforces_frames_and_bytes():
    budget = ResourceBudget(ResourceLimits(max_frames=1, max_payload_bytes=4, max_total_bytes=4))
    budget.consume_frame(4)
    with pytest.raises(ResourceLimitExceeded):
        budget.consume_frame(0)


def test_resource_budget_enforces_per_payload_and_total_bytes():
    budget = ResourceBudget(ResourceLimits(max_frames=10, max_payload_bytes=4, max_total_bytes=5))
    budget.consume_frame(3)
    with pytest.raises(ResourceLimitExceeded):
        budget.consume_frame(3)


def test_manifest_round_trip_and_digest_verification(tmp_path):
    manifest = RunManifest.create(
        run_id="run-1", scenario_id="scenario-1",
        scenario_definition_digest=_digest("scenario"),
        variation_digest=_digest("variation"), seed=42,
        runner_version="c2.3", resource_policy={"max_frames": 10},
    )
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        manifest.write(workspace)
        path = workspace.path / "manifest.json"
        verified = verify_manifest(path)
        assert verified["scenario_id"] == "scenario-1" and verified["seed"] == 42


def test_manifest_detects_tampering(tmp_path):
    manifest = RunManifest.create(
        run_id="run-1", scenario_id="scenario-1",
        scenario_definition_digest=_digest("scenario"),
        variation_digest=_digest("variation"), seed=42,
        runner_version="c2.3", resource_policy={"max_frames": 10},
    )
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        manifest.write(workspace)
        path = workspace.path / "manifest.json"
        envelope = json.loads(path.read_text())
        envelope["manifest"]["seed"] = 999
        path.write_text(json.dumps(envelope))
        with pytest.raises(ManifestError):
            verify_manifest(path)


def test_manifest_rejects_invalid_digest():
    with pytest.raises(ManifestError):
        RunManifest.create(
            run_id="r", scenario_id="s", scenario_definition_digest="bad",
            variation_digest=_digest("v"), seed=1, runner_version="v",
            resource_policy={},
        )


def test_workspace_enforces_cumulative_storage_budget(tmp_path):
    limits = ResourceLimits(max_frames=10, max_payload_bytes=4, max_total_bytes=5)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        workspace.write_bytes("first.bin", b"123")
        with pytest.raises(WorkspaceError, match="total staging byte budget"):
            workspace.write_bytes("second.bin", b"456")
        assert not (workspace.path / "second.bin").exists()


def test_workspace_refuses_overwrite_to_keep_byte_accounting_sound(tmp_path):
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        workspace.write_bytes("capture.bin", b"first")
        with pytest.raises(WorkspaceError, match="overwrite"):
            workspace.write_bytes("capture.bin", b"second")
        assert (workspace.path / "capture.bin").read_bytes() == b"first"


def test_workspace_cleanup_rejects_nested_symlink_when_supported(tmp_path):
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        outside = tmp_path / "outside"
        outside.mkdir()
        link = workspace.path / "linked-dir"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unavailable on this platform")
        with pytest.raises(WorkspaceError, match="nested symlink"):
            workspace.cleanup()
        assert outside.exists()
        link.unlink()
        workspace.cleanup()


def test_manifest_resource_policy_is_deeply_immutable(tmp_path):
    policy = {"limits": {"max_frames": 10}, "tags": ["offline", "lab"]}
    manifest = RunManifest.create(
        run_id="r", scenario_id="s", scenario_definition_digest=_digest("s"),
        variation_digest=_digest("v"), seed=1, runner_version="c2.3",
        resource_policy=policy,
    )
    before = manifest.digest()
    policy["limits"]["max_frames"] = 999
    assert manifest.digest() == before
    with pytest.raises(TypeError):
        manifest.resource_policy["limits"]["max_frames"] = 999
    detached = manifest.payload()
    detached["resource_policy"]["limits"]["max_frames"] = 888
    assert manifest.digest() == before


def test_manifest_rejects_arbitrary_path_persistence(tmp_path):
    manifest = RunManifest.create(
        run_id="r", scenario_id="s", scenario_definition_digest=_digest("s"),
        variation_digest=_digest("v"), seed=1, runner_version="c2.3",
        resource_policy={},
    )
    with pytest.raises(ManifestError, match="StagingWorkspace"):
        manifest.write(tmp_path / "outside.json")


def test_manifest_persistence_obeys_workspace_and_manifest_size_limit(tmp_path):
    limits = ResourceLimits(max_payload_bytes=4096, max_total_bytes=8192, max_manifest_bytes=32)
    manifest = RunManifest.create(
        run_id="r", scenario_id="s", scenario_definition_digest=_digest("s"),
        variation_digest=_digest("v"), seed=1, runner_version="c2.3",
        resource_policy={},
    )
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        with pytest.raises(ManifestError, match="manifest exceeds"):
            manifest.write(workspace)
        assert not (workspace.path / "manifest.json").exists()


def test_write_frame_accounts_frame_and_bytes_once(tmp_path):
    limits = ResourceLimits(max_frames=1, max_payload_bytes=4, max_total_bytes=4)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        workspace.write_frame("frames/0001.bin", b"1234")
        snapshot = workspace.budget.snapshot()
        assert snapshot["frames"] == 1
        assert snapshot["total_bytes"] == 4
        with pytest.raises(WorkspaceError, match="frame-count"):
            workspace.write_frame("frames/0002.bin", b"")


def test_additional_evidence_shares_the_same_total_byte_budget(tmp_path):
    limits = ResourceLimits(max_frames=5, max_payload_bytes=4, max_total_bytes=5)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        workspace.write_frame("frames/one.bin", b"123")
        with pytest.raises(WorkspaceError, match="total staging byte budget"):
            workspace.write_bytes("reports/extra.bin", b"456")


def test_failed_workspace_write_refunds_reserved_budget(tmp_path):
    limits = ResourceLimits(max_frames=2, max_payload_bytes=8, max_total_bytes=8)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        workspace.write_bytes("same.bin", b"1234")
        with pytest.raises(WorkspaceError, match="overwrite"):
            workspace.write_bytes("same.bin", b"5678")
        assert workspace.budget.snapshot()["total_bytes"] == 4
        workspace.write_bytes("other.bin", b"5678")
        assert workspace.budget.snapshot()["total_bytes"] == 8


# C2.3 corrective hardening regression tests

def test_runtime_limit_rejects_non_finite_values():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite positive"):
            ResourceLimits(max_runtime_seconds=value)


def test_manifest_rejects_non_string_nested_mapping_keys():
    with pytest.raises(ManifestError, match="keys must be strings"):
        RunManifest.create(
            run_id="r", scenario_id="s",
            scenario_definition_digest=_digest("s"),
            variation_digest=_digest("v"), seed=1, runner_version="c2.3",
            resource_policy={"nested": {1: "ambiguous"}},
        )


def test_manifest_verification_reads_only_configured_limit(tmp_path, monkeypatch):
    manifest_path = tmp_path / "oversized.json"
    manifest_path.write_bytes(b"x" * 100)

    class BoundedReader:
        def __init__(self, data):
            self.data = data
            self.requested = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size=-1):
            self.requested.append(size)
            return self.data[:size]

    reader = BoundedReader(b"x" * 100)
    original_open = Path.open
    def controlled_open(path, *args, **kwargs):
        if Path(path) == manifest_path:
            return reader
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", controlled_open)
    with pytest.raises(ManifestError, match="size limit"):
        verify_manifest(manifest_path, max_bytes=17)
    assert reader.requested == [18]


def test_manifest_verification_rejects_invalid_size_limit(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    for value in (0, -1, True, 1.5):
        with pytest.raises(ManifestError, match="positive integer"):
            verify_manifest(path, max_bytes=value)


def test_failed_atomic_workspace_write_refunds_frame_and_cleans_partial_files(
    tmp_path, monkeypatch
):
    import simulation.lab.staging.workspace as workspace_module
    limits = ResourceLimits(max_frames=2, max_payload_bytes=8, max_total_bytes=8)
    with StagingWorkspace(StagingConfig(tmp_path / "lab", resource_limits=limits)) as workspace:
        def fail_link(source, destination):
            # Simulate a concurrent writer winning the destination race.
            Path(destination).write_bytes(b"other-writer")
            raise FileExistsError("simulated concurrent destination creation")
        monkeypatch.setattr(workspace_module.os, "link", fail_link)
        with pytest.raises(FileExistsError, match="simulated concurrent destination creation"):
            workspace.write_frame("frames/fail.bin", b"1234")
        snapshot = workspace.budget.snapshot()
        assert snapshot["frames"] == 0
        assert snapshot["total_bytes"] == 0
        # The concurrent writer's destination must never be deleted by rollback.
        assert (workspace.path / "frames" / "fail.bin").read_bytes() == b"other-writer"
        assert not list(workspace.path.rglob(".lumen-staging-*.tmp"))


def test_atomic_workspace_write_publishes_complete_file_without_overwrite(tmp_path):
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        target = workspace.write_bytes("nested/evidence.bin", b"complete")
        assert target.read_bytes() == b"complete"
        assert not list(workspace.path.rglob(".lumen-staging-*.tmp"))
        with pytest.raises(WorkspaceError, match="overwrite|existing"):
            workspace.write_bytes("nested/evidence.bin", b"replacement")
        assert target.read_bytes() == b"complete"


def test_failed_atomic_evidence_write_refunds_budget(tmp_path, monkeypatch):
    import simulation.lab.staging.workspace as workspace_module
    with StagingWorkspace(StagingConfig(tmp_path / "lab")) as workspace:
        def fail_link(source, destination):
            raise OSError("publish denied")
        monkeypatch.setattr(workspace_module.os, "link", fail_link)
        with pytest.raises(OSError, match="publish denied"):
            workspace.write_bytes("evidence.bin", b"abc")
        assert workspace.budget.snapshot()["total_bytes"] == 0
        assert not list(workspace.path.rglob(".lumen-staging-*.tmp"))

# C2.3 final manifest-construction and envelope-shape hardening tests

def test_direct_manifest_constructor_deep_freezes_resource_policy():
    policy = {"nested": {"max_frames": 10}, "tags": ["offline", "lab"]}
    manifest = RunManifest(
        run_id="r",
        scenario_id="s",
        scenario_definition_digest=_digest("s"),
        variation_digest=_digest("v"),
        seed=1,
        runner_version="c2.3",
        resource_policy=policy,
        created_utc="2026-01-01T00:00:00+00:00",
    )
    before = manifest.digest()
    policy["nested"]["max_frames"] = 999
    policy["tags"].append("mutated")
    assert manifest.digest() == before
    with pytest.raises(TypeError):
        manifest.resource_policy["nested"]["max_frames"] = 999
    with pytest.raises(AttributeError):
        manifest.resource_policy["tags"].append("mutated")


def test_direct_manifest_constructor_rejects_non_mapping_resource_policy():
    with pytest.raises(ManifestError, match="resource_policy must be a mapping"):
        RunManifest(
            run_id="r",
            scenario_id="s",
            scenario_definition_digest=_digest("s"),
            variation_digest=_digest("v"),
            seed=1,
            runner_version="c2.3",
            resource_policy=[],  # type: ignore[arg-type]
            created_utc="2026-01-01T00:00:00+00:00",
        )


@pytest.mark.parametrize("manifest_value", [[], None, "not-an-object", 42])
def test_manifest_verification_rejects_non_object_manifest(tmp_path, manifest_value):
    canonical = json.dumps(
        manifest_value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    envelope = {
        "manifest": manifest_value,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    path = tmp_path / "malformed-manifest.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ManifestError, match="manifest must be an object"):
        verify_manifest(path)


@pytest.mark.parametrize("envelope_value", [[], None, "not-an-object", 42])
def test_manifest_verification_rejects_non_object_envelope(tmp_path, envelope_value):
    path = tmp_path / "malformed-envelope.json"
    path.write_text(json.dumps(envelope_value), encoding="utf-8")
    with pytest.raises(ManifestError, match="expected an object"):
        verify_manifest(path)
