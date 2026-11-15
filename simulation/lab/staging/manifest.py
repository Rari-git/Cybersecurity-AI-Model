"""Canonical, deeply immutable, workspace-persisted staging manifests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .workspace import StagingWorkspace


class ManifestError(ValueError):
    pass


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError("manifest mapping keys must be strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ManifestError(f"unsupported manifest value type: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _thaw(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"manifest contains non-canonical data: {exc}") from exc


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    scenario_id: str
    scenario_definition_digest: str
    variation_digest: str
    seed: int
    runner_version: str
    resource_policy: Mapping[str, Any]
    created_utc: str
    schema_version: str = "c2.staging-manifest.v1"

    def __post_init__(self) -> None:
        # Enforce deep immutability even when callers bypass create().
        if not isinstance(self.resource_policy, Mapping):
            raise ManifestError("resource_policy must be a mapping")
        frozen_policy = _freeze(self.resource_policy)
        _canonical_json({"resource_policy": frozen_policy})
        object.__setattr__(self, "resource_policy", frozen_policy)

    @classmethod
    def create(
        cls, *, run_id: str, scenario_id: str,
        scenario_definition_digest: str, variation_digest: str,
        seed: int, runner_version: str, resource_policy: Mapping[str, Any],
    ) -> "RunManifest":
        if not run_id or not scenario_id or not runner_version:
            raise ManifestError("run_id, scenario_id and runner_version are required")
        for name, digest in (
            ("scenario_definition_digest", scenario_definition_digest),
            ("variation_digest", variation_digest),
        ):
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
            ):
                raise ManifestError(f"{name} must be a SHA-256 hex digest")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestError("seed must be an integer")
        frozen_policy = _freeze(dict(resource_policy))
        # Validate serializability, finite floats and string-compatible mapping keys.
        _canonical_json({"resource_policy": frozen_policy})
        return cls(
            run_id, scenario_id, scenario_definition_digest.lower(),
            variation_digest.lower(), seed, runner_version, frozen_policy,
            datetime.now(timezone.utc).isoformat(),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_definition_digest": self.scenario_definition_digest,
            "variation_digest": self.variation_digest,
            "seed": self.seed,
            "runner_version": self.runner_version,
            "resource_policy": _thaw(self.resource_policy),
            "created_utc": self.created_utc,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload())).hexdigest()

    def write(self, workspace: "StagingWorkspace", relative_path: str = "manifest.json") -> str:
        """Write through a staging workspace; arbitrary filesystem paths are rejected."""
        from .workspace import StagingWorkspace, WorkspaceError

        if not isinstance(workspace, StagingWorkspace):
            raise ManifestError("manifest persistence requires a StagingWorkspace")
        envelope = {
            "manifest": self.payload(),
            "manifest_sha256": self.digest(),
        }
        encoded = _canonical_json(envelope)
        try:
            workspace.write_manifest(relative_path, encoded)
        except WorkspaceError as exc:
            raise ManifestError(str(exc)) from exc
        return envelope["manifest_sha256"]


def verify_manifest(path: Path, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ManifestError("max_bytes must be a positive integer")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ManifestError(f"unable to read manifest: {exc}") from exc
    if len(raw) > max_bytes:
        raise ManifestError("manifest exceeds configured size limit")
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("invalid manifest envelope") from exc

    if not isinstance(envelope, dict):
        raise ManifestError("invalid manifest envelope: expected an object")
    manifest = envelope.get("manifest")
    expected = envelope.get("manifest_sha256")
    if not isinstance(manifest, dict):
        raise ManifestError("invalid manifest envelope: manifest must be an object")
    if not isinstance(expected, str):
        raise ManifestError("invalid manifest envelope: manifest_sha256 must be a string")

    actual = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    if actual != expected:
        raise ManifestError("manifest SHA-256 integrity verification failed")
    if manifest.get("schema_version") != "c2.staging-manifest.v1":
        raise ManifestError("unsupported staging manifest schema")
    return manifest
