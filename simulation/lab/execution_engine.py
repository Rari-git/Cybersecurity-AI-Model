"""C2.4 offline scenario orchestration for canonical development attacks.

This engine generates deterministic frame evidence locally. It does not transmit
frames, invoke Zenoh, launch subprocesses, access production services, or claim
OS-level containment. It is not an attack execution or deployment mechanism.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from attacks.parameterized_generators import (
    ParameterizedGeneratorError,
    build_parameterized_scenario,
)
from simulation.lab.scenario_registry import (
    ScenarioRegistry,
    build_default_registry,
)
from simulation.lab.staging import (
    ManifestError,
    ResourceLimits,
    RunManifest,
    StagingConfig,
    StagingWorkspace,
    WorkspaceError,
)

ENGINE_VERSION = "c2.offline-scenario-engine.v1"
FRAME_SCHEMA_VERSION = "c2.generated-frame.v1"
SUMMARY_SCHEMA_VERSION = "c2.offline-run-summary.v1"
MAX_STEPS = 10_000


class ScenarioExecutionError(RuntimeError):
    """A local offline generation run failed or violated its contract."""


@dataclass(frozen=True)
class ScenarioRunResult:
    run_id: str
    scenario_id: str
    variation_id: str
    variation_digest: str
    frame_count: int
    frames_sha256: str
    manifest_sha256: str
    output_directory: Path
    total_bytes: int
    elapsed_seconds: float
    execution_mode: str = "offline_frame_generation_only"
    external_execution_performed: bool = False
    os_containment_verified: bool = False


class OfflineScenarioEngine:
    """Generate and persist frames for canonical development attack scenarios.

    `run_development_attack` is an offline data-generation operation only.
    Frame timing values are recorded as metadata; no real-time sleeping,
    network I/O, subprocess execution, Zenoh publication, or production access
    is performed.
    """

    def __init__(
        self,
        config: StagingConfig,
        *,
        registry: ScenarioRegistry | None = None,
    ) -> None:
        if not isinstance(config, StagingConfig):
            raise TypeError("config must be a StagingConfig")
        if config.allow_network or config.allow_subprocess or config.allow_production_access:
            raise ScenarioExecutionError("offline scenario engine requires denied network, subprocess and production access")
        if registry is not None and not isinstance(registry, ScenarioRegistry):
            raise TypeError("registry must be a ScenarioRegistry")
        self.config = config
        self.registry = registry if registry is not None else build_default_registry()

    def run_development_attack(
        self,
        scenario_id: str,
        *,
        variation_seed: int,
        steps: int = 120,
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> ScenarioRunResult:
        """Generate one deterministic variation from a canonical DEV attack.

        Held-out, regression, fault, protocol, clean, modified and relabeled
        scenario definitions are rejected by the canonical parameterized
        generator. The engine never executes a generated frame externally.
        """
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ScenarioExecutionError("scenario_id must be a non-empty string")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
            raise ScenarioExecutionError(f"steps must be an integer between 1 and {MAX_STEPS}")
        if steps > self.config.resource_limits.max_frames:
            raise ScenarioExecutionError("requested steps exceed the configured frame budget")

        try:
            source = self.registry.get(scenario_id)
            variation = build_parameterized_scenario(
                source,
                variation_seed=variation_seed,
                parameter_overrides=parameter_overrides,
            )
        except (KeyError, ParameterizedGeneratorError, ValueError, TypeError) as exc:
            raise ScenarioExecutionError(f"scenario preflight rejected: {exc}") from exc

        workspace = StagingWorkspace(self.config)
        try:
            workspace.create()
            frames_hasher = hashlib.sha256()
            frame_count = 0

            for expected_step, frame in enumerate(variation.iter_frames(steps)):
                if frame.step != expected_step:
                    raise ScenarioExecutionError("generator emitted a non-contiguous frame sequence")
                if frame.scenario_id != variation.variation_id:
                    raise ScenarioExecutionError("generator emitted an unexpected scenario identifier")
                if frame.ground_truth_label != "attack":
                    raise ScenarioExecutionError("development attack generator emitted an unexpected ground-truth label")
                if not isinstance(frame.payload, bytes):
                    raise ScenarioExecutionError("generator emitted a non-bytes payload")
                if isinstance(frame.delta_t_sec, bool) or not isinstance(frame.delta_t_sec, (int, float)):
                    raise ScenarioExecutionError("generator emitted an invalid timing delta")

                record = {
                    "schema_version": FRAME_SCHEMA_VERSION,
                    "step": frame.step,
                    "device_id": frame.device_id,
                    "topic": frame.topic,
                    "operation": frame.operation,
                    "delta_t_sec": float(frame.delta_t_sec),
                    "payload_base64": base64.b64encode(frame.payload).decode("ascii"),
                    "ground_truth_label": frame.ground_truth_label,
                    "scenario_id": frame.scenario_id,
                    "dropped": bool(frame.dropped),
                }
                encoded = json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                # Length-prefix records so the digest is unambiguous.
                frames_hasher.update(len(encoded).to_bytes(8, "big"))
                frames_hasher.update(encoded)
                workspace.write_frame(f"frames/frame_{expected_step:06d}.json", encoded)
                frame_count += 1

            if frame_count != steps:
                raise ScenarioExecutionError(
                    f"generator produced {frame_count} frames; expected {steps}"
                )

            frames_digest = frames_hasher.hexdigest()
            budget_snapshot = workspace.budget.snapshot()
            summary = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "engine_version": ENGINE_VERSION,
                "execution_mode": "offline_frame_generation_only",
                "external_execution_performed": False,
                "os_containment_verified": False,
                "run_id": workspace.run_id,
                "source_scenario_id": source.scenario_id,
                "source_definition_digest": source.definition_digest,
                "variation_id": variation.variation_id,
                "variation_digest": variation.variation_digest,
                "variation_seed": variation.variation_seed,
                "frame_count": frame_count,
                "frames_sha256": frames_digest,
                "budget_snapshot_stage": "after_frames_before_summary_and_manifest",
                "budget_snapshot_excludes": ["run_summary.json", "run_manifest.json"],
                "budget": budget_snapshot,
            }
            summary_bytes = json.dumps(
                summary,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            workspace.write_bytes("run_summary.json", summary_bytes)

            manifest = RunManifest.create(
                run_id=workspace.run_id,
                scenario_id=source.scenario_id,
                scenario_definition_digest=source.definition_digest,
                variation_digest=variation.variation_digest,
                seed=variation.variation_seed,
                runner_version=ENGINE_VERSION,
                resource_policy={
                    "max_frames": self.config.resource_limits.max_frames,
                    "max_payload_bytes": self.config.resource_limits.max_payload_bytes,
                    "max_total_bytes": self.config.resource_limits.max_total_bytes,
                    "max_runtime_seconds": self.config.resource_limits.max_runtime_seconds,
                    "max_manifest_bytes": self.config.resource_limits.max_manifest_bytes,
                    "requested_steps": steps,
                    "execution_mode": "offline_frame_generation_only",
                    "external_execution_performed": False,
                    "os_containment_verified": False,
                },
            )
            manifest_digest = manifest.write(workspace, "run_manifest.json")
            final_budget = workspace.budget.snapshot()

            return ScenarioRunResult(
                run_id=workspace.run_id,
                scenario_id=source.scenario_id,
                variation_id=variation.variation_id,
                variation_digest=variation.variation_digest,
                frame_count=frame_count,
                frames_sha256=frames_digest,
                manifest_sha256=manifest_digest,
                output_directory=workspace.path,
                total_bytes=int(final_budget["total_bytes"]),
                elapsed_seconds=float(final_budget["elapsed_seconds"]),
            )
        except Exception as exc:
            try:
                workspace.cleanup()
            except Exception as cleanup_exc:
                raise ScenarioExecutionError(
                    f"offline generation failed ({exc}); workspace cleanup also failed ({cleanup_exc})"
                ) from exc
            if isinstance(exc, ScenarioExecutionError):
                raise
            if isinstance(exc, (WorkspaceError, ManifestError)):
                raise ScenarioExecutionError(f"offline generation failed: {exc}") from exc
            raise ScenarioExecutionError(f"offline generation failed: {exc}") from exc
