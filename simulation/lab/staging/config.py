"""Immutable configuration for offline Lab staging."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class ResourceLimits:
    max_frames: int = 100_000
    max_payload_bytes: int = 1_048_576
    max_total_bytes: int = 268_435_456
    max_runtime_seconds: float = 300.0
    max_manifest_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for name in ("max_frames", "max_payload_bytes", "max_total_bytes", "max_manifest_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_runtime_seconds, bool)
            or not isinstance(self.max_runtime_seconds, (int, float))
            or not math.isfinite(self.max_runtime_seconds)
            or self.max_runtime_seconds <= 0
        ):
            raise ValueError("max_runtime_seconds must be a finite positive number")
        if self.max_payload_bytes > self.max_total_bytes:
            raise ValueError("max_payload_bytes cannot exceed max_total_bytes")

@dataclass(frozen=True)
class StagingConfig:
    """Dedicated Lab staging root; explicitly list known production roots as forbidden."""
    workspace_root: Path
    forbidden_roots: Tuple[Path, ...] = field(default_factory=tuple)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    require_os_containment: bool = True
    allow_network: bool = False
    allow_subprocess: bool = False
    allow_production_access: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).expanduser())
        object.__setattr__(self, "forbidden_roots", tuple(Path(p).expanduser() for p in self.forbidden_roots))
        if self.allow_network or self.allow_subprocess or self.allow_production_access:
            raise ValueError("Lab staging must deny network, subprocess and production access")
        if not isinstance(self.resource_limits, ResourceLimits):
            raise TypeError("resource_limits must be a ResourceLimits instance")
