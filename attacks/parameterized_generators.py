"""Deterministic, seed-parameterized attack variations for the C2 lab.

This module is deliberately separate from attacks.advanced_generators:
the historical frozen generator and its output remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from attacks.advanced_generators import FROZEN_SCENARIOS, GeneratedFrame
from simulation.lab.scenario_schema import (
    SCHEMA_VERSION,
    ScenarioDefinition,
    canonical_json,
)

GENERATOR_VERSION = "c2.parameterized-attack.v1"
VARIATION_SCHEMA_VERSION = "c2.variation.v1"


class ParameterizedGeneratorError(ValueError):
    """Invalid source scenario, variation seed, or generator parameters."""


@dataclass(frozen=True)
class ParameterizedScenario:
    source_scenario_id: str
    source_definition_digest: str
    variation_id: str
    variation_seed: int
    effective_seed: int
    variation_digest: str
    family: str
    target_device: str
    target_topic: str
    parameters: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": VARIATION_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "source_split": "dev",
            "source_category": "attack",
            "source_scenario_id": self.source_scenario_id,
            "source_definition_digest": self.source_definition_digest,
            "variation_id": self.variation_id,
            "variation_seed": self.variation_seed,
            "effective_seed": self.effective_seed,
            "variation_digest": self.variation_digest,
            "family": self.family,
            "target": {"device": self.target_device, "topic": self.target_topic},
            "parameters": _thaw(self.parameters),
        }

    def iter_frames(self, steps: int = 120) -> Iterator[GeneratedFrame]:
        _validate_steps(steps)
        rng = random.Random(self.effective_seed)
        p = _thaw(self.parameters)
        family = self.family

        if family == "stealth_value_spoofing":
            for step in range(steps):
                value = (
                    p["base_ppm"]
                    + step * p["ramp_per_step"]
                    + rng.gauss(0.0, p["noise_std"])
                )
                value = min(1950.0, max(0.0, value))
                delta = max(0.85, min(1.15, 1.0 + rng.gauss(0.0, 0.03)))
                yield _frame(
                    self, step, "put", delta, f"{value:.4f}".encode("utf-8")
                )

        elif family == "authenticated_replay_timing_skew":
            # The phase and small jitter make variation_seed meaningful even
            # though the historical frozen replay generator was seed-insensitive.
            phase = rng.uniform(-0.5, 0.5)
            for step in range(steps):
                is_cross = step % p["skew_period"] == 0
                topic = p["cross_topic"] if is_cross else self.target_topic
                if is_cross:
                    delta = 0.12 + rng.uniform(-0.02, 0.02)
                else:
                    delta = 1.0 + 0.04 * math.sin(step + phase) + rng.uniform(-0.005, 0.005)
                yield _frame(
                    self,
                    step,
                    "put",
                    delta,
                    p["replayed_payload"].encode("utf-8"),
                    topic=topic,
                )

        elif family == "low_and_slow_reconnaissance":
            targets = p["probe_targets"]
            for step in range(steps):
                topic = targets[step % len(targets)]
                delta = max(0.05, p["probe_interval_sec"] + rng.uniform(-0.4, 0.6))
                operation = "query" if step % 2 == 0 else "declare_subscriber"
                payload = json.dumps(
                    {"probe_seq": step, "node": self.target_device},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                yield _frame(self, step, operation, delta, payload, topic=topic)

        elif family == "adaptive_burst_entropy_drift":
            for step in range(steps):
                in_burst = step % p["burst_every"] < p["burst_len"]
                if in_burst:
                    delta = max(0.05, 0.18 + rng.uniform(-0.015, 0.015))
                    token = f"{rng.getrandbits(32):08x}"
                    value = int(round(350.0 + p["drift_offset"] + rng.gauss(0.0, 8.0)))
                    payload = f"{value}|{token}".encode("utf-8")
                else:
                    delta = max(0.05, 1.0 + rng.gauss(0.0, 0.03))
                    value = int(round(352.0 + rng.gauss(0.0, 5.0)))
                    payload = str(value).encode("utf-8")
                yield _frame(self, step, "put", delta, payload)

        else:  # Defensive guard; construction already rejects unsupported families.
            raise ParameterizedGeneratorError(f"unsupported attack family: {family!r}")


def build_parameterized_scenario(
    source: ScenarioDefinition,
    *,
    variation_seed: int,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> ParameterizedScenario:
    """Build a deterministic variation from one canonical frozen DEV attack.

    Only the four canonical frozen development attack definitions are accepted.
    Held-out, regression, fault, clean, relabeled, or modified source definitions
    are rejected. Parameter changes must be supplied explicitly as overrides.
    """
    if not isinstance(source, ScenarioDefinition):
        raise ParameterizedGeneratorError("source must be a validated ScenarioDefinition")
    if isinstance(variation_seed, bool) or not isinstance(variation_seed, int):
        raise ParameterizedGeneratorError("variation_seed must be an integer")
    if variation_seed < 0 or variation_seed > (2**63 - 1):
        raise ParameterizedGeneratorError("variation_seed must be between 0 and 2**63-1")

    spec = FROZEN_SCENARIOS.get(source.scenario_id)
    if spec is None:
        raise ParameterizedGeneratorError("source scenario is not in the canonical frozen catalog")
    if spec.category != "attack" or spec.split != "dev":
        raise ParameterizedGeneratorError("only canonical development attack scenarios are allowed")

    # Rebuild the trusted registry entry and compare the complete immutable
    # definition, including its independently computed definition_digest.
    # This prevents a caller from supplying canonical-looking fields with a
    # forged or stale digest.
    from simulation.lab.scenario_registry import build_default_registry

    canonical_source = build_default_registry().get(source.scenario_id)
    if canonical_source is None or source != canonical_source:
        raise ParameterizedGeneratorError(
            "source definition does not match its canonical frozen definition and digest"
        )
    if (
        source.schema_version != SCHEMA_VERSION
        or source.scenario_version != 1
        or source.generator != "attacks.advanced_generators.iter_scenario_frames"
        or source.complexity != 1
        or source.category != "attack"
        or source.corpus_class != "adversarial"
        or source.ground_truth != "confirmed_attack"
        or source.split != "dev"
        or source.family != spec.attack_family
        or source.seed != spec.seed
        or dict(source.target) != {"device": spec.target_device, "topic": spec.target_topic}
        or _thaw(source.parameters) != spec.params
        or dict(source.provenance) != {
            "source": "attacks.advanced_generators.FROZEN_SCENARIOS",
            "source_scenario_id": spec.scenario_id,
            "frozen_historical_definition": True,
        }
    ):
        raise ParameterizedGeneratorError(
            "source definition does not match its canonical frozen development definition"
        )

    if parameter_overrides is not None and not isinstance(parameter_overrides, Mapping):
        raise ParameterizedGeneratorError("parameter_overrides must be a mapping")
    overrides = {} if parameter_overrides is None else dict(parameter_overrides)
    if any(not isinstance(key, str) or not key for key in overrides):
        raise ParameterizedGeneratorError("parameter override keys must be non-empty strings")
    parameters = _thaw(source.parameters)
    unknown = sorted(set(overrides) - set(parameters))
    if unknown:
        raise ParameterizedGeneratorError(
            f"unsupported parameter override(s): {', '.join(unknown)}"
        )
    parameters.update(overrides)
    parameters = _validate_family_parameters(source.family, parameters)
    if (
        source.family == "authenticated_replay_timing_skew"
        and parameters["cross_topic"] == source.target["topic"]
    ):
        raise ParameterizedGeneratorError(
            "cross_topic must differ from the source target topic"
        )

    seed_material = {
        "source_definition_digest": source.definition_digest,
        "variation_seed": variation_seed,
        "parameters": parameters,
        "generator_version": GENERATOR_VERSION,
    }
    digest = hashlib.sha256(canonical_json(seed_material).encode("utf-8")).hexdigest()
    effective_seed = int(digest[:16], 16)
    variation_id = f"{source.scenario_id}__var_{variation_seed}_{digest[:12]}"

    return ParameterizedScenario(
        source_scenario_id=source.scenario_id,
        source_definition_digest=source.definition_digest,
        variation_id=variation_id,
        variation_seed=variation_seed,
        effective_seed=effective_seed,
        variation_digest=digest,
        family=source.family,
        target_device=source.target["device"],
        target_topic=source.target["topic"],
        parameters=_freeze(parameters),
    )


def _validate_family_parameters(family: str, params: dict[str, Any]) -> dict[str, Any]:
    schemas = {
        "stealth_value_spoofing": {
            "base_ppm": ("number", 100.0, 1000.0),
            "ramp_per_step": ("number", 0.1, 15.0),
            "noise_std": ("number", 0.0, 20.0),
        },
        "authenticated_replay_timing_skew": {
            "replayed_payload": ("numeric_string",),
            "skew_period": ("int", 2, 60),
            "cross_topic": ("string",),
        },
        "low_and_slow_reconnaissance": {
            "probe_interval_sec": ("number", 1.0, 30.0),
            "probe_targets": ("topics", 1, 32),
        },
        "adaptive_burst_entropy_drift": {
            "burst_len": ("int", 1, 20),
            "burst_every": ("int", 2, 120),
            "drift_offset": ("number", 1.0, 1000.0),
        },
    }
    schema = schemas.get(family)
    if schema is None:
        raise ParameterizedGeneratorError(f"unsupported attack family: {family!r}")
    if set(params) != set(schema):
        raise ParameterizedGeneratorError(
            f"{family} parameters must be exactly: {', '.join(sorted(schema))}"
        )

    normalized = dict(params)
    for key, rule in schema.items():
        value = normalized[key]
        kind = rule[0]
        if kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParameterizedGeneratorError(f"{key} must be a finite number")
            if not math.isfinite(float(value)):
                raise ParameterizedGeneratorError(f"{key} must be a finite number")
            if not rule[1] <= float(value) <= rule[2]:
                raise ParameterizedGeneratorError(f"{key} must be between {rule[1]} and {rule[2]}")
            normalized[key] = float(value)
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ParameterizedGeneratorError(f"{key} must be an integer")
            if not rule[1] <= value <= rule[2]:
                raise ParameterizedGeneratorError(f"{key} must be between {rule[1]} and {rule[2]}")
        elif kind == "string":
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ParameterizedGeneratorError(f"{key} must be a non-empty string of at most 256 characters")
        elif kind == "numeric_string":
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ParameterizedGeneratorError(f"{key} must be a numeric string")
            try:
                parsed = float(value)
            except ValueError as exc:
                raise ParameterizedGeneratorError(f"{key} must be a numeric string") from exc
            if not math.isfinite(parsed) or parsed < 0:
                raise ParameterizedGeneratorError(f"{key} must encode a finite non-negative number")
        elif kind == "topics":
            if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= rule[2]:
                raise ParameterizedGeneratorError(f"{key} must contain between {rule[1]} and {rule[2]} topics")
            if any(not isinstance(topic, str) or not topic.strip() or len(topic) > 256 for topic in value):
                raise ParameterizedGeneratorError(f"{key} entries must be non-empty topic strings")
            if len(set(value)) != len(value):
                raise ParameterizedGeneratorError(f"{key} entries must be unique")
            normalized[key] = list(value)

    if family == "adaptive_burst_entropy_drift" and normalized["burst_len"] >= normalized["burst_every"]:
        raise ParameterizedGeneratorError("burst_len must be smaller than burst_every")
    return normalized


def _validate_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise ParameterizedGeneratorError("steps must be an integer between 1 and 10000")


def _frame(
    scenario: ParameterizedScenario,
    step: int,
    operation: str,
    delta_t_sec: float,
    payload: bytes,
    *,
    topic: str | None = None,
) -> GeneratedFrame:
    return GeneratedFrame(
        step=step,
        device_id=scenario.target_device,
        topic=topic or scenario.target_topic,
        operation=operation,
        delta_t_sec=round(float(delta_t_sec), 4),
        payload=payload,
        ground_truth_label="attack",
        scenario_id=scenario.variation_id,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
