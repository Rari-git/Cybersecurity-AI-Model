"""C2.1 versioned scenario schema and immutable definitions.
Pure module: no filesystem, database, network, Zenoh, PKI, model, or runtime operations.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = "c2.scenario.v1"
CATEGORY_RULES = {
    "attack": ("adversarial", "confirmed_attack"),
    "protocol": ("protocol_violation", "confirmed_protocol_violation"),
    "fault": ("hardware_fault", "confirmed_hardware_fault"),
    "clean": ("clean_control", "confirmed_normal"),
}
ALLOWED_SPLITS = frozenset({"dev", "held_out", "regression"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

class ScenarioValidationError(ValueError):
    """A scenario definition violates the C2 scenario contract."""

def _validate_json_value(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScenarioValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ScenarioValidationError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ScenarioValidationError(f"{path} contains unsupported type {type(value).__name__}")

def canonical_json(value: Any) -> str:
    _validate_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

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

@dataclass(frozen=True)
class ScenarioDefinition:
    schema_version: str
    scenario_id: str
    scenario_version: int
    family: str
    category: str
    corpus_class: str
    split: str
    generator: str
    seed: int
    parameters: Mapping[str, Any]
    target: Mapping[str, Any]
    ground_truth: str
    complexity: int
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "family": self.family,
            "category": self.category,
            "corpus_class": self.corpus_class,
            "split": self.split,
            "generator": self.generator,
            "seed": self.seed,
            "parameters": _thaw(self.parameters),
            "target": _thaw(self.target),
            "ground_truth": self.ground_truth,
            "complexity": self.complexity,
            "provenance": _thaw(self.provenance),
        }

    @property
    def definition_digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

def validate_scenario_definition(raw: Mapping[str, Any]) -> ScenarioDefinition:
    if not isinstance(raw, Mapping):
        raise ScenarioValidationError("scenario definition must be a mapping")
    data = dict(raw)
    required = {
        "schema_version", "scenario_id", "scenario_version", "family", "category",
        "corpus_class", "split", "generator", "seed", "parameters", "target",
        "ground_truth", "complexity", "provenance",
    }
    missing = sorted(required - set(data))
    extra = sorted(set(data) - required)
    if missing:
        raise ScenarioValidationError(f"missing required fields: {', '.join(missing)}")
    if extra:
        raise ScenarioValidationError(f"unsupported fields: {', '.join(extra)}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ScenarioValidationError(f"unsupported schema_version: {data['schema_version']!r}")
    for field in ("scenario_id", "family", "generator"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ScenarioValidationError(f"{field} must be a non-empty string")
    if not _ID_RE.fullmatch(data["scenario_id"]):
        raise ScenarioValidationError("scenario_id contains unsupported characters")
    for field in ("scenario_version", "seed", "complexity"):
        if isinstance(data[field], bool) or not isinstance(data[field], int):
            raise ScenarioValidationError(f"{field} must be an integer")
    if data["scenario_version"] < 1:
        raise ScenarioValidationError("scenario_version must be >= 1")
    if data["seed"] < 0:
        raise ScenarioValidationError("seed must be >= 0")
    if data["complexity"] < 1:
        raise ScenarioValidationError("complexity must be >= 1")
    category = data["category"]
    if not isinstance(category, str) or category not in CATEGORY_RULES:
        raise ScenarioValidationError(f"unsupported category: {category!r}")
    expected_class, expected_truth = CATEGORY_RULES[category]
    if data["corpus_class"] != expected_class:
        raise ScenarioValidationError(f"category {category!r} requires corpus_class {expected_class!r}")
    if data["ground_truth"] != expected_truth:
        raise ScenarioValidationError(f"category {category!r} requires ground_truth {expected_truth!r}")
    if not isinstance(data["split"], str) or data["split"] not in ALLOWED_SPLITS:
        raise ScenarioValidationError(f"unsupported split: {data['split']!r}")
    if category == "protocol" and data["split"] != "regression":
        raise ScenarioValidationError("protocol scenarios must use the regression split")
    if category == "fault" and data["split"] == "dev":
        raise ScenarioValidationError("frozen hardware-fault scenarios are not development data")
    for field in ("parameters", "target", "provenance"):
        if not isinstance(data[field], Mapping):
            raise ScenarioValidationError(f"{field} must be an object")
        _validate_json_value(dict(data[field]), field)
    target = dict(data["target"])
    if set(target) != {"device", "topic"}:
        raise ScenarioValidationError("target must contain exactly device and topic")
    if any(not isinstance(target[key], str) or not target[key].strip() for key in ("device", "topic")):
        raise ScenarioValidationError("target.device and target.topic must be non-empty strings")
    detached = json.loads(canonical_json(data))
    return ScenarioDefinition(
        schema_version=detached["schema_version"], scenario_id=detached["scenario_id"],
        scenario_version=detached["scenario_version"], family=detached["family"],
        category=detached["category"], corpus_class=detached["corpus_class"],
        split=detached["split"], generator=detached["generator"], seed=detached["seed"],
        parameters=_freeze(detached["parameters"]), target=_freeze(detached["target"]),
        ground_truth=detached["ground_truth"], complexity=detached["complexity"],
        provenance=_freeze(detached["provenance"]),
    )
