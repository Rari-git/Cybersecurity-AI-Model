"""Read-only C2.1 registry built from frozen historical scenario definitions."""
from __future__ import annotations
from collections import Counter
from types import MappingProxyType
from typing import Iterable
from attacks.advanced_generators import FROZEN_SCENARIOS
from .scenario_schema import ScenarioDefinition, validate_scenario_definition

_PROTOCOL_CATALOG = {
    "authenticated_replay_duplicate_sequence": {
        "family": "authenticated_replay_duplicate_sequence",
        "generator": "storage.admission_gate.AdmissionGate.evaluate_v6_and_ingest",
        "target": {"device": "gas_sensor_01", "topic": "home/sensors/gas/ppm"},
        "expected_reasons": ["REPLAY_OR_OUT_OF_ORDER_SEQUENCE"],
    },
    "unregistered_boot_epoch": {
        "family": "unregistered_boot_epoch",
        "generator": "storage.admission_gate.AdmissionGate.evaluate_v6_and_ingest",
        "target": {"device": "gas_sensor_01", "topic": "home/sensors/gas/ppm"},
        "expected_reasons": ["BOOT_NOT_REGISTERED", "STALE_BOOT_ID"],
    },
}

class ScenarioRegistryError(ValueError):
    """Registry lookup or construction failed."""

class ScenarioRegistry:
    """Immutable, in-memory index of validated scenario definitions."""
    def __init__(self, definitions: Iterable[dict] | None = None):
        raw = list(definitions) if definitions is not None else _default_catalog()
        validated = [validate_scenario_definition(item) for item in raw]
        by_id = {}
        for definition in validated:
            if definition.scenario_id in by_id:
                raise ScenarioRegistryError(f"duplicate scenario_id: {definition.scenario_id}")
            by_id[definition.scenario_id] = definition
        self._by_id = MappingProxyType(by_id)

    def get(self, scenario_id: str) -> ScenarioDefinition:
        try:
            return self._by_id[scenario_id]
        except KeyError as exc:
            raise ScenarioRegistryError(f"unknown scenario_id: {scenario_id}") from exc

    def list(self, *, category=None, corpus_class=None, split=None, family=None):
        result = []
        for item in self._by_id.values():
            if category is not None and item.category != category:
                continue
            if corpus_class is not None and item.corpus_class != corpus_class:
                continue
            if split is not None and item.split != split:
                continue
            if family is not None and item.family != family:
                continue
            result.append(item)
        return tuple(sorted(result, key=lambda entry: entry.scenario_id))

    def development_scenarios(self):
        return self.list(split="dev")

    def held_out_scenarios(self):
        return self.list(split="held_out")

    def regression_scenarios(self):
        return self.list(split="regression")

    def report(self) -> dict:
        counts = Counter(item.corpus_class for item in self._by_id.values())
        splits = Counter(item.split for item in self._by_id.values())
        return {
            "schema_version": "c2.scenario.v1",
            "scenario_count": len(self._by_id),
            "counts_by_corpus_class": dict(sorted(counts.items())),
            "counts_by_split": dict(sorted(splits.items())),
            "scenario_ids": sorted(self._by_id),
        }

    def __len__(self):
        return len(self._by_id)

def _default_catalog():
    definitions = []
    for scenario_id, spec in FROZEN_SCENARIOS.items():
        if spec.category == "attack":
            category, corpus_class, ground_truth = "attack", "adversarial", "confirmed_attack"
        elif spec.category == "fault":
            category, corpus_class, ground_truth = "fault", "hardware_fault", "confirmed_hardware_fault"
        else:
            raise ScenarioRegistryError(f"unsupported frozen generator category {spec.category!r}")
        definitions.append({
            "schema_version": "c2.scenario.v1",
            "scenario_id": scenario_id,
            "scenario_version": 1,
            "family": spec.attack_family,
            "category": category,
            "corpus_class": corpus_class,
            "split": spec.split,
            "generator": "attacks.advanced_generators.iter_scenario_frames",
            "seed": int(spec.seed),
            "parameters": spec.params,
            "target": {"device": spec.target_device, "topic": spec.target_topic},
            "ground_truth": ground_truth,
            "complexity": 1,
            "provenance": {
                "source": "attacks.advanced_generators.FROZEN_SCENARIOS",
                "source_scenario_id": scenario_id,
                "frozen_historical_definition": True,
            },
        })
    for scenario_id, item in _PROTOCOL_CATALOG.items():
        definitions.append({
            "schema_version": "c2.scenario.v1",
            "scenario_id": scenario_id,
            "scenario_version": 1,
            "family": item["family"],
            "category": "protocol",
            "corpus_class": "protocol_violation",
            "split": "regression",
            "generator": item["generator"],
            "seed": 0,
            "parameters": {},
            "target": item["target"],
            "ground_truth": "confirmed_protocol_violation",
            "complexity": 1,
            "provenance": {
                "source": "simulation.capture_regression_b4.PROTOCOL_SCENARIOS",
                "runner_version": "stage-b4-regression-capture-v1",
                "seed_policy": "runner_supplied_at_execution",
                "expected_reasons": item["expected_reasons"],
            },
        })
    return definitions

def build_default_registry() -> ScenarioRegistry:
    return ScenarioRegistry()
