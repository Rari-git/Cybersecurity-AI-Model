from __future__ import annotations
import copy
import pytest
from simulation.lab.scenario_registry import ScenarioRegistry, ScenarioRegistryError, build_default_registry
from simulation.lab.scenario_schema import ScenarioValidationError, validate_scenario_definition

def _example():
    return {
        "schema_version": "c2.scenario.v1", "scenario_id": "example_attack",
        "scenario_version": 1, "family": "stealth_value_spoofing",
        "category": "attack", "corpus_class": "adversarial", "split": "dev",
        "generator": "parameterized", "seed": 101,
        "parameters": {"nested": {"values": [1, 2]}},
        "target": {"device": "sensor_01", "topic": "ingest/sensors/temperature"},
        "ground_truth": "confirmed_attack", "complexity": 1, "provenance": {},
    }

def test_default_registry_preserves_historical_catalog():
    registry = build_default_registry()
    report = registry.report()
    assert report["scenario_count"] == 12
    assert report["counts_by_corpus_class"] == {
        "adversarial": 8, "hardware_fault": 2, "protocol_violation": 2,
    }
    assert report["counts_by_split"] == {"dev": 4, "held_out": 6, "regression": 2}
    assert len(registry.list(category="attack")) == 8

def test_dev_heldout_and_regression_are_separate():
    registry = build_default_registry()
    assert len(registry.development_scenarios()) == 4
    assert all(x.split == "dev" for x in registry.development_scenarios())
    assert len(registry.held_out_scenarios()) == 6
    assert all(x.split == "held_out" for x in registry.held_out_scenarios())
    assert len(registry.regression_scenarios()) == 2
    assert all(x.category == "protocol" for x in registry.regression_scenarios())

def test_protocol_is_not_mislabeled_as_adversarial():
    items = build_default_registry().list(category="protocol")
    assert {x.corpus_class for x in items} == {"protocol_violation"}
    assert {x.ground_truth for x in items} == {"confirmed_protocol_violation"}

def test_digest_is_deterministic():
    raw = _example()
    assert validate_scenario_definition(raw).definition_digest == validate_scenario_definition(copy.deepcopy(raw)).definition_digest
    assert len(validate_scenario_definition(raw).definition_digest) == 64

def test_nested_data_is_immutable_and_export_is_detached():
    definition = validate_scenario_definition(_example())
    with pytest.raises(TypeError):
        definition.parameters["new"] = "value"
    with pytest.raises(TypeError):
        definition.parameters["nested"]["new"] = "value"
    exported = definition.to_dict()
    exported["parameters"]["nested"]["values"].append(3)
    assert definition.to_dict()["parameters"]["nested"]["values"] == [1, 2]

def test_caller_mutation_does_not_change_definition():
    raw = _example()
    definition = validate_scenario_definition(raw)
    digest = definition.definition_digest
    raw["parameters"]["nested"]["values"].append(999)
    assert definition.definition_digest == digest

def test_duplicate_ids_rejected():
    raw = _example()
    with pytest.raises(ScenarioRegistryError, match="duplicate"):
        ScenarioRegistry([raw, copy.deepcopy(raw)])

def test_unknown_id_rejected():
    with pytest.raises(ScenarioRegistryError, match="unknown"):
        build_default_registry().get("not_registered")

@pytest.mark.parametrize("field,value,message", [
    ("schema_version", "c2.scenario.v99", "unsupported schema_version"),
    ("category", "unknown", "unsupported category"),
    ("corpus_class", "protocol_violation", "requires corpus_class"),
    ("ground_truth", "confirmed_normal", "requires ground_truth"),
    ("split", "test", "unsupported split"),
    ("scenario_version", True, "must be an integer"),
    ("seed", -1, "seed must be >= 0"),
    ("complexity", 0, "complexity must be >= 1"),
])
def test_invalid_fields_rejected(field, value, message):
    raw = _example()
    raw[field] = value
    with pytest.raises(ScenarioValidationError, match=message):
        validate_scenario_definition(raw)

def test_protocol_requires_regression_split():
    raw = _example()
    raw.update(category="protocol", corpus_class="protocol_violation",
               ground_truth="confirmed_protocol_violation", split="dev")
    with pytest.raises(ScenarioValidationError, match="regression split"):
        validate_scenario_definition(raw)

def test_fault_cannot_be_development_data():
    raw = _example()
    raw.update(category="fault", corpus_class="hardware_fault",
               ground_truth="confirmed_hardware_fault", split="dev")
    with pytest.raises(ScenarioValidationError, match="not development data"):
        validate_scenario_definition(raw)

def test_non_finite_parameters_rejected():
    raw = _example()
    raw["parameters"] = {"bad": float("nan")}
    with pytest.raises(ScenarioValidationError, match="non-finite"):
        validate_scenario_definition(raw)

def test_target_shape_is_strict():
    raw = _example()
    raw["target"] = {"device": "sensor_01"}
    with pytest.raises(ScenarioValidationError, match="exactly device and topic"):
        validate_scenario_definition(raw)
