"""C2 attack simulation laboratory foundations (registry-only in C2.1)."""
from .scenario_registry import ScenarioRegistry, ScenarioRegistryError, build_default_registry
from .scenario_schema import (
    SCHEMA_VERSION, ScenarioDefinition, ScenarioValidationError, validate_scenario_definition,
)
__all__ = [
    "SCHEMA_VERSION", "ScenarioDefinition", "ScenarioRegistry", "ScenarioRegistryError",
    "ScenarioValidationError", "build_default_registry", "validate_scenario_definition",
]
