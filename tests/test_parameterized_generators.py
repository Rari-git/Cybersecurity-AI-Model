from __future__ import annotations

from dataclasses import replace

import pytest

from attacks.advanced_generators import FROZEN_SCENARIOS, iter_scenario_frames
from attacks.parameterized_generators import (
    ParameterizedGeneratorError,
    build_parameterized_scenario,
)
from simulation.lab.scenario_registry import build_default_registry


@pytest.fixture()
def registry():
    return build_default_registry()


def _frames(scenario, steps=40):
    return list(scenario.iter_frames(steps))


@pytest.mark.parametrize("scenario_id", [
    "atk1_stealth_spoof_dev",
    "atk2_replay_skew_dev",
    "atk3_low_slow_recon_dev",
    "atk4_adaptive_burst_dev",
])
def test_all_canonical_dev_attack_families_generate(scenario_id, registry):
    source = registry.get(scenario_id)
    variation = build_parameterized_scenario(source, variation_seed=17)
    frames = _frames(variation, 25)
    assert len(frames) == 25
    assert all(frame.ground_truth_label == "attack" for frame in frames)
    assert all(frame.scenario_id == variation.variation_id for frame in frames)
    assert [frame.step for frame in frames] == list(range(25))


@pytest.mark.parametrize("scenario_id", [
    "atk1_stealth_spoof_dev",
    "atk2_replay_skew_dev",
    "atk3_low_slow_recon_dev",
    "atk4_adaptive_burst_dev",
])
def test_same_seed_is_bitwise_deterministic(scenario_id, registry):
    source = registry.get(scenario_id)
    left = build_parameterized_scenario(source, variation_seed=987654)
    right = build_parameterized_scenario(source, variation_seed=987654)
    assert left.metadata() == right.metadata()
    assert _frames(left, 80) == _frames(right, 80)


@pytest.mark.parametrize("scenario_id", [
    "atk1_stealth_spoof_dev",
    "atk2_replay_skew_dev",
    "atk3_low_slow_recon_dev",
    "atk4_adaptive_burst_dev",
])
def test_different_seeds_change_generated_stream_for_every_family(scenario_id, registry):
    source = registry.get(scenario_id)
    left = build_parameterized_scenario(source, variation_seed=1001)
    right = build_parameterized_scenario(source, variation_seed=1002)
    assert _frames(left, 80) != _frames(right, 80)
    assert left.variation_id != right.variation_id


def test_held_out_source_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_heldout")
    with pytest.raises(ParameterizedGeneratorError, match="only canonical development"):
        build_parameterized_scenario(source, variation_seed=1)


def test_held_out_definition_cannot_be_relabelled_as_dev(registry):
    heldout = registry.get("atk1_stealth_spoof_heldout")
    relabelled = replace(heldout, split="dev")
    with pytest.raises(ParameterizedGeneratorError):
        build_parameterized_scenario(relabelled, variation_seed=1)


def test_hardware_fault_source_is_rejected(registry):
    source = registry.get("fault1_frozen_sensor")
    with pytest.raises(ParameterizedGeneratorError, match="only canonical development"):
        build_parameterized_scenario(source, variation_seed=1)


def test_modified_frozen_source_definition_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    altered = replace(source, seed=source.seed + 1)
    with pytest.raises(ParameterizedGeneratorError, match="does not match"):
        build_parameterized_scenario(altered, variation_seed=1)


def test_noncanonical_scenario_id_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    altered = replace(source, scenario_id="renamed_dev_scenario")
    with pytest.raises(ParameterizedGeneratorError, match="not in the canonical"):
        build_parameterized_scenario(altered, variation_seed=1)


def test_extra_source_provenance_field_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    altered_provenance = {
        **dict(source.provenance),
        "unexpected": "tampered",
    }
    altered = replace(source, provenance=altered_provenance)
    with pytest.raises(ParameterizedGeneratorError, match="does not match"):
        build_parameterized_scenario(altered, variation_seed=1)


def test_valid_family_parameter_override_changes_variation(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    base = build_parameterized_scenario(source, variation_seed=7)
    changed = build_parameterized_scenario(
        source, variation_seed=7, parameter_overrides={"ramp_per_step": 4.0}
    )
    assert _frames(base, 30) != _frames(changed, 30)
    assert changed.parameters["ramp_per_step"] == 4.0


def test_unknown_parameter_override_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    with pytest.raises(ParameterizedGeneratorError, match="unsupported parameter"):
        build_parameterized_scenario(
            source, variation_seed=1, parameter_overrides={"unexpected": 1}
        )


def test_parameter_overrides_must_be_a_mapping(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    with pytest.raises(ParameterizedGeneratorError, match="must be a mapping"):
        build_parameterized_scenario(source, variation_seed=1, parameter_overrides=[1, 2])


def test_parameter_override_keys_must_be_strings(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    with pytest.raises(ParameterizedGeneratorError, match="keys must be"):
        build_parameterized_scenario(source, variation_seed=1, parameter_overrides={1: "bad"})


@pytest.mark.parametrize("overrides, message", [
    ({"base_ppm": float("nan")}, "finite number"),
    ({"base_ppm": 99.0}, "between 100.0 and 1000.0"),
    ({"noise_std": True}, "finite number"),
    ({"ramp_per_step": float("inf")}, "finite number"),
])
def test_invalid_stealth_parameters_are_rejected(registry, overrides, message):
    source = registry.get("atk1_stealth_spoof_dev")
    with pytest.raises(ParameterizedGeneratorError, match=message):
        build_parameterized_scenario(source, variation_seed=1, parameter_overrides=overrides)


def test_replay_parameters_are_validated(registry):
    source = registry.get("atk2_replay_skew_dev")
    with pytest.raises(ParameterizedGeneratorError, match="numeric string"):
        build_parameterized_scenario(
            source, variation_seed=1, parameter_overrides={"replayed_payload": "not-numeric"}
        )
    with pytest.raises(ParameterizedGeneratorError, match="between 2 and 60"):
        build_parameterized_scenario(
            source, variation_seed=1, parameter_overrides={"skew_period": 1}
        )
    with pytest.raises(
        ParameterizedGeneratorError, match="cross_topic must differ from the source target topic"
    ):
        build_parameterized_scenario(
            source,
            variation_seed=1,
            parameter_overrides={"cross_topic": source.target["topic"]},
        )


def test_recon_topics_must_be_unique_nonempty_topics(registry):
    source = registry.get("atk3_low_slow_recon_dev")
    with pytest.raises(ParameterizedGeneratorError, match="unique"):
        build_parameterized_scenario(
            source,
            variation_seed=1,
            parameter_overrides={"probe_targets": ["home/a", "home/a"]},
        )


def test_burst_relationship_is_validated(registry):
    source = registry.get("atk4_adaptive_burst_dev")
    with pytest.raises(ParameterizedGeneratorError, match="smaller than"):
        build_parameterized_scenario(
            source,
            variation_seed=1,
            parameter_overrides={"burst_len": 12, "burst_every": 12},
        )


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_variation_seed_is_rejected(seed, registry):
    source = registry.get("atk1_stealth_spoof_dev")
    with pytest.raises(ParameterizedGeneratorError, match="variation_seed"):
        build_parameterized_scenario(source, variation_seed=seed)


@pytest.mark.parametrize("steps", [0, -1, 10001, True, 1.5])
def test_invalid_step_count_is_rejected(steps, registry):
    variation = build_parameterized_scenario(
        registry.get("atk1_stealth_spoof_dev"), variation_seed=1
    )
    with pytest.raises(ParameterizedGeneratorError, match="steps"):
        list(variation.iter_frames(steps))


def test_modified_source_definition_is_rejected(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    altered = replace(source, seed=source.seed + 1)

    # The digest is a computed property over the immutable definition fields.
    assert altered.definition_digest != source.definition_digest

    with pytest.raises(ParameterizedGeneratorError, match="canonical frozen definition and digest"):
        build_parameterized_scenario(altered, variation_seed=1)


def test_metadata_contains_source_and_variation_provenance(registry):
    source = registry.get("atk1_stealth_spoof_dev")
    variation = build_parameterized_scenario(source, variation_seed=77)
    metadata = variation.metadata()
    assert metadata["schema_version"] == "c2.variation.v1"
    assert metadata["source_scenario_id"] == source.scenario_id
    assert metadata["source_definition_digest"] == source.definition_digest
    assert metadata["variation_seed"] == 77
    assert metadata["effective_seed"] == variation.effective_seed
    assert len(metadata["variation_digest"]) == 64
    assert metadata["generator_version"] == "c2.parameterized-attack.v1"


def test_variation_parameters_are_immutable_and_metadata_is_detached(registry):
    variation = build_parameterized_scenario(
        registry.get("atk3_low_slow_recon_dev"), variation_seed=77
    )
    with pytest.raises(TypeError):
        variation.parameters["probe_interval_sec"] = 8.0
    exported = variation.metadata()
    exported["parameters"]["probe_targets"].append("tampered/topic")
    assert len(variation.parameters["probe_targets"]) == 3


def test_historical_frozen_generator_output_is_unchanged(registry):
    before = list(iter_scenario_frames("atk1_stealth_spoof_dev", steps=30))
    build_parameterized_scenario(
        registry.get("atk1_stealth_spoof_dev"), variation_seed=123
    )
    after = list(iter_scenario_frames("atk1_stealth_spoof_dev", steps=30))
    assert before == after
    assert all(frame.scenario_id == "atk1_stealth_spoof_dev" for frame in after)
