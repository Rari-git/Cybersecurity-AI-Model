from __future__ import annotations

import json

from simulation.capture_regression_b4 import run_capture_suite
from storage.telemetry_db import DBPaths, LumenStorageManager
from training.adaptive_retrainer import evaluate_regression_evidence_preflight


def test_b4_suite_captures_independent_attack_protocol_and_fault_families(tmp_path):
    db_dir = tmp_path / "db"
    results = run_capture_suite(
        db_dir, seed=43, attack_steps=24, fault_steps=36
    )
    assert len(results) == 8
    assert len({item["scenario_id"] for item in results}) == 8

    counts = {
        category: sum(item["corpus_class"] == category for item in results)
        for category in ("adversarial", "protocol_violation", "hardware_fault")
    }
    assert counts == {
        "adversarial": 4,
        "protocol_violation": 2,
        "hardware_fault": 2,
    }

    storage = LumenStorageManager(paths=DBPaths.from_dir(db_dir), enable_writers=True)
    try:
        classes = ("adversarial", "protocol_violation", "hardware_fault")
        corpus = storage.load_adversarial_regression_corpus(classes)
        assert corpus
        families_by_class = {category: set() for category in classes}
        scenarios_by_class = {category: set() for category in classes}
        for row in corpus:
            category = str(row["corpus_class"])
            scenario_id = str(row["scenario_id"])
            manifest = json.loads(str(row["scenario_manifest_json"]))
            family = manifest.get("family")
            assert isinstance(family, str) and family
            families_by_class[category].add(family)
            scenarios_by_class[category].add(scenario_id)

        assert len(families_by_class["adversarial"]) == 4
        assert len(families_by_class["protocol_violation"]) == 2
        assert len(families_by_class["hardware_fault"]) == 2
        assert len(scenarios_by_class["adversarial"]) == 4
        assert len(scenarios_by_class["protocol_violation"]) == 2
        assert len(scenarios_by_class["hardware_fault"]) == 2
        assert {
            item["fault_type"] for item in results
            if item["corpus_class"] == "hardware_fault"
        } == {"FROZEN_SENSOR", "PACKET_DROP_DEGRADATION"}

        preflight = evaluate_regression_evidence_preflight(storage)
        assert preflight["ready"] is True
        assert preflight["deficits"] == {}
        assert preflight["family_deficits"] == {}
        assert preflight["family_counts"] == {
            "adversarial": 4,
            "protocol_violation": 2,
            "hardware_fault": 2,
        }
    finally:
        storage.close()


def test_promotion_evaluation_prefers_captured_faults_without_double_counting_generated_fallback(tmp_path, monkeypatch):
    import training.promotion_gate as gate

    captured_fault = gate.EvaluationGroup(
        "fault:captured", "fault", "gas_sensor_01", "home/sensors/gas/ppm", tuple()
    )
    captured_attack = gate.EvaluationGroup(
        "attack:captured", "attack", "gas_sensor_01", "home/sensors/gas/ppm", tuple()
    )
    clean_group = gate.EvaluationGroup(
        "clean:heldout", "clean", "gas_sensor_01", "home/sensors/gas/ppm", tuple()
    )
    monkeypatch.setattr(gate, "_load_clean_groups", lambda *_a, **_k: [clean_group])
    monkeypatch.setattr(gate, "_load_incident_groups", lambda *_a, **_k: ([captured_attack], [], [captured_fault]))
    monkeypatch.setattr(
        gate, "_load_fault_scenarios",
        lambda: (_ for _ in ()).throw(AssertionError("generated fault fallback duplicated captured corpus")),
    )
    observed_group_counts = []
    monkeypatch.setattr(
        gate, "_score_bundle",
        lambda _bundle, _key, groups, **_kwargs: observed_group_counts.append(len(groups)) or {"ok": True},
    )

    result = gate.evaluate_promotion_metrics(
        object(), tmp_path / "candidate", tmp_path / "active", tmp_path / "public.pem", ["heldout"]
    )
    assert result["group_count"] == 3
    assert observed_group_counts == [3, 3]
