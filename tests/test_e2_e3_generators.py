"""Teste automate pentru Etapa E2 (Baseline Capture) și Etapa E3 (Atacuri & Defecte înghețate)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.capture_baseline import run_baseline_capture
from storage.telemetry_db import LumenStorageManager, DBPaths
from attacks.advanced_generators import FROZEN_SCENARIOS, iter_scenario_frames


def test_e2_baseline_capture_populates_approved_sessions():
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e2-"))
    try:
        stats = run_baseline_capture(db_dir=tmp_dir, sessions=2, events_per_sensor=20, seed=42)
        assert stats["sessions"] == 2
        assert stats["admitted"] == 2 * 20 * 5
        assert stats["rejected"] == 0

        mgr = LumenStorageManager(paths=DBPaths.from_dir(tmp_dir), enable_writers=False)
        ro = mgr.connect_readonly("normal")
        try:
            approved = ro.execute("SELECT COUNT(*) FROM baseline_sessions WHERE approved_for_training = 1").fetchone()[0]
            raw_cnt = ro.execute("SELECT COUNT(*) FROM raw_events WHERE source_type = 'baseline_lab'").fetchone()[0]
            assert approved == 2
            assert raw_cnt == 200
        finally:
            ro.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_e3_frozen_scenarios_are_deterministic_and_cover_4_attacks_plus_2_faults():
    dev_attacks = [s for s in FROZEN_SCENARIOS.values() if s.split == "dev" and s.category == "attack"]
    heldout_attacks = [s for s in FROZEN_SCENARIOS.values() if s.split == "held_out" and s.category == "attack"]
    faults = [s for s in FROZEN_SCENARIOS.values() if s.category == "fault"]

    assert len(dev_attacks) == 4
    assert len(heldout_attacks) == 4
    assert len(faults) == 2

    # Verificăm determinsimul (același seed produce exact aceleași cadre)
    seq_a = list(iter_scenario_frames("atk1_stealth_spoof_dev", steps=30))
    seq_b = list(iter_scenario_frames("atk1_stealth_spoof_dev", steps=30))
    assert seq_a == seq_b
    assert float(seq_a[-1].payload) > float(seq_a[0].payload)

    # Verificăm defectul 1 (senzor înghețat -> varianță 0)
    frozen = list(iter_scenario_frames("fault1_frozen_sensor", steps=25))
    assert len({f.payload for f in frozen}) == 1