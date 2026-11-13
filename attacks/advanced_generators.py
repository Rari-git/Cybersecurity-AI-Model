"""Etapa E3: Generatoare deterministe (parametri înghețați) pentru:
- 4 scenarii avansate de atac (Stealth Slow-Ramp, Authenticated Replay + Timing Skew,
  Low-and-Slow Reconnaissance, Adaptive Rate-Burst + Entropy Drift)
- 2 scenarii de defecțiune tehnică (Frozen Sensor, Packet Drop 40%)
- Separare strictă între setul de dezvoltare/calibrare și setul held-out de evaluare.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str  # 'dev' | 'held_out'
    category: str  # 'attack' | 'fault'
    attack_family: str
    target_device: str
    target_topic: str
    seed: int
    params: dict


@dataclass(frozen=True)
class GeneratedFrame:
    step: int
    device_id: str
    topic: str
    operation: str  # 'put' | 'query' | 'declare_subscriber'
    delta_t_sec: float
    payload: bytes
    ground_truth_label: str  # 'attack' | 'fault' | 'normal'
    scenario_id: str
    dropped: bool = False


# Parametri înghețați înainte de E4/E5 pentru a preveni contaminarea evaluării
FROZEN_SCENARIOS: dict[str, ScenarioSpec] = {
    # ---- SET DE DEZVOLTARE / CALIBRARE (DEV) ----
    "atk1_stealth_spoof_dev": ScenarioSpec(
        scenario_id="atk1_stealth_spoof_dev",
        split="dev",
        category="attack",
        attack_family="stealth_value_spoofing",
        target_device="gas_sensor_01",
        target_topic="home/sensors/gas/ppm",
        seed=101,
        params={"base_ppm": 350.0, "ramp_per_step": 2.8, "noise_std": 1.15},
    ),
    "atk2_replay_skew_dev": ScenarioSpec(
        scenario_id="atk2_replay_skew_dev",
        split="dev",
        category="attack",
        attack_family="authenticated_replay_timing_skew",
        target_device="gas_sensor_01",
        target_topic="home/sensors/gas/ppm",
        seed=102,
        params={"replayed_payload": "362", "skew_period": 5, "cross_topic": "home/sensors/door/state"},
    ),
    "atk3_low_slow_recon_dev": ScenarioSpec(
        scenario_id="atk3_low_slow_recon_dev",
        split="dev",
        category="attack",
        attack_family="low_and_slow_reconnaissance",
        target_device="vibration_sensor_01",
        target_topic="home/admin/config",
        seed=103,
        params={"probe_interval_sec": 3.5, "probe_targets": ["home/admin/config", "home/security/keys", "system/internal/logs"]},
    ),
    "atk4_adaptive_burst_dev": ScenarioSpec(
        scenario_id="atk4_adaptive_burst_dev",
        split="dev",
        category="attack",
        attack_family="adaptive_burst_entropy_drift",
        target_device="gas_sensor_02",
        target_topic="home/sensors/gas/ppm",
        seed=104,
        params={"burst_len": 4, "burst_every": 12, "drift_offset": 190.0},
    ),

    # ---- SET HELD-OUT (NEATINS PÂNĂ LA EVALUAREA FINALĂ DIN E7) ----
    "atk1_stealth_spoof_heldout": ScenarioSpec(
        scenario_id="atk1_stealth_spoof_heldout",
        split="held_out",
        category="attack",
        attack_family="stealth_value_spoofing",
        target_device="gas_sensor_02",
        target_topic="home/sensors/gas/ppm",
        seed=201,
        params={"base_ppm": 345.0, "ramp_per_step": 2.2, "noise_std": 0.95},
    ),
    "atk2_replay_skew_heldout": ScenarioSpec(
        scenario_id="atk2_replay_skew_heldout",
        split="held_out",
        category="attack",
        attack_family="authenticated_replay_timing_skew",
        target_device="vibration_sensor_01",
        target_topic="home/sensors/vibration/shock",
        seed=202,
        params={"replayed_payload": "0.018", "skew_period": 4, "cross_topic": "home/sensors/gas/ppm"},
    ),
    "atk3_low_slow_recon_heldout": ScenarioSpec(
        scenario_id="atk3_low_slow_recon_heldout",
        split="held_out",
        category="attack",
        attack_family="low_and_slow_reconnaissance",
        target_device="pir_sensor_01",
        target_topic="home/security/keys",
        seed=203,
        params={"probe_interval_sec": 4.2, "probe_targets": ["home/security/keys", "shadow/sandbox/gas_sensor_01/data", "home/admin/config"]},
    ),
    "atk4_adaptive_burst_heldout": ScenarioSpec(
        scenario_id="atk4_adaptive_burst_heldout",
        split="held_out",
        category="attack",
        attack_family="adaptive_burst_entropy_drift",
        target_device="gas_sensor_01",
        target_topic="home/sensors/gas/ppm",
        seed=204,
        params={"burst_len": 5, "burst_every": 15, "drift_offset": 240.0},
    ),

    # ---- DEFECȚIUNI TEHNICE (TEST NEGATIV: TREBUIE CLASIFICATE CA FAULT, NU CARANTINĂ) ----
    "fault1_frozen_sensor": ScenarioSpec(
        scenario_id="fault1_frozen_sensor",
        split="held_out",
        category="fault",
        attack_family="operational_frozen_sensor",
        target_device="gas_sensor_01",
        target_topic="home/sensors/gas/ppm",
        seed=301,
        params={"stuck_value": "354"},
    ),
    "fault2_packet_drop_40": ScenarioSpec(
        scenario_id="fault2_packet_drop_40",
        split="held_out",
        category="fault",
        attack_family="operational_packet_drop",
        target_device="vibration_sensor_01",
        target_topic="home/sensors/vibration/shock",
        seed=302,
        params={"drop_probability": 0.40},
    ),
}


def iter_scenario_frames(scenario_id: str, steps: int = 120) -> Iterator[GeneratedFrame]:
    """Generează secvența deterministă de cadre pentru un scenariu înghețat."""
    if scenario_id not in FROZEN_SCENARIOS:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")
    spec = FROZEN_SCENARIOS[scenario_id]
    rng = random.Random(spec.seed)

    for step in range(steps):
        if spec.attack_family == "stealth_value_spoofing":
            # Creștere lentă + zgomot gaussian continuu (fără cuantizare ADC întreagă -> detectabil în E4)
            val = spec.params["base_ppm"] + step * spec.params["ramp_per_step"] + rng.gauss(0.0, spec.params["noise_std"])
            val = min(1950.0, max(0.0, val))
            dt = max(0.85, min(1.15, 1.0 + rng.gauss(0.0, 0.03)))
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=spec.target_topic,
                operation="put",
                delta_t_sec=round(dt, 4),
                payload=f"{val:.4f}".encode("utf-8"),
                ground_truth_label="attack",
                scenario_id=spec.scenario_id,
            )

        elif spec.attack_family == "authenticated_replay_timing_skew":
            # Retransmitere periodică cu deviație de timp + încercare de cross-topic injection
            is_cross = (step % spec.params["skew_period"] == 0)
            topic = spec.params["cross_topic"] if is_cross else spec.target_topic
            dt = 0.12 if is_cross else 1.0 + 0.04 * math.sin(step)
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=topic,
                operation="put",
                delta_t_sec=round(dt, 4),
                payload=str(spec.params["replayed_payload"]).encode("utf-8"),
                ground_truth_label="attack",
                scenario_id=spec.scenario_id,
            )

        elif spec.attack_family == "low_and_slow_reconnaissance":
            # Sondare lentă a topicurilor interne la intervale mari (sub pragul de rate-limit)
            targets = spec.params["probe_targets"]
            probe_topic = targets[step % len(targets)]
            dt = spec.params["probe_interval_sec"] + rng.uniform(-0.4, 0.6)
            op = "query" if step % 2 == 0 else "declare_subscriber"
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=probe_topic,
                operation=op,
                delta_t_sec=round(dt, 4),
                payload=json.dumps({"probe_seq": step, "node": spec.target_device}).encode("utf-8"),
                ground_truth_label="attack",
                scenario_id=spec.scenario_id,
            )

        elif spec.attack_family == "adaptive_burst_entropy_drift":
            # Rafală scurtă sub Token Bucket + schimbare de structură/entropie a payload-ului
            in_burst = (step % spec.params["burst_every"]) < spec.params["burst_len"]
            if in_burst:
                dt = 0.18
                # Payload cu entropie modificată (padding hex aleatoriu atașat)
                token = f"{rng.getrandbits(32):08x}"
                val = int(round(350.0 + spec.params["drift_offset"] + rng.gauss(0.0, 8.0)))
                payload = f"{val}|{token}".encode("utf-8")
            else:
                dt = 1.0 + rng.gauss(0.0, 0.03)
                val = int(round(352.0 + rng.gauss(0.0, 5.0)))
                payload = str(val).encode("utf-8")
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=spec.target_topic,
                operation="put",
                delta_t_sec=round(max(0.05, dt), 4),
                payload=payload,
                ground_truth_label="attack",
                scenario_id=spec.scenario_id,
            )

        elif spec.attack_family == "operational_frozen_sensor":
            # Defecțiune tehnică 1: senzor înghețat (varianță 0 pe valoare, ritm normal)
            dt = max(0.90, min(1.10, 1.0 + rng.gauss(0.0, 0.02)))
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=spec.target_topic,
                operation="put",
                delta_t_sec=round(dt, 4),
                payload=str(spec.params["stuck_value"]).encode("utf-8"),
                ground_truth_label="fault",
                scenario_id=spec.scenario_id,
            )

        elif spec.attack_family == "operational_packet_drop":
            # Defecțiune tehnică 2: pierdere 40% din pachete, dar valorile transmise sunt normale
            dropped = rng.random() < spec.params["drop_probability"]
            val = round(max(0.0, 0.015 + abs(rng.gauss(0.0, 0.007))), 3)
            yield GeneratedFrame(
                step=step,
                device_id=spec.target_device,
                topic=spec.target_topic,
                operation="put",
                delta_t_sec=1.0,
                payload=f"{val:.3f}".encode("utf-8"),
                ground_truth_label="fault",
                scenario_id=spec.scenario_id,
                dropped=dropped,
            )