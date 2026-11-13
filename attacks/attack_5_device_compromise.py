"""Bounded device-identity rate-abuse and mTLS quarantine test."""
from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.cert_lifecycle import CertAuthority, CertRevocationLedger
from utils.device_registry import DeviceRegistry
from utils.gateway_metrics import calculate_delta, read_latest_snapshot
from utils.mtls_hot_reload import open_mtls_client
from utils.logger import get_script_logger

logger = get_script_logger("attack_5_device_compromise")
METRICS_PATH = os.environ.get("LUMEN_GATEWAY_METRICS_PATH", "logs/gateway_metrics.jsonl")


def _payload(payload_type: str) -> bytes:
    return {
        "door": b"OPEN", "gas": b"350", "vibration": b"0.020",
        "pir": b"MOTION_DETECTED", "generic": b"OK",
    }[payload_type]


def _is_cert_quarantined(device_id: str) -> bool:
    pointer_path = os.environ.get("LUMEN_CERT_POINTER")
    ledger_path = os.environ.get("LUMEN_CERT_REVOCATIONS")
    if not pointer_path or not ledger_path:
        return False
    try:
        pointer, _, _ = CertAuthority.read_active_bundle(pointer_path)
        if device_id in set(pointer.get("quarantined", [])):
            return True
        ledger = CertRevocationLedger(ledger_path)
        return len(ledger.revoked_serials(device_id)) > 0
    except Exception:
        return False


def _metric_wait(device_id: str, peer_id: str, baseline: dict, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    latest = None
    block_key = f"device:{device_id}"
    while time.monotonic() < deadline:
        candidate = read_latest_snapshot(METRICS_PATH)
        if candidate and candidate.get("run_id") == baseline.get("run_id"):
            latest = candidate
            blocks = candidate.get("device_blocks", {})
            accepted = candidate.get("device_publish_accepted", {})
            own_blocked = (isinstance(blocks, dict) and block_key in blocks) or _is_cert_quarantined(device_id)
            peer_accepted = int(accepted.get(peer_id, 0) or 0) > int(
                baseline.get("device_publish_accepted", {}).get(peer_id, 0) or 0
            ) if peer_id else True
            if own_blocked and peer_accepted:
                return candidate
        time.sleep(0.25)
    return latest


def main() -> int:
    registry = DeviceRegistry.load(os.environ.get("LUMEN_DEVICE_REGISTRY", "config/devices.json"))
    device_id = os.environ.get("LUMEN_COMPROMISED_DEVICE", os.environ.get("LUMEN_DOS_DEVICE_ID", "gas_sensor_01")).strip()
    spec = registry.get(device_id)
    if spec is None or not spec.enabled:
        raise ValueError(f"device is not enabled in registry: {device_id}")
    try:
        rate = float(os.environ.get("LUMEN_DEVICE_ATTACK_RATE_PPS", "40"))
        duration = float(os.environ.get("LUMEN_DEVICE_ATTACK_DURATION_SEC", "8"))
    except ValueError as exc:
        raise ValueError("attack rate and duration must be numeric") from exc
    if not 1 <= rate <= 200 or not 1 <= duration <= 30:
        raise ValueError("attack rate must be 1..200 pps and duration 1..30 seconds")

    peer_id = os.environ.get("LUMEN_DOS_PEER_DEVICE", "gas_sensor_02").strip()
    if peer_id == device_id:
        peer_id = ""
    baseline = read_latest_snapshot(METRICS_PATH)
    if baseline is None:
        raise RuntimeError(f"no gateway metrics snapshot at {METRICS_PATH}; start Patch 03/04 gateway first")

    session = open_mtls_client(spec.cert_cn)
    sent = errors = 0
    start_at = time.monotonic()
    stop_at = start_at + duration
    period = 1.0 / rate
    next_send = start_at
    payload = _payload(spec.payload_type)
    logger.warning("[DEVICE COMPROMISE TEST] device=%s cert_cn=%s key=%s rate=%.1fpps duration=%.1fs",
                   device_id, spec.cert_cn, spec.ingress_key, rate, duration)
    try:
        while time.monotonic() < stop_at:
            now = time.monotonic()
            if next_send > now:
                time.sleep(min(next_send - now, 0.01))
                continue
            try:
                session.put(spec.ingress_key, payload)
                sent += 1
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    logger.warning("[DEVICE COMPROMISE TEST] put rejected/failed (expected after quarantine): %s", exc)
            next_send += period
            if next_send < time.monotonic() - period:
                next_send = time.monotonic() + period
    finally:
        session.close()

    logger.info("[DEVICE COMPROMISE TEST] put_returned=%d errors=%d achieved=%.1fpps",
                sent, errors, sent / max(0.001, time.monotonic() - start_at))
    latest = _metric_wait(device_id, peer_id, baseline)
    delta = calculate_delta(baseline, latest) if latest else None
    if delta is None:
        logger.error("[STATUS] INCONCLUSIVE: no same-process gateway metrics after the attack")
        return 1

    own_block_key = f"device:{device_id}"
    blocks = delta.get("device_blocks", {})
    cert_quarantined = _is_cert_quarantined(device_id)
    own_blocked = (isinstance(blocks, dict) and own_block_key in blocks) or cert_quarantined
    other_device_blocks = {
        key for key in blocks if str(key).startswith("device:") and key != own_block_key
    } if isinstance(blocks, dict) else set()
    peer_blocked = bool(peer_id and (f"device:{peer_id}" in blocks or _is_cert_quarantined(peer_id)))
    accepted = delta.get("device_publish_accepted", {})
    peer_accepted = int(accepted.get(peer_id, 0) or 0) if peer_id else 0
    logger.info("[DEVICE METRICS] run_id=%s ingress=%d rate_dropped=%d block_dropped=%d blocks_issued=%d cert_quarantined=%s",
                delta["run_id"], delta["ingress_received"], delta["device_rate_dropped"],
                delta["device_blocklist_dropped"], delta["device_blocks_issued"], cert_quarantined)
    if other_device_blocks:
        logger.error("[DEVICE COMPROMISE TEST] unexpected additional device blocks: %s",
                     sorted(other_device_blocks))
    if peer_id:
        logger.info("[SAME-TOPIC PEER] peer=%s gateway_put_accepted_delta=%d blocked=%s",
                    peer_id, peer_accepted, peer_blocked)

    ok = bool(
        delta["device_rate_dropped"] > 0
        and delta["device_blocks_issued"] > 0
        and own_blocked
        and not other_device_blocks
        and not peer_blocked
        and (not peer_id or peer_accepted > 0)
    )
    logger.info("[STATUS] %s", "PASS: only the compromised device was quarantined" if ok else
                "FAIL/INCONCLUSIVE: inspect ACL, device metrics, and healthy peer simulator")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("[DEVICE COMPROMISE TEST] setup/runtime failed")
        raise SystemExit(2)