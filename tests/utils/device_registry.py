"""Strict registry binding an mTLS certificate CN to one exact Zenoh ingress key."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class DeviceRegistryError(ValueError):
    """Raised when the device registry is missing, malformed, or ambiguous."""


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    device_id: str
    cert_cn: str
    ingress_key: str
    target_topic: str
    enabled: bool = True
    payload_type: str = "generic"
    rate_pps: float = 8.0
    burst: float = 12.0
    strike_threshold: int = 3
    strike_window_sec: float = 30.0
    strike_cooldown_sec: float = 1.0
    block_ttl_sec: float = 45.0


class DeviceRegistry:
    def __init__(self, devices: list[DeviceSpec]):
        if not devices:
            raise DeviceRegistryError("registry must contain at least one device")
        by_id: dict[str, DeviceSpec] = {}
        by_ingress: dict[str, DeviceSpec] = {}
        by_cn: dict[str, DeviceSpec] = {}
        for spec in devices:
            if spec.device_id in by_id:
                raise DeviceRegistryError(f"duplicate device_id: {spec.device_id}")
            if spec.ingress_key in by_ingress:
                raise DeviceRegistryError(f"duplicate ingress_key: {spec.ingress_key}")
            if spec.cert_cn in by_cn:
                raise DeviceRegistryError(f"duplicate cert_cn: {spec.cert_cn}")
            by_id[spec.device_id] = spec
            by_ingress[spec.ingress_key] = spec
            by_cn[spec.cert_cn] = spec
        self._by_id = by_id
        self._by_ingress = by_ingress
        self._by_cn = by_cn

    @classmethod
    def load(cls, path: str | Path = "config/devices.json") -> "DeviceRegistry":
        file_path = Path(path).expanduser()
        try:
            doc = json.loads(file_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DeviceRegistryError(f"cannot read device registry {file_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DeviceRegistryError(f"invalid JSON in {file_path}: {exc}") from exc
        if not isinstance(doc, dict) or doc.get("version") != 1:
            raise DeviceRegistryError("device registry must be an object with version=1")
        defaults = doc.get("defaults", {})
        if not isinstance(defaults, dict):
            raise DeviceRegistryError("defaults must be an object")
        raw_devices = doc.get("devices")
        if not isinstance(raw_devices, list):
            raise DeviceRegistryError("devices must be an array")
        specs = []
        for index, raw in enumerate(raw_devices):
            if not isinstance(raw, dict):
                raise DeviceRegistryError(f"devices[{index}] must be an object")
            merged = {**defaults, **raw}
            specs.append(_parse_spec(merged, index))
        return cls(specs)

    def get(self, device_id: str) -> Optional[DeviceSpec]:
        return self._by_id.get(device_id)

    def get_by_cn(self, cert_cn: str) -> Optional[DeviceSpec]:
        return self._by_cn.get(cert_cn)

    def enabled_devices(self) -> tuple[DeviceSpec, ...]:
        return tuple(spec for spec in self._by_id.values() if spec.enabled)

    def resolve_ingress(self, key: str) -> Optional[tuple[DeviceSpec, str]]:
        """Resolve only exact, explicitly registered ingress expressions."""
        spec = self._by_ingress.get(key)
        if spec is None or not spec.enabled:
            return None
        return spec, spec.target_topic

    def __len__(self) -> int:
        return len(self._by_id)


def _finite_number(value, field: str, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeviceRegistryError(f"{field} must be numeric") from exc
    import math
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise DeviceRegistryError(f"{field} must be between {minimum} and {maximum}")
    return number


def _parse_spec(raw: dict, index: int) -> DeviceSpec:
    prefix = f"devices[{index}]"
    device_id = raw.get("device_id")
    cert_cn = raw.get("cert_cn")
    ingress_key = raw.get("ingress_key")
    target_topic = raw.get("target_topic")
    if not isinstance(device_id, str) or not _DEVICE_ID_RE.fullmatch(device_id):
        raise DeviceRegistryError(f"{prefix}.device_id is invalid")
    if cert_cn != device_id or not isinstance(cert_cn, str) or not _DEVICE_ID_RE.fullmatch(cert_cn):
        raise DeviceRegistryError(f"{prefix}.cert_cn must exactly match device_id")
    if not isinstance(ingress_key, str) or not ingress_key.startswith(f"ingest/devices/{device_id}/"):
        raise DeviceRegistryError(f"{prefix}.ingress_key must be namespaced under this device id")
    if not isinstance(target_topic, str) or not target_topic.startswith("home/sensors/"):
        raise DeviceRegistryError(f"{prefix}.target_topic must be under home/sensors/")
    ingress_suffix = ingress_key[len(f"ingest/devices/{device_id}/"):]
    target_suffix = target_topic[len("home/"):]
    if ingress_suffix != target_suffix:
        raise DeviceRegistryError(f"{prefix}.ingress_key sensor suffix must match target_topic")
    if any(ch in ingress_key + target_topic for ch in "*$?#") or not ingress_key.isprintable() or not target_topic.isprintable():
        raise DeviceRegistryError(f"{prefix} contains a wildcard or control character")
    if not isinstance(raw.get("enabled", True), bool):
        raise DeviceRegistryError(f"{prefix}.enabled must be boolean")
    payload_type = raw.get("payload_type", "generic")
    if not isinstance(payload_type, str) or payload_type not in {"door", "gas", "vibration", "pir", "generic"}:
        raise DeviceRegistryError(f"{prefix}.payload_type is unsupported")
    expected_suffix = {
        "door": "sensors/door/state",
        "gas": "sensors/gas/ppm",
        "vibration": "sensors/vibration/shock",
        "pir": "sensors/pir/motion",
    }.get(payload_type)
    if expected_suffix is not None and ingress_suffix != expected_suffix:
        raise DeviceRegistryError(f"{prefix}.payload_type does not match its sensor topic")
    rate = _finite_number(raw.get("rate_pps", 8), f"{prefix}.rate_pps", minimum=0.1, maximum=10000)
    burst = _finite_number(raw.get("burst", 12), f"{prefix}.burst", minimum=1, maximum=100000)
    threshold_raw = raw.get("strike_threshold", 3)
    if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, int) or not 1 <= threshold_raw <= 100:
        raise DeviceRegistryError(f"{prefix}.strike_threshold must be an integer in 1..100")
    window = _finite_number(raw.get("strike_window_sec", 30), f"{prefix}.strike_window_sec", minimum=1, maximum=3600)
    cooldown = _finite_number(raw.get("strike_cooldown_sec", 1), f"{prefix}.strike_cooldown_sec", minimum=0.1, maximum=60)
    ttl = _finite_number(raw.get("block_ttl_sec", 45), f"{prefix}.block_ttl_sec", minimum=1, maximum=3600)
    return DeviceSpec(
        device_id=device_id, cert_cn=cert_cn, ingress_key=ingress_key,
        target_topic=target_topic, enabled=raw.get("enabled", True),
        payload_type=payload_type, rate_pps=rate, burst=burst,
        strike_threshold=threshold_raw, strike_window_sec=window,
        strike_cooldown_sec=cooldown, block_ttl_sec=ttl,
    )
