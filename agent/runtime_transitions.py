"""Bounded, descriptive runtime transition observations for the IDS."""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _payload_bucket(size: int) -> str:
    if size == 0:
        return "empty"
    if size <= 16:
        return "tiny"
    if size <= 64:
        return "small"
    if size <= 256:
        return "medium"
    if size <= 1024:
        return "large"
    return "very_large"


def _interval_class(seconds: float, rapid: float, normal: float) -> str:
    if seconds == 0:
        return "simultaneous"
    if seconds <= rapid:
        return "rapid"
    if seconds <= normal:
        return "normal"
    return "delayed"


@dataclass(frozen=True)
class RuntimeTransitionObservation:
    state_id: str
    previous_state_id: str | None
    transition_id: str | None
    interval_class: str
    state_changed: bool
    changed_dimensions: tuple[str, ...]

    @property
    def explanation(self) -> str:
        if self.previous_state_id is None:
            return f"transition=baseline_state:{self.state_id}"
        if not self.state_changed:
            return f"transition=self:{self.transition_id} interval={self.interval_class}"
        dimensions = ",".join(self.changed_dimensions)
        return f"transition=state_change:{self.transition_id} dimensions={dimensions}"


class RuntimeTransitionObserver:
    """Track recent device/topic states without assigning anomaly or policy scores.

    The bounded stream cache prevents untrusted topic diversity from growing
    process memory indefinitely. A generation change creates an independent
    stream and cannot inherit the prior generation's state.
    """

    def __init__(
        self,
        *,
        rapid_interval_seconds: float = 0.1,
        normal_interval_seconds: float = 1.0,
        max_streams: int = 4096,
    ) -> None:
        if (isinstance(rapid_interval_seconds, bool)
                or not isinstance(rapid_interval_seconds, (int, float))
                or not math.isfinite(float(rapid_interval_seconds))
                or rapid_interval_seconds <= 0):
            raise ValueError("rapid interval must be finite and positive")
        if (isinstance(normal_interval_seconds, bool)
                or not isinstance(normal_interval_seconds, (int, float))
                or not math.isfinite(float(normal_interval_seconds))
                or normal_interval_seconds <= rapid_interval_seconds):
            raise ValueError("normal interval must be finite and greater than rapid interval")
        if isinstance(max_streams, bool) or not isinstance(max_streams, int) or max_streams < 1:
            raise ValueError("max_streams must be a positive integer")
        self.rapid_interval_seconds = float(rapid_interval_seconds)
        self.normal_interval_seconds = float(normal_interval_seconds)
        self.max_streams = max_streams
        self._last_descriptors: OrderedDict[tuple[str, int, str], dict[str, Any]] = OrderedDict()

    def observe(
        self,
        *,
        device_id: str,
        generation: int,
        topic: str,
        payload_len: int,
        delta_t_seconds: float,
    ) -> RuntimeTransitionObservation:
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")
        if isinstance(payload_len, bool) or not isinstance(payload_len, int) or payload_len < 0:
            raise ValueError("payload_len must be a non-negative integer")
        if (isinstance(delta_t_seconds, bool)
                or not isinstance(delta_t_seconds, (int, float))
                or not math.isfinite(float(delta_t_seconds))
                or delta_t_seconds < 0):
            raise ValueError("delta_t_seconds must be finite and non-negative")
        stream = (device_id, generation, topic)
        interval_class = _interval_class(
            float(delta_t_seconds),
            self.rapid_interval_seconds,
            self.normal_interval_seconds,
        )
        descriptor = {
            "device_id": device_id,
            "generation": generation,
            "topic": topic,
            "payload_size_bucket": _payload_bucket(payload_len),
        }
        state_id = "state_" + hashlib.sha256(_canonical(descriptor)).hexdigest()
        previous = self._last_descriptors.pop(stream, None)
        self._last_descriptors[stream] = descriptor
        if len(self._last_descriptors) > self.max_streams:
            self._last_descriptors.popitem(last=False)

        if previous is None:
            return RuntimeTransitionObservation(
                state_id, None, None, interval_class, False, (),
            )
        previous_id = "state_" + hashlib.sha256(_canonical(previous)).hexdigest()
        transition_material = {
            "from_state_id": previous_id,
            "to_state_id": state_id,
            "inter_arrival_class": interval_class,
        }
        transition_id = "transition_" + hashlib.sha256(_canonical(transition_material)).hexdigest()
        changed = tuple(key for key in descriptor if previous[key] != descriptor[key])
        return RuntimeTransitionObservation(
            state_id,
            previous_id,
            transition_id,
            interval_class,
            previous_id != state_id,
            changed,
        )
