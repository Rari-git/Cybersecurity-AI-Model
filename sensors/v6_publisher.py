"""V6 virtual/edge sensor publisher with authenticated boot registration."""
from __future__ import annotations

import secrets
import time
from typing import Any

from utils.boot_registration import perform_boot_registration, boot_topic
from utils.sensor_payload import encode_sensor_frame


class V6SensorPublisher:
    """Own one sensor's authenticated boot epoch and monotonic local sequence."""

    def __init__(
        self,
        session: Any,
        device_id: str,
        ingress_key: str,
        target_topic: str,
        cert_pointer: str,
        *,
        revocation_ledger: str | None = None,
        timeout_sec: float = 10.0,
        retry_interval_sec: float = 0.5,
        boot_id: str | None = None,
        registration_fn=perform_boot_registration,
    ):
        self.session = session
        self.device_id = device_id
        self.ingress_key = ingress_key
        self.target_topic = target_topic
        self.boot_id = boot_id or secrets.token_hex(16)
        self.sequence = 0
        self.boot_request_topic = boot_topic("request", device_id)
        self.boot_challenge_topic = boot_topic("challenge", device_id)
        self.boot_proof_topic = boot_topic("proof", device_id)
        self.boot_result_topic = boot_topic("result", device_id)
        self.registration = registration_fn(
            session,
            device_id=device_id,
            cert_pointer=cert_pointer,
            revocation_ledger=revocation_ledger,
            timeout_sec=timeout_sec,
            retry_interval_sec=retry_interval_sec,
            boot_id=self.boot_id,
        )
        if (self.registration.get("device_id") != device_id
                or self.registration.get("boot_id") != self.boot_id
                or self.registration.get("sequence") != 0):
            raise RuntimeError("Gateway returned an invalid V6 boot registration")

    def build_frame(self, payload: bytes | str | int | float, *, device_timestamp_ns: int | None = None) -> bytes:
        next_sequence = self.sequence + 1
        frame = encode_sensor_frame(
            device_id=self.device_id,
            boot_id=self.boot_id,
            sequence=next_sequence,
            device_timestamp_ns=device_timestamp_ns or time.time_ns(),
            topic=self.target_topic,
            payload=payload,
        )
        self.sequence = next_sequence
        return frame

    def publish(self, payload: bytes | str | int | float) -> int:
        frame = self.build_frame(payload)
        self.session.put(self.ingress_key, frame)
        return self.sequence
