"""Resolve real device certificate metadata from the active local PKI bundle."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.x509.oid import NameOID

from utils.cert_lifecycle import CertAuthority

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class ActiveDeviceIdentity:
    device_id: str
    generation: int
    common_name: str
    certificate_fingerprint: bytes
    serial_number_hex: str


def load_active_device_identity(pointer_path: str | Path, device_id: str) -> ActiveDeviceIdentity:
    """Derive CN, generation and SHA-256 from the active cert file, never from a device ID."""
    if not _SAFE_ID.fullmatch(device_id):
        raise ValueError("invalid stable device id")
    _pointer, bundle_dir, manifest = CertAuthority.inspect_active(pointer_path, verify_files=True)
    entry = manifest.get("identities", {}).get(device_id)
    if not isinstance(entry, dict):
        raise KeyError(f"active certificate bundle has no identity {device_id!r}")
    filename = entry.get("cert_file")
    if filename != f"{device_id}.pem":
        raise ValueError("unsafe active certificate filename")
    cert_path = (bundle_dir / filename).resolve(strict=True)
    if bundle_dir.resolve() not in cert_path.parents:
        raise ValueError("certificate path escapes active bundle")
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if len(names) != 1 or names[0].value != entry.get("common_name"):
        raise ValueError("active certificate CN does not match its signed manifest")
    if format(cert.serial_number, "x") != str(entry.get("serial", "")).lower().lstrip("0"):
        # Cert manifest stores the serial as hex; normalize leading zeroes.
        raise ValueError("active certificate serial does not match its manifest")
    generation = entry.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("active certificate generation is invalid")
    return ActiveDeviceIdentity(
        device_id=device_id,
        generation=generation,
        common_name=names[0].value,
        certificate_fingerprint=cert.fingerprint(hashes.SHA256()),
        serial_number_hex=format(cert.serial_number, "x"),
    )
