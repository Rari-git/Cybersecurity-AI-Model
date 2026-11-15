"""Central Zenoh connection policy for Lumen Edge Security.

TLS clients resolve credentials exclusively through Patch 04's active.json
pointer and revoked_certs.json ledger.  They use HotReloadZenohClient so a
certificate rotation replaces the Zenoh session without stopping the process.

Plain TCP remains available only for explicitly opted-in loopback development.
There is deliberately no fallback from mTLS to static config/certs or TCP.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ENDPOINT = "tls/127.0.0.1:7447"


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _split_endpoint(endpoint: str) -> tuple[str, str]:
    """Return (scheme, host) for Zenoh's ``tls/host:port`` / ``tcp/host:port``."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("LUMEN_ZENOH_ENDPOINT must be a non-empty string")
    endpoint = endpoint.strip()
    if endpoint.startswith("tls/"):
        scheme, authority = "tls", endpoint[4:]
    elif endpoint.startswith("tcp/"):
        scheme, authority = "tcp", endpoint[4:]
    else:
        raise ValueError("Zenoh endpoint must start with 'tls/' or 'tcp/'")

    if not authority or "/" in authority or "@" in authority:
        raise ValueError(f"Invalid Zenoh endpoint authority: {authority!r}")
    if authority.startswith("["):
        close = authority.find("]")
        if close < 0 or close + 1 >= len(authority) or authority[close + 1] != ":":
            raise ValueError(f"Invalid bracketed IPv6 endpoint: {authority!r}")
        host, port_text = authority[1:close], authority[close + 2:]
    else:
        host, separator, port_text = authority.rpartition(":")
        if not separator or not host:
            raise ValueError(f"Endpoint must include a host and port: {authority!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid Zenoh endpoint port: {port_text!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Zenoh endpoint port out of range: {port}")
    if not host or any(ch.isspace() for ch in host):
        raise ValueError("Zenoh endpoint host is invalid")
    return scheme, host


def validate_endpoint_policy(endpoint: str, environment: str,
                              allow_insecure_local: bool) -> str:
    """Production requires TLS; dev TCP requires explicit opt-in and loopback."""
    env = str(environment).strip().lower()
    if env not in {"development", "production"}:
        raise ValueError("LUMEN_ENV must be either 'development' or 'production'")
    scheme, host = _split_endpoint(endpoint)
    if scheme == "tls":
        return scheme
    if env != "development":
        raise ValueError("Plain TCP is forbidden unless LUMEN_ENV=development")
    if not allow_insecure_local:
        raise ValueError("Plain TCP requires LUMEN_ALLOW_INSECURE_LOCAL=1 in development mode")
    if not _is_loopback_host(host):
        raise ValueError("Insecure development TCP is allowed only to localhost/loopback")
    return scheme


def _identity_for_call(identity: str) -> str:
    configured = os.environ.get("LUMEN_CERT_IDENTITY", "").strip()
    requested = str(identity or configured).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", requested):
        raise ValueError("TLS identity must contain only letters, digits, '.', '_' or '-' (1-64 chars)")
    allowed_raw = os.environ.get("LUMEN_CERT_ALLOWED_IDENTITIES", "").strip()
    if allowed_raw:
        allowed = {item.strip() for item in allowed_raw.split(",") if item.strip()}
        if configured:
            allowed.add(configured)
        if requested not in allowed:
            raise ValueError(
                f"requested TLS identity {requested!r} is not in LUMEN_CERT_ALLOWED_IDENTITIES"
            )
    elif configured and configured != requested:
        raise ValueError(
            f"requested TLS identity {requested!r} differs from LUMEN_CERT_IDENTITY={configured!r}"
        )
    return requested


def _active_client_material(identity: str) -> tuple[Path, Path, Path]:
    """Resolve and validate the current bundle; never consult legacy cert_dir."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    from utils.cert_lifecycle import (
        CertAuthority, CertRevocationLedger, CertStoreError,
        _cert_times, _spki_der, _verify_certificate_signature,
    )

    pointer_value = os.environ.get("LUMEN_CERT_POINTER", "").strip()
    ledger_value = os.environ.get("LUMEN_CERT_REVOCATIONS", "").strip()
    if not pointer_value or not ledger_value:
        raise CertStoreError(
            "Patch 04 environment is missing: LUMEN_CERT_POINTER and "
            "LUMEN_CERT_REVOCATIONS must be supplied by run_all.py"
        )

    pointer_path, ledger_path = Path(pointer_value), Path(ledger_value)
    pointer, bundle_dir, manifest = CertAuthority.inspect_active(pointer_path)
    if identity in set(pointer.get("quarantined", [])):
        raise CertStoreError(f"mTLS identity {identity!r} is quarantined")
    info = manifest.get("identities", {}).get(identity)
    if not isinstance(info, dict):
        raise CertStoreError(f"identity {identity!r} is absent from active certificate bundle")
    if CertRevocationLedger(ledger_path).is_revoked(identity, info["serial"]):
        raise CertStoreError(f"active certificate for {identity!r} is revoked")

    cert_path = bundle_dir / info["cert_file"]
    key_path = bundle_dir / info["key_file"]
    ca_path = bundle_dir / "ca.pem"
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        if _spki_der(cert.public_key()) != _spki_der(key.public_key()):
            raise CertStoreError("client certificate/private-key mismatch")
        _verify_certificate_signature(cert, ca)
        not_before, not_after = _cert_times(cert)
        now = datetime.now(timezone.utc)
        if not not_before <= now <= not_after:
            raise CertStoreError("client certificate is outside its validity window")
        cns = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if len(cns) != 1 or cns[0].value != info.get("common_name"):
            raise CertStoreError("client certificate CN differs from bundle manifest")
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        if ExtendedKeyUsageOID.CLIENT_AUTH not in eku:
            raise CertStoreError("client certificate lacks clientAuth EKU")
        if cert.serial_number != int(info["serial"], 16):
            raise CertStoreError("client certificate serial differs from bundle manifest")
    except CertStoreError:
        raise
    except Exception as exc:
        raise CertStoreError(f"active mTLS material is invalid for {identity!r}: {exc}") from exc

    return ca_path.resolve(), cert_path.resolve(), key_path.resolve()


def build_client_config(identity: str = ""):
    """Build a snapshot Zenoh config for diagnostics/one-shot callers.

    Application processes should call ``open_client_session`` instead: for TLS
    that returns a hot-reload proxy rather than a one-shot Zenoh session.
    """
    import zenoh  # Keep policy helpers importable without the Zenoh runtime.

    environment = os.environ.get("LUMEN_ENV", "production")
    allow_insecure = os.environ.get("LUMEN_ALLOW_INSECURE_LOCAL", "0") == "1"
    endpoint = os.environ.get("LUMEN_ZENOH_ENDPOINT", DEFAULT_ENDPOINT)
    scheme = validate_endpoint_policy(endpoint, environment, allow_insecure)

    cfg = zenoh.Config()
    cfg.insert_json5("mode", json.dumps("client"))
    cfg.insert_json5("connect/endpoints", json.dumps([endpoint]))
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "false")

    if scheme == "tls":
        client_identity = _identity_for_call(identity)
        ca_path, cert_path, key_path = _active_client_material(client_identity)
        for name, value in (
            ("root_ca_certificate", ca_path),
            ("connect_certificate", cert_path),
            ("connect_private_key", key_path),
        ):
            cfg.insert_json5(f"transport/link/tls/{name}", json.dumps(str(value)))
        cfg.insert_json5("transport/link/tls/enable_mtls", "true")
        cfg.insert_json5("transport/link/tls/verify_name_on_connect", "true")
        cfg.insert_json5("transport/link/tls/close_link_on_expiration", "true")
    else:
        print("[LUMEN SECURITY WARNING] Zenoh is using plain TCP on loopback (development only).")
    return cfg


def open_client_session(identity: str = ""):
    """Open a Zenoh client; TLS uses Patch 04's live pointer/ledger watcher."""
    environment = os.environ.get("LUMEN_ENV", "production")
    allow_insecure = os.environ.get("LUMEN_ALLOW_INSECURE_LOCAL", "0") == "1"
    endpoint = os.environ.get("LUMEN_ZENOH_ENDPOINT", DEFAULT_ENDPOINT)
    scheme = validate_endpoint_policy(endpoint, environment, allow_insecure)

    if scheme == "tls":
        from utils.mtls_hot_reload import open_mtls_client
        return open_mtls_client(_identity_for_call(identity), endpoint=endpoint)

    import zenoh
    return zenoh.open(build_client_config(identity))