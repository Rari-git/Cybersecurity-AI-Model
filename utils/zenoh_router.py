"""Generate a generation-bound Zenoh router config from the active mTLS bundle."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from utils.cert_lifecycle import CertAuthority, CertRevocationLedger, CertStoreError
from utils.key_lifecycle import atomic_write_bytes


def _replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    text2, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise CertStoreError(f"router config template must contain exactly one {label} field")
    return text2


def build_router_config(template_path, pointer_path, ledger_path, output_path) -> tuple[str, Path]:
    """Render TLS paths and per-role cert CNs; return config digest and bundle dir.

    The ACL subject IDs and policy references stay stable (``ids``, ``gateway``),
    but the cert_common_names set becomes generation-specific. A revoked current
    serial is mapped to a non-issued sentinel CN, never to an empty/wildcard
    subject, because Zenoh interprets omitted subject match lists as wildcards.
    """
    pointer, bundle_dir, manifest = CertAuthority.inspect_active(pointer_path)
    ledger = CertRevocationLedger(ledger_path)
    revoked = ledger.revoked_serials()  # validates/fails closed before rendering
    identities = manifest.get("identities", {})
    quarantine = set(pointer.get("quarantined", []))

    router_info = identities.get("router")
    if not isinstance(router_info, dict) or ("router", router_info.get("serial")) in revoked:
        raise CertStoreError("router certificate is revoked or missing")
    ca_path = (bundle_dir / "ca.pem").resolve()
    router_cert = (bundle_dir / router_info["cert_file"]).resolve()
    router_key = (bundle_dir / router_info["key_file"]).resolve()
    for path in (ca_path, router_cert, router_key):
        if not path.is_file():
            raise CertStoreError(f"router TLS file missing: {path}")

    text = Path(template_path).read_text(encoding="utf-8")
    # JSON5 path strings are valid JSON string literals after json.dumps.
    replacements = {
        "root_ca_certificate": ca_path.as_posix(),
        "listen_private_key": router_key.as_posix(),
        "listen_certificate": router_cert.as_posix(),
    }
    for field, value in replacements.items():
        pattern = rf"(\b{field}\s*:\s*)('[^']*'|\"[^\"]*\")"
        text = _replace_one(text, pattern, lambda m, v=json.dumps(value): m.group(1) + v,
                            field)

    # Force expiration-driven link closure on TLS mTLS listeners (keep valid JSON5 comma).
    if re.search(r"\bclose_link_on_expiration\s*:", text):
        text = re.sub(r"(\bclose_link_on_expiration\s*:\s*)(true|false)",
                      r"\1true", text, count=1)
    else:
        text = _replace_one(
            text,
            r"(\benable_mtls\s*:\s*true)\s*,?",
            r"\1,\n        close_link_on_expiration: true,",
            "enable_mtls",
        )

    for identity, info in identities.items():
        if identity == "router":
            continue
        if identity in quarantine or (identity, info.get("serial")) in revoked:
            allowed = [f"__lumen_denied__{identity}"]
        else:
            allowed = [str(info["common_name"])]
        pattern = (r"(\{\s*id\s*:\s*[\"']?" + re.escape(identity) +
                   r"[\"']?\s*,\s*cert_common_names\s*:\s*)\[[^\]]*\](\s*\})")
        text = _replace_one(text, pattern,
                            lambda m, values=json.dumps(allowed): m.group(1) + values + m.group(2),
                            f"ACL subject {identity}")

    # Make publication atomic; router reads the fully rendered version at exec.
    data = text.encode("utf-8")
    atomic_write_bytes(Path(output_path), data)
    digest = hashlib.sha256(data).hexdigest()
    state_digest = hashlib.sha256((
        pointer["bundle_id"] + "|" + str(pointer.get("revision", 0)) + "|" +
        json.dumps(sorted(revoked)) + "|" + digest
    ).encode("utf-8")).hexdigest()
    return state_digest, bundle_dir