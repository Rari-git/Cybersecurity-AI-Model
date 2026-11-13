"""Compatibility CLI for Patch 04 session-scoped mTLS certificate provisioning.

Do not generate a persistent CA in config/certs. The session CA must live in the
KeyAuthority-owned ephemeral session directory and is wiped during cleanup.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils.cert_lifecycle import CertAuthority, default_identities


def generate_mtls_certificates(session_dir=None, **kwargs) -> CertAuthority:
    if not session_dir:
        raise ValueError(
            "Patch 04 no longer writes a persistent config/certs CA. "
            "Pass --session-dir or start the stack with run_all.py."
        )
    return CertAuthority.create_session(
        Path(session_dir), identities=default_identities(),
        leaf_ttl_days=int(os.environ.get("LUMEN_CERT_TTL_DAYS", "14")),
        ca_validity_days=int(os.environ.get("LUMEN_CA_VALIDITY_DAYS", "3650")),
        **kwargs,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default=os.environ.get("LUMEN_SESSION_DIR"))
    args = parser.parse_args(argv)
    if not args.session_dir:
        parser.error("--session-dir or LUMEN_SESSION_DIR is required")
    authority = generate_mtls_certificates(args.session_dir)
    print(f"mTLS session provisioned: {authority.pointer_path}")
    print(f"active identities: {', '.join(sorted(authority.status()))}")
    # Do not cleanup here: the session is managed by agent.key_authority/run_all.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
