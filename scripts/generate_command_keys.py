"""Generate an Ed25519 command signer keypair and update the verifier trust store."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signer-id", default="controller")
    parser.add_argument("--key-id", default="controller-2026-01")
    parser.add_argument("--private-out", default="config/command_keys/controller.private.pem")
    parser.add_argument("--public-out", default="config/command_keys/controller.public.pem")
    parser.add_argument("--trust-store", default="config/trusted_command_signers.json")
    parser.add_argument("--topics", nargs="+", required=True,
                        help="Exact canonical topics, e.g. home/actuators/door/lock")
    parser.add_argument("--force", action="store_true", help="Allow overwriting output key files")
    args = parser.parse_args()

    if not SAFE_ID.fullmatch(args.signer_id) or not SAFE_ID.fullmatch(args.key_id):
        parser.error("signer-id and key-id may contain only letters, numbers, dot, underscore and dash")
    topics = sorted(set(args.topics))
    if not topics or any(not t.startswith("home/actuators/") or any(c in t for c in "*$?#") for t in topics):
        parser.error("topics must be exact home/actuators/... topic names; wildcards are not accepted")

    private_path = Path(args.private_out).expanduser().resolve()
    public_path = Path(args.public_out).expanduser().resolve()
    trust_path = Path(args.trust_store).expanduser().resolve()
    for path in (private_path, public_path):
        if path.exists() and not args.force:
            parser.error(f"{path} already exists; choose new paths or use --force")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    if os.name == "posix":
        os.chmod(private_path, stat.S_IRUSR | stat.S_IWUSR)
        os.chmod(public_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    if trust_path.exists():
        try:
            doc = json.loads(trust_path.read_text(encoding="utf-8"))
        except Exception as exc:
            parser.error(f"cannot read existing trust store: {exc}")
        if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("signers"), dict):
            parser.error("existing trust store must have version=1 and signers object")
    else:
        doc = {"version": 1, "signers": {}}

    relative_public = os.path.relpath(public_path, trust_path.parent).replace(os.sep, "/")
    signer = doc["signers"].setdefault(args.signer_id, {"active_key_id": args.key_id, "keys": {}})
    if not isinstance(signer, dict):
        parser.error(f"existing signer entry {args.signer_id!r} is malformed")
    signer.setdefault("keys", {})
    if args.key_id in signer["keys"] and not args.force:
        parser.error(f"key id {args.key_id!r} already exists in trust store")
    signer["keys"][args.key_id] = {
        "public_key": relative_public,
        "allowed_topics": topics,
    }
    signer["active_key_id"] = args.key_id
    trust_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    print("Ed25519 command identity generated.")
    print(f"  signer id : {args.signer_id}")
    print(f"  key id    : {args.key_id}")
    print(f"  private   : {private_path}")
    print(f"  public    : {public_path}")
    print(f"  trust     : {trust_path}")
    print("IMPORTANT: distribute only the public key/trust store to gateway, IDS and actuators.")
    print("Keep the private key only on the trusted command issuer. For the local attack simulator,")
    print("the attack harness temporarily needs the controller key to create a valid captured frame;")
    print("do not copy that private key to the Raspberry Pi gateway or ESP32 actuator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
