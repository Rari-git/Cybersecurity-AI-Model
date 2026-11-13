"""Key Authority daemon for Patch 03.1 + Patch 04.

  python -m agent.key_authority serve --session-dir <dir>
  python -m agent.key_authority status --session-dir <dir>
  python -m agent.key_authority revoke --session-dir <dir> --signer controller
  python -m agent.key_authority revoke --session-dir <dir> --cert-identity gas_sensor_01
  python -m agent.key_authority revoke --session-dir <dir> --cert-identity ids --quarantine-cert
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.key_lifecycle import KeyAuthority, RevocationLedger, RevocationStoreError, load_revocations
from utils.cert_lifecycle import (
    CertAuthority, CertRevocationLedger, CertRevocationStoreError, CertStoreError,
    CertRotationSuspended,
)
from utils.logger import get_script_logger

logger = get_script_logger("key_authority")
POLL_INTERVAL_SEC = 0.2
HEARTBEAT_INTERVAL_SEC = 30.0
ERROR_BACKOFF_SEC = 2.0
LOG_REPEAT_INTERVAL_SEC = 10.0

_CTRL_BREAK_EVENT = 1
_CTRL_CLOSE_EVENT = 2
_CTRL_LOGOFF_EVENT = 5
_CTRL_SHUTDOWN_EVENT = 6
_CONSOLE_STOP_EVENTS = (_CTRL_BREAK_EVENT, _CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT)
_CONSOLE_GRACE_SEC = 4.0
_cleanup_done = threading.Event()


def _session_dir(arg) -> str:
    value = arg or os.environ.get("LUMEN_SESSION_DIR", "")
    if not value:
        raise SystemExit("session dir required (--session-dir or LUMEN_SESSION_DIR)")
    return value


def _install_console_handler(request_stop):
    if os.name != "nt":
        return None
    import ctypes
    handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    def _handler(ctrl_type):
        if ctrl_type in _CONSOLE_STOP_EVENTS:
            request_stop()
            _cleanup_done.wait(_CONSOLE_GRACE_SEC)
            return True
        return False

    callback = handler_type(_handler)
    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, True):
        logger.error("[KEY AUTHORITY] cannot install console handler; session cleanup on console close is not guaranteed")
        return None
    return callback


class _RateLimitedLog:
    def __init__(self, interval: float = LOG_REPEAT_INTERVAL_SEC):
        self.interval = interval
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if now - self._last.get(key, float("-inf")) >= self.interval:
            self._last[key] = now
            return True
        return False


def _adopt_authorities(session_dir: str):
    key_authority = KeyAuthority.adopt(
        session_dir,
        max_rotations_per_hour=int(os.environ.get("LUMEN_REKEY_MAX_PER_HOUR", "6")),
        max_key_age_sec=float(os.environ.get("LUMEN_KEY_MAX_AGE_SEC", "0")),
        audit_path=os.environ.get("LUMEN_KEY_AUDIT_LOG") or None,
    )
    cert_authority = CertAuthority.from_env(session_dir)
    key_authority.attach_cert_authority(cert_authority)
    return key_authority, cert_authority


def serve(session_dir: str) -> int:
    authority, cert_authority = _adopt_authorities(session_dir)
    stop = threading.Event()

    def _stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)
    _console_cb = _install_console_handler(_stop)  # noqa: F841

    quiet = _RateLimitedLog()
    backoff_until = 0.0
    last_beat = float("-inf")
    snapshot = authority.status()
    logger.info("[KEY AUTHORITY] ready session=%s signers=%s certs=%s",
                os.path.basename(session_dir),
                sorted(k for k in snapshot if k != "certificates"),
                sorted(snapshot.get("certificates", {})))
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_INTERVAL_SEC:
                try:
                    os.utime(session_dir, None)
                except OSError:
                    pass
                last_beat = now
            if now >= backoff_until:
                try:
                    for event in authority.poll():
                        if event.get("component") == "certificate":
                            if event["event"] == "rotated":
                                logger.warning("[CERT AUTHORITY] rotated identity=%s reason=%s cn=%s router_restart=%s",
                                               event["identity"], event["reason"], event["common_name"],
                                               event.get("router_restart_required", False))
                            else:
                                logger.critical("[CERT AUTHORITY] ROTATION SUSPENDED identity=%s (%s)",
                                                event["identity"], event["detail"])
                        elif event["event"] == "rotated":
                            logger.warning("[KEY AUTHORITY] rotated signer=%s reason=%s new_key_id=%s",
                                           event["signer"], event["reason"], event["key_id"])
                        else:
                            logger.critical("[KEY AUTHORITY] ROTATION SUSPENDED signer=%s (%s). Signer remains locked out.",
                                            event["signer"], event["detail"])
                except (RevocationStoreError, CertRevocationStoreError, CertStoreError):
                    if quiet.allow("lifecycle_store"):
                        logger.critical("[KEY AUTHORITY] key/certificate store unreadable; lifecycle is fail-closed; not rotating",
                                        exc_info=True)
                except Exception as exc:
                    backoff_until = now + ERROR_BACKOFF_SEC
                    if quiet.allow("poll_error"):
                        logger.critical("[KEY AUTHORITY] lifecycle poll failed (%s); retrying in %.0fs; affected identities remain denied",
                                        type(exc).__name__, ERROR_BACKOFF_SEC, exc_info=True)
            stop.wait(POLL_INTERVAL_SEC)
    finally:
        try:
            # KeyAuthority.cleanup() first wipes the X.509 CA key then removes the
            # entire per-session store, including leaf keys and Ed25519 keys.
            authority.cleanup()
        finally:
            logger.info("[KEY AUTHORITY] session key material wiped")
            _cleanup_done.set()
    return 0


def status(session_dir: str) -> int:
    authority, cert_authority = _adopt_authorities(session_dir)
    snapshot = authority.status()
    for signer, info in snapshot.items():
        if signer != "certificates":
            print(f"signer/{signer}: {info}")
    for identity, info in cert_authority.status().items():
        print(f"certificate/{identity}: {info}")
    print("revoked command keys:", sorted(load_revocations(authority.revocation_path)))
    print("revoked certificates:", sorted(CertRevocationLedger(cert_authority.revocation_path).revoked_serials()))
    return 0


def revoke(session_dir: str, signer: str, reason: str,
           cert_identity: str | None = None, quarantine_cert: bool = False) -> int:
    authority, cert_authority = _adopt_authorities(session_dir)
    if signer and signer in authority._state:
        key_id = authority.status()[signer]["active_key_id"]
        RevocationLedger(authority.revocation_path).revoke(signer, key_id, reason)
        print(f"revoked command key {signer}/{key_id}")
    if cert_identity:
        if quarantine_cert:
            cert_authority.quarantine(cert_identity, reason)
            print(f"quarantined mTLS identity {cert_identity}; router ACL will deny its current CN")
        else:
            new_cn = cert_authority.revoke(cert_identity, reason, rotate=True)
            print(f"revoked previous mTLS certificate for {cert_identity}; replacement CN={new_cn}")
    elif not signer or signer not in authority._state:
        raise SystemExit("select an existing --signer and/or --cert-identity")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent.key_authority")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "status", "revoke"):
        p = sub.add_parser(name)
        p.add_argument("--session-dir")
        if name == "revoke":
            p.add_argument("--signer", default="")
            p.add_argument("--cert-identity")
            p.add_argument("--reason", default="operator")
            p.add_argument("--quarantine-cert", action="store_true")
    args = parser.parse_args(argv)
    session_dir = _session_dir(args.session_dir)
    if args.cmd == "serve":
        return serve(session_dir)
    if args.cmd == "status":
        return status(session_dir)
    signer = args.signer or ""
    return revoke(session_dir, signer, args.reason, args.cert_identity, args.quarantine_cert)


if __name__ == "__main__":
    raise SystemExit(main())
