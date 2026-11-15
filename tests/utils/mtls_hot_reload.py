"""Hot-reloading Zenoh mTLS client bound to the active certificate pointer.

Use this instead of a one-shot ``zenoh.open`` call. The object is a small
session proxy: publishers/queries always target the current session, and
subscriber declarations are recreated after a certificate bundle changes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from utils.cert_lifecycle import (
    CertAuthority, CertRevocationLedger, CertRevocationStoreError, CertStoreError,
    _cert_times, _spki_der, _verify_certificate_signature,
)

log = logging.getLogger("lumen.mtls_hot_reload")


class _DeclarationProxy:
    """Re-declarable Zenoh entity proxy (subscriber, queryable, publisher)."""

    def __init__(self, owner: "HotReloadZenohClient", method: str, *args, **kwargs):
        self.owner = owner
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._handle = None
        self._closed = False

    def _bind(self, session) -> None:
        if self._closed:
            return
        self._handle = getattr(session, self.method)(*self.args, **self.kwargs)

    def _unbind(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.undeclare()
            except Exception:
                log.debug("old Zenoh %s already closed", self.method, exc_info=True)

    def undeclare(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unbind()
        self.owner._remove_declaration(self)

    def __getattr__(self, name):
        handle = self._handle
        if handle is None:
            raise RuntimeError(f"Zenoh {self.method} declaration is not active")
        return getattr(handle, name)


class HotReloadZenohClient:
    """A single logical Zenoh client that reconnects on active-pointer updates."""

    def __init__(self, identity: str, *, endpoint: Optional[str] = None,
                 pointer_path: os.PathLike | str | None = None,
                 ledger_path: os.PathLike | str | None = None,
                 poll_interval: float = 0.5,
                 session_factory: Optional[Callable[[Path, Path, Path, str], object]] = None,
                 clock: Callable[[], float] = time.time,
                 start_monitor: bool = True):
        self.identity = str(identity)
        if not self.identity or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in self.identity):
            raise ValueError("invalid mTLS identity")
        self.endpoint = endpoint or os.environ.get("LUMEN_ZENOH_ENDPOINT", "tls/127.0.0.1:7447")
        if not self.endpoint.startswith("tls/"):
            raise ValueError("mTLS client endpoint must use tls/...")
        pointer_value = pointer_path or os.environ.get("LUMEN_CERT_POINTER")
        ledger_value = ledger_path or os.environ.get("LUMEN_CERT_REVOCATIONS")
        if not pointer_value or not ledger_value:
            raise CertStoreError("LUMEN_CERT_POINTER and LUMEN_CERT_REVOCATIONS are required")
        self.pointer_path = Path(pointer_value)
        self.ledger_path = Path(ledger_value)
        self._clock = clock
        self.poll_interval = max(0.1, float(poll_interval))
        self._factory = session_factory or self._open_zenoh_session
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._session = None
        self._signature = None
        self._declarations: list[_DeclarationProxy] = []
        self._stop = threading.Event()
        self._thread = None
        self._refresh(force=True)
        if start_monitor:
            self._thread = threading.Thread(target=self._monitor, name=f"mtls-reload-{identity}", daemon=True)
            self._thread.start()

    @staticmethod
    def _open_zenoh_session(cert_path: Path, key_path: Path, ca_path: Path, endpoint: str):
        try:
            import zenoh
        except ImportError as exc:
            raise RuntimeError("eclipse-zenoh Python binding is required for mTLS clients") from exc
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("scouting/gossip/enabled", "false")
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("transport/link/tls", json.dumps({
            "root_ca_certificate": str(ca_path.resolve()),
            "enable_mtls": True,
            "connect_private_key": str(key_path.resolve()),
            "connect_certificate": str(cert_path.resolve()),
            "close_link_on_expiration": True,
        }))
        return zenoh.open(config)

    def _resolve_active(self):
        pointer, bundle_dir, manifest = CertAuthority.inspect_active(self.pointer_path)
        info = manifest.get("identities", {}).get(self.identity)
        if not isinstance(info, dict):
            raise CertStoreError(f"identity {self.identity!r} absent from active certificate bundle")
        if self.identity in set(pointer.get("quarantined", [])):
            raise CertStoreError(f"identity {self.identity!r} is quarantined")
        ledger = CertRevocationLedger(self.ledger_path)
        if ledger.is_revoked(self.identity, info["serial"]):
            raise CertStoreError(f"active certificate for {self.identity!r} has been revoked")
        try:
            cert = x509.load_pem_x509_certificate((bundle_dir / info["cert_file"]).read_bytes())
            key = serialization.load_pem_private_key((bundle_dir / info["key_file"]).read_bytes(), password=None)
            ca = x509.load_pem_x509_certificate((bundle_dir / "ca.pem").read_bytes())
            if _spki_der(cert.public_key()) != _spki_der(key.public_key()):
                raise CertStoreError("client certificate/private-key mismatch")
            _verify_certificate_signature(cert, ca)
            nb, na = _cert_times(cert)
            now = datetime.fromtimestamp(float(self._clock()), timezone.utc)
            if not nb <= now <= na:
                raise CertStoreError("client certificate is outside its validity window")
            cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            if len(cn) != 1 or cn[0].value != info["common_name"]:
                raise CertStoreError("client certificate CN differs from bundle manifest")
            eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            if ExtendedKeyUsageOID.CLIENT_AUTH not in eku:
                raise CertStoreError("client certificate lacks clientAuth EKU")
        except CertStoreError:
            raise
        except Exception as exc:
            raise CertStoreError(f"active client keypair/certificate is invalid: {exc}") from exc
        return pointer, bundle_dir, info

    @staticmethod
    def _pointer_signature(pointer: dict) -> tuple:
        return (pointer.get("bundle_id"), int(pointer.get("revision", 0)),
                pointer.get("manifest_sha256"))

    def _close_session(self, session) -> None:
        if session is None:
            return
        try:
            session.close()
        except Exception:
            log.debug("Zenoh session close failed during mTLS reload", exc_info=True)

    def _refresh(self, *, force: bool = False) -> bool:
        # Multiple publisher threads and the watcher can observe the same pointer
        # change. Serialize session creation/swap to avoid duplicate connections.
        with self._refresh_lock:
            return self._refresh_unlocked(force=force)

    def _refresh_unlocked(self, *, force: bool = False) -> bool:
        try:
            pointer, bundle_dir, info = self._resolve_active()
            signature = self._pointer_signature(pointer)
        except (CertStoreError, CertRevocationStoreError, OSError, ValueError) as exc:
            with self._lock:
                old = self._session
                self._session = None
                self._signature = None
                for declaration in self._declarations:
                    declaration._unbind()
            self._close_session(old)
            log.error("mTLS identity %s is fail-closed: %s", self.identity, exc)
            return False

        with self._lock:
            if not force and signature == self._signature and self._session is not None:
                return True

        try:
            new_session = self._factory(
                bundle_dir / info["cert_file"], bundle_dir / info["key_file"],
                bundle_dir / "ca.pem", self.endpoint,
            )
            for declaration in list(self._declarations):
                declaration._bind(new_session)
        except Exception as exc:
            # Old session is already revoked whenever a legitimate rotation changed
            # the pointer. Never keep it alive as a fallback after incident/expiry.
            self._close_session(locals().get("new_session"))
            with self._lock:
                old = self._session
                self._session = None
                self._signature = None
                for declaration in self._declarations:
                    declaration._unbind()
            self._close_session(old)
            log.error("mTLS reconnect failed for %s; client remains disconnected: %s", self.identity, exc)
            return False

        with self._lock:
            old = self._session
            self._session = new_session
            self._signature = signature
        self._close_session(old)
        log.info("mTLS session active identity=%s cn=%s bundle=%s",
                 self.identity, info["common_name"], pointer["bundle_id"])
        return True

    def _monitor(self) -> None:
        while not self._stop.wait(self.poll_interval):
            self._refresh()

    def _remove_declaration(self, declaration: _DeclarationProxy) -> None:
        with self._lock:
            try:
                self._declarations.remove(declaration)
            except ValueError:
                pass

    def _declare(self, method: str, *args, **kwargs):
        proxy = _DeclarationProxy(self, method, *args, **kwargs)
        with self._lock:
            self._declarations.append(proxy)
            session = self._session
        if session is None:
            self._remove_declaration(proxy)
            raise RuntimeError(f"mTLS session for {self.identity} is not connected")
        try:
            proxy._bind(session)
        except Exception:
            self._remove_declaration(proxy)
            raise
        return proxy

    def declare_subscriber(self, key_expr, callback, **kwargs):
        return self._declare("declare_subscriber", key_expr, callback, **kwargs)

    def _current_session(self):
        self._refresh()
        with self._lock:
            if self._session is None:
                raise RuntimeError(f"mTLS session for {self.identity} is not connected (fail-closed)")
            return self._session

    def put(self, *args, **kwargs):
        return self._current_session().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self._current_session().get(*args, **kwargs)

    def declare_queryable(self, *args, **kwargs):
        return self._declare("declare_queryable", *args, **kwargs)

    def declare_publisher(self, *args, **kwargs):
        return self._declare("declare_publisher", *args, **kwargs)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.poll_interval * 4))
        with self._refresh_lock, self._lock:
            old = self._session
            self._session = None
            for declaration in self._declarations:
                declaration._unbind()
            self._declarations.clear()
        self._close_session(old)

    def __getattr__(self, name):
        # Preserve the rest of the zenoh.Session API for callers, while resolving
        # it against the currently active session on each access.
        return getattr(self._current_session(), name)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def open_mtls_client(identity: str, **kwargs) -> HotReloadZenohClient:
    """Factory used by Gateway, IDS, sensors, dashboard, controller, actuator."""
    return HotReloadZenohClient(identity, **kwargs)
