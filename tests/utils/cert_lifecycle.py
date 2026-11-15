"""Session-scoped X.509/mTLS lifecycle for Lumen Edge Security (Patch 04).

The CA signing key never leaves the KeyAuthority session directory. Clients only
receive paths to the active, verified certificate bundle and the monotone
revocation ledger. Bundles are immutable/versioned; active.json is the sole
atomic publication point (no OS symlinks are used).
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from utils.key_lifecycle import atomic_write_bytes, restrict_to_owner, _FileLock

log = logging.getLogger("lumen.cert_lifecycle")

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_BUNDLE_RE = re.compile(r"^b[0-9]{6}-[a-f0-9]{12}$")
CERT_DIRNAME = "certificates"
POINTER_FILE = "active.json"
REVOCATION_FILE = "revoked_certs.json"
DEFAULT_IDENTITIES = (
    "router", "gateway", "ids", "honeypot", "sensors", "dashboard",
    "lab_tester", "controller", "actuator", "door_sensor_01",
    "gas_sensor_01", "gas_sensor_02", "vibration_sensor_01", "pir_sensor_01",
)
SEPARATED_IDENTITIES = frozenset(("ids", "honeypot"))


class CertStoreError(RuntimeError):
    """Certificate pointer, manifest, bundle, or keypair is missing/corrupt."""


class CertRevocationStoreError(RuntimeError):
    """The certificate revocation ledger is unreadable or corrupt."""


class CertRotationSuspended(RuntimeError):
    """Rotation rate limit exceeded; identity remains quarantined."""


def _json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode("utf-8")


def _utc_timestamp(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), timezone.utc)


def _cert_times(cert: x509.Certificate) -> tuple[datetime, datetime]:
    nb = getattr(cert, "not_valid_before_utc", None)
    na = getattr(cert, "not_valid_after_utc", None)
    if nb is None:
        nb = cert.not_valid_before.replace(tzinfo=timezone.utc)
    if na is None:
        na = cert.not_valid_after.replace(tzinfo=timezone.utc)
    return nb, na


def _spki_der(public_key) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_identity(identity: str) -> str:
    if not isinstance(identity, str) or not _ID_RE.fullmatch(identity):
        raise ValueError(f"invalid certificate identity: {identity!r}")
    return identity


def _read_json(path: Path, exc_type, description: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise exc_type(f"{description} unreadable/corrupt: {exc}") from exc
    if not isinstance(value, dict):
        raise exc_type(f"{description} must be a JSON object")
    return value


def _verify_certificate_signature(cert: x509.Certificate, issuer: x509.Certificate) -> None:
    if cert.issuer != issuer.subject:
        raise CertStoreError("leaf issuer does not match session CA")
    direct = getattr(cert, "verify_directly_issued_by", None)
    if callable(direct):
        try:
            direct(issuer)
            return
        except Exception as exc:
            raise CertStoreError(f"certificate signature validation failed: {exc}") from exc
    # Compatibility path for older cryptography releases.
    pub = issuer.public_key()
    try:
        from cryptography.hazmat.primitives.asymmetric import padding, ec, dsa
        if isinstance(pub, rsa.RSAPublicKey):
            pub.verify(cert.signature, cert.tbs_certificate_bytes,
                       padding.PKCS1v15(), cert.signature_hash_algorithm)
        elif isinstance(pub, ec.EllipticCurvePublicKey):
            pub.verify(cert.signature, cert.tbs_certificate_bytes,
                       ec.ECDSA(cert.signature_hash_algorithm))
        elif isinstance(pub, dsa.DSAPublicKey):
            pub.verify(cert.signature, cert.tbs_certificate_bytes, cert.signature_hash_algorithm)
        else:
            raise CertStoreError("unsupported CA public-key type")
    except CertStoreError:
        raise
    except Exception as exc:
        raise CertStoreError("certificate signature validation failed") from exc


def _parse_revocation_doc(doc: dict) -> dict:
    if doc.get("version") != 1 or not isinstance(doc.get("revoked"), dict):
        raise ValueError("ledger requires version=1 and a revoked object")
    for identity, entries in doc["revoked"].items():
        _safe_identity(identity)
        if not isinstance(entries, dict):
            raise ValueError(f"revoked[{identity!r}] must be an object")
        for serial, meta in entries.items():
            if not isinstance(serial, str) or not re.fullmatch(r"[0-9a-f]{1,128}", serial):
                raise ValueError(f"invalid revoked serial for {identity!r}")
            if not isinstance(meta, dict) or not isinstance(meta.get("ts"), (int, float)):
                raise ValueError(f"invalid revocation metadata for {identity!r}/{serial}")
    return doc


class CertRevocationLedger:
    """Monotone identity/serial revocations; a corrupt ledger is never overwritten."""

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_doc(self) -> dict:
        try:
            doc = _read_json(self.path, CertRevocationStoreError, "certificate revocation ledger")
        except FileNotFoundError as exc:
            # The authority initializes the ledger once during provisioning.
            # Its later disappearance must not mean "nothing revoked".
            raise CertRevocationStoreError("certificate revocation ledger is missing") from exc
        try:
            return _parse_revocation_doc(doc)
        except Exception as exc:
            raise CertRevocationStoreError(f"corrupt certificate revocation ledger: {exc}") from exc

    def ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, _FileLock(self.path):
            if not self.path.exists():
                atomic_write_bytes(self.path, _json_bytes({"version": 1, "revoked": {}}))
            else:
                self._read_doc()

    def revoke(self, identity: str, serial: int | str, reason: str,
               *, common_name: str = "", now: Optional[float] = None) -> bool:
        identity = _safe_identity(identity)
        serial_hex = format(int(serial), "x") if isinstance(serial, int) else str(serial).lower()
        if not re.fullmatch(r"[0-9a-f]{1,128}", serial_hex):
            raise ValueError("invalid certificate serial")
        safe_reason = "".join(c if c.isprintable() else "?" for c in str(reason))[:160]
        with self._lock, _FileLock(self.path):
            doc = self._read_doc()
            entries = doc["revoked"].setdefault(identity, {})
            if serial_hex in entries:
                return False
            entries[serial_hex] = {
                "ts": round(float(time.time() if now is None else now), 3),
                "reason": safe_reason,
                "common_name": str(common_name)[:128],
            }
            atomic_write_bytes(self.path, _json_bytes(doc))
            return True

    def revoked_serials(self, identity: Optional[str] = None) -> frozenset:
        doc = self._read_doc()
        if identity is not None:
            return frozenset(doc["revoked"].get(_safe_identity(identity), {}).keys())
        return frozenset((sid, serial) for sid, entries in doc["revoked"].items() for serial in entries)

    def is_revoked(self, identity: str, serial: int | str) -> bool:
        value = format(int(serial), "x") if isinstance(serial, int) else str(serial).lower()
        return value in self.revoked_serials(identity)

    @classmethod
    def revoke_active(cls, pointer_path: os.PathLike | str, ledger_path: os.PathLike | str,
                      identity: str, reason: str, *, now: Optional[float] = None,
                      quarantine_only: bool = False) -> dict:
        """Gateway/IDS-safe operation: revoke current serial and quarantine identity.

        This writer needs no CA private key and cannot mint certificates. The
        quarantine pointer update and ledger write are both atomic; the router
        supervisor consumes either change and restarts Zenoh with an ACL deny.
        """
        identity = _safe_identity(identity)
        pointer_path = Path(pointer_path)
        # Serialize pointer read/modify/write with bundle publishers; otherwise a
        # racing publication could roll back to an older active bundle.
        with _FileLock(pointer_path):
            pointer, bundle_dir, manifest = CertAuthority.read_active_bundle(pointer_path)
            info = manifest.get("identities", {}).get(identity)
            if not isinstance(info, dict):
                raise KeyError(f"unknown certificate identity: {identity}")
            ledger = cls(ledger_path)
            ledger_error = None
            try:
                ledger.revoke(identity, info["serial"], reason,
                              common_name=info["common_name"], now=now)
            except Exception as exc:
                # A failed ledger write must not leave the compromised CN
                # authorized. The independently atomic pointer quarantine is
                # consumed by both the client watcher and router ACL renderer.
                ledger_error = exc
            quarantined = set(pointer.get("quarantined", []))
            quarantined.add(identity)
            pointer["quarantined"] = sorted(quarantined)
            pending = set(pointer.get("pending_rotation", []))
            if quarantine_only or ledger_error is not None:
                # Do not auto-issue a replacement when persistent revocation
                # could not be committed. Operator must repair the ledger first.
                pending.discard(identity)
            else:
                pending.add(identity)
            pointer["pending_rotation"] = sorted(pending)
            pointer["revision"] = int(pointer.get("revision", 0)) + 1
            pointer["updated_at"] = float(time.time() if now is None else now)
            atomic_write_bytes(pointer_path, _json_bytes(pointer))
            if ledger_error is not None:
                raise CertRevocationStoreError(
                    f"revocation ledger write failed; {identity} quarantined by active pointer"
                ) from ledger_error
        return {"identity": identity, "serial": info["serial"],
                "common_name": info["common_name"], "quarantined": True}


@dataclass(frozen=True)
class ActiveCertificate:
    identity: str
    common_name: str
    serial: int
    generation: int
    not_before: float
    not_after: float
    cert_path: Path
    key_path: Path


class CertAuthority:
    """Session-scoped mTLS CA and leaf certificate authority.

    ``identity`` is the stable application role (e.g. ``ids``); certificate CNs
    are generation-bound after the first rotation (e.g. ``ids-g2-a1b2c3d4``).
    The stable role is also placed in DNS SAN. Zenoh ACLs must match the current
    generation CN, so an old same-role certificate is denied after rotation.
    """

    def __init__(self, session_dir: os.PathLike | str, *,
                 identities: Iterable[str] = DEFAULT_IDENTITIES,
                 leaf_ttl_days: int = 14, ca_validity_days: int = 3650,
                 renew_fraction: float = 2.0 / 3.0,
                 max_rotations_per_hour: int = 6,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 audit_path: os.PathLike | str | None = None):
        self.session_dir = Path(session_dir).resolve()
        self.root = self.session_dir / CERT_DIRNAME
        self.ca_dir = self.root / "ca"
        self.bundles_dir = self.root / "bundles"
        self.pointer_path = self.root / POINTER_FILE
        self.revocation_path = self.root / REVOCATION_FILE
        self.identities = tuple(dict.fromkeys(_safe_identity(i) for i in identities))
        if "router" not in self.identities:
            raise ValueError("router identity is mandatory")
        if not SEPARATED_IDENTITIES.issubset(set(self.identities)):
            raise ValueError("both 'ids' and 'honeypot' identities are mandatory and separate")
        if not 7 <= int(leaf_ttl_days) <= 30:
            raise ValueError("leaf_ttl_days must be between 7 and 30")
        if int(ca_validity_days) < int(leaf_ttl_days) * 2:
            raise ValueError("CA validity must exceed leaf validity")
        if not 0.5 <= float(renew_fraction) < 1.0:
            raise ValueError("renew_fraction must be in [0.5, 1.0)")
        if int(max_rotations_per_hour) < 1:
            raise ValueError("max_rotations_per_hour must be >= 1")
        self.leaf_ttl_days = int(leaf_ttl_days)
        self.ca_validity_days = int(ca_validity_days)
        self.renew_fraction = float(renew_fraction)
        self.max_rotations_per_hour = int(max_rotations_per_hour)
        self._clock, self._mono = clock, monotonic
        self.audit_path = Path(audit_path) if audit_path else self.root / "cert_events.jsonl"
        self._rotations: dict[str, deque] = defaultdict(deque)
        self._suspended: set[str] = set()
        self._lock = threading.RLock()
        self.ledger = CertRevocationLedger(self.revocation_path)

    @property
    def ca_key_path(self) -> Path:
        return self.ca_dir / "ca.key"

    @property
    def ca_cert_path(self) -> Path:
        return self.ca_dir / "ca.pem"

    @classmethod
    def create_session(cls, session_dir: os.PathLike | str, **kwargs) -> "CertAuthority":
        authority = cls(session_dir, **kwargs)
        authority.root.mkdir(parents=True, exist_ok=False)
        authority.ca_dir.mkdir()
        authority.bundles_dir.mkdir()
        restrict_to_owner(authority.root, directory=True)
        restrict_to_owner(authority.ca_dir, directory=True)
        restrict_to_owner(authority.bundles_dir, directory=True)
        try:
            ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
            ca_name = x509.Name([
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cybersecurity AI Lab"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Lumen Session mTLS CA"),
            ])
            now = _utc_timestamp(authority._clock())
            ca_cert = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
                       .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
                       .not_valid_before((now - timedelta(minutes=1)).replace(tzinfo=None))
                       .not_valid_after((now + timedelta(days=authority.ca_validity_days)).replace(tzinfo=None))
                       .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                       .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                           key_encipherment=False, data_encipherment=False, key_agreement=False,
                           key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False),
                           critical=True)
                       .sign(ca_key, hashes.SHA256()))
            atomic_write_bytes(authority.ca_key_path, ca_key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()), private=True)
            atomic_write_bytes(authority.ca_cert_path,
                               ca_cert.public_bytes(serialization.Encoding.PEM))
            authority.ledger.ensure()
            authority._publish_bundle(initial=True)
            authority._audit("session_ca_provisioned", identities=list(authority.identities),
                             leaf_ttl_days=authority.leaf_ttl_days)
        except BaseException:
            authority.cleanup()
            shutil.rmtree(authority.root, ignore_errors=True)
            raise
        return authority

    @classmethod
    def adopt(cls, session_dir: os.PathLike | str, **kwargs) -> "CertAuthority":
        pointer_path = Path(session_dir).resolve() / CERT_DIRNAME / POINTER_FILE
        pointer, _, manifest = cls.read_active_bundle(pointer_path)
        identities = tuple(manifest.get("identities", {}).keys())
        authority = cls(session_dir, identities=identities, **kwargs)
        try:
            ca_cert = x509.load_pem_x509_certificate(authority.ca_cert_path.read_bytes())
            ca_key = serialization.load_pem_private_key(authority.ca_key_path.read_bytes(), password=None)
            if _spki_der(ca_key.public_key()) != _spki_der(ca_cert.public_key()):
                raise CertStoreError("session CA private key does not match ca.pem")
            authority.ledger._read_doc()
            authority._verify_bundle(pointer, manifest)
        except Exception as exc:
            if isinstance(exc, (CertStoreError, CertRevocationStoreError)):
                raise
            raise CertStoreError(f"cannot adopt certificate session: {exc}") from exc
        return authority

    @classmethod
    def from_env(cls, session_dir: os.PathLike | str) -> "CertAuthority":
        return cls.adopt(
            session_dir,
            leaf_ttl_days=int(os.environ.get("LUMEN_CERT_TTL_DAYS", "14")),
            ca_validity_days=int(os.environ.get("LUMEN_CA_VALIDITY_DAYS", "3650")),
            renew_fraction=float(os.environ.get("LUMEN_CERT_RENEW_FRACTION", str(2 / 3))),
            max_rotations_per_hour=int(os.environ.get("LUMEN_CERT_MAX_PER_HOUR", "6")),
            audit_path=os.environ.get("LUMEN_CERT_AUDIT_LOG") or None,
        )

    def env_public(self, identity: str, endpoint: Optional[str] = None) -> dict:
        identity = _safe_identity(identity)
        data = {
            "LUMEN_CERT_POINTER": str(self.pointer_path),
            "LUMEN_CERT_REVOCATIONS": str(self.revocation_path),
            "LUMEN_CERT_IDENTITY": identity,
        }
        if endpoint:
            data["LUMEN_ZENOH_ENDPOINT"] = str(endpoint)
        return data

    @staticmethod
    def read_active_bundle(pointer_path: os.PathLike | str) -> tuple[dict, Path, dict]:
        pointer_path = Path(pointer_path).resolve()
        pointer = _read_json(pointer_path, CertStoreError, "active certificate pointer")
        if pointer.get("version") != 1:
            raise CertStoreError("unsupported certificate pointer version")
        if (isinstance(pointer.get("revision"), bool) or
                not isinstance(pointer.get("revision"), int) or pointer["revision"] < 1):
            raise CertStoreError("invalid active certificate pointer revision")
        for field in ("quarantined", "pending_rotation"):
            value = pointer.get(field, [])
            if not isinstance(value, list) or any(not isinstance(v, str) or not _ID_RE.fullmatch(v) for v in value):
                raise CertStoreError(f"invalid {field} field in active certificate pointer")
        manifest_digest = pointer.get("manifest_sha256")
        if not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
            raise CertStoreError("invalid active certificate manifest digest")
        bundle_id = pointer.get("bundle_id")
        if not isinstance(bundle_id, str) or not _BUNDLE_RE.fullmatch(bundle_id):
            raise CertStoreError("invalid active bundle id")
        bundle_dir = pointer_path.parent / "bundles" / bundle_id
        try:
            bundle_dir.resolve().relative_to((pointer_path.parent / "bundles").resolve())
        except ValueError as exc:
            raise CertStoreError("bundle path escapes certificate store") from exc
        try:
            manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != pointer.get("manifest_sha256"):
                raise CertStoreError("active bundle manifest digest mismatch")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except CertStoreError:
            raise
        except Exception as exc:
            raise CertStoreError(f"active bundle manifest unreadable: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise CertStoreError("invalid active bundle manifest")
        if manifest.get("bundle_id") != bundle_id or not isinstance(manifest.get("identities"), dict):
            raise CertStoreError("active bundle manifest identity/bundle mismatch")
        for identity, info in manifest["identities"].items():
            _safe_identity(identity)
            if not isinstance(info, dict):
                raise CertStoreError(f"invalid identity manifest entry: {identity}")
            if not isinstance(info.get("common_name"), str) or not _ID_RE.fullmatch(info["common_name"]):
                raise CertStoreError(f"invalid common name for identity {identity}")
            if not isinstance(info.get("serial"), str) or not re.fullmatch(r"[0-9a-f]{1,128}", info["serial"]):
                raise CertStoreError(f"invalid serial for identity {identity}")
            for numeric in ("generation", "not_before", "not_after"):
                if isinstance(info.get(numeric), bool) or not isinstance(info.get(numeric), (int, float)):
                    raise CertStoreError(f"invalid {numeric} for identity {identity}")
            if int(info["generation"]) < 1 or float(info["not_after"]) <= float(info["not_before"]):
                raise CertStoreError(f"invalid validity/generation for identity {identity}")
            if info.get("cert_file") != f"{identity}.pem" or info.get("key_file") != f"{identity}.key":
                raise CertStoreError(f"unsafe certificate/key filename for identity {identity}")
        return pointer, bundle_dir, manifest

    @classmethod
    def inspect_active(cls, pointer_path: os.PathLike | str, *, verify_files: bool = True) -> tuple[dict, Path, dict]:
        pointer, bundle_dir, manifest = cls.read_active_bundle(pointer_path)
        if verify_files:
            hashes_doc = manifest.get("files")
            if not isinstance(hashes_doc, dict) or not hashes_doc:
                raise CertStoreError("bundle manifest has no file-integrity map")
            for rel, expected in hashes_doc.items():
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise CertStoreError("unsafe file path in certificate manifest")
                path = bundle_dir / rel_path
                try:
                    actual = _sha256(path)
                except Exception as exc:
                    raise CertStoreError(f"bundle file missing/unreadable: {rel}") from exc
                if not secrets.compare_digest(actual, str(expected)):
                    raise CertStoreError(f"bundle file digest mismatch: {rel}")
        return pointer, bundle_dir, manifest

    def _load_ca(self) -> tuple[object, x509.Certificate]:
        try:
            key = serialization.load_pem_private_key(self.ca_key_path.read_bytes(), password=None)
            cert = x509.load_pem_x509_certificate(self.ca_cert_path.read_bytes())
        except Exception as exc:
            raise CertStoreError(f"session CA unavailable/corrupt: {exc}") from exc
        if _spki_der(key.public_key()) != _spki_der(cert.public_key()):
            raise CertStoreError("session CA keypair mismatch")
        if not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca:
            raise CertStoreError("session certificate is not a CA")
        return key, cert

    def _router_sans(self):
        values = [x509.DNSName("localhost")]
        for raw in os.environ.get("LUMEN_ROUTER_IP", "127.0.0.1").split(","):
            raw = raw.strip()
            if raw:
                values.append(x509.IPAddress(ipaddress.ip_address(raw)))
        return values

    def _issue_leaf(self, identity: str, generation: int, ca_key, ca_cert) -> tuple[bytes, bytes, dict]:
        identity = _safe_identity(identity)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        token = secrets.token_hex(4)
        common_name = identity if generation == 1 else f"{identity}-g{generation}-{token}"
        now = _utc_timestamp(self._clock())
        issuer = ca_cert.subject
        subject = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cybersecurity AI Lab"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        builder = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before((now - timedelta(minutes=1)).replace(tzinfo=None))
                   .not_valid_after((now + timedelta(days=self.leaf_ttl_days)).replace(tzinfo=None)))
        if identity == "router":
            sans = self._router_sans()
            eku = ExtendedKeyUsageOID.SERVER_AUTH
        else:
            sans = [x509.DNSName(identity)]
            eku = ExtendedKeyUsageOID.CLIENT_AUTH
        cert = (builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
                .add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                    key_encipherment=True, data_encipherment=False, key_agreement=False,
                    key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False),
                    critical=True)
                .sign(ca_key, hashes.SHA256()))
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = key.private_bytes(serialization.Encoding.PEM,
                                    serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())
        nb, na = _cert_times(cert)
        info = {
            "common_name": common_name,
            "generation": int(generation),
            "serial": format(cert.serial_number, "x"),
            "not_before": nb.timestamp(),
            "not_after": na.timestamp(),
            "cert_file": f"{identity}.pem",
            "key_file": f"{identity}.key",
        }
        return cert_pem, key_pem, info

    def _verify_leaf(self, bundle_dir: Path, identity: str, info: dict,
                     ca_cert: x509.Certificate, now: Optional[float] = None) -> tuple[bytes, bytes]:
        try:
            cert_bytes = (bundle_dir / info["cert_file"]).read_bytes()
            key_bytes = (bundle_dir / info["key_file"]).read_bytes()
            cert = x509.load_pem_x509_certificate(cert_bytes)
            key = serialization.load_pem_private_key(key_bytes, password=None)
            cns = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            if len(cns) != 1 or cns[0].value != info["common_name"]:
                raise CertStoreError(f"{identity}: certificate CN/manifest mismatch")
            if cert.serial_number != int(info["serial"], 16):
                raise CertStoreError(f"{identity}: certificate serial/manifest mismatch")
            _verify_certificate_signature(cert, ca_cert)
            if _spki_der(key.public_key()) != _spki_der(cert.public_key()):
                raise CertStoreError(f"{identity}: private key does not match certificate")
            nb, na = _cert_times(cert)
            current = _utc_timestamp(self._clock() if now is None else now)
            if not nb <= current <= na:
                raise CertStoreError(f"{identity}: certificate outside validity window")
            eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            required = (ExtendedKeyUsageOID.SERVER_AUTH if identity == "router"
                        else ExtendedKeyUsageOID.CLIENT_AUTH)
            if required not in eku:
                raise CertStoreError(f"{identity}: wrong ExtendedKeyUsage")
            sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            dns = {x.value for x in sans if isinstance(x, x509.DNSName)}
            if identity not in dns and not (identity == "router" and "localhost" in dns):
                raise CertStoreError(f"{identity}: stable DNS SAN missing")
            return cert_bytes, key_bytes
        except CertStoreError:
            raise
        except Exception as exc:
            raise CertStoreError(f"{identity}: invalid certificate/key material: {exc}") from exc

    def _verify_bundle(self, pointer: dict, manifest: dict) -> dict:
        _, bundle_dir, check_manifest = self.inspect_active(self.pointer_path)
        if check_manifest != manifest:
            raise CertStoreError("active bundle changed while it was being verified")
        ca_cert = x509.load_pem_x509_certificate((bundle_dir / "ca.pem").read_bytes())
        identities = manifest.get("identities")
        if not isinstance(identities, dict) or set(identities) != set(self.identities):
            raise CertStoreError("bundle identity set does not match configured identities")
        pubkeys, serials = set(), set()
        now = self._clock()
        for identity, info in identities.items():
            cert_bytes, key_bytes = self._verify_leaf(bundle_dir, identity, info, ca_cert, now)
            cert = x509.load_pem_x509_certificate(cert_bytes)
            pub = _spki_der(cert.public_key())
            if pub in pubkeys:
                raise CertStoreError("duplicate public keys across certificate identities")
            pubkeys.add(pub)
            serial = cert.serial_number
            if serial in serials:
                raise CertStoreError("duplicate certificate serials")
            serials.add(serial)
        for identity in SEPARATED_IDENTITIES:
            if identity not in identities:
                raise CertStoreError(f"required separated identity missing: {identity}")
        return identities

    def _publish_bundle(self, *, initial: bool = False,
                        replacement: Optional[tuple[str, bytes, bytes, dict]] = None,
                        clear_quarantine: Optional[str] = None) -> dict:
        with self._lock, _FileLock(self.pointer_path):
            old_pointer = None
            old_manifest = None
            old_dir = None
            if not initial and self.pointer_path.exists():
                old_pointer, old_dir, old_manifest = self.inspect_active(self.pointer_path)
            revision = int(old_pointer.get("revision", 0) if old_pointer else 0) + 1
            bundle_id = f"b{revision:06d}-{secrets.token_hex(6)}"
            final_dir = self.bundles_dir / bundle_id
            stage_dir = self.bundles_dir / f".stage-{secrets.token_hex(8)}"
            stage_dir.mkdir()
            restrict_to_owner(stage_dir, directory=True)
            try:
                ca_pem = self.ca_cert_path.read_bytes()
                atomic_write_bytes(stage_dir / "ca.pem", ca_pem)
                identities_doc: dict = {}
                if old_manifest:
                    for identity, info in old_manifest["identities"].items():
                        cert_data, key_data = self._verify_leaf(old_dir, identity, info,
                            x509.load_pem_x509_certificate(ca_pem), self._clock())
                        atomic_write_bytes(stage_dir / f"{identity}.pem", cert_data)
                        atomic_write_bytes(stage_dir / f"{identity}.key", key_data, private=True)
                        identities_doc[identity] = dict(info)
                ca_key, ca_cert = self._load_ca()
                if initial:
                    targets = self.identities
                else:
                    targets = (replacement[0],) if replacement else ()
                for identity in targets:
                    generation = 1 if initial else int(identities_doc[identity]["generation"]) + 1
                    if replacement and identity == replacement[0]:
                        cert_data, key_data, info = replacement[1], replacement[2], dict(replacement[3])
                    else:
                        cert_data, key_data, info = self._issue_leaf(identity, generation, ca_key, ca_cert)
                    atomic_write_bytes(stage_dir / f"{identity}.pem", cert_data)
                    atomic_write_bytes(stage_dir / f"{identity}.key", key_data, private=True)
                    identities_doc[identity] = info
                if set(identities_doc) != set(self.identities):
                    raise CertStoreError("staged certificate bundle is incomplete")
                hashes_doc = {"ca.pem": _sha256(stage_dir / "ca.pem")}
                for identity in self.identities:
                    hashes_doc[f"{identity}.pem"] = _sha256(stage_dir / f"{identity}.pem")
                    hashes_doc[f"{identity}.key"] = _sha256(stage_dir / f"{identity}.key")
                manifest = {"version": 1, "bundle_id": bundle_id,
                            "created_at": float(self._clock()),
                            "leaf_ttl_days": self.leaf_ttl_days,
                            "identities": identities_doc, "files": hashes_doc}
                atomic_write_bytes(stage_dir / "manifest.json", _json_bytes(manifest))
                # Verify the complete package before it can become visible.
                for rel, expected in hashes_doc.items():
                    if not secrets.compare_digest(_sha256(stage_dir / rel), expected):
                        raise CertStoreError(f"staged package integrity failure: {rel}")
                ca_cert_stage = x509.load_pem_x509_certificate((stage_dir / "ca.pem").read_bytes())
                ca_constraints = ca_cert_stage.extensions.get_extension_for_class(x509.BasicConstraints).value
                if not ca_constraints.ca:
                    raise CertStoreError("staged ca.pem is not a CA certificate")
                pubs, serials = set(), set()
                now_dt = _utc_timestamp(self._clock())
                for identity, info in identities_doc.items():
                    cert = x509.load_pem_x509_certificate((stage_dir / info["cert_file"]).read_bytes())
                    key = serialization.load_pem_private_key((stage_dir / info["key_file"]).read_bytes(), password=None)
                    _verify_certificate_signature(cert, ca_cert_stage)
                    if _spki_der(cert.public_key()) != _spki_der(key.public_key()):
                        raise CertStoreError(f"staged keypair mismatch for {identity}")
                    cn_values = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
                    if len(cn_values) != 1 or cn_values[0].value != info["common_name"]:
                        raise CertStoreError(f"staged CN does not match manifest for {identity}")
                    if format(cert.serial_number, "x") != info["serial"]:
                        raise CertStoreError(f"staged serial does not match manifest for {identity}")
                    nb, na = _cert_times(cert)
                    if not nb <= now_dt <= na:
                        raise CertStoreError(f"staged certificate is outside validity for {identity}")
                    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
                    expected_eku = (ExtendedKeyUsageOID.SERVER_AUTH if identity == "router"
                                    else ExtendedKeyUsageOID.CLIENT_AUTH)
                    if expected_eku not in eku:
                        raise CertStoreError(f"staged ExtendedKeyUsage invalid for {identity}")
                    sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
                    dns_names = {x.value for x in sans if isinstance(x, x509.DNSName)}
                    if identity not in dns_names and not (identity == "router" and "localhost" in dns_names):
                        raise CertStoreError(f"staged stable DNS SAN missing for {identity}")
                    pub = _spki_der(cert.public_key())
                    if pub in pubs or cert.serial_number in serials:
                        raise CertStoreError("staged identities are not cryptographically isolated")
                    pubs.add(pub)
                    serials.add(cert.serial_number)
                if not initial and old_pointer:
                    quarantined = set(old_pointer.get("quarantined", []))
                    pending_rotation = set(old_pointer.get("pending_rotation", []))
                else:
                    quarantined = set()
                    pending_rotation = set()
                if clear_quarantine:
                    quarantined.discard(clear_quarantine)
                    pending_rotation.discard(clear_quarantine)
                os.replace(stage_dir, final_dir)
                new_pointer = {
                    "version": 1, "bundle_id": bundle_id, "revision": revision,
                    "manifest_sha256": hashlib.sha256((final_dir / "manifest.json").read_bytes()).hexdigest(),
                    "quarantined": sorted(quarantined),
                    "pending_rotation": sorted(pending_rotation),
                    "updated_at": float(self._clock()),
                }
                atomic_write_bytes(self.pointer_path, _json_bytes(new_pointer))
                self._audit("bundle_published", bundle_id=bundle_id, revision=revision,
                            identities=len(identities_doc))
                return new_pointer
            except BaseException:
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise

    def _new_replacement(self, identity: str) -> tuple[bytes, bytes, dict]:
        pointer, bundle_dir, manifest = self.inspect_active(self.pointer_path)
        if identity not in manifest["identities"]:
            raise KeyError(f"unknown certificate identity: {identity}")
        old_info = manifest["identities"][identity]
        ca_key, ca_cert = self._load_ca()
        return self._issue_leaf(identity, int(old_info["generation"]) + 1, ca_key, ca_cert)

    def rotate(self, identity: str, reason: str = "scheduled", *, old_already_revoked: bool = False) -> str:
        identity = _safe_identity(identity)
        with self._lock:
            if identity in self._suspended:
                raise CertRotationSuspended(f"{identity}: rotation is suspended")
            pointer, _, manifest = self.inspect_active(self.pointer_path)
            old_info = manifest["identities"].get(identity)
            if not isinstance(old_info, dict):
                raise KeyError(identity)
            if self.ledger.is_revoked(identity, old_info["serial"]):
                old_already_revoked = True
            now_mono = self._mono()
            window = self._rotations[identity]
            while window and now_mono - window[0] > 3600.0:
                window.popleft()
            if len(window) >= self.max_rotations_per_hour:
                self.ledger.revoke(identity, old_info["serial"], "rotation-rate-limit",
                                   common_name=old_info["common_name"], now=self._clock())
                blocked = set(pointer.get("quarantined", []))
                blocked.add(identity)
                with _FileLock(self.pointer_path):
                    # Reload under the inter-process lock so no concurrent IDS
                    # revocation or bundle publish can be overwritten.
                    current_pointer, _, _ = self.inspect_active(self.pointer_path)
                    blocked = set(current_pointer.get("quarantined", []))
                    blocked.add(identity)
                    current_pointer["quarantined"] = sorted(blocked)
                    pending = set(current_pointer.get("pending_rotation", []))
                    pending.discard(identity)
                    current_pointer["pending_rotation"] = sorted(pending)
                    current_pointer["revision"] = int(current_pointer.get("revision", 0)) + 1
                    current_pointer["updated_at"] = float(self._clock())
                    atomic_write_bytes(self.pointer_path, _json_bytes(current_pointer))
                self._suspended.add(identity)
                self._audit("rotation_suspended", identity=identity, reason=reason)
                raise CertRotationSuspended(f"{identity}: {len(window)} rotations in the last hour")
            if not old_already_revoked:
                self.ledger.revoke(identity, old_info["serial"], reason,
                                   common_name=old_info["common_name"], now=self._clock())
            replacement = self._new_replacement(identity)
            self._publish_bundle(replacement=(identity, *replacement), clear_quarantine=identity)
            window.append(now_mono)
            self._audit("certificate_rotated", identity=identity,
                        old_serial=old_info["serial"], new_serial=replacement[2]["serial"], reason=reason)
            return replacement[2]["common_name"]

    def revoke(self, identity: str, reason: str, *, rotate: bool = True) -> Optional[str]:
        result = CertRevocationLedger.revoke_active(
            self.pointer_path, self.revocation_path, identity, reason, now=self._clock())
        self._audit("certificate_revoked", **result, reason=reason)
        if not rotate:
            self._suspended.add(identity)
            return None
        return self.rotate(identity, reason=f"incident:{reason}", old_already_revoked=True)

    def quarantine(self, identity: str, reason: str = "operator-quarantine") -> None:
        identity = _safe_identity(identity)
        CertRevocationLedger.revoke_active(self.pointer_path, self.revocation_path,
                                             identity, reason, now=self._clock(),
                                             quarantine_only=True)
        self._suspended.add(identity)
        self._audit("identity_quarantined", identity=identity, reason=reason)

    def poll(self) -> list[dict]:
        # Read the ledger before making decisions: corruption always aborts polling.
        revoked = self.ledger.revoked_serials()
        pointer, _, manifest = self.inspect_active(self.pointer_path)
        quarantined = set(pointer.get("quarantined", []))
        pending_rotation = set(pointer.get("pending_rotation", []))
        events = []
        for identity, info in list(manifest["identities"].items()):
            if identity in self._suspended or (identity in quarantined and identity not in pending_rotation):
                continue
            reason = None
            if identity in pending_rotation or (identity, info["serial"]) in revoked:
                reason = "incident"
            else:
                lifetime = float(info["not_after"]) - float(info["not_before"])
                renew_at = float(info["not_before"]) + lifetime * self.renew_fraction
                if float(self._clock()) >= renew_at:
                    reason = "scheduled"
            if not reason:
                continue
            try:
                new_cn = self.rotate(identity, reason=reason, old_already_revoked=(reason == "incident"))
                events.append({"event": "rotated", "identity": identity,
                               "reason": reason, "common_name": new_cn,
                               "router_restart_required": identity == "router"})
            except CertRotationSuspended as exc:
                events.append({"event": "suspended", "identity": identity, "detail": str(exc)})
        return events

    def status(self) -> dict:
        revoked = self.ledger.revoked_serials()
        pointer, _, manifest = self.inspect_active(self.pointer_path)
        result = {}
        quarantined = set(pointer.get("quarantined", []))
        now = float(self._clock())
        for identity, info in manifest["identities"].items():
            lifetime = float(info["not_after"]) - float(info["not_before"])
            renew_at = float(info["not_before"]) + lifetime * self.renew_fraction
            result[identity] = {
                "common_name": info["common_name"], "serial": info["serial"],
                "generation": int(info["generation"]),
                "not_after": float(info["not_after"]),
                "renewal_due": now >= renew_at,
                "revoked": (identity, info["serial"]) in revoked,
                "quarantined": identity in quarantined,
                "pending_rotation": identity in set(pointer.get("pending_rotation", [])),
                "suspended": identity in self._suspended,
            }
        return result

    def active_certificate(self, identity: str) -> ActiveCertificate:
        identity = _safe_identity(identity)
        pointer, bundle_dir, manifest = self.inspect_active(self.pointer_path)
        info = manifest["identities"].get(identity)
        if not info:
            raise KeyError(identity)
        if identity in set(pointer.get("quarantined", [])):
            raise CertStoreError(f"identity {identity} is quarantined")
        if self.ledger.is_revoked(identity, info["serial"]):
            raise CertStoreError(f"active certificate for {identity} is revoked")
        return ActiveCertificate(identity, info["common_name"], int(info["serial"], 16),
                                  int(info["generation"]), float(info["not_before"]),
                                  float(info["not_after"]), bundle_dir / info["cert_file"],
                                  bundle_dir / info["key_file"])

    def _audit(self, event: str, **fields) -> None:
        record = {"ts": round(float(self._clock()), 3), "event": event, **fields}
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            log.exception("certificate audit log write failed")
        log.warning("[CERT LIFECYCLE] %s %s", event, fields)

    def cleanup(self) -> None:
        """Best-effort overwrite and remove the session CA private key only."""
        path = self.ca_key_path
        try:
            if path.is_file():
                size = path.stat().st_size
                with path.open("r+b") as handle:
                    handle.write(b"\0" * size)
                    handle.flush()
                    os.fsync(handle.fileno())
                path.unlink()
        except OSError:
            log.exception("failed to wipe session CA private key")


def revoke_active_certificate(pointer_path: os.PathLike | str,
                               ledger_path: os.PathLike | str,
                               identity: str, reason: str, *,
                               quarantine_only: bool = False) -> dict:
    """Privilege-limited API for Gateway/IDS incident handlers (no CA signing).

    ``quarantine_only=True`` prevents the authority daemon from automatically
    minting a replacement for a device that may still be compromised.
    """
    return CertRevocationLedger.revoke_active(pointer_path, ledger_path, identity, reason,
                                               quarantine_only=quarantine_only)


def default_identities() -> tuple[str, ...]:
    """Return default system identities; additional sensor identities can be appended."""
    extra = [item.strip() for item in os.environ.get("LUMEN_CERT_EXTRA_IDENTITIES", "").split(",") if item.strip()]
    return tuple(dict.fromkeys((*DEFAULT_IDENTITIES, *(_safe_identity(i) for i in extra))))
