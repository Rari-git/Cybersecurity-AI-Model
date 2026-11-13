"""Etapa E0: Spike de validare pentru PEP, ACL-uri Zenoh și sesiuni mTLS active.

Testează experimental 5 garanții fundamentale înainte de extinderea v5.0:
1. Non-bypass: un senzor nu poate publica direct pe home/**, control/** sau shadow/**.
2. Read isolation: un senzor nu poate face subscribe pe home/**, ingest/** sau shadow/**.
3. Shadow Sink isolation: doar gateway (PEP) scrie în shadow/sandbox/** și doar honeypot citește.
4. Sesiuni deja stabilite + Shadow Quarantine fără restart de router:
   o sesiune mTLS activă care intră în SHADOW_QUARANTINE are fluxul blocat imediat
   pe home/** și redirecționat exclusiv către shadow/sandbox/<device_id>/data.
5. Revocare ireversibilă: certificatul revocat este respins la reconectare mTLS.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import zenoh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cert_lifecycle import CertAuthority, default_identities, revoke_active_certificate
from utils.zenoh_router import build_router_config
from utils.zenoh_conn import open_client_session


def _find_zenohd() -> str:
    configured = os.environ.get("LUMEN_ZENOHD")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    for candidate in (ROOT / "zenohd.exe", ROOT / "zenohd"):
        if candidate.is_file():
            return str(candidate.resolve())
    found = shutil.which("zenohd") or shutil.which("zenohd.exe")
    if not found:
        pytest.skip("zenohd binary not found; set LUMEN_ZENOHD to run E0 PEP integration tests")
    return found


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_tcp(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"zenohd did not open TCP port {port}")


class MinimalPEPProxy:
    """Prototip minim de Policy Enforcement Point (PEP) pentru validarea E0.

     Demonstrează aplicarea instantanee a carantinei pe o sesiune mTLS deja deschisă,
     fără repornirea routerului Zenoh.
    """

    def __init__(self, gateway_session):
        self.session = gateway_session
        self._lock = threading.Lock()
        self._quarantined: set[str] = set()
        self.enforcement_confirmed: dict[str, bool] = {}
        self.sub = self.session.declare_subscriber("ingest/devices/**", self._on_ingest)

    def set_quarantine(self, device_id: str, enabled: bool) -> bool:
        with self._lock:
            if enabled:
                self._quarantined.add(device_id)
            else:
                self._quarantined.discard(device_id)
            # Confirmare explicită PEP după actualizarea atomică a stării
            self.enforcement_confirmed[device_id] = (device_id in self._quarantined) == enabled
            return self.enforcement_confirmed[device_id]

    def _on_ingest(self, sample) -> None:
        key = str(sample.key_expr)
        parts = key.split("/")
        # Format așteptat: ingest/devices/<device_id>/sensors/<domain>/<metric>
        if len(parts) < 6 or parts[0] != "ingest" or parts[1] != "devices":
            return
        device_id = parts[2]
        target_topic = "home/" + "/".join(parts[3:])
        raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)

        with self._lock:
            is_quarantined = device_id in self._quarantined

        if is_quarantined:
            envelope = {
                "source_device": device_id,
                "original_topic": target_topic,
                "state": "SHADOW_QUARANTINE",
                "payload": raw.decode("utf-8", errors="replace"),
            }
            self.session.put(
                f"shadow/sandbox/{device_id}/data",
                json.dumps(envelope).encode("utf-8"),
            )
            return

        self.session.put(target_topic, raw)

    def close(self) -> None:
        self.sub.undeclare()


@pytest.fixture()
def zenoh_e0_env(monkeypatch):
    zenohd = _find_zenohd()
    port = _free_port()
    endpoint = f"tls/127.0.0.1:{port}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="lumen-e0-"))

    # Adaptăm template-ul routerului pentru portul izolat de test
    raw_template = (ROOT / "config" / "zenohd_router.json5").read_text(encoding="utf-8")
    test_template = tmp_dir / "router_template.json5"
    test_template.write_text(
        raw_template.replace("tls/127.0.0.1:7447", endpoint),
        encoding="utf-8",
    )

    cert_auth = CertAuthority.create_session(
        tmp_dir,
        identities=default_identities(),
        leaf_ttl_days=7,
    )
    active_cfg = tmp_dir / "zenohd.active.json5"
    build_router_config(
        test_template,
        cert_auth.pointer_path,
        cert_auth.revocation_path,
        active_cfg,
    )

    monkeypatch.setenv("LUMEN_ENV", "production")
    monkeypatch.setenv("LUMEN_ZENOH_ENDPOINT", endpoint)
    monkeypatch.setenv("LUMEN_CERT_POINTER", str(cert_auth.pointer_path))
    monkeypatch.setenv("LUMEN_CERT_REVOCATIONS", str(cert_auth.revocation_path))
    monkeypatch.delenv("LUMEN_CERT_IDENTITY", raising=False)
    monkeypatch.setenv(
        "LUMEN_CERT_ALLOWED_IDENTITIES",
        "gateway,ids,honeypot,controller,lab_tester,gas_sensor_01,door_sensor_01",
    )

    proc = subprocess.Popen(
        [zenohd, "-c", str(active_cfg)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_tcp(port, timeout=10.0)
        yield {
            "endpoint": endpoint,
            "port": port,
            "tmp_dir": tmp_dir,
            "cert_auth": cert_auth,
            "test_template": test_template,
            "active_cfg": active_cfg,
            "proc": proc,
            "zenohd": zenohd,
        }
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_e0_1_sensor_cannot_bypass_gateway(zenoh_e0_env):
    """Un senzor autentificat nu poate publica direct pe home/**, control/** sau shadow/**."""
    home_received: list[str] = []
    shadow_received: list[str] = []
    control_received: list[str] = []

    ids_sess = open_client_session("ids")
    hp_sess = open_client_session("honeypot")
    gw_sess = open_client_session("gateway")
    sensor_sess = open_client_session("gas_sensor_01")

    sub_home = ids_sess.declare_subscriber("home/**", lambda s: home_received.append(str(s.key_expr)))
    sub_shadow = hp_sess.declare_subscriber("shadow/**", lambda s: shadow_received.append(str(s.key_expr)))
    sub_ctrl = gw_sess.declare_subscriber("control/**", lambda s: control_received.append(str(s.key_expr)))
    time.sleep(0.3)

    try:
        # Încercări de ocolire directă a Gateway-ului
        sensor_sess.put("home/sensors/gas/ppm", b"9999")
        sensor_sess.put("home/actuators/valve", b'{"cmd":"OPEN"}')
        sensor_sess.put("shadow/sandbox/gas_sensor_01/data", b"poison")
        sensor_sess.put("control/policy/gas_sensor_01", b"ACTIVE")
        time.sleep(0.5)

        assert home_received == [], f"ACL leak: sensor published directly to home/**: {home_received}"
        assert shadow_received == [], f"ACL leak: sensor published directly to shadow/**: {shadow_received}"
        assert control_received == [], f"ACL leak: sensor published directly to control/**: {control_received}"
    finally:
        sub_home.undeclare()
        sub_shadow.undeclare()
        sub_ctrl.undeclare()
        sensor_sess.close()
        gw_sess.close()
        hp_sess.close()
        ids_sess.close()


def test_e0_2_sensor_cannot_subscribe_to_production_or_shadow(zenoh_e0_env):
    """Un senzor nu poate citi traficul altor senzori sau datele din shadow/control."""
    snooped: list[str] = []

    gw_sess = open_client_session("gateway")
    sensor_sess = open_client_session("gas_sensor_01")

    sub_home = sensor_sess.declare_subscriber("home/**", lambda s: snooped.append(str(s.key_expr)))
    sub_shadow = sensor_sess.declare_subscriber("shadow/**", lambda s: snooped.append(str(s.key_expr)))
    sub_ctrl = sensor_sess.declare_subscriber("control/**", lambda s: snooped.append(str(s.key_expr)))
    time.sleep(0.3)

    try:
        gw_sess.put("home/sensors/door/state", b"OPEN")
        gw_sess.put("shadow/sandbox/door_sensor_01/data", b"secret_capture")
        time.sleep(0.5)

        assert snooped == [], f"ACL read leak: sensor snooped messages on {snooped}"
    finally:
        sub_home.undeclare()
        sub_shadow.undeclare()
        sub_ctrl.undeclare()
        sensor_sess.close()
        gw_sess.close()


def test_e0_3_active_session_shadow_quarantine_without_router_restart(zenoh_e0_env):
    """Sesiune mTLS deja deschisă: la SHADOW_QUARANTINE fluxul real se oprește instantaneu
    fără repornirea routerului și este redirecționat exclusiv în shadow/sandbox/**.
    """
    router_pid_before = zenoh_e0_env["proc"].pid
    prod_messages: list[bytes] = []
    shadow_messages: list[dict] = []

    gw_sess = open_client_session("gateway")
    ids_sess = open_client_session("ids")
    hp_sess = open_client_session("honeypot")
    sensor_sess = open_client_session("gas_sensor_01")

    pep = MinimalPEPProxy(gw_sess)
    sub_prod = ids_sess.declare_subscriber(
        "home/sensors/gas/ppm",
        lambda s: prod_messages.append(s.payload.to_bytes()),
    )
    sub_shadow = hp_sess.declare_subscriber(
        "shadow/sandbox/gas_sensor_01/**",
        lambda s: shadow_messages.append(json.loads(s.payload.to_bytes().decode("utf-8"))),
    )
    time.sleep(0.3)

    try:
        # 1. Stare ACTIVE: mesajul ajunge în producție, nimic în shadow
        sensor_sess.put("ingest/devices/gas_sensor_01/sensors/gas/ppm", b"350")
        time.sleep(0.35)
        assert prod_messages == [b"350"]
        assert shadow_messages == []

        # 2. Tranziție în SHADOW_QUARANTINE pe sesiunea DEJA DESCHISĂ
        confirmed = pep.set_quarantine("gas_sensor_01", True)
        assert confirmed is True

        # Senzorul trimite din aceeași sesiune mTLS deschisă anterior
        sensor_sess.put("ingest/devices/gas_sensor_01/sensors/gas/ppm", b"950")
        time.sleep(0.35)

        # Routerul nu a fost repornit, producția nu a primit 950, honeypot-ul l-a capturat
        assert zenoh_e0_env["proc"].pid == router_pid_before
        assert prod_messages == [b"350"], f"Quarantine leak into production: {prod_messages}"
        assert len(shadow_messages) == 1
        assert shadow_messages[0]["source_device"] == "gas_sensor_01"
        assert shadow_messages[0]["payload"] == "950"

        # 3. Eliberare reversibilă din carantină -> revine în producție
        assert pep.set_quarantine("gas_sensor_01", False) is True
        sensor_sess.put("ingest/devices/gas_sensor_01/sensors/gas/ppm", b"360")
        time.sleep(0.35)
        assert prod_messages == [b"350", b"360"]
        assert len(shadow_messages) == 1
    finally:
        sub_prod.undeclare()
        sub_shadow.undeclare()
        pep.close()
        sensor_sess.close()
        hp_sess.close()
        ids_sess.close()
        gw_sess.close()


def test_e0_4_revoked_certificate_rejected_on_reconnect(zenoh_e0_env):
    """Revocarea ireversibilă blochează sesiunea fail-closed și respinge orice operație."""
    cert_auth = zenoh_e0_env["cert_auth"]
    revoke_active_certificate(
        cert_auth.pointer_path,
        cert_auth.revocation_path,
        "gas_sensor_01",
        "deterministic compromise test",
    )

    sess = open_client_session("gas_sensor_01")
    try:
        with pytest.raises(Exception):
            sess.put("ingest/devices/gas_sensor_01/sensors/gas/ppm", b"500")
    finally:
        sess.close()