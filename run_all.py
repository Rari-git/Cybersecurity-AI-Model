"""Portable Lumen Edge Security orchestrator (Windows + Linux/POSIX).

The orchestrator owns the ephemeral session directory and the zenohd process.
It watches the mTLS active pointer + revocation ledger and restarts zenohd when
its TLS certificate or generation-bound ACL changes.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

from utils.cert_lifecycle import CertAuthority, default_identities
from utils.zenoh_router import build_router_config, validate_router_acl_template
from utils.key_lifecycle import KeyAuthority, sweep_stale_sessions
from models.bundle_loader import verify_and_load_bundle
from models.model_lifecycle import resolve_active_bundle

ROOT = Path(__file__).resolve().parent
ZENOH_ENDPOINT = os.environ.get("LUMEN_ZENOH_ENDPOINT", "tls/127.0.0.1:7447")
SIGNER_TEMPLATE = Path(os.environ.get("LUMEN_COMMAND_SIGNER_TEMPLATE", ROOT / "config/trusted_command_signers.json"))
SIGNER_ID = os.environ.get("LUMEN_COMMAND_SIGNER_ID", "controller")
ROUTER_TEMPLATE = Path(os.environ.get("LUMEN_ZENOH_ROUTER_CONFIG", ROOT / "config/zenohd_router.json5"))
DEVICE_REGISTRY = Path(os.environ.get("LUMEN_DEVICE_REGISTRY", ROOT / "config/devices.json"))
MODEL_ROOT = Path(os.environ.get("LUMEN_MODEL_ROOT", ROOT / "models")).resolve()
ACTIVE_MODEL_POINTER = Path(os.environ.get(
    "LUMEN_ACTIVE_MODEL_POINTER", MODEL_ROOT / "active_model.json"
)).resolve()
MODEL_PUBLIC_KEY = Path(os.environ.get(
    "LUMEN_MODEL_PUBKEY", ROOT / "config/model_keys/model_signing_ed25519.pub"
)).resolve()
POLL_INTERVAL_SEC = float(os.environ.get("LUMEN_ROUTER_SUPERVISOR_POLL_SEC", "0.5"))

REQUIRED_IDENTITIES = (
    "gateway", "ids", "honeypot", "sensors", "dashboard", "lab_tester",
    "controller", "actuator", "door_sensor_01", "gas_sensor_01", "gas_sensor_02",
    "vibration_sensor_01", "pir_sensor_01",
)
REQUIRED_CERTS = ["ca.pem", "router.pem", "router.key"] + [
    f"{cn}.{ext}" for cn in REQUIRED_IDENTITIES for ext in ("pem", "key")
]

GRID_SLOTS = (
    "Zenoh Router",
    "Key Authority",
    "Inline Gateway",
    "IDS Agent AI",
    "Honeypot Sink",
    "Virtual Sensors",
    "Dashboard Streamlit",
    "Meniu Atacuri",
)

_STARTF_USESIZE = getattr(subprocess, "STARTF_USESIZE", 0x00000002)
_STARTF_USEPOSITION = getattr(subprocess, "STARTF_USEPOSITION", 0x00000004)

logging.basicConfig(
    level=os.environ.get("LUMEN_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lumen.run_all")


def _get_work_area() -> tuple[int, int, int, int]:
    """Return usable primary monitor rect (left, top, width, height) excluding taskbar."""
    if os.name != "nt":
        return (0, 0, 1920, 1080)
    try:
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        rect = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
            return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    except Exception:
        pass
    return (0, 0, 1920, 1040)


def _grid_rect(slot_index: int, total_slots: int = 8, cols: int = 4) -> tuple[int, int, int, int]:
    """Compute (x, y, width, height) for a compact 4x2 grid cell."""
    left, top, work_w, work_h = _get_work_area()
    rows = max(1, (total_slots + cols - 1) // cols)
    col = slot_index % cols
    row = slot_index // cols
    cell_w = max(320, work_w // cols)
    cell_h = max(220, work_h // rows)
    x = left + col * cell_w
    y = top + row * cell_h
    w = (left + work_w - x) if col == cols - 1 else cell_w
    h = (top + work_h - y) if row == rows - 1 else cell_h
    return (x, y, w, h)


def _win32_visible_windows() -> dict[int, tuple[int, str]]:
    """Return {hwnd: (pid, title)} for all visible top-level windows on Windows."""
    if os.name != "nt":
        return {}
    user32 = ctypes.windll.user32
    found: dict[int, tuple[int, str]] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.GetWindow(hwnd, 4):  # GW_OWNER != 0 (skip tool/owned popups)
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found[int(hwnd)] = (int(pid.value), buf.value)
        return True

    user32.EnumWindows(callback, 0)
    return found


def _place_window_in_slot(pid: int, title: str, slot_index: int,
                          before_hwnds: set[int], timeout: float = 2.0) -> None:
    """Move and resize the spawned console/terminal window into its 4x2 grid slot."""
    if os.name != "nt":
        return
    x, y, w, h = _grid_rect(slot_index, len(GRID_SLOTS))
    user32 = ctypes.windll.user32
    deadline = time.monotonic() + timeout
    target_hwnd = None
    while time.monotonic() < deadline:
        windows = _win32_visible_windows()
        # 1. Direct PID match (classic conhost.exe console)
        for hwnd, (win_pid, _win_title) in windows.items():
            if win_pid == pid and hwnd not in before_hwnds:
                target_hwnd = hwnd
                break
        # 2. Newly appeared window matching title or WindowsTerminal host
        if target_hwnd is None:
            for hwnd, (_win_pid, win_title) in windows.items():
                if hwnd not in before_hwnds and (title in win_title or win_title):
                    target_hwnd = hwnd
                    break
        if target_hwnd is not None:
            break
        time.sleep(0.05)

    if target_hwnd is not None:
        try:
            user32.ShowWindow(target_hwnd, 9)  # SW_RESTORE
            user32.MoveWindow(target_hwnd, x, y, w, h, True)
        except Exception:
            pass


def _spawn_console_window(command: list[str], *, cwd: Path, env: dict[str, str],
                          title: str, slot_index: int) -> subprocess.Popen:
    """Spawn a process in its own positioned, compact console window."""
    if os.name != "nt":
        return subprocess.Popen(command, cwd=str(cwd), env=env)

    before_hwnds = set(_win32_visible_windows().keys())
    x, y, w, h = _grid_rect(slot_index, len(GRID_SLOTS))
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= _STARTF_USEPOSITION | _STARTF_USESIZE
    startupinfo.dwX = x
    startupinfo.dwY = y
    startupinfo.dwXSize = w
    startupinfo.dwYSize = h
    startupinfo.lpTitle = title

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo,
    )
    _place_window_in_slot(proc.pid, title, slot_index, before_hwnds)
    return proc


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """Parse a Zenoh tls/host:port or tcp/host:port endpoint."""
    if not isinstance(endpoint, str) or "/" not in endpoint:
        raise ValueError(f"invalid Zenoh endpoint: {endpoint!r}")
    scheme, authority = endpoint.split("/", 1)
    if scheme not in {"tls", "tcp"}:
        raise ValueError(f"unsupported Zenoh endpoint scheme: {scheme!r}")
    if authority.startswith("["):
        close = authority.find("]")
        if close < 0 or close + 1 >= len(authority) or authority[close + 1] != ":":
            raise ValueError(f"invalid bracketed IPv6 endpoint: {endpoint!r}")
        host, port_text = authority[1:close], authority[close + 2:]
    else:
        host, separator, port_text = authority.rpartition(":")
        if not separator or not host:
            raise ValueError(f"endpoint must include host and port: {endpoint!r}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"endpoint port out of range: {port}")
    return host, port


def wait_for_router_listener(endpoint: str, timeout: float = 15.0) -> bool:
    """Wait for the TCP listener before starting dependent Zenoh clients.

    This confirms socket readiness only; TLS/mTLS authentication is performed
    by the clients and remains fail-closed if certificates or ACLs are invalid.
    """
    host, port = _endpoint_host_port(endpoint)
    deadline = time.monotonic() + max(0.1, float(timeout))
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    log.error("Zenoh listener did not become ready at %s:%d: %s", host, port, last_error)
    return False


def _find_zenohd() -> str:
    configured = os.environ.get("LUMEN_ZENOHD")
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(configured)
        if found:
            return found
        raise FileNotFoundError(f"LUMEN_ZENOHD not found: {configured}")
    candidates = [ROOT / ("zenohd.exe" if os.name == "nt" else "zenohd"), ROOT / "zenohd"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    found = shutil.which("zenohd") or shutil.which("zenohd.exe")
    if found:
        return found
    raise FileNotFoundError("zenohd binary not found; set LUMEN_ZENOHD")


def preflight_or_exit() -> None:
    problems = []
    if not os.environ.get("LUMEN_HMAC_KEY"):
        import secrets
        os.environ["LUMEN_HMAC_KEY"] = secrets.token_urlsafe(48)
    elif len(os.environ["LUMEN_HMAC_KEY"].encode("utf-8")) < 32:
        problems.append("LUMEN_HMAC_KEY must contain at least 32 UTF-8 bytes")
    if not SIGNER_TEMPLATE.is_file():
        problems.append(f"signer template missing: {SIGNER_TEMPLATE}")
    if not ROUTER_TEMPLATE.is_file():
        problems.append(f"Zenoh router ACL template missing: {ROUTER_TEMPLATE}")
    else:
        try:
            validate_router_acl_template(ROUTER_TEMPLATE)
        except Exception as exc:
            problems.append(f"Zenoh router ACL template failed E1/E2 validation: {exc}")
    if not DEVICE_REGISTRY.is_file():
        problems.append(f"device registry missing: {DEVICE_REGISTRY}")
    if not ZENOH_ENDPOINT.startswith("tls/"):
        problems.append(f"LUMEN_ZENOH_ENDPOINT must use tls/... (got {ZENOH_ENDPOINT!r})")
    try:
        _find_zenohd()
    except FileNotFoundError as exc:
        problems.append(str(exc))
    try:
        resolve_verified_active_model_bundle()
    except Exception as exc:
        problems.append(f"signed active ONNX bundle unavailable: {exc}")
    if problems:
        raise SystemExit("Startup cancelled (fail-closed):\n  - " + "\n  - ".join(problems))


def resolve_verified_active_model_bundle() -> Path:
    """Resolve the active pointer and verify its Ed25519-signed ONNX bundle.

    The launcher deliberately does not accept ``LUMEN_MODEL_BUNDLE_DIR`` as an
    override: production must follow the atomic active-model pointer so a
    candidate cannot be loaded merely because an inherited shell variable is
    set.  IDS repeats bundle verification immediately before creating its ONNX
    Runtime session.
    """
    if not MODEL_PUBLIC_KEY.is_file():
        raise FileNotFoundError(f"model signing public key missing: {MODEL_PUBLIC_KEY}")
    bundle_dir = resolve_active_bundle(
        model_root=MODEL_ROOT,
        pointer_path=ACTIVE_MODEL_POINTER,
        fallback_bundle=MODEL_ROOT / "lumen_model_bundle",
    )
    verified = verify_and_load_bundle(bundle_dir, MODEL_PUBLIC_KEY)
    if not verified.onnx_path.is_file():
        raise FileNotFoundError(f"verified ONNX artifact is missing: {verified.onnx_path}")
    return Path(bundle_dir).resolve()


def provision_ephemeral_session(project_dir: Path):
    sweep_stale_sessions()
    authority = KeyAuthority.create_session(
        SIGNER_TEMPLATE,
        signers=[SIGNER_ID],
        audit_path=project_dir / "logs" / "key_events.jsonl",
    )
    try:
        cert_authority = CertAuthority.create_session(
            authority.session_dir,
            identities=default_identities(),
            leaf_ttl_days=int(os.environ.get("LUMEN_CERT_TTL_DAYS", "14")),
            ca_validity_days=int(os.environ.get("LUMEN_CA_VALIDITY_DAYS", "3650")),
            renew_fraction=float(os.environ.get("LUMEN_CERT_RENEW_FRACTION", str(2 / 3))),
            max_rotations_per_hour=int(os.environ.get("LUMEN_CERT_MAX_PER_HOUR", "6")),
            audit_path=project_dir / "logs" / "cert_events.jsonl",
        )
    except BaseException:
        authority.cleanup()
        raise
    authority.attach_cert_authority(cert_authority)
    pointer, bundle_dir, _manifest = CertAuthority.inspect_active(cert_authority.pointer_path)
    missing = [name for name in REQUIRED_CERTS if not (bundle_dir / name).is_file()]
    if missing:
        authority.cleanup()
        raise RuntimeError("initial certificate bundle is incomplete: " + ", ".join(missing))
    return authority, cert_authority


def build_window_env(authority: KeyAuthority, cert_authority: CertAuthority, project_dir: Path) -> dict[str, dict[str, str]]:
    public_base = {
        "LUMEN_TRUSTED_COMMAND_SIGNERS": str(authority.trust_path),
        "LUMEN_REVOKED_KEYS": str(authority.revocation_path),
        "LUMEN_REKEY_ENABLED": "1",
        "LUMEN_CERT_POINTER": str(cert_authority.pointer_path.resolve()),
        "LUMEN_CERT_REVOCATIONS": str(cert_authority.revocation_path.resolve()),
        "LUMEN_ZENOH_ENDPOINT": ZENOH_ENDPOINT,
    }
    # Pin IDS to the verified active pointer/key and disallow the legacy
    # unbundled CUSUM-only fallback in the live stack.  Do not pass a resolved
    # bundle directory: IDS must independently call resolve_active_bundle().
    ids_model_env = {
        "LUMEN_MODEL_ROOT": str(MODEL_ROOT),
        "LUMEN_ACTIVE_MODEL_POINTER": str(ACTIVE_MODEL_POINTER),
        "LUMEN_MODEL_PUBKEY": str(MODEL_PUBLIC_KEY),
        "LUMEN_ALLOW_UNBUNDLED_FALLBACK": "0",
    }

    device_identities = [
        name for name in cert_authority.identities
        if name.endswith("_sensor_01") or name.endswith("_sensor_02")
    ]
    device_identities_csv = ",".join(device_identities)
    attack_identities_csv = ",".join(
        dict.fromkeys(["controller", "lab_tester", "sensors", *device_identities])
    )

    def client(identity: str) -> dict[str, str]:
        return {**public_base, "LUMEN_CERT_IDENTITY": identity}

    gateway_ready_file = authority.session_dir / "gateway_ready.json"
    gateway_ready_file.parent.mkdir(parents=True, exist_ok=True)
    gateway_ready_file.unlink(missing_ok=True)

    return {
        "Inline Gateway": {
            **client("gateway"),
            "LUMEN_GATEWAY_READY_FILE": str(gateway_ready_file.resolve()),
        },
        "IDS Agent AI": {**client("ids"), **ids_model_env},
        "Honeypot Sink": client("honeypot"),
        "Virtual Sensors": {
            **client("sensors"),
            "LUMEN_CERT_DEVICE_IDENTITIES": device_identities_csv,
            "LUMEN_CERT_ALLOWED_IDENTITIES": device_identities_csv,
        },
        "Dashboard Streamlit": client("dashboard"),
        "Meniu Atacuri": {
            **client("controller"),
            "LUMEN_CERT_ALLOWED_IDENTITIES": attack_identities_csv,
            **authority.env_signer(SIGNER_ID),
        },
        "Key Authority": {
            "LUMEN_SESSION_DIR": str(authority.session_dir),
            "LUMEN_KEY_AUDIT_LOG": str(project_dir / "logs" / "key_events.jsonl"),
            "LUMEN_CERT_AUDIT_LOG": str(project_dir / "logs" / "cert_events.jsonl"),
            "LUMEN_CERT_TTL_DAYS": str(cert_authority.leaf_ttl_days),
            "LUMEN_CA_VALIDITY_DAYS": str(cert_authority.ca_validity_days),
            "LUMEN_CERT_MAX_PER_HOUR": str(cert_authority.max_rotations_per_hour),
            "LUMEN_ZENOH_ENDPOINT": ZENOH_ENDPOINT,
        },
    }


def build_child_env(title: str, envs: dict[str, dict[str, str]],
                    parent_env: dict[str, str] | None = None) -> dict[str, str]:
    """Merge managed environment for one child, removing unsafe model overrides."""
    child_env = dict(os.environ if parent_env is None else parent_env)
    child_env.update(envs.get(title, {}))
    if title == "IDS Agent AI":
        # An inherited override would bypass the signed active-model pointer.
        child_env.pop("LUMEN_MODEL_BUNDLE_DIR", None)
        child_env.pop("LUMEN_ALLOW_EXPLICIT_MODEL_BUNDLE", None)
        child_env["LUMEN_ALLOW_UNBUNDLED_FALLBACK"] = "0"
    if title == "Virtual Sensors" and child_env.get("LUMEN_RUN_LIVE_SMOKE", "0") == "1":
        # The live smoke opens a dedicated mTLS session for gas_sensor_01 and
        # registers a fresh boot epoch to test sequence/replay handling. Keep
        # the background simulator from concurrently registering/publishing
        # the same identity, which would intentionally supersede that epoch.
        # Preserve both inherited and managed skip lists rather than allowing
        # one environment layer to silently discard the other.
        inherited = parent_env or os.environ
        skipped = {
            item.strip()
            for source in (
                inherited.get("LUMEN_SIM_SKIP_DEVICES", ""),
                envs.get(title, {}).get("LUMEN_SIM_SKIP_DEVICES", ""),
                child_env.get("LUMEN_SIM_SKIP_DEVICES", ""),
            )
            for item in source.split(",")
            if item.strip()
        }
        skipped.add("gas_sensor_01")
        child_env["LUMEN_SIM_SKIP_DEVICES"] = ",".join(sorted(skipped))
    if title == "Meniu Atacuri":
        child_env["LUMEN_ATTACK_PREFIX"] = "ingest"
    return child_env


class RouterSupervisor:
    def __init__(self, authority: CertAuthority, *, template_path: Path,
                 zenohd: str, endpoint: str, cwd: Path, slot_index: int = 0):
        self.authority = authority
        self.template_path = Path(template_path)
        self.zenohd = zenohd
        self.endpoint = endpoint
        self.cwd = Path(cwd)
        self.slot_index = slot_index
        self.config_path = authority.root / "zenohd.active.json5"
        self.process: subprocess.Popen | None = None
        self._digest: str | None = None
        self._stop = threading.Event()

    def _spawn(self) -> None:
        env = os.environ.copy()
        self.process = _spawn_console_window(
            [self.zenohd, "-c", str(self.config_path)],
            cwd=self.cwd,
            env=env,
            title="Zenoh Router",
            slot_index=self.slot_index,
        )
        log.warning("[ZENOH ROUTER] started pid=%s config=%s", self.process.pid, self.config_path)

    def _stop_process(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def refresh(self, *, force: bool = False) -> bool:
        try:
            digest, _bundle_dir = build_router_config(
                self.template_path, self.authority.pointer_path,
                self.authority.revocation_path, self.config_path,
            )
        except Exception:
            # If config/ledger is corrupt, do not leave a previously authorized
            # router accepting clients. This is an intentional fail-closed stop.
            self._stop_process()
            raise
        changed = force or digest != self._digest
        if changed:
            self._stop_process()
            self._spawn()
            self._digest = digest
            return True
        if self.process is None or self.process.poll() is not None:
            self._spawn()
            return True
        return False

    def run(self) -> None:
        self.refresh(force=True)
        while not self._stop.wait(POLL_INTERVAL_SEC):
            try:
                was_dead = self.process is None or self.process.poll() is not None
                if self.refresh():
                    if was_dead:
                        log.warning("[ZENOH ROUTER] process exited unexpectedly; restarted")
                    else:
                        log.warning("[ZENOH ROUTER] certificate/ACL state changed; controlled restart completed")
            except Exception:
                log.critical("[ZENOH ROUTER] certificate state invalid; router stopped (fail-closed)", exc_info=True)
                self._stop.set()
                raise

    def stop(self) -> None:
        self._stop.set()
        self._stop_process()


def wait_for_gateway_ready_marker(path: str | Path, timeout: float = 90.0) -> dict:
    """Wait until Gateway has declared its authenticated boot subscribers."""
    ready_path = Path(path).expanduser().resolve()
    required = {
        "request_subscriber": "control/boot/request/**",
        "proof_subscriber": "control/boot/proof/**",
        "challenge_publisher_prefix": "control/boot/challenge/",
        "result_publisher_prefix": "control/boot/result/",
    }
    deadline = time.monotonic() + timeout
    last_error = "marker not created yet"
    while time.monotonic() < deadline:
        try:
            document = json.loads(ready_path.read_text(encoding="utf-8"))
            if document.get("status") == "READY" and document.get("boot_topics") == required:
                return document
            last_error = f"unexpected Gateway readiness marker: {document!r}"
        except FileNotFoundError:
            last_error = "Gateway readiness marker not created yet"
        except (OSError, json.JSONDecodeError) as exc:
            last_error = f"invalid Gateway readiness marker: {exc}"
        time.sleep(0.1)
    raise TimeoutError(f"Gateway boot subscribers did not become ready within {timeout:.1f}s: {last_error}")


def _client_processes(project_dir: Path):
    py = sys.executable
    return [
        ("Key Authority", [py, "-m", "agent.key_authority", "serve"]),
        ("Inline Gateway", [py, "-m", "agent.gateway_ids"]),
        ("IDS Agent AI", [py, "-m", "agent.ids_agent"]),
        ("Honeypot Sink", [py, "-m", "agent.honeypot"]),
        ("Virtual Sensors", [py, "-m", "simulation.virtual_sensors_devices"]),
        ("Dashboard Streamlit", [py, "-m", "streamlit", "run", "ui/app.py"]),
        ("Meniu Atacuri", [py, "run_attacks.py"]),
    ]




def recover_deployment_journal_or_exit() -> list[dict]:
    """Recover an interrupted model-pointer transaction before startup preflight.

    The deployment authority's opt-in flag is enabled only for this narrowly
    scoped recovery call and is restored before any child process is launched.
    Recovery validates the journal, pointer state, signed bundles, and
    deployment attestation; ambiguous or corrupt states cancel startup.
    """
    authority_var = "LUMEN_MODEL_DEPLOYMENT_AUTHORITY"
    previous_value = os.environ.get(authority_var)
    os.environ[authority_var] = "1"
    try:
        from training.deployment_authority import recover_deployment_transactions

        outcomes = recover_deployment_transactions(
            model_root=MODEL_ROOT,
            public_key_path=MODEL_PUBLIC_KEY,
            pointer_path=ACTIVE_MODEL_POINTER,
        )
    except Exception as exc:
        log.critical(
            "Startup cancelled (fail-closed): deployment journal recovery failed: %s",
            exc,
            exc_info=True,
        )
        raise SystemExit(
            "Startup cancelled (fail-closed): deployment journal recovery failed; "
            "inspect the deployment journal and model pointers before retrying."
        ) from exc
    finally:
        if previous_value is None:
            os.environ.pop(authority_var, None)
        else:
            os.environ[authority_var] = previous_value

    if outcomes:
        log.warning("[MODEL DEPLOYMENT RECOVERY] reconciled transactions: %s", outcomes)
    else:
        log.info("[MODEL DEPLOYMENT RECOVERY] no incomplete deployment transactions")
    return outcomes



def launch_and_arrange_grid() -> int:
    recover_deployment_journal_or_exit()
    preflight_or_exit()
    project_dir = ROOT
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    key_authority, cert_authority = provision_ephemeral_session(project_dir)
    envs = build_window_env(key_authority, cert_authority, project_dir)
    processes: list[tuple[str, subprocess.Popen]] = []
    supervisor = RouterSupervisor(
        cert_authority, template_path=ROUTER_TEMPLATE, zenohd=_find_zenohd(),
        endpoint=ZENOH_ENDPOINT, cwd=project_dir, slot_index=0,
    )
    stop = threading.Event()

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    router_thread = threading.Thread(target=supervisor.run, name="zenoh-router-supervisor", daemon=True)
    try:
        router_thread.start()
        # Wait for the initial router start before launching dependent clients.
        deadline = time.monotonic() + 10.0
        while supervisor.process is None and time.monotonic() < deadline and router_thread.is_alive():
            time.sleep(0.05)
        if supervisor.process is None or supervisor.process.poll() is not None:
            raise RuntimeError("zenohd did not start")
        if not wait_for_router_listener(ZENOH_ENDPOINT, timeout=15.0):
            raise RuntimeError(
                f"zenohd process exists but listener {ZENOH_ENDPOINT!r} is not ready; "
                "inspect the zenohd console for TLS/configuration errors"
            )

        for idx, (title, command) in enumerate(_client_processes(project_dir), start=1):
            env = build_child_env(title, envs)
            process = _spawn_console_window(
                command,
                cwd=project_dir,
                env=env,
                title=title,
                slot_index=idx,
            )
            processes.append((title, process))
            log.info("launched %s pid=%s", title, process.pid)
            if title == "Inline Gateway":
                ready_file = envs["Inline Gateway"].get("LUMEN_GATEWAY_READY_FILE")
                if not ready_file:
                    raise RuntimeError("Gateway readiness marker path was not provisioned")
                ready_timeout = float(os.environ.get("LUMEN_GATEWAY_READY_TIMEOUT_SEC", "90"))
                ready = wait_for_gateway_ready_marker(ready_file, timeout=ready_timeout)
                log.info(
                    "[GATEWAY READY BARRIER] run_id=%s boot_request=%s boot_proof=%s",
                    ready.get("run_id"), ready["boot_topics"]["request_subscriber"],
                    ready["boot_topics"]["proof_subscriber"],
                )

        if os.environ.get("LUMEN_RUN_LIVE_SMOKE", "0") == "1":
            # Exercise the actual mTLS/Zenoh route only after Router, Gateway,
            # IDS, and the virtual sensors have been started.  This short-lived
            # verifier is intentionally not part of the long-running process
            # monitor list.
            smoke_env = os.environ.copy()
            smoke_env.update({
                "LUMEN_CERT_POINTER": str(cert_authority.pointer_path.resolve()),
                "LUMEN_CERT_REVOCATIONS": str(cert_authority.revocation_path.resolve()),
                "LUMEN_ZENOH_ENDPOINT": ZENOH_ENDPOINT,
                "LUMEN_CERT_IDENTITY": "ids",
                "LUMEN_CERT_ALLOWED_IDENTITIES": "ids,gas_sensor_01,gas_sensor_02",
                "LUMEN_ENV": "production",
                "LUMEN_BOOT_REGISTRATION_TIMEOUT_SEC": os.environ.get("LUMEN_BOOT_REGISTRATION_TIMEOUT_SEC", "60"),
                "LUMEN_BOOT_REGISTRATION_RETRY_SEC": os.environ.get("LUMEN_BOOT_REGISTRATION_RETRY_SEC", "0.25"),
            })
            log.warning("[LIVE SMOKE] running real mTLS/Zenoh envelope + PEP ACK verification")
            smoke_timeout = float(os.environ.get("LUMEN_LIVE_SMOKE_PROCESS_TIMEOUT_SEC", "180"))
            completed = subprocess.run(
                [sys.executable, str(project_dir / "scripts" / "live_runtime_smoke.py")],
                cwd=str(project_dir),
                env=smoke_env,
                timeout=smoke_timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"live mTLS/Zenoh smoke failed with exit code {completed.returncode}"
                )
            log.info("[LIVE SMOKE] PASS: shared-topic identities and correlated PEP ACKs verified")

        log.info("Lumen Edge Security running. Press Ctrl+C to shut down.")
        while not stop.wait(0.5):
            dead = [(title, p.returncode) for title, p in processes if p.poll() is not None]
            for title, code in dead:
                log.error("component exited title=%s code=%s", title, code)
                stop.set()
                break
            if not router_thread.is_alive():
                log.critical("router supervisor stopped; shutting down the stack")
                stop.set()
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        supervisor.stop()
        # Stop clients first; Key Authority is last so it can observe revocation
        # state while consumers are shutting down. The orchestrator owns cleanup.
        for title, process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        router_thread.join(timeout=6.0)
        key_authority.cleanup()  # wipes CA + leaf certs + Ed25519 session keys
        log.info("Lumen session cleaned up")
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_and_arrange_grid())