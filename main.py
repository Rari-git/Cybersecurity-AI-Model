"""Legacy interactive menu.

The secure live topology is orchestrated by run_all.py.  This menu remains as
an entrypoint for offline training/evaluation and guarded one-off tools, but it
must not start live Zenoh clients without the managed mTLS session variables.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _managed_session_ready(required_identity: str | None = None) -> bool:
    """Fail closed when a live component is started outside run_all's PKI env."""
    pointer = os.environ.get("LUMEN_CERT_POINTER", "").strip()
    ledger = os.environ.get("LUMEN_CERT_REVOCATIONS", "").strip()
    endpoint = os.environ.get("LUMEN_ZENOH_ENDPOINT", "").strip()
    identity = os.environ.get("LUMEN_CERT_IDENTITY", "").strip()
    missing = []
    if not pointer or not Path(pointer).is_file():
        missing.append("LUMEN_CERT_POINTER (active PKI pointer)")
    if not ledger or not Path(ledger).is_file():
        missing.append("LUMEN_CERT_REVOCATIONS (certificate revocation ledger)")
    if not endpoint.startswith("tls/"):
        missing.append("LUMEN_ZENOH_ENDPOINT (tls/...) ")
    if required_identity and identity != required_identity:
        missing.append(f"LUMEN_CERT_IDENTITY={required_identity}")
    if missing:
        print("Live component launch refused; missing managed mTLS context:")
        for item in missing:
            print(f"  - {item}")
        print("Start the complete secured topology with: python run_all.py")
        return False
    return True


def main() -> int:
    print("==================================================")
    print(" 🛡️  LUMEN EDGE SECURITY - V5.0 🛡️")
    print("==================================================")
    print("1. Antrenează Modelul Autoencoder AI (PyTorch & ONNX)")
    print("2. Pornește stack-ul live securizat (Router + PKI + Gateway + IDS + Sensor)")
    print("3. Evaluează bundle-ul V5 activ (semnătură Ed25519 + benchmark)")
    print("4. Pornește Dashboard-ul UI Streamlit")
    print("5. IDS standalone (necesită sesiune mTLS deja provisionată)")
    print("6. Simulator senzori mTLS (necesită sesiune mTLS deja provisionată)")
    print("7. Attack 5 (necesită sesiune mTLS deja provisionată)")
    print("0. Ieșire")
    print("==================================================")

    choice = input("Alege o opțiune (0-7): ").strip()

    if choice == "1":
        return subprocess.call([sys.executable, "-m", "training.train_autoencoder"], cwd=ROOT)
    if choice == "2":
        return subprocess.call([sys.executable, "run_all.py"], cwd=ROOT)
    if choice == "3":
        return subprocess.call([sys.executable, "-m", "training.evaluate_v5"], cwd=ROOT)
    if choice == "4":
        if not _managed_session_ready("dashboard"):
            return 2
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "ui/app.py"], cwd=ROOT)
    if choice == "5":
        if not _managed_session_ready("ids"):
            return 2
        return subprocess.call([sys.executable, "-m", "agent.ids_agent"], cwd=ROOT)
    if choice == "6":
        if not _managed_session_ready("sensors"):
            return 2
        if not os.environ.get("LUMEN_CERT_DEVICE_IDENTITIES"):
            print("Live sensor launch refused: LUMEN_CERT_DEVICE_IDENTITIES is not configured.")
            print("Start the complete secured topology with: python run_all.py")
            return 2
        return subprocess.call([sys.executable, "-m", "simulation.virtual_sensors_devices"], cwd=ROOT)
    if choice == "7":
        if not _managed_session_ready():
            return 2
        return subprocess.call([sys.executable, "-m", "attacks.attack_5_device_compromise"], cwd=ROOT)
    if choice == "0":
        print("Ieșire.")
        return 0
    print("Opțiune invalidă!")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
