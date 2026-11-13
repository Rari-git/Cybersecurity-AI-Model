"""Windows/cross-platform attack launcher inheriting ephemeral Patch 03.1 + 04 session env."""
import os
import subprocess
import sys
import time


def launch_attacks_menu():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("==================================================")
    print(" LUMEN EDGE SECURITY - ATTACK LAUNCHER")
    print("==================================================")
    print("1. Device-attributed DoS / per-device quarantine")
    print("2. HMAC Replay / Tampering")
    print("3. Reconnaissance / Honeypot Scan")
    print("4. Launch attacks 1-3 in parallel (not for device metrics validation)")
    print("5. Single-device compromise / per-device quarantine")
    print("0. Exit")
    print("==================================================")
    choice = input("Choose (0-5): ").strip()

    def run_cmd(title, module):
        env = os.environ.copy()
        env.setdefault("LUMEN_ZENOH_ENDPOINT", "tls/127.0.0.1:7447")
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=project_dir,
            env=env,
            creationflags=flags,
        )
        print(f"[+] Started: {title}")

    if choice == "1":
        run_cmd("Attack 1 - Device DoS", "attacks.attack_1_dos")
    elif choice == "2":
        run_cmd("Attack 2 - HMAC Replay", "attacks.attack_2_replay")
    elif choice == "3":
        run_cmd("Attack 3 - Honeypot Scan", "attacks.attack_3_scan")
    elif choice == "4":
        run_cmd("Attack 1 - Device DoS", "attacks.attack_1_dos")
        time.sleep(1)
        run_cmd("Attack 2 - HMAC Replay", "attacks.attack_2_replay")
        time.sleep(1)
        run_cmd("Attack 3 - Honeypot Scan", "attacks.attack_3_scan")
    elif choice == "5":
        run_cmd("Attack 5 - Device Compromise", "attacks.attack_5_device_compromise")
    elif choice == "0":
        print("Exit.")
    else:
        print("Invalid option.")


if __name__ == "__main__":
    launch_attacks_menu()