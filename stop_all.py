import os
import subprocess
import sys


def stop_lumen_stack():
    print("🛑 Se opresc toate procesele Lumen Edge Security...\n")

    # 1. Oprire procese Python legate de proiect
    python_scripts = [
        "agent.gateway_ids",
        "agent.ids_agent",
        "agent.ids_agent_2",
        "agent.ids_agent_3",
        "agent.honeypot",
        "agent.honeypot_2",
        "simulation.virtual_sensors",
        "simulation.virtual_sensors_devices",
        "attacks.attack_5_device_compromise",
        "app.py",
        "run_attacks.py",
        "attack_1_dos",
        "attack_2_replay",
        "attack_3_scan"
    ]

    for script in python_scripts:
        cmd = f"wmic process where \"commandline like '%{script}%' and name='python.exe'\" call terminate"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [-] Închis script Python: {script}")

    # 2. Oprire executabil router Zenoh
    print("  [-] Închidere router zenohd.exe...")
    subprocess.run("taskkill /F /IM zenohd.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Curățare ferestre de PowerShell lansate cu titlurile din grid
    window_titles = [
        "Zenoh Router",
        "Inline Gateway",
        "IDS Agent AI",
        "Honeypot Sink",
        "Virtual Sensors",
        "Dashboard Streamlit",
        "Meniu Atacuri"
    ]

    for title in window_titles:
        cmd = f'taskkill /FI "WINDOWTITLE eq {title}*" /F'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("\n✅ Toate procesele și resursele au fost eliberate cu succes!")


if __name__ == "__main__":
    stop_lumen_stack()
