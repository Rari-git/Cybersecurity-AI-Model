import os
import sys

def main():
    print("==================================================")
    print(" 🛡️  CYBERSECURITY AI LAB - ZENOH IDS ROUTER 🛡️")
    print("==================================================")
    print("1. Antrenează Modelul Autoencoder AI (PyTorch & ONNX)")
    print("2. Pornește Generatorul de Senzori Virtuali (legacy)")
    print("3. Pornește Agentul AI IDS (Inspecție în timp real)")
    print("4. Pornește Dashboard-ul UI Streamlit")
    print("5. Execută Atac 1 (DoS / Topic Flooding)")
    print("6. Execută Atac 2 (Replay Attack)")
    print("7. Execută Atac 3 (Reconnaissance / Scanning)")
    print("8. Pornește simularea Patch 03 cu identități mTLS separate")
    print("9. Execută Attack 5 (Device Compromise / per-device quarantine)")
    print("0. Ieșire")
    print("==================================================")
    
    choice = input("Alege o opțiune (0-9): ").strip()
    
    if choice == "1":
        os.system("python -m training.train_autoencoder")
    elif choice == "2":
        os.system("python -m simulation.virtual_sensors")
    elif choice == "3":
        os.system("python -m agent.ids_agent")
    elif choice == "8":
        os.system("python -m simulation.virtual_sensors_devices")
    elif choice == "9":
        os.system("python -m attacks.attack_5_device_compromise")
    elif choice == "4":
        os.system("streamlit run ui/app.py")
    elif choice == "5":
        os.system("python -m attacks.attack_1_dos")
    elif choice == "6":
        os.system("python -m attacks.attack_2_replay")
    elif choice == "7":
        os.system("python -m attacks.attack_3_scan")
    elif choice == "0":
        print("Ieșire.")
    else:
        print("Opțiune invalidă!")

if __name__ == "__main__":
    main()
