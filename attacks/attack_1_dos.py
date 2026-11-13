"""Patch 03 Attack 1 compatibility entrypoint: device-attributed bounded DoS."""
from attacks.attack_5_device_compromise import main


if __name__ == "__main__":
    raise SystemExit(main())
