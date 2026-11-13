"""Archive stale Lumen logs before a clean validation run; never deletes replay state."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(os.environ.get("LUMEN_LOG_DIR", "logs"))
FILES = (
    "gateway_ids.log",
    "gateway_metrics.jsonl",
    "ids_agent.log",
    "enforcement.log",
    "honeypot.log",
    "attack_1_dos.log",
    "attack_2_replay.log",
    "attack_3_scan.log",
    "attack_4_persistent_replay.log",
)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = LOG_DIR / "archive" / f"validation_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    # Archive the known names and any manually suffixed copies such as
    # gateway_ids_2.log / attack_1_dos_2.log from previous consolidated runs.
    candidates = {LOG_DIR / name for name in FILES}
    candidates.update(LOG_DIR.glob("*.log"))
    candidates.update(LOG_DIR.glob("*.jsonl"))
    for source in sorted(candidates):
        if source.is_file():
            destination = archive / source.name
            shutil.move(str(source), str(destination))
            moved.append(source.name)
    manifest = {
        "prepared_at": datetime.now().astimezone().isoformat(),
        "archived_files": moved,
        "note": "Replay SQLite databases under state/ were intentionally not touched.",
    }
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    active = {
        "prepared_at": manifest["prepared_at"],
        "archive_dir": str(archive),
        "archived_files": moved,
    }
    (LOG_DIR / "validation_run.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
    print(f"Archived {len(moved)} old log file(s) to: {archive}")
    print("Replay state under state/ was preserved.")
    print("Start run_all.py now. Confirm the new gateway log contains [GATEWAY BUILD] patch02-completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
