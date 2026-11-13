"""Etapa E7: Componentă minimală de Topologie Live & XAI Inspector pentru Streamlit."""
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from storage.telemetry_db import LumenStorageManager, DBPaths

STATE_COLORS: dict[str, str] = {
    "ACTIVE": "#2ecc71",
    "MONITORING": "#3498db",
    "ENFORCEMENT_PENDING": "#f1c40f",
    "SHADOW_QUARANTINE": "#e67e22",
    "REVOKED": "#e74c3c",
    "FAULT": "#9b59b6",
    "OFFLINE": "#7f8c8d",
    "ENFORCEMENT_FAILED": "#c0392b",
}


def render_network_topology(db_dir: str | Path = "logs/db") -> None:
    st.subheader("Lumen Zero-Trust Live Topology & PEP State")
    paths = DBPaths.from_dir(db_dir)
    if not paths.control_db.is_file():
        st.info("Baza de date security_control.db nu a fost încă inițializată.")
        return

    storage = LumenStorageManager(paths=paths, enable_writers=False)
    conn = storage.connect_readonly("control")
    try:
        devices = conn.execute(
            "SELECT device_id, generation, device_role, criticality, current_state FROM device_registry ORDER BY device_id"
        ).fetchall()
        audits = conn.execute(
            "SELECT decision_id, device_id, old_state, target_state, pep_status, enforcement_latency_ms, detail "
            "FROM enforcement_audit ORDER BY audit_id DESC LIMIT 10"
        ).fetchall()
    finally:
        conn.close()

    # Generăm graful Graphviz cu stările confirmate de PEP
    lines = [
        "digraph LumenTopology {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="white"];',
        '  zenohd [label="Eclipse Zenoh Router\\n(mTLS + ACL)", fillcolor="#1f2937"];',
        '  pep [label="Inline Gateway (PEP)\\nPolicy Enforcement", fillcolor="#111827"];',
        '  sink [label="Forensic Shadow Sink\\n(shadow/sandbox/**)", fillcolor="#7c2d12"];',
        "  pep -> zenohd [label=\"home/**\"];",
        "  pep -> sink [label=\"quarantined\", style=dashed, color=\"#e67e22\"];",
    ]

    for d in devices:
        dev_id = d["device_id"]
        state = d["current_state"]
        color = STATE_COLORS.get(state, "#34495e")
        crit = " [CRITICAL]" if d["criticality"] == "safety_critical" else ""
        lines.append(f'  "{dev_id}" [label="{dev_id} (g{d["generation"]}){crit}\\n[{state}]", fillcolor="{color}"];')
        if state == "SHADOW_QUARANTINE":
            lines.append(f'  "{dev_id}" -> pep [color="#e67e22", penwidth=2.0];')
        elif state == "REVOKED":
            lines.append(f'  "{dev_id}" -> pep [color="#e74c3c", style=dotted];')
        else:
            lines.append(f'  "{dev_id}" -> pep [color="#2ecc71"];')

    lines.append("}")
    st.graphviz_chart("\n".join(lines))

    if audits:
        st.markdown("**Ultimele confirmări PEP (Enforcement Audit)**")
        st.dataframe([dict(r) for r in audits], use_container_width=True)