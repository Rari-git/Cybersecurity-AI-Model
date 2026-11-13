import json
import math
import os
import sys
import time
from queue import Queue, Full, Empty

# `streamlit run ui/app.py` pune în sys.path doar ui/, nu rădăcina proiectului.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.mtls_hot_reload import open_mtls_client

st.set_page_config(page_title="Cybersecurity AI Lab - Zenoh IDS", page_icon="🛡️", layout="wide")
st.title("🛡️ Cybersecurity AI Lab - Real-Time IDS Dashboard")
st.markdown("Monitorizarea traficului pe magistrala **Zenoh** și detectarea atacurilor cu **Temporal LSTM Autoencoder ONNX** + verificare criptografică.")

META_PATH = "models/saved/model_metadata.json"
SENTINEL_LOSS = 50.0   # alertele euristice (crypto/rată/honeypot) folosesc loss=99.0, nu sunt scoruri ML
DASHBOARD_IDENTITY = os.environ.get("LUMEN_CERT_IDENTITY", "dashboard")
ZENOH_ENDPOINT = os.environ.get("LUMEN_ZENOH_ENDPOINT", "tls/127.0.0.1:7447")

MAX_HISTORY = 500
MAX_BLOCK_TTL = 3600.0
ML_CLUSTER_GAP_SEC = 30.0   # două alerte ML la mai puțin de atât fac parte din același episod
ML_CLUSTER_MIN_ALERTS = 3   # episod "contaminat" = cel puțin atâtea alerte ML consecutive
SEVERITY_COLORS = {"critical": "#FF4B4B", "normal": "#4B8BFF"}
FALLBACK_COLORS = ["#FFB84B", "#9B59B6", "#2ECC71", "#95A5A6"]


@st.cache_resource
def _open_zenoh_listener():
    """Ridică excepție la eșec, ca un eșec de conexiune să NU fie păstrat în cache."""
    alert_queue = Queue(maxsize=5000)
    session = open_mtls_client(DASHBOARD_IDENTITY)

    def alert_handler(sample):
        try:
            alert = json.loads(sample.payload.to_bytes().decode("utf-8"))
            if isinstance(alert, dict):
                alert["_rx"] = time.time()   # momentul recepției (ceasul dashboard-ului), pentru countdown TTL
                alert_queue.put_nowait(alert)
        except (Full, ValueError, UnicodeDecodeError):
            pass

    sub = session.declare_subscriber("home/security/alerts", alert_handler)
    return session, sub, alert_queue


def init_zenoh_listener():
    try:
        return _open_zenoh_listener()
    except Exception as e:
        st.error(f"Eroare Conexiune Zenoh ({ZENOH_ENDPOINT}): {e}")
        return None, None, Queue(maxsize=5000)


def load_threshold():
    try:
        with open(META_PATH) as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return None


session, sub, alert_queue = init_zenoh_listener()

if "alerts_history" not in st.session_state:
    st.session_state.alerts_history = []
if "blocks" not in st.session_state:
    # topic -> {"expires": epoch local, "ttl": float, "attack": str, "source": str}
    st.session_state.blocks = {}


# ---------------------------------------------------------------- Blocklist (reconstruit din alerte BLOCK)
def _apply_block(alert: dict):
    """Reconstruct topic- or device-level TTL blocks from gateway alerts."""
    if str(alert.get("action", "")).upper() != "BLOCK":
        return
    try:
        ttl = float(alert.get("block_ttl", 0))
        rx = float(alert.get("_rx", time.time()))
    except (TypeError, ValueError):
        return
    if not math.isfinite(ttl) or not math.isfinite(rx) or ttl <= 0:
        return
    ttl = min(ttl, MAX_BLOCK_TTL)
    entity_type = str(alert.get("entity_type", "topic"))[:20]
    device_id = str(alert.get("device_id", ""))[:64]
    topic = str(alert.get("topic", "?"))[:80]
    entity_id = (
        str(alert.get("entity_id") or f"device:{device_id}")[:100]
        if entity_type == "device" and device_id else topic
    )
    expires = rx + ttl
    prev = st.session_state.blocks.get(entity_id)
    if prev is not None:
        expires = max(expires, prev["expires"])
    st.session_state.blocks[entity_id] = {
        "expires": expires, "ttl": ttl,
        "attack": str(alert.get("attack", "?"))[:60],
        "source": str(alert.get("source", "?"))[:40],
        "entity_type": entity_type, "device_id": device_id, "topic": topic,
    }


def _clear_blocklist():
    st.session_state.blocks = {}


# ---------------------------------------------------------------- Pregătire date
def _prepare_df(history: list) -> pd.DataFrame:
    df = pd.DataFrame(history)
    if df.empty:
        return df
    for col, default in (("attack", "?"), ("topic", "?"), ("action", "?"),
                         ("severity", "normal"), ("source", "?"), ("detail", ""),
                         ("entity_type", "topic"), ("device_id", "")):
        df[col] = df[col].fillna(default) if col in df.columns else default
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce") if "loss" in df.columns else float("nan")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") if "timestamp" in df.columns else float("nan")
    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    df["severity"] = df["severity"].astype(str).str.lower()
    src = df["source"].astype(str)
    honey = src.str.contains("honeypot", case=False, na=False) | (df["attack"] == "Reconnaissance / Scanning")
    df["origin"] = src.where(~honey, "Honeypot")
    return df


def _contaminated_zones(ml: pd.DataFrame):
    """Episoade în care alertele ML vin în rafală (gap <= ML_CLUSTER_GAP_SEC, >= ML_CLUSTER_MIN_ALERTS)."""
    zones, cur = [], []

    def flush():
        if len(cur) >= ML_CLUSTER_MIN_ALERTS:
            zones.append((cur[0], cur[-1]))

    for t in ml["time"].tolist():
        if cur and (t - cur[-1]).total_seconds() > ML_CLUSTER_GAP_SEC:
            flush()
            cur = []
        cur.append(t)
    flush()
    return zones


# ---------------------------------------------------------------- Grafice
def _stacked_bar(df: pd.DataFrame, x_col: str, title: str, x_title: str, key: str):
    counts = df.groupby([x_col, "severity"]).size().unstack(fill_value=0)
    fig = go.Figure()
    extra = iter(FALLBACK_COLORS)
    ordered = [s for s in ("critical", "normal") if s in counts.columns] + \
              [s for s in counts.columns if s not in ("critical", "normal")]
    for sev in ordered:
        fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts[sev].tolist(), name=sev.capitalize(),
                             marker_color=SEVERITY_COLORS.get(sev) or next(extra, "#999999")))
    fig.update_layout(barmode="stack", title=title, template="plotly_dark", height=340,
                      xaxis_title=x_title, yaxis_title="Nr. alerte", legend_title="Severitate")
    st.plotly_chart(fig, use_container_width=True, key=key)


def _ml_timeline(df: pd.DataFrame):
    ml = df[df["loss"] < SENTINEL_LOSS].sort_values("time")
    if ml.empty:
        st.info("Nicio alertă ML (reconstruction loss) înregistrată încă.")
        return
    fig = go.Figure()
    for x0, x1 in _contaminated_zones(ml):
        pad = pd.Timedelta(seconds=2)
        fig.add_shape(type="rect", x0=x0 - pad, x1=x1 + pad, y0=0, y1=1, yref="paper",
                      fillcolor="rgba(255,75,75,0.18)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=ml["time"], y=ml["loss"], mode="lines+markers",
                             name="Reconstruction loss (alerte ML)", line=dict(color="#FF4B4B", width=2)))
    blocked = ml[ml["action"].astype(str).str.upper() == "BLOCK"]
    if not blocked.empty:
        fig.add_trace(go.Scatter(x=blocked["time"], y=blocked["loss"], mode="markers", name="BLOCK emis",
                                 marker=dict(symbol="x", size=12, color="#FFFFFF",
                                             line=dict(width=2, color="#FF4B4B"))))
    thr = load_threshold()
    if thr is not None:
        fig.add_hline(y=thr, line_dash="dash", annotation_text="Prag global")
    fig.update_layout(title="Timeline anomalii ML: Loss vs Threshold (zonele roșii = episoade contaminate)",
                      template="plotly_dark", height=340, xaxis_title="Timp", yaxis_title="Reconstruction Loss")
    st.plotly_chart(fig, use_container_width=True, key="chart_ml_timeline")


# ---------------------------------------------------------------- Dashboard
@st.fragment(run_every=1.0)
def render_dashboard_fragment():
    drained = []
    while True:
        try:
            drained.append(alert_queue.get_nowait())
        except Empty:
            break
    for a in drained:
        _apply_block(a)
    if drained:
        st.session_state.alerts_history = (list(reversed(drained)) + st.session_state.alerts_history)[:MAX_HISTORY]

    now = time.time()
    st.session_state.blocks = {k: v for k, v in st.session_state.blocks.items() if v["expires"] > now}
    blocks = st.session_state.blocks

    df = _prepare_df(st.session_state.alerts_history)

    st.sidebar.header("Status Edge System")
    if session is not None:
        st.sidebar.success(f"🟢 Broker Zenoh: {ZENOH_ENDPOINT}")
    else:
        st.sidebar.error(f"🔴 Broker Zenoh indisponibil: {ZENOH_ENDPOINT}")
    st.sidebar.info("🤖 Model: LSTM Autoencoder ONNX")
    st.sidebar.warning(f"📊 Alerte Înregistrate: {len(df)}")
    st.sidebar.error(f"⛔ Entități / Dispozitive Blocate Activ: {len(blocks)}")

    def count(name):
        return int((df["attack"] == name).sum()) if not df.empty else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("DoS Mitigate", count("DoS / Topic Flooding"))
    c2.metric("Replay Blocat", count("Replay Attack"))
    c3.metric("Comenzi Falsificate", count("Command Tampering"))
    c4.metric("Scanări (Honeypot)", count("Reconnaissance / Scanning"))
    c5.metric("Payload Anormal (ML)", count("Anomalous Payload"))
    c6.metric("Blocări Active", len(blocks))

    st.markdown("---")

    tab_enf, tab_analytics, tab_log = st.tabs(
        ["⛔ Active Enforcement & Blocklist", "📈 Analytics", "🗂️ Jurnal Alerte"])

    with tab_enf:
        left, right = st.columns([3, 1])
        left.subheader("Entități aflate în stare BLOCK (TTL activ)")
        right.button("🧹 Clear Active Blocklist", on_click=_clear_blocklist, key="clear_blocklist",
                     help="Doar pentru laborator/demo: golește lista afișată în acest dashboard.")
        if blocks:
            max_ttl = max(v["ttl"] for v in blocks.values())
            rows = [{"Entitate": k, "Tip entitate": v.get("entity_type", "topic"),
                     "Device ID": v.get("device_id", ""), "Topic": v.get("topic", k),
                     "Atac": v["attack"], "Sursă": v["source"],
                     "TTL total (s)": round(v["ttl"], 1),
                     "Rămas (s)": round(max(0.0, v["expires"] - now), 1)}
                    for k, v in blocks.items()]
            bdf = pd.DataFrame(rows).sort_values("Rămas (s)", ascending=False)
            st.dataframe(
                bdf, use_container_width=True, hide_index=True,
                column_config={"Rămas (s)": st.column_config.ProgressColumn(
                    "Rămas (s)", min_value=0.0, max_value=float(max(max_ttl, 1.0)), format="%.0f s")})
        else:
            st.success("Nicio entitate blocată în acest moment.")
        st.caption("Lista este reconstruită din alertele `action=BLOCK` primite pe `home/security/alerts`. "
                   "Blocările de dispozitiv sunt independente chiar și când două dispozitive împart același topic. "
                   "Identitatea `dashboard` este read-only; resetarea golește doar vizualizarea locală, "
                   "iar blocarea reală expiră în gateway la finalul TTL-ului.")

    with tab_analytics:
        if df.empty:
            st.info("Nicio alertă detectată până în prezent. Trafic normal pe magistrală.")
        else:
            a1, a2 = st.columns(2)
            with a1:
                _stacked_bar(df, "attack", "Distribuția atacurilor pe severitate", "Tip atac", "chart_attack_sev")
            with a2:
                _stacked_bar(df, "origin", "Alerte pe sursă (Inline Gateway / Passive IDS / Honeypot)",
                             "Sursă", "chart_origin_sev")
            _ml_timeline(df)

    with tab_log:
        if not df.empty:
            cols = [c for c in ("time", "attack", "entity_type", "device_id", "topic", "action",
                                "severity", "repeats", "loss", "block_ttl", "detail", "source")
                    if c in df.columns]
            st.dataframe(df[cols], use_container_width=True)
        else:
            st.info("Nicio alertă detectată până în prezent. Trafic normal pe magistrală.")


render_dashboard_fragment()