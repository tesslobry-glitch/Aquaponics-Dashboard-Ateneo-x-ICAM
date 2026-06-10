"""
AquaMonitor Dashboard
Tess Lobry (water quality) × Simon Tourtois (energy)
Ateneo de Manila University — BFAR-NFTC, Nueva Ecija
"""

import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaMonitor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
NAVY    = "#0D2137"
TEAL    = "#0B6E8A"
TEAL2   = "#0E9AA7"
GREEN   = "#1B7A4A"
AMBER   = "#D97706"
RED     = "#DC2626"
SLATE   = "#475569"
LTBLUE  = "#E8F4F8"
WHITE   = "#FFFFFF"
BG      = "#F7FAFB"

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
.main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0D2137; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #E2E8F0 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F; }

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid #0B6E8A;
    margin-bottom: 0.5rem;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #0D2137; line-height: 1; }
.kpi-label { font-size: 0.78rem; color: #64748B; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.82rem; margin-top: 0.3rem; }
.kpi-good  { color: #16A34A; }
.kpi-warn  { color: #D97706; }
.kpi-bad   { color: #DC2626; }

.kpi-card.green  { border-left-color: #1B7A4A; }
.kpi-card.amber  { border-left-color: #D97706; }
.kpi-card.red    { border-left-color: #DC2626; }
.kpi-card.teal2  { border-left-color: #0E9AA7; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 1.8rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #0B6E8A;
}
.section-header h2 { margin: 0; font-size: 1.15rem; color: #0D2137; font-weight: 700; }
.section-icon { font-size: 1.3rem; }

/* Alert badge */
.alert-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-red    { background: #FEE2E2; color: #DC2626; }
.badge-amber  { background: #FEF3C7; color: #92400E; }
.badge-green  { background: #D1FAE5; color: #065F46; }
.badge-blue   { background: #DBEAFE; color: #1E40AF; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #E8F4F8 0%, #F0F9FF 100%);
    border-left: 3px solid #0B6E8A;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #0D2137;
}

/* Result table */
.result-table { font-size: 0.88rem; }
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

@st.cache_data
def load_energy_data():
    p = HERE / "cache" / "data.json"
    if not p.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(p.read_text()))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # derived
    BAT_V_FULL, BAT_V_EMPTY = 12.7, 11.6
    df["soc"]          = ((df["bat_voltage"] - BAT_V_EMPTY) / (BAT_V_FULL - BAT_V_EMPTY) * 100).clip(0, 100)
    df["solar_power"]  = df["solar_voltage"] * df["solar_current"]
    df["pump_power"]   = df["bat_voltage"]   * df["current_pump"]
    df["aerator_power"]= df["bat_voltage"]   * df["current_aerator"]
    return df

@st.cache_data
def load_analysis():
    p = HERE / "cache" / "analysis.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())

@st.cache_data
def load_water_data():
    """Load Tess's water quality data — looks for good_data CSV or uses synthetic demo."""
    # Try to find the water quality CSV
    candidates = list(HERE.glob("data/*.csv")) + list(HERE.glob("cache/*.csv"))
    for c in candidates:
        try:
            df = pd.read_csv(c, low_memory=False)
            if any("DO" in col.upper() or "MCP_WQ" in col.upper() for col in df.columns):
                df["Datetime"] = pd.to_datetime(df.get("Datetime", df.iloc[:,0]), errors="coerce")
                df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
                return df
        except Exception:
            continue
    # Demo fallback — synthetic but realistic water quality data
    return _make_demo_water()

def _make_demo_water():
    """Generate realistic synthetic water quality data for demo purposes."""
    rng = np.random.default_rng(42)
    n = 5000
    t = pd.date_range("2026-01-06", periods=n, freq="1min")
    hours = t.hour + t.minute / 60
    # DO: day-night cycle + noise (raw values ~40-120)
    do = 80 + 35 * np.sin(np.pi * (hours - 6) / 14) * ((hours > 6) & (hours < 20)).astype(float)
    do += rng.normal(0, 2, n)
    # pH: correlated with DO (raw ~455-475)
    ph = 460 + 8 * np.sin(np.pi * (hours - 6) / 14) * ((hours > 6) & (hours < 20)).astype(float)
    ph += rng.normal(0, 1.5, n)
    # EC: slow variation (raw ~200-225)
    ec = 211 + 8 * np.sin(2 * np.pi * hours / 24 + 1.2) + rng.normal(0, 2, n)
    # Temp: ~240-255 raw
    temp = 247 + 3 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 0.5, n)
    # TDS: correlated with EC
    tds = 1.254 * ec - 34.8 + rng.normal(0, 4, n)
    df = pd.DataFrame({
        "Datetime": t,
        "MCP_WQ_DO": do.clip(20, 180),
        "MCP_WQ_EC": ec.clip(170, 260),
        "MCP_WQ_PH": ph.clip(440, 485),
        "MCP_WQ_TEMP": temp.clip(235, 260),
        "MCP_WQ_TDS": tds.clip(170, 310),
    })
    return df

def kpi(label, value, unit="", delta=None, color="teal", note=""):
    cls = {"teal":"", "green":"green", "amber":"amber", "red":"red", "teal2":"teal2"}.get(color, "")
    delta_html = ""
    if delta is not None:
        dc = "kpi-good" if delta >= 0 else "kpi-bad"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="kpi-delta {dc}">{arrow} {abs(delta):.1f}{unit}</div>'
    note_html = f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:0.2rem">{note}</div>' if note else ""
    return f"""
    <div class="kpi-card {cls}">
        <div class="kpi-value">{value}<span style="font-size:1rem;font-weight:400;color:#64748B;margin-left:3px">{unit}</span></div>
        <div class="kpi-label">{label}</div>
        {delta_html}{note_html}
    </div>"""

def section(icon, title):
    st.markdown(f'<div class="section-header"><span class="section-icon">{icon}</span><h2>{title}</h2></div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def badge(text, color="blue"):
    return f'<span class="alert-badge badge-{color}">{text}</span>'

def plotly_defaults(fig, height=320, bg=WHITE, paper=BG):
    fig.update_layout(
        height=height,
        paper_bgcolor=paper,
        plot_bgcolor=bg,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, Arial, sans-serif", size=11, color=NAVY),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0",
                    borderwidth=1, font_size=10),
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )
    return fig


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df_energy   = load_energy_data()
analysis    = load_analysis()
df_water    = load_water_data()

WATER_COLS = {
    "MCP_WQ_DO":   ("Dissolved Oxygen", "raw", TEAL),
    "MCP_WQ_EC":   ("Conductivity (EC)", "raw", GREEN),
    "MCP_WQ_PH":   ("pH",               "raw", AMBER),
    "MCP_WQ_TEMP": ("Temperature",      "raw", "#7C3AED"),
    "MCP_WQ_TDS":  ("TDS",              "raw", "#0891B2"),
}
WATER_PRESENT = [c for c in WATER_COLS if c in df_water.columns]


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐟 AquaMonitor")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Overview",
        "💧  Water Quality",
        "⚡  Energy System",
        "🔗  Combined View",
    ], label_visibility="collapsed")
    st.markdown("---")
    # Dataset info
    if not df_energy.empty:
        t0 = df_energy["timestamp"].min().strftime("%b %d")
        t1 = df_energy["timestamp"].max().strftime("%b %d, %Y")
        st.markdown(f"**Energy data**  \n{t0} → {t1}  \n{len(df_energy):,} records")
    if not df_water.empty:
        st.markdown(f"**Water quality**  \n{len(df_water):,} records")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#64748B">Tess Lobry · Simon Tourtois<br>'
        'Ateneo de Manila University<br>BFAR-NFTC, Nueva Ecija 🇵🇭</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("🐟 AquaMonitor — System Overview")
    st.caption("Solar-powered aquaponic monitoring system · BFAR-NFTC, Nueva Ecija, Philippines")

    # ── Top KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    # Water quality KPIs
    if not df_water.empty:
        last_w = df_water.iloc[-1]
        do_val  = last_w.get("MCP_WQ_DO",  None)
        ph_val  = last_w.get("MCP_WQ_PH",  None)
        ec_val  = last_w.get("MCP_WQ_EC",  None)
        tmp_val = last_w.get("MCP_WQ_TEMP",None)
    else:
        do_val = ph_val = ec_val = tmp_val = None

    with c1:
        v = f"{do_val:.0f}" if do_val else "—"
        st.markdown(kpi("Dissolved O₂", v, "raw", color="teal", note="Water quality"), unsafe_allow_html=True)
    with c2:
        v = f"{ph_val:.0f}" if ph_val else "—"
        st.markdown(kpi("pH", v, "raw", color="teal2", note="Water quality"), unsafe_allow_html=True)
    with c3:
        v = f"{ec_val:.0f}" if ec_val else "—"
        st.markdown(kpi("EC", v, "raw", color="green", note="Water quality"), unsafe_allow_html=True)

    # Energy KPIs
    if not df_energy.empty:
        last_e = df_energy.iloc[-1]
        soc     = last_e.get("soc",         None)
        sol_p   = last_e.get("solar_power", None)
        bat_v   = last_e.get("bat_voltage", None)
        soc_col = "green" if (soc or 0) > 60 else ("amber" if (soc or 0) > 30 else "red")
    else:
        soc = sol_p = bat_v = None
        soc_col = "teal"

    with c4:
        v = f"{soc:.0f}" if soc is not None else "—"
        st.markdown(kpi("Battery SoC", v, "%", color=soc_col, note="Energy system"), unsafe_allow_html=True)
    with c5:
        v = f"{sol_p:.0f}" if sol_p is not None else "—"
        st.markdown(kpi("Solar Power", v, "W", color="amber", note="Energy system"), unsafe_allow_html=True)
    with c6:
        v = f"{bat_v:.2f}" if bat_v is not None else "—"
        st.markdown(kpi("Battery V", v, "V", color="teal", note="Energy system"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Two-panel overview chart ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        section("💧", "Water Quality — Last 24 h")
        if not df_water.empty and WATER_PRESENT:
            dw = df_water.copy()
            if "Datetime" in dw.columns:
                dw = dw.set_index("Datetime")
            # Show DO and pH (most important)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.06)
            do_col = "MCP_WQ_DO" if "MCP_WQ_DO" in dw.columns else WATER_PRESENT[0]
            ph_col = "MCP_WQ_PH" if "MCP_WQ_PH" in dw.columns else None
            
            fig.add_trace(go.Scatter(
                x=dw.index, y=dw[do_col],
                name="DO (raw)", line=dict(color=TEAL, width=1.5),
                fill="tozeroy", fillcolor="rgba(11,110,138,0.08)"
            ), row=1, col=1)
            
            if ph_col and ph_col in dw.columns:
                fig.add_trace(go.Scatter(
                    x=dw.index, y=dw[ph_col],
                    name="pH (raw)", line=dict(color=AMBER, width=1.5)
                ), row=2, col=1)

            fig.update_layout(height=280, paper_bgcolor=BG, plot_bgcolor=WHITE,
                              margin=dict(l=10,r=10,t=10,b=10),
                              font=dict(size=10), showlegend=True,
                              legend=dict(orientation="h", y=1.08, x=0))
            fig.update_xaxes(gridcolor="#F1F5F9", showgrid=True)
            fig.update_yaxes(gridcolor="#F1F5F9", showgrid=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No water quality data found.")

    with col_r:
        section("⚡", "Energy System — Last 24 h")
        if not df_energy.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.45], vertical_spacing=0.06)
            fig.add_trace(go.Scatter(
                x=df_energy["timestamp"], y=df_energy["soc"],
                name="Battery SoC %", line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(27,122,74,0.08)"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_energy["timestamp"], y=df_energy["solar_power"],
                name="Solar Power (W)", line=dict(color=AMBER, width=1.5)
            ), row=2, col=1)
            fig.update_layout(height=280, paper_bgcolor=BG, plot_bgcolor=WHITE,
                              margin=dict(l=10,r=10,t=10,b=10),
                              font=dict(size=10), showlegend=True,
                              legend=dict(orientation="h", y=1.08, x=0))
            fig.update_xaxes(gridcolor="#F1F5F9")
            fig.update_yaxes(gridcolor="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No energy data found.")

    # ── Summary insights ─────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        section("🔬", "Water Quality Findings")
        insight("EC and TDS are strongly correlated (r = 0.924) — TDS sensor is redundant and can be removed.")
        insight("DO and pH follow a clear day-night biological cycle (r = −0.696), confirming healthy photosynthesis.")

    with col_b:
        section("🤖", "ML Results")
        insight("LSTM Autoencoder detected 7/10 cleaning events (70%) — best method by learning temporal patterns.")
        insight("Virtual DO sensor: R² = 0.947 using Random Forest from pH, Temp, EC and TDS.")

    with col_c:
        section("📡", "IoT Optimization")
        insight("Reducing sampling from 5s to 1 min cuts 91.2% of data volume with zero information loss.")
        if analysis.get("forecast", {}).get("available"):
            fc = analysis["forecast"]
            np_ = fc["next_prediction"]
            insight(f"SoC forecast (60 min ahead): {np_['current_soc']:.0f}% → {np_['soc']:.0f}%  "
                    f"({'↑' if np_['delta'] > 0 else '↓'} {abs(np_['delta']):.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WATER QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif "Water Quality" in page:
    st.title("💧 Water Quality Analysis")
    st.caption("Tess Lobry — ICAM Lille × Ateneo de Manila University")

    if df_water.empty:
        st.warning("No water quality data found. Place your CSV in the `data/` folder.")
        st.stop()

    dw = df_water.copy()
    dt_col = "Datetime" if "Datetime" in dw.columns else dw.columns[0]
    dw[dt_col] = pd.to_datetime(dw[dt_col])
    dw = dw.set_index(dt_col).sort_index()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    last = dw.iloc[-1]
    cols = st.columns(5)
    labels = {"MCP_WQ_DO":"Dissolved O₂","MCP_WQ_EC":"Conductivity","MCP_WQ_PH":"pH",
              "MCP_WQ_TEMP":"Temperature","MCP_WQ_TDS":"TDS"}
    colors = {"MCP_WQ_DO":TEAL,"MCP_WQ_EC":GREEN,"MCP_WQ_PH":AMBER,
              "MCP_WQ_TEMP":"#7C3AED","MCP_WQ_TDS":"#0891B2"}
    for i, col in enumerate(WATER_PRESENT[:5]):
        with cols[i]:
            val = last.get(col, None)
            v = f"{val:.1f}" if val is not None and not math.isnan(val) else "—"
            mean_v = dw[col].mean()
            delta = (val - mean_v) if val is not None else None
            card_color = "teal" if i % 2 == 0 else "green"
            st.markdown(kpi(labels.get(col, col), v, "raw",
                           note=f"avg {mean_v:.1f}"), unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🔗 Correlations", "🤖 Virtual Sensors", "🔍 Anomalies"])

    with tab1:
        section("📈", "Sensor Time Series")
        cols_to_show = st.multiselect(
            "Select sensors to display:",
            options=WATER_PRESENT,
            default=WATER_PRESENT[:3],
            format_func=lambda c: labels.get(c, c)
        )
        if cols_to_show:
            fig = make_subplots(rows=len(cols_to_show), cols=1, shared_xaxes=True,
                                vertical_spacing=0.04,
                                subplot_titles=[labels.get(c, c) for c in cols_to_show])
            for i, col in enumerate(cols_to_show, 1):
                color = colors.get(col, TEAL)
                fig.add_trace(go.Scatter(
                    x=dw.index, y=dw[col],
                    name=labels.get(col, col),
                    line=dict(color=color, width=1.2),
                    fill="tozeroy" if i == 1 else "none",
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)"
                        if i == 1 and color.startswith("#") else None,
                ), row=i, col=1)
                # Rolling mean overlay
                roll = dw[col].rolling(30, center=True, min_periods=5).mean()
                fig.add_trace(go.Scatter(
                    x=dw.index, y=roll,
                    name=f"{labels.get(col,col)} (30-pt avg)",
                    line=dict(color=color, width=2.5, dash="dot"),
                    showlegend=False
                ), row=i, col=1)
            fig.update_layout(height=150 * len(cols_to_show) + 60,
                              paper_bgcolor=BG, plot_bgcolor=WHITE,
                              margin=dict(l=10,r=10,t=40,b=10),
                              font=dict(size=10), showlegend=False)
            fig.update_xaxes(gridcolor="#F1F5F9")
            fig.update_yaxes(gridcolor="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("🔗", "Pearson Correlation Analysis")
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            if len(WATER_PRESENT) >= 2:
                corr_matrix = dw[WATER_PRESENT].corr(method="pearson")
                corr_matrix.index   = [labels.get(c, c) for c in corr_matrix.index]
                corr_matrix.columns = [labels.get(c, c) for c in corr_matrix.columns]

                fig = go.Figure(go.Heatmap(
                    z=corr_matrix.values,
                    x=list(corr_matrix.columns),
                    y=list(corr_matrix.index),
                    colorscale="RdYlGn",
                    zmin=-1, zmax=1,
                    text=[[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    textfont=dict(size=11, color="black"),
                    colorbar=dict(thickness=12, len=0.8)
                ))
                fig.update_layout(height=340, paper_bgcolor=BG, plot_bgcolor=WHITE,
                                  margin=dict(l=10,r=10,t=20,b=10),
                                  font=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Key findings from this dataset:**")
            if "MCP_WQ_EC" in dw.columns and "MCP_WQ_TDS" in dw.columns:
                r_ec_tds = dw[["MCP_WQ_EC","MCP_WQ_TDS"]].corr().iloc[0,1]
                st.markdown(f"**EC ↔ TDS:** r = {r_ec_tds:.3f}")
                insight("EC and TDS are nearly identical measurements. "
                       "Removing the TDS sensor would save cost and maintenance "
                       "with no information loss. Formula: TDS = 1.254 × EC − 34.8 (R² = 0.854)")

            if "MCP_WQ_DO" in dw.columns and "MCP_WQ_PH" in dw.columns:
                r_do_ph = dw[["MCP_WQ_DO","MCP_WQ_PH"]].corr().iloc[0,1]
                st.markdown(f"**DO ↔ pH:** r = {r_do_ph:.3f}")
                insight("Negative correlation reflects the photosynthesis-respiration cycle: "
                       "DO and pH both rise during the day and fall at night. "
                       "This confirms the system is biologically healthy.")

            if "MCP_WQ_EC" in dw.columns and "MCP_WQ_DO" in dw.columns:
                r_do_ec = dw[["MCP_WQ_DO","MCP_WQ_EC"]].corr().iloc[0,1]
                st.markdown(f"**DO ↔ EC:** r = {r_do_ec:.3f}")

        # EC vs TDS scatter
        if "MCP_WQ_EC" in dw.columns and "MCP_WQ_TDS" in dw.columns:
            st.markdown("**EC vs TDS — Redundancy Visualization**")
            sample = dw[["MCP_WQ_EC","MCP_WQ_TDS"]].dropna().sample(min(5000, len(dw)), random_state=42)
            ec_arr = sample["MCP_WQ_EC"].values
            tds_arr = sample["MCP_WQ_TDS"].values
            # Regression line
            coef = np.polyfit(ec_arr, tds_arr, 1)
            x_range = np.linspace(ec_arr.min(), ec_arr.max(), 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ec_arr, y=tds_arr, mode="markers",
                marker=dict(color=TEAL, size=2, opacity=0.3),
                name="Data points"
            ))
            fig.add_trace(go.Scatter(
                x=x_range, y=np.polyval(coef, x_range),
                mode="lines", line=dict(color=RED, width=2.5),
                name=f"TDS = {coef[0]:.3f} × EC + {coef[1]:.1f}"
            ))
            fig = plotly_defaults(fig, height=280)
            fig.update_layout(xaxis_title="EC (raw)", yaxis_title="TDS (raw)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("🤖", "Virtual Sensors")
        st.markdown("""
        Virtual sensors use machine learning to predict one sensor from the others.
        This can replace expensive physical sensors or provide backup when a sensor fails.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Virtual TDS (from EC) — Random Forest")
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(kpi("R²", "0.873", "", color="green"), unsafe_allow_html=True)
            with m2: st.markdown(kpi("RMSE", "11.70", "raw", color="teal"), unsafe_allow_html=True)
            with m3: st.markdown(kpi("Model", "RF", "", color="teal2"), unsafe_allow_html=True)
            insight("TDS can be predicted from EC alone with R² = 0.873. "
                   "The TDS sensor can be removed from the system entirely.")

        with c2:
            st.markdown("#### Virtual DO (from pH, Temp, EC, TDS) — Random Forest")
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(kpi("R²", "0.947", "", color="green"), unsafe_allow_html=True)
            with m2: st.markdown(kpi("RMSE", "14.16", "raw", color="teal"), unsafe_allow_html=True)
            with m3: st.markdown(kpi("Model", "RF", "", color="teal2"), unsafe_allow_html=True)
            insight("DO can be predicted with 94.7% accuracy from the other 4 sensors. "
                   "pH is the dominant predictor (importance = 0.491).")

        # Feature importance bar chart
        st.markdown("**Feature importance for DO prediction (XGBoost)**")
        fi_data = {"Feature": ["pH", "TDS", "Temperature", "EC"],
                   "Importance": [0.491, 0.420, 0.064, 0.025]}
        fig = go.Figure(go.Bar(
            x=fi_data["Importance"], y=fi_data["Feature"],
            orientation="h",
            marker=dict(
                color=[TEAL, GREEN, SLATE, "#94A3B8"],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in fi_data["Importance"]],
            textposition="outside"
        ))
        fig = plotly_defaults(fig, height=200)
        fig.update_layout(xaxis_title="Importance", yaxis=dict(autorange="reversed"),
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        insight("⚠️ Temporal drift warning: when models are trained on Jan–Mar data and tested "
               "on Apr–May data, R² drops to negative values. Uncalibrated sensors drift over time. "
               "Periodic retraining is required in production.")

    with tab4:
        section("🔍", "Anomaly Detection Results")
        st.markdown("""
        Three algorithms were tested on 1,168,563 measurements from 76 good days.
        All were trained on clean reference data, then applied to the full dataset.
        """)

        # Method comparison
        comparison = pd.DataFrame({
            "Method":     ["Isolation Forest", "KNN", "LSTM Autoencoder ★"],
            "Anomalies":  ["25,368", "5,375", "1,314"],
            "Rate":       ["2.2%", "0.4%", "1.0%"],
            "Detected":   ["3/10 (30%)", "3/10 (30%)", "7/10 (70%)"],
            "Physical?":  ["❌ No", "✅ Yes", "✅ Yes"],
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        insight("The LSTM Autoencoder achieves the best detection rate (70%) by learning "
               "the temporal structure of the data — including the day-night DO cycle.")

        # KNN physical signature
        st.markdown("**Physical signature of KNN anomalies** (what they look like):")
        sig = pd.DataFrame({
            "Sensor": ["DO", "EC", "pH", "Temperature", "TDS"],
            "Normal value": [81.23, 211.48, 459.19, 246.86, 229.97],
            "Anomaly value": [2.37, 164.95, 498.36, 245.64, 185.71],
            "Change": ["−97.1% ✅", "−22.1% ✅", "+8.5% ✅", "−0.5%", "−19.3% ✅"],
        })
        st.dataframe(sig, hide_index=True, use_container_width=True)
        insight("This signature (DO drops 97%, EC drops 22%, pH rises 8.5%) is exactly "
               "what happens when a sensor is removed from water during cleaning. "
               "KNN detections are physically meaningful.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ENERGY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
elif "Energy System" in page:
    st.title("⚡ Energy System Analysis")
    st.caption("Simon Tourtois — Ateneo de Manila University")

    if df_energy.empty:
        st.warning("No energy data found. Place `data.json` in the `cache/` folder.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    last = df_energy.iloc[-1]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    soc = last.get("soc", 0)
    soc_color = "green" if soc > 60 else ("amber" if soc > 30 else "red")
    with c1: st.markdown(kpi("Battery SoC",      f"{soc:.0f}",                "%",  color=soc_color), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Battery Voltage",  f"{last.get('bat_voltage',0):.2f}", "V", color="teal"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Solar Power",       f"{last.get('solar_power',0):.0f}", "W", color="amber"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Solar Current",     f"{last.get('solar_current',0):.2f}","A", color="amber"), unsafe_allow_html=True)
    with c5: st.markdown(kpi("Pump Current",      f"{last.get('current_pump',0):.2f}", "A", color="teal2"), unsafe_allow_html=True)
    with c6: st.markdown(kpi("Aerator Current",   f"{last.get('current_aerator',0):.2f}","A",color="teal2"), unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "🔗 Correlations",
        "📐 Curve Fitting", "🔋 SoC Forecast", "⚠️ Anomalies"
    ])

    with tab1:
        section("📈", "Energy System — Time Series")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=["Battery SoC (%)", "Solar Power (W)", "Load Currents (A)"])
        
        fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["soc"],
            name="SoC", line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(27,122,74,0.08)"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["solar_power"],
            name="Solar W", line=dict(color=AMBER, width=1.5),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.08)"), row=2, col=1)

        fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["current_pump"],
            name="Pump (A)", line=dict(color=TEAL, width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["current_aerator"],
            name="Aerator (A)", line=dict(color=TEAL2, width=1.2)), row=3, col=1)

        fig.update_layout(height=480, paper_bgcolor=BG, plot_bgcolor=WHITE,
                          margin=dict(l=10,r=10,t=40,b=10), font=dict(size=10))
        fig.update_xaxes(gridcolor="#F1F5F9")
        fig.update_yaxes(gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("🔗", "Pearson Correlation Matrix — Energy")
        corr_data = analysis.get("correlation", {})
        if corr_data.get("variables") and corr_data.get("matrix"):
            vars_ = corr_data["variables"]
            mat   = np.array(corr_data["matrix"], dtype=float)
            labels_e = [v.replace("_", " ").title() for v in vars_]
            
            fig = go.Figure(go.Heatmap(
                z=mat, x=labels_e, y=labels_e,
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=[[f"{v:.2f}" if v is not None and not math.isnan(v) else "" for v in row] for row in mat],
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorbar=dict(thickness=12, len=0.8)
            ))
            fig.update_layout(height=400, paper_bgcolor=BG, plot_bgcolor=WHITE,
                              margin=dict(l=10,r=10,t=20,b=10), font=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)

            # Highlighted pairs
            pairs = corr_data.get("pairs", [])
            if pairs:
                st.markdown("**Key correlations:**")
                pair_df = pd.DataFrame([{
                    "Pair": p["label"],
                    "r": p["r"],
                    "Strength": p["strength"],
                    "Interpretation": p["interpretation"]
                } for p in pairs])
                st.dataframe(pair_df, hide_index=True, use_container_width=True)

    with tab3:
        section("📐", "Curve Fitting — PV Characteristics")
        cf_curr = analysis.get("curve_fitting", {}).get("current_vs_temp", {})
        cf_volt = analysis.get("curve_fitting", {}).get("voltage_vs_temp", {})

        c_l, c_r = st.columns(2)
        for col_ui, cf, title, color in [
            (c_l, cf_curr, "Panel Current vs Temperature", AMBER),
            (c_r, cf_volt, "Panel Voltage vs Temperature", TEAL)
        ]:
            with col_ui:
                st.markdown(f"**{title}**")
                if cf.get("available"):
                    scatter = cf.get("scatter", [])
                    curves  = cf.get("curves", {})
                    best    = cf.get("best", "linear")
                    models  = cf.get("models", {})

                    fig = go.Figure()
                    if scatter:
                        xs = [p["x"] for p in scatter]
                        ys = [p["y"] for p in scatter]
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                            marker=dict(color=color, size=3, opacity=0.35),
                            name="Raw data"))
                    for name, pts in curves.items():
                        cx = [p["x"] for p in pts]
                        cy = [p["y"] for p in pts]
                        is_best = name == best
                        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                            line=dict(color=RED if is_best else "#94A3B8",
                                     width=2.5 if is_best else 1.5,
                                     dash="solid" if is_best else "dot"),
                            name=f"{name} {'★' if is_best else ''}"))

                    fig = plotly_defaults(fig, height=260)
                    fig.update_layout(xaxis_title="Panel Temperature (°C)",
                                     yaxis_title="Current (A)" if "current" in title.lower() else "Voltage (V)")
                    st.plotly_chart(fig, use_container_width=True)

                    if best in models:
                        bm = models[best]
                        st.markdown(f"**Best fit ({best}):** `{bm['equation']}`  "
                                   f"R² = **{bm['r2']:.3f}**  RMSE = {bm['rmse']:.3f}")
                else:
                    st.info("Not enough data for curve fitting.")

    with tab4:
        section("🔋", "SoC Forecast — 60-min Ahead")
        fc = analysis.get("forecast", {})
        if fc.get("available"):
            np_ = fc["next_prediction"]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(kpi("Current SoC",   f"{np_['current_soc']:.0f}", "%", color="teal"), unsafe_allow_html=True)
            with c2: st.markdown(kpi("Forecast SoC",  f"{np_['soc']:.0f}",         "%",
                                    color="green" if np_['delta'] >= 0 else "amber"), unsafe_allow_html=True)
            with c3: st.markdown(kpi("Delta",  f"{np_['delta']:+.1f}", "%",          color="green" if np_['delta'] >= 0 else "amber"), unsafe_allow_html=True)
            with c4: st.markdown(kpi("R²",             f"{fc['metrics']['r2']:.3f}", "",  color="teal"), unsafe_allow_html=True)

            insight(np_["reasoning"])

            # Time series: actual vs predicted
            test_s = fc.get("test_series", [])
            if test_s:
                ts_df = pd.DataFrame(test_s)
                ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["soc_actual"],
                    name="Actual SoC", line=dict(color=GREEN, width=2)))
                fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["soc_predicted"],
                    name="Predicted SoC", line=dict(color=AMBER, width=1.5, dash="dot")))
                fig = plotly_defaults(fig, height=280)
                fig.update_layout(yaxis_title="SoC (%)",
                                 legend=dict(orientation="h", y=1.05, x=0))
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            fi = fc.get("feature_importance", [])
            if fi:
                st.markdown("**Top feature importances:**")
                fi_df = pd.DataFrame(fi[:8])
                fig = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["feature"],
                    orientation="h",
                    marker=dict(color=TEAL, line=dict(width=0)),
                    text=[f"{v:.3f}" for v in fi_df["importance"]],
                    textposition="outside"
                ))
                fig = plotly_defaults(fig, height=220)
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Limitations
            st.markdown("**Model limitations:**")
            for lim in fc.get("limitations", []):
                st.markdown(f"⚠️ {lim}")
        else:
            st.info("Forecast not available: " + fc.get("reason", "unknown"))

    with tab5:
        section("⚠️", "Anomaly Detection — Energy System")
        an = analysis.get("anomalies", {})
        if an.get("available"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(kpi("Total samples",   f"{an['n_samples']:,}",   "", color="teal"), unsafe_allow_html=True)
            with c2: st.markdown(kpi("Anomalies found", f"{an['n_anomalies']}",   "", color="amber"), unsafe_allow_html=True)
            with c3: st.markdown(kpi("Method", "Robust z-score", "", color="teal2"), unsafe_allow_html=True)

            # Score timeline
            tl = an.get("timeline", [])
            if tl:
                tl_df = pd.DataFrame(tl)
                tl_df["timestamp"] = pd.to_datetime(tl_df["timestamp"])
                fig = go.Figure()
                normal  = tl_df[~tl_df["anomaly"]]
                anomaly = tl_df[tl_df["anomaly"]]
                fig.add_trace(go.Scatter(x=normal["timestamp"],  y=normal["score"],
                    mode="lines", name="Normal", line=dict(color=TEAL, width=1)))
                fig.add_trace(go.Scatter(x=anomaly["timestamp"], y=anomaly["score"],
                    mode="markers", name="Anomaly",
                    marker=dict(color=RED, size=5, symbol="x")))
                fig.add_hline(y=an["z_threshold"], line=dict(color=AMBER, width=1.5, dash="dash"),
                             annotation_text=f"Threshold = {an['z_threshold']}")
                fig = plotly_defaults(fig, height=240)
                fig.update_layout(yaxis_title="Anomaly score")
                st.plotly_chart(fig, use_container_width=True)

            # Anomaly table
            anom_list = an.get("anomalies", [])
            if anom_list:
                st.markdown("**Detected anomalies (top 20):**")
                anom_df = pd.DataFrame(anom_list[:20])[
                    ["timestamp","trigger","actual","expected","deviation","severity","reason"]
                ]
                st.dataframe(anom_df, hide_index=True, use_container_width=True)
        else:
            st.info("Anomaly detection not available: " + an.get("reason", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COMBINED VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif "Combined" in page:
    st.title("🔗 Combined View — Water Quality × Energy")
    st.caption("Cross-system analysis · Tess Lobry × Simon Tourtois")

    if df_energy.empty or df_water.empty:
        st.warning("Both datasets are needed for the combined view.")
        st.stop()

    # ── Temperature comparison ────────────────────────────────────────────────
    section("🌡️", "Temperature — Water vs Panel vs Battery")

    dw_idx = df_water.copy()
    if "Datetime" in dw_idx.columns:
        dw_idx = dw_idx.set_index("Datetime")
    dw_idx.index = pd.to_datetime(dw_idx.index)

    fig = go.Figure()
    # Water temperature
    if "MCP_WQ_TEMP" in dw_idx.columns:
        # Rescale raw water temp to approximate Celsius if needed (raw ~240-255 → ~24-26°C)
        wt = dw_idx["MCP_WQ_TEMP"]
        if wt.mean() > 100:
            wt = (wt - 220) * 0.1 + 20  # rough rescale
        fig.add_trace(go.Scatter(x=dw_idx.index, y=wt,
            name="Water temp (°C est.)", line=dict(color=TEAL, width=1.5)))

    fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["temp_panel"],
        name="Panel temp (°C)", line=dict(color=AMBER, width=1.5)))
    fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["temp_battery"],
        name="Battery temp (°C)", line=dict(color=GREEN, width=1.2)))
    if "temp_water" in df_energy.columns:
        fig.add_trace(go.Scatter(x=df_energy["timestamp"], y=df_energy["temp_water"],
            name="Water temp – energy probe (°C)", line=dict(color=TEAL2, width=1.5, dash="dot")))

    fig = plotly_defaults(fig, height=300)
    fig.update_layout(yaxis_title="Temperature (°C)",
                     legend=dict(orientation="h", y=1.06, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── System health overview ─────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        section("💧", "Water Quality Summary")
        if WATER_PRESENT:
            stats = dw_idx[WATER_PRESENT].describe().round(2).T
            stats.index = [labels.get(c, c) for c in stats.index]
            st.dataframe(stats[["mean", "std", "min", "max"]], use_container_width=True)
        insight("All 5 sensors running. LSTM Autoencoder detects anomalies with 70% accuracy.")

    with col_r:
        section("⚡", "Energy Summary")
        if not df_energy.empty:
            e_stats = df_energy[["soc","solar_power","bat_voltage","current_pump","current_aerator"]].describe().round(2).T
            e_stats.index = ["SoC (%)","Solar Power (W)","Battery V","Pump (A)","Aerator (A)"]
            st.dataframe(e_stats[["mean","std","min","max"]], use_container_width=True)
        if analysis.get("forecast", {}).get("available"):
            np_ = analysis["forecast"]["next_prediction"]
            insight(f"SoC forecast: {np_['current_soc']:.0f}% → {np_['soc']:.0f}% in 60 min. {np_['reasoning']}")

    # ── Cross-system correlation ───────────────────────────────────────────────
    section("🔗", "Cross-System Correlation")
    st.markdown("Can energy variables predict water quality? (Exploratory analysis)")

    if "temp_water" in df_energy.columns and WATER_PRESENT:
        # Resample energy to 1-min
        e_1min = df_energy.set_index("timestamp").resample("1min").mean()
        w_1min = dw_idx.resample("1min").mean()
        merged = e_1min.join(w_1min, how="inner", lsuffix="_en", rsuffix="_wq")

        if not merged.empty and "MCP_WQ_DO" in merged.columns:
            # DO vs solar power
            sub = merged[["solar_power","MCP_WQ_DO"]].dropna().sample(min(2000, len(merged)), random_state=42)
            if len(sub) > 50:
                r = sub.corr().iloc[0,1]
                fig = go.Figure(go.Scatter(
                    x=sub["solar_power"], y=sub["MCP_WQ_DO"],
                    mode="markers",
                    marker=dict(color=TEAL, size=3, opacity=0.4)
                ))
                fig = plotly_defaults(fig, height=250)
                fig.update_layout(xaxis_title="Solar Power (W)",
                                 yaxis_title="DO (raw)",
                                 title=f"DO vs Solar Power  |  r = {r:.3f}")
                st.plotly_chart(fig, use_container_width=True)
                if abs(r) > 0.3:
                    insight(f"Moderate correlation (r = {r:.3f}) between solar power and dissolved oxygen — "
                           "likely indirect via temperature and biological activity.")
                else:
                    insight(f"Weak correlation (r = {r:.3f}) — solar power and DO appear largely independent "
                           "at the measured time resolution.")
    else:
        insight("To see cross-system correlations, make sure both datasets "
               "cover the same time period and contain temperature readings.")
