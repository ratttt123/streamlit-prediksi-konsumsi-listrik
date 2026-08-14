import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import numpy as np

# Tambahkan root folder ke path utils untuk sidebar
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_css, render_sidebar

st.set_page_config(page_title="Dashboard", page_icon="⚡", layout="wide")
load_css()
render_sidebar()

# ── CSS KHUSUS DASHBOARD ─────────────────────────────────────
with open(Path(__file__).parent.parent / "styles" / "dashboard.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel("data/data_listrik_clean.xlsx")

df = load_data()

# Data efektif setelah lag (lag_1, lag_2, lag_3 -> 3 baris pertama jadi NaN)
df_efektif = df.dropna(subset=["lag_1", "lag_2", "lag_3"])

# ── LOAD HASIL OPTIMASI & HITUNG MAPE ────────────────────────
@st.cache_data
def load_hasil_optimasi():
    return pd.read_excel("data/hasil_optimasi_timeseries.xlsx")

hasil_optimasi = load_hasil_optimasi()
mape_optimasi = (
    abs((hasil_optimasi["Actual"] - hasil_optimasi["Prediksi"]) / hasil_optimasi["Actual"]).mean()
) * 100

# ── INFORMASI MODEL (HASIL OPTIMASI) ────────────────────────
MODEL_INFO = {
    "model_terbaik": "Random Forest Regressor (Tuned)",
    "akurasi_mape": f"{mape_optimasi:.2f}%".replace(".", ","),
    "total_data": f"{len(df)} bulan",
    "data_training": f"{int(len(df_efektif) * 0.8)} bulan (80%)",
    "data_testing": f"{len(df_efektif) - int(len(df_efektif) * 0.8)} bulan (20%)",
}

# ── WRAPPER SCOPE HALAMAN DASHBOARD ────────────────────────────
st.markdown('<div class="dashboard-page">', unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Dashboard</h1>
    <p>Gambaran umum data beban listrik bulanan PT Inti Bumi Perkasa</p>
</div>
""", unsafe_allow_html=True)

# ── METRIK ───────────────────────────────────────────────────
ICON_DATABASE = """<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>"""
ICON_TREND_UP = """<svg viewBox="0 0 24 24"><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></svg>"""
ICON_TREND_DOWN = """<svg viewBox="0 0 24 24"><polyline points="3 7 9 13 13 9 21 17"/><polyline points="14 17 21 17 21 10"/></svg>"""
ICON_GAUGE = """<svg viewBox="0 0 24 24"><path d="M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20Z"/><path d="M12 12 16 8"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/></svg>"""
ICON_CHART_LINE = """<svg viewBox="0 0 24 24"><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></svg>"""
ICON_TABLE = """<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="3" y1="10" x2="21" y2="10"/><line x1="9" y1="10" x2="9" y2="20"/></svg>"""

c1, c2, c3, c4 = st.columns(4)
for col, label, value, icon, color in [
    (c1, "Total Data",      f"{len(df)} bulan",        ICON_DATABASE,   "blue"),
    (c2, "Beban Tertinggi", f"{df['Beban'].max():,.0f} kWh", ICON_TREND_UP,   "red"),
    (c3, "Beban Terendah",  f"{df['Beban'].min():,.0f} kWh", ICON_TREND_DOWN, "green"),
    (c4, "Rata-rata Beban", f"{df['Beban'].mean():,.0f} kWh", ICON_GAUGE,      "purple"),
]:
    col.markdown(
        f'<div class="metric-card {color}">'
        f'<div class="icon">{icon}</div>'
        f'<div class="text">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── GRAFIK TREN + INFORMASI MODEL ────────────────────────────
col_chart, col_info = st.columns([2.4, 1])

with col_chart:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title"><span class="section-icon">{ICON_CHART_LINE}</span>Tren Beban Listrik Bulanan</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["periode"], y=df["Beban"],
        mode="lines+markers",
        name="Beban Listrik",
        line=dict(color="#2E4DB5", width=2.5),
        marker=dict(size=6, color="#1B2A6B"),
        hovertemplate="<b>%{x}</b><br>Beban: %{y:,.0f} kWh<extra></extra>"
    ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            tickangle=-45,
            gridcolor="#F0F4FF",
            rangeslider=dict(visible=True, thickness=0.08),
            type="category"
        ),
        yaxis=dict(gridcolor="#F0F4FF", title="Beban Listrik (kWh)"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── TUTUP WRAPPER SCOPE HALAMAN DASHBOARD ──────────────────────
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="model-info-title">Informasi Model '
        '<span class="badge-optimized">Optimasi</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-info-card highlight">'
        f'<div class="label">Model Terbaik</div>'
        f'<div class="value">{MODEL_INFO["model_terbaik"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-card">'
        f'<div class="label">Akurasi (MAPE)</div>'
        f'<div class="value">{MODEL_INFO["akurasi_mape"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-card">'
        f'<div class="label">Total Data</div>'
        f'<div class="value">{MODEL_INFO["total_data"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-card">'
        f'<div class="label">Data Latih (Training) </div>'
        f'<div class="value">{MODEL_INFO["data_training"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="model-info-card">'
        f'<div class="label">Data Uji (Testing)</div>'
        f'<div class="value">{MODEL_INFO["data_testing"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── TABEL ────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title"><span class="section-icon">{ICON_TABLE}</span>Tabel Data Historis</div>', unsafe_allow_html=True)
st.dataframe(
    df[['periode', 'Tahun', 'Bulan', 'Beban', 'lag_1', 'lag_2', 'lag_3']],
    use_container_width=True, hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)