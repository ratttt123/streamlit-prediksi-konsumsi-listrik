import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_css, render_sidebar, get_image_base64

st.set_page_config(page_title="Prediksi", page_icon="⚡", layout="wide")
load_css()
render_sidebar()

with open(Path(__file__).parent.parent / "styles" / "dashboard.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open(Path(__file__).parent.parent / "styles" / "prediksi.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL & DATA
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("model/model_rf.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("data/data_listrik_clean.xlsx")

model = load_model()
df    = load_data()
df['periode'] = df['periode'].astype(str)

last_row   = df.iloc[-1]
last_tahun = int(last_row['Tahun'])
last_bulan = int(last_row['Bulan'])

NAMA_BULAN = {
    1:'Januari', 2:'Februari', 3:'Maret',    4:'April',
    5:'Mei',     6:'Juni',     7:'Juli',      8:'Agustus',
    9:'September',10:'Oktober',11:'November', 12:'Desember'
}
NAMA_BULAN_SINGKAT = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun',
    7:'Jul', 8:'Agt', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'
}

# Buat daftar 12 bulan ke depan
bulan_list = []
for i in range(1, 13):
    b = last_bulan + i
    t = last_tahun
    if b > 12:
        b -= 12
        t += 1
    bulan_list.append({'label': f"{NAMA_BULAN[b]} {t}", 'tahun': t, 'bulan': b, 'step': i})

PARAM_GRID = {
    'n_estimators':    [50, 100],
    'max_depth':       [3, 5],
    'min_samples_split': [2, 5],
}
N_SPLITS = 3


# ══════════════════════════════════════════════════════════════
# HELPER: STEP BAR HTML
# ══════════════════════════════════════════════════════════════
def render_step_bar(active: int):
    """Render step bar. active=1 (pilih), 2 (hitung), 3 (hasil)."""
    steps = ["Pilih bulan", "Hitung prediksi", "Lihat hasil"]
    items = []
    for i, label in enumerate(steps, start=1):
        if i < active:
            circle  = '<span class="step-circle done">✓</span>'
            txt_cls = "step-text"
        elif i == active:
            circle  = f'<span class="step-circle active">{i}</span>'
            txt_cls = "step-text active"
        else:
            circle  = f'<span class="step-circle pending">{i}</span>'
            txt_cls = "step-text"

        item = f'<div class="step-item">{circle}<span class="{txt_cls}">{label}</span></div>'
        if i < len(steps):
            item += '<div class="step-line"></div>'
        items.append(item)

    st.markdown(
        f'<div class="step-bar">{"".join(items)}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
# GRIDSEARCHCV + EVALUASI
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def jalankan_gridsearch(_df):
    X = _df[['lag_1', 'lag_2', 'lag_3']]
    y = _df['Beban']

    split   = int(len(_df) * 0.8)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    grid = GridSearchCV(
        estimator  = RandomForestRegressor(random_state=42),
        param_grid = PARAM_GRID,
        cv         = tscv,
        scoring    = 'neg_mean_absolute_error',
        n_jobs     = -1
    )
    grid.fit(X_train, y_train)

    best_index  = grid.best_index_
    cv_results  = grid.cv_results_
    fold_scores = [
        -cv_results[f'split{i}_test_score'][best_index]
        for i in range(N_SPLITS)
    ]

    best_model  = RandomForestRegressor(random_state=42, **grid.best_params_)
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)

    mae_test  = abs(y_test.values - y_pred_test).mean()
    mape_test = (abs((y_test.values - y_pred_test) / y_test.values)).mean() * 100

    # MAE & MAPE baseline (model asli) pada test set yang sama
    y_pred_base  = model.predict(X_test)
    mae_base     = abs(y_test.values - y_pred_base).mean()
    mape_base    = (abs((y_test.values - y_pred_base) / y_test.values)).mean() * 100

    return {
        'best_estimator':  best_model,
        'best_params':     grid.best_params_,
        'best_mae_cv':     -grid.best_score_,     # MAE rata-rata CV (train)
        'fold_scores':     fold_scores,
        'n_combinations':  len(cv_results['params']),
        # Evaluasi test set
        'mae_baseline':    mae_base,
        'mape_baseline':   mape_base,
        'mae_optimasi':    mae_test,
        'mape_optimasi':   mape_test,
    }


# ══════════════════════════════════════════════════════════════
# PREDIKSI BERANTAI
# ══════════════════════════════════════════════════════════════
def hitung_chain(mdl, n_steps):
    history   = list(df['Beban'].tail(3).values[::-1])
    results   = []
    cur_bulan = last_bulan
    cur_tahun = last_tahun

    for _ in range(n_steps):
        lag1, lag2, lag3 = float(history[0]), float(history[1]), float(history[2])
        input_df   = pd.DataFrame([[lag1, lag2, lag3]], columns=['lag_1','lag_2','lag_3'])
        tree_preds = np.array([t.predict(input_df)[0] for t in mdl.estimators_])
        pred       = tree_preds.mean()

        cur_bulan += 1
        if cur_bulan > 12:
            cur_bulan = 1
            cur_tahun += 1

        results.append({
            'Periode':  f"{NAMA_BULAN[cur_bulan]} {cur_tahun}",
            'Singkat':  f"{NAMA_BULAN_SINGKAT[cur_bulan]} {str(cur_tahun)[2:]}",
            'lag_1': lag1, 'lag_2': lag2, 'lag_3': lag3,
            '_pred':       pred,
            '_tree_preds': tree_preds,
        })
        history = [pred, history[0], history[1]]

    return results


# ══════════════════════════════════════════════════════════════
# HELPER: BUAT GRAFIK PLOTLY
# ══════════════════════════════════════════════════════════════
LAYOUT_BASE = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        tickangle=-45,
        gridcolor='#F0F4FF',
        type='category',
        tickfont=dict(size=11, color='#6B7A99'),
    ),
    yaxis=dict(
        gridcolor='#F0F4FF',
        title=dict(text='Beban Listrik (kWh)', font=dict(size=12, color='#6B7A99')),
        tickfont=dict(size=11, color='#6B7A99'),
    ),
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1,
        font=dict(size=12),
        bgcolor='rgba(0,0,0,0)',
    ),
    hovermode='x unified',
    margin=dict(l=10, r=10, t=10, b=10),
    height=420,
)


def buat_grafik_tunggal(chain, warna_pred, nama_pred, fill_pred=None, puncak=None):
    # Ambil seluruh data historis (sama seperti dashboard)
    hist_periode = df['periode'].tolist()
    hist_beban   = df['Beban'].tolist()
    pred_vals    = [r['_pred'] for r in chain]

    # Sambung titik terakhir histori ke prediksi agar kurva tidak putus
    x_pred = [hist_periode[-1]] + [r['Periode'] for r in chain]
    y_pred = [hist_beban[-1]]   + pred_vals

    fig = go.Figure()

    # Trace data aktual — persis seperti dashboard
    fig.add_trace(go.Scatter(
        x=hist_periode, y=hist_beban,
        mode='lines+markers',
        name='Data Aktual',
        line=dict(color='#2E4DB5', width=2.5, shape='spline', smoothing=0.6),
        marker=dict(size=6, color='#1B2A6B'),
        hovertemplate='<b>%{x}</b><br>Beban: %{y:,.2f} kWh<extra>Data Aktual</extra>',
    ))

    # Trace prediksi — warna berbeda, garis putus-putus
    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        mode='lines+markers',
        name=nama_pred,
        line=dict(color=warna_pred, width=2.5, dash='dot', shape='spline', smoothing=0.6),
        marker=dict(size=8, color=warna_pred, symbol='diamond',
                    line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Prediksi: %{y:,.2f} kWh<extra>' + nama_pred + '</extra>',
    ))

    # Garis pemisah aktual vs prediksi
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=hist_periode[-1], x1=hist_periode[-1],
        y0=0, y1=0.94,
        line=dict(color='#CBD5E1', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        xref='x', yref='paper',
        x=hist_periode[-1], y=0.97,
        text='← Aktual · Prediksi →',
        showarrow=False,
        font=dict(size=10, color='#94A3B8'),
        xanchor='center',
    )

    if puncak:
        fig.add_annotation(
            x=puncak['Periode'], y=puncak['_pred'],
            text=(f"<b>{puncak['Periode']}</b><br>"
                  f"Tertinggi<br>"
                  f"<b>{puncak['_pred']:,.2f} kWh</b>"),
            showarrow=True, arrowhead=2, arrowcolor=warna_pred,
            ax=0, ay=-56,
            bgcolor='white', bordercolor=warna_pred,
            borderwidth=1.5, borderpad=6,
            font=dict(size=11, color='#1B2A6B'),
        )

    layout = dict(**LAYOUT_BASE)
    layout['xaxis'] = dict(
        **LAYOUT_BASE['xaxis'],
        rangeslider=dict(visible=True, thickness=0.06),
    )
    fig.update_layout(**layout)
    return fig


def buat_grafik_perbandingan(chain_b, chain_o):
    hist_periode = df['periode'].tolist()
    hist_beban   = df['Beban'].tolist()

    x_b = [hist_periode[-1]] + [r['Periode'] for r in chain_b]
    y_b = [hist_beban[-1]]   + [r['_pred']   for r in chain_b]
    x_o = [hist_periode[-1]] + [r['Periode'] for r in chain_o]
    y_o = [hist_beban[-1]]   + [r['_pred']   for r in chain_o]

    fig = go.Figure()

    # Data aktual
    fig.add_trace(go.Scatter(
        x=hist_periode, y=hist_beban,
        mode='lines+markers',
        name='Data Aktual',
        line=dict(color='#2E4DB5', width=2.5, shape='spline', smoothing=0.6),
        marker=dict(size=6, color='#1B2A6B'),
        hovertemplate='<b>%{x}</b><br>Beban: %{y:,.2f} kWh<extra>Data Aktual</extra>',
    ))

    # Model Standar — oranye putus-putus
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b,
        mode='lines+markers',
        name='Model Standar',
        line=dict(color='#F5821F', width=2.5, dash='dot', shape='spline', smoothing=0.6),
        marker=dict(size=8, color='#F5821F', symbol='diamond',
                    line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Prediksi: %{y:,.2f} kWh<extra>Model Standar</extra>',
    ))

    # Model Ditingkatkan — hijau putus-putus berbeda
    fig.add_trace(go.Scatter(
        x=x_o, y=y_o,
        mode='lines+markers',
        name='Model Ditingkatkan',
        line=dict(color='#16A34A', width=2.5, dash='dash', shape='spline', smoothing=0.6),
        marker=dict(size=8, color='#16A34A', symbol='circle',
                    line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Prediksi: %{y:,.2f} kWh<extra>Model Ditingkatkan</extra>',
    ))

    # Garis pemisah
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=hist_periode[-1], x1=hist_periode[-1],
        y0=0, y1=0.94,
        line=dict(color='#CBD5E1', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        xref='x', yref='paper',
        x=hist_periode[-1], y=0.97,
        text='← Aktual · Prediksi →',
        showarrow=False,
        font=dict(size=10, color='#94A3B8'),
        xanchor='center',
    )

    layout = dict(**LAYOUT_BASE)
    layout['xaxis'] = dict(
        **LAYOUT_BASE['xaxis'],
        rangeslider=dict(visible=True, thickness=0.06),
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# HELPER: TABEL PREDIKSI
# ══════════════════════════════════════════════════════════════
def buat_tabel_prediksi(chain):
    rows     = []
    prev_val = df['Beban'].iloc[-1]
    for r in chain:
        pct = ((r['_pred'] - prev_val) / prev_val) * 100
        rows.append({
            "Bulan":          r['Periode'],
            "Prediksi (kWh)":  f"{r['_pred']:,.2f}",
            "Perubahan":      f"↑ {abs(pct):.2f}%" if pct >= 0 else f"↓ {abs(pct):.2f}%",
        })
        prev_val = r['_pred']
    return pd.DataFrame(rows)

def buat_tabel_chain(chain):
    rows = []
    for r in chain:
        rows.append({
            "Bulan":               r['Periode'],
            "Bulan lalu (kWh)":    f"{r['lag_1']:,.0f}",
            "2 bulan lalu (kWh)":  f"{r['lag_2']:,.0f}",
            "3 bulan lalu (kWh)":  f"{r['lag_3']:,.0f}",
            "Hasil prediksi (kWh)": f"{r['_pred']:,.2f}",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# HELPER: CARD MAE & MAPE
# ══════════════════════════════════════════════════════════════
def render_eval_cards(mae, mape, prefix=""):
    c1, c2 = st.columns(2)
    c1.markdown(f"""
    <div class="metric-card yellow">
        <div class="text">
            <div class="label">Rata-rata Selisih Prediksi (MAE)</div>
            <div class="value">{mae:,.2f} <span class="unit">kWh</span></div>
        </div>
    </div>
    <p class="hint-text">Rata-rata, prediksi model meleset dari nilai aktual sebesar ini.</p>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-card purple">
        <div class="text">
            <div class="label">Persentase Meleset (MAPE)</div>
            <div class="value">{mape:.2f}<span class="unit"> %</span></div>
        </div>
    </div>
    <p class="hint-text">Rata-rata prediksi model tersebut meleset {mape:.2f}% dari nilai beban aktual.</p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1> Prediksi Beban Listrik </h1>
    <p>Bandingkan dua model sekaligus, pilih bulan yang ingin diprediksi lalu lihat hasilnya</p>
</div>
""", unsafe_allow_html=True)

# Tentukan step aktif
step_aktif = 1
if st.session_state.get('opt') is not None:
    step_aktif = 3
render_step_bar(step_aktif)


# ══════════════════════════════════════════════════════════════
# FORM PILIH BULAN
# ══════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        st.markdown(
            '<p style="font-size:0.82rem;font-weight:700;color:#6B7A99;'
            'text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">'
            'Pilih bulan yang ingin diprediksi, lalu klik tombol "Prediksi Sekarang".</p>',
            unsafe_allow_html=True
        )
        pilihan_label = [b['label'] for b in bulan_list]
        selected      = st.selectbox("", pilihan_label, label_visibility="collapsed")
        selected_info = next(b for b in bulan_list if b['label'] == selected)
        n_steps       = selected_info['step']
        st.markdown(
            f'<p class="hint-text">Data terakhir yang tersedia: '
            f'{NAMA_BULAN[last_bulan]} {last_tahun}. '
            f'Model akan menghitung prediksi secara berurutan hingga bulan yang dipilih.</p>',
            unsafe_allow_html=True
        )
    with col_btn:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        run = st.button("⚡Prediksi Sekarang", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# JALANKAN PREDIKSI
# ══════════════════════════════════════════════════════════════
if run:
    render_step_bar(2)
    with st.spinner("Sedang mencari pengaturan terbaik untuk model… sebentar ya."):
        st.session_state['opt']      = jalankan_gridsearch(df)
        st.session_state['n_steps']  = n_steps
        st.session_state['selected'] = selected
    st.rerun()

opt      = st.session_state.get('opt')
n_steps  = st.session_state.get('n_steps', n_steps)
selected = st.session_state.get('selected', selected)


# ══════════════════════════════════════════════════════════════
# TAMPILKAN HASIL
# ══════════════════════════════════════════════════════════════
if opt is None:
    # ── PLACEHOLDER ──────────────────────────────────────────
    icon_b64, icon_mime = get_image_base64("icon_chart")
    icon_html = (
        f'<img src="data:{icon_mime};base64,{icon_b64}" alt="icon"/>'
        if icon_b64 else '<div class="fallback-icon">📊</div>'
    )
    st.markdown(f"""
    <div class="predict-placeholder">
        {icon_html}
        <div class="title">Hasil Prediksi Akan Muncul Di sini</div>
        <div class="Pilih bulan di atas, lalu klik tombol "Prediksi Sekarang".</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── HITUNG CHAIN ─────────────────────────────────────────
    chain_b = hitung_chain(model,              n_steps)
    chain_o = hitung_chain(opt['best_estimator'], n_steps)

    final_b    = chain_b[-1]
    final_o    = chain_o[-1]
    prev_beban = chain_b[-2]['_pred'] if n_steps > 1 else df['Beban'].iloc[-1]
    prev_label = chain_b[-2]['Periode'] if n_steps > 1 else f"{NAMA_BULAN[last_bulan]} {last_tahun}"

    delta_b = ((final_b['_pred'] - prev_beban) / prev_beban) * 100
    delta_o = ((final_o['_pred'] - prev_beban) / prev_beban) * 100

    if n_steps > 1:
        nilai_b      = [r['_pred'] for r in chain_b]
        nilai_o      = [r['_pred'] for r in chain_o]
        idx_puncak_b = int(np.argmax(nilai_b))
        idx_puncak_o = int(np.argmax(nilai_o))
        puncak_b     = chain_b[idx_puncak_b]
        puncak_o     = chain_o[idx_puncak_o]
        total_b      = sum(nilai_b)
        total_o      = sum(nilai_o)

    # ── TABS ─────────────────────────────────────────────────
    tab_standar, tab_ditingkatkan, tab_perbandingan = st.tabs([
        "📊 Model Standar",
        "⚙️ Model Ditingkatkan",
        "🔍 Perbandingan",
    ])


    # ════════════════════════════════════════════════════════
    # TAB 1 — MODEL STANDAR (BASELINE)
    # ════════════════════════════════════════════════════════
    with tab_standar:

        # Hero result card
        dc_b = "up" if delta_b >= 0 else "down"
        st.markdown(f"""
        <div class="section-card predict-result">
            <div class="eyebrow">Prediksi beban listrik · Model Standar</div>
            <div class="period">{selected}</div>
            <div class="value">{final_b['_pred']:,.2f}<span class="unit"> kWh</span></div>
            <div class="delta {dc_b}">
                {'↑' if delta_b >= 0 else '↓'} {abs(delta_b):.1f}%
                compared to the previous month ({prev_label})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MAE & MAPE cards
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📏 Seberapa akurat model ini?</div>', unsafe_allow_html=True)
        render_eval_cards(opt['mae_baseline'], opt['mape_baseline'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Ringkasan puncak & total (hanya multi-step)
        if n_steps > 1:
            mc1, mc2 = st.columns(2)
            mc1.markdown(f"""
            <div class="metric-card yellow">
                <div class="text">
                    <div class="label">Bulan dengan beban terbesar</div>
                    <div class="value">{puncak_b['_pred']:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            <p class="hint-text" style="margin-top:-10px">Diperkirakan terjadi pada {puncak_b['Periode']}</p>
            """, unsafe_allow_html=True)
            mc2.markdown(f"""
            <div class="metric-card">
                <div class="text">
                    <div class="label">Total beban seluruh periode</div>
                    <div class="value">{total_b:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            <p class="hint-text" style="margin-top:-10px">{chain_b[0]['Periode']} – {chain_b[-1]['Periode']}</p>
            """, unsafe_allow_html=True)

        # Grafik
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Grafik Prediksi Tren </div>', unsafe_allow_html=True)
        fig_b = buat_grafik_tunggal(
            chain_b, '#F5821F', 'Model Standar',
            puncak=puncak_b if n_steps > 1 else None
        )
        st.plotly_chart(fig_b, use_container_width=True)
        if n_steps == 1:
            st.markdown(
                '<p class="hint-text">💡 Garis putus-putus oranye adalah hasil prediksi, '
                'melanjutkan data aktual (batang biru).</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabel prediksi & rantai
        if n_steps > 1:
            col_tbl, col_chain = st.columns([1, 1], gap="large")
            with col_tbl:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📋 Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
                st.dataframe(buat_tabel_prediksi(chain_b), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_chain:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">🔗 Urutan Hitung Bulan per Bulan '
                    '<span class="info-tooltip" title="Karena prediksi lebih dari 1 bulan, '
                    'hasil prediksi bulan sebelumnya dipakai sebagai data masukan bulan berikutnya.">ⓘ</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="lag-callout">Model hanya bisa membaca 3 bulan terakhir. '
                    'Jika menebak lebih jauh, hasil prediksi sebelumnya otomatis dijadikan "data bulan lalu".</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(buat_tabel_chain(chain_b), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Single step: tampilkan data masukan saja
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">📥 Data yang Dipakai Model</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="lag-callout">Model membaca konsumsi listrik dari 3 bulan terakhir '
                'untuk menghasilkan prediksi bulan berikutnya.</div>',
                unsafe_allow_html=True
            )
            detail_df = pd.DataFrame({
                'Periode':       [NAMA_BULAN[last_bulan] + f' {last_tahun}',
                                  NAMA_BULAN[(last_bulan-2) % 12 + 1] + f' {last_tahun}',
                                  NAMA_BULAN[(last_bulan-3) % 12 + 1] + f' {last_tahun}'],
                'Posisi':        ['Bulan lalu', '2 bulan lalu', '3 bulan lalu'],
                'Beban (kWh)':   [f"{chain_b[0]['lag_1']:,.0f}",
                                  f"{chain_b[0]['lag_2']:,.0f}",
                                  f"{chain_b[0]['lag_3']:,.0f}"],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Info model (expander)
        tp_b = final_b['_tree_preds']
        with st.expander("ℹ️ Detail cara kerja Model Standar"):
            st.markdown(f"""
            <table class="param-table">
              <tr><td class="key">Nama model</td>
                  <td class="val">Random Forest</td></tr>
              <tr><td class="key">Jumlah pohon keputusan</td>
                  <td class="val">{model.n_estimators}</td></tr>
              <tr><td class="key">Kedalaman tiap pohon</td>
                  <td class="val">{model.max_depth if model.max_depth else 'Tidak dibatasi'}</td></tr>
              <tr><td class="key">Min. data per cabang</td>
                  <td class="val">{model.min_samples_split}</td></tr>
              <tr><td class="key">Rata-rata prediksi pohon</td>
                  <td class="val">{tp_b.mean():,.2f} kWh</td></tr>
              <tr><td class="key">Std. deviasi antar pohon</td>
                  <td class="val">{tp_b.std():,.2f} kWh</td></tr>
            </table>
            <p class="hint-text" style="margin-top:10px">
              Model ini menggunakan 100 "pohon keputusan" yang masing-masing memberikan prediksi,
              lalu semua prediksi dirata-rata menjadi hasil akhir — tanpa penyetelan tambahan.
            </p>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════
    # TAB 2 — MODEL DITINGKATKAN (OPTIMASI)
    # ════════════════════════════════════════════════════════
    with tab_ditingkatkan:

        best_params   = opt['best_params']
        model_optimal = opt['best_estimator']

        # Hero result card
        dc_o = "up" if delta_o >= 0 else "down"
        st.markdown(f"""
        <div class="section-card predict-result">
            <div class="eyebrow">Prediksi beban listrik · Model Ditingkatkan
                <span class="badge-optimized" style="margin-left:8px">Disempurnakan</span>
            </div>
            <div class="period">{selected}</div>
            <div class="value">{final_o['_pred']:,.2f}<span class="unit"> kWh</span></div>
            <div class="delta {dc_o}">
                {'↑' if delta_o >= 0 else '↓'} {abs(delta_o):.1f}%
                dibanding bulan sebelumnya ({prev_label})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MAE & MAPE cards
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📏 Seberapa Akurat Model Ini?</div>', unsafe_allow_html=True)
        render_eval_cards(opt['mae_optimasi'], opt['mape_optimasi'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Info penyetelan otomatis
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Bagaimana Model Ini Disempurnakan?</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="lag-callout">'
            f'Sistem mencoba <b>{opt["n_combinations"]} kombinasi pengaturan</b> yang berbeda '
            f'secara otomatis, lalu memilih yang menghasilkan selisih prediksi terkecil. '
            f'Proses pengujiannya dibagi menjadi <b>{N_SPLITS} putaran</b> agar hasilnya adil '
            f'dan tidak hanya kebetulan cocok dengan satu periode data.</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Ringkasan puncak & total (hanya multi-step)
        if n_steps > 1:
            mc1, mc2 = st.columns(2)
            mc1.markdown(f"""
            <div class="metric-card yellow">
                <div class="text">
                    <div class="label">Bulan dengan beban terbesar</div>
                    <div class="value">{puncak_o['_pred']:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            <p class="hint-text" style="margin-top:-10px">Diperkirakan terjadi pada {puncak_o['Periode']}</p>
            """, unsafe_allow_html=True)
            mc2.markdown(f"""
            <div class="metric-card">
                <div class="text">
                    <div class="label">Total beban seluruh periode</div>
                    <div class="value">{total_o:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            <p class="hint-text" style="margin-top:-10px">{chain_o[0]['Periode']} – {chain_o[-1]['Periode']}</p>
            """, unsafe_allow_html=True)

        # Grafik
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Grafik Tren Prediksi</div>', unsafe_allow_html=True)
        fig_o = buat_grafik_tunggal(
            chain_o, '#16A34A', 'Model Ditingkatkan',
            puncak=puncak_o if n_steps > 1 else None
        )
        st.plotly_chart(fig_o, use_container_width=True)
        if n_steps == 1:
            st.markdown(
                '<p class="hint-text">💡 Garis putus-putus hijau adalah hasil prediksi, '
                'melanjutkan data aktual (batang biru).</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabel prediksi & rantai
        if n_steps > 1:
            col_tbl2, col_chain2 = st.columns([1, 1], gap="large")
            with col_tbl2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📋 Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
                st.dataframe(buat_tabel_prediksi(chain_o), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_chain2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">🔗 Urutan Hitung Bulan per Bulan '
                    '<span class="info-tooltip" title="Sama seperti model standar — hasil prediksi '
                    'bulan sebelumnya dipakai sebagai data masukan bulan berikutnya.">ⓘ</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="lag-callout">Cara hitung yang sama dengan model standar, '
                    'hanya menggunakan pengaturan yang sudah disempurnakan.</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(buat_tabel_chain(chain_o), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📥 Data yang Dipakai Model</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="lag-callout">Model membaca konsumsi listrik dari 3 bulan terakhir '
                'untuk menghasilkan prediksi bulan berikutnya.</div>',
                unsafe_allow_html=True
            )
            detail_df2 = pd.DataFrame({
                'Periode':     [NAMA_BULAN[last_bulan] + f' {last_tahun}',
                                NAMA_BULAN[(last_bulan-2) % 12 + 1] + f' {last_tahun}',
                                NAMA_BULAN[(last_bulan-3) % 12 + 1] + f' {last_tahun}'],
                'Posisi':      ['Bulan lalu', '2 bulan lalu', '3 bulan lalu'],
                'Beban (kWh)': [f"{chain_o[0]['lag_1']:,.0f}",
                                f"{chain_o[0]['lag_2']:,.0f}",
                                f"{chain_o[0]['lag_3']:,.0f}"],
            })
            st.dataframe(detail_df2, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Info model (expander)
        tp_o = final_o['_tree_preds']
        with st.expander("ℹ️ Detail cara kerja Model Ditingkatkan"):
            st.markdown(f"""
            <table class="param-table">
              <tr><td class="key">Nama model</td>
                  <td class="val">Random Forest (disempurnakan)</td></tr>
              <tr><td class="key">Jumlah pohon keputusan</td>
                  <td class="val">{model_optimal.n_estimators}</td></tr>
              <tr><td class="key">Kedalaman tiap pohon</td>
                  <td class="val">{best_params['max_depth']}</td></tr>
              <tr><td class="key">Min. data per cabang</td>
                  <td class="val">{best_params['min_samples_split']}</td></tr>
              <tr><td class="key">Rata-rata prediksi pohon</td>
                  <td class="val">{tp_o.mean():,.2f} kWh</td></tr>
              <tr><td class="key">Std. deviasi antar pohon</td>
                  <td class="val">{tp_o.std():,.2f} kWh</td></tr>
            </table>
            <p class="hint-text" style="margin-top:10px">
              Model ini bekerja sama persis dengan model standar, bedanya pengaturannya
              sudah dipilih secara otomatis untuk menghasilkan prediksi yang lebih presisi.
            </p>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════
    # TAB 3 — PERBANDINGAN
    # ════════════════════════════════════════════════════════
    with tab_perbandingan:

        selisih            = abs(final_b['_pred'] - final_o['_pred'])
        model_lebih_rendah = "Ditingkatkan" if final_o['_pred'] <= final_b['_pred'] else "Standar"
        is_opt_winner      = final_o['_pred'] <= final_b['_pred']
        is_opt_more_acc    = opt['mape_optimasi'] <= opt['mape_baseline']
        pct_selisih        = (selisih / final_b['_pred']) * 100

        # ── HERO: 3 kartu hasil prediksi ─────────────────────
        ka, kb, kc = st.columns(3)
        ka.markdown(f"""
        <div class="section-card" style="text-align:center;padding:20px 16px">
            <div style="font-size:.7rem;font-weight:700;color:#6B7A99;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:6px">
                Model Standar&nbsp;<span class="badge-baseline">Standar</span>
            </div>
            <div style="font-size:2rem;font-weight:800;color:#1B2A6B;line-height:1.1">
                {final_b['_pred']:,.2f}
            </div>
            <div style="font-size:.82rem;color:#6B7A99;margin-top:2px">kWh</div>
            <div style="font-size:.78rem;color:#F5821F;font-weight:600;margin-top:8px">
                {"\u2191" if delta_b >= 0 else "\u2193"} {abs(delta_b):.1f}% vs bulan lalu
            </div>
        </div>
        """, unsafe_allow_html=True)

        winner_style = (
            "border:1.5px solid #86EFAC;background:linear-gradient(160deg,#fff 0%,#F0FDF4 100%)"
            if is_opt_winner else "border:1.5px solid #E8EDF8"
        )
        winner_mark = "&nbsp;&#10003; Lebih rendah" if is_opt_winner else ""
        kb.markdown(f"""
        <div class="section-card" style="text-align:center;padding:20px 16px;{winner_style}">
            <div style="font-size:.7rem;font-weight:700;color:#6B7A99;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:6px">
                Model Ditingkatkan&nbsp;<span class="badge-optimized">Disempurnakan</span>
            </div>
            <div style="font-size:2rem;font-weight:800;color:#1B2A6B;line-height:1.1">
                {final_o['_pred']:,.2f}
            </div>
            <div style="font-size:.82rem;color:#6B7A99;margin-top:2px">kWh</div>
            <div style="font-size:.78rem;color:#16A34A;font-weight:600;margin-top:8px">
                {"\u2191" if delta_o >= 0 else "\u2193"} {abs(delta_o):.1f}% vs bulan lalu{winner_mark}
            </div>
        </div>
        """, unsafe_allow_html=True)

        kc.markdown(f"""
        <div class="section-card" style="text-align:center;padding:20px 16px">
            <div style="font-size:.7rem;font-weight:700;color:#6B7A99;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:6px">Selisih Kedua Model</div>
            <div style="font-size:2rem;font-weight:800;color:#F5821F;line-height:1.1">
                {selisih:,.2f}
            </div>
            <div style="font-size:.82rem;color:#6B7A99;margin-top:2px">kWh</div>
            <div style="font-size:.78rem;color:#6B7A99;font-weight:600;margin-top:8px">
                ({pct_selisih:.2f}% dari prediksi standar)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── GRAFIK (selalu tampil, termasuk n_steps == 1) ────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">&#128202; Grafik Perbandingan Kedua Model</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(buat_grafik_perbandingan(chain_b, chain_o), use_container_width=True)
        if n_steps == 1:
            st.markdown(
                '<p class="hint-text">&#128161; Titik berlian oranye = Model Standar, '                'titik lingkaran hijau = Model Ditingkatkan. '                'Geser range di bawah grafik untuk zoom ke area prediksi.</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── AKURASI: 2 kolom Standar vs Ditingkatkan ─────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">&#128207; Perbandingan Akurasi Kedua Model</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="hint-text" style="margin-bottom:16px">'            'Dihitung dari data uji yang tidak dipakai saat pelatihan model.</p>',
            unsafe_allow_html=True
        )

        col_acc_b, col_acc_sep, col_acc_o = st.columns([5, 1, 5])

        with col_acc_b:
            st.markdown(
                '<div style="font-size:.72rem;font-weight:700;color:#6B7A99;'                'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">'                'Model Standar</div>',
                unsafe_allow_html=True
            )
            ab1, ab2 = st.columns(2)
            ab1.markdown(f"""
            <div class="metric-card yellow" style="margin-bottom:0">
                <div class="text">
                    <div class="label">MAE</div>
                    <div class="value">{opt['mae_baseline']:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            ab2.markdown(f"""
            <div class="metric-card" style="margin-bottom:0">
                <div class="text">
                    <div class="label">MAPE</div>
                    <div class="value">{opt['mape_baseline']:.2f} <span class="unit">%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_acc_sep:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'                'height:80px;font-size:.75rem;font-weight:700;color:#8A96B5">VS</div>',
                unsafe_allow_html=True
            )

        with col_acc_o:
            better = ' <span style="color:#16A34A;font-size:.68rem">&#10003; Lebih akurat</span>' \
                     if is_opt_more_acc else ''
            st.markdown(
                f'<div style="font-size:.72rem;font-weight:700;color:#6B7A99;'                f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">'                f'Model Ditingkatkan{better}</div>',
                unsafe_allow_html=True
            )
            ao1, ao2 = st.columns(2)
            mae_cls  = "green" if is_opt_more_acc else "yellow"
            mape_cls = "green" if is_opt_more_acc else ""
            ao1.markdown(f"""
            <div class="metric-card {mae_cls}" style="margin-bottom:0">
                <div class="text">
                    <div class="label">MAE</div>
                    <div class="value">{opt['mae_optimasi']:,.2f} <span class="unit">kWh</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            ao2.markdown(f"""
            <div class="metric-card {mape_cls}" style="margin-bottom:0">
                <div class="text">
                    <div class="label">MAPE</div>
                    <div class="value">{opt['mape_optimasi']:.2f} <span class="unit">%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── RINGKASAN ─────────────────────────────────────────
        pill_mape_b = f'<span class="compare-pill yellow">{opt["mape_baseline"]:.2f}%</span>'
        pill_mape_o = f'<span class="compare-pill green">{opt["mape_optimasi"]:.2f}%</span>'
        acc_text = (
            f"Model Ditingkatkan lebih presisi dengan MAPE {pill_mape_o} "
            f"dibanding {pill_mape_b} pada model standar."
            if is_opt_more_acc else
            f"Kedua model memiliki keakuratan yang hampir setara "
            f"(MAPE Standar: {pill_mape_b}, Ditingkatkan: {pill_mape_o})."
        )
        st.markdown(f"""
<div class="compare-summary">
<p>
Untuk bulan <b>{selected}</b>, model standar memperkirakan
<span class="compare-pill orange">{final_b['_pred']:,.2f} kWh</span>
dan model yang telah disempurnakan memperkirakan
<span class="compare-pill green">{final_o['_pred']:,.2f} kWh</span>
— selisih <span class="compare-pill yellow">{selisih:,.2f} kWh ({pct_selisih:.2f}%)</span>.
</p>
<p>{acc_text}</p>
<div class="compare-cta">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
Untuk analisis performa model secara lengkap, buka halaman <b>Analisis &amp; Kesimpulan</b>.
</div>
</div>
""", unsafe_allow_html=True)