# ── 3_Prediksi_Optimasi.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_css, render_sidebar, get_image_base64

st.set_page_config(page_title="Prediksi", page_icon="⚡", layout="wide")
load_css()
render_sidebar()

# ── CSS KHUSUS DASHBOARD & PREDIKSI ──────────────────────────
with open(Path(__file__).parent.parent / "styles" / "dashboard.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open(Path(__file__).parent.parent / "styles" / "prediksi.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── LOAD MODEL & DATA ─────────────────────────────────────────
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

nama_bulan_map = {
    1:'Januari', 2:'Februari', 3:'Maret', 4:'April',
    5:'Mei', 6:'Juni', 7:'Juli', 8:'Agustus',
    9:'September', 10:'Oktober', 11:'November', 12:'Desember'
}

# Generate 12 bulan ke depan
bulan_list = []
for i in range(1, 13):
    b = last_bulan + i
    t = last_tahun
    if b > 12:
        b -= 12
        t += 1
    bulan_list.append({'label': f"{nama_bulan_map[b]} {t}", 'tahun': t, 'bulan': b, 'step': i})

# ── PARAM GRID UNTUK GRIDSEARCHCV ─────────────────────────────
PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
}
N_SPLITS = 3

@st.cache_resource(show_spinner=False)
def jalankan_gridsearch(_df):
    """
    Optimasi Random Forest Regressor menggunakan GridSearchCV
    dengan validasi TimeSeriesSplit (n_splits=3),
    mengikuti pembagian data train-test 80/20 seperti pada notebook.
    """
    X = _df[['lag_1', 'lag_2', 'lag_3']]
    y = _df['Beban']

    split   = int(len(_df) * 0.8)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_index  = grid.best_index_
    cv_results  = grid.cv_results_
    fold_scores = [
        -cv_results[f'split{i}_test_score'][best_index]
        for i in range(N_SPLITS)
    ]

    # Latih ulang model terbaik pada seluruh X_train, evaluasi MAPE pada X_test
    best_model = RandomForestRegressor(random_state=42, **grid.best_params_)
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    mape_test   = (abs((y_test.values - y_pred_test) / y_test.values)).mean() * 100

    return {
        'best_estimator': best_model,
        'best_params':    grid.best_params_,
        'best_mae':       -grid.best_score_,
        'best_mape':      mape_test,
        'fold_scores':    fold_scores,
        'n_combinations': len(cv_results['params']),
    }

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Prediksi Beban Listrik</h1>
    <p>Pilih bulan yang ingin diprediksi menggunakan model Random Forest Regressor</p>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT DUA KOLOM ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Pilih Bulan Prediksi</div>', unsafe_allow_html=True)

    pilihan_label = [b['label'] for b in bulan_list]
    selected      = st.selectbox("", pilihan_label, label_visibility="collapsed")
    selected_info = next(b for b in bulan_list if b['label'] == selected)
    n_steps       = selected_info['step']

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Prediksi", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── JALANKAN GRIDSEARCHCV (SEKALI, SAAT TOMBOL DIKLIK) ───────
if run:
    with st.spinner("Menjalankan GridSearchCV dengan TimeSeriesSplit (3 fold)..."):
        st.session_state['opt'] = jalankan_gridsearch(df)

opt = st.session_state.get('opt')

with left:
    # Info model — dinamis: tampilkan hasil tuning jika sudah ada, default jika belum
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    if opt is not None:
        bp = opt['best_params']
        st.markdown(
            '<div class="section-title">🤖 Parameter Model '
            '<span class="badge-optimized">Hasil Tuning</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <table class="param-table">
          <tr>
            <td class="key">Algoritma</td>
            <td class="val">Random Forest Regressor (Tuned)</td>
          </tr>
          <tr>
            <td class="key">n_estimators</td>
            <td class="val">{bp["n_estimators"]}</td>
          </tr>
          <tr>
            <td class="key">max_depth</td>
            <td class="val">{bp["max_depth"]}</td>
          </tr>
          <tr>
            <td class="key">min_samples_split</td>
            <td class="val">{bp["min_samples_split"]}</td>
          </tr>
          <tr>
            <td class="key">Fitur Input</td>
            <td class="val">lag_1, lag_2, lag_3</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">🤖 Parameter Model</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="param-table">
          <tr>
            <td class="key">Algoritma</td>
            <td class="val">Random Forest Regressor</td>
          </tr>
          <tr>
            <td class="key">n_estimators</td>
            <td class="val">{model.n_estimators}</td>
          </tr>
          <tr>
            <td class="key">max_depth</td>
            <td class="val">{model.max_depth if model.max_depth else 'None (tidak dibatasi)'}</td>
          </tr>
          <tr>
            <td class="key">min_samples_split</td>
            <td class="val">{model.min_samples_split}</td>
          </tr>
          <tr>
            <td class="key">Fitur Input</td>
            <td class="val">lag_1, lag_2, lag_3</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if opt is not None:
        model_optimal = opt['best_estimator']
        best_params   = opt['best_params']

        # ── SECTION: PROSES OPTIMASI MODEL ────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛠️ Optimasi Model — GridSearchCV + TimeSeriesSplit</div>', unsafe_allow_html=True)
        st.markdown(
            f"Pencarian kombinasi parameter terbaik dari **{opt['n_combinations']} kombinasi** "
            f"menggunakan validasi **TimeSeriesSplit (n_splits={N_SPLITS})**, "
            f"dengan metrik evaluasi **MAE (Mean Absolute Error)**."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Param grid yang diuji
        st.markdown('<div class="model-info-title">Grid Parameter yang Diuji</div>', unsafe_allow_html=True)
        grid_df = pd.DataFrame({
            'Parameter': list(PARAM_GRID.keys()),
            'Kandidat Nilai': [str(v) for v in PARAM_GRID.values()],
        })
        st.dataframe(grid_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Best params
        bp1, bp2, bp3, bp4, bp5 = st.columns(5)
        bp1.markdown(
            f'<div class="model-info-card highlight"><div class="label">n_estimators</div>'
            f'<div class="value">{best_params["n_estimators"]}</div></div>',
            unsafe_allow_html=True
        )
        bp2.markdown(
            f'<div class="model-info-card highlight"><div class="label">max_depth</div>'
            f'<div class="value">{best_params["max_depth"]}</div></div>',
            unsafe_allow_html=True
        )
        bp3.markdown(
            f'<div class="model-info-card highlight"><div class="label">min_samples_split</div>'
            f'<div class="value">{best_params["min_samples_split"]}</div></div>',
            unsafe_allow_html=True
        )
        bp4.markdown(
            f'<div class="model-info-card highlight"><div class="label">MAE Rata-rata CV</div>'
            f'<div class="value">{opt["best_mae"]:,.2f}</div></div>',
            unsafe_allow_html=True
        )
        bp5.markdown(
            f'<div class="model-info-card highlight"><div class="label">MAPE (Test Set)</div>'
            f'<div class="value">{opt["best_mape"]:.2f}%</div></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Skor per fold
        st.markdown('<div class="model-info-title">MAE per Fold (TimeSeriesSplit)</div>', unsafe_allow_html=True)
        fold_df = pd.DataFrame({
            'Fold': [f"Fold {i+1}" for i in range(N_SPLITS)],
            'MAE': [f"{score:,.2f}" for score in opt['fold_scores']],
        })
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── PREDIKSI BERANTAI (MENGGUNAKAN MODEL HASIL OPTIMASI) ──
        history = list(df['Beban'].tail(3).values[::-1])  # [bulan-1, bulan-2, bulan-3]

        chain_results = []
        cur_bulan = last_bulan
        cur_tahun = last_tahun

        for step in range(1, n_steps + 1):
            lag1 = float(history[0])
            lag2 = float(history[1])
            lag3 = float(history[2])

            input_df   = pd.DataFrame([[lag1, lag2, lag3]], columns=['lag_1','lag_2','lag_3'])
            tree_preds = np.array([t.predict(input_df)[0] for t in model_optimal.estimators_])
            pred       = tree_preds.mean()

            cur_bulan += 1
            if cur_bulan > 12:
                cur_bulan = 1
                cur_tahun += 1

            chain_results.append({
                'Periode':        f"{nama_bulan_map[cur_bulan]} {cur_tahun}",
                'lag_1':          f"{lag1:,.0f}",
                'lag_2':          f"{lag2:,.0f}",
                'lag_3':          f"{lag3:,.0f}",
                'Prediksi (kWh)': round(pred, 2),
                '_pred':          pred,
                '_tree_preds':    tree_preds,
            })

            history = [pred, history[0], history[1]]

        final        = chain_results[-1]
        hasil        = final['_pred']
        tree_preds   = final['_tree_preds']
        beban_prev   = chain_results[-2]['_pred'] if n_steps > 1 else df['Beban'].iloc[-1]
        delta        = ((hasil - beban_prev) / beban_prev) * 100
        delta_icon   = "↑" if delta >= 0 else "↓"
        prev_label   = chain_results[-2]['Periode'] if n_steps > 1 else f"{nama_bulan_map[last_bulan]} {last_tahun}"

        # ── HASIL UTAMA ───────────────────────────────────────
        delta_class = "up" if delta >= 0 else "down"
        st.markdown(f"""
        <div class="section-card predict-result">
            <div class="eyebrow">Prediksi Beban Listrik untuk</div>
            <div class="period">{selected}</div>
            <div class="value">
                {hasil:,.2f}
                <span class="unit"> kWh</span>
            </div>
            <div class="delta {delta_class}">
                <span>{delta_icon} {abs(delta):.1f}% dibanding bulan sebelumnya ({prev_label})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TABEL RANTAI PREDIKSI (jika > 1 langkah) ─────────
        if n_steps > 1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔗 Rantai Prediksi Berantai</div>', unsafe_allow_html=True)
            st.markdown("Setiap hasil prediksi digunakan sebagai lag untuk bulan berikutnya.")
            tabel_chain = pd.DataFrame([{
                'Periode':        r['Periode'],
                'lag_1 (kWh)':    r['lag_1'],
                'lag_2 (kWh)':    r['lag_2'],
                'lag_3 (kWh)':    r['lag_3'],
                'Prediksi (kWh)': f"{r['Prediksi (kWh)']:,.2f}"
            } for r in chain_results])
            st.dataframe(tabel_chain, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── PROSES RF (MODEL HASIL OPTIMASI) ──────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌲 Proses Random Forest (Model Hasil Optimasi)</div>', unsafe_allow_html=True)
        st.markdown(f"Hasil akhir merupakan rata-rata dari **{model_optimal.n_estimators} pohon keputusan**.")

        p1, p2, p3 = st.columns(3)
        p1.markdown(
            f'<div class="metric-card"><div class="text"><div class="label">Jumlah Pohon</div>'
            f'<div class="value">{model_optimal.n_estimators}</div></div></div>',
            unsafe_allow_html=True
        )
        p2.markdown(
            f'<div class="metric-card purple"><div class="text"><div class="label">Rata-rata</div>'
            f'<div class="value">{tree_preds.mean():,.2f}</div></div></div>',
            unsafe_allow_html=True
        )
        p3.markdown(
            f'<div class="metric-card green"><div class="text"><div class="label">Std. Deviasi</div>'
            f'<div class="value">{tree_preds.std():,.2f}</div></div></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        icon_b64, icon_mime = get_image_base64("icon_chart")
        icon_html = (
            f'<img src="data:{icon_mime};base64,{icon_b64}" alt="icon"/>'
            if icon_b64 else '<div class="fallback-icon">📊</div>'
        )
        st.markdown(f"""
        <div class="section-card predict-placeholder">
            {icon_html}
            <div class="title">Hasil prediksi akan muncul di sini</div>
            <div class="subtitle">Pilih bulan di sebelah kiri lalu klik tombol Prediksi</div>
        </div>
        """, unsafe_allow_html=True)