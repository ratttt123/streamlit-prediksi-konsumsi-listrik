import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_css, render_sidebar

st.set_page_config(page_title="Analisis & Kesimpulan", page_icon="📊", layout="wide")
load_css()
render_sidebar()

with open(Path(__file__).parent.parent / "styles" / "dashboard.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open(Path(__file__).parent.parent / "styles" / "prediksi.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open(Path(__file__).parent.parent / "styles" / "evaluasi.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD DATA HISTORIS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_excel("data/data_listrik_clean.xlsx")

df = load_data()
df['periode'] = df['periode'].astype(str)

NAMA_BULAN = {
    1:'Januari', 2:'Februari', 3:'Maret',    4:'April',
    5:'Mei',     6:'Juni',     7:'Juli',      8:'Agustus',
    9:'September',10:'Oktober',11:'November', 12:'Desember'
}


# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1><svg style="vertical-align:-3px;margin-right:6px" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/></svg>Analisis &amp; Kesimpulan</h1>
    <p>Analisis mendalam hasil prediksi beserta perbandingan performa kedua model</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CEK SESSION STATE — tampilkan placeholder jika belum prediksi
# ══════════════════════════════════════════════════════════════
opt      = st.session_state.get('opt')
n_steps  = st.session_state.get('n_steps')
selected = st.session_state.get('selected')

if opt is None:
    st.markdown("""
    <div class="predict-placeholder" style="margin-top:16px">
        <div class="fallback-icon"><svg style="vertical-align:-3px;margin-right:6px" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/></svg></div>
        <div class="title">Belum ada hasil prediksi</div>
        <div class="subtitle">
            Jalankan prediksi terlebih dahulu di halaman <b>Prediksi</b>,
            lalu kembali ke halaman ini untuk melihat analisis lengkapnya.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
# AMBIL DATA DARI SESSION STATE
# ══════════════════════════════════════════════════════════════
import joblib
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_model():
    return joblib.load("model/model_rf.pkl")

model = load_model()

NAMA_BULAN_SINGKAT = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun',
    7:'Jul', 8:'Agt', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'
}

last_row   = df.iloc[-1]
last_tahun = int(last_row['Tahun'])
last_bulan = int(last_row['Bulan'])

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
            'Periode': f"{NAMA_BULAN[cur_bulan]} {cur_tahun}",
            'Tahun': cur_tahun, 'Bulan': cur_bulan,
            'lag_1': lag1, 'lag_2': lag2, 'lag_3': lag3,
            '_pred': pred,
        })
        history = [pred, history[0], history[1]]
    return results

chain_b = hitung_chain(model,                  n_steps)
chain_o = hitung_chain(opt['best_estimator'],  n_steps)

final_b    = chain_b[-1]
final_o    = chain_o[-1]
nilai_b    = [r['_pred'] for r in chain_b]
nilai_o    = [r['_pred'] for r in chain_o]
selisih    = abs(final_b['_pred'] - final_o['_pred'])
pct_selisih = (selisih / final_b['_pred']) * 100

mae_b   = opt['mae_baseline']
mape_b  = opt['mape_baseline']
mae_o   = opt['mae_optimasi']
mape_o  = opt['mape_optimasi']

is_opt_more_acc    = mape_o <= mape_b
model_terbaik_nama = "Model Ditingkatkan" if is_opt_more_acc else "Model Standar"
best_mae           = mae_o  if is_opt_more_acc else mae_b
best_mape          = mape_o if is_opt_more_acc else mape_b

tanggal_analisis = datetime.now().strftime("%d %B %Y, %H:%M")


# ══════════════════════════════════════════════════════════════
# HELPER: GRAFIK (konsisten dengan prediksi.py)
# ══════════════════════════════════════════════════════════════
LAYOUT_BASE = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    xaxis=dict(
        tickangle=-45, gridcolor='#F0F4FF', type='category',
        tickfont=dict(size=11, color='#6B7A99'),
    ),
    yaxis=dict(
        gridcolor='#F0F4FF',
        title=dict(text='Beban Listrik (kWh)', font=dict(size=12, color='#6B7A99')),
        tickfont=dict(size=11, color='#6B7A99'),
    ),
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1, font=dict(size=12), bgcolor='rgba(0,0,0,0)',
    ),
    hovermode='x unified',
    margin=dict(l=10, r=10, t=10, b=10), height=400,
)

def buat_grafik_analisis():
    """Grafik historis + kedua prediksi + rangeslider."""
    hist_periode = df['periode'].tolist()
    hist_beban   = df['Beban'].tolist()
    pred_periode = [r['Periode'] for r in chain_b]

    x_b = [hist_periode[-1]] + pred_periode
    y_b = [hist_beban[-1]]   + nilai_b
    x_o = [hist_periode[-1]] + [r['Periode'] for r in chain_o]
    y_o = [hist_beban[-1]]   + nilai_o

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_periode, y=hist_beban, mode='lines+markers',
        name='Data Aktual',
        line=dict(color='#2E4DB5', width=2.5, shape='spline', smoothing=0.6),
        marker=dict(size=6, color='#1B2A6B'),
        hovertemplate='<b>%{x}</b><br>Aktual: %{y:,.2f} kWh<extra>Data Aktual</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b, mode='lines+markers',
        name='Model Standar',
        line=dict(color='#F5821F', width=2.5, dash='dot', shape='spline', smoothing=0.6),
        marker=dict(size=8, color='#F5821F', symbol='diamond',
                    line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Model Standar: %{y:,.2f} kWh<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_o, y=y_o, mode='lines+markers',
        name='Model Ditingkatkan',
        line=dict(color='#16A34A', width=2.5, dash='dash', shape='spline', smoothing=0.6),
        marker=dict(size=8, color='#16A34A', symbol='circle',
                    line=dict(color='white', width=1.5)),
        hovertemplate='<b>%{x}</b><br>Model Ditingkatkan: %{y:,.2f} kWh<extra></extra>',
    ))
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=hist_periode[-1], x1=hist_periode[-1], y0=0, y1=0.94,
        line=dict(color='#CBD5E1', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        xref='x', yref='paper', x=hist_periode[-1], y=0.97,
        text='← Aktual · Prediksi →', showarrow=False,
        font=dict(size=10, color='#94A3B8'), xanchor='center',
    )
    layout = dict(**LAYOUT_BASE)
    layout['xaxis'] = dict(**LAYOUT_BASE['xaxis'],
                            rangeslider=dict(visible=True, thickness=0.06))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# HELPER: GENERATE EXCEL LAPORAN
# ══════════════════════════════════════════════════════════════
def buat_excel_laporan():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:

        # Sheet 1: Ringkasan
        ringkasan = pd.DataFrame({
            'Keterangan': [
                'Tanggal Analisis',
                'Bulan yang Diprediksi',
                'Jumlah Bulan ke Depan',
                'Model Terbaik',
                'MAE Model Standar (kWh)',
                'MAPE Model Standar (%)',
                'MAE Model Ditingkatkan (kWh)',
                'MAPE Model Ditingkatkan (%)',
                'Prediksi Akhir — Model Standar (kWh)',
                'Prediksi Akhir — Model Ditingkatkan (kWh)',
                'Selisih Kedua Model (kWh)',
            ],
            'Nilai': [
                tanggal_analisis,
                selected,
                n_steps,
                model_terbaik_nama,
                f"{mae_b:,.2f}",
                f"{mape_b:.2f}",
                f"{mae_o:,.2f}",
                f"{mape_o:.2f}",
                f"{final_b['_pred']:,.2f}",
                f"{final_o['_pred']:,.2f}",
                f"{selisih:,.2f}",
            ]
        })
        ringkasan.to_excel(writer, sheet_name='Ringkasan', index=False)

        # Sheet 2: Hasil Prediksi Model Standar
        rows_b = []
        prev = df['Beban'].iloc[-1]
        for r in chain_b:
            pct = ((r['_pred'] - prev) / prev) * 100
            rows_b.append({
                'Bulan': r['Periode'],
                'Bulan lalu (kWh)': f"{r['lag_1']:,.0f}",
                '2 bulan lalu (kWh)': f"{r['lag_2']:,.0f}",
                '3 bulan lalu (kWh)': f"{r['lag_3']:,.0f}",
                'Prediksi (kWh)': round(r['_pred'], 2),
                'Perubahan (%)': f"{'↑' if pct >= 0 else '↓'} {abs(pct):.2f}%",
            })
            prev = r['_pred']
        pd.DataFrame(rows_b).to_excel(writer, sheet_name='Prediksi Model Standar', index=False)

        # Sheet 3: Hasil Prediksi Model Ditingkatkan
        rows_o = []
        prev = df['Beban'].iloc[-1]
        for r in chain_o:
            pct = ((r['_pred'] - prev) / prev) * 100
            rows_o.append({
                'Bulan': r['Periode'],
                'Bulan lalu (kWh)': f"{r['lag_1']:,.0f}",
                '2 bulan lalu (kWh)': f"{r['lag_2']:,.0f}",
                '3 bulan lalu (kWh)': f"{r['lag_3']:,.0f}",
                'Prediksi (kWh)': round(r['_pred'], 2),
                'Perubahan (%)': f"{'↑' if pct >= 0 else '↓'} {abs(pct):.2f}%",
            })
            prev = r['_pred']
        pd.DataFrame(rows_o).to_excel(writer, sheet_name='Prediksi Model Ditingkatkan', index=False)

        # Sheet 4: Data Historis
        df[['periode','Tahun','Bulan','Beban','lag_1','lag_2','lag_3']].to_excel(
            writer, sheet_name='Data Historis', index=False
        )

    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# HELPER: GENERATE PDF LAPORAN
# ══════════════════════════════════════════════════════════════
def buat_pdf_laporan():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    NAVY   = colors.HexColor('#1B2A6B')
    ORANGE = colors.HexColor('#F5821F')
    GREEN  = colors.HexColor('#16A34A')
    GRAY   = colors.HexColor('#6B7A99')
    LIGHT  = colors.HexColor('#F0F4FF')

    style_title = ParagraphStyle(
        'Title2', parent=styles['Title'],
        textColor=NAVY, fontSize=18, spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        textColor=GRAY, fontSize=10, spaceAfter=16,
    )
    style_h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        textColor=NAVY, fontSize=13, spaceBefore=14, spaceAfter=6,
    )
    style_body = ParagraphStyle(
        'Body2', parent=styles['Normal'],
        fontSize=10, leading=16, textColor=colors.HexColor('#374151'),
    )
    style_small = ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontSize=9, textColor=GRAY,
    )

    def tbl_style(header_bg=NAVY, alt=LIGHT):
        return TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), header_bg),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('TOPPADDING',    (0,0), (-1,0), 8),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, alt]),
            ('FONTSIZE',      (0,1), (-1,-1), 9),
            ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#E2E8F0')),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('TOPPADDING',    (0,1), (-1,-1), 6),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ])

    story = []

    # ── COVER ─────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Laporan Analisis Prediksi", style_title))
    story.append(Paragraph("Beban Listrik Bulanan — PT Inti Bumi Perkasa", style_subtitle))
    story.append(HRFlowable(width='100%', thickness=2, color=NAVY, spaceAfter=12))

    meta = [
        ['Tanggal Laporan', tanggal_analisis],
        ['Periode Prediksi', selected],
        ['Jumlah Bulan ke Depan', str(n_steps)],
        ['Model Terbaik', model_terbaik_nama],
    ]
    t_meta = Table([[Paragraph(k, style_body), Paragraph(v, style_body)] for k, v in meta],
                   colWidths=[6*cm, 10*cm])
    t_meta.setStyle(TableStyle([
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#E2E8F0')),
        ('BACKGROUND',    (0,0), (0,-1), LIGHT),
        ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 0.6*cm))

    # ── BAGIAN 1: HASIL PREDIKSI ───────────────────────────────
    story.append(Paragraph("1. Hasil Prediksi", style_h2))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=8))

    pred_data = [['Model', 'Prediksi (kWh)', 'Perubahan vs Bulan Lalu']]
    prev = df['Beban'].iloc[-1]
    delta_b_val = ((final_b['_pred'] - prev) / prev) * 100
    delta_o_val = ((final_o['_pred'] - prev) / prev) * 100
    pred_data.append([
        'Model Standar',
        f"{final_b['_pred']:,.2f}",
        f"{'naik' if delta_b_val >= 0 else 'turun'} {abs(delta_b_val):.1f}%",
    ])
    pred_data.append([
        'Model Ditingkatkan',
        f"{final_o['_pred']:,.2f}",
        f"{'naik' if delta_o_val >= 0 else 'turun'} {abs(delta_o_val):.1f}%",
    ])
    pred_data.append(['Selisih Kedua Model', f"{selisih:,.2f}", f"{pct_selisih:.2f}%"])

    t_pred = Table(pred_data, colWidths=[6*cm, 5*cm, 5*cm])
    t_pred.setStyle(tbl_style())
    story.append(t_pred)
    story.append(Spacer(1, 0.5*cm))

    # ── BAGIAN 2: AKURASI MODEL ────────────────────────────────
    story.append(Paragraph("2. Perbandingan Akurasi Model", style_h2))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=8))
    story.append(Paragraph(
        "Metrik dihitung pada data uji (test set) yang tidak digunakan saat pelatihan model.",
        style_small
    ))
    story.append(Spacer(1, 0.3*cm))

    acc_data = [
        ['Metrik', 'Model Standar', 'Model Ditingkatkan', 'Lebih Baik'],
        ['MAE (kWh)', f"{mae_b:,.2f}", f"{mae_o:,.2f}",
         'Ditingkatkan' if mae_o <= mae_b else 'Standar'],
        ['MAPE (%)', f"{mape_b:.2f}%", f"{mape_o:.2f}%",
         'Ditingkatkan' if mape_o <= mape_b else 'Standar'],
    ]
    t_acc = Table(acc_data, colWidths=[4*cm, 4*cm, 4.5*cm, 3.5*cm])
    t_acc.setStyle(tbl_style())
    story.append(t_acc)
    story.append(Spacer(1, 0.5*cm))

    # ── BAGIAN 3: DETAIL PREDIKSI BULANAN ─────────────────────
    if n_steps > 1:
        story.append(Paragraph("3. Rincian Prediksi Bulan per Bulan", style_h2))
        story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=8))

        # Model Standar
        story.append(Paragraph("Model Standar", ParagraphStyle(
            'H3', parent=styles['Normal'],
            textColor=ORANGE, fontSize=10, fontName='Helvetica-Bold', spaceAfter=4,
        )))
        det_b = [['Bulan', 'Prediksi (kWh)', 'Perubahan']]
        prev = df['Beban'].iloc[-1]
        for r in chain_b:
            pct = ((r['_pred'] - prev) / prev) * 100
            det_b.append([r['Periode'], f"{r['_pred']:,.2f}",
                           f"{'naik' if pct >= 0 else 'turun'} {abs(pct):.2f}%"])
            prev = r['_pred']
        t_det_b = Table(det_b, colWidths=[5*cm, 5*cm, 6*cm])
        t_det_b.setStyle(tbl_style(header_bg=ORANGE))
        story.append(t_det_b)
        story.append(Spacer(1, 0.4*cm))

        # Model Ditingkatkan
        story.append(Paragraph("Model Ditingkatkan", ParagraphStyle(
            'H3g', parent=styles['Normal'],
            textColor=GREEN, fontSize=10, fontName='Helvetica-Bold', spaceAfter=4,
        )))
        det_o = [['Bulan', 'Prediksi (kWh)', 'Perubahan']]
        prev = df['Beban'].iloc[-1]
        for r in chain_o:
            pct = ((r['_pred'] - prev) / prev) * 100
            det_o.append([r['Periode'], f"{r['_pred']:,.2f}",
                           f"{'naik' if pct >= 0 else 'turun'} {abs(pct):.2f}%"])
            prev = r['_pred']
        t_det_o = Table(det_o, colWidths=[5*cm, 5*cm, 6*cm])
        t_det_o.setStyle(tbl_style(header_bg=GREEN))
        story.append(t_det_o)
        story.append(Spacer(1, 0.5*cm))

    # ── BAGIAN 4: KESIMPULAN ───────────────────────────────────
    section_num = 4 if n_steps > 1 else 3
    story.append(Paragraph(f"{section_num}. Kesimpulan", style_h2))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=8))

    selisih_mape = abs(mape_b - mape_o)
    if selisih_mape < 1.0:
        catatan_selisih = (
            "Selisih akurasi yang sangat kecil ini mengindikasikan bahwa proses penyetelan "
            "parameter tidak memberikan peningkatan yang signifikan, kemungkinan karena "
            "ukuran dataset yang relatif kecil (data bulanan 5 tahun)."
        )
    else:
        catatan_selisih = (
            "Proses penyetelan parameter secara otomatis memberikan perbedaan akurasi "
            "yang cukup terlihat dibanding model standar."
        )

    kesimpulan = (
        f"Berdasarkan hasil evaluasi, {model_terbaik_nama} menunjukkan performa terbaik "
        f"dengan MAE {best_mae:,.2f} kWh dan MAPE {best_mape:.2f}%. "
        f"Untuk periode prediksi {selected}, model standar memperkirakan "
        f"{final_b['_pred']:,.2f} kWh sedangkan model yang telah disempurnakan "
        f"memperkirakan {final_o['_pred']:,.2f} kWh "
        f"(selisih {selisih:,.2f} kWh atau {pct_selisih:.2f}%). "
        f"{catatan_selisih}"
    )
    story.append(Paragraph(kesimpulan, style_body))
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=NAVY, spaceAfter=8))
    story.append(Paragraph(
        f"Laporan dibuat otomatis oleh Aplikasi Prediksi Beban Listrik PT Inti Bumi Perkasa — {tanggal_analisis}",
        style_small
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# KONTEN HALAMAN
# ══════════════════════════════════════════════════════════════

# ── INFO KONTEKS ──────────────────────────────────────────────
st.markdown(f"""
<div class="lag-callout" style="margin-bottom:20px">
    <b>Analisis berdasarkan prediksi yang dijalankan pada {tanggal_analisis}</b> —
    Periode: <b>{selected}</b> ({n_steps} bulan ke depan).
    Untuk memperbarui analisis, jalankan prediksi baru di halaman Prediksi.
</div>
""", unsafe_allow_html=True)

# ── TOMBOL DOWNLOAD ───────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Unduh Laporan</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint-text" style="margin-bottom:14px">'
    'Unduh hasil analisis ini sebagai file Excel (data lengkap) atau PDF (ringkasan laporan).</p>',
    unsafe_allow_html=True
)
dl_col1, dl_col2, _ = st.columns([1, 1, 2])
with dl_col1:
    excel_bytes = buat_excel_laporan()
    st.download_button(
        label="Unduh Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"laporan_prediksi_{selected.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with dl_col2:
    pdf_bytes = buat_pdf_laporan()
    st.download_button(
        label="Unduh PDF (.pdf)",
        data=pdf_bytes,
        file_name=f"laporan_prediksi_{selected.replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
st.markdown('</div>', unsafe_allow_html=True)


# ── RINGKASAN METRIK ──────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>Ringkasan Hasil Prediksi</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"""
<div class="metric-card" style="margin-bottom:0">
    <div class="text">
        <div class="label">Prediksi — Model Standar</div>
        <div class="value">{final_b['_pred']:,.2f} <span class="unit">kWh</span></div>
    </div>
</div>""", unsafe_allow_html=True)
m2.markdown(f"""
<div class="metric-card green" style="margin-bottom:0">
    <div class="text">
        <div class="label">Prediksi — Model Ditingkatkan</div>
        <div class="value">{final_o['_pred']:,.2f} <span class="unit">kWh</span></div>
    </div>
</div>""", unsafe_allow_html=True)
m3.markdown(f"""
<div class="metric-card yellow" style="margin-bottom:0">
    <div class="text">
        <div class="label">Selisih Kedua Model</div>
        <div class="value">{selisih:,.2f} <span class="unit">kWh</span></div>
    </div>
</div>""", unsafe_allow_html=True)
m4.markdown(f"""
<div class="metric-card purple" style="margin-bottom:0">
    <div class="text">
        <div class="label">Model Terbaik</div>
        <div class="value" style="font-size:1rem">{model_terbaik_nama}</div>
    </div>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── PERBANDINGAN AKURASI ──────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3" cy="6" r="1" fill="#1B2A6B"/><circle cx="3" cy="12" r="1" fill="#1B2A6B"/><circle cx="3" cy="18" r="1" fill="#1B2A6B"/></svg>Perbandingan Akurasi Kedua Model</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint-text" style="margin-bottom:16px">'
    'Dihitung dari data uji yang tidak dipakai saat pelatihan model.</p>',
    unsafe_allow_html=True
)

col_l, col_sep, col_r = st.columns([5, 1, 5])
with col_l:
    st.markdown(
        '<div style="font-size:.72rem;font-weight:700;color:#6B7A99;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:10px">Model Standar</div>',
        unsafe_allow_html=True
    )
    al1, al2 = st.columns(2)
    al1.markdown(f"""
    <div class="metric-card yellow" style="margin-bottom:0">
        <div class="text"><div class="label">MAE</div>
        <div class="value">{mae_b:,.2f} <span class="unit">kWh</span></div></div>
    </div>
    <p class="hint-text">Rata-rata selisih prediksi</p>
    """, unsafe_allow_html=True)
    al2.markdown(f"""
    <div class="metric-card" style="margin-bottom:0">
        <div class="text"><div class="label">MAPE</div>
        <div class="value">{mape_b:.2f} <span class="unit">%</span></div></div>
    </div>
    <p class="hint-text">Persentase meleset rata-rata</p>
    """, unsafe_allow_html=True)

with col_sep:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'height:90px;font-size:.75rem;font-weight:700;color:#8A96B5">VS</div>',
        unsafe_allow_html=True
    )

with col_r:
    better_label = ' <span style="color:#16A34A;font-size:.68rem">✓ Lebih akurat</span>' \
                   if is_opt_more_acc else ''
    st.markdown(
        f'<div style="font-size:.72rem;font-weight:700;color:#6B7A99;text-transform:uppercase;'
        f'letter-spacing:.06em;margin-bottom:10px">Model Ditingkatkan{better_label}</div>',
        unsafe_allow_html=True
    )
    ar1, ar2 = st.columns(2)
    mae_cls  = "green" if is_opt_more_acc else "yellow"
    mape_cls = "green" if is_opt_more_acc else ""
    ar1.markdown(f"""
    <div class="metric-card {mae_cls}" style="margin-bottom:0">
        <div class="text"><div class="label">MAE</div>
        <div class="value">{mae_o:,.2f} <span class="unit">kWh</span></div></div>
    </div>
    <p class="hint-text">Rata-rata selisih prediksi</p>
    """, unsafe_allow_html=True)
    ar2.markdown(f"""
    <div class="metric-card {mape_cls}" style="margin-bottom:0">
        <div class="text"><div class="label">MAPE</div>
        <div class="value">{mape_o:.2f} <span class="unit">%</span></div></div>
    </div>
    <p class="hint-text">Persentase meleset rata-rata</p>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── GRAFIK UTAMA ──────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>Grafik Tren: Historis + Prediksi Kedua Model</div>',
            unsafe_allow_html=True)
st.plotly_chart(buat_grafik_analisis(), use_container_width=True)
st.markdown(
    '<p class="hint-text">Geser range slider di bawah grafik untuk zoom ke periode prediksi. '
    'Klik nama di legenda untuk menyembunyikan/menampilkan trace tertentu.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)


# ── TABEL DETAIL (multi-step saja) ────────────────────────────
if n_steps > 1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="9" x2="9" y2="21"/></svg>Rincian Prediksi Bulan per Bulan</div>',
                unsafe_allow_html=True)
    tab_b, tab_o = st.tabs(["&#9193; Model Standar", "&#9881; Model Ditingkatkan"])

    with tab_b:
        rows_b = []
        prev = df['Beban'].iloc[-1]
        for r in chain_b:
            pct = ((r['_pred'] - prev) / prev) * 100
            rows_b.append({
                'Bulan': r['Periode'],
                'Prediksi (kWh)': f"{r['_pred']:,.2f}",
                'Perubahan': f"↑ {abs(pct):.2f}%" if pct >= 0 else f"↓ {abs(pct):.2f}%",
                'Bulan lalu (kWh)': f"{r['lag_1']:,.0f}",
                '2 bulan lalu (kWh)': f"{r['lag_2']:,.0f}",
                '3 bulan lalu (kWh)': f"{r['lag_3']:,.0f}",
            })
            prev = r['_pred']
        st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)

    with tab_o:
        rows_o = []
        prev = df['Beban'].iloc[-1]
        for r in chain_o:
            pct = ((r['_pred'] - prev) / prev) * 100
            rows_o.append({
                'Bulan': r['Periode'],
                'Prediksi (kWh)': f"{r['_pred']:,.2f}",
                'Perubahan': f"↑ {abs(pct):.2f}%" if pct >= 0 else f"↓ {abs(pct):.2f}%",
                'Bulan lalu (kWh)': f"{r['lag_1']:,.0f}",
                '2 bulan lalu (kWh)': f"{r['lag_2']:,.0f}",
                '3 bulan lalu (kWh)': f"{r['lag_3']:,.0f}",
            })
            prev = r['_pred']
        st.dataframe(pd.DataFrame(rows_o), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── KESIMPULAN ────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><svg style="vertical-align:-3px;margin-right:6px" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2A6B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>Kesimpulan Analisis</div>', unsafe_allow_html=True)

selisih_mape_val = abs(mape_b - mape_o)
if selisih_mape_val < 1.0:
    catatan = (
        "Selisih akurasi yang sangat kecil ini mengindikasikan bahwa proses penyetelan "
        "parameter tidak memberikan peningkatan yang signifikan, kemungkinan karena "
        "ukuran dataset yang relatif kecil (data bulanan 5 tahun)."
    )
else:
    catatan = (
        "Proses penyetelan parameter secara otomatis memberikan perbedaan akurasi "
        "yang cukup terlihat dibanding model standar."
    )

st.markdown(f"""
<div class="summary-box">
<div class="verdict-banner">
<div class="verdict-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
</div>
<div class="verdict-text">
<div class="verdict-label">Model Terbaik</div>
<div class="verdict-value">{model_terbaik_nama}</div>
</div>
<div class="verdict-stats">
<div class="verdict-stat">
<div class="vs-num">{best_mae:,.2f}</div>
<div class="vs-lbl">MAE (kWh)</div>
</div>
<div class="vs-divider"></div>
<div class="verdict-stat">
<div class="vs-num">{best_mape:.2f}%</div>
<div class="vs-lbl">MAPE</div>
</div>
</div>
</div>
<div class="summary-body">
<p>
Untuk periode <b>{selected}</b>, model standar memperkirakan
<span class="inline-pill orange">{final_b['_pred']:,.2f} kWh</span>
sedangkan model yang telah disempurnakan memperkirakan
<span class="inline-pill green">{final_o['_pred']:,.2f} kWh</span>
— selisih <span class="inline-pill yellow">{selisih:,.2f} kWh ({pct_selisih:.2f}%)</span>.
</p>
<p>{catatan}</p>
</div>
<div class="summary-footnote">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
Prediksi bersifat estimasi berdasarkan pola historis 5 tahun terakhir menggunakan
3 lag fitur. Nilai aktual dapat berbeda akibat faktor eksternal yang tidak tercakup dalam model.
</div>
</div>
""", unsafe_allow_html=True)