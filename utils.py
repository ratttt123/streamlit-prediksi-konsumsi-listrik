import streamlit as st
import base64
from pathlib import Path

def load_css():
    css_path = Path("styles/main.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_image_base64(filename):
    base_dir = Path(__file__).parent
    for ext, mime in [
        ("png",  "image/png"),
        ("jpg",  "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("svg",  "image/svg+xml"),
    ]:
        img_file = base_dir / "assets" / f"{filename}.{ext}"
        if img_file.exists():
            with open(img_file, "rb") as f:
                return base64.b64encode(f.read()).decode(), mime
    return None, None

def get_logo_base64():
    return get_image_base64("logo")


def render_forest_accent():
    svg_html = (
        '<div class="sidebar-accent">'
        '<svg viewBox="0 0 260 105" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="20" y1="91" x2="240" y2="91" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>'
        '<polygon points="132,2 122,20 129,20 125,34 141,14 133,14 137,2" fill="#F5821F"/>'
        '<g transform="translate(50,15)">'
        '<polygon points="0,20 -7,32 7,32" fill="#F5821F"/>'
        '<polygon points="0,28 -10,44 10,44" fill="#F5821F" opacity="0.85"/>'
        '<polygon points="0,38 -13,54 13,54" fill="#F5821F" opacity="0.7"/>'
        '<rect x="-3" y="54" width="6" height="16" rx="1" fill="#ffffff" opacity="0.5"/>'
        '</g>'
        '<g transform="translate(130,15)">'
        '<polygon points="0,8 -9,22 9,22" fill="#ffffff"/>'
        '<polygon points="0,16 -13,36 13,36" fill="#ffffff" opacity="0.85"/>'
        '<polygon points="0,28 -17,52 17,52" fill="#ffffff" opacity="0.7"/>'
        '<rect x="-4" y="52" width="8" height="18" rx="1" fill="#F5821F" opacity="0.7"/>'
        '</g>'
        '<g transform="translate(210,15)">'
        '<polygon points="0,14 -8,27 8,27" fill="#F5821F"/>'
        '<polygon points="0,22 -12,40 12,40" fill="#F5821F" opacity="0.85"/>'
        '<polygon points="0,33 -16,54 16,54" fill="#F5821F" opacity="0.7"/>'
        '<rect x="-3.5" y="54" width="7" height="16" rx="1" fill="#ffffff" opacity="0.5"/>'
        '</g>'
        '</svg>'
        '<div class="sidebar-accent-label">Random Forest &middot; Prediksi Beban Listrik</div>'
        '</div>'
    )
    st.markdown(svg_html, unsafe_allow_html=True)


def render_sidebar():
    logo_b64, mime = get_logo_base64()

    with st.sidebar:

        # ── BLOK ATAS: LOGO ──────────────────────────────────
        with st.container(key="sidebar_top"):
            if logo_b64:
                st.markdown(
                    f'<div class="sidebar-logo">'
                    f'<img src="data:{mime};base64,{logo_b64}" alt="Logo"/>'
                    f'<div class="sidebar-title">Prediksi Beban Listrik<br>Bulanan</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="sidebar-logo">'
                    '<div style="font-size:3rem">⚡</div>'
                    '<div class="sidebar-title">Prediksi Beban Listrik<br>Bulanan</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            st.markdown("---")

        # ── BLOK TENGAH: MENU (otomatis center) ─────────────
        with st.container(key="sidebar_menu"):
            st.page_link("pages/1_Dashboard.py", label="Dashboard", icon=":material/dashboard:")
            st.page_link("pages/2_Prediksi.py", label="Prediksi", icon=":material/online_prediction:")
            st.page_link(
                "pages/4_Analisis_Kesimpulan.py",
                label="Evaluasi",
                icon=":material/insights:"
            )

        # ── BLOK BAWAH: AKSEN RANDOM FOREST (paling bawah, tanpa footer) ─
        with st.container(key="sidebar_bottom"):
            st.markdown("---")
            render_forest_accent()