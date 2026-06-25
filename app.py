import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import os
from tensorflow.keras.models import load_model

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Smart Factory Command Center — Telkom University",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi session state
if 'masuk' not in st.session_state:
    st.session_state.masuk = False

# =============================================================================
# GLOBAL CSS — Enterprise Dark/Light Adaptive Design
# =============================================================================
st.markdown("""
<style>
/* ─── TYPOGRAPHY ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], .main,
p, h1, h2, h3, h4, h5, h6, span, label, div, button {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Bypass padding kosong bawaan Streamlit biar ga perlu scroll */
[data-testid="block-container"] {
    padding-top: 2.5rem !important;
    padding-bottom: 1rem !important;
}

/* ─── DESIGN TOKENS (CSS Variables) ─────────────────────────────────────── */
:root {
    --accent:        #e11d48;
    --accent-dim:    rgba(225, 29, 72, 0.12);
    --accent-glow:   rgba(225, 29, 72, 0.25);
    --green:         #10b981;
    --green-dim:     rgba(16, 185, 129, 0.12);
    --green-glow:    rgba(16, 185, 129, 0.25);
    --border:        rgba(148, 163, 184, 0.12);
    --border-strong: rgba(148, 163, 184, 0.22);
    --surface:       rgba(15, 23, 42, 0.04);
    --radius:        10px;
    --radius-lg:     16px;
}

/* ─── SIDEBAR OVERHAUL ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-strong) !important;
    background: transparent !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 28px !important;
}

/* Sidebar nav radio — make it look like an app menu */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px !important;
    margin-top: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: var(--radius) !important;
    padding: 9px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: background 0.15s !important;
    border: 1px solid transparent !important;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}
/* Hide the radio circle dot */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
}

/* ─── GLOBAL BUTTON RESET & OVERHAUL ────────────────────────────────────── */
.stButton { display: flex !important; justify-content: center !important; }
.stButton > button {
    height: 44px !important;
    border-radius: var(--radius) !important;
    background: var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px var(--accent-glow) !important;
}
.stButton > button:hover {
    background: #be123c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px var(--accent-glow) !important;
}

/* Keluar button — ghost style */
.btn-ghost .stButton > button {
    background: transparent !important;
    color: inherit !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: none !important;
}
.btn-ghost .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ─── SLIDER ACCENT ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ─── METRIC CARDS ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 500 !important; opacity: 0.65 !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 700 !important; }

/* ─── TABLE STYLING ──────────────────────────────────────────────────────── */
[data-testid="stTable"] table {
    border-collapse: collapse !important;
    font-size: 13.5px !important;
    width: 100% !important;
}
[data-testid="stTable"] th {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 10px 14px !important;
    border: none !important;
}
[data-testid="stTable"] td {
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 13px !important;
}

/* ─── SECTION DIVIDER ────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border-strong) !important; margin: 20px 0 !important; }

/* ─── SPLASH SCREEN CARD (DIOPTIMALKAN BIAR GA SCROLL) ───────────────────── */
.splash-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 75vh; /* Fix height biar ga bocor ke bawah */
}
.splash-card {
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 40px 40px 32px; /* Padding dikurangin biar pas di layar */
    max-width: 460px;
    width: 100%;
    text-align: center;
    background: var(--surface);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
}
.splash-eyebrow {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid rgba(225,29,72,0.18);
    border-radius: 99px;
    padding: 4px 14px;
    margin-bottom: 18px;
}
.splash-title {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.6px;
    line-height: 1.25;
    margin: 0 0 10px 0;
}
.splash-sub {
    font-size: 13.5px;
    line-height: 1.6;
    opacity: 0.55;
    margin-bottom: 24px;
}
.splash-divider {
    width: 40px;
    height: 3px;
    background: var(--accent);
    border-radius: 99px;
    margin: 14px auto 20px;
}
.splash-btn-wrap .stButton > button {
    width: 220px !important;
    font-size: 13.5px !important;
    letter-spacing: 0.5px !important;
}

/* ─── DASHBOARD HEADER ───────────────────────────────────────────────────── */
.dh-wrap {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--border-strong);
    margin-bottom: 28px;
    gap: 16px;
}
.dh-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}
.dh-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
    line-height: 1.2;
}
.dh-sub {
    font-size: 13.5px;
    opacity: 0.55;
    margin: 0;
    line-height: 1.6;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-dim);
    border: 1px solid var(--green-glow);
    border-radius: 99px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--green);
    white-space: nowrap;
    flex-shrink: 0;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.45; transform:scale(0.75); }
}

/* ─── SECTION LABEL ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 12px;
    margin-top: 0;
}

/* ─── DIAGNOSIS OUTPUT PANELS ────────────────────────────────────────────── */
.diag-panel {
    border-radius: var(--radius-lg);
    padding: 28px 28px 22px;
    margin-bottom: 16px;
}
.diag-panel.risk {
    background: var(--accent-dim);
    border: 1.5px solid rgba(225,29,72,0.28);
}
.diag-panel.safe {
    background: var(--green-dim);
    border: 1.5px solid rgba(16,185,129,0.28);
}
.diag-status-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.diag-status-label.risk { color: var(--accent); }
.diag-status-label.safe { color: var(--green); }
.diag-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.4px;
    margin: 0 0 10px;
    line-height: 1.25;
}
.diag-title.risk { color: var(--accent); }
.diag-title.safe { color: var(--green); }
.diag-body {
    font-size: 13.5px;
    line-height: 1.75;
    opacity: 0.75;
    margin: 0;
}

/* ─── STANDBY STATE ──────────────────────────────────────────────────────── */
.standby-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 48px 24px;
    gap: 14px;
}
.standby-icon {
    width: 52px; height: 52px;
    border-radius: 50%;
    border: 2px solid var(--border-strong);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    opacity: 0.35;
}
.standby-title { font-size: 15px; font-weight: 600; opacity: 0.5; margin: 0; }
.standby-sub   { font-size: 12.5px; opacity: 0.35; margin: 0; line-height: 1.6; }

/* ─── KPI STRIP ──────────────────────────────────────────────────────────── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    padding: 14px 16px;
    background: var(--surface);
}
.kpi-label { font-size: 10.5px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; opacity: 0.45; margin-bottom: 6px; }
.kpi-value { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; font-family: 'JetBrains Mono', monospace; }
.kpi-sub   { font-size: 11px; opacity: 0.4; margin-top: 2px; }

/* ─── FEATURE IMPORTANCE INSIGHT BOX ────────────────────────────────────── */
.insight-box {
    background: var(--accent-dim);
    border: 1px solid rgba(225,29,72,0.2);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.65;
    margin-top: 16px;
}

/* ─── ALGO SPEC BULLET ───────────────────────────────────────────────────── */
.spec-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
    line-height: 1.6;
}
.spec-item:last-child { border-bottom: none; }
.spec-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    margin-top: 8px;
    flex-shrink: 0;
}

/* ─── WATERMARK ──────────────────────────────────────────────────────────── */
.watermark {
    position: fixed;
    bottom: 14px;
    right: 18px;
    z-index: 9999;
    font-size: 10.5px;
    font-weight: 500;
    opacity: 0.5;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.2px;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    padding: 5px 12px;
    backdrop-filter: blur(8px);
}

/* ─── SIDEBAR TEAM LIST ──────────────────────────────────────────────────── */
.team-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    font-size: 12px;
    line-height: 1.5;
    border-bottom: 1px solid var(--border);
}
.team-item:last-child { border-bottom: none; }
.team-num {
    min-width: 18px;
    font-weight: 700;
    color: var(--accent);
    font-size: 11px;
    margin-top: 1px;
}
.team-name  { font-weight: 500; }
.team-nim   { opacity: 0.45; font-size: 10.5px; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# Floating Watermark
st.markdown('<div class="watermark">ML Project · Multi-Model AI</div>', unsafe_allow_html=True)


# =============================================================================
# SPLASH SCREEN
# =============================================================================
if not st.session_state.masuk:
    st.markdown('<div class="splash-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="splash-card">', unsafe_allow_html=True)

    try:
        col_logo = st.columns([1, 1.5, 1])[1]
        with col_logo:
            st.image("logo_telu.png", use_container_width=True)
    except:
        st.markdown("<p style='font-size:11px; font-weight:800; letter-spacing:3px; color:#e11d48; margin:0 0 6px;'>TELKOM UNIVERSITY</p>", unsafe_allow_html=True)

    st.markdown('<div class="splash-eyebrow">Fakultas Teknik Elektro</div>', unsafe_allow_html=True)

    st.markdown("""
        <h1 class="splash-title">Smart Factory<br>Command Center</h1>
        <div class="splash-divider"></div>
        <p class="splash-sub">
            Sistem Predictive Maintenance berbasis AI untuk<br>
            deteksi dini kerusakan komponen manufaktur industri.
        </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="splash-btn-wrap">', unsafe_allow_html=True)
    if st.button("BUKA DASHBOARD SISTEM", key="masuk_btn"):
        st.session_state.masuk = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()


# =============================================================================
# LOAD MODEL ASSETS
# =============================================================================
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rf_path = os.path.join(BASE_DIR, 'model_mesin_rf.pkl')
    lstm_path = os.path.join(BASE_DIR, 'model_mesin_lstm.h5')
    scaler_path = os.path.join(BASE_DIR, 'scaler_mesin.pkl')
    
    model_rf = joblib.load(rf_path)
    model_lstm = load_model(lstm_path)
    scaler = joblib.load(scaler_path)
    return model_rf, model_lstm, scaler

model_rf, model_lstm, scaler = load_assets()


# =============================================================================
# SIDEBAR NAVIGASI
# =============================================================================
with st.sidebar:
    try:
        st.image("logo_telu.png", width=80)
    except:
        st.markdown("<p style='font-size:13px; font-weight:800; color:#e11d48; margin:0;'>TELKOM UNIVERSITY</p>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["Dashboard Pemantauan", "Analisis Performa Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Team Members
    st.markdown("<p class='section-label' style='padding:0 4px;'>Tim Penyusun</p>", unsafe_allow_html=True)

    team = [
        ("Reynaldhi Ananda Rahardian", "1102223226"),
        ("Fikri Naufal Hakim",         "1102220107"),
        ("Rendi Febrian",               "1102223112"),
        ("Muhammad Ghoisan H.A.",       "1102223228"),
        ("I Wayan Rivan",               "1102223168"),
        ("Ahmad Diva Sihombing",        "1102223050"),
    ]

    for i, (name, nim) in enumerate(team, 1):
        st.markdown(f"""
            <div class="team-item">
                <span class="team-num">{i:02d}</span>
                <div>
                    <div class="team-name">{name}</div>
                    <div class="team-nim">{nim}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
    if st.button("Keluar Sistem", use_container_width=True):
        st.session_state.masuk = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: DASHBOARD PEMANTAUAN
# =============================================================================
if menu == "Dashboard Pemantauan":

    # Page Header
    st.markdown("""
        <div class="dh-wrap">
            <div>
                <div class="dh-tag">Smart Factory · Predictive Maintenance AI</div>
                <h1 class="dh-title">Dashboard Pemantauan Mesin</h1>
                <p class="dh-sub">Pemantauan parameter termal & mekanik real-time untuk pencegahan kegagalan operasional.</p>
            </div>
            <div>
                <div class="live-badge">
                    <div class="live-dot"></div>
                    Sistem Aktif
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Banner image (optional)
    try:
        st.image("banner_industri.jpg", use_container_width=True)
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    except:
        pass

    # KPI Strip
    st.markdown("""
        <div class="kpi-strip">
            <div class="kpi-card">
                <div class="kpi-label">Model AI</div>
                <div class="kpi-value">RF & LSTM</div>
                <div class="kpi-sub">Multi-Model Classifier</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Akurasi Model</div>
                <div class="kpi-value">98.3%</div>
                <div class="kpi-sub">AI4I 2020 · 10K data poin</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Sensor Aktif</div>
                <div class="kpi-value">5</div>
                <div class="kpi-sub">Termal & Mekanik</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main 2-column layout
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        with st.container(border=True):
            st.markdown("<p class='section-label'>Parameter Sensor Operasional</p>", unsafe_allow_html=True)

            # Sub-columns for compact layout
            r1a, r1b = st.columns(2)
            with r1a:
                air_temp = st.slider("Air Temperature (K)", 250.0, 350.0, 298.1, 0.1,
                                     help="Suhu udara ambien di sekitar mesin (Kelvin)")
            with r1b:
                process_temp = st.slider("Process Temperature (K)", 250.0, 350.0, 308.6, 0.1,
                                          help="Suhu proses operasional mesin (Kelvin)")

            r2a, r2b = st.columns(2)
            with r2a:
                rot_speed = st.slider("Rotational Speed (rpm)", 1000, 3000, 1551, 1,
                                      help="Kecepatan putaran poros mesin (RPM)")
            with r2b:
                torque = st.slider("Torque (Nm)", 10.0, 100.0, 42.8, 0.1,
                                   help="Momen torsi pada poros (Newton-meter)")

            tool_wear = st.slider("Tool Wear (min)", 0, 500, 0, 1,
                                  help="Akumulasi waktu pemakaian pahat/tool (menit)")

            # Delta Thermal (computed hint)
            delta_t = round(process_temp - air_temp, 1)
            dt_color = "#e11d48" if delta_t > 12 else "#10b981"
            st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; padding:10px 14px;
                            border:1px solid var(--border); border-radius:8px; margin-top:8px;
                            font-size:12.5px;">
                    <span style="opacity:0.5;">Delta Termal (ΔT):</span>
                    <code style="color:{dt_color}; font-family:'JetBrains Mono',monospace; font-weight:700;">
                        {delta_t} K
                    </code>
                    <span style="opacity:0.35; font-size:11.5px;">
                        {'⚠ Tinggi' if delta_t > 12 else '✓ Normal'}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # --- TAMBAHAN PILIHAN MODEL AI ---
            st.markdown("<p class='section-label' style='margin-top:18px;'>Pengaturan Model AI</p>", unsafe_allow_html=True)
            pilihan_model = st.radio(
                "Pilih Algoritma Diagnosis:",
                ("Random Forest (Rekomendasi)", "LSTM (Deep Learning)"),
                label_visibility="collapsed"
            )

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            submit = st.button("JALANKAN DIAGNOSIS", use_container_width=True)

    # ── Output Column ──────────────────────────────────────────────────────
    with col_out:
        with st.container(border=True):
            st.markdown("<p class='section-label'>Hasil Evaluasi Diagnosis</p>", unsafe_allow_html=True)

            if submit:
                with st.spinner("Memproses data sensor melalui model AI…"):
                    time.sleep(0.6)

                    input_df = pd.DataFrame({
                        'Air temperature [K]':      [air_temp],
                        'Process temperature [K]':  [process_temp],
                        'Rotational speed [rpm]':   [rot_speed],
                        'Torque [Nm]':              [torque],
                        'Tool wear [min]':          [tool_wear],
                    })
                    input_scaled = scaler.transform(input_df)
                    
                    # Logika Pemilihan Model RF / LSTM
                    if "Random Forest" in pilihan_model:
                        prediction   = model_rf.predict(input_scaled)
                        pred_proba   = model_rf.predict_proba(input_scaled)[0]
                        is_risk = (prediction[0] == 1)
                        
                        if is_risk:
                            confidence = round(pred_proba[1] * 100, 1)
                        else:
                            confidence = round(pred_proba[0] * 100, 1)
                            
                    else:
                        input_lstm = np.reshape(input_scaled, (1, 1, input_scaled.shape[1]))
                        pred_prob_lstm = model_lstm.predict(input_lstm)[0][0]
                        is_risk = (pred_prob_lstm > 0.5)
                        
                        if is_risk:
                            confidence = round(pred_prob_lstm * 100, 1)
                        else:
                            confidence = round((1 - pred_prob_lstm) * 100, 1)

                if is_risk:
                    st.markdown(f"""
                        <div class="diag-panel risk">
                            <div class="diag-status-label risk">⚠ Peringatan Kritis</div>
                            <div class="diag-title risk">Risiko Kegagalan Terdeteksi</div>
                            <p class="diag-body">
                                Model AI mendeteksi anomali kritis pada kombinasi parameter sensor aktif.
                                Segera lakukan pemeliharaan korektif atau hentikan sementara lini produksi
                                untuk mencegah breakdown mekanikal secara total.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Status Komponen", "RISK DETECTED", delta=None)
                    with col_m2:
                        st.metric("Confidence Model", f"{confidence}%", delta=None)
                else:
                    st.markdown(f"""
                        <div class="diag-panel safe">
                            <div class="diag-status-label safe">✓ Sistem Normal</div>
                            <div class="diag-title safe">Komponen Beroperasi Optimal</div>
                            <p class="diag-body">
                                Seluruh parameter sensor berada dalam batas toleransi aman yang diizinkan.
                                Kinerja termal dan mekanikal komponen berada pada tingkat efisiensi optimal.
                                Proses manufaktur dapat dilanjutkan secara penuh.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Status Komponen", "STABLE")
                    with col_m2:
                        st.metric("Confidence Model", f"{confidence}%")

                # Sensor snapshot
                st.markdown("<p class='section-label' style='margin-top:18px;'>Snapshot Nilai Sensor</p>", unsafe_allow_html=True)
                snap_df = pd.DataFrame({
                    "Parameter": ["Air Temp", "Proc. Temp", "Speed", "Torque", "Tool Wear"],
                    "Nilai":     [f"{air_temp} K", f"{process_temp} K", f"{rot_speed} rpm", f"{torque} Nm", f"{tool_wear} min"]
                })
                st.dataframe(snap_df, use_container_width=True, hide_index=True)

            else:
               st.markdown("""
                <div class="insight-box">
                    <strong>Kesimpulan Analisis Fitur</strong><br>
                    Variabel <strong>Torque</strong> dan <strong>Rotational Speed</strong> adalah indikator sensor paling
                    kritis, berkontribusi gabungan sebesar <strong>63%</strong> terhadap akurasi penentuan
                    degradasi komponen internal mesin.
                </div>
            """, unsafe_allow_html=True)
               # =============================================================================
# PAGE: ANALISIS PERFORMA MODEL
# =============================================================================
elif menu == "Analisis Performa Model":

    # Page Header
    st.markdown("""
        <div class="dh-wrap">
            <div>
                <div class="dh-tag">Model Evaluation · Random Forest vs LSTM</div>
                <h1 class="dh-title">Analisis Komparasi Algoritma</h1>
                <p class="dh-sub">Evaluasi performa model berdasarkan metrik klasifikasi pada dataset tabular operasional pabrik.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-label'>Tabel Komparasi Metrik (Class 1 : Risiko Kegagalan)</p>", unsafe_allow_html=True)

    # Bikin DataFrame buat nampilin metrik yang persis sama kayak hasil terminal lo kemaren
    metric_df = pd.DataFrame({
        "Algoritma AI": ["Random Forest (Tabular)", "LSTM (Sequential / Deep Learning)"],
        "Accuracy": ["98.3%", "97.2%"],
        "Precision": ["0.80", "0.56"],
        "Recall": ["0.59", "0.38"],
        "F1-Score": ["0.68", "0.45"]
    })
    
    # Nampilin tabel
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    
    # Kotak Insight buat nyamain sama narasi paper IEEE lo
    st.markdown("""
        <div class="insight-box" style="border-left: 3px solid var(--accent);">
            <h4 style='margin-top:0; margin-bottom:10px; font-weight:800; color:var(--accent); letter-spacing:-0.3px;'>
                🧠 Insight Akademis & Justifikasi Model
            </h4>
            <div style='font-size: 13.5px; opacity: 0.85; line-height: 1.7;'>
                <strong>Analisis Trade-off:</strong> Walaupun LSTM merupakan arsitektur <em>Deep Learning</em> yang sangat kuat untuk memori data deret waktu (<em>time-series</em>), model ini mengalami <em>underperforming</em> pada metrik <b>Recall (0.38)</b>. Hal ini terjadi secara natural karena dataset sensor industri yang digunakan bersifat tabular diskrit (<em>lack of temporal dependencies</em> antar baris).
                <br><br>
                Sebaliknya, pendekatan <b>Random Forest</b> terbukti jauh lebih <em>robust</em> dalam menangkap pola non-linear pada fitur independen, dibuktikan dengan nilai <b>Recall (0.59)</b> dan <b>Precision (0.80)</b> yang jauh memimpin. Dalam konteks <em>Predictive Maintenance</em>, Random Forest dipilih sebagai model final yang di-deploy karena lebih efektif meminimalisir <em>False Negative</em> (mesin diprediksi aman padahal rusak) sekaligus menjaga tingkat alarm palsu tetap rendah.
            </div>
        </div>
    """, unsafe_allow_html=True)