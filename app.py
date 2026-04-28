"""
app.py
======
ECG-HRV Analysis Dashboard  —  BSP Lab OEL
==========================================

Interactive Streamlit dashboard for:
  • ECG Signal Processing & Visualization
  • Time-Domain, Frequency-Domain, and Non-linear HRV Analysis
  • Automated Physiological Interpretation

Run with:
    streamlit run app.py

Author  : BSP Lab OEL
Version : 1.0
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from processing import full_hrv_analysis, preprocess_ecg, detect_r_peaks, extract_rr_intervals
from utils import (load_ecg_file, generate_synthetic_ecg,
                   format_metric, ms, interpret_hrv)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration & Global Theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG-HRV Analysis Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "**ECG-HRV Analysis Dashboard** | BSP Lab OEL | University Project"
    }
)

# ── Theme Configuration ────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

def set_theme():
    if st.session_state.theme == "Light":
        return """
        :root {
            --bg-primary    : #f8fafc;
            --bg-card       : #ffffff;
            --bg-card2      : #f1f5f9;
            --accent-blue   : #2563eb;
            --accent-cyan   : #0891b2;
            --accent-green  : #059669;
            --accent-purple : #7c3aed;
            --accent-red    : #dc2626;
            --accent-amber  : #d97706;
            --text-primary  : #0f172a;
            --text-secondary: #475569;
            --border        : #e2e8f0;
        }
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f8fafc 100%); color: var(--text-primary); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #e2e8f0 0%, #ffffff 100%); border-right: 1px solid var(--border); }
        .dashboard-header { background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 50%, #ffffff 100%); border: 1px solid #cbd5e1; }
        """
    else:
        return """
        :root {
            --bg-primary    : #0a0e1a;
            --bg-card       : #111827;
            --bg-card2      : #1a2235;
            --accent-blue   : #3b82f6;
            --accent-cyan   : #06b6d4;
            --accent-green  : #10b981;
            --accent-purple : #8b5cf6;
            --accent-red    : #ef4444;
            --accent-amber  : #f59e0b;
            --text-primary  : #f1f5f9;
            --text-secondary: #94a3b8;
            --border        : #1e293b;
        }
        .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1424 50%, #0a0e1a 100%); color: var(--text-primary); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1424 0%, #111827 100%); border-right: 1px solid var(--border); }
        .dashboard-header { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); border: 1px solid #312e81; }
        """

st.markdown(f"<style>{set_theme()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── App background ──────────────────────────────────── */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-cyan) !important;
}

/* ── Metric cards ────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 6px 0;
    transition: transform 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.metric-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.metric-card.red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #67e8f9); }

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #334155;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-unit {
    font-size: 13px;
    font-weight: 400;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* ── Section headers ─────────────────────────────────── */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    padding: 12px 0 6px 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Tab styling ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 13px;
    padding: 8px 16px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8) !important;
    color: white !important;
}

/* ── Header banner ───────────────────────────────────── */
.dashboard-header {
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at center, rgba(99,102,241,0.08) 0%, transparent 60%);
}
.dashboard-title {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.dashboard-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 6px;
}

/* ── Status badges ───────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-amber  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-blue   { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }

/* ── Interpretation panel ────────────────────────────── */
.interp-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e293b;
    border-left: 4px solid var(--accent-blue);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
}
.interp-title {
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent-cyan);
    margin-bottom: 5px;
}

/* ── Selectbox / sliders ─────────────────────────────── */
.stSelectbox label, .stSlider label,
.stCheckbox label, .stRadio label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Expander ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Coloring constants for Plotly charts
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color= '#475569' if st.session_state.theme == 'Light' else '#94a3b8', size=12),
    title_font=dict(family='Inter', color= '#0f172a' if st.session_state.theme == 'Light' else '#f1f5f9', size=15),
    xaxis=dict(
        gridcolor='#e2e8f0' if st.session_state.theme == 'Light' else '#1e293b', gridwidth=1,
        zerolinecolor='#e2e8f0' if st.session_state.theme == 'Light' else '#1e293b',
        tickfont=dict(color='#475569' if st.session_state.theme == 'Light' else '#94a3b8', size=11),
    ),
    yaxis=dict(
        gridcolor='#e2e8f0' if st.session_state.theme == 'Light' else '#1e293b', gridwidth=1,
        zerolinecolor='#e2e8f0' if st.session_state.theme == 'Light' else '#1e293b',
        tickfont=dict(color='#475569' if st.session_state.theme == 'Light' else '#94a3b8', size=11),
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.8)' if st.session_state.theme == 'Light' else 'rgba(17,24,39,0.8)',
        bordercolor='#e2e8f0' if st.session_state.theme == 'Light' else '#1e293b', borderwidth=1,
        font=dict(color='#475569' if st.session_state.theme == 'Light' else '#94a3b8', size=11),
    ),
    margin=dict(l=50, r=30, t=50, b=50),
    hovermode='x unified',
)

COLOR_RAW      = '#475569'
COLOR_FILTERED = '#3b82f6'
COLOR_RPEAK    = '#ef4444'
COLOR_RR       = '#06b6d4'
COLOR_VLF      = 'rgba(139,92,246,0.35)'
COLOR_LF       = 'rgba(59,130,246,0.40)'
COLOR_HF       = 'rgba(16,185,129,0.40)'


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Metric Card HTML
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, unit: str = "", color: str = "blue") -> str:
    return f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>"""


def interp_card(title: str, message: str) -> str:
    return f"""
    <div class="interp-card">
        <div class="interp-title">{title}</div>
        {message}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="url(#sidebar-grad)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
            <defs>
                <linearGradient id="sidebar-grad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#3b82f6"/>
                    <stop offset="100%" stop-color="#8b5cf6"/>
                </linearGradient>
            </defs>
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
        <div style='font-size:18px; font-weight:800; color:#f8fafc; letter-spacing: 0.5px;'>ECG-HRV Dashboard</div>
        <div style='font-size:12px; font-weight:500; color:#64748b; margin-top:2px; letter-spacing: 1px; text-transform: uppercase;'>BSP Lab OEL</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    st.markdown("### 🎨 Appearance")
    theme_choice = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_radio", 
                            index=0 if st.session_state.theme == "Dark" else 1)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()
    st.divider()

    # ── Data Source ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Select input",
        ["Upload File (.mat / .csv / .dat)", "Use Demo Signal"],
        key="data_source"
    )

    ecg_raw, fs_val = None, 360.0

    if data_source == "Upload File (.mat / .csv / .dat)":
        uploaded = st.file_uploader(
            "Upload ECG file",
            type=["mat", "csv", "dat"],
            help="Supports PhysioNet MIT-BIH (.mat or .dat+.hea) and generic CSV files."
        )

        # NEW ADDITION: Allow manual override of sampling frequency
        manual_fs = st.number_input("Override Sampling Frequency (Hz) - Optional", min_value=0.0, max_value=10000.0, value=0.0, step=1.0, help="Leave as 0.0 to auto-detect")

        if uploaded:
            try:
                ecg_raw, fs_val = load_ecg_file(uploaded)
                if manual_fs > 0:
                    fs_val = float(manual_fs)
                st.success(f"✅ Loaded {len(ecg_raw):,} samples @ {fs_val:.0f} Hz")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    else:
        st.markdown("**Demo Signal Parameters**")
        demo_hr   = st.slider("Heart Rate (bpm)", 45, 120, 72, 1)
        demo_dur  = st.slider("Duration (seconds)", 30, 300, 120, 10)
        demo_hrv  = st.slider("HRV Level (ms)", 10, 80, 40, 5)
        ecg_raw, fs_val = generate_synthetic_ecg(
            duration=float(demo_dur),
            fs=360.0,
            heart_rate=float(demo_hr),
            hrv_noise_std=float(demo_hrv) / 1000.0
        )
        st.info(f"🎲 Synthetic ECG: {demo_dur}s, {demo_hr} bpm")

    st.divider()

    # ── Signal Window ─────────────────────────────────────────────────────────
    st.markdown("### ⏱ Display Window")
    if ecg_raw is not None:
        max_dur = float(len(ecg_raw) / fs_val)
        
        # Safely compute slider bounds to ensure min_value < max_value
        s_max = max(0.01, max_dur - 1.0)
        win_start = st.slider("Start (s)", min_value=0.0, max_value=s_max, value=0.0)
        
        d_min = min(1.0, max_dur / 2.0)
        d_max = max(d_min + 0.01, min(60.0, max_dur))
        win_dur = st.slider("Window (s)", min_value=d_min, max_value=d_max, value=min(30.0, d_max))
    else:
        win_start, win_dur = 0.0, 30.0

    st.divider()

    # ── Preprocessing Toggles ─────────────────────────────────────────────────
    st.markdown("### 🔧 Preprocessing")
    apply_bp  = st.checkbox("Bandpass Filter (0.5–40 Hz)", value=True,
                             help="Butterworth BPF removes EMG noise > 40 Hz and motion artifact < 0.5 Hz")
    apply_bw  = st.checkbox("Baseline Wander Removal", value=True,
                             help="High-pass filter at 0.5 Hz removes respiratory drift")
    lowcut    = st.number_input("Low cutoff (Hz)", 0.1, 5.0, 0.5, 0.1)
    highcut   = st.number_input("High cutoff (Hz)", 10.0, 100.0, 40.0, 5.0)

    st.divider()

    # ── Detection & Analysis Settings ────────────────────────────────────────
    st.markdown("### 🎯 Detection & Analysis")
    peak_method = st.selectbox(
        "R-Peak Detection",
        ["neurokit", "pantompkins", "scipy"],
        index=0,
        help="NeuroKit2 (most robust), Pan-Tompkins (classic), SciPy (simple)"
    )
    entropy_type = st.selectbox(
        "Entropy Measure",
        ["sample", "approximate"],
        index=0,
        help="Sample Entropy is more robust for HRV; Approximate Entropy is faster"
    )

    st.divider()
    st.markdown(
        "<div style='font-size:10px;color:#334155;text-align:center;'>"
        "ECG-HRV Dashboard v1.0<br>BSP Lab · PhysioNet Compatible"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="url(#main-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 12px; margin-bottom: 4px;">
            <defs>
                <linearGradient id="main-grad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#3b82f6"/>
                    <stop offset="100%" stop-color="#06b6d4"/>
                </linearGradient>
            </defs>
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>ECG & HRV Analysis Dashboard
    </div>
    <div class="dashboard-subtitle">
        <span style="color:#60a5fa; font-weight: 500;">Biomedical Signal Processing Lab</span> &nbsp;|&nbsp;
        Time-Domain · Frequency-Domain · Non-Linear HRV Analysis &nbsp;|&nbsp;
        PhysioNet Compatible
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Guard: No data yet
# ─────────────────────────────────────────────────────────────────────────────
if ecg_raw is None:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <div style='font-size:64px; margin-bottom:16px;'>📁</div>
        <div style='font-size:20px; font-weight:700; color:#f1f5f9; margin-bottom:8px;'>
          No Signal Loaded
        </div>
        <div style='font-size:14px; color:#64748b;'>
          Upload an ECG file from the sidebar or choose <b>Use Demo Signal</b> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run Full Analysis Pipeline (with caching)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_analysis(ecg_bytes, fs, apply_bp, apply_bw, lowcut, highcut,
                 peak_method, entropy_type):
    """Cached wrapper around full_hrv_analysis."""
    ecg = np.frombuffer(ecg_bytes, dtype=np.float64)
    return full_hrv_analysis(
        ecg, fs,
        apply_bandpass=apply_bp,
        apply_baseline=apply_bw,
        lowcut=lowcut, highcut=highcut,
        peak_method=peak_method,
        entropy_type=entropy_type
    )

# NEW ADDITION: Data validation before analysis
if len(ecg_raw) < fs_val * 10:
    st.warning("⚠️ Signal is very short (< 10 seconds). HRV metrics may be unreliable.")

if np.any(np.isnan(ecg_raw)) or np.any(np.isinf(ecg_raw)):
    st.warning("⚠️ Signal contains NaN or Infinite values. Attempting to clean...")
    ecg_raw = np.nan_to_num(ecg_raw, nan=np.nanmean(ecg_raw), posinf=np.nanmax(ecg_raw), neginf=np.nanmin(ecg_raw))

if np.max(ecg_raw) - np.min(ecg_raw) < 1e-6:
    st.error("🚨 Flatline signal detected. Please check your data source.")
    st.stop()

with st.spinner("⚙️ Running ECG-HRV analysis pipeline..."):
    # NEW ADDITION: Robust error handling
    try:
        results = run_analysis(
            ecg_raw.astype(np.float64).tobytes(), fs_val,
            apply_bp, apply_bw, lowcut, highcut,
            peak_method, entropy_type
        )
    except Exception as e:
        st.error(f"🚨 Analysis pipeline failed: {e}")
        st.stop()

ecg_clean   = results['ecg_clean']
r_peaks     = results['r_peaks']
rr_intervals= results['rr_intervals']
nn_intervals= results.get('nn_intervals', rr_intervals)
ectopic_indices = results.get('ectopic_indices', [])
rr_times    = results['rr_times']
time_m      = results['time_metrics']
freq_m      = results['freq_metrics']
nl_m        = results['nonlinear_metrics']
fs          = results['fs']


# ─────────────────────────────────────────────────────────────────────────────
# Quick Stats Banner (top row)
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

def _t(key, unit="", dec=1):
    v = time_m.get(key, None)
    return format_metric(v, unit, dec) if v is not None else "N/A"

with col1:
    st.markdown(metric_card("Heart Rate", _t('mean_hr', 'bpm'), "Mean HR", "blue"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Mean RR", _t('mean_rr_ms', 'ms'), "RR Interval", "cyan"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("SDNN", _t('sdnn_ms', 'ms'), "Overall HRV", "green"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("RMSSD", _t('rmssd_ms', 'ms'), "Vagal Tone", "purple"), unsafe_allow_html=True)
with col5:
    lf_hf_val = freq_m.get('lf_hf_ratio', None)
    st.markdown(metric_card("LF/HF", format_metric(lf_hf_val, "", 2), "Sympathovagal", "amber"), unsafe_allow_html=True)
with col6:
    st.markdown(metric_card("Beats", str(len(r_peaks)), "R-Peaks Found", "red"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈  ECG Signal",
    "📉  Tachogram",
    "⏱  Time-Domain HRV",
    "📊  Frequency-Domain HRV",
    "🔵  Non-Linear HRV",
    "🧠  Interpretation"
])


# ════════════════════════════════════════════════════════════════════════
# TAB 1: ECG Signal View
# ════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📈 ECG Signal Analysis</div>', unsafe_allow_html=True)

    # ── Windowing ──────────────────────────────────────────────────────────
    i_start = int(win_start * fs)
    i_end   = min(int((win_start + win_dur) * fs), len(ecg_raw))
    t_axis  = np.arange(i_start, i_end) / fs

    raw_seg      = ecg_raw[i_start:i_end]
    clean_seg    = ecg_clean[i_start:i_end]

    # R-peaks in window
    win_peaks = r_peaks[(r_peaks >= i_start) & (r_peaks < i_end)]

    # ── Toggle: show raw vs clean ─────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        show_raw     = st.checkbox("Show Raw ECG", value=True)
    with c2:
        show_filtered = st.checkbox("Show Filtered ECG", value=True)
    with c3:
        show_rpeaks  = st.checkbox("Highlight R-Peaks", value=True)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig_ecg = go.Figure()

    if show_raw:
        fig_ecg.add_trace(go.Scatter(
            x=t_axis, y=raw_seg,
            name='Raw ECG',
            line=dict(color=COLOR_RAW, width=1.2),
            opacity=0.7
        ))

    if show_filtered:
        fig_ecg.add_trace(go.Scatter(
            x=t_axis, y=clean_seg,
            name='Filtered ECG',
            line=dict(color=COLOR_FILTERED, width=1.8),
        ))

    if show_rpeaks and len(win_peaks) > 0:
        peak_x = win_peaks / fs
        peak_y = ecg_clean[win_peaks]
        fig_ecg.add_trace(go.Scatter(
            x=peak_x, y=peak_y,
            name='R-Peaks',
            mode='markers',
            marker=dict(
                color=COLOR_RPEAK, size=9,
                symbol='triangle-up',
                line=dict(color='white', width=1)
            )
        ))
        # Vertical dashed lines at R-peaks
        for px_idx in win_peaks:
            fig_ecg.add_vline(
                x=px_idx / fs,
                line_color='rgba(239,68,68,0.2)',
                line_width=1,
                line_dash='dot'
            )

    layout = dict(PLOTLY_LAYOUT)
    layout['title'] = dict(
        text=f"ECG Signal — Window: {win_start:.1f}s to {win_start + win_dur:.1f}s  "
             f"({len(win_peaks)} R-peaks in view)",
        font=dict(color='#f1f5f9', size=14)
    )
    layout['xaxis_title'] = 'Time (seconds)'
    layout['yaxis_title'] = 'Amplitude (mV / a.u.)'
    layout['height'] = 400
    fig_ecg.update_layout(**layout)
    fig_ecg.update_xaxes(rangeslider_visible=True,
                          rangeslider=dict(bgcolor='#111827', thickness=0.06))
    st.plotly_chart(fig_ecg, use_container_width=True)

    # NEW ADDITION: Download Filtered ECG Export
    csv_filtered = pd.DataFrame({'Time (s)': t_axis, 'Raw ECG': raw_seg, 'Filtered ECG': clean_seg}).to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Filtered ECG Segment", data=csv_filtered, file_name="ecg_segment.csv", mime="text/csv", key="btn_export_ecg")

    # ── Signal Stats ────────────────────────────────────────────────────────
    with st.expander("📋 Signal Information", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Sampling Rate", f"{fs:.0f} Hz")
            st.metric("Duration", f"{len(ecg_raw)/fs:.1f} s")
        with col_b:
            st.metric("Total Samples", f"{len(ecg_raw):,}")
            st.metric("R-Peaks Detected", f"{len(r_peaks)}")
        with col_c:
            # NEW ADDITION: Improved Power-based Signal Quality Index (SNR)
            try:
                signal_power = np.mean(ecg_clean**2)
                noise_power = np.mean((ecg_raw - ecg_clean)**2)
                sqi_db = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-9 else float('inf')
                sqi_display = f"{sqi_db:.1f} dB" if sqi_db != float('inf') else "Perfect"
            except Exception:
                sqi_display = "N/A"
            st.metric("Signal Quality Index", sqi_display, help="Power-based SNR")
            
            st.metric("RR Intervals", f"{len(rr_intervals)}")
        with col_d:
            filt_status = "ON" if (apply_bp or apply_bw) else "OFF"
            st.metric("Preprocessing", filt_status)
            st.metric("Detection Method", peak_method.capitalize())


# ════════════════════════════════════════════════════════════════════════
# TAB 2: RR Tachogram
# ════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">📉 RR Interval Tachogram</div>', unsafe_allow_html=True)

    if len(rr_intervals) < 3:
        st.warning("⚠️ Not enough RR intervals detected. Try adjusting filter settings or detection method.")
    else:
        rr_ms_arr = rr_intervals * 1000.0
        nn_ms_arr = nn_intervals * 1000.0

        fig_tacho = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.10,
            shared_xaxes=False,
            subplot_titles=["RR Interval Tachogram (with Ectopic Correction)", "NN Interval Distribution (Histogram)"]
        )

        # ── Tachogram line (Raw RR) ───────────────────────────────────────────
        fig_tacho.add_trace(
            go.Scatter(
                x=rr_times, y=rr_ms_arr,
                name='Raw RR Interval',
                line=dict(color='rgba(148, 163, 184, 0.5)', width=2),
                hovertemplate='Time: %{x:.2f}s<br>Raw RR: %{y:.1f} ms<extra></extra>'
            ), row=1, col=1
        )

        # ── Tachogram line (Corrected NN) ─────────────────────────────────────
        fig_tacho.add_trace(
            go.Scatter(
                x=rr_times, y=nn_ms_arr,
                name='Corrected NN Interval',
                line=dict(color=COLOR_RR, width=2),
                fill='tozeroy',
                fillcolor='rgba(6,182,212,0.08)',
                hovertemplate='Time: %{x:.2f}s<br>NN: %{y:.1f} ms<extra></extra>'
            ), row=1, col=1
        )

        # ── Highlight Ectopic Beats ───────────────────────────────────────────
        if len(ectopic_indices) > 0:
            fig_tacho.add_trace(
                go.Scatter(
                    x=rr_times[ectopic_indices], y=rr_ms_arr[ectopic_indices],
                    name='Ectopic Beats',
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    hovertemplate='Time: %{x:.2f}s<br>Ectopic RR: %{y:.1f} ms<extra></extra>'
                ), row=1, col=1
            )

        # Mean NN line
        mean_nn_ms = float(np.mean(nn_ms_arr))
        fig_tacho.add_hline(
            y=mean_nn_ms, line_color='#f59e0b',
            line_dash='dash', line_width=1.5,
            annotation_text=f"Mean NN = {mean_nn_ms:.1f} ms",
            annotation_font_color='#f59e0b',
            row=1, col=1
        )

        # ±1 SD band
        sd_nn = float(np.std(nn_ms_arr, ddof=1))
        fig_tacho.add_hrect(
            y0=mean_nn_ms - sd_nn, y1=mean_nn_ms + sd_nn,
            fillcolor='rgba(245,158,11,0.05)',
            line_color='rgba(245,158,11,0.3)', line_width=1, line_dash='dot',
            row=1, col=1
        )

        # ── Histogram ─────────────────────────────────────────────────────────
        fig_tacho.add_trace(
            go.Histogram(
                x=nn_ms_arr, nbinsx=30,
                name='NN Distribution',
                marker=dict(
                    color='rgba(6,182,212,0.6)',
                    line=dict(color='rgba(6,182,212,0.9)', width=1)
                ),
                hovertemplate='NN: %{x:.0f} ms<br>Count: %{y}<extra></extra>'
            ), row=2, col=1
        )

        tacho_layout = dict(PLOTLY_LAYOUT)
        tacho_layout['height'] = 540
        tacho_layout['showlegend'] = False
        fig_tacho.update_layout(**tacho_layout)
        fig_tacho.update_xaxes(title_text='Time (s)', row=1, col=1,
                                 gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))
        fig_tacho.update_yaxes(title_text='RR Interval (ms)', row=1, col=1,
                                 gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))
        fig_tacho.update_xaxes(title_text='RR Interval (ms)', row=2, col=1,
                                 gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))
        fig_tacho.update_yaxes(title_text='Count', row=2, col=1,
                                 gridcolor='#1e293b', tickfont=dict(color='#94a3b8'))
        fig_tacho.update_annotations(font_color='#94a3b8')
        st.plotly_chart(fig_tacho, use_container_width=True)

        # NEW ADDITION: Download RR Intervals Export
        csv_rr = pd.DataFrame({'Time (s)': np.round(rr_times, 3), 'RR Interval (ms)': np.round(rr_ms_arr, 2), 'Heart Rate (bpm)': np.round(60000 / rr_ms_arr, 1)}).to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download All RR Intervals", data=csv_rr, file_name="rr_intervals.csv", mime="text/csv", key="btn_export_rr")

        # ── Table of first 20 RR intervals ────────────────────────────────────
        with st.expander("📋 RR Interval Table (first 50 beats)"):
            df_rr = pd.DataFrame({
                'Beat #': range(1, min(51, len(rr_ms_arr) + 1)),
                'Time (s)': np.round(rr_times[:50], 3),
                'RR Interval (ms)': np.round(rr_ms_arr[:50], 2),
                'Heart Rate (bpm)': np.round(60000 / rr_ms_arr[:50], 1)
            })
            st.dataframe(df_rr, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3: Time-Domain HRV
# ════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">⏱ Time-Domain HRV Analysis</div>', unsafe_allow_html=True)

    if not time_m:
        st.warning("⚠️ Insufficient RR intervals for time-domain analysis.")
    else:
        # ── Metric Cards Grid ─────────────────────────────────────────────────
        st.markdown("**Standard HRV Metrics (Task Force, 1996)**")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown(metric_card(
                "Mean RR Interval",
                f"{time_m.get('mean_rr_ms', 0):.1f}",
                "milliseconds", "blue"
            ), unsafe_allow_html=True)
            st.markdown(metric_card(
                "Total Beats",
                str(int(time_m.get('n_beats', 0))),
                "R-R pairs", "cyan"
            ), unsafe_allow_html=True)

        with r1c2:
            st.markdown(metric_card(
                "SDNN",
                f"{time_m.get('sdnn_ms', 0):.2f}",
                "ms — Overall HRV (Normal: 50-100)", "green"
            ), unsafe_allow_html=True)
            st.markdown(metric_card(
                "pNN50",
                f"{time_m.get('pnn50', 0):.1f}",
                "% — Proportion > 50 ms diff", "purple"
            ), unsafe_allow_html=True)

        with r1c3:
            st.markdown(metric_card(
                "RMSSD",
                f"{time_m.get('rmssd_ms', 0):.2f}",
                "ms — Short-term HRV (Normal: >20)", "amber"
            ), unsafe_allow_html=True)
            st.markdown(metric_card(
                "NN50",
                str(int(time_m.get('nn50', 0))),
                "pairs differing > 50 ms", "red"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Heart Rate Statistics**")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            st.markdown(metric_card("Mean HR", f"{time_m.get('mean_hr', 0):.1f}", "bpm", "blue"), unsafe_allow_html=True)
        with r2c2:
            st.markdown(metric_card("Min HR", f"{time_m.get('min_hr', 0):.1f}", "bpm", "green"), unsafe_allow_html=True)
        with r2c3:
            st.markdown(metric_card("Max HR", f"{time_m.get('max_hr', 0):.1f}", "bpm", "red"), unsafe_allow_html=True)
        with r2c4:
            st.markdown(metric_card("HR Std Dev", f"{time_m.get('hr_std', 0):.2f}", "bpm", "amber"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Instantaneous HR Timeline ─────────────────────────────────────────
        st.markdown("**Instantaneous Heart Rate Over Time**")
        inst_hr = 60.0 / rr_intervals
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=rr_times, y=inst_hr,
            name='Instantaneous HR',
            line=dict(color='#8b5cf6', width=2),
            fill='tozeroy', fillcolor='rgba(139,92,246,0.08)',
            hovertemplate='Time: %{x:.2f}s<br>HR: %{y:.1f} bpm<extra></extra>'
        ))
        fig_hr.add_hline(y=time_m.get('mean_hr', 0), line_color='#f59e0b',
                          line_dash='dash', line_width=1.5,
                          annotation_text=f"Mean = {time_m.get('mean_hr', 0):.1f} bpm",
                          annotation_font_color='#f59e0b')
        hr_layout = dict(PLOTLY_LAYOUT)
        hr_layout.update({'height': 300, 'xaxis_title': 'Time (s)',
                           'yaxis_title': 'Heart Rate (bpm)'})
        fig_hr.update_layout(**hr_layout)
        st.plotly_chart(fig_hr, use_container_width=True)

        # ── Formulas Expander ─────────────────────────────────────────────────
        with st.expander("📐 HRV Metric Formulas & Reference Ranges"):
            st.markdown("""
| Metric | Formula | Normal Range |
|--------|---------|-------------|
| **SDNN** | std(NN intervals) | 50–100 ms (healthy adult) |
| **RMSSD** | √(mean(ΔNN²)) | > 20 ms (higher = better vagal tone) |
| **pNN50** | (NN pairs > 50 ms) / total × 100 | > 5% normal; > 20% athletic |
| **Mean HR** | 60 / Mean_RR | 60–100 bpm (normal sinus) |
| **NN50** | Count pairs with ΔRRI > 50 ms | Context-dependent |

> **Reference:** Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. *Heart rate variability: standards of measurement, physiological interpretation, and clinical use.* Eur Heart J. 1996;17:354–381.
            """)


# ════════════════════════════════════════════════════════════════════════
# TAB 4: Frequency-Domain HRV
# ════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📊 Frequency-Domain HRV Analysis (Welch PSD)</div>', unsafe_allow_html=True)

    if not freq_m or 'frequencies' not in freq_m:
        st.warning("⚠️ Insufficient RR intervals for frequency analysis (need ≥ 16 beats).")
    else:
        freqs  = freq_m['frequencies']
        psd    = freq_m['psd']

        # ── Power metrics ─────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.markdown(metric_card("VLF Power",
                f"{freq_m.get('vlf_power', 0):.1f}",
                "ms²  (0.003–0.04 Hz)", "purple"), unsafe_allow_html=True)
        with fc2:
            st.markdown(metric_card("LF Power",
                f"{freq_m.get('lf_power', 0):.1f}",
                "ms²  (0.04–0.15 Hz)", "blue"), unsafe_allow_html=True)
        with fc3:
            st.markdown(metric_card("HF Power",
                f"{freq_m.get('hf_power', 0):.1f}",
                "ms²  (0.15–0.40 Hz)", "green"), unsafe_allow_html=True)
        with fc4:
            st.markdown(metric_card("LF/HF Ratio",
                f"{freq_m.get('lf_hf_ratio', 0):.3f}",
                "Sympathovagal (Normal: 1-2)", "amber"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fc5, fc6, fc7 = st.columns(3)
        with fc5:
            st.markdown(metric_card("LF n.u.",
                f"{freq_m.get('lf_nu', 0):.1f}",
                "Normalised Units", "blue"), unsafe_allow_html=True)
        with fc6:
            st.markdown(metric_card("HF n.u.",
                f"{freq_m.get('hf_nu', 0):.1f}",
                "Normalised Units", "green"), unsafe_allow_html=True)
        with fc7:
            st.markdown(metric_card("Total Power",
                f"{freq_m.get('total_power', 0):.1f}",
                "ms² (VLF+LF+HF)", "cyan"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PSD Plot ───────────────────────────────────────────────────────────
        # Clip to 0.5 Hz max for clarity
        freq_mask = freqs <= 0.5
        f_plot = freqs[freq_mask]
        p_plot = psd[freq_mask]

        fig_psd = go.Figure()

        # Shaded frequency bands
        def _add_band(fig, f, p, fmin, fmax, color, label):
            mask = (f >= fmin) & (f <= fmax)
            if mask.sum() < 2:
                return
            fig.add_trace(go.Scatter(
                x=f[mask], y=p[mask],
                name=label, fill='tozeroy',
                fillcolor=color,
                line=dict(color=color.replace('0.35', '0.9').replace('0.40', '0.9'), width=1.5),
                hovertemplate=f'[{label}] f=%{{x:.3f}} Hz<br>PSD=%{{y:.2f}} ms²/Hz<extra></extra>'
            ))

        _add_band(fig_psd, f_plot, p_plot, 0.003, 0.04,  COLOR_VLF, 'VLF (0.003–0.04 Hz)')
        _add_band(fig_psd, f_plot, p_plot, 0.04,  0.15,  COLOR_LF,  'LF  (0.04–0.15 Hz)')
        _add_band(fig_psd, f_plot, p_plot, 0.15,  0.40,  COLOR_HF,  'HF  (0.15–0.40 Hz)')

        # Full PSD line overlay
        fig_psd.add_trace(go.Scatter(
            x=f_plot, y=p_plot,
            name='PSD (Welch)',
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            hoverinfo='skip'
        ))

        # Band boundary lines
        for freq_line, label in [(0.04, 'VLF|LF'), (0.15, 'LF|HF'), (0.40, 'HF limit')]:
            fig_psd.add_vline(x=freq_line, line_color='rgba(255,255,255,0.15)',
                               line_width=1, line_dash='dot',
                               annotation_text=f'{freq_line} Hz',
                               annotation_font_color='#64748b',
                               annotation_position='top right')

        psd_layout = dict(PLOTLY_LAYOUT)
        psd_layout.update({
            'title': 'Power Spectral Density — Welch Method (RR Tachogram)',
            'xaxis_title': 'Frequency (Hz)',
            'yaxis_title': 'PSD (ms² / Hz)',
            'height': 420,
            'yaxis_type': 'linear'
        })
        fig_psd.update_layout(**psd_layout)
        st.plotly_chart(fig_psd, use_container_width=True)

        # ── Pie chart: Band Power Distribution ────────────────────────────────
        st.markdown("**Band Power Distribution**")
        pie_col, table_col = st.columns([1, 1])

        with pie_col:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['VLF', 'LF', 'HF'],
                values=[
                    max(freq_m.get('vlf_power', 0), 0.001),
                    max(freq_m.get('lf_power',  0), 0.001),
                    max(freq_m.get('hf_power',  0), 0.001)
                ],
                hole=0.55,
                marker=dict(colors=['#7c3aed', '#2563eb', '#059669'],
                            line=dict(color='#0a0e1a', width=2)),
                textfont=dict(color='white', size=12),
                hovertemplate='%{label}: %{value:.1f} ms² (%{percent})<extra></extra>'
            )])
            pie_layout = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#94a3b8'),
                              legend=dict(font=dict(color='#94a3b8')), height=300,
                              margin=dict(l=10, r=10, t=10, b=10),
                              annotations=[dict(text='Band<br>Power', x=0.5, y=0.5,
                                               font_size=13, font_color='#f1f5f9',
                                               showarrow=False)])
            fig_pie.update_layout(**pie_layout)
            st.plotly_chart(fig_pie, use_container_width=True)

        with table_col:
            df_freq = pd.DataFrame({
                'Band': ['VLF', 'LF', 'HF', 'Total'],
                'Range (Hz)': ['0.003–0.04', '0.04–0.15', '0.15–0.40', '0.003–0.40'],
                'Power (ms²)': [
                    f"{freq_m.get('vlf_power', 0):.2f}",
                    f"{freq_m.get('lf_power', 0):.2f}",
                    f"{freq_m.get('hf_power', 0):.2f}",
                    f"{freq_m.get('total_power', 0):.2f}"
                ],
                '% Total': [
                    f"{freq_m.get('vlf_pct', 0):.1f}%",
                    f"{freq_m.get('lf_pct', 0):.1f}%",
                    f"{freq_m.get('hf_pct', 0):.1f}%",
                    "100.0%"
                ],
                'n.u.': ['—',
                          f"{freq_m.get('lf_nu', 0):.1f}",
                          f"{freq_m.get('hf_nu', 0):.1f}", '—']
            })
            st.dataframe(df_freq, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 5: Non-Linear HRV
# ════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🔵 Non-Linear HRV Analysis</div>', unsafe_allow_html=True)

    if not nl_m or 'rr_n' not in nl_m:
        st.warning("⚠️ Insufficient RR intervals for non-linear analysis.")
    else:
        rr_n  = nl_m['rr_n']
        rr_n1 = nl_m['rr_n1']
        sd1   = nl_m.get('sd1_ms', 0)
        sd2   = nl_m.get('sd2_ms', 0)

        # ── Metric cards ───────────────────────────────────────────────────────
        nl_c1, nl_c2, nl_c3, nl_c4 = st.columns(4)
        with nl_c1:
            st.markdown(metric_card("SD1", f"{sd1:.2f}", "ms — Short-term variability", "cyan"), unsafe_allow_html=True)
        with nl_c2:
            st.markdown(metric_card("SD2", f"{sd2:.2f}", "ms — Long-term variability", "blue"), unsafe_allow_html=True)
        with nl_c3:
            ratio = nl_m.get('sd1_sd2_ratio', 0)
            st.markdown(metric_card("SD1/SD2", f"{ratio:.4f}", "Complexity Ratio", "purple"), unsafe_allow_html=True)
        with nl_c4:
            en_v = nl_m.get('entropy_value', float('nan'))
            en_l = nl_m.get('entropy_type', 'Entropy')
            en_str = f"{en_v:.4f}" if (en_v is not None and not (isinstance(en_v, float) and np.isnan(en_v))) else "N/A"
            en_type_short = "SampEn" if "Sample" in en_l else "ApEn"
            st.markdown(metric_card(en_type_short, en_str, en_l, "amber"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Poincaré Plot ─────────────────────────────────────────────────────
        nl_left, nl_right = st.columns([3, 2])
        with nl_left:
            fig_pc = go.Figure()

            # Identity line (RR_n = RR_{n+1})
            rr_min, rr_max = float(np.min(rr_n)) - 30, float(np.max(rr_n)) + 30
            fig_pc.add_trace(go.Scatter(
                x=[rr_min, rr_max], y=[rr_min, rr_max],
                name='Line of Identity (RRₙ = RRₙ₊₁)',
                line=dict(color='rgba(255,255,255,0.15)', width=1.5, dash='dash'),
                hoverinfo='skip'
            ))

            # Scatter density
            fig_pc.add_trace(go.Scatter(
                x=rr_n, y=rr_n1,
                name='RR Points',
                mode='markers',
                marker=dict(
                    color=np.arange(len(rr_n)),
                    colorscale='Viridis',
                    size=5, opacity=0.7,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text='Beat #', font=dict(color='#94a3b8', size=11)),
                        tickfont=dict(color='#94a3b8', size=10),
                        bgcolor='rgba(17,24,39,0.8)',
                        outlinecolor='#1e293b'
                    )
                ),
                hovertemplate='RRₙ: %{x:.1f} ms<br>RRₙ₊₁: %{y:.1f} ms<extra></extra>'
            ))

            # SD1 / SD2 Ellipse
            centroid_x = float(np.mean(rr_n))
            centroid_y = float(np.mean(rr_n1))
            theta = np.linspace(0, 2 * np.pi, 200)
            ellipse_x = centroid_x + sd2 * np.cos(theta) * np.cos(np.pi/4) - sd1 * np.sin(theta) * np.sin(np.pi/4)
            ellipse_y = centroid_y + sd2 * np.cos(theta) * np.sin(np.pi/4) + sd1 * np.sin(theta) * np.cos(np.pi/4)
            fig_pc.add_trace(go.Scatter(
                x=ellipse_x, y=ellipse_y,
                name='SD1/SD2 Ellipse',
                line=dict(color='#ef4444', width=2, dash='dot'),
                hoverinfo='skip'
            ))

            # Centroid
            fig_pc.add_trace(go.Scatter(
                x=[centroid_x], y=[centroid_y],
                name='Centroid',
                mode='markers',
                marker=dict(color='#ef4444', size=10, symbol='cross',
                             line=dict(color='white', width=2)),
                hovertemplate=f'Centroid: ({centroid_x:.1f}, {centroid_y:.1f}) ms<extra></extra>'
            ))

            pc_layout = dict(PLOTLY_LAYOUT)
            pc_layout.update({
                'title': f'Poincaré Plot (RRₙ vs RRₙ₊₁)  |  SD1={sd1:.1f} ms, SD2={sd2:.1f} ms',
                'xaxis_title': 'RRₙ (ms)',
                'yaxis_title': 'RRₙ₊₁ (ms)',
                'height': 480,
                'xaxis_scaleanchor': 'y',  # force square aspect
            })
            fig_pc.update_layout(**pc_layout)
            st.plotly_chart(fig_pc, use_container_width=True)

        with nl_right:
            st.markdown("**What is the Poincaré Plot?**")
            st.markdown("""
<div style='font-size:13px; color:#94a3b8; line-height:1.7;'>
Each point represents a pair of consecutive RR intervals:
<br><br>
• <b style='color:#06b6d4;'>SD1</b> — spread perpendicular to the line of identity → <em>short-term (beat-to-beat) variability</em>. Reflects parasympathetic modulation.
<br><br>
• <b style='color:#3b82f6;'>SD2</b> — spread along the line of identity → <em>long-term variability</em>. Reflects overall autonomic modulation.
<br><br>
• <b>SD1/SD2 ratio</b> — complexity index. Higher ratio = greater short-term relative to long-term variability.
<br><br>
• The <b style='color:#ef4444;'>red ellipse</b> fits the SD1-SD2 axes and summarises HRV geometry.
</div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            en_v = nl_m.get('entropy_value', float('nan'))
            en_l = nl_m.get('entropy_type', 'Entropy')
            en_ok = en_v is not None and not (isinstance(en_v, float) and np.isnan(en_v))

            st.markdown(f"**{en_l}**")
            st.markdown(f"""
<div style='font-size:13px; color:#94a3b8; line-height:1.7;'>
Value: <b style='font-size:22px; color:#f1f5f9; font-family:JetBrains Mono,monospace;'>
{"N/A" if not en_ok else f"{en_v:.4f}"}</b>
<br><br>
{"Low entropy (<1.0) → highly regular, predictable signal (e.g., cardiac pathology, paced rhythm)" if en_ok and en_v < 1.0 else
 "Moderate entropy (1.0–1.5) → healthy, complex HRV dynamics typical of normal sinus rhythm" if en_ok and 1.0 <= en_v <= 1.5 else
 "High entropy (>1.5) → highly complex / irregular rhythm (may indicate noise or arrhythmia)" if en_ok else ""}
<br><br>
<b>Reference:</b> Richman & Moorman (2000). Am J Physiol Heart Circ Physiol.
</div>
            """, unsafe_allow_html=True)

            if nl_m.get('ellipse_area'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Poincaré Ellipse Area**")
                st.markdown(f"""
<div style='font-size:13px; color:#94a3b8;'>
Area = π × SD1 × SD2 = <b style='color:#f1f5f9;'>{nl_m['ellipse_area']:.1f} ms²</b>
<br>Larger area → greater overall HRV complexity.
</div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 6: Interpretation
# ════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🧠 Automated Clinical Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:13px; color:#64748b; margin-bottom:16px;'>
⚠️ <em>This automated interpretation is for academic and research purposes only.
It does not constitute medical advice. All findings should be evaluated by a qualified clinician.</em>
</div>
    """, unsafe_allow_html=True)

    # Build combined metrics dict for interpret_hrv()
    combined = {}
    combined.update(time_m)
    combined.update({k: v for k, v in freq_m.items() if not isinstance(v, np.ndarray)})
    combined.update({k: v for k, v in nl_m.items() if not isinstance(v, np.ndarray)})

    interpretations = interpret_hrv(combined)

    if not interpretations:
        st.info("Run the analysis first to see interpretations.")
    else:
        for title, message in interpretations.items():
            st.markdown(interp_card(title, message), unsafe_allow_html=True)

    # ── Summary Report ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📄 HRV Summary Report**")
    with st.expander("Click to view full metrics report", expanded=True):
        report_data = []

        # Time domain
        for k, label, unit in [
            ('mean_rr_ms', 'Mean RR Interval', 'ms'),
            ('sdnn_ms', 'SDNN', 'ms'),
            ('rmssd_ms', 'RMSSD', 'ms'),
            ('pnn50', 'pNN50', '%'),
            ('nn50', 'NN50', 'count'),
            ('mean_hr', 'Mean HR', 'bpm'),
            ('min_hr', 'Min HR', 'bpm'),
            ('max_hr', 'Max HR', 'bpm'),
        ]:
            v = time_m.get(k, None)
            if v is not None:
                report_data.append({
                    'Domain': 'Time',
                    'Metric': label,
                    'Value': f"{v:.3f}",
                    'Unit': unit
                })

        # Frequency domain
        for k, label, unit in [
            ('vlf_power', 'VLF Power', 'ms²'),
            ('lf_power', 'LF Power', 'ms²'),
            ('hf_power', 'HF Power', 'ms²'),
            ('total_power', 'Total Power', 'ms²'),
            ('lf_nu', 'LF (n.u.)', 'n.u.'),
            ('hf_nu', 'HF (n.u.)', 'n.u.'),
            ('lf_hf_ratio', 'LF/HF Ratio', '-'),
        ]:
            v = freq_m.get(k, None)
            if v is not None:
                report_data.append({
                    'Domain': 'Frequency',
                    'Metric': label,
                    'Value': f"{v:.4f}",
                    'Unit': unit
                })

        # Non-linear
        for k, label, unit in [
            ('sd1_ms', 'SD1', 'ms'),
            ('sd2_ms', 'SD2', 'ms'),
            ('sd1_sd2_ratio', 'SD1/SD2', '-'),
            ('ellipse_area', 'Poincaré Ellipse Area', 'ms²'),
            ('entropy_value', nl_m.get('entropy_type', 'Entropy'), 'nats'),
        ]:
            v = nl_m.get(k, None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                report_data.append({
                    'Domain': 'Non-Linear',
                    'Metric': label,
                    'Value': f"{v:.4f}",
                    'Unit': unit
                })

        if report_data:
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = df_report.to_csv(index=False)
            st.download_button(
                label="⬇ Download Report as CSV",
                data=csv_data,
                file_name="hrv_analysis_report.csv",
                mime="text/csv"
            )

    # ── PDF Report Generation ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Export PDF Medical Report")
    st.markdown("Generate a comprehensive, clinical-grade PDF report containing all analysis results and visual plots.")
    
    if st.button("Download Professional PDF Report", type="primary"):
        with st.spinner("Generating high-quality PDF report... (This may take a moment to render plots)"):
            import tempfile
            from report_generator import generate_pdf_report
            
            try:
                # Save plots as images
                plots_dict = {}
                tmp_dir = tempfile.mkdtemp()
                
                # 1. ECG Signal (First 10 seconds)
                fig_ecg_pdf = go.Figure(fig_ecg)
                fig_ecg_pdf.update_layout(width=800, height=400, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
                fig_ecg_pdf.update_xaxes(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0', tickfont=dict(color='black'))
                fig_ecg_pdf.update_yaxes(gridcolor='#e2e8f0', zerolinecolor='#e2e8f0', tickfont=dict(color='black'))
                ecg_path = os.path.join(tmp_dir, "ecg.png")
                fig_ecg_pdf.write_image(ecg_path)
                plots_dict["ECG Signal Segment"] = ecg_path
                
                # 2. Tachogram
                fig_tacho_pdf = go.Figure(fig_tacho)
                fig_tacho_pdf.update_layout(width=800, height=500, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
                fig_tacho_pdf.update_xaxes(gridcolor='#e2e8f0', tickfont=dict(color='black'))
                fig_tacho_pdf.update_yaxes(gridcolor='#e2e8f0', tickfont=dict(color='black'))
                tacho_path = os.path.join(tmp_dir, "tacho.png")
                fig_tacho_pdf.write_image(tacho_path)
                plots_dict["RR Interval Tachogram & Distribution"] = tacho_path
                
                # 3. Frequency Spectrum
                if 'fig_psd' in locals():
                    fig_psd_pdf = go.Figure(fig_psd)
                    fig_psd_pdf.update_layout(width=800, height=400, paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
                    fig_psd_pdf.update_xaxes(gridcolor='#e2e8f0', tickfont=dict(color='black'))
                    fig_psd_pdf.update_yaxes(gridcolor='#e2e8f0', tickfont=dict(color='black'))
                    psd_path = os.path.join(tmp_dir, "psd.png")
                    fig_psd_pdf.write_image(psd_path)
                    plots_dict["Power Spectral Density (Welch)"] = psd_path
                
                # Generate PDF
                pdf_path = os.path.join(tmp_dir, "ECG_HRV_Report.pdf")
                generate_pdf_report(combined, interpretations, plots_dict, pdf_path)
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="⬇ Click Here to Download PDF",
                        data=pdf_file,
                        file_name="ECG_HRV_Report.pdf",
                        mime="application/pdf",
                        key="download_pdf_final"
                    )
                st.success("PDF Generated Successfully!")
                
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:12px; padding:8px 0;'>
    🫀 <b style='color:#3b82f6;'>ECG-HRV Analysis Dashboard</b> &nbsp;|&nbsp;
    Biomedical Signal Processing Lab &nbsp;|&nbsp;
    Algorithms: Pan-Tompkins (1985) · Welch PSD · Poincaré Analysis · Sample Entropy &nbsp;|&nbsp;
    Data: PhysioNet MIT-BIH Compatible
</div>
""", unsafe_allow_html=True)
