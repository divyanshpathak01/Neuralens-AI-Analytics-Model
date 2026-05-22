# -*- coding: utf-8 -*-
"""
NeuraLens v4.0 — Premium ML Analytics Dashboard
Converted from React/JSX to Streamlit Python
Theme: Obsidian & Violet · Aurora Accents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuraLens · ML Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONSTANTS ─────────────────────────────────────────────────
VALID_USER = "ABCD"
VALID_PASS = "12345678"

PALETTES = {
    "Obsidian Violet": ["#a78bfa", "#6366f1", "#67e8f9", "#f472b6"],
    "Aurora Borealis": ["#34d399", "#22d3ee", "#a78bfa", "#fb923c"],
    "Neon Cosmos":     ["#f0abfc", "#67e8f9", "#bef264", "#fb7185"],
    "Rose Prism":      ["#fb7185", "#f0abfc", "#c084fc", "#34d399"],
}

# ── SESSION STATE INIT ────────────────────────────────────────
for k, v in [("logged_in", False), ("login_error", ""), ("theme", "dark")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── GLOBAL CSS ────────────────────────────────────────────────
def inject_css(theme: str):
    is_dark = theme == "dark"
    bg          = "#090C15"      if is_dark else "#F0F2F8"
    sidebar_bg  = "#0D1120"      if is_dark else "#E4E8F4"
    card_bg     = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.04)"
    text_pri    = "#F0ECFF"      if is_dark else "#0F1020"
    text_sec    = "rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.45)"
    border      = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.1)"
    input_bg    = "#110e22"      if is_dark else "#ffffff"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
    background-color: {bg} !important;
    color: {text_pri} !important;
}}

.main .block-container {{
    background: {bg} !important;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {border} !important;
}}
section[data-testid="stSidebar"] * {{ color: {text_pri} !important; }}
section[data-testid="stSidebar"] label {{
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    color: {text_sec} !important;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {'rgba(130,80,255,0.06)' if is_dark else 'rgba(0,0,0,0.04)'};
    padding: 5px;
    border-radius: 14px;
    border: 1px solid {border};
    flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    color: {text_sec};
    padding: 0.45rem 1rem;
    transition: all 0.18s;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.4) !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}}
.stButton > button:hover {{
    box-shadow: 0 6px 28px rgba(124,58,237,0.55) !important;
    transform: translateY(-2px) !important;
}}

/* ── INPUTS ── */
.stTextInput input, .stNumberInput input {{
    background: {input_bg} !important;
    border: 1px solid rgba(130,80,255,0.25) !important;
    color: {text_pri} !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 2px rgba(168,85,247,0.2) !important;
}}

/* ── SELECTBOX ── */
.stSelectbox > div > div {{
    background: {input_bg} !important;
    border: 1px solid rgba(130,80,255,0.2) !important;
    border-radius: 10px !important;
    color: {text_pri} !important;
}}

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"] {{
    accent-color: #a855f7;
}}
.stSlider [data-baseweb="thumb"] {{
    background: #a855f7 !important;
}}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {{
    border: 2px dashed rgba(130,80,255,0.35) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    background: rgba(130,80,255,0.04) !important;
    transition: all 0.2s;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: rgba(168,85,247,0.6) !important;
    background: rgba(130,80,255,0.08) !important;
}}

/* ── DATAFRAME ── */
.stDataFrame {{
    border-radius: 14px !important;
    border: 1px solid {border} !important;
    overflow: hidden;
}}
.stDataFrame [data-testid="stDataFrameResizable"] {{
    background: {card_bg} !important;
}}

/* ── METRICS ── */
[data-testid="stMetric"] {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.2rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: {text_sec} !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #a78bfa !important;
}}

/* ── GENERAL TEXT ── */
p, li, .stMarkdown {{ color: {text_pri} !important; }}
h1, h2, h3 {{ color: {text_pri} !important; font-family: 'Outfit', sans-serif !important; }}
.stCaption {{ color: {text_sec} !important; font-size: 0.78rem !important; }}
hr {{ border-color: {border} !important; }}

/* ── SUCCESS / ERROR / INFO ── */
.stSuccess {{ background: rgba(52,211,153,0.1) !important; border-color: rgba(52,211,153,0.3) !important; border-radius: 12px !important; }}
.stError   {{ background: rgba(248,113,113,0.1) !important; border-color: rgba(248,113,113,0.3) !important; border-radius: 12px !important; }}
.stInfo    {{ background: rgba(96,165,250,0.1)  !important; border-color: rgba(96,165,250,0.3)  !important; border-radius: 12px !important; }}
.stWarning {{ background: rgba(251,191,36,0.1)  !important; border-color: rgba(251,191,36,0.3)  !important; border-radius: 12px !important; }}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(130,80,255,0.3); border-radius: 3px; }}

/* ── CUSTOM CARDS ── */
.nl-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}}
.nl-section-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {text_sec};
    font-weight: 700;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.nl-metric {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}}
.nl-metric::before {{
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), transparent);
}}
.nl-metric-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {text_sec};
    margin-bottom: 0.5rem;
    font-weight: 700;
}}
.nl-metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.1;
}}

/* ── HERO ── */
.nl-hero {{
    background: {'linear-gradient(135deg,#0e0b20 0%,#150f2e 50%,#1a0d35 100%)' if is_dark else 'linear-gradient(135deg,#E8E4FF 0%,#D4D8FF 100%)'};
    border: 1px solid {'rgba(130,80,255,0.25)' if is_dark else border};
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.nl-hero-title {{
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.05;
    margin-bottom: 0.4rem;
    font-family: 'Outfit', sans-serif;
    color: {text_pri};
}}
.nl-hero-accent {{
    background: linear-gradient(90deg, #a855f7, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.nl-pill {{
    display: inline-block;
    background: rgba(130,80,255,0.12);
    border: 1px solid rgba(130,80,255,0.25);
    color: #b090e0;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 4px;
}}
.nl-badge-excellent {{ background:rgba(52,211,153,0.15);  color:#34d399; border:1px solid rgba(52,211,153,0.35);  padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }}
.nl-badge-good      {{ background:rgba(96,165,250,0.15);  color:#60a5fa; border:1px solid rgba(96,165,250,0.35);  padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }}
.nl-badge-average   {{ background:rgba(251,191,36,0.15);  color:#fbbf24; border:1px solid rgba(251,191,36,0.35);  padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }}
.nl-badge-weak      {{ background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.35); padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }}

.nl-pred-box {{
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(99,102,241,0.08));
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(124,58,237,0.15);
}}
.nl-pred-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}}

/* ── LOGIN ── */
.nl-login-card {{
    background: linear-gradient(160deg, #0e0b20, #150f2e, #130c28);
    border: 1px solid rgba(130,80,255,0.22);
    border-radius: 24px;
    padding: 2.8rem 2.6rem;
    box-shadow: 0 0 80px rgba(124,58,237,0.15);
}}
.nl-logo {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #f0eaff;
}}

/* ── AI INSIGHT CARD ── */
.nl-insight-card {{
    background: {card_bg};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)


# ── HELPER: PERFORMANCE BADGE ─────────────────────────────────
def perf_badge(r2: float) -> str:
    if r2 >= 0.9:  return '<span class="nl-badge-excellent">✦ Excellent</span>'
    if r2 >= 0.75: return '<span class="nl-badge-good">✦ Good</span>'
    if r2 >= 0.5:  return '<span class="nl-badge-average">✦ Average</span>'
    return '<span class="nl-badge-weak">✦ Weak</span>'


def perf_label(r2: float) -> str:
    if r2 >= 0.9:  return "Excellent"
    if r2 >= 0.75: return "Good"
    if r2 >= 0.5:  return "Average"
    return "Weak"


# ── HELPER: METRIC CARD HTML ──────────────────────────────────
def metric_card(label: str, value: str, color: str) -> str:
    return f"""
<div class="nl-metric">
  <div class="nl-metric-label">{label}</div>
  <div class="nl-metric-value" style="color:{color}">{value}</div>
</div>"""


# ── HELPER: PLOTLY THEME ──────────────────────────────────────
def plotly_layout(title: str = "", theme: str = "dark") -> dict:
    is_dark = theme == "dark"
    bg = "#0b0b18" if is_dark else "#F8F9FC"
    grid = "rgba(255,255,255,0.07)" if is_dark else "rgba(0,0,0,0.07)"
    text_color = "rgba(255,255,255,0.5)" if is_dark else "rgba(0,0,0,0.5)"
    return dict(
        title=dict(text=title, font=dict(family="Outfit", size=13, color=text_color)) if title else {},
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family="Outfit", color=text_color, size=11),
        margin=dict(t=30 if title else 16, r=16, b=40, l=50),
        xaxis=dict(showgrid=True, gridcolor=grid, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=grid, zeroline=False, tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, font=dict(size=11)),
    )


# ── HELPER: BUILD MODEL ───────────────────────────────────────
def build_model(model_type: str, alpha: float):
    if model_type == "Ridge Regression":
        return Ridge(alpha=alpha)
    elif model_type == "Lasso Regression":
        return Lasso(alpha=alpha, max_iter=10000)
    return LinearRegression()


# ── HELPER: EXPORT CSV ────────────────────────────────────────
def predictions_csv(y_test, y_pred) -> bytes:
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Residual": y_test - y_pred})
    return df.to_csv(index=False).encode()


# ══════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    inject_css("dark")
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("""
        <div class="nl-login-card">
          <div class="nl-logo">
            Neura<span style="background:linear-gradient(90deg,#a855f7,#60a5fa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;">Lens</span>
          </div>
          <div style="color:rgba(255,255,255,0.3);font-size:13px;margin-bottom:16px;">
            ML Analytics · Regression Intelligence Platform
          </div>
          <div style="margin-bottom:20px;">
            <span class="nl-pill">RIDGE</span>
            <span class="nl-pill">LASSO</span>
            <span class="nl-pill">LINEAR</span>
          </div>
          <div style="height:1px;background:linear-gradient(90deg,rgba(168,85,247,0.4),transparent);margin-bottom:8px;"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        uid = st.text_input("User ID", placeholder="Enter your User ID")
        pwd = st.text_input("Password", placeholder="Enter your Password", type="password")

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        if st.button("Sign In →", use_container_width=True):
            if uid == VALID_USER and pwd == VALID_PASS:
                st.session_state.logged_in = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Invalid credentials. Please try again."
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🔐 Authorized access only · NeuraLens v4.0")
    st.stop()


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
inject_css(st.session_state.theme)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:900;
                letter-spacing:-0.03em;padding-bottom:2px;">
      Neura<span style="background:linear-gradient(90deg,#a855f7,#60a5fa);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;">Lens</span>
    </div>
    <div style="font-size:10px;opacity:0.4;margin-bottom:18px;font-weight:500;">
      ML Analytics Platform
    </div>
    <div style="height:1px;background:linear-gradient(90deg,rgba(168,85,247,0.4),transparent);
                margin-bottom:18px;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;opacity:0.45;font-weight:700;margin-bottom:8px;">⚙️ Model Config</div>', unsafe_allow_html=True)
    model_type   = st.selectbox("Regression Model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
    test_size    = st.slider("Test Split (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random Seed", 0, 999, 42)

    alpha_val = 1.0
    if model_type in ("Ridge Regression", "Lasso Regression"):
        alpha_val = st.slider("Alpha (λ)", 0.01, 10.0, 1.0, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;opacity:0.45;font-weight:700;margin-bottom:8px;">🎨 Chart Palette</div>', unsafe_allow_html=True)
    palette_name = st.selectbox("Color Palette", list(PALETTES.keys()))
    colors = PALETTES[palette_name]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(168,85,247,0.3),transparent);margin-bottom:14px;"></div>', unsafe_allow_html=True)

    theme_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    st.caption("NeuraLens v4.0 · Obsidian & Violet")


# ── HERO SECTION ──────────────────────────────────────────────
st.markdown(f"""
<div class="nl-hero">
  <div style="position:absolute;top:-80px;right:-80px;width:300px;height:300px;
    background:radial-gradient(circle,rgba(160,80,255,0.12) 0%,transparent 70%);
    pointer-events:none;"></div>
  <div class="nl-hero-title">
    Neura<span class="nl-hero-accent">Lens</span>
  </div>
  <div style="opacity:0.45;font-size:14px;margin-bottom:12px;">
    Regression Intelligence Platform · Universal ML Analytics
  </div>
  <span class="nl-pill">🧠 Multi-Model</span>
  <span class="nl-pill">📊 Interactive Charts</span>
  <span class="nl-pill">✨ AI Insights</span>
  <span class="nl-pill">⚡ Real-Time</span>
</div>
""", unsafe_allow_html=True)


# ── FILE UPLOAD ───────────────────────────────────────────────
st.markdown('<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.12em;opacity:0.45;font-weight:700;margin-bottom:8px;">📂 Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("⬆️  Drop any CSV file here to begin. Select your target (Y) column after upload.")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────
data_raw = pd.read_csv(uploaded_file)
all_cols       = list(data_raw.columns)
numeric_cols   = data_raw.select_dtypes(include=np.number).columns.tolist()
cat_cols       = data_raw.select_dtypes(include="object").columns.tolist()

if len(numeric_cols) < 2:
    st.error("Your dataset needs at least 2 numeric columns to run regression.")
    st.stop()

# Success bar
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:0.65rem 1rem;
  background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.25);
  border-radius:12px;margin-bottom:16px;font-size:13px;">
  <span style="color:#34d399;">✅</span>
  <span style="color:#34d399;font-weight:700;">Dataset loaded:</span>
  <span style="opacity:0.6;">{data_raw.shape[0]:,} rows × {data_raw.shape[1]} columns &nbsp;·&nbsp;
  {len(numeric_cols)} numeric &nbsp;·&nbsp; {len(cat_cols)} categorical</span>
</div>
""", unsafe_allow_html=True)

# Target selector in sidebar
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;opacity:0.45;font-weight:700;margin-bottom:8px;">🎯 Target Column</div>', unsafe_allow_html=True)
    target_col = st.selectbox("Target (Y)", numeric_cols, index=len(numeric_cols)-1)

feature_cols = [c for c in numeric_cols if c != target_col]

# ── PREPROCESS ────────────────────────────────────────────────
data_enc = pd.get_dummies(data_raw, drop_first=True)
X_full   = data_enc.drop(columns=[target_col], errors="ignore")
y_full   = data_enc[target_col]

# ── TRAIN MODEL ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=test_size/100, random_state=int(random_state)
)
model = build_model(model_type, alpha_val)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_val   = r2_score(y_test, y_pred)
mae_val  = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
cv_scores = cross_val_score(model, X_full, y_full, cv=min(5, len(X_full)//2), scoring="r2")
residuals = y_test.values - y_pred

r2_color = "#34d399" if r2_val >= 0.9 else "#60a5fa" if r2_val >= 0.75 else "#f87171"

# ── KPI ROW ───────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_data = [
    (k1, "R² Score",     f"{r2_val:.4f}",            r2_color),
    (k2, "MAE",          f"{mae_val:,.2f}",           colors[1]),
    (k3, "RMSE",         f"{rmse_val:,.2f}",          colors[2]),
    (k4, "Dataset Size", f"{data_raw.shape[0]:,}",    colors[3]),
    (k5, "Features",     str(len(feature_cols)),       colors[0]),
    (k6, "CV Mean R²",   f"{cv_scores.mean():.4f}",   colors[0]),
]
for col, label, val, color in kpi_data:
    col.markdown(metric_card(label, val, color), unsafe_allow_html=True)

# Performance badge + export
badge_col, export_col, _ = st.columns([1.5, 1.5, 5])
with badge_col:
    st.markdown(f'<div style="margin-top:10px;">{perf_badge(r2_val)} &nbsp; <span style="font-size:11px;opacity:0.4;">{model_type}</span></div>', unsafe_allow_html=True)
with export_col:
    csv_bytes = predictions_csv(y_test.values, y_pred)
    st.download_button(
        label="⬇ Export Predictions CSV",
        data=csv_bytes,
        file_name="neuralens_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data", "📊 Visuals", "🤖 Model", "🔮 Predict", "📈 Compare", "✨ AI Insights"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="nl-section-label">📐 Summary Statistics</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.dataframe(data_raw.describe().round(3), use_container_width=True, height=260)

    with col_b:
        st.markdown('<div class="nl-section-label">🧹 Data Quality</div>', unsafe_allow_html=True)
        missing = data_raw.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["Complete %"] = ((1 - missing["Missing"] / len(data_raw)) * 100).round(1)
        missing["Status"] = missing["Missing"].apply(lambda x: "✅ Clean" if x == 0 else "⚠️ Has Nulls")
        st.dataframe(missing, use_container_width=True, height=260)

    st.markdown('<div class="nl-section-label" style="margin-top:1.2rem;">🗃️ Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(data_raw.head(50), use_container_width=True, height=300)

    if cat_cols:
        st.markdown('<div class="nl-section-label" style="margin-top:1.2rem;">🏷️ Categorical Distribution</div>', unsafe_allow_html=True)
        cat_choice = st.selectbox("Select categorical column", cat_cols, key="cat_dist")
        cat_counts = data_raw[cat_choice].value_counts().reset_index()
        cat_counts.columns = [cat_choice, "Count"]
        fig_cat = px.bar(cat_counts, x="Count", y=cat_choice, orientation="h",
                         color_discrete_sequence=[colors[0]])
        fig_cat.update_layout(**plotly_layout(theme=st.session_state.theme))
        st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="nl-section-label">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Feature vs Target · Scatter**")
        x_feat1 = st.selectbox("X Feature", feature_cols, index=0, key="scatter1")
        fig1 = px.scatter(data_raw, x=x_feat1, y=target_col,
                          color=target_col, color_continuous_scale="Purples",
                          opacity=0.8, hover_data=data_raw.columns[:4].tolist())
        fig1.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.3)")))
        fig1.update_layout(**plotly_layout(theme=st.session_state.theme))
        fig1.update_coloraxes(showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with v2:
        st.markdown("**Correlation Heatmap**")
        corr_data = data_enc.select_dtypes(include=np.number).corr().round(2)
        fig2 = px.imshow(corr_data, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig2.update_layout(**plotly_layout(theme=st.session_state.theme))
        fig2.update_traces(textfont_size=9)
        st.plotly_chart(fig2, use_container_width=True)

    v3, v4 = st.columns(2)

    with v3:
        st.markdown("**Second Feature vs Target**")
        default_idx = min(1, len(feature_cols)-1)
        x_feat2 = st.selectbox("X Feature", feature_cols, index=default_idx, key="scatter2")
        fig3 = px.scatter(data_raw, x=x_feat2, y=target_col,
                          color_discrete_sequence=[colors[2]], opacity=0.8)
        fig3.update_traces(marker=dict(size=7))
        fig3.update_layout(**plotly_layout(theme=st.session_state.theme))
        st.plotly_chart(fig3, use_container_width=True)

    with v4:
        st.markdown(f"**{target_col} · Distribution**")
        fig4 = px.histogram(data_raw, x=target_col, nbins=18,
                             color_discrete_sequence=[colors[0]], opacity=0.85)
        fig4.add_vline(x=data_raw[target_col].mean(), line_dash="dash",
                       line_color=colors[3],
                       annotation_text=f"Mean: {data_raw[target_col].mean():,.1f}",
                       annotation_font_color=colors[3])
        fig4.update_layout(**plotly_layout(theme=st.session_state.theme))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="nl-section-label" style="margin-top:0.5rem;">🔗 Custom Pair Plot</div>', unsafe_allow_html=True)
    px_col, py_col = st.columns(2)
    x_axis = px_col.selectbox("X Axis", numeric_cols, index=0, key="pair_x")
    y_axis = py_col.selectbox("Y Axis", numeric_cols, index=len(numeric_cols)-1, key="pair_y")
    fig5 = px.scatter(data_raw, x=x_axis, y=y_axis,
                      color=cat_cols[0] if cat_cols else None,
                      color_discrete_sequence=colors,
                      opacity=0.82, hover_data=data_raw.columns[:3].tolist())
    fig5.update_traces(marker=dict(size=7))
    fig5.update_layout(**plotly_layout(theme=st.session_state.theme))
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL & RESULTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="nl-section-label">Model Training & Evaluation</div>', unsafe_allow_html=True)

    r3a, r3b = st.columns(2)

    with r3a:
        st.markdown("**Actual vs Predicted**")
        mn_val = min(float(y_test.min()), float(y_pred.min()))
        mx_val = max(float(y_test.max()), float(y_pred.max()))
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                   marker=dict(color=colors[0], size=7, opacity=0.8,
                                               line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
                                   name="Predictions", hovertemplate="Actual: %{x:.2f}<br>Pred: %{y:.2f}"))
        fig6.add_trace(go.Scatter(x=[mn_val, mx_val], y=[mn_val, mx_val],
                                   mode="lines", line=dict(color=colors[3], dash="dash", width=2),
                                   name="Perfect Fit"))
        fig6.update_layout(**plotly_layout(theme=st.session_state.theme),
                            xaxis_title=f"Actual {target_col}",
                            yaxis_title=f"Predicted {target_col}")
        st.plotly_chart(fig6, use_container_width=True)

    with r3b:
        st.markdown("**Residuals Distribution**")
        fig7 = px.histogram(x=residuals, nbins=16, color_discrete_sequence=[colors[1]], opacity=0.85)
        fig7.add_vline(x=0, line_dash="dash", line_color=colors[3], annotation_text="Zero Error",
                       annotation_font_color=colors[3])
        fig7.update_layout(**plotly_layout(theme=st.session_state.theme),
                            xaxis_title="Residuals", yaxis_title="Frequency")
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="nl-section-label" style="margin-top:0.5rem;">🧬 Feature Coefficients</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature":     X_full.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=True)

    bar_colors = [colors[0] if v >= 0 else colors[3] for v in coef_df["Coefficient"]]
    fig8 = go.Figure(go.Bar(
        y=coef_df["Feature"], x=coef_df["Coefficient"],
        orientation="h", marker_color=bar_colors, marker_line_width=0,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig8.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig8.update_layout(**plotly_layout(theme=st.session_state.theme),
                        height=max(300, len(coef_df) * 28 + 60),
                        xaxis_title="Coefficient Value")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="nl-section-label">📋 Coefficient Table</div>', unsafe_allow_html=True)
    st.dataframe(coef_df.sort_values("Coefficient", ascending=False).reset_index(drop=True),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="nl-section-label">🔮 Make a Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter values for each feature to get a real-time prediction from the trained model.")

    input_vals = {}
    pred_cols = st.columns(3)
    for i, feat in enumerate(X_full.columns):
        def_val = float(X_full[feat].mean())
        with pred_cols[i % 3]:
            input_vals[feat] = st.number_input(
                feat[:24] + ("…" if len(feat) > 24 else ""),
                value=round(def_val, 4),
                key=f"pred_{feat}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Generate Prediction", use_container_width=False):
        input_arr = np.array([[input_vals[f] for f in X_full.columns]])
        pred_val  = model.predict(input_arr)[0]

        _, pcol, _ = st.columns([1, 2, 1])
        with pcol:
            st.markdown(f"""
            <div class="nl-pred-box">
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.14em;
                          opacity:0.4;margin-bottom:8px;font-weight:700;">
                Predicted {target_col}
              </div>
              <div class="nl-pred-value">{pred_val:,.4f}</div>
              <div style="font-size:11px;opacity:0.35;margin-top:8px;font-family:'JetBrains Mono',monospace;">
                Model: {model_type}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="nl-section-label">📈 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Comparing Linear, Ridge, and Lasso regression across key metrics.")

    results = {}
    for name, mdl in [
        ("Linear", LinearRegression()),
        ("Ridge",  Ridge(alpha=alpha_val)),
        ("Lasso",  Lasso(alpha=alpha_val, max_iter=10000)),
    ]:
        mdl.fit(X_train, y_train)
        yp = mdl.predict(X_test)
        cv_s = cross_val_score(mdl, X_full, y_full, cv=min(5, len(X_full)//2), scoring="r2")
        results[name] = {
            "R²":      r2_score(y_test, yp),
            "MAE":     mean_absolute_error(y_test, yp),
            "RMSE":    np.sqrt(mean_squared_error(y_test, yp)),
            "CV Mean": cv_s.mean(),
            "CV Std":  cv_s.std(),
        }

    res_df = pd.DataFrame(results).T.round(4)
    st.dataframe(res_df, use_container_width=True)

    # Performance badges row
    badge_cols = st.columns(3)
    for i, (name, vals) in enumerate(results.items()):
        with badge_cols[i]:
            r2c = "#34d399" if vals["R²"]>=0.9 else "#60a5fa" if vals["R²"]>=0.75 else "#f87171"
            st.markdown(f"""
            <div class="nl-metric" style="margin-top:10px;">
              <div style="font-size:15px;font-weight:700;color:{colors[i]};margin-bottom:6px;">{name}</div>
              <div style="font-size:11px;opacity:0.5;margin-bottom:3px;">R² &nbsp;
                <span style="font-family:monospace;font-weight:700;color:{r2c};">{vals["R²"]:.4f}</span>
              </div>
              <div style="font-size:11px;opacity:0.5;margin-bottom:3px;">MAE &nbsp;
                <span style="font-family:monospace;">{vals["MAE"]:.4f}</span>
              </div>
              <div style="font-size:11px;opacity:0.5;margin-bottom:8px;">RMSE &nbsp;
                <span style="font-family:monospace;">{vals["RMSE"]:.4f}</span>
              </div>
              {perf_badge(vals["R²"])}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_a, c_b = st.columns(2)

    with c_a:
        st.markdown("**R² Score Comparison**")
        fig9 = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[v["R²"] for v in results.values()],
            marker_color=colors[:3], marker_line_width=0,
            text=[f'{v["R²"]:.4f}' for v in results.values()],
            textposition="outside", textfont=dict(size=10)
        ))
        fig9.update_layout(**plotly_layout(theme=st.session_state.theme), yaxis_title="R² Score")
        st.plotly_chart(fig9, use_container_width=True)

    with c_b:
        st.markdown("**MAE Comparison**")
        fig10 = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[v["MAE"] for v in results.values()],
            marker_color=colors[:3], marker_line_width=0,
            text=[f'{v["MAE"]:,.2f}' for v in results.values()],
            textposition="outside", textfont=dict(size=10)
        ))
        fig10.update_layout(**plotly_layout(theme=st.session_state.theme), yaxis_title="MAE")
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("**Cross-Validation R² (5-Fold) — All Models**")
    def hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
        """Convert #rrggbb to rgba(r,g,b,a)."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig11 = go.Figure()
    for i, (name, mdl) in enumerate([
        ("Linear", LinearRegression()),
        ("Ridge",  Ridge(alpha=alpha_val)),
        ("Lasso",  Lasso(alpha=alpha_val, max_iter=10000)),
    ]):
        mdl.fit(X_train, y_train)
        cv_s = cross_val_score(mdl, X_full, y_full, cv=min(5, len(X_full)//2), scoring="r2")
        fill_color = hex_to_rgba(colors[i], 0.08) if colors[i].startswith("#") else colors[i]
        fig11.add_trace(go.Scatter(
            x=list(range(1, len(cv_s)+1)), y=cv_s, mode="lines+markers",
            name=name, line=dict(color=colors[i], width=2),
            marker=dict(size=7, color=colors[i]),
            fill="tozeroy", fillcolor=fill_color,
        ))
    layout11 = plotly_layout(theme=st.session_state.theme)
layout11["xaxis"].update(tickmode="linear", tick0=1, dtick=1)
fig11.update_layout(**layout11, xaxis_title="Fold", yaxis_title="R² Score")
st.plotly_chart(fig11, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="nl-section-label">✨ AI-Powered Dataset Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;opacity:0.45;margin-bottom:16px;">'
        'Powered by Claude · Automatically analyzes your dataset, model performance, and feature relationships.'
        '</div>',
        unsafe_allow_html=True
    )

    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = None

    if st.button("✨ Generate AI Insights", use_container_width=False):
        import anthropic
        sample_rows = data_raw.head(5).to_dict(orient="records")
        prompt = f"""You are an ML analytics expert. Given this dataset summary, provide concise, sharp AI insights.

Dataset: {data_raw.shape[0]} rows, {data_raw.shape[1]} columns
Target: {target_col}
Features: {', '.join(feature_cols)}
Sample rows: {sample_rows}
Model metrics: R²={r2_val:.4f}, MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, Model={model_type}

Respond ONLY with a JSON object (no markdown, no backticks) with this exact structure:
{{
  "summary": "2-sentence dataset overview",
  "strongestPredictor": "feature name and why in 1-2 sentences",
  "anomaly": "one data anomaly or notable pattern",
  "modelInterpretation": "1-2 sentence model performance interpretation",
  "recommendation": "one actionable recommendation"
}}"""

        with st.spinner("Generating AI insights with Claude…"):
            try:
                client = anthropic.Anthropic(api_key="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")
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                import json
                raw_text = message.content[0].text.strip()
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                st.session_state.ai_insights = json.loads(clean_text)
            except Exception as e:
                st.error(f"Could not generate insights: {e}")

    if st.session_state.ai_insights:
        insights = st.session_state.ai_insights
        insight_items = [
            ("🔭 Dataset Overview",      insights.get("summary", "—"),              colors[0]),
            ("🎯 Strongest Predictor",   insights.get("strongestPredictor", "—"),    colors[1]),
            ("🔍 Notable Pattern",       insights.get("anomaly", "—"),               colors[2]),
            ("🤖 Model Interpretation",  insights.get("modelInterpretation", "—"),   colors[3]),
            ("💡 Recommendation",        insights.get("recommendation", "—"),        "#fbbf24"),
        ]

        col_a_ins, col_b_ins = st.columns(2)
        for idx, (label, text, ic) in enumerate(insight_items):
            target_col_ins = col_a_ins if idx % 2 == 0 else col_b_ins
            with target_col_ins:
                st.markdown(f"""
                <div class="nl-insight-card" style="border-left:3px solid {ic};">
                  <div style="font-size:11px;color:{ic};font-weight:700;
                              letter-spacing:0.08em;margin-bottom:6px;text-transform:uppercase;">
                    {label}
                  </div>
                  <div style="font-size:13px;line-height:1.6;opacity:0.8;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("↻ Regenerate Insights", use_container_width=False):
            st.session_state.ai_insights = None
            st.rerun()
    else:
        st.markdown(
            '<div style="opacity:0.4;font-size:13px;">Click the button above to generate AI-powered insights about your dataset and model.</div>',
            unsafe_allow_html=True
        )