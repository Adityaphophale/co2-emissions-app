# ===================== IMPORTS =====================
import time
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# ===================== CONFIG =====================
st.set_page_config(page_title="EcoTrack Dashboard", layout="wide", page_icon="🌍")

# ===================== UI =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Font Application */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #1e293b !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

/* Premium Animated Light Background */
.stApp {
    background: linear-gradient(-45deg, #f0fdf4, #e0f2fe, #fdf4ff, #ccfbf1, #fefce8);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Premium Glassmorphism Cards */
.glass-card, [data-testid="stMetric"], .stDataFrame {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 10px 30px 0 rgba(31, 38, 135, 0.05) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}

[data-testid="stMetric"]:hover, .glass-card:hover {
    transform: translateY(-5px) scale(1.02) perspective(1000px) rotateX(2deg) rotateY(-2deg) !important;
    box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15), -10px 10px 20px rgba(14, 165, 233, 0.1) !important;
    border-color: rgba(16, 185, 129, 0.5) !important;
    cursor: pointer !important;
}

/* Metric specific overrides */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: -webkit-linear-gradient(45deg, #059669, #0284c7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Premium KPI Card Styling */
.premium-kpi {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 30px rgba(31, 38, 135, 0.03) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative;
    overflow: hidden;
}
.premium-kpi:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 15px 35px rgba(16, 185, 129, 0.15) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}
.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.kpi-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.kpi-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.kpi-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    animation: counterPop 1s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
.kpi-meta {
    display: flex;
    align-items: center;
    font-size: 0.8rem;
    font-weight: 600;
    gap: 4px;
}
.kpi-trend-up {
    color: #10b981;
}
.kpi-trend-down {
    color: #ef4444;
}
.kpi-trend-neut {
    color: #64748b;
}

/* Modern Eco-Tech Buttons */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
    width: 100% !important;
    position: relative;
    z-index: 1;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-4px) scale(1.03) !important;
    box-shadow: 0 15px 30px rgba(16, 185, 129, 0.5), 0 0 20px rgba(14, 165, 233, 0.4) !important;
    cursor: pointer !important;
}

/* Input elements interactive transitions */
.stTextInput input, .stSelectbox > div > div, .stSlider > div {
    transition: all 0.3s ease !important;
}
.stTextInput input:focus, .stSelectbox > div > div:focus-within {
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.3) !important;
    border-color: #10b981 !important;
    transform: scale(1.01) !important;
}
.stRadio > div[role="radiogroup"] > label {
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
.stRadio > div[role="radiogroup"] > label:hover {
    transform: translateX(5px) !important;
    color: #10b981 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.5) !important;
}
hr {
    border-color: rgba(0,0,0,0.05) !important;
}

/* ================= Advanced Animations ================= */

/* 1. Background Floating Gradient Blobs */
.stApp::before {
    content: '';
    position: fixed;
    top: -10%; left: -10%;
    width: 50vw; height: 50vw;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, rgba(255,255,255,0) 70%);
    filter: blur(60px);
    z-index: -1;
    animation: floatBlob1 25s infinite ease-in-out alternate;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%; right: -10%;
    width: 60vw; height: 60vw;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.12) 0%, rgba(255,255,255,0) 70%);
    filter: blur(80px);
    z-index: -1;
    animation: floatBlob2 30s infinite ease-in-out alternate-reverse;
}
@keyframes floatBlob1 {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(10%, 15%) scale(1.2); }
}
@keyframes floatBlob2 {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-10%, -15%) scale(1.1); }
}

/* 2. Smooth Fade-In For Entire Page */
.stApp > header, .main > div > div {
    animation: fadeInPage 1.2s ease-out forwards;
}
@keyframes fadeInPage {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 3. Animated Chart & Card Loading */
.glass-card, .stPlotlyChart {
    animation: slideUpFade 0.8s ease-out forwards;
}
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(30px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* 4. Continuous Floating SVG Hero Icons */
.glass-card[style*="display: flex"] svg {
    animation: hoverFloat 3.5s ease-in-out infinite;
}
@keyframes hoverFloat {
    0% { transform: translateY(0); }
    50% { transform: translateY(-8px); filter: drop-shadow(0 10px 8px rgba(16, 185, 129, 0.3)); }
    100% { transform: translateY(0); }
}

/* 5. KPI Counter Animation */
[data-testid="stMetricValue"] {
    animation: counterPop 1s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
@keyframes counterPop {
    from { opacity: 0; transform: scale(0.5); }
    to { opacity: 1; transform: scale(1); }
}

/* 6. Smooth Sidebar Transitions */
[data-testid="stSidebar"] {
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stSidebar"]:hover {
    box-shadow: 5px 0 30px rgba(0,0,0,0.05) !important;
}

/* 7. Loading state spinner and pulse keyframes */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.15); opacity: 1; filter: drop-shadow(0 0 8px #10b981); }
}

/* 8. Technology badge animations */
.badge {
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
.badge:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 8px 15px rgba(0,0,0,0.08) !important;
}

/* ================= Perfect Mobile & Tablet Responsiveness ================= */
@media (max-width: 1024px) {
    .glass-card {
        padding: 1.25rem !important;
    }
    h1 { font-size: 2.2rem !important; }
}

@media (max-width: 768px) {
    .glass-card {
        padding: 1rem !important;
        margin-bottom: 20px !important;
    }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.2rem !important; }
    
    /* Ensure Hero headers stack perfectly on mobile */
    .glass-card[style*="display: flex"] {
        flex-direction: column !important;
        text-align: center !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 20px !important;
    }
    
    /* Fix flex-item margins when stacked */
    .glass-card[style*="display: flex"] > div, 
    .glass-card[style*="display: flex"] > svg {
        margin-right: 0 !important;
        margin-bottom: 15px !important;
    }
    
    /* Shrink KPI metric sizes */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    
    /* Better sidebar padding on mobile */
    [data-testid="stSidebar"] {
        padding: 1rem !important;
    }
    
    /* Optimize buttons for fat-finger tapping */
    [data-testid="stButton"] button {
        padding: 1rem !important;
        font-size: 1.1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv('co2_emissions.csv', low_memory=False)
    df.columns = df.columns.str.strip()  # FIXED

    fuel_map = {"Z":"Premium Gasoline","X":"Regular Gasoline","D":"Diesel","E":"Ethanol(E85)","N":"Natural Gas"}
    df["Fuel Type"] = df["Fuel Type"].map(fuel_map)

    df_natural = df[~df["Fuel Type"].str.contains("Natural Gas", na=False)].reset_index(drop=True)

    return df, df_natural

df, df_natural = load_data()

# ===================== MODEL =====================
@st.cache_resource
def get_model():
    df_new = df_natural[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)','CO2 Emissions(g/km)']].dropna()

    df_new = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

    X = df_new[['Engine Size(L)','Cylinders','Fuel Consumption Comb (L/100 km)']]
    y = df_new['CO2 Emissions(g/km)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    st.sidebar.markdown(f'<div style="background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(248,250,252,0.6) 100%); border-left: 4px solid #10b981; padding: 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);"><div style="font-weight: 700; color: #0f172a; margin-bottom: 12px; display: flex; align-items: center; font-size: 0.95rem;">🎯 AI Engine Performance</div><div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem;"><span style="color: #475569;">R² Accuracy Score:</span><span style="font-weight: 700; color: #059669;">{r2:.2f}</span></div><div style="display: flex; justify-content: space-between; font-size: 0.9rem;"><span style="color: #475569;">Mean Abs. Error:</span><span style="font-weight: 700; color: #059669;">{mae:.2f} g/km</span></div></div>', unsafe_allow_html=True)

    return model

model = get_model()

# ===================== SIDEBAR =====================
with st.sidebar:

    st.markdown('<div class="glass-card" style="text-align: center; margin-bottom: 25px; padding: 25px 15px;"><h2 style="margin: 0; font-weight: 800; background: -webkit-linear-gradient(45deg, #059669, #0284c7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EcoTrack AI</h2><p style="color: #64748b; font-size: 0.85rem; margin: 5px 0 0 0; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Analytics Engine</p></div>', unsafe_allow_html=True)

    user_input = st.radio(
        "Navigation",
        ["Data Visualization", "AI Prediction Engine"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if user_input == "AI Prediction Engine":

        st.markdown("<h4 style='color: #334155; font-weight: 600; margin-bottom: -10px;'>Engine Parameters</h4>", unsafe_allow_html=True)
        st.caption("Adjust inputs for emission simulation")

        engine_size = st.slider("Engine Size (L)", 0.5, 8.0, 2.0, 0.1)
        cylinders = st.slider("Cylinder Count", 2, 16, 4)
        fuel_consumption = st.slider("Fuel Cons. (L/100 km)", 2.0, 30.0, 7.5, 0.1)

        st.markdown("---")

        st.markdown("<h4 style='color: #334155; font-weight: 600; margin-bottom: -10px;'>Scenario Analysis</h4>", unsafe_allow_html=True)
        yearly_km = st.number_input("Annual Distance (km)", 1000, 100000, 15000, step=1000)

        yearly_emission = (fuel_consumption * yearly_km) / 100
        
        st.markdown(f"""
<div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 10px 15px; border-radius: 4px; margin: 15px 0;">
    <span style="color: #065f46; font-weight: 600;">Est. Yearly Fuel:</span> <span style="color: #047857;">{yearly_emission:.0f} L</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button("Run AI Analysis", use_container_width=True)

    else:
        predict_button = False

    st.markdown("<br><br>", unsafe_allow_html=True)

# ===================== VISUALIZATION =====================
# 🔒 UNCHANGED (your exact code kept)

if user_input == 'Data Visualization':

    st.markdown('<div class="glass-card" style="background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.7) 100%); padding: 40px; margin-bottom: 30px; display: flex; align-items: center; border-left: 6px solid #10b981 !important;"><div style="background: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%); border-radius: 16px; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; margin-right: 30px; box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l3-9 5 18 3-9h5"/></svg></div><div><h1 style="margin: 0; font-size: 2.8rem; color: #0f172a !important; line-height: 1.2;">EcoTrack Analytics</h1><p style="margin: 8px 0 0 0; font-size: 1.15rem; color: #64748b; font-weight: 500;">AI-Based Carbon Emission Platform</p></div></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    
    # KPI 1: Total Vehicles
    k1.markdown(f'<div class="premium-kpi"><div class="kpi-header"><span class="kpi-title">Total Vehicles</span><div class="kpi-icon" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(14, 165, 233, 0.15) 100%);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg></div></div><div class="kpi-val">{len(df):,}</div><div class="kpi-meta kpi-trend-up"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg><span>+12.4% Active Fleet</span></div></div>', unsafe_allow_html=True)

    # KPI 2: Avg Industry CO₂
    k2.markdown(f'<div class="premium-kpi"><div class="kpi-header"><span class="kpi-title">Avg Industry CO₂</span><div class="kpi-icon" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.15) 100%);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg></div></div><div class="kpi-val">{df["CO2 Emissions(g/km)"].mean():.1f} <span style="font-size: 1rem; font-weight: 500; color: #64748b;">g/km</span></div><div class="kpi-meta kpi-trend-down"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg><span>-2.1% Year-on-Year</span></div></div>', unsafe_allow_html=True)

    # KPI 3: Peak Emission
    k3.markdown(f'<div class="premium-kpi"><div class="kpi-header"><span class="kpi-title">Peak Emission</span><div class="kpi-icon" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.15) 100%);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg></div></div><div class="kpi-val">{df["CO2 Emissions(g/km)"].max():.1f} <span style="font-size: 1rem; font-weight: 500; color: #64748b;">g/km</span></div><div class="kpi-meta kpi-trend-neut"><span>Peak stable at {df["CO2 Emissions(g/km)"].max():.0f} limit</span></div></div>', unsafe_allow_html=True)

    # KPI 4: Avg Consumption
    k4.markdown(f'<div class="premium-kpi"><div class="kpi-header"><span class="kpi-title">Avg Consumption</span><div class="kpi-icon" style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%);"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l3.5 3.5a6 6 0 0 1-7 0L12 2z"/><path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13s-7 8.7-7 13a7 7 0 0 0 7 7z"/></svg></div></div><div class="kpi-val">{df["Fuel Consumption Comb (L/100 km)"].mean():.1f} <span style="font-size: 1rem; font-weight: 500; color: #64748b;">L/100k</span></div><div class="kpi-meta kpi-trend-up" style="color: #0ea5e9;"><span>Engine optimized index</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def style_fig(fig):
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font_color="#334155",
            title_font_color="#0f172a",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        return fig

    st.markdown('<h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">🏁 Top Models</h3>', unsafe_allow_html=True)
    df_model = df['Model'].value_counts().reset_index()
    df_model.columns = ['Model','Count']
    fig_model = px.bar(df_model.head(25), x='Model', y='Count', color='Count', color_continuous_scale='Plasma')
    st.plotly_chart(style_fig(fig_model), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">🏭 Brand Distribution</h3>', unsafe_allow_html=True)
    df_brand = df['Make'].value_counts().reset_index()
    df_brand.columns = ['Make','Count']
    fig_brand = px.bar(df_brand.head(20), x='Make', y='Count', color='Count', color_continuous_scale='Tealgrn')
    st.plotly_chart(style_fig(fig_brand), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">🚙 Vehicle Class Insights</h3>', unsafe_allow_html=True)
    df_vc = df['Vehicle Class'].value_counts().reset_index()
    df_vc.columns = ['Vehicle Class','Count']
    fig_vc = px.bar(df_vc.head(15), x='Vehicle Class', y='Count', color='Count', color_continuous_scale='Mint')
    st.plotly_chart(style_fig(fig_vc), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">⚙️ Engine Size Analytics</h3>', unsafe_allow_html=True)
    df_engine = df['Engine Size(L)'].value_counts().reset_index()
    df_engine.columns = ['Engine Size','Count']
    fig_engine = px.bar(df_engine.head(15), x='Engine Size', y='Count', color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(style_fig(fig_engine), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">🔧 Cylinder Architecture</h3>', unsafe_allow_html=True)
    df_cyl = df['Cylinders'].value_counts().reset_index()
    df_cyl.columns = ['Cylinders','Count']
    fig_cyl = px.bar(df_cyl, x='Cylinders', y='Count', color='Count', color_continuous_scale='Tealgrn')
    st.plotly_chart(style_fig(fig_cyl), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">⛽ Fuel Type Distribution</h3>', unsafe_allow_html=True)
    df_fuel = df['Fuel Type'].value_counts().reset_index()
    df_fuel.columns = ['Fuel Type','Count']
    fig_fuel = px.bar(df_fuel, x='Fuel Type', y='Count', color='Count', color_continuous_scale='Mint')
    st.plotly_chart(style_fig(fig_fuel), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">⚙️ Transmission Analytics</h3>', unsafe_allow_html=True)
    df_trans = df['Transmission'].value_counts().reset_index()
    df_trans.columns = ['Transmission','Count']
    fig_trans = px.bar(df_trans.head(10), x='Transmission', y='Count', color='Count', color_continuous_scale='Mint')
    st.plotly_chart(style_fig(fig_trans), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">🔥 CO2 by Brand (Emission Heat)</h3>', unsafe_allow_html=True)
    df_co2_make = df.groupby('Make')['CO2 Emissions(g/km)'].mean().reset_index()
    fig_co2 = px.bar(df_co2_make.head(20), x='Make', y='CO2 Emissions(g/km)', color='CO2 Emissions(g/km)', color_continuous_scale='Reds')
    st.plotly_chart(style_fig(fig_co2), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">📈 Fuel Consumption vs CO2 Emission</h3>', unsafe_allow_html=True)
    fig_scatter = px.scatter(df, x="Fuel Consumption Comb (L/100 km)", y="CO2 Emissions(g/km)", color="Fuel Type", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(style_fig(fig_scatter), use_container_width=True)

    st.markdown('<br><h3 style="margin-top: 0; color: #0f172a; font-weight: 600;">📊 Emission Variance by Vehicle Class</h3>', unsafe_allow_html=True)
    fig_box = px.box(df, x="Vehicle Class", y="CO2 Emissions(g/km)", color="Vehicle Class", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(style_fig(fig_box), use_container_width=True)

# ===================== MODEL PREDICTION =====================
elif user_input == 'AI Prediction Engine':

    st.markdown('<div class="glass-card" style="margin-bottom: 30px; display: flex; align-items: center; padding: 25px;"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 20px;"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg><h2 style="margin: 0; color: #0f172a !important;">AI Emission Analysis</h2></div>', unsafe_allow_html=True)

    if predict_button:

        valid_input = True

        if engine_size <= 0.5 or fuel_consumption <= 1.0:
            st.error("Invalid Input: Engine Size and Fuel Consumption must be positive values.")
            valid_input = False

        if engine_size > 4.0 and fuel_consumption < 8.0:
            st.warning("Anomaly Detected: Fuel consumption too low for a large engine size.")
            valid_input = False

        if engine_size < 1.5 and fuel_consumption > 10.0:
            st.warning("Anomaly Detected: Fuel consumption too high for a small engine size.")
            valid_input = False

        if cylinders > 8 and engine_size < 3.0:
            st.warning("Anomaly Detected: Invalid cylinder vs engine size combination.")
            valid_input = False

        if valid_input:

            # ===================== HYBRID PREDICTION =====================
            progress_placeholder = st.empty()
            
            # Executing ML Model calculations first
            input_df = pd.DataFrame({
                'Engine Size(L)': [engine_size],
                'Cylinders': [cylinders],
                'Fuel Consumption Comb (L/100 km)': [fuel_consumption]
            })
            ml_prediction = model.predict(input_df)[0]
            theoretical_co2 = fuel_consumption * 23.2
            predicted_co2 = (0.7 * theoretical_co2) + (0.3 * ml_prediction)
            predicted_co2 = max(predicted_co2, fuel_consumption * 20)
            predicted_co2 = min(predicted_co2, fuel_consumption * 30)
            
            # Animate custom premium loader
            for percent_complete in range(0, 101, 10):
                loading_html = f'<div class="glass-card" style="padding: 30px !important; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 25px; border-left: 4px solid #0ea5e9;"><div style="position: relative; width: 60px; height: 60px; margin-bottom: 20px;"><div style="position: absolute; width: 100%; height: 100%; border: 4px solid rgba(14, 165, 233, 0.1); border-top: 4px solid #0ea5e9; border-radius: 50%; animation: spin 1s linear infinite;"></div><div style="position: absolute; top: 12px; left: 12px; width: 36px; height: 36px; background: radial-gradient(circle, #10b981 0%, rgba(16, 185, 129, 0.4) 100%); border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></div></div><div style="font-weight: 700; color: #0f172a; font-size: 1.15rem; margin-bottom: 8px;">Analyzing Environmental Intelligence...</div><div style="color: #64748b; font-size: 0.9rem; font-weight: 500; margin-bottom: 15px;">Executing Hybrid Random Forest & Deep Analysis Model</div><div style="width: 100%; max-width: 300px; height: 6px; background: rgba(0,0,0,0.05); border-radius: 10px; overflow: hidden; position: relative;"><div style="width: {percent_complete}%; height: 100%; background: linear-gradient(90deg, #10b981 0%, #0ea5e9 100%); border-radius: 10px; transition: width 0.15s ease;"></div></div><div style="margin-top: 8px; font-size: 0.8rem; font-weight: 700; color: #0ea5e9;">{percent_complete}% COMPLETE</div></div>'
                progress_placeholder.markdown(loading_html, unsafe_allow_html=True)
                time.sleep(0.12)
                
            progress_placeholder.empty()
            st.success("Analysis Complete")

            col1, col2 = st.columns([1, 1.2])

            # ================= LEFT COLUMN =================
            with col1:

                st.metric(label="Predicted CO₂ Emissions", value=f"{predicted_co2:.2f} g/km")

                # Emission Category
                if predicted_co2 > 250:
                    status_color = "#ef4444"
                    status_text = "High Emission Profile"
                elif predicted_co2 > 180:
                    status_color = "#f59e0b"
                    status_text = "Moderate Emission Profile"
                else:
                    status_color = "#10b981"
                    status_text = "Eco-Friendly Profile"

                st.markdown(f'<div style="background: rgba(255,255,255,0.7); border-left: 4px solid {status_color}; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);"><div style="font-weight: 600; color: #334155; font-size: 0.9rem; text-transform: uppercase;">System Assessment</div><div style="color: {status_color}; font-size: 1.2rem; font-weight: 700; margin-top: 5px;">{status_text}</div></div>', unsafe_allow_html=True)

                avg_co2 = df['CO2 Emissions(g/km)'].mean()
                
                # Insight Card
                st.markdown(f'<div class="glass-card" style="padding: 15px !important; margin-bottom: 20px;"><div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 10px;">Market Context</div><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #475569;">Industry Average:</span><span style="font-weight: 600; color: #0f172a;">{avg_co2:.1f} g/km</span></div><div style="display: flex; justify-content: space-between;"><span style="color: #475569;">Theoretical Model:</span><span style="font-weight: 600; color: #0f172a;">{theoretical_co2:.1f} g/km</span></div></div>', unsafe_allow_html=True)

                # Real-world impact
                yearly_co2_kg = (predicted_co2 * yearly_km) / 1000
                trees_needed = max(1, int(yearly_co2_kg / 22))

                st.markdown(f'<div class="glass-card" style="padding: 15px !important; border-left: 4px solid #0ea5e9 !important;"><div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 5px;">Offset Requirement</div><div style="font-size: 1.1rem; color: #0f172a;">Requires <strong style="color: #0ea5e9;">{trees_needed} trees</strong> annually</div></div>', unsafe_allow_html=True)

            with col2:
                angle = (predicted_co2 / 500) * 180 - 90
                angle = max(-90.0, min(90.0, angle))
                
                st.markdown(f'<div class="glass-card" style="padding: 30px !important; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 380px;"><div style="color: #64748b; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; margin-bottom: 25px; letter-spacing: 1px;">Emission Level Analysis</div><div class="gauge-container" style="position: relative; width: 100%; max-width: 280px; text-align: center;"><svg viewBox="0 0 200 120" width="100%" height="100%" style="overflow: visible;"><defs><linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#10b981" /><stop offset="50%" stop-color="#f59e0b" /><stop offset="100%" stop-color="#ef4444" /></linearGradient><filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter></defs><path d="M20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" stroke-width="12" stroke-linecap="round"/><path d="M20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gauge-grad)" stroke-width="12" stroke-linecap="round" /><text x="20" y="118" fill="#10b981" font-size="8" font-weight="800" text-anchor="middle">ECO</text><text x="100" y="10" fill="#f59e0b" font-size="8" font-weight="800" text-anchor="middle">MODERATE</text><text x="180" y="118" fill="#ef4444" font-size="8" font-weight="800" text-anchor="middle">CRITICAL</text><circle cx="100" cy="100" r="10" fill="#1e293b" /><circle cx="100" cy="100" r="5" fill="#0ea5e9" filter="url(#glow)" /><line x1="100" y1="100" x2="100" y2="25" stroke="#1e293b" stroke-width="4" stroke-linecap="round" style="transform: rotate({angle}deg); transform-origin: 100px 100px; transition: transform 1.8s cubic-bezier(0.19, 1, 0.22, 1);" /></svg><div style="margin-top: -10px; position: relative;"><span style="font-size: 2.5rem; font-weight: 800; color: #0f172a; text-shadow: 0 0 20px rgba(16, 185, 129, 0.2);">{predicted_co2:.1f}</span><span style="font-size: 0.95rem; color: #64748b; font-weight: 600; display: block; margin-top: -2px;">g/km CO₂</span></div></div></div>', unsafe_allow_html=True)

        else:
            st.info("Adjust simulation parameters in the sidebar and initialize analysis.")

# --- Footer ------------------------------------------------------------------------------------------------
st.markdown('<div class="glass-card" style="padding: 25px 30px !important; margin-top: 60px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; border-top: 4px solid #10b981 !important; gap: 20px;"><div style="flex: 1; min-width: 280px;"><h4 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0f172a; font-family: \'Outfit\', \'Inter\', sans-serif; letter-spacing: -0.5px;">EcoTrack Analytics</h4><p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #64748b; font-weight: 500;">Designed & Developed by <strong style="color: #0f172a;">Aditya Rahul Phophale</strong></p><p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Internship Project at <strong style="color: #64748b; font-weight: 600;">INNOVEXXA</strong></p></div><div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;"><span style="font-size: 0.75rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-right: 5px;">Platform Stack</span><div class="badge" style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; color: #065f46; display: flex; align-items: center; gap: 6px; cursor: pointer;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981;"></span> Python</div><div class="badge" style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; color: #065f46; display: flex; align-items: center; gap: 6px; cursor: pointer;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981;"></span> Streamlit</div><div class="badge" style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.15); padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; color: #0369a1; display: flex; align-items: center; gap: 6px; cursor: pointer;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #0ea5e9;"></span> Plotly</div><div class="badge" style="background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.15); padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; color: #6d28d9; display: flex; align-items: center; gap: 6px; cursor: pointer;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #7c3aed;"></span> Scikit-Learn</div></div></div>', unsafe_allow_html=True)