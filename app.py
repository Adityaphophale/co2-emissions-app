# ===================== IMPORTS =====================
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
.stApp {
    background: linear-gradient(-45deg, #caf0f8, #90e0ef, #ade8f4, #ffdd99);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
}
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.main-title {
    background: rgba(255,255,255,0.8);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 2.3rem;
    font-weight: bold;
}
.section-title {
    font-size: 1.4rem;
    font-weight: bold;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv('co2 Emissions.csv', low_memory=False)
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

    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.write(f"R2 Score: {r2_score(y_test, preds):.2f}")
    st.sidebar.write(f"MAE: {mean_absolute_error(y_test, preds):.2f}")

    return model

model = get_model()

# ===================== SIDEBAR =====================
with st.sidebar:

    st.markdown("""
        <div style="text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.4);
        border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h2 style="color: #03506F; margin: 0; font-weight: 800;">🌿 EcoTrack </h2>
            <p style="color: #555; font-size: 0.85rem; margin: 0;">
                AI Emission Intelligence
            </p>
        </div>
    """, unsafe_allow_html=True)

    user_input = st.radio(
        "Navigation",
        ["📊 Visualization", "🎯 AI Model Prediction"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if user_input == "🎯 AI Model Prediction":

        st.markdown("### ⚙️ Engine Parameters")
        st.caption("Adjust sliders for simulation")

        engine_size = st.slider("Engine Size (L)", 0.5, 8.0, 2.0, 0.1)
        cylinders = st.slider("Cylinder Count", 2, 16, 4)
        fuel_consumption = st.slider("Fuel Cons. (L/100 km)", 2.0, 30.0, 7.5, 0.1)

        st.markdown("---")

        st.markdown("### 🛣️ Scenario Analysis")
        yearly_km = st.number_input("Annual Distance (km)", 1000, 100000, 15000, step=1000)

        yearly_emission = (fuel_consumption * yearly_km) / 100
        st.success(f"🌍 Yearly Fuel Usage: {yearly_emission:.0f} L")

        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button("🚀 Run AI Prediction", use_container_width=True)

    else:
        predict_button = False

    st.markdown("<br><br>", unsafe_allow_html=True)

# ===================== VISUALIZATION =====================
# 🔒 UNCHANGED (your exact code kept)

if user_input == '📊 Visualization':

    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.85);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;">
        <h1 style="text-align:center; color:#03506F;">
            🌍 CO2 Emission Intelligence Dashboard
        </h1>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚘 Total Cars", len(df))
    k2.metric("📊 Avg CO2", f"{df['CO2 Emissions(g/km)'].mean():.1f} g/km")
    k3.metric("⚠️ Max CO2", f"{df['CO2 Emissions(g/km)'].max():.1f} g/km")
    k4.metric("⛽ Avg Fuel", f"{df['Fuel Consumption Comb (L/100 km)'].mean():.1f} L/100km")

    st.markdown("---")

    st.markdown("### 💽 Dataset")
    st.dataframe(df)

    st.markdown("### 🚘 Brands")
    df_brand = df['Make'].value_counts().reset_index()
    df_brand.columns = ['Make','Count']

    fig1 = px.bar(df_brand, x='Make', y='Count', color='Count', color_continuous_scale='Turbo')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🏁 Top Models")
    df_model = df['Model'].value_counts().reset_index()
    df_model.columns = ['Model','Count']

    fig2 = px.bar(df_model.head(25), x='Model', y='Count', color='Count', color_continuous_scale='Plasma')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🚙 Vehicle Class")
    df_vc = df['Vehicle Class'].value_counts().reset_index()
    df_vc.columns = ['Vehicle Class','Count']

    fig3 = px.bar(df_vc, x='Vehicle Class', y='Count', color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🛠️ Engine Size")
    df_engine = df['Engine Size(L)'].value_counts().reset_index()
    df_engine.columns = ['Engine Size','Count']

    fig4 = px.bar(df_engine, x='Engine Size', y='Count', color='Count', color_continuous_scale='Magma')
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🔩 Cylinders")
    df_cyl = df['Cylinders'].value_counts().reset_index()
    df_cyl.columns = ['Cylinders','Count']

    fig5 = px.bar(df_cyl, x='Cylinders', y='Count', color='Count', color_continuous_scale='Plasma')
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### ⚙️ Transmission")
    df_trans = df['Transmission'].value_counts().reset_index()
    df_trans.columns = ['Transmission','Count']

    fig6 = px.bar(df_trans, x='Transmission', y='Count', color='Count', color_continuous_scale='Turbo')
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### ⛽ Fuel Type")
    df_fuel = df['Fuel Type'].value_counts().reset_index()
    df_fuel.columns = ['Fuel Type','Count']

    fig7 = px.bar(df_fuel, x='Fuel Type', y='Count', color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### 📉 CO2 by Brand")
    df_co2_make = df.groupby('Make')['CO2 Emissions(g/km)'].mean().reset_index()

    fig8 = px.bar(df_co2_make, x='Make', y='CO2 Emissions(g/km)', color='CO2 Emissions(g/km)', color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("### 📊 Fuel vs CO2")
    fig9 = px.scatter(df, x="Fuel Consumption Comb (L/100 km)", y="CO2 Emissions(g/km)", color="Fuel Type")
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("### 📦 Box Plot")
    fig10 = px.box(df, x="Vehicle Class", y="CO2 Emissions(g/km)", color="Vehicle Class")
    st.plotly_chart(fig10, use_container_width=True)

# ===================== MODEL PREDICTION =====================
elif user_input == '🎯 AI Model Prediction':

    st.markdown("<div class='main-title'>🚗 Predict CO2 Emissions</div>", unsafe_allow_html=True)

    if predict_button:

        valid_input = True

        if engine_size <= 0.5 or fuel_consumption <= 1.0:
            st.error("Engine Size and Fuel Consumption must be positive values.")
            valid_input = False

        if engine_size > 4.0 and fuel_consumption < 8.0:
            st.warning("Fuel consumption too low for large engine.")
            valid_input = False

        if engine_size < 1.5 and fuel_consumption > 10.0:
            st.warning("Fuel consumption too high for small engine.")
            valid_input = False

        if cylinders > 8 and engine_size < 3.0:
            st.warning("Invalid cylinder vs engine size combination.")
            valid_input = False

        if valid_input:

            # ===================== HYBRID PREDICTION =====================
            input_df = pd.DataFrame({
                'Engine Size(L)': [engine_size],
                'Cylinders': [cylinders],
                'Fuel Consumption Comb (L/100 km)': [fuel_consumption]
            })

            ml_prediction = model.predict(input_df)[0]

            # 🔬 Real-world formula
            theoretical_co2 = fuel_consumption * 23.2

            # 🔥 Final accurate prediction (HYBRID)
            predicted_co2 = (0.7 * theoretical_co2) + (0.3 * ml_prediction)

            # Safety bounds
            predicted_co2 = max(predicted_co2, fuel_consumption * 20)
            predicted_co2 = min(predicted_co2, fuel_consumption * 30)

            st.success("✅ Prediction successful!")

            col1, col2 = st.columns(2)

            # ================= LEFT COLUMN =================
            with col1:

                st.metric(label="Predicted CO₂ Emissions", value=f"{predicted_co2:.2f} g/km")

                # Emission Category
                if predicted_co2 > 250:
                    st.error("High Emission Vehicle 🚨")
                elif predicted_co2 > 180:
                    st.warning("Moderate Emission ⚠️")
                else:
                    st.success("Eco Friendly Vehicle 🌱")

                # Comparison
                st.write(f"🔬 Theoretical CO₂: {theoretical_co2:.2f} g/km")

                avg_co2 = df['CO2 Emissions(g/km)'].mean()
                st.info(f"📊 Industry Avg CO₂: {avg_co2:.2f} g/km")

                # 🔥 Insight
                if predicted_co2 > avg_co2:
                    st.warning("🚨 This vehicle emits above industry average CO₂.")
                else:
                    st.success("🌱 This vehicle performs better than industry average.")

                st.markdown("<br>", unsafe_allow_html=True)

                # 🌳 Real-world impact
                yearly_co2_kg = (predicted_co2 * yearly_km) / 1000
                trees_needed = max(1, int(yearly_co2_kg / 22))

                st.info(f"🌳 **Offset Requirement:** You need approx **{trees_needed} trees** to offset yearly emissions.")

                # Optional visual
                st.progress(min(trees_needed / 100, 1.0))

            # ================= RIGHT COLUMN =================
            with col2:

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_co2,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Emission Level"},
                    gauge={
                        'axis': {'range': [0, 500]},
                        'bar': {'color': "#03506F"},
                        'steps': [
                            {'range': [0, 150], 'color': "#a7f3d0"},
                            {'range': [150, 300], 'color': "#fef08a"},
                            {'range': [300, 500], 'color': "#fecaca"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'value': predicted_co2
                        }
                    }
                ))

                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Enter vehicle specifications in the sidebar and click Run AI Prediction.")

# --- Footer ------------------------------------------------------------------------------------------------
st.markdown("""
<hr>
<center>
    <i>🌍 Developed by Aditya Rahul Phophale</i>
</center>
""", unsafe_allow_html=True)