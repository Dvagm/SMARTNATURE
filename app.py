import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import pytz

# ========== CONFIG ==========
st.set_page_config(page_title="🌾 Smart Farming Dashboard", layout="wide")
st.title("🌱 Smart Farming Dashboard - Desa Kihiyang")
st.caption("📡 Terintegrasi dengan AI & Cuaca Real-Time dari Tomorrow.io")
st_autorefresh(interval=60 * 60 * 1000, key="refresh")

# ========== API Setup ==========
API_KEY = "TEVUVSLhMhDhVyN6F5CZwUAf3R40nvlU"
LAT, LON = -6.4539, 107.7631
url = "https://api.tomorrow.io/v4/weather/forecast"
params = {"location": f"{LAT},{LON}", "apikey": API_KEY}
response = requests.get(url, params=params)
data = response.json()
hourly = data['timelines']['hourly']
df = pd.json_normalize(hourly)

# ========== Preprocessing ==========
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].dt.tz_convert('Asia/Jakarta')
df.columns = [col.replace('values.', '') for col in df.columns]
df.drop(columns=[col for col in df.columns if 'snow' in col.lower()], inplace=True)

# ========== Feature Engineering ==========
def categorize_uv(uv):
    if uv <= 2: return "Rendah"
    elif 3 <= uv <= 5: return "Sedang"
    else: return "Tinggi"

def stress_label(row):
    return 1 if (row['temperature'] > 33 or row['humidity'] < 40 or row['uvIndex'] > 6 or row['evapotranspiration'] > 0.3) else 0

def wind_risk(row):
    return 1 if row.get('windSpeed', 0) > 30 else 0

def uv_risk(row):
    return 1 if row.get('uvIndex', 0) > 7 else 0

df['uv_category'] = df['uvIndex'].apply(categorize_uv)
df['plant_stress'] = df.apply(stress_label, axis=1)
df['wind_risk'] = df.apply(wind_risk, axis=1)
df['uv_risk'] = df.apply(uv_risk, axis=1)
df['evapo_risk'] = df['evapotranspiration'].apply(lambda x: 1 if x > 0.4 else 0)
df['rain_risk'] = df['rainIntensity'].apply(lambda x: 1 if x > 2 else 0)

# ========== Load Models ==========
model1 = joblib.load("model_stress_tanaman.pkl")
model2 = joblib.load("model_risiko_angin.pkl")
model3 = joblib.load("model_risiko_uv.pkl")
model4 = joblib.load("model_evapotranspirasi.pkl")
model5 = joblib.load("model_curah_hujan.pkl")

# ========== Predict ==========
df['pred_stress'] = model1.predict(df[['temperature', 'humidity', 'uvIndex', 'evapotranspiration', 'cloudCover', 'rainIntensity']])
df['pred_wind'] = model2.predict(df[['windSpeed', 'windGust', 'temperature', 'humidity']])
df['pred_uv'] = model3.predict(df[['uvIndex', 'cloudCover', 'humidity', 'temperature']])
df['pred_evapo'] = model4.predict(df[['temperature', 'humidity', 'cloudCover', 'windSpeed']])
df['pred_rain'] = model5.predict(df[['precipitationProbability', 'rainIntensity', 'humidity', 'temperature']])

# ========== Sidebar ==========
with st.sidebar:
    st.header("📁 Ekspor Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="smart_farming_forecast.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("🕒 Data terakhir di-update:")
    wib_now = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M WIB')
    st.info(f"{wib_now}")

# ========== Tabs ==========
tabs = st.tabs([
    "📊 Ringkasan", "🌡️ Suhu & Kelembapan", "🔆 UV & Awan",
    "🌬️ Angin", "💧 Evapotranspirasi", "🌧️ Hujan", "📌 Korelasi"
])

# === [Tab 1] Ringkasan ===
with tabs[0]:
    st.subheader("📊 Ringkasan Prediksi Model AI")
    st.dataframe(df[['time', 'temperature', 'humidity', 'uvIndex',
                     'pred_stress', 'pred_wind', 'pred_uv', 'pred_evapo', 'pred_rain']].head(24))

    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 Stres Tanaman", f"{df['pred_stress'].sum()} kejadian")
    col2.metric("🌬️ Risiko Angin", f"{df['pred_wind'].sum()} kejadian")
    col3.metric("🔆 Paparan UV Tinggi", f"{df['pred_uv'].sum()} kejadian")

# === [Tab 2] Suhu & Kelembapan ===
with tabs[1]:
    st.subheader("🌡️ Tren Suhu & Kelembapan")
    fig = px.line(df, x="time", y=["temperature", "humidity"], title="Per Jam")
    st.plotly_chart(fig, use_container_width=True)

# === [Tab 3] UV & Awan ===
with tabs[2]:
    st.subheader("🔆 Indeks UV & Cloud Cover")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['uvIndex'], name="UV Index", mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['cloudCover'], name="Cloud Cover", mode='lines+markers'))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(df, x="uv_category", title="Distribusi Kategori UV", color="uv_category")
    st.plotly_chart(fig2, use_container_width=True)

# === [Tab 4] Angin ===
with tabs[3]:
    st.subheader("🌬️ Kecepatan Angin dan Prediksi Risiko")
    fig = px.bar(df, x="time", y="windSpeed", color="pred_wind", title="Risiko Angin")
    st.plotly_chart(fig, use_container_width=True)

# === [Tab 5] Evapotranspirasi ===
with tabs[4]:
    st.subheader("💧 Evapotranspirasi (ET) dan Risiko")
    fig = px.area(df, x="time", y="evapotranspiration", color="pred_evapo", title="Evapotranspirasi")
    st.plotly_chart(fig, use_container_width=True)

# === [Tab 6] Curah Hujan ===
with tabs[5]:
    st.subheader("🌧️ Intensitas Hujan & Probabilitas")
    fig1 = px.line(df, x="time", y="rainIntensity", color="pred_rain", title="Curah Hujan vs Risiko")
    st.plotly_chart(fig1, use_container_width=True)

# === [Tab 7] Korelasi ===
with tabs[6]:
    st.subheader("📌 Korelasi Antar Variabel Cuaca")
    selected_cols = ['temperature', 'humidity', 'uvIndex', 'windSpeed',
                     'evapotranspiration', 'rainIntensity']
    corr = df[selected_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Heatmap Korelasi")
    st.plotly_chart(fig, use_container_width=True)
