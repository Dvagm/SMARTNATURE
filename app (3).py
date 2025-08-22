%%writefile app.py
import os
import io
import threading
import time
from datetime import datetime, timedelta
import pytz
import zipfile
from pytz import timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import requests
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from telegram import Bot
import joblib

# =========================
# KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="🌾 Smart Farming Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palet warna 'Smart Farming Sky Palette' yang lembut dan harmonis
PALETTE = {
    'green_soft': '#8BC34A',
    'blue_sky': '#81D4FA',
    'yellow_sun': '#FFEB3B',
    'orange_sunset': '#FF9800',
    'red_danger': '#F44336',
    'blue_rain': '#03A9F4',
    'grey_cloud': '#BDBDBD',
    'text_primary': '#212121',
    'text_secondary': '#616161',
}

# --- PERUBAHAN UTAMA: DATA LOKASI DAN API KEY DIJADIKAN KAMUS ---
LOCATIONS = {
    "Desa Kihiyang": {"lat": -6.4539, "lon": 107.7631},
    "Desa Cikande": {"lat": -6.4025, "lon": 107.8284},
    "Desa Gantar": {"lat": -6.3768, "lon": 107.9734},
}

TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY", "TEVUVSLhMhDhVyN6F5CZwUAf3R40nvlU")
TOMORROW_URL = "https://api.tomorrow.io/v4/weather/forecast"

# Telegram Bot
DEFAULT_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8093589762:AAE70b-D8q-W8jASLT8BEkEAsr7pV6jsjvM")
DEFAULT_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Data dummy untuk tanaman
PLANT_DATA = {
    "cabai": {
        "nama": "Cabai",
        "kategori": "Sayuran",
        "kebutuhan_air": "Sedang",
        "kebutuhan_cahaya": "Tinggi",
        "kondisi_ideal": {
            "moisture_min": 45, "moisture_max": 65,
            "ph_min": 6.0, "ph_max": 6.8
        },
        "rekomendasi_perawatan": "Siram saat tanah kering, hindari kelembaban berlebihan. Lindungi dari panas ekstrem."
    },
    "tomat": {
        "nama": "Tomat",
        "kategori": "Sayuran",
        "kebutuhan_air": "Tinggi",
        "kebutuhan_cahaya": "Tinggi",
        "kondisi_ideal": {
            "moisture_min": 50, "moisture_max": 70,
            "ph_min": 6.0, "ph_max": 6.8
        },
        "rekomendasi_perawatan": "Pastikan drainase baik, airi secara teratur. Rentan terhadap penyakit jamur."
    },
    "mangga": {
        "nama": "Mangga",
        "kategori": "Buah",
        "kebutuhan_air": "Rendah",
        "kebutuhan_cahaya": "Tinggi",
        "kondisi_ideal": {
            "moisture_min": 30, "moisture_max": 50,
            "ph_min": 5.5, "ph_max": 6.5
        },
        "rekomendasi_perawatan": "Penyiraman dikurangi saat berbunga. Pohon dewasa lebih tahan kering."
    },
}

# =========================
# FUNGSI UTILITY & DATA
# =========================
def safe_to_datetime_localized(series, tz_target='Asia/Jakarta'):
    s = pd.to_datetime(series, errors='coerce')
    if s.dt.tz is None:
        s = s.dt.tz_localize('UTC').dt.tz_convert(tz_target)
    else:
        s = s.dt.tz_convert(tz_target)
    return s

@st.cache_data(ttl=3600)
def fetch_forecast(lat, lon):
    """Mengambil data cuaca per jam dari API Tomorrow.io untuk lokasi tertentu."""
    params = {"location": f"{lat},{lon}", "apikey": TOMORROW_API_KEY, "timesteps": "1h"}
    try:
        r = requests.get(TOMORROW_URL, params=params, timeout=15)
        r.raise_for_status()
        df = pd.json_normalize(r.json()["timelines"]["hourly"])
    except Exception as e:
        st.warning(f"Gagal mengambil data: {e} — kembali ke frame 24 jam kosong.")
        idx = pd.date_range(start=datetime.now(), periods=24, freq="H")
        df = pd.DataFrame({"time": idx})

    df['time'] = safe_to_datetime_localized(df.get('time', pd.Series(dtype='datetime64[ns]')))
    df.columns = [c.replace("values.", "") for c in df.columns]
    needed_cols = ['temperature', 'humidity', 'uvIndex', 'cloudCover', 'evapotranspiration',
                   'windSpeed', 'windGust', 'rainIntensity', 'precipitationProbability']
    for n in needed_cols:
        if n not in df.columns:
            df[n] = 0.0
    for c in needed_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df.sort_values('time').reset_index(drop=True)

def categorize_uv(uv):
    try:
        uv = float(uv)
    except (ValueError, TypeError):
        return "Rendah"
    if uv <= 2: return "Rendah"
    elif 3 <= uv <= 5: return "Sedang"
    elif 6 <= uv <= 7: return "Tinggi"
    elif 8 <= uv <= 10: return "Sangat Tinggi"
    else: return "Ekstrem"

def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def add_engineered_columns(df, mode_ai):
    df_copy = df.copy()
    df_copy['uv_category'] = df_copy['uvIndex'].apply(categorize_uv)
    
    # --- Pilihan mode AI ---
    if mode_ai == "model":
        features_map = {
            'stress': (['temperature', 'humidity', 'uvIndex', 'evapotranspiration', 'cloudCover', 'rainIntensity'], 'pred_stress'),
            'wind': (['windSpeed', 'windGust', 'temperature', 'humidity'], 'pred_wind'),
            'uv': (['uvIndex', 'cloudCover', 'humidity', 'temperature'], 'pred_uv'),
            'evapo': (['temperature', 'humidity', 'cloudCover', 'windSpeed'], 'pred_evapo'),
            'rain': (['precipitationProbability', 'rainIntensity', 'humidity', 'temperature'], 'pred_rain')
        }
        
        for model_name, (features, pred_col) in features_map.items():
            # PERUBAHAN UTAMA DI SINI
            # Mengubah path untuk menunjuk ke folder 'models'
            model_path_base = f"models/model_risiko_{model_name}.pkl"
            
            # Khusus untuk model curah hujan dan stress tanaman, nama filenya berbeda
            if model_name == 'rain':
                model_path = "models/model_curah_hujan.pkl"
            elif model_name == 'stress':
                model_path = "models/model_stress_tanaman.pkl"
            else:
                model_path = model_path_base
            
            model = try_load_model(model_path)
            
            if model:
                try:
                    missing_features = [f for f in features if f not in df_copy.columns]
                    if missing_features:
                        st.warning(f"Model '{model_name}' missing features: {missing_features}. Cannot predict.")
                        df_copy[pred_col] = 0
                        continue
                    
                    for f in features:
                        df_copy[f] = pd.to_numeric(df_copy[f], errors='coerce').fillna(df_copy[f].mean() if df_copy[f].mean() is not np.nan else 0.0)
                    
                    df_copy[pred_col] = model.predict(df_copy[features])
                    
                except Exception as e:
                    st.warning(f"Gagal memprediksi dengan model '{model_name}': {e}. Menggunakan heuristik.")
                    # Jika model gagal, fallback ke heuristik
                    if pred_col == 'pred_stress':
                        df_copy[pred_col] = ((df_copy['temperature'] > 33) | (df_copy['humidity'] < 40) | (df_copy['uvIndex'] > 7) | (df_copy['evapotranspiration'] > 0.4)).astype(int)
                    elif pred_col == 'pred_wind':
                        df_copy[pred_col] = (df_copy['windSpeed'] > 25).astype(int)
                    elif pred_col == 'pred_uv':
                        df_copy[pred_col] = (df_copy['uvIndex'] > 7).astype(int)
                    elif pred_col == 'pred_evapo':
                        df_copy[pred_col] = (df_copy['evapotranspiration'] > 0.4).astype(int)
                    elif pred_col == 'pred_rain':
                        df_copy[pred_col] = (df_copy['rainIntensity'] > 2).astype(int)
            else:
                st.warning(f"File model '{model_path}' tidak ditemukan. Menggunakan heuristik.")
                # Fallback ke heuristik jika model tidak ditemukan
                if pred_col == 'pred_stress':
                    df_copy[pred_col] = ((df_copy['temperature'] > 33) | (df_copy['humidity'] < 40) | (df_copy['uvIndex'] > 7) | (df_copy['evapotranspiration'] > 0.4)).astype(int)
                elif pred_col == 'pred_wind':
                    df_copy[pred_col] = (df_copy['windSpeed'] > 25).astype(int)
                elif pred_col == 'pred_uv':
                    df_copy[pred_col] = (df_copy['uvIndex'] > 7).astype(int)
                elif pred_col == 'pred_evapo':
                    df_copy[pred_col] = (df_copy['evapotranspiration'] > 0.4).astype(int)
                elif pred_col == 'pred_rain':
                    df_copy[pred_col] = (df_copy['rainIntensity'] > 2).astype(int)
    
    else: # mode_ai == "heuristik"
        df_copy['pred_stress'] = ((df_copy['temperature'] > 33) | (df_copy['humidity'] < 40) | (df_copy['uvIndex'] > 7) | (df_copy['evapotranspiration'] > 0.4)).astype(int)
        df_copy['pred_wind'] = (df_copy['windSpeed'] > 25).astype(int)
        df_copy['pred_uv'] = (df_copy['uvIndex'] > 7).astype(int)
        df_copy['pred_evapo'] = (df_copy['evapotranspiration'] > 0.4).astype(int)
        df_copy['pred_rain'] = (df_copy['rainIntensity'] > 2).astype(int)

    # Memastikan kolom prediksi bertipe numerik
    for c in ['pred_stress', 'pred_wind', 'pred_uv', 'pred_evapo', 'pred_rain']:
        df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce').fillna(0.0)

    return df_copy

def get_conclusion(df, category):
    if category == 'uv':
        high_hours = (df['pred_uv'] > 0).sum()
        if high_hours >= 4: return f"🔆 **UV sangat tinggi** selama {high_hours} jam. Gunakan shading dan perlindungan daun."
        elif high_hours >= 1: return f"🔆 Terdapat {high_hours} jam UV > 7. Siapkan perlindungan di periode siang."
        else: return "✅ UV aman, tidak ada tindakan khusus."
    elif category == 'rain':
        heavy_hours = (df['pred_rain'] > 0).sum()
        if heavy_hours >= 3: return f"🌧 **Hujan intens** selama {heavy_hours} jam. Pastikan drainase dan hindari irigasi berlebihan."
        elif df['rainIntensity'].mean() < 0.2: return "⚠️ **Curah hujan rendah**. Pertimbangkan irigasi tambahan."
        else: return "✅ Curah hujan normal."
    elif category == 'evapo':
        mean_evapo = df['evapotranspiration'].mean()
        if mean_evapo > 0.35: return "💧 **Evapotranspirasi tinggi**. Tanaman membutuhkan penyiraman lebih sering."
        else: return "✅ Evapotranspirasi normal."
    elif category == 'wind':
        gust_hours = (df['pred_wind'] > 0).sum()
        if gust_hours >= 1: return f"🌬 **Angin kencang** ({gust_hours} jam). Amankan tutupan dan tanaman muda."
        elif df['windSpeed'].mean() > 20: return "⚠️ Kecepatan angin rata-rata tinggi. Waspadai tanaman muda."
        else: return "✅ Kondisi angin aman."
    elif category == 'stress':
        total_stress_hours = int(df['pred_stress'].sum())
        if total_stress_hours >= 6: return f"⚠️ **Prediksi stres tanaman tinggi** ({total_stress_hours} jam). Cek kesehatan tanaman dan siram jika perlu."
        elif total_stress_hours >= 2: return f"⚠️ Prediksi stres muncul ({total_stress_hours} jam). Pantau lebih dekat."
        else: return "✅ Prediksi stres rendah."
    return "ℹ️ Tidak ada analisis spesifik."

# =========================
# FUNGSI UNTUK DATA TANAH ARDUINO
# =========================
def generate_mock_arduino_data(hours=24):
    """Generates mock data for Arduino smart farming sensors."""
    now = datetime.now(timezone("Asia/Jakarta"))
    timestamps = [now + timedelta(hours=i) for i in range(hours)]

    np.random.seed(42) # for reproducibility

    soil_moisture = np.random.uniform(30, 70, hours)
    soil_temperature = np.random.uniform(25, 35, hours)
    soil_ph = np.random.uniform(5.5, 7.5, hours)
    nitrogen = np.random.uniform(40, 100, hours)
    phosphorus = np.random.uniform(10, 30, hours)
    potassium = np.random.uniform(80, 200, hours)
    light_intensity = np.random.uniform(5000, 50000, hours)

    df_arduino = pd.DataFrame({
        'timestamp': timestamps,
        'soil_moisture': soil_moisture,
        'soil_temperature': soil_temperature,
        'soil_ph': soil_ph,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'light_intensity': light_intensity
    })
    return df_arduino

def get_soil_recommendations(df_soil):
    """Provides AI-like recommendations based on mock soil data."""
    recommendations = []
    
    avg_moisture = df_soil['soil_moisture'].mean()
    if avg_moisture < 40:
        recommendations.append("💧 **Kelembaban tanah rendah.** Pertimbangkan irigasi tambahan.")
    elif avg_moisture > 65:
        recommendations.append("💧 **Kelembaban tanah tinggi.** Pastikan drainase baik, hindari penyiraman berlebihan.")
    else:
        recommendations.append("💧 Kelembaban tanah optimal.")

    avg_ph = df_soil['soil_ph'].mean()
    if avg_ph < 6.0:
        recommendations.append("🧪 **pH tanah asam.** Pertimbangkan penambahan kapur pertanian.")
    elif avg_ph > 7.0:
        recommendations.append("🧪 **pH tanah basa.** Pertimbangkan penambahan bahan organik atau sulfur.")
    else:
        recommendations.append("🧪 pH tanah optimal.")

    avg_nitrogen = df_soil['nitrogen'].mean()
    if avg_nitrogen < 60:
        recommendations.append("🌿 **Kadar Nitrogen rendah.** Berikan pupuk kaya Nitrogen (Urea/ZA).")
    else:
        recommendations.append("🌿 Kadar Nitrogen cukup.")

    avg_phosphorus = df_soil['phosphorus'].mean()
    if avg_phosphorus < 15:
        recommendations.append("🌿 **Kadar Fosfor rendah.** Berikan pupuk kaya Fosfor (SP-36/TSP).")
    else:
        recommendations.append("🌿 Kadar Fosfor cukup.")

    avg_potassium = df_soil['potassium'].mean()
    if avg_potassium < 100:
        recommendations.append("🌿 **Kadar Kalium rendah.** Berikan pupuk kaya Kalium (KCL/NPK).")
    else:
        recommendations.append("🌿 Kadar Kalium cukup.")

    avg_light = df_soil['light_intensity'].mean()
    if avg_light < 10000:
        recommendations.append("💡 **Intensitas cahaya rendah.** Pertimbangkan pencahayaan tambahan jika tanaman membutuhkan banyak cahaya.")
    elif avg_light > 40000:
        recommendations.append("💡 **Intensitas cahaya tinggi.** Pertimbangkan shading untuk tanaman sensitif.")
    else:
        recommendations.append("💡 Intensitas cahaya optimal.")

    return "\n".join(recommendations)

def create_soil_figs(df_soil):
    """Creates Plotly figures for soil data."""
    figs = {}
    figs['soil_moisture_line'] = px.line(df_soil, x='timestamp', y='soil_moisture',
                                         title='💧 Kelembaban Tanah per Jam',
                                         labels={'soil_moisture': 'Kelembaban (%)'},
                                         color_discrete_sequence=[PALETTE['blue_sky']])
    figs['soil_temperature_line'] = px.line(df_soil, x='timestamp', y='soil_temperature',
                                             title='🌡️ Suhu Tanah per Jam',
                                             labels={'soil_temperature': 'Suhu (°C)'},
                                             color_discrete_sequence=[PALETTE['orange_sunset']])
    figs['soil_ph_line'] = px.line(df_soil, x='timestamp', y='soil_ph',
                                   title='🧪 pH Tanah per Jam',
                                   labels={'soil_ph': 'pH'},
                                   color_discrete_sequence=[PALETTE['green_soft']])
    figs['soil_npk_bar'] = go.Figure()
    figs['soil_npk_bar'].add_trace(go.Bar(x=df_soil['timestamp'], y=df_soil['nitrogen'], name='Nitrogen (ppm)', marker_color='#4CAF50'))
    figs['soil_npk_bar'].add_trace(go.Bar(x=df_soil['timestamp'], y=df_soil['phosphorus'], name='Fosfor (ppm)', marker_color='#FFC107'))
    figs['soil_npk_bar'].add_trace(go.Bar(x=df_soil['timestamp'], y=df_soil['potassium'], name='Kalium (ppm)', marker_color='#2196F3'))
    figs['soil_npk_bar'].update_layout(barmode='group', title='🌱 Kadar NPK Tanah per Jam')
    figs['light_intensity_area'] = px.area(df_soil, x='timestamp', y='light_intensity',
                                             title='💡 Intensitas Cahaya per Jam',
                                             labels={'light_intensity': 'Intensitas (Lux)'},
                                             color_discrete_sequence=[PALETTE['yellow_sun']])
    figs['soil_scatter_temp_ph'] = px.scatter(df_soil, x='soil_temperature', y='soil_ph',
                                             color='soil_moisture', size='light_intensity',
                                             title='🌡️🧪 Hubungan Suhu Tanah & pH (Ukuran=Cahaya, Warna=Kelembaban)',
                                             color_continuous_scale='Viridis')
    return figs

# =========================
# VISUALISASI & PEMBUATAN FILE
# =========================
def create_all_figs(df):
    figs = {}
    figs['1_temperatur_per_jam'] = px.line(df, x='time', y='temperature', title='🌡 Temperatur per Jam', labels={'temperature': 'Suhu (°C)'}, color_discrete_sequence=[PALETTE['orange_sunset']]).update_traces(line=dict(width=3))
    figs['2_kelembapan_per_jam'] = px.line(df, x='time', y='humidity', title='💧 Kelembapan per Jam', labels={'humidity': 'Kelembaban (%)'}, color_discrete_sequence=[PALETTE['blue_sky']]).update_traces(line=dict(width=3))
    figs['3_uv_index_per_jam'] = px.line(df, x='time', y='uvIndex', title='🔆 UV Index per Jam', color_discrete_sequence=[PALETTE['yellow_sun']]).update_traces(line=dict(width=3))
    figs['4_distribusi_suhu_per_uv'] = px.violin(df, x='uv_category', y='temperature', box=True, points="all", title='🌞 Distribusi Suhu per Kategori UV', color='uv_category', color_discrete_map={'Rendah':PALETTE['green_soft'], 'Sedang':PALETTE['blue_sky'], 'Tinggi':PALETTE['yellow_sun'], 'Sangat Tinggi':PALETTE['orange_sunset'], 'Ekstrem':PALETTE['red_danger']})
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(x=df['time'], y=df['rainIntensity'], name='Curah Hujan', marker_color=PALETTE['blue_rain']))
    fig_rain.add_trace(go.Scatter(x=df['time'], y=df['precipitationProbability'], name='Probabilitas Presipitasi', mode='lines+markers', line=dict(color=PALETTE['blue_sky'], width=2)))
    fig_rain.update_layout(title='🌧 Curah Hujan & Probabilitas Presipitasi', barmode='overlay')
    figs['5_curah_hujan_probabilitas'] = fig_rain
    figs['6_evapotranspirasi'] = px.area(df, x='time', y='evapotranspiration', title='💧 Evapotranspirasi per Jam', labels={'evapotranspiration': 'ET (mm)'}, color_discrete_sequence=[PALETTE['blue_sky']])
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(x=df['time'], y=df['windSpeed'], mode='lines', name='Kecepatan Angin', line=dict(color=PALETTE['grey_cloud'])))
    fig_wind.add_trace(go.Scatter(x=df['time'], y=df['windGust'], mode='lines', name='Wind Gust', line=dict(color=PALETTE['red_danger'], dash='dash')))
    fig_wind.update_layout(title='🌬 Kecepatan Angin & Gust')
    figs['7_kecepatan_angin'] = fig_wind
    df['stress_roll'] = df['pred_stress'].rolling(4, min_periods=1).mean()
    figs['8_prediksi_stres'] = px.area(df, x='time', y='stress_roll', title='🌿 Prediksi Stres (Rolling Mean)', labels={'stress_roll': 'Skor Stres'}, color_discrete_sequence=[PALETTE['red_danger']])
    figs['9_prediksi_uv_scatter'] = px.scatter(df, x='time', y='pred_uv', size='uvIndex', color='pred_uv', title='🔆 Prediksi UV (Ukuran=UV Index)', color_continuous_scale='YlOrRd')
    figs['10_prediksi_risiko_angin'] = px.scatter(df, x='time', y='pred_wind', size='windSpeed', color='pred_wind', title='🌬 Prediksi Risiko Angin', color_continuous_scale='Purples')
    cols_corr = ['temperature', 'humidity', 'uvIndex', 'windSpeed', 'evapotranspiration', 'rainIntensity', 'precipitationProbability']
    corr = df[cols_corr].corr()
    figs['11_korelasi_variabel'] = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='📌 Korelasi Variabel Cuaca')
    figs['12_et_vs_temp'] = px.scatter(df, x='temperature', y='evapotranspiration', color='humidity', size='evapotranspiration', title='💧 Evapotranspirasi vs Temperatur', color_continuous_scale='Viridis')
    dfv = df.copy()
    dfv['ndvi_sim'] = ((dfv['humidity'] / 100) - (dfv['temperature'] / 50)).clip(-0.2, 0.9)
    figs['13_simulasi_ndvi'] = px.line(dfv, x='time', y='ndvi_sim', title='🌿 Simulasi NDVI (sederhana)', color_discrete_sequence=[PALETTE['green_soft']])
    figs['14_distribusi_suhu'] = px.histogram(df, x='temperature', nbins=20, title='🌡 Distribusi Suhu', color_discrete_sequence=[PALETTE['orange_sunset']])
    figs['15_boxplot_kelembapan'] = px.box(df, y='humidity', points='all', title='💧 Boxplot Kelembapan', color_discrete_sequence=[PALETTE['blue_sky']])
    kpi_df = pd.DataFrame({
        'metric': ['Suhu (°C) Rata-rata', 'Kelembapan (%) Rata-rata', 'Jam UV > 7', 'Total Curah Hujan (mm)'],
        'value': [df['temperature'].mean(), df['humidity'].mean(), (df['uvIndex'] > 7).sum(), df['rainIntensity'].sum()]
    })
    figs['16_kpi_ringkas'] = px.bar(kpi_df, x='metric', y='value', title='📊 KPI Ringkas', color='metric', color_discrete_map={
        'Suhu (°C) Rata-rata': PALETTE['orange_sunset'],
        'Kelembapan (%) Rata-rata': PALETTE['blue_sky'],
        'Jam UV > 7': PALETTE['yellow_sun'],
        'Total Curah Hujan (mm)': PALETTE['blue_rain']
    })
    return figs

def generate_report_zip(figs_dict):
    """Membuat file ZIP berisi laporan HTML interaktif dari semua visualisasi."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for name, fig in figs_dict.items():
            try:
                html_content = pio.to_html(fig, full_html=False, include_plotlyjs='cdn').encode('utf-8')
                zip_file.writestr(f"{name.replace('_', ' ').title()}.html", html_content)
            except Exception as e:
                print(f"Gagal memproses visualisasi {name}: {e}")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def get_summary_text(df, location_name):
    """Membuat teks ringkasan untuk pesan Telegram."""
    now_wib = datetime.now(timezone("Asia/Jakarta"))
    summary = (
        f"🌱 **Smart Farming Update {location_name}**\n"
        f"🗓️ {now_wib.strftime('%d-%m-%Y %H:%M WIB')}\n\n"
        f"**Ringkasan Cuaca 24 Jam ke Depan:**\n"
        f"- Suhu rata-rata: {df['temperature'].mean():.2f} °C\n"
        f"- Kelembapan rata-rata: {df['humidity'].mean():.2f} %\n"
        f"- Total curah hujan: {df['rainIntensity'].sum():.2f} mm\n\n"
        f"**Prediksi AI dan Rekomendasi Cuaca:**\n"
        f"- {get_conclusion(df, 'stress')}\n"
        f"- {get_conclusion(df, 'uv')}\n"
        f"- {get_conclusion(df, 'rain')}\n"
        f"- {get_conclusion(df, 'wind')}\n"
        f"- {get_conclusion(df, 'evapo')}\n"
        f"\nLaporan lengkap berupa visualisasi HTML interaktif tersedia dalam file ZIP terlampir."
    )
    return summary

def send_telegram_file(bot, chat_id, file_content, file_name, file_type, caption=None):
    """Mengirim file ke Telegram."""
    try:
        buf = io.BytesIO(file_content)
        if file_type == "html_zip":
            bot.send_document(chat_id=chat_id, document=buf, filename=file_name, caption=caption)
        elif file_type == "message":
            bot.send_message(chat_id=chat_id, text=caption, parse_mode='Markdown')
        return True
    except Exception as e:
        st.error(f"Gagal mengirim laporan: {e}")
        return False

def polling_loop(token, chat_override=None, lat=None, lon=None, location_name=None, mode_ai=None):
    """
    Polling loop to listen for Telegram messages.
    Responds to the "laporan" command with a full report.
    """
    bot = Bot(token=token)
    offset = None
    st.session_state.polling_running = True

    st.sidebar.success("Bot polling aktif. Bot akan merespons perintah 'laporan' di Telegram.")

    while st.session_state.polling_running:
        try:
            updates = bot.get_updates(offset=offset, timeout=10)
            for update in updates:
                offset = update.update_id + 1
                chat_id = update.message.chat.id
                text = update.message.text

                if text and text.strip().lower() == 'laporan':
                    bot.send_message(chat_id=chat_id, text="Membuat laporan interaktif... Mohon tunggu sebentar.")

                    df_processed_polled = add_engineered_columns(fetch_forecast(lat, lon), mode_ai)
                    figs_dict_polled = create_all_figs(df_processed_polled)

                    summary_text = get_summary_text(df_processed_polled, location_name)
                    zip_bytes = generate_report_zip(figs_dict_polled)

                    bot.send_message(chat_id=chat_id, text=summary_text, parse_mode='Markdown')
                    bot.send_document(chat_id=chat_id, document=io.BytesIO(zip_bytes), filename="smart_farming_report.zip")

                    st.session_state.last_report_sent = datetime.now()
        except Exception as e:
            time.sleep(2)

# =========================
# TATA LETAK APLIKASI STREAMLIT
# =========================
st_autorefresh(interval=60*60*1000, key="auto_refresh")

st.sidebar.header("📍 Pilih Lokasi")
location_name = st.sidebar.selectbox("Pilih Desa", list(LOCATIONS.keys()))
current_location = LOCATIONS[location_name]
LAT, LON = current_location['lat'], current_location['lon']

st.sidebar.markdown("---")
st.sidebar.header("🧠 Pengaturan AI")
mode_ai_selection = st.sidebar.radio(
    "Pilih Mode AI:",
    ["Menggunakan Model AI", "Heuristik Sederhana"],
    help="Pilih 'Menggunakan Model AI' jika file model .pkl tersedia di folder yang sama."
)
mode_ai = "model" if mode_ai_selection == "Menggunakan Model AI" else "heuristik"

if 'data_frame' not in st.session_state or st.session_state.get('current_location') != location_name or st.session_state.get('mode_ai') != mode_ai:
    raw_data = fetch_forecast(LAT, LON)
    st.session_state.data_frame = add_engineered_columns(raw_data, mode_ai)
    st.session_state.visuals = create_all_figs(st.session_state.data_frame)
    st.session_state.soil_data_loaded = False
    st.session_state.current_location = location_name
    st.session_state.mode_ai = mode_ai

df_processed = st.session_state.data_frame
figs_dict = st.session_state.visuals

st.title(f"🌱 Smart Farming Dashboard — {location_name}")
st.caption("📡 Data Cuaca Real-time & Prediksi AI · Diperbarui otomatis per jam · Laporan Telegram")

st.sidebar.header("⚙️ Kontrol Aplikasi")
st.sidebar.markdown("### 🚜 Kontrol Pertanian Manual")
col1_sb, col2_sb = st.sidebar.columns(2)
if col1_sb.button("💧 Irigasi ON/OFF", key="irigasi_manual_button"):
    st.session_state.irrigation_manual = not st.session_state.get('irrigation_manual', False)
    st.sidebar.success(f"Irigasi Manual: {'AKTIF' if st.session_state.irrigation_manual else 'NONAKTIF'}")
if col2_sb.button("☀️ UV Light ON/OFF", key="uv_light_manual_button"):
    st.session_state.uv_light_manual = not st.session_state.get('uv_light_manual', False)
    st.sidebar.success(f"UV Light: {'AKTIF' if st.session_state.uv_light_manual else 'NONAKTIF'}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📧 Pengaturan Telegram")
telegram_token = st.sidebar.text_input("🔑 Token Bot", value=DEFAULT_TELEGRAM_TOKEN, type="password", key="telegram_token_input")
telegram_chat = st.sidebar.text_input("👥 Chat ID Penerima", value=DEFAULT_TELEGRAM_CHAT_ID, key="telegram_chat_input")

if st.sidebar.button("📤 Kirim Laporan via Telegram", key="send_report_telegram_button"):
    if not telegram_token or not telegram_chat:
        st.sidebar.error("Token atau Chat ID Telegram kosong. Mohon diisi.")
    else:
        try:
            bot = Bot(token=telegram_token)
            summary_text = get_summary_text(df_processed, location_name)
            zip_bytes = generate_report_zip(figs_dict)

            st.sidebar.info("Mengirim pesan dan file ke Telegram...")
            send_telegram_file(bot, telegram_chat, None, None, "message", summary_text)
            send_telegram_file(bot, telegram_chat, zip_bytes, "smart_farming_report.zip", "html_zip")
            st.sidebar.success("Laporan berhasil dikirim ke Telegram! Cek file ZIP berisi HTML interaktif.")
        except Exception as e:
            st.sidebar.error(f"Gagal mengirim laporan: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("▶️ Aktifkan Polling Bot Telegram"):
    if 'polling_thread' not in st.session_state or not st.session_state.get('polling_thread_alive', False):
        t = threading.Thread(target=polling_loop, args=(telegram_token, telegram_chat, LAT, LON, location_name, mode_ai), daemon=True)
        t.start()
        st.session_state.polling_thread = t
        st.session_state.polling_thread_alive = True
    else:
        st.sidebar.info("Bot polling sudah berjalan.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⬇️ Download Laporan & Data")
csv = df_processed.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("💾 Download Data Cuaca (CSV)", csv, "forecast_data.csv", key="download_csv_button")
zip_bytes_html = generate_report_zip(figs_dict)
st.sidebar.download_button("📦 Download Laporan Lengkap (ZIP)", zip_bytes_html, "smart_farming_report.zip", mime="application/zip", key="download_zip_button")
st.sidebar.write("Terakhir diupdate (WIB):", df_processed['time'].iloc[0].strftime("%d-%m-%Y %H:%M"))

tab_overview, tab_weather, tab_stress, tab_land, tab_ai, tab_soil_data = st.tabs([
    "📊 Ringkasan Utama", "☁️ Analisis Cuaca", "🌿 Stres Tanaman", "💧 Manajemen Lahan", "🧠 Prediksi AI", "🌱 Data Tanah"
])

with tab_overview:
    st.header(f"Ringkasan Utama 24 Jam ke Depan untuk {location_name}")
    st.markdown("---")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Stres Tanaman (jam)", int(df_processed['pred_stress'].sum()), help="Prediksi jam-jam di mana tanaman berpotensi mengalami stres.")
    kpi_cols[1].metric("Risiko Angin (jam)", int(df_processed['pred_wind'].sum()), help="Prediksi jam-jam di mana angin kencang berisiko merusak tanaman.")
    kpi_cols[2].metric("Jam UV>7", int((df_processed['pred_uv'] > 0).sum()), help="Total jam dengan indeks UV di atas level berbahaya (7).")
    kpi_cols[3].metric("Total Curah Hujan", f"{df_processed['rainIntensity'].sum():.2f} mm", help="Total curah hujan kumulatif yang diprediksi.")
    st.plotly_chart(figs_dict['16_kpi_ringkas'], use_container_width=True)
    st.subheader("Kesimpulan Cepat")
    st.info(get_summary_text(df_processed, location_name).split("Prediksi AI dan Rekomendasi Cuaca:")[0])

with tab_weather:
    st.header("☁️ Analisis Cuaca Mendalam")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs_dict['1_temperatur_per_jam'], use_container_width=True)
        st.info("🌡 **Analisis:** Grafik ini menunjukkan fluktuasi suhu. Suhu ekstrem (terlalu tinggi atau rendah) dapat memengaruhi pertumbuhan dan kesehatan tanaman.")
    with col2:
        st.plotly_chart(figs_dict['2_kelembapan_per_jam'], use_container_width=True)
        st.info("💧 **Analisis:** Kelembaban penting untuk transpirasi dan fotosintesis. Kelembaban rendah bisa meningkatkan kebutuhan air, sementara kelembaban tinggi berisiko memicu penyakit jamur.")
    st.plotly_chart(figs_dict['7_kecepatan_angin'], use_container_width=True)
    st.info(f"🌬️ **Analisis:** Grafik ini membedakan kecepatan angin normal dan hembusan angin (*gust*). Hembusan angin lebih berbahaya dan dapat merusak struktur tanaman. {get_conclusion(df_processed, 'wind')}")
    st.plotly_chart(figs_dict['3_uv_index_per_jam'], use_container_width=True)
    st.info(f"🔆 **Analisis:** Indeks UV tinggi dapat menyebabkan kerusakan sel pada tanaman. Grafik ini membantu Anda mengidentifikasi jam-jam dengan indeks UV berbahaya untuk merencanakan tindakan perlindungan, seperti penggunaan naungan.")
    st.plotly_chart(figs_dict['5_curah_hujan_probabilitas'], use_container_width=True)
    st.info(f"🌧️ **Analisis:** Grafik ini menampilkan intensitas curah hujan dan probabilitasnya. Informasi ini sangat penting untuk manajemen irigasi dan pencegahan banjir di lahan pertanian. {get_conclusion(df_processed, 'rain')}")
    st.plotly_chart(figs_dict['6_evapotranspirasi'], use_container_width=True)
    st.info(f"💧 **Analisis:** Evapotranspirasi adalah gabungan dari penguapan air dari tanah dan transpirasi dari tanaman. Angka yang tinggi menunjukkan kebutuhan air yang lebih besar. {get_conclusion(df_processed, 'evapo')}")
    st.plotly_chart(figs_dict['12_et_vs_temp'], use_container_width=True)
    st.info("🔄 **Analisis:** Grafik ini menunjukkan hubungan antara temperatur dan evapotranspirasi. Umumnya, semakin tinggi suhu, semakin tinggi laju evapotranspirasi.")

with tab_stress:
    st.header("🌿 Prediksi Stres Tanaman")
    st.markdown("---")
    st.plotly_chart(figs_dict['8_prediksi_stres'], use_container_width=True)
    st.info(f"🌱 **Analisis:** Grafik ini memprediksi tingkat stres tanaman berdasarkan gabungan faktor cuaca. Nilai yang lebih tinggi menunjukkan risiko stres yang lebih besar. {get_conclusion(df_processed, 'stress')}")
    col_stress1, col_stress2 = st.columns(2)
    with col_stress1:
        st.plotly_chart(figs_dict['9_prediksi_uv_scatter'], use_container_width=True)
        st.info("🔆 **Analisis:** Titik yang lebih besar menunjukkan indeks UV yang lebih tinggi, yang berpotensi menyebabkan stres pada tanaman.")
    with col_stress2:
        st.plotly_chart(figs_dict['10_prediksi_risiko_angin'], use_container_width=True)
        st.info("🌬️ **Analisis:** Grafik ini menyoroti jam-jam dengan risiko angin kencang yang dapat merusak struktur fisik tanaman.")
    st.markdown("---")
    st.subheader("Simulasi NDVI (Normalized Difference Vegetation Index)")
    st.plotly_chart(figs_dict['13_simulasi_ndvi'], use_container_width=True)
    st.info("🌿 **Analisis:** NDVI adalah indikator kesehatan tanaman. Nilai NDVI antara -1 hingga 1. Semakin mendekati 1, semakin sehat tanaman. Grafik ini adalah simulasi sederhana dari NDVI berdasarkan data cuaca.")

with tab_land:
    st.header("💧 Manajemen Lahan & Irigasi")
    st.markdown("---")
    st.plotly_chart(figs_dict['5_curah_hujan_probabilitas'], use_container_width=True)
    st.plotly_chart(figs_dict['6_evapotranspirasi'], use_container_width=True)
    st.subheader("Rekomendasi Irigasi")
    # Jika ada curah hujan tinggi, rekomendasi adalah tidak menyiram
    if df_processed['pred_rain'].sum() > 2:
        st.success("✅ **Irigasi Tidak Diperlukan:** Prediksi curah hujan tinggi. Matikan sistem irigasi untuk mencegah genangan.")
    # Jika evapotranspirasi tinggi, merekomendasikan untuk menyiram
    elif df_processed['pred_evapo'].mean() > 0.4:
        st.warning("💧 **Irigasi Diperlukan:** Laju evapotranspirasi tinggi. Pertimbangkan irigasi tambahan untuk mengganti air yang hilang.")
    # Jika kondisi normal
    else:
        st.info("🌱 **Irigasi Normal:** Kondisi cuaca stabil, ikuti jadwal irigasi rutin.")
    
    # Menampilkan kontrol manual (dari sidebar) di dalam tab
    col_irr_status, col_uv_status = st.columns(2)
    col_irr_status.markdown(f"**Status Irigasi:** {'💧 AKTIF' if st.session_state.get('irrigation_manual', False) else '💧 NONAKTIF'}")
    col_uv_status.markdown(f"**Status UV Light:** {'☀️ AKTIF' if st.session_state.get('uv_light_manual', False) else '☀️ NONAKTIF'}")
    
with tab_ai:
    st.header("🧠 Detail Prediksi AI & Korelasi")
    st.markdown("---")
    st.plotly_chart(figs_dict['11_korelasi_variabel'], use_container_width=True)
    st.info("📌 **Analisis:** Heatmap ini menunjukkan hubungan (korelasi) antara berbagai variabel cuaca. Nilai mendekati 1 (biru tua) menunjukkan korelasi positif kuat, sedangkan -1 (merah tua) menunjukkan korelasi negatif kuat.")
    st.markdown("---")
    st.subheader("Perbandingan Distribusi Cuaca")
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        st.plotly_chart(figs_dict['14_distribusi_suhu'], use_container_width=True)
    with col_ai2:
        st.plotly_chart(figs_dict['15_boxplot_kelembapan'], use_container_width=True)
    
with tab_soil_data:
    st.header("🌱 Data Sensor Tanah")
    st.markdown("---")
    st.warning("⚠️ **Peringatan:** Data di tab ini adalah data simulasi (mock). Diimplementasi riil, data ini akan berasal dari sensor Arduino.")
    if st.button("🔄 Perbarui Data Sensor Tanah Simulasi", key="refresh_soil_data"):
        st.session_state.soil_data_loaded = False
    
    if not st.session_state.get('soil_data_loaded', False):
        df_soil_data = generate_mock_arduino_data()
        st.session_state.soil_data = df_soil_data
        st.session_state.soil_figs = create_soil_figs(df_soil_data)
        st.session_state.soil_data_loaded = True
        
    soil_data = st.session_state.soil_data
    soil_figs = st.session_state.soil_figs

    st.markdown("### Ringkasan Kondisi Tanah")
    st.info(get_soil_recommendations(soil_data))
    
    col_soil1, col_soil2 = st.columns(2)
    with col_soil1:
        st.plotly_chart(soil_figs['soil_moisture_line'], use_container_width=True)
    with col_soil2:
        st.plotly_chart(soil_figs['soil_temperature_line'], use_container_width=True)
    
    col_soil3, col_soil4 = st.columns(2)
    with col_soil3:
        st.plotly_chart(soil_figs['soil_ph_line'], use_container_width=True)
    with col_soil4:
        st.plotly_chart(soil_figs['soil_npk_bar'], use_container_width=True)
    
    st.plotly_chart(soil_figs['light_intensity_area'], use_container_width=True)
    st.plotly_chart(soil_figs['soil_scatter_temp_ph'], use_container_width=True)
