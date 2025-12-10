# dashboard/streamlit_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib # <--- TAMBAHKAN INI
from tensorflow.keras.models import load_model
from src.utils import scale_data, inverse_transform_prediction
from src.predictor_agent import decision_logic # Import logika keputusan

# --- KONFIGURASI DAN KONSTANTA ---
st.set_page_config(page_title="Smart Pre-Cooling UPI FPMIPA C", layout="wide")

DATA_FILE = os.path.join('data', 'simulated_data.csv')
MODEL_FILE = os.path.join('models', 'lstm_model.keras')
SCALER_FILE = os.path.join('models', 'scaler.joblib') # <--- DEFINISI SCALER FILE
N_STEPS_IN = 48
N_STEPS_OUT = 12 
TARGET_COMFORT_TIME = '07:00:00' 
COMFORT_THRESHOLD = 25.0 
PRE_COOLING_WINDOW = 60

# --- FUNGSI MEMUAT ASSET ---
@st.cache_resource
def load_assets():
    """Memuat model, scaler, dan data hanya sekali."""
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE) # <--- MEMUAT SCALER
    except Exception:
        st.error(f"Gagal memuat Model atau Scaler. Pastikan 02_model_trainer.py sudah dijalankan.")
        return None, None, None

    try:
        df = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan: {DATA_FILE}. Pastikan 01_data_generator.py sudah dijalankan.")
        return None, None, None

    return model, df, scaler # <--- KEMBALIKAN SCALER

# --- FUNGSI PREDIKSI DAN KEPUTUSAN DASHBOARD ---
def run_prediction_and_decision_for_ui(model, df, scaler, current_time_index):
    """Menjalankan prediksi pada waktu yang disimulasikan dan menghasilkan output UI."""
    
    end_index = df.index.get_loc(current_time_index) + 1
    start_index = end_index - N_STEPS_IN
    
    if start_index < 0:
        return "Not enough data for prediction", None, None, None

    latest_data = df[['Tin', 'Tex']].iloc[start_index:end_index]
    latest_timestamp = latest_data.index[-1]
    
    # Scaling, Reshape, dan Prediksi
    scaled_latest_data = scaler.transform(latest_data)
    X_input = scaled_latest_data.reshape((1, N_STEPS_IN, 2))
    scaled_prediction = model.predict(X_input, verbose=0)
    predicted_temp_original = inverse_transform_prediction(scaled_prediction, scaler)[0]
    
    # Logika Keputusan
    action, predicted_target_temp = decision_logic(
        latest_timestamp, 
        predicted_temp_original, 
        COMFORT_THRESHOLD
    )

    pred_timestamps = pd.date_range(
        start=latest_timestamp + pd.Timedelta(minutes=5), 
        periods=N_STEPS_OUT, 
        freq='5min'
    )
    
    return action, predicted_target_temp, latest_data, pd.Series(predicted_temp_original, index=pred_timestamps)

# --- TAMPILAN UTAMA STREAMLIT ---
def main():
    st.title("💡 Intelligent Pre-Cooling Agent - FPMIPA C UPI")
    st.markdown("Dashboard Monitoring dan Rekomendasi Aksi AC Gedung")
    
    model, df, scaler = load_assets()

    if model is None or df is None:
        return

    # Sidebar Kontrol Waktu Simulasi
    st.sidebar.header("Kontrol Waktu Observasi")
    
    # Filter waktu pagi hari untuk simulasi
    sim_data_subset = df.between_time('05:00', '08:00')
    
    latest_time_sim = st.sidebar.select_slider(
        "Pilih Waktu Observasi (Simulasi Data Terbaru)",
        options=sim_data_subset.index.tolist(),
        value=sim_data_subset.index[10] # Default di sekitar 06:00
    )
    
    # Jalankan Agen
    action, predicted_target_temp, historical_data, prediction_series = \
        run_prediction_and_decision_for_ui(model, df, scaler, latest_time_sim)

    # --- Kolom Tampilan ---
    col1, col2 = st.columns([1, 2.5])

    # KOTAK STATUS DAN AKSI
    with col1:
        st.header("Status dan Aksi Rekomendasi")
        
        st.metric(label="Waktu Observasi Saat Ini", 
                  value=latest_time_sim.strftime("%Y-%m-%d %H:%M:%S"))
        
        st.metric(label="Suhu Internal Terbaru (Tin)", 
                  value=f"{historical_data['Tin'].iloc[-1]:.2f} °C")
        
        st.subheader("REKOMENDASI AGEN")
        
        if action == "PRECOOL_ON":
            st.success("🟢 NYALAKAN AC SEKARANG (PRECOOL_ON)")
            st.markdown(f"**Alasan:** Diprediksi suhu pukul {TARGET_COMFORT_TIME} akan mencapai **{predicted_target_temp:.2f}°C** (di atas {COMFORT_THRESHOLD}°C).")
        elif action == "STANDBY":
            st.info("🔵 AC TETAP STANDBY (STANDBY)")
            st.markdown(f"**Alasan:** Diprediksi suhu pukul {TARGET_COMFORT_TIME} akan mencapai **{predicted_target_temp:.2f}°C** (nyaman).")
        else:
            st.warning(f"🟠 STATUS: {action.replace('_', ' ')}")
            st.markdown(f"**Tindakan:** Tunggu hingga masuk jendela waktu pra-pendinginan.")
            
        st.metric(label=f"Suhu Diprediksi pada {TARGET_COMFORT_TIME}", 
                  value=f"{predicted_target_temp:.2f} °C", 
                  delta=f"Threshold: {COMFORT_THRESHOLD} °C")
        
    # VISUALISASI TIME SERIES
    with col2:
        st.header("Grafik Histori dan Peramalan Suhu")
        
        if prediction_series is not None:
            # Gabungkan data untuk plotting
            plot_df = pd.DataFrame({
                'Tin Histori': historical_data['Tin'],
                'Tex Historis': historical_data['Tex'],
            })
            
            # Tambahkan prediksi pada kolom baru
            plot_df['Tin Prediksi'] = prediction_series
            
            # Garis Threshold
            threshold_df = pd.DataFrame({'Threshold (25°C)': COMFORT_THRESHOLD}, index=plot_df.index)
            final_chart_df = pd.concat([plot_df, threshold_df], axis=1)
            
            st.line_chart(final_chart_df[['Tin Histori', 'Tex Historis', 'Tin Prediksi', 'Threshold (25°C)']])
            
            st.markdown(f"""
            *Grafik menunjukkan **4 jam histori** dan **60 menit prediksi** Tin. Waktu Target Kenyamanan (akhir prediksi) adalah **{prediction_series.index[-1].strftime('%H:%M')}**.*
            """)


if __name__ == '__main__':
    main()
