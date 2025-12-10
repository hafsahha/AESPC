# src/predictor_agent.py

import pandas as pd
import numpy as np
import os
import joblib # <--- TAMBAHKAN INI
from tensorflow.keras.models import load_model
from src.utils import scale_data, inverse_transform_prediction

# --- Konfigurasi ---
DATA_FILE = os.path.join('data', 'simulated_data.csv')
MODEL_FILE = os.path.join('models', 'lstm_model.keras')
SCALER_FILE = os.path.join('models', 'scaler.joblib') # <--- DEFINISI SCALER FILE
N_STEPS_IN = 48
N_STEPS_OUT = 12 
TARGET_COMFORT_TIME = '07:00:00' 
COMFORT_THRESHOLD = 25.0 
PRE_COOLING_WINDOW = 60 # Jendela waktu pre-cooling dalam menit

def decision_logic(latest_time, predicted_temp_60min, target_temp_threshold):
    """
    Mengimplementasikan Logika Keputusan (Flowchart Bab III).
    """
    target_dt = pd.to_datetime(TARGET_COMFORT_TIME).time()
    pre_cool_start_time = (pd.to_datetime(TARGET_COMFORT_TIME) - pd.Timedelta(minutes=PRE_COOLING_WINDOW)).time()
    current_time = latest_time.time()

    is_in_pre_cool_window = (current_time >= pre_cool_start_time) and (current_time < target_dt)
    
    # Ambil prediksi suhu tepat di waktu target (setelah 60 menit)
    predicted_temp_at_target = predicted_temp_60min[-1] 
    is_predicted_hot = predicted_temp_at_target > target_temp_threshold
    
    action = "STANDBY"
    
    if is_in_pre_cool_window and is_predicted_hot:
        action = "PRECOOL_ON"
    elif current_time < pre_cool_start_time:
        action = "NO_ACTION_YET"
    elif current_time >= target_dt:
        action = "NO_ACTION"
        
    return action, predicted_temp_at_target

def run_predictor_agent():
    """Memuat model, mengambil data terbaru, dan membuat keputusan aksi."""
    
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE) # <--- MEMUAT SCALER TERSIMPAN
    except Exception as e:
        print(f"❌ Aset (Model/Scaler) tidak ditemukan atau gagal dimuat. Pastikan 02_model_trainer.py dijalankan. Error: {e}")
        return

    try:
        df = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ File data tidak ditemukan.")
        return

    # Ambil N_STEPS_IN data historis terakhir (simulasi data terbaru)
    latest_data = df[['Tin', 'Tex']].tail(N_STEPS_IN)
    latest_timestamp = latest_data.index[-1]

    if len(latest_data) < N_STEPS_IN:
        print(f"❌ Data historis tidak cukup.")
        return

    # 1. Scaling Data (Gunakan scaler yang dimuat)
    scaled_latest_data = scaler.transform(latest_data) # <--- HANYA TRANSFORM
    
    # 2. Reshape dan Prediksi
    X_input = scaled_latest_data.reshape((1, N_STEPS_IN, 2))
    scaled_prediction = model.predict(X_input, verbose=0)
    
    # 3. Inverse Transform
    predicted_temp_original = inverse_transform_prediction(scaled_prediction, scaler)[0]
    
    # 4. Logika Keputusan
    action, predicted_target_temp = decision_logic(
        latest_timestamp, 
        predicted_temp_original, 
        COMFORT_THRESHOLD
    )
    
    print("\n=============================================")
    print(f"Waktu Observasi: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suhu Internal (Tin) Terbaru: {latest_data['Tin'].iloc[-1]:.2f}°C")
    print(f"Suhu Diprediksi pada {TARGET_COMFORT_TIME}: {predicted_target_temp:.2f}°C")
    print(f"REKOMENDASI AKSI: {action}")
    print("=============================================")

if __name__ == '__main__':
    run_predictor_agent()
