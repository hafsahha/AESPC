# src/01_data_generator.py

import pandas as pd
import numpy as np
import os
import requests


# --- Parameter Simulasi ---
START_DATE = '2025-10-01'
NUM_DAYS = 30
INTERVAL_MINUTES = 5
TOTAL_SAMPLES = NUM_DAYS * (24 * 60 // INTERVAL_MINUTES)
OUTPUT_FILE = os.path.join('data', 'simulated_data.csv')

# --- Parameter API Cuaca (OpenWeatherMap) ---
USE_WEATHER_API = True  # Set ke False jika ingin full simulasi
OWM_API_KEY = os.getenv('OWM_API_KEY', '')  # Atur API key Anda di environment variable
BANDUNG_SETIABUDI_COORD = {'lat': -6.8607, 'lon': 107.6107}
OWM_URL = 'https://api.openweathermap.org/data/2.5/weather'

# Konstanta termal
BASE_TIN = 24.0
BASE_TEX = 28.0
TIN_TEX_CORRELATION = 0.4
NOISE_LEVEL = 0.3


def fetch_tex_from_api():
    """Ambil suhu luar (Tex) real-time dari OpenWeatherMap untuk Bandung Setiabudi."""
    try:
        params = {
            'lat': BANDUNG_SETIABUDI_COORD['lat'],
            'lon': BANDUNG_SETIABUDI_COORD['lon'],
            'appid': OWM_API_KEY,
            'units': 'metric'
        }
        resp = requests.get(OWM_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data['main']['temp']
    except Exception as e:
        print(f"Gagal fetch Tex dari API: {e}. Gunakan simulasi Tex.")
        return None

def generate_data():
    """Menghasilkan data time series Tin dan Tex (simulasi atau real API)."""
    timestamps = pd.date_range(
        start=START_DATE,
        periods=TOTAL_SAMPLES,
        freq=f'{INTERVAL_MINUTES}min',
        tz='Asia/Jakarta'
    )
    time_series = np.arange(TOTAL_SAMPLES)

    if USE_WEATHER_API and OWM_API_KEY:
        print("🔗 Mengambil Tex dari OpenWeatherMap API (Bandung Setiabudi)...")
        # Ambil Tex real-time hanya untuk 1x (snapshot), lalu gunakan untuk seluruh data (atau bisa diimprove fetch per jam)
        tex_api = fetch_tex_from_api()
        if tex_api is not None:
            Tex = np.full(TOTAL_SAMPLES, tex_api)
        else:
            # Fallback ke simulasi jika gagal
            cycle_period = 24 * 60 / INTERVAL_MINUTES
            tex_cycle = np.sin(2 * np.pi * time_series / cycle_period)
            Tex = BASE_TEX + 4.0 * tex_cycle
            Tex += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, TOTAL_SAMPLES)
    else:
        # Simulasi Tex
        cycle_period = 24 * 60 / INTERVAL_MINUTES
        tex_cycle = np.sin(2 * np.pi * time_series / cycle_period)
        Tex = BASE_TEX + 4.0 * tex_cycle
        Tex += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, TOTAL_SAMPLES)

    # 2. Pola Suhu Internal (Tin) - Mengikuti Tex dengan inersia
    Tin = np.zeros(TOTAL_SAMPLES)
    Tin[0] = BASE_TIN
    for i in range(1, TOTAL_SAMPLES):
        thermal_change = TIN_TEX_CORRELATION * (Tex[i] - Tin[i-1])
        Tin[i] = Tin[i-1] + thermal_change
    Tin += np.random.uniform(-NOISE_LEVEL/2, NOISE_LEVEL/2, TOTAL_SAMPLES)
    Tin = np.clip(Tin, BASE_TIN - 2, BASE_TEX + 1)

    # 3. Simpan
    df = pd.DataFrame({'timestamp': timestamps, 'Tin': Tin, 'Tex': Tex})
    df = df.set_index('timestamp')
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUTPUT_FILE)
    print(f"✅ Data {'real' if USE_WEATHER_API and OWM_API_KEY and tex_api is not None else 'simulasi'} {TOTAL_SAMPLES} sampel berhasil dibuat dan disimpan di {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_data()
