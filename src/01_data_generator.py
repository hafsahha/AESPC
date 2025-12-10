# src/01_data_generator.py

import pandas as pd
import numpy as np
import os

# --- Parameter Simulasi ---
START_DATE = '2025-10-01'
NUM_DAYS = 30
INTERVAL_MINUTES = 5
TOTAL_SAMPLES = NUM_DAYS * (24 * 60 // INTERVAL_MINUTES)
OUTPUT_FILE = os.path.join('data', 'simulated_data.csv')

# Konstanta termal
BASE_TIN = 24.0
BASE_TEX = 28.0
TIN_TEX_CORRELATION = 0.4
NOISE_LEVEL = 0.3

def generate_data():
    """Menghasilkan data time series Tin dan Tex yang disimulasikan."""
    
    timestamps = pd.date_range(start=START_DATE, periods=TOTAL_SAMPLES, freq=f'{INTERVAL_MINUTES}min')
    time_series = np.arange(TOTAL_SAMPLES)
    
    # 1. Pola Suhu Eksternal (Tex) - Siklus harian
    cycle_period = 24 * 60 / INTERVAL_MINUTES
    tex_cycle = np.sin(2 * np.pi * time_series / cycle_period)
    Tex = BASE_TEX + 4.0 * tex_cycle # Amplitudo 4.0
    Tex += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, TOTAL_SAMPLES)
    
    # 2. Pola Suhu Internal (Tin) - Mengikuti Tex dengan inersia
    Tin = np.zeros(TOTAL_SAMPLES)
    Tin[0] = BASE_TIN
    
    for i in range(1, TOTAL_SAMPLES):
        # Model sederhana inersia termal
        thermal_change = TIN_TEX_CORRELATION * (Tex[i] - Tin[i-1])
        Tin[i] = Tin[i-1] + thermal_change
        
    Tin += np.random.uniform(-NOISE_LEVEL/2, NOISE_LEVEL/2, TOTAL_SAMPLES)
    Tin = np.clip(Tin, BASE_TIN - 2, BASE_TEX + 1)
    
    # 3. Simpan
    df = pd.DataFrame({'timestamp': timestamps, 'Tin': Tin, 'Tex': Tex})
    df = df.set_index('timestamp')
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUTPUT_FILE)
    print(f"✅ Data simulasi {TOTAL_SAMPLES} sampel berhasil dibuat dan disimpan di {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_data()
