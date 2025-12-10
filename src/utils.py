# src/utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def create_sequences(data: np.ndarray, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengubah deret waktu menjadi format sequence-to-sequence (sliding window).
    """
    X, y = list(), list()
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_ix = end_ix + n_steps_out
        
        if out_ix > len(data):
            break
            
        # X: data historis (Tin dan Tex)
        seq_x = data[i:end_ix, :]
        
        # y: Tin target 60 menit ke depan (kolom indeks 0)
        seq_y = data[end_ix:out_ix, 0] 
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

def scale_data(df: pd.DataFrame) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Melakukan normalisasi data menggunakan MinMaxScaler pada kolom 'Tin' dan 'Tex'.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Tin', 'Tex']])
    return scaler, scaled_data

def inverse_transform_prediction(scaled_preds, scaler):
    """Mengembalikan prediksi Tin yang dinormalisasi ke skala suhu asli."""
    n_samples, n_steps = scaled_preds.shape
    # Buat array dummy Tex (kolom 1) karena hanya Tin (kolom 0) yang diprediksi
    dummy_tex = np.zeros((n_samples, n_steps))
    
    # Gabungkan Tin prediksi dan dummy Tex
    dummy_data = np.dstack((scaled_preds, dummy_tex)) 
    
    # Inverse transform dan ambil kembali kolom Tin (indeks 0)
    original_preds = scaler.inverse_transform(dummy_data.reshape(-1, 2))[:, 0]
    return original_preds.reshape(n_samples, n_steps)
