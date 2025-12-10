# src/02_model_trainer.py

import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import create_sequences, scale_data, inverse_transform_prediction

# --- Konfigurasi ---
DATA_FILE = os.path.join('data', 'simulated_data.csv')
MODEL_FILE = os.path.join('models', 'lstm_model.keras')
N_STEPS_IN = 48  # 4 jam historis
N_STEPS_OUT = 12 # 60 menit prediksi
TRAIN_RATIO = 0.8 
VAL_RATIO = 0.1
EPOCHS = 50
BATCH_SIZE = 32

def train_model():
    """Memuat data, melatih model LSTM, dan menyimpan model."""
    try:
        df = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ File data tidak ditemukan. Jalankan 01_data_generator.py terlebih dahulu.")
        return

    print("Memuat dan Preprocessing Data...")
    
    # 1. Scaling Data
    scaler, scaled_data = scale_data(df)
    
    # 2. Membuat Sequences
    X, y = create_sequences(scaled_data, N_STEPS_IN, N_STEPS_OUT)
    
    # 3. Chronological Split Data
    n_train = int(TRAIN_RATIO * len(X))
    n_val = int(VAL_RATIO * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"Ukuran Data Latih: {X_train.shape}, Data Validasi: {X_val.shape}, Data Uji: {X_test.shape}")
    
    # 4. Membangun Model LSTM
    n_features = scaled_data.shape[1]
    
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(N_STEPS_IN, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(N_STEPS_OUT)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    print("✅ Model LSTM berhasil dikompilasi.")

    # 5. Pelatihan Model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Memulai Pelatihan Model...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Evaluasi dan Penyimpanan
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nLoss (MSE) pada data uji: {loss:.4f}")
    
    # Prediksi dan inverse transform untuk metrik
    y_pred_scaled = model.predict(X_test)
    y_test_original = inverse_transform_prediction(y_test, scaler).flatten()
    y_pred_original = inverse_transform_prediction(y_pred_scaled, scaler).flatten()
    
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    print(f"MAE (Mean Absolute Error) pada data uji: {mae:.2f} °C")
    print(f"RMSE (Root Mean Squared Error) pada data uji: {rmse:.2f} °C")
    
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_FILE)
    print(f"\n✅ Model LSTM berhasil dilatih dan disimpan di {MODEL_FILE}")

if __name__ == '__main__':
    train_model()
