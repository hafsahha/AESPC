# AESPC: Automated Energy Saving Pre-Cooling Classroom

## Proyek
Sistem Cerdas Prediktif untuk mengoptimalkan konsumsi energi AC di Gedung FPMIPA C, UPI, dengan melakukan pra-pendinginan (pre-cooling) secara presisi. Sistem menggunakan model Long Short-Term Memory (LSTM) untuk meramalkan Suhu Internal (Tin) 60 menit ke depan (Waktu Target Kenyamanan 07:00 WIB).

## Komponen Utama
1.  **Data Generation (`01_data_generator.py`):** Membuat data simulasi Tin dan Tex.
2.  **Model Training (`02_model_trainer.py`):** Melatih model LSTM multivariate.
3.  **Predictor Agent (`03_predictor_agent.py`):** Mengambil data terbaru, memprediksi, dan menerapkan Logika Keputusan.
4.  **Dashboard (`streamlit_app.py`):** Antarmuka visual untuk rekomendasi aksi (ON/STANDBY) bagi penjaga gedung.

## Langkah Menjalankan Proyek

### 1. Setup Environment
Install semua dependensi (termasuk joblib untuk persistence scaler):
```bash
pip install -r requirements.txt
```

### 2. Generate Data Simulasi
Buat data simulasi suhu:
```bash
python src/01_data_generator.py
```


### 3. Latih Model & Simpan Scaler
Latih model LSTM dan simpan scaler (wajib agar prediksi konsisten):
```bash
python -m src.02_model_trainer
```
File model (`models/lstm_model.keras`) dan scaler (`models/scaler.joblib`) akan otomatis dibuat.

### 4. Jalankan Dashboard Visualisasi & Agen
Jalankan dashboard untuk monitoring dan rekomendasi aksi:
```bash
streamlit run dashboard/streamlit_app.py
```


### 5. (Opsional) Jalankan Agen Prediksi via CLI
Untuk menjalankan agen prediksi dan logika keputusan dari terminal:
```bash
python -m src.predictor_agent
```

---

**Catatan Penting:**
- Pastikan urutan eksekusi: data → training → dashboard/agen.
- File scaler (`models/scaler.joblib`) wajib ada agar prediksi real-time konsisten dengan training.
- Jika ada error asset tidak ditemukan, ulangi langkah training.
- Jika muncul error `ModuleNotFoundError: No module named 'src'`, jalankan perintah dengan `python -m src.02_model_trainer` atau `python -m src.predictor_agent` dari root folder proyek.
