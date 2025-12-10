# AESPC: Automated Energy Saving Pre-Cooling Classroom

## Proyek
Sistem Cerdas Prediktif untuk mengoptimalkan konsumsi energi AC di Gedung FPMIPA C, UPI, dengan melakukan pra-pendinginan (pre-cooling) secara presisi. Sistem menggunakan model Long Short-Term Memory (LSTM) untuk meramalkan Suhu Internal (Tin) 60 menit ke depan (Waktu Target Kenyamanan 07:00 WIB).

## Komponen Utama
1.  **Data Generation (`01_data_generator.py`):** Membuat data simulasi Tin dan Tex.
2.  **Model Training (`02_model_trainer.py`):** Melatih model LSTM multivariate.
3.  **Predictor Agent (`03_predictor_agent.py`):** Mengambil data terbaru, memprediksi, dan menerapkan Logika Keputusan.
4.  **Dashboard (`streamlit_app.py`):** Antarmuka visual untuk rekomendasi aksi (ON/STANDBY) bagi penjaga gedung.

## Langkah Inisiasi dan Menjalankan Proyek
1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Hasilkan Data Simulasi:**
    ```bash
    python src/01_data_generator.py
    ```
3.  **Latih Model LSTM:**
    ```bash
    python src/02_model_trainer.py
    ```
4.  **Jalankan Dashboard:**
    ```bash
    streamlit run dashboard/streamlit_app.py
    ```
