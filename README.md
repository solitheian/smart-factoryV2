# 🏭 Smart Factory Command Center - Predictive Maintenance AI

Sistem **Predictive Maintenance** interaktif berbasis Machine Learning untuk mendeteksi dini risiko kerusakan komponen manufaktur industri berdasarkan data sensor termal dan mekanikal secara *real-time*.

### 🌐 Live Deployment
Aplikasi ini telah di-deploy dan dapat diuji secara interaktif tanpa perlu instalasi lokal.
**👉 Akses Dashboard AI di sini: [Smart Factory Web App](https://smart-factoryv2.streamlit.app/)**

---

## 👥 Tim Pengembang
Proyek ini disusun oleh Mahasiswa Program Studi Teknik Elektro, **Telkom University**:
1. **Reynaldhi Ananda Rahardian** (1102223226)
2. **Fikri Naufal Hakim** (1102220107)
3. **Rendi Febrian** (1102223112)
4. **Muhammad Ghoisan H.A.** (1102223228)
5. **I Wayan Rivan** (1102223168)
6. **Ahmad Diva Sihombing** (1102223050)

---

## 🚀 Arsitektur & Algoritma Sistem
Sistem ini menggunakan dataset industri yang berisi 5 parameter kritis (Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear) dan membandingkan dua pendekatan algoritma:
* **Random Forest Classifier (Model Utama):** Telah melalui proses *Hyperparameter Tuning* (GridSearchCV) dan *5-Fold Cross-Validation*. Memiliki akurasi 98.3% dan metrik *Recall* yang sangat stabil untuk meminimalisir *False Negative* pada data tabular industri.
* **Long Short-Term Memory (LSTM):** Digunakan sebagai algoritma pembanding dari arsitektur *Deep Learning* (*Sequential Memory*).

---

## 💻 Panduan Instalasi & Eksekusi Lokal (Local Development)

Jika Anda ingin menjalankan sistem *inference* ini secara lokal di komputer Anda, ikuti langkah-langkah berikut:

**1. Clone Repository**
Buka terminal/command prompt dan jalankan perintah berikut untuk mengunduh *source code*:
```bash
git clone [https://github.com/solitheian/smart-factoryV2.git](https://github.com/solitheian/smart-factoryV2.git)
cd smart-factoryV2

2. Install Dependencies
Pastikan Python (versi 3.10 atau 3.11 disarankan) sudah terinstal. Jalankan perintah ini untuk menginstal seluruh library yang dibutuhkan:

Bash
pip install -r requirements.txt
3. Jalankan Aplikasi Web (Streamlit)
Eksekusi perintah berikut untuk menyalakan server lokal:

Bash
streamlit run app.py
Aplikasi akan otomatis terbuka di browser pada alamat http://localhost:8501.

Dibuat untuk memenuhi Tugas Besar Machine Learning - 2026
