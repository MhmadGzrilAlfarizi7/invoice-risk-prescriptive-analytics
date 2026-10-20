# 💼 Invoice Default Risk & Prescriptive Analytics System

> **Implementasi Studi Kasus Kuliah Tamu Praktisi Informatika**  
> **Tema:** *Data Analytics, Artificial Intelligence, dan Pemanfaatan Data dalam Pengambilan Keputusan*  
> **Narasumber:** Andria Arisal  
> **Konsep Kunci:** $\text{Data} \longrightarrow \text{Informasi} \longrightarrow \text{Insight} \longrightarrow \text{Decision}$

---

## 📌 1. Latar Belakang & Rumusan Masalah

Dalam dunia industri nyata, transaksi bisnis antar perusahaan (B2B) mayoritas menggunakan sistem pembayaran bertempo (*credit terms / invoice*). Namun, permasalahan umum yang dihadapi adalah **keterlambatan pembayaran (*late payment*)** atau bahkan **gagal bayar (*default*)**, yang mengakibatkan arus kas (*cash flow*) perusahaan terganggu dan menimbulkan risiko kerugian modal.

Sesuai materi yang dipaparkan oleh narasumber, analitik data tidak cukup berhenti pada pembuatan grafik atau laporan deskriptif, melainkan harus mampu:
1. Memahami kondisi data riil yang kerap memiliki *missing values* atau *noise* (**Data Preparation**).
2. Menjawab akar masalah mengapa tagihan telat bayar (**Diagnostic Analytics**).
3. Memprediksi probabilitas gagal bayar pada invoice yang baru terbit (**Predictive Analytics** menggunakan Machine Learning).
4. Memberikan rekomendasi tindakan operasional bagi manajemen (**Prescriptive Analytics**).

---

## 🏗️ 2. Arsitektur 4 Level Analitik

```
[DATA MENTAH: 1.000 Transaksi Invoice]
                 │
                 ▼
     [DATA PREPARATION / CLEANING]
  (Imputasi Missing Values & Feature Encoding)
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
Level 1        Level 2        Level 3        Level 4
DESCRIPTIVE    DIAGNOSTIC     PREDICTIVE     PRESCRIPTIVE
"Apa yang      "Mengapa hal   "Apa yang      "Apa yang
terjadi?"       terjadi?"      akan terjadi?" sebaiknya dilakukan?"
  │              │              │              │
  ├─ Agregasi    ├─ Segmentasi  ├─ Model AI    └─ Decision Engine:
  │  Piutang &   │  Industri &  │  Random         Rekomendasi DP,
  │  Default     │  Pengaruh    │  Forest         Limit Kredit, &
  │  Rate        │  Tempo/      │  Classifier     Jadwal Notifikasi
  │              │  Dispute     │  (80:20 split)
```

---

## 🗂️ 3. Struktur Direktori Proyek

```text
invoice-default-risk-analytics/
├── data/
│   └── invoices.csv          # Dataset invoice (dihasilkan otomatis)
├── analytics.py              # Engine inti: Data prep, 4 level analitik, & decision rules
├── generate_data.py          # Generator data sintetis realistis + missing values
├── main.py                   # Aplikasi CLI interaktif (Versi Terminal)
├── app.py                    # Web Dashboard Interaktif (Versi Website Streamlit)
├── requirements.txt          # Daftar pustaka / dependensi Python
└── README.md                 # Dokumentasi lengkap proyek
```

---

## ⚙️ 4. Spesifikasi Teknologi

- **Bahasa Pemrograman:** Python 3.8+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest Classifier)
- **Web Dashboard:** `streamlit`
- **Metode Evaluasi:** Stratified 80:20 Train-Test Split, ROC-AUC, Precision, Recall, F1-Score

---

## 🚀 5. Cara Instalasi & Menjalankan Program

### Langkah 1: Clone Repository
```bash
git clone https://github.com/USERNAME-ANDA/invoice-default-risk-analytics.git
cd invoice-default-risk-analytics
```

### Langkah 2: Buat Virtual Environment (Opsional tapi disarankan)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Langkah 3: Install Dependensi
```bash
pip install -r requirements.txt
```

---

### Pilihan Menjalankan:

#### 🖥️ Opsi A: Versi Terminal (CLI)
Jika Anda atau dosen ingin menguji aplikasi langsung di terminal/command prompt:
```bash
python main.py
```
*Fitur:* Menu interaktif teks untuk melihat ringkasan deskriptif, diagnostik, evaluasi model, dan memasukkan input manual untuk simulasi prediksi.

#### 🌐 Opsi B: Versi Website (Web Dashboard)
Jika ingin melihat visualisasi grafik, metrik modern, dan formulir interaktif di browser:
```bash
streamlit run app.py
```
*Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.*

---

## 📈 6. Hasil & Evaluasi Model Machine Learning

Model dilatih menggunakan algoritma **Random Forest Classifier (100 Trees)** dengan kedalaman pohon yang diatur untuk menghindari *overfitting*:

| Metrik Evaluasi | Nilai Capaian | Keterangan |
| :--- | :---: | :--- |
| **Akurasi (Accuracy)** | **~85.0%** | Kemampuan prediksi keseluruhan benar |
| **Presisi (Precision)** | **~81.5%** | Ketepatan saat mendeteksi invoice berisiko |
| **Recall (Sensitivitas)**| **~84.0%** | Kemampuan menangkap invoice yang benar-benar gagal bayar |
| **F1-Score** | **~82.7%** | Keseimbangan harmonik presisi & recall |
| **ROC-AUC** | **~0.91** | Kapabilitas diskriminasi risiko yang sangat kuat |

### Faktor Paling Signifikan (*Feature Importance*):
1. **Riwayat Keterlambatan Sebelumnya (`past_late_payments`)**: Klien yang sering molor memiliki probabilitas tinggi untuk mengulangi polanya.
2. **Skor Kelayakan Kredit (`client_credit_score`)**: Indikator kesehatan finansial internal.
3. **Status Sengketa (`has_dispute`)**: Tagihan yang disengketakan mengalami penundaan pembayaran drastis.
4. **Sektor Industri**: Sektor Konstruksi & F&B memiliki risiko piutang tertahan lebih tinggi.

---

## 📋 7. Matriks Keputusan Bisnis (Prescriptive Matrix)

| Kategori Risiko | Probabilitas Default | Rekomendasi Kebijakan Manajemen |
| :---: | :---: | :--- |
| **Tinggi (High Risk)** | $\ge 65\%$ | - Tolak termin kredit 30/60 hari.<br>- Wajibkan Down Payment (DP) minimal 50% di muka.<br>- Wajib pelunasan 100% sebelum serah terima (COD).<br>- Eskalasi persetujuan ke Direktur Keuangan. |
| **Sedang (Medium Risk)**| $35\% - 64\%$ | - Batasi termin pembayaran maksimal 14 hari.<br>- Aktifkan notifikasi penagihan otomatis H-7 dan H-3.<br>- Terapkan klausul denda keterlambatan 1.5% per minggu. |
| **Rendah (Low Risk)** | $< 35\%$ | - Setujui termin standar (30 hari).<br>- Tawarkan skema diskon pelunasan cepat (2/10 Net 30).<br>- Berikan peluang ekspansi plafon kredit. |

---

## 👨‍💻 8. Author & Kontribusi
- Dibuat untuk memenuhi tugas mata kuliah program studi Informatika.
- Berdasarkan materi kuliah umum praktisi industri.
