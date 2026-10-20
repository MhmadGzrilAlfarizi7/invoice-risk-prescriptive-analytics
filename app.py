import os
import streamlit as st
import pandas as pd
import numpy as np

import analytics
import generate_data

# Page configuration
st.set_page_config(
    page_title="Invoice Risk & Decision Analytics",
    page_icon="💼",
    layout="wide"
)

# Cegah Google Translate / browser auto-translate merusak React DOM
st.markdown(
    """
    <meta name="google" content="notranslate" />
    <style>
        body, html {
            -webkit-user-select: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invoices.csv')

def load_data():
    if not os.path.exists(DATA_FILE):
        df = generate_data.generate_invoice_data(1000)
        df.to_csv(DATA_FILE, index=False)
        return df
    return pd.read_csv(DATA_FILE)

def format_rupiah(val):
    return f"Rp {val:,.0f}".replace(",", ".")

df = load_data()

# Header
st.title("💼 Invoice Default Risk Analytics & Decision Support System")
st.markdown("""
**Implementasi Studi Kasus Kuliah Tamu Informatika:** *Data Analytics, AI, & Pemanfaatan Data dalam Pengambilan Keputusan*  
**Pemateri:** Andria Arisal | **Konsep Alur:** Data $\\to$ Informasi $\\to$ Insight $\\to$ Keputusan
""")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
st.sidebar.title("Navigasi Analitik")
page = st.sidebar.radio(
    "Pilih Tingkatan Analisis:",
    [
        "1. Executive Dashboard (Descriptive)",
        "2. Analisis Akar Masalah (Diagnostic)",
        "3. Evaluasi Model AI (Predictive)",
        "4. Simulator Keputusan Bisnis (Prescriptive)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Struktur Proyek:**
- **Level 1:** Apa yang terjadi?
- **Level 2:** Mengapa terjadi?
- **Level 3:** Apa yang akan terjadi?
- **Level 4:** Apa yang harus dilakukan?
""")

# ==========================================
# PAGE 1: DESCRIPTIVE ANALYTICS
# ==========================================
if "1. Executive Dashboard" in page:
    st.header("📊 Level 1: Descriptive Analytics (Apa yang Terjadi?)")
    st.write("Agregasi data historis invoice untuk memantau performa arus kas piutang perusahaan.")
    
    desc = analytics.get_descriptive_analytics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invoice", f"{desc['total_invoices']} Dokumen")
    col2.metric("Total Nilai Piutang", format_rupiah(desc['total_receivables']))
    col3.metric("Tingkat Default", f"{desc['default_rate_pct']:.1f}%")
    col4.metric("Total Dana Macet", format_rupiah(desc['total_at_risk_amount']), delta="-Risiko Kerugian", delta_color="inverse")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Status Pembayaran")
        status_counts = df['is_defaulted'].value_counts().rename(index={0: 'Lunas Tepat Waktu', 1: 'Default / Telat Bayar'})
        st.bar_chart(status_counts)
        
    with col_b:
        st.subheader("Perbandingan Rata-rata Nilai Invoice")
        avg_df = pd.DataFrame({
            'Status': ['Lunas Tepat Waktu', 'Macet / Default'],
            'Rata-rata Nominal (Rp)': [desc['avg_amount_paid'], desc['avg_amount_default']]
        }).set_index('Status')
        st.bar_chart(avg_df)

    st.success(f"""
    **💡 Insight Deskriptif:**
    Perusahaan memiliki total piutang sebesar **{format_rupiah(desc['total_receivables'])}**, namun sebesar **{format_rupiah(desc['total_at_risk_amount'])} ({desc['default_rate_pct']:.1f}%)** berada dalam kondisi macet atau terlambat serius.
    """)

# ==========================================
# PAGE 2: DIAGNOSTIC ANALYTICS
# ==========================================
elif "2. Analisis Akar Masalah" in page:
    st.header("🔍 Level 2: Diagnostic Analytics (Mengapa Hal Tersebut Terjadi?)")
    st.write("Menganalisis faktor-faktor penyebab keterlambatan pembayaran berdasarkan segmen industri, tempo, dan sengketa.")
    
    diag = analytics.get_diagnostic_analytics(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Tingkat Default Berdasarkan Sektor Industri")
        ind_chart = diag['industry_analysis'].set_index('client_industry')['default_rate']
        st.bar_chart(ind_chart)
        st.dataframe(diag['industry_analysis'])
        
    with col2:
        st.subheader("2. Pengaruh Tempo Pembayaran (Payment Terms)")
        terms_chart = diag['terms_analysis'].set_index('payment_terms_days')['default_rate']
        st.line_chart(terms_chart)
        st.caption("Semakin panjang tempo kredit yang diberikan, semakin tinggi risiko keterlambatan.")
        
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Pengaruh Sengketa / Komplain Barang (Dispute)")
        disp_df = diag['dispute_analysis'].copy()
        disp_df['Status Sengketa'] = disp_df['has_dispute'].map({0: 'Tanpa Sengketa', 1: 'Ada Sengketa'})
        st.bar_chart(disp_df.set_index('Status Sengketa')['default_rate'])
        
    with col4:
        st.subheader("4. Profil Skor Kredit & Keterlambatan")
        st.dataframe(diag['profile_summary'].rename(columns={
            'is_defaulted': 'Status Default (1=Ya, 0=Tidak)',
            'client_credit_score': 'Rata-rata Skor Kredit',
            'past_late_payments': 'Rata-rata Keterlambatan Masa Lalu'
        }))

    st.warning("""
    **💡 Insight Diagnostik:**
    - Klien di industri **Konstruksi** dan **F&B** menunjukkan persentase gagal bayar tertinggi.
    - Tagihan dengan durasi tempo di atas **45 hari** dan yang mengalami **sengketa barang** berpotensi gagal bayar 3 kali lipat lebih besar dibanding tagihan normal.
    """)

# ==========================================
# PAGE 3: PREDICTIVE ANALYTICS
# ==========================================
elif "3. Evaluasi Model AI" in page:
    st.header("🤖 Level 3: Predictive Analytics (Apa yang Kemungkinan Terjadi?)")
    st.write("Melatih model Machine Learning (*Random Forest Classifier*) untuk memprediksi probabilitas gagal bayar invoice baru.")
    
    with st.spinner("Melatih model Random Forest pada 800 data train dan mengevaluasi pada 200 data test..."):
        model, feature_names, metrics, fi = analytics.train_predictive_model(df)
        
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Akurasi (Accuracy)", f"{metrics['accuracy']*100:.1f}%")
    col2.metric("Presisi (Precision)", f"{metrics['precision']*100:.1f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    col4.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    st.markdown("---")
    st.subheader("Faktor Paling Berpengaruh dalam Prediksi (Feature Importance)")
    st.bar_chart(fi.set_index('feature'))
    
    st.info("""
    **Catatan Metodologi:**
    Model dilatih dengan algoritma **Random Forest (100 Trees)** menggunakan validasi *stratified train-test split* 80:20.
    Faktor riwayat keterlambatan masa lalu (*past_late_payments*) dan skor kredit (*client_credit_score*) merupakan prediktor paling signifikan.
    """)

# ==========================================
# PAGE 4: PRESCRIPTIVE ANALYTICS
# ==========================================
elif "4. Simulator Keputusan Bisnis" in page:
    st.header("⚡ Level 4: Prescriptive Analytics (Apa yang Sebaiknya Dilakukan?)")
    st.write("Simulasikan pengajuan tagihan baru untuk mendapatkan estimasi risiko AI dan rekomendasi tindakan kebijakan manajemen.")
    
    # Cache model
    @st.cache_resource
    def get_trained_model():
        model, feature_names, _, _ = analytics.train_predictive_model(df)
        return model, feature_names
    
    model, feature_names = get_trained_model()
    
    with st.form("simulation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            client_name = st.text_input("Nama Klien / Perusahaan", value="PT Maju Bersama Abadi")
            industry = st.selectbox("Sektor Industri", ['Konstruksi', 'Retail', 'Teknologi', 'Manufaktur', 'F&B', 'Logistik'])
            amount = st.number_input("Nominal Tagihan (Rp)", min_value=1_000_000, max_value=500_000_000, value=75_000_000, step=5_000_000)
            terms = st.selectbox("Pengajuan Tempo Pembayaran (Hari)", [14, 30, 45, 60], index=1)
            
        with col2:
            past_delays = st.slider("Jumlah Riwayat Keterlambatan Sebelumnya", min_value=0, max_value=7, value=2)
            credit_score = st.slider("Skor Kelayakan Kredit Klien (300-850)", min_value=300, max_value=850, value=580)
            has_dispute = st.radio("Apakah ada catatan sengketa/komplain barang sebelumnya?", ["Tidak Ada", "Ada Sengketa"], index=0)
            dispute_val = 1 if has_dispute == "Ada Sengketa" else 0
            
        submitted = st.form_submit_button("🔍 Jalankan Analisis & Dapatkan Rekomendasi", use_container_width=True)
        
    if submitted:
        # Siapkan payload feature berdasarkan kolom yang diharapkan model
        input_data = {col: 0 for col in feature_names}
        input_data['invoice_amount'] = amount
        input_data['payment_terms_days'] = terms
        input_data['past_late_payments'] = past_delays
        input_data['client_credit_score'] = credit_score
        input_data['has_dispute'] = dispute_val
        
        # Set dummy variable industri jika kolomnya ada di feature_names
        ind_col = f'ind_{industry}'
        if ind_col in input_data:
            input_data[ind_col] = 1
            
        input_df = pd.DataFrame([input_data])[feature_names]
        prob_default = model.predict_proba(input_df)[0][1]
        
        decision = analytics.get_prescriptive_decision(prob_default, amount)
        
        st.markdown("---")
        st.subheader("🎯 Hasil Keputusan Sistem (Prescriptive Decision)")
        
        if decision['color'] == 'red':
            st.error(f"### Kategori: {decision['risk_category']} — Probabilitas Default: {decision['risk_percentage']}%")
        elif decision['color'] == 'yellow':
            st.warning(f"### Kategori: {decision['risk_category']} — Probabilitas Default: {decision['risk_percentage']}%")
        else:
            st.success(f"### Kategori: {decision['risk_category']} — Probabilitas Default: {decision['risk_percentage']}%")
            
        st.markdown("#### 📋 Tindakan Rekomendasi Manajemen (Actionable Items):")
        for i, rec in enumerate(decision['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")
