import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def clean_and_prepare_data(df: pd.DataFrame):
    """
    Tahap Data Preparation:
    Membersihkan missing values dan menyiapkan fitur untuk pemodelan (CRISP-DM Stage 2).
    """
    df_clean = df.copy()
    
    # Imputasi Missing Values
    if df_clean['client_credit_score'].isnull().sum() > 0:
        median_cs = df_clean['client_credit_score'].median()
        df_clean['client_credit_score'].fillna(median_cs, inplace=True)
        
    if df_clean['past_late_payments'].isnull().sum() > 0:
        mode_pl = df_clean['past_late_payments'].mode()[0]
        df_clean['past_late_payments'].fillna(mode_pl, inplace=True)
        
    return df_clean

def get_descriptive_analytics(df: pd.DataFrame) -> dict:
    """
    Level 1: Descriptive Analytics (Apa yang terjadi?)
    Menghitung agregasi data, rasio keterlambatan, dan total piutang.
    """
    total_invoices = len(df)
    total_receivables = df['invoice_amount'].sum()
    default_count = df['is_defaulted'].sum()
    default_rate = (default_count / total_invoices) * 100
    total_at_risk = df[df['is_defaulted'] == 1]['invoice_amount'].sum()
    
    avg_amount_paid = df[df['is_defaulted'] == 0]['invoice_amount'].mean()
    avg_amount_default = df[df['is_defaulted'] == 1]['invoice_amount'].mean()
    
    return {
        'total_invoices': total_invoices,
        'total_receivables': total_receivables,
        'default_count': default_count,
        'default_rate_pct': default_rate,
        'total_at_risk_amount': total_at_risk,
        'avg_amount_paid': avg_amount_paid,
        'avg_amount_default': avg_amount_default
    }

def get_diagnostic_analytics(df: pd.DataFrame) -> dict:
    """
    Level 2: Diagnostic Analytics (Mengapa hal tersebut terjadi?)
    Mencari faktor penyebab invoice macet/telat berdasarkan industri, tempo, dan riwayat.
    """
    # 1. Tingkat default per industri
    industry_default = df.groupby('client_industry')['is_defaulted'].agg(
        total_invoices='count',
        default_count='sum',
        default_rate=lambda x: (x.sum() / x.count()) * 100
    ).sort_values(by='default_rate', ascending=False).reset_index()

    # 2. Tingkat default per tempo pembayaran (payment terms)
    terms_default = df.groupby('payment_terms_days')['is_defaulted'].agg(
        total_invoices='count',
        default_rate=lambda x: (x.sum() / x.count()) * 100
    ).reset_index()

    # 3. Pengaruh adanya sengketa/dispute barang
    dispute_default = df.groupby('has_dispute')['is_defaulted'].agg(
        total='count',
        default_rate=lambda x: (x.sum() / x.count()) * 100
    ).reset_index()

    # 4. Rata-rata credit score & riwayat keterlambatan
    profile_summary = df.groupby('is_defaulted')[['client_credit_score', 'past_late_payments']].mean().reset_index()

    return {
        'industry_analysis': industry_default,
        'terms_analysis': terms_default,
        'dispute_analysis': dispute_default,
        'profile_summary': profile_summary
    }

def train_predictive_model(df: pd.DataFrame):
    """
    Level 3: Predictive Analytics (Apa yang kemungkinan akan terjadi?)
    Melatih model Random Forest Classifier untuk memprediksi probabilitas gagal bayar.
    """
    df_clean = clean_and_prepare_data(df)
    
    # Feature Engineering & Encoding
    features = ['invoice_amount', 'payment_terms_days', 'past_late_payments', 'client_credit_score', 'has_dispute']
    X_num = df_clean[features]
    X_cat = pd.get_dummies(df_clean['client_industry'], prefix='ind', drop_first=True)
    
    X = pd.concat([X_num, X_cat], axis=1)
    y = df_clean['is_defaulted']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model: Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    return model, X.columns.tolist(), metrics, feature_importances

def get_prescriptive_decision(prob_default: float, invoice_amount: float) -> dict:
    """
    Level 4: Prescriptive Analytics (Apa yang sebaiknya dilakukan?)
    Decision Engine yang mengubah probabilitas prediksi menjadi rekomendasi kebijakan bisnis nyata.
    """
    risk_pct = round(prob_default * 100, 1)
    
    if prob_default >= 0.65:
        category = "RISIKO TINGGI (HIGH RISK)"
        color = "red"
        recommendations = [
            "TOLAK pengajuan termin kredit 30/60 hari.",
            "Wajibkan uang muka (Down Payment) minimal 50% di awal sebelum pesanan diproses.",
            "Wajibkan pelunasan 100% saat barang/jasa diserahterimakan (Cash on Delivery).",
            "Minta jaminan pembayaran (Letter of Credit / Bank Garansi) jika nominal > Rp 50 Juta.",
            "Eskalasi ke Direktur Keuangan untuk persetujuan khusus."
        ]
    elif prob_default >= 0.35:
        category = "RISIKO SEDANG (MEDIUM RISK)"
        color = "yellow"
        recommendations = [
            "Batasi termin pembayaran maksimal 14 hari.",
            "Kirimkan notifikasi penagihan otomatis via Email & WhatsApp pada H-7 dan H-3 sebelum jatuh tempo.",
            "Terapkan denda keterlambatan 1.5% per minggu untuk mencegah molor.",
            "Pastikan Berita Acara Serah Terima (BAST) ditandatangani basah oleh klien."
        ]
    else:
        category = "RISIKO RENDAH (LOW RISK)"
        color = "green"
        recommendations = [
            "Setujui permohonan termin pembayaran standar (30 hari).",
            "Tawarkan skema potongan diskon 2% jika klien melunasi dalam 10 hari (Skema 2/10 Net 30).",
            "Pertimbangkan menaikkan limit plafon kredit klien untuk meningkatkan volume transaksi penjualan."
        ]
        
    return {
        'risk_percentage': risk_pct,
        'risk_category': category,
        'color': color,
        'recommendations': recommendations
    }
