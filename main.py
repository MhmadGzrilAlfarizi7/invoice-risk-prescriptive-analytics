import os
import sys
import pandas as pd
import numpy as np

# Import modul analitik internal
import analytics
import generate_data

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invoices.csv')

def format_rupiah(val):
    return f"Rp {val:,.0f}".replace(",", ".")

def ensure_data():
    if not os.path.exists(DATA_FILE):
        print("\n[!] File invoices.csv belum ditemukan. Menghasilkan dataset otomatis...")
        df = generate_data.generate_invoice_data(1000)
        df.to_csv(DATA_FILE, index=False)
        print("[✓] Dataset berhasil dibuat.")
    return pd.read_csv(DATA_FILE)

def display_descriptive(df):
    desc = analytics.get_descriptive_analytics(df)
    print("\n" + "="*65)
    print("📊 LEVEL 1: DESCRIPTIVE ANALYTICS (Apa yang sedang terjadi?)")
    print("="*65)
    print(f"• Total Invoice Tercatat        : {desc['total_invoices']} transaksi")
    print(f"• Total Nilai Piutang           : {format_rupiah(desc['total_receivables'])}")
    print(f"• Jumlah Invoice Gagal/Telat    : {desc['default_count']} invoice")
    print(f"• Tingkat Default (Default Rate): {desc['default_rate_pct']:.2f}%")
    print(f"• Total Uang Tertahan / Macet   : {format_rupiah(desc['total_at_risk_amount'])}")
    print(f"• Rata-rata Nominal Lunas Tepat : {format_rupiah(desc['avg_amount_paid'])}")
    print(f"• Rata-rata Nominal Macet/Telat : {format_rupiah(desc['avg_amount_default'])}")
    print("="*65)
    print("💡 INSIGHT DESKRIPTIF:")
    print(f"Sekitar {desc['default_rate_pct']:.1f}% invoice mengalami gagal bayar/keterlambatan serius,")
    print(f"dengan total uang tertahan sebesar {format_rupiah(desc['total_at_risk_amount'])}.")
    print("="*65)

def display_diagnostic(df):
    diag = analytics.get_diagnostic_analytics(df)
    print("\n" + "="*65)
    print("🔍 LEVEL 2: DIAGNOSTIC ANALYTICS (Mengapa hal itu terjadi?)")
    print("="*65)
    
    print("\n[1] Tingkat Default Berdasarkan Sektor Industri:")
    print("-" * 55)
    print(f"{'Industri':<15} | {'Total Invoice':<15} | {'Default Rate':<15}")
    print("-" * 55)
    for _, row in diag['industry_analysis'].iterrows():
        print(f"{row['client_industry']:<15} | {row['total_invoices']:<15} | {row['default_rate']:.2f}%")
        
    print("\n[2] Tingkat Default Berdasarkan Termin Pembayaran (Tempo):")
    print("-" * 55)
    print(f"{'Tempo (Hari)':<15} | {'Total Invoice':<15} | {'Default Rate':<15}")
    print("-" * 55)
    for _, row in diag['terms_analysis'].iterrows():
        print(f"{row['payment_terms_days']:<15} | {row['total_invoices']:<15} | {row['default_rate']:.2f}%")

    print("\n[3] Pengaruh Sengketa/Dispute Barang:")
    print("-" * 55)
    for _, row in diag['dispute_analysis'].iterrows():
        status = "Ada Sengketa (Dispute)" if row['has_dispute'] == 1 else "Normal (Tanpa Sengketa)"
        print(f"• {status:<25}: Default Rate = {row['default_rate']:.2f}%")
        
    print("="*65)
    print("💡 INSIGHT DIAGNOSTIK:")
    print("1. Industri Konstruksi & F&B memiliki risiko keterlambatan tertinggi.")
    print("2. Semakin panjang tempo pembayaran (45-60 hari), risiko default naik signifikan.")
    print("3. Invoice yang mengalami sengketa memiliki kemungkinan macet hingga 3x lipat lebih tinggi.")
    print("="*65)

def train_and_evaluate(df):
    print("\n" + "="*65)
    print("🤖 LEVEL 3: PREDICTIVE ANALYTICS (Melatih Model AI / Machine Learning)")
    print("="*65)
    print("[*] Melakukan data preparation (imputasi missing values & encoding)...")
    print("[*] Melatih model Random Forest Classifier (100 estimators)...")
    
    model, feature_names, metrics, fi = analytics.train_predictive_model(df)
    
    print("\n[✓] Evaluasi Kinerja Model AI pada Data Uji (Test Set 20%):")
    print("-" * 45)
    print(f"• Akurasi (Accuracy)  : {metrics['accuracy'] * 100:.2f}%")
    print(f"• Presisi (Precision) : {metrics['precision'] * 100:.2f}%")
    print(f"• Recall              : {metrics['recall'] * 100:.2f}%")
    print(f"• F1-Score            : {metrics['f1_score'] * 100:.2f}%")
    print(f"• ROC-AUC Score       : {metrics['roc_auc']:.4f}")
    print("-" * 45)
    
    print("\nTop 4 Faktor Paling Berpengaruh (Feature Importance):")
    for idx, row in fi.head(4).iterrows():
        print(f"  {row['feature']:<22}: {row['importance']*100:.1f}% pengaruh")
    print("="*65)
    return model, feature_names

def interactive_prediction(model, feature_names):
    print("\n" + "="*65)
    print("⚡ LEVEL 4: PRESCRIPTIVE ANALYTICS (Simulasi Prediksi & Keputusan Bisnis)")
    print("="*65)
    print("Masukkan data invoice baru untuk dianalisis oleh AI:\n")
    
    try:
        client_name = input("1. Nama Klien / Perusahaan: ") or "PT Contoh Nusantara"
        
        print("\nPilih Industri Klien:")
        industries = ['Konstruksi', 'Retail', 'Teknologi', 'Manufaktur', 'F&B', 'Logistik']
        for i, ind in enumerate(industries, 1):
            print(f"   [{i}] {ind}")
        ind_idx = int(input("Pilih nomor (1-6) [default: 1]: ") or "1") - 1
        ind_chosen = industries[max(0, min(ind_idx, 5))]
        
        amount_input = input("2. Nominal Tagihan / Invoice (Rp) [default: 75000000]: ") or "75000000"
        amount = float(amount_input.replace(".", "").replace(",", ""))
        
        terms = int(input("3. Tempo Pembayaran yang diminta (hari, misal: 30) [default: 30]: ") or "30")
        delays = int(input("4. Riwayat telat bayar sebelumnya (0-7 kali) [default: 2]: ") or "2")
        credit_score = int(input("5. Skor Kredit Internal Klien (300-850) [default: 600]: ") or "600")
        dispute_str = input("6. Apakah ada keluhan/sengketa barang? (y/n) [default: n]: ").lower()
        has_dispute = 1 if dispute_str == 'y' else 0
        
        # Susun feature row
        input_data = {col: 0 for col in feature_names}
        input_data['invoice_amount'] = amount
        input_data['payment_terms_days'] = terms
        input_data['past_late_payments'] = delays
        input_data['client_credit_score'] = credit_score
        input_data['has_dispute'] = has_dispute
        
        ind_col = f'ind_{ind_chosen}'
        if ind_col in input_data:
            input_data[ind_col] = 1
            
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Prediksi probabilitas
        prob_default = model.predict_proba(input_df)[0][1]
        decision = analytics.get_prescriptive_decision(prob_default, amount)
        
        print("\n" + "="*65)
        print("🎯 HASIL KEPUTUSAN SISTEM (PRESCRIPTIVE DECISION)")
        print("="*65)
        print(f"Klien                  : {client_name} ({ind_chosen})")
        print(f"Nominal Tagihan        : {format_rupiah(amount)}")
        print(f"Probabilitas Default   : {decision['risk_percentage']}%")
        print(f"Status Kategori Risiko : [{decision['risk_category']}]")
        print("-" * 65)
        print("📋 REKOMENDASI TINDAKAN BISNIS (ACTIONABLE DECISION):")
        for i, rec in enumerate(decision['recommendations'], 1):
            print(f"  {i}. {rec}")
        print("="*65)
        
    except Exception as e:
        print(f"\n[!] Terjadi kesalahan input: {e}")

def main():
    df = ensure_data()
    model = None
    feature_names = None
    
    while True:
        print("\n" + "="*65)
        print("   SISTEM ANALISIS RISIKO TAGIHAN & PENGAMBILAN KEPUTUSAN")
        print("   Implementasi Materi Kuliah Tamu: Andria Arisal")
        print("="*65)
        print(" [1] Level 1: Descriptive Analytics (Apa yang Terjadi?)")
        print(" [2] Level 2: Diagnostic Analytics (Mengapa Terjadi?)")
        print(" [3] Level 3: Predictive Analytics (Latih Model AI)")
        print(" [4] Level 4: Prescriptive Analytics (Simulasi Prediksi & Keputusan)")
        print(" [5] Jalankan Seluruh Alur Analitik Sekaligus")
        print(" [0] Keluar")
        print("="*65)
        
        choice = input("Pilih menu (0-5): ").strip()
        
        if choice == '1':
            display_descriptive(df)
        elif choice == '2':
            display_diagnostic(df)
        elif choice == '3':
            model, feature_names = train_and_evaluate(df)
        elif choice == '4':
            if model is None:
                print("\n[*] Menyiapkan dan melatih model Machine Learning terlebih dahulu...")
                model, feature_names = train_and_evaluate(df)
            interactive_prediction(model, feature_names)
        elif choice == '5':
            display_descriptive(df)
            display_diagnostic(df)
            model, feature_names = train_and_evaluate(df)
            interactive_prediction(model, feature_names)
        elif choice == '0':
            print("\nTerima kasih! Program selesai.")
            break
        else:
            print("\n[!] Pilihan tidak valid, silakan coba lagi.")

if __name__ == '__main__':
    main()
