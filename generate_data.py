import os
import numpy as np
import pandas as pd

def generate_invoice_data(num_samples=1000, random_state=42):
    """
    Menghasilkan dataset invoice realistis dengan sedikit noise dan missing values
    untuk merefleksikan kondisi data dunia industri nyata (sesuai materi kuliah umum).
    """
    np.random.seed(random_state)
    
    industries = ['Konstruksi', 'Retail', 'Teknologi', 'Manufaktur', 'F&B', 'Logistik']
    industry_risk_weights = {
        'Konstruksi': 0.35, # Sektor dengan piutang macet lebih sering
        'Retail': 0.25,
        'Teknologi': 0.15,
        'Manufaktur': 0.20,
        'F&B': 0.30,
        'Logistik': 0.22
    }
    
    client_names = [f"PT Mitra Sejahtera {i}" for i in range(1, 201)]
    
    data = []
    for i in range(1, num_samples + 1):
        inv_id = f"INV-{10000 + i}"
        client = np.random.choice(client_names)
        industry = np.random.choice(industries)
        
        # Nominal tagihan antara 5 Juta s/d 250 Juta Rupiah
        amount = np.random.choice([
            np.random.uniform(5_000_000, 25_000_000),
            np.random.uniform(25_000_000, 100_000_000),
            np.random.uniform(100_000_000, 250_000_000)
        ], p=[0.5, 0.35, 0.15])
        
        # Termin pembayaran (tempo) dalam hari
        terms_days = int(np.random.choice([14, 30, 45, 60], p=[0.2, 0.5, 0.2, 0.1]))
        
        # Riwayat keterlambatan sebelumnya (0 s/d 7 kali)
        past_delays = int(np.random.poisson(lam=1.2))
        past_delays = min(past_delays, 7)
        
        # Skor kelayakan kredit internal (300 - 850)
        credit_score = int(np.clip(np.random.normal(loc=650 - (past_delays * 30), scale=70), 300, 850))
        
        # Apakah ada sengketa tagihan / barang bermasalah (dispute)
        disputed = int(np.random.choice([0, 1], p=[0.90, 0.10]))
        
        # Rumus probabilitas default/keterlambatan (ground truth logika bisnis)
        base_risk = industry_risk_weights[industry]
        risk_score = (
            base_risk * 0.25 +
            (amount / 250_000_000) * 0.20 +
            (terms_days / 60) * 0.15 +
            (past_delays / 7) * 0.25 +
            (1 - (credit_score - 300) / 550) * 0.20 +
            (0.35 if disputed == 1 else 0.0)
        )
        
        prob_default = np.clip(risk_score, 0.05, 0.95)
        is_default = int(np.random.binomial(1, prob_default))
        
        data.append({
            'invoice_id': inv_id,
            'client_name': client,
            'client_industry': industry,
            'invoice_amount': round(amount, -3), # Pembulatan ribuan
            'payment_terms_days': terms_days,
            'past_late_payments': past_delays,
            'client_credit_score': credit_score,
            'has_dispute': disputed,
            'is_defaulted': is_default
        })
        
    df = pd.DataFrame(data)
    
    # Menambahkan ~3% missing values secara sengaja untuk menunjukkan proses Data Preparation / Cleaning
    mask_missing_cs = np.random.rand(len(df)) < 0.03
    df.loc[mask_missing_cs, 'client_credit_score'] = np.nan
    
    mask_missing_pl = np.random.rand(len(df)) < 0.02
    df.loc[mask_missing_pl, 'past_late_payments'] = np.nan

    return df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'invoices.csv')
    
    print("Membuat dataset invoice realistis...")
    df = generate_invoice_data(1000)
    df.to_csv(output_path, index=False)
    print(f"Sukses! Dataset tersimpan di: {output_path}")
    print(f"Total baris: {len(df)}, Fitur: {list(df.columns)}")
