"""800 TDA Özelliği için Çoklu Test Düzeltmesi (FDR Benjamini-Hochberg & Bonferroni) Analizi.

Saf numpy/scipy ile dış kütüphane bağımlılığı olmadan FDR ve Bonferroni hesaplar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FEATURES_PATH = "data/processed/features/X_zone_features.npy"
SPLITS_PATH = "data/processed/splits/zone_splits.csv"

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-değerlerini hesaplar."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    q_values = np.zeros(n)
    # q_i = min_{k >= i} (n / (k + 1)) * p_{(k)}
    cummin = float("inf")
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q_val = (n / rank) * sorted_p[i]
        cummin = min(cummin, q_val)
        q_values[i] = min(cummin, 1.0)
        
    # Orijinal sıralamaya geri dön
    original_q = np.zeros(n)
    original_q[sorted_indices] = q_values
    return original_q

def main():
    print("Veriler yükleniyor...")
    X = np.load(FEATURES_PATH)
    splits = pd.read_csv(SPLITS_PATH)

    print("\n" + "=" * 80)
    print("TDA ÖZELLİKLERİ İÇİN ÇOKLU TEST DÜZELTMESİ (FDR & BONFERRONI) ANALİZİ")
    print("=" * 80)

    for zid in range(1, 6):
        # Yalnızca train kümesindeki verilere bakıyoruz (veri sızıntısı olmaması için)
        mask = (splits["split"] == "train") & (splits["zone_id"] == zid)
        y = splits.loc[mask, "zone_label"].values
        X_z = X[mask.values]

        pos_mask = (y == 1)
        neg_mask = (y == 0)

        p_values = []
        for feat_idx in range(800):
            val_pos = X_z[pos_mask, feat_idx]
            val_neg = X_z[neg_mask, feat_idx]
            if np.all(val_pos == val_pos[0]) and np.all(val_neg == val_neg[0]) and val_pos[0] == val_neg[0]:
                p_values.append(1.0)
            else:
                try:
                    stat, p = mannwhitneyu(val_pos, val_neg, alternative='two-sided')
                    p_values.append(p)
                except Exception:
                    p_values.append(1.0)

        p_values = np.array(p_values)

        # Ham p < 0.05
        raw_sig = np.sum(p_values < 0.05)

        # Benjamini-Hochberg FDR
        q_values = benjamini_hochberg(p_values)
        fdr_sig = np.sum(q_values < 0.05)

        # Bonferroni
        bonf_threshold = 0.05 / 800
        bonf_sig = np.sum(p_values < bonf_threshold)

        min_p = np.min(p_values)
        min_q = np.min(q_values)

        print(f"Zon {zid}:")
        print(f"  Toplam Pozitif: {pos_mask.sum()}, Negatif: {neg_mask.sum()}")
        print(f"  Ham p < 0.05 geçen özellik sayısı:           {raw_sig:>3} / 800 (Beklenen rastgele: ~40)")
        print(f"  FDR (Benjamini-Hochberg) q < 0.05 sayısı:    {fdr_sig:>3} / 800")
        print(f"  Bonferroni p < (0.05/800) sayısı:            {bonf_sig:>3} / 800")
        print(f"  En küçük p-değeri: {min_p:.2e} | En küçük FDR q-değeri: {min_q:.2e}")
        print("-" * 80)

if __name__ == "__main__":
    main()
