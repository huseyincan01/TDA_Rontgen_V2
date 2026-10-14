import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

print("=" * 80)
print("TDA & VERİ SETİ TEŞHİS ANALİZİ")
print("=" * 80)

# Load data
X = np.load("data/processed/features/X_zone_features.npy")
meta = pd.read_csv("data/processed/splits/zone_splits.csv")

print(f"Toplam özellik matrisi: {X.shape}")

# -------------------------------------------------------------
# 1. TDA ÖZELLİK SPARSITY VE AKTİF BİN SAYISI
# -------------------------------------------------------------
h0 = X[:, :400]
h1 = X[:, 400:]

non_zero_h0 = (h0 > 0).mean(axis=0)
non_zero_h1 = (h1 > 0).mean(axis=0)

active_h0_bins = (non_zero_h0 > 0.01).sum()
active_h1_bins = (non_zero_h1 > 0.01).sum()

print("\n--- 1. TDA ÖZELLİK ANALİZİ (20x20 Izgara Doluluğu) ---")
print(f"H0 toplam 400 bin'den aktif olan (en az %1 örnekte >0): {active_h0_bins} / 400")
print(f"H1 toplam 400 bin'den aktif olan (en az %1 örnekte >0): {active_h1_bins} / 400")
print(f"H0 ortalama sıfır oranı: %{((h0 == 0).mean() * 100):.1f}")
print(f"H1 ortalama sıfır oranı: %{((h1 == 0).mean() * 100):.1f}")

# -------------------------------------------------------------
# 2. ZON BAZINDA POZİTİF vs NEGATİF ANLAMLILIK (Mann-Whitney U)
# -------------------------------------------------------------
print("\n--- 2. ZON BAZINDA POZİTİF vs NEGATİF AYRIŞMA GÜCÜ (TRAIN KÜMESİ) ---")
for z in range(1, 6):
    z_mask = (meta["split"] == "train") & (meta["zone_id"] == z) & (meta["is_train_balanced_sample"])
    z_pos = X[z_mask & (meta["zone_label"] == 1)]
    z_neg = X[z_mask & (meta["zone_label"] == 0)]
    
    # Her özellik için Mann-Whitney U p-değeri
    p_vals = []
    for f in range(800):
        try:
            stat, p = stats.mannwhitneyu(z_pos[:, f], z_neg[:, f], alternative="two-sided")
            p_vals.append(p)
        except Exception:
            p_vals.append(1.0)
    p_vals = np.array(p_vals)
    sig_05 = (p_vals < 0.05).sum()
    sig_01 = (p_vals < 0.01).sum()
    print(f"Zon {z}: Toplam 800 özellikten p < 0.05 olan: {sig_05:>3} adet (%{sig_05/800*100:.1f}) | p < 0.01 olan: {sig_01:>3} adet")

# -------------------------------------------------------------
# 3. VERİ KAYNAKLARI (DOMAIN SHIFT) ANALİZİ
# -------------------------------------------------------------
print("\n--- 3. VERİ KAYNAKLARI (DOMAIN SHIFT) ANALİZİ ---")
# PCA ilk 2 bileşen
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
print(f"PCA ilk 5 bileşenin açıkladığı varyans: {pca.explained_variance_ratio_.round(4)}")

# Kaynaklara göre ilk temel bileşenin ortalaması
meta["pca_1"] = X_pca[:, 0]
meta["pca_2"] = X_pca[:, 1]
meta["pca_3"] = X_pca[:, 2]

print("\nKaynak Veri Kümelerine Göre Ortalama Özellik Büyüklüğü (Norm):")
meta["feat_norm"] = np.linalg.norm(X, axis=1)
norm_by_src = meta.groupby("source_dataset")["feat_norm"].agg(["mean", "std", "count"])
print(norm_by_src)

print("\nEtikete (0 vs 1) Göre Ortalama Özellik Normu:")
print(meta.groupby("patient_label")["feat_norm"].agg(["mean", "std", "count"]))

# -------------------------------------------------------------
# 4. ZON 4 ve ZON 5 SINIR ÇAKIŞMASI
# -------------------------------------------------------------
print("\n--- 4. ZON SINIRLARI VE NODÜL KONUMLARI ---")
mapping_df = pd.read_csv("data/processed/zones/reports/nodule_5zones_mapping.csv")
print(f"Toplam nodül kaydı: {len(mapping_df)}")
print("Nodül merkezlerinin zonlara göre dağılımı:")
print(mapping_df["assigned_zone_id"].value_counts().sort_index())
print(f"Akciğer dışı / mediastinal fallback sayısı: {mapping_df['is_fallback'].sum()}")
