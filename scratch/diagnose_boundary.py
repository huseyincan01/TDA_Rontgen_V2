import pandas as pd
import numpy as np

# Bbox sınır analizi
mapping_df = pd.read_csv("data/processed/zones/reports/nodule_5zones_mapping.csv")
z_summary = pd.read_csv("data/processed/zones/reports/patients_zones_summary.csv")

print("\n--- BBOX VE ZON BOYUTLARI ANALİZİ ---")
print("Bbox genişlik ve yükseklik istatistikleri (piksel):")
print(mapping_df[["bbox_w", "bbox_h"]].describe())

# Zon 4 ve Zon 5 nodüllerinin y koordinatları
z4_nodules = mapping_df[mapping_df["assigned_zone_id"] == 4]
z5_nodules = mapping_df[mapping_df["assigned_zone_id"] == 5]

print("\nZon 4 (Sol Üst) nodül merkez y koordinatları (min, median, max):")
print(z4_nodules["center_y"].min(), z4_nodules["center_y"].median(), z4_nodules["center_y"].max())

print("\nZon 5 (Sol Alt) nodül merkez y koordinatları (min, median, max):")
print(z5_nodules["center_y"].min(), z5_nodules["center_y"].median(), z5_nodules["center_y"].max())

# Sol akciğer zon sınırları nerede?
# zones.py içinde sol akciğer Ymin..Ymax arası tam 2 eşit parçaya bölünüyor:
# y_mid = (ymin + ymax) // 2
# y_mid civarında (+/- 30 piksel) kaç nodül var?
# Bakalım:
print("\nZon 4 nodüllerinin alt kenarı (bbox_y + bbox_h):")
print("Zon 4 nodüllerinden center_y > 450 olan sayısı:", (z4_nodules["center_y"] > 450).sum(), "/", len(z4_nodules))
print("Zon 5 nodüllerinden center_y < 550 olan sayısı:", (z5_nodules["center_y"] < 550).sum(), "/", len(z5_nodules))
