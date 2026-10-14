import pandas as pd
import numpy as np

# Load overlap results and zone splits
overlap_df = pd.read_csv("data/processed/zones/reports/nodule_pixel_overlap.csv")
splits_df = pd.read_csv("data/processed/splits/zone_splits.csv")

print("\n--- ZON ETİKETLEME VE TEMİZ NEGATİF ANALİZİ ---")

# Her hasta ve her zon için nodül örtüşme oranını hesapla
# patient_id x zone_id -> max overlap ratio
patient_zone_max_overlap = {}

for _, row in overlap_df.iterrows():
    pid = row["image"]
    for z in range(1, 6):
        r = row[f"z{z}_ratio"]
        key = (pid, z)
        patient_zone_max_overlap[key] = max(patient_zone_max_overlap.get(key, 0.0), r)

# Şimdi zone_splits tablosuna bakalım
splits_df["max_nodule_overlap"] = splits_df.apply(
    lambda r: patient_zone_max_overlap.get((r["patient_id"], r["zone_id"]), 0.0), axis=1
)

print("\nNodüllü hastalarda (patient_label == 1), zone_label == 0 (negatif etiketlenmiş) zonların nodül örtüşmesi:")
neg_in_pos_patients = splits_df[(splits_df["patient_label"] == 1) & (splits_df["zone_label"] == 0)]
print(f"Toplam nodüllü hasta negatif zon sayısı: {len(neg_in_pos_patients)}")
print(f"Bu zonlardan max_nodule_overlap >= 0.20 olan (YANLIŞ NEGATİF / TÜMÖRLÜ) sayısı: {(neg_in_pos_patients['max_nodule_overlap'] >= 0.20).sum()}")
print(f"Bu zonlardan 0.05 <= max_nodule_overlap < 0.20 olan (ŞÜPHELİ / SINIRDA) sayısı: {((neg_in_pos_patients['max_nodule_overlap'] >= 0.05) & (neg_in_pos_patients['max_nodule_overlap'] < 0.20)).sum()}")
print(f"Bu zonlardan max_nodule_overlap == 0 olan (TAM TEMİZ NEGATİF) sayısı: {(neg_in_pos_patients['max_nodule_overlap'] == 0).sum()}")

# Train kümesindeki etki:
train_neg = splits_df[(splits_df["split"] == "train") & (splits_df["is_train_balanced_sample"]) & (splits_df["zone_label"] == 0)]
print(f"\nEski SVM Train dengeli negatif örnekleri ({len(train_neg)} adet) içinde:")
print(f"Nodül örtüşmesi >= 0.20 olan (Tümörlü zon negatif diye eğitilmiş!): {(train_neg['max_nodule_overlap'] >= 0.20).sum()}")
print(f"Nodül örtüşmesi > 0 olan (Sınırda doku içeren): {(train_neg['max_nodule_overlap'] > 0).sum()}")

# Zon 4 özelinde durum:
z4_train = splits_df[(splits_df["zone_id"] == 4) & (splits_df["split"] == "train")]
z4_neg_leak = z4_train[(z4_train["zone_label"] == 0) & (z4_train["max_nodule_overlap"] >= 0.20)]
print(f"\nZon 4 Train kümesinde negatif etiketli ama >= %20 nodül içeren vaka sayısı: {len(z4_leak := z4_neg_leak)}")
