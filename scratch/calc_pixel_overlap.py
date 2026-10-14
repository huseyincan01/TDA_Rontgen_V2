import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

print("=" * 80)
print("BOUNDING BOX - GERÇEK ZON MASKESİ PİKSEL ÖRTÜŞME ANALİZİ")
print("=" * 80)

mapping_path = Path("data/processed/zones/reports/nodule_5zones_mapping.csv")
mapping_df = pd.read_csv(mapping_path)
patients_dir = Path("data/processed/zones/patients")

records = []
t0 = time.time()

# Group nodules by patient to read each mask only once
grouped = mapping_df.groupby("image")
print(f"Toplam nodüllü hasta sayısı: {len(grouped)}")
print(f"Toplam nodül (bbox) sayısı:  {len(mapping_df)}")

for patient_id, group in tqdm(grouped, desc="Maskeler inceleniyor"):
    mask_path = patients_dir / patient_id / "mask_5zones.mha"
    if not mask_path.exists():
        continue
    
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[0]
        
    H, W = mask_arr.shape
    
    for _, row in group.iterrows():
        bx = int(row["bbox_x"])
        by = int(row["bbox_y"])
        bw = int(row["bbox_w"])
        bh = int(row["bbox_h"])
        assigned_z = int(row["assigned_zone_id"])
        
        y0 = max(0, by)
        y1 = min(H, by + bh)
        x0 = max(0, bx)
        x1 = min(W, bx + bw)
        
        roi = mask_arr[y0:y1, x0:x1]
        total_bbox_pixels = bw * bh
        
        zone_pixels = {}
        zone_ratios = {}
        for z in range(1, 6):
            cnt = int((roi == z).sum())
            zone_pixels[z] = cnt
            zone_ratios[z] = round(cnt / total_bbox_pixels, 4)
            
        outside_pixels = int((roi == 0).sum())
        lung_pixels = total_bbox_pixels - outside_pixels
        
        # Overlapping zones with >= 20% of bbox area
        zones_ge_20pct = [z for z, r in zone_ratios.items() if r >= 0.20]
        zones_ge_10pct = [z for z, r in zone_ratios.items() if r >= 0.10]
        zones_ge_05pct = [z for z, r in zone_ratios.items() if r >= 0.05]
        
        records.append({
            "image": patient_id,
            "dataset": row["dataset"],
            "nodule_id": row["nodule_id"],
            "bbox_area": total_bbox_pixels,
            "assigned_zone_id": assigned_z,
            "z1_ratio": zone_ratios[1],
            "z2_ratio": zone_ratios[2],
            "z3_ratio": zone_ratios[3],
            "z4_ratio": zone_ratios[4],
            "z5_ratio": zone_ratios[5],
            "outside_ratio": round(outside_pixels / total_bbox_pixels, 4),
            "zones_ge_20pct": zones_ge_20pct,
            "n_zones_ge_20pct": len(zones_ge_20pct),
            "zones_ge_10pct": zones_ge_10pct,
            "n_zones_ge_10pct": len(zones_ge_10pct),
        })

overlap_df = pd.DataFrame(records)
overlap_out = Path("data/processed/zones/reports/nodule_pixel_overlap.csv")
overlap_df.to_csv(overlap_out, index=False)

print(f"\nAnaliz tamamlandı ({time.time()-t0:.2f}s). Sonuçlar kaydedildi: {overlap_out}")

print("\n--- ÖZET BULGULAR ---")
print("1. Bbox alanı en az %20 olan zon sayısı dağılımı (1476 nodül):")
print(overlap_df["n_zones_ge_20pct"].value_counts().sort_index())

print("\n2. Bbox alanı en az %10 olan zon sayısı dağılımı:")
print(overlap_df["n_zones_ge_10pct"].value_counts().sort_index())

# Multi-zone nodüller
multi_20 = overlap_df[overlap_df["n_zones_ge_20pct"] > 1]
print(f"\nBirden fazla zona >= %20 taşan nodül sayısı: {len(multi_20)} / {len(overlap_df)} (%{len(multi_20)/len(overlap_df)*100:.1f})")

# Hangi zonlar arası taşma en çok?
overlap_pairs = []
for _, r in multi_20.iterrows():
    zs = tuple(sorted(r["zones_ge_20pct"]))
    overlap_pairs.append(zs)
print("\nEn çok taşan zon çiftleri (>= %20 örtüşme):")
print(pd.Series(overlap_pairs).value_counts().head(10))

# Merkez ataması ile %20 zonları uyuşmayanlar
disagreements = []
for _, r in overlap_df.iterrows():
    assigned = r["assigned_zone_id"]
    if assigned not in r["zones_ge_20pct"] and assigned != 0:
        disagreements.append(r)
print(f"\nMerkezi bir zona atanmış ama o zonda %20'den az örtüşen nodül sayısı: {len(disagreements)}")
