import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load features, splits, and overlap data
X_all = np.load("data/processed/features/X_zone_features.npy")
splits_df = pd.read_csv("data/processed/splits/zone_splits.csv")
overlap_df = pd.read_csv("data/processed/zones/reports/nodule_pixel_overlap.csv")

# Create max overlap mapping
patient_zone_max_overlap = {}
for _, row in overlap_df.iterrows():
    pid = row["image"]
    for z in range(1, 6):
        r = row[f"z{z}_ratio"]
        key = (pid, z)
        patient_zone_max_overlap[key] = max(patient_zone_max_overlap.get(key, 0.0), r)

splits_df["max_overlap"] = splits_df.apply(
    lambda r: patient_zone_max_overlap.get((r["patient_id"], r["zone_id"]), 0.0), axis=1
)

# 1. KURAL: Bbox örtüşmesi >= 0.20 olan zonları pozitif yap
# Eski etiket 1 olanlar da zaten pozitif kalır
splits_df["clean_zone_label"] = splits_df["zone_label"]
updated_pos = (splits_df["max_overlap"] >= 0.20) & (splits_df["clean_zone_label"] == 0)
splits_df.loc[updated_pos, "clean_zone_label"] = 1
print(f"Pozitif yapılan sınır zonu sayısı (>= %20 örtüşme): {updated_pos.sum()}")

# 2. KURAL: Temiz Sağlıklı Negatifler (patient_label == 0 olan kontrol hastaları)
# Belirsiz zonlar: 0 < max_overlap < 0.20 olanlar eğitim negatif havuzundan ÇIKARILIR
ambiguous_mask = (splits_df["max_overlap"] > 0) & (splits_df["max_overlap"] < 0.20)
print(f"Eğitimden çıkarılacak belirsiz sınır zonu sayısı: {ambiguous_mask.sum()}")

rng = np.random.RandomState(42)

# Yeni dengeli train seçimi:
# Her zon için pozitifler (clean_zone_label == 1)
# Negatifler: SADECE patient_label == 0 olan temiz kontrol hastalarının o zonundan rastgele seçilir
splits_df["is_clean_train_balanced"] = False

for z in range(1, 6):
    tr_z = splits_df[(splits_df["split"] == "train") & (splits_df["zone_id"] == z)]
    pos_idx = tr_z[tr_z["clean_zone_label"] == 1].index
    
    # Temiz negatif havuzu: patient_label == 0 ve max_overlap == 0
    clean_neg_idx = tr_z[(tr_z["patient_label"] == 0) & (tr_z["max_overlap"] == 0)].index
    
    n_pos = len(pos_idx)
    sel_neg_idx = rng.choice(clean_neg_idx, size=n_pos, replace=False)
    
    splits_df.loc[pos_idx, "is_clean_train_balanced"] = True
    splits_df.loc[sel_neg_idx, "is_clean_train_balanced"] = True
    print(f"Zon {z}: Yeni Train Pos={n_pos}, Temiz Neg={len(sel_neg_idx)}, Toplam={n_pos*2}")

# SVM'yi bu temiz veride eğitip değerlendirelim:
print("\n" + "=" * 60)
print("TEMİZ VERİ ÜZERİNDE RBF-SVM ÇALIŞTIRILIYOR")
print("=" * 60)

zone_test_results = {}
val_preds = []
test_preds = []

# SVM parametreleri (her zon için önceki en iyi C/gamma parametreleri veya hızlı grid)
best_params_per_zone = {
    1: {"C": 50.0, "gamma": 0.01},
    2: {"C": 50.0, "gamma": "scale"},
    3: {"C": 100.0, "gamma": "auto"},
    4: {"C": 100.0, "gamma": 0.001},
    5: {"C": 10.0, "gamma": 0.001},
}

for z in range(1, 6):
    tr_mask = (splits_df["split"] == "train") & (splits_df["zone_id"] == z) & (splits_df["is_clean_train_balanced"])
    vl_mask = (splits_df["split"] == "val") & (splits_df["zone_id"] == z)
    ts_mask = (splits_df["split"] == "test") & (splits_df["zone_id"] == z)
    
    X_tr = X_all[splits_df[tr_mask]["vector_index"].values]
    y_tr = splits_df.loc[tr_mask, "clean_zone_label"].values
    
    X_vl = X_all[splits_df[vl_mask]["vector_index"].values]
    y_vl = splits_df.loc[vl_mask, "clean_zone_label"].values
    
    X_ts = X_all[splits_df[ts_mask]["vector_index"].values]
    y_ts = splits_df.loc[ts_mask, "clean_zone_label"].values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_ts_s = scaler.transform(X_ts)
    
    p = best_params_per_zone[z]
    clf = SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], probability=True, random_state=42)
    clf.fit(X_tr_s, y_tr)
    
    v_probs = clf.predict_proba(X_vl_s)[:, 1]
    t_probs = clf.predict_proba(X_ts_s)[:, 1]
    
    v_auc = roc_auc_score(y_vl, v_probs)
    t_auc = roc_auc_score(y_ts, t_probs)
    
    print(f"Zon {z}: Val ROC-AUC = {v_auc:.4f} | TEST ROC-AUC = {t_auc:.4f}")
    
    for pid, prob, yt in zip(splits_df.loc[vl_mask, "patient_id"], v_probs, y_vl):
        val_preds.append({"patient_id": pid, "zone_id": z, "prob": prob, "true": yt})
    for pid, prob, yt in zip(splits_df.loc[ts_mask, "patient_id"], t_probs, y_ts):
        test_preds.append({"patient_id": pid, "zone_id": z, "prob": prob, "true": yt})

# Hasta Düzeyinde Max Pooling
v_df = pd.DataFrame(val_preds)
v_meta = splits_df[splits_df["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
v_pat_probs = v_df.groupby("patient_id")["prob"].max()
v_pat_true = v_meta.loc[v_pat_probs.index, "patient_label"].values

t_df = pd.DataFrame(test_preds)
t_meta = splits_df[splits_df["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")
t_pat_probs = t_df.groupby("patient_id")["prob"].max()
t_pat_true = t_meta.loc[t_pat_probs.index, "patient_label"].values

val_patient_auc = roc_auc_score(v_pat_true, v_pat_probs)
test_patient_auc = roc_auc_score(t_pat_true, t_pat_probs)

print("\n" + "=" * 60)
print(f"HASTA DÜZEYİNDE VAL ROC-AUC:  {val_patient_auc:.4f}")
print(f"HASTA DÜZEYİNDE TEST ROC-AUC: {test_patient_auc:.4f}")
print("=" * 60)
