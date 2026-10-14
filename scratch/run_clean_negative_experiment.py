"""Temiz Sağlıklı Negatif ve %20 BBox Örtüşme Çözümü — RBF-SVM Karşılaştırmalı Deneyi.

Bu betik:
1. Nodule pixel overlap CSV'sini okuyarak her hasta-zon için max_overlap ve dirty durumunu hesaplar.
2. 4 farklı stratejiyi aynı splitler ve aynı hiperparametre aralığıyla eğitip karşılaştırır:
   - Strateji 0 (Baseline): Mevcut durum (merkez nokta etiketi, rastgele zone_label==0 train negatifleri)
   - Strateji 1 (Overlap >= %20 Düzeltmesi): Taşma yapan zonlar pozitif (label=1)
   - Strateji 2 (Temiz Sağlıklı Negatifler): Train negatifleri SADECE patient_label==0 olan kontrol hastalarından seçilir
   - Strateji 3 (Tam Temiz Çözüm): Hem overlap >= %20 düzeltmesi hem de SADECE kontrol hastalarından train negatifleri
3. Her zon ve hasta düzeyinde Val/Test ROC-AUC, Sens, Spec, Karışıklık matrisi üretir.
"""

from __future__ import annotations
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc

PROJECT_ROOT = Path("c:/Users/Huseyincan/codes/rontgen_V2")
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "X_zone_features.npy"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "zone_splits.csv"
OVERLAP_PATH = PROJECT_ROOT / "data" / "processed" / "zones" / "reports" / "nodule_pixel_overlap.csv"

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec) if len(np.unique(y_true)) > 1 else 0.0
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / len(y_true)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 100)
    best_j = -1.0
    best_th = 0.5
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_th = float(th)
    return best_th

def main():
    print("Veriler yükleniyor...")
    X_all = np.load(FEATURES_PATH)
    splits_df = pd.read_csv(SPLITS_PATH)
    overlap_df = pd.read_csv(OVERLAP_PATH)

    # 1. Her hasta ve zon için max örtüşme oranını hesapla
    patient_zone_overlap = {}  # (patient_id, zone_id) -> max_ratio
    for _, row in overlap_df.iterrows():
        pid = row["image"]
        for zid in range(1, 6):
            ratio = float(row[f"z{zid}_ratio"])
            key = (pid, zid)
            if key not in patient_zone_overlap:
                patient_zone_overlap[key] = 0.0
            if ratio > patient_zone_overlap[key]:
                patient_zone_overlap[key] = ratio

    splits_df["max_overlap"] = splits_df.apply(
        lambda r: patient_zone_overlap.get((r["patient_id"], r["zone_id"]), 0.0),
        axis=1
    )

    # Örtüşme analizleri:
    # Eski etiketi 0 olup max_overlap >= 0.20 olan kaç zon var?
    dirty_negatives = splits_df[(splits_df["zone_label"] == 0) & (splits_df["max_overlap"] >= 0.20)]
    print(f"Eski etiketi 0 (negatif) olup gerçekte nodülün >= %20'sini barındıran zon sayısı: {len(dirty_negatives)}")
    print(dirty_negatives["zone_id"].value_counts().sort_index().to_dict())

    # Belirsiz zonlar: 0 < max_overlap < 0.20
    ambiguous = splits_df[(splits_df["zone_label"] == 0) & (splits_df["max_overlap"] > 0.0) & (splits_df["max_overlap"] < 0.20)]
    print(f"Belirsiz (0 < overlap < %20) zon sayısı: {len(ambiguous)}")

    strategies = [
        {"name": "Strat 0: Orijinal Baseline", "relabel_20": False, "clean_controls_only": False, "drop_ambiguous": False},
        {"name": "Strat 1: Overlap >= %20 Düzeltmesi", "relabel_20": True, "clean_controls_only": False, "drop_ambiguous": False},
        {"name": "Strat 2: Sadece Temiz Kontrol Negatifleri", "relabel_20": False, "clean_controls_only": True, "drop_ambiguous": False},
        {"name": "Strat 3: Tam Temiz Çözüm (%20 Relabel + Temiz Kontrol)", "relabel_20": True, "clean_controls_only": True, "drop_ambiguous": True},
    ]

    param_grid = {"C": [1.0, 10.0, 50.0, 100.0], "gamma": ["scale", "auto", 0.01, 0.001]}

    results = []

    for strat in strategies:
        print("\n" + "=" * 80)
        print(f"ÇALIŞTIRILIYOR: {strat['name']}")
        print("=" * 80)

        df = splits_df.copy()

        # Etiket güncelleme
        if strat["relabel_20"]:
            df.loc[df["max_overlap"] >= 0.20, "zone_label"] = 1

        # Train kümesinde negatif seçimi
        rng = np.random.RandomState(42)
        df["train_selected"] = False

        for zid in range(1, 6):
            tr_z = df[(df["split"] == "train") & (df["zone_id"] == zid)]
            pos_idx = tr_z[tr_z["zone_label"] == 1].index

            if strat["clean_controls_only"]:
                # Sadece gerçek sağlıklı kontrol hastaları (patient_label == 0)
                neg_candidates = tr_z[(tr_z["zone_label"] == 0) & (tr_z["patient_label"] == 0)].index
            else:
                neg_candidates = tr_z[tr_z["zone_label"] == 0].index

            if strat["drop_ambiguous"]:
                # Belirsiz zonları aday negatiflerden çıkar
                neg_candidates = neg_candidates.difference(df[df["max_overlap"] > 0.0].index)

            n_pos = len(pos_idx)
            sel_neg = rng.choice(neg_candidates, size=n_pos, replace=False)

            df.loc[pos_idx, "train_selected"] = True
            df.loc[sel_neg, "train_selected"] = True

        # Modelleri eğit ve değerlendir
        zone_models = {}
        zone_thresholds = {}
        val_zone_probs = {pid: {} for pid in df[df["split"] == "val"]["patient_id"].unique()}
        test_zone_probs = {pid: {} for pid in df[df["split"] == "test"]["patient_id"].unique()}

        zone_metrics_summary = {}

        for zid in range(1, 6):
            tr_mask = (df["split"] == "train") & (df["zone_id"] == zid) & (df["train_selected"])
            val_mask = (df["split"] == "val") & (df["zone_id"] == zid)
            test_mask = (df["split"] == "test") & (df["zone_id"] == zid)

            X_tr, y_tr = X_all[tr_mask.values], df.loc[tr_mask, "zone_label"].values
            X_vl, y_vl = X_all[val_mask.values], df.loc[val_mask, "zone_label"].values
            X_ts, y_ts = X_all[test_mask.values], df.loc[test_mask, "zone_label"].values

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_vl_sc = scaler.transform(X_vl)
            X_ts_sc = scaler.transform(X_ts)

            # Tuning
            best_auc = -1.0
            best_model = None
            best_params = {}
            for c in param_grid["C"]:
                for g in param_grid["gamma"]:
                    clf = SVC(kernel="rbf", C=c, gamma=g, probability=True, random_state=42, cache_size=1000)
                    clf.fit(X_tr_sc, y_tr)
                    vp = clf.predict_proba(X_vl_sc)[:, 1]
                    s = roc_auc_score(y_vl, vp)
                    if s > best_auc:
                        best_auc = s
                        best_model = clf
                        best_params = {"C": c, "gamma": g}

            val_p = best_model.predict_proba(X_vl_sc)[:, 1]
            test_p = best_model.predict_proba(X_ts_sc)[:, 1]

            z_th = find_optimal_threshold(y_vl, val_p)
            zone_thresholds[zid] = z_th

            val_m = compute_binary_metrics(y_vl, val_p, z_th)
            test_m = compute_binary_metrics(y_ts, test_p, z_th)

            zone_metrics_summary[f"zone_{zid}"] = {
                "best_params": best_params,
                "val_auc": val_m["roc_auc"],
                "test_auc": test_m["roc_auc"],
                "test_sens": test_m["sensitivity"],
                "test_spec": test_m["specificity"],
            }

            # Hasta düzeyinde olasılıkları sakla
            for pid, prob in zip(df.loc[val_mask, "patient_id"], val_p):
                val_zone_probs[pid][zid] = prob
            for pid, prob in zip(df.loc[test_mask, "patient_id"], test_p):
                test_zone_probs[pid][zid] = prob

        # Hasta düzeyinde Max-Pooling
        val_patients = df[df["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
        test_patients = df[df["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")

        y_val_pat = val_patients["patient_label"].values
        p_val_pat = np.array([max(val_zone_probs[pid].values()) for pid in val_patients.index])

        y_test_pat = test_patients["patient_label"].values
        p_test_pat = np.array([max(test_zone_probs[pid].values()) for pid in test_patients.index])

        patient_th = find_optimal_threshold(y_val_pat, p_val_pat)
        val_pat_m = compute_binary_metrics(y_val_pat, p_val_pat, patient_th)
        test_pat_m = compute_binary_metrics(y_test_pat, p_test_pat, patient_th)

        print(f"-> Hasta Val ROC-AUC:  {val_pat_m['roc_auc']:.4f}")
        print(f"-> Hasta Test ROC-AUC: {test_pat_m['roc_auc']:.4f} (PR-AUC: {test_pat_m['pr_auc']:.4f})")
        print(f"-> Test Duyarlılık:    %{test_pat_m['sensitivity']*100:.2f} | Özgüllük: %{test_pat_m['specificity']*100:.2f} | Acc: %{test_pat_m['accuracy']*100:.2f}")
        print(f"-> Karışıklık: TP={test_pat_m['tp']}, FP={test_pat_m['fp']}, FN={test_pat_m['fn']}, TN={test_pat_m['tn']}")

        for zid in range(1, 6):
            zm = zone_metrics_summary[f"zone_{zid}"]
            print(f"   Zon {zid}: Val AUC={zm['val_auc']:.4f} | Test AUC={zm['test_auc']:.4f} | {zm['best_params']}")

        results.append({
            "strategy": strat["name"],
            "val_auc": val_pat_m["roc_auc"],
            "test_auc": test_pat_m["roc_auc"],
            "test_pr_auc": test_pat_m["pr_auc"],
            "test_acc": test_pat_m["accuracy"],
            "test_sens": test_pat_m["sensitivity"],
            "test_spec": test_pat_m["specificity"],
            "threshold": patient_th,
            "tp": test_pat_m["tp"],
            "fp": test_pat_m["fp"],
            "fn": test_pat_m["fn"],
            "tn": test_pat_m["tn"],
            "zone_metrics": zone_metrics_summary
        })

    with open(PROJECT_ROOT / "data" / "processed" / "zones" / "reports" / "clean_negative_experiments_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRapor kaydedildi: data/processed/zones/reports/clean_negative_experiments_report.json")

if __name__ == "__main__":
    main()
