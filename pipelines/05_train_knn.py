"""5-Zon k-NN (k-En Yakın Komşu) Modelleri Eğitim, Validasyon Ayarlama ve Test Değerlendirme Boru Hattı.

Bu betik:
1. Her zon için dengelenmiş train kümesinde (1:1 pos/neg) StandardScaler fit eder.
2. Her zon için k in [1, 3, 5, 7, 11] değerlerini Validation ROC-AUC ile dener ve en iyi k'yı seçer.
3. Validation kümesinde Youden's J indeksi ile zon karar eşiğini (threshold) belirler.
4. Dokunulmamış Test kümesinde dondurulmuş k ve karar eşikleriyle nihai zon ve hasta başarımını değerlendirir.
5. 5 zon olasılığının maksimumu (Max Pooling) ile hasta skorunu hesaplar.
6. Sonuçları artifacts/ altına kaydeder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold
from node21_tda.models.patient_predictor import ZoneModelBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-Zon k-NN eğitim boru hattı.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features_v2" / "X_zone_features.npy",
        help="X_zone_features.npy dosya yolu.",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv",
        help="zone_splits.csv dosya yolu.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models.yaml",
        help="models.yaml yapılandırma dosya yolu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Model ve raporların kaydedileceği dizin.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_and_tune_zone_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_candidates: List[int],
    weights: str = "uniform",
) -> Tuple[KNeighborsClassifier, Dict[str, Any], float, Dict[int, float]]:
    """Validation ROC-AUC üzerinde en iyi k hiperparametresini arar."""
    best_score = -1.0
    best_k = k_candidates[0]
    best_model = None
    k_scores = {}

    for k in k_candidates:
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
        clf.fit(X_train, y_train)

        # predict_proba pozitif sınıf olasılıkları
        val_probs = clf.predict_proba(X_val)
        # binary sınıfta 2 sütun (0 ve 1)
        if val_probs.shape[1] == 2:
            val_pos_probs = val_probs[:, 1]
        else:
            val_pos_probs = val_probs[:, 0]

        val_metrics = compute_binary_metrics(y_val, val_pos_probs)
        score = val_metrics["roc_auc"]
        k_scores[k] = round(score, 4)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = clf

    assert best_model is not None
    best_params = {"n_neighbors": best_k, "weights": weights}
    return best_model, best_params, best_score, k_scores


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 80)
    print("NODE21 TDA — 5-ZON k-NN EĞİTİM VE TEST DEĞERLENDİRME")
    print("=" * 80)

    cfg = load_yaml(args.config_path).get("zone_knn", {})
    k_candidates = cfg.get("k_neighbors", [1, 3, 5, 7, 11])
    weights = cfg.get("weights", "uniform")
    patient_agg = cfg.get("patient_aggregation", "max")

    print(f"Denenen k değerleri: {k_candidates}")
    print(f"Ağırlık fonksiyonu:   {weights}")
    print(f"Hasta birleştirme:    {patient_agg.upper()}")

    # Verileri yükle
    print(f"\nÖzellikler yükleniyor: {args.features_path}")
    X_all = np.load(args.features_path)
    print(f"X boyutu: {X_all.shape}")

    print(f"Zon splitleri yükleniyor: {args.splits_path}")
    splits_df = pd.read_csv(args.splits_path)

    bundle = ZoneModelBundle(aggregation=patient_agg)
    zone_results = {}

    test_predictions_list = []
    val_predictions_list = []

    print("\n" + "-" * 80)
    print("ZON BAZLI k-NN EĞİTİMİ VE k SEÇİMİ BAŞLIYOR")
    print("-" * 80)

    for zone_id in range(1, 6):
        z_start = time.time()
        zone_name = splits_df[splits_df["zone_id"] == zone_id]["zone_name"].iloc[0]

        # 1. Filtreleme
        # Train: Dengelenmiş 1:1 temiz alt küme
        train_mask = (
            (splits_df["split"] == "train")
            & (splits_df["zone_id"] == zone_id)
            & (splits_df["is_train_balanced_sample"])
        )
        # Validation ve Test zon değerlendirmesi: YALNIZCA is_zone_metric_sample == True
        val_metric_mask = (
            (splits_df["split"] == "val")
            & (splits_df["zone_id"] == zone_id)
            & (splits_df["is_zone_metric_sample"])
        )
        test_metric_mask = (
            (splits_df["split"] == "test")
            & (splits_df["zone_id"] == zone_id)
            & (splits_df["is_zone_metric_sample"])
        )
        # Hasta düzeyinde Max-Pooling birleştirmesi için (tüm zonlar)
        val_all_mask = (splits_df["split"] == "val") & (splits_df["zone_id"] == zone_id)
        test_all_mask = (splits_df["split"] == "test") & (splits_df["zone_id"] == zone_id)

        train_indices = splits_df[train_mask]["vector_index"].values
        val_metric_indices = splits_df[val_metric_mask]["vector_index"].values
        test_metric_indices = splits_df[test_metric_mask]["vector_index"].values
        val_all_indices = splits_df[val_all_mask]["vector_index"].values
        test_all_indices = splits_df[test_all_mask]["vector_index"].values

        X_tr = X_all[train_indices]
        y_tr = splits_df.loc[train_mask, "zone_label"].values

        X_vl_metric = X_all[val_metric_indices]
        y_vl_metric = splits_df.loc[val_metric_mask, "zone_label"].values

        X_ts_metric = X_all[test_metric_indices]
        y_ts_metric = splits_df.loc[test_metric_mask, "zone_label"].values

        X_vl_all = X_all[val_all_indices]
        X_ts_all = X_all[test_all_indices]

        # 2. StandardScaler fit YALNIZCA TRAIN verisine
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_vl_metric_scaled = scaler.transform(X_vl_metric)
        X_ts_metric_scaled = scaler.transform(X_ts_metric)
        X_vl_all_scaled = scaler.transform(X_vl_all)
        X_ts_all_scaled = scaler.transform(X_ts_all)

        # 3. Validation üzerinde en iyi k seçimi (yalnızca temiz metric satırları)
        best_clf, best_params, best_val_auc, k_scores = train_and_tune_zone_knn(
            X_tr_scaled, y_tr, X_vl_metric_scaled, y_vl_metric, k_candidates, weights=weights
        )

        # 4. Validation üzerinde karar eşiği belirleme
        val_metric_probs = best_clf.predict_proba(X_vl_metric_scaled)[:, 1]
        opt_thresh, j_score = find_optimal_threshold(y_vl_metric, val_metric_probs, method="youden")

        val_metrics = compute_binary_metrics(y_vl_metric, val_metric_probs, threshold=opt_thresh)

        # 5. Dondurulmuş eşikle Test kümesi değerlendirmesi
        test_metric_probs = best_clf.predict_proba(X_ts_metric_scaled)[:, 1]
        test_metrics = compute_binary_metrics(y_ts_metric, test_metric_probs, threshold=opt_thresh)

        # Model paketine kaydet
        bundle.set_zone_model(
            zone_id=zone_id,
            model=best_clf,
            scaler=scaler,
            threshold=opt_thresh,
            params=best_params,
        )

        # Hasta tahmini için tüm zonların olasılıkları
        val_all_probs = best_clf.predict_proba(X_vl_all_scaled)[:, 1]
        test_all_probs = best_clf.predict_proba(X_ts_all_scaled)[:, 1]

        for pid, prob in zip(splits_df.loc[val_all_mask, "patient_id"], val_all_probs):
            val_predictions_list.append({
                "patient_id": pid, "zone_id": zone_id, "zone_prob": prob
            })

        for pid, prob in zip(splits_df.loc[test_all_mask, "patient_id"], test_all_probs):
            test_predictions_list.append({
                "patient_id": pid, "zone_id": zone_id, "zone_prob": prob
            })

        elapsed = time.time() - z_start
        zone_results[f"zone_{zone_id}"] = {
            "zone_name": zone_name,
            "train_samples": len(y_tr),
            "k_scores_val_auc": k_scores,
            "best_params": best_params,
            "optimal_threshold": opt_thresh,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

        print(
            f"Zon {zone_id} ({zone_name}): "
            f"Train={len(y_tr)} | En İyi k={best_params['n_neighbors']} (k_skorları={k_scores}) | Eşik={opt_thresh:.4f}\n"
            f"  -> Validation: ROC-AUC={val_metrics['roc_auc']:.4f}, Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}\n"
            f"  -> TEST:       ROC-AUC={test_metrics['roc_auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f} ({elapsed:.1f}s)"
        )

    # 6. Hasta Düzeyinde Değerlendirme (Max Pooling)
    print("\n" + "-" * 80)
    print(f"HASTA DÜZEYİNDE BİRLEŞTİRME VE DEĞERLENDİRME (Yöntem: {patient_agg.upper()})")
    print("-" * 80)

    # Validation hasta tahmini
    val_pred_df = pd.DataFrame(val_predictions_list)
    val_patient_meta = splits_df[splits_df["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    val_patient_probs = val_pred_df.groupby("patient_id")["zone_prob"].max()
    val_patient_true = val_patient_meta.loc[val_patient_probs.index, "patient_label"].values

    patient_thresh, _ = find_optimal_threshold(val_patient_true, val_patient_probs.values, method="youden")
    bundle.patient_threshold = patient_thresh

    patient_val_metrics = compute_binary_metrics(val_patient_true, val_patient_probs.values, threshold=patient_thresh)

    # Test hasta tahmini
    test_pred_df = pd.DataFrame(test_predictions_list)
    test_patient_meta = splits_df[splits_df["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")
    test_patient_probs = test_pred_df.groupby("patient_id")["zone_prob"].max()
    test_patient_true = test_patient_meta.loc[test_patient_probs.index, "patient_label"].values

    patient_test_metrics = compute_binary_metrics(test_patient_true, test_patient_probs.values, threshold=patient_thresh)

    print(f"Belirlenen Hasta Karar Eşiği (Validation'dan): {patient_thresh:.4f}")
    print(
        f"Validation Hasta Başarımı: ROC-AUC={patient_val_metrics['roc_auc']:.4f}, "
        f"Sens={patient_val_metrics['sensitivity']:.4f}, Spec={patient_val_metrics['specificity']:.4f}, F1={patient_val_metrics['f1']:.4f}"
    )
    print(
        f"TEST HASTA BAŞARIMI:       ROC-AUC={patient_test_metrics['roc_auc']:.4f}, "
        f"Sens={patient_test_metrics['sensitivity']:.4f}, Spec={patient_test_metrics['specificity']:.4f}, F1={patient_test_metrics['f1']:.4f}"
    )
    print(f"Test Hasta Karışıklık Matrisi: TP={patient_test_metrics['tp']}, FP={patient_test_metrics['fp']}, TN={patient_test_metrics['tn']}, FN={patient_test_metrics['fn']}")

    # 7. Çıktıları Kaydetme
    models_dir = args.output_dir / "models" / "zone_knn"
    runs_dir = args.output_dir / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = models_dir / "zone_knn_bundle.joblib"
    bundle.save(bundle_path)

    full_report = {
        "model_type": "zone_knn",
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "k_candidates": k_candidates,
        "weights": weights,
        "patient_aggregation": patient_agg,
        "patient_threshold": round(patient_thresh, 4),
        "patient_val_metrics": patient_val_metrics,
        "patient_test_metrics": patient_test_metrics,
        "zone_results": zone_results,
    }

    report_path = runs_dir / "zone_knn_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    test_pred_out_path = runs_dir / "test_patient_predictions_knn.csv"
    test_summary_df = pd.DataFrame({
        "patient_id": test_patient_probs.index,
        "patient_label": test_patient_true,
        "predicted_patient_prob": test_patient_probs.values.round(4),
        "predicted_patient_label": (test_patient_probs.values >= patient_thresh).astype(int),
    })
    test_summary_df.to_csv(test_pred_out_path, index=False)

    print("\n" + "=" * 80)
    print("k-NN EĞİTİM VE TEST BAŞARIYLA TAMAMLANDI")
    print("=" * 80)
    print(f"Model Paketi:    {bundle_path}")
    print(f"Rapor Dosyası:   {report_path}")
    print(f"Test Tahminleri: {test_pred_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
