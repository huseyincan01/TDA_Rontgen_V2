"""5-Zon Persistence Landscapes RBF-SVM Modelleri Eğitim ve Değerlendirme Boru Hattı.

Bu betik:
1. `features_landscape_v2/X_zone_features.npy` (800 boyutlu PL vektörleri) kullanır.
2. Her zon için 1:1 dengelenmiş train kümesinde StandardScaler fit eder.
3. Grid search ile Validation ROC-AUC skorunu maksimize eden (C, gamma) hiperparametrelerini seçer.
4. Validation kümesinde Youden's J indeksi ile zon ve hasta karar eşiğini kilitler.
5. Dokunulmamış Test kümesinde (734 hasta) dondurulmuş modeller ve eşiklerle nihai başarım değerlendirir.
6. Çıktıları `artifacts/models/zone_landscape_svm/` ve `artifacts/runs/` altına kaydeder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold
from node21_tda.models.patient_predictor import ZoneModelBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-Zon Persistence Landscapes RBF-SVM Eğitim Boru Hattı")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features_landscape_v2" / "X_zone_features.npy",
        help="Landscape X_zone_features.npy dosya yolu.",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv",
        help="zone_splits.csv dosya yolu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Çıktı dizini.",
    )
    return parser.parse_args()


def tune_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, List[Any]],
    random_state: int = 42,
) -> Tuple[SVC, Dict[str, Any], float]:
    best_score = -1.0
    best_params = {}
    best_model = None

    c_values = param_grid.get("C", [0.1, 1.0, 10.0, 100.0])
    gamma_values = param_grid.get("gamma", ["scale", "auto", 0.01, 0.001])

    for c in c_values:
        for g in gamma_values:
            clf = SVC(
                kernel="rbf",
                C=c,
                gamma=g,
                probability=True,
                random_state=random_state,
                cache_size=1000,
            )
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_metrics = compute_binary_metrics(y_val, val_probs)
            score = val_metrics["roc_auc"]

            if score > best_score:
                best_score = score
                best_params = {"C": c, "gamma": g}
                best_model = clf

    assert best_model is not None
    return best_model, best_params, best_score


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 80)
    print("5-ZON PERSISTENCE LANDSCAPES (PL) — RBF-SVM EĞİTİM VE TEST")
    print("=" * 80)

    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.001]}

    print(f"Özellikler Yükleniyor: {args.features_path}")
    X_all = np.load(args.features_path)
    print(f"X Şekli: {X_all.shape}")

    print(f"Splitler Yükleniyor:    {args.splits_path}")
    splits_df = pd.read_csv(args.splits_path)

    bundle = ZoneModelBundle(aggregation="max")
    zone_results = {}
    test_predictions_list = []
    val_predictions_list = []

    print("\n" + "-" * 80)
    print("ZON BAZLI MODELLER EĞİTİLİYOR...")
    print("-" * 80)

    for zone_id in range(1, 6):
        z_start = time.time()
        zone_name = splits_df[splits_df["zone_id"] == zone_id]["zone_name"].iloc[0]

        train_mask = (
            (splits_df["split"] == "train")
            & (splits_df["zone_id"] == zone_id)
            & (splits_df["is_train_balanced_sample"])
        )
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

        X_tr = X_all[train_mask]
        y_tr = splits_df.loc[train_mask, "zone_label"].values.astype(int)

        X_val_m = X_all[val_metric_mask]
        y_val_m = splits_df.loc[val_metric_mask, "zone_label"].values.astype(int)

        X_test_m = X_all[test_metric_mask]
        y_test_m = splits_df.loc[test_metric_mask, "zone_label"].values.astype(int)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val_m)
        X_test_sc = scaler.transform(X_test_m)

        best_clf, best_params, best_val_auc = tune_rbf_svm(
            X_tr_sc, y_tr, X_val_sc, y_val_m, param_grid=param_grid
        )

        val_probs_m = best_clf.predict_proba(X_val_sc)[:, 1]
        opt_thresh, max_j = find_optimal_threshold(y_val_m, val_probs_m)

        val_metrics = compute_binary_metrics(y_val_m, val_probs_m, threshold=opt_thresh)
        test_probs_m = best_clf.predict_proba(X_test_sc)[:, 1]
        test_metrics = compute_binary_metrics(y_test_m, test_probs_m, threshold=opt_thresh)

        bundle.set_zone_model(
            zone_id=zone_id,
            model=best_clf,
            scaler=scaler,
            threshold=opt_thresh,
            params=best_params,
        )

        zone_results[f"zone_{zone_id}"] = {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "best_params": best_params,
            "optimal_threshold": round(opt_thresh, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "elapsed_seconds": round(time.time() - z_start, 2),
        }

        print(
            f"[Zon {zone_id} - {zone_name}] "
            f"Param={best_params}, Eşik={opt_thresh:.4f} | "
            f"Val AUC={val_metrics['roc_auc']:.4f} | "
            f"Test AUC={test_metrics['roc_auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f}"
        )

        # Hasta tahmini için tüm val/test zonlarını dönüştür
        val_all_mask = (splits_df["split"] == "val") & (splits_df["zone_id"] == zone_id)
        val_pids = splits_df.loc[val_all_mask, "patient_id"].values
        val_all_sc = scaler.transform(X_all[val_all_mask])
        val_all_probs = best_clf.predict_proba(val_all_sc)[:, 1]

        val_df_z = pd.DataFrame({
            "patient_id": val_pids,
            f"prob_zone_{zone_id}": val_all_probs,
        }).set_index("patient_id")
        val_predictions_list.append(val_df_z)

        test_all_mask = (splits_df["split"] == "test") & (splits_df["zone_id"] == zone_id)
        test_pids = splits_df.loc[test_all_mask, "patient_id"].values
        test_all_sc = scaler.transform(X_all[test_all_mask])
        test_all_probs = best_clf.predict_proba(test_all_sc)[:, 1]

        test_df_z = pd.DataFrame({
            "patient_id": test_pids,
            f"prob_zone_{zone_id}": test_all_probs,
        }).set_index("patient_id")
        test_predictions_list.append(test_df_z)

    # Hasta Düzeyinde Max-Pooling
    print("\n" + "-" * 80)
    print("HASTA DÜZEYİNDE MAX-POOLING BİRLEŞTİRMESİ")
    print("-" * 80)

    val_pat_df = pd.concat(val_predictions_list, axis=1)
    val_patient_probs = val_pat_df.max(axis=1)

    pat_splits = splits_df[["patient_id", "patient_label", "split"]].drop_duplicates().set_index("patient_id")
    val_patient_true = pat_splits.loc[val_patient_probs.index, "patient_label"].values.astype(int)

    patient_thresh, _ = find_optimal_threshold(val_patient_true, val_patient_probs.values)
    patient_val_metrics = compute_binary_metrics(val_patient_true, val_patient_probs.values, threshold=patient_thresh)

    test_pat_df = pd.concat(test_predictions_list, axis=1)
    test_patient_probs = test_pat_df.max(axis=1)
    test_patient_true = pat_splits.loc[test_patient_probs.index, "patient_label"].values.astype(int)
    patient_test_metrics = compute_binary_metrics(test_patient_true, test_patient_probs.values, threshold=patient_thresh)

    print(f"Validation Hasta Karar Eşiği: {patient_thresh:.4f}")
    print(
        f"Validation: ROC-AUC={patient_val_metrics['roc_auc']:.4f}, PR-AUC={patient_val_metrics['pr_auc']:.4f}, "
        f"Sens={patient_val_metrics['sensitivity']:.4f}, Spec={patient_val_metrics['specificity']:.4f}, F1={patient_val_metrics['f1']:.4f}"
    )
    print(
        f"TEST:       ROC-AUC={patient_test_metrics['roc_auc']:.4f}, PR-AUC={patient_test_metrics['pr_auc']:.4f}, "
        f"Sens={patient_test_metrics['sensitivity']:.4f}, Spec={patient_test_metrics['specificity']:.4f}, F1={patient_test_metrics['f1']:.4f}"
    )
    print(f"Karışıklık Matrisi: TP={patient_test_metrics['tp']}, FP={patient_test_metrics['fp']}, TN={patient_test_metrics['tn']}, FN={patient_test_metrics['fn']}")

    # Çıktıları Kaydet
    models_dir = args.output_dir / "models" / "zone_landscape_svm"
    runs_dir = args.output_dir / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = models_dir / "zone_landscape_svm_bundle.joblib"
    bundle.save(bundle_path)

    full_report = {
        "model_type": "zone_landscape_rbf_svm",
        "representation": "persistence_landscape",
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "param_grid": param_grid,
        "patient_aggregation": "max",
        "patient_threshold": round(patient_thresh, 4),
        "patient_val_metrics": patient_val_metrics,
        "patient_test_metrics": patient_test_metrics,
        "zone_results": zone_results,
    }

    report_path = runs_dir / "zone_landscape_svm_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    test_pred_out_path = runs_dir / "test_patient_predictions_landscape_svm.csv"
    test_summary_df = pd.DataFrame({
        "patient_id": test_patient_probs.index,
        "patient_label": test_patient_true,
        "predicted_patient_prob": test_patient_probs.values.round(4),
        "predicted_patient_label": (test_patient_probs.values >= patient_thresh).astype(int),
    })
    test_summary_df.to_csv(test_pred_out_path, index=False)

    print("\n" + "=" * 80)
    print("LANDSCAPE RBF-SVM MODELİ BAŞARIYLA TAMAMLANDI VE KAYDEDİLDİ")
    print("=" * 80)
    print(f"Model:           {bundle_path}")
    print(f"Rapor:           {report_path}")
    print(f"Test Tahminleri: {test_pred_out_path}")


if __name__ == "__main__":
    main()
