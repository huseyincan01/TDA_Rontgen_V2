"""Test Kümesi Üzerinde Bootstrap ile %95 Güven Aralıkları (Confidence Intervals) Hesaplayıcı.

Bu betik:
1. RBF-SVM, k-NN ve Küçük CNN test tahminlerini yükler (734 hasta).
2. N=2000 bootstrap yinelemesi ile ROC-AUC, PR-AUC, Duyarlılık, Özgüllük, F1 ve Doğruluk
   metrikleri için %95 Güven Aralığı ([2.5. persentil, 97.5. persentil]) hesaplar.
3. Sonuçları artifacts/runs/bootstrap_confidence_intervals.json dosyasına kaydeder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tek bir örneklem için tüm metrikleri hesaplar."""
    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(np.sum((recall_curve[:-1] - recall_curve[1:]) * precision_curve[:-1]))

    # Karışıklık öğeleri
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "accuracy": acc,
    }


def run_bootstrap(
    df: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verilen tahmin tablosu üzerinde bootstrap analizi yapar."""
    rng = np.random.RandomState(seed)
    n = len(df)
    y_true_orig = df["patient_label"].values
    y_prob_orig = df["predicted_patient_prob"].values
    y_pred_orig = df["predicted_patient_label"].values

    base_metrics = compute_metrics(y_true_orig, y_prob_orig, y_pred_orig)

    metric_keys = list(base_metrics.keys())
    boot_distributions = {k: [] for k in metric_keys}

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        # En az bir pozitif ve bir negatif olmalı
        if len(np.unique(y_true_orig[indices])) < 2:
            continue

        b_metrics = compute_metrics(
            y_true_orig[indices],
            y_prob_orig[indices],
            y_pred_orig[indices],
        )
        for k in metric_keys:
            if not np.isnan(b_metrics[k]):
                boot_distributions[k].append(b_metrics[k])

    results = {}
    for k in metric_keys:
        vals = np.array(boot_distributions[k])
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))

        results[k] = {
            "point_estimate": round(base_metrics[k], 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_formatted": f"{base_metrics[k]:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]",
            "ci_percent_formatted": f"%{base_metrics[k]*100:.2f} [%{ci_lower*100:.2f}, %{ci_upper*100:.2f}]",
            "bootstrap_mean": round(mean_val, 4),
            "bootstrap_std": round(std_val, 4),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap %95 Güven Aralığı Hesaplama")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    models = {
        "TDA Küçük CNN": runs_dir / "test_patient_predictions_cnn.csv",
        "Ham Görüntü CNN": runs_dir / "test_patient_predictions_raw_cnn.csv",
        "RBF-SVM": runs_dir / "test_patient_predictions.csv",
        "k-NN": runs_dir / "test_patient_predictions_knn.csv",
        "Landscape Küçük CNN": runs_dir / "test_patient_predictions_landscape_cnn.csv",
        "Landscape RBF-SVM": runs_dir / "test_patient_predictions_landscape_svm.csv",
    }

    all_results = {}

    print("=" * 110)
    print(f"NODE21 TDA — TEST KÜMESİ %95 BOOTSTRAP GÜVEN ARALIKLARI (N={args.n_bootstraps} yineleme)")
    print("=" * 110)

    for name, path in models.items():
        if not path.exists():
            print(f"Uyarı: {path} bulunamadı, atlanıyor.")
            continue

        df = pd.read_csv(path)
        boot_res = run_bootstrap(df, n_bootstraps=args.n_bootstraps, seed=args.seed)
        all_results[name] = boot_res

    # Konsol tablosu
    print(f"{'Model':<12} | {'ROC-AUC [%95 CI]':<26} | {'PR-AUC [%95 CI]':<26} | {'Duyarlılık [%95 CI]':<26} | {'Özgüllük [%95 CI]':<26}")
    print("-" * 110)
    for name, res in all_results.items():
        roc_str = res["roc_auc"]["ci_formatted"]
        pr_str = res["pr_auc"]["ci_formatted"]
        sens_str = res["sensitivity"]["ci_percent_formatted"]
        spec_str = res["specificity"]["ci_percent_formatted"]
        print(f"{name:<12} | {roc_str:<26} | {pr_str:<26} | {sens_str:<26} | {spec_str:<26}")

    print("\n" + f"{'Model':<12} | {'F1-Skoru [%95 CI]':<26} | {'Doğruluk [%95 CI]':<26} | {'Hassasiyet [%95 CI]':<26}")
    print("-" * 85)
    for name, res in all_results.items():
        f1_str = res["f1"]["ci_formatted"]
        acc_str = res["accuracy"]["ci_percent_formatted"]
        prec_str = res["precision"]["ci_percent_formatted"]
        print(f"{name:<12} | {f1_str:<26} | {acc_str:<26} | {prec_str:<26}")

    # JSON çıktısı
    out_path = runs_dir / "bootstrap_confidence_intervals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("=" * 110)
    print(f"Rapor Kaydedildi: {out_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
