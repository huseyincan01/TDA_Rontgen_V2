"""Sınıflandırma performans metrikleri ve eşik (threshold) optimizasyon modülü."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> Tuple[float, float]:
    """Validation kümesi üzerinde en uygun karar eşiğini bulur.
    
    Args:
        y_true: Gerçek ikili etiketler (0 veya 1).
        y_prob: Pozitif sınıf (1) olasılıkları [0, 1].
        method: 'youden' (Duyarlılık + Özgüllük - 1 maksimizasyonu) veya 'f1'.
        
    Returns:
        (optimal_threshold, best_score)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        # Güvenlik: eşik [0, 1] sınırlarında kalsın
        best_threshold = float(np.clip(best_threshold, 0.01, 0.99))
        return best_threshold, float(j_scores[best_idx])

    elif method == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        denom = precisions + recalls
        f1_scores = np.divide(
            2 * precisions * recalls,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        # precision_recall_curve'da thresholds dizisi 1 eleman kısadır
        if len(thresholds) > 0:
            best_idx = int(np.argmax(f1_scores[: len(thresholds)]))
            best_threshold = float(thresholds[best_idx])
            best_score = float(f1_scores[best_idx])
        else:
            best_threshold = 0.5
            best_score = 0.0
        return float(np.clip(best_threshold, 0.01, 0.99)), best_score

    else:
        raise ValueError(f"Bilinmeyen eşik yöntemi: {method}. 'youden' veya 'f1' kullanın.")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """İkili sınıflandırma için ROC-AUC, PR-AUC, Duyarlılık, Özgüllük ve hata matrisini hesaplar."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC (Average Precision)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    # trapezoidal integral for PR curve
    pr_auc = float(np.sum((recall_curve[:-1] - recall_curve[1:]) * precision_curve[:-1]))

    # Karışıklık matrisi
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_samples": len(y_true),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }
