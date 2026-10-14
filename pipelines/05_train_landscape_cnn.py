"""5-Zon Persistence Landscapes Küçük CNN Modelleri Eğitim ve Değerlendirme Boru Hattı.

Bu betik:
1. `features_landscape_v2/X_zone_features.npy` (800 boyutlu PL vektörleri) kullanır.
2. Her zon için 1:1 dengelenmiş train kümesinde StandardScaler fit eder.
3. Vektörleri (2, 20, 20) tensörüne dönüştürerek SmallZoneCNN mimarisiyle eğitir.
4. Validation ROC-AUC ile en iyi epoch ağırlıklarını kilitler.
5. Validation kümesinde Youden's J indeksi ile zon ve hasta karar eşiğini belirler.
6. Dokunulmamış Test kümesinde (734 hasta) dondurulmuş modeller ve eşiklerle nihai başarım değerlendirir.
7. Çıktıları `artifacts/models/zone_landscape_cnn/` ve `artifacts/runs/` altına kaydeder.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-Zon Persistence Landscapes Küçük CNN Eğitim Boru Hattı")
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
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu.")
    return parser.parse_args()


def train_zone_cnn(
    model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val: np.ndarray,
    max_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, float, int]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    X_val_dev = X_val_t.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_dev)
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
            v_metrics = compute_binary_metrics(y_val, v_probs)
            val_auc = v_metrics["roc_auc"]

            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch

    model.load_state_dict(best_weights)
    return model, best_auc, best_epoch


def main() -> None:
    args = parse_args()
    start_time = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("5-ZON PERSISTENCE LANDSCAPES (PL) — KÜÇÜK CNN EĞİTİM VE TEST")
    print("=" * 80)

    dropout = 0.3
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 32
    max_epochs = 40
    device = torch.device("cpu")

    print(f"Özellikler Yükleniyor: {args.features_path}")
    X_all = np.load(args.features_path)
    print(f"X Şekli: {X_all.shape}")

    print(f"Splitler Yükleniyor:    {args.splits_path}")
    splits_df = pd.read_csv(args.splits_path)

    zone_models = {}
    zone_scalers = {}
    zone_thresholds = {}
    zone_results = {}
    val_predictions_list = []
    test_predictions_list = []

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

        # Tensöre dönüştür
        X_tr_img = torch.tensor(vector_to_2channel_image(X_tr_sc), dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        X_val_img = torch.tensor(vector_to_2channel_image(X_val_sc), dtype=torch.float32)
        X_test_img = torch.tensor(vector_to_2channel_image(X_test_sc), dtype=torch.float32)

        model = SmallZoneCNN(in_channels=2, dropout=dropout)
        trained_model, best_val_auc, best_epoch = train_zone_cnn(
            model=model,
            X_train_t=X_tr_img,
            y_train_t=y_tr_t,
            X_val_t=X_val_img,
            y_val=y_val_m,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        trained_model.eval()
        with torch.no_grad():
            val_probs_m = trained_model.predict_proba(X_val_img.to(device)).cpu().numpy()
            opt_thresh, _ = find_optimal_threshold(y_val_m, val_probs_m)
            val_metrics = compute_binary_metrics(y_val_m, val_probs_m, threshold=opt_thresh)

            test_probs_m = trained_model.predict_proba(X_test_img.to(device)).cpu().numpy()
            test_metrics = compute_binary_metrics(y_test_m, test_probs_m, threshold=opt_thresh)

        zone_models[zone_id] = trained_model
        zone_scalers[zone_id] = scaler
        zone_thresholds[zone_id] = opt_thresh

        zone_results[f"zone_{zone_id}"] = {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "best_epoch": best_epoch,
            "optimal_threshold": round(opt_thresh, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "elapsed_seconds": round(time.time() - z_start, 2),
        }

        print(
            f"[Zon {zone_id} - {zone_name}] "
            f"BestEpoch={best_epoch}, Eşik={opt_thresh:.4f} | "
            f"Val AUC={val_metrics['roc_auc']:.4f} | "
            f"Test AUC={test_metrics['roc_auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f}"
        )

        # Hasta tahmini için tüm val/test zonlarını dönüştür
        val_all_mask = (splits_df["split"] == "val") & (splits_df["zone_id"] == zone_id)
        val_pids = splits_df.loc[val_all_mask, "patient_id"].values
        val_all_sc = scaler.transform(X_all[val_all_mask])
        val_all_img = torch.tensor(vector_to_2channel_image(val_all_sc), dtype=torch.float32)

        with torch.no_grad():
            val_all_probs = trained_model.predict_proba(val_all_img.to(device)).cpu().numpy()

        val_df_z = pd.DataFrame({
            "patient_id": val_pids,
            f"prob_zone_{zone_id}": val_all_probs,
        }).set_index("patient_id")
        val_predictions_list.append(val_df_z)

        test_all_mask = (splits_df["split"] == "test") & (splits_df["zone_id"] == zone_id)
        test_pids = splits_df.loc[test_all_mask, "patient_id"].values
        test_all_sc = scaler.transform(X_all[test_all_mask])
        test_all_img = torch.tensor(vector_to_2channel_image(test_all_sc), dtype=torch.float32)

        with torch.no_grad():
            test_all_probs = trained_model.predict_proba(test_all_img.to(device)).cpu().numpy()

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
    models_dir = args.output_dir / "models" / "zone_landscape_cnn"
    runs_dir = args.output_dir / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bundle_dict = {
        "models": {z: zone_models[z].state_dict() for z in range(1, 6)},
        "scalers": {z: zone_scalers[z] for z in range(1, 6)},
        "thresholds": zone_thresholds,
        "patient_threshold": patient_thresh,
    }
    torch.save(bundle_dict, models_dir / "zone_landscape_cnn_bundle.pt")

    full_report = {
        "model_type": "zone_landscape_cnn",
        "representation": "persistence_landscape",
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "hyperparameters": {
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        },
        "patient_aggregation": "max",
        "patient_threshold": round(patient_thresh, 4),
        "patient_val_metrics": patient_val_metrics,
        "patient_test_metrics": patient_test_metrics,
        "zone_results": zone_results,
    }

    report_path = runs_dir / "zone_landscape_cnn_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    test_pred_out_path = runs_dir / "test_patient_predictions_landscape_cnn.csv"
    test_summary_df = pd.DataFrame({
        "patient_id": test_patient_probs.index,
        "patient_label": test_patient_true,
        "predicted_patient_prob": test_patient_probs.values.round(4),
        "predicted_patient_label": (test_patient_probs.values >= patient_thresh).astype(int),
    })
    test_summary_df.to_csv(test_pred_out_path, index=False)

    print("\n" + "=" * 80)
    print("LANDSCAPE KÜÇÜK CNN MODELİ BAŞARIYLA TAMAMLANDI VE KAYDEDİLDİ")
    print("=" * 80)
    print(f"Model:           {models_dir / 'zone_landscape_cnn_bundle.pt'}")
    print(f"Rapor:           {report_path}")
    print(f"Test Tahminleri: {test_pred_out_path}")


if __name__ == "__main__":
    main()
