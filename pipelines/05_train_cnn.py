"""5-Zon Küçük CNN Modelleri Eğitim, Validasyon Doğrulama ve Test Değerlendirme Boru Hattı.

Bu betik:
1. Her zon için 800 boyutlu TDA vektörünü (2, 20, 20) tensörüne dönüştürür:
   - Kanal 0: H0 Persistence Image (20x20)
   - Kanal 1: H1 Persistence Image (20x20)
2. Yalnızca 1:1 dengelenmiş train kümesine StandardScaler fit eder.
3. Her zon için ayrı bir SmallZoneCNN eğitir; en iyi ağırlıkları Validation ROC-AUC'ye göre kaydeder.
4. Validation kümesinde Youden's J indeksi ile karar eşiklerini dondurur.
5. Dokunulmamış Test kümesinde dondurulmuş modeller ve eşiklerle nihai zon ve hasta başarımını değerlendirir.
6. Hasta skorunu beş zon olasılığının maksimumu (Max Pooling) ile hesaplar.
7. Sonuçları artifacts/ altına kaydeder ve RBF-SVM ile karşılaştırır.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="5-Zon Küçük CNN eğitim boru hattı.")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele tohum değeri.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    """SmallZoneCNN modelini eğitir ve en yüksek Validation ROC-AUC veren ağırlıkları döner."""
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

        # Validation değerlendirmesi
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
    print("NODE21 TDA — 5-ZON KÜÇÜK CNN (2x20x20) EĞİTİM VE TEST DEĞERLENDİRME")
    print("=" * 80)

    cfg = load_yaml(args.config_path).get("zone_cnn", {})
    dropout = cfg.get("dropout", 0.3)
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    batch_size = cfg.get("batch_size", 32)
    max_epochs = cfg.get("max_epochs", 40)
    patient_agg = cfg.get("patient_aggregation", "max")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hesaplama Cihazı:     {device}")
    print(f"Giriş Temsili:        2-Kanallı Persistence Image (H0: 20x20, H1: 20x20)")
    print(f"Eğitim Parametreleri: Epochs={max_epochs}, Batch={batch_size}, LR={lr}, WD={weight_decay}, Dropout={dropout}")
    print(f"Hasta Birleştirme:    {patient_agg.upper()}")

    # Verileri yükle
    print(f"\nÖzellikler yükleniyor: {args.features_path}")
    X_all = np.load(args.features_path)
    print(f"X boyutu: {X_all.shape}")

    print(f"Zon splitleri yükleniyor: {args.splits_path}")
    splits_df = pd.read_csv(args.splits_path)

    zone_models: Dict[int, nn.Module] = {}
    zone_scalers: Dict[int, StandardScaler] = {}
    zone_thresholds: Dict[int, float] = {}
    zone_results = {}

    test_predictions_list = []
    val_predictions_list = []

    print("\n" + "-" * 80)
    print("ZON BAZLI KÜÇÜK CNN EĞİTİMİ BAŞLIYOR")
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
        y_tr = splits_df.loc[train_mask, "zone_label"].values.astype(np.float32)

        X_vl_metric = X_all[val_metric_indices]
        y_vl_metric = splits_df.loc[val_metric_mask, "zone_label"].values.astype(np.float32)

        X_ts_metric = X_all[test_metric_indices]
        y_ts_metric = splits_df.loc[test_metric_mask, "zone_label"].values.astype(np.float32)

        X_vl_all = X_all[val_all_indices]
        X_ts_all = X_all[test_all_indices]

        # 2. StandardScaler fit YALNIZCA TRAIN verisine
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_vl_metric_scaled = scaler.transform(X_vl_metric)
        X_ts_metric_scaled = scaler.transform(X_ts_metric)
        X_vl_all_scaled = scaler.transform(X_vl_all)
        X_ts_all_scaled = scaler.transform(X_ts_all)

        # 3. (N, 2, 20, 20) tensör dönüşümü
        X_tr_t = torch.tensor(vector_to_2channel_image(X_tr_scaled), dtype=torch.float32)
        X_vl_metric_t = torch.tensor(vector_to_2channel_image(X_vl_metric_scaled), dtype=torch.float32)
        X_ts_metric_t = torch.tensor(vector_to_2channel_image(X_ts_metric_scaled), dtype=torch.float32)
        X_vl_all_t = torch.tensor(vector_to_2channel_image(X_vl_all_scaled), dtype=torch.float32)
        X_ts_all_t = torch.tensor(vector_to_2channel_image(X_ts_all_scaled), dtype=torch.float32)

        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        # 4. Modeli eğit ve Validation ROC-AUC'ye göre checkpoint al (yalnızca temiz metric satırları)
        model = SmallZoneCNN(in_channels=2, dropout=dropout)
        best_model, best_val_auc, best_epoch = train_zone_cnn(
            model=model,
            X_train_t=X_tr_t,
            y_train_t=y_tr_t,
            X_val_t=X_vl_metric_t,
            y_val=y_vl_metric,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        # 5. Validation üzerinde karar eşiği belirleme
        best_model.eval()
        with torch.no_grad():
            val_logits = best_model(X_vl_metric_t.to(device))
            val_metric_probs = torch.sigmoid(val_logits).cpu().numpy()

        opt_thresh, _ = find_optimal_threshold(y_vl_metric, val_metric_probs, method="youden")
        val_metrics = compute_binary_metrics(y_vl_metric, val_metric_probs, threshold=opt_thresh)

        # 6. Dondurulmuş en iyi model ve eşikle Test kümesi değerlendirmesi
        with torch.no_grad():
            test_logits = best_model(X_ts_metric_t.to(device))
            test_metric_probs = torch.sigmoid(test_logits).cpu().numpy()

        test_metrics = compute_binary_metrics(y_ts_metric, test_metric_probs, threshold=opt_thresh)

        zone_models[zone_id] = best_model
        zone_scalers[zone_id] = scaler
        zone_thresholds[zone_id] = float(opt_thresh)

        # Hasta tahmini için tüm zonların olasılıkları
        with torch.no_grad():
            val_all_logits = best_model(X_vl_all_t.to(device))
            val_all_probs = torch.sigmoid(val_all_logits).cpu().numpy()
            test_all_logits = best_model(X_ts_all_t.to(device))
            test_all_probs = torch.sigmoid(test_all_logits).cpu().numpy()

        for pid, prob in zip(splits_df.loc[val_all_mask, "patient_id"], val_all_probs):
            val_predictions_list.append({
                "patient_id": pid, "zone_id": zone_id, "zone_prob": float(prob)
            })

        for pid, prob in zip(splits_df.loc[test_all_mask, "patient_id"], test_all_probs):
            test_predictions_list.append({
                "patient_id": pid, "zone_id": zone_id, "zone_prob": float(prob)
            })

        elapsed = time.time() - z_start
        zone_results[f"zone_{zone_id}"] = {
            "zone_name": zone_name,
            "train_samples": len(y_tr),
            "best_epoch": best_epoch,
            "optimal_threshold": float(opt_thresh),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

        print(
            f"Zon {zone_id} ({zone_name}): "
            f"Train={len(y_tr)} | En İyi Epoch={best_epoch}/{max_epochs} | Eşik={opt_thresh:.4f}\n"
            f"  -> Validation: ROC-AUC={val_metrics['roc_auc']:.4f}, Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}\n"
            f"  -> TEST:       ROC-AUC={test_metrics['roc_auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f} ({elapsed:.1f}s)"
        )

    # 7. Hasta Düzeyinde Değerlendirme (Max Pooling)
    print("\n" + "-" * 80)
    print(f"HASTA DÜZEYİNDE BİRLEŞTİRME VE DEĞERLENDİRME (Yöntem: {patient_agg.upper()})")
    print("-" * 80)

    # Validation hasta tahmini
    val_pred_df = pd.DataFrame(val_predictions_list)
    val_patient_meta = splits_df[splits_df["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    val_patient_probs = val_pred_df.groupby("patient_id")["zone_prob"].max()
    val_patient_true = val_patient_meta.loc[val_patient_probs.index, "patient_label"].values

    patient_thresh, _ = find_optimal_threshold(val_patient_true, val_patient_probs.values, method="youden")
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

    # 8. Çıktıları Kaydetme
    models_dir = args.output_dir / "models" / "zone_cnn"
    runs_dir = args.output_dir / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch bundle kaydı
    bundle_data = {
        "models_state_dict": {z: m.state_dict() for z, m in zone_models.items()},
        "scalers": zone_scalers,
        "thresholds": zone_thresholds,
        "patient_threshold": float(patient_thresh),
        "config": cfg,
    }
    bundle_path = models_dir / "zone_cnn_bundle.pt"
    torch.save(bundle_data, bundle_path)

    full_report = {
        "model_type": "zone_cnn",
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "config": cfg,
        "patient_aggregation": patient_agg,
        "patient_threshold": round(patient_thresh, 4),
        "patient_val_metrics": patient_val_metrics,
        "patient_test_metrics": patient_test_metrics,
        "zone_results": zone_results,
    }

    report_path = runs_dir / "zone_cnn_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    test_pred_out_path = runs_dir / "test_patient_predictions_cnn.csv"
    test_summary_df = pd.DataFrame({
        "patient_id": test_patient_probs.index,
        "patient_label": test_patient_true,
        "predicted_patient_prob": test_patient_probs.values.round(4),
        "predicted_patient_label": (test_patient_probs.values >= patient_thresh).astype(int),
    })
    test_summary_df.to_csv(test_pred_out_path, index=False)

    print("\n" + "=" * 80)
    print("KÜÇÜK CNN EĞİTİM VE TEST BAŞARIYLA TAMAMLANDI")
    print("=" * 80)
    print(f"Model Paketi:    {bundle_path}")
    print(f"Rapor Dosyası:   {report_path}")
    print(f"Test Tahminleri: {test_pred_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
