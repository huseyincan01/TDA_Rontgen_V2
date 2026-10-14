"""5-Zon Ham Görüntü (Raw Zone Image) Küçük CNN Eğitim ve Test Değerlendirme Boru Hattı.

Bu betik:
1. Akciğer zonlarının ham .mha kesitlerini okur, doku bölgesini kırpar,
   min-max normalize eder ve (1, 64, 64) tensörüne yeniden boyutlandırır (TDA kullanılmaz).
2. v2.1 bölünmesi üzerinde 5 bağımsız RawZoneCNN eğitir:
   - Train: is_train_balanced_sample == True (TDA ile birebir aynı 1:1 dengeli alt küme)
   - Val / Test Zon Değerlendirmesi: is_zone_metric_sample == True
3. Her zon için en yüksek Validation ROC-AUC veren ağırlıkları kaydeder.
4. Validation kümesinde Youden's J ile eşik kilitler.
5. Dokunulmamış 734 test hastasında Max Pooling ile hasta düzeyinde nihai başarıyı değerlendirir.
6. Çıktıları artifacts/ altına kaydeder.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold

ZONE_FILENAMES = {
    1: "zone_1_sag_ust.mha",
    2: "zone_2_sag_orta.mha",
    3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha",
    5: "zone_5_sol_alt.mha",
}

ZONE_NAMES = {
    1: "Sağ Üst (R-Upper)",
    2: "Sağ Orta (R-Mid)",
    3: "Sağ Alt (R-Lower)",
    4: "Sol Üst (L-Upper)",
    5: "Sol Alt (L-Lower)",
}


class RawZoneCNN(nn.Module):
    """(1, 64, 64) ham gri seviye zon kesiti için hafif CNN."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        # 1. Blok: 64x64 -> 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 2. Blok: 32x32 -> 16x16
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 3. Blok: 16x16 -> 8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 4. Havuzlama: 8x8 -> 4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 5. Sınıflandırıcı Başlık
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        return self.classifier(x).squeeze(-1)


def load_and_preprocess_mha(path_str: str, target_size: int = 64) -> np.ndarray:
    """Tek bir .mha kesitini okur, kırpar, normalize eder ve (target_size, target_size) yapar."""
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if arr.ndim == 3:
            arr = arr[0]
        valid = arr > 0
        if np.any(valid):
            y_coords, x_coords = np.where(valid)
            arr = arr[y_coords.min() : y_coords.max() + 1, x_coords.min() : x_coords.max() + 1]
        arr = arr.astype(np.float32)
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
        t_resized = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return t_resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
    except Exception:
        return np.zeros((target_size, target_size), dtype=np.float32)


def train_zone_raw_cnn(
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
    """RawZoneCNN modelini eğitir ve en yüksek Validation ROC-AUC veren ağırlıkları döner."""
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
    parser = argparse.ArgumentParser(description="5-Zon Ham Görüntü CNN Eğitim Boru Hattı.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("NODE21 TDA — 5-ZON HAM GÖRÜNTÜ KÜÇÜK CNN (RAW BASELINE) EĞİTİMİ")
    print("=" * 80)
    print(f"Giriş Temsili:        Ham Zon Kesiti ({args.image_size}x{args.image_size}, 1-Kanal Gri Seviye)")
    print(f"Eğitim Parametreleri: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}")

    splits_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv"
    patients_dir = PROJECT_ROOT / "data" / "processed" / "zones" / "patients"
    splits_df = pd.read_csv(splits_path)

    zone_models: Dict[int, nn.Module] = {}
    zone_thresholds: Dict[int, float] = {}
    zone_results = {}

    test_predictions_list = []
    val_predictions_list = []

    device = torch.device("cpu")

    print("\n" + "-" * 80)
    print("ZON BAZLI HAM GÖRÜNTÜ CNN EĞİTİMİ BAŞLIYOR")
    print("-" * 80)

    for zone_id in range(1, 6):
        z_start = time.time()
        zone_name = ZONE_NAMES[zone_id]
        zone_filename = ZONE_FILENAMES[zone_id]

        # 1. Filtreler
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
        val_all_mask = (splits_df["split"] == "val") & (splits_df["zone_id"] == zone_id)
        test_all_mask = (splits_df["split"] == "test") & (splits_df["zone_id"] == zone_id)

        # Hasta ID'leri
        train_pids = splits_df.loc[train_mask, "patient_id"].values
        y_tr = splits_df.loc[train_mask, "zone_label"].values.astype(np.float32)

        val_metric_pids = splits_df.loc[val_metric_mask, "patient_id"].values
        y_vl_metric = splits_df.loc[val_metric_mask, "zone_label"].values.astype(np.float32)

        test_metric_pids = splits_df.loc[test_metric_mask, "patient_id"].values
        y_ts_metric = splits_df.loc[test_metric_mask, "zone_label"].values.astype(np.float32)

        val_all_pids = splits_df.loc[val_all_mask, "patient_id"].values
        test_all_pids = splits_df.loc[test_all_mask, "patient_id"].values

        # 2. Resimleri Yükleme (ThreadPool ile çok hızlı)
        all_pids_needed = list(dict.fromkeys(list(train_pids) + list(val_all_pids) + list(test_all_pids)))
        file_paths = [str(patients_dir / pid / zone_filename) for pid in all_pids_needed]

        load_t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            images = list(ex.map(lambda p: load_and_preprocess_mha(p, args.image_size), file_paths))
        load_time = time.time() - load_t0

        pid_to_img = {pid: img for pid, img in zip(all_pids_needed, images)}

        # Tensörleri Oluşturma
        X_tr_np = np.stack([pid_to_img[pid] for pid in train_pids])[:, np.newaxis, :, :]
        X_vl_metric_np = np.stack([pid_to_img[pid] for pid in val_metric_pids])[:, np.newaxis, :, :]
        X_ts_metric_np = np.stack([pid_to_img[pid] for pid in test_metric_pids])[:, np.newaxis, :, :]
        X_vl_all_np = np.stack([pid_to_img[pid] for pid in val_all_pids])[:, np.newaxis, :, :]
        X_ts_all_np = np.stack([pid_to_img[pid] for pid in test_all_pids])[:, np.newaxis, :, :]

        X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_vl_metric_t = torch.tensor(X_vl_metric_np, dtype=torch.float32)
        X_ts_metric_t = torch.tensor(X_ts_metric_np, dtype=torch.float32)
        X_vl_all_t = torch.tensor(X_vl_all_np, dtype=torch.float32)
        X_ts_all_t = torch.tensor(X_ts_all_np, dtype=torch.float32)

        # 3. Model Eğitimi
        model = RawZoneCNN(in_channels=1, dropout=0.3)
        best_model, best_val_auc, best_epoch = train_zone_raw_cnn(
            model=model,
            X_train_t=X_tr_t,
            y_train_t=y_tr_t,
            X_val_t=X_vl_metric_t,
            y_val=y_vl_metric,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        # 4. Validation Üzerinde Eşik Belirleme
        best_model.eval()
        with torch.no_grad():
            val_logits = best_model(X_vl_metric_t.to(device))
            val_metric_probs = torch.sigmoid(val_logits).cpu().numpy()

        opt_thresh, _ = find_optimal_threshold(y_vl_metric, val_metric_probs, method="youden")
        val_metrics = compute_binary_metrics(y_vl_metric, val_metric_probs, threshold=opt_thresh)

        # 5. Dondurulmuş Eşikle Test Değerlendirmesi
        with torch.no_grad():
            test_logits = best_model(X_ts_metric_t.to(device))
            test_metric_probs = torch.sigmoid(test_logits).cpu().numpy()

        test_metrics = compute_binary_metrics(y_ts_metric, test_metric_probs, threshold=opt_thresh)

        zone_models[zone_id] = best_model
        zone_thresholds[zone_id] = float(opt_thresh)

        # Hasta tahminleri için tüm zon olasılıkları
        with torch.no_grad():
            val_all_logits = best_model(X_vl_all_t.to(device))
            val_all_probs = torch.sigmoid(val_all_logits).cpu().numpy()
            test_all_logits = best_model(X_ts_all_t.to(device))
            test_all_probs = torch.sigmoid(test_all_logits).cpu().numpy()

        for pid, prob in zip(val_all_pids, val_all_probs):
            val_predictions_list.append({"patient_id": pid, "zone_id": zone_id, "zone_prob": float(prob)})

        for pid, prob in zip(test_all_pids, test_all_probs):
            test_predictions_list.append({"patient_id": pid, "zone_id": zone_id, "zone_prob": float(prob)})

        elapsed = time.time() - z_start
        zone_results[f"zone_{zone_id}"] = {
            "zone_name": zone_name,
            "train_samples": len(y_tr),
            "best_epoch": best_epoch,
            "optimal_threshold": float(opt_thresh),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "load_time_seconds": round(load_time, 2),
            "elapsed_seconds": round(elapsed, 2),
        }

        print(
            f"Zon {zone_id} ({zone_name}): "
            f"Train={len(y_tr)} | Epoch={best_epoch}/{args.epochs} | Eşik={opt_thresh:.4f}\n"
            f"  -> Validation: ROC-AUC={val_metrics['roc_auc']:.4f}, Sens={val_metrics['sensitivity']:.4f}, Spec={val_metrics['specificity']:.4f}\n"
            f"  -> TEST:       ROC-AUC={test_metrics['roc_auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f} ({elapsed:.1f}s)"
        )

    # 6. Hasta Düzeyinde Değerlendirme (Max Pooling)
    print("\n" + "-" * 80)
    print("HASTA DÜZEYİNDE BİRLEŞTİRME VE DEĞERLENDİRME (MAX POOLING)")
    print("-" * 80)

    val_pred_df = pd.DataFrame(val_predictions_list)
    val_patient_meta = splits_df[splits_df["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    val_patient_probs = val_pred_df.groupby("patient_id")["zone_prob"].max()
    val_patient_true = val_patient_meta.loc[val_patient_probs.index, "patient_label"].values

    patient_thresh, _ = find_optimal_threshold(val_patient_true, val_patient_probs.values, method="youden")
    patient_val_metrics = compute_binary_metrics(val_patient_true, val_patient_probs.values, threshold=patient_thresh)

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
    models_dir = PROJECT_ROOT / "artifacts" / "models" / "zone_raw_cnn"
    runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = models_dir / "zone_raw_cnn_bundle.pt"
    bundle_data = {
        "models_state_dict": {z: m.state_dict() for z, m in zone_models.items()},
        "thresholds": zone_thresholds,
        "patient_threshold": float(patient_thresh),
        "config": {"epochs": args.epochs, "image_size": args.image_size, "lr": args.lr},
    }
    torch.save(bundle_data, bundle_path)

    full_report = {
        "model_type": "zone_raw_cnn",
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "config": {"epochs": args.epochs, "image_size": args.image_size, "lr": args.lr},
        "patient_aggregation": "max",
        "patient_threshold": round(patient_thresh, 4),
        "patient_val_metrics": patient_val_metrics,
        "patient_test_metrics": patient_test_metrics,
        "zone_results": zone_results,
    }

    report_path = runs_dir / "zone_raw_cnn_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    test_pred_out_path = runs_dir / "test_patient_predictions_raw_cnn.csv"
    test_summary_df = pd.DataFrame({
        "patient_id": test_patient_probs.index,
        "patient_label": test_patient_true,
        "predicted_patient_prob": test_patient_probs.values.round(4),
        "predicted_patient_label": (test_patient_probs.values >= patient_thresh).astype(int),
    })
    test_summary_df.to_csv(test_pred_out_path, index=False)

    print("\n" + "=" * 80)
    print("HAM GÖRÜNTÜ CNN EĞİTİM VE TEST BAŞARIYLA TAMAMLANDI")
    print("=" * 80)
    print(f"Model Paketi:    {bundle_path}")
    print(f"Rapor Dosyası:   {report_path}")
    print(f"Test Tahminleri: {test_pred_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
