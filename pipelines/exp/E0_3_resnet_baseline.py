"""E0.3 — Güçlü aynı-split baseline: donmuş pretrainli ResNet18 + linear probe.

Neden bu deney:
    Literatür (literature.md §2.8, §5.3) hakemin en olası itirazını şöyle
    özetliyor: "TDA neden global bir CNN baseline'ını geçmiyor?" Mevcut ham
    görüntü baseline'ı (0.7043) 64x64 girdili **küçük** bir CNN'dir ve adil
    değildir. Bu deney, aynı split üzerinde **ImageNet-pretrainli ResNet18**
    (donmuş omurga) + lojistik regresyon ile güçlü bir karşılaştırma noktası kurar.

NOT: CPU'da uçtan uca fine-tuning pratik değil (venv `torch+cpu`, GPU kapalı —
roadmap E0.0). Donmuş omurga + linear probe, tıbbi görüntülemede standart ve
savunulabilir bir "güçlü baseline"dır.

Çıktılar:
    data/processed/features_resnet18/X_zone_features.npy   (24410, 512)  [E3.1 füzyonu için]
    artifacts/runs/E0.3_resnet_baseline/val_report.json

Kullanım:
    python pipelines/exp/E0_3_resnet_baseline.py            # çıkarım + değerlendirme
    python pipelines/exp/E0_3_resnet_baseline.py --reuse-features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "E0.3"
SLUG = "resnet_baseline"
SEED = 42
INPUT_SIZE = 224
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
FEATURES_PATH = PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0.3 ResNet18 frozen baseline")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reuse-features", action="store_true")
    return p.parse_args()


def preprocess(path: Path) -> np.ndarray | None:
    """Zon kesitini 0-1 aralığına normalize edip INPUT_SIZE'a ölçekler."""
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    tissue = img[img > 0]
    if tissue.size < 100:
        return None
    lo, hi = np.percentile(tissue, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return img.astype(np.float32, copy=False)


def build_backbone() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    body = nn.Sequential(*list(model.children())[:-1])   # avgpool'a kadar
    body.eval()
    for param in body.parameters():
        param.requires_grad = False
    return body


def extract_features(splits: pd.DataFrame, zones_dir: Path, batch_size: int) -> np.ndarray:
    import torchvision.transforms.functional as TF

    body = build_backbone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    n = len(splits)
    feats = np.zeros((n, 512), dtype=np.float32)
    buffer, indices = [], []
    t0 = time.time()

    def flush():
        if not buffer:
            return
        x = torch.from_numpy(np.stack(buffer)).float()    # (B,H,W)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)             # gri -> 3 kanal
        x = (x - mean) / std
        with torch.no_grad():
            out = body(x).flatten(1)
        for k, idx in enumerate(indices):
            feats[idx] = out[k].numpy()
        buffer.clear()
        indices.clear()

    for pos, row in enumerate(splits.itertuples(index=False)):
        path = zones_dir / row.patient_id / ZONE_FILES[int(row.zone_id)]
        img = preprocess(path) if path.exists() else None
        if img is None:
            buffer.append(np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32))
        else:
            t = torch.from_numpy(img).unsqueeze(0)
            t = TF.resize(t, [INPUT_SIZE, INPUT_SIZE], antialias=True)
            buffer.append(t.squeeze(0).numpy())
        indices.append(pos)
        if len(buffer) >= batch_size:
            flush()
            if (pos + 1) % (batch_size * 20) == 0:
                done = pos + 1
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done}/{n} (%{100*done/n:.0f}) | {rate:.0f} zon/s | "
                      f"kalan ~{(n-done)/max(rate,1e-9)/60:.1f} dk", flush=True)
    flush()
    print(f"  Cikarim tamamlandi: {n} zon, {time.time()-t0:.0f} s")
    return feats


def evaluate(feats: np.ndarray, splits: pd.DataFrame) -> dict:
    y = splits["zone_label"].values
    train_m = splits["is_train_balanced_sample"].values.astype(bool)

    zone_models, zone_scalers, zone_thresholds = {}, {}, {}
    val_rows, test_rows = [], []

    for zone_id in range(1, 6):
        zm = splits["zone_id"] == zone_id
        tr = zm & train_m
        va = zm & (splits["split"] == "val")
        te = zm & (splits["split"] == "test")

        scaler = StandardScaler().fit(feats[tr.values])
        Xtr, Xva, Xte = scaler.transform(feats[tr.values]), scaler.transform(feats[va.values]), scaler.transform(feats[te.values])

        best = {"C": None, "auc": -1.0, "model": None}
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
            clf.fit(Xtr, y[tr.values])
            pv = clf.predict_proba(Xva)[:, 1]
            from sklearn.metrics import roc_auc_score
            a = float(roc_auc_score(y[va.values], pv))
            if a > best["auc"]:
                best = {"C": C, "auc": a, "model": clf}

        clf = best["model"]
        pv = clf.predict_proba(Xva)[:, 1]
        pt = clf.predict_proba(Xte)[:, 1]
        thr, _ = find_optimal_threshold(y[va.values], pv, method="youden")
        zone_models[zone_id], zone_scalers[zone_id], zone_thresholds[zone_id] = clf, scaler, thr

        for pid, p in zip(splits.loc[va, "patient_id"], pv):
            val_rows.append({"patient_id": pid, "p": p})
        for pid, p in zip(splits.loc[te, "patient_id"], pt):
            test_rows.append({"patient_id": pid, "p": p})

    def patient_probs(rows, split_name):
        df = pd.DataFrame(rows).groupby("patient_id")["p"].max()
        meta = splits[splits["split"] == split_name].drop_duplicates("patient_id").set_index("patient_id")
        return df.reindex(meta.index), meta["patient_label"]

    vp, vy = patient_probs(val_rows, "val")
    tp, ty = patient_probs(test_rows, "test")
    thr, _ = find_optimal_threshold(vy.values, vp.values, method="youden")
    return {
        "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        "best_C_per_zone": {z: zone_models[z].C for z in zone_models},
        "patient_threshold": float(thr),
        "val_auc": float(compute_binary_metrics(vy.values, vp.values)["roc_auc"]),
        "test_metrics": compute_binary_metrics(ty.values, tp.values, threshold=thr),
        "val_probs": vp, "test_probs": tp, "y_val": vy, "y_test": ty,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)

    print("=" * 78)
    print("E0.3 — DONMUS PRETRAINLI ResNet18 + LINEAR PROBE (GUCLU BASELINE)")
    print("=" * 78)
    print(f"Zon satiri: {len(splits)} | girdi: {INPUT_SIZE}x{INPUT_SIZE} | cihaz: cpu")

    if args.reuse_features and FEATURES_PATH.exists():
        feats = np.load(FEATURES_PATH)
        print(f"Kayitli ozellikler yuklendi: {feats.shape}")
    else:
        feats = extract_features(splits, args.zones_dir, args.batch_size)
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(FEATURES_PATH, feats)
        print(f"Ozellikler kaydedildi: {FEATURES_PATH} {feats.shape}")

    res = evaluate(feats, splits)
    tm = res["test_metrics"]
    print("\n--- ResNet18 (donmus) + lojistik regresyon ---")
    print(f"  En iyi C (zon basina): {res['best_C_per_zone']}")
    print(f"  Val  ROC-AUC: {res['val_auc']:.4f}")
    print(f"  Test ROC-AUC: {tm['roc_auc']:.4f} | Sens={tm['sensitivity']:.4f} "
          f"Spec={tm['specificity']:.4f} F1={tm['f1']:.4f}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Ayni split'te ImageNet-pretrainli ResNet18 tabanli guclu baseline kurulabilir.",
        "input_size": INPUT_SIZE,
        "feature_dim": int(feats.shape[1]),
        "val_auc": res["val_auc"],
        "test_metrics": tm,
        "best_C_per_zone": res["best_C_per_zone"],
        "patient_threshold": res["patient_threshold"],
        "comparison": {"tda_pi_cnn_test_auc": 0.8631, "ensemble_test_auc": 0.8776,
                       "small_raw_cnn_test_auc": 0.7043},
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision="BASELINE",
        metrics={"val_auc": res["val_auc"], "test_roc_auc": tm["roc_auc"]},
        notes=f"Donmus ResNet18 + lojistik regresyon, {INPUT_SIZE}px. Test ROC-AUC={tm['roc_auc']:.4f}.",
        extra={"features_path": str(FEATURES_PATH), "feature_dim": int(feats.shape[1])},
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
