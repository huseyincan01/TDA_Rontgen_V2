"""E0.6 — BELİRLEYİCİ TEST: TDA, ince ayarlı (güçlü) görüntü modeline de katkı sağlıyor mu?

Neden bu deney (ve neden son test hakkı):
    E3.1'de ana iddiamız şuydu: *"TDA + görüntü füzyonu, tek başına görüntü modelinden
    +0.0862 yüksek"*. Ancak o görüntü modeli **DONMUŞ** ImageNet ResNet18'di (test
    0.8349) ve ciddi genelleme kaybı yaşıyordu. E0.4/E0.5, **ince ayarlı** ResNet18'in
    val hasta AUC'sini **0.94–0.95**'e çıkardı — füzyonun val değeri 0.9572'ye çok yakın.

    Yani füzyon kazancının büyük kısmı **zayıf baseline**'dan geliyor olabilir. Bu,
    literatür taramasında öngörülen **1 numaralı hakem itirazının** ta kendisidir.

Ön-kayıtlı hipotez (ANA İDDİANIN SINAVI):
    TDA dalı, **ince ayarlı** görüntü modeline de test ROC-AUC'sinde **≥ +0.015** katkı
    sağlar (bootstrap %95 CI sıfırı dışlar).

Tasarım:
    * Görüntü dalı: ince ayarlı ResNet18 (E0.5 'mild' tarifi — en iyi val).
    * TDA dalı: mevcut 5-zon CNN + SVM ensemble (E3.4 konfigürasyonu, w=0.55).
    * Füzyon ağırlığı ve eşiği **yalnızca validation'da** seçilir.
    * Karşılaştırmalar: dondurulmuş ResNet (0.8349), füzyon-dondurulmuş (0.9190).

Kapsam:
    **Son test değerlendirmesi buraya ayrıldı** (bütçe 5/6 idi → 6/6).

Kullanım:
    python pipelines/exp/E0_6_decisive_test.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E0.6"
SLUG = "decisive_test"
SEED = 42
INPUT_SIZE = 224
EPOCHS = 25
LR = 1e-4
WD = 1e-4
DROPOUT = 0.3
BATCH = 64
TDA_CWT = 0.55
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_BOOTSTRAP = 2000
MIN_DELTA = 0.015
FROZEN_REF = {"image_test": 0.8349, "fusion_test": 0.9190, "tda_test": 0.8776}
CACHE = PROJECT_ROOT / "data/processed/features_resnet18/trainval_gray224.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0.6 belirleyici test")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--tda-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--cnn-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt")
    p.add_argument("--svm-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    return p.parse_args()


def raw_batch(gray, device):
    return torch.from_numpy(gray).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1).to(device)


def normalize(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def augment_hflip(x):
    b = x.shape[0]
    f = torch.rand(b, device=x.device) < 0.5
    if f.any():
        x[f] = torch.flip(x[f], dims=[3])
    return x


@torch.no_grad()
def predict(model, gray, idx, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out.append(torch.sigmoid(model(normalize(raw_batch(gray[b], device))).squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def bootstrap_auc_diff(y, pa, pb, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, len(y), len(y))
        yi = y[idx]
        if yi.min() == yi.max():
            d[i] = np.nan
            continue
        d[i] = roc_auc_score(yi, np.asarray(pa)[idx]) - roc_auc_score(yi, np.asarray(pb)[idx])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p2 = float(min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))))
    return float(lo), float(hi), p2


def main() -> None:
    args = parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = (zs["is_train_balanced_sample"].values.astype(bool)
            | (zs["split"] == "val").values | (zs["split"] == "test").values)
    sub = zs[need].reset_index(drop=True)
    y = sub["zone_label"].values.astype(np.float32)

    print("=" * 78)
    print("E0.6 — BELIRLEYICI TEST: TDA, INCE AYARLI GORUNTU MODELINE DE KATKI SAGLIYOR MU?")
    print("=" * 78)
    gray = np.load(CACHE)
    if len(gray) != len(sub):
        raise ValueError(f"Onbellek uyusmuyor: {len(gray)} vs {len(sub)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zon: {len(sub)} | cihaz: {device}")

    tr = sub["is_train_balanced_sample"].values.astype(bool)
    va = (sub["split"] == "val").values
    te = (sub["split"] == "test").values
    idx_tr = np.where(tr)[0]

    # --- Goruntu dali: ince ayarli ResNet18 ---
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, 1))
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()

    best_auc, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.RandomState(SEED + ep)
        rng.shuffle(idx_tr)
        for i in range(0, len(idx_tr), BATCH):
            b = idx_tr[i:i + BATCH]
            xb = normalize(augment_hflip(raw_batch(gray[b], device)))
            yb = torch.tensor(y[b], device=device)
            opt.zero_grad(); crit(model(xb).squeeze(-1), yb).backward(); opt.step()
        pv = predict(model, gray, np.where(va & sub["is_zone_metric_sample"].values)[0], device)
        a = roc_auc_score(y[va & sub["is_zone_metric_sample"].values], pv)
        if a > best_auc:
            best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  epoch {ep:2d}/{EPOCHS} | val zon AUC={a:.4f} | en iyi={best_auc:.4f}", flush=True)
    model.load_state_dict(best_state)

    def img_patient_probs(mask):
        idx = np.where(mask)[0]
        p = predict(model, gray, idx, device)
        t = sub.loc[mask, ["patient_id"]].copy(); t["p"] = p
        return t.groupby("patient_id")["p"].max()

    img_val = img_patient_probs(va)
    img_test = img_patient_probs(te)

    # --- TDA dali: mevcut 5-zon CNN + SVM (E3.4 konfigurasyonu) ---
    X_tda = np.load(args.tda_features)
    cb = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    sb = joblib.load(args.svm_bundle)
    drop = float(cb["config"].get("dropout", 0.3))
    tda = {}
    for name in ("val", "test"):
        rows = []
        for z in range(1, 6):
            m = ((sub["split"] == name) & (sub["zone_id"] == z)).values
            idx = sub[m]["vector_index"].values
            Xz = X_tda[idx]
            sc = cb["scalers"][z]
            mm = SmallZoneCNN(in_channels=2, dropout=drop); mm.load_state_dict(cb["models_state_dict"][z]); mm.eval()
            with torch.no_grad():
                pc = torch.sigmoid(mm(torch.tensor(vector_to_2channel_image(sc.transform(Xz)),
                                                     dtype=torch.float32))).numpy()
            ps = sb.predict_zone_proba(z, Xz)
            fused = TDA_CWT * pc + (1 - TDA_CWT) * ps
            rows.append(pd.DataFrame({"patient_id": sub.loc[m, "patient_id"].values, "p": fused}))
        tda[name] = pd.concat(rows).groupby("patient_id")["p"].max()

    meta_val = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    meta_test = sub[sub["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")
    y_val = meta_val["patient_label"].values
    y_test = meta_test["patient_label"].values

    iv = img_val.reindex(meta_val.index).values
    it = img_test.reindex(meta_test.index).values
    tv = tda["val"].reindex(meta_val.index).values
    tt = tda["test"].reindex(meta_test.index).values

    img_test_auc = float(roc_auc_score(y_test, it))
    tda_test_auc = float(roc_auc_score(y_test, tt))
    best = {"a": None, "auc": -1.0}
    for a in A_GRID:
        s = roc_auc_score(y_val, a * iv + (1 - a) * tv)
        if s > best["auc"]:
            best = {"a": float(a), "auc": float(s)}
    a = best["a"]
    fv = a * iv + (1 - a) * tv
    ft = a * it + (1 - a) * tt
    thr, _ = find_optimal_threshold(y_val, fv, method="youden")
    fus_auc = float(roc_auc_score(y_test, ft))
    metrics = compute_binary_metrics(y_test, ft, threshold=thr)
    lo, hi, p2 = bootstrap_auc_diff(y_test, ft, it)
    delta = fus_auc - img_test_auc
    passed = delta >= MIN_DELTA and lo > 0

    print(f"\n--- TEST SONUCLARI (734 hasta) ---")
    print(f"  Ince ayarli ResNet18 (tek)     : {img_test_auc:.4f}   (dondurulmus {FROZEN_REF['image_test']})")
    print(f"  TDA ensemble (tek)             : {tda_test_auc:.4f}   (dondurulmus donem {FROZEN_REF['tda_test']})")
    print(f"  FUZYON (a={a:.2f})                 : {fus_auc:.4f}   (fuzyon-dondurulmus {FROZEN_REF['fusion_test']})")
    print(f"  Fusion dogruluk/esik           : {metrics['accuracy']:.4f} / {thr:.4f}")
    print(f"\n  TDA katkisi (fuzyon - ince ayarli goruntu): {delta:+.4f} "
          f"[%95 CI {lo:+.4f}, {hi:+.4f}] p={p2:.4f}")
    print(f"  Kriter: >= +{MIN_DELTA} ve CI>0 -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => UYARI: TDA'nin katkisi, GUCU ince ayarli goruntu modeli karsisinda zayifliyor.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"TDA, ince ayarli goruntu modeline de test AUC'sinde >= +{MIN_DELTA} katki saglar.",
        "scope": "SON test degerlendirmesi (butce 6/6)",
        "image_branch": {"type": "fine-tuned ResNet18 (layer4+fc)", "epochs": EPOCHS, "lr": LR,
                         "wd": WD, "dropout": DROPOUT, "aug": "hflip", "zone_val_auc": float(best_auc),
                         "test_auc": img_test_auc},
        "tda_branch": {"type": "5-zone CNN+SVM ensemble", "w": TDA_CWT, "test_auc": tda_test_auc},
        "fusion": {"a": a, "patient_threshold": float(thr), "test_auc": fus_auc, "test_metrics": metrics},
        "tda_contribution": {"delta": delta, "bootstrap_ci95": [lo, hi], "p_value": p2},
        "references_frozen_period": FROZEN_REF,
        "criteria": {"delta_ge_0.015": bool(delta >= MIN_DELTA), "ci_excludes_zero": bool(lo > 0)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.tda_features], seed=SEED,
        decision=report["decision"],
        metrics={"image_test_auc": img_test_auc, "tda_test_auc": tda_test_auc,
                 "fusion_test_auc": fus_auc, "tda_delta": delta},
        notes=f"Belirleyici test: ince ayarli goruntu {img_test_auc:.4f}, fuzyon {fus_auc:.4f}, TDA katkisi {delta:+.4f}.",
    )
    print(f"\nRapor    : {run_dir / 'test_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
