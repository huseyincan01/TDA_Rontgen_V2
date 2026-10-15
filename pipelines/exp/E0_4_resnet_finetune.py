"""E0.4 — ResNet18 son-blok ince ayarı: "hazır model" eleştirisini kapatma denemesi.

Neden bu deney:
    E0.3'te görüntü dalı **donmuş** ImageNet ResNet18 + lojistik regresyondu (val 0.8884,
    test 0.8349). Bu, "görüntü tarafında model EĞİTMEDİNİZ" eleştirisine açıktır. E0.4,
    ağın **son bloğunu (layer4) + sınıflandırıcı başlığını** ince ayar yaparak *gerçekten
    eğitilmiş* bir görüntü modeli kurar.

    Neden sadece son blok: venv'de `torch+cpu` var (GPU yok) ve dengeli train yalnızca
    2.040 zon. Tüm ağı eğitmek hem saatler sürer hem de 11,7M parametreyle aşırı uyum
    yapar. `layer4 + fc` ince ayarı CPU'da ~20 dakikada biter ve transfer öğrenmede
    standart bir yaklaşımdır.

Ön-kayıtlı hipotez:
    Son-blok ince ayarı, dondurulmuş versiyonun val ROC-AUC'sini (0.8884) **≥ +0.015**
    artırır — yani E0.3 baseline'ı gerçekten "zayıf" kalmıştır.

Tasarım:
    Tek değişken: `layer4+fc` eğitilebilir mi (dondurulmuş) — mimari, ön-işleme, split ve
    değerlendirme protokolü E0.3 ile **aynı**. Zon kimliği E3.5'te eşdeğer bulunduğu için
    **havuzlanmış tek model** eğitilir (5× veri, daha stabil).

Kapsam (test bütçesi korunur):
    Yalnızca **validation** kararı; test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E0_4_resnet_finetune.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
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

EXPERIMENT_ID = "E0.4"
SLUG = "resnet_finetune"
SEED = 42
INPUT_SIZE = 224
EPOCHS = 15
LR = 1e-4
BATCH = 64
FROZEN_VAL_AUC = 0.8884
MIN_DELTA = 0.015
CACHE = PROJECT_ROOT / "data/processed/features_resnet18/trainval_gray224.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0.4 ResNet18 son-blok ince ayari")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--reuse-cache", action="store_true")
    return p.parse_args()


def preprocess(path: Path) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    tissue = img[img > 0]
    if tissue.size >= 100:
        lo, hi = np.percentile(tissue, [1.0, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    t = torch.from_numpy(img).unsqueeze(0)
    return (TF.resize(t, [INPUT_SIZE, INPUT_SIZE], antialias=True).squeeze(0).numpy() * 255.0).astype(np.uint8)


def build_or_load_cache(sub: pd.DataFrame, zones_dir: Path, reuse: bool) -> np.ndarray:
    if reuse and CACHE.exists():
        arr = np.load(CACHE)
        print(f"Onbellek yuklendi: {arr.shape}")
        return arr
    t0 = time.time()
    arr = np.zeros((len(sub), INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    for i, row in enumerate(sub.itertuples(index=False)):
        p = zones_dir / row.patient_id / ZONE_FILES[int(row.zone_id)]
        if p.exists():
            arr[i] = preprocess(p)
        if (i + 1) % 500 == 0:
            rate = (i + 1) / max(1e-9, time.time() - t0)
            print(f"  {i+1}/{len(sub)} | {rate:.0f} zon/s | kalan ~{(len(sub)-i-1)/max(rate,1e-9)/60:.1f} dk",
                  flush=True)
    print(f"  On-isleme tamamlandi: {len(sub)} zon, {time.time()-t0:.0f} s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, arr)
    return arr


def to_batch(gray_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(gray_u8).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x.to(device) - mean) / std


@torch.no_grad()
def predict(model: nn.Module, gray: np.ndarray, idx: np.ndarray, device: torch.device,
            bs: int = 256) -> np.ndarray:
    """Toplu cikarim — tum val setini tek forward'da gecirmek 6 GB VRAM'i asar (OOM)."""
    model.eval()
    outs = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        outs.append(torch.sigmoid(model(to_batch(gray[b], device)).squeeze(-1)).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty(0)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = zs["is_train_balanced_sample"].values.astype(bool) | (zs["split"] == "val").values | (zs["split"] == "test").values
    sub = zs[need].reset_index(drop=True)
    y = sub["zone_label"].values.astype(np.float32)

    print("=" * 78)
    print("E0.4 — ResNet18 SON-BLOK INCE AYARI (dondurulmus baseline'a karsi)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | dondurulmus val referansi: {FROZEN_VAL_AUC:.4f} | "
          f"epochs={EPOCHS} lr={LR}")
    print("TEST SETINE DOKUNULMUYOR.")

    gray = build_or_load_cache(sub, args.zones_dir, args.reuse_cache)

    tr = sub["is_train_balanced_sample"].values.astype(bool)
    va = (sub["split"] == "val").values
    vt = va & sub["is_zone_metric_sample"].values.astype(bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    print(f"  Cihaz: {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    print(f"  Egitilebilir parametre: {n_train_p:,} (toplam {sum(p.numel() for p in model.parameters()):,})")

    opt = torch.optim.Adam(params, lr=LR)
    crit = nn.BCEWithLogitsLoss()
    idx_tr = np.where(tr)[0]
    idx_vt = np.where(vt)[0]

    best_auc, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        rng = np.random.RandomState(SEED + ep)
        rng.shuffle(idx_tr)
        losses = []
        for i in range(0, len(idx_tr), BATCH):
            b = idx_tr[i:i + BATCH]
            xb = to_batch(gray[b], device); yb = torch.tensor(y[b], device=device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        pv = predict(model, gray, idx_vt, device)
        a = roc_auc_score(y[idx_vt], pv)
        if a > best_auc:
            best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}/{EPOCHS} | loss={np.mean(losses):.4f} | val zon AUC={a:.4f} "
              f"| en iyi={best_auc:.4f} | {time.time()-t0:.0f} s", flush=True)

    model.load_state_dict(best_state)
    model.eval()

    # hasta duzeyi val AUC
    pv_all = predict(model, gray, np.where(va)[0], device)
    tmp = sub.loc[va, ["patient_id"]].copy(); tmp["p"] = pv_all
    pdf = tmp.groupby("patient_id")["p"].max()
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv_pat = pdf.reindex(meta.index)
    val_patient_auc = float(roc_auc_score(meta["patient_label"].values, pv_pat.values))

    print(f"\n--- SONUC ---")
    print(f"  zon duzeyi val AUC (en iyi epoch): {best_auc:.4f}")
    print(f"  HASTA duzeyi val AUC            : {val_patient_auc:.4f}")
    print(f"  dondurulmus versiyon (E0.3)     : {FROZEN_VAL_AUC:.4f}")
    delta = val_patient_auc - FROZEN_VAL_AUC
    print(f"  fark                            : {delta:+.4f}")

    passed = delta >= MIN_DELTA
    decision = "KEEP" if passed else "KILL"
    print(f"\n  Kriter: fark >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {decision}")
    if not passed:
        print("  => Dondurulmus baseline YETERLIYDI; 'hazir model' elestirisi gecerli degil.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Son-blok ince ayari, dondurulmus val AUC'yi >= +{MIN_DELTA} artirir.",
        "scope": "validation only; test setine dokunulmadi",
        "epochs": EPOCHS, "lr": LR, "batch": BATCH,
        "trainable_params": int(n_train_p),
        "frozen_reference_val_patient_auc": FROZEN_VAL_AUC,
        "finetuned_zone_val_auc": float(best_auc),
        "finetuned_patient_val_auc": val_patient_auc,
        "delta": delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": decision,
        "note": "Havuzlanmis tek model (E3.5 zon-ozel ile esdeger bulundu); augmentasyon yok.",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=decision,
        metrics={"finetuned_patient_val_auc": val_patient_auc, "delta": delta},
        notes=f"ResNet18 son-blok ince ayari: hasta val AUC={val_patient_auc:.4f}; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
