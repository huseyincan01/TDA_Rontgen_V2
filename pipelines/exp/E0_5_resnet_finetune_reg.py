"""E0.5 — Düzenlileştirilmiş ResNet18 ince ayarı: aşırı uyumu giderme denemesi.

Neden bu deney:
    E0.4'te son-blok ince ayarı val hasta AUC'sini 0.8884 → **0.9487**'ye çıkardı (KEEP),
    ANCAK **ciddi aşırı uyum** vardı: eğitim kaybı 0.5894 → **0.0012** (ezberleme) ve
    zon düzeyi val AUC **0.859'da plato**. Bu yüzden E0.4'ün val kazancı **güvenilmez**:
    hasta düzeyindeki sıçrama muhtemelen max-pooling + aşırı özgüven artefaktıdır.

    E0.5 aynı ince ayarı **veri artırma + güçlü düzenlileştirme** ile tekrarlar ve
    aşırı uyumun gerçekten giderilip giderilmediğini ölçer.

Ön-kayıtlı hipotez (İKİ koşul birlikte):
    (a) Aşırı uyum imzası kaybolur: **son eğitim kaybı > 0.05** (E0.4'te 0.0012 idi), VE
    (b) Temsil gerçekten iyileşir: **zon düzeyi val AUC ≥ 0.87** (E0.4'te 0.8590).

Tek değişken:
    Eğitim tarifi (artırma + düzenlileştirme) — mimari, ön-işleme, split, seed ve
    değerlendirme protokolü E0.4 ile **aynı**.

Kapsam (test bütçesi korunur):
    Yalnızca **validation**; test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E0_5_resnet_finetune_reg.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

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

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "E0.5"
SLUG = "resnet_finetune_reg"
SEED = 42
INPUT_SIZE = 224
EPOCHS = 25
LR = 1e-4
WD = 1e-2
BATCH = 64
DROPOUT = 0.5
MAX_TRAIN_LOSS = 0.05
MIN_ZONE_AUC = 0.87
E04_TRAIN_LOSS = 0.0012
E04_ZONE_AUC = 0.8590
CACHE = PROJECT_ROOT / "data/processed/features_resnet18/trainval_gray224.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0.5 duzenlilestirilmis ince ayar")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--wd", type=float, default=WD, help="Adam weight decay")
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--aug", choices=["none", "hflip", "full"], default="full",
                   help="none: artirma yok | hflip: yalniz yatay cevirme | full: + crop&resize + parlaklik")
    p.add_argument("--tag", type=str, default="", help="rapor etiketi")
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


def load_cache(sub: pd.DataFrame, zones_dir: Path, reuse: bool) -> np.ndarray:
    if reuse and CACHE.exists():
        arr = np.load(CACHE)
        # KRITIK: onbellek E0.4 ile AYNI satir kumesine ait olmali; aksi halde
        # goruntuler etiketlerle KAYAR ve model sans duzeyinde kalir.
        if len(arr) != len(sub):
            raise ValueError(
                f"Onbellek satir sayisi uyusmuyor: onbellek {len(arr)}, beklenen {len(sub)}. "
                f"Onbellegi silip yeniden uretin.")
        print(f"Onbellek yuklendi: {arr.shape}")
        return arr
    t0 = time.time()
    arr = np.zeros((len(sub), INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    for i, row in enumerate(sub.itertuples(index=False)):
        p = zones_dir / row.patient_id / ZONE_FILES[int(row.zone_id)]
        if p.exists():
            arr[i] = preprocess(p)
    print(f"  On-isleme: {len(sub)} zon, {time.time()-t0:.0f} s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, arr)
    return arr


def raw_batch(gray_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B,224,224) uint8 -> (B,3,224,224) float [0,1], cihazda."""
    x = torch.from_numpy(gray_u8).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    return x.to(device)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, level: str = "full") -> torch.Tensor:
    """level: 'none' | 'hflip' (sadece yatay cevirme) | 'full' (+ crop&resize + parlaklik)."""
    if level == "none":
        return x
    b, _, h, w = x.shape
    flip = torch.rand(b, device=x.device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])
    if level == "hflip":
        return x
    f = random.uniform(0.85, 1.0)
    ch, cw = max(8, int(h * f)), max(8, int(w * f))
    top, left = random.randint(0, h - ch), random.randint(0, w - cw)
    x = TF.resize(x[:, :, top:top + ch, left:left + cw], [h, w], antialias=True)
    x = x * (1.0 + 0.2 * torch.randn(b, 1, 1, 1, device=x.device)) \
          + 0.1 * torch.randn(b, 1, 1, 1, device=x.device)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def predict(model: nn.Module, gray: np.ndarray, idx: np.ndarray, device: torch.device,
            bs: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        outs.append(torch.sigmoid(model(normalize(raw_batch(gray[b], device))).squeeze(-1)).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty(0)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    # E0.4 ile AYNI satir kumesi (test dahil) — onbellek bununla uretildi
    need = zs["is_train_balanced_sample"].values.astype(bool) | (zs["split"] == "val").values | (zs["split"] == "test").values
    sub = zs[need].reset_index(drop=True)
    y = sub["zone_label"].values.astype(np.float32)

    print("=" * 78)
    print("E0.5 — DUZENLILESTIRILMIS ResNet18 INCE AYARI (asiri uyum giderme)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | epochs={EPOCHS} lr={LR} wd={WD} dropout={DROPOUT} | artirma: ACIK")
    print(f"E0.4 (asiri uyumlu): train loss={E04_TRAIN_LOSS} | zon AUC={E04_ZONE_AUC}")
    print("TEST SETINE DOKUNULMUYOR.")

    gray = load_cache(sub, args.zones_dir, args.reuse_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Cihaz: {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    tr = sub["is_train_balanced_sample"].values.astype(bool)
    va = (sub["split"] == "val").values
    vt = va & sub["is_zone_metric_sample"].values.astype(bool)
    idx_tr, idx_vt = np.where(tr)[0], np.where(vt)[0]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(model.fc.in_features, 1))
    model.to(device)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Egitilebilir parametre: {n_train_p:,} | wd={args.wd} dropout={args.dropout} aug={args.aug}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=args.wd)
    crit = nn.BCEWithLogitsLoss()

    best_auc, best_state, final_loss = -1.0, None, float("nan")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        rng = np.random.RandomState(SEED + ep)
        rng.shuffle(idx_tr)
        losses = []
        for i in range(0, len(idx_tr), BATCH):
            b = idx_tr[i:i + BATCH]
            xb = normalize(augment(raw_batch(gray[b], device), args.aug))
            yb = torch.tensor(y[b], device=device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses))
        model.eval()
        pv = predict(model, gray, idx_vt, device)
        a = roc_auc_score(y[idx_vt], pv)
        if a > best_auc:
            best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}/{EPOCHS} | train loss={final_loss:.4f} | val zon AUC={a:.4f} "
              f"| en iyi={best_auc:.4f} | {time.time()-t0:.0f} s", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    pv_all = predict(model, gray, np.where(va)[0], device)
    tmp = sub.loc[va, ["patient_id"]].copy(); tmp["p"] = pv_all
    pdf = tmp.groupby("patient_id")["p"].max()
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    val_patient_auc = float(roc_auc_score(meta["patient_label"].values, pdf.reindex(meta.index).values))

    no_overfit = final_loss > MAX_TRAIN_LOSS
    better_repr = best_auc >= MIN_ZONE_AUC
    passed = no_overfit and better_repr
    decision = "KEEP" if passed else "KILL"

    print(f"\n--- SONUC ---")
    print(f"  son train loss        : {final_loss:.4f}  (esik > {MAX_TRAIN_LOSS}) -> {no_overfit}")
    print(f"  zon duzeyi val AUC    : {best_auc:.4f}  (esik >= {MIN_ZONE_AUC}) -> {better_repr}")
    print(f"  HASTA duzeyi val AUC  : {val_patient_auc:.4f}")
    print(f"  E0.4 karsilastirma    : train loss {E04_TRAIN_LOSS} -> {final_loss:.4f} | "
          f"zon AUC {E04_ZONE_AUC} -> {best_auc:.4f}")
    print(f"\n  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Asiri uyum giderilir (son train loss > {MAX_TRAIN_LOSS}) VE zon AUC >= {MIN_ZONE_AUC}.",
        "scope": "validation only; test setine dokunulmadi",
        "config": {"epochs": EPOCHS, "lr": LR, "weight_decay": args.wd, "dropout": args.dropout,
                   "augmentation": args.aug, "batch": BATCH},
        "final_train_loss": final_loss,
        "zone_val_auc": float(best_auc),
        "patient_val_auc": val_patient_auc,
        "e04_reference": {"train_loss": E04_TRAIN_LOSS, "zone_auc": E04_ZONE_AUC},
        "criteria": {"no_overfit": bool(no_overfit), "better_repr": bool(better_repr)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG + (f"_{args.tag}" if args.tag else ""))
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=decision,
        metrics={"final_train_loss": final_loss, "zone_val_auc": float(best_auc),
                 "patient_val_auc": val_patient_auc},
        notes=f"Duzenlilestirilmis ince ayar: train loss={final_loss:.4f}, zon AUC={best_auc:.4f}, "
              f"hasta AUC={val_patient_auc:.4f}; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
