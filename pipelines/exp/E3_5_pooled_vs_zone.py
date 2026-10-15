"""E3.5 — Zon-özel modelleme gerekli mi? Havuzlanmış tek model vs 5-zon modeli.

Neden bu deney:
    E4.7, zon-bazlı modellerin nodülü **lokalize ETMEDİĞİNİ** gösterdi (zon düzeyi AUC
    ~0.48–0.53). Bu durumda "her zon için ayrı model" mimarisinin bir katkısı olup
    olmadığı sorgulanmalı: tek bir **havuzlanmış** (zon kimliğini yok sayan) model
    aynı başarımı veriyorsa, zon-özel modelleme gereksiz karmaşıklıktır.

Ön-kayıtlı hipotez (EŞDEĞERLİK):
    Havuzlanmış tek modelin ortalama val ROC-AUC'si, 5-zon modelinden **0.015'ten az**
    sapar (yani zon-özel modelleme ölçülebilir bir katkı sağlamaz).

Tasarım:
    Tek değişken: modelleme şeması (5 ayrı zon modeli vs tek havuzlanmış model).
    Özellikler, split, mimari (SmallZoneCNN), eğitim protokolü, seed'ler **sabit**.
    Karşılaştırma referansı: E2.8'in 3-seed ortalaması (**0.8697**, σ=1.0/ızgara=20).

Kapsam (test bütçesi korunur):
    Yalnızca **validation**; test setine dokunulmaz.

Kullanım:
    python pipelines/exp/E3_5_pooled_vs_zone.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E3.5"
SLUG = "pooled_vs_zone"
SEEDS = [42, 43, 44]
TOLERANCE = 0.015
# E2.8 ile AYNI protokol/seed'lerle olculen 5-zon referansi (3 seed ortalamasi)
ZONE_REFERENCE_MEAN = 0.8697
ZONE_REFERENCE_SEED_AUCS = [0.8736, 0.8546, 0.8701]   # E2.8 s1.0_g20 seed 42/43/44
MAX_EPOCHS = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.5 havuzlanmis vs zon-ozel model")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--features-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    return p.parse_args()


def train_pooled(feats: np.ndarray, zs: pd.DataFrame, seed: int) -> Dict[str, Any]:
    """Zon kimliğini YOK SAYAN tek model; hasta skoru = zon olasılıklarının maksimumu."""
    y = zs["zone_label"].values.astype(np.float32)
    tr = zs["is_train_balanced_sample"].values.astype(bool)
    va_all = (zs["split"] == "val").values
    va_metric = va_all & zs["is_zone_metric_sample"].values.astype(bool)

    scaler = StandardScaler().fit(feats[tr])
    Xtr, Xvm, Xva = scaler.transform(feats[tr]), scaler.transform(feats[va_metric]), scaler.transform(feats[va_all])

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallZoneCNN(in_channels=2, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32),
                                      torch.tensor(y[tr], dtype=torch.float32)),
                        batch_size=32, shuffle=True)
    Xvm_t = torch.tensor(vector_to_2channel_image(Xvm), dtype=torch.float32)

    best_auc, best_state = -1.0, None
    for _ in range(MAX_EPOCHS):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xvm_t)).numpy()
        a = compute_binary_metrics(y[va_metric], p)["roc_auc"]
        if a > best_auc:
            best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(torch.tensor(
            vector_to_2channel_image(Xva), dtype=torch.float32))).numpy()

    tmp = zs.loc[va_all, ["patient_id", "zone_id"]].copy()
    tmp["p"] = pv
    pdf = tmp.groupby("patient_id")["p"].max()
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv_patient = pdf.reindex(meta.index)
    thr, _ = find_optimal_threshold(meta["patient_label"].values, pv_patient.values, method="youden")
    return {
        "val_patient_auc": float(compute_binary_metrics(meta["patient_label"].values, pv_patient.values)["roc_auc"]),
        "val_zone_auc": float(best_auc),
        "patient_threshold": float(thr),
        "n_train": int(tr.sum()),
    }


def main() -> None:
    args = parse_args()
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    feats = np.load(args.features_path)

    print("=" * 78)
    print("E3.5 — HAVUZLANMIS TEK MODEL vs 5-ZON OZEL MODELI")
    print("=" * 78)
    print(f"Ozellikler: {feats.shape} | seed'ler: {SEEDS} | TEST SETINE DOKUNULMUYOR")
    print(f"5-zon referansi (E2.8, ayni protokol): {ZONE_REFERENCE_MEAN:.4f}")

    pooled = {}
    for seed in SEEDS:
        r = train_pooled(feats, zs, seed)
        pooled[seed] = r
        print(f"  seed={seed}: havuzlanmis val hasta AUC = {r['val_patient_auc']:.4f} "
              f"(zon AUC {r['val_zone_auc']:.4f}, n_train={r['n_train']})", flush=True)

    pooled_aucs = [pooled[s]["val_patient_auc"] for s in SEEDS]
    pooled_mean = float(np.mean(pooled_aucs))
    zone_mean = float(np.mean(ZONE_REFERENCE_SEED_AUCS))
    delta = pooled_mean - zone_mean

    print("\n--- Karsilastirma (3 seed ortalamasi, hasta duzeyi val ROC-AUC) ---")
    print(f"  5-zon ozel modeli : {zone_mean:.4f}  (seed'ler {ZONE_REFERENCE_SEED_AUCS})")
    print(f"  Havuzlanmis model : {pooled_mean:.4f}  (seed'ler {[round(a,4) for a in pooled_aucs]})")
    print(f"  Fark              : {delta:+.4f}")

    equivalent = abs(delta) < TOLERANCE
    decision = "KEEP" if equivalent else "KILL"
    print(f"\n  Esdegerlik kriteri: |fark| < {TOLERANCE} -> {equivalent}")
    print(f"  KARAR: {decision}")
    if equivalent:
        print("  => Sonuc: zon-ozel modelleme OLCULEBILIR bir katki saglamiyor;"
              " mimari 'havuzlanmis tek model + max-pooling' olarak basitlestirilebilir.")
    else:
        print("  => Sonuc: zon-ozel modelleme olculebilir katki sagliyor.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Havuzlanmis tek model, 5-zon modelinden 0.015'ten az sapar (esdegerlik).",
        "scope": "validation only; test setine dokunulmadi",
        "seeds": SEEDS, "tolerance": TOLERANCE,
        "zone_specialized": {"mean_val_auc": zone_mean, "seed_aucs": ZONE_REFERENCE_SEED_AUCS,
                             "source": "E2.8 s1.0_g20 (ayni protokol)"},
        "pooled": {"mean_val_auc": pooled_mean, "seed_aucs": pooled_aucs,
                   "per_seed": {str(s): pooled[s] for s in SEEDS}},
        "delta_pooled_minus_zone": delta,
        "criteria": {"abs_delta_lt_tolerance": bool(equivalent)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.features_path],
        seed=42, decision=decision,
        metrics={"pooled_mean_val_auc": pooled_mean, "zone_mean_val_auc": zone_mean, "delta": delta},
        notes=f"Havuzlanmis vs zon-ozel: fark {delta:+.4f}; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
