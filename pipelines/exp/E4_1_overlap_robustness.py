"""E4.1 — Etiketleme kuralı sağlamlık (robustness) analizi: %10 / %20 / %30 örtüşme.

Ön-kayıtlı hipotez:
    %20 piksel-örtüşme kuralı ile elde edilen model sıralaması/sonuçları, %10 ve
    %30 kuralları altında **anlamlı biçimde değişmez** (val ROC-AUC farkı < 0.015).

Tek değişken:
    Zon pozitiflik eşiği `t` (bounding-box piksel örtüşme oranı).
    **Hasta split'i DONDURULMUŞTUR** (splits_v2/patient_splits.csv) → aynı
    train/val/test hastaları; yalnızca zon etiketleri ve 1:1 dengeli alt küme
    yeniden türetilir.

Kapsam (test bütçesi korunur):
    Karar **yalnızca validation** ROC-AUC üzerinden verilir; **test setine
    dokunulmaz**.

Dayanak:
    literature.md §5.3 / roadmap §12 — hakemin etiketleme kuralına dair sorusu.

Kullanım:
    python pipelines/exp/E4_1_overlap_robustness.py
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

from node21_tda.data.labels_v2 import LABEL_RULE_VERSION, generate_zone_labels_v2  # noqa: E402
from node21_tda.data.zone_splits_v2 import build_zone_splits_v2  # noqa: E402
from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E4.1"
SLUG = "overlap_robustness"
SEED = 42
THRESHOLDS = [0.10, 0.20, 0.30]
BASELINE_THRESHOLD = 0.20
TOLERANCE = 0.015                     # bu kadar fark "anlamli degisim" sayilir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4.1 overlap kurali saglamlik analizi")
    p.add_argument("--overlap-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/reports/nodule_pixel_overlap.csv")
    p.add_argument("--patients-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--features-meta-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/features_metadata.csv")
    p.add_argument("--features-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--patient-splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv")
    p.add_argument("--max-epochs", type=int, default=40)
    return p.parse_args()


def train_eval(feats: np.ndarray, zs: pd.DataFrame, max_epochs: int, seed: int) -> Dict[str, Any]:
    """Zon splitleri uzerinde 5-zon CNN egitir; val hasta ROC-AUC ve esikleri doner."""
    y_zone = zs["zone_label"].values.astype(np.float32)
    val_preds: List[Dict[str, Any]] = []
    zone_auc: Dict[str, float] = {}

    for zone_id in range(1, 6):
        zm = (zs["zone_id"] == zone_id).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va_m = zm & (zs["split"] == "val").values
        va_all = va_m & zs["is_zone_metric_sample"].values.astype(bool)

        scaler = StandardScaler().fit(feats[tr])
        Xtr = scaler.transform(feats[tr])
        Xva = scaler.transform(feats[va_all])
        Xva_all = scaler.transform(feats[va_m])
        ytr = torch.tensor(y_zone[tr], dtype=torch.float32)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32), ytr),
            batch_size=32, shuffle=True)
        Xva_t = torch.tensor(vector_to_2channel_image(Xva), dtype=torch.float32)

        best_auc, best_state = -1.0, None
        for _ in range(max_epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xva_t)).numpy()
            a = compute_binary_metrics(y_zone[va_all], p)["roc_auc"]
            if a > best_auc:
                best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv_all = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(Xva_all), dtype=torch.float32))).numpy()
        zone_auc[f"zone_{zone_id}"] = float(best_auc)
        for pid, p in zip(zs.loc[va_m, "patient_id"], pv_all):
            val_preds.append({"patient_id": pid, "p": float(p)})

    pdf = pd.DataFrame(val_preds).groupby("patient_id")["p"].max()
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv = pdf.reindex(meta.index)
    yv = meta["patient_label"].values
    thr, _ = find_optimal_threshold(yv, pv.values, method="youden")

    per_zone = {}
    for zone_id in range(1, 6):
        z = zs[zs["zone_id"] == zone_id]
        per_zone[f"zone_{zone_id}"] = {
            "train_positive": int(((z["split"] == "train") & (z["label_state"] == "positive")).sum()),
            "train_clean_negative_pool": int(((z["split"] == "train") & (z["label_state"] == "negative")).sum()),
            "train_balanced_total": int(z["is_train_balanced_sample"].sum()),
            "val_positive": int(((z["split"] == "val") & (z["label_state"] == "positive")).sum()),
            "val_ambiguous": int(((z["split"] == "val") & (z["label_state"] == "ambiguous")).sum()),
        }
    return {
        "val_patient_auc": float(compute_binary_metrics(yv, pv.values)["roc_auc"]),
        "val_patient_threshold": float(thr),
        "zone_val_auc": zone_auc,
        "per_zone": per_zone,
        "n_train_balanced": int(zs["is_train_balanced_sample"].sum()),
    }


def main() -> None:
    args = parse_args()
    feats = np.load(args.features_path)
    base_meta = pd.read_csv(args.features_meta_path)
    patient_splits = pd.read_csv(args.patient_splits_path)

    print("=" * 78)
    print("E4.1 — ETIKETLEME KURALI SAGLAMLIK ANALIZI (%10 / %20 / %30)")
    print("=" * 78)
    print(f"Ozellikler: {feats.shape} | hasta split'i DONDURULMUS ({len(patient_splits)} hasta)")
    print("Karar yalnizca VALIDATION uzerinden; TEST SETINE DOKUNULMUYOR.")

    results: Dict[str, Any] = {}
    for t in THRESHOLDS:
        key = f"overlap{int(round(t*100)):02d}"
        print(f"\n--- Esik %{int(round(t*100))} ---")
        labels, rep = generate_zone_labels_v2(
            overlap_csv_path=args.overlap_path,
            patients_dir=args.patients_dir,
            threshold=t,
            rule_version=f"v2_overlap{int(round(t*100))}",
        )
        meta = base_meta.drop(columns=["zone_label", "label_state", "max_bbox_overlap", "label_rule_version"]).merge(
            labels[["patient_id", "zone_id", "zone_label", "label_state", "max_bbox_overlap", "label_rule_version"]],
            on=["patient_id", "zone_id"], how="left", validate="one_to_one")
        if meta["label_state"].isna().any():
            raise ValueError(f"{key}: etiket birlestirmesi basarisiz")

        zs, _ = build_zone_splits_v2(features_meta_df=meta, patient_splits_df=patient_splits, seed=SEED)
        res = train_eval(feats, zs, args.max_epochs, SEED)
        res["label_counts"] = {
            "positive": int((zs["label_state"] == "positive").sum()),
            "negative": int((zs["label_state"] == "negative").sum()),
            "ambiguous": int((zs["label_state"] == "ambiguous").sum()),
            "unassigned_nodule": int((zs["label_state"] == "unassigned_nodule").sum()),
        }
        results[key] = res
        print(f"  Etiketler: poz={res['label_counts']['positive']} amb={res['label_counts']['ambiguous']} "
              f"| dengeli train={res['n_train_balanced']}")
        print(f"  VAL hasta ROC-AUC = {res['val_patient_auc']:.4f}")

    base_key = f"overlap{int(round(BASELINE_THRESHOLD*100)):02d}"
    base_auc = results[base_key]["val_patient_auc"]
    print("\n--- Ozet (val hasta ROC-AUC) ---")
    deltas = {}
    for key, r in results.items():
        d = r["val_patient_auc"] - base_auc
        deltas[key] = d
        print(f"  {key}: {r['val_patient_auc']:.4f}  (%20'ye fark {d:+.4f})")

    max_abs_delta = max(abs(d) for k, d in deltas.items() if k != base_key)
    stable = max_abs_delta < TOLERANCE
    decision = "KEEP" if stable else "KILL"
    print(f"\n  En buyuk |fark| = {max_abs_delta:.4f} | tolerans {TOLERANCE} -> {'SAGLAM' if stable else 'DEGISKEN'}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "%20 kuraliyla elde edilen sonuclar %10 ve %30 altinda anlamli degismez (< 0.015).",
        "design": "Hasta split'i DONDURULMUS; yalnizca zon etiketleri ve dengeli alt kume yeniden turetildi.",
        "scope": "validation only; test setine dokunulmadi",
        "thresholds": THRESHOLDS,
        "baseline_threshold": BASELINE_THRESHOLD,
        "tolerance": TOLERANCE,
        "results": results,
        "deltas_vs_baseline": deltas,
        "max_abs_delta": max_abs_delta,
        "criteria": {"max_abs_delta_lt_tolerance": bool(stable)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.features_path, args.features_meta_path,
                                        args.patient_splits_path, args.overlap_path],
        seed=SEED, decision=decision,
        metrics={f"val_auc_{k}": v["val_patient_auc"] for k, v in results.items()},
        notes=f"Esik saglamligi: en buyuk |fark|={max_abs_delta:.4f} (tolerans {TOLERANCE}); karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
