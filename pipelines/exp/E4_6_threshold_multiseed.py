"""E4.6 — %10 vs %20 örtüşme kuralı: çok-seed (eşleştirilmiş) doğrulama.

Neden bu deney:
    E4.1, %10 kuralının val ROC-AUC'yi **+0.0186** artırdığını gösterdi — ama bu
    **tek seed**'lik bir ölçümdü ve CNN eğitim gürültüsü ±0.01–0.02 mertebesinde
    olabilir. Bu deney, aynı farkı **eşleştirilmiş (paired) çok-seed** tasarımıyla
    doğrular veya çürütür.

Ön-kayıtlı hipotez:
    %10 kuralı, %20'ye kıyasla **ortalama val ROC-AUC'yi ≥ +0.015 artırır** ve
    bu fark seed'ler boyunca **tutarlıdır** (5 seed'in en az 4'ünde pozitif).

Tek değişken:
    Zon pozitiflik eşiği (0.10 vs 0.20). Hasta split'i **dondurulmuştur**.
    Her seed hem 1:1 dengeleme seçimini hem CNN eğitim tohumunu kontrol eder.

Kapsam (test bütçesi korunur):
    Yalnızca **validation**; test setine dokunulmaz.

Kullanım:
    python pipelines/exp/E4_6_threshold_multiseed.py
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

from node21_tda.data.labels_v2 import generate_zone_labels_v2  # noqa: E402
from node21_tda.data.zone_splits_v2 import build_zone_splits_v2  # noqa: E402
from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E4.6"
SLUG = "threshold_multiseed"
SEEDS = [42, 43, 44, 45, 46]
RULES = {"overlap10": 0.10, "overlap20": 0.20}
BASELINE_RULE = "overlap20"
TOLERANCE = 0.015


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4.6 %10 vs %20 cok-seed dogrulama")
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


def train_val_auc(feats: np.ndarray, zs: pd.DataFrame, max_epochs: int, seed: int) -> float:
    """5-zon CNN egitir; val hasta ROC-AUC'sini doner (agirlikli ortalama degil, max-pooling)."""
    y_zone = zs["zone_label"].values.astype(np.float32)
    val_preds: List[Dict[str, Any]] = []

    for zone_id in range(1, 6):
        zm = (zs["zone_id"] == zone_id).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va_all = zm & (zs["split"] == "val").values
        va_metric = va_all & zs["is_zone_metric_sample"].values.astype(bool)

        scaler = StandardScaler().fit(feats[tr])
        Xtr = scaler.transform(feats[tr])
        Xvm = scaler.transform(feats[va_metric])
        Xva = scaler.transform(feats[va_all])

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32),
                          torch.tensor(y_zone[tr], dtype=torch.float32)),
            batch_size=32, shuffle=True)
        Xvm_t = torch.tensor(vector_to_2channel_image(Xvm), dtype=torch.float32)

        best_auc, best_state = -1.0, None
        for _ in range(max_epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xvm_t)).numpy()
            a = compute_binary_metrics(y_zone[va_metric], p)["roc_auc"]
            if a > best_auc:
                best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(Xva), dtype=torch.float32))).numpy()
        for pid, p in zip(zs.loc[va_all, "patient_id"], pv):
            val_preds.append({"patient_id": pid, "p": float(p)})

    pdf = pd.DataFrame(val_preds).groupby("patient_id")["p"].max()
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv = pdf.reindex(meta.index)
    return float(compute_binary_metrics(meta["patient_label"].values, pv.values)["roc_auc"])


def main() -> None:
    args = parse_args()
    feats = np.load(args.features_path)
    base_meta = pd.read_csv(args.features_meta_path)
    patient_splits = pd.read_csv(args.patient_splits_path)

    print("=" * 78)
    print("E4.6 — %10 vs %20 ORTUSME KURALI: COK-SEED (ESLESTIRILMIS) DOGRULAMA")
    print("=" * 78)
    print(f"Seed'ler: {SEEDS} | Kuralar: {list(RULES)} | hasta split'i DONDURULMUS")
    print("Karar yalnizca VALIDATION uzerinden; TEST SETINE DOKUNULMUYOR.")

    # Kurallar icin etiket + metadata bir kez hazirlanir
    rule_meta: Dict[str, pd.DataFrame] = {}
    for rule, t in RULES.items():
        labels, _ = generate_zone_labels_v2(
            overlap_csv_path=args.overlap_path, patients_dir=args.patients_dir,
            threshold=t, rule_version=f"v2_overlap{int(round(t*100))}")
        meta = base_meta.drop(columns=["zone_label", "label_state", "max_bbox_overlap", "label_rule_version"]).merge(
            labels[["patient_id", "zone_id", "zone_label", "label_state", "max_bbox_overlap", "label_rule_version"]],
            on=["patient_id", "zone_id"], how="left", validate="one_to_one")
        if meta["label_state"].isna().any():
            raise ValueError(f"{rule}: etiket birlestirmesi basarisiz")
        rule_meta[rule] = meta
        print(f"  {rule}: pozitif zon = {int((meta['label_state']=='positive').sum())}")

    per_seed: Dict[str, Dict[int, float]] = {rule: {} for rule in RULES}
    for seed in SEEDS:
        line = [f"  seed={seed}"]
        for rule in RULES:
            zs, _ = build_zone_splits_v2(features_meta_df=rule_meta[rule],
                                         patient_splits_df=patient_splits, seed=seed)
            a = train_val_auc(feats, zs, args.max_epochs, seed)
            per_seed[rule][seed] = a
            line.append(f"{rule}={a:.4f}")
        print(" | ".join(line), flush=True)

    deltas = [per_seed["overlap10"][s] - per_seed[BASELINE_RULE][s] for s in SEEDS]
    mean10 = float(np.mean([per_seed["overlap10"][s] for s in SEEDS]))
    mean20 = float(np.mean([per_seed[BASELINE_RULE][s] for s in SEEDS]))
    mean_delta = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))

    print("\n--- Eslestirilmis farklar (%10 - %20) ---")
    for s, d in zip(SEEDS, deltas):
        print(f"  seed={s}: {d:+.4f}")
    print(f"\n  Ortalama: %10={mean10:.4f} | %20={mean20:.4f} | fark={mean_delta:+.4f} "
          f"(std={np.std(deltas):.4f})")
    print(f"  Pozitif seed sayisi: {n_pos}/{len(SEEDS)}")

    passed = mean_delta >= TOLERANCE and n_pos >= 4
    decision = "KEEP" if passed else "KILL"
    print(f"\n  Kriter: ortalama fark >= +{TOLERANCE} -> {mean_delta >= TOLERANCE} | "
          f"pozitif seed >= 4/5 -> {n_pos >= 4}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "%10 kurali %20'ye kiyasla ortalama val ROC-AUC'yi >= +0.015 artirir ve 5 seed'in >=4'unde pozitiftir.",
        "design": "Eslestirilmis (paired) cok-seed; hasta split'i dondurulmus; seed hem dengelemeyi hem egitimi kontrol eder.",
        "scope": "validation only; test setine dokunulmadi",
        "seeds": SEEDS,
        "rules": RULES,
        "per_seed": {rule: {str(s): v for s, v in per_seed[rule].items()} for rule in RULES},
        "paired_deltas": {str(s): d for s, d in zip(SEEDS, deltas)},
        "mean_overlap10": mean10, "mean_overlap20": mean20,
        "mean_delta": mean_delta, "std_delta": float(np.std(deltas)), "n_positive_seeds": n_pos,
        "criteria": {"mean_delta_ge_0.015": bool(mean_delta >= TOLERANCE), "n_positive_ge_4": bool(n_pos >= 4)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID,
        inputs=[args.features_path, args.features_meta_path, args.patient_splits_path, args.overlap_path],
        seed=42, decision=decision,
        metrics={"mean_overlap10": mean10, "mean_overlap20": mean20, "mean_delta": mean_delta,
                 "n_positive_seeds": n_pos},
        notes=f"Eslestirilmis cok-seed: ortalama fark={mean_delta:+.4f}, {n_pos}/5 pozitif; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
