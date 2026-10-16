"""E2.2 — Çok ölçekli Kalıcılık İmgesi (multi-scale PI).

Neden bu deney:
    E2.8, PI'ın Gaussian genişliğine duyarlı olduğunu gösterdi: σ = 0.5 → 0.8460,
    **1.0 → 0.8697**, 1.5 → 0.8734, 2.0 → 0.8712. Tek bir σ seçmek zorunda değiliz —
    **birden çok ölçekte PI hesaplayıp birleştirmek** farklı ölçeklerdeki topolojik
    yapıyı aynı temsilde kodlayabilir. Faz 3'ün kalan adayı.

Ön-kayıtlı hipotez:
    Çok ölçekli temsil (σ = 0.5/1.0/1.5/2.0, 3200 boyut), tek ölçekli PI'ın (σ=1.0,
    800 boyut) **val** ROC-AUC'sini **≥ +0.015** geçer.

Tasarım (E2.1'deki dersler uygulanır):
    * A  = tek ölçek, σ=1.0 (projenin varsayılanı, 800)
    * A' = tek ölçek, σ=1.5 (E2.8'de en iyi tek ölçek, 800)
    * B  = çok ölçek, concat(0.5, 1.0, 1.5, 2.0) = 3200
    * C  = **BOYUT KONTROLÜ**: concat(σ=1.0, diğer üç ölçeğin KARIŞTIRILMIŞ hâli) = 3200
           → boyut aynı, ek ölçeklerin bilgisi yok. E2.1b ile aynı mantık.
    Sınıflandırıcı, split, zon yapısı ve hasta max-pooling **sabit** (RBF-SVM C=10).

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.
    Özellikler E2.8 önbelleğinden okunur; yeniden çıkarım YOK.

Kullanım:
    python pipelines/exp/E2_2_multiscale_pi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "E2.2"
SLUG = "multiscale_pi"
SEED = 42
MIN_DELTA = 0.015
FEATURE_DIR = PROJECT_ROOT / "data/processed/features_sigma/E2_8"
SCALES = ["s0.5_g20", "s1.0_g20", "s1.5_g20", "s2.0_g20"]
BASE = "s1.0_g20"
BEST_SINGLE = "s1.5_g20"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.2 cok olcekli PI")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def svm_val_patient_auc(X: np.ndarray, zs: pd.DataFrame, meta: pd.DataFrame) -> float:
    y = zs["zone_label"].values
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va = zm & (zs["split"] == "val").values
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clf.fit(X[tr], y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[va, "patient_id"].values,
                                  "p": clf.predict_proba(X[va])[:, 1]}))
    pdf = pd.concat(rows).groupby("patient_id")["p"].max().reindex(meta.index)
    return float(roc_auc_score(meta["patient_label"].values, pdf.values))


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = zs["is_train_balanced_sample"].values.astype(bool) | (zs["split"] == "val").values
    sub = zs[need].reset_index(drop=True)
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")

    print("=" * 78)
    print("E2.2 — COK OLCEKLI PI vs TEK OLCEKLI PI")
    print("=" * 78)
    print(f"Zon: {len(sub)} | olcekler: {SCALES} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR. Ozellikler E2.8 onbelleginden okunur.")

    idx = np.load(FEATURE_DIR / "vector_index.npy")
    pos = {int(v): i for i, v in enumerate(idx)}
    order = np.array([pos[int(v)] for v in sub["vector_index"].values])
    feats: Dict[str, np.ndarray] = {}
    for s in SCALES:
        a = np.load(FEATURE_DIR / f"{s}.npy")
        feats[s] = a[order]
        print(f"  {s}: {feats[s].shape}")

    X_A = feats[BASE]
    X_Ap = feats[BEST_SINGLE]
    rng = np.random.RandomState(SEED)
    others = [s for s in SCALES if s != BASE]
    X_C = np.concatenate([X_A] + [feats[s][rng.permutation(len(sub))] for s in others], axis=1)
    X_B = np.concatenate([feats[s] for s in SCALES], axis=1)
    print(f"  Temsiller: A {X_A.shape} | A' {X_Ap.shape} | B {X_B.shape} | C(kontrol) {X_C.shape}")

    print("\n  SVM (A: tek olcek sigma=1.0)...", flush=True)
    auc_A = svm_val_patient_auc(X_A, sub, meta)
    print(f"    A  = {auc_A:.4f}", flush=True)
    print("  SVM (A': tek olcek sigma=1.5)...", flush=True)
    auc_Ap = svm_val_patient_auc(X_Ap, sub, meta)
    print(f"    A' = {auc_Ap:.4f}", flush=True)
    print("  SVM (C: BOYUT KONTROLU, karistirilmis)...", flush=True)
    auc_C = svm_val_patient_auc(X_C, sub, meta)
    print(f"    C  = {auc_C:.4f}", flush=True)
    print("  SVM (B: cok olcek)...", flush=True)
    auc_B = svm_val_patient_auc(X_B, sub, meta)
    print(f"    B  = {auc_B:.4f}", flush=True)

    best_single = max(auc_A, auc_Ap)
    delta = auc_B - best_single
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    print(f"  A  tek olcek (sigma=1.0)        : {auc_A:.4f}  (800 boyut)")
    print(f"  A' tek olcek (sigma=1.5, en iyi): {auc_Ap:.4f}  (800 boyut)")
    print(f"  B  COK OLCEK                    : {auc_B:.4f}  ({X_B.shape[1]} boyut)")
    print(f"  C  boyut kontrolu (karistirilmis): {auc_C:.4f}  ({X_C.shape[1]} boyut)")
    print(f"  B - en iyi tek olcek            : {delta:+.4f}   (kriter >= +{MIN_DELTA} -> {passed})")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Cok olcek ek bilgi katmiyor; tek olcek yeterli.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Cok olcekli PI, en iyi tek olcekli PI'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi; ozellikler E2.8 onbelleginden",
        "config": {"scales": SCALES, "classifier": "RBF-SVM C=10 gamma=scale",
                   "dimension_control": "C = base + shuffled copies of the other scales"},
        "single_scale_s1.0": auc_A,
        "single_scale_s1.5": auc_Ap,
        "multiscale": auc_B,
        "dimension_control_shuffled": auc_C,
        "best_single": best_single,
        "delta_multiscale_minus_best_single": delta,
        "feature_dims": {"single": 800, "multiscale": int(X_B.shape[1]), "control": int(X_C.shape[1])},
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG + (f"_{args.tag}" if args.tag else ""))
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=report["decision"],
        metrics={"multiscale_val_auc": auc_B, "best_single_val_auc": best_single, "delta": delta},
        notes=f"Cok olcekli {auc_B:.4f} vs en iyi tek olcek {best_single:.4f}; fark {delta:+.4f}; "
              f"karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
