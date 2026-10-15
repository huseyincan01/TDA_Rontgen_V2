"""E3.2 — Orta seviye füzyon: özellik birleştirme vs geç (karar) füzyonu.

Neden bu deney:
    E3.1 ana sonucumuz **geç füzyon**dur: iki dal ayrı ayrı hasta olasılığı üretir,
    sonra ağırlıklı ortalamayla birleştirilir (val 0.9572). Literatürde **orta seviye
    füzyon** (özellikleri birleştirip tek sınıflandırıcı eğitmek) genellikle daha güçlü
    kabul edilir. E3.2 bunu aynı split ve protokolde test eder — Faz 2'nin son maddesi.

Ön-kayıtlı hipotez:
    Orta seviye füzyon (PI ⊕ ResNet özellikleri → tek lojistik regresyon), geç füzyonun
    **val** ROC-AUC'sini (0.9572) **≥ +0.015** geçer.

Tek değişken:
    Füzyon seviyesi (özellik birleştirme vs karar birleştirme). Özellikler, split,
    zon yapısı ve hasta düzeyi max-pooling **sabit**.

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E3_2_midlevel_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E3.2"
SLUG = "midlevel_fusion"
SEED = 42
TDA_CWT = 0.55
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
LATE_FUSION_VAL_REF = 0.9572
MIN_DELTA = 0.015


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.2 orta seviye füzyon")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--tda-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    p.add_argument("--cnn-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt")
    p.add_argument("--svm-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    return p.parse_args()


def probe_patient_probs(X: np.ndarray, zs: pd.DataFrame, meta: pd.DataFrame,
                        tag: str) -> pd.Series:
    """Zon bazli lojistik regresyon sondasi -> hasta max-pooling (val)."""
    y = zs["zone_label"].values
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va = zm & (zs["split"] == "val").values
        vm = va & zs["is_zone_metric_sample"].values.astype(bool)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        best = (-1.0, None)
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit(Xtr, y[tr])
            a = roc_auc_score(y[vm], clf.predict_proba(sc.transform(X[vm]))[:, 1])
            if a > best[0]:
                best = (a, clf)
        clf = best[1]
        rows.append(pd.DataFrame({"patient_id": zs.loc[va, "patient_id"].values,
                                  "p": clf.predict_proba(Xva)[:, 1]}))
    return pd.concat(rows).groupby("patient_id")["p"].max().reindex(meta.index)


def main() -> None:
    args = parse_args()
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    y_pat = meta["patient_label"].values

    print("=" * 78)
    print("E3.2 — ORTA SEVIYE FUZYON (ozellik birlestirme) vs GEC FUZYON")
    print("=" * 78)
    print(f"Gec fuzyon val referansi: {LATE_FUSION_VAL_REF} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR.")

    # --- Tekil dallar (referans) ---
    p_tda_only = probe_patient_probs(X_tda, zs, meta, "tda")
    p_res_only = probe_patient_probs(X_res, zs, meta, "res")
    auc_tda = float(roc_auc_score(y_pat, p_tda_only.values))
    auc_res = float(roc_auc_score(y_pat, p_res_only.values))

    # --- Orta seviye fuzyon: ozellikleri birlestir ---
    X_cat = np.concatenate([X_tda, X_res], axis=1)
    p_mid = probe_patient_probs(X_cat, zs, meta, "mid")
    auc_mid = float(roc_auc_score(y_pat, p_mid.values))

    # --- Gec fuzyon (ayni kosuda yeniden hesapla; tutarlilik icin) ---
    cb = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    sb = joblib.load(args.svm_bundle)
    drop = float(cb["config"].get("dropout", 0.3))
    rows = []
    for z in range(1, 6):
        m = ((zs["split"] == "val") & (zs["zone_id"] == z)).values
        idx = zs[m]["vector_index"].values
        Xz = X_tda[idx]
        sc = cb["scalers"][z]
        mm = SmallZoneCNN(in_channels=2, dropout=drop)
        mm.load_state_dict(cb["models_state_dict"][z]); mm.eval()
        with torch.no_grad():
            pc = torch.sigmoid(mm(torch.tensor(vector_to_2channel_image(sc.transform(Xz)),
                                                 dtype=torch.float32))).numpy()
        ps = sb.predict_zone_proba(z, Xz)
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": TDA_CWT * pc + (1 - TDA_CWT) * ps}))
    p_tda_ens = pd.concat(rows).groupby("patient_id")["p"].max().reindex(meta.index)
    auc_tda_ens = float(roc_auc_score(y_pat, p_tda_ens.values))

    best = {"a": None, "auc": -1.0}
    for a in A_GRID:
        s = roc_auc_score(y_pat, a * p_res_only.values + (1 - a) * p_tda_ens.values)
        if s > best["auc"]:
            best = {"a": float(a), "auc": float(s)}
    p_late = best["a"] * p_res_only.values + (1 - best["a"]) * p_tda_ens.values
    auc_late = float(roc_auc_score(y_pat, p_late))

    print(f"\n--- VAL sonuclari (734 hasta) ---")
    print(f"  TDA lojistik sondasi (tek)     : {auc_tda:.4f}")
    print(f"  ResNet18 sondasi (tek)         : {auc_res:.4f}")
    print(f"  TDA ensemble (CNN+SVM)         : {auc_tda_ens:.4f}")
    print(f"  GEC fuzyon (a={best['a']:.2f})            : {auc_late:.4f}   (E3.1 referansi {LATE_FUSION_VAL_REF})")
    print(f"  ORTA seviye (PI+ResNet -> LR)  : {auc_mid:.4f}")

    delta = auc_mid - auc_late
    passed = delta >= MIN_DELTA
    print(f"\n  Orta - Gec fark: {delta:+.4f}  (kriter >= +{MIN_DELTA} -> {passed})")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Gec fuzyon yeterli; ozellik birlestirme ek kazanc saglamiyor.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Orta seviye fuzyon, gec fuzyonun val AUC'sini >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi (butce tukendi)",
        "variants": {
            "tda_probe": auc_tda, "resnet_probe": auc_res, "tda_ensemble": auc_tda_ens,
            "late_fusion": {"auc": auc_late, "a": best["a"]},
            "mid_level_concat": auc_mid,
        },
        "late_fusion_reference": LATE_FUSION_VAL_REF,
        "delta_mid_minus_late": delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.tda_features, args.resnet_features],
        seed=SEED, decision=report["decision"],
        metrics={"mid_level_val_auc": auc_mid, "late_fusion_val_auc": auc_late, "delta": delta},
        notes=f"Orta seviye fuzyon {auc_mid:.4f} vs gec fuzyon {auc_late:.4f}; fark {delta:+.4f}; "
              f"karar={report['decision']}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
