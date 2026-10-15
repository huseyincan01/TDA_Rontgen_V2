"""E4.3 — Zon bazlı hata analizi: hangi zonlar kararı sürüklüyor, hangileri zayıf?

Ön-kayıtlı hipotez:
    Hasta düzeyi max-pooling kararı **tek bir zona** bağımlı değildir; zayıf zonlar
    (val ROC-AUC < 0.70) vardır ve yanlış negatiflerin çoğundan sorumludur.

Ölçütler:
    1. Zon bazlı hasta düzeyi val ROC-AUC (her zon TEK BAŞINA kullanılırsa).
    2. "Sorumlu zon" dağılımı: her hasta için argmax zonu, sonuca göre (TP/FP/FN/TN).
    3. Zayıf zon eşiği: < 0.70 olan zon sayısı.
    4. Yanlış negatiflerde (FN) maksimum olasılık dağılımı.

Kapsam (test bütçesi korunur):
    Yalnızca **validation**; test setine dokunulmaz. Bu betik yeni model EĞİTMEZ —
    kayıtlı paketlerden çıkarım yapar (açıklayıcı analiz, model seçimi değil).

Kullanım:
    python pipelines/exp/E4_3_zone_error_analysis.py
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

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E4.3"
SLUG = "zone_error_analysis"
SEED = 42
WEAK_ZONE_AUC = 0.70
ZONE_NAMES = {1: "Sağ Üst", 2: "Sağ Orta", 3: "Sağ Alt", 4: "Sol Üst", 5: "Sol Alt"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4.3 zon bazli hata analizi")
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


def val_zone_probs(args) -> Dict[str, pd.DataFrame]:
    """VAL zonları için CNN / SVM / ResNet olasılıkları (patient_id, zone_id, p)."""
    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)
    cb = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    sb = joblib.load(args.svm_bundle)
    dropout = float(cb["config"].get("dropout", 0.3))

    rows: Dict[str, List[Dict[str, Any]]] = {"cnn": [], "svm": [], "resnet": []}
    y_zone = splits["zone_label"].values
    train_m = splits["is_train_balanced_sample"].values.astype(bool)

    for z in range(1, 6):
        zm = (splits["zone_id"] == z).values
        va = zm & (splits["split"] == "val").values
        tr = zm & train_m

        # CNN
        sc = cb["scalers"][z]
        Xs = sc.transform(X_tda[va])
        model = SmallZoneCNN(in_channels=2, dropout=dropout)
        model.load_state_dict(cb["models_state_dict"][z])
        model.eval()
        with torch.no_grad():
            p_cnn = torch.sigmoid(model(torch.tensor(vector_to_2channel_image(Xs), dtype=torch.float32))).numpy()
        p_svm = sb.predict_zone_proba(z, X_tda[va])

        # ResNet linear probe (E0.3 protokolu: C val'de secilir)
        rsc = StandardScaler().fit(X_res[tr])
        Xtr, Xva = rsc.transform(X_res[tr]), rsc.transform(X_res[va])
        best = {"auc": -1.0, "model": None}
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, y_zone[tr])
            a = roc_auc_score(y_zone[va], clf.predict_proba(Xva)[:, 1])
            if a > best["auc"]:
                best = {"auc": a, "model": clf}
        p_res = best["model"].predict_proba(Xva)[:, 1]

        for pid, a, b, c in zip(splits.loc[va, "patient_id"], p_cnn, p_svm, p_res):
            rows["cnn"].append({"patient_id": pid, "zone_id": z, "p": float(a)})
            rows["svm"].append({"patient_id": pid, "zone_id": z, "p": float(b)})
            rows["resnet"].append({"patient_id": pid, "zone_id": z, "p": float(c)})
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def main() -> None:
    args = parse_args()
    splits = pd.read_csv(args.splits_path)
    meta = splits[splits["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    y_patient = meta["patient_label"]

    print("=" * 78)
    print("E4.3 — ZON BAZLI HATA ANALIZI (validation)")
    print("=" * 78)
    print(f"Val hastası: {len(meta)} | TEST SETINE DOKUNULMUYOR")

    probs = val_zone_probs(args)

    # Ensemble (E3.4 w=0.55) ve Fusion (E3.1 a=0.55) hasta duzeyi skorlari
    def patient_scores(df: pd.DataFrame) -> pd.Series:
        return df.groupby("patient_id")["p"].max().reindex(meta.index)

    cnn_p, svm_p, res_p = (patient_scores(probs[k]) for k in ("cnn", "svm", "resnet"))
    ens_p = 0.55 * cnn_p + 0.45 * svm_p
    fus_p = 0.55 * res_p + 0.45 * ens_p
    models = {"cnn": cnn_p, "svm": svm_p, "resnet": res_p, "ensemble": ens_p, "fusion": fus_p}

    print("\n--- Hasta duzeyi val ROC-AUC ---")
    model_auc = {}
    for name, p in models.items():
        a = float(roc_auc_score(y_patient.values, p.values))
        model_auc[name] = a
        print(f"  {name:9s}: {a:.4f}")

    # 1) Zon bazli hasta duzeyi AUC (her zon TEK BASINA)
    print("\n--- Zon bazli hasta duzeyi val ROC-AUC (zon tek basina) ---")
    print(f"{'Zon':<12} | {'CNN':>7} | {'SVM':>7} | {'ResNet':>7}")
    print("-" * 48)
    zone_only_auc: Dict[str, Dict[str, float]] = {}
    weak_zones = []
    for z in range(1, 6):
        row = {}
        for name in ("cnn", "svm", "resnet"):
            p = probs[name][probs[name]["zone_id"] == z].set_index("patient_id")["p"].reindex(meta.index)
            row[name] = float(roc_auc_score(y_patient.values, p.values))
        zone_only_auc[f"zone_{z}"] = row
        flag = ""
        if row["cnn"] < WEAK_ZONE_AUC:
            weak_zones.append(z)
            flag = "  <- ZAYIF"
        print(f"Zon {z} {ZONE_NAMES[z]:<7} | {row['cnn']:7.4f} | {row['svm']:7.4f} | {row['resnet']:7.4f}{flag}")

    # 2) Sorumlu zon dagilimi (fusion modeli icin)
    fused_zone = probs["resnet"].copy()
    ens_zone = probs["cnn"].merge(probs["svm"], on=["patient_id", "zone_id"], suffixes=("_cnn", "_svm"))
    ens_zone["p"] = 0.55 * ens_zone["p_cnn"] + 0.45 * ens_zone["p_svm"]
    fused_zone = fused_zone.merge(ens_zone[["patient_id", "zone_id", "p"]],
                                 on=["patient_id", "zone_id"], suffixes=("_res", "_ens"))
    fused_zone["p"] = 0.55 * fused_zone["p_res"] + 0.45 * fused_zone["p_ens"]

    thr, _ = find_optimal_threshold(y_patient.values, fus_p.values, method="youden")
    pred = (fus_p.values >= thr).astype(int)
    outcome = np.where((y_patient.values == 1) & (pred == 1), "TP",
              np.where((y_patient.values == 0) & (pred == 1), "FP",
              np.where((y_patient.values == 0) & (pred == 0), "TN", "FN")))
    resp = fused_zone.loc[fused_zone.groupby("patient_id")["p"].idxmax()].set_index("patient_id")
    resp = resp.reindex(meta.index)
    analysis = pd.DataFrame({"patient_label": y_patient.values, "outcome": outcome,
                             "fusion_prob": fus_p.values, "responsible_zone": resp["zone_id"].values})

    print(f"\n--- Sorumlu zon dagilimi (fusion, esik={thr:.4f}) ---")
    table = pd.crosstab(analysis["responsible_zone"], analysis["outcome"])
    table = table.reindex(columns=["TP", "FP", "TN", "FN"], fill_value=0)
    print(table.to_string())

    print("\n--- Zon basina karar orani ---")
    share = analysis["responsible_zone"].value_counts(normalize=True).sort_index()
    for z, v in share.items():
        print(f"  Zon {z} {ZONE_NAMES[int(z)]:<7}: %{100*v:.1f}")

    fn = analysis[analysis["outcome"] == "FN"]
    tp = analysis[analysis["outcome"] == "TP"]
    print(f"\n--- Yanlis negatifler (FN={len(fn)}) ---")
    if len(fn):
        print(f"  FN fusion olasiligi: ort={fn['fusion_prob'].mean():.4f} "
              f"maks={fn['fusion_prob'].max():.4f}")
        print(f"  FN sorumlu zonlari: {fn['responsible_zone'].value_counts().sort_index().to_dict()}")
    print(f"  TP fusion olasiligi: ort={tp['fusion_prob'].mean():.4f}")

    passed = len(weak_zones) > 0 and len(fn) > 0
    decision = "KEEP" if passed else "KILL"
    print(f"\n  Zayif zon (<{WEAK_ZONE_AUC}) sayisi: {len(weak_zones)} -> {weak_zones}")
    print(f"  KARAR: {decision} (hipotez: zayif zon var VE FN'lerden sorumlu)")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Hasta duzeyi karar tek zona bagimli degildir; zayif zonlar vardir ve FN'lerden sorumludur.",
        "scope": "validation only; test setine dokunulmadi; yeni model egitilmedi",
        "n_val_patients": int(len(meta)),
        "patient_level_val_auc": model_auc,
        "zone_only_val_auc": zone_only_auc,
        "weak_zone_threshold": WEAK_ZONE_AUC,
        "weak_zones": weak_zones,
        "fusion_threshold": float(thr),
        "responsible_zone_by_outcome": table.to_dict(),
        "responsible_zone_share": {str(k): float(v) for k, v in share.items()},
        "fn_count": int(len(fn)),
        "fn_mean_prob": float(fn["fusion_prob"].mean()) if len(fn) else None,
        "fn_responsible_zones": {str(k): int(v) for k, v in fn["responsible_zone"].value_counts().items()},
        "tp_mean_prob": float(tp["fusion_prob"].mean()) if len(tp) else None,
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    analysis.to_csv(run_dir / "val_patient_outcomes.csv", index=False)
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.tda_features, args.resnet_features],
        seed=SEED, decision=decision,
        metrics={"fusion_val_auc": model_auc["fusion"], "n_weak_zones": len(weak_zones), "n_fn": int(len(fn))},
        notes="Zon bazli hata analizi; yeni model egitilmedi, test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
