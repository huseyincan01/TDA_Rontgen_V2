"""E3.7 — Ana sonucun çok-seed doğrulaması (Protokol Kural 10 uyumu).

Neden bu deney:
    E4.6'da CNN'in seed varyansı **±0.02–0.03** ölçüldü ve **Kural 10** eklendi:
    makalede raporlanan model performansları **≥3 seed ortalaması** olmalı. Ancak ana
    sonucumuz (füzyon 0.9211) hâlâ **tek seed**'lik. Bu deney o boşluğu kapatır.

Hipotez (ön-kayıtlı):
    Füzyonun 3-seed ortalama test ROC-AUC'si, tek-seed değerinden (0.9211) **0.03'ten az**
    sapar (yani ana sonuç seed'e karşı kararlıdır) ve ortalaması **> 0.89** kalır.

Tasarım:
    Tek değişken: CNN eğitim tohumu (42/43/44). TDA özellikleri, split, mimari, SVM dalı,
    ResNet dalı (deterministik) ve füzyon protokolü **sabit**. Her seed için füzyon ağırlığı
    ve eşiği **yalnızca validation'da** seçilir.

Kapsam:
    Test seti bu deney için **bir kez** kullanılır (ana sonucun seed belirsizliğini ölçmek için).

Kullanım:
    python pipelines/exp/E3_7_headline_multiseed.py
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E3.7"
SLUG = "headline_multiseed"
SEEDS = [42, 43, 44]
TDA_CWT = 0.55
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SINGLE_SEED_REF = 0.9211
MAX_EPOCHS = 40
STABILITY_TOL = 0.03
MIN_MEAN_AUC = 0.89


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.7 ana sonuc cok-seed")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--tda-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    p.add_argument("--svm-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    return p.parse_args()


def tda_cnn_patient_probs(X, zs, seed) -> Dict[str, pd.Series]:
    """5-zon CNN'i verilen seed ile egitir; val/test hasta olasiliklarini doner."""
    y = zs["zone_label"].values.astype(np.float32)
    out = {k: [] for k in ("val", "test")}
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        vm = zm & (zs["split"] == "val").values & zs["is_zone_metric_sample"].values.astype(bool)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xvm = sc.transform(X[tr]), sc.transform(X[vm])
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32),
                                          torch.tensor(y[tr], dtype=torch.float32)),
                            batch_size=32, shuffle=True)
        Xvm_t = torch.tensor(vector_to_2channel_image(Xvm), dtype=torch.float32)
        best_auc, best = -1.0, None
        for _ in range(MAX_EPOCHS):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xvm_t)).numpy()
            a = compute_binary_metrics(y[vm], p)["roc_auc"]
            if a > best_auc:
                best_auc, best = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best)
        model.eval()
        for name in ("val", "test"):
            m = zm & (zs["split"] == name).values
            Xa = sc.transform(X[m])
            with torch.no_grad():
                pv = torch.sigmoid(model(torch.tensor(
                    vector_to_2channel_image(Xa), dtype=torch.float32))).numpy()
            out[name].append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values, "p": pv}))
    return {k: pd.concat(v).groupby("patient_id")["p"].max() for k, v in out.items()}


def resnet_patient_probs(X_res, zs) -> Dict[str, pd.Series]:
    y = zs["zone_label"].values
    train_m = zs["is_train_balanced_sample"].values.astype(bool)
    out = {k: [] for k in ("val", "test")}
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr, va, te = zm & train_m, zm & (zs["split"] == "val").values, zm & (zs["split"] == "test").values
        sc = StandardScaler().fit(X_res[tr])
        Xtr, Xva, Xte = sc.transform(X_res[tr]), sc.transform(X_res[va]), sc.transform(X_res[te])
        best = (-1.0, None)
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(Xtr, y[tr])
            a = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
            if a > best[0]:
                best = (a, clf)
        clf = best[1]
        out["val"].append(pd.DataFrame({"patient_id": zs.loc[va, "patient_id"].values,
                                        "p": clf.predict_proba(Xva)[:, 1]}))
        out["test"].append(pd.DataFrame({"patient_id": zs.loc[te, "patient_id"].values,
                                         "p": clf.predict_proba(Xte)[:, 1]}))
    return {k: pd.concat(v).groupby("patient_id")["p"].max() for k, v in out.items()}


def main() -> None:
    args = parse_args()
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)
    svm = joblib.load(args.svm_bundle)

    meta_val = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    meta_test = zs[zs["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")

    print("=" * 78)
    print("E3.7 — ANA SONUCUN COK-SEED DOGRULAMASI (Kural 10 uyumu)")
    print("=" * 78)
    print(f"Seed'ler: {SEEDS} | tek-seed referans: {SINGLE_SEED_REF}")

    # SVM dali deterministik -> bir kez
    svm_p = {}
    for name in ("val", "test"):
        rows = []
        for z in range(1, 6):
            m = ((zs["split"] == name) & (zs["zone_id"] == z)).values
            idx = zs[m]["vector_index"].values
            pr = svm.predict_zone_proba(z, X_tda[idx])
            rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values, "p": pr}))
        svm_p[name] = pd.concat(rows).groupby("patient_id")["p"].max()
    res_p = resnet_patient_probs(X_res, zs)

    results: Dict[str, Any] = {}
    for seed in SEEDS:
        cnn_p = tda_cnn_patient_probs(X_tda, zs, seed)
        tda_val = (TDA_CWT * cnn_p["val"] + (1 - TDA_CWT) * svm_p["val"]).reindex(meta_val.index)
        tda_test = (TDA_CWT * cnn_p["test"] + (1 - TDA_CWT) * svm_p["test"]).reindex(meta_test.index)
        r_val = res_p["val"].reindex(meta_val.index)
        r_test = res_p["test"].reindex(meta_test.index)

        best = {"a": None, "auc": -1.0}
        for a in A_GRID:
            s = roc_auc_score(meta_val["patient_label"].values, (a * r_val + (1 - a) * tda_val).values)
            if s > best["auc"]:
                best = {"a": float(a), "auc": float(s)}
        a = best["a"]
        thr, _ = find_optimal_threshold(meta_val["patient_label"].values,
                                        (a * r_val + (1 - a) * tda_val).values, method="youden")
        pt = (a * r_test + (1 - a) * tda_test).values
        tm = compute_binary_metrics(meta_test["patient_label"].values, pt, threshold=thr)
        results[seed] = {"best_a": a, "val_auc": best["auc"], "patient_threshold": float(thr),
                         "test_auc": float(tm["roc_auc"]), "test_metrics": tm,
                         "tda_only_test_auc": float(roc_auc_score(meta_test["patient_label"].values,
                                                                  tda_test.values))}
        print(f"  seed={seed}: a={a:.2f} | val={best['auc']:.4f} | TEST AUC={tm['roc_auc']:.4f} "
              f"| acc={tm['accuracy']:.4f} | TDA-tek={results[seed]['tda_only_test_auc']:.4f}", flush=True)

    aucs = np.array([results[s]["test_auc"] for s in SEEDS])
    mean_auc, std_auc = float(aucs.mean()), float(aucs.std())
    tda_aucs = np.array([results[s]["tda_only_test_auc"] for s in SEEDS])
    delta_vs_single = mean_auc - SINGLE_SEED_REF

    print("\n--- OZET (test ROC-AUC, 734 hasta) ---")
    print(f"  Fuzyon 3-seed : {mean_auc:.4f} +/- {std_auc:.4f}   (tek-seed referans {SINGLE_SEED_REF})")
    print(f"  TDA-tek 3-seed: {tda_aucs.mean():.4f} +/- {tda_aucs.std():.4f}")
    print(f"  Tek-seed'e fark: {delta_vs_single:+.4f}")

    stable = abs(delta_vs_single) < STABILITY_TOL
    strong = mean_auc > MIN_MEAN_AUC
    decision = "KEEP" if (stable and strong) else "KILL"
    print(f"\n  Kriter: |fark| < {STABILITY_TOL} -> {stable} | ortalama > {MIN_MEAN_AUC} -> {strong}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Fuzyonun 3-seed ortalamasi tek-seed'den <0.03 sapar ve >0.89 kalir.",
        "scope": "test seti ana sonucun seed belirsizligini olcmek icin bir kez kullanildi",
        "seeds": SEEDS, "tda_internal_weight": TDA_CWT,
        "single_seed_reference": SINGLE_SEED_REF,
        "per_seed": {str(s): results[s] for s in SEEDS},
        "fusion_test_auc_mean": mean_auc, "fusion_test_auc_std": std_auc,
        "tda_only_test_auc_mean": float(tda_aucs.mean()), "tda_only_test_auc_std": float(tda_aucs.std()),
        "delta_vs_single_seed": delta_vs_single,
        "criteria": {"abs_delta_lt_0.03": bool(stable), "mean_gt_0.89": bool(strong)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.tda_features, args.resnet_features],
        seed=42, decision=decision,
        metrics={"fusion_test_auc_mean": mean_auc, "fusion_test_auc_std": std_auc,
                 "delta_vs_single_seed": delta_vs_single},
        notes=f"Ana sonuc 3-seed: {mean_auc:.4f} +/- {std_auc:.4f} (tek-seed {SINGLE_SEED_REF}); karar={decision}.",
    )
    print(f"\nRapor    : {run_dir / 'test_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
