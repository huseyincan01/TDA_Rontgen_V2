"""E3.1 — Pretrainli görüntü omurgası (ResNet18) + TDA geç füzyonu.

Ön-kayıtlı hipotez:
    Güçlü bir görüntü temsili (donmuş ImageNet ResNet18 + lojistik regresyon) ile
    TDA temsilinin (E3.4 ensemble) hasta düzeyinde geç füzyonu, **tek başına
    ResNet baseline'ından en az +0.015 val ROC-AUC** verir — yani TDA, güçlü bir
    görüntü modeline **tamamlayıcı** bilgi ekler.

Tek değişken:
    ResNet/TDA füzyon ağırlığı `a` (p = a*p_resnet + (1-a)*p_tda).
    TDA dalı E3.4 konfigürasyonunda **sabit** tutulur (CNN+SVM, w=0.55).
    Diğer her şey (v2.1 hattı, split, ölçekleyiciler) sabittir.

Başarı eşiği:
    val ROC-AUC >= ResNet_tek + 0.015  VE
    test AUC farkı için N=2000 bootstrap %95 CI sıfırı dışlar.

Kullanım:
    python pipelines/exp/E3_1_tda_image_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
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

EXPERIMENT_ID = "E3.1"
SLUG = "tda_image_fusion"
SEED = 42
N_BOOTSTRAP = 2000
TDA_CWT = 0.55                       # E3.4'te val'de secilen CNN/SVM agirligi (sabit)
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RESNET_FEATURES = PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.1 TDA + goruntu fuzyonu")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--tda-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path, default=RESNET_FEATURES)
    p.add_argument("--cnn-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt")
    p.add_argument("--svm-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    return p.parse_args()


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def bootstrap_auc_diff(y, pa, pb, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, len(y), len(y))
        yi = y[idx]
        if yi.min() == yi.max():
            diffs[i] = np.nan
            continue
        diffs[i] = auc(yi, np.asarray(pa)[idx]) - auc(yi, np.asarray(pb)[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = float(min(1.0, 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))))
    return float(lo), float(hi), p_two


def tda_zone_probs(args, splits, X_tda):
    """E3.4 konfigurasyonu: zon basina CNN ve SVM olasiliklari."""
    cnn_bundle = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    svm_bundle = joblib.load(args.svm_bundle)
    dropout = float(cnn_bundle["config"].get("dropout", 0.3))
    out = {name: {"cnn": [], "svm": []} for name in ("val", "test")}

    for zone_id in range(1, 6):
        for split_name in ("val", "test"):
            mask = (splits["split"] == split_name) & (splits["zone_id"] == zone_id)
            idx = splits[mask]["vector_index"].values
            Xz = X_tda[idx]

            scaler = cnn_bundle["scalers"][zone_id]
            Xs = scaler.transform(Xz)
            model = SmallZoneCNN(in_channels=2, dropout=dropout)
            model.load_state_dict(cnn_bundle["models_state_dict"][zone_id])
            model.eval()
            with torch.no_grad():
                p_cnn = torch.sigmoid(model(torch.tensor(
                    vector_to_2channel_image(Xs), dtype=torch.float32))).numpy()
            p_svm = svm_bundle.predict_zone_proba(zone_id, Xz)
            pid = splits.loc[mask, "patient_id"].values
            out[split_name]["cnn"].append(pd.DataFrame({"patient_id": pid, "p": p_cnn}))
            out[split_name]["svm"].append(pd.DataFrame({"patient_id": pid, "p": p_svm}))

    res = {}
    for name in ("val", "test"):
        cnn = pd.concat(out[name]["cnn"]).groupby("patient_id")["p"].max()
        svm = pd.concat(out[name]["svm"]).groupby("patient_id")["p"].max()
        res[name] = TDA_CWT * cnn + (1 - TDA_CWT) * svm
    return res


def resnet_patient_probs(args, splits, X_res) -> dict:
    """E0.3 protokolu: zon basina lojistik regresyon (C val'de secilir)."""
    y = splits["zone_label"].values
    train_m = splits["is_train_balanced_sample"].values.astype(bool)
    out = {"val": [], "test": []}
    for zone_id in range(1, 6):
        zm = (splits["zone_id"] == zone_id).values
        tr, va, te = zm & train_m, zm & (splits["split"] == "val").values, zm & (splits["split"] == "test").values
        scaler = StandardScaler().fit(X_res[tr])
        Xtr, Xva, Xte = scaler.transform(X_res[tr]), scaler.transform(X_res[va]), scaler.transform(X_res[te])
        best = {"auc": -1.0, "model": None}
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, y[tr])
            a = auc(y[va], clf.predict_proba(Xva)[:, 1])
            if a > best["auc"]:
                best = {"auc": a, "model": clf}
        clf = best["model"]
        out["val"].append(pd.DataFrame({"patient_id": splits.loc[va, "patient_id"].values,
                                        "p": clf.predict_proba(Xva)[:, 1]}))
        out["test"].append(pd.DataFrame({"patient_id": splits.loc[te, "patient_id"].values,
                                         "p": clf.predict_proba(Xte)[:, 1]}))
    return {k: pd.concat(v).groupby("patient_id")["p"].max() for k, v in out.items()}


def labels_for(splits, split_name):
    meta = splits[splits["split"] == split_name].drop_duplicates("patient_id").set_index("patient_id")
    return meta["patient_label"]


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)

    print("=" * 78)
    print("E3.1 — PRETRAINLI GORUNTU (ResNet18) + TDA GEC FUZYONU")
    print("=" * 78)
    print(f"TDA ozellikleri: {X_tda.shape} | ResNet ozellikleri: {X_res.shape}")

    y_val = labels_for(splits, "val")
    y_test = labels_for(splits, "test")

    tda = tda_zone_probs(args, splits, X_tda)
    res = resnet_patient_probs(args, splits, X_res)

    tda_val = tda["val"].reindex(y_val.index)
    tda_test = tda["test"].reindex(y_test.index)
    res_val = res["val"].reindex(y_val.index)
    res_test = res["test"].reindex(y_test.index)

    refs = {
        "resnet": {"val_auc": auc(y_val.values, res_val.values), "test_auc": auc(y_test.values, res_test.values)},
        "tda_ensemble": {"val_auc": auc(y_val.values, tda_val.values), "test_auc": auc(y_test.values, tda_test.values)},
    }

    best = {"a": None, "val_auc": -1.0}
    for a in A_GRID:
        pv = a * res_val + (1 - a) * tda_val
        s = auc(y_val.values, pv.values)
        if s > best["val_auc"]:
            best = {"a": float(a), "val_auc": s}

    a = best["a"]
    pv = (a * res_val + (1 - a) * tda_val).values
    pt = (a * res_test + (1 - a) * tda_test).values
    thr, _ = find_optimal_threshold(y_val.values, pv, method="youden")
    tm = compute_binary_metrics(y_test.values, pt, threshold=thr)

    print("\n--- Referanslar ---")
    for k, v in refs.items():
        print(f"  {k:14s}: val={v['val_auc']:.4f} | test={v['test_auc']:.4f}")
    print(f"\n--- Fuzyon (en iyi a={a:.2f}) ---")
    print(f"  val={best['val_auc']:.4f} | test={tm['roc_auc']:.4f} | esik={thr:.4f}")

    delta_val = best["val_auc"] - refs["resnet"]["val_auc"]
    lo, hi, p_two = bootstrap_auc_diff(y_test.values, pt, res_test.values)
    print(f"\n  Fuzyon - ResNet: val fark={delta_val:+.4f} | "
          f"test AUC fark={tm['roc_auc'] - refs['resnet']['test_auc']:+.4f} "
          f"[%95 CI {lo:+.4f}, {hi:+.4f}] | p={p_two:.4f}")

    passed_val = delta_val >= 0.015
    passed_ci = lo > 0.0
    decision = "KEEP" if (passed_val and passed_ci) else "KILL"
    print(f"\n  Karar: val fark>=+0.015 -> {passed_val} | CI sifiri disliyor -> {passed_ci}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "ResNet18 + TDA gec fuzyonu, tek basina ResNet'ten >= +0.015 val ROC-AUC verir.",
        "tda_internal_weight_fixed": TDA_CWT,
        "best_fusion_a": a,
        "references": refs,
        "fusion": {"val_auc": best["val_auc"], "test_auc": tm["roc_auc"],
                   "patient_threshold": float(thr), "test_metrics": tm},
        "fusion_minus_resnet": {"val_delta": delta_val,
                                "test_auc_delta": tm["roc_auc"] - refs["resnet"]["test_auc"],
                                "bootstrap_ci95": [lo, hi], "p_value": p_two},
        "criteria": {"val_delta_ge_0.015": bool(passed_val), "ci_excludes_zero": bool(passed_ci)},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID,
        inputs=[args.splits_path, args.tda_features, args.resnet_features],
        seed=SEED, decision=decision,
        metrics={"fusion_val_auc": best["val_auc"], "fusion_test_auc": tm["roc_auc"],
                 "resnet_val_auc": refs["resnet"]["val_auc"], "resnet_test_auc": refs["resnet"]["test_auc"]},
        notes=f"Gec fuzyon a={a:.2f}; ResNet tek={refs['resnet']['test_auc']:.4f}, "
              f"TDA tek={refs['tda_ensemble']['test_auc']:.4f}, fuzyon={tm['roc_auc']:.4f}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
