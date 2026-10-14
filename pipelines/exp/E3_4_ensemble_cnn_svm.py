"""E3.4 — 5-Zon TDA Küçük CNN + RBF-SVM soft-voting (ensemble) deneyi.

Ön-kayıtlı hipotez:
    CNN ve SVM hasta düzeyi olasılıklarının ağırlıklı ortalaması, tek modellerin
    her birinden en az **+0.015 val ROC-AUC** verir.

Tek değişken:
    Birleştirme ağırlığı `w` (p = w * p_cnn + (1 - w) * p_svm). Veri hattı,
    özellikler, split, ölçekleyiciler ve max-pooling **sabittir** (v2.1 kilidi).

Başarı eşiği:
    val ROC-AUC >= (en iyi tek model) + 0.015  VE
    test AUC farkı için N=2000 bootstrap %95 CI sıfırı dışlar.

İki varyant raporlanır (karar ön-kayıtlı A varyantına göre verilir):
    A) Hasta düzeyinde soft voting : p = w*max_z(cnn_z) + (1-w)*max_z(svm_z)
    B) Zon düzeyinde füzyon        : p_z = w*cnn_z + (1-w)*svm_z, sonra max_z

Kullanım:
    python pipelines/exp/E3_4_ensemble_cnn_svm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E3.4"
SLUG = "ensemble_cnn_svm"
N_BOOTSTRAP = 2000
SEED = 42
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3.4 CNN+SVM ensemble deneyi")
    p.add_argument("--features-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--cnn-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt")
    p.add_argument("--svm-bundle", type=Path,
                   default=PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib")
    return p.parse_args()


def zone_probabilities(args, X_all, splits):
    """Her zon için val/test olasılıklarını (CNN ve SVM) üretir."""
    cnn_bundle = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    svm_bundle = joblib.load(args.svm_bundle)
    dropout = float(cnn_bundle["config"].get("dropout", 0.3))

    rows = {name: {m: [] for m in ("cnn", "svm")} for name in ("val", "test")}

    for zone_id in range(1, 6):
        for split_name in ("val", "test"):
            mask = (splits["split"] == split_name) & (splits["zone_id"] == zone_id)
            idx = splits[mask]["vector_index"].values
            Xz = X_all[idx]

            # --- CNN ---
            scaler = cnn_bundle["scalers"][zone_id]
            Xz_scaled = scaler.transform(Xz)
            model = SmallZoneCNN(in_channels=2, dropout=dropout)
            model.load_state_dict(cnn_bundle["models_state_dict"][zone_id])
            model.eval()
            tensor = torch.tensor(vector_to_2channel_image(Xz_scaled), dtype=torch.float32)
            with torch.no_grad():
                p_cnn = torch.sigmoid(model(tensor)).numpy()

            # --- SVM ---
            p_svm = svm_bundle.predict_zone_proba(zone_id, Xz)

            pid = splits.loc[mask, "patient_id"].values
            rows[split_name]["cnn"].append(pd.DataFrame({"patient_id": pid, "zone_id": zone_id, "p": p_cnn}))
            rows[split_name]["svm"].append(pd.DataFrame({"patient_id": pid, "zone_id": zone_id, "p": p_svm}))

    return {
        name: {m: pd.concat(rows[name][m], ignore_index=True) for m in ("cnn", "svm")}
        for name in ("val", "test")
    }


def patient_frame(splits, split_name):
    """Hasta düzeyi doğru etiketleri (split başına) döner."""
    meta = splits[splits["split"] == split_name].drop_duplicates("patient_id").set_index("patient_id")
    return meta["patient_label"]


def patient_probs(df_cnn, df_svm, w, variant):
    """Varyanta göre hasta düzeyi birleşik olasılıkları üretir."""
    if variant == "A":  # hasta düzeyi soft voting
        a = df_cnn.groupby("patient_id")["p"].max()
        b = df_svm.groupby("patient_id")["p"].max()
        return (w * a + (1 - w) * b)
    # variant == "B": zon düzeyi füzyon, sonra max
    merged = df_cnn.merge(df_svm, on=["patient_id", "zone_id"], suffixes=("_cnn", "_svm"))
    merged["p_fused"] = w * merged["p_cnn"] + (1 - w) * merged["p_svm"]
    return merged.groupby("patient_id")["p_fused"].max()


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def bootstrap_auc_diff(y, p_a, p_b, n=N_BOOTSTRAP, seed=SEED):
    """AUC(p_a) - AUC(p_b) için bootstrap %95 CI ve iki yönlü p-değeri."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n_obs = len(y)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, n_obs, n_obs)
        yi = y[idx]
        if yi.min() == yi.max():          # tek sınıflı örneklem -> atla
            diffs[i] = np.nan
            continue
        diffs[i] = auc(yi, np.asarray(p_a)[idx]) - auc(yi, np.asarray(p_b)[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_below0 = float((diffs <= 0).mean())
    frac_above0 = float((diffs >= 0).mean())
    p_two_sided = float(min(1.0, 2 * min(frac_below0, frac_above0)))
    return float(lo), float(hi), p_two_sided, float(diffs.mean())


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_all = np.load(args.features_path)
    splits = pd.read_csv(args.splits_path)
    print("=" * 78)
    print("E3.4 — 5-ZON TDA CNN + RBF-SVM SOFT-VOTING (ENSEMBLE)")
    print("=" * 78)
    print(f"Ozellikler: {X_all.shape} | split satirlari: {len(splits)}")

    probs = zone_probabilities(args, X_all, splits)

    y_val = patient_frame(splits, "val")
    y_test = patient_frame(splits, "test")

    results = {}
    for variant in ("A", "B"):
        best = {"w": None, "val_auc": -1.0}
        for w in W_GRID:
            pv = patient_probs(probs["val"]["cnn"], probs["val"]["svm"], float(w), variant)
            pv = pv.reindex(y_val.index)
            if pv.isna().any():
                continue
            a = auc(y_val.values, pv.values)
            if a > best["val_auc"]:
                best = {"w": float(w), "val_auc": a}

        w = best["w"]
        pv = patient_probs(probs["val"]["cnn"], probs["val"]["svm"], w, variant).reindex(y_val.index)
        pt = patient_probs(probs["test"]["cnn"], probs["test"]["svm"], w, variant).reindex(y_test.index)
        thr, _ = find_optimal_threshold(y_val.values, pv.values, method="youden")
        test_metrics = compute_binary_metrics(y_test.values, pt.values, threshold=thr)
        results[variant] = {
            "best_w": w, "val_auc": best["val_auc"], "patient_threshold": float(thr),
            "test_auc": test_metrics["roc_auc"], "test_metrics": test_metrics,
            "val_probs": pv, "test_probs": pt,
        }

    # --- Referans tek modeller (kayitli patient_threshold ile) ---
    refs = {}
    for name in ("cnn", "svm"):
        pv = probs["val"][name].groupby("patient_id")["p"].max().reindex(y_val.index)
        pt = probs["test"][name].groupby("patient_id")["p"].max().reindex(y_test.index)
        thr, _ = find_optimal_threshold(y_val.values, pv.values, method="youden")
        tm = compute_binary_metrics(y_test.values, pt.values, threshold=thr)
        refs[name] = {"val_auc": auc(y_val.values, pv.values), "test_auc": tm["roc_auc"],
                      "patient_threshold": float(thr), "test_metrics": tm,
                      "test_probs": pt}

    print("\n--- Referans tek modeller (val'de esik, test'te dondurulmus) ---")
    for name, r in refs.items():
        print(f"  {name.upper():4s}: val AUC={r['val_auc']:.4f} | test AUC={r['test_auc']:.4f} "
              f"| esik={r['patient_threshold']:.4f}")

    print("\n--- Ensemble varyantlari ---")
    for variant, r in results.items():
        print(f"  Varyant {variant}: en iyi w={r['best_w']:.2f} | val AUC={r['val_auc']:.4f} "
              f"| test AUC={r['test_auc']:.4f} | esik={r['patient_threshold']:.4f}")

    best_single_name = max(refs, key=lambda k: refs[k]["val_auc"])
    best_single_val = refs[best_single_name]["val_auc"]
    print(f"\n  En iyi tek model (val): {best_single_name.upper()} (val AUC={best_single_val:.4f})")

    print("\n--- Bootstrap %95 CI (test AUC farki) ---")
    # Ensemble A vs en iyi tek model: AUC farki
    primary = "A"
    lo, hi, p_two, mean_diff = bootstrap_auc_diff(
        y_test.values, results[primary]["test_probs"].values, refs[best_single_name]["test_probs"].values
    )
    delta_val = results[primary]["val_auc"] - best_single_val
    print(f"  Varyant {primary} - {best_single_name.upper()}: "
          f"val fark={delta_val:+.4f} | test AUC fark={results[primary]['test_auc'] - refs[best_single_name]['test_auc']:+.4f} "
          f"[%95 CI {lo:+.4f}, {hi:+.4f}] | p={p_two:.4f}")

    passed_val = delta_val >= 0.015
    passed_ci = lo > 0.0
    decision = "KEEP" if (passed_val and passed_ci) else "KILL"
    print(f"\n  Karar kriteri: val fark >= +0.015 -> {passed_val} | CI sifiri disliyor -> {passed_ci}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "CNN + SVM soft-voting tek modellerden >= +0.015 val ROC-AUC verir.",
        "reference_single_models": {k: {"val_auc": v["val_auc"], "test_auc": v["test_auc"],
                                        "patient_threshold": v["patient_threshold"],
                                        "test_metrics": v["test_metrics"]} for k, v in refs.items()},
        "ensembles": {k: {"best_w": v["best_w"], "val_auc": v["val_auc"], "test_auc": v["test_auc"],
                          "patient_threshold": v["patient_threshold"], "test_metrics": v["test_metrics"]}
                      for k, v in results.items()},
        "primary_variant": primary,
        "best_single_model": best_single_name,
        "vertical_gain": {"val_delta": delta_val,
                          "test_auc_delta": results[primary]["test_auc"] - refs[best_single_name]["test_auc"],
                          "bootstrap_ci95": [lo, hi], "p_value": p_two},
        "criteria": {"val_delta_ge_0.015": bool(passed_val), "ci_excludes_zero": bool(passed_ci)},
        "decision": decision,
    }

    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID,
        inputs=[args.features_path, args.splits_path, args.cnn_bundle, args.svm_bundle],
        seed=SEED, decision=decision,
        metrics={"ensemble_A_val_auc": results["A"]["val_auc"], "ensemble_A_test_auc": results["A"]["test_auc"],
                 "cnn_val_auc": refs["cnn"]["val_auc"], "cnn_test_auc": refs["cnn"]["test_auc"],
                 "svm_val_auc": refs["svm"]["val_auc"], "svm_test_auc": refs["svm"]["test_auc"],
                 "val_delta_vs_best_single": delta_val},
        notes=f"Ensemble varyant A (hasta duzeyi soft voting), w={results['A']['best_w']:.2f}; karar={decision}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
