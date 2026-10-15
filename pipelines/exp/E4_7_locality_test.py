"""E4.7 — Lokalite testi: model gerçekten nodülün OLDUĞU zonu mu işaretliyor?

Neden bu deney:
    E4.3, her zonun TEK BAŞINA hastayı 0.74–0.84 AUC ile ayırt edebildiğini gösterdi.
    Sinyal tamamen lokal olsaydı, nodül İÇERMEYEN zonlar şans düzeyinde kalırdı. Bu
    deney, zon-bazlı sinyalin **lokal** mi yoksa **hasta-global** mı olduğunu ölçer.

Ön-kayıtlı hipotez (LOKALİTE):
    Nodül-pozitif val hastalarında model,
      (a) argmax zonunu **gerçek nodül zonuna** şanstan daha sık yerleştirir, VE
      (b) gerçek nodül zonlarına, aynı hastanın nodülsüz zonlarından **daha yüksek**
          olasılık atar (eşleştirilmiş fark > 0, bootstrap CI sıfırı dışlar).

Ölçütler:
    1. İsabet oranı (argmax ∈ gerçek pozitif zonlar) vs şans tabanı (n_pos/5 ortalaması)
       + iki terimli (binomial) test.
    2. Eşleştirilmiş fark: p(gerçek pozitif zon) − p(gerçek negatif zon), bootstrap %95 CI.
    3. Zon düzeyi AUC: yalnızca nodül-pozitif hastalar içinde, zon olasılığının
       gerçek-pozitif/gerçek-negatif zonu ayırt etme gücü (0.5 = lokalizasyon yok).

Kapsam (test bütçesi korunur):
    Yalnızca **validation**; test setine dokunulmaz. Yeni model eğitilmez.

Kullanım:
    python pipelines/exp/E4_7_locality_test.py
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
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E4.7"
SLUG = "locality_test"
SEED = 42
N_BOOTSTRAP = 2000
ZONE_NAMES = {1: "Sağ Üst", 2: "Sağ Orta", 3: "Sağ Alt", 4: "Sol Üst", 5: "Sol Alt"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4.7 lokalite testi")
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


def val_zone_frame(args) -> pd.DataFrame:
    """VAL zonları için patient_id, zone_id, zone_label, label_state, p_cnn, p_svm, p_res."""
    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)
    cb = torch.load(args.cnn_bundle, map_location="cpu", weights_only=False)
    sb = joblib.load(args.svm_bundle)
    dropout = float(cb["config"].get("dropout", 0.3))
    y_zone = splits["zone_label"].values
    train_m = splits["is_train_balanced_sample"].values.astype(bool)

    frames = []
    for z in range(1, 6):
        zm = (splits["zone_id"] == z).values
        va = zm & (splits["split"] == "val").values
        tr = zm & train_m

        scaler = cb["scalers"][z]
        model = SmallZoneCNN(in_channels=2, dropout=dropout)
        model.load_state_dict(cb["models_state_dict"][z])
        model.eval()
        with torch.no_grad():
            p_cnn = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(scaler.transform(X_tda[va])), dtype=torch.float32))).numpy()
        p_svm = sb.predict_zone_proba(z, X_tda[va])

        rsc = StandardScaler().fit(X_res[tr])
        Xtr, Xva = rsc.transform(X_res[tr]), rsc.transform(X_res[va])
        best = {"auc": -1.0, "m": None}
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, y_zone[tr])
            a = roc_auc_score(y_zone[va], clf.predict_proba(Xva)[:, 1])
            if a > best["auc"]:
                best = {"auc": a, "m": clf}
        p_res = best["m"].predict_proba(Xva)[:, 1]

        frames.append(pd.DataFrame({
            "patient_id": splits.loc[va, "patient_id"].values,
            "zone_id": z,
            "zone_label": splits.loc[va, "zone_label"].values,
            "label_state": splits.loc[va, "label_state"].values,
            "p_cnn": p_cnn, "p_svm": p_svm, "p_res": p_res,
        }))
    return pd.concat(frames, ignore_index=True)


def bootstrap_mean_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=float)
    means = np.array([v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(df: pd.DataFrame, score_col: str, name: str) -> Dict[str, Any]:
    """Nodül-pozitif val hastalarında lokalite ölçütleri."""
    pos_patients = df[df["zone_label"] == 1]["patient_id"].unique()
    hits, chances, diffs, zone_scores, zone_labels = [], [], [], [], []

    for pid in pos_patients:
        sub = df[df["patient_id"] == pid]
        if len(sub) != 5:
            continue
        true_pos = set(sub[sub["zone_label"] == 1]["zone_id"].tolist())
        if not true_pos:
            continue
        p = sub.set_index("zone_id")[score_col]
        argmax_zone = int(p.idxmax())
        hits.append(1 if argmax_zone in true_pos else 0)
        chances.append(len(true_pos) / 5.0)
        diffs.append(float(p[list(true_pos)].mean() - p[[z for z in range(1, 6) if z not in true_pos]].mean()))
        for z in sub["zone_id"]:
            zone_scores.append(float(p[z]))
            zone_labels.append(1 if z in true_pos else 0)

    hits = np.asarray(hits)
    chance = float(np.mean(chances))
    hit_rate = float(hits.mean())
    n = len(hits)
    bt = binomtest(int(hits.sum()), n, chance, alternative="greater")
    lo, hi = bootstrap_mean_ci(np.asarray(diffs))
    zone_auc = float(roc_auc_score(zone_labels, zone_scores)) if len(set(zone_labels)) > 1 else float("nan")

    return {
        "model": name,
        "n_nodule_positive_patients": int(n),
        "hit_rate": hit_rate,
        "chance_rate": chance,
        "hit_rate_minus_chance": hit_rate - chance,
        "binomial_p_greater": float(bt.pvalue),
        "paired_mean_diff": float(np.mean(diffs)),
        "paired_ci95": [lo, hi],
        "paired_positive": bool(lo > 0.0),
        "zone_level_auc": zone_auc,
        "localizes": bool(hit_rate > chance and lo > 0.0 and zone_auc > 0.5),
    }


def main() -> None:
    args = parse_args()
    df = val_zone_frame(args)
    df["p_ens"] = 0.55 * df["p_cnn"] + 0.45 * df["p_svm"]
    df["p_fus"] = 0.55 * df["p_res"] + 0.45 * df["p_ens"]

    print("=" * 78)
    print("E4.7 — LOKALITE TESTI: argmax zonu gercek nodul zonuna mi dusuyor?")
    print("=" * 78)
    print(f"Val zon satiri: {len(df)} | TEST SETINE DOKUNULMUYOR")

    results = {}
    for col, name in [("p_cnn", "TDA CNN"), ("p_ens", "TDA ensemble"),
                      ("p_res", "ResNet18"), ("p_fus", "Fusion")]:
        r = analyse(df, col, name)
        results[name] = r
        print(f"\n--- {name} ---")
        print(f"  Nodul-pozitif hasta : {r['n_nodule_positive_patients']}")
        print(f"  Isabet orani        : {r['hit_rate']:.4f}  (sans tabani {r['chance_rate']:.4f}, "
              f"fark {r['hit_rate_minus_chance']:+.4f})")
        print(f"  Binomial p (>)      : {r['binomial_p_greater']:.3e}")
        print(f"  Esles. fark (pos-neg): {r['paired_mean_diff']:+.4f} "
              f"[%95 CI {r['paired_ci95'][0]:+.4f}, {r['paired_ci95'][1]:+.4f}]")
        print(f"  Zon duzeyi AUC      : {r['zone_level_auc']:.4f}")
        print(f"  -> LOKALIZE EDİYOR  : {'EVET' if r['localizes'] else 'HAYIR'}")

    # Kontrol: nodul-NEGATIF hastalarda ayni olcut anlamsiz olmali
    print("\n--- Kontrol: nodul-NEGATIF val hastalari (gercek pozitif zon yok) ---")
    neg_ids = df[df["label_state"] == "negative"]["patient_id"].unique()
    print(f"  Temiz-negatif zon iceren hasta sayisi: {len(neg_ids)} (beklenen: sans ~0.5)")

    deciders = [k for k, v in results.items() if v["localizes"]]
    decision = "KEEP" if len(deciders) >= 3 else "KILL"
    print(f"\n  Lokalize eden model sayisi: {len(deciders)}/4 ({deciders})")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Model argmax'i gercek nodul zonuna sanstan sik duser ve o zona daha yuksek olasilik atar.",
        "scope": "validation only; test setine dokunulmadi; yeni model egitilmedi",
        "results": results,
        "localizing_models": deciders,
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.tda_features, args.resnet_features],
        seed=SEED, decision=decision,
        metrics={f"hit_rate_{k}": v["hit_rate"] for k, v in results.items()} |
                {f"zone_auc_{k}": v["zone_level_auc"] for k, v in results.items()},
        notes=f"Lokalite testi: {len(deciders)}/4 model lokalize ediyor; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
