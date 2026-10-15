"""E5.3 — Kaynak-dışı genelleme (leave-one-source-out).

Neden bu deney:
    Splitimiz **rastgele** (hasta düzeyi). Bir hakemin soracağı soru: "Model, eğitimde
    görmediği bir **kaynak veri setinden** gelen nodülleri tanıyor mu?" Bu, rastgele
    split'ten çok daha güçlü bir genelleme kanıtıdır ve **yeni veri indirmeyi
    gerektirmez** (kaynak bilgisi patient_info.json'da mevcut).

Tasarım (her kaynak için bir kez):
    * **Test**  : o kaynağın nodüllü hastaları + orana uygun kontrol örneklemi
    * **Val**   : kalan nodüllü hastaların %15'i + kalan kontrollerin %15'i
    * **Train** : geri kalan herkes (zon bazında 1:1 temiz negatif dengelemesi)
    Kaynaklar: chestxray14 (617 hasta) · padchest (314) · jsrt (149)

Ön-kayıtlı hipotez:
    3 kaynağın ortalamasında test ROC-AUC **> 0.75** (rastgele split'in 0.9190'ına
    kıyasla makul bir düşüşle) ⇒ model kaynak-dışı genelliyor.

Tek değişken:
    Test edilen kaynak. Özellikler, mimari, eğitim protokolü ve dengeleme kuralı sabit.

Kullanım:
    python pipelines/exp/E5_3_cross_source.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.data.zone_splits_v2 import build_zone_splits_v2  # noqa: E402
from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E5.3"
SLUG = "cross_source"
SEEDS = [42, 43, 44]
HELD_OUT = ["chestxray14", "padchest", "jsrt"]
CONTROL_RATIO = 3.27          # orijinal test setindeki kontrol/nodul orani
VAL_FRAC = 0.15
TDA_CWT = 0.55
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RANDOM_SPLIT_REF = {"tda_cnn": 0.8631, "fusion": 0.9190}
SUCCESS_AUC, FAIL_AUC = 0.75, 0.65
MAX_EPOCHS = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5.3 kaynak-disi genelleme")
    p.add_argument("--meta-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/features_metadata.csv")
    p.add_argument("--tda-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    p.add_argument("--patients-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    return p.parse_args()


def patient_sources(patients_dir: Path) -> Dict[str, str]:
    out = {}
    for pdir in patients_dir.iterdir():
        if not pdir.is_dir():
            continue
        f = pdir / "patient_info.json"
        if f.exists():
            out[pdir.name] = json.loads(f.read_text(encoding="utf-8")).get("source_dataset", "?")
    return out


def make_patient_splits(meta: pd.DataFrame, src: Dict[str, str], holdout: str, seed: int):
    piv = meta.pivot(index="patient_id", columns="zone_id", values="zone_label").sort_index(axis=1)
    pattern = piv.astype(int).astype(str).agg("".join, axis=1)
    poscnt = piv.sum(axis=1)
    plabel = meta.groupby("patient_id")["patient_label"].first()
    pids = plabel.index
    rng = np.random.RandomState(seed)

    nod = np.array([p for p in pids if plabel[p] == 1])
    ctrl = np.array([p for p in pids if plabel[p] == 0])
    nod_held = np.array([p for p in nod if src.get(p) == holdout])
    nod_other = np.array([p for p in nod if src.get(p) != holdout])

    n_test_ctrl = min(len(ctrl), int(round(len(nod_held) * CONTROL_RATIO)))
    test_ctrl = rng.choice(ctrl, size=n_test_ctrl, replace=False)
    rest_ctrl = np.setdiff1d(ctrl, test_ctrl)

    n_val_nod = max(1, int(round(len(nod_other) * VAL_FRAC)))
    n_val_ctrl = max(1, int(round(len(rest_ctrl) * VAL_FRAC)))
    val_nod = rng.choice(nod_other, size=n_val_nod, replace=False)
    val_ctrl = rng.choice(rest_ctrl, size=n_val_ctrl, replace=False)

    assign = {}
    for p in nod_held: assign[p] = "test"
    for p in test_ctrl: assign[p] = "test"
    for p in val_nod: assign[p] = "val"
    for p in val_ctrl: assign[p] = "val"
    for p in pids:
        assign.setdefault(p, "train")

    df = pd.DataFrame({"patient_id": list(assign.keys()), "split": list(assign.values())})
    df = df.set_index("patient_id")
    df["zone_pattern"] = pattern
    df["pos_zone_count"] = poscnt
    df = df.reset_index()
    return df


def train_tda_cnn(X, zs, seed) -> Dict[str, pd.Series]:
    y = zs["zone_label"].values.astype(np.float32)
    out = {k: [] for k in ("val", "test")}
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        if tr.sum() < 20:
            continue
        vm = zm & (zs["split"] == "val").values & zs["is_zone_metric_sample"].values.astype(bool)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xvm = sc.transform(X[tr]), sc.transform(X[vm])
        torch.manual_seed(seed); np.random.seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32),
                                          torch.tensor(y[tr], dtype=torch.float32)), batch_size=32, shuffle=True)
        Xvm_t = torch.tensor(vector_to_2channel_image(Xvm), dtype=torch.float32)
        best_auc, best = -1.0, None
        for _ in range(MAX_EPOCHS):
            model.train()
            for bx, by in loader:
                opt.zero_grad(); crit(model(bx), by).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xvm_t)).numpy()
            if len(set(y[vm])) > 1:
                a = compute_binary_metrics(y[vm], p)["roc_auc"]
                if a > best_auc:
                    best_auc, best = a, {k: v.clone() for k, v in model.state_dict().items()}
        if best is None:
            best = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best)
        model.eval()
        for name in ("val", "test"):
            m = zm & (zs["split"] == name).values
            if not m.any():
                continue
            with torch.no_grad():
                pv = torch.sigmoid(model(torch.tensor(
                    vector_to_2channel_image(sc.transform(X[m])), dtype=torch.float32))).numpy()
            out[name].append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values, "p": pv}))
    return {k: (pd.concat(v).groupby("patient_id")["p"].max() if v else pd.Series(dtype=float))
            for k, v in out.items()}


def resnet_probe(X, zs, seed=42) -> Dict[str, pd.Series]:
    y = zs["zone_label"].values
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    out = {k: [] for k in ("val", "test")}
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr, va = zm & trm, zm & (zs["split"] == "val").values
        if tr.sum() < 20:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        best = (-1.0, None)
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xtr, y[tr])
            a = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
            if a > best[0]:
                best = (a, clf)
        clf = best[1]
        for name, m in (("val", va), ("test", zm & (zs["split"] == "test").values)):
            if not m.any():
                continue
            out[name].append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                           "p": clf.predict_proba(sc.transform(X[m]))[:, 1]}))
    return {k: (pd.concat(v).groupby("patient_id")["p"].max() if v else pd.Series(dtype=float))
            for k, v in out.items()}


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.meta_path).sort_values("vector_index").reset_index(drop=True)
    X_tda = np.load(args.tda_features)
    X_res = np.load(args.resnet_features)
    src = patient_sources(args.patients_dir)

    print("=" * 78)
    print("E5.3 — KAYNAK-DISI GENELLEME (leave-one-source-out)")
    print("=" * 78)
    print(f"Rastgele-split referansi: TDA CNN {RANDOM_SPLIT_REF['tda_cnn']:.4f} | "
          f"Fusion {RANDOM_SPLIT_REF['fusion']:.4f}")

    results: Dict[str, Any] = {}
    for s in HELD_OUT:
        ps = make_patient_splits(meta, src, s, seed=42)
        zs, _ = build_zone_splits_v2(features_meta_df=meta, patient_splits_df=ps, seed=42)
        print(f"\n--- Kaynak disi: {s} ---")
        print(f"  train hasta={(ps['split']=='train').sum()} | val={(ps['split']=='val').sum()} "
              f"| test={(ps['split']=='test').sum()} | dengeli train zon={int(zs['is_train_balanced_sample'].sum())}")

        tda_aucs, fus_aucs = [], []
        for seed in SEEDS:
            cnn = train_tda_cnn(X_tda, zs, seed)
            meta_te = zs[zs["split"] == "test"].drop_duplicates("patient_id").set_index("patient_id")
            meta_va = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
            y_te = meta_te["patient_label"].values
            y_va = meta_va["patient_label"].values
            if len(np.unique(y_te)) < 2:
                continue
            tda_te = cnn["test"].reindex(meta_te.index)   # TDA dali = CNN (SVM dali bu calismada disarida)
            tda_aucs.append(float(roc_auc_score(y_te, tda_te.values)))

            res = resnet_probe(X_res, zs)
            r_va = res["val"].reindex(meta_va.index); r_te = res["test"].reindex(meta_te.index)
            tda_va = cnn["val"].reindex(meta_va.index)
            best = {"a": None, "auc": -1.0}
            for a in A_GRID:
                sc_ = roc_auc_score(y_va, (a * r_va + (1 - a) * tda_va).values)
                if sc_ > best["auc"]:
                    best = {"a": float(a), "auc": float(sc_)}
            a = best["a"]
            fus_aucs.append(float(roc_auc_score(y_te, (a * r_te + (1 - a) * tda_te).values)))

        results[s] = {
            "n_test_patients": int((ps["split"] == "test").sum()),
            "tda_cnn_test_auc_mean": float(np.mean(tda_aucs)) if tda_aucs else None,
            "tda_cnn_test_auc_std": float(np.std(tda_aucs)) if tda_aucs else None,
            "fusion_test_auc_mean": float(np.mean(fus_aucs)) if fus_aucs else None,
            "fusion_test_auc_std": float(np.std(fus_aucs)) if fus_aucs else None,
        }
        r = results[s]
        print(f"  TDA CNN test AUC={r['tda_cnn_test_auc_mean']:.4f} | "
              f"Fusion test AUC={r['fusion_test_auc_mean']:.4f}", flush=True)

    tda_means = [results[s]["tda_cnn_test_auc_mean"] for s in HELD_OUT if results[s]["tda_cnn_test_auc_mean"]]
    fus_means = [results[s]["fusion_test_auc_mean"] for s in HELD_OUT if results[s]["fusion_test_auc_mean"]]
    tda_avg = float(np.mean(tda_means)) if tda_means else 0.0
    fus_avg = float(np.mean(fus_means)) if fus_means else 0.0

    print("\n--- OZET (3 kaynak disi) ---")
    print(f"  TDA CNN ortalama test AUC : {tda_avg:.4f}  (rastgele split {RANDOM_SPLIT_REF['tda_cnn']:.4f})")
    print(f"  Fusion  ortalama test AUC : {fus_avg:.4f}  (rastgele split {RANDOM_SPLIT_REF['fusion']:.4f})")

    decision = "KEEP" if fus_avg > SUCCESS_AUC else ("KILL" if fus_avg < FAIL_AUC else "KILL")
    print(f"\n  Kriter: >{SUCCESS_AUC} KEEP | <{FAIL_AUC} KILL")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Kaynak-disi ortalama test ROC-AUC > {SUCCESS_AUC}.",
        "design": "leave-one-source-out; kontroller her kaynak icin orana uygun orneklenir; zon bazinda 1:1 dengeleme",
        "held_out_sources": HELD_OUT, "seeds": SEEDS,
        "random_split_reference": RANDOM_SPLIT_REF,
        "results": results,
        "tda_cnn_mean_test_auc": tda_avg, "fusion_mean_test_auc": fus_avg,
        "thresholds": {"success": SUCCESS_AUC, "fail": FAIL_AUC},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.meta_path, args.tda_features, args.resnet_features],
        seed=42, decision=decision,
        metrics={"tda_cnn_mean_test_auc": tda_avg, "fusion_mean_test_auc": fus_avg},
        notes=f"Kaynak-disi genelleme (LOO): TDA {tda_avg:.4f}, fusion {fus_avg:.4f}; karar={decision}.",
    )
    print(f"\nRapor    : {run_dir / 'test_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
