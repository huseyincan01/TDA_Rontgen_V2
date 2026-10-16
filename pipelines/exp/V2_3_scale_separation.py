"""V2.3 — Ölçek ayrışması: kaba vs ince ızgara PI, FÜZYON ölçütüyle.

Neden bu deney (v4 Faz V2'nin son adayı):
    E2.8 ızgara 15/20/25 test etti ve 20'yi en iyi buldu — ama bunlar hep **ince**
    ölçeklerdi. v4'ün kalan hipotezi **ölçek ayrışması**: kaba bir PI (ızgara 6-10)
    **organ düzeyi / düşük frekanslı** yapıyı kodlar; görüntü dalı ise dokuya bakar.
    Farklı ölçek ⇒ **düşük yedeklilik** beklentisi (V2.1'de bu beklenti çürümüştü,
    burada tekrar test ediliyor).

Ön-kayıtlı hipotez (v4 ölçütü):
    Kaba ızgara (6 veya 10), ince ızgaraya (20) göre füzyona **daha fazla** katkı
    sağlar (d_fuzyon farkı ≥ +0.015) — çünkü farklı bir ölçekte bilgi taşır.

Tasarım:
    * Aynı diyagramlar (E2.6 önbelleğindeki t=0 = standart filtrasyon), yalnızca
      **ızgara çözünürlüğü** değişir: {6, 10, 15, 20}. Aralıklar her koşuda train'den.
    * Tek değişken: ızgara. Sınıflandırıcı/split/zon/max-pooling sabit.
    * Her ızgara için TDA-tek AUC, **füzyon AUC** ve **r(TDA, ResNet)** raporlanır.
    * Kontrol: ızgara 6'nın **karıştırılmış** hâli (boyut aynı, bilgi yok).

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test setine DOKUNULMAZ. Yeniden çıkarım YOK.

Kullanım:
    python pipelines/exp/V2_3_scale_separation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "V2.3"
SLUG = "scale_separation"
SEED = 42
GRIDS = [6, 10, 15, 20]
SIGMA = 1.0
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CACHE = PROJECT_ROOT / "data/processed/features_multiparam/E2_6_diagrams.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2.3 olcek ayrimi (kaba vs ince izgara)")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    return p.parse_args()


def rasterize(dgm, br, pr, grid) -> np.ndarray:
    img = np.zeros((grid, grid), dtype=np.float64)
    if len(dgm) == 0:
        return img
    b, pers = dgm[:, 0], dgm[:, 1] - dgm[:, 0]
    ok = np.isfinite(b) & np.isfinite(pers) & (pers > 0)
    if not np.any(ok):
        return img
    b, pers = np.clip(b[ok], *br), np.clip(pers[ok], *pr)
    bw, pw = max(br[1] - br[0], 1e-9), max(pr[1] - pr[0], 1e-9)
    cols = np.minimum(((b - br[0]) / bw * grid).astype(int), grid - 1)
    rows = np.minimum(((pers - pr[0]) / pw * grid).astype(int), grid - 1)
    np.add.at(img, (rows, cols), pers)
    return gaussian_filter(img, sigma=SIGMA, mode="constant", truncate=3.0)


def ranges_from(diags, trm):
    ds = [d for d, m in zip(diags, trm) if m and len(d)]
    if not ds:
        return (0.0, 1.0), (0.0, 1.0)
    ab = np.concatenate([d[:, 0] for d in ds]); ap = np.concatenate([d[:, 1] - d[:, 0] for d in ds])
    return (float(np.percentile(ab, 0.5)), float(np.percentile(ab, 99.5))), \
           (float(np.percentile(ap, 0.5)), float(np.percentile(ap, 99.5)))


def svm_probs(X, zs, name) -> pd.Series:
    y = zs["zone_label"].values
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        m = zm & (zs["split"] == name).values
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clf.fit(X[tr], y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clf.predict_proba(X[m])[:, 1]}))
    return pd.concat(rows).groupby("patient_id")["p"].max()


def resnet_probs(X, zs, name) -> pd.Series:
    y = zs["zone_label"].values
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & trm
        m = zm & (zs["split"] == name).values
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clf.predict_proba(sc.transform(X[m]))[:, 1]}))
    return pd.concat(rows).groupby("patient_id")["p"].max()


def best_fusion(img: pd.Series, tda: pd.Series, meta: pd.DataFrame) -> Tuple[float, float]:
    y = meta["patient_label"].values
    iv, tv = img.reindex(meta.index).values, tda.reindex(meta.index).values
    best = (-1.0, None)
    for a in A_GRID:
        s = roc_auc_score(y, a * iv + (1 - a) * tv)
        if s > best[0]:
            best = (s, float(a))
    return best[0], best[1]


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = zs["is_train_balanced_sample"].values.astype(bool) | (zs["split"] == "val").values
    sub = zs[need].reset_index(drop=True)
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    trm = sub["is_train_balanced_sample"].values.astype(bool)
    n = len(sub)

    print("=" * 78)
    print("V2.3 — OLCEK AYRIMI: kaba vs ince izgara PI (FUZYON olcutu)")
    print("=" * 78)
    print(f"Zon: {n} | izgaralar: {GRIDS} | esik d_fuzyon >= +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR. Yeniden cikarim YOK (E2.6 onbellegi).")

    z = np.load(CACHE, allow_pickle=True)
    h0, h1 = z["t0_h0"], z["t0_h1"]
    r0, r1 = ranges_from(h0, trm), ranges_from(h1, trm)

    reps: Dict[str, np.ndarray] = {}
    for g in GRIDS:
        reps[f"grid{g}"] = np.stack([np.concatenate([rasterize(h0[i], *r0, g).ravel(),
                                                     rasterize(h1[i], *r1, g).ravel()])
                                     for i in range(n)])
    rng = np.random.RandomState(SEED)
    gmin = min(GRIDS)
    reps[f"grid{gmin}_kontrol"] = reps[f"grid{gmin}"][rng.permutation(n)]
    for k, v in reps.items():
        print(f"  {k}: {v.shape}")

    print("\n  Goruntu dali (donmus ResNet18)...", flush=True)
    Xr = np.load(args.resnet_features)[sub["vector_index"].values]
    img = resnet_probs(Xr, sub, "val")
    img_auc = float(roc_auc_score(meta["patient_label"].values, img.reindex(meta.index).values))
    print(f"    goruntu tek basina = {img_auc:.4f}")

    out: Dict[str, Dict[str, float]] = {}
    for tag, X in reps.items():
        print(f"\n  SVM ({tag}, {X.shape[1]} boyut)...", flush=True)
        tda = svm_probs(X, sub, "val")
        tv = tda.reindex(meta.index).values
        tda_auc = float(roc_auc_score(meta["patient_label"].values, tv))
        fus_auc, a = best_fusion(img, tda, meta)
        r_corr = float(np.corrcoef(img.reindex(meta.index).values, tv)[0, 1])
        out[tag] = {"tda_alone_val_auc": tda_auc, "fusion_val_auc": fus_auc,
                    "fusion_a": a, "r_tda_resnet": r_corr, "dim": int(X.shape[1])}
        print(f"    TDA-tek={tda_auc:.4f} | FUZYON={fus_auc:.4f} (a={a:.2f}) | r={r_corr:.3f}",
              flush=True)

    fine = out[f"grid{max(GRIDS)}"]["fusion_val_auc"]
    print(f"\n--- SONUC (val) ---")
    print(f"  {'Temsil':16s} {'boyut':>6} {'TDA-tek':>9} {'FUZYON':>9} {'d_vs_ince':>10} {'r':>7}")
    best_tag, best_delta = None, -9.0
    for tag, r in out.items():
        d = r["fusion_val_auc"] - fine
        print(f"  {tag:16s} {r['dim']:6d} {r['tda_alone_val_auc']:9.4f} {r['fusion_val_auc']:9.4f} "
              f"{d:+10.4f} {r['r_tda_resnet']:7.3f}")
        if tag.startswith("grid") and "kontrol" not in tag and d > best_delta:
            best_delta, best_tag = d, tag
    passed = best_delta >= MIN_DELTA
    print(f"\n  En iyi: {best_tag} (ince ızgaraya göre {best_delta:+.4f}) | kriter >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Kaba izgara, ince izgaraya gore fuzyona >= +{MIN_DELTA} daha fazla katki saglar.",
        "scope": "validation only; test setine dokunulmadi; yeniden cikarim yok",
        "grids": GRIDS, "single_variable": "izgara cozunurlugu",
        "image_branch_alone_val_auc": img_auc,
        "representations": out,
        "fine_grid_fusion_auc": fine,
        "best_coarse": best_tag, "best_delta_vs_fine": best_delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.resnet_features], seed=SEED,
        decision=report["decision"],
        metrics={"fine_grid_fusion_auc": fine, "best_delta_vs_fine": best_delta},
        notes=f"Olcek ayrimi: en iyi {best_tag} (ince izgaraya gore {best_delta:+.4f}); karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
