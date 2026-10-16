"""V2.1 + V2.2 — Konum tabanlı filtrasyonlar (yön ve radyal), FÜZYON ölçütüyle.

Neden bu deney (v4 Faz V2):
    V1.1/V1.2 gösterdi ki TDA dalının **tek başına** başarımını artırmak füzyona
    yansımıyor (E2.6: +0.0480 → +0.0072; E2.1: +0.0210 → **−0.0019**). Çünkü o
    kazançlar **ResNet18 ile yedekli**. v4 bu yüzden **d_fuzyon** ölçütünü kullanır.

    Bu deney **konum tabanlı** ikinci parametreleri test eder:
      * `f + t·y_norm`  (dikey/yükseklik)
      * `f + t·x_norm`  (yatay)
      * `f + t·r_norm`  (zon merkezine radyal uzaklık)
    **Neden umut verici:** konum bilgisi ResNet18'in global average pooling'inde
    **kaybolur** ⇒ bu bilginin görüntü dalıyla **yedekli olma ihtimali düşüktür**
    (E2.6'nın gradyanı ise ResNet'te zaten vardı).

Ön-kayıtlı hipotez (v4 ölçütü):
    En az bir konum filtrasyonu, mevcut PI füzyonunu val'de **d_fuzyon ≥ +0.015** ile geçer.

Tasarım (tek değişken: ikinci filtrasyon fonksiyonu; görüntü dalı sabit):
    * f_norm = mevcut filtrasyon [0,1] ölçekli
    * y_norm, x_norm: satır/sütun indeksi [0,1]  |  r_norm: centroid'e uzaklık [0,1]
    * Dilimler t ∈ {0.5, 1.5}
    * Aday temsiller: concat(proje PI, iki dilim) = 2400 boyut, her yön için ayrı
    * Her aday için **TDA-tek** ve **FÜZYON** AUC'si; ayrıca karıştırılmış kontrol

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/V2_1_positional_filtrations.py
    python pipelines/exp/V2_1_positional_filtrations.py --reuse-cache   # analiz only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Tuple

import gudhi
import numpy as np
import pandas as pd
import SimpleITK as sitk
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
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402

EXPERIMENT_ID = "V2.1"
SLUG = "positional_filtrations"
SEED = 42
GRID, SIGMA = 20, 1.0
T_VALUES = [0.5, 1.5]
ARMS = ["y", "x", "r"]
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CACHE = PROJECT_ROOT / "data/processed/features_positional/V2_1_diagrams.npz"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2.1/V2.2 konum filtrasyonlari")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--reuse-cache", action="store_true")
    return p.parse_args()


_W: Dict[str, Any] = {}


def _init_worker() -> None:
    _W["ex"] = ZoneTDAExtractor(TDASettings())


def _finite(iv) -> np.ndarray:
    if iv is None or len(iv) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    d = np.asarray(iv, dtype=np.float64)
    d = d[np.isfinite(d).all(axis=1)]
    return d[(d[:, 1] - d[:, 0]) > 0]


def _ph(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cc = gudhi.CubicalComplex(dimensions=f.shape,
                              top_dimensional_cells=f.astype(np.float64).ravel())
    cc.compute_persistence()
    return _finite(cc.persistence_intervals_in_dimension(0)), \
           _finite(cc.persistence_intervals_in_dimension(1))


def _process(task: Tuple[int, str]):
    vi, path_str = task
    ex: ZoneTDAExtractor = _W["ex"]
    n = len(ARMS) * len(T_VALUES)
    empty = tuple(np.zeros((0, 2)) for _ in range(2 * n))
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str)).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        f = ex.prepare_filtration(img)
        if f is None:
            return (vi,) + empty
        f = np.asarray(f, dtype=np.float32)
        rg = float(f.max() - f.min())
        f_norm = (f - f.min()) / rg if rg > 0 else np.zeros_like(f)

        h, w = f_norm.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        y_norm = yy / max(h - 1, 1)
        x_norm = xx / max(w - 1, 1)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r = np.hypot(yy - cy, xx - cx)
        r_norm = r / max(float(r.max()), 1e-6)
        coords = {"y": y_norm, "x": x_norm, "r": r_norm}

        out = []
        for arm in ARMS:
            for t in T_VALUES:
                h0, h1 = _ph(f_norm + np.float32(t) * coords[arm])
                out.extend([h0, h1])
        return (vi,) + tuple(out)
    except Exception:
        return (vi,) + empty


def rasterize(dgm, br, pr) -> np.ndarray:
    img = np.zeros((GRID, GRID), dtype=np.float64)
    if len(dgm) == 0:
        return img
    b, pers = dgm[:, 0], dgm[:, 1] - dgm[:, 0]
    ok = np.isfinite(b) & np.isfinite(pers) & (pers > 0)
    if not np.any(ok):
        return img
    b, pers = np.clip(b[ok], *br), np.clip(pers[ok], *pr)
    bw, pw = max(br[1] - br[0], 1e-9), max(pr[1] - pr[0], 1e-9)
    cols = np.minimum(((b - br[0]) / bw * GRID).astype(int), GRID - 1)
    rows = np.minimum(((pers - pr[0]) / pw * GRID).astype(int), GRID - 1)
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


def best_fusion(img: pd.Series, tda: pd.Series, meta: pd.DataFrame):
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
    print("V2.1/V2.2 — KONUM TABANLI FILTRASYONLAR (yon ve radyal), FUZYON olcutu")
    print("=" * 78)
    print(f"Zon: {n} | kollar: {ARMS} | t={T_VALUES} | esik d_fuzyon >= +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR.")

    labels = [f"{a}_t{t}" for a in ARMS for t in T_VALUES]
    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        D = {k: (z[f"{k}_h0"], z[f"{k}_h1"]) for k in labels}
        print("Onbellek yuklendi.")
    else:
        tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
                 for r in sub.itertuples(index=False)]
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        D = {k: (np.empty(n, object), np.empty(n, object)) for k in labels}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for res in pool.map(_process, tasks, chunksize=8):
                i = pos[res[0]]
                for j, k in enumerate(labels):
                    D[k][0][i] = res[1 + 2 * j]
                    D[k][1][i] = res[2 + 2 * j]
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
        print(f"  Cikarim: {len(tasks)} zon x {len(labels)} filtrasyon, {time.time()-t0:.0f} s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{f"{k}_{d}": D[k][j] for k in labels for j, d in enumerate(("h0", "h1"))})

    blocks = {}
    for k in labels:
        r0, r1 = ranges_from(D[k][0], trm), ranges_from(D[k][1], trm)
        blocks[k] = np.stack([np.concatenate([rasterize(D[k][0][i], *r0).ravel(),
                                              rasterize(D[k][1][i], *r1).ravel()])
                              for i in range(n)])

    X_pi = np.load(args.pi_features)[sub["vector_index"].values]
    X_res = np.load(args.resnet_features)[sub["vector_index"].values]
    reps: Dict[str, np.ndarray] = {"A_PI": X_pi}
    for a in ARMS:
        reps[f"B_{a}"] = np.concatenate([X_pi] + [blocks[f"{a}_t{t}"] for t in T_VALUES], axis=1)
    rng = np.random.RandomState(SEED)
    reps["C_kontrol_y"] = np.concatenate(
        [X_pi] + [blocks[f"y_t{t}"][rng.permutation(n)] for t in T_VALUES], axis=1)

    print("\n  Goruntu dali (donmus ResNet18 + lojistik)...", flush=True)
    img = resnet_probs(X_res, sub, "val")
    img_auc = float(roc_auc_score(meta["patient_label"].values, img.reindex(meta.index).values))
    print(f"    goruntu tek basina = {img_auc:.4f}")

    out: Dict[str, Dict[str, float]] = {}
    for tag, X in reps.items():
        print(f"\n  SVM ({tag}, {X.shape[1]} boyut)...", flush=True)
        tda = svm_probs(X, sub, "val")
        tv = tda.reindex(meta.index).values
        tda_auc = float(roc_auc_score(meta["patient_label"].values, tv))
        fus_auc, a = best_fusion(img, tda, meta)
        iv = img.reindex(meta.index).values
        r_corr = float(np.corrcoef(iv, tv)[0, 1])
        out[tag] = {"tda_alone_val_auc": tda_auc, "fusion_val_auc": fus_auc, "fusion_a": a,
                    "r_tda_resnet": r_corr, "dim": int(X.shape[1])}
        print(f"    TDA-tek={tda_auc:.4f} | FUZYON={fus_auc:.4f} (a={a:.2f}) | r={r_corr:.3f}", flush=True)

    base = out["A_PI"]["fusion_val_auc"]
    print(f"\n--- SONUC (val) — v4 olcutu: d_fuzyon ---")
    print(f"  {'Temsil':18s} {'boyut':>6} {'TDA-tek':>9} {'FUZYON':>9} {'d_fuzyon':>9} {'d_tek':>9} {'r':>7}")
    best_tag, best_delta = None, -9.0
    for tag, r in out.items():
        d = r["fusion_val_auc"] - base
        dt = r["tda_alone_val_auc"] - out["A_PI"]["tda_alone_val_auc"]
        print(f"  {tag:18s} {r['dim']:6d} {r['tda_alone_val_auc']:9.4f} {r['fusion_val_auc']:9.4f} "
              f"{d:+9.4f} {dt:+9.4f} {r['r_tda_resnet']:7.3f}")
        if tag.startswith("B_") and d > best_delta:
            best_delta, best_tag = d, tag
    passed = best_delta >= MIN_DELTA
    print(f"\n  En iyi aday: {best_tag} (d_fuzyon {best_delta:+.4f}) | kriter >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Konum filtrasyonlari fuzyona yeterli katki saglamiyor.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"En az bir konum filtrasyonu fuzyonu val'de d_fuzyon >= +{MIN_DELTA} ile gecer.",
        "scope": "validation only; test setine dokunulmadi",
        "criterion": "v4: d_fuzyon (TDA-tek AUC karar olcutu DEGIL)",
        "arms": ARMS, "t_values": T_VALUES,
        "image_branch_alone_val_auc": img_auc,
        "representations": out,
        "baseline_fusion_val_auc": base,
        "best_candidate": best_tag,
        "best_fusion_delta": best_delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features, args.resnet_features],
        seed=SEED, decision=report["decision"],
        metrics={"baseline_fusion_val_auc": base, "best_fusion_delta": best_delta},
        notes=f"Konum filtrasyonlari: en iyi {best_tag} (d_fuzyon {best_delta:+.4f}); karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
