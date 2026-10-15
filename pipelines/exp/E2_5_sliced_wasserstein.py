"""E2.5 — Sliced Wasserstein (SW) gömlemesi vs Kalıcılık İmgesi (PI).

Neden bu deney:
    E2.8, PI'ın ızgara çözünürlüğüne duyarlı olduğunu gösterdi (15/20/25 → 0.8545/0.8697/0.8469):
    ızgara hem bilgi kaybettiriyor hem yapay duyarlılık getiriyor. **Sliced Wasserstein**
    diyagramları ızgarasız, permütasyon-değişmez ve ölçeklenebilir biçimde temsil eder.
    Faz 4'ün (yüksek risk) ilk deneyi.

Yöntem:
    1. Her zon için H0/H1 diyagramları yeniden hesaplanır (önbellekte diyagram yok).
    2. **SW gömlemesi:** N_θ yön boyunca 1B Wasserstein mesafesi; her diyagram için
       **K çapa diyagramına** olan SW mesafeleri özellik vektörü yapılır.
    3. Aynı protokolde RBF-SVM ile sınıflandırılır; **PI + RBF-SVM** ile karşılaştırılır.

Ön-kayıtlı hipotez:
    SW gömlemesi + SVM, PI + SVM'in **val** ROC-AUC'sini **≥ +0.015** geçer.

Tek değişken:
    Diyagram temsili (SW gömlemesi vs PI). Sınıflandırıcı (RBF-SVM), split, zon yapısı
    ve hasta düzeyi max-pooling **sabit**.

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E2_5_sliced_wasserstein.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gudhi
import numpy as np
import pandas as pd
import SimpleITK as sitk
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

EXPERIMENT_ID = "E2.5"
SLUG = "sliced_wasserstein"
SEED = 42
N_THETA = 24
Q_GRID = 128
N_ANCHORS = 96
MIN_DELTA = 0.015
PI_VAL_REF = None  # ayni kosuda yeniden hesaplanir
CACHE = PROJECT_ROOT / "data/processed/features_sw/E2_5_diagrams.npz"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.5 sliced Wasserstein vs PI")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--reuse-cache", action="store_true")
    return p.parse_args()


_W: Dict[str, Any] = {}


def _init_worker() -> None:
    _W["ex"] = ZoneTDAExtractor(TDASettings())


def _finite(intervals: np.ndarray) -> np.ndarray:
    if intervals is None or len(intervals) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    d = np.asarray(intervals, dtype=np.float64)
    d = d[np.isfinite(d).all(axis=1)]
    d = d[(d[:, 1] - d[:, 0]) > 0]
    return d


def _process(task: Tuple[int, str]) -> Tuple[int, np.ndarray, np.ndarray]:
    vi, path_str = task
    ex: ZoneTDAExtractor = _W["ex"]
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if img.ndim == 3:
            img = img[0]
        f = ex.prepare_filtration(img)
        if f is None:
            return vi, np.zeros((0, 2)), np.zeros((0, 2))
        cc = gudhi.CubicalComplex(dimensions=f.shape,
                                  top_dimensional_cells=f.astype(np.float64).ravel())
        cc.compute_persistence()
        return vi, _finite(cc.persistence_intervals_in_dimension(0)), \
                   _finite(cc.persistence_intervals_in_dimension(1))
    except Exception:
        return vi, np.zeros((0, 2)), np.zeros((0, 2))


def quantile_profile(dgm: np.ndarray, thetas: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    """(n_theta, Q) — her yonde projeksiyonun ters-CDF'i (ızgara uzerinde)."""
    out = np.zeros((len(thetas), len(qgrid)), dtype=np.float32)
    if len(dgm) == 0:
        return out
    b, p = dgm[:, 0], dgm[:, 1]
    for i, th in enumerate(thetas):
        proj = np.sort(b * np.cos(th) + p * np.sin(th))
        out[i] = np.quantile(proj, qgrid)
    return out


def build_cache(sub: pd.DataFrame, args) -> Dict[str, np.ndarray]:
    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"Onbellek yuklendi: {z['h0'].shape}")
        return {"h0": z["h0"], "h1": z["h1"]}
    tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
             for r in sub.itertuples(index=False)]
    pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
    h0 = np.empty(len(sub), dtype=object); h1 = np.empty(len(sub), dtype=object)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        done = 0
        for vi, a, b in pool.map(_process, tasks, chunksize=8):
            i = pos[vi]; h0[i] = a; h1[i] = b
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
    print(f"  Diyagram cikarimi: {len(tasks)} zon, {time.time()-t0:.0f} s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, h0=h0, h1=h1)
    return {"h0": h0, "h1": h1}


def sw_features(prof: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """prof: (N, n_theta, Q) | anchors: (K, n_theta, Q) -> (N, K) SW^2 ozellikleri."""
    # (N,1,T,Q) - (1,K,T,Q) -> (N,K,T,Q); Q uzerinden ortalama, T uzerinden ortalama
    d = prof[:, None, :, :] - anchors[None, :, :, :]
    return (d * d).mean(axis=(2, 3)).astype(np.float32)


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
    print("E2.5 — SLICED WASSERSTEIN GOMLEMESI vs KALICILIK IMGESI (PI)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | N_theta={N_THETA} Q={Q_GRID} K_anchor={N_ANCHORS} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR.")

    dg = build_cache(sub, args)
    thetas = np.linspace(0.0, np.pi, N_THETA, endpoint=False)
    qgrid = np.linspace(0.0, 1.0, Q_GRID)

    print("  Kantil profilleri hesaplaniyor...", flush=True)
    t0 = time.time()
    feats = {}
    for dim in ("h0", "h1"):
        P = np.stack([quantile_profile(d, thetas, qgrid) for d in dg[dim]])
        tr_idx = np.where(sub["is_train_balanced_sample"].values.astype(bool))[0]
        rng = np.random.RandomState(SEED)
        a_idx = rng.choice(tr_idx, size=min(N_ANCHORS, len(tr_idx)), replace=False)
        feats[dim] = sw_features(P, P[a_idx])
        print(f"    {dim}: profil {P.shape} -> SW ozellik {feats[dim].shape} ({time.time()-t0:.0f} s)",
              flush=True)
    X_sw = np.concatenate([feats["h0"], feats["h1"]], axis=1)
    print(f"  SW ozellik matrisi: {X_sw.shape}")

    X_pi = np.load(args.pi_features)[sub["vector_index"].values]

    print("\n  SVM egitimi (PI)...", flush=True)
    auc_pi = svm_val_patient_auc(X_pi, sub, meta)
    print(f"    PI + RBF-SVM val hasta AUC = {auc_pi:.4f}", flush=True)
    print("  SVM egitimi (SW)...", flush=True)
    auc_sw = svm_val_patient_auc(X_sw, sub, meta)
    print(f"    SW + RBF-SVM val hasta AUC = {auc_sw:.4f}", flush=True)

    delta = auc_sw - auc_pi
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    print(f"  PI + SVM : {auc_pi:.4f}")
    print(f"  SW + SVM : {auc_sw:.4f}   (fark {delta:+.4f})")
    print(f"  Kriter: >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"SW gomlemesi + SVM, PI + SVM'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi (butce tukendi)",
        "config": {"n_theta": N_THETA, "q_grid": Q_GRID, "n_anchors": N_ANCHORS,
                   "classifier": "RBF-SVM C=10 gamma=scale"},
        "pi_svm_val_patient_auc": auc_pi,
        "sw_svm_val_patient_auc": auc_sw,
        "delta": delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features], seed=SEED,
        decision=report["decision"],
        metrics={"pi_svm_val_auc": auc_pi, "sw_svm_val_auc": auc_sw, "delta": delta},
        notes=f"SW gomlemesi {auc_sw:.4f} vs PI {auc_pi:.4f}; fark {delta:+.4f}; karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
