"""E2.1 — Çift yönlü (diphasic) filtrasyon: iki polaritenin topolojisi birlikte.

Neden bu deney:
    Mevcut hat, kübik kompleksi **tek polaritede** kurar (mevcut `prepare_filtration`):
    H0 karanlık kümeleri, H1 döngüleri izler. Ancak bir görüntünün **ters polaritesi**
    (parlak yapılar) farklı topolojik bilgi taşır — sublevel ve superlevel kümeleri
    aynı şey değildir. E2.1 her iki polariteyi hesaplayıp birleştirir.

Ön-kayıtlı hipotez:
    Çift yönlü temsil (2 × 800 = 1600 boyut), tek polariteli PI'ın **val** ROC-AUC'sini
    **≥ +0.015** geçer.

Tasarım:
    * A = mevcut filtrasyon, B = ters filtrasyon (`f_max − f`).
    * Her polarite için H0/H1 → PI (20×20), **kendi train dağılımından** türetilen
      eksen aralıklarıyla (sabit sayı uydurulmaz).
    * Sınıflandırıcı: RBF-SVM (C=10) — E2.5/E2.7/E2.8 ile aynı protokol.
    * Tek değişken: temsil (tek polarite vs çift polarite). Split, zon, max-pooling sabit.

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E2_1_diphasic_filtration.py
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

EXPERIMENT_ID = "E2.1"
SLUG = "diphasic_filtration"
SEED = 42
GRID = 20
SIGMA = 1.0
MIN_DELTA = 0.015
CACHE = PROJECT_ROOT / "data/processed/features_diphasic/E2_1_diagrams.npz"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.1 diphasic filtration")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--use-project-pi", action="store_true",
                   help="A kolu icin PROJENIN PI ozelliklerini kullan (dogru baseline). "
                        "Aksi halde araliklar bu betikte yeniden turetilir ve baseline zayiflar.")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--shuffle-b", action="store_true",
                   help="B kolunu zonlar arasinda karistir: BOYUT ayni kalir, B'nin bilgisi yok olur. "
                        "Kazanc boyuttan mi geliyor, bilgiden mi — bunu ayirt eder.")
    p.add_argument("--tag", type=str, default="", help="rapor etiketi (ayri run dizini)")
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
    empty = (np.zeros((0, 2)),) * 4
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if img.ndim == 3:
            img = img[0]
        f = ex.prepare_filtration(img)
        if f is None:
            return (vi,) + empty
        a0, a1 = _ph(f)
        b0, b1 = _ph(np.float32(f.max()) - f)   # ters polarite
        return vi, a0, a1, b0, b1
    except Exception:
        return (vi,) + empty


def rasterize(dgm: np.ndarray, br, pr) -> np.ndarray:
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


def ranges_from(diags, train_mask) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ds = [d for d, m in zip(diags, train_mask) if m and len(d)]
    if not ds:
        return (0.0, 1.0), (0.0, 1.0)
    allb = np.concatenate([d[:, 0] for d in ds])
    allp = np.concatenate([d[:, 1] - d[:, 0] for d in ds])
    return (float(np.percentile(allb, 0.5)), float(np.percentile(allb, 99.5))), \
           (float(np.percentile(allp, 0.5)), float(np.percentile(allp, 99.5)))


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
    print("E2.1 — CIFT YONLU (DIPHASIC) FILTRASYON vs TEK POLARITE")
    print("=" * 78)
    print(f"Zon: {len(sub)} | izgara {GRID}x{GRID} sigma={SIGMA} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR.")

    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        A0, A1, B0, B1 = z["A0"], z["A1"], z["B0"], z["B1"]
        print("Onbellek yuklendi.")
    else:
        tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
                 for r in sub.itertuples(index=False)]
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        A0 = np.empty(len(sub), object); A1 = np.empty(len(sub), object)
        B0 = np.empty(len(sub), object); B1 = np.empty(len(sub), object)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for vi, a0, a1, b0, b1 in pool.map(_process, tasks, chunksize=8):
                i = pos[vi]
                A0[i], A1[i], B0[i], B1[i] = a0, a1, b0, b1
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
        print(f"  Cikarim: {len(tasks)} zon x 2 polarite, {time.time()-t0:.0f} s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, A0=A0, A1=A1, B0=B0, B1=B1)

    trm = sub["is_train_balanced_sample"].values.astype(bool)
    print("  Eksen araliklari train'den turetiliyor...", flush=True)
    rA0, rA1 = ranges_from(A0, trm), ranges_from(A1, trm)
    rB0, rB1 = ranges_from(B0, trm), ranges_from(B1, trm)
    def _fmt(r):
        return f"birth[{r[0][0]:.1f},{r[0][1]:.1f}] pers[{r[1][0]:.2f},{r[1][1]:.2f}]"
    print(f"    A (standart): H0 {_fmt(rA0)} | H1 {_fmt(rA1)}")
    print(f"    B (ters)    : H0 {_fmt(rB0)} | H1 {_fmt(rB1)}")

    if args.use_project_pi:
        # DOGRU baseline: projenin dondurulmus araliklarla uretilmis PI ozellikleri
        X_A = np.load(args.pi_features)[sub["vector_index"].values]
        print(f"  A kolu = PROJENIN PI ozellikleri {X_A.shape} (dondurulmus araliklar)")
    else:
        X_A = np.stack([np.concatenate([rasterize(A0[i], *rA0).ravel(),
                                        rasterize(A1[i], *rA1).ravel()]) for i in range(len(sub))])
    X_B = np.stack([np.concatenate([rasterize(B0[i], *rB0).ravel(),
                                    rasterize(B1[i], *rB1).ravel()]) for i in range(len(sub))])
    if args.shuffle_b:
        # BOYUT KONTROLU: B kolunun zon eslesmesini boz -> boyut ayni, bilgi yok
        rng = np.random.RandomState(SEED)
        X_B = X_B[rng.permutation(len(X_B))]
        print("  [KONTROL] B kolu zonlar arasinda KARISTIRILDI (boyut ayni, bilgi yok)")
    X_di = np.concatenate([X_A, X_B], axis=1)
    print(f"  Temsiller: tek polarite {X_A.shape} | cift polarite {X_di.shape}")

    print("\n  SVM (tek polarite)...", flush=True)
    auc_a = svm_val_patient_auc(X_A, sub, meta)
    print(f"    tek polarite val hasta AUC = {auc_a:.4f}", flush=True)
    print("  SVM (cift polarite)...", flush=True)
    auc_di = svm_val_patient_auc(X_di, sub, meta)
    print(f"    cift polarite val hasta AUC = {auc_di:.4f}", flush=True)

    delta = auc_di - auc_a
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    print(f"  Tek polarite  : {auc_a:.4f}  ({X_A.shape[1]} boyut)")
    print(f"  Cift polarite : {auc_di:.4f}  ({X_di.shape[1]} boyut)")
    print(f"  Fark          : {delta:+.4f}   (kriter >= +{MIN_DELTA} -> {passed})")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Ters polarite ek bilgi katmiyor; tek polarite yeterli.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Cift polariteli temsil, tek polariteli PI'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi (butce tukendi)",
        "config": {"grid": GRID, "sigma": SIGMA, "classifier": "RBF-SVM C=10 gamma=scale",
                   "single_polarity_source": ("proje PI (dondurulmus araliklar)" if args.use_project_pi
                                              else "bu betikte turetilen araliklar"),
                   "reversed_polarity_ranges": "train percentiles [0.5, 99.5]"},
        "single_polarity_val_auc": auc_a,
        "diphasic_val_auc": auc_di,
        "delta": delta,
        "feature_dims": {"single": int(X_A.shape[1]), "diphasic": int(X_di.shape[1])},
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG + (f"_{args.tag}" if args.tag else ""))
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=report["decision"],
        metrics={"single_polarity_val_auc": auc_a, "diphasic_val_auc": auc_di, "delta": delta},
        notes=f"Cift yonlu filtrasyon {auc_di:.4f} vs tek polarite {auc_a:.4f}; fark {delta:+.4f}; "
              f"karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
