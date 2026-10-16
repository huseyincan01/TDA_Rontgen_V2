"""E2.6 — Çok parametreli persistence (multiparameter), dilim yaklaşımı.

Neden bu deney:
    E2.6 roadmap'te "ertelendi" idi. Gerekçe: `multipers`/MMA olgunluğu belirsiz ve
    hesaplama ağır. Ancak **modülün dilimleri** (slices) pratikte uygulanabilir bir
    yaklaşımdır: iki fonksiyonlu filtrasyon `f_t = f + t·g` için her `t`'de **tek
    parametreli** PH hesaplanır ve dilimler birleştirilir. Bu, E2.1/E2.2'de
    doğruladığımız "temsil tasarımı" ekseninde **üçüncü ve farklı** bir yöndür.

    İkinci fonksiyon olarak **gradyan büyüklüğü** (Sobel) seçildi: E4.2, kalıcılık
    sınıflarının gradyanla ilişkisini ölçmüştü; gradyanı **ikinci parametre** yapmak
    sınıfları elemek yerine **katmanlaştırır** (E1.2'nin "elemek zarar veriyor"
    bulgusundan sonra mantıklı bir yön).

Ön-kayıtlı hipotez:
    Çok dilimli temsil (3 dilim, 2400 boyut), tek dilimli PI'ı (t=0, 800 boyut)
    **val** ROC-AUC'sinde **≥ +0.015** geçer.

Tasarım (E2.2'deki boyut kontrolü uygulanır):
    * f_norm = `prepare_filtration` çıktısı [0,1]'e ölçeklenmiş
    * g_norm = Sobel gradyan büyüklüğü [0,1]'e ölçeklenmiş
    * Dilimler: t ∈ {0.0, 0.5, 1.5}  →  f_t = f_norm + t·g_norm
    * **A** = t=0 (tek dilim, 800) · **B** = concat(3 dilim, 2400)
    * **C** = **boyut kontrolü**: t=0 + diğer iki dilimin **karıştırılmış** hâli (2400)
    Sınıflandırıcı/split/zon/max-pooling sabit (RBF-SVM C=10).

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E2_6_multiparameter_slices.py
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
from scipy.ndimage import gaussian_filter, sobel
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

EXPERIMENT_ID = "E2.6"
SLUG = "multiparameter_slices"
SEED = 42
GRID = 20
SIGMA = 1.0
T_VALUES = [0.0, 0.5, 1.5]
MIN_DELTA = 0.015
CACHE = PROJECT_ROOT / "data/processed/features_multiparam/E2_6_diagrams.npz"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.6 cok parametreli persistence (dilimler)")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--use-project-pi", action="store_true",
                   help="t=0 kolunu PROJENIN PI ozellikleriyle kur (dogru baseline). "
                        "Aksi halde araliklar bu betikte turetilir ve baseline zayiflar "
                        "(E2.1'de oldugu gibi kazanc sisirilir).")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
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
    n_slice = len(T_VALUES)
    empty = tuple(np.zeros((0, 2)) for _ in range(2 * n_slice))
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str)).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        f = ex.prepare_filtration(img)
        if f is None:
            return (vi,) + empty
        f = np.asarray(f, dtype=np.float32)
        rng = float(f.max() - f.min())
        f_norm = (f - f.min()) / rng if rng > 0 else np.zeros_like(f)
        g = np.hypot(sobel(f_norm, axis=0), sobel(f_norm, axis=1))
        gm = float(g.max())
        g_norm = (g / gm).astype(np.float32) if gm > 0 else np.zeros_like(g)

        out = []
        for t in T_VALUES:
            h0, h1 = _ph(f_norm + np.float32(t) * g_norm)
            out.extend([h0, h1])
        return (vi,) + tuple(out)
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


def ranges_from(diags, train_mask):
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
    print("E2.6 — COK PARAMETRELI PERSISTENCE (dilim yaklasimi)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | dilimler t={T_VALUES} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR.")

    keys = [f"t{i}" for i in range(len(T_VALUES))]
    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        D = {k: (z[f"{k}_h0"], z[f"{k}_h1"]) for k in keys}
        print("Onbellek yuklendi.")
    else:
        tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
                 for r in sub.itertuples(index=False)]
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        D = {k: (np.empty(len(sub), object), np.empty(len(sub), object)) for k in keys}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for res in pool.map(_process, tasks, chunksize=8):
                i = pos[res[0]]
                for j, k in enumerate(keys):
                    D[k][0][i] = res[1 + 2 * j]
                    D[k][1][i] = res[2 + 2 * j]
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
        print(f"  Cikarim: {len(tasks)} zon x {len(T_VALUES)} dilim, {time.time()-t0:.0f} s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{f"{k}_{d}": D[k][j] for k in keys for j, d in enumerate(("h0", "h1"))})

    trm = sub["is_train_balanced_sample"].values.astype(bool)
    blocks = {}
    for k in keys:
        r0, r1 = ranges_from(D[k][0], trm), ranges_from(D[k][1], trm)
        blocks[k] = np.stack([np.concatenate([rasterize(D[k][0][i], *r0).ravel(),
                                              rasterize(D[k][1][i], *r1).ravel()])
                              for i in range(len(sub))])
    rng = np.random.RandomState(SEED)
    if args.use_project_pi:
        X_A = np.load(args.pi_features)[sub["vector_index"].values]
        X_B = np.concatenate([X_A] + [blocks[k] for k in keys[1:]], axis=1)
        print(f"  t=0 kolu = PROJENIN PI ozellikleri (dondurulmus araliklar), {X_A.shape}")
    else:
        X_A = blocks[keys[0]]
        X_B = np.concatenate([blocks[k] for k in keys], axis=1)
    X_C = np.concatenate([X_A] + [blocks[k][rng.permutation(len(sub))] for k in keys[1:]], axis=1)
    print(f"  Temsiller: A (tek dilim) {X_A.shape} | B (cok dilim) {X_B.shape} | C (kontrol) {X_C.shape}")

    print("\n  SVM (A: tek dilim)...", flush=True)
    auc_A = svm_val_patient_auc(X_A, sub, meta)
    print(f"    A = {auc_A:.4f}", flush=True)
    print("  SVM (C: boyut kontrolu, karistirilmis)...", flush=True)
    auc_C = svm_val_patient_auc(X_C, sub, meta)
    print(f"    C = {auc_C:.4f}", flush=True)
    print("  SVM (B: cok dilim)...", flush=True)
    auc_B = svm_val_patient_auc(X_B, sub, meta)
    print(f"    B = {auc_B:.4f}", flush=True)

    delta = auc_B - auc_A
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    print(f"  A tek dilim (t=0)      : {auc_A:.4f}  ({X_A.shape[1]} boyut)")
    print(f"  B cok dilim (3 t)      : {auc_B:.4f}  ({X_B.shape[1]} boyut)")
    print(f"  C boyut kontrolu       : {auc_C:.4f}  ({X_C.shape[1]} boyut)")
    print(f"  B - A                  : {delta:+.4f}   (kriter >= +{MIN_DELTA} -> {passed})")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Cok parametreli dilimler ek bilgi katmiyor; tek dilim yeterli.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Cok dilimli temsil, tek dilimli PI'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi",
        "design": {"slices": T_VALUES, "second_function": "Sobel gradyan buyuklugu (normalize)",
                   "filtration": "f_norm + t * g_norm", "classifier": "RBF-SVM C=10",
                   "single_slice_source": ("proje PI (dondurulmus araliklar)" if args.use_project_pi
                                           else "bu betikte turetilen araliklar"),
                   "control": "C = t=0 + shuffled copies of the other slices"},
        "single_slice_val_auc": auc_A,
        "multi_slice_val_auc": auc_B,
        "dimension_control_val_auc": auc_C,
        "delta": delta,
        "feature_dims": {"single": int(X_A.shape[1]), "multi": int(X_B.shape[1])},
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=report["decision"],
        metrics={"single_slice_val_auc": auc_A, "multi_slice_val_auc": auc_B, "delta": delta},
        notes=f"Cok parametreli dilimler {auc_B:.4f} vs tek dilim {auc_A:.4f}; fark {delta:+.4f}; "
              f"karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
