"""V2.4 — Hasta düzeyi global kalıcı homoloji (tüm akciğer, TEK diyagram).

Neden bu deney (v4 Faz V2):
    V1.1/V1.2/V2.1 gösterdi ki **tek-zon** temsilini iyileştirmek füzyona yansımıyor —
    çünkü görüntü dalı da aynı zonu görüyor ⇒ **yapısal olarak yedekli**.

    Bu deney **yapısal olarak farklı** bir temsil kurar: 5 zon ayrı ayrı değil,
    **birleşik akciğer bölgesi** üzerinde **tek** kalıcı homoloji hesabı. Böylece
    zonlar **arası** ilişkiler ve organ düzeyi yapı kodlanır — görüntü dalının
    (zon başına bağımsız + max-pool) temsil **edemediği** bir şey.

    Ayrıca bu, v3'ün en güçlü **olumsuz** bulgusuyla tutarlıdır (E4.7 + E1.2 + E3.6:
    sinyal lokal değil **global**).

Ön-kayıtlı hipotez (v4 ölçütü):
    Global PH temsili, mevcut PI füzyonunu val'de **d_fuzyon ≥ +0.015** ile geçer.
    Ya da en azından: korelasyonu **r < 0.29** (mevcut taban) tutarken rekabetçi AUC verir.

Tasarım:
    * Hasta düzeyi: train 3414 hasta (patient_label), val 734 hasta. **TEST YOK.**
    * Görüntü: birleşik akciğer maskesi uygulanmış, akciğer bbox'ına kırpılmış,
      en büyük kenarı ~320'ye yeniden ölçeklenmiş.
    * Kübik PH → H0/H1 → PI (2×20×20 = 800), aralıklar train'den.
    * SVM (sınıf dengesi ağırlıklı) → hasta olasılığı (max-pool YOK — zaten hasta düzeyi).
    * Görüntü dalı: donmuş ResNet18 zon sondası + max-pool (sabit, v3 protokolü).

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/V2_4_patient_level_global_ph.py
    python pipelines/exp/V2_4_patient_level_global_ph.py --reuse-cache
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

EXPERIMENT_ID = "V2.4"
SLUG = "patient_level_global_ph"
SEED = 42
GRID, SIGMA = 20, 1.0
MAX_SIDE = 320
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CACHE = PROJECT_ROOT / "data/processed/features_global_ph/V2_4_diagrams.npz"
IMG_DIR = PROJECT_ROOT / "dataset_node21/cxr_images/proccessed_data/images"
PAT_DIR = PROJECT_ROOT / "data/processed/zones/patients"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2.4 hasta duzeyi global PH")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv")
    p.add_argument("--zone-splits", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    p.add_argument("--workers", type=int, default=8)
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


def _process(task: Tuple[str, str, str]):
    pid, img_path, mask_path = task
    ex: ZoneTDAExtractor = _W["ex"]
    empty = (np.zeros((0, 2)), np.zeros((0, 2)))
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        msk = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        if img.ndim == 3:
            img = img[0]
        if msk.ndim == 3:
            msk = msk[0]
        if img.shape != msk.shape:
            return (pid,) + empty
        lung = msk > 0
        if lung.sum() < 500:
            return (pid,) + empty
        ys, xs = np.where(lung)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = img[y0:y1, x0:x1].copy()
        cmask = lung[y0:y1, x0:x1]
        crop[~cmask] = 0.0                     # akciger disi sifirlanir
        # Olcekle: en buyuk kenar <= MAX_SIDE. sitk.Resample bu surumde spacing
        # hatasi verdigi icin bagimliliksiz BLOK ORTALAMASI kullanilir.
        h, w = crop.shape
        fac = int(np.ceil(max(h, w) / MAX_SIDE))
        if fac > 1:
            nh, nw = (h // fac) * fac, (w // fac) * fac
            crop = crop[:nh, :nw].reshape(nh // fac, fac, nw // fac, fac).mean(axis=(1, 3))
            crop = crop.astype(np.float32)
        f = ex.prepare_filtration(crop)
        if f is None:
            return (pid,) + empty
        cc = gudhi.CubicalComplex(dimensions=f.shape,
                                  top_dimensional_cells=np.asarray(f, dtype=np.float64).ravel())
        cc.compute_persistence()
        return pid, _finite(cc.persistence_intervals_in_dimension(0)), \
                    _finite(cc.persistence_intervals_in_dimension(1))
    except Exception:
        return (pid,) + empty


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


def ranges_from(diags, keep):
    ds = [d for d, k in zip(diags, keep) if k and len(d)]
    if not ds:
        return (0.0, 1.0), (0.0, 1.0)
    ab = np.concatenate([d[:, 0] for d in ds]); ap = np.concatenate([d[:, 1] - d[:, 0] for d in ds])
    return (float(np.percentile(ab, 0.5)), float(np.percentile(ab, 99.5))), \
           (float(np.percentile(ap, 0.5)), float(np.percentile(ap, 99.5)))


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    ps = pd.read_csv(args.splits_path)
    ps = ps[ps["split"].isin(["train", "val"])].copy()
    ps = ps[ps["split"].isin(["train", "val"])]
    ps = ps.dropna(subset=["patient_label"])
    ps["patient_label"] = ps["patient_label"].astype(int)

    print("=" * 78)
    print("V2.4 — HASTA DUZEYI GLOBAL PH (birlesik akciger, TEK diyagram)")
    print("=" * 78)
    print(f"Hasta: {len(ps)} (train {int((ps['split']=='train').sum())}, "
          f"val {int((ps['split']=='val').sum())}) | olcek: en buyuk kenar {MAX_SIDE}")
    print("TEST HASTALARI KULLANILMIYOR.")

    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        pids = list(z["pids"]); h0 = z["h0"]; h1 = z["h1"]
        print("Onbellek yuklendi.")
    else:
        tasks = [(r.patient_id, str(IMG_DIR / f"{r.patient_id}.mha"),
                  str(PAT_DIR / r.patient_id / "mask_5zones.mha")) for r in ps.itertuples(index=False)]
        h0 = np.empty(len(tasks), object); h1 = np.empty(len(tasks), object)
        pids = [t[0] for t in tasks]
        pos = {p: i for i, p in enumerate(pids)}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for pid, a, b in pool.map(_process, tasks, chunksize=8):
                i = pos[pid]; h0[i] = a; h1[i] = b
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
        print(f"  Cikarim: {len(tasks)} hasta, {time.time()-t0:.0f} s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, pids=np.array(pids, dtype=object), h0=h0, h1=h1)

    # SANITY: sessiz except'in sistematik hatayi gizlemesini engelle
    n_empty = int(sum(1 for a in h0 if len(a) == 0))
    print(f"  SANITY: bos diyagram {n_empty}/{len(h0)}")
    if n_empty > 0.05 * len(h0):
        raise RuntimeError(
            f"COK FAZLA BOS DIYAGRAM ({n_empty}/{len(h0)}) — cikarim sistematik olarak "
            f"basarisiz. 'except Exception' hatayi gizliyor olabilir; once teshis et.")

    info = ps.set_index("patient_id").reindex(pids)
    y = info["patient_label"].values
    tr = (info["split"] == "train").values
    va = (info["split"] == "val").values
    print(f"  train hasta: {int(tr.sum())} (poz {int(y[tr].sum())}) | "
          f"val hasta: {int(va.sum())} (poz {int(y[va].sum())})")

    r0, r1 = ranges_from(h0, tr), ranges_from(h1, tr)
    print(f"  araliklar: H0 birth[{r0[0][0]:.1f},{r0[0][1]:.1f}] pers[{r0[1][0]:.2f},{r0[1][1]:.2f}] | "
          f"H1 birth[{r1[0][0]:.1f},{r1[0][1]:.1f}] pers[{r1[1][0]:.2f},{r1[1][1]:.2f}]")
    X = np.stack([np.concatenate([rasterize(h0[i], *r0).ravel(),
                                  rasterize(h1[i], *r1).ravel()]) for i in range(len(pids))])
    print(f"  Global PH ozellikleri: {X.shape}")

    # --- TDA dali: hasta duzeyi ---
    print("\n  SVM (hasta duzeyi global PH, sinif dengeli)...", flush=True)
    clf = make_pipeline(StandardScaler(),
                        SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced",
                            random_state=SEED))
    clf.fit(X[tr], y[tr])
    p_glob = clf.predict_proba(X[va])[:, 1]
    yv = y[va]
    auc_glob = float(roc_auc_score(yv, p_glob))
    print(f"    global PH tek basina val AUC = {auc_glob:.4f}")

    # --- Goruntu dali: donmus ResNet18 zon sondasi + max-pool (v3 protokolu) ---
    zs = pd.read_csv(args.zone_splits).sort_values("vector_index").reset_index(drop=True)
    Xr = np.load(args.resnet_features)
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for zz in range(1, 6):
        zm = (zs["zone_id"] == zz).values
        m = zm & (zs["split"] == "val").values
        sc = StandardScaler().fit(Xr[zm & trm])
        lr = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(
            sc.transform(Xr[zm & trm]), zs["zone_label"].values[zm & trm])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": lr.predict_proba(sc.transform(Xr[m]))[:, 1]}))
    img = pd.concat(rows).groupby("patient_id")["p"].max()

    # eslesme: val hasta sirasi = pids[va]
    val_pids = np.array(pids)[va]
    img_v = img.reindex(val_pids).values
    img_auc = float(roc_auc_score(yv, img_v))
    print(f"    goruntu dali tek basina val AUC = {img_auc:.4f}")

    best = (-1.0, None)
    for a in A_GRID:
        s = roc_auc_score(yv, a * img_v + (1 - a) * p_glob)
        if s > best[0]:
            best = (s, float(a))
    fus_auc, a_star = best
    r_corr = float(np.corrcoef(img_v, p_glob)[0, 1])
    print(f"\n--- SONUC (val) ---")
    print(f"  Global PH tek basina : {auc_glob:.4f}  [karsilastirma: PI tabani 0.8399]")
    print(f"  Goruntu dali         : {img_auc:.4f}")
    print(f"  FUZYON (a={a_star:.2f})       : {fus_auc:.4f}  [PI fuzyon tabani 0.9430]")
    print(f"  d_fuzyon             : {fus_auc - 0.9430:+.4f}")
    print(f"  r(globalPH, ResNet)  : {r_corr:.3f}  [PI tabani 0.293]")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Global (tum akciger) PH temsili, tek-zon temsilinden daha az yedekli olup fuzyona katki saglar.",
        "scope": "validation only; TEST HASTALARI KULLANILMADI",
        "design": {"level": "hasta duzeyi (tek diyagram)", "mask": "5 zonun birlesimi",
                   "max_side": MAX_SIDE, "classifier": "RBF-SVM C=1 class_weight=balanced",
                   "baseline_fusion_ref": 0.9430, "baseline_r_ref": 0.293},
        "global_ph_val_auc": auc_glob,
        "image_branch_val_auc": img_auc,
        "fusion_val_auc": fus_auc,
        "fusion_a": a_star,
        "delta_fusion_vs_PI_baseline": fus_auc - 0.9430,
        "r_global_ph_vs_resnet": r_corr,
        "n_train_patients": int(tr.sum()),
        "n_val_patients": int(va.sum()),
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.zone_splits], seed=SEED,
        decision="KEEP" if (fus_auc - 0.9430) >= MIN_DELTA else "KILL",
        metrics={"global_ph_val_auc": auc_glob, "fusion_val_auc": fus_auc,
                 "delta_fusion": fus_auc - 0.9430, "r": r_corr},
        notes=f"Hasta duzeyi global PH: tek {auc_glob:.4f}, fuzyon {fus_auc:.4f}, d={fus_auc-0.9430:+.4f}, r={r_corr:.3f}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
