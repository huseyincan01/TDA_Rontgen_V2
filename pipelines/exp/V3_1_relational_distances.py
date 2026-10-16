"""V3.1 — İlişkisel temsil: 5 zon arasındaki 10 ikili topolojik mesafe.

Neden bu deney (v4'ün asıl fırsatı):
    v3/v4 boyunca TDA dalını **zon İÇİNDE** iyileştirmeye çalıştık; hepsi füzyonda
    başarısız oldu çünkü görüntü dalı da aynı zonu görüyor ⇒ **yapısal yedeklilik**
    (ölçtük: TDA-tek ↑ ⇒ r ↑ ⇒ d_fuzyon ↓).

    Bu deney **zonlar ARASI** bilgi kullanır: her hasta için 5 zonun **10 ikili
    topolojik mesafesi**. Görüntü dalı her zonu **bağımsız** işleyip max-pool yaptığı
    için bu mesafeleri **yapısal olarak üretemez**.

Ön-kayıtlı hipotez (v4 ölçütü):
    İlişkisel temsil, mevcut PI füzyonunu (0.9430) **d_fuzyon ≥ +0.015** ile geçer
    **VEYA** PI ile birleştiğinde geçer.

Tasarım:
    * Her hasta için 5 zonun diyagramları (H0, H1) — en kalıcı N sınıf.
    * **İlişkisel özellik (20)**: 10 ikili **sliced Wasserstein** mesafesi (H0 için 10, H1 için 10).
    * **Kontrol (20)**: aynı sayıda mesafe, ama zonlar **FARKLI hastalardan** rastgele
      eşleştirilmiş ⇒ dağılım aynı, **ilişki yok**. Sinyalin gerçekten ilişkiden mi
      geldiğini ayırır.
    * Kollar (hepsi hasta düzeyi): A = PI zon dalı (taban) · B = ilişkisel · C = PI + ilişkisel
    * Her kol için: hasta AUC, **yeni AUC'si (a validation'da)** ve **r(ResNet)**.

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test hastaları KULLANILMAZ.

Kullanım:
    python pipelines/exp/V3_1_relational_distances.py
    python pipelines/exp/V3_1_relational_distances.py --reuse-cache
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

EXPERIMENT_ID = "V3.1"
SLUG = "relational_distances"
SEED = 42
N_THETA = 24
N_POINTS = 100          # diyagram basina en kalici N sinif
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BASELINE_FUSION = 0.9430
CACHE = PROJECT_ROOT / "data/processed/features_relational/V3_1_diagrams.npz"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3.1 iliskisel mesafeler")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv")
    p.add_argument("--zone-splits", type=Path,
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


def _top(d: np.ndarray, n: int) -> np.ndarray:
    """En kalici n sinifi tut (gurultuyu at)."""
    if len(d) <= n:
        return d
    pers = d[:, 1] - d[:, 0]
    idx = np.argsort(pers)[::-1][:n]
    return d[np.sort(idx)]


def _process(task: Tuple[str, str]):
    key, path_str = task
    ex: ZoneTDAExtractor = _W["ex"]
    empty = (np.zeros((0, 2)), np.zeros((0, 2)))
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path_str)).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        f = ex.prepare_filtration(img)
        if f is None:
            return (key,) + empty
        cc = gudhi.CubicalComplex(dimensions=f.shape,
                                  top_dimensional_cells=np.asarray(f, dtype=np.float64).ravel())
        cc.compute_persistence()
        return key, _top(_finite(cc.persistence_intervals_in_dimension(0)), N_POINTS), \
                    _top(_finite(cc.persistence_intervals_in_dimension(1)), N_POINTS)
    except Exception:
        return (key,) + empty


def quantile_profile(dgm: np.ndarray, thetas: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(thetas), len(qgrid)), dtype=np.float32)
    if len(dgm) == 0:
        return out
    b, p = dgm[:, 0], dgm[:, 1]
    for i, th in enumerate(thetas):
        proj = np.sort(b * np.cos(th) + p * np.sin(th))
        out[i] = np.quantile(proj, qgrid)
    return out


def sw(prof_a: np.ndarray, prof_b: np.ndarray) -> float:
    """Sliced Wasserstein mesafesi (p=2): sqrt(mean_theta mean_q (q_a - q_b)^2).

    Kare degil KENDISI dondurulur — boylece deger araligi makul kalir (asiri
    buyuk SW^2 degerleri SVM'i zorlastiriyordu).
    """
    d = prof_a - prof_b
    return float(np.sqrt((d * d).mean()))


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    ps = pd.read_csv(args.splits_path)
    ps = ps[ps["split"].isin(["train", "val"])].dropna(subset=["patient_label"]).copy()
    ps["patient_label"] = ps["patient_label"].astype(int)
    pids = list(ps["patient_id"])

    print("=" * 78)
    print("V3.1 — ILISKISEL TEMSIL: 5 ZON ARASINDAKI 10 IKILI TOPOLOJIK MESAFE")
    print("=" * 78)
    print(f"Hasta: {len(pids)} (train {int((ps['split']=='train').sum())}, "
          f"val {int((ps['split']=='val').sum())}) | zon basina {len(ZONE_FILES)}")
    print(f"Diyagram: en kalici {N_POINTS} sinif | SW yon sayisi {N_THETA}")
    print("TEST HASTALARI KULLANILMIYOR.")

    keys = [f"{p}|{z}" for p in pids for z in range(1, 6)]
    if args.reuse_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        D = {k: (z[f"{k}_h0"], z[f"{k}_h1"]) for k in keys}
        print("Onbellek yuklendi.")
    else:
        tasks = [(f"{p}|{z}", str(args.zones_dir / p / ZONE_FILES[z]))
                 for p in pids for z in range(1, 6)]
        D = {k: (np.zeros((0, 2)), np.zeros((0, 2))) for k in keys}
        t0 = time.time(); done = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            for key, a, b in pool.map(_process, tasks, chunksize=8):
                D[key] = (a, b)
                done += 1
                if done % 2000 == 0:
                    print(f"  {done}/{len(tasks)} | {time.time()-t0:.0f} s", flush=True)
        print(f"  Cikarim: {len(tasks)} zon, {time.time()-t0:.0f} s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{f"{k}_{d}": D[k][j] for k in keys for j, d in enumerate(("h0", "h1"))})

    n_empty = int(sum(1 for k in keys if len(D[k][0]) == 0))
    print(f"  SANITY: bos diyagram {n_empty}/{len(keys)}")
    if n_empty > 0.05 * len(keys):
        raise RuntimeError(f"COK FAZLA BOS DIYAGRAM ({n_empty}/{len(keys)})")

    thetas = np.linspace(0.0, np.pi, N_THETA, endpoint=False)
    qgrid = np.linspace(0.0, 1.0, 64)
    print("  Kantil profilleri hesaplaniyor...", flush=True)
    prof: Dict[str, Dict[int, np.ndarray]] = {}
    for p in pids:
        prof[p] = {}
        for zz in range(1, 6):
            h0, h1 = D[f"{p}|{zz}"]
            prof[p][zz] = (quantile_profile(h0, thetas, qgrid), quantile_profile(h1, thetas, qgrid))

    pairs = [(i, j) for i in range(1, 6) for j in range(i + 1, 6)]
    print(f"  {len(pairs)} ikili karsilastirma x 2 boyut = {2*len(pairs)} ozellik")

    def relational(patient_ids: List[str], shuffle_within_dim: bool, rng) -> np.ndarray:
        """20 iliskisel ozellik. shuffle_within_dim=True ise partner zonlar KARISTIRILIR."""
        out = np.zeros((len(patient_ids), 2 * len(pairs)), dtype=np.float32)
        for a, p in enumerate(patient_ids):
            for b in range(2):                     # 0 = H0, 1 = H1
                for c, (i, j) in enumerate(pairs):
                    pa = prof[p][i][b]
                    if shuffle_within_dim:
                        q = patient_ids[rng.randint(0, len(patient_ids))]
                        pb = prof[q][j][b]
                    else:
                        pb = prof[p][j][b]
                    out[a, b * len(pairs) + c] = sw(pa, pb)
        return out

    rng = np.random.RandomState(SEED)
    X_rel = relational(pids, False, rng)
    X_ctl = relational(pids, True, np.random.RandomState(SEED + 1))
    print(f"  iliskisel {X_rel.shape} | kontrol {X_ctl.shape}")
    print(f"  ornek hasta ilk 5 iliskisel mesafe: {np.round(X_rel[0,:5],3)}")
    print(f"  kontrol   ilk 5                 : {np.round(X_ctl[0,:5],3)}")

    y = ps["patient_label"].values
    tr = (ps["split"] == "train").values
    va = (ps["split"] == "val").values

    def pat_svm(X, C=1.0):
        clf = make_pipeline(StandardScaler(),
                            SVC(C=C, gamma="scale", probability=True,
                                class_weight="balanced", random_state=SEED))
        clf.fit(X[tr], y[tr])
        return clf.predict_proba(X[va])[:, 1]

    # TDA zon dali (PI) — TUM hastalar icin hasta olasiligi (train birlesik kol icin gerekli)
    zs = pd.read_csv(args.zone_splits).sort_values("vector_index").reset_index(drop=True)
    Xp = np.load(args.pi_features)
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for zz in range(1, 6):
        zm = (zs["zone_id"] == zz).values
        m = zm & zs["split"].isin(["train", "val"]).values
        clfz = make_pipeline(StandardScaler(),
                             SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clfz.fit(Xp[zm & trm], zs["zone_label"].values[zm & trm])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clfz.predict_proba(Xp[m])[:, 1]}))
    pi_pat_all = pd.concat(rows).groupby("patient_id")["p"].max()

    # Goruntu dali
    Xr = np.load(args.resnet_features)
    rows = []
    for zz in range(1, 6):
        zm = (zs["zone_id"] == zz).values
        m = zm & (zs["split"] == "val").values
        sc = StandardScaler().fit(Xr[zm & trm])
        lr = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(
            sc.transform(Xr[zm & trm]), zs["zone_label"].values[zm & trm])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": lr.predict_proba(sc.transform(Xr[m]))[:, 1]}))
    img_pat = pd.concat(rows).groupby("patient_id")["p"].max()

    val_pids = ps.loc[va, "patient_id"].values
    yv = y[va]
    iv = img_pat.reindex(val_pids).values
    piv = pi_pat_all.reindex(val_pids).values
    pi_all = pi_pat_all.reindex(pids).values

    print("\n  SVM'ler egitiliyor...", flush=True)
    p_rel = pat_svm(X_rel)
    p_ctl = pat_svm(X_ctl)

    # PI + iliskisel: egitim TUM hastalar uzerinde, degerlendirme val
    def pat_svm_full(X):
        clf = make_pipeline(StandardScaler(),
                            SVC(C=1.0, gamma="scale", probability=True,
                                class_weight="balanced", random_state=SEED))
        clf.fit(X[tr], y[tr])
        return clf.predict_proba(X)[:, 1]
    p_pi_rel_full = pat_svm_full(np.concatenate([X_rel, pi_all[:, None]], axis=1))
    p_pi_ctl_full = pat_svm_full(np.concatenate([X_ctl, pi_all[:, None]], axis=1))
    p_pi_rel = p_pi_rel_full[va]
    p_pi_ctl = p_pi_ctl_full[va]

    print(f"\n--- SONUC (val, hasta duzeyi) ---")
    print(f"  goruntu dali            : {roc_auc_score(yv, iv):.4f}")
    print(f"  PI zon dali             : {roc_auc_score(yv, piv):.4f}")

    def fusion(tda_vec, label):
        best = (-1.0, None)
        for a in A_GRID:
            s = roc_auc_score(yv, a * iv + (1 - a) * tda_vec)
            if s > best[0]:
                best = (s, float(a))
        r = float(np.corrcoef(iv, tda_vec)[0, 1])
        print(f"  {label:24s} tek={roc_auc_score(yv, tda_vec):.4f} | "
              f"FUZYON={best[0]:.4f} (a={best[1]:.2f}) | r={r:.3f} | "
              f"d_fuzyon={best[0]-BASELINE_FUSION:+.4f}")
        return {"tda_alone_val_auc": float(roc_auc_score(yv, tda_vec)),
                "fusion_val_auc": best[0], "fusion_a": best[1], "r": r,
                "delta_fusion_vs_PI_baseline": best[0] - BASELINE_FUSION}

    results = {
        "A_PI_zonal": fusion(piv, "A PI (taban)"),
        "B_iliskisel": fusion(p_rel, "B iliskisel (20)"),
        "C_PI+iliskisel": fusion(p_pi_rel, "C PI + iliskisel"),
        "D_PI+kontrol": fusion(p_pi_ctl, "D PI + KONTROL"),
        "E_iliskisel_kontrol": fusion(p_ctl, "E iliskisel KONTROL"),
    }

    best_tag = max([k for k in results if k.startswith(("B_", "C_"))],
                   key=lambda k: results[k]["fusion_val_auc"])
    best_delta = results[best_tag]["delta_fusion_vs_PI_baseline"]
    passed = best_delta >= MIN_DELTA
    print(f"\n  En iyi iliskisel kol: {best_tag} (d_fuzyon {best_delta:+.4f})")
    print(f"  Kriter >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Iliskisel temsil (zonlar arasi mesafeler) fuzyonu >= +{MIN_DELTA} iyilestirir.",
        "scope": "validation only; TEST HASTALARI KULLANILMADI",
        "design": {"features": "5 zon arasinda 10 ikili sliced-Wasserstein mesafesi x (H0,H1) = 20",
                   "diagram_points": N_POINTS, "n_theta": N_THETA,
                   "control": "D/E kollari: partner zonlar FARKLI hastalardan (iliski yok)",
                   "baseline_fusion_ref": BASELINE_FUSION},
        "results": results, "best_relational_arm": best_tag,
        "best_delta_fusion": best_delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.zone_splits], seed=SEED,
        decision=report["decision"],
        metrics={"best_delta_fusion": best_delta,
                 "B_iliskisel_fusion": results["B_iliskisel"]["fusion_val_auc"],
                 "C_PI+iliskisel_fusion": results["C_PI+iliskisel"]["fusion_val_auc"]},
        notes=f"Iliskisel mesafeler: en iyi {best_tag} (d_fuzyon {best_delta:+.4f}); karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
