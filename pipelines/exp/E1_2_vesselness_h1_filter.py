"""E1.2 — Frangi damar-bastırma ile H1 diyagram filtrelmesi.

Neden bu deney:
    E4.4, H1 üst-zarfının her zonda **≥5 çok uzun ömürlü arka plan sınıfı** (bronkovasküler
    döngüler + kaburga kenarları) tarafından doyurulduğunu ve sınıf sinyalinin zarfın
    yalnızca %1–2'si kaldığını gösterdi. E1.2, bu arka plan sınıflarını **diyagram
    düzeyinde** süzer: görüntüyü bozmadan, yalnızca yüksek damarsallık (Frangi vesselness)
    bölgesinde doğan H1 sınıflarını eler.

Ön-kayıtlı hipotez:
    Damar-baskın H1 sınıflarının elenmesi, 5-zon CNN'in ortalama val ROC-AUC'sini
    **≥ +0.015** artırır (3 seed ortalaması).

Tek değişken:
    H1 filtrelme eşiği `q` — doğum hücresinin Frangi vesselness değeri, doku vesselness
    dağılımının `q` kantilini aşıyorsa sınıf ELENİR.
      * `base` : filtre yok (mevcut hat)
      * `v90`  : en damarsal %10 elenir (hafif)
      * `v75`  : en damarsal %25 elenir
      * `v50`  : en damarsal %50 elenir (agresif)
    Diyagramlar, ızgara aralıkları, split, mimari ve eğitim protokolü **sabit**.
    (Görüntü DEĞİŞTİRİLMEZ; yalnızca H1 diyagramı süzülür.)

Kapsam (test bütçesi korunur):
    Yalnızca train(dengeli)+val; **test setine dokunulmaz**.

Kullanım:
    python pipelines/exp/E1_2_vesselness_h1_filter.py
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
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E1.2"
SLUG = "vesselness_h1_filter"
SEEDS = [42, 43, 44]
VARIANTS = {"base": None, "v90": 0.90, "v75": 0.75, "v50": 0.50}
TOLERANCE = 0.015
GRID, SIGMA = 20, 1.0
H0_RANGES = ((0.0, 248.0), (0.0, 248.0))
H1_RANGES = ((239.0, 250.0), (0.0, 7.0))
FEATURE_DIR = PROJECT_ROOT / "data/processed/features_vessel/E1_2"
INDEX_PATH = FEATURE_DIR / "vector_index.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}


def rasterize(diagram: np.ndarray, birth_range, persistence_range) -> np.ndarray:
    image = np.zeros((GRID, GRID), dtype=np.float64)
    if len(diagram) == 0:
        return image
    d = np.asarray(diagram, dtype=np.float64)
    births, pers = d[:, 0], d[:, 1] - d[:, 0]
    valid = np.isfinite(births) & np.isfinite(pers) & (pers > 0)
    if not np.any(valid):
        return image
    births = np.clip(births[valid], *birth_range)
    pers = np.clip(pers[valid], *persistence_range)
    bw = birth_range[1] - birth_range[0]
    pw = persistence_range[1] - persistence_range[0]
    cols = np.minimum(((births - birth_range[0]) / bw * GRID).astype(int), GRID - 1)
    rows = np.minimum(((pers - persistence_range[0]) / pw * GRID).astype(int), GRID - 1)
    np.add.at(image, (rows, cols), pers)
    return gaussian_filter(image, sigma=SIGMA, mode="constant", truncate=3.0)


_W: Dict[str, Any] = {}


def _init_worker() -> None:
    _W["ex"] = ZoneTDAExtractor(TDASettings())


def _empty() -> Dict[str, np.ndarray]:
    return {name: np.zeros(800, dtype=np.float32) for name in VARIANTS}


def _process(task: Tuple[int, str]) -> Tuple[int, Dict[str, np.ndarray]]:
    vi, path_str = task
    ex: ZoneTDAExtractor = _W["ex"]
    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if image.ndim == 3:
            image = image[0]
        # extractor ile birebir ayni crop (drift korumasi asagida)
        valid = np.isfinite(image) & (image > 0)
        if not np.any(valid):
            return vi, _empty()
        ys, xs = np.where(valid)
        tissue = np.asarray(image[ys.min():ys.max() + 1, xs.min():xs.max() + 1], dtype=np.float32)
        tmask = valid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        filtration = ex.prepare_filtration(image)
        if filtration is None or filtration.shape != tissue.shape:
            return vi, _empty()

        vessel = frangi(tissue, sigmas=[1, 2, 3, 4], black_ridges=False).astype(np.float64)
        vessel_q = {q: float(np.quantile(vessel[tmask], q)) for q in (0.90, 0.75, 0.50)}

        cc = gudhi.CubicalComplex(dimensions=filtration.shape,
                                  top_dimensional_cells=filtration.astype(np.float64).ravel())
        cc.compute_persistence()
        h0 = cc.persistence_intervals_in_dimension(0)
        h1 = cc.persistence_intervals_in_dimension(1)
        h0 = h0[np.isfinite(h0[:, 1])]
        h1f = h1[np.isfinite(h1[:, 1])]
        pairs_h1 = np.asarray(cc.cofaces_of_persistence_pairs()[0][1]) if len(h1f) else np.empty((0, 2))

        # H1 siniflarinin DOGUM hucrelerindeki vesselness
        bv = np.full(len(h1f), np.nan)
        if len(pairs_h1) == len(h1f):
            shape = filtration.shape
            flat = filtration.ravel()
            for i, cells in enumerate(pairs_h1):
                vals = flat[cells]
                birth_cell = cells[int(np.argmin(vals))]
                r, c = np.unravel_index(int(birth_cell), shape)
                bv[i] = vessel[r, c]

        out: Dict[str, np.ndarray] = {}
        for name, q in VARIANTS.items():
            if q is None or not np.isfinite(bv).any():
                h1_keep = h1f
            else:
                h1_keep = h1f[bv <= vessel_q[q]]
            out[name] = np.concatenate([rasterize(h0, *H0_RANGES).ravel(),
                                        rasterize(h1_keep, *H1_RANGES).ravel()]).astype(np.float32)
        return vi, out
    except Exception:
        return vi, _empty()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1.2 vesselness H1 filtreleme")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--reuse-features", action="store_true")
    return p.parse_args()


def train_val_auc(X: np.ndarray, zs: pd.DataFrame, seed: int, max_epochs: int) -> float:
    y = zs["zone_label"].values.astype(np.float32)
    preds: List[Dict[str, Any]] = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va = zm & (zs["split"] == "val").values
        vm = va & zs["is_zone_metric_sample"].values.astype(bool)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xvm, Xva = sc.transform(X[tr]), sc.transform(X[vm]), sc.transform(X[va])
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32),
                                          torch.tensor(y[tr], dtype=torch.float32)),
                            batch_size=32, shuffle=True)
        Xvm_t = torch.tensor(vector_to_2channel_image(Xvm), dtype=torch.float32)
        best_auc, best_state = -1.0, None
        for _ in range(max_epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(model(bx), by).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xvm_t)).numpy()
            a = compute_binary_metrics(y[vm], p)["roc_auc"]
            if a > best_auc:
                best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(Xva), dtype=torch.float32))).numpy()
        for pid, p in zip(zs.loc[va, "patient_id"], pv):
            preds.append({"patient_id": pid, "p": float(p)})
    pdf = pd.DataFrame(preds).groupby("patient_id")["p"].max()
    meta = zs[zs["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv = pdf.reindex(meta.index)
    return float(compute_binary_metrics(meta["patient_label"].values, pv.values)["roc_auc"])


def main() -> None:
    args = parse_args()
    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    needed = splits["is_train_balanced_sample"].values.astype(bool) | (splits["split"] == "val").values
    sub = splits[needed].reset_index(drop=True)

    print("=" * 78)
    print("E1.2 — FRANGI DAMAR-BASTIRMA ile H1 DIYAGRAM FILTRELEMESI")
    print("=" * 78)
    print(f"Kapsam: train(dengeli)+val = {len(sub)} zon | TEST SETINE DOKUNULMUYOR")
    print(f"Varyantlar: {VARIANTS} | seed'ler: {SEEDS}")

    if args.reuse_features and INDEX_PATH.exists():
        feats = {n: np.load(FEATURE_DIR / f"{n}.npy") for n in VARIANTS}
        print("Kayitli ozellikler yuklendi.")
    else:
        tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
                 for r in sub.itertuples(index=False)]
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        feats = {n: np.zeros((len(sub), 800), dtype=np.float32) for n in VARIANTS}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for vi, d in pool.map(_process, tasks, chunksize=8):
                i = pos[vi]
                for n, v in d.items():
                    feats[n][i] = v
                done += 1
                if done % 500 == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"  {done}/{len(tasks)} | {rate:.1f} zon/s | kalan ~{(len(tasks)-done)/max(rate,1e-9)/60:.1f} dk",
                          flush=True)
        print(f"  Cikarim tamamlandi: {len(tasks)} zon, {time.time()-t0:.0f} s")
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_PATH, sub["vector_index"].values)
        for n, a in feats.items():
            np.save(FEATURE_DIR / f"{n}.npy", a)

    results: Dict[str, Any] = {}
    for name in VARIANTS:
        aucs = [train_val_auc(feats[name], sub, s, args.max_epochs) for s in SEEDS]
        results[name] = {"seed_aucs": aucs, "mean_val_auc": float(np.mean(aucs)), "std": float(np.std(aucs))}
        print(f"  {name:5s}: ortalama val AUC = {np.mean(aucs):.4f} (std {np.std(aucs):.4f})", flush=True)

    base = results["base"]["mean_val_auc"]
    print(f"\n--- Referans (base, filtre yok) = {base:.4f} ---")
    best_name, best_auc = max(results.items(), key=lambda kv: kv[1]["mean_val_auc"])
    for name in VARIANTS:
        d = results[name]["mean_val_auc"] - base
        print(f"  {name:5s}: {results[name]['mean_val_auc']:.4f}  (fark {d:+.4f})")
    delta = best_auc["mean_val_auc"] - base
    passed = best_name != "base" and delta >= TOLERANCE
    decision = "KEEP" if passed else "KILL"
    print(f"\n  En iyi: {best_name} ({best_auc['mean_val_auc']:.4f}) | fark={delta:+.4f} | esik +{TOLERANCE}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Damar-baskin H1 siniflarinin elenmesi ortalama val ROC-AUC'yi >= +0.015 artirir.",
        "scope": "train(dengeli)+val only; test setine dokunulmadi; goruntu degistirilmedi (yalniz H1 diyagrami suzuldu)",
        "method": "Frangi vesselness (sigmas 1-4); H1 sinifinin DOGUM hucresindeki vesselness degeri doku kantilini asarsa elenir.",
        "variants": VARIANTS, "seeds": SEEDS, "tolerance": TOLERANCE,
        "results": results, "baseline_mean_val_auc": base,
        "best_variant": best_name, "delta": delta,
        "criteria": {"delta_ge_0.015": bool(delta >= TOLERANCE), "not_baseline": bool(best_name != "base")},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=42, decision=decision,
        metrics={f"mean_val_auc_{k}": v["mean_val_auc"] for k, v in results.items()},
        notes=f"En iyi varyant={best_name} (fark {delta:+.4f}); karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
