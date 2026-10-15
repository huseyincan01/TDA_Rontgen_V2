"""E2.8 — Persistence Image σ (Gaussian genişliği) ve ızgara çözünürlüğü duyarlılığı.

Ön-kayıtlı hipotez (SAĞLAMLIK):
    PI sonuçları, Gaussian genişliği σ ve ızgara çözünürlüğü seçimine karşı
    **sağlamdır**: her varyantın ortalama val ROC-AUC'si referans konfigürasyondan
    (σ=1.0, ızgara=20) **en fazla 0.03** sapar.

Toleransın gerekçesi (ön-kayıtlı):
    E4.6'da CNN seed varyansı **±0.02–0.03** ölçüldü. Tek-seed'lik 0.015 toleransı
    gürültünün altında kalırdı; bu yüzden bu deney **3 seed ortalaması** kullanır ve
    tolerans **0.03** olarak sabitlenmiştir (gürültüden büyük).

Tek değişken:
    İki kol ayrı ayrı değerlendirilir —
      A) σ ∈ {0.5, 1.0, 1.5, 2.0}, ızgara=20 sabit
      B) ızgara ∈ {15, 20, 25}, σ=1.0 sabit
    Diyagramlar, doğum/kalıcılık aralıkları, split, mimari ve eğitim protokolü sabit.
    Diyagramlar **bir kez** hesaplanır, her konfigürasyon için yeniden vektörleştirilir.

Kapsam (test bütçesi korunur):
    Yalnızca train(dengeli)+val; **test setine dokunulmaz**.

Kullanım:
    python pipelines/exp/E2_8_sigma_grid_sensitivity.py
    python pipelines/exp/E2_8_sigma_grid_sensitivity.py --reuse-features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN  # noqa: E402

EXPERIMENT_ID = "E2.8"
SLUG = "sigma_grid_sensitivity"
SEEDS = [42, 43, 44]
TOLERANCE = 0.03
REFERENCE = "s1.0_g20"
FEATURE_DIR = PROJECT_ROOT / "data/processed/features_sigma/E2_8"
INDEX_PATH = FEATURE_DIR / "vector_index.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
# kol A: sigma sweep (grid=20) · kol B: grid sweep (sigma=1.0)
CONFIGS: Dict[str, Tuple[int, float]] = {
    "s0.5_g20": (20, 0.5), "s1.0_g20": (20, 1.0), "s1.5_g20": (20, 1.5), "s2.0_g20": (20, 2.0),
    "s1.0_g15": (15, 1.0), "s1.0_g25": (25, 1.0),
}
ARM_A = ["s0.5_g20", "s1.0_g20", "s1.5_g20", "s2.0_g20"]
ARM_B = ["s1.0_g15", "s1.0_g20", "s1.0_g25"]
H0_RANGES = ((0.0, 248.0), (0.0, 248.0))
H1_RANGES = ((239.0, 250.0), (0.0, 7.0))


def rasterize(diagram: np.ndarray, birth_range, persistence_range, grid: int, sigma: float) -> np.ndarray:
    image = np.zeros((grid, grid), dtype=np.float64)
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
    cols = np.minimum(((births - birth_range[0]) / bw * grid).astype(int), grid - 1)
    rows = np.minimum(((pers - persistence_range[0]) / pw * grid).astype(int), grid - 1)
    np.add.at(image, (rows, cols), pers)
    return gaussian_filter(image, sigma=sigma, mode="constant", truncate=3.0)


_WORKER: Dict[str, Any] = {}


def _init_worker() -> None:
    _WORKER["extractor"] = ZoneTDAExtractor(TDASettings())


def _process(task: Tuple[int, str]) -> Tuple[int, Dict[str, np.ndarray]]:
    vector_index, path_str = task
    ex: ZoneTDAExtractor = _WORKER["extractor"]
    empty = {name: np.zeros(2 * g * g, dtype=np.float32) for name, (g, _) in CONFIGS.items()}
    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if image.ndim == 3:
            image = image[0]
        filtration = ex.prepare_filtration(image)
        if filtration is None:
            return vector_index, empty
        h0, h1 = ex.compute_diagrams(filtration)
        out: Dict[str, np.ndarray] = {}
        for name, (grid, sigma) in CONFIGS.items():
            v = np.concatenate([rasterize(h0, *H0_RANGES, grid, sigma).ravel(),
                                rasterize(h1, *H1_RANGES, grid, sigma).ravel()])
            out[name] = v.astype(np.float32, copy=False)
        return vector_index, out
    except Exception:
        return vector_index, empty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.8 sigma/izgara duyarlilik")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--reuse-features", action="store_true")
    return p.parse_args()


def to_tensor(X: np.ndarray, grid: int) -> torch.Tensor:
    n = X.shape[0]
    h0 = X[:, : grid * grid].reshape(n, grid, grid)
    h1 = X[:, grid * grid:].reshape(n, grid, grid)
    return torch.tensor(np.stack([h0, h1], axis=1), dtype=torch.float32)


def train_val_auc(X: np.ndarray, zs: pd.DataFrame, grid: int, max_epochs: int, seed: int) -> float:
    y = zs["zone_label"].values.astype(np.float32)
    preds: List[Dict[str, Any]] = []
    for zone_id in range(1, 6):
        zm = (zs["zone_id"] == zone_id).values
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
        loader = DataLoader(TensorDataset(to_tensor(Xtr, grid), torch.tensor(y[tr], dtype=torch.float32)),
                            batch_size=32, shuffle=True)
        Xvm_t = to_tensor(Xvm, grid)
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
            pv = torch.sigmoid(model(to_tensor(Xva, grid))).numpy()
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
    print("E2.8 — PI SIGMA / IZGARA COZUNURLUK DUYARLILIK TARAMASI")
    print("=" * 78)
    print(f"Kapsam: train(dengeli)+val = {len(sub)} zon | TEST SETINE DOKUNULMUYOR")
    print(f"Konfigurasyonlar: {list(CONFIGS)}")
    print(f"Seed'ler: {SEEDS} | tolerans: {TOLERANCE} (E4.6'da olculen seed varyansi nedeniyle)")

    if args.reuse_features and INDEX_PATH.exists():
        idx = np.load(INDEX_PATH)
        feats = {name: np.load(FEATURE_DIR / f"{name}.npy") for name in CONFIGS}
        print("Kayitli ozellikler yuklendi.")
    else:
        tasks = [(int(r.vector_index), str(args.zones_dir / r.patient_id / ZONE_FILES[int(r.zone_id)]))
                 for r in sub.itertuples(index=False)]
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        feats = {name: np.zeros((len(sub), 2 * g * g), dtype=np.float32) for name, (g, _) in CONFIGS.items()}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for vi, d in pool.map(_process, tasks, chunksize=8):
                i = pos[vi]
                for name, v in d.items():
                    feats[name][i] = v
                done += 1
                if done % 500 == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"  {done}/{len(tasks)} | {rate:.1f} zon/s | kalan ~{(len(tasks)-done)/max(rate,1e-9)/60:.1f} dk",
                          flush=True)
        print(f"  Cikarim tamamlandi: {len(tasks)} zon, {time.time()-t0:.0f} s")
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_PATH, sub["vector_index"].values)
        for name, arr in feats.items():
            np.save(FEATURE_DIR / f"{name}.npy", arr)

    results: Dict[str, Dict[str, Any]] = {}
    for name, (grid, sigma) in CONFIGS.items():
        aucs = []
        for seed in SEEDS:
            a = train_val_auc(feats[name], sub, grid, args.max_epochs, seed)
            aucs.append(a)
        results[name] = {"grid": grid, "sigma": sigma, "seed_aucs": aucs,
                         "mean_val_auc": float(np.mean(aucs)), "std": float(np.std(aucs))}
        print(f"  {name:10s} (grid={grid:2d}, sigma={sigma:.1f}): "
              f"ortalama val AUC = {np.mean(aucs):.4f} (std {np.std(aucs):.4f})", flush=True)

    ref = results[REFERENCE]["mean_val_auc"]
    print(f"\n--- Referans ({REFERENCE}) = {ref:.4f} ---")
    arms = {"A_sigma": ARM_A, "B_grid": ARM_B}
    decisions = {}
    for arm, members in arms.items():
        print(f"  Kol {arm}:")
        deltas = {}
        for m in members:
            d = results[m]["mean_val_auc"] - ref
            deltas[m] = d
            print(f"    {m:10s}: {results[m]['mean_val_auc']:.4f}  (fark {d:+.4f})")
        max_abs = max(abs(v) for k, v in deltas.items() if k != REFERENCE) if len(members) > 1 else 0.0
        stable = max_abs < TOLERANCE
        decisions[arm] = {"deltas": deltas, "max_abs_delta": max_abs, "stable": bool(stable)}
        print(f"    -> en buyuk |fark| = {max_abs:.4f} | tolerans {TOLERANCE} -> "
              f"{'SAGLAM' if stable else 'DUYARLI'}")

    overall_stable = all(v["stable"] for v in decisions.values())
    decision = "KEEP" if overall_stable else "KILL"
    print(f"\n  KARAR: {decision} (tum kollar saglam: {overall_stable})")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"PI sonuclari sigma/izgara secimine karsi saglamdir (|fark| < {TOLERANCE}).",
        "tolerance_rationale": "E4.6'da olculen CNN seed varyansi (+/-0.02-0.03) nedeniyle tolerans 0.03; 3 seed ortalamasi.",
        "scope": "train(dengeli)+val only; test setine dokunulmadi",
        "seeds": SEEDS, "tolerance": TOLERANCE, "reference": REFERENCE,
        "configs": {k: {"grid": v[0], "sigma": v[1]} for k, v in CONFIGS.items()},
        "results": results, "arms": decisions, "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=42, decision=decision,
        metrics={f"mean_val_auc_{k}": v["mean_val_auc"] for k, v in results.items()},
        notes=f"Sigma/izgara duyarliligi: en buyuk |fark|={max(v['max_abs_delta'] for v in decisions.values()):.4f}, "
              f"tolerans={TOLERANCE}; karar={decision}. Test kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
