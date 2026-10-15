"""E2.7 — Persistence Image ağırlık fonksiyonu karşılaştırması.

Ön-kayıtlı hipotez:
    Adams ağırlığı `w(p) = p` dışında bir ağırlık fonksiyonu, aynı veri hattı ve
    mimariyle **val ROC-AUC'yi en az +0.015** artırır.

Tek değişken:
    Yalnızca PI hücre birikim ağırlığı w(p). Diyagramlar, ızgara aralıkları
    (`tda_ranges_v2.json`), split, mimari, eğitim protokolü **sabittir**.

Dayanak:
    Moon et al. (arXiv:2012.12102) üç ağırlık fonksiyonu karşılaştırıyor;
    literatür taraması (literature.md §5.4) bunu ucuz bir sağlamlık deneyi olarak
    öneriyor.

Kapsam (test bütçesi korunur):
    Yalnızca **train (dengeli) + validation** zonları işlenir; test setine
    DOKUNULMAZ. Karar validation ROC-AUC üzerinden verilir.

Kullanım:
    python pipelines/exp/E2_7_pi_weight_functions.py
    python pipelines/exp/E2_7_pi_weight_functions.py --reuse-features
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
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "E2.7"
SLUG = "pi_weight_functions"
SEED = 42
GRID = 20
SIGMA = 1.0
FEATURES_PATH = PROJECT_ROOT / "data/processed/features_weights/E2_7_zone_features.npy"
INDEX_PATH = PROJECT_ROOT / "data/processed/features_weights/E2_7_vector_index.npy"
ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
H0_RANGES = ((0.0, 248.0), (0.0, 248.0))
H1_RANGES = ((239.0, 250.0), (0.0, 7.0))

WEIGHT_NAMES = ["adams_p", "constant", "arctan", "sqrt", "square"]


def apply_weight(name: str, p: np.ndarray) -> np.ndarray:
    if name == "adams_p":
        return p
    if name == "constant":
        return np.ones_like(p)
    if name == "arctan":
        return np.arctan(p)
    if name == "sqrt":
        return np.sqrt(p)
    if name == "square":
        return p * p
    raise ValueError(f"Bilinmeyen agirlik: {name}")


def rasterize(diagram: np.ndarray, birth_range, persistence_range, weight: str) -> np.ndarray:
    """Motordan bagimsiz, agirlikli PI rasterlestirme (extractor ile ayni mantik)."""
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
    np.add.at(image, (rows, cols), apply_weight(weight, pers))
    return gaussian_filter(image, sigma=SIGMA, mode="constant", truncate=3.0)


_WORKER: Dict[str, Any] = {}


def _init_worker() -> None:
    _WORKER["extractor"] = ZoneTDAExtractor(TDASettings())


def _process(task: Tuple[int, str]) -> Tuple[int, np.ndarray]:
    vector_index, path_str = task
    extractor: ZoneTDAExtractor = _WORKER["extractor"]
    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(path_str))
        if image.ndim == 3:
            image = image[0]
        filtration = extractor.prepare_filtration(image)
        if filtration is None:
            return vector_index, np.zeros((len(WEIGHT_NAMES), 800), dtype=np.float32)
        h0, h1 = extractor.compute_diagrams(filtration)
        out = np.zeros((len(WEIGHT_NAMES), 800), dtype=np.float32)
        for wi, wname in enumerate(WEIGHT_NAMES):
            v = np.concatenate([
                rasterize(h0, *H0_RANGES, wname).ravel(),
                rasterize(h1, *H1_RANGES, wname).ravel(),
            ])
            out[wi] = v.astype(np.float32, copy=False)
        return vector_index, out
    except Exception:
        return vector_index, np.zeros((len(WEIGHT_NAMES), 800), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.7 PI agirlik fonksiyonu karsilastirmasi")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--reuse-features", action="store_true")
    return p.parse_args()


def train_variant(X_sub: np.ndarray, sub: pd.DataFrame, max_epochs: int, seed: int) -> Dict[str, Any]:
    """Tek bir agirlik varyanti icin 5-zon CNN egitir ve val hasta AUC'sini doner."""
    y_zone = sub["zone_label"].values.astype(np.float32)
    zone_models, zone_thresholds = {}, {}
    val_preds: List[Dict[str, Any]] = []

    for zone_id in range(1, 6):
        zm = (sub["zone_id"] == zone_id).values
        tr = zm & sub["is_train_balanced_sample"].values.astype(bool)
        va = zm & (sub["split"] == "val").values

        scaler = StandardScaler().fit(X_sub[tr])
        Xtr = scaler.transform(X_sub[tr])
        Xva = scaler.transform(X_sub[va])
        ytr = torch.tensor(y_zone[tr], dtype=torch.float32)

        torch.manual_seed(seed)
        model = SmallZoneCNN(in_channels=2, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(
            torch.tensor(vector_to_2channel_image(Xtr), dtype=torch.float32), ytr),
            batch_size=32, shuffle=True)

        Xva_t = torch.tensor(vector_to_2channel_image(Xva), dtype=torch.float32)
        best_auc, best_state = -1.0, None
        for _ in range(max_epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xva_t)).numpy()
            a = compute_binary_metrics(y_zone[va], p)["roc_auc"]
            if a > best_auc:
                best_auc, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).numpy()
        thr, _ = find_optimal_threshold(y_zone[va], pv, method="youden")
        zone_models[zone_id], zone_thresholds[zone_id] = model, thr
        for pid, p in zip(sub.loc[va, "patient_id"], pv):
            val_preds.append({"patient_id": pid, "p": float(p)})

    pdf = pd.DataFrame(val_preds).groupby("patient_id")["p"].max()
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    pv = pdf.reindex(meta.index)
    yv = meta["patient_label"].values
    return {"val_auc": float(compute_binary_metrics(yv, pv.values)["roc_auc"])}


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    splits = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)

    needed = splits["is_train_balanced_sample"].values.astype(bool) | (splits["split"] == "val").values
    sub = splits[needed].reset_index(drop=True)
    print("=" * 78)
    print("E2.7 — PERSISTENCE IMAGE AGIRLIK FONKSIYONU KARSILASTIRMASI")
    print("=" * 78)
    print(f"Kapsam: train(dengeli)+val = {len(sub)} zon (TEST SETINE DOKUNULMUYOR)")
    print(f"Agirliklar: {WEIGHT_NAMES}")

    if args.reuse_features and FEATURES_PATH.exists():
        feats = np.load(FEATURES_PATH)
        idx = np.load(INDEX_PATH)
        print(f"Kayitli ozellikler yuklendi: {feats.shape}")
    else:
        tasks = []
        for row in sub.itertuples(index=False):
            path = args.zones_dir / row.patient_id / ZONE_FILES[int(row.zone_id)]
            tasks.append((int(row.vector_index), str(path)))
        t0 = time.time()
        feats = np.zeros((len(sub), len(WEIGHT_NAMES), 800), dtype=np.float32)
        pos = {int(v): i for i, v in enumerate(sub["vector_index"].values)}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            done = 0
            for vi, arr in pool.map(_process, tasks, chunksize=8):
                feats[pos[vi]] = arr
                done += 1
                if done % 500 == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"  {done}/{len(tasks)} | {rate:.1f} zon/s | kalan ~{(len(tasks)-done)/max(rate,1e-9)/60:.1f} dk",
                          flush=True)
        print(f"  Cikarim tamamlandi: {len(tasks)} zon, {time.time()-t0:.0f} s")
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(FEATURES_PATH, feats)
        np.save(INDEX_PATH, sub["vector_index"].values)

    results = {}
    for wi, wname in enumerate(WEIGHT_NAMES):
        r = train_variant(feats[:, wi, :], sub, args.max_epochs, SEED)
        results[wname] = r
        print(f"  {wname:10s}: val hasta ROC-AUC = {r['val_auc']:.4f}", flush=True)

    base = results["adams_p"]["val_auc"]
    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best_auc = results[best_name]["val_auc"]
    delta = best_auc - base
    passed = best_name != "adams_p" and delta >= 0.015
    decision = "KEEP" if passed else "KILL"

    print("\n--- Ozet ---")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["val_auc"]):
        print(f"  {k:10s} {v['val_auc']:.4f}  (adams_p'ye fark {v['val_auc']-base:+.4f})")
    print(f"\n  En iyi: {best_name} ({best_auc:.4f}) | fark={delta:+.4f} | esik +0.015 -> {passed}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Adams w(p)=p disinda bir agirlik fonksiyonu val ROC-AUC'yi >= +0.015 artirir.",
        "scope": "train(dengeli)+val only; test setine dokunulmadi",
        "n_zones_processed": int(len(sub)),
        "weights": WEIGHT_NAMES,
        "results": results,
        "baseline_adams_p_val_auc": base,
        "best_weight": best_name, "best_val_auc": best_auc, "delta": delta,
        "criteria": {"delta_ge_0.015": bool(delta >= 0.015), "not_baseline": bool(best_name != "adams_p")},
        "decision": decision,
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=decision,
        metrics={f"val_auc_{k}": v["val_auc"] for k, v in results.items()},
        notes=f"En iyi agirlik={best_name} (val {best_auc:.4f}, adams_p'ye {delta:+.4f}); karar={decision}. Test seti kullanilmadi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()
