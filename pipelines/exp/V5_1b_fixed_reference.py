"""V5.1b — Sabit (tüm-akciğer) referans kontrolü: uyarlanabilir pencere artefaktı mı?

NEDEN
    V5.1'deki `a_density` kolu, p1–p99'u **her zon kırpıntısının kendi** dokusundan
    hesaplıyordu ("uyarlanabilir pencere"). Bir nodül, kırpıntının p99'unu hafifçe
    yükseltip normalizasyonu etkileyebilir (etiket sızıntısı değildir, ama istatistiksel
    bir çiftleşme yaratabilir). V5.2'deki 0,9856'lık füzyon sıçramasının bu artefakttan
    gelmediğini doğrulamak için kontrol kolu: p1–p99 **aynı hastanın tüm akciğer dokusu**
    üzerinden (5 zonun birleşimi) hesaplanır ve o tek eşleme o hastanın tüm zonlarına uygulanır.

PROTOKOL (ön-kayıt)
    * Kol `c_lungref`: `scaled = (v − p1_akciğer)/(p99_akciğer − p1_akciğer) × 255`, doku maskesinde 255 − scaled.
    * Karşılaştırma: (i) V5.1 üçlü kapısı (val label AUC · A2 kaynak AUC · LOO), (ii) V5.2'nin
      5-zon CNN + SVM ensemble + ince ayarlı ResNet18 füzyonu.
    * KARAR KURALI: `c_lungref` füzyon val AUC'si, `a_density`'nin (0,9856) ±0,005'i içindeyse
      **uyarlanabilir pencere artefaktı YOK** (KEEP doğrulanır). Eğer v2 seviyesine (~0,973)
      düşerse, 0,9856 pencere artefaktıydı ⇒ KEEP, `c_lungref`'e devredilir.
    * Test bütçesi 6/6 kapalı ⇒ yalnızca validation.

Kullanım:
    python pipelines/exp/V5_1b_fixed_reference.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EXPERIMENT_ID = "V5.1b"
SLUG = "fixed_reference"
SEED = 42
GRID = 20
SIGMA = 1.0
MAX_WORKERS = 10
SHARD = 2048
LOWER_PCT, UPPER_PCT = 1.0, 99.0

ZONE_FILES = {
    1: "zone_1_sag_ust.mha", 2: "zone_2_sag_orta.mha", 3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha", 5: "zone_5_sol_alt.mha",
}
FEATURES_META = PROJECT_ROOT / "data/processed/features_v2/features_metadata.csv"
PATIENTS_DIR = PROJECT_ROOT / "data/processed/zones/patients"
TDA_RANGES_V2 = PROJECT_ROOT / "data/processed/tda_config/tda_ranges_v2.json"
CACHE_DIR = PROJECT_ROOT / "data/processed/features_v5_1"
REPORT_JSON = PROJECT_ROOT / "data/external_vindr/v5_1b_fixed_reference_report.json"
REF_A_DENSITY_FUSION = 0.9856
REF_V2_FUSION = 0.9729

_W: dict = {}


def _v51():
    if _W.get("v51") is None:
        spec = importlib.util.spec_from_file_location(
            "v51", PROJECT_ROOT / "pipelines/exp/V5_1_physical_canonicalization.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _W["v51"] = m
    return _W["v51"]


def _v52():
    if _W.get("v52") is None:
        spec = importlib.util.spec_from_file_location(
            "v52", PROJECT_ROOT / "pipelines/exp/V5_2_corrected_tda_ensemble_fusion.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _W["v52"] = m
    return _W["v52"]


# ------------------------------------------------------------------ tüm-akciğer yüzdeleri
def _patient_p1p99(pid: str):
    vals = []
    for z in range(1, 6):
        p = PATIENTS_DIR / pid / ZONE_FILES[z]
        if not p.exists():
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        if arr.ndim == 3:
            arr = arr[0]
        t = arr[arr > 0]
        if t.size:
            vals.append(t.astype(np.float64))
    if not vals:
        return float("nan"), float("nan")
    allv = np.concatenate(vals)
    p1, p99 = np.percentile(allv, [LOWER_PCT, UPPER_PCT])
    return float(p1), float(p99)


def build_lungref_filtration(image: np.ndarray, p1: float, p99: float):
    if image.ndim == 3:
        image = image[0]
    image = np.asarray(image, dtype=np.float32)
    mask = np.isfinite(image) & (image > 0)
    if not np.any(mask):
        return None
    y, x = np.where(mask)
    tissue = image[y.min():y.max() + 1, x.min():x.max() + 1]
    tm = mask[y.min():y.max() + 1, x.min():x.max() + 1]
    if p99 <= p1:
        p99 = p1 + 1.0
    vals = tissue[tm]
    scaled = np.clip(np.rint((vals - p1) * 255.0 / (p99 - p1)), 0, 255)
    fil = np.zeros(tissue.shape, dtype=np.uint8)
    fil[tm] = 255 - scaled.astype(np.uint8)
    return fil


def _worker_diagram(task):
    path, p1, p99 = task
    v51 = _v51()
    ex = v51.ZoneTDAExtractor(v51.TDASettings(grid_size=GRID, pi_sigma_pixels=SIGMA))
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    fil = build_lungref_filtration(arr, p1, p99)
    if fil is None:
        return None
    h0, h1 = ex.compute_diagrams(fil)
    return h0, h1


def _worker_feature(task):
    path, p1, p99, ranges = task
    v51 = _v51()
    ex = v51.ZoneTDAExtractor(v51._settings(ranges))
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    fil = build_lungref_filtration(arr, p1, p99)
    if fil is None:
        return np.zeros(2 * GRID * GRID, dtype=np.float32)
    h0, h1 = ex.compute_diagrams(fil)
    return ex.vectorize(h0, h1)


# ------------------------------------------------------------------ aralıklar + çıkarım
def derive_ranges(lungref: dict) -> dict:
    v51 = _v51()
    ref = json.loads(TDA_RANGES_V2.read_text(encoding="utf-8"))
    tasks = []
    for pid in ref["sample_info"]["patient_ids"]:
        p1, p99 = lungref.get(pid, (float("nan"), float("nan")))
        for z in range(1, 6):
            tasks.append((str(PATIENTS_DIR / pid / ZONE_FILES[z]), p1, p99))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        res = list(pool.map(_worker_diagram, tasks, chunksize=8))
    h0b, h0p, h1b, h1p = [], [], [], []
    for r in res:
        if r is None:
            continue
        h0, h1 = r
        if len(h0):
            h0b.extend(h0[:, 0].tolist()); h0p.extend((h0[:, 1] - h0[:, 0]).tolist())
        if len(h1):
            h1b.extend(h1[:, 0].tolist()); h1p.extend((h1[:, 1] - h1[:, 0]).tolist())

    def b(arr, is_birth):
        a = np.asarray(arr, float)
        if not a.size:
            return [0.0, 10.0]
        lo = float(np.floor(np.percentile(a, LOWER_PCT))) if is_birth else 0.0
        hi = float(np.ceil(np.percentile(a, UPPER_PCT)))
        if hi <= lo:
            hi = lo + 10.0
        return [lo, hi]

    ranges = {"h0_birth": b(h0b, True), "h0_pers": b(h0p, False),
              "h1_birth": b(h1b, True), "h1_pers": b(h1p, False),
              "n_h0": len(h0b), "n_h1": len(h1b)}
    print(f"  [ranges] H0 b={ranges['h0_birth']} p={ranges['h0_pers']} | "
          f"H1 b={ranges['h1_birth']} p={ranges['h1_pers']} | H1 nokta {len(h1b):,}")
    return ranges


def extract(meta: pd.DataFrame, lungref: dict) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x_path = CACHE_DIR / "X_c_lungref.npy"
    r_path = CACHE_DIR / "ranges_c_lungref.json"
    parts_dir = CACHE_DIR / "parts_c_lungref"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if x_path.exists() and r_path.exists():
        print(f"  [cache] {x_path.name} yüklendi.")
        return np.load(x_path)
    ranges = json.loads(r_path.read_text(encoding="utf-8")) if r_path.exists() else derive_ranges(lungref)
    if not r_path.exists():
        r_path.write_text(json.dumps(ranges, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = []
    for row in meta.itertuples(index=False):
        p1, p99 = lungref.get(row.patient_id, (float("nan"), float("nan")))
        tasks.append((str(PATIENTS_DIR / row.patient_id / ZONE_FILES[int(row.zone_id)]),
                      p1, p99, ranges))
    n_shards = (len(tasks) + SHARD - 1) // SHARD
    print(f"  [extract] {len(tasks)} zon | {n_shards} parça | {MAX_WORKERS} işçi")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for k in range(n_shards):
            chunk = tasks[k * SHARD:(k + 1) * SHARD]
            part = parts_dir / f"{k:04d}.npy"
            if part.exists():
                try:
                    if np.load(part).shape == (len(chunk), 2 * GRID * GRID):
                        done += len(chunk)
                        continue
                except Exception:
                    pass
                part.unlink()
            arr = np.stack(list(pool.map(_worker_feature, chunk, chunksize=16)))
            np.save(part, arr)
            done += len(chunk)
            rate = done / max(1e-9, time.time() - t0)
            print(f"    parça {k + 1}/{n_shards} | {done}/{len(tasks)} zon | "
                  f"{rate * 60:.0f} zon/dk | kalan ~{(len(tasks) - done) / max(rate, 1e-9) / 60:.1f} dk",
                  flush=True)
    X = np.concatenate([np.load(parts_dir / f"{k:04d}.npy") for k in range(n_shards)])
    np.save(x_path, X)
    print(f"  [extract] tamamlandı {X.shape} ({time.time() - t0:.0f} s)")
    return X


# ------------------------------------------------------------------ değerlendirme
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 82)
    print("V5.1b — SABİT (TÜM-AKCİĞER) REFERANS KONTROLÜ: uyarlanabilir pencere artefaktı mı?")
    print("=" * 82)

    import argparse

    parser = argparse.ArgumentParser(description="V5.1b sabit referans kontrolü")
    parser.add_argument("--limit", type=int, default=0, help="duman testi: ilk N hasta")
    args = parser.parse_args()

    v51, v52 = _v51(), _v52()
    meta = pd.read_csv(FEATURES_META)
    if args.limit:
        sel = meta["patient_id"].unique()[: args.limit]
        meta = meta[meta["patient_id"].isin(sel)].reset_index(drop=True)
    pids = sorted(meta["patient_id"].unique())

    # tüm-akciğer yüzdeleri (paralel + önbellek)
    ref_path = CACHE_DIR / "lung_p1p99.json"
    if ref_path.exists():
        lungref = json.loads(ref_path.read_text(encoding="utf-8"))
        print(f"Tüm-akciğer yüzdeleri önbellekten ({len(lungref)} hasta).")
    else:
        print(f"Tüm-akciğer p1/p99 hesaplanıyor ({len(pids)} hasta)...")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            vals = list(pool.map(_patient_p1p99, pids, chunksize=16))
        lungref = {pid: [v[0], v[1]] for pid, v in zip(pids, vals)}
        ref_path.write_text(json.dumps(lungref), encoding="utf-8")
        print(f"  tamamlandı ({time.time() - t0:.0f} s)")

    X = extract(meta, lungref)

    # --- (i) V5.1 üçlü kapısı (SVM protokolü) ---
    print("\n[üçlü kapı — SVM]")
    gates = {
        "val_label": v51.val_label_auc(X, tta=False),
        "source": v51.source_auc(X, tta=False),
        "loo": v51.loo_auc(X, tta=False),
        "val_label_tta": v51.val_label_auc(X, tta=True),
        "loo_tta": v51.loo_auc(X, tta=True),
    }

    # --- (ii) CNN ensemble + füzyon (V5.2 protokolü) ---
    print("\n[CNN ensemble + füzyon]")
    splits = pd.read_csv(v51.ZONE_SPLITS).sort_values("vector_index").reset_index(drop=True)
    need = (splits["is_train_balanced_sample"].values.astype(bool)
            | (splits["split"] == "val").values | (splits["split"] == "test").values)
    sub = splits[need].reset_index(drop=True)
    img_val_df = v52.resnet_patient_probs(sub, (sub["split"] == "val").values)
    img_val = img_val_df.set_index("patient_id")["p"]
    ensemble = v52.evaluate_arm("c_lungref", X, splits, img_val, save_bundles=False)

    report = {
        "experiment": EXPERIMENT_ID,
        "question": "V5.2'nin 0.9856 füzyonu, kırpıntı-başına uyarlanabilir pencere artefaktı mı?",
        "arm": "c_lungref (tüm-akciğer tek p1/p99, o hastanın tüm zonlarına uygulanır)",
        "gates_svm": gates,
        "ensemble_fusion": ensemble,
        "references": {"a_density_fusion_val": REF_A_DENSITY_FUSION, "v2_fusion_val": REF_V2_FUSION},
        "comparison": {
            "fusion_delta_vs_a_density": ensemble["val_fusion_best_a"] - REF_A_DENSITY_FUSION,
            "fusion_delta_vs_v2": ensemble["val_fusion_best_a"] - REF_V2_FUSION,
        },
    }
    tol = 0.005
    d_a = ensemble["val_fusion_best_a"] - REF_A_DENSITY_FUSION
    decision = ("KEEP (pencere artefaktı YOK)" if abs(d_a) <= tol
                else ("ARTEFAKT VAR → KEEP c_lungref'e devredilir" if d_a < -tol
                      else "c_lungref daha da iyi (açıklama gerekir)"))
    report["decision"] = decision

    print("\n" + "=" * 82)
    print(f"  val label AUC : {gates['val_label']['patient_auc']:.4f}   "
          f"(V5.1 a_density: 0.9211)")
    print(f"  A2 kaynak AUC : {gates['source']['macro_ovr_auc']:.4f}   (a_density: 0.9554)")
    print(f"  LOO           : {gates['loo']['mean_auc']:.4f}   (a_density: 0.7861)")
    print(f"  TDA ensemble  : {ensemble['val_tda_ensemble']:.4f}   (a_density: 0.9522)")
    print(f"  FÜZYON        : {ensemble['val_fusion_best_a']:.4f} (a={ensemble['fusion_best_a']:.2f})   "
          f"(a_density: {REF_A_DENSITY_FUSION}, v2: {REF_V2_FUSION})")
    print(f"  KARAR         : {decision}")
    print("=" * 82)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        from node21_tda.experiment import new_run_dir, write_manifest

        run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
        write_manifest(run_dir, EXPERIMENT_ID,
                       config={"arm": "c_lungref", "p1_p99": "tüm-akciğer birleşimi",
                               "tol": tol},
                       inputs=[FEATURES_META, CACHE_DIR / "X_c_lungref.npy"],
                       seed=SEED,
                       metrics={"val_label_auc": gates["val_label"]["patient_auc"],
                                "source_auc": gates["source"]["macro_ovr_auc"],
                                "loo_mean_auc": gates["loo"]["mean_auc"],
                                "fusion_val_auc": ensemble["val_fusion_best_a"]},
                       decision=decision, notes="Sabit (tüm-akciğer) referans kontrolü.")
        print(f"Manifest: {run_dir / 'manifest.json'}")
    except Exception as exc:
        print(f"Manifest yazılamadı: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
