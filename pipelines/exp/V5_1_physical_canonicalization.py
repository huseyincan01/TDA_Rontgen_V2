"""V5.1 — Fiziksel kanonikleştirme: yoğunluk (p1–p99) + ölçek (akciğer yüksekliği) + etiketsiz TTA.

AMAÇ
    Alan kaymasının kök nedeni iki fizik büyüklüğüdür (bkz. A0/A1 + A2 teşhisi):
      (1) mutlak yoğunluk/bit-derinliği  (16-bit NODE21 ↔ 8-bit VinDr; sabit `intensity_max`)
      (2) piksel ölçeği (spacing 0.147–0.615 mm/px)
    Bu deney, **5-zon mimarisi korunarak** bu iki büyüklüğü kanonikleştirmenin
    (i) iç val başarımını, (ii) cihaz imzasını (A2), (iii) kaynak-dışı genellemeyi (LOO/E5.3b)
    nasıl etkilediğini ölçer.

KURALLAR (kullanıcı kararları)
    * 5-zon mimarisi **korunur**; tek kanonik akciğer görüntüsüne geçilmez.
    * Üç basamaklı tablo:  (a) yoğunluk · (b) yoğunluk+ölçek · (c) (b)+TTA
    * **TTA ayrı satır**: etiketsiz hedef kümesi istatistiğiyle yeniden standardizasyon
      (BN running-stats recalibrasyonunun SVM karşılığı). Eğitim girdileri/etiketleri kilitli.
    * Tek değişken: her basamak bir öncekine TEK müdahale ekler.
    * Değerlendirme kapısı üçlüsü: **Val Label AUC · A2 Kaynak AUC · LOO (E5.3b, eşit boyut)**.
    * Test hastaları bu deneyde **karar için kullanılmaz**; LOO katları E5.3 protokolünü birebir izler.

ÖN-KAYITLI KARAR KURALI (deney başlamadan):
    (b) KEEP sayılır ancak ve ancak:
        val_label_auc >= baseline_val - 0.005
        AND loo_mean_auc >= baseline_loo + 0.015
        AND source_auc <= baseline_source - 0.03
    Aksi hâlde KILL. (TTA satırı (c) ayrıca raporlanır; karar (b) üzerinden verilir.)

Kullanım:
    python pipelines/exp/V5_1_physical_canonicalization.py [--sanity 200] [--arms a_density b_density_scale]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

from node21_tda.data.zone_splits_v2 import build_zone_splits_v2  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402

EXPERIMENT_ID = "V5.1"
SLUG = "physical_canonicalization"
SEED = 42
GRID = 20
SIGMA = 1.0
MAX_WORKERS = 10
LOWER_PCT, UPPER_PCT = 1.0, 99.0
SHARD = 2048              # her parça ~2-3 dk; kesintide en fazla bu kadar iş kaybolur
LOO_CAP = 1034            # E5.3b eşit-boyut
LOO_SOURCES = ["chestxray14", "padchest", "jsrt"]
SVM_C, SVM_GAMMA = 10.0, "scale"   # tek değişken için SABİT (arm başına tuning yapılmaz)

ZONE_FILES = {
    1: "zone_1_sag_ust.mha",
    2: "zone_2_sag_orta.mha",
    3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha",
    5: "zone_5_sol_alt.mha",
}

FEATURES_META = PROJECT_ROOT / "data/processed/features_v2/features_metadata.csv"
FROZEN_FEATURES = PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy"
ZONE_SPLITS = PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv"
PATIENT_SPLITS = PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv"
INVENTORY = PROJECT_ROOT / "data/processed/metadata_inventory_v2.csv"
TDA_RANGES_V2 = PROJECT_ROOT / "data/processed/tda_config/tda_ranges_v2.json"
PATIENTS_DIR = PROJECT_ROOT / "data/processed/zones/patients"
CACHE_DIR = PROJECT_ROOT / "data/processed/features_v5_1"
REPORT_JSON = PROJECT_ROOT / "data/external_vindr/v5_1_physical_canonicalization_report.json"

# Yoğunluk kanonikleştirmesinden etkilenen filtre: tepe (parlak) → 0, doku → 255 (v2 ile aynı yön)
_E53 = {"mod": None}


# ---------------------------------------------------------------------------------------
# Filtreleme (filtration) — tek değişkenin kalbi
# ---------------------------------------------------------------------------------------
def _resize_pair(crop: np.ndarray, mask: np.ndarray, scale: float):
    from skimage.transform import resize as sk_resize

    h, w = crop.shape
    new_h, new_w = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
    if new_h * new_w > 6_000_000:  # bellek sigortası
        f = np.sqrt(6_000_000 / (new_h * new_w))
        new_h, new_w = max(16, int(new_h * f)), max(16, int(new_w * f))
    c = sk_resize(crop.astype(np.float32), (new_h, new_w), order=1,
                  preserve_range=True, anti_aliasing=True)
    m = sk_resize(mask.astype(np.float32), (new_h, new_w), order=0,
                  preserve_range=True, anti_aliasing=False) > 0.5
    return np.where(m, c, 0.0), m


def build_filtration(image: np.ndarray, arm: str, scale: float = 1.0) -> np.ndarray | None:
    """Zon kesitini, kol tanımına göre filtrasyona çevirir.

    * baseline      : v2 projesi — sabit `intensity_max=65535` (mutlak yoğunluk bağımlı)
    * a_density     : doku p1–p99 → [0,255] kanonikleştirme (mutlak yoğunluk bağımsız)
    * b_density_scale: a + akciğer yüksekliğine göre ölçek normalizasyonu (piksel ölçeği bağımsız)
    """
    if image.ndim == 3:
        image = image[0]
    image = np.asarray(image, dtype=np.float32)
    mask = np.isfinite(image) & (image > 0)
    if not np.any(mask):
        return None

    if arm == "b_density_scale" and np.isfinite(scale) and abs(scale - 1.0) > 1e-3:
        y_idx, x_idx = np.where(mask)
        y0, y1, x0, x1 = y_idx.min(), y_idx.max() + 1, x_idx.min(), x_idx.max() + 1
        crop = image[y0:y1, x0:x1]
        cmask = mask[y0:y1, x0:x1]
        image_r, mask = _resize_pair(crop, cmask, float(np.clip(scale, 0.4, 2.5)))
    else:
        image_r = image

    y_idx, x_idx = np.where(mask)
    y0, y1, x0, x1 = y_idx.min(), y_idx.max() + 1, x_idx.min(), x_idx.max() + 1
    tissue = image_r[y0:y1, x0:x1]
    tmask = mask[y0:y1, x0:x1]
    vals = tissue[tmask]

    if arm == "baseline":
        scaled = np.clip(np.rint(vals * 255.0 / 65535.0), 0, 255)
    else:
        p1, p99 = np.percentile(vals, [LOWER_PCT, UPPER_PCT])
        if p99 <= p1:
            p99 = p1 + 1.0
        scaled = np.clip(np.rint((vals - p1) * 255.0 / (p99 - p1)), 0, 255)

    filtration = np.zeros(tissue.shape, dtype=np.uint8)
    filtration[tmask] = (255 - scaled.astype(np.uint8))
    return filtration


def _read_zone(path: Path) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return arr[0] if arr.ndim == 3 else arr


def _settings(ranges: dict) -> TDASettings:
    return TDASettings(
        grid_size=GRID,
        intensity_max=65535.0,
        h0_birth_range=tuple(ranges["h0_birth"]),
        h0_persistence_range=tuple(ranges["h0_pers"]),
        h1_birth_range=tuple(ranges["h1_birth"]),
        h1_persistence_range=tuple(ranges["h1_pers"]),
        pi_sigma_pixels=SIGMA,
    )


# --- İşçiler (Windows: modül düzeyi) ---
def _diagram_worker(task):
    path, arm, scale = task
    try:
        ex = ZoneTDAExtractor(TDASettings(grid_size=GRID, pi_sigma_pixels=SIGMA))
        fil = build_filtration(_read_zone(Path(path)), arm, scale)
        if fil is None:
            return None
        h0, h1 = ex.compute_diagrams(fil)
        return h0, h1
    except Exception:
        return None


def _feature_worker(task):
    path, arm, scale, ranges = task
    try:
        ex = ZoneTDAExtractor(_settings(ranges))
        fil = build_filtration(_read_zone(Path(path)), arm, scale)
        if fil is None:
            return np.zeros(2 * GRID * GRID, dtype=np.float32)
        h0, h1 = ex.compute_diagrams(fil)
        return ex.vectorize(h0, h1)
    except Exception:
        return np.zeros(2 * GRID * GRID, dtype=np.float32)


# ---------------------------------------------------------------------------------------
# Akciğer yüksekliği (ölçek normalizasyonu)
# ---------------------------------------------------------------------------------------
def _lung_heights(patient_ids) -> dict:
    out = {}
    for pid in patient_ids:
        p = PATIENTS_DIR / str(pid) / "patient_info.json"
        if not p.exists():
            continue
        zones = json.loads(p.read_text(encoding="utf-8")).get("zones", {})
        try:
            right = zones["3"]["crop_bbox"]["max_y"] - zones["1"]["crop_bbox"]["min_y"]
            left = zones["5"]["crop_bbox"]["max_y"] - zones["4"]["crop_bbox"]["min_y"]
            out[str(pid)] = float(max(right, left))
        except (KeyError, TypeError):
            continue
    return out


def target_lung_height() -> float:
    splits = pd.read_csv(PATIENT_SPLITS)
    train_ids = sorted(splits.loc[splits["split"] == "train", "patient_id"].astype(str))
    rng = np.random.RandomState(SEED)
    sample = rng.choice(train_ids, size=min(400, len(train_ids)), replace=False)
    h = _lung_heights(sample)
    return float(np.median(list(h.values())))


# ---------------------------------------------------------------------------------------
# Izgara aralıklarının train'den türetilmesi (v2 protokolü, kol bazında)
# ---------------------------------------------------------------------------------------
def derive_ranges(arm: str, scales: dict) -> dict:
    ref = json.loads(TDA_RANGES_V2.read_text(encoding="utf-8"))
    sample_ids = ref["sample_info"]["patient_ids"]
    tasks = []
    for pid in sample_ids:
        for z in range(1, 6):
            tasks.append((str(PATIENTS_DIR / pid / ZONE_FILES[z]), arm,
                          scales.get(str(pid), 1.0)))
    print(f"    [ranges] {len(tasks)} örnek zon üzerinden aralıklar türetiliyor...", flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        res = list(pool.map(_diagram_worker, tasks, chunksize=8))

    h0b, h0p, h1b, h1p = [], [], [], []
    for r in res:
        if r is None:
            continue
        h0, h1 = r
        if len(h0):
            h0b.extend(h0[:, 0].tolist())
            h0p.extend((h0[:, 1] - h0[:, 0]).tolist())
        if len(h1):
            h1b.extend(h1[:, 0].tolist())
            h1p.extend((h1[:, 1] - h1[:, 0]).tolist())

    def bounds(arr, low):
        arr = np.asarray(arr, dtype=np.float64)
        vol = arr[(arr > 0) if low == 0 else np.isfinite(arr)] if low == 0 else arr
        hi = float(np.percentile(arr, UPPER_PCT)) if arr.size else 10.0
        lo = float(np.floor(np.percentile(arr, LOWER_PCT))) if (low == "pct" and arr.size) else 0.0
        if hi <= lo:
            hi = lo + 10.0
        return [lo, float(np.ceil(hi))]

    ranges = {
        "h0_birth": bounds(h0b, "pct"),
        "h0_pers": bounds(h0p, 0.0),
        "h1_birth": bounds(h1b, "pct"),
        "h1_pers": bounds(h1p, 0.0),
        "n_h0_points": len(h0b), "n_h1_points": len(h1b),
        "source": f"train-only 50-hasta örneklemi (tda_ranges_v2.json listesi), {arm}",
    }
    print(f"    [ranges] H0 b={ranges['h0_birth']} p={ranges['h0_pers']} | "
          f"H1 b={ranges['h1_birth']} p={ranges['h1_pers']}", flush=True)
    return ranges


# ---------------------------------------------------------------------------------------
# Özellik çıkarımı (önbellekli)
# ---------------------------------------------------------------------------------------
def extract_arm(arm: str, meta: pd.DataFrame, scales: dict) -> tuple[np.ndarray, dict]:
    """Kolun özellik matrisini üretir — **parçalı (shard) + devam edebilir**.

    Her shard ayrı `.npy` olarak yazılır; koşu kesilirse (kapatma/çökme) mevcut
    shard'lar yeniden hesaplanmaz, en fazla tek shard (~2-3 dk) kaybedilir.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x_path = CACHE_DIR / f"X_{arm}.npy"
    r_path = CACHE_DIR / f"ranges_{arm}.json"
    parts_dir = CACHE_DIR / f"parts_{arm}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if x_path.exists() and r_path.exists():
        print(f"    [cache] {x_path.name} yüklendi.", flush=True)
        return np.load(x_path), json.loads(r_path.read_text(encoding="utf-8"))

    if r_path.exists():
        ranges = json.loads(r_path.read_text(encoding="utf-8"))
        print("    [cache] ızgara aralıkları yüklendi.", flush=True)
    else:
        ranges = derive_ranges(arm, scales)
        r_path.write_text(json.dumps(ranges, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = []
    for row in meta.itertuples(index=False):
        tasks.append((str(PATIENTS_DIR / row.patient_id / ZONE_FILES[int(row.zone_id)]),
                      arm, scales.get(str(row.patient_id), 1.0), ranges))

    n_shards = (len(tasks) + SHARD - 1) // SHARD
    print(f"    [extract] {len(tasks)} zon | {n_shards} parça × {SHARD} | {MAX_WORKERS} işçi",
          flush=True)
    t0 = time.time()
    done_zones = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for k in range(n_shards):
            chunk = tasks[k * SHARD:(k + 1) * SHARD]
            part = parts_dir / f"{k:04d}.npy"
            if part.exists():
                try:
                    arr = np.load(part)
                    if arr.shape == (len(chunk), 2 * GRID * GRID):
                        done_zones += len(chunk)
                        continue
                    print(f"    [extract] parça {k} bozuk → yeniden", flush=True)
                    part.unlink()
                except Exception:
                    part.unlink()
            feats = np.stack(list(pool.map(_feature_worker, chunk, chunksize=16)))
            np.save(part, feats)
            done_zones += len(chunk)
            rate = done_zones / max(1e-9, time.time() - t0)
            eta = (len(tasks) - done_zones) / max(1e-9, rate)
            print(f"    [extract] parça {k + 1}/{n_shards} | {done_zones}/{len(tasks)} zon | "
                  f"{rate * 60:.0f} zon/dk | kalan ~{eta / 60:.1f} dk", flush=True)

    X = np.concatenate([np.load(parts_dir / f"{k:04d}.npy") for k in range(n_shards)])
    np.save(x_path, X)
    print(f"    [extract] tamamlandı {X.shape} ({time.time() - t0:.0f} s) → {x_path.name}", flush=True)
    return X, ranges


def sanity_check(arm: str, n: int, meta: pd.DataFrame, scales: dict) -> None:
    """`baseline` filtrasyonumuzun projenin dondurulmuş özelliklerini birebir ürettiğini doğrular."""
    frozen = np.load(FROZEN_FEATURES, mmap_mode="r")
    ranges = json.loads(TDA_RANGES_V2.read_text(encoding="utf-8"))["ranges"]
    ranges = {
        "h0_birth": ranges["h0"]["birth_range"], "h0_pers": ranges["h0"]["persistence_range"],
        "h1_birth": ranges["h1"]["birth_range"], "h1_pers": ranges["h1"]["persistence_range"],
    }
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(meta), size=n, replace=False)
    tasks, keep = [], []
    for i in idx:
        row = meta.iloc[i]
        if arm == "baseline":
            tasks.append((str(PATIENTS_DIR / row.patient_id / ZONE_FILES[int(row.zone_id)]),
                          arm, 1.0, ranges))
            keep.append(i)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        got = np.stack(list(pool.map(_feature_worker, tasks, chunksize=8)))
    ref = np.asarray(frozen[keep])
    diff = float(np.max(np.abs(got - ref)))
    print(f"  [sanity] {len(keep)} zon | maks |Δ| = {diff:.3e} "
          f"| {'OK (eşdeğer)' if diff < 1e-3 else 'FARKLI — filtre hattı uyuşmuyor!'}")
    if diff >= 1e-3:
        raise SystemExit("Sanity başarısız: baseline filtrasyonu dondurulmuş özellikleri üretmiyor.")


# ---------------------------------------------------------------------------------------
# Değerlendirme kapıları
# ---------------------------------------------------------------------------------------
def _svm():
    import warnings

    from sklearn.svm import SVC

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
    return SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=True,
               random_state=SEED, cache_size=1000)


def _fit_scaler(X_tr, X_target, tta: bool):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    if tta:
        sc.fit(X_target)          # etiketsiz hedef istatistiği (TTA)
    else:
        sc.fit(X_tr)              # eğitim istatistiği kilitli (saf zero-shot)
    return sc


def val_label_auc(X: np.ndarray, tta: bool) -> dict:
    """Standart v2 splitleri: train dengeli zonlar → val hasta AUC."""
    zs = pd.read_csv(ZONE_SPLITS).sort_values("vector_index").reset_index(drop=True)
    y = zs["zone_label"].values.astype(float)
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        te = zm & (zs["split"] == "val").values & zs["is_zone_metric_sample"].values.astype(bool)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        sc = _fit_scaler(X[tr], X[te], tta)
        clf = _svm().fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        rows.append(pd.DataFrame({
            "patient_id": zs.loc[te, "patient_id"].values,
            "patient_label": zs.loc[te, "patient_label"].values,
            "p": p,
        }))
    df = pd.concat(rows)
    pat = df.groupby(["patient_id", "patient_label"])["p"].max().reset_index()
    from sklearn.metrics import roc_auc_score
    return {
        "patient_auc": float(roc_auc_score(pat["patient_label"], pat["p"])),
        "n_patients": int(len(pat)),
        "n_pos": int((pat["patient_label"] == 1).sum()),
    }


def source_auc(X: np.ndarray, tta: bool) -> dict:
    """A2 protokolü: yalnız pozitiflerde, etiketten bağımsız kaynak (cihaz) tahmini."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    meta = pd.read_csv(FEATURES_META)
    inv = pd.read_csv(INVENTORY)[["patient_id", "source_dataset", "label"]]
    splits = pd.read_csv(PATIENT_SPLITS)[["patient_id", "split"]]
    df = pd.DataFrame(X)
    df["patient_id"] = meta["patient_id"].values
    agg = df.groupby("patient_id").mean()
    info = inv.merge(splits, on="patient_id").set_index("patient_id")
    pos = info[(info["label"] == 1) & (~info["source_dataset"].isin(["node21_control", "unknown"]))]
    agg = agg.loc[agg.index.intersection(pos.index)]
    pos = pos.loc[agg.index]
    tr = (pos["split"] == "train").values
    va = (pos["split"] == "val").values
    y = pos["source_dataset"].values
    classes = sorted(set(y))
    if tta:
        sc = StandardScaler().fit(agg.values[va])
        Xtr, Xva = sc.transform(agg.values[tr]), sc.transform(agg.values[va])
    else:
        sc = StandardScaler().fit(agg.values[tr])
        Xtr, Xva = sc.transform(agg.values[tr]), sc.transform(agg.values[va])
    clf = LogisticRegression(max_iter=3000, random_state=SEED, class_weight="balanced").fit(Xtr, y[tr])
    proba = clf.predict_proba(Xva)
    aucs = [roc_auc_score((y[va] == c).astype(int), proba[:, i]) for i, c in enumerate(classes)]
    return {
        "macro_ovr_auc": float(np.mean(aucs)),
        "per_class": {c: float(a) for c, a in zip(classes, aucs)},
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
    }


def loo_auc(X: np.ndarray, tta: bool) -> dict:
    """E5.3b: kaynak sırayla dışarıda, katlar 1034 dengeli zona eşitlenir."""
    if _E53["mod"] is None:
        spec = importlib.util.spec_from_file_location(
            "e53", PROJECT_ROOT / "pipelines/exp/E5_3_cross_source.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _E53["mod"] = mod
    e53 = _E53["mod"]
    from sklearn.metrics import roc_auc_score

    meta = pd.read_csv(FEATURES_META)
    src = e53.patient_sources(PATIENTS_DIR)
    folds = {}
    for holdout in LOO_SOURCES:
        ps = e53.make_patient_splits(meta, src, holdout, seed=SEED)
        zs, _ = build_zone_splits_v2(meta, ps, seed=SEED)
        zs = e53.cap_balanced_train(zs, LOO_CAP, SEED).reset_index(drop=True)
        y = zs["zone_label"].values.astype(float)
        rows = []
        for z in range(1, 6):
            zm = (zs["zone_id"] == z).values
            tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
            te = zm & (zs["split"] == "test").values
            if tr.sum() < 20 or te.sum() == 0:
                continue
            sc = _fit_scaler(X[tr], X[te], tta)
            clf = _svm().fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            rows.append(pd.DataFrame({
                "patient_id": zs.loc[te, "patient_id"].values,
                "patient_label": zs.loc[te, "patient_label"].values,
                "p": p,
            }))
        pat = pd.concat(rows).groupby(["patient_id", "patient_label"])["p"].max().reset_index()
        folds[holdout] = {
            "patient_auc": float(roc_auc_score(pat["patient_label"], pat["p"])),
            "n_test_patients": int(len(pat)),
            "n_train_balanced_zones": int(zs["is_train_balanced_sample"].sum()),
        }
        print(f"    [LOO] dışarıda={holdout:12s} AUC={folds[holdout]['patient_auc']:.4f} "
              f"(n_test={folds[holdout]['n_test_patients']})")
    return {
        "mean_auc": float(np.mean([f["patient_auc"] for f in folds.values()])),
        "folds": folds,
    }


# ---------------------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="V5.1 fiziksel kanonikleştirme")
    parser.add_argument("--arms", nargs="+", default=["a_density", "b_density_scale"])
    parser.add_argument("--sanity", type=int, default=0, help="Baseline filtre eşdeğerlik testi (N zon)")
    args = parser.parse_args()
    args.arms = [a for a in args.arms if a in ("a_density", "b_density_scale")]

    print("=" * 82)
    print("V5.1 — FİZİKSEL KANONİKLEŞTİRME (yoğunluk + ölçek) · 5-zon mimarisi korunuyor")
    print("=" * 82)

    meta = pd.read_csv(FEATURES_META)
    print(f"Zon: {len(meta)} | hasta: {meta['patient_id'].nunique()}")

    if args.sanity:
        sanity_check("baseline", args.sanity, meta, {})

    tgt = target_lung_height()
    heights = _lung_heights(sorted(meta["patient_id"].unique()))
    scales = {pid: float(np.clip(tgt / h, 0.4, 2.5)) for pid, h in heights.items() if h > 0}
    print(f"NODE21 medyan akciğer yüksekliği (hedef): {tgt:.0f} px | ölçek uygulanan hasta: {len(scales)}")

    features = {"baseline": np.load(FROZEN_FEATURES)}
    for arm in args.arms:
        print(f"\n[çıkarım] kol: {arm}", flush=True)
        X, _ = extract_arm(arm, meta, scales)
        features[arm] = X

    print("\n[değerlendirme]")
    report = {"experiment": EXPERIMENT_ID, "arms": {}, "decisions": {}}
    t0 = time.time()
    for name, X in features.items():
        entry = {
            "val_label": val_label_auc(X, tta=False),
            "source": source_auc(X, tta=False),
            "loo": loo_auc(X, tta=False),
            "val_label_tta": val_label_auc(X, tta=True),
            "source_tta": source_auc(X, tta=True),
            "loo_tta": loo_auc(X, tta=True),
        }
        report["arms"][name] = entry
        print(f"  [{name}] val={entry['val_label']['patient_auc']:.4f} "
              f"source={entry['source']['macro_ovr_auc']:.4f} "
              f"loo={entry['loo']['mean_auc']:.4f} | "
              f"+TTA: val={entry['val_label_tta']['patient_auc']:.4f} "
              f"source={entry['source_tta']['macro_ovr_auc']:.4f} "
              f"loo={entry['loo_tta']['mean_auc']:.4f}  ({time.time() - t0:.0f} s)")

    base = report["arms"]["baseline"]
    rules = {}
    for arm in [a for a in features if a != "baseline"]:
        e = report["arms"][arm]
        ok = (e["val_label"]["patient_auc"] >= base["val_label"]["patient_auc"] - 0.005
              and e["loo"]["mean_auc"] >= base["loo"]["mean_auc"] + 0.015
              and e["source"]["macro_ovr_auc"] <= base["source"]["macro_ovr_auc"] - 0.03)
        rules[arm] = {
            "val_ok": e["val_label"]["patient_auc"] >= base["val_label"]["patient_auc"] - 0.005,
            "loo_ok": e["loo"]["mean_auc"] >= base["loo"]["mean_auc"] + 0.015,
            "source_ok": e["source"]["macro_ovr_auc"] <= base["source"]["macro_ovr_auc"] - 0.03,
            "delta_val": e["val_label"]["patient_auc"] - base["val_label"]["patient_auc"],
            "delta_loo": e["loo"]["mean_auc"] - base["loo"]["mean_auc"],
            "delta_source": e["source"]["macro_ovr_auc"] - base["source"]["macro_ovr_auc"],
            "decision": "KEEP" if ok else "KILL",
        }
        print(f"  KARAR {arm}: {rules[arm]['decision']} "
              f"(Δval={rules[arm]['delta_val']:+.4f}, Δloo={rules[arm]['delta_loo']:+.4f}, "
              f"Δsource={rules[arm]['delta_source']:+.4f})")
    report["decisions"] = rules
    report["config"] = {
        "grid": GRID, "sigma": SIGMA, "svm": {"C": SVM_C, "gamma": SVM_GAMMA},
        "loo_cap": LOO_CAP, "loo_sources": LOO_SOURCES, "seed": SEED,
        "target_lung_height_px": tgt, "rules": "val>=-0.005 & loo>=+0.015 & source<=-0.03",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {REPORT_JSON}")

    try:
        from node21_tda.experiment import new_run_dir, write_manifest

        run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
        (run_dir / "v5_1_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        write_manifest(
            run_dir, EXPERIMENT_ID,
            config=report["config"],
            inputs=[FEATURES_META, ZONE_SPLITS, PATIENT_SPLITS, INVENTORY, FROZEN_FEATURES],
            seed=SEED,
            metrics={f"{a}_{k}": report["arms"][a][k]["patient_auc"]
                     if "patient_auc" in report["arms"][a][k] else report["arms"][a][k]["mean_auc"]
                     for a in features for k in ("val_label", "loo")},
            decision=", ".join(f"{a}:{d['decision']}" for a, d in rules.items()) or "n/a",
            notes="Fiziksel kanonikleştirme (yoğunluk p1-p99 [+ölçek]) ve etiketsiz TTA; 5-zon korunur.",
        )
        print(f"Manifest: {run_dir / 'manifest.json'}")
    except Exception as exc:
        print(f"Manifest yazılamadı: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
