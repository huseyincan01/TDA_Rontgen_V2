"""V5.3 — VinDr-CXR harici doğrulamasının DÜZELTİLMİŞ TDA ile tekrarı.

NEDEN
    V4'te harici doğrulama, o günkü temsille yapıldı: filtrasyon `intensity_max=65535`
    yüzünden fiilen ~4-bit'e kuantalanıyordu. V5.1 bu niceleme hatasını düzeltti
    (doku p1–p99 → 255 seviye) ve iç val'de TDA ensemble 0,8811 → **0,9522**,
    füzyon 0,9729 → **0,9856** çıktı (V5.2). Bu deney aynı düzeltmeyi **harici** veride sınar:
    TDA dalı VinDr'de hâlâ şans düzeyinde mi, yoksa taşınır hâle geldi mi?

PROTOKOL
    * VinDr'de **sıfır eğitim**: ağırlıklar NODE21'de eğitilmiş V5.1 bundle'ları
      (`artifacts/models/zone_cnn_v5_1`, `zone_svm_v5_1`) ve E0.6 ResNet18 checkpoint'i.
    * Izgara aralıkları NODE21-train türevi (`ranges_a_density.json`) — sabit.
    * Kalibrasyon: **kanonik yoğunluk** (per-kırpıntı doku p1–p99). Histogram eşleme
      artık gereksizdir (kanonikleştirme mutlak yoğunluk bağımlılığını kaldırır).
    * Karşılaştırma referansı: V4'ün `matched` kolu (eski temsil): TDA 0,5409 · ResNet 0,6204 · füzyon 0,5438 · optimal a = 1,00.

Kullanım:
    python pipelines/exp/V5_3_corrected_tda_vindr.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.exposure import match_histograms  # noqa: F401  (referans: V4 karşılaştırması)
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

EXPERIMENT_ID = "V5.3"
SLUG = "corrected_tda_vindr"
SEED = 42
N_BOOTSTRAP = 2000
MAX_WORKERS = 10
INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EXT = PROJECT_ROOT / "data/external_vindr"
COHORT = EXT / "cohort_with_diameters.csv"
IMAGES = EXT / "images"
ZONE_MAPS = PROJECT_ROOT / "data/processed/features_vindr/vindr_zone_maps.npz"
ZONE_MAPS_META = PROJECT_ROOT / "data/processed/features_vindr/vindr_zone_maps_meta.json"
RANGES = PROJECT_ROOT / "data/processed/features_v5_1/ranges_a_density.json"
CNN_BUNDLE = PROJECT_ROOT / "artifacts/models/zone_cnn_v5_1/zone_cnn_v5_1_bundle.pt"
SVM_BUNDLE = PROJECT_ROOT / "artifacts/models/zone_svm_v5_1/zone_svm_v5_1_bundle.joblib"
RESNET_CKPT = PROJECT_ROOT / "artifacts/models/zone_resnet18/resnet18_e0_6.pt"
REPORT_JSON = PROJECT_ROOT / "data/external_vindr/v5_3_corrected_tda_vindr_report.json"
V4_REFERENCE = {"tda": 0.5409, "resnet": 0.6204, "fusion": 0.5438, "a_opt": 1.00}

_W: dict = {}


# --- V5.1 filtrasyon tanımını tek kaynaktan al ---
def _v51():
    if _W.get("v51") is None:
        spec = importlib.util.spec_from_file_location(
            "v51", PROJECT_ROOT / "pipelines/exp/V5_1_physical_canonicalization.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _W["v51"] = mod
    return _W["v51"]


def _tda_worker(args):
    path, ranges = args
    v51 = _v51()
    if path is None:
        return np.zeros(2 * v51.GRID * v51.GRID, dtype=np.float32)
    ex = ZoneTDAExtractor(v51._settings(ranges))
    arr = np.array(Image.open(path))
    fil = v51.build_filtration(arr, "a_density", 1.0)
    if fil is None:
        return np.zeros(2 * v51.GRID * v51.GRID, dtype=np.float32)
    h0, h1 = ex.compute_diagrams(fil)
    return ex.vectorize(h0, h1)


def _crop_zone(img: np.ndarray, zone_map: np.ndarray, z: int):
    zmask = zone_map == z
    if not np.any(zmask):
        return None
    zimg = np.zeros_like(img)
    zimg[zmask] = img[zmask]
    y, x = np.where(zmask)
    y0, y1 = max(0, y.min() - 5), min(img.shape[0], y.max() + 6)
    x0, x1 = max(0, x.min() - 5), min(img.shape[1], x.max() + 6)
    return zimg[y0:y1, x0:x1]


def _resnet_patch(crop) -> np.ndarray:
    if crop is None or crop.size < 50:
        return np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    tissue = crop[crop > 0]
    lo, hi = np.percentile(tissue, [1.0, 99.5]) if tissue.size > 10 else (0, 3600)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    pil = Image.fromarray((norm * 255.0).astype(np.uint8)).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return np.array(pil, dtype=np.uint8)


@torch.no_grad()
def _resnet_probs(patches: np.ndarray) -> np.ndarray:
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    ckpt = torch.load(RESNET_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = []
    for i in range(0, len(patches), 128):
        b = patches[i:i + 128]
        x = torch.from_numpy(b).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        m = torch.tensor(MEAN).view(1, 3, 1, 1)
        s = torch.tensor(STD).view(1, 3, 1, 1)
        out.append(torch.sigmoid(model((x - m) / s).squeeze(-1)).numpy())
    return np.concatenate(out)


def bootstrap_ci(y, p, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 82)
    print("V5.3 — VINDR-CXR: DÜZELTİLMİŞ TDA (65535 niceleme düzeltmesi) ile harici test")
    print("=" * 82)

    cohort = pd.read_csv(COHORT)
    cohort["image_id"] = cohort["image_id"].astype(str)
    y = cohort["patient_label"].values.astype(int)
    ranges = json.loads(RANGES.read_text(encoding="utf-8"))
    print(f"Kohort: {len(cohort)} görüntü | izgara aralıkları: {ranges['h1_birth']} / {ranges['h1_pers']}")

    zmap_data = np.load(ZONE_MAPS)
    zone_maps = {k: zmap_data[k] for k in zmap_data.files}
    print(f"Segmentasyon önbelleği: {len(zone_maps)} görüntü")

    # --- VinDr kesitleri + kanonik TDA özellikleri + ResNet yamaları ---
    print("\n[1/4] Kesitler, yamalar ve TDA özellikleri (kanonik yoğunluk)...")
    t0 = time.time()
    rows, patches, tasks = [], [], []
    for img_id in cohort["image_id"]:
        img = np.array(Image.open(IMAGES / f"{img_id}.jpg"))
        zmap = zone_maps[img_id]
        for z in range(1, 6):
            crop = _crop_zone(img, zmap, z)
            rows.append({"image_id": img_id, "zone_id": z, "patient_label": int(
                cohort.loc[cohort["image_id"] == img_id, "patient_label"].iloc[0])})
            patches.append(_resnet_patch(crop))
            if crop is not None:
                tmp = (IMAGES / f"{img_id}.jpg").as_posix(), ranges
                tasks.append(tmp)
            else:
                tasks.append((None, ranges))
    zone_df = pd.DataFrame(rows)
    resnet_patches = np.stack(patches)

    # TDA: kesitleri yeniden okuyup kanonik filtrasyon uygula (paralel)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        feats = list(pool.map(_tda_worker, tasks, chunksize=16))
    X = np.stack(feats)
    print(f"  TDA özellikleri: {X.shape} | doluluk {(X > 0).mean():.3f} ({time.time() - t0:.0f} s)")

    # --- TDA ensemble (V5.1 bundle'ları, sıfır eğitim) ---
    print("\n[2/4] Dondurulmuş V5.1 TDA bundle'larıyla çıkarım...")
    cnn_b = torch.load(CNN_BUNDLE, map_location="cpu", weights_only=False)
    svm_b = joblib.load(SVM_BUNDLE)
    p_tda_zone = np.zeros(len(X))
    for z in range(1, 6):
        m = (zone_df["zone_id"] == z).values
        sc = cnn_b["scalers"][z]
        Xz = sc.transform(X[m])
        model = SmallZoneCNN(in_channels=2, dropout=float(cnn_b["config"]["dropout"]))
        model.load_state_dict(cnn_b["models_state_dict"][z])
        model.eval()
        with torch.no_grad():
            p_cnn = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(Xz), dtype=torch.float32))).numpy().flatten()
        p_svm = svm_b[z].predict_proba(Xz)[:, 1]
        p_tda_zone[m] = 0.55 * p_cnn + 0.45 * p_svm
    zone_df["p_tda"] = p_tda_zone

    # --- ResNet (E0.6 checkpoint, sıfır eğitim) ---
    print("\n[3/4] ResNet18 (E0.6 checkpoint) çıkarımı...")
    zone_df["p_res"] = _resnet_probs(resnet_patches)

    # --- Değerlendirme ---
    print("\n[4/4] Değerlendirme...")
    p_tda = zone_df.groupby("image_id")["p_tda"].max().reindex(cohort["image_id"]).values
    p_res = zone_df.groupby("image_id")["p_res"].max().reindex(cohort["image_id"]).values

    tda_auc = float(roc_auc_score(y, p_tda))
    res_auc = float(roc_auc_score(y, p_res))
    a_sweep = {f"{a:.2f}": float(roc_auc_score(y, a * p_res + (1 - a) * p_tda))
               for a in np.round(np.arange(0.0, 1.0001, 0.05), 2)}
    best_a = max(a_sweep, key=a_sweep.get)
    fus_auc = a_sweep["0.45"]
    corr = float(np.corrcoef(p_res, p_tda)[0, 1])

    pos = cohort["patient_label"].values == 1
    rel = cohort["rel_size"].values
    q1, q2 = np.nanpercentile(rel[pos], [100 / 3, 200 / 3])
    subs = {}
    for name, pm in (("kucuk_tercile", pos & (rel <= q1)),
                     ("orta_tercile", pos & (rel > q1) & (rel <= q2)),
                     ("buyuk_tercile", pos & (rel > q2))):
        m = (~pos) | pm
        subs[name] = {"n_positive": int(pm.sum()),
                      "tda_auc": float(roc_auc_score(y[m], p_tda[m])),
                      "resnet_auc": float(roc_auc_score(y[m], p_res[m])),
                      "fusion_auc": float(roc_auc_score(y[m], (0.45 * p_res + 0.55 * p_tda)[m]))}

    report = {
        "experiment": EXPERIMENT_ID,
        "question": "65535 niceleme düzeltmesi TDA dalını harici veride taşınır hâle getirdi mi?",
        "cohort": {"n_images": int(len(cohort)), "n_positive": int(y.sum()),
                   "n_negative": int((y == 0).sum()), "level": "image",
                   "source": "VinDr-CXR 3x downsampled JPEG"},
        "protocol": {"training_on_vindr": False,
                     "tda_bundles": "V5.1 (NODE21, kanonik yoğunluk)",
                     "resnet": "E0.6 checkpoint",
                     "calibration": "kanonik yoğunluk (per-kırpıntı doku p1–p99); histogram eşleme YOK",
                     "ranges": "ranges_a_density.json (NODE21-train türevi)",
                     "features_fill_ratio": float((X > 0).mean())},
        "results": {"tda_auc": tda_auc, "tda_ci95": bootstrap_ci(y, p_tda),
                    "resnet_auc": res_auc, "resnet_ci95": bootstrap_ci(y, p_res),
                    "fusion_a045": fus_auc, "fusion_a045_ci95": bootstrap_ci(y, 0.45 * p_res + 0.55 * p_tda),
                    "a_sweep": a_sweep, "a_best": {"a": float(best_a), "auc": a_sweep[best_a]},
                    "correlation_r": corr},
        "subgroups_scale_free": subs,
        "tertile_boundaries_rel_size": [float(q1), float(q2)],
        "v4_reference_matched_arm": V4_REFERENCE,
        "comparison_vs_v4": {
            "delta_tda": tda_auc - V4_REFERENCE["tda"],
            "delta_resnet": res_auc - V4_REFERENCE["resnet"],
            "delta_fusion_a045": fus_auc - V4_REFERENCE["fusion"],
            "a_opt_old": V4_REFERENCE["a_opt"], "a_opt_new": float(best_a),
        },
    }

    print("\n" + "=" * 82)
    print(f"  TDA ensemble (kanonik) : {tda_auc:.4f} [CI {report['results']['tda_ci95'][0]:.4f}, "
          f"{report['results']['tda_ci95'][1]:.4f}]   (V4 eski temsil: {V4_REFERENCE['tda']:.4f})")
    print(f"  ResNet18 (E0.6)        : {res_auc:.4f}   (V4: {V4_REFERENCE['resnet']:.4f})")
    print(f"  Füzyon (a=0.45)        : {fus_auc:.4f}   (V4: {V4_REFERENCE['fusion']:.4f})")
    print(f"  VinDr'de optimal a     : {best_a} → {a_sweep[best_a]:.4f}   (V4: a=1.00 → 0.6204)")
    print(f"  r(ResNet,TDA)          : {corr:.4f}")
    print("=" * 82)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    pred = cohort[["image_id", "patient_label", "group", "rel_size"]].copy()
    pred["p_tda"] = p_tda
    pred["p_resnet"] = p_res
    pred.to_csv(run_dir / "vindr_predictions_v5_3.csv", index=False)
    write_manifest(run_dir, EXPERIMENT_ID, config=report["protocol"],
                   inputs=[COHORT, RANGES, CNN_BUNDLE, SVM_BUNDLE, RESNET_CKPT], seed=SEED,
                   metrics={"tda_auc": tda_auc, "resnet_auc": res_auc, "fusion_a045": fus_auc,
                            "a_best": float(best_a)},
                   decision="EXTERNAL",
                   notes="Düzeltilmiş TDA ile VinDr harici testi (sıfır eğitim).")
    print(f"Rapor: {REPORT_JSON}")


if __name__ == "__main__":
    main()
