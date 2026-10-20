"""V4 — VinDr-CXR harici doğrulama (v3 E0.6 mimarisi, sıfır eğitim).

Bu betik, daha önce iki ayrı kopya hâlinde bulunan VinDr değerlendirmesinin
**tek kaynağıdır** ve şu düzeltmeleri içerir:

1. **Kalibrasyon kolları** (aynı koşuda, aynı segmentasyon üzerinde):
   * ``linear``          : 8-bit → 12-bit doğrusal ölçekleme (×16). Eşlemesiz baseline.
   * ``matched``         : 200 hastalık NODE21 havuzuna histogram eşleme.
   * ``matched_scalenorm``: histogram eşleme + **akciğer yüksekliğine göre ölçek
     normalizasyonu** (piksel-ölçeği confound'unu sınamak için).
2. **Füzyon a-duyarlılığı**: E0.6'nın kilitli a=0.45'i (görüntü ağırlığı) ana
   sonuçtur; ayrıca 0–1 arası tam tarama raporlanır (harici kümede optimal a
   değişebilir; okuyucu bunu görebilsin).
3. **NODE21 sanity kontrolü**: ResNet18, E0.6 tarifiyle (seed 42) yeniden eğitilir
   ve val zon AUC'sinin E0.6 ile **birebir** eşleştiği doğrulanır (0.8703).
   Böylece "dondurulmuş E0.6 modeli" iddiası denetlenebilir olur. Model
   ``artifacts/models/zone_resnet18/`` altına kaydedilir ve ``--reuse-resnet``
   ile yeniden eğitilmeden yüklenebilir. **VinDr üzerinde hiçbir eğitim yoktur.**
4. **Ölçek-bağımsız alt gruplar**: nodül boyutu mm/px yerine göreli boyut
   (``rel_size``) üçte birlikleri ile raporlanır; VinDr kohortunda spacing
   olmadığı için mm dönüşümü yapılmaz.
5. **Kayıt**: hasta düzeyi tahmin CSV'leri + ``artifacts/runs/V4_*/manifest.json``.

Kullanım:
    python pipelines/exp/V4_external_validation_vindr.py [--reuse-resnet]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage.exposure import match_histograms
from skimage.transform import resize as sk_resize
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.evaluation.metrics import compute_binary_metrics, find_optimal_threshold  # noqa: E402
from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

EXPERIMENT_ID = "V4"
SLUG = "vindr_external_validation"
SEED = 42

# --- E0.6 tarifi (birebir) ---
INPUT_SIZE = 224
EPOCHS = 25
LR = 1e-4
WD = 1e-4
DROPOUT = 0.3
BATCH = 64
TDA_CWT = 0.55                 # E3.4: 5-zon TDA ensemble (CNN + SVM) CNN ağırlığı
E0_6_VAL_ZONE_AUC = 0.8703325946876191
E0_6_FUSION_A = 0.45           # E0.6 kilitli görüntü ağırlığı
E0_6_THRESHOLD = 0.7172159718111142
N_BOOTSTRAP = 2000
MAX_CPU_WORKERS = 6

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COHORT_CSV = PROJECT_ROOT / "data/external_vindr/cohort_with_diameters.csv"
IMAGES_DIR = PROJECT_ROOT / "data/external_vindr/images"
REF_NPZ = PROJECT_ROOT / "data/processed/node21_200pt_ref_histograms.npz"
SPLITS_PATH = PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv"
SPLITS_PATIENT = PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv"
RESNET_CACHE = PROJECT_ROOT / "data/processed/features_resnet18/trainval_gray224.npy"
TDA_FEATURES = PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy"
CNN_BUNDLE = PROJECT_ROOT / "artifacts/models/zone_cnn/zone_cnn_bundle.pt"
SVM_BUNDLE = PROJECT_ROOT / "artifacts/models/zone_svm/zone_svm_bundle.joblib"
RESNET_CKPT = PROJECT_ROOT / "artifacts/models/zone_resnet18/resnet18_e0_6.pt"
CACHE_DIR = PROJECT_ROOT / "data/processed/features_vindr"
ZONES_DIR = PROJECT_ROOT / "data/processed/zones/patients"

ZONE_FILES = {
    1: "zone_1_sag_ust.mha",
    2: "zone_2_sag_orta.mha",
    3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha",
    5: "zone_5_sol_alt.mha",
}

TDA_SETTINGS = TDASettings(
    grid_size=20,
    intensity_max=65535.0,
    h0_birth_range=(0.0, 248.0),
    h0_persistence_range=(0.0, 248.0),
    h1_birth_range=(239.0, 250.0),
    h1_persistence_range=(0.0, 7.0),
    pi_sigma_pixels=1.0,
)

# --------------------------------------------------------------------------------------
# İşçi süreçler (Windows: modül düzeyinde olmalı)
# --------------------------------------------------------------------------------------
_W: dict[str, object] = {}


def _tda_worker(args):
    crop, settings = args
    if crop is None or crop.size < 50:
        return np.zeros(settings.vector_size, dtype=np.float32)
    ex = _W.get("ex")
    if ex is None:
        ex = ZoneTDAExtractor(settings)
        _W["ex"] = ex
    vec, _ = ex.extract(crop)
    return vec.astype(np.float32)


def set_low_process_priority() -> None:
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
    except Exception:
        pass


def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_pred[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_auc_diff(y, pa, pb, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    d = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        d.append(roc_auc_score(yi, np.asarray(pa)[idx]) - roc_auc_score(yi, np.asarray(pb)[idx]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p2 = float(min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))))
    return float(lo), float(hi), p2


# --------------------------------------------------------------------------------------
# Girdi hazırlığı
# --------------------------------------------------------------------------------------
def node21_median_lung_height(n_patients: int = 200) -> float:
    """NODE21 train örnekleminden medyan akciğer yüksekliği (piksel).

    Zone bbox'larından: sağ akciğer = zon1.min_y → zon3.max_y,
    sol akciğer = zon4.min_y → zon5.max_y; hasta için ikisinin maksimumu.
    """
    splits = pd.read_csv(SPLITS_PATIENT)
    train_ids = sorted(splits.loc[splits["split"] == "train", "patient_id"].astype(str))
    rng = np.random.RandomState(SEED)
    picked = rng.choice(train_ids, size=min(n_patients, len(train_ids)), replace=False)
    heights = []
    for pid in picked:
        p = ZONES_DIR / str(pid) / "patient_info.json"
        if not p.exists():
            continue
        info = json.loads(p.read_text(encoding="utf-8"))
        zones = info.get("zones", {})
        try:
            right = zones["3"]["crop_bbox"]["max_y"] - zones["1"]["crop_bbox"]["min_y"]
            left = zones["5"]["crop_bbox"]["max_y"] - zones["4"]["crop_bbox"]["min_y"]
            heights.append(max(right, left))
        except (KeyError, TypeError):
            continue
    if not heights:
        raise RuntimeError("NODE21 akciğer yükseklikleri hesaplanamadı.")
    return float(np.median(heights))


def lung_height_from_masks(seg: dict) -> float:
    heights = []
    for key in ("right_lung", "left_lung"):
        mask = seg.get(key)
        if mask is None or not np.any(mask):
            continue
        rows = np.where(mask.any(axis=1))[0]
        heights.append(float(rows[-1] - rows[0] + 1))
    return float(max(heights)) if heights else float("nan")


def build_or_load_zone_maps(cohort: pd.DataFrame, device: torch.device):
    """Segmentasyonları önbelleğe alır: zone_map + akciğer yüksekliği."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    maps_path = CACHE_DIR / "vindr_zone_maps.npz"
    meta_path = CACHE_DIR / "vindr_zone_maps_meta.json"
    ids = cohort["image_id"].astype(str).tolist()
    if maps_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("image_ids") == ids:
            data = np.load(maps_path)
            print(f"  Segmentasyon önbellekten yüklendi ({len(ids)} görüntü).")
            return {k: data[k] for k in data.files}, meta["lung_heights"]

    print("  PSPNet 5-zon segmentasyonu çalıştırılıyor...")
    segmenter = PSPNet5ZoneSegmenter(device=device)
    zone_maps: dict[str, np.ndarray] = {}
    lung_heights: dict[str, float] = {}
    t0 = time.time()
    for i, img_id in enumerate(ids, start=1):
        img = np.array(Image.open(IMAGES_DIR / f"{img_id}.jpg"))
        seg = segmenter.segment(img)
        zone_maps[img_id] = seg["zone_map"].astype(np.uint8)
        lung_heights[img_id] = lung_height_from_masks(seg)
        if i % 50 == 0:
            print(f"    {i}/{len(ids)} ({time.time() - t0:.0f} s)")
    np.savez_compressed(maps_path, **zone_maps)
    meta_path.write_text(
        json.dumps({"image_ids": ids, "lung_heights": lung_heights}, ensure_ascii=False),
        encoding="utf-8",
    )
    return zone_maps, lung_heights


def crop_zone(img: np.ndarray, zone_map: np.ndarray, z: int):
    zmask = zone_map == z
    if not np.any(zmask):
        return None
    zimg = np.zeros_like(img)
    zimg[zmask] = img[zmask]
    y_c, x_c = np.where(zmask)
    y0, y1 = max(0, y_c.min() - 5), min(img.shape[0], y_c.max() + 6)
    x0, x1 = max(0, x_c.min() - 5), min(img.shape[1], x_c.max() + 6)
    return zimg[y0:y1, x0:x1]


def to_resnet_patch(crop: np.ndarray) -> np.ndarray:
    """E0.4/E0.6 ile birebir aynı ön işleme (doku yüzdelikleri → 224)."""
    if crop is None or crop.size < 50:
        return np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    tissue = crop[crop > 0]
    lo, hi = np.percentile(tissue, [1.0, 99.5]) if tissue.size > 10 else (0, 3600)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    pil = Image.fromarray((norm * 255.0).astype(np.uint8)).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    return np.array(pil, dtype=np.uint8)


def apply_arm(crop8: np.ndarray | None, arm: str, zone_id: int, ref_zone: np.ndarray,
              scale: float) -> np.ndarray | None:
    """Bir zon kesitini kol tanımına göre NODE21 aralığına getirir.

    ÖNEMLİ: Eşleme **yalnızca doku piksellerine** uygulanır ve arka plan ``0``
    olarak korunur. Aksi hâlde (tüm kesiti havuzlanmış *doku* referansına eşlemek)
    sıfır arka plan referansın en düşük değerine (~1080) taşınır; extractor'ın
    ``image > 0`` doku maskesi bozulur ve TDA'ya yapay sınır yapısı eklenir.
    """
    if crop8 is None:
        return None
    if arm == "linear":
        # 8-bit → NODE21'in 12-bit aralığı (×16 ≈ 0–4080)
        return (crop8.astype(np.uint16) * 16)

    tissue_mask = crop8 > 0
    matched = np.zeros(crop8.shape, dtype=np.uint16)
    if np.any(tissue_mask):
        mapped = match_histograms(crop8[tissue_mask], ref_zone).astype(np.uint16)
        matched[tissue_mask] = mapped

    if arm == "matched":
        return matched
    if arm == "matched_scalenorm":
        if not np.isfinite(scale) or scale <= 0:
            return matched
        h, w = matched.shape
        new_h, new_w = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        # Büyük ölçeklerde bellek patlamasını önle
        if new_h * new_w > 4_000_000:
            f = np.sqrt(4_000_000 / (new_h * new_w))
            new_h, new_w = int(new_h * f), int(new_w * f)
        out = sk_resize(matched.astype(np.float32), (new_h, new_w), order=1,
                        preserve_range=True, anti_aliasing=True)
        mask = sk_resize(tissue_mask.astype(np.float32), (new_h, new_w), order=0,
                         preserve_range=True, anti_aliasing=False)
        out = np.where(mask > 0.5, out, 0.0)
        return np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    raise ValueError(f"Bilinmeyen kol: {arm}")


def build_arm_features(arm: str, cohort: pd.DataFrame, zone_maps: dict, ref_zones: dict,
                       lung_heights: dict, target_lung_h: float, need_patches: bool):
    ids = cohort["image_id"].astype(str).tolist()
    tasks = []
    rows = []
    patches = [] if need_patches else None

    for img_id in ids:
        img = np.array(Image.open(IMAGES_DIR / f"{img_id}.jpg"))
        zmap = zone_maps[img_id]
        lh = lung_heights.get(img_id, float("nan"))
        scale = target_lung_h / lh if lh and np.isfinite(lh) and lh > 0 else float("nan")
        if np.isfinite(scale):
            scale = float(np.clip(scale, 0.4, 2.5))
        for z in range(1, 6):
            raw = crop_zone(img, zmap, z)
            crop = apply_arm(raw, arm, z, ref_zones[z], scale)
            tasks.append((crop, TDA_SETTINGS))
            if need_patches:
                patches.append(to_resnet_patch(raw))
            rows.append({"image_id": img_id, "zone_id": z})

    print(f"    TDA özellik çıkarımı: {len(tasks)} zon, {MAX_CPU_WORKERS} işçi...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as pool:
        feats = list(pool.map(_tda_worker, tasks, chunksize=25))
    X = np.stack(feats)
    print(f"    Tamamlandı: {X.shape} ({time.time() - t0:.0f} s)")
    patch_arr = np.stack(patches) if need_patches else None
    return pd.DataFrame(rows), X, patch_arr


# --------------------------------------------------------------------------------------
# ResNet18 (E0.6 modeli)
# --------------------------------------------------------------------------------------
def raw_batch(gray, device):
    return torch.from_numpy(gray).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1).to(device)


def normalize_img(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def augment_hflip(x):
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])
    return x


@torch.no_grad()
def predict_resnet(model, gray, idx, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out.append(torch.sigmoid(model(normalize_img(raw_batch(gray[b], device))).squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def set_determinism() -> dict:
    """ResNet eğitimini mümkün olduğunca deterministik kılar.

    Not: E0.6'nın ağırlıkları hiç kaydedilmediği için bit-bit yeniden üretim
    **imkânsızdır** (GPU konvolüsyonları + val üzerinden en iyi epoch seçimi).
    Amaç, *bizim* koşumuzun tekrar edilebilir olması ve E0.6 ile uyumun
    raporlanmasıdır.
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def train_or_load_resnet(device: torch.device, reuse: bool):
    splits = pd.read_csv(SPLITS_PATH).sort_values("vector_index").reset_index(drop=True)
    need = (splits["is_train_balanced_sample"].values.astype(bool)
            | (splits["split"] == "val").values | (splits["split"] == "test").values)
    sub = splits[need].reset_index(drop=True)
    y = sub["zone_label"].values.astype(np.float32)
    gray = np.load(RESNET_CACHE)
    if len(gray) != len(sub):
        raise ValueError(f"ResNet önbelleği uyuşmuyor: {len(gray)} vs {len(sub)}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(model.fc.in_features, 1))
    model.to(device)

    idx_tr = np.where(sub["is_train_balanced_sample"].values.astype(bool))[0]
    idx_va = np.where((sub["split"] == "val").values & sub["is_zone_metric_sample"].values)[0]

    if reuse and RESNET_CKPT.exists():
        ckpt = torch.load(RESNET_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"  ResNet18 checkpoint'ten yüklendi (val zon AUC={ckpt['val_zone_auc']:.4f}, "
              f"Δ(E0.6)={ckpt.get('delta_vs_e0_6', float('nan')):.2e}).")
        return model, float(ckpt["val_zone_auc"]), bool(ckpt.get("delta_vs_e0_6", 1.0) <= 5e-3)

    print("  ResNet18 E0.6 tarifiyle NODE21'de yeniden eğitiliyor (VinDr'de eğitim YOK)...")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.RandomState(SEED + ep)
        rng.shuffle(idx_tr)
        for i in range(0, len(idx_tr), BATCH):
            b = idx_tr[i:i + BATCH]
            xb = normalize_img(augment_hflip(raw_batch(gray[b], device)))
            yb = torch.tensor(y[b], device=device)
            opt.zero_grad()
            crit(model(xb).squeeze(-1), yb).backward()
            opt.step()
        pv = predict_resnet(model, gray, idx_va, device)
        auc = roc_auc_score(y[idx_va], pv)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"    epoch {ep:2d}/{EPOCHS} | val zon AUC={auc:.4f} | en iyi={best_auc:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    # ---- SANITY: E0.6 ile uyum (bit-bit değil; ağırlıklar saklanmadı) ----
    delta = abs(best_auc - E0_6_VAL_ZONE_AUC)
    print(f"  Sanity (NODE21 val zon AUC): {best_auc:.6f} | E0.6 kaydı: {E0_6_VAL_ZONE_AUC:.6f} | Δ={delta:.2e}")
    print("  Not: E0.6 ağırlıkları kaydedilmediği için bit-bit eşitlik beklenmez;")
    print("       bu koşumun checkpoint'i sabitlenir ve VinDr'de bu model kullanılır.")
    if delta > 5e-3:
        raise RuntimeError(
            "ResNet18 yeniden eğitimi E0.6 ile uyumsuz "
            f"(Δ={delta:.2e} > 5e-3). Eğitim tarifinin/ortamın E0.6 ile aynı olduğu "
            "doğrulanmalı; aksi hâlde VinDr sonuçları E0.6 mimarisine atfedilemez."
        )

    RESNET_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "val_zone_auc": float(best_auc),
        "e0_6_reference_val_zone_auc": float(E0_6_VAL_ZONE_AUC),
        "delta_vs_e0_6": float(delta),
        "bitwise_identical_possible": False,
        "recipe": {"epochs": EPOCHS, "lr": LR, "wd": WD, "dropout": DROPOUT,
                   "batch": BATCH, "seed": SEED, "aug": "hflip"},
    }, RESNET_CKPT)
    print(f"  Checkpoint kaydedildi: {RESNET_CKPT}")
    return model, float(best_auc), bool(delta <= 5e-3)


# --------------------------------------------------------------------------------------
# Değerlendirme
# --------------------------------------------------------------------------------------
def evaluate_arm(arm: str, p_tda: np.ndarray, p_res: np.ndarray, y: np.ndarray,
                 cohort: pd.DataFrame) -> dict:
    a = E0_6_FUSION_A
    p_fus = a * p_res + (1 - a) * p_tda

    a_sweep = {}
    for a_val in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        a_sweep[f"{a_val:.2f}"] = float(roc_auc_score(y, a_val * p_res + (1 - a_val) * p_tda))
    best_a = max(a_sweep, key=a_sweep.get)

    tda_auc = float(roc_auc_score(y, p_tda))
    res_auc = float(roc_auc_score(y, p_res))
    fus_auc = float(roc_auc_score(y, p_fus))
    delta = fus_auc - res_auc
    d_lo, d_hi, p_val = bootstrap_auc_diff(y, p_fus, p_res)

    youden_thr, _ = find_optimal_threshold(y, p_fus, method="youden")

    # Ölçek-bağımsız alt gruplar (göreli boyut üçte birlikleri)
    pos = cohort["patient_label"].values == 1
    rel = cohort["rel_size"].values
    q1, q2 = np.nanpercentile(rel[pos], [100 / 3, 200 / 3])
    neg = ~pos
    subgroups = {}
    defs = {
        "kucuk_tercile": pos & (rel <= q1),
        "orta_tercile": pos & (rel > q1) & (rel <= q2),
        "buyuk_tercile": pos & (rel > q2),
    }
    for name, pos_mask in defs.items():
        m = neg | pos_mask
        if pos_mask.sum() < 5:
            continue
        subgroups[name] = {
            "definition": "rel_size = bbox_max_dim_px / image_h üçte birlik dilimi",
            "n_positive": int(pos_mask.sum()),
            "n_negative": int(neg.sum()),
            "rel_size_range": [float(np.nanmin(rel[pos_mask])), float(np.nanmax(rel[pos_mask]))],
            "tda_auc": float(roc_auc_score(y[m], p_tda[m])),
            "resnet_auc": float(roc_auc_score(y[m], p_res[m])),
            "fusion_auc": float(roc_auc_score(y[m], p_fus[m])),
        }

    return {
        "arm": arm,
        "a_locked": a,
        "auc": {"tda": tda_auc, "resnet": res_auc, "fusion": fus_auc},
        "ci95": {
            "tda": bootstrap_ci(y, p_tda),
            "resnet": bootstrap_ci(y, p_res),
            "fusion": bootstrap_ci(y, p_fus),
        },
        "pr_auc": {
            "tda": float(average_precision_score(y, p_tda)),
            "resnet": float(average_precision_score(y, p_res)),
            "fusion": float(average_precision_score(y, p_fus)),
        },
        "correlation_r_resnet_tda": float(np.corrcoef(p_res, p_tda)[0, 1]),
        "fusion_delta_over_resnet": delta,
        "fusion_delta_ci95": [d_lo, d_hi],
        "fusion_delta_p": p_val,
        "a_sweep": a_sweep,
        "a_best_in_vindr": {"a": float(best_a), "auc": a_sweep[best_a]},
        "thresholds": {
            f"e0.6_locked_{E0_6_THRESHOLD:.4f}": compute_binary_metrics(y, p_fus, threshold=E0_6_THRESHOLD),
            "vindr_youden": {"threshold": float(youden_thr),
                             "metrics": compute_binary_metrics(y, p_fus, threshold=youden_thr)},
        },
        "subgroups_scale_free": subgroups,
        "subgroup_tertile_boundaries_rel_size": [float(q1), float(q2)],
    }


def main() -> None:
    try:  # Windows konsolu cp1254 olabilir; sembolleri güvenli yazdır
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="V4 VinDr-CXR harici doğrulama")
    parser.add_argument("--reuse-resnet", action="store_true",
                        help="E0.6 ResNet checkpoint'i varsa yeniden eğitme.")
    parser.add_argument("--arms", nargs="+", default=["linear", "matched", "matched_scalenorm"],
                        help="Değerlendirilecek kalibrasyon kolları.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Yalnızca ilk N görüntüyle çalış (duman testi). 0 = tümü.")
    args = parser.parse_args()

    set_low_process_priority()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    determinism = set_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("V4 — VINDR-CXR HARİCİ DOĞRULAMA (v3 E0.6 mimarisi, sıfır eğitim)")
    print("=" * 78)
    print(f"Cihaz: {device} | CPU işçi: {MAX_CPU_WORKERS}")

    cohort = pd.read_csv(COHORT_CSV)
    cohort["image_id"] = cohort["image_id"].astype(str)
    if args.limit:
        # Stratifiye alt küme (duman testi): yarı pozitif, yarı kontrol
        half = max(1, args.limit // 2)
        pos = cohort[cohort["patient_label"] == 1].head(half)
        neg = cohort[cohort["patient_label"] == 0].head(args.limit - len(pos))
        cohort = pd.concat([pos, neg]).reset_index(drop=True)
    y = cohort["patient_label"].values.astype(int)
    print(f"Kohort: {len(cohort)} görüntü ({int(y.sum())} nodül/kitle, {int((y == 0).sum())} kontrol)")
    print("Not: VinDr metadata'sında hasta kimliği yok → değerlendirme GÖRÜNTÜ düzeyindedir.")

    ref = np.load(REF_NPZ)
    ref_zones = {z: ref[f"z{z}"] for z in range(1, 6)}

    print("\n[1/5] Segmentasyon (önbellekli)...")
    zone_maps, lung_heights = build_or_load_zone_maps(cohort, device)

    target_lh = node21_median_lung_height()
    vh = np.array([v for v in lung_heights.values() if np.isfinite(v)])
    print(f"  NODE21 medyan akciğer yüksekliği: {target_lh:.0f} px | "
          f"VinDr medyanı: {np.median(vh) if vh.size else float('nan'):.0f} px | "
          f"ölçek ≈ {target_lh / np.median(vh) if vh.size else float('nan'):.2f}×")

    # ResNet18 (E0.6 tarifi) — ucuz; sanity kontrolü pahalı TDA kollarından ÖNCE çalışsın.
    print("\n[2/5] ResNet18 (E0.6 tarifi; VinDr'de eğitim yok)...")
    model_res, val_auc, reproduced = train_or_load_resnet(device, args.reuse_resnet)

    # TDA dalı (dondurulmuş, NODE21'de eğitilmiş)
    cnn_bundle = torch.load(CNN_BUNDLE, map_location="cpu", weights_only=False)
    svm_bundle = joblib.load(SVM_BUNDLE)
    drop = float(cnn_bundle["config"].get("dropout", 0.3))

    print("\n[3/5] VinDr özellikleri (kollar)...")
    per_arm = {}
    res_patient_zone = None
    for arm in args.arms:
        print(f"  Kol: {arm}")
        need_patches = res_patient_zone is None
        zone_df, X, patches = build_arm_features(
            arm, cohort, zone_maps, ref_zones, lung_heights, target_lh, need_patches
        )
        if need_patches:
            res_patient_zone = (zone_df, patches)

        # TDA ensemble (CNN + SVM; E3.4)
        p_tda_zone = np.zeros(len(X), dtype=np.float64)
        for z in range(1, 6):
            m = (zone_df["zone_id"] == z).values
            Xz = X[m]
            sc = cnn_bundle["scalers"][z]
            model = SmallZoneCNN(in_channels=2, dropout=drop)
            model.load_state_dict(cnn_bundle["models_state_dict"][z])
            model.eval()
            with torch.no_grad():
                p_cnn = torch.sigmoid(model(torch.tensor(
                    vector_to_2channel_image(sc.transform(Xz)), dtype=torch.float32))).numpy().flatten()
            p_svm = svm_bundle.predict_zone_proba(z, Xz).flatten()
            p_tda_zone[m] = TDA_CWT * p_cnn + (1 - TDA_CWT) * p_svm

        tmp = zone_df.copy()
        tmp["p_tda"] = p_tda_zone
        per_arm[arm] = tmp.groupby("image_id")["p_tda"].max()

    print("\n[4/5] VinDr ResNet18 çıkarımı...")
    res_zone_df, res_patches = res_patient_zone
    res_idx = np.arange(len(res_patches))
    p_res_zone = predict_resnet(model_res, res_patches, res_idx, device)
    tmp = res_zone_df.copy()
    tmp["p_res"] = p_res_zone
    p_res_patient = tmp.groupby("image_id")["p_res"].max().reindex(cohort["image_id"]).values

    print("\n[5/5] Değerlendirme...")
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    reports = {}
    print("\n" + "=" * 78)
    for arm in args.arms:
        p_tda = per_arm[arm].reindex(cohort["image_id"]).values
        rep = evaluate_arm(arm, p_tda, p_res_patient, y, cohort)
        reports[arm] = rep

        pred = cohort[["image_id", "patient_label", "group", "rel_size"]].copy()
        pred["p_tda"] = p_tda
        pred["p_resnet"] = p_res_patient
        pred["p_fusion_a045"] = E0_6_FUSION_A * p_res_patient + (1 - E0_6_FUSION_A) * p_tda
        pred.to_csv(run_dir / f"vindr_predictions_{arm}.csv", index=False)
        (run_dir / f"vindr_report_{arm}.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Şeffaf, küçük raporlar (git'te tutulabilir)
        (PROJECT_ROOT / f"data/external_vindr/vindr_report_{arm}.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"KOL: {arm}")
        print(f"  TDA ensemble  : {rep['auc']['tda']:.4f} "
              f"[{rep['ci95']['tda'][0]:.4f}, {rep['ci95']['tda'][1]:.4f}]")
        print(f"  ResNet18      : {rep['auc']['resnet']:.4f} "
              f"[{rep['ci95']['resnet'][0]:.4f}, {rep['ci95']['resnet'][1]:.4f}]")
        print(f"  Füzyon (a={rep['a_locked']}) : {rep['auc']['fusion']:.4f} "
              f"[{rep['ci95']['fusion'][0]:.4f}, {rep['ci95']['fusion'][1]:.4f}] | "
              f"Δ(resnet)={rep['fusion_delta_over_resnet']:+.4f}")
        print(f"  Füzyonun VinDr'de en iyi a'sı: {rep['a_best_in_vindr']['a']:.2f} "
              f"→ {rep['a_best_in_vindr']['auc']:.4f}")
        print("-" * 78)

    summary = {
        "experiment": "V4 — VinDr-CXR external validation (zero-shot)",
        "cohort": {
            "n_images": int(len(cohort)),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "level": "image (VinDr metadata has no patient identifier)",
            "source": "VinDr-CXR, 3x downsampled JPEG, no DICOM spacing",
            "sample_seed": SEED,
        },
        "protocol": {
            "training_on_vindr": False,
            "tda_branch": "frozen NODE21 bundles (SmallZoneCNN + SVM, w=0.55)",
            "resnet_branch": ("E0.6 recipe re-trained deterministically on NODE21. "
                              "E0.6 weights were never saved, so bit-identity is impossible; "
                              "the fixed checkpoint produced here is used for VinDr inference."),
            "resnet_val_zone_auc": val_auc,
            "resnet_val_auc_delta_vs_e0_6": abs(val_auc - E0_6_VAL_ZONE_AUC),
            "resnet_val_consistent_with_e0_6": reproduced,
            "e0_6_reference_val_zone_auc": E0_6_VAL_ZONE_AUC,
            "determinism": determinism,
            "fusion_definition": "a*resnet + (1-a)*tda",
            "fusion_a_locked": E0_6_FUSION_A,
            "fusion_a_source": "E0.6 (artifacts/runs/E0.6_decisive_test/test_report.json)",
            "intensity_scale": "linear: uint8*16 (≈NODE21 12-bit); matched: pooled 200-patient reference",
            "scale_normalization": "matched_scalenorm: crop resized by target_lung_h / patient_lung_h",
            "node21_median_lung_height_px": target_lh,
            "subgroup_definition": "tertiles of rel_size = bbox_max_dim_px / image_h (no mm)",
        },
        "reports": reports,
        "artifacts": {
            "predictions": [f"vindr_predictions_{a}.csv" for a in args.arms],
            "reports": [f"vindr_report_{a}.json" for a in args.arms],
        },
    }
    (run_dir / "vindr_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Eski (silinen) kopya betiğin dosya adlarıyla geriye dönük uyumluluk:
    (PROJECT_ROOT / "data/external_vindr/vindr_summary_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_manifest(
        run_dir,
        EXPERIMENT_ID,
        config={
            "arms": args.arms,
            "seed": SEED,
            "tda_settings": {
                "grid_size": TDA_SETTINGS.grid_size,
                "intensity_max": TDA_SETTINGS.intensity_max,
                "h0_birth_range": TDA_SETTINGS.h0_birth_range,
                "h0_persistence_range": TDA_SETTINGS.h0_persistence_range,
                "h1_birth_range": TDA_SETTINGS.h1_birth_range,
                "h1_persistence_range": TDA_SETTINGS.h1_persistence_range,
                "pi_sigma_pixels": TDA_SETTINGS.pi_sigma_pixels,
            },
            "tda_cwt": TDA_CWT,
            "fusion_a": E0_6_FUSION_A,
            "determinism": determinism,
            "resnet": {"epochs": EPOCHS, "lr": LR, "wd": WD, "dropout": DROPOUT,
                       "batch": BATCH, "val_zone_auc": val_auc,
                       "delta_vs_e0_6": abs(val_auc - E0_6_VAL_ZONE_AUC)},
        },
        inputs=[COHORT_CSV, REF_NPZ, SPLITS_PATH, RESNET_CACHE],
        seed=SEED,
        metrics={
            **{f"{arm}_tda_auc": reports[arm]["auc"]["tda"] for arm in args.arms},
            **{f"{arm}_resnet_auc": reports[arm]["auc"]["resnet"] for arm in args.arms},
            **{f"{arm}_fusion_auc": reports[arm]["auc"]["fusion"] for arm in args.arms},
        },
        decision="EXTERNAL",
        notes="VinDr-CXR zero-shot; kararlar yalnızca dış veri üzerinde tanımlayıcıdır.",
    )
    print(f"\nRaporlar: {run_dir}")
    print(f"Özet    : {PROJECT_ROOT / 'data/external_vindr/vindr_summary_report.json'}")


if __name__ == "__main__":
    main()
