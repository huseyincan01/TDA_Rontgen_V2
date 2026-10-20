"""V6.0 — 1. Aşama Aday Üretici Benchmark: ortak altyapı (RAM'de, disk yazımı YOK).

Tüm görüntü/maske işlemleri on-the-fly (RAM) yapılır; diske yalnızca JSON rapor yazılır.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_IMAGES = PROJECT_ROOT / "dataset_node21" / "cxr_images" / "proccessed_data" / "images"
PATIENTS_DIR = PROJECT_ROOT / "data" / "processed" / "zones" / "patients"
PATIENT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv"
LUNG_P1P99 = PROJECT_ROOT / "data" / "processed" / "features_v5_1" / "lung_p1p99.json"

import os

DILATION_PX = int(os.environ.get("V6_DILATION", "15"))   # V6.0 varsayılanı; V6.0b için V6_DILATION=25
BOX_HALF = 32          # 64x64 aday kutusu yarı kenarı
SIGMAS = [3, 6, 10, 15, 22]

_lung_p1p99_cache: dict | None = None


def get_lung_p1p99() -> dict:
    global _lung_p1p99_cache
    if _lung_p1p99_cache is None:
        _lung_p1p99_cache = json.loads(LUNG_P1P99.read_text(encoding="utf-8"))
    return _lung_p1p99_cache


def load_image(pid: str) -> np.ndarray:
    """1024x1024 uint16 (0-4095) tam görüntü → float32."""
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC_IMAGES / f"{pid}.mha")))
    if img.ndim == 3:
        img = img[0]
    return img.astype(np.float32)


def load_lung_mask(pid: str) -> np.ndarray:
    """mask_5zones.mha → akciğer (zone>0) maskesi, 15 px morfolojik dilasyon."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(PATIENTS_DIR / pid / "mask_5zones.mha")))
    if m.ndim == 3:
        m = m[0]
    lung = m > 0
    return ndi.binary_dilation(lung, structure=ndi.generate_binary_structure(2, 1),
                               iterations=DILATION_PX)


def canonicalize(img: np.ndarray, mask: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """c_lungref (tüm-akciğer p1-p99) kanonikleştirme → [0,255], akciğer dışı 0."""
    if p99 <= p1:
        p99 = p1 + 1.0
    out = np.zeros_like(img)
    out[mask] = np.clip((img[mask] - p1) * 255.0 / (p99 - p1), 0, 255)
    return out


def load_gt(pid: str) -> list[dict]:
    info = json.loads((PATIENTS_DIR / pid / "patient_info.json").read_text(encoding="utf-8"))
    spacing = info.get("spacing", [1.0, 1.0])
    sx = float(spacing[0])
    gts = []
    for nd in info.get("nodules", []):
        w, h = float(nd["bbox_w"]), float(nd["bbox_h"])
        gts.append({
            "x": float(nd["bbox_x"]), "y": float(nd["bbox_y"]),
            "w": w, "h": h,
            "cx": float(nd["center_x"]), "cy": float(nd["center_y"]),
            "size_mm": max(w, h) * sx,       # px * mm/px
        })
    return gts


def _bbox_radius(gt: dict) -> float:
    return float(np.hypot(gt["w"], gt["h"]) / 2.0)


def hit_criterion1(cx: float, cy: float, gt: dict) -> bool:
    """Kriter 1 (FROC): merkez bbox içinde VEYA merkeze uzaklık <= max(16, 0.5*R_bbox)."""
    inside = (gt["x"] <= cx <= gt["x"] + gt["w"]) and (gt["y"] <= cy <= gt["y"] + gt["h"])
    d = float(np.hypot(cx - gt["cx"], cy - gt["cy"]))
    return bool(inside or d <= max(16.0, 0.5 * _bbox_radius(gt)))


def hit_criterion2(cx: float, cy: float, gt: dict) -> bool:
    """Kriter 2 (kapsama): 64x64 kutu nodül alanının >=%50'sini (IoBB) VEYA IoU >= 0.15."""
    x0, x1 = cx - BOX_HALF, cx + BOX_HALF
    y0, y1 = cy - BOX_HALF, cy + BOX_HALF
    gx0, gx1 = gt["x"], gt["x"] + gt["w"]
    gy0, gy1 = gt["y"], gt["y"] + gt["h"]
    ix0, ix1 = max(x0, gx0), min(x1, gx1)
    iy0, iy1 = max(y0, gy0), min(y1, gy1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    gt_area = gt["w"] * gt["h"]
    box_area = (x1 - x0) * (y1 - y0)
    union = gt_area + box_area - inter
    iobb = inter / gt_area if gt_area > 0 else 0.0
    iou = inter / union if union > 0 else 0.0
    return bool(iobb >= 0.50 or iou >= 0.15)


HIT_FNS = {"kriter1_froc": hit_criterion1, "kriter2_containment": hit_criterion2}


def subgroup_of(size_mm: float) -> str:
    if size_mm < 15.0:
        return "kucuk"
    if size_mm <= 30.0:
        return "orta"
    return "buyuk"


def recall_at_K(cands: list[tuple], gts: list[dict], K: int, hit_fn) -> float:
    """cands: [(cx, cy, score)] skor azalan sıralı. GT'lerden kaçı Top-K'da yakalanıyor."""
    if not gts:
        return 1.0
    top = cands[:K]
    hits = 0
    for gt in gts:
        if any(hit_fn(cx, cy, gt) for cx, cy, _ in top):
            hits += 1
    return hits / len(gts)


def split_patients() -> dict[str, list[str]]:
    import pandas as pd

    df = pd.read_csv(PATIENT_SPLITS)
    return {s: df.loc[df["split"] == s, "patient_id"].astype(str).tolist()
            for s in ("train", "val", "test")}
