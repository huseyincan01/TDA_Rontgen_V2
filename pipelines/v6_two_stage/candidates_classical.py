"""V6.0 — AİLE 1: Eğitimsiz matematiksel/radyolojik aday üreticiler (M1–M4).

Her `generate_*` fonksiyonu (cx, cy, score) listesi döndürür (skor azalan sıralı).
Girdi: `canon` (c_lungref kanonik, [0,255], akciğer dışı 0) ve `mask` (dilate akciğer).
Hepsi RAM'de çalışır; diske hiçbir şey yazmaz.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _local_maxima(resp: np.ndarray, mask: np.ndarray, min_sep: int = 13, top: int = 1500):
    """mask içinde, pozitif yanıtın yerel maksimumlarını bulur (plateau-güvenli + NMS)."""
    resp = np.where(mask, resp, -np.inf)
    maxf = ndi.maximum_filter(resp, size=min_sep)
    lm = (resp >= maxf) & (resp > 1e-6)
    ys, xs = np.where(lm)
    scores = resp[lm]
    order = np.argsort(-scores)
    kept: list = []
    kept_pts: list = []
    for k in order:
        px, py = int(xs[k]), int(ys[k])
        if any((px - a) ** 2 + (py - b) ** 2 < min_sep ** 2 for a, b in kept_pts):
            continue
        kept.append((float(px), float(py), float(scores[k])))
        kept_pts.append((px, py))
        if len(kept) >= top:
            break
    return kept


def _nms(cands: list, min_dist: float = 20.0, top: int = 1000):
    """Izgara tabanlı hızlı NMS (O(n)): skor azalan sıralı varsayılır."""
    grid: dict = {}
    out = []
    for cx, cy, s in cands:
        gx, gy = int(cx // min_dist), int(cy // min_dist)
        if any((gx + dx, gy + dy) in grid for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        grid[(gx, gy)] = True
        out.append((cx, cy, s))
        if len(out) >= top:
            break
    return out


# ------------------------------------------------------------------ M1: LoG + CLAHE
def generate_m1(canon: np.ndarray, mask: np.ndarray) -> list:
    from skimage.exposure import equalize_adapthist
    from scipy.ndimage import gaussian_laplace

    clahe = equalize_adapthist(np.clip(canon, 0, 255).astype(np.uint8), clip_limit=0.02).astype(np.float32)
    cands = []
    for sigma in (3, 6, 10, 15, 22):
        # parlak nodül → -LoG tepe; sigma^2 ölçek normalizasyonu
        resp = -(sigma ** 2) * gaussian_laplace(clahe, sigma)
        cands.extend(_local_maxima(resp, mask, min_sep=int(2 * sigma + 1)))
    cands.sort(key=lambda c: -c[2])
    return _nms(cands, min_dist=20.0)


# ------------------------------------------------------------------ M2: DoH + damar bastırma
def generate_m2(canon: np.ndarray, mask: np.ndarray) -> list:
    from scipy.ndimage import gaussian_filter

    # Determinant of Hessian (scipy tabanlı, versiyon-bağımsız)
    sm = gaussian_filter(canon, 2.0)
    Hxx = gaussian_filter(sm, 3.0, order=[2, 0])
    Hyy = gaussian_filter(sm, 3.0, order=[0, 2])
    Hxy = gaussian_filter(sm, 3.0, order=[1, 1])
    detH = Hxx * Hyy - Hxy * Hxy
    tr = Hxx + Hyy
    disc = np.sqrt(np.maximum((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy, 0.0))
    l1 = (tr + disc) / 2.0
    l2 = (tr - disc) / 2.0
    a, b = np.abs(l1), np.abs(l2)
    # blobness oranı: küre/kubbe → 1, uzun damar sırtı → 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(np.maximum(a, b) > 1e-9, np.minimum(a, b) / np.maximum(a, b), 0.0)
    score = np.clip(detH, 0, None) * np.clip(rb, 0, 1)
    cands = _local_maxima(score, mask, min_sep=13)
    cands.sort(key=lambda c: -c[2])
    return _nms(cands, min_dist=20.0)


# ------------------------------------------------------------------ M3: radyal simetri (gradyan yakınsaması)
def generate_m3(canon: np.ndarray, mask: np.ndarray) -> list:
    from scipy.ndimage import gaussian_filter, sobel

    gx = sobel(canon, axis=1)
    gy = sobel(canon, axis=0)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag[mask], 82)
    edge = (mag > thresh) & mask
    ux = gx / (mag + 1e-6)
    uy = gy / (mag + 1e-6)
    ys, xs = np.where(edge)
    me = mag[edge]
    H, W = canon.shape
    acc = np.zeros((H, W), dtype=np.float32)
    for r in (4, 7, 11, 16):
        cx = np.clip((xs + ux[edge] * r).astype(int), 0, W - 1)
        cy = np.clip((ys + uy[edge] * r).astype(int), 0, H - 1)
        np.add.at(acc, (cy, cx), me)
    acc = gaussian_filter(acc, 3.0)
    cands = _local_maxima(acc, mask, min_sep=13)
    cands.sort(key=lambda c: -c[2])
    return _nms(cands, min_dist=20.0)


# ------------------------------------------------------------------ M4: bilateral asimetri + LoG
def generate_m4(canon: np.ndarray, mask: np.ndarray) -> list:
    from scipy.ndimage import gaussian_laplace

    mirrored = canon[:, ::-1]
    asym = np.abs(canon - mirrored) * mask
    cands = []
    for sigma in (4, 7, 11, 16):
        resp = -(sigma ** 2) * gaussian_laplace(asym, sigma)
        cands.extend(_local_maxima(resp, mask, min_sep=int(2 * sigma + 1)))
    cands.sort(key=lambda c: -c[2])
    return _nms(cands, min_dist=20.0)


GENERATORS = {
    "M1_log_clahe": generate_m1,
    "M2_doh_vesselsuppress": generate_m2,
    "M3_radial_symmetry": generate_m3,
    "M4_bilateral_asymmetry": generate_m4,
}
