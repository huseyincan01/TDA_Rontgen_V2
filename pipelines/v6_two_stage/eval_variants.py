"""V6.0 — Aday birleştirme varyantları (val, eval-only): TTA + RRF füzyonu + kota.

Amaç: K=20/30'da %98 recall kapısına ulaşmak. Yeniden eğitim YOK.
Varyantlar:
  M5            : tek model
  M5+TTA        : M5 + yatay çevrim (RRF füzyonu)
  M5+M6         : iki eğitimli model (RRF)
  M5+M6+TTA     : üçü (RRF)
  M5+M6+M2      : eğitimli + geometrik (RRF)
  M7a_quota     : mevcut ortogonal kota (M5 %70 + M2 %30)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import candidates_classical as cc  # noqa: E402
from detector_models import CenterNetHead, FCOSHead, STRIDE, INPUT  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
KS = (10, 20, 30, 50)


def _predict_flip(model, canon, device, kind):
    """Yatay çevrilmiş girdiyle tahmin; aday x koordinatı geri çevrilir."""
    model.eval()
    x = p2._to_input(canon, True).unsqueeze(0).to(device)
    with torch.no_grad():
        if kind == "M5":
            h, _ = model(x)
            heat = torch.sigmoid(h)[0, 0].cpu().numpy()
            maxf = ndi.maximum_filter(heat, size=3)
            ys, xs = np.where((heat >= maxf) & (heat > 0.01))
            scores = heat[ys, xs]
            cands = [(float(xs[k] * STRIDE + STRIDE / 2) / 0.5,
                      float(ys[k] * STRIDE + STRIDE / 2) / 0.5,
                      float(scores[k])) for k in np.argsort(-scores)]
        else:
            o, b = model(x)
            obj = torch.sigmoid(o)[0, 0].cpu().numpy()
            ys, xs = np.where(obj > 0.02)
            cands = [(float(xs[k] * STRIDE + STRIDE / 2) / 0.5,
                      float(ys[k] * STRIDE + STRIDE / 2) / 0.5,
                      float(obj[ys[k], xs[k]])) for k in np.argsort(-obj[ys, xs])]
    # geri çevir (1024 uzayı)
    cands = [(1024.0 - cx, cy, s) for cx, cy, s in cands]
    return cc._nms(cands, min_dist=20, top=1000)


def _rrf(lists, k=60, top=1000):
    """Reciprocal Rank Fusion: skor kalibrasyonu gerektirmez."""
    acc: dict = {}
    for lst in lists:
        for r, (cx, cy, _) in enumerate(lst[:top], start=1):
            key = (int(round(cx / 8)), int(round(cy / 8)))   # 8px kaba anahtar
            if key not in acc:
                acc[key] = [0.0, cx, cy]
            acc[key][0] += 1.0 / (k + r)
    items = [(v[1], v[2], v[0]) for v in acc.values()]
    items.sort(key=lambda c: -c[2])
    return cc._nms(items, min_dist=20, top=top)


def _predict_heat_tta(model, canon, device, kind):
    """Isı haritası düzeyinde TTA: x ve çevrilmiş x'in çıktı haritaları ortalanır."""
    model.eval()
    x = p2._to_input(canon, False).unsqueeze(0).to(device)
    xf = torch.flip(x, dims=[3])
    with torch.no_grad():
        o1 = model(x)
        o2 = model(xf)
    if kind == "M5":
        h1 = torch.sigmoid(o1[0])[0, 0]
        h2 = torch.flip(torch.sigmoid(o2[0])[0, 0], dims=[1])
        heat = ((h1 + h2) / 2).cpu().numpy()
    else:
        h1 = torch.sigmoid(o1[0])[0, 0]
        h2 = torch.flip(torch.sigmoid(o2[0])[0, 0], dims=[1])
        heat = ((h1 + h2) / 2).cpu().numpy()
    maxf = ndi.maximum_filter(heat, size=3)
    thr = 0.01 if kind == "M5" else 0.02
    ys, xs = np.where((heat >= maxf) & (heat > thr))
    scores = heat[ys, xs]
    order = np.argsort(-scores)
    cands = [(float(xs[k] * STRIDE + STRIDE / 2) / 0.5,
              float(ys[k] * STRIDE + STRIDE / 2) / 0.5, float(scores[k])) for k in order]
    return cc._nms(cands, min_dist=20, top=1000)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()

    pids = p2._nodule_pids("val")
    print(f"Varyant taraması | val {len(pids)} nodül hastası | device={device}")

    variants = ["M5", "M5_heatTTA", "M5_heatTTA+M6", "M5_heatTTA+M6hTTA", "M5+M6", "M5+M6+TTA"]
    agg = {v: {str(k): 0 for k in KS} for v in variants}
    n_les = 0
    t0 = time.time()

    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        mask = common.load_lung_mask(pid)
        gts = common.load_gt(pid)
        n_les += len(gts)

        c5 = p2.predict(m5, canon, device, "M5")
        c5h = _predict_heat_tta(m5, canon, device, "M5")
        c5f = _predict_flip(m5, canon, device, "M5")
        c6 = p2.predict(m6, canon, device, "M6")
        c6h = _predict_heat_tta(m6, canon, device, "M6")

        cand = {
            "M5": c5,
            "M5_heatTTA": c5h,
            "M5_heatTTA+M6": _rrf([c5h, c6]),
            "M5_heatTTA+M6hTTA": _rrf([c5h, c6h]),
            "M5+M6": _rrf([c5, c6]),
            "M5+M6+TTA": _rrf([c5, c5f, c6]),
        }
        for v, c in cand.items():
            for k in KS:
                agg[v][str(k)] += int(round(common.recall_at_K(c, gts, k, common.hit_criterion1) * len(gts)))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pids)}", flush=True)

    print(f"\n=== VAL (n_lesions={n_les}, {time.time() - t0:.0f} s) ===")
    print(f"  {'varyant':14s} " + "  ".join(f"K{k}" for k in KS))
    best = None
    for v in variants:
        r = {k: agg[v][str(k)] / max(1, n_les) for k in KS}
        print(f"  {v:14s} " + "  ".join(f"{r[k]:.4f}" for k in KS))
        if best is None or r[30] > best[1][30]:
            best = (v, r)

    out = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_0_variant_sweep.json"
    out.write_text(json.dumps({
        "n_lesions": n_les, "n_patients": len(pids),
        "variants": {v: {k: agg[v][str(k)] / max(1, n_les) for k in KS} for v in variants},
        "best_at_k30": best[0],
        "gate_98": {k: bool(agg[best[0]][str(k)] / max(1, n_les) >= 0.98) for k in (20, 30)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEn iyi (K30): {best[0]} | K20={best[1][20]:.4f} K30={best[1][30]:.4f}")
    print(f"Rapor: {out}")


if __name__ == "__main__":
    main()
