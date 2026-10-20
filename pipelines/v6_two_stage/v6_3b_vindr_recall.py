"""V6.3b — VinDr-CXR bbox recall@K30/K50 (3x indirgeme doğrulandı: fit_ratio=1.0).

Annotasyon bbox'ları (orijinal uzay) /3 ile JPEG uzayına taşınır; Kriter-1 ile
görüntü başına "en az bir nodül yakalandı mı" recall'u ölçülür.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402
from detector_models import CenterNetHead, FCOSHead  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_3_vindr_recall.json"


def _gt_from_ann(ann_img: pd.DataFrame, scale=3.0):
    gts = []
    for _, b in ann_img.iterrows():
        x0, x1 = float(b["x_min"]) / scale, float(b["x_max"]) / scale
        y0, y1 = float(b["y_min"]) / scale, float(b["y_max"]) / scale
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        gts.append({"x": x0, "y": y0, "w": w, "h": h, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                    "size_mm": max(w, h)})
    return gts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()
    seg = PSPNet5ZoneSegmenter(device=device)

    import torchxrayvision as xrv
    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(PROJECT_ROOT / "data/external_vindr/cohort_with_diameters.csv")
    pos = cohort[cohort["patient_label"] == 1]

    stats = {v: {str(K): [0, 0] for K in (10, 20, 30, 50)} for v in ("M5_heatTTA", "M5M6TTA")}
    t0 = time.time()
    for i, (_, row) in enumerate(pos.iterrows()):
        pid = str(row["image_id"])
        img = np.array(Image.open(PROJECT_ROOT / "data/external_vindr/images" / f"{pid}.jpg"))
        gts = _gt_from_ann(nod[nod["image_id"] == pid])
        if not gts:
            continue
        try:
            zmap = seg.segment(img)["zone_map"]
            raw = zmap > 0
            ys, xs = np.where(raw)
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                                       iterations=common.DILATION_PX)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            H, W = canon.shape
            sx, sy = W / 1024.0, H / 1024.0
            c5 = [(x * sx, y2 * sy, s) for x, y2, s in ev._predict_heat_tta(m5, canon, device, "M5")]
            c6 = [(x * sx, y2 * sy, s) for x, y2, s in p2.predict(m6, canon, device, "M6")]
            cand5 = c5[:50]
            candm = ev._rrf([c5, c6])[:50]
        except Exception:
            continue
        for vname, cands in (("M5_heatTTA", cand5), ("M5M6TTA", candm)):
            for K in (10, 20, 30, 50):
                hit = any(common.hit_criterion1(cx, cy, g) for g in gts for cx, cy, _ in cands[:K])
                stats[vname][str(K)][0] += int(hit)
                stats[vname][str(K)][1] += 1
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(pos)} ({time.time() - t0:.0f}s)", flush=True)

    out = {"n_positive_with_bbox": None, "assumption": "original = 3x JPEG (fit_ratio=1.0)",
           "metric": "görüntü başına >=1 nodül yakalandı (Kriter-1)", "recall": {}}
    for v, ks in stats.items():
        out["recall"][v] = {k: (round(a / b, 4) if b else None) for k, (a, b) in ks.items()}
        out["n_positive_with_bbox"] = max(vals[1] for vals in ks.values())
    (PROJECT_ROOT / "artifacts" / "v6_two_stage").mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Rapor: {OUT}")


if __name__ == "__main__":
    main()
