"""V6.4b — VinDr: AdaBN'li M5 (M6 YOK) + lokal TDA-HGB (e_fusion_tda) + V5.3 birleşimi.

Amaç: AdaBN M6'yı bozduğu için, yalnız M5 (AdaBN) + TDA dalının VinDr AUC'sini ve
V5.3 global ile birleşik AUC'yi ölçmek.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
from detector_models import CenterNetHead, STRIDE  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
EXT = PROJECT_ROOT / "data/external_vindr"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_4b_vindr_adabn_tda.json"
A45 = 0.45
INPUT = 512


def rank_norm(x):
    return (rankdata(np.asarray(x, float)) / len(x)).astype(np.float32)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    scaler, hgb, lr, ranges = b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"]
    ex = ZoneTDAExtractor(s2._settings(ranges))

    import torchxrayvision as xrv
    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)

    # ---- Geçiş 1: kanonik + mask + 512 girdi (RAM)
    print("[1] VinDr hazırlık (kanonik + mask + 512)...")
    data = []
    t0 = time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"]); lab = int(row["patient_label"])
        img = np.array(Image.open(EXT / "images" / f"{pid}.jpg"))
        gts = []
        for _, bb in nod[nod["image_id"] == pid].iterrows():
            x0, x1 = bb["x_min"] / 3.0, bb["x_max"] / 3.0
            y0, y1 = bb["y_min"] / 3.0, bb["y_max"] / 3.0
            if x1 > x0 and y1 > y0:
                gts.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
        try:
            raw = seg.segment(img)["zone_map"] > 0
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                                       iterations=common.DILATION_PX)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            canon_u8 = np.clip(np.rint(canon), 0, 255).astype(np.uint8)
            x512 = np.array(Image.fromarray(canon_u8).resize((INPUT, INPUT), Image.Resampling.BILINEAR))
        except Exception:
            raw = np.zeros(img.shape, bool); canon_u8 = np.zeros(img.shape, np.uint8)
            x512 = np.zeros((INPUT, INPUT), np.uint8)
        data.append({"pid": pid, "y": lab, "gts": gts, "canon": canon_u8,
                     "raw": raw.astype(np.uint8), "x512": x512})
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- AdaBN (yalnız M5)
    print("[2] AdaBN (M5) — VinDr etiketsiz yığın...")
    tensors = [m64._norm512(d["x512"]) for d in data]
    m5a = m64.adabn(m5, tensors, device)

    # ---- Geçiş 2: adaylar + lokal TDA + e_fusion
    print("[3] AdaBN'li M5 adayları + lokal TDA-HGB...")
    t0 = time.time()
    y, s1_img, tda_img, fus_img = [], [], [], []
    for i, d in enumerate(data):
        H, W = d["canon"].shape
        try:
            x = d["x512"]
            with torch.no_grad():
                h, _ = m5a(m64._norm512(x).unsqueeze(0).to(device))
            heat = torch.sigmoid(h)[0, 0].cpu().numpy()
            maxf = ndi.maximum_filter(heat, size=3)
            yy, xx = np.where((heat >= maxf) & (heat > 0.01))
            sc = heat[yy, xx]
            order = np.argsort(-sc)
            cands = [(float((xx[k] * STRIDE + STRIDE / 2) * W / INPUT),
                      float((yy[k] * STRIDE + STRIDE / 2) * H / INPUT), float(sc[k])) for k in order][:50]
            raw = d["raw"] > 0
            ys, xs = np.where(raw)
            bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
            dist = ndi.distance_transform_edt(raw).astype(np.float32)
            canon_f = d["canon"].astype(np.float32)
            tda = np.zeros((len(cands), 2 * 2 * s2.GRID * s2.GRID), np.float32)
            spa = np.zeros((len(cands), 4), np.float32)
            for k, (cx, cy, _) in enumerate(cands):
                tda[k] = np.concatenate([ex.extract(s2._crop(canon_f, cx, cy, s2.INNER))[0],
                                         ex.extract(s2._crop(canon_f, cx, cy, s2.OUTER))[0]])
                spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
            X = scaler.transform(np.concatenate([tda, spa], 1))
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda = hgb.predict_proba(X)[:, 1]
            p_fus = lr.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
        except Exception:
            y.append(d["y"]); s1_img.append(0.0); tda_img.append(0.0); fus_img.append(0.0); continue
        y.append(d["y"])
        s1_img.append(float(1 - np.prod(1 - np.clip(np.sort(s1)[::-1][:5], 0, 1))))
        tda_img.append(float(1 - np.prod(1 - np.clip(np.sort(p_tda)[::-1][:5], 0, 1))))
        fus_img.append(float(1 - np.prod(1 - np.clip(np.sort(p_fus)[::-1][:5], 0, 1))))
        if (i + 1) % 100 == 0:
            print(f"   {i + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)

    y = np.array(y); s1_img = np.array(s1_img); tda_img = np.array(tda_img); fus_img = np.array(fus_img)
    out = {"n_images": len(y), "n_positive": int(y.sum()),
           "adabn_m5_stage1_auc": round(float(roc_auc_score(y, s1_img)), 4),
           "adabn_m5_tda_hgb_auc": round(float(roc_auc_score(y, tda_img)), 4),
           "adabn_m5_e_fusion_tda_auc": round(float(roc_auc_score(y, fus_img)), 4)}

    # V5.3 global + birleşim
    v53 = pd.read_csv(PROJECT_ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv")
    v53 = v53.set_index("image_id").reindex(cohort["image_id"])
    g = (A45 * v53["p_resnet"] + (1 - A45) * v53["p_tda"]).values
    g = np.nan_to_num(g, nan=0.0)
    out["v5_3_global_auc"] = round(float(roc_auc_score(y, g)), 4)
    comb = 0.5 * rank_norm(g) + 0.5 * rank_norm(fus_img)
    out["combined_v53_adabn_m5_tda_auc"] = round(float(roc_auc_score(y, comb)), 4)
    out["r_v53_vs_v6"] = round(float(np.corrcoef(g, fus_img)[0, 1]), 4)
    out["note"] = ("M6 kullanılmadı (AdaBN M6'yı bozuyordu). Aşama-2 = donmuş "
                   "stage2_final_tda_hgb (TDA-HGB + LR e_fusion).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
