"""V6.5b — Few-shot kalibrasyonun iki zayıflığını kapatma.

(1) Kompakt özellik: 1604-boyutlu LR yerine NODE21 referansında fit edilmiş
    PCA-128 uzayında kalibrasyon (hedef etiketi kullanılmaz → sızıntı yok).
(2) Bbox-türevli aday etiketleri: kalibrasyon LR'si görüntü etiketi yerine
    GT kutusuna isabet eden adaylar üzerinden eğitilir → FROC anlamlı olur.

Protokol V6.5 ile aynı: 5-fold stratified CV, N ∈ {0,10,20,40,80,160},
Aşama-1 donmuş, skor dondurulmuş lr_fusion üzerinden.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import candidates_classical as cc  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as m62  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
import v6_5_vindr_transfer_boost as v65  # noqa: E402
from detector_models import CenterNetHead  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
EXT = ROOT / "data/external_vindr"
OUT = ROOT / "artifacts" / "v6_two_stage" / "v6_5b_fewshot_compact.json"
INPUT = 512
A45 = 0.45
SEED = 42
N_FEW = (0, 10, 20, 40, 80, 160)
NW = max(2, min(10, (os.cpu_count() or 4) - 2))
KEY = ("scale", 1.0)  # KOL1'in en iyi konfigürasyonu (denoise 1.0 + ölçek)


def _extract_one(chunk, ranges):
    ex = ZoneTDAExtractor(s2._settings(ranges))
    out = []
    for d in chunk:
        c = d["cands"]
        raw = d["raw"] > 0
        ys, xs = np.where(raw)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, 0, 0, 0)
        dist = ndi.distance_transform_edt(raw).astype(np.float32) if ys.size else np.zeros(raw.shape, np.float32)
        canon = ndi.gaussian_filter(d["canon"].astype(np.float32), KEY[1])
        s1_, s2_ = v65._sizes(KEY[0], d["lung_h"])
        tda = np.zeros((len(c), 1600), np.float32)
        spa = np.zeros((len(c), 4), np.float32)
        for k, (cx, cy, _) in enumerate(c):
            tda[k] = np.concatenate([ex.extract(v65._crop(canon, cx, cy, s1_))[0],
                                     ex.extract(v65._crop(canon, cx, cy, s2_))[0]])
            if ys.size:
                spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
        out.append({"tda": tda, "spa": spa})
    return [d["pid"] for d in chunk], out


def _w(args):
    return _extract_one(*args)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    hgb, lr, ranges = b["hgb"], b["lr_fusion"], b["ranges"]

    import torchxrayvision as xrv

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    print("[0] VinDr hazırlık...", flush=True)
    data, t0 = [], time.time()
    for i, row in cohort.iterrows():
        pid, lab = str(row["image_id"]), int(row["patient_label"])
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
            ys = np.where(raw.any(axis=1))[0]
            lung_h = int(ys.max() - ys.min() + 1) if ys.size else 570
        except Exception:
            raw = np.zeros(img.shape, bool)
            canon_u8 = np.zeros(img.shape, np.uint8)
            x512 = np.zeros((INPUT, INPUT), np.uint8)
            lung_h = 570
        data.append({"pid": pid, "y": lab, "gts": gts, "canon": canon_u8,
                     "raw": raw.astype(np.uint8), "x512": x512, "lung_h": lung_h})
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)

    print("[1] AdaBN + adaylar...", flush=True)
    m5a = m64.adabn(m5, [m64._norm512(d["x512"]) for d in data], device)
    cands = {}
    for d in data:
        try:
            cands[d["pid"]] = cc._nms(m64._pred_from_512(m5a, d["x512"], d["canon"].shape, device, "M5"), 20, 50)
        except Exception:
            cands[d["pid"]] = []

    print("[2] Referans + kompakt özellikler...", flush=True)
    from concurrent.futures import ProcessPoolExecutor

    tr = list(np.random.RandomState(SEED).choice(p2._nodule_pids("train"), 100, replace=False)) + \
        list(np.random.RandomState(SEED + 1).choice(p2._control_pids("train"), 100, replace=False))
    c_tr = s2.stage1_candidates(tr, m5, m64.load_models(device)[1], device, top=20)
    ref_jobs = [(list(p), {q: c_tr[q]["M5_heatTTA"] for q in p}, ranges) for p in np.array_split(np.array(tr), NW)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as ex:
        ref = list(ex.map(v65._w_ref, ref_jobs))
    X_ref = np.stack([np.concatenate([t, s]) for r in ref for t, s in zip(r[0], r[1])]).astype(np.float32)
    print(f"    referans: {len(X_ref)} ({time.time() - t0:.0f}s)", flush=True)

    chunks = []
    for part in np.array_split(np.arange(len(data)), NW):
        chunks.append(([{"pid": data[i]["pid"], "canon": data[i]["canon"], "raw": data[i]["raw"],
                         "lung_h": data[i]["lung_h"], "cands": cands[data[i]["pid"]]} for i in part], ranges))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as ex:
        res = list(ex.map(_w, chunks))
    feats = {}
    for pids, out in res:
        feats.update(dict(zip(pids, out)))
    print(f"    özellikler: {time.time() - t0:.0f}s", flush=True)

    aligner = v65.Aligner(X_ref)
    blocks, counts = [], []
    for d in data:
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([feats[d["pid"]]["tda"], feats[d["pid"]]["spa"]], 1))
    X = aligner.coral(np.concatenate(blocks))
    Xref_al = aligner.coral(X_ref)
    pca = PCA(n_components=128, random_state=SEED).fit(Xref_al)
    Xp = pca.transform(X).astype(np.float32)
    print(f"    PCA-128 açıklanan varyans: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    hits = {d["pid"]: s2._labels_for(cands[d["pid"]], d["gts"]) for d in data}
    y_img = np.array([d["y"] for d in data])
    Xall, Xpall, s1all, k = [], [], [], 0
    for d, n in zip(data, counts):
        Xall.append(X[k:k + n])
        Xpall.append(Xp[k:k + n])
        s1all.append(np.array([c[2] for c in cands[d["pid"]]], np.float32))
        k += n

    v53 = pd.read_csv(ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv").set_index("image_id")
    v53 = v53.reindex([d["pid"] for d in data])
    g53 = np.nan_to_num((A45 * v53["p_resnet"] + (1 - A45) * v53["p_tda"]).values, nan=0.0)

    rng = np.random.RandomState(SEED)
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5

    def fused_img_score(p_tda, i):
        if len(p_tda) == 0:
            return 0.0
        return v65.image_score_from_cands(lr.predict_proba(np.stack([s1all[i], p_tda], 1))[:, 1])

    rows = {N: {"bbox": [], "image": [], "comb_bbox": [], "froc": []} for N in N_FEW}
    for N in N_FEW:
        for f in range(5):
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]
            if N == 0:
                p_te = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx]
                p_te_img = p_te
            else:
                pos = [i for i in tr_idx if y_img[i] == 1]
                neg = [i for i in tr_idx if y_img[i] == 0]
                r2 = np.random.RandomState(SEED + N + f)
                npos = N // 2
                pick = list(r2.choice(pos, size=min(npos, len(pos)), replace=False)) + \
                    list(r2.choice(neg, size=min(N - npos, len(neg)), replace=False))
                pick = [i for i in pick if len(Xpall[i])]
                Xc = np.concatenate([Xpall[i] for i in pick])
                # (a) bbox-türevli aday etiketleri
                yc_box = np.concatenate([np.array([1.0 if h is not None else 0.0
                                                   for h in hits[data[i]["pid"]]], np.float32) for i in pick])
                # (b) görüntü etiketi (V6.5 ile aynı)
                yc_img = np.concatenate([np.full(len(Xpall[i]), y_img[i], np.float32) for i in pick])
                def _fit(yc):
                    if len(set(yc.tolist())) < 2:
                        return None
                    return LogisticRegression(class_weight="balanced", max_iter=500, C=0.1,
                                              random_state=SEED).fit(Xc, yc)
                m_box, m_img = _fit(yc_box), _fit(yc_img)
                p_te = ([m_box.predict_proba(Xpall[i])[:, 1] if len(Xpall[i]) else np.zeros(0) for i in te_idx]
                        if m_box is not None else
                        [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx])
                p_te_img = ([m_img.predict_proba(Xpall[i])[:, 1] if len(Xpall[i]) else np.zeros(0) for i in te_idx]
                            if m_img is not None else p_te)
            te_y = y_img[te_idx]
            sc_box = np.array([fused_img_score(p_te[k], i) for k, i in enumerate(te_idx)])
            sc_img = np.array([fused_img_score(p_te_img[k], i) for k, i in enumerate(te_idx)])
            recs = []
            for kk, i in enumerate(te_idx):
                c_ = cands[data[i]["pid"]]
                lb = hits[data[i]["pid"]]
                pf = (lr.predict_proba(np.stack([s1all[i], p_te[kk]], 1))[:, 1]
                      if len(c_) else np.zeros(0))
                recs.append({"pid": data[i]["pid"], "is_nodule": y_img[i] == 1,
                             "cands": [(c_[j][0], c_[j][1], float(pf[j]), lb[j]) for j in range(len(c_))]})
            fr, _ = m62.froc_fixed(recs, len(te_idx), sum(len(data[i]["gts"]) for i in te_idx))
            rows[N]["froc"].append(fr)
            ok = len(set(te_y)) > 1
            rows[N]["bbox"].append(roc_auc_score(te_y, sc_box) if ok else np.nan)
            rows[N]["image"].append(roc_auc_score(te_y, sc_img) if ok else np.nan)
            comb = 0.5 * rankdata(g53[te_idx]) / len(te_idx) + 0.5 * rankdata(sc_box) / len(te_idx)
            rows[N]["comb_bbox"].append(roc_auc_score(te_y, comb) if ok else np.nan)

    curve = {}
    for N in N_FEW:
        def ms(key):
            a = np.array(rows[N][key], float)
            return round(float(np.nanmean(a)), 4), round(float(np.nanstd(a)), 4)
        bm, bs = ms("bbox")
        im, isd = ms("image")
        cm, cs = ms("comb_bbox")
        curve[str(N)] = {
            "bbox_labels_mean": bm, "bbox_labels_std": bs,
            "image_labels_mean": im, "image_labels_std": isd,
            "combined_bbox_mean": cm, "combined_bbox_std": cs,
            "froc_mean": {k2: round(float(np.mean([fr[k2] for fr in rows[N]["froc"]])), 4)
                          for k2 in ("0.5", "1", "2", "4")},
        }
        print(f"    N={N:3d} | bbox {bm:.4f}±{bs:.4f} | image {im:.4f}±{isd:.4f} "
              f"| comb {cm:.4f}±{cs:.4f} | FROC@2 {curve[str(N)]['froc_mean']['2']:.3f}", flush=True)

    report = {"experiment": "V6.5b", "config": KEY, "pca_components": 128,
              "pca_explained_var": round(float(pca.explained_variance_ratio_.sum()), 4),
              "folds": 5, "N": list(N_FEW), "curve": curve,
              "note": "Kalibrasyon LR'si PCA-128 uzayında; bbox sütunu GT kutusuna isabet eden "
                      "adaylarla, image sütunu görüntü etiketiyle eğitilmiştir. Aşama-1 donmuş."}
    OUT.write_text(json.dumps(v65._san(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}", flush=True)


if __name__ == "__main__":
    main()
