"""V6.4 — Domain stabilizasyonu (AdaBN + SNR) + domain-generalized M5 + NODE21 test mühürlemesi.

ADIM 1 (VinDr, eğitimsiz): (a) SNR/arka plan normalizasyonu, (b) AdaBN
ADIM 2: Fiziksel domain augmentation ile M5 yeniden eğitimi (NODE21 train only)
ADIM 3: NODE21 TEST (734 hasta) tek seferlik mühürleme: V6, V5.2 ve birleşik

Kullanım: python pipelines/v6_two_stage/v6_4_domain_and_seal.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as m62  # noqa: E402
import v6_3_final as m63  # noqa: E402
from detector_models import CenterNetHead, FCOSHead, build_targets_heat, center_focal_loss, INPUT, STRIDE  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_4_domain_and_test_seal.json"
EXT = PROJECT_ROOT / "data/external_vindr"
SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_models(device):
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device))
    m5.eval(); m6.eval()
    return m5, m6


def _norm512(arr_u8):
    a = torch.from_numpy(arr_u8.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (a - m) / s


def adabn(model, tensors, device, momentum=None, bs=8):
    """BN running istatistiklerini hedef (etiketsiz) yığınla günceller.
    momentum=None → kümülatif ortalama (az batch'te bile kararlı)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = momentum
    model.train()
    with torch.no_grad():
        for i in range(0, len(tensors), bs):
            model(torch.stack(tensors[i:i + bs]).to(device))
    model.eval()
    return model


def snr_rerank(cands, mode="diff"):
    if len(cands) < 16:
        return cands
    s = np.array([c[2] for c in cands], float)
    bg = s[14:30]
    new = s - float(np.median(bg)) if mode == "diff" else (s - float(bg.mean())) / (float(bg.std()) + 1e-4)
    out = [(cands[i][0], cands[i][1], float(new[i])) for i in range(len(cands))]
    out.sort(key=lambda c: -c[2])
    return out


def image_score(cands, agg="top1"):
    if not cands:
        return 0.0
    ss = sorted([c[2] for c in cands], reverse=True)
    if agg == "top1":
        return float(ss[0])
    if agg == "top3":
        return float(np.mean(ss[:3]))
    return float(1 - np.prod(1 - np.clip(ss[:5], 0, 1)))


# ---------------------------------------------------------------- VinDr geçişi
def vindr_pass(m5, m6, device, seg, limit=0):
    import torchxrayvision as xrv
    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    if limit:
        cohort = cohort.head(limit)
    recs = []
    t0 = time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"]); lab = int(row["patient_label"])
        img = np.array(Image.open(EXT / "images" / f"{pid}.jpg"))
        gts = []
        for _, b in nod[nod["image_id"] == pid].iterrows():
            x0, x1 = b["x_min"] / 3.0, b["x_max"] / 3.0
            y0, y1 = b["y_min"] / 3.0, b["y_max"] / 3.0
            if x1 > x0 and y1 > y0:
                gts.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
        H, W = img.shape
        try:
            raw = seg.segment(img)["zone_map"] > 0
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                                       iterations=common.DILATION_PX)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            sx, sy = W / 1024.0, H / 1024.0
            c5 = [(x * sx, y * sy, s) for x, y, s in ev._predict_heat_tta(m5, canon, device, "M5")]
            c6 = [(x * sx, y * sy, s) for x, y, s in p2.predict(m6, canon, device, "M6")]
            cm = ev._rrf([c5, c6])
            x512 = np.array(Image.fromarray(np.clip(canon, 0, 255).astype(np.uint8)).resize(
                (INPUT, INPUT), Image.Resampling.BILINEAR))
        except Exception:
            c5, cm, x512 = [], [], np.zeros((INPUT, INPUT), np.uint8)
        recs.append({"pid": pid, "y": lab, "gts": gts, "c5": c5, "cm": cm, "x512": x512, "shape": (H, W)})
        if (i + 1) % 50 == 0:
            print(f"    VinDr {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)
    return recs


def _pred_from_512(model, x512, shape, device, kind="M5"):
    """512 girdiden tahmin; koordinatlar görüntü uzayına ölçeklenir."""
    with torch.no_grad():
        if kind == "M5":
            h, _ = model(_norm512(x512).unsqueeze(0).to(device))
            heat = torch.sigmoid(h)[0, 0].cpu().numpy()
            maxf = ndi.maximum_filter(heat, size=3)
            ys, xs = np.where((heat >= maxf) & (heat > 0.01))
            sc = heat[ys, xs]
            order = np.argsort(-sc)
            H, W = shape
            return [(float((xs[k] * STRIDE + STRIDE / 2) * W / INPUT),
                     float((ys[k] * STRIDE + STRIDE / 2) * H / INPUT), float(sc[k])) for k in order]
        o, b = model(_norm512(x512).unsqueeze(0).to(device))
        obj = torch.sigmoid(o)[0, 0].cpu().numpy()
        ys, xs = np.where(obj > 0.02)
        sc = obj[ys, xs]
        order = np.argsort(-sc)
        H, W = shape
        return [(float((xs[k] * STRIDE + STRIDE / 2) * W / INPUT),
                 float((ys[k] * STRIDE + STRIDE / 2) * H / INPUT), float(sc[k])) for k in order]


def eval_vindr(recs, key="c5", snr=None):
    y, sc = [], []
    for r in recs:
        cands = r[key]
        if snr:
            cands = snr_rerank(cands, snr)
        y.append(r["y"]); sc.append(image_score(cands, "top1"))
    auc = float(roc_auc_score(y, sc)) if len(set(y)) > 1 else None
    out = {"auc": round(auc, 4) if auc else None, "recall": {}}
    for K in (30, 50):
        a = b = 0
        for r in recs:
            if not r["gts"]:
                continue
            cands = r[key]
            if snr:
                cands = snr_rerank(cands, snr)
            hit = any(common.hit_criterion1(cx, cy, g) for g in r["gts"] for cx, cy, _ in cands[:K])
            a += int(hit); b += 1
        out["recall"][str(K)] = round(a / b, 4) if b else None
    return out


# ---------------------------------------------------------------- domain-gen M5
def _aug_input(canon, centers, radii, rng):
    from skimage.exposure import equalize_adapthist
    im = np.array(Image.fromarray(np.clip(canon, 0, 255).astype(np.uint8)).resize(
        (INPUT, INPUT), Image.Resampling.BILINEAR), dtype=np.float32)
    g = rng.uniform(0.6, 1.6)
    im = 255.0 * np.clip(im / 255.0, 0, 1) ** g
    if rng.rand() < 0.3:
        im = equalize_adapthist(np.clip(im, 0, 255).astype(np.uint8), clip_limit=0.02).astype(np.float32)
    if rng.rand() < 0.5:
        im = ndi.gaussian_filter(im, rng.uniform(0.4, 1.4))
    if rng.rand() < 0.5:
        im = im + rng.normal(0, rng.uniform(1.0, 4.0), im.shape)
    im = np.clip(im, 0, 255)
    s = rng.uniform(0.75, 1.25)
    n = int(round(INPUT * s))
    zoom = np.array(Image.fromarray(im.astype(np.uint8)).resize((n, n), Image.Resampling.BILINEAR), np.float32)
    out = np.zeros((INPUT, INPUT), np.float32)
    if s >= 1.0:
        off = (n - INPUT) // 2
        out = zoom[off:off + INPUT, off:off + INPUT]
    else:
        pad = (INPUT - n) // 2
        out[pad:pad + n, pad:pad + n] = zoom
    cen = [((cx - INPUT / 2) * s + INPUT / 2, (cy - INPUT / 2) * s + INPUT / 2) for cx, cy in centers]
    rad = [r * s for r in radii]
    keep = [(c, r) for c, r in zip(cen, rad) if 0 <= c[0] < INPUT and 0 <= c[1] < INPUT]
    if not keep:
        return None, [], []
    cen, rad = [k[0] for k in keep], [k[1] for k in keep]
    if rng.rand() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1])
        cen = [(INPUT - cx, cy) for cx, cy in cen]
    a = torch.from_numpy(out / 255.0).unsqueeze(0).repeat(3, 1, 1)
    m = torch.tensor(MEAN).view(3, 1, 1); sd = torch.tensor(STD).view(3, 1, 1)
    return (a - m) / sd, cen, rad


def _loss(model, xb, Y, R):
    h, r = model(xb)
    return center_focal_loss(h, Y) + 0.1 * F.smooth_l1_loss(r, R, reduction="none").mul(
        (Y > 0).float()).sum() / (Y > 0).sum().clamp_min(1)


def train_m5_dg(pos, neg, epochs, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = CenterNetHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    items = [(p, True) for p in pos] + [(p, False) for p in neg]
    B = 4
    for ep in range(1, epochs + 1):
        model.train(); rng = np.random.RandomState(SEED + ep)
        xs, ts, tot, nb = [], [], 0.0, 0
        for i in rng.permutation(len(items)):
            try:
                canon, centers, radii = p2._prep(items[i][0])
            except Exception:
                continue
            x, cen, rad = _aug_input(canon, centers, radii, rng)
            if x is None:
                continue
            Y, R = build_targets_heat(cen, rad, device)
            xs.append(x); ts.append((Y, R))
            if len(xs) == B:
                Y = torch.stack([t[0] for t in ts]); R = torch.stack([t[1] for t in ts])
                loss = _loss(model, torch.stack(xs).to(device), Y, R)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
                xs, ts = [], []
        if xs:
            Y = torch.stack([t[0] for t in ts]); R = torch.stack([t[1] for t in ts])
            loss = _loss(model, torch.stack(xs).to(device), Y, R)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        print(f"  M5_dg epoch {ep}/{epochs} | loss {tot / max(1, nb):.4f}", flush=True)
    model.eval(); torch.save(model.state_dict(), CKPT / "M5_domain_gen.pt")
    return model


# ---------------------------------------------------------------- V5.2 skorları
def v52_scores(split: str) -> pd.DataFrame:
    zs = pd.read_csv(m63.V53_ZONE_SPLITS).sort_values("vector_index").reset_index(drop=True)
    X = np.load(m63.V53_FEATURES)
    cb = torch.load(m63.V53_CNN, map_location="cpu", weights_only=False)
    sb = joblib.load(m63.V53_SVM)
    rows = []
    for z in range(1, 6):
        m = ((zs["split"] == split) & (zs["zone_id"] == z)).values
        Xz = X[m]; sc = cb["scalers"][z]
        mm = SmallZoneCNN(in_channels=2, dropout=float(cb["config"]["dropout"]))
        mm.load_state_dict(cb["models_state_dict"][z]); mm.eval()
        with torch.no_grad():
            pc = torch.sigmoid(mm(torch.tensor(vector_to_2channel_image(sc.transform(Xz)),
                                                dtype=torch.float32))).numpy().flatten()
        ps = sb[z].predict_proba(sc.transform(Xz))[:, 1]
        rows.append(pd.DataFrame({"pid": zs.loc[m, "patient_id"].values, "p": 0.55 * pc + 0.45 * ps}))
    tda = pd.concat(rows).groupby("pid")["p"].max()
    need = (zs["is_train_balanced_sample"].values.astype(bool)
            | (zs["split"] == "val").values | (zs["split"] == "test").values)
    sub = zs[need].reset_index(drop=True)
    img = m63._v52().resnet_patient_probs(sub, (sub["split"] == split).values).set_index("patient_id")["p"]
    out = pd.DataFrame({"pid": tda.index})
    out["tda"] = tda.reindex(out["pid"]).values
    out["img"] = img.reindex(out["pid"]).values
    out["global"] = 0.45 * out["img"] + 0.55 * out["tda"]
    return out


# ---------------------------------------------------------------- test mühürü
def _safe_auc(y, p):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def test_seal(m5, m6, device, scaler, hgb, lr, ranges, limit=0):
    pids = common.split_patients()["test"]
    if limit:
        pids = pids[:limit]
    print(f"  test: {len(pids)} hasta")
    c = s2.stage1_candidates(pids, m5, m6, device, top=50)
    gts = {p: common.load_gt(p) for p in pids}
    labs = {p: s2._labels_for(c[p]["M5M6TTA"], gts[p]) for p in pids}
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(p, c[p]["M5M6TTA"], ranges, False) for p in pids]
    feats = {}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
            feats[pid] = res
    recs, v6p = [], {}
    for p in pids:
        f = feats.get(p); cands = c[p]["M5M6TTA"][:50]; lb = labs[p][:50]
        if f is None or not cands:
            recs.append({"pid": p, "is_nodule": len(gts[p]) > 0, "cands": []}); v6p[p] = 0.0; continue
        X = scaler.transform(np.concatenate([f["tda"][:50], f["spa"][:50]], 1))
        s1 = np.array([x[2] for x in cands], np.float32)
        p_tda = hgb.predict_proba(X)[:, 1]
        s_fus = lr.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
        recs.append({"pid": p, "is_nodule": len(gts[p]) > 0,
                     "cands": [(cands[i][0], cands[i][1], float(s_fus[i]), lb[i]) for i in range(len(cands))]})
        v6p[p] = float(1 - np.prod(1 - np.clip(np.sort(s_fus)[::-1][:5], 0, 1)))
    n_les = sum(len(g) for g in gts.values())
    froc, auc = m62.froc_fixed(recs, len(recs), n_les)
    v52 = v52_scores("test").set_index("pid")
    order = [r["pid"] for r in recs]
    y = np.array([1 if r["is_nodule"] else 0 for r in recs])
    g52 = np.nan_to_num(v52["global"].reindex(order).values, nan=0.0)
    v6v = np.array([v6p[p] for p in order])
    combined = m63._combined(g52, v6v)
    return {"n_patients": len(pids), "n_lesions": n_les,
            "v6": {"froc": froc, "patient_auc": auc},
            "v5_2_global": {"patient_auc": _safe_auc(y, g52)},
            "combined_rankavg": {"patient_auc": _safe_auc(y, combined),
                                 "r": round(float(np.corrcoef(g52, v6v)[0, 1]), 4)}}


# ---------------------------------------------------------------- ana
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sm = args.smoke
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5, m6 = load_models(device)
    seg = PSPNet5ZoneSegmenter(device=device)
    report = {"experiment": "V6.4", "smoke": sm}

    print("\n[ADIM 1] VinDr stabilizasyon...")
    vrecs = vindr_pass(m5, m6, device, seg, limit=20 if sm else 0)
    report["step1"] = {
        "baseline": {"m5": eval_vindr(vrecs, "c5"), "m6m5": eval_vindr(vrecs, "cm")},
        "snr_diff": {"m5": eval_vindr(vrecs, "c5", "diff"), "m6m5": eval_vindr(vrecs, "cm", "diff")},
        "snr_z": {"m5": eval_vindr(vrecs, "c5", "z"), "m6m5": eval_vindr(vrecs, "cm", "z")},
    }
    tensors = [_norm512(r["x512"]) for r in vrecs]
    m5a = adabn(load_models(device)[0], tensors, device)
    m6a = adabn(load_models(device)[1], tensors, device)
    ada = []
    for r in vrecs:
        rec = dict(r)
        try:
            rec["c5"] = _pred_from_512(m5a, r["x512"], r["shape"], device, "M5")
            rec["c6"] = _pred_from_512(m6a, r["x512"], r["shape"], device, "M6")
            rec["cm"] = ev._rrf([rec["c5"], rec["c6"]])
        except Exception:
            pass
        ada.append(rec)
    report["step1"]["adabn"] = {"m5": eval_vindr(ada, "c5"), "m6m5": eval_vindr(ada, "cm")}

    print("\n[ADIM 2] Domain-generalized M5...")
    tr_nod = p2._nodule_pids("train"); tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(SEED)
    pos = list(rng.choice(tr_nod, 40 if sm else 790, replace=False))
    neg = list(rng.choice(tr_ctl, 40 if sm else 790, replace=False))
    m5dg = train_m5_dg(pos, neg, 1 if sm else 15, device)
    val_pids = p2._nodule_pids("val")
    if sm:
        val_pids = val_pids[:20]
    c_val = s2.stage1_candidates(val_pids, m5dg, m6, device, top=50)
    gts_val = {p: common.load_gt(p) for p in val_pids}
    node21 = {}
    for v in ("M5_heatTTA", "M5M6TTA"):
        for K in (30, 50):
            a = b = 0
            for p in val_pids:
                if not gts_val[p]:
                    continue
                hit = any(common.hit_criterion1(cx, cy, gt) for gt in gts_val[p]
                          for cx, cy, _ in c_val[p][v][:K])
                a += int(hit); b += 1
            node21[f"{v}_{K}"] = round(a / b, 4) if b else None
    vrecs_dg = vindr_pass(m5dg, m6, device, seg, limit=20 if sm else 0)
    report["step2"] = {"node21_val_recall": node21,
                       "vindr": {"m5": eval_vindr(vrecs_dg, "c5"), "m6m5": eval_vindr(vrecs_dg, "cm")}}

    print("\n[ADIM 3] NODE21 test mühürü...")
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    report["step3_test_seal"] = test_seal(load_models(device)[0], m6, device,
                                          b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"],
                                          limit=30 if sm else 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_sanitize(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}")


if __name__ == "__main__":
    main()
