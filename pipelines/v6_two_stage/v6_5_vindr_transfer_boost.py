"""V6.5 — VinDr harici transfer güçlendirmesi.

KOL 1 (zero-shot, 0 etiket): AdaBN-M5 adayları üzerinde
  (a) JPEG-denoise (Gaussian σ ∈ {0.6, 1.0}) — kübik homoloji öncesi
  (b) Anatomik ölçek-eşlemeli kırpıntı: S = round(sabit * H_lung / 810)
  (c) TDA özellik hizalama: quantile (rank-gauss) ve CORAL → NODE21 train dağılımı
KOL 2 (few-shot): 5-fold stratified CV, N ∈ {10,20,40,80,160} etiketli hedef görüntü
  (i) yalnız 2. aşama TDA-HGB kalibrasyonu (Aşama-1 donmuş)
  (ii) V5.3 global + V6 lokal birleşik

Hız: 5 özellik konfigürasyonu 10 süreçle paralel çıkarılır (tek çekirdek yerine).
"""

from __future__ import annotations

import argparse
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
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import candidates_classical as cc  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as m62  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
from detector_models import CenterNetHead  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
EXT = PROJECT_ROOT / "data/external_vindr"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_5_vindr_transfer_boost.json"
INPUT = 512
A45 = 0.45
SEED = 42
N_FEW = (0, 10, 20, 40, 80, 160)
NW = max(2, min(10, (os.cpu_count() or 4) - 2))

CONFIGS = [
    ("base_z", "fixed", 0.0, "z"),
    ("denoise0.6_z", "fixed", 0.6, "z"),
    ("denoise1.0_z", "fixed", 1.0, "z"),
    ("scale_z", "scale", 0.0, "z"),
    ("denoise1.0_scale_z", "scale", 1.0, "z"),
    ("base_quantile", "fixed", 0.0, "quantile"),
    ("base_coral", "fixed", 0.0, "coral"),
    ("denoise1.0_scale_coral", "scale", 1.0, "coral"),
]
# özellik çıkarımı gereken benzersiz (ölçek türü, sigma) kombinasyonları
KEYS = [("fixed", 0.0), ("fixed", 0.6), ("fixed", 1.0), ("scale", 0.0), ("scale", 1.0)]


def _sizes(kind: str, lung_h: int) -> tuple[int, int]:
    if kind == "fixed":
        return 32, 64
    s = max(0.5, min(2.0, lung_h / 810.0))
    return int(np.clip(round(32 * s), 16, 96)), int(np.clip(round(64 * s), 32, 192))


def _crop(canon, cx, cy, size):
    return s2._crop(canon, cx, cy, size)


# --------------------------------------------------------------------------- #
# Paralel özellik çıkarımı (VinDr adayları)
# --------------------------------------------------------------------------- #
def _extract_chunk(chunk, ranges):
    """Bir görüntü yığını için 5 konfigürasyonun TDA+uzamsal özellikleri."""
    ex = ZoneTDAExtractor(s2._settings(ranges))
    out = {k: [] for k in KEYS}
    for d in chunk:
        c = d["cands"]
        raw = d["raw"] > 0
        ys, xs = np.where(raw)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, 0, 0, 0)
        dist = ndi.distance_transform_edt(raw).astype(np.float32) if ys.size else np.zeros(raw.shape, np.float32)
        canon0 = d["canon"].astype(np.float32)
        for key in KEYS:
            kind, sigma = key
            canon = ndi.gaussian_filter(canon0, sigma) if sigma > 0 else canon0
            s1, s2_ = _sizes(kind, d["lung_h"])
            tda = np.zeros((len(c), 2 * 2 * s2.GRID * s2.GRID), np.float32)
            spa = np.zeros((len(c), 4), np.float32)
            for k, (cx, cy, _) in enumerate(c):
                tda[k] = np.concatenate([ex.extract(_crop(canon, cx, cy, s1))[0],
                                         ex.extract(_crop(canon, cx, cy, s2_))[0]])
                if ys.size:
                    spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
            out[key].append({"tda": tda, "spa": spa})
    return [d["pid"] for d in chunk], out


def _w_extract(args):
    chunk, ranges = args
    return _extract_chunk(chunk, ranges)


# --------------------------------------------------------------------------- #
# Paralel NODE21 referans özellikleri (quantile / CORAL hedefi)
# --------------------------------------------------------------------------- #
def _w_ref(args):
    pids, ctr, ranges = args
    ex = ZoneTDAExtractor(s2._settings(ranges))
    tda, spa = [], []
    for p in pids:
        try:
            img = common.load_image(p)
            mask_d = common.load_lung_mask(p)
            p1, p99 = common.get_lung_p1p99()[p]
            canonical = common.canonicalize(img, mask_d, float(p1), float(p99))
            raw = s2._raw_mask(p)
        except Exception:
            continue
        ys, xs = np.where(raw)
        if not ys.size:
            continue
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        dist = ndi.distance_transform_edt(raw).astype(np.float32)
        for cx, cy, _ in ctr[p]:
            tda.append(np.concatenate([ex.extract(_crop(canonical, cx, cy, 32))[0],
                                       ex.extract(_crop(canonical, cx, cy, 64))[0]]))
            spa.append(s2._spatial(cx, cy, raw, dist, bbox))
    return tda, spa


class Aligner:
    """NODE21 train aday dağılımına z-score / quantile / CORAL hizalama."""

    def __init__(self, X_ref: np.ndarray):
        self.mean = X_ref.mean(0)
        self.std = X_ref.std(0) + 1e-6
        self.sorted = np.sort(X_ref, axis=0)
        self.n = len(X_ref)
        Xc = (X_ref - self.mean) / self.std
        self.cov = np.cov(Xc, rowvar=False)

    def z(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def quantile(self, X):
        out = np.empty_like(X, dtype=np.float32)
        for j in range(X.shape[1]):
            col = self.sorted[:, j]
            r = np.searchsorted(col, X[:, j], side="left")
            q = np.clip((r + 0.5) / self.n, 1e-4, 1 - 1e-4)
            out[:, j] = norm.ppf(q)
        return out

    def coral(self, X):
        Xs = X - X.mean(0)
        cov_s = np.cov(Xs, rowvar=False) + np.eye(X.shape[1]) * 1e-3
        ev, V = np.linalg.eigh(cov_s)
        W = V @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-6, None))) @ V.T
        ev2, V2 = np.linalg.eigh(self.cov)
        C = V2 @ np.diag(np.sqrt(np.clip(ev2, 0, None))) @ V2.T
        return ((Xs @ W) @ C + self.mean).astype(np.float32)


def image_score_from_cands(scores) -> float:
    if len(scores) == 0:
        return 0.0
    return float(1 - np.prod(1 - np.clip(np.sort(scores)[::-1][:5], 0, 1)))


def eval_config(data, cands, feats, aligner, align_mode, hgb, lr):
    """Hizalamayı TÜM aday matrisine tek seferde uygular (görüntü başına değil).

    Not: görüntü başına CORAL çağırmak 400×2 adet 1604×1604 özdeğer ayrışımı
    demekti; tek çağrıda 2 ayrışım yeterli ve aynı sonucu verir.
    """
    blocks, counts, s1s = [], [], []
    for d, f in zip(data, feats):
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([f["tda"], f["spa"]], 1))
            s1s.append(np.array([x[2] for x in c], np.float32))
    if not blocks:
        return {"auc": None}
    X = np.concatenate(blocks)
    if align_mode == "z":
        X = aligner.z(X)
    elif align_mode == "quantile":
        X = aligner.quantile(X)
    else:
        X = aligner.coral(X)
    p_tda = hgb.predict_proba(X)[:, 1]
    p_fus = lr.predict_proba(np.stack([np.concatenate(s1s), p_tda], 1))[:, 1]
    y, scores, k = [], [], 0
    for d, n in zip(data, counts):
        y.append(d["y"])
        scores.append(image_score_from_cands(p_fus[k:k + n]))
        k += n
    return {"auc": round(float(roc_auc_score(y, scores)), 4) if len(set(y)) > 1 else None}


def _san(obj):
    if isinstance(obj, dict):
        return {k: _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sm = args.smoke
    n_few = (0, 10) if sm else N_FEW

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    scaler, hgb, lr, ranges = b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"]

    # ---------------- VinDr hazırlık
    import torchxrayvision as xrv

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    if sm:
        cohort = pd.concat([cohort[cohort["patient_label"] == 1].head(10),
                            cohort[cohort["patient_label"] == 0].head(10)]).reset_index(drop=True)
    seg = PSPNet5ZoneSegmenter(device=device)
    print("[0] VinDr hazırlık...", flush=True)
    data, t0 = [], time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"])
        lab = int(row["patient_label"])
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

    # ---------------- AdaBN-M5 + adaylar
    print("[1] AdaBN (M5) + adaylar...", flush=True)
    tensors = [m64._norm512(d["x512"]) for d in data]
    m5a = m64.adabn(m5, tensors, device)
    cands = {}
    for d in data:
        try:
            c = m64._pred_from_512(m5a, d["x512"], d["canon"].shape, device, "M5")
            cands[d["pid"]] = cc._nms(c, 20, 50)
        except Exception:
            cands[d["pid"]] = []
    print(f"    aday: {sum(len(v) for v in cands.values())}", flush=True)

    # ---------------- KOL 1
    print("[2] KOL 1 — zero-shot konfigürasyonları...", flush=True)
    from concurrent.futures import ProcessPoolExecutor

    # NODE21 referans (quantile/CORAL hedefi) — paralel
    n_ref = 10 if sm else 100
    tr_pids = list(np.random.RandomState(SEED).choice(p2._nodule_pids("train"), n_ref, replace=False)) + \
        list(np.random.RandomState(SEED + 1).choice(p2._control_pids("train"), n_ref, replace=False))
    c_tr = s2.stage1_candidates(tr_pids, m5, m64.load_models(device)[1], device, top=5 if sm else 20)
    ref_parts = np.array_split(np.array(tr_pids), NW)
    ref_jobs = [(list(part), {p: c_tr[p]["M5_heatTTA"] for p in part}, ranges) for part in ref_parts]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as pool:
        ref_res = list(pool.map(_w_ref, ref_jobs))
    ref_tda = [x for r in ref_res for x in r[0]]
    ref_spa = [x for r in ref_res for x in r[1]]
    X_ref = np.stack([np.concatenate([t, s]) for t, s in zip(ref_tda, ref_spa)]).astype(np.float32)
    print(f"    referans aday: {len(X_ref)} ({time.time() - t0:.0f}s)", flush=True)
    aligner = Aligner(X_ref)

    # VinDr özellikleri — paralel (her süreç kendi dilimini 5 konfig için işler)
    parts = np.array_split(np.arange(len(data)), NW)
    jobs = []
    for part in parts:
        chunk = [{"pid": data[i]["pid"], "canon": data[i]["canon"], "raw": data[i]["raw"],
                  "lung_h": data[i]["lung_h"], "cands": cands[data[i]["pid"]]} for i in part]
        jobs.append((chunk, ranges))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as pool:
        res = list(pool.map(_w_extract, jobs))
    feats_cache = {k: {} for k in KEYS}
    for pids, out in res:
        for k in KEYS:
            for pid, f in zip(pids, out[k]):
                feats_cache[k][pid] = f
    print(f"    özellikler ({len(KEYS)} konfig, {NW} süreç): {time.time() - t0:.0f}s", flush=True)

    kol1 = {}
    for name, skind, sigma, amode in CONFIGS:
        feats = [feats_cache[(skind, sigma)][d["pid"]] for d in data]
        res_c = eval_config(data, cands, feats, aligner, amode, hgb, lr)
        kol1[name] = res_c
        print(f"    {name:24s} AUC={res_c['auc']}", flush=True)

    # ---------------- KOL 2
    print("[3] KOL 2 — few-shot kalibrasyon (5-fold CV)...", flush=True)
    best_name = max(kol1, key=lambda k: kol1[k]["auc"] or 0)
    bkind, bsigma, amode_best = next((c[1], c[2], c[3]) for c in CONFIGS if c[0] == best_name)
    feats = [feats_cache[(bkind, bsigma)][d["pid"]] for d in data]
    hits = {d["pid"]: s2._labels_for(cands[d["pid"]], d["gts"]) for d in data}
    # hizalama tüm aday matrisine tek seferde, sonra görüntülere böl
    blocks, counts = [], []
    for d, f in zip(data, feats):
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([f["tda"], f["spa"]], 1))
    X = np.concatenate(blocks) if blocks else np.zeros((0, 1604), np.float32)
    X = aligner.z(X) if amode_best == "z" else (aligner.quantile(X) if amode_best == "quantile" else aligner.coral(X))
    Xall, kk = [], 0
    for n in counts:
        Xall.append(X[kk:kk + n])
        kk += n
    y_img = np.array([d["y"] for d in data])

    v53 = pd.read_csv(PROJECT_ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv").set_index("image_id")
    v53 = v53.reindex([d["pid"] for d in data])
    g53 = np.nan_to_num((A45 * v53["p_resnet"] + (1 - A45) * v53["p_tda"]).values, nan=0.0)

    rng = np.random.RandomState(SEED)
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5

    rows = {N: {"stage2_only": [], "combined": [], "froc": []} for N in n_few}
    for N in n_few:
        for f in range(5):
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]
            if N == 0:
                p_te = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx]
            else:
                pos = [i for i in tr_idx if y_img[i] == 1]
                neg = [i for i in tr_idx if y_img[i] == 0]
                r2 = np.random.RandomState(SEED + N + f)
                npos = N // 2
                pick = list(r2.choice(pos, size=min(npos, len(pos)), replace=False)) + \
                    list(r2.choice(neg, size=min(N - npos, len(neg)), replace=False))
                Xc = np.concatenate([Xall[i] for i in pick if len(Xall[i])])
                yc = np.concatenate([np.full(len(Xall[i]), y_img[i]) for i in pick if len(Xall[i])])
                if len(set(yc.tolist())) < 2:
                    p_te = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx]
                else:
                    lr_cal = LogisticRegression(class_weight="balanced", max_iter=300, C=0.1,
                                                random_state=SEED).fit(Xc, yc)
                    p_te = [lr_cal.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx]
            s2_img = np.array([
                image_score_from_cands(
                    lr.predict_proba(np.stack([
                        np.array([x[2] for x in cands[data[i]["pid"]]], np.float32), p_te[k]
                    ], 1))[:, 1]
                ) if len(p_te[k]) else 0.0
                for k, i in enumerate(te_idx)
            ])
            te_y = y_img[te_idx]
            recs = []
            for k, i in enumerate(te_idx):
                c_ = cands[data[i]["pid"]]
                lb = hits[data[i]["pid"]]
                s1_ = np.array([x[2] for x in c_], np.float32)
                pf = lr.predict_proba(np.stack([s1_, p_te[k]], 1))[:, 1] if len(c_) else np.zeros(0)
                recs.append({"pid": data[i]["pid"], "is_nodule": y_img[i] == 1,
                             "cands": [(c_[j][0], c_[j][1], float(pf[j]), lb[j]) for j in range(len(c_))]})
            n_les = sum(len(data[i]["gts"]) for i in te_idx)
            fr, _ = m62.froc_fixed(recs, len(te_idx), n_les)
            rows[N]["froc"].append(fr)
            rows[N]["stage2_only"].append(roc_auc_score(te_y, s2_img) if len(set(te_y)) > 1 else np.nan)
            comb = 0.5 * rankdata(g53[te_idx]) / len(te_idx) + 0.5 * rankdata(s2_img) / len(te_idx)
            rows[N]["combined"].append(roc_auc_score(te_y, comb) if len(set(te_y)) > 1 else np.nan)

    kol2 = {}
    for N in n_few:
        a1 = np.array(rows[N]["stage2_only"], float)
        a2 = np.array(rows[N]["combined"], float)
        kol2[str(N)] = {
            "stage2_only_mean": round(float(np.nanmean(a1)), 4),
            "stage2_only_std": round(float(np.nanstd(a1)), 4),
            "combined_mean": round(float(np.nanmean(a2)), 4),
            "combined_std": round(float(np.nanstd(a2)), 4),
            "froc_mean": {k: round(float(np.mean([fr[k] for fr in rows[N]["froc"]])), 4)
                          for k in ("0.5", "1", "2", "4")},
        }
        print(f"    N={N:3d} | stage2 {kol2[str(N)]['stage2_only_mean']:.4f}±{kol2[str(N)]['stage2_only_std']:.4f} "
              f"| combined {kol2[str(N)]['combined_mean']:.4f}±{kol2[str(N)]['combined_std']:.4f} "
              f"| FROC@2 {kol2[str(N)]['froc_mean']['2']:.3f}", flush=True)

    report = {
        "experiment": "V6.5",
        "kol1_zero_shot": kol1,
        "kol2_few_shot": {"config": best_name, "folds": 5, "N": list(n_few), "curve": kol2},
        "config": {"workers": NW, "ref_patients": 2 * n_ref,
                   "note": "KOL2: 2. aşama tablosal modeli hedef merkezde kalibre edilir; "
                           "Aşama-1 ağırlıkları donmuştur. Aday etiketi = görüntü etiketi (kaba)."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_san(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}", flush=True)


if __name__ == "__main__":
    main()
