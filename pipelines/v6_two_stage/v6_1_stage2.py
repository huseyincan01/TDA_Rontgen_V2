"""V6.1 — 2. Aşama: Lokal TDA + uzamsal öncül + lokal görüntü ile FP eleme.

1. Aşama adayları (K=30 ve K=50) üzerinde, RAM'de (diske görüntü YAZMADAN):
  * Çift ölçekli lokal TDA: 32x32 (çekirdek) ve 64x64 (çevre) kırpıntılar üzerinde
    H0/H1 Persistence Image — **c_lungref ölçeği korunur** (yerel min-max YOK).
  * Uzamsal öncül: rel_x, rel_y (akciğer bbox), hilus uzaklığı, plevra uzaklığı.
  * Lokal görüntü yaması: 64x64 (küçük CNN).
Modeller:
  (a) yalnız 1. Aşama skoru (baseline)   (b) lokal TDA + uzamsal (HGB)
  (c) lokal görüntü CNN                  (d) tam füzyon (LR)
Değerlendirme: lezyon FROC (0.5/1/2/4 FP/hasta) + hasta düzeyi ROC-AUC (Top-1/Top-3/Noisy-OR).

Kullanım:
    python pipelines/v6_two_stage/v6_1_stage2.py --train-nodule 400 --train-control 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402
from detector_models import CenterNetHead, FCOSHead, INPUT  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT_JSON = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_1_stage2_fp_reduction.json"
GRID = 20
SIGMA = 1.0
INNER, OUTER = 32, 64
K_BUDGETS = (30, 50)
FP_TARGETS = (0.5, 1.0, 2.0, 4.0)
SEED = 42
_IDS = {"s": {}, "r": {}}


# ------------------------------------------------------------------ lokal TDA
def _settings(ranges):
    return TDASettings(grid_size=GRID, intensity_max=255.0,
                       h0_birth_range=tuple(ranges["h0_birth"]),
                       h0_persistence_range=tuple(ranges["h0_pers"]),
                       h1_birth_range=tuple(ranges["h1_birth"]),
                       h1_persistence_range=tuple(ranges["h1_pers"]),
                       pi_sigma_pixels=SIGMA)


def _crop(canon, cx, cy, size):
    h = size // 2
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    out = np.zeros((size, size), dtype=np.float32)
    sy0, sy1 = max(0, y0), min(canon.shape[0], y0 + size)
    sx0, sx1 = max(0, x0), min(canon.shape[1], x0 + size)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)] = canon[sy0:sy1, sx0:sx1]
    return out


def _spatial(cx, cy, raw_mask, dist_map, bbox):
    y0, y1, x0, x1 = bbox
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    rel_x = (cx - x0) / w
    rel_y = (cy - y0) / h
    ctr_x, ctr_y = (x0 + x1) / 2, (y0 + y1) / 2
    diag = np.hypot(w, h)
    hilus = float(np.hypot(cx - ctr_x, cy - ctr_y) / diag) if diag > 0 else 0.0
    iy, ix = int(np.clip(round(cy), 0, raw_mask.shape[0] - 1)), int(np.clip(round(cx), 0, raw_mask.shape[1] - 1))
    pleura = float(dist_map[iy, ix] / 30.0)
    return [rel_x, rel_y, hilus, pleura]


def _patient_features(args):
    """pid + aday listesi → lokal TDA (2 kırpıntı), uzamsal, 64x64 yama."""
    pid, cands, ranges, with_patch = args
    try:
        img = common.load_image(pid)
        mask_dil = common.load_lung_mask(pid)
        p1, p99 = common.get_lung_p1p99()[pid]
        canon = common.canonicalize(img, mask_dil, float(p1), float(p99))
        raw = _raw_mask(pid)
        dist = _distance_map(raw)
        ys, xs = np.where(raw)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, 0, 0, 0)
    except Exception as exc:
        return pid, None, f"{type(exc).__name__}: {exc}"

    ex = ZoneTDAExtractor(_settings(ranges))
    n = len(cands)
    tda = np.zeros((n, 2 * 2 * GRID * GRID), dtype=np.float32)   # 2 kırpıntı × (H0+H1)
    spa = np.zeros((n, 4), dtype=np.float32)
    pat = np.zeros((n, OUTER * OUTER), dtype=np.uint8) if with_patch else None
    for i, (cx, cy, _) in enumerate(cands):
        c_in = _crop(canon, cx, cy, INNER)
        c_out = _crop(canon, cx, cy, OUTER)
        v_in, _ = ex.extract(c_in)
        v_out, _ = ex.extract(c_out)
        tda[i] = np.concatenate([v_in, v_out])
        spa[i] = _spatial(cx, cy, raw, dist, bbox)
        if with_patch:
            pat[i] = np.clip(c_out, 0, 255).astype(np.uint8).ravel()
    return pid, {"tda": tda, "spa": spa, "pat": pat}, None


def _raw_mask(pid):
    import SimpleITK as sitk
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(common.PATIENTS_DIR / pid / "mask_5zones.mha")))
    return (m[0] if m.ndim == 3 else m) > 0


def _distance_map(raw):
    from scipy import ndimage as ndi
    return ndi.distance_transform_edt(raw).astype(np.float32)


def derive_ranges(sample_cands) -> dict:
    """Eğitim adaylarından 32/64 kırpıntı için ayrı PI aralıkları (p1-p99)."""
    v51 = _v51()
    tasks = []
    for pid, cands in sample_cands:
        try:
            img = common.load_image(pid)
            mask = common.load_lung_mask(pid)
            p1, p99 = common.get_lung_p1p99()[pid]
            canon = common.canonicalize(img, mask, float(p1), float(p99))
        except Exception:
            continue
        for cx, cy, _ in cands[:6]:
            tasks.append((canon, cx, cy))
    h0b, h0p, h1b, h1p = [], [], [], []
    ex = ZoneTDAExtractor(TDASettings(grid_size=GRID, intensity_max=255.0, pi_sigma_pixels=SIGMA))
    for canon, cx, cy in tasks:
        for size in (INNER, OUTER):
            fil = _fil(canon, cx, cy, size)
            if fil is None:
                continue
            h0, h1 = ex.compute_diagrams(fil)
            if len(h0):
                h0b.extend(h0[:, 0].tolist()); h0p.extend((h0[:, 1] - h0[:, 0]).tolist())
            if len(h1):
                h1b.extend(h1[:, 0].tolist()); h1p.extend((h1[:, 1] - h1[:, 0]).tolist())

    def bnd(a, birth):
        a = np.asarray(a, float)
        if not a.size:
            return [0.0, 10.0]
        lo = float(np.floor(np.percentile(a, 1))) if birth else 0.0
        hi = float(np.ceil(np.percentile(a, 99)))
        return [lo, hi if hi > lo else lo + 10.0]

    return {"h0_birth": bnd(h0b, True), "h0_pers": bnd(h0p, False),
            "h1_birth": bnd(h1b, True), "h1_pers": bnd(h1p, False),
            "n_h0": len(h0b), "n_h1": len(h1b)}


def _v51():
    if _IDS["s"].get("v51") is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v51", PROJECT_ROOT / "pipelines/exp/V5_1_physical_canonicalization.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _IDS["s"]["v51"] = m
    return _IDS["s"]["v51"]


def _fil(canon, cx, cy, size):
    crop = _crop(canon, cx, cy, size)
    m = crop > 0
    if not m.any():
        return None
    fil = np.zeros(crop.shape, dtype=np.uint8)
    fil[m] = (255 - np.clip(np.rint(crop[m]), 0, 255)).astype(np.uint8)
    return fil


# ------------------------------------------------------------------ aşama 1 adayları
def stage1_candidates(pids, m5, m6, device, top=50):
    out = {}
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        c5h = ev._predict_heat_tta(m5, canon, device, "M5")
        c6 = p2.predict(m6, canon, device, "M6")
        out[pid] = {
            "M5_heatTTA": c5h[:top],
            "M5M6TTA": ev._rrf([c5h, c6])[:top],
        }
        if (i + 1) % 100 == 0:
            print(f"    aday: {i + 1}/{len(pids)}", flush=True)
    return out


def _labels_for(cands, gts):
    """Her aday için: hangi GT'yi vuruyor (Kriter1) veya None."""
    lab = []
    for cx, cy, _ in cands:
        hit = None
        for gi, gt in enumerate(gts):
            if common.hit_criterion1(cx, cy, gt):
                hit = gi
                break
        lab.append(hit)
    return lab


# ------------------------------------------------------------------ değerlendirme
def froc_and_auc(records, n_patients, n_lesions):
    """records: [{pid, is_nodule, cands:[(score, hit_or_None)]}] → FROC + hasta AUC."""
    flat = []
    for r in records:
        for s, hit in r["cands"]:
            flat.append((s, r["pid"], hit))
    flat.sort(key=lambda x: -x[0])
    detected = {}
    fp_per_patient = {r["pid"]: 0 for r in records}
    pts = []  # (fp/patient, sensitivity)
    n_det = 0
    seen_fp = 0
    for s, pid, hit in flat:
        if hit is not None and hit not in detected.setdefault(pid, set()):
            detected[pid].add(hit)
            n_det += 1
        else:
            fp_per_patient[pid] += 1
            seen_fp += 1
        pts.append((seen_fp / max(1, n_patients), n_det / max(1, n_lesions)))
    # hedef FP/patient seviyelerinde duyarlılık (adım fonksiyonu)
    out = {}
    for t in FP_TARGETS:
        sens = 0.0
        for fp, sn in pts:
            if fp <= t + 1e-9:
                sens = max(sens, sn)
        out[f"{t:g}"] = round(sens, 4)
    # hasta düzeyi AUC
    nod = [r for r in records if r["is_nodule"]]
    ctl = [r for r in records if not r["is_nodule"]]
    aucs = {}
    for agg in ("top1", "top3", "noisyor"):
        def score(r):
            ss = sorted([s for s, _ in r["cands"]], reverse=True)
            if not ss:
                return 0.0
            if agg == "top1":
                return ss[0]
            if agg == "top3":
                return float(np.mean(ss[:3]))
            return float(1.0 - np.prod([1.0 - min(1.0, max(0.0, s)) for s in ss[:5]]))
        y = [1] * len(nod) + [0] * len(ctl)
        pred = [score(r) for r in nod] + [score(r) for r in ctl]
        aucs[agg] = round(float(roc_auc_score(y, pred)), 4) if len(set(y)) > 1 else None
    return out, aucs


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_patch_cnn(X, y, device, epochs=6, bs=256):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = PatchCNN().to(device)
    pos = max(1, int(y.sum())); neg = max(1, int(len(y) - pos))
    pos_weight = torch.tensor([neg / pos], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X.reshape(-1, 1, OUTER, OUTER), dtype=torch.float32) / 255.0
    yt = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            tot += float(crit(model(xb), yb)) * len(yb); n += len(yb)
        print(f"  patchCNN epoch {ep}/{epochs} | loss {tot / max(1, n):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict_patch(model, X, device, bs=512):
    out = []
    Xt = torch.tensor(X.reshape(-1, 1, OUTER, OUTER), dtype=torch.float32) / 255.0
    for i in range(0, len(Xt), bs):
        out.append(torch.sigmoid(model(Xt[i:i + bs].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# ------------------------------------------------------------------ ana akış
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-nodule", type=int, default=400)
    ap.add_argument("--train-control", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()
    print("V6.1 Stage-2 | 1. Aşama donduruldu (M5_heatTTA + M5M6TTA)")

    tr_nod = p2._nodule_pids("train"); tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(SEED)
    tr_pids = list(rng.choice(tr_nod, size=min(args.train_nodule, len(tr_nod)), replace=False)) + \
              list(rng.choice(tr_ctl, size=min(args.train_control, len(tr_ctl)), replace=False))
    va_pids = common.split_patients()["val"]
    print(f"Train aday hastası: {len(tr_pids)} | Val: {len(va_pids)}")

    print("\n[1] Aşama-1 adayları (train)...")
    c_tr = stage1_candidates(tr_pids, m5, m6, device)
    print("[1] Aşama-1 adayları (val)...")
    c_va = stage1_candidates([p for p in va_pids], m5, m6, device)

    # GT + etiketler
    gt_tr = {p: common.load_gt(p) for p in tr_pids}
    gt_va = {p: common.load_gt(p) for p in va_pids}
    lab_tr = {p: {v: _labels_for(c_tr[p][v], gt_tr[p]) for v in c_tr[p]} for p in tr_pids}
    lab_va = {p: {v: _labels_for(c_va[p][v], gt_va[p]) for v in c_va[p]} for p in va_pids}

    print("\n[2] Lokal TDA aralıkları (train adaylarından)...")
    sample = [(p, c_tr[p]["M5_heatTTA"]) for p in tr_pids if gt_tr[p]][:20]
    ranges = derive_ranges(sample)
    print(f"   {ranges}")

    def build(pids, cand_dict, variant):
        """Belirli bir 1. aşama varyantının adayları için özellikleri çıkarır."""
        jobs = [(pid, cand_dict[pid][variant], ranges, True) for pid in pids]
        out = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for pid, res, err in pool.map(_patient_features, jobs, chunksize=8):
                if err or res is None:
                    print(f"   ! {pid}: {err}")
                    out[pid] = None
                else:
                    out[pid] = res
        return out

    print("\n[3] Lokal özellikler (train)...")
    f_tr = {p: {} for p in tr_pids}
    for variant in ("M5_heatTTA", "M5M6TTA"):
        got = build(tr_pids, c_tr, variant)
        for p in tr_pids:
            f_tr[p][variant] = got.get(p)
    print("[3] Lokal özellikler (val)...")
    f_va = {p: {} for p in va_pids}
    for variant in ("M5_heatTTA", "M5M6TTA"):
        got = build(va_pids, c_va, variant)
        for p in va_pids:
            f_va[p][variant] = got.get(p)

    # eğitim matrisi (M5_heatTTA adayları; top-50)
    Xt_tda, Xt_spa, Xt_pat, yt, s1_t = [], [], [], [], []
    for p in tr_pids:
        f = f_tr[p].get("M5_heatTTA")
        if f is None:
            continue
        for i, (cx, cy, s) in enumerate(c_tr[p]["M5_heatTTA"]):
            Xt_tda.append(f["tda"][i]); Xt_spa.append(f["spa"][i]); Xt_pat.append(f["pat"][i])
            yt.append(1 if lab_tr[p]["M5_heatTTA"][i] is not None else 0)
            s1_t.append(s)
    Xt_tda = np.asarray(Xt_tda, np.float32); Xt_spa = np.asarray(Xt_spa, np.float32)
    Xt_pat = np.asarray(Xt_pat, np.uint8); yt = np.asarray(yt, int); s1_t = np.asarray(s1_t, np.float32)
    print(f"   train aday: {len(yt)} | pozitif: {int(yt.sum())} ({100 * yt.mean():.2f}%)")

    Xt_ts = np.concatenate([Xt_tda, Xt_spa], axis=1)
    sc = StandardScaler().fit(Xt_ts)
    print("\n[4] Modeller...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=None,
                                         class_weight="balanced", random_state=SEED)
    hgb.fit(sc.transform(Xt_ts), yt)
    print(f"   HGB (TDA+uzamsal) eğitildi ({time.time() - t0:.0f}s)")
    cnn = train_patch_cnn(Xt_pat, yt, device, epochs=6)

    # füzyon için eğitim skorları → LR
    p_tda_t = hgb.predict_proba(sc.transform(Xt_ts))[:, 1]
    p_img_t = predict_patch(cnn, Xt_pat, device)
    Zt = np.stack([s1_t, p_tda_t, p_img_t], axis=1)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(Zt, yt)
    print("   Füzyon LR eğitildi")

    # ------------------------------------------------------------------ değerlendirme
    def eval_setting(variant: str, K: int, model_name: str):
        recs = []
        for p in va_pids:
            f = f_va[p].get(variant)
            if f is None:
                continue
            cands = c_va[p][variant][:K]
            labs = lab_va[p][variant][:K]
            n = len(cands)
            if n == 0:
                recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0, "cands": []})
                continue
            tda = f["tda"][:K]; spa = f["spa"][:K]; pat = f["pat"][:K]
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda = hgb.predict_proba(sc.transform(np.concatenate([tda, spa], 1)))[:, 1]
            p_img = predict_patch(cnn, pat, device)
            p_fus = lr.predict_proba(np.stack([s1, p_tda, p_img], 1))[:, 1]
            score_map = {"a_stage1": s1, "b_tda_spatial": p_tda, "c_patch_cnn": p_img, "d_fusion": p_fus}
            recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0,
                         "cands": [(float(score_map[model_name][i]), labs[i]) for i in range(n)]})
        n_les = sum(len(g) for g in gt_va.values())
        return froc_and_auc(recs, len(recs), n_les)

    results = {}
    for variant in ("M5_heatTTA", "M5M6TTA"):
        for K in K_BUDGETS:
            for model in ("a_stage1", "b_tda_spatial", "c_patch_cnn", "d_fusion"):
                froc, auc = eval_setting(variant, K, model)
                results[f"{variant}|K{K}|{model}"] = {"froc": froc, "patient_auc": auc}
                print(f"  {variant} K{K} {model:14s} FROC {froc} | AUC {auc}")

    # kayıt
    import joblib
    joblib.dump({"hgb": hgb, "scaler": sc, "lr": lr}, CKPT / "stage2_tda_spatial_lr.joblib")
    torch.save(cnn.state_dict(), CKPT / "stage2_patch_cnn.pt")
    report = {
        "experiment": "V6.1", "stage": "2. asama lokal TDA + FP eleme (val)",
        "config": {"inner": INNER, "outer": OUTER, "grid": GRID, "kd_tree": "c_lungref korunur (yerel min-max yok)",
                   "train_candidates": int(len(yt)), "train_positive": int(yt.sum()),
                   "positive_ratio": round(float(yt.mean()), 4),
                   "ranges": ranges, "train_patients": len(tr_pids), "val_patients": len(va_pids),
                   "froc_convention": "duplicate hit'ler FP sayılır",
                   "K_budgets": list(K_BUDGETS)},
        "results": results,
        "notes": "1. asama dondurulmus checkpoint'ler kullanildi; disk yazimi yok (yalniz JSON+model).",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT_JSON}")


if __name__ == "__main__":
    main()
