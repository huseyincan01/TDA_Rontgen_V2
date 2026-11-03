"""V6.9t — Kutu-içi Geometrik ve Graf TDA ablasyonu (Radyal, EDT kalınlık, Vietoris-Rips graf).

32x32 ve 64x64 aday yamalarında, ham parlaklık kübik TDA'ya (Pure1600) EK olarak üç yeni
filtrasyon/graf temsili denenir:
  M1 (424)  : kutu-merkezli radyal kübik filtrasyon  f = (255 - I) + alpha * r_norm
  M2 (560)  : geometrik kalınlık (EDT) kübik TDA + yuvarlaklık/merkezilik özetleri
  M3 (296)  : nokta bulutu Vietoris-Rips graf TDA + kovaryans (izotropi) özetleri

İki kulvar (aynı 8 kol):
  Kulvar 1 (birincil, kalibresiz): fit=V6.9r fit (1366, 0 JSRT) · z-ref=V6.9r va (693, 0 JSRT)
                                   · test=242 Saf JSRT (149 jp + 93 jn)
  Kulvar 2 (kurum içi)           : fit=NODE21 train (1580) · z-ref ve değerlendirme=NODE21 val (734)

Ana mimari checkpoint'lerine ve mevcut önbelleklere DOKUNULMAZ (SHA-256 koşu başı/sonu).
Tuzak B: s1 HGB'ye özellik olarak verilmez; yalnız z-füzyonda kullanılır (w_s1=0,55 / w_tda=0,45).

Kullanım:
    python pipelines/v6_two_stage/v6_9t_geometric_graph_tda.py --limit 20     # duman (önbellek YAZMAZ)
    python pipelines/v6_two_stage/v6_9t_geometric_graph_tda.py                 # tam koşu
    python pipelines/v6_two_stage/v6_9t_geometric_graph_tda.py --eval-test ARM # kilitli test (tek atım)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_SITE = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import gudhi  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from v6_6_optimization_audit import iou, official_metric  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
CACHE_L = ART / "_v6_9l_cache.npz"
CACHE_M = ART / "_v6_9m_test_cache.npz"
CACHE_R = ART / "_v6_9r_cache.npz"
CACHE_GEO = ART / "_v6_9t_geo_cache.npz"
OUT_JSON = ART / "v6_9t_geometric_graph_tda.json"

BOX_HALF = 28.0
W_TDA = 0.45
SEED = 42
GEO_DIM = 1280
SL_M1, SL_M2, SL_M3 = slice(0, 424), slice(424, 984), slice(984, 1280)

READONLY_FILES = [
    CKPT / "M5_foundation.pt",
    CKPT / "stage2_foundation_tda_hgb_pure1600.joblib",
    CKPT / "stage2_foundation_tda_hgb.joblib",
    CKPT / "M5_nojsrt_full.pt",
    CKPT / "stage2_nojsrt_full_tda_hgb_pure1600.joblib",
    CACHE_L, CACHE_M, CACHE_R,
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "YOK"
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            d.update(chunk)
    return d.hexdigest()


# --------------------------------------------------------------------------- #
# PI / PH yardımcıları
# --------------------------------------------------------------------------- #
def _finite(diag) -> np.ndarray:
    if diag is None or len(diag) == 0:
        return np.empty((0, 2), dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    return diag[np.isfinite(diag[:, 1])]


def _pi(diag, grid, b_range, p_range, sigma=1.0) -> np.ndarray:
    img = np.zeros((grid, grid), dtype=np.float64)
    if len(diag) == 0:
        return img.ravel().astype(np.float32)
    b = diag[:, 0]
    p = diag[:, 1] - diag[:, 0]
    ok = np.isfinite(b) & np.isfinite(p) & (p > 0)
    if not np.any(ok):
        return img.ravel().astype(np.float32)
    b = np.clip(b[ok], b_range[0], b_range[1])
    p = np.clip(p[ok], p_range[0], p_range[1])
    bw = max(1e-6, b_range[1] - b_range[0])
    pw = max(1e-6, p_range[1] - p_range[0])
    cols = np.minimum(((b - b_range[0]) / bw * grid).astype(int), grid - 1)
    rows = np.minimum(((p - p_range[0]) / pw * grid).astype(int), grid - 1)
    np.add.at(img, (rows, cols), p)
    return gaussian_filter(img, sigma=sigma, mode="constant", truncate=3.0).ravel().astype(np.float32)


def _diag_stats6(diag) -> np.ndarray:
    if len(diag) == 0:
        return np.zeros(6, dtype=np.float32)
    b = diag[:, 0]
    p = diag[:, 1] - diag[:, 0]
    ok = np.isfinite(b) & np.isfinite(p) & (p > 0)
    if not np.any(ok):
        return np.zeros(6, dtype=np.float32)
    b, p = b[ok], p[ok]
    sp = float(p.sum())
    wb = float((b * p).sum() / (sp + 1e-9))
    return np.array([float(len(p)), float(p.max()), sp, float(p.mean()), float(b.min()), wb], dtype=np.float32)


def _cubical_h0_h1(field: np.ndarray):
    cc = gudhi.CubicalComplex(dimensions=field.shape,
                              top_dimensional_cells=field.astype(np.float64, copy=False).ravel())
    cc.compute_persistence()
    return (_finite(cc.persistence_intervals_in_dimension(0)),
            _finite(cc.persistence_intervals_in_dimension(1)))


def _r_norm_grid(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    return (np.hypot(yy - c, xx - c) / max(1.0, size / 2.0)).astype(np.float32)


_RNORM = {32: _r_norm_grid(32), 64: _r_norm_grid(64)}


# --------------------------------------------------------------------------- #
# M1 / M2 / M3
# --------------------------------------------------------------------------- #
def extract_m1_radial_patch(patch: np.ndarray, alpha: float = 60.0) -> np.ndarray:
    """212 boyut (2 yama = 424)."""
    valid = np.isfinite(patch) & (patch > 0)
    if np.count_nonzero(valid) < 4:
        return np.zeros(212, dtype=np.float32)
    rn = _RNORM[patch.shape[0]]
    f_rad = np.full(patch.shape, 255.0, dtype=np.float32)
    f_rad[valid] = np.clip((255.0 - patch[valid]) + alpha * rn[valid], 0.0, 255.0)
    h0, h1 = _cubical_h0_h1(f_rad)
    return np.concatenate([
        _pi(h0, 10, (0.0, 255.0), (0.0, 255.0)),
        _pi(h1, 10, (0.0, 255.0), (0.0, 255.0)),
        _diag_stats6(h0), _diag_stats6(h1),
    ]).astype(np.float32)


def extract_m2_edt_patch(patch: np.ndarray) -> np.ndarray:
    """280 boyut (2 yama = 560)."""
    valid = np.isfinite(patch) & (patch > 0)
    if np.count_nonzero(valid) < 4:
        return np.zeros(280, dtype=np.float32)
    vals = patch[valid]
    qs = np.percentile(vals, [50, 70, 85])
    rn = _RNORM[patch.shape[0]]
    h = patch.shape[0] // 2
    max_r = float(patch.shape[0] / 2.0)

    q_stats = []
    d70 = np.zeros(patch.shape, dtype=np.float32)
    for q_idx, thr in enumerate(qs):
        fg = valid & (patch >= thr)
        area = float(np.count_nonzero(fg))
        if area < 1.0:
            q_stats.append(np.zeros(8, dtype=np.float32))
            continue
        dt = distance_transform_edt(fg).astype(np.float32)
        if q_idx == 1:
            d70 = dt
        c_val = float(dt[h, h])
        c3 = float(dt[max(0, h - 3):h + 4, max(0, h - 3):h + 4].max())
        g_max = float(dt.max())
        cent = c3 / (g_max + 1e-5)
        round_ = float(np.clip(np.pi * g_max ** 2 / (area + 1.0), 0.0, 2.0))
        h0q, _ = _cubical_h0_h1(-dt)
        pq = (h0q[:, 1] - h0q[:, 0]) if len(h0q) else np.zeros(0)
        pq = np.sort(pq[pq > 0])[::-1] if len(pq) else np.zeros(0)
        p1 = float(pq[0]) if len(pq) >= 1 else g_max
        p2 = float(pq[1]) if len(pq) >= 2 else 0.0
        q_stats.append(np.array([c_val, c3, g_max, cent, round_, p1, p2, p2 / (p1 + 1e-5)], dtype=np.float32))

    f_edt = -d70
    f_geo = -(d70 * (patch / 255.0) * np.exp(-0.5 * rn ** 2)).astype(np.float32)
    h0e, h1e = _cubical_h0_h1(f_edt)
    h0g, h1g = _cubical_h0_h1(f_geo)
    br, pr = (-max_r, 0.0), (0.0, max_r)
    return np.concatenate([
        _pi(h0e, 8, br, pr), _pi(h1e, 8, br, pr), _pi(h0g, 8, br, pr), _pi(h1g, 8, br, pr), *q_stats,
    ]).astype(np.float32)


def extract_m3_vr_graph_patch(patch: np.ndarray, n_pts: int = 40) -> np.ndarray:
    """148 boyut (2 yama = 296)."""
    valid = np.isfinite(patch) & (patch > 0)
    ys, xs = np.where(valid)
    if len(ys) < 4:
        return np.zeros(148, dtype=np.float32)
    vals = patch[ys, xs].astype(np.float32)
    if len(vals) > n_pts:
        idx = np.argpartition(vals, -n_pts)[-n_pts:]
        ys, xs, vals = ys[idx], xs[idx], vals[idx]
    half = max(1.0, patch.shape[0] / 2.0)
    u = (ys.astype(np.float32) - half) / half
    v = (xs.astype(np.float32) - half) / half
    w_int = (vals - vals.min()) / (vals.max() - vals.min() + 1e-5) + 0.1
    w = w_int / w_int.sum()

    mu_u, mu_v = float((w * u).sum()), float((w * v).sum())
    du, dv = u - mu_u, v - mu_v
    c_uu = float((w * du * du).sum()) + 1e-6
    c_vv = float((w * dv * dv).sum()) + 1e-6
    c_uv = float((w * du * dv).sum())
    ev = np.sort(np.linalg.eigvalsh(np.array([[c_uu, c_uv], [c_uv, c_vv]], dtype=np.float64)))[::-1]
    l1, l2 = float(max(1e-6, ev[0])), float(max(0.0, ev[1]))
    geo6 = np.array([l2 / l1, np.sqrt(l1 + l2), np.hypot(mu_u, mu_v),
                     float(np.mean(np.hypot(u, v) <= 0.5)), l1, l2], dtype=np.float32)

    pts3 = np.stack([u, v, (vals / 255.0) * 1.2], axis=1).astype(np.float64)
    diff = pts3[:, None, :] - pts3[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    st = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=2.5).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h0 = _finite(st.persistence_intervals_in_dimension(0))
    h1 = _finite(st.persistence_intervals_in_dimension(1))
    if len(h0) > 0:
        d0 = h0[:, 1]
        mst8 = np.array([d0.max(), d0.mean(), d0.std(), np.percentile(d0, 25), np.percentile(d0, 50),
                         np.percentile(d0, 75), np.percentile(d0, 90), d0.sum()], dtype=np.float32)
    else:
        mst8 = np.zeros(8, dtype=np.float32)
    return np.concatenate([
        _pi(h0, 8, (0.0, 2.0), (0.0, 2.5)), _pi(h1, 8, (0.0, 2.5), (0.0, 2.5)),
        mst8, _diag_stats6(h1), geo6,
    ]).astype(np.float32)


def box_geo_vec(c_in: np.ndarray, c_out: np.ndarray) -> np.ndarray:
    m1 = np.concatenate([extract_m1_radial_patch(c_in), extract_m1_radial_patch(c_out)])
    m2 = np.concatenate([extract_m2_edt_patch(c_in), extract_m2_edt_patch(c_out)])
    m3 = np.concatenate([extract_m3_vr_graph_patch(c_in), extract_m3_vr_graph_patch(c_out)])
    return np.concatenate([m1, m2, m3]).astype(np.float32)


def _patient_geo_worker(args):
    pid, cx, cy = args
    try:
        img = common.load_image(pid)
        mask_dil = common.load_lung_mask(pid)
        p1, p99 = common.get_lung_p1p99()[pid]
        canon = common.canonicalize(img, mask_dil, float(p1), float(p99))
    except Exception as exc:  # noqa: BLE001
        return pid, None, f"{type(exc).__name__}: {exc}"
    out = np.zeros((len(cx), GEO_DIM), dtype=np.float32)
    for i in range(len(cx)):
        c_in = s2._crop(canon, float(cx[i]), float(cy[i]), 32)
        c_out = s2._crop(canon, float(cx[i]), float(cy[i]), 64)
        out[i] = box_geo_vec(c_in, c_out)
    return pid, np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), None


def extract_geo(pids, off, cx, cy, tag, limit_patients=None) -> np.ndarray:
    n_tot = int(off[len(pids)])
    out = np.zeros((n_tot, GEO_DIM), dtype=np.float32)
    jobs = []
    for i, p in enumerate(pids):
        lo, hi = int(off[i]), int(off[i + 1])
        jobs.append((p, cx[lo:hi], cy[lo:hi], lo, hi))
    t0 = time.time()
    index = {p: (lo, hi) for p, _, _, lo, hi in jobs}
    try:
        with ProcessPoolExecutor(max_workers=10) as pool:
            for k, (pid, res, err) in enumerate(pool.map(_patient_geo_worker, [j[:3] for j in jobs], chunksize=4), 1):
                if err:
                    print(f"    ! {pid}: {err}", flush=True)
                    continue
                lo, hi = index[pid]
                out[lo:hi] = res
                if k % 100 == 0:
                    print(f"    geo {tag} {k}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    (ProcessPool yok: {type(exc).__name__} → ThreadPool)", flush=True)
        with ThreadPoolExecutor(max_workers=8) as tp:
            for pid, res, err in tp.map(_patient_geo_worker, [j[:3] for j in jobs]):
                if err:
                    print(f"    ! {pid}: {err}", flush=True)
                    continue
                lo, hi = index[pid]
                out[lo:hi] = res
    print(f"    geo {tag}: {len(pids)} hasta, {n_tot} kutu ({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Kohort yükleme
# --------------------------------------------------------------------------- #
def _slice_cohort(z, tag, limit=None):
    pids = [str(p) for p in z[f"{tag}_pids"]]
    off = z[f"{tag}_off"].astype(np.int64)
    if limit:
        pids = pids[:limit]
        off = off[:limit + 1]
    n = int(off[-1])
    return dict(
        pids=pids, off=off,
        cx=z[f"{tag}_cx"][:n].astype(np.float32), cy=z[f"{tag}_cy"][:n].astype(np.float32),
        s1=z[f"{tag}_s1"][:n].astype(np.float32), tda=z[f"{tag}_tda"][:n].astype(np.float32),
    )


def _labels(c):
    ys = []
    for i, p in enumerate(c["pids"]):
        lo, hi = int(c["off"][i]), int(c["off"][i + 1])
        cands = [(float(c["cx"][j]), float(c["cy"][j]), float(c["s1"][j])) for j in range(lo, hi)]
        ys.extend([1 if lb is not None else 0 for lb in s2._labels_for(cands, common.load_gt(p))])
    return np.asarray(ys, dtype=np.int64)


def _gts(pids):
    g = {p: [(d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]) for d in common.load_gt(p)] for p in pids}
    return g, {p: bool(g[p]) for p in pids}


def merge_cohorts(a, b):
    off = np.concatenate([a["off"], b["off"][1:] + a["off"][-1]])
    return dict(
        pids=a["pids"] + b["pids"], off=off,
        **{k: np.concatenate([a[k], b[k]]) for k in ("cx", "cy", "s1", "tda", "geo") if k in a and k in b},
    )


# --------------------------------------------------------------------------- #
# Değerlendirme (v6_9s ile aynı protokol)
# --------------------------------------------------------------------------- #
def evaluate_predictions(c, y, p_tda, p_ref, ref):
    pids, off, cx, cy, s1 = c["pids"], c["off"], c["cx"], c["cy"], c["s1"]
    gts, is_nod = c["gts"], c["is_nod"]
    n_les = sum(len(gts[p]) for p in pids)
    out = {
        "candidate_auc": round(float(roc_auc_score(y, p_tda)), 4),
        "corr_with_s1": round(float(np.corrcoef(s1, p_tda)[0, 1]), 4),
        "corr_with_pure1600": round(float(np.corrcoef(p_ref, p_tda)[0, 1]), 4),
    }
    for mode, w in (("tda_only", 1.0), ("fusion_w045", W_TDA)):
        flat, hit, n_pos = [], 0, 0
        for i, p in enumerate(pids):
            lo, hi = int(off[i]), int(off[i + 1])
            if hi <= lo:
                continue
            s1_i, td_i = s1[lo:hi], p_tda[lo:hi]
            if w == 1.0:
                sc = td_i
            else:
                z1 = (s1_i - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
                zt = (td_i - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
                sc = 1.0 / (1.0 + np.exp(-((1.0 - w) * z1 + w * zt)))
            top = int(np.argmax(sc))
            if is_nod[p]:
                n_pos += 1
                bx, by = float(cx[lo + top]), float(cy[lo + top])
                b = (bx - BOX_HALF, by - BOX_HALF, bx + BOX_HALF, by + BOX_HALF)
                hit += int(any(iou(b, g) > 0.2 for g in gts[p]))
            for j in range(lo, hi):
                flat.append((float(sc[j - lo]), p, float(cx[j]), float(cy[j]), 0.0))
        r = official_metric(flat, len(pids), n_les, gts, is_nod, pids, box_half=BOX_HALF)
        out[mode] = {
            "patient_auc": round(float(r["auc"]), 4),
            "S@0.125": round(float(r["sens"]["0.125"]), 4),
            "S@0.25": round(float(r["sens"]["0.25"]), 4),
            "S@0.5": round(float(r["sens"]["0.5"]), 4),
            "rank_metric": round(float(r["rank_metric"]), 4),
            "top1_box_hit_rate": round(hit / max(1, n_pos), 4),
        }
    return out


# --------------------------------------------------------------------------- #
# Kollar
# --------------------------------------------------------------------------- #
ARM_COLS = {
    "Arm_0_Baseline_Pure1600": ("tda",),
    "Arm_1_Radial_Only": ("m1",),
    "Arm_2_DistTransform_Only": ("m2",),
    "Arm_3_VR_Graph_Only": ("m3",),
    "Arm_4_GeoGraph_Combined": ("m1", "m2", "m3"),
    "Arm_5_Pure1600_Plus_Radial": ("tda", "m1"),
    "Arm_6_Pure1600_Plus_AllGeo": ("tda", "m1", "m2", "m3"),
}


def blocks(c) -> dict:
    g = c["geo"]
    return {"tda": c["tda"], "m1": g[:, SL_M1], "m2": g[:, SL_M2], "m3": g[:, SL_M3]}


def design(c, cols):
    b = blocks(c)
    return np.concatenate([b[k] for k in cols], axis=1)


def fit_hgb(X, y):
    sc = StandardScaler().fit(X)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                         random_state=SEED).fit(sc.transform(X), y)
    return sc, hgb


def predict(model, X):
    sc, hgb = model
    return hgb.predict_proba(sc.transform(X))[:, 1].astype(np.float32)


def _ref(s1_cal, p_cal):
    return {"s1_mean": float(s1_cal.mean()), "s1_std": float(s1_cal.std()),
            "tda_mean": float(p_cal.mean()), "tda_std": float(p_cal.std())}


def _z(p, ref_p):
    return (p - ref_p.mean()) / (ref_p.std() + 1e-9)


def run_track(name, fit, cal, ev, only_arms=None):
    """fit/cal/ev: kohort sözlükleri (geo+tda dahil). ev, cal ile aynı olabilir."""
    print(f"\n=== {name}: fit {len(fit['y'])} aday · cal {len(cal['s1'])} · eval {len(ev['s1'])} ===", flush=True)
    res, models, preds = {}, {}, {}
    for arm, cols in ARM_COLS.items():
        if only_arms and arm not in only_arms and arm != "Arm_0_Baseline_Pure1600":
            continue
        t0 = time.time()
        m = fit_hgb(design(fit, cols), fit["y"])
        models[arm] = m
        p_cal = predict(m, design(cal, cols))
        p_ev = p_cal if ev is cal else predict(m, design(ev, cols))
        preds[arm] = (p_cal, p_ev)
        print(f"  [{arm}] eğitildi ({time.time() - t0:.0f}s)", flush=True)
    p0_ev = preds["Arm_0_Baseline_Pure1600"][1]
    for arm, (p_cal, p_ev) in preds.items():
        res[arm] = evaluate_predictions(ev, ev["y"], p_ev, p0_ev, _ref(cal["s1"], p_cal))
        res[arm]["n_features"] = int(design(cal, ARM_COLS[arm]).shape[1])
        print(f"  {arm}: cand_auc={res[arm]['candidate_auc']} r_s1={res[arm]['corr_with_s1']} "
              f"TDA-tek S@.25={res[arm]['tda_only']['S@0.25']} hit={res[arm]['tda_only']['top1_box_hit_rate']} | "
              f"füzyon AUC={res[arm]['fusion_w045']['patient_auc']} S@.25={res[arm]['fusion_w045']['S@0.25']} "
              f"rank={res[arm]['fusion_w045']['rank_metric']}", flush=True)
    if "Arm_4_GeoGraph_Combined" in preds:
        pc0, pe0 = preds["Arm_0_Baseline_Pure1600"]
        pc4, pe4 = preds["Arm_4_GeoGraph_Combined"]
        for alpha in (0.35, 0.50):
            def blend(p0, p4):
                return (1 - alpha) * _z(p0, pc0) + alpha * _z(p4, pc4)
            bc = blend(pc0, pc4)
            be = bc if ev is cal else blend(pe0, pe4)
            # blend z-skoru → (0,1) olasılık benzeri skor
            sc_c = (1.0 / (1.0 + np.exp(-bc))).astype(np.float32)
            sc_e = (1.0 / (1.0 + np.exp(-be))).astype(np.float32)
            arm = f"Arm_7_DualHGB_ProbBlend_alpha{alpha:.2f}"
            res[arm] = evaluate_predictions(ev, ev["y"], sc_e, p0_ev, _ref(cal["s1"], sc_c))
            res[arm]["n_features"] = "1600 + 1280 (2 HGB)"
            print(f"  {arm}: cand_auc={res[arm]['candidate_auc']} r_s1={res[arm]['corr_with_s1']} "
                  f"TDA-tek S@.25={res[arm]['tda_only']['S@0.25']} hit={res[arm]['tda_only']['top1_box_hit_rate']} | "
                  f"füzyon AUC={res[arm]['fusion_w045']['patient_auc']} S@.25={res[arm]['fusion_w045']['S@0.25']} "
                  f"rank={res[arm]['fusion_w045']['rank_metric']}", flush=True)
    return res


def load_or_extract(limit):
    zl = np.load(CACHE_L, allow_pickle=True)
    zr = np.load(CACHE_R, allow_pickle=True)
    co = {
        "tr": _slice_cohort(zl, "tr", limit), "va": _slice_cohort(zl, "va", limit),
        "rfit": _slice_cohort(zr, "fit", limit), "rva": _slice_cohort(zr, "va", limit),
        "jp": _slice_cohort(zr, "jp", limit), "jn": _slice_cohort(zr, "jn", limit),
    }
    cache = {}
    if CACHE_GEO.exists() and not limit:
        cz = np.load(CACHE_GEO, allow_pickle=True)
        cache = {k: cz[k] for k in cz.files}
    for name, c in co.items():
        key = f"{name}_geo"
        if key in cache and cache[key].shape[0] == int(c["off"][-1]):
            c["geo"] = cache[key]
            print(f"  önbellek: {key} {cache[key].shape}", flush=True)
            continue
        print(f"\n[geo] {name}: {len(c['pids'])} hasta", flush=True)
        c["geo"] = extract_geo(c["pids"], c["off"], c["cx"], c["cy"], name)
        cache[key] = c["geo"]
        if not limit:
            np.savez_compressed(CACHE_GEO, **cache)
            print(f"  önbellek yazıldı: {CACHE_GEO.name}", flush=True)
    # etiketler: l-önbelleğinde hazır; r-önbelleğinde hesaplanır
    co["tr"]["y"] = zl["tr_y"][:int(co["tr"]["off"][-1])].astype(np.int64)
    co["va"]["y"] = zl["va_y"][:int(co["va"]["off"][-1])].astype(np.int64)
    co["rfit"]["y"] = zr["fit_y"][:int(co["rfit"]["off"][-1])].astype(np.int64)
    co["rva"]["y"] = _labels(co["rva"])
    co["jp"]["y"] = _labels(co["jp"])
    co["jn"]["y"] = _labels(co["jn"])
    for k in ("va", "rva", "jp", "jn"):
        co[k]["gts"], co[k]["is_nod"] = _gts(co[k]["pids"])
    return co


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-test", default="", help="kilitli test: kazanan kol adı (tek atım)")
    args = ap.parse_args()

    before = {p.name: sha256(p) for p in READONLY_FILES}
    print("V6.9t — kutu-içi geometrik/graf TDA ablasyonu · ana mimari SALT-OKUNUR", flush=True)
    t_start = time.time()
    co = load_or_extract(args.limit or None)

    # Kulvar 1: kalibresiz 242 saf JSRT
    ev1 = merge_cohorts(co["jp"], co["jn"])
    ev1["y"] = np.concatenate([co["jp"]["y"], co["jn"]["y"]])
    ev1["gts"] = {**co["jp"]["gts"], **co["jn"]["gts"]}
    ev1["is_nod"] = {**co["jp"]["is_nod"], **co["jn"]["is_nod"]}
    report = {"experiment": "V6.9t-geometric-graph-tda", "limit": args.limit, "fusion_w_tda": W_TDA}

    if args.eval_test:
        zm = np.load(CACHE_M, allow_pickle=True)
        te = _slice_cohort(zm, "te", args.limit or None)
        te["geo"] = extract_geo(te["pids"], te["off"], te["cx"], te["cy"], "te")
        te["y"] = zm["te_y"][:int(te["off"][-1])].astype(np.int64)
        te["gts"], te["is_nod"] = _gts(te["pids"])
        report["track_3_node21_test_locked"] = run_track(
            "Kulvar 3 (kilitli NODE21 Test, tek atım)", co["tr"], co["va"], te, only_arms={args.eval_test})
        out = ART / "v6_9t_test_oneshot.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"yazıldı: {out}", flush=True)
        return

    report["track_1_uncalibrated_jsrt242"] = run_track(
        "Kulvar 1 (kalibresiz 242 saf JSRT)", co["rfit"], co["rva"], ev1)
    report["track_2_node21_val734"] = run_track(
        "Kulvar 2 (NODE21 val 734)", co["tr"], co["va"], co["va"])

    after = {p.name: sha256(p) for p in READONLY_FILES}
    report["sha256_unchanged"] = before == after
    report["elapsed_s"] = round(time.time() - t_start, 1)
    if args.limit:
        print("\n(duman: JSON yazılmadı)", flush=True)
    else:
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nyazıldı: {OUT_JSON}", flush=True)
    print(f"SHA-256 değişmedi: {report['sha256_unchanged']} · süre {report['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()

