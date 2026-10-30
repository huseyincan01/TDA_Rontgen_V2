"""V6.9l — UZAMSAL KONUM ÖZELLİKLERİ ABLASYONU (yalnız VAL; kilitli test KAPALI).

SORU
----
2. aşamanın 1604 boyutlu vektörünün son 4 sütunu uzamsal: `rel_x`, `rel_y`,
**akciğer bbox merkezine uzaklık (hilus vekili)**, **akciğer maskesi sınırına uzaklık
(plevra vekili)**. Bu 4 sayı gerçekten katkı sağlıyor mu, ve daha "anatomik" bir konum
kodlaması daha iyi mi?

KOLLAR (TDA 1600 özelliği SABİT — bir kez hesaplanır, önbelleğe alınır)
----------------------------------------------------------------------
    Kol A (1600) : yalnız TDA — konum sütunları YOK
    Kol B (1604) : mevcut geometrik vekiller  [rel_x, rel_y, bbox_merkez, maske_sınırı]
    Kol C (1606) : rafine anatomik konum      [rel_x, rel_y, hilus_uzaklığı,
                                               iç(mediasten) uzaklığı, dış(lateral) uzaklığı,
                                               taraf]

Kol C nasıl hesaplanır (GUDHI YOK, saniyeler sürer)
--------------------------------------------------
`mask_5zones.mha` sol/sağ akciğeri AYIRIR (1-3 = sağ, 4-5 = sol). Bunu kullanarak her
akciğerin **iç (medial) kenar** ve **dış (lateral) kenarı** ayrı ayrı bulunur:

  * Sağ akciğer görüntüde küçük x'te durur ⇒ medial yön **+x**, lateral yön **−x**
    (hangisinin küçük x'te olduğu maskeden ÖLÇÜLÜR, varsayılmaz).
  * `iç uzaklığı`  = adaydan medial yönde ilk arka plan pikseline yatay mesafe
  * `dış uzaklığı` = lateral yönde ilk arka plan pikseline yatay mesafe
    (yatay koşu uzunlukları satır bazında cumsum ile **vektörel** hesaplanır)
  * `hilus_uzaklığı` = adayın KENDİ akciğerinin **iç kenar orta noktasına** Öklid uzaklığı;
    iç kenar = medial sınır pikselleri (koşu ≤ 2 px), orta nokta = bunların medyanı

MÜHÜR
-----
Yalnız **train (fit) + val (ölçüm)** kullanılır. **Kilitli test bölmesine dokunulmaz**
(V6.9k sonrası hâlâ kapalı). Eğitim: 790 nodüllü + 790 kontrol; tüm kollarda AYNI.
HGB hiperparametreleri sabit (`max_iter=200, lr=0.08, balanced, rs=42`), Tuzak B korunur
(1. aşama skoru özellik olarak VERİLMEZ), füzyon w=0,55 kilitli.

Kullanım:
    python pipelines/v6_two_stage/v6_9l_spatial_ablation.py --limit 40   # duman
    python pipelines/v6_two_stage/v6_9l_spatial_ablation.py --refresh    # önbelleği yenile
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import (dist_nms, heat_maps, official_metric,  # noqa: E402
                                     pick_peaks)
import v6_9k_leave_jsrt_out as k  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
OUT = ART / "v6_9l_spatial_ablation.json"
CACHE = ART / "_v6_9l_cache.npz"
CKPT = ART / "checkpoints"

SCALES = k.SCALES
WIN, NMS_PX, TOPK = k.WIN, k.NMS_PX, k.TOPK
BOX_HALF, W = k.BOX_HALF, k.W
SEED = 42
MAIN_TAG = k.MAIN_TAG
MAIN_BUNDLE = k.MAIN_BUNDLE

SPA_B_NAMES = ["rel_x", "rel_y", "bbox_merkez_uzakligi", "maske_sinir_uzakligi"]
SPA_C_NAMES = ["rel_x", "rel_y", "hilus_uzakligi", "ic_mediasten_uzakligi",
               "dis_lateral_uzakligi", "taraf"]


# --------------------------------------------------------------------------- #
# Kol C — anatomik konum (sol/sağ ayrımı maskeden)
# --------------------------------------------------------------------------- #
def _zones(pid: str) -> np.ndarray:
    import SimpleITK as sitk
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(common.PATIENTS_DIR / pid / "mask_5zones.mha")))
    return (m[0] if m.ndim == 3 else m)


def _horiz_runs(mask: np.ndarray):
    """Satır bazında yatay koşu uzunlukları.

    d_right[y,x]: (x dahil) sağa doğru ilk arka plan pikseline KADAR maskeli piksel sayısı
    d_left[y,x] : (x dahil) sola doğru aynı

    Uygulama: sağa doğru ilk arka plan = sonek MINİMUMU, sola doğru son arka plan = önek
    MAKSİMUMU. (Sonek maksimumu yanlış olur — satır sonundaki arka planı yakalar.)
    """
    H, Wd = mask.shape
    idx = np.arange(Wd, dtype=np.int32)[None, :].repeat(H, 0)
    nxt = np.minimum.accumulate(np.where(mask, np.int32(Wd), idx)[:, ::-1],
                                axis=1)[:, ::-1]
    d_right = (nxt - idx).astype(np.float32)
    prv = np.maximum.accumulate(np.where(mask, np.int32(-1), idx), axis=1)
    d_left = (idx - prv).astype(np.float32)
    return d_left, d_right


def _inner_point(mask: np.ndarray, d_left, d_right, medial_right: bool):
    """Akciğerin medial (iç) kenarının ORTA NOKTASI → (x, y)."""
    inner = d_right if medial_right else d_left
    ys, xs = np.where(mask & (inner <= 2.0))
    if ys.size == 0:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
    return float(np.median(xs)), float(np.median(ys))


def _spatial_anatomical(cx, cy, zone, runs, inner_pt, bbox):
    """Kol C: 6 boyut.

    runs: (d_left, d_right) — o adayın akciğer maskesi için önceden hesaplanmış yatay
    koşu uzunlukları (bkz. `_horiz_runs`). Sağ akciğer küçük x'te ise medial = +x.
    """
    y0, y1, x0, x1 = bbox
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    diag = float(np.hypot(w, h)) or 1.0
    rel_x = (cx - x0) / w
    rel_y = (cy - y0) / h
    side = 0.0 if (1 <= zone <= 3) else (1.0 if zone >= 4 else float(cx > (x0 + x1) / 2))
    if inner_pt is None or runs is None:
        return [rel_x, rel_y, 0.0, 0.0, 0.0, side]

    medial_right = side == 0.0
    d_left, d_right = runs
    iy = int(np.clip(round(cy), 0, d_left.shape[0] - 1))
    ix = int(np.clip(round(cx), 0, d_left.shape[1] - 1))
    d_in = float(d_right[iy, ix] if medial_right else d_left[iy, ix])
    d_out = float(d_left[iy, ix] if medial_right else d_right[iy, ix])
    hil = float(np.hypot(cx - inner_pt[0], cy - inner_pt[1]))
    return [rel_x, rel_y, hil / diag, d_in / diag, d_out / diag, side]


# --------------------------------------------------------------------------- #
# özellik çıkarımı (TDA bir kez; B ve C birlikte)
# --------------------------------------------------------------------------- #
def _worker(args):
    pid, cands, ranges = args
    try:
        img = common.load_image(pid)
        mask_dil = common.load_lung_mask(pid)
        p1, p99 = common.get_lung_p1p99()[pid]
        canon = common.canonicalize(img, mask_dil, float(p1), float(p99))
        zone = _zones(pid)
        right, left = (zone >= 1) & (zone <= 3), zone >= 4
        raw = zone > 0
        dist = s2._distance_map(raw)
        ys, xs = np.where(raw)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, 0, 0, 0)
        # medial yönü VARSAYMA: sağ akciğer küçük x'te mi? (ölçülür)
        right_is_left = bool(right.any() and left.any()
                             and np.where(right)[1].mean() < np.where(left)[1].mean())
        runs_r = _horiz_runs(right) if right.any() else None
        runs_l = _horiz_runs(left) if left.any() else None
        inner_r = _inner_point(right, *runs_r, right_is_left) if runs_r else None
        inner_l = _inner_point(left, *runs_l, not right_is_left) if runs_l else None
    except Exception as exc:
        return pid, None, f"{type(exc).__name__}: {exc}"

    ex = ZoneTDAExtractor(s2._settings(ranges))
    n = len(cands)
    tda = np.zeros((n, 2 * 2 * s2.GRID * s2.GRID), np.float32)
    spa_b = np.zeros((n, 4), np.float32)
    spa_c = np.zeros((n, 6), np.float32)
    for i, (cx, cy, _s) in enumerate(cands):
        c_in = s2._crop(canon, cx, cy, s2.INNER)
        c_out = s2._crop(canon, cx, cy, s2.OUTER)
        v_in, _ = ex.extract(c_in)
        v_out, _ = ex.extract(c_out)
        tda[i] = np.concatenate([v_in, v_out])
        spa_b[i] = s2._spatial(cx, cy, raw, dist, bbox)
        iy = int(np.clip(round(cy), 0, zone.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, zone.shape[1] - 1))
        z = int(zone[iy, ix])
        inline = z if z else (1 if cx < (bbox[2] + bbox[3]) / 2 else 4)
        inner_pt = inner_r if inline <= 3 else inner_l
        runs = runs_r if inline <= 3 else runs_l
        spa_c[i] = _spatial_anatomical(cx, cy, inline, runs, inner_pt, bbox)
    return pid, {"tda": tda, "spa_b": spa_b, "spa_c": spa_c}, None


def _candidates(model, pids, device, tag) -> dict:
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
        if (i + 1) % 200 == 0:
            print(f"    aday {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    aday {tag}: {sum(len(v) for v in out.values())} kutu "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def _features(pids, cands, ranges, tag) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(pid, cands[pid], ranges) for pid in pids]
    out, t0 = {}, time.time()
    try:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for pid, res, err in pool.map(_worker, jobs, chunksize=8):
                out[pid] = res
                if err:
                    print(f"    ! {pid}: {err}", flush=True)
    except Exception as exc:
        print(f"    (süreç havuzu yok: {type(exc).__name__} → seri)", flush=True)
        for job in jobs:
            pid, res, err = _worker(job)
            out[pid] = res
    print(f"    özellik {tag}: {len(pids)} hasta ({time.time() - t0:.0f}s)", flush=True)
    return out


def pack(pids, cands, feats):
    """Hasta bazlı listeleri tek matrise paketle (+offset, n, s1)."""
    tda, sb, sc, s1, n = [], [], [], [], []
    for pid in pids:
        f = feats.get(pid)
        cs = cands[pid]
        m = len(cs)
        n.append(m)
        s1.extend([c[2] for c in cs])
        if f is None:
            tda.append(np.zeros((m, 1600), np.float32))
            sb.append(np.zeros((m, 4), np.float32))
            sc.append(np.zeros((m, 6), np.float32))
        else:
            tda.append(f["tda"][:m])
            sb.append(f["spa_b"][:m])
            sc.append(f["spa_c"][:m])
    off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    return {"tda": np.concatenate(tda), "sb": np.concatenate(sb),
            "sc": np.concatenate(sc), "s1": np.asarray(s1, np.float32),
            "n": np.asarray(n, np.int64), "off": off,
            "pids": np.asarray(pids)}


def labels_for(pids, cands) -> np.ndarray:
    out = []
    for pid in pids:
        gts = common.load_gt(pid)
        out.extend([1 if lb is not None else 0 for lb in s2._labels_for(cands[pid], gts)])
    return np.asarray(out, np.int64)


def gts_of(pids) -> list[list[tuple]]:
    return [[(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in common.load_gt(p)]
            for p in pids]


def img_max(scores: np.ndarray, off: np.ndarray) -> np.ndarray:
    return np.asarray([scores[off[i]:off[i + 1]].max() if off[i + 1] > off[i] else 0.0
                       for i in range(len(off) - 1)])
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="duman: hasta sayısı sınırı")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9l — uzamsal konum ablasyonu (yalnız VAL)", flush=True)
    print(f"  device={device} | kilitli TEST KAPALI | ana checkpoint'lere DOKUNULMAZ", flush=True)

    tr_nod = p2._nodule_pids("train")
    tr_ctl = p2._control_pids("train")[:len(tr_nod)]
    va_pids = list(common.split_patients()["val"])
    if args.limit:
        tr_nod, tr_ctl = tr_nod[:args.limit], tr_ctl[:args.limit]
        va_pids = va_pids[:args.limit]
    tr_pids = list(tr_nod) + list(tr_ctl)
    print(f"  fit {len(tr_pids)} hasta ({len(tr_nod)}+{len(tr_ctl)}) · val {len(va_pids)}", flush=True)

    if CACHE.exists() and not args.refresh:
        print(f"\n[0] önbellek: {CACHE.name}", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        d = {key: z[key] for key in z.files}
    else:
        import joblib
        ranges = joblib.load(MAIN_BUNDLE)["ranges"]
        model = load_foundation(MAIN_TAG, device)
        print("\n[1] Adaylar (fit)…", flush=True)
        c_tr = _candidates(model, tr_pids, device, "fit")
        print("\n[2] Adaylar (val)…", flush=True)
        c_va = _candidates(model, va_pids, device, "val")
        print("\n[3] Özellikler (fit)…", flush=True)
        f_tr = _features(tr_pids, c_tr, ranges, "fit")
        print("\n[4] Özellikler (val)…", flush=True)
        f_va = _features(va_pids, c_va, ranges, "val")
        d = {"tr": pack(tr_pids, c_tr, f_tr), "va": pack(va_pids, c_va, f_va),
             "tr_pids": np.asarray(tr_pids), "va_pids": np.asarray(va_pids)}
        # aday koordinatları da gerekiyor (resmî metrik IoU ile çalışır)
        d["tr"]["cx"] = np.asarray([c[0] for p in tr_pids for c in c_tr[p]], np.float32)
        d["tr"]["cy"] = np.asarray([c[1] for p in tr_pids for c in c_tr[p]], np.float32)
        d["va"]["cx"] = np.asarray([c[0] for p in va_pids for c in c_va[p]], np.float32)
        d["va"]["cy"] = np.asarray([c[1] for p in va_pids for c in c_va[p]], np.float32)
        d["tr"]["y"] = labels_for(tr_pids, c_tr)
        d["va"]["y"] = labels_for(va_pids, c_va)
        flat = {}
        for s in ("tr", "va"):
            for key, v in d[s].items():
                flat[f"{s}_{key}"] = (v.astype(np.float32) if v.dtype == object else v)
        flat["tr_pids"] = d["tr_pids"]
        flat["va_pids"] = d["va_pids"]
        np.savez_compressed(CACHE, **flat)
        d = flat
        print(f"  önbellek yazıldı: {CACHE.name}", flush=True)

    tr = {kk[3:]: vv for kk, vv in d.items() if kk.startswith("tr_")}
    va = {kk[3:]: vv for kk, vv in d.items() if kk.startswith("va_")}
    va_pids = [str(p) for p in d["va_pids"]]
    gts = gts_of(va_pids)
    n_les = sum(len(g) for g in gts)
    print(f"\n  fit aday {len(tr['y'])} (%{100*tr['y'].mean():.2f} poz) · "
          f"val aday {len(va['y'])} · {n_les} lezyon", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    arms = {
        "A_1600_konum_yok": (tr["tda"], va["tda"]),
        "B_1604_geometrik_vekil": (np.concatenate([tr["tda"], tr["sb"]], 1),
                                   np.concatenate([va["tda"], va["sb"]], 1)),
        "C_1606_anatomik_konum": (np.concatenate([tr["tda"], tr["sc"]], 1),
                                  np.concatenate([va["tda"], va["sc"]], 1)),
    }

    offline = {"pids": va_pids, "gts": {p: g for p, g in zip(va_pids, gts)},
               "is_nod": {p: bool(g) for p, g in zip(va_pids, gts)}}
    rep = {"experiment": "V6.9l-spatial-ablation",
           "question": ("2. aşamanın 4 uzamsal konum özelliği katkı sağlıyor mu, ve daha "
                        "anatomik (sol/sağ ayrımlı) bir kodlama daha iyi mi?"),
           "seal": "YALNIZ train(fit) + val. KİLİTLİ TEST KAPALI. Ana checkpoint'lere dokunulmadı.",
           "protocol": {"stage1": MAIN_TAG, "features": "1600 TDA (sabit) + kol-özgü konum",
                        "hgb": "max_iter=200 lr=0.08 balanced seed=42",
                        "stage1_score_included": False, "w_fusion": W,
                        "box_px": int(2 * BOX_HALF), "z_ref": "her kol için val'den donduruldu"},
           "feature_names": {"B": SPA_B_NAMES, "C": SPA_C_NAMES},
           "counts": {"fit_patients": len(d["tr_pids"]), "fit_cands": int(len(tr["y"])),
                      "val_patients": len(va_pids), "val_cands": int(len(va["y"])),
                      "val_lesions": int(n_les)},
           "arms": {}}

    for name, (Xtr, Xva) in arms.items():
        t0 = time.time()
        sc = StandardScaler().fit(Xtr)
        hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                             class_weight="balanced",
                                             random_state=SEED).fit(sc.transform(Xtr),
                                                                    tr["y"])
        p_tda = hgb.predict_proba(sc.transform(Xva))[:, 1].astype(np.float32)
        s1 = va["s1"]
        ref = {"s1_mean": float(s1.mean()), "s1_std": float(s1.std()),
               "tda_mean": float(p_tda.mean()), "tda_std": float(p_tda.std()), "w": W}
        fus = k.fuse(s1, p_tda, ref, W)

        def official(scores, sigmoid=False):
            arr = 1.0 / (1.0 + np.exp(-scores)) if sigmoid else scores
            flat = [(float(arr[i]), va_pids[p_i], float(va["cx"][i]), float(va["cy"][i]), 0.0)
                    for p_i, i in _iter_index(va["off"])]
            return official_metric(flat, len(va_pids), n_les, offline["gts"],
                                   offline["is_nod"], va_pids, box_half=BOX_HALF)

        m_s1 = official(s1)
        m_tda = official(p_tda)
        m_fus = official(fus, sigmoid=True)
        rep["arms"][name] = {"n_features": int(Xtr.shape[1]),
                             "stage1_only": m_s1, "tda_only": m_tda, "fusion": m_fus,
                             "z_ref": ref, "seconds": round(time.time() - t0, 1)}
        print(f"\n  [{name}] ({Xtr.shape[1]} özellik)", flush=True)
        print(f"    TDA-tek  AUC={m_tda['auc']} rank={m_tda['rank_metric']} "
              f"S@.25={m_tda['sens']['0.25']}", flush=True)
        print(f"    FÜZYON   AUC={m_fus['auc']} rank={m_fus['rank_metric']} "
              f"S@.25={m_fus['sens']['0.25']} S@.5={m_fus['sens']['0.5']}", flush=True)

    base = rep["arms"]["B_1604_geometrik_vekil"]
    rep["delta_vs_B"] = {
        name: {"tda_auc": _d(a["tda_only"]["auc"], base["tda_only"]["auc"]),
               "fusion_auc": _d(a["fusion"]["auc"], base["fusion"]["auc"]),
               "fusion_rank": _d(a["fusion"]["rank_metric"], base["fusion"]["rank_metric"]),
               "fusion_S025": _d(a["fusion"]["sens"]["0.25"], base["fusion"]["sens"]["0.25"])}
        for name, a in rep["arms"].items() if name != "B_1604_geometrik_vekil"}
    print(f"\n  Δ vs Kol B: {json.dumps(rep['delta_vs_B'], ensure_ascii=False)}", flush=True)

    # ---- karar ----
    c = rep["delta_vs_B"]["C_1606_anatomik_konum"]
    a = rep["delta_vs_B"]["A_1600_konum_yok"]
    rep["verdict"] = {
        "spatial_features_help": (None if a["fusion_auc"] is None
                                  else ("EVET" if a["fusion_auc"] < -0.002 else
                                        ("HAYIR" if a["fusion_auc"] > 0.002 else "NÖTR"))),
        "anatomical_better_than_proxy": (None if c["fusion_auc"] is None
                                         else ("EVET" if c["fusion_auc"] > 0.002 else
                                               ("HAYIR" if c["fusion_auc"] < -0.002 else "NÖTR"))),
        "keep_main_checkpoint": True,
        "note": ("Karar eşiği ±0,002 AUC (gürültü bandı). Ana checkpoint'ler DEĞİŞTİRİLMEDİ; "
                 "bu betik yalnız ölçüm yapar."),
    }
    print(f"\n  HÜKÜM: konum özellikleri → {rep['verdict']['spatial_features_help']} | "
          f"anatomik > vekil → {rep['verdict']['anatomical_better_than_proxy']}", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


def _iter_index(off):
    for i in range(len(off) - 1):
        for j in range(off[i], off[i + 1]):
            yield i, j


def _d(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


if __name__ == "__main__":
    main()
