"""V6.6 Optimizasyon Denetimi — resmî NODE21 Detection metriği + NMS/tepe + çok ölçekli TTA.

YENİ EĞİTİM YOK. Yalnız mevcut `M5_foundation.pt` üzerinde çıkarım sonrası analiz.

RESMÎ NODE21 DETECTION PROTOKOLÜ (node21.grand-challenge.org/Details/ — birebir)
--------------------------------------------------------------------------------
* AUC: görüntü skoru = o görüntüdeki tahminlerin EN YÜKSEK olasılığı (tahmin yoksa 0).
* FROC eşleştirme: bir referans kutuyla **IoU > 0.2** çakışan birden çok tahmin varsa
  yalnız **en yüksek olasılıklı** olan tutulur; diğerleri ne TP ne FP sayılır.
  TP: referansla IoU > 0.2, aksi hâlde FP.
* Duyarlılık **3 ön-tanımlı FP/görüntü oranında**: 1/8 (0.125), 1/4 (0.25), 1/2 (0.5).
  (1.0 FP/görüntü resmî FROC ortalamasında YOKTUR.)
* Nihai sıralama:  S = Sens@0.25 FP/görüntü
                   **rank_metric = 0.75·AUC + 0.25·S**

Denetlenen optimizasyonlar
--------------------------
  1. Tepe penceresi (3x3 vs 5x5) + mesafe-tabanlı NMS yarıçapı (Δd taraması)
  2. Çok ölçekli TTA: {1.0} / {0.9,1.0} / {0.9,1.0,1.1} (her ölçekte + yatay çevirme)
  3. Resmî rank_metric simülasyonu (düşük-FP bölgesi)

Kutu sözleşmesi: CenterNet yarıçap kafası → kutu = (cx ± 4R, cy ± 4R) [1024 uzayı]
(R: 256 ızgaradaki yarıçap; 1 ızgara px = 4 görüntü px). Sabit 64x64 kutu kontrolü de var.

Kullanım:
    python pipelines/v6_two_stage/v6_6_optimization_audit.py
    python pipelines/v6_two_stage/v6_6_optimization_audit.py --limit 150   (hızlı)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
from detector_models_foundation import INPUT, canon_to_512, foundation_input  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_optimization_audit.json"
HMP = 256
FROC_OFFICIAL = (0.125, 0.25, 0.5)      # RESMÎ
FROC_LEGACY = (1.0, 2.0, 4.0)           # projenin eski protokolü (karşılaştırma)
SCALE_SETS = {"single_1.0": (1.0,), "multi_0.9_1.0": (0.9, 1.0),
              "multi_0.9_1.0_1.1": (0.9, 1.0, 1.1)}
PEAK_WINDOWS = (3, 5)
NMS_DIST = (8, 12, 16, 20, 24, 28)
BOX_HALVES = (20, 24, 28, 32, 36, 40, 48, 56)   # 1024 uzayı → 40..112 px kutu
TOP_PEAKS, TOP_KEEP = 400, 50


@torch.no_grad()
def heat_maps(model, c512: np.ndarray, device, scales):
    """Ölçek seti için 256x256 ısı ve yarıçap haritalarını ortalar (flip TTA dahil)."""
    heat = np.zeros((HMP, HMP), np.float32)
    rad = np.zeros((HMP, HMP), np.float32)
    for s in scales:
        size = int(round(INPUT * s))
        img = (c512 if size == INPUT else
               np.array(Image.fromarray(c512).resize((size, size),
                                                     Image.Resampling.BILINEAR)))
        for flip in (False, True):
            t = foundation_input(img, flip).unsqueeze(0).to(device)
            h, r = model(t)
            hh = torch.sigmoid(h)[0, 0].cpu().numpy()
            rr = r[0, 0].cpu().numpy()
            if flip:
                hh, rr = hh[:, ::-1], rr[:, ::-1]
            if hh.shape != (HMP, HMP):
                hh = np.array(Image.fromarray(hh).resize((HMP, HMP),
                                                         Image.Resampling.BILINEAR))
                rr = np.array(Image.fromarray(rr).resize((HMP, HMP),
                                                         Image.Resampling.BILINEAR))
            heat += hh
            rad += rr
    n = 2 * len(scales)
    return heat / n, rad / n


def pick_peaks(heat, rad, window, top=TOP_PEAKS, thr=0.01):
    """Yerel tepeler → (cx, cy, skor, r_1024). [1024 uzayı]

    Ölçek: ızgara px x → 512 uzayı (x·STRIDE + STRIDE/2) → 1024 uzayı ×2
           ⇒ cx = (x·2 + 1)·2 = 4x + 2.
    Yarıçap: R (ızgara) → r512 = R·2 → r1024 = R·4.
    """
    maxf = ndi.maximum_filter(heat, size=window)
    ys, xs = np.where((heat >= maxf) & (heat > thr))
    if len(ys) == 0:
        return []
    sc = heat[ys, xs]
    order = np.argsort(-sc)[:top]
    return [(float(xs[k] * 4 + 2), float(ys[k] * 4 + 2), float(sc[k]),
             float(rad[ys[k], xs[k]]) * 4.0) for k in order]


def dist_nms(cands, dd, top=TOP_KEEP):
    """Skor sıralı mesafe-tabanlı NMS: merkez mesafesi < dd olan düşük skorlular elenir."""
    grid, out = {}, []
    for cx, cy, s, r in cands:                       # cands zaten skor sıralı
        gx, gy = int(cx // dd), int(cy // dd)
        dup = False
        for dx in (-1, 0, 1):
            if dup:
                break
            for dy in (-1, 0, 1):
                for (ox, oy) in grid.get((gx + dx, gy + dy), ()):
                    if (cx - ox) ** 2 + (cy - oy) ** 2 < dd * dd:
                        dup = True
                        break
                if dup:
                    break
        if dup:
            continue
        grid.setdefault((gx, gy), []).append((cx, cy))
        out.append((cx, cy, s, r))
        if len(out) >= top:
            break
    return out


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def official_metric(flat, n_images, n_lesions, gt_of, is_nod_map, pids, box_half=32.0,
                    use_radius=False):
    """flat: [(score, pid, cx, cy, r)] → resmî FROC + AUC + rank_metric.

    box_half: sabit yarı-kenar (1024 uzayı). use_radius=True ise yarıçap kafası kullanılır.
    """
    boxes = []
    for i, (s, pid, cx, cy, r) in enumerate(flat):
        h = max(1.0, r) if use_radius else box_half
        boxes.append((s, pid, (cx - h, cy - h, cx + h, cy + h)))
    boxes.sort(key=lambda t: -t[0])
    matched = {p: set() for p in pids}
    fps, n_tp = 0, 0
    pts, img_score = [], {}
    for s, pid, box in boxes:
        img_score[pid] = max(img_score.get(pid, 0.0), s)
        best, hit = 0.0, None
        for gi, g in enumerate(gt_of[pid]):
            v = iou(box, g)
            if v > best:
                best, hit = v, gi
        if hit is not None and best > 0.2:
            if hit in matched[pid]:
                continue                              # çakışan 2.+ tahmin: ne TP ne FP
            matched[pid].add(hit)
            n_tp += 1
        else:
            fps += 1
        pts.append((fps / max(1, n_images), n_tp / max(1, n_lesions)))
    sens = {}
    for t in FROC_OFFICIAL + FROC_LEGACY:
        cand = [sn for fp, sn in pts if fp <= t + 1e-9]
        sens[f"{t:g}"] = round(max(cand) if cand else 0.0, 4)
    y = [1 if is_nod_map[p] else 0 for p in pids]
    sc = [img_score.get(p, 0.0) for p in pids]
    order = np.argsort(-np.array(sc))
    yy = np.array(y, float)[order]
    P, N = yy.sum(), len(yy) - yy.sum()
    auc = None
    if P > 0 and N > 0:
        tpr = np.concatenate([[0], np.cumsum(yy) / P])
        fpr = np.concatenate([[0], np.cumsum(1 - yy) / N])
        auc = round(float(np.trapezoid(tpr, fpr)), 4)
    froc3 = round(float(np.mean([sens[f"{t:g}"] for t in FROC_OFFICIAL])), 4)
    rank = round(0.75 * auc + 0.25 * sens["0.25"], 4) if auc is not None else None
    return {"sens": sens, "auc": auc, "froc_mean_3pt": froc3, "rank_metric": rank}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="M5_foundation")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.6 Optimizasyon Denetimi | ckpt={args.ckpt} | resmî NODE21 protokolü",
          flush=True)

    model = load_foundation(args.ckpt, device)
    val_all = common.split_patients()["val"]
    if args.limit:                       # dengeli alt küme (nodülsüz dilimde metrik tanımsız)
        nod = [p for p in val_all if common.load_gt(p)]
        ctl = [p for p in val_all if not common.load_gt(p)]
        h = args.limit // 2
        val_pids = nod[:h] + ctl[:args.limit - h]
    else:
        val_pids = val_all
    gts_map = {p: common.load_gt(p) for p in val_pids}
    gt_boxes = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in gts_map[p]]
                for p in val_pids}
    is_nod = {p: len(gts_map[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gts_map.values())
    print(f"  {len(val_pids)} hasta · {n_les} lezyon", flush=True)

    vkeys = [(sn, w, dd) for sn in SCALE_SETS for w in PEAK_WINDOWS for dd in NMS_DIST]
    vkeys += [("single_1.0", 3, 0)]
    flat = {k: [] for k in vkeys}
    kept = {k: 0 for k in vkeys}

    print("\n[1] Isı haritaları + tepe/NMS varyantları (hasta-hasta)...", flush=True)
    t0 = time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        c512 = canon_to_512(canon)
        for sname, scales in SCALE_SETS.items():
            heat, rad = heat_maps(model, c512, device, scales)
            for win in PEAK_WINDOWS:
                peaks = pick_peaks(heat, rad, win)
                for dd in NMS_DIST:
                    cs = dist_nms(peaks, dd)
                    kept[(sname, win, dd)] += len(cs)
                    flat[(sname, win, dd)].extend((s, pid, cx, cy, r) for cx, cy, s, r in cs)
                if sname == "single_1.0" and win == 3:
                    kept[("single_1.0", 3, 0)] += len(peaks)
                    flat[("single_1.0", 3, 0)].extend((s, pid, cx, cy, r)
                                                      for cx, cy, s, r in peaks)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[2] Resmî metrik — TAM IZGARA (varyant × kutu boyutu) ({time.time() - t0:.0f}s)...",
          flush=True)
    grid = {}
    for k in vkeys:
        for bh in BOX_HALVES:
            r = official_metric(flat[k], len(val_pids), n_les, gt_boxes, is_nod, val_pids,
                                box_half=float(bh))
            r["kept_per_image"] = round(kept[k] / max(1, len(val_pids)), 1)
            grid[(k, bh)] = r
    print(f"    {len(grid)} değerlendirme bitti ({time.time() - t0:.0f}s)", flush=True)

    # en iyi kutu boyutu (tüm varyantlar üzerinden)
    best_box = max(BOX_HALVES, key=lambda bh: max(grid[(k, bh)]["rank_metric"] or 0
                                                  for k in vkeys))
    print(f"\n  --- kutu boyutu {2 * best_box}px'te en iyi 10 varyant ---", flush=True)
    top = sorted(vkeys, key=lambda k: -(grid[(k, best_box)]["rank_metric"] or 0))[:10]
    for k in top:
        r = grid[(k, best_box)]
        print(f"  {str(k):30s} rank={r['rank_metric']} AUC={r['auc']} FROC3={r['froc_mean_3pt']}"
              f" | @.125={r['sens']['0.125']} @.25={r['sens']['0.25']} @.5={r['sens']['0.5']}"
              f" @1={r['sens']['1']} | n/img={r['kept_per_image']}", flush=True)

    # ölçek seti başına en iyi (çok ölçekli TTA sorusu)
    scale_best = {}
    for sname in SCALE_SETS:
        cands = [(k, bh) for (k, bh) in grid if k[0] == sname]
        bk = max(cands, key=lambda kb: grid[kb]["rank_metric"] or 0)
        scale_best[sname] = {"variant": list(bk[0]), "box_px": 2 * bk[1],
                             **grid[bk]}
    print("\n  --- ölçek seti başına en iyi (kutu dahil) ---", flush=True)
    for sname, v in scale_best.items():
        print(f"  {sname:20s} rank={v['rank_metric']} AUC={v['auc']} S@0.25={v['sens']['0.25']}"
              f" FROC3={v['froc_mean_3pt']} | {v['variant']} kutu={v['box_px']}px", flush=True)

    best_key = max(grid, key=lambda kb: grid[kb]["rank_metric"] or 0)
    best = best_key[0]

    print("\n[3] Yarıçap kafası kontrolü (resmî IoU>0.2)...", flush=True)
    ctrl_radius = official_metric(flat[("single_1.0", 3, 20)], len(val_pids), n_les,
                                  gt_boxes, is_nod, val_pids, use_radius=True)
    print(f"  yarıçap kafası kutusu  rank={ctrl_radius['rank_metric']} "
          f"S@0.25={ctrl_radius['sens']['0.25']}  (BOZUK)", flush=True)

    rep = {
        "experiment": "V6.6-optimization-audit",
        "protocol": {
            "source": "node21.grand-challenge.org/Details/ (resmî detection değerlendirmesi)",
            "matching": "IoU > 0.2; aynı referansa çakışan 2.+ tahmin ne TP ne FP",
            "froc_points_official": list(FROC_OFFICIAL),
            "froc_points_legacy": list(FROC_LEGACY),
            "rank_metric": "0.75·AUC + 0.25·Sens@0.25 FP/görüntü",
            "box_primary": "SABİT kare kutu (yarıçap kafası bozuk — bkz. radius_head_control)",
        },
        "n_patients": len(val_pids), "n_lesions": int(n_les),
        "best_box_half": int(best_box),
        "best_overall": {"variant": list(best), "box_px": 2 * best_key[1],
                         **grid[best_key]},
        "scale_set_best": scale_best,
        "top10_at_best_box": [{"variant": list(k), **grid[(k, best_box)]} for k in top],
        "grid": {"|".join(str(x) for x in k) + f"|box{2 * bh}": v
                 for (k, bh), v in grid.items()},
        "radius_head_control": ctrl_radius,
        "scale_sets": {k: list(v) for k, v in SCALE_SETS.items()},
        "peak_windows": list(PEAK_WINDOWS), "nms_distances": list(NMS_DIST),
        "box_halves": list(BOX_HALVES),
        "note": ("YENİ EĞİTİM YOK. Kural 9 geçerli: verimiz NODE21'in HALK AÇIK EĞİTİM "
                 "setidir; resmî test setleri özeldir ⇒ leaderboard'a gönderim/sıralama "
                 "MÜMKÜN DEĞİLDİR. Bu tablo yalnız kendi val bölmemizde resmî protokolün "
                 "simülasyonudur."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEN İYİ: {best_key[0]} kutu={2 * best_key[1]}px → "
          f"rank_metric={grid[best_key]['rank_metric']} "
          f"(S@0.25={grid[best_key]['sens']['0.25']}, AUC={grid[best_key]['auc']})", flush=True)
    print(f"Rapor: {OUT}", flush=True)


if __name__ == "__main__":
    main()
