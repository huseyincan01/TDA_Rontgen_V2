"""V6.9h — Yarıçap-uyarlamalı 2. aşama (temiz varyant): tahmin edilen yarıçap, 5. uzamsal özellik olarak.

SORU (kullanıcının yakaladığı nokta)
------------------------------------
V6.9f, dinamik yarıçapın yalnızca DIŞ çerçeveyi değiştirdiğini, 2. aşamanın TDA penceresinin
(32×32 + 64×64) hâlâ SABİT olduğunu gösterdi. Peki tahmin edilen yarıçap R̂, 2. aşamaya
**ek bir özellik** olarak verilirse (1600 PI + 4 uzamsal + 1 yarıçap = 1605 özellik) nodül
adayı ile yanlış-alarmı ayırmak iyileşir mi?

NEDEN BU VARYANT (riskli olanı DEĞİL)
-------------------------------------
* TDA penceresi DEĞİŞMEZ (32/64) ⇒ V5.1'in "bilinear resize TDA'yı bozar" tuzağına ve
  E3.6'nın "yama boyutu confound'u"na GİRMEYİZ.
* Yalnızca yarıçap (R̂) bir sayı olarak HGB'ye eklenir ⇒ sıfır interpolasyon, sıfır boyut değişimi.
* R̂, 1. aşamanın (donuk) yarıçap kafasından gelir; aday başına tek skaler.

PROTOKOL
--------
* Model: M5_foundation + V6.8 radius_head (R>0 maske). Adaylar: heat-map TTA, NMS 8 px, top-50.
* Eğitim: train split'ten 400 nodüllü + 400 kontrol (v6_2 ile aynı, seed 42) — HGB class_weight balanced.
* Değerlendirme: **yalnız VAL** (test bölmesi 3 kez kullanıldı — KAPALI).
* Karşılaştırma: aynı adaylar üzerinde HGB(1604) vs HGB(1605) → TDA-tek ve füzyon (w=0,55).

Kullanım: python pipelines/v6_two_stage/v6_9h_radius_feature.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, official_metric, pick_peaks  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
WORK = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9h_radius_feature.json"
SCALES, WIN, NMS_PX, TOPK = (0.9, 1.0, 1.1), 3, 8, 50
BOX_HALF, W = 28.0, 0.55


class RadiusHead(nn.Module):
    def __init__(self, in_ch=64, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, f):
        return self.net(f)


def _candidates(model, pids, device, tag):
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s), float(r)) for cx, cy, s, r in cs]
        if (i + 1) % 150 == 0:
            print(f"    {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _features(pids, cands, ranges):
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(pid, [(cx, cy, s) for cx, cy, s, _r in cands[pid]], ranges, False) for pid in pids]
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=4):
            out[pid] = res
    print(f"    özellikler {len(pids)} hasta ({time.time() - t0:.0f}s)", flush=True)
    return out


def _matrix(pids, cands, f, labs):
    tda, spa, rad, y, s1 = [], [], [], [], []
    for pid in pids:
        if f[pid] is None:
            continue
        for i, (cx, cy, s, r) in enumerate(cands[pid]):
            tda.append(f[pid]["tda"][i])
            spa.append(f[pid]["spa"][i])
            rad.append(float(max(0.0, r)))      # yarıçap ≥ 0 (negatif ise 0)
            y.append(1 if labs[pid][i] is not None else 0)
            s1.append(s)
    return (np.asarray(tda, np.float32), np.asarray(spa, np.float32),
            np.asarray(rad, np.float32), np.asarray(y, np.int64), np.asarray(s1, np.float32))


def _flat(pids, cands, s1, p_tda, key):
    """key: 's1', 'tda', 'fus' → flat [(score, pid, cx, cy, 0)] (fus için z-ref gerekli)."""
    zr_s1 = (s1.mean(), s1.std())
    zr_td = (p_tda.mean(), p_tda.std())
    flat, idx = [], 0
    for pid in pids:
        for (cx, cy, _, _) in cands[pid]:
            a, b = s1[idx], p_tda[idx]
            if key == "s1":
                sc = a
            elif key == "tda":
                sc = b
            else:
                z = W * (a - zr_s1[0]) / (zr_s1[1] + 1e-9) + (1 - W) * (b - zr_td[0]) / (zr_td[1] + 1e-9)
                sc = 1.0 / (1.0 + np.exp(-z))
            flat.append((float(sc), pid, cx, cy, 0.0))
            idx += 1
    return flat


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9h — yarıçap 5. özellik olarak 2. aşamaya (temiz varyant)", flush=True)

    model = load_foundation("M5_foundation", device)
    head = RadiusHead(64, 32).to(device)
    head.load_state_dict(torch.load(WORK / "radius_head_weight1_center_only.pt",
                                    map_location=device))
    head.eval()
    model.radius = head
    model.radius_detach = True
    for p in model.parameters():
        p.requires_grad = False
    import joblib
    ranges = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")["ranges"]
    print("  model + yarıçap kafası + ranges yüklendi", flush=True)

    # ---- veri
    tr_nod = p2._nodule_pids("train")
    tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(42)
    tr_pids = list(rng.choice(tr_nod, 400, replace=False)) + \
              list(rng.choice(tr_ctl, 400, replace=False))
    va_pids = list(common.split_patients()["val"])
    print(f"  train {len(tr_pids)} · val {len(va_pids)}", flush=True)

    print("[1] Adaylar...", flush=True)
    c_tr = _candidates(model, tr_pids, device, "train")
    c_va = _candidates(model, va_pids, device, "val")

    gt_tr = {p: common.load_gt(p) for p in tr_pids}
    gt_va = {p: common.load_gt(p) for p in va_pids}
    lab_tr = {p: s2._labels_for([(cx, cy, s) for cx, cy, s, _ in c_tr[p]], gt_tr[p]) for p in tr_pids}
    lab_va = {p: s2._labels_for([(cx, cy, s) for cx, cy, s, _ in c_va[p]], gt_va[p]) for p in va_pids}

    print("[2] Özellikler (TDA 32/64 + uzamsal)...", flush=True)
    f_tr = _features(tr_pids, c_tr, ranges)
    f_va = _features(va_pids, c_va, ranges)

    tda_tr, spa_tr, rad_tr, y_tr, s1_tr = _matrix(tr_pids, c_tr, f_tr, lab_tr)
    tda_va, spa_va, rad_va, y_va, s1_va = _matrix(va_pids, c_va, f_va, lab_va)
    print(f"  train: {len(y_tr)} aday, %{100 * y_tr.mean():.2f} pozitif | "
          f"val: {len(y_va)} aday, %{100 * y_va.mean():.2f} pozitif", flush=True)

    X4_tr = np.concatenate([tda_tr, spa_tr], 1)
    X5_tr = np.concatenate([tda_tr, spa_tr, rad_tr[:, None]], 1)
    X4_va = np.concatenate([tda_va, spa_va], 1)
    X5_va = np.concatenate([tda_va, spa_va, rad_va[:, None]], 1)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    print("[3] HGB eğitimi (1604 vs 1605, class_weight balanced)...", flush=True)
    def train(X_tr, X_va, y):
        sc = StandardScaler().fit(X_tr)
        h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                           class_weight="balanced", random_state=42).fit(
            sc.transform(X_tr), y)
        return sc, h, h.predict_proba(sc.transform(X_va))[:, 1]

    sc4, hgb4, p4 = train(X4_tr, X4_va, y_tr)
    sc5, hgb5, p5 = train(X5_tr, X5_va, y_tr)

    gts = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in gt_va[p]]
           for p in va_pids}
    is_nod = {p: len(gts[p]) > 0 for p in va_pids}
    n_les = sum(len(g) for g in gts.values())

    print("[4] Resmî metrik (val)...", flush=True)
    res = {}
    for name, p in (("tda_1604", p4), ("tda_1605", p5)):
        f_s1 = _flat(va_pids, c_va, s1_va, p, "s1")
        f_td = _flat(va_pids, c_va, s1_va, p, "tda")
        f_fu = _flat(va_pids, c_va, s1_va, p, "fus")
        res[name] = {
            "stage1_only": official_metric(f_s1, len(va_pids), n_les, gts, is_nod, va_pids,
                                           box_half=BOX_HALF),
            "tda_only": official_metric(f_td, len(va_pids), n_les, gts, is_nod, va_pids,
                                        box_half=BOX_HALF),
            "fusion": official_metric(f_fu, len(va_pids), n_les, gts, is_nod, va_pids,
                                      box_half=BOX_HALF),
        }
        print(f"  {name:10s} TDA-tek AUC={res[name]['tda_only']['auc']} | "
              f"füzyon rank={res[name]['fusion']['rank_metric']} "
              f"AUC={res[name]['fusion']['auc']} "
              f"S@.25={res[name]['fusion']['sens']['0.25']}", flush=True)

    rep = {
        "experiment": "V6.9h-radius-as-feature",
        "question": "Tahmin edilen yarıçap R̂, 5. uzamsal özellik olarak HGB'ye verilirse 2. aşama iyileşir mi?",
        "protocol": "TDA penceresi SABİT (32/64); yalnızca R̂ skaleri eklenir (1605 özellik). train 400+400, val 734. Test KAPALI (3. kullanım geçildi).",
        "n_train_cands": int(len(y_tr)), "n_val_cands": int(len(y_va)),
        "train_pos_ratio": round(float(y_tr.mean()), 4),
        "val_pos_ratio": round(float(y_va.mean()), 4),
        "results": res,
        "delta": {
            "fusion_rank": round((res["tda_1605"]["fusion"]["rank_metric"] or 0)
                                 - (res["tda_1604"]["fusion"]["rank_metric"] or 0), 4),
            "fusion_auc": round((res["tda_1605"]["fusion"]["auc"] or 0)
                                - (res["tda_1604"]["fusion"]["auc"] or 0), 4),
            "tda_auc": round((res["tda_1605"]["tda_only"]["auc"] or 0)
                             - (res["tda_1604"]["tda_only"]["auc"] or 0), 4),
        },
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Δ(füzyon rank) = {rep['delta']['fusion_rank']:+.4f} | "
          f"Δ(füzyon AUC) = {rep['delta']['fusion_auc']:+.4f} | "
          f"Δ(TDA AUC) = {rep['delta']['tda_auc']:+.4f}", flush=True)
    print(f"  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
