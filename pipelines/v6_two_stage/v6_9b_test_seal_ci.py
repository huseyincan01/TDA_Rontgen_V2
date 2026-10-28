"""V6.9b — Test mührü için belirsizlik aralığı (hasta düzeyi bootstrap).

MÜHÜRÜ BOZMAZ: hiçbir parametre seçilmez, hiçbir eşik değiştirilmez. Aynı kilitli
protokol **yeniden koşulur** (deterministik) ve ürettiği toplam metrikler mühür
kaydıyla (v6_9_test_seal.json) **birebir** karşılaştırılır — uyuşmazsa betik durur.
Sonra 734 test hastası yerine-yerine örneklenerek (1.000 tekrar) rank_metric / AUC /
duyarlılık için %95 aralık hesaplanır.

Neden gerekli: 229 lezyonda Sens@0,25'in standart hatası ≈ 0,03 ⇒ val–test farkının
(−0,029) "seçim yanlılığı" mı "örnekleme gürültüsü" mü olduğu ancak aralıkla söylenebilir.

Kullanım: python pipelines/v6_two_stage/v6_9b_test_seal_ci.py --reps 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_9_test_seal as seal  # noqa: E402
from v6_6_optimization_audit import iou  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

SEALED = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9_test_seal.json"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9b_test_seal_ci.json"
FROC = (0.125, 0.25, 0.5)


def _metric_slots(slots) -> dict:
    """slots: [(gt_boxes, is_nodule, cands[(score,cx,cy)])] → resmî metrik.

    `official_metric` ile aynı eşleştirme mantığı (IoU>0,2; çakışan 2.+ ne TP ne FP),
    ama hasta yerine **örnekleme yuvası** üzerinde çalışır ⇒ bootstrap mümkün.
    """
    flat = []
    for si, (_gt, _isn, cands) in enumerate(slots):
        for sc, cx, cy in cands:
            flat.append((sc, si, cx, cy))
    flat.sort(key=lambda t: -t[0])
    matched = [set() for _ in slots]
    fp, tp = 0, 0
    pts, img_score = [], {}
    n_les = sum(len(s[0]) for s in slots)
    for sc, si, cx, cy in flat:
        gt = slots[si][0]
        img_score[si] = max(img_score.get(si, 0.0), sc)
        box = (cx - seal.BOX_HALF, cy - seal.BOX_HALF,
               cx + seal.BOX_HALF, cy + seal.BOX_HALF)
        best, hit = 0.0, None
        for gi, g in enumerate(gt):
            v = iou(box, g)
            if v > best:
                best, hit = v, gi
        if hit is not None and best > 0.2:
            if hit in matched[si]:
                continue
            matched[si].add(hit)
            tp += 1
        else:
            fp += 1
        pts.append((fp / len(slots), tp / max(1, n_les)))
    sens = {}
    for t in FROC:
        cand = [sn for f, sn in pts if f <= t + 1e-9]
        sens[f"{t:g}"] = float(max(cand) if cand else 0.0)
    y = np.array([1.0 if s[1] else 0.0 for s in slots])
    sc = np.array([img_score.get(i, 0.0) for i in range(len(slots))])
    order = np.argsort(-sc)
    yy = y[order]
    P, N = yy.sum(), len(yy) - yy.sum()
    auc = None
    if P > 0 and N > 0:
        tpr = np.concatenate([[0], np.cumsum(yy) / P])
        fpr = np.concatenate([[0], np.cumsum(1 - yy) / N])
        auc = float(np.trapezoid(tpr, fpr))
    return {"auc": auc, "sens": sens,
            "rank_metric": (0.75 * auc + 0.25 * sens["0.25"]) if auc else None}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    if not sealed.get("sealed"):
        raise SystemExit("Mühür kaydı yok/geçersiz — önce v6_9_test_seal.py koşmalı.")
    ref = sealed["frozen_z_reference"]
    print("V6.9b — test mührü belirsizlik aralığı")
    print(f"  dondurulmuş z referansı (val'den): {ref}", flush=True)

    import joblib
    bundle = joblib.load(seal.CKPT / seal.STAGE2_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_foundation(seal.STAGE1_TAG, device)
    sp = common.split_patients()
    test_pids = list(sp["test"])

    recs = seal._pass(test_pids, model, device, bundle["ranges"], bundle, "TEST")

    # ---- mühürle birebir uyuşma kontrolü (mühür bozulmadı)
    a1 = seal._official(recs, test_pids, "s1")
    a2 = seal._official(recs, test_pids, "fus", ref)
    for name, got, want in (("A1", a1, sealed["test"]["A1_stage1_only"]),
                            ("A2", a2, sealed["test"]["A2a_fusion_tda"])):
        for k in ("rank_metric", "auc"):
            if abs((got[k] or 0) - (want[k] or 0)) > 1e-6:
                raise SystemExit(f"MÜHÜR UYUŞMAZLIĞI {name}.{k}: {got[k]} vs {want[k]}")
        print(f"  ✔ {name} mühürle aynı: rank={got['rank_metric']} auc={got['auc']}",
              flush=True)

    # ---- yuvarlar
    slots = []
    for pid in test_pids:
        r = recs[pid]
        gt = [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in r["gts"]]
        s1 = [(float(r["s1"][i]), float(r["cs"][i][0]), float(r["cs"][i][1]))
              for i in range(len(r["cs"]))]
        fus_sc = seal._fuse(r, ref, seal.W_FUSION)
        fus = [(float(1.0 / (1.0 + np.exp(-fus_sc[i]))), float(r["cs"][i][0]),
                float(r["cs"][i][1])) for i in range(len(r["cs"]))]
        slots.append({"pid": pid, "gt": gt, "is_nodule": r["is_nodule"],
                      "s1": s1, "fus": fus})

    rng = np.random.RandomState(args.seed)
    n = len(slots)
    out = {}
    for key in ("s1", "fus"):
        base = _metric_slots([(s["gt"], s["is_nodule"], s[key]) for s in slots])
        boot = {"rank_metric": [], "auc": [], **{f"sens_{t:g}": [] for t in FROC}}
        for _ in range(args.reps):
            idx = rng.randint(0, n, n)
            m = _metric_slots([(slots[i]["gt"], slots[i]["is_nodule"], slots[i][key])
                               for i in idx])
            if m["rank_metric"] is not None:
                boot["rank_metric"].append(m["rank_metric"])
            if m["auc"] is not None:
                boot["auc"].append(m["auc"])
            for t in FROC:
                boot[f"sens_{t:g}"].append(m["sens"][f"{t:g}"])
        ci = {}
        for k, v in boot.items():
            a = np.array(v)
            ci[k] = {"mean": round(float(a.mean()), 4),
                     "lo95": round(float(np.percentile(a, 2.5)), 4),
                     "hi95": round(float(np.percentile(a, 97.5)), 4)}
        out[key] = {"point": {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in base.items()}, "ci95": ci}
        print(f"\n  [{key}] rank_metric {base['rank_metric']:.4f} "
              f"(95% {ci['rank_metric']['lo95']:.4f}–{ci['rank_metric']['hi95']:.4f})"
              f" · AUC {base['auc']:.4f} "
              f"(95% {ci['auc']['lo95']:.4f}–{ci['auc']['hi95']:.4f})", flush=True)
        for t in FROC:
            k = f"sens_{t:g}"
            print(f"      S@{t:g} {base['sens'][f'{t:g}']:.4f} "
                  f"(95% {ci[k]['lo95']:.4f}–{ci[k]['hi95']:.4f})", flush=True)

    rep = {"experiment": "V6.9b-test-seal-ci",
           "seal_record": str(SEALED.relative_to(PROJECT_ROOT)),
           "assertion": "yeniden koşulan toplam metrikler mühür kaydıyla birebir aynı",
           "n_patients": n, "reps": args.reps, "seed": args.seed,
           "frozen_z_reference": ref, "results": out,
           "note": ("Hiçbir parametre seçilmedi; yalnız örnekleme belirsizliği ölçüldü. "
                    "rank_metric = 0,75·AUC + 0,25·Sens@0,25 (resmî NODE21).")}
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kayıt: {OUT}", flush=True)


if __name__ == "__main__":
    main()
