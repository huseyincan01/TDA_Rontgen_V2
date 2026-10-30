"""V6.9q — JSRT hasta AUC'sinin KONTROL HAVUZUNA duyarlılığı (eğitim YOK).

ŞÜPHE
-----
JSRT'nin 149 görüntüsünün **tamamı pozitif** (label=1). Hasta AUC'si için negatifler zorunlu
olduğundan `node21_control` havuzundan 250 kontrol kullanılıyor. Ancak model eğitimde
`node21_control` dağılımını **bastırmayı öğrendi**, JSRT arka planını hiç görmedi ⇒ bu asimetri
AUC'yi (özellikle 50 kutuyu birleştiren **noisy-OR**'u) yapay olarak şişirebilir.

BU BETİK
--------
Yeni model eğitmez; mevcut V6.9k4 önbelleğinden (`_v6_9k4_cache.npz`) yalnız **skorları** alır ve
üç şeyi ölçer:

  A) **KONTROL HAVUZUNDAN BAĞIMSIZ** metrikler (JSRT'nin kendi içinde):
     Recall@K, aday düzeyi AUC (TP adaylar vs aynı görüntünün diğer adayları), top-1'in TP olma oranı.
  B) **Kontrol havuzu duyarlılığı**: tüm 250 kontrol · yalnız **spacing ≥ 0,28 mm/px**
     (JSRT'nin fiziksel piksel boyutuna benzeyen) kontroller.
  C) **Hasta-içi arka plan normalizasyonu**: her hastanın 50 aday skoru kendi medyan/std'si ile
     z-normalize edilip öyle birleştirilir ⇒ hastalar arası kalibrasyon farkı kalkar.

Kullanım: python pipelines/v6_two_stage/v6_9q_jsrt_control_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_9k_leave_jsrt_out as k  # noqa: E402
from v6_6_optimization_audit import iou  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CACHE = ART / "_v6_9k4_cache.npz"
BUNDLE = ART / "checkpoints" / "stage2_nojsrt_tda_hgb_pure1600.joblib"
OUT = ART / "v6_9q_jsrt_control_sensitivity.json"
BOX_HALF = 28.0
SPACING_MIN = 0.28
N_CTRL = 250
KS = (10, 20, 50)


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return None
    o = np.argsort(-s)
    y2 = y[o]
    tpr = np.concatenate([[0], np.cumsum(y2) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - y2) / N])
    return round(float(np.trapezoid(tpr, fpr)), 4)


def spacing_of(pid: str) -> float:
    i = json.loads((common.PATIENTS_DIR / pid / "patient_info.json").read_text(encoding="utf-8"))
    s = i.get("spacing", [1.0, 1.0])
    return float(max(s[0], s[1]))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import joblib

    z = np.load(CACHE, allow_pickle=True)
    b = joblib.load(BUNDLE)
    js_pids = [str(p) for p in z["js_pids"]]
    va_pids = [str(p) for p in z["va_pids"]]
    print(f"JSRT {len(js_pids)} · val havuzu {len(va_pids)}", flush=True)

    def per_patient(tag, pids):
        off = z[f"{tag}_off"]
        s1 = z[f"{tag}_s1"].astype(np.float32)
        p_tda = b["hgb"].predict_proba(
            b["scaler"].transform(z[f"{tag}_tda"].astype(np.float32)))[:, 1]
        fus = k.fuse(s1, p_tda, b["fusion_ref"], b["fusion_ref"]["w"])
        cx, cy = z[f"{tag}_cx"], z[f"{tag}_cy"]
        out = {}
        for i, p in enumerate(pids):
            lo, hi = int(off[i]), int(off[i + 1])
            out[p] = {"s1": s1[lo:hi], "fus": fus[lo:hi],
                      "cx": cx[lo:hi], "cy": cy[lo:hi]}
        return out

    js = per_patient("js", js_pids)
    va = per_patient("va", va_pids)
    gts = {p: common.load_gt(p) for p in js_pids}
    gts_box = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in gts[p]]
               for p in js_pids}

    # ---------------- A) kontrol havuzundan BAĞIMSIZ ----------------
    n_les = sum(len(g) for g in gts.values())
    rec = {}
    for rk in ("s1", "fus"):
        rec[rk] = {}
        for K in KS:
            hit = 0
            for p in js_pids:
                top = np.argsort(-js[p][rk])[:K]
                for g in common.load_gt(p):
                    if any(common.hit_criterion1(float(js[p]["cx"][i]),
                                                 float(js[p]["cy"][i]), g) for i in top):
                        hit += 1
            rec[rk][f"K{K}"] = round(hit / max(1, n_les), 4)
    # top-1 kutusu gerçekten tümörün üstünde mi (IoU>0,2)
    top1_tp = 0
    for p in js_pids:
        i = int(np.argmax(js[p]["fus"]))
        box = (float(js[p]["cx"][i]) - BOX_HALF, float(js[p]["cy"][i]) - BOX_HALF,
               float(js[p]["cx"][i]) + BOX_HALF, float(js[p]["cy"][i]) + BOX_HALF)
        if any(iou(box, g) > 0.2 for g in gts_box[p]):
            top1_tp += 1
    # aday düzeyi AUC (aynı görüntünün TP adayları vs diğer adaylar) — kontrol havuzu YOK
    cand = {}
    for key in ("s1", "fus"):
        y, s = [], []
        for p in js_pids:
            labs = [1 if any(common.hit_criterion1(float(cx_), float(cy_), g)
                             for g in common.load_gt(p)) else 0
                    for cx_, cy_ in zip(js[p]["cx"], js[p]["cy"])]
            y.extend(labs)
            s.extend(js[p][key].tolist())
        cand[key] = auc(y, s)
    control_free = {"n_patients": len(js_pids), "n_lesions": n_les,
                    "recall_at_K": rec,
                    "top1_box_is_tumor_rate": round(top1_tp / len(js_pids), 4),
                    "candidate_auc_within_jsrt": cand,
                    "note": "Bu metriklerde node21_control HİÇ kullanılmaz."}

    # ---------------- B/C) kontrol havuzu duyarlılığı ----------------
    ctrl_all = [p for p in va_pids if not common.load_gt(p)]
    rng = np.random.RandomState(42)
    ctrl = list(rng.permutation(ctrl_all)[:N_CTRL])
    sp_c = {p: spacing_of(p) for p in ctrl}
    ctrl_sp = [p for p in ctrl if sp_c[p] >= SPACING_MIN]
    sp_js = [spacing_of(p) for p in js_pids]
    print(f"  kontroller: tümü {len(ctrl)} · spacing≥{SPACING_MIN}: {len(ctrl_sp)}", flush=True)

    def score(pid, store, kind, agg, selfnorm):
        v = store[pid][kind].astype(np.float64)
        if len(v) == 0:
            return 0.0
        if selfnorm:                                  # hasta-içi arka plan normalizasyonu
            v = (v - np.median(v)) / (v.std() + 1e-9)
            sig = 1.0 / (1.0 + np.exp(-v))
            return float(sig.max()) if agg == "top1" else \
                float(1 - np.prod(1 - np.clip(np.sort(sig)[::-1][:5], 0, 1)))
        if agg == "top1":
            return float(v.max())
        sig = 1.0 / (1.0 + np.exp(-v))                # füzyon z uzayında
        return float(1 - np.prod(1 - np.clip(np.sort(sig)[::-1][:5], 0, 1)))

    grids = {}
    for kind in ("s1", "fus"):
        for agg in ("top1", "noisyor"):
            for selfnorm in (False, True):
                for nm, pool in (("all250", ctrl), (f"spacing_ge_{SPACING_MIN}", ctrl_sp)):
                    key = f"{kind}|{agg}|{'selfnorm' if selfnorm else 'raw'}|{nm}"
                    y = [1] * len(js_pids) + [0] * len(pool)
                    s = ([score(p, js, kind, agg, selfnorm) for p in js_pids]
                         + [score(p, va, kind, agg, selfnorm) for p in pool])
                    grids[key] = {"auc": auc(y, s), "n_control": len(pool)}
    print("\n=== kontrol havuzu duyarlılığı (hasta AUC) ===", flush=True)
    for key, v in grids.items():
        print(f"  {key:52s} AUC={v['auc']}  (n_ctrl={v['n_control']})", flush=True)

    rep = {"experiment": "V6.9q-jsrt-control-sensitivity",
           "concern": ("JSRT'nin 149 görüntüsü de pozitif; AUC için kullanılan node21_control "
                       "havuzu modelin eğitimde bastırmayı öğrendiği bir dağılımdır ⇒ noisy-OR "
                       "şişebilir. Kontrol havuzundan bağımsız metrikler ve duyarlılık testi."),
           "model": "M5_nojsrt + stage2_nojsrt_tda_hgb_pure1600 (1600, saf topolojik)",
           "spacing": {"jsrt_p5_p50_p95": np.percentile(sp_js, [5, 50, 95]).round(3).tolist(),
                       "jsrt_ge_0.28_share": round(float(np.mean(np.array(sp_js) >= SPACING_MIN)), 3),
                       "control_ge_0.28_share": round(
                           float(np.mean([sp_c[p] >= SPACING_MIN for p in ctrl])), 3)},
           "A_control_free": control_free,
           "B_C_control_pool": grids,
           "verdict_note": ("Asıl güvenilir dış ölçüt kontrol havuzundan bağımsız olanlardır: "
                            "kutu yakalama (K10/K50) ve JSRT-içi aday AUC.")}
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kayıt: {OUT.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"\n=== KONTROL HAVUZUNDAN BAĞIMSIZ ===\n  Recall@K {rec} · top-1 tümörde "
          f"%{100*top1_tp/len(js_pids):.1f} · aday AUC {cand}", flush=True)


if __name__ == "__main__":
    main()
