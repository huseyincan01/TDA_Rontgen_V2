"""V6.9s — Kutu-içi doğrusal birleştirmede TDA ağırlığı (w_tda) ablasyonu.

Kutu skoru:  z_loc = (1 - w_tda) * z(s1) + w_tda * z(p_TDA)   (z-ref: kilitli fusion_ref)
Hasta skoru: max_i z_loc  (Top-1, resmî).

HİÇBİR model yeniden eğitilmez; yalnız hazır önbellekler (kutu koordinatları, s1, TDA özellikleri)
ve kilitli 2. aşama paketleri okunur. Kilitli ağırlık w_tda = 0,45 (w_s1 = 0,55) KORUNUR; bu betik
yalnız ablasyon kaydı üretir. Kümeler: NODE21 Val, NODE21 Test (kilitli), 242 Saf JSRT.

Mekanizma tanılaması (aynı çalıştırmada): her görüntüde birleşik skorla seçilen 1. kutunun
1. aşama sıralamasındaki yeri (s1-rank) ve pozitiflerde 1. kutunun gerçek nodüle isabet oranı.

Çıktı: artifacts/v6_two_stage/v6_9s_weight_sweep.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import common  # noqa: E402
from v6_6_optimization_audit import iou, official_metric  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
BUNDLE_MAIN = CKPT / "stage2_foundation_tda_hgb_pure1600.joblib"
BUNDLE_R = CKPT / "stage2_nojsrt_full_tda_hgb_pure1600.joblib"
BOX_HALF = 28.0
W_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LOCKED_W = 0.45
OUT = ART / "v6_9s_weight_sweep.json"


def _split(z, tag, p_tda, pids_key=None):
    pids = [str(p) for p in z[pids_key or f"{tag}_pids"]]
    off = z[f"{tag}_off"]
    s1 = z[f"{tag}_s1"].astype(np.float32)
    cx = z[f"{tag}_cx"].astype(np.float32)
    cy = z[f"{tag}_cy"].astype(np.float32)
    cands, scored = {}, {}
    for i, p in enumerate(pids):
        lo, hi = int(off[i]), int(off[i + 1])
        cands[p] = [(float(cx[j]), float(cy[j])) for j in range(lo, hi)]
        scored[p] = (s1[lo:hi], p_tda[lo:hi])
    return pids, cands, scored


def _p_tda(z, tag, bundle):
    return bundle["hgb"].predict_proba(bundle["scaler"].transform(z[f"{tag}_tda"].astype(np.float32)))[:, 1].astype(np.float32)


def load_sets() -> list[dict]:
    sets = []
    bm = joblib.load(BUNDLE_MAIN)
    for name, cache, tag in (("NODE21_Val", "_v6_9l_cache.npz", "va"), ("NODE21_Test", "_v6_9m_test_cache.npz", "te")):
        z = np.load(ART / cache, allow_pickle=True)
        pids, c, s = _split(z, tag, _p_tda(z, tag, bm))
        sets.append(dict(name=name, pids=pids, cands=c, scored=s, ref=bm["fusion_ref"]))
    br = joblib.load(BUNDLE_R)
    z = np.load(ART / "_v6_9r_cache.npz", allow_pickle=True)
    pids, c, s = [], {}, {}
    for tag in ("jp", "jn"):
        pp, cc, ss = _split(z, tag, _p_tda(z, tag, br))
        pids += pp
        c.update(cc)
        s.update(ss)
    sets.append(dict(name="JSRT_242", pids=pids, cands=c, scored=s, ref=br["fusion_ref"]))
    for d in sets:
        d["gts"] = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in common.load_gt(p)] for p in d["pids"]}
        d["is_nod"] = {p: bool(d["gts"][p]) for p in d["pids"]}
    return sets


def evaluate(d: dict, w_tda: float) -> dict:
    ref, gts, pids = d["ref"], d["gts"], d["pids"]
    n_les = sum(len(gts[p]) for p in pids)
    flat, hit, n_pos, low_rank, n_img, rank_sum = [], 0, 0, 0, 0, 0.0
    for p in pids:
        s1, td = d["scored"][p]
        if not len(s1):
            continue
        z1 = (s1 - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
        zt = (td - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
        zl = (1.0 - w_tda) * z1 + w_tda * zt
        top = int(np.argmax(zl))
        s1_rank = int((s1 > s1[top]).sum())  # 0 = 1. aşamanın da en iyi kutusu
        n_img += 1
        rank_sum += s1_rank
        low_rank += int(s1_rank >= 20)
        if d["is_nod"][p]:
            n_pos += 1
            cx, cy = d["cands"][p][top]
            b = (cx - BOX_HALF, cy - BOX_HALF, cx + BOX_HALF, cy + BOX_HALF)
            hit += int(any(iou(b, g) > 0.2 for g in gts[p]))
        for i in range(len(s1)):
            cx, cy = d["cands"][p][i]
            flat.append((float(1.0 / (1.0 + np.exp(-zl[i]))), p, cx, cy, 0.0))
    r = official_metric(flat, len(pids), n_les, gts, d["is_nod"], pids, box_half=BOX_HALF)
    return {
        "w_tda": w_tda,
        "w_s1": round(1 - w_tda, 2),
        "patient_auc": round(float(r["auc"]), 4),
        "S@0.125": round(float(r["sens"]["0.125"]), 4),
        "S@0.25": round(float(r["sens"]["0.25"]), 4),
        "S@0.5": round(float(r["sens"]["0.5"]), 4),
        "rank_metric": round(float(r["rank_metric"]), 4),
        "top1_box_hit_rate": round(hit / max(1, n_pos), 4),
        "top1_box_s1_rank_ge20_frac": round(low_rank / max(1, n_img), 4),
        "top1_box_mean_s1_rank": round(rank_sum / max(1, n_img), 2),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    sets = load_sets()
    out = {
        "experiment": "V6.9s_weight_sweep",
        "locked_w_tda": LOCKED_W,
        "note": "Yeniden eğitim yok; kilitli w_tda=0.45 korunur. Yalnız ablasyon kaydı.",
        "mean_candidates_per_image": {d["name"]: round(float(np.mean([len(d["cands"][p]) for p in d["pids"]])), 1) for d in sets},
        "results": {d["name"]: [evaluate(d, w) for w in W_GRID] for d in sets},
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'w_tda':>5} | {'küme':11} | AUC    S@.25  S@.50  rank   top1hit s1rank>=20  meanS1rank")
    for w_i, w in enumerate(W_GRID):
        for d in sets:
            r = out["results"][d["name"]][w_i]
            print(f"{w:5.2f} | {d['name']:11} | {r['patient_auc']:.4f} {r['S@0.25']:.4f} {r['S@0.5']:.4f} "
                  f"{r['rank_metric']:.4f} {r['top1_box_hit_rate']:.4f}  {r['top1_box_s1_rank_ge20_frac']:.4f}      {r['top1_box_mean_s1_rank']}")
    print("mean cands:", out["mean_candidates_per_image"])
    print(f"Süre: {time.time() - t0:.1f} sn → {OUT}")


if __name__ == "__main__":
    main()

