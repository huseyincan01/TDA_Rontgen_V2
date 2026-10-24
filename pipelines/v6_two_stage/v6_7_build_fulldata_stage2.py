"""V6.7 — %100 Tam Veri AŞAMA-2: TDA-HGB_fulldata (TUZAK B KORUMALI).

Tuzak B (Stacking Overfitting) nedir ve nasıl korunuyor?
--------------------------------------------------------
2. aşama modelini, 1. aşamanın KENDİ eğitim verisi üzerinde ürettiği şişirilmiş
(in-sample) skorlarla eğitmek yığma-uyumu (stacking overfit) yaratır. Bu betik:

  (a) HGB'yi **yalnız saf topolojik/uzamsal özellikler** üzerinde eğitir:
      1600 PI (32/64 çift ölçekli H0/H1) + 4 uzamsal öncül. **1. aşama skoru
      (s1) matrise HİÇ girmez** ⇒ 1. aşamanın ezberlediği bilgi sızamaz.
  (b) Füzyon ağırlığını **YENİDEN EĞİTMEZ**. V6.6.6'da val üzerinde doğrulanmış ve
      miras alınan LR katsayılarını domine eden **sabit w = 0,55** kullanılır
      (S = w·z(s1) + (1−w)·z(p_tda), sigmoid ile [0,1]'e eşlenir). Sabit katsayı
      olduğu için hiçbir veri üzerinde yeniden uydurulmaz ⇒ Tuzak B imkânsız.

Çıktı: checkpoints/submission_fulldata/stage2_tda_hgb_fulldata.joblib
Bilimsel checkpoint'lere DOKUNULMAZ.

Kullanım:
    python pipelines/v6_two_stage/v6_7_build_fulldata_stage2.py
    python pipelines/v6_two_stage/v6_7_build_fulldata_stage2.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, pick_peaks  # noqa: E402

SUBDIR = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
OUT = SUBDIR / "stage2_tda_hgb_fulldata.joblib"
LOG = SUBDIR / "stage2_fulldata_build_log.json"

# ---- KİLİTLİ çıkarım standardı (V6.6.7 denetimiyle sabitlendi)
SCALES = (0.9, 1.0, 1.1)
PEAK_WINDOW = 3
NMS_PX = 8
TOP_K = 50
SEED = 42
NW = 10


def all_pids():
    sp = common.split_patients()
    return list(sp["train"]) + list(sp["val"]) + list(sp["test"])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="M5_foundation_fulldata")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=NW)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SUBDIR.mkdir(parents=True, exist_ok=True)
    print("V6.7 TAM VERİ Aşama-2 | Tuzak B korumalı (s1 matrise girmez)", flush=True)

    model = load_foundation(args.ckpt, device, subdir=True)
    pids = all_pids()
    if args.limit:
        pids = pids[:args.limit]
    print(f"  {len(pids)} hasta (train+val+test)", flush=True)

    # ---------------------------------------------------------------- adaylar
    # NOT: Aşama-2 fonksiyonları (derive_ranges / _patient_features / _labels_for)
    # 3'lü demet (cx, cy, skor) bekler ⇒ yarıçap bileşeni burada düşürülür.
    CACHE_CANDS = SUBDIR / "fulldata_candidates.pkl"
    if CACHE_CANDS.exists() and not args.limit:
        print(f"\n[1] Adaylar ÖNBELLEKTEN okunuyor: {CACHE_CANDS.name}", flush=True)
        cands = joblib.load(CACHE_CANDS)
        n_cand = sum(len(v) for v in cands.values())
        print(f"    {n_cand} aday · {len(cands)} hasta", flush=True)
    else:
        print(f"\n[1] Adaylar — KİLİTLİ standart: ölçek{SCALES} + flip, tepe {PEAK_WINDOW}x"
              f"{PEAK_WINDOW}, NMS {NMS_PX}px, top-{TOP_K}", flush=True)
        cands, t0 = {}, time.time()
        for i, pid in enumerate(pids):
            canon, _, _ = p2._prep(pid)
            heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
            pk = pick_peaks(heat, rad, PEAK_WINDOW)
            cands[pid] = [(cx, cy, s) for cx, cy, s, _ in dist_nms(pk, NMS_PX, top=TOP_K)]
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
        n_cand = sum(len(v) for v in cands.values())
        print(f"    {n_cand} aday ({time.time() - t0:.0f}s)", flush=True)
        joblib.dump(cands, CACHE_CANDS)
        print(f"    aday önbelleği: {CACHE_CANDS}", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- aralıklar
    print("\n[2] PI aralıkları (tam veri adaylarından)...", flush=True)
    nod = [p for p in pids if common.load_gt(p)][:60]
    ranges = s2.derive_ranges([(p, cands[p]) for p in nod])
    print(f"    {ranges}", flush=True)

    # ---------------------------------------------------------------- özellikler
    print(f"\n[3] Lokal TDA + uzamsal özellikler ({args.workers} süreç)...", flush=True)
    jobs = [(pid, cands[pid], ranges, False) for pid in pids]
    t0 = time.time()
    X_parts, y_parts, s1_parts, n_done = [], [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=4):
            n_done += 1
            if err or res is None or len(cands[pid]) == 0:
                continue
            gts = common.load_gt(pid)
            labs = s2._labels_for(cands[pid], gts)
            k = len(cands[pid])
            X_parts.append(np.concatenate([res["tda"][:k], res["spa"][:k]], 1))
            y_parts.append(np.array([0 if l is None else 1 for l in labs], np.int8))
            s1_parts.append(np.array([c[2] for c in cands[pid]], np.float32))
            if n_done % 500 == 0:
                print(f"    {n_done}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    s1 = np.concatenate(s1_parts).astype(np.float32)
    print(f"    matris {X.shape} | pozitif {int(y.sum())} (%{100 * y.mean():.2f}) "
          f"({time.time() - t0:.0f}s)", flush=True)
    assert X.shape[1] == 1604, f"beklenen 1604 özellik, gelen {X.shape[1]}"
    del X_parts, y_parts, s1_parts

    # ---------------------------------------------------------------- eğitim
    print("\n[4] StandardScaler + HGB (yalnız 1600 PI + 4 uzamsal; s1 YOK)...", flush=True)
    t0 = time.time()
    sc = StandardScaler().fit(X)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         max_depth=None, class_weight="balanced",
                                         random_state=SEED)
    hgb.fit(sc.transform(X), y)
    print(f"    HGB eğitildi ({time.time() - t0:.0f}s)", flush=True)

    # ---- KİLİTLİ füzyon referansı (sabit z-skor sabitleri).
    # HGB'ye s1 GİRMEZ; burada yalnız nihai doğrusal füzyonun normalizasyon
    # sabitleri tam-veri aday havuzundan bir kez hesaplanıp DONDURULUR.
    p_tda = hgb.predict_proba(sc.transform(X))[:, 1]
    fusion_ref = {
        "s1_mean": float(s1.mean()), "s1_std": float(s1.std() + 1e-9),
        "tda_mean": float(p_tda.mean()), "tda_std": float(p_tda.std() + 1e-9),
        "w": 0.55,
        "formula": "S = sigmoid(w·z(s1) + (1−w)·z(p_tda)); z, bu sabitlerle",
        "source": "V6.6.6 val-doğrulanmış sabit ağırlık (yeniden uydurulmadı)",
    }
    print(f"    füzyon referansı: {fusion_ref}", flush=True)
    del X, p_tda

    joblib.dump({"hgb": hgb, "scaler": sc, "ranges": ranges, "fusion_ref": fusion_ref,
                 "meta": {"experiment": "V6.7-fulldata", "features": "1600 PI + 4 uzamsal",
                          "stage1_score_included": False, "trap_b_protected": True,
                          "label_rule": "hit_criterion1",
                          "n_samples": int(len(y)), "n_positive": int(y.sum()),
                          "n_patients": len(pids), "candidate_standard": {
                              "scales": list(SCALES), "peak_window": PEAK_WINDOW,
                              "nms_px": NMS_PX, "top_k": TOP_K},
                          "fusion_locked": "w=0.55 (V6.6.6 val-doğrulanmış, sabit)"}},
                OUT)
    LOG.write_text(json.dumps({"n_patients": len(pids), "n_candidates": int(n_cand),
                               "n_samples": int(len(y)), "n_positive": int(y.sum()),
                               "ranges": ranges, "fusion_ref": fusion_ref},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {OUT}", flush=True)
    print("Bilimsel checkpoint'lere DOKUNULMADI.", flush=True)


if __name__ == "__main__":
    main()
