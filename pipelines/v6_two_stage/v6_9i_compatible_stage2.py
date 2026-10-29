"""V6.9i — Uyumlu 2. aşama checkpoint sabitleme + kilitli test ölçümü.

SORUN (V6.9h bonus bulgusu)
---------------------------
Mevcut `stage2_final_tda_hgb.joblib`, **eski M5 (ImageNet) dedektörünün** aday dağılımı üzerinde
eğitilmişti; oysa 1. aşamada artık `M5_foundation` (CXR vakıf omurgası) kullanıyoruz. HGB'yi
foundation'ın **kendi train adaylarıyla** yeniden eğitince val füzyonu 0,9174 → **0,9238**'e çıktı
⇒ eğitim/çıkarım aday dağılımı uyumsuzluğu vardı.

BU BETİK NE YAPAR
-----------------
1. HGB'yi M5_foundation'ın **TÜM train split** adayları üzerinde eğitir (790 nodüllü + 790 kontrol;
   V6.2 ile aynı 1:1 denge, seed 42).
   * Özellikler: 1600 PI (32/64, c_lungref) + 4 uzamsal = **1604**; **yarıçap YOK**; **s1 YOK**
     (Tuzak B: `stage1_score_included=False`).
2. Kalıcı bundle olarak `checkpoints/stage2_foundation_tda_hgb.joblib` kaydeder (eski bundle'a
   DOKUNMAZ — o korumalı bilimsel artifact'tir).
3. Füzyon referansını (z-normalizasyon: s1_mean/std, tda_mean/std) **val** adaylarından dondurur
   (V6.9 mühürüyle aynı yöntem) ve bundle'a yazar.
4. **Kilitli test**te (734 hasta) nihai skoru ölçer: AUC, Sens@0.125/0.25/0.5, rank_metric
   (kilitli w=0,55 + sabit 56×56 kutu).

⚠️ TEST BÜTÇESİ BEYANI: Bu, test bölmesinin **DÖRDÜNCÜ** kullanımıdır (V6.4 · V6.9 · V6.9f ·
V6.9i). Gerekçe: dağılım-uyumluluğu (kompatibilite) düzeltmesinin tek-atımlık doğrulamasıdır;
hiçbir hiperparametre test üzerinde seçilmez. Bu koşudan sonra test bölmesi **tamamen kapanır**.

Kullanım: python pipelines/v6_two_stage/v6_9i_compatible_stage2.py
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
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, official_metric, pick_peaks  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9i_compatible_stage2.json"
NEW_BUNDLE = CKPT / "stage2_foundation_tda_hgb.joblib"
OLD_BUNDLE = CKPT / "stage2_final_tda_hgb.joblib"
SCALES, WIN, NMS_PX, TOPK = (0.9, 1.0, 1.1), 3, 8, 50
BOX_HALF, W = 28.0, 0.55


def _candidates(model, pids, device, tag):
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s), float(r)) for cx, cy, s, r in cs]
        if (i + 1) % 200 == 0:
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
    tda, spa, y, s1 = [], [], [], []
    for pid in pids:
        if f[pid] is None:
            continue
        for i, (cx, cy, s, r) in enumerate(cands[pid]):
            tda.append(f[pid]["tda"][i])
            spa.append(f[pid]["spa"][i])
            y.append(1 if labs[pid][i] is not None else 0)
            s1.append(s)
    return (np.asarray(tda, np.float32), np.asarray(spa, np.float32),
            np.asarray(y, np.int64), np.asarray(s1, np.float32))


def _fuse_scores(pids, cands, s1, p_tda, ref):
    """ref: {s1_mean, s1_std, tda_mean, tda_std, w}. Döner: (flat_fus, flat_s1)."""
    f_fus, f_s1, idx = [], [], 0
    for pid in pids:
        for (cx, cy, _, _) in cands[pid]:
            a, b = s1[idx], p_tda[idx]
            z = ref["w"] * (a - ref["s1_mean"]) / (ref["s1_std"] + 1e-9) \
                + (1 - ref["w"]) * (b - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
            f_fus.append((float(1.0 / (1.0 + np.exp(-z))), pid, cx, cy, 0.0))
            f_s1.append((float(a), pid, cx, cy, 0.0))
            idx += 1
    return f_fus, f_s1


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--smoke", action="store_true", help="yalnız 40+40 hasta (hız kontrolü)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9i — uyumlu 2. aşama checkpoint + kilitli test", flush=True)
    print("  ⚠ DÖRDÜNCÜ test kullanımı — dağılım-uyum düzeltmesinin tek-atımlık doğrulaması",
          flush=True)

    import joblib
    ranges = joblib.load(OLD_BUNDLE)["ranges"]
    model = load_foundation("M5_foundation", device)
    print("  model + ranges yüklendi (eski bundle'a dokunulmadı)", flush=True)

    # ---- eğitim verisi: TÜM train split, 1:1
    tr_nod = p2._nodule_pids("train")
    tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(42)
    tr_pids = list(tr_nod) + list(rng.permutation(tr_ctl)[:len(tr_nod)])
    if args.smoke:
        tr_pids = tr_nod[:40] + rng.permutation(tr_ctl)[:40]
    print(f"  train: {len(tr_pids)} hasta ({len(tr_nod)} nodüllü + eşit kontrol)", flush=True)

    print("[1] Train adayları + özellikler...", flush=True)
    c_tr = _candidates(model, tr_pids, device, "train")
    gt_tr = {p: common.load_gt(p) for p in tr_pids}
    lab_tr = {p: s2._labels_for([(cx, cy, s) for cx, cy, s, _ in c_tr[p]], gt_tr[p]) for p in tr_pids}
    f_tr = _features(tr_pids, c_tr, ranges)
    tda_tr, spa_tr, y_tr, s1_tr = _matrix(tr_pids, c_tr, f_tr, lab_tr)
    X_tr = np.concatenate([tda_tr, spa_tr], 1)
    print(f"    {len(y_tr)} aday, %{100 * y_tr.mean():.2f} pozitif", flush=True)

    print("[2] HGB eğitimi (1604 özellik, s1 YOK, class_weight balanced)...", flush=True)
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    sc = StandardScaler().fit(X_tr)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         class_weight="balanced", random_state=42).fit(
        sc.transform(X_tr), y_tr)
    print("    HGB eğitildi", flush=True)

    # ---- val: z-referansını dondur
    print("[3] Val (z-referansı dondurma)...", flush=True)
    va_pids = list(common.split_patients()["val"])
    c_va = _candidates(model, va_pids, device, "val")
    gt_va = {p: common.load_gt(p) for p in va_pids}
    lab_va = {p: s2._labels_for([(cx, cy, s) for cx, cy, s, _ in c_va[p]], gt_va[p]) for p in va_pids}
    f_va = _features(va_pids, c_va, ranges)
    tda_va, spa_va, y_va, s1_va = _matrix(va_pids, c_va, f_va, lab_va)
    p_tda_va = hgb.predict_proba(sc.transform(np.concatenate([tda_va, spa_va], 1)))[:, 1]
    ref = {"s1_mean": float(s1_va.mean()), "s1_std": float(s1_va.std()),
           "tda_mean": float(p_tda_va.mean()), "tda_std": float(p_tda_va.std()), "w": W}
    print(f"    dondurulan ref: {json.dumps(ref, ensure_ascii=False)}", flush=True)

    # ---- kalıcı bundle kaydet (eski bundle KORUNUR)
    bundle = {"hgb": hgb, "scaler": sc, "ranges": ranges, "fusion_ref": ref,
              "stage1_score_included": False,
              "meta": {"experiment": "V6.9i-compatible-stage2",
                       "trained_on": "M5_foundation train split adayları (790+790)",
                       "features": "1600 PI (32/64) + 4 uzamsal = 1604 (yarıçap YOK, s1 YOK)",
                       "fusion": "w=0.55 kilitli, z-ref val'den donduruldu",
                       "protects": "eski stage2_final_tda_hgb.joblib DEĞİŞMEDİ"}}
    joblib.dump(bundle, NEW_BUNDLE)
    print(f"    kaydedildi: {NEW_BUNDLE.name}", flush=True)

    # ---- kilitli test
    print("[4] Kilitli test (734 hasta)...", flush=True)
    te_pids = list(common.split_patients()["test"])
    c_te = _candidates(model, te_pids, device, "test")
    gt_te = {p: common.load_gt(p) for p in te_pids}
    lab_te = {p: s2._labels_for([(cx, cy, s) for cx, cy, s, _ in c_te[p]], gt_te[p]) for p in te_pids}
    f_te = _features(te_pids, c_te, ranges)
    tda_te, spa_te, y_te, s1_te = _matrix(te_pids, c_te, f_te, lab_te)
    p_tda_te = hgb.predict_proba(sc.transform(np.concatenate([tda_te, spa_te], 1)))[:, 1]

    gts = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]) for g in gt_te[p]]
           for p in te_pids}
    is_nod = {p: len(gts[p]) > 0 for p in te_pids}
    n_les = sum(len(g) for g in gts.values())
    f_fus, f_s1 = _fuse_scores(te_pids, c_te, s1_te, p_tda_te, ref)
    fus = official_metric(f_fus, len(te_pids), n_les, gts, is_nod, te_pids, box_half=BOX_HALF)
    s1m = official_metric(f_s1, len(te_pids), n_les, gts, is_nod, te_pids, box_half=BOX_HALF)

    print(f"\n  KİLİTLİ TEST | 1. aşama: rank={s1m['rank_metric']} AUC={s1m['auc']} | "
          f"FÜZYON (uyumlu 2. aşama): rank={fus['rank_metric']} AUC={fus['auc']} "
          f"S@.125={fus['sens']['0.125']} S@.25={fus['sens']['0.25']} S@.5={fus['sens']['0.5']}",
          flush=True)

    rep = {
        "experiment": "V6.9i-compatible-stage2",
        "test_budget_declaration": "DÖRDÜNCÜ test kullanımı — dağılım-uyum düzeltmesinin tek-atımlık doğrulaması; test üzerinde seçim yok. Bu koşudan sonra test KAPANIR.",
        "bundle": "checkpoints/stage2_foundation_tda_hgb.joblib",
        "old_bundle_protected": "stage2_final_tda_hgb.joblib DEĞİŞMEDİ (SHA-256 korumalı)",
        "trained_on": f"train split {len(tr_pids)} hasta (1:1) · {len(y_tr)} aday · %{100*y_tr.mean():.2f} pozitif",
        "features": "1604 (1600 PI + 4 uzamsal) · s1 YOK (Tuzak B) · yarıçap YOK",
        "fusion": {"w": W, "z_ref": ref, "box": "sabit 56×56 px"},
        "test": {"stage1_only": s1m, "fusion": fus},
        "smoke": args.smoke,
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
