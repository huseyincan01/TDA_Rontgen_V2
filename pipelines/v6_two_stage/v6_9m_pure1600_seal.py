"""V6.9m — SAF 1.600 TDA STANDARDİZASYONU + kilitli test doğrulaması.

GEREKÇE (V6.9l ablasyonu)
------------------------
V6.9l'de 1604 boyutlu temsilin son 4 uzamsal vekili (rel_x, rel_y, "hilus vekili",
"plevra vekili") **aday düzeyinde zararlı** çıktı (TDA-tek AUC 0,9387 → 0,9242) ve hasta
düzeyinde katkısı **gösterilemedi** (füzyon AUC ±0,001, rank ±0,0025, işaretler ters).
Occam'ın usturası gereği 2. aşama temsili **%100 saf topolojik** hâle getirilir:

    2 kesit (32/64) × 2 homoloji (H0/H1) × 20×20 PI = **1.600 boyut**

BU BETİK NE YAPAR
-----------------
1. `_v6_9l_cache.npz` içindeki **ilk 1.600 sütunu** kullanarak (TDA yeniden HESAPLANMAZ)
   HGB'yi eğitir; z-referansını **val**'den dondurur ve
   `checkpoints/stage2_foundation_tda_hgb_pure1600.joblib` olarak kaydeder.
   ⇒ `stage2_foundation_tda_hgb.joblib` (1604) **SİLİNMEZ**, yedek olarak korunur.
2. **Kilitli testte (734 hasta)** saf 1.600 modelin TDA-tek ve füzyon metriklerini ölçer;
   aynı aday kümesinde **deploy edilmiş 1604 bundle'ı** ile yan yana raporlar.
   Test TDA özellikleri `_v6_9m_test_cache.npz` olarak önbelleğe alınır (tekrar hesap yok).

⚠️ TEST BÜTÇESİ BEYANI — **BEŞİNCİ** kullanım
---------------------------------------------
Öncekiler: V6.4 · V6.9 · V6.9f · V6.9i. Bu koşuda **hiçbir hiperparametre test üzerinde
seçilmez**: mimari sadeleştirme kararı V6.9l'de yalnız **val**'de alınmıştır; test yalnız
tek-atımlık **doğrulama**dır. Çıktı ne olursa olsun olduğu gibi raporlanır; saf 1.600 model
testte belirgin biçimde geri kalırsa **deploy edilmez** ve 1604 bundle'ı korunur.

Kullanım:
    python pipelines/v6_two_stage/v6_9m_pure1600_seal.py
    python pipelines/v6_two_stage/v6_9m_pure1600_seal.py --limit 30   # duman (mühür DEĞİL)
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

import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_9k_leave_jsrt_out as k  # noqa: E402
import v6_9l_spatial_ablation as L  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import official_metric  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
OUT = ART / "v6_9m_pure1600_seal.json"
CACHE_L = ART / "_v6_9l_cache.npz"
CACHE_TEST = ART / "_v6_9m_test_cache.npz"
NEW_BUNDLE = CKPT / "stage2_foundation_tda_hgb_pure1600.joblib"
OLD_BUNDLE = CKPT / "stage2_foundation_tda_hgb.joblib"
SEED = 42
W = k.W
BOX_HALF = k.BOX_HALF


def _flat(scores, pids, off, cx, cy):
    out = []
    for p_i in range(len(pids)):
        for i in range(off[p_i], off[p_i + 1]):
            out.append((float(scores[i]), pids[p_i], float(cx[i]), float(cy[i]), 0.0))
    return out


def _metrics(scores, pids, off, cx, cy, gts, n_les):
    gt_map = {p: g for p, g in zip(pids, gts)}
    is_nod = {p: bool(g) for p, g in zip(pids, gts)}
    return official_metric(_flat(scores, pids, off, cx, cy), len(pids), n_les, gt_map,
                           is_nod, pids, box_half=BOX_HALF)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9m — saf 1.600 TDA standardizasyonu", flush=True)
    print("  ⚠️ BEŞİNCİ test kullanımı — tek-atımlık doğrulama; test üzerinde seçim YOK",
          flush=True)
    print(f"  device={device}", flush=True)

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # ---------------- 1) saf 1.600 HGB (önbellekten) ---------------- #
    if not CACHE_L.exists():
        raise SystemExit(f"V6.9l önbelleği yok: {CACHE_L} — önce v6_9l koşulmalı")
    z = np.load(CACHE_L, allow_pickle=True)
    Xtr = z["tr_tda"].astype(np.float32)
    ytr = z["tr_y"]
    Xva = z["va_tda"].astype(np.float32)
    s1_va = z["va_s1"].astype(np.float32)
    print(f"\n[1] Önbellekten: fit {Xtr.shape} · val {Xva.shape} · "
          f"%{100 * ytr.mean():.2f} pozitif", flush=True)

    sc = StandardScaler().fit(Xtr)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         class_weight="balanced",
                                         random_state=SEED).fit(sc.transform(Xtr), ytr)
    p_va = hgb.predict_proba(sc.transform(Xva))[:, 1].astype(np.float32)
    ref = {"s1_mean": float(s1_va.mean()), "s1_std": float(s1_va.std()),
           "tda_mean": float(p_va.mean()), "tda_std": float(p_va.std()), "w": W}
    print(f"    val z-ref donduruldu: {json.dumps(ref, ensure_ascii=False)}", flush=True)

    old = joblib.load(OLD_BUNDLE)
    bundle = {"hgb": hgb, "scaler": sc, "ranges": old["ranges"], "fusion_ref": ref,
              "stage1_score_included": False,
              "meta": {"experiment": "V6.9m-pure1600",
                       "features": "SAF 1600 TDA (2 kesit × 2 homoloji × 20×20); "
                                   "uzamsal konum özelliği YOK",
                       "trained_on": "M5_foundation train adayları (790+790)",
                       "why": "V6.9l: konum özellikleri aday düzeyinde zararlı, hasta "
                              "düzeyinde katkısız ⇒ Occam (mimari sadelik)",
                       "protects": "stage2_foundation_tda_hgb.joblib (1604) SİLİNMEDİ"}}
    joblib.dump(bundle, NEW_BUNDLE)
    print(f"    kaydedildi: {NEW_BUNDLE.name}  (eski 1604 bundle korundu)", flush=True)

    # ---------------- 2) kilitli test ---------------- #
    te_pids = list(common.split_patients()["test"])
    if args.limit:
        te_pids = te_pids[:args.limit]
    gts = L.gts_of(te_pids)
    n_les = sum(len(g) for g in gts)
    print(f"\n[2] Kilitli test: {len(te_pids)} hasta · {n_les} lezyon", flush=True)

    if CACHE_TEST.exists() and not args.refresh and not args.limit:
        print(f"    önbellek: {CACHE_TEST.name}", flush=True)
        t = {kk: vv for kk, vv in np.load(CACHE_TEST, allow_pickle=True).items()}
    else:
        model = load_foundation(k.MAIN_TAG, device)
        c_te = L._candidates(model, te_pids, device, "test")
        f_te = L._features(te_pids, c_te, old["ranges"], "test")
        pk = L.pack(te_pids, c_te, f_te)
        t = {f"te_{kk}": vv for kk, vv in pk.items()}
        t["te_cx"] = np.asarray([c[0] for p in te_pids for c in c_te[p]], np.float32)
        t["te_cy"] = np.asarray([c[1] for p in te_pids for c in c_te[p]], np.float32)
        t["te_y"] = L.labels_for(te_pids, c_te)
        t["te_pids"] = np.asarray(te_pids)
        np.savez_compressed(CACHE_TEST, **t)
        print(f"    önbellek yazıldı: {CACHE_TEST.name}", flush=True)

    te_pids = [str(p) for p in t["te_pids"]]
    tx, ty = t["te_cx"], t["te_cy"]
    off = t["te_off"]
    s1 = t["te_s1"].astype(np.float32)
    print(f"    aday {len(s1)} · %{100 * t['te_y'].mean():.2f} pozitif", flush=True)

    arms = {}
    # --- saf 1.600
    p_pure = hgb.predict_proba(sc.transform(t["te_tda"].astype(np.float32)))[:, 1]
    fus_pure = k.fuse(s1, p_pure, ref, W)
    # --- deploy edilmiş 1604 (yeniden eğitilmez; doğrudan uygulanır)
    X_old = np.concatenate([t["te_tda"].astype(np.float32), t["te_sb"].astype(np.float32)], 1)
    p_old = old["hgb"].predict_proba(old["scaler"].transform(X_old))[:, 1]
    ref_old = old["fusion_ref"]
    fus_old = k.fuse(s1, p_old, ref_old, W)

    for name, p_tda, fus in (("pure1600", p_pure, fus_pure), ("ref1604", p_old, fus_old)):
        arms[name] = {"n_features": 1600 if name == "pure1600" else 1604,
                      "tda_only": _metrics(p_tda, te_pids, off, tx, ty, gts, n_les),
                      "fusion": _metrics(1.0 / (1.0 + np.exp(-fus)), te_pids, off, tx, ty,
                                         gts, n_les)}
        a = arms[name]
        print(f"\n  [{name}] TDA-tek AUC={a['tda_only']['auc']} "
              f"rank={a['tda_only']['rank_metric']}", flush=True)
        print(f"  [{name}] FÜZYON  AUC={a['fusion']['auc']} rank={a['fusion']['rank_metric']} "
              f"S@.125={a['fusion']['sens']['0.125']} S@.25={a['fusion']['sens']['0.25']} "
              f"S@.5={a['fusion']['sens']['0.5']}", flush=True)

    d = {kk: _delta(arms["pure1600"]["fusion"][kk], arms["ref1604"]["fusion"][kk])
         for kk in ("auc", "rank_metric")}
    d["tda_auc"] = _delta(arms["pure1600"]["tda_only"]["auc"],
                          arms["ref1604"]["tda_only"]["auc"])
    d["S025"] = _delta(arms["pure1600"]["fusion"]["sens"]["0.25"],
                       arms["ref1604"]["fusion"]["sens"]["0.25"])
    d["S05"] = _delta(arms["pure1600"]["fusion"]["sens"]["0.5"],
                      arms["ref1604"]["fusion"]["sens"]["0.5"])
    print(f"\n  Δ(saf1600 − 1604): {json.dumps(d, ensure_ascii=False)}", flush=True)

    verdict = ("DEPLOY: saf 1600" if (d["auc"] or 0) >= -0.003
               else "DEPLOY ETME: 1604 korunur")
    rep = {
        "experiment": "V6.9m-pure1600-seal",
        "question": ("2. aşama temsili saf 1.600 topolojik boyuta indirildiğinde kilitli testte "
                     "ne olur? (V6.9l: konum özellikleri katkısız/zararlı ⇒ Occam)"),
        "test_budget_declaration": ("BEŞİNCİ test kullanımı (öncekiler V6.4 · V6.9 · V6.9f · "
                                    "V6.9i). Mimari karar V6.9l'de yalnız VAL'de alındı; bu koşu "
                                    "tek-atımlık doğrulamadır. Test üzerinde seçim YOK."),
        "representation": "2 kesit (32/64) × 2 homoloji (H0/H1) × 20×20 PI = 1600",
        "spatial_features_removed": ["rel_x", "rel_y", "akciger_bbox_merkezine_uzaklik",
                                     "akciger_maskesi_sinirina_uzaklik"],
        "bundles": {"pure1600": str(NEW_BUNDLE.relative_to(PROJECT_ROOT)),
                    "ref1604_protected": str(OLD_BUNDLE.relative_to(PROJECT_ROOT))},
        "trap_B": "stage1_score_included = False (her iki kolda)",
        "fusion": {"w": W, "box_px": int(2 * BOX_HALF),
                   "z_ref_pure1600": ref, "z_ref_ref1604": ref_old},
        "test": {"n_patients": len(te_pids), "n_lesions": int(n_les),
                 "n_candidates": int(len(s1))},
        "arms": arms,
        "delta_pure1600_minus_ref1604": d,
        "verdict": verdict,
        "smoke": bool(args.limit),
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  HÜKÜM: {verdict}", flush=True)
    print(f"  kayıt: {args.out}", flush=True)


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


if __name__ == "__main__":
    main()
