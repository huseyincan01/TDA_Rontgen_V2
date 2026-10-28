"""V6.9 — NODE21 TEST MÜHRÜ (tek atım, ön-kayıtlı).

NEDEN VAR
---------
V6.1–V6.8 boyunca **her** tasarım kararı (kutu boyu 296 değerlendirmelik tarama,
füzyon ağırlığı ızgarası, TTA ölçek seti, NMS penceresi) NODE21 **val** bölmesinde
alındı ve **aynı** bölmede raporlandı. Bu sızıntı değil — ama *seçim yanlılığı*dır:
231 lezyonluk val'de FROC@0,5'in standart hatası ≈ 0,024 olduğu için 0,9131'lik
rank_metric'in gerçek dış-değeri birkaç puan altında olabilir.

Bu betik, kilitli protokolü **hiç değiştirmeden** dokunulmamış **test** bölmesinde
(734 hasta / 229 lezyon) bir kez koşar ve makalenin dürüst dış-değerini üretir.

MÜHÜR KURALLARI (ihlal = koşu geçersiz)
---------------------------------------
1. Hiçbir hiperparametre test üzerinde seçilmez; seçim yapacak kod **yoktur**.
2. Füzyon ağırlığı **w=0,55** ve z-normalizasyon referansı **val'de dondurulmuştur**
   (test dağılımından ortalama/standart sapma hesaplanmaz — bu transductive olurdu).
3. Betik kendi SHA-256'sını `test_seal_preregistration.json` ile karşılaştırır;
   uyuşmazsa **koşmayı reddeder** (ön-kayıttan sonra kod değişmediğinin kanıtı).
4. Kilitli artifact'lerin (model + 2. aşama bundle'ı) SHA-256'sı da doğrulanır.
5. Çıktı ne olursa olsun **olduğu gibi** raporlanır; yeniden koşu/ayar yok.

ÖN-KAYIT: `artifacts/v6_two_stage/test_seal_preregistration.json` (koşudan ÖNCE commit).
Kullanım:
    python pipelines/v6_two_stage/v6_9_test_seal.py            # tam mühür
    python pipelines/v6_two_stage/v6_9_test_seal.py --limit 12 # yalnız duman (mühür DEĞİL)
"""

from __future__ import annotations

import argparse
import hashlib
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
import v6_2_stage2_v2 as v62  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import (dist_nms, heat_maps, official_metric,  # noqa: E402
                                     pick_peaks)

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
PREREG = ART / "test_seal_preregistration.json"
OUT = ART / "v6_9_test_seal.json"
SELF = Path(__file__).resolve()

# ---- KİLİTLİ PROTOKOL (V6.6.7 / submission_config.json ile birebir) ---------- #
SCALES = (0.9, 1.0, 1.1)
FLIP_TTA = True
WIN, NMS_PX, TOPK = 3, 8, 50
BOX_HALF = 28.0                     # 1024 uzayında yarı-kenar ⇒ 56x56 px kutu
W_FUSION = 0.55                     # V6.6.6'da val üzerinde kilitlendi
STAGE1_TAG = "M5_foundation"
STAGE2_FILE = "stage2_final_tda_hgb.joblib"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while (chunk := f.read(1 << 20)):
            d.update(chunk)
    return d.hexdigest()


def _guard(prereg: dict) -> None:
    """Mühür bütünlüğü: betik + artifact hash'leri ön-kayıtla aynı olmalı."""
    want = prereg["sealed_code"]["v6_9_test_seal.py"]
    got = sha256(SELF)
    if want != got:
        raise SystemExit(
            f"MÜHÜR İHLALİ — betik hash'i ön-kayıttan farklı.\n"
            f"  ön-kayıt: {want}\n  şu anki : {got}\n"
            f"Ön-kayıt, koşudan önce commit edilmiş kodu tanımlar; kod değiştiyse "
            f"mühür geçersizdir.")
    for name, meta in prereg["sealed_artifacts"].items():
        p = PROJECT_ROOT / meta["path"]
        if not p.exists():
            raise SystemExit(f"MÜHÜR İHLALİ — eksik artifact: {p}")
        if sha256(p) != meta["sha256"]:
            raise SystemExit(f"MÜHÜR İHLALİ — {name} hash'i değişmiş: {p}")
    print("  🔒 mühür bütünlüğü doğrulandı (betik + artifact hash'leri)", flush=True)


# --------------------------------------------------------------------------- #
# bölme koşusu
# --------------------------------------------------------------------------- #
def _pass(pids, model, device, ranges, bundle, tag: str) -> dict:
    """Aday üretimi + 2. aşama özellikleri. Döner: pid → sözlük."""
    from concurrent.futures import ProcessPoolExecutor

    print(f"\n[{tag}] 1. aşama adayları ({len(pids)} hasta)...", flush=True)
    cands, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        cands[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    bitti: {sum(len(v) for v in cands.values())} aday "
          f"({time.time() - t0:.0f}s)", flush=True)

    print(f"[{tag}] 2. aşama lokal TDA özellikleri...", flush=True)
    jobs = [(pid, cands[pid], ranges, False) for pid in pids]
    feats, t1 = {}, time.time()
    try:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
                feats[pid] = res
    except Exception as exc:                       # kısıtlı sandbox: süreç kurulamaz
        print(f"    (süreç havuzu yok: {type(exc).__name__} → seri)", flush=True)
        for job in jobs:
            pid, res, err = s2._patient_features(job)
            feats[pid] = res
    bad = [p for p in pids if feats.get(p) is None]
    if bad:
        print(f"    ⚠️ özellik çıkarılamayan {len(bad)} hasta: {bad[:5]}", flush=True)
    print(f"    bitti ({time.time() - t1:.0f}s)", flush=True)

    sc, hgb = bundle["scaler"], bundle["hgb"]
    gts_map = {p: common.load_gt(p) for p in pids}
    out = {}
    for pid in pids:
        cs = cands[pid]
        f = feats.get(pid)
        rec = {"pid": pid, "is_nodule": bool(gts_map[pid]), "gts": gts_map[pid],
               "cs": cs, "labs": s2._labels_for(cs, gts_map[pid]),
               "s1": np.zeros(0), "tda": np.zeros(0)}
        if f is not None and cs:
            X = sc.transform(np.concatenate([f["tda"], f["spa"]], 1))
            rec["s1"] = np.array([c[2] for c in cs], np.float32)
            rec["tda"] = hgb.predict_proba(X[:len(cs)])[:, 1].astype(np.float32)
        out[pid] = rec
    return out


def _zstat(recs: dict) -> dict:
    s1 = np.concatenate([r["s1"] for r in recs.values() if len(r["s1"])])
    td = np.concatenate([r["tda"] for r in recs.values() if len(r["tda"])])
    return {"s1_mean": float(s1.mean()), "s1_std": float(s1.std()),
            "tda_mean": float(td.mean()), "tda_std": float(td.std())}


def _fuse(rec: dict, ref: dict, w: float) -> np.ndarray:
    """S = w·z(s1) + (1−w)·z(tda); z referansı VAL'de dondurulmuş."""
    if len(rec["s1"]) == 0:
        return np.zeros(0)
    z1 = (rec["s1"] - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
    zt = (rec["tda"] - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
    return (w * z1 + (1.0 - w) * zt).astype(np.float32)


def _flat(recs, pids, key, ref=None, sigmoid=False):
    flat = []
    for pid in pids:
        r = recs[pid]
        if key == "s1":
            sc = r["s1"]
        elif key == "tda":
            sc = r["tda"]
        else:
            sc = _fuse(r, ref, W_FUSION)
            if sigmoid:
                sc = 1.0 / (1.0 + np.exp(-sc))
        for i, (cx, cy, _old) in enumerate(r["cs"]):
            flat.append((float(sc[i]), pid, float(cx), float(cy), 0.0))
    return flat


def _official(recs, pids, key, ref=None):
    gts = {p: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
               for g in recs[p]["gts"]] for p in pids}
    is_nod = {p: recs[p]["is_nodule"] for p in pids}
    n_les = sum(len(g) for g in gts.values())
    return official_metric(_flat(recs, pids, key, ref, sigmoid=True), len(pids), n_les,
                           gts, is_nod, pids, box_half=BOX_HALF)


def _legacy(recs, pids, key, ref=None):
    """Eski protokol (FROC 1/2/4 FP) — yayımlanmış val sayılarıyla kıyas için."""
    n_les = sum(len(recs[p]["gts"]) for p in pids)
    rl = []
    for pid in pids:
        r = recs[pid]
        sc = r["s1"] if key == "s1" else (
            r["tda"] if key == "tda" else _fuse(r, ref, W_FUSION))
        rl.append({"pid": pid, "is_nodule": r["is_nodule"],
                   "cands": [(r["cs"][i][0], r["cs"][i][1], float(sc[i]), r["labs"][i])
                             for i in range(len(r["cs"]))]})
    fr, au = v62.froc_fixed(rl, len(rl), n_les)
    fh, ah = v62.froc_fixed(v62.hierarchical(rl), len(rl), n_les)
    return {"froc": fr, "patient_auc": au, "hier_froc": fh, "hier_auc": ah}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="yalnız duman testi (mühür DEĞİL)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-val", action="store_true",
                    help="z referansını ön-kayıttan oku (val koşusunu atla)")
    args = ap.parse_args()

    if not PREREG.exists():
        raise SystemExit(f"Ön-kayıt yok: {PREREG} — mühür koşusu yapılamaz.")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    print("V6.9 TEST MÜHÜRÜ")
    if args.limit:
        print("  ⚠️ DUMAN MODU — sonuç mühür sayılmaz, rapora 'sealed': false yazılır")
    _guard(prereg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import joblib
    bundle = joblib.load(CKPT / STAGE2_FILE)
    ranges = bundle["ranges"]
    model = load_foundation(STAGE1_TAG, device)
    print(f"  model={STAGE1_TAG} · bundle={STAGE2_FILE} · device={device} · "
          f"w={W_FUSION} · kutu={int(2 * BOX_HALF)}px", flush=True)

    sp = common.split_patients()
    val_pids, test_pids = list(sp["val"]), list(sp["test"])
    if args.limit:
        val_pids, test_pids = val_pids[:args.limit], test_pids[:args.limit]

    # ---- 1) VAL: yalnız z referansını dondurmak + üretimi doğrulamak için
    if args.skip_val:
        ref = prereg["frozen_z_reference"]
        print(f"\n[val] atlandı, dondurulmuş referans: {ref}", flush=True)
        val_recs = None
    else:
        val_recs = _pass(val_pids, model, device, ranges, bundle, "VAL")
        ref = _zstat(val_recs)
        print(f"\n[val] DONDURULAN z referansı: {json.dumps(ref)}", flush=True)

    # ---- 2) TEST: mühür — tek geçiş
    test_recs = _pass(test_pids, model, device, ranges, bundle, "TEST")

    rep = {"experiment": "V6.9-test-seal",
           "purpose": prereg["purpose"],
           "sealed": not args.limit,
           "n_test_patients": len(test_pids),
           "n_test_lesions": sum(len(test_recs[p]["gts"]) for p in test_pids),
           "frozen_z_reference": ref,
           "locked": {"w_fusion": W_FUSION, "box_px": int(2 * BOX_HALF),
                      "scales": list(SCALES), "flip_tta": FLIP_TTA,
                      "peak_window": WIN, "nms_px": NMS_PX, "top_k": TOPK},
           "artifacts": {k: v["sha256"] for k, v in prereg["sealed_artifacts"].items()}}

    # ---- 3) ölçümler (A1: yalnız 1. aşama · A2: iki aşamalı)
    for name, recs, pids in (("val", val_recs, val_pids), ("test", test_recs, test_pids)):
        if recs is None:
            continue
        block = {"n_patients": len(pids),
                 "n_lesions": sum(len(recs[p]["gts"]) for p in pids),
                 "A1_stage1_only": _official(recs, pids, "s1"),
                 "A2a_fusion_tda": _official(recs, pids, "fus", ref),
                 "A2b_tda_only": _official(recs, pids, "tda"),
                 "legacy_stage1": _legacy(recs, pids, "s1"),
                 "legacy_fusion": _legacy(recs, pids, "fus", ref)}
        rep[name] = block
        o = block["A1_stage1_only"]
        f = block["A2a_fusion_tda"]
        print(f"\n  [{name}] A1 (yalnız 1. aşama)  rank_metric={o['rank_metric']} "
              f"AUC={o['auc']} S@.125={o['sens']['0.125']} S@.25={o['sens']['0.25']} "
              f"S@.5={o['sens']['0.5']}", flush=True)
        print(f"  [{name}] A2 (TDA füzyonu w={W_FUSION}) rank_metric={f['rank_metric']} "
              f"AUC={f['auc']} S@.125={f['sens']['0.125']} S@.25={f['sens']['0.25']} "
              f"S@.5={f['sens']['0.5']}", flush=True)

    # ---- 4) mühür hükmü
    if rep.get("test"):
        o, f = rep["test"]["A1_stage1_only"], rep["test"]["A2a_fusion_tda"]
        rep["verdict"] = {
            "stage1_rank_metric": o["rank_metric"],
            "fusion_rank_metric": f["rank_metric"],
            "stage1_auc": o["auc"],
            "vs_val_stage1_rank_metric": round(
                (o["rank_metric"] or 0) - 0.9131, 4) if o["rank_metric"] else None,
            "vs_val_fusion_auc": round((f["auc"] or 0) - 0.9879, 4) if f["auc"] else None,
        }
        def f(x):
            return "n/a" if x is None else f"{x:+.4f}"

        print(f"\n  VAL→TEST farkı (1. aşama rank_metric): "
              f"{f(rep['verdict']['vs_val_stage1_rank_metric'])} (val 0,9131)", flush=True)
        print(f"  VAL→TEST farkı (füzyon AUC): "
              f"{f(rep['verdict']['vs_val_fusion_auc'])} (val 0,9879)", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False,
                                         default=str), encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()
