"""V6.5d — STRICT / INDUCTIVE zero-shot denetimi: AdaBN OLMADAN fiziksel hizalama.

Motivasyon
----------
V6.5 KOL 1 tablosunda taban adayları `AdaBN-M5` idi (VinDr'in ETİKETSİZ BN
istatistikleri kullanılıyordu ⇒ transductive). Bu yüzden fiziksel hizalamanın
(denoise + ölçek + CORAL) hiçbir hedef istatistiği kullanmayan **tekil-görüntü
(inductive) zero-shot** etkisi ayrı olarak ölçülmemişti.

Bu betik aynı 5 konfigürasyonu DONMUŞ NODE21 BatchNorm istatistikleriyle
(AdaBN KAPALI) koşar ve sonucu mevcut V6.5 raporuna `strict_no_adabn_ablation`
anahtarıyla ekler. Önceki tüm sonuçlar korunur.

Konfigürasyonlar
----------------
  strict_base_z              no-AdaBN + base_z (saf sıfır-öğrenme tabanı)
  strict_denoise1.0_z        no-AdaBN + JPEG-denoise (sigma=1.0)
  strict_scale_z             no-AdaBN + ölçek-eşlemeli yama
  strict_denoise1.0_scale_z  no-AdaBN + denoise + ölçek (inductive tavan)
  strict_denoise1.0_scale_coral  no-AdaBN + denoise + ölçek + CORAL
      NOT: CORAL VinDr kovaryansını kullanır ⇒ bu satır inductive DEĞİL,
      karşılaştırma için "tek kanal transductive" olarak raporlanır.

Kullanım:
    python pipelines/v6_two_stage/v6_5d_strict_no_adabn.py
    python pipelines/v6_two_stage/v6_5d_strict_no_adabn.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import candidates_classical as cc  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
import v6_5_vindr_transfer_boost as v65  # noqa: E402
from detector_models import INPUT, CenterNetHead  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT_JSON = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_5_vindr_transfer_boost.json"
EXT = PROJECT_ROOT / "data/external_vindr"
SEED = 42
NW = v65.NW

# Bu denetim için gereken benzersiz (ölçek türü, sigma) kombinasyonları.
# (fixed, 0.6) atlanır — istenen 5 konfigürasyonda yok, hesaplama tasarrufu.
STRICT_KEYS = [("fixed", 0.0), ("fixed", 1.0), ("scale", 0.0), ("scale", 1.0)]

# isim, ölçek türü, sigma, hizalama modu, inductive mi?
STRICT_CONFIGS = [
    ("strict_base_z", "fixed", 0.0, "z", True),
    ("strict_denoise1.0_z", "fixed", 1.0, "z", True),
    ("strict_scale_z", "scale", 0.0, "z", True),
    ("strict_denoise1.0_scale_z", "scale", 1.0, "z", True),
    ("strict_denoise1.0_scale_coral", "scale", 1.0, "coral", False),
]

# ÖNEMLİ: ProcessPoolExecutor (spawn) altında çocuk süreç v6_5 modülünü YENİDEN
# import eder; bu yüzden anahtar kısıtı modül düzeyinde yapılmalıdır, aksi hâlde
# çocuk süreç gereksiz (fixed, 0.6) anahtarını da hesaplar.
v65.KEYS = list(STRICT_KEYS)

# V6.5 KOL 1'deki AdaBN'li karşılıklar (rapor karşılaştırması için, yeniden koşulmaz).
ADABN_COUNTERPART = {
    "strict_base_z": "base_z",
    "strict_denoise1.0_z": "denoise1.0_z",
    "strict_scale_z": "scale_z",
    "strict_denoise1.0_scale_z": "denoise1.0_scale_z",
    "strict_denoise1.0_scale_coral": "denoise1.0_scale_coral",
}


def build_vindr(seg, cohort, nod, limit=0):
    """VinDr kohortu: kanonik görüntü (RAM'de; diske YAZILMAZ) + GT + x512."""
    data = []
    if limit:
        cohort = cohort.head(limit)
    t0 = time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"])
        lab = int(row["patient_label"])
        img = np.array(Image.open(EXT / "images" / f"{pid}.jpg"))
        gts = []
        for _, b in nod[nod["image_id"] == pid].iterrows():
            x0, x1 = b["x_min"] / 3.0, b["x_max"] / 3.0
            y0, y1 = b["y_min"] / 3.0, b["y_max"] / 3.0
            if x1 > x0 and y1 > y0:
                gts.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
        try:
            raw = seg.segment(img)["zone_map"] > 0
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi_dilate(raw)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            canon_u8 = np.clip(np.rint(canon), 0, 255).astype(np.uint8)
            x512 = np.array(Image.fromarray(canon_u8).resize((INPUT, INPUT),
                                                             Image.Resampling.BILINEAR))
            ys = np.where(raw.any(axis=1))[0]
            lung_h = int(ys.max() - ys.min() + 1) if ys.size else 570
        except Exception:
            raw = np.zeros(img.shape, bool)
            canon_u8 = np.zeros(img.shape, np.uint8)
            x512 = np.zeros((INPUT, INPUT), np.uint8)
            lung_h = 570
        data.append({"pid": pid, "y": lab, "gts": gts, "canon": canon_u8,
                     "raw": raw.astype(np.uint8), "x512": x512, "lung_h": lung_h})
        if (i + 1) % 100 == 0:
            print(f"    VinDr {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)
    return data


def ndi_dilate(raw):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                               iterations=common.DILATION_PX)


def detector_only_auc(data, cands):
    """Aşama-1 tek başına hasta AUC (AdaBN'siz) — V6.4'ün 0,5443'ü ile çapraz kontrol."""
    y = np.array([d["y"] for d in data])
    out = {}
    for agg in ("top1", "top3", "noisyor"):
        sc = []
        for d in data:
            ss = sorted([c[2] for c in cands[d["pid"]]], reverse=True)
            if not ss:
                sc.append(0.0)
            elif agg == "top1":
                sc.append(float(ss[0]))
            elif agg == "top3":
                sc.append(float(np.mean(ss[:3])))
            else:
                sc.append(float(1 - np.prod([1 - min(1.0, max(0.0, s)) for s in ss[:5]])))
        from sklearn.metrics import roc_auc_score
        out[agg] = round(float(roc_auc_score(y, sc)), 4)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sm = args.smoke

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()
    bundle = __import__("joblib").load(CKPT / "stage2_final_tda_hgb.joblib")
    scaler, hgb, lr, ranges = bundle["scaler"], bundle["hgb"], bundle["lr_fusion"], bundle["ranges"]
    print("V6.5d | STRICT (AdaBN KAPALI) fiziksel hizalama denetimi", flush=True)

    # ---------------- VinDr hazırlık
    import torchxrayvision as xrv
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    if sm:  # dengeli küçük alt küme (tek sınıflı dilim AUC hesaplayamaz)
        cohort = pd.concat([cohort[cohort["patient_label"] == 1].head(10),
                            cohort[cohort["patient_label"] == 0].head(10)]).reset_index(drop=True)
    seg = PSPNet5ZoneSegmenter(device=device)
    print("\n[0] VinDr hazırlık...", flush=True)
    data = build_vindr(seg, cohort, nod)

    # ---------------- Aşama-1 adayları — AdaBN YOK
    print("\n[1] Aşama-1 adayları (AdaBN KAPALI, dondurulmuş NODE21 BN)...", flush=True)
    cands = {}
    for d in data:
        try:
            c = m64._pred_from_512(m5, d["x512"], d["canon"].shape, device, "M5")
            cands[d["pid"]] = cc._nms(c, 20, 50)
        except Exception:
            cands[d["pid"]] = []
    n_cand = sum(len(v) for v in cands.values())
    print(f"    aday: {n_cand}", flush=True)
    det = detector_only_auc(data, cands)
    print(f"    yalnız dedektör hasta AUC (AdaBN'siz): {det}", flush=True)

    # ---------------- NODE21 referans özellikleri (CORAL hedefi)
    print("\n[2] NODE21 referans özellikleri (CORAL hedefi)...", flush=True)
    n_ref = 10 if sm else 100
    tr_pids = list(np.random.RandomState(SEED).choice(p2._nodule_pids("train"), n_ref, replace=False)) + \
        list(np.random.RandomState(SEED + 1).choice(p2._control_pids("train"), n_ref, replace=False))
    _m5b, m6 = m64.load_models(device)
    c_tr = s2.stage1_candidates(tr_pids, m5, m6, device, top=5 if sm else 20)
    ref_parts = np.array_split(np.array(tr_pids), NW)
    ref_jobs = [(list(part), {p: c_tr[p]["M5_heatTTA"] for p in part}, ranges) for part in ref_parts]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as pool:
        ref_res = list(pool.map(v65._w_ref, ref_jobs))
    ref_tda = [x for r in ref_res for x in r[0]]
    ref_spa = [x for r in ref_res for x in r[1]]
    X_ref = np.stack([np.concatenate([t, s]) for t, s in zip(ref_tda, ref_spa)]).astype(np.float32)
    print(f"    referans aday: {len(X_ref)} ({time.time() - t0:.0f}s)", flush=True)
    aligner = v65.Aligner(X_ref)

    # ---------------- VinDr özellikleri (yalnız gerekli anahtarlar)
    print("\n[3] VinDr lokal TDA özellikleri (paralel)...", flush=True)
    parts = np.array_split(np.arange(len(data)), NW)
    jobs = []
    for part in parts:
        chunk = [{"pid": data[i]["pid"], "canon": data[i]["canon"], "raw": data[i]["raw"],
                  "lung_h": data[i]["lung_h"], "cands": cands[data[i]["pid"]]} for i in part]
        jobs.append((chunk, ranges))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as pool:
        res = list(pool.map(v65._w_extract, jobs))
    feats_cache = {k: {} for k in STRICT_KEYS}
    for pids, out in res:
        for k in STRICT_KEYS:
            for pid, f in zip(pids, out[k]):
                feats_cache[k][pid] = f
    print(f"    özellikler ({len(STRICT_KEYS)} konfig, {NW} süreç): {time.time() - t0:.0f}s", flush=True)

    # ---------------- konfigürasyon değerlendirmesi
    print("\n[4] Strict konfigürasyonlar...", flush=True)
    strict = {}
    for name, skind, sigma, amode, inductive in STRICT_CONFIGS:
        feats = [feats_cache[(skind, sigma)][d["pid"]] for d in data]
        r = v65.eval_config(data, cands, feats, aligner, amode, hgb, lr)
        strict[name] = {"auc": r["auc"], "inductive": inductive,
                        "scale": skind, "denoise_sigma": sigma, "align": amode}
        print(f"    {name:32s} AUC={r['auc']}  (inductive={inductive})", flush=True)

    # ---------------- raporu GÜNCELLE (mevcut anahtarlar korunur)
    report = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    prev = report.get("kol1_zero_shot", {})
    cmp_rows = {}
    for name, meta in strict.items():
        cp = ADABN_COUNTERPART[name]
        a_prev = (prev.get(cp) or {}).get("auc")
        cmp_rows[name] = {"no_adabn_auc": meta["auc"], "adabn_auc": a_prev,
                          "delta_adabn_minus_strict": (round(a_prev - meta["auc"], 4)
                                                       if a_prev is not None and meta["auc"] is not None else None),
                          "inductive": meta["inductive"]}
    report["strict_no_adabn_ablation"] = {
        "experiment": "V6.5d",
        "date": time.strftime("%Y-%m-%d"),
        "goal": "AdaBN KAPALI (dondurulmuş NODE21 BN) fiziksel hizalama denetimi",
        "protocol": "VinDr-CXR 400 görüntü, adaylar M5_centernet (AdaBN'siz), K=50; "
                    "aynı dondurulmuş 2. aşama (HGB + füzyon LR); referans 200 NODE21 train hastası",
        "n_candidates": int(n_cand),
        "detector_only_patient_auc_no_adabn": det,
        "configs": strict,
        "comparison_vs_adabn": cmp_rows,
        "inductive_ceiling": "strict_denoise1.0_scale_z",
        "note": "CORAL satırı VinDr kovaryansını kullandığı için inductive değildir "
                "(tek kanal transductive); diğer dört satır hiçbir hedef istatistiği kullanmaz.",
        "classification": "strict (inductive) zero-shot",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor güncellendi: {OUT_JSON}", flush=True)
    print("\n=== STRICT vs AdaBN ===", flush=True)
    for k, v in cmp_rows.items():
        print(f"  {k:32s} strict={v['no_adabn_auc']}  adabn={v['adabn_auc']}  "
              f"delta={v['delta_adabn_minus_strict']}", flush=True)


if __name__ == "__main__":
    main()
