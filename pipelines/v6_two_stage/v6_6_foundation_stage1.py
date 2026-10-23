"""V6.6 — Foundation X-Ray Backbone Stage-1 DEĞERLENDİRMESİ (elma-elma).

Aşamalar (`--stage`):
  val_stage1   NODE21 val (734 hasta): M5 vs M5_foundation — Oracle Recall@K
               (Kriter1 + Kriter2), yalnız-Aşama-1 FROC ve hasta AUC.
  vindr_stage1 VinDr-CXR (400 görüntü): M5 vs M5_foundation — yalnız-Aşama-1
               hasta AUC (AdaBN'siz = strict zero-shot, ve AdaBN'li) + bbox Recall@K.
  val_stage2   NODE21 val: foundation adayları üzerinde lokal TDA-HGB entegrasyonu
               (e_fusion_tda + f_hierarchical), dondurulmuş 2. aşama bundle'ı ile.
  vindr_e2e    VinDr uçtan uca: strict zero-shot / transductive / few-shot.

Tüm sonuçlar artifacts/v6_two_stage/v6_6_foundation_stage1.json içinde birikir.
Disk kuralı: hiçbir kırpıntı/görüntü diske YAZILMAZ; TDA çıkarımı RAM'de on-the-fly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as v62  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
import v6_5_vindr_transfer_boost as v65  # noqa: E402
import v6_5d_strict_no_adabn as v65d  # noqa: E402
from detector_models import CenterNetHead  # noqa: E402
from detector_models_foundation import (FoundationCenterNet, canon_to_512,  # noqa: E402
                                        foundation_input, predict_single, predict_tta)

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_foundation_stage1.json"
KS = (10, 20, 30, 50)
NW = v65.NW
FND_KEY = "FND_heatTTA"


# --------------------------------------------------------------------------- #
# yardımcılar
# --------------------------------------------------------------------------- #
def load_foundation(tag: str, device):
    m = FoundationCenterNet(weights="resnet50-res512-all").to(device)
    m.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location=device))
    m.eval()
    return m


def adabn_foundation(model, canons, device, bs=4):
    """AdaBN: BN running istatistiklerini hedef (etiketsiz) VinDr yığınıyla günceller.

    VinDr görüntüleri değişken boyutludur ⇒ her kanonik önce 512x512'ye indirilir.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for i in range(0, len(canons), bs):
            xs = torch.stack([foundation_input(canon_to_512(c), False)
                              for c in canons[i:i + bs]]).to(device)
            model(xs)
    model.eval()
    return model


def save_report(rep: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")


def load_report() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"experiment": "V6.6", "stage1_backbone": "resnet50-res512-all (torchxrayvision)"}


def records_from(cands_map, gts_map, pids, is_nod_map):
    recs = []
    for p in pids:
        cs = cands_map.get(p, [])
        labs = s2._labels_for(cs, gts_map.get(p, []))
        recs.append({"pid": p, "is_nodule": is_nod_map[p],
                     "cands": [(cs[i][0], cs[i][1], float(cs[i][2]), labs[i])
                               for i in range(len(cs))]})
    return recs


# --------------------------------------------------------------------------- #
# AŞAMA 1 — NODE21 val: M5 vs Foundation
# --------------------------------------------------------------------------- #
def stage_val_stage1(m5, fnd, device, rep):
    print("\n" + "=" * 70)
    print("AŞAMA 1 — NODE21 val (734 hasta): M5 vs M5_foundation")
    print("=" * 70, flush=True)
    val_pids = common.split_patients()["val"]
    gts_map = {p: common.load_gt(p) for p in val_pids}
    is_nod = {p: len(gts_map[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gts_map.values())
    nod_pids = [p for p in val_pids if is_nod[p]]

    variants = ["M5_heatTTA", FND_KEY, "M5", "FND"]
    cands_map = {v: {} for v in variants}
    hits = {v: {str(k): {"c1": 0, "c2": 0} for k in KS} for v in variants}

    t0 = time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        gts = gts_map[pid]
        c = {
            "M5_heatTTA": ev._predict_heat_tta(m5, canon, device, "M5"),
            FND_KEY: predict_tta(fnd, canon, device),
            "M5": p2.predict(m5, canon, device, "M5"),
            "FND": predict_single(fnd, canon, device),
        }
        for v, lst in c.items():
            cands_map[v][pid] = lst
        if gts:
            for v, lst in c.items():
                for k in KS:
                    hits[v][str(k)]["c1"] += int(round(
                        common.recall_at_K(lst, gts, k, common.hit_criterion1) * len(gts)))
                    hits[v][str(k)]["c2"] += int(round(
                        common.recall_at_K(lst, gts, k, common.hit_criterion2) * len(gts)))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)", flush=True)

    out = {"n_val_patients": len(val_pids), "n_nodule_patients": len(nod_pids),
           "n_lesions": n_les, "seconds": round(time.time() - t0, 1),
           "recall_at_K": {}, "stage1_only": {}}
    for v in variants:
        out["recall_at_K"][v] = {
            "criterion1": {k: round(hits[v][str(k)]["c1"] / max(1, n_les), 4) for k in KS},
            "criterion2": {k: round(hits[v][str(k)]["c2"] / max(1, n_les), 4) for k in KS},
            "n_candidates_mean": round(np.mean([len(cands_map[v][p]) for p in val_pids]), 1),
        }
        print(f"  {v:12s} K1: " + " ".join(f"K{k}={hits[v][str(k)]['c1'] / max(1, n_les):.4f}"
                                          for k in KS), flush=True)
        recs = records_from(cands_map[v], gts_map, val_pids, is_nod)
        f_, a_ = v62.froc_fixed(recs, len(recs), n_les)
        out["stage1_only"][v] = {"froc": f_, "patient_auc": a_}
        print(f"  {v:12s} FROC {f_} | AUC {a_}", flush=True)
    rep["val_stage1"] = out
    save_report(rep)
    print(f"  -> kaydedildi: {OUT}", flush=True)


# --------------------------------------------------------------------------- #
# AŞAMA 1b — VinDr-CXR: M5 vs Foundation (strict + AdaBN)
# --------------------------------------------------------------------------- #
def stage_vindr_stage1(m5, fnd, device, rep):
    print("\n" + "=" * 70)
    print("AŞAMA 1b — VinDr-CXR (400 görüntü): M5 vs M5_foundation")
    print("=" * 70, flush=True)
    import pandas as pd
    import torchxrayvision as xrv
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(v65d.EXT / "cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    print("[0] VinDr hazırlık...", flush=True)
    data = v65d.build_vindr(seg, cohort, nod)
    y = np.array([d["y"] for d in data])
    canons = [d["canon"] for d in data]

    out = {"n_images": len(data), "n_positive": int(y.sum()), "models": {}}
    from sklearn.metrics import roc_auc_score

    def auc_from(cands):
        r = {}
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
            r[agg] = round(float(roc_auc_score(y, sc)), 4)
        return r

    def bbox_recall(cands):
        """VinDr bbox recall@K (yalnız pozitif görüntüler, Kriter 1)."""
        res = {}
        pos = [d for d in data if d["gts"]]
        n_les = sum(len(d["gts"]) for d in pos)
        for k in (30, 50):
            hit = sum(int(round(common.recall_at_K(cands[d["pid"]], d["gts"], k,
                                                   common.hit_criterion1) * len(d["gts"])))
                      for d in pos)
            res[f"K{k}"] = round(hit / max(1, n_les), 4)
        return res

    # --- M5 (AdaBN'siz)
    print("[1] M5 (AdaBN KAPALI)...", flush=True)
    c_m5 = {}
    for d in data:
        try:
            c_m5[d["pid"]] = m64._pred_from_512(m5, d["x512"], d["canon"].shape, device, "M5")
        except Exception:
            c_m5[d["pid"]] = []
    out["models"]["M5"] = {"strict_no_adabn": {"patient_auc": auc_from(c_m5),
                                               "bbox_recall": bbox_recall(c_m5)}}

    # --- Foundation (AdaBN'siz)
    print("[2] Foundation (AdaBN KAPALI)...", flush=True)
    c_fn = {}
    for d in data:
        c_fn[d["pid"]] = predict_tta(fnd, d["canon"].astype(np.float32), device,
                                     shape=d["canon"].shape)
    out["models"]["FND"] = {"strict_no_adabn": {"patient_auc": auc_from(c_fn),
                                                "bbox_recall": bbox_recall(c_fn)}}

    # --- AdaBN'li sürümler (m5'i YERİNDE bozmamak için taze model)
    print("[3] AdaBN (M5 ve Foundation)...", flush=True)
    m5_fresh = m64.load_models(device)[0]
    m5a = m64.adabn(m5_fresh, [m64._norm512(d["x512"]) for d in data], device)
    c_m5a = {}
    for d in data:
        try:
            c_m5a[d["pid"]] = m64._pred_from_512(m5a, d["x512"], d["canon"].shape, device, "M5")
        except Exception:
            c_m5a[d["pid"]] = []
    out["models"]["M5"]["adabn"] = {"patient_auc": auc_from(c_m5a),
                                    "bbox_recall": bbox_recall(c_m5a)}

    fnda = adabn_foundation(load_foundation(rep.get("ckpt_tag", "M5_foundation"), device),
                            canons, device)
    c_fna = {d["pid"]: predict_tta(fnda, d["canon"].astype(np.float32), device,
                                   shape=d["canon"].shape) for d in data}
    out["models"]["FND"]["adabn"] = {"patient_auc": auc_from(c_fna),
                                     "bbox_recall": bbox_recall(c_fna)}

    for m in out["models"]:
        print(f"  {m:4s} strict AUC {out['models'][m]['strict_no_adabn']['patient_auc']} | "
              f"AdaBN AUC {out['models'][m]['adabn']['patient_auc']} | "
              f"strict bbox {out['models'][m]['strict_no_adabn']['bbox_recall']}", flush=True)
    rep["vindr_stage1"] = out
    save_report(rep)
    print(f"  -> kaydedildi: {OUT}", flush=True)


# --------------------------------------------------------------------------- #
# AŞAMA 2 — Stage-2 entegrasyonu (NODE21 val)
# --------------------------------------------------------------------------- #
def stage_val_stage2(m5, fnd, device, rep):
    print("\n" + "=" * 70)
    print("AŞAMA 2 — NODE21 val: lokal TDA-HGB entegrasyonu (dondurulmuş 2. aşama)")
    print("=" * 70, flush=True)
    import joblib
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    sc, hgb, lr, ranges = b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"]

    val_pids = common.split_patients()["val"]
    gts_map = {p: common.load_gt(p) for p in val_pids}
    is_nod = {p: len(gts_map[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gts_map.values())

    print("[1] Foundation adayları (val)...", flush=True)
    cands = {}
    t0 = time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        cands[pid] = predict_tta(fnd, canon, device)[:50]
        if (i + 1) % 150 == 0:
            print(f"    {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)", flush=True)

    print("[2] Lokal TDA özellikleri (paralel)...", flush=True)
    jobs = [(pid, cands[pid], ranges, False) for pid in val_pids]
    feats = {}
    with ProcessPoolExecutor(max_workers=NW) as pool:
        for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
            feats[pid] = res
    print(f"    bitti ({time.time() - t0:.0f}s)", flush=True)

    out = {}
    for K in (30, 50):
        base = []
        for pid in val_pids:
            f = feats.get(pid)
            cs = cands[pid][:K]
            if f is None or len(cs) == 0:
                base.append({"pid": pid, "is_nodule": is_nod[pid], "cs": [], "labs": [],
                             "s1": np.zeros(0), "tda": np.zeros(0), "fus": np.zeros(0)})
                continue
            labs = s2._labels_for(cs, gts_map[pid])
            X = sc.transform(np.concatenate([f["tda"][:K], f["spa"][:K]], 1))
            s1 = np.array([c[2] for c in cs], np.float32)
            p_tda = hgb.predict_proba(X)[:, 1]
            p_fus = lr.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
            base.append({"pid": pid, "is_nodule": is_nod[pid], "cs": cs, "labs": labs,
                         "s1": s1, "tda": p_tda, "fus": p_fus})

        KEYMAP = {"a_stage1": "s1", "b_tda_hgb": "tda", "e_fusion_tda": "fus"}

        def as_recs(name):
            key = KEYMAP[name]
            return [{"pid": r["pid"], "is_nodule": r["is_nodule"],
                     "cands": [(r["cs"][i][0], r["cs"][i][1], float(r[key][i]), r["labs"][i])
                               for i in range(len(r["cs"]))]} for r in base]

        for kind in ("a_stage1", "b_tda_hgb", "e_fusion_tda"):
            recs = as_recs(kind)
            f_, a_ = v62.froc_fixed(recs, len(recs), n_les)
            out[f"FND_heatTTA|K{K}|{kind}"] = {"froc": f_, "patient_auc": a_}
            print(f"  FND K{K} {kind:14s} FROC {f_} | AUC {a_}", flush=True)
        recs = as_recs("e_fusion_tda")
        fh, ah = v62.froc_fixed(v62.hierarchical(recs), len(recs), n_les)
        out[f"FND_heatTTA|K{K}|f_hierarchical"] = {"froc": fh, "patient_auc": ah}
        print(f"  FND K{K} {'f_hierarchical':14s} FROC {fh} | AUC {ah}", flush=True)
    rep["val_stage2_foundation"] = out
    save_report(rep)
    print(f"  -> kaydedildi: {OUT}", flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val_stage1",
                    choices=["val_stage1", "vindr_stage1", "val_stage2", "vindr_e2e"])
    ap.add_argument("--ckpt", default="M5_foundation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = load_report()
    rep["ckpt_tag"] = args.ckpt
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()

    if args.stage == "val_stage1":
        fnd = load_foundation(args.ckpt, device)
        stage_val_stage1(m5, fnd, device, rep)
    elif args.stage == "vindr_stage1":
        fnd = load_foundation(args.ckpt, device)
        stage_vindr_stage1(m5, fnd, device, rep)
    elif args.stage == "val_stage2":
        fnd = load_foundation(args.ckpt, device)
        stage_val_stage2(m5, fnd, device, rep)
    else:
        raise SystemExit("vindr_e2e ayrı betikte (v6_6_vindr_e2e.py)")


if __name__ == "__main__":
    main()
