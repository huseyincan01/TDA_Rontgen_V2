"""V6.6 — VinDr-CXR UÇTAN UCA transfer (foundation Stage-1 + lokal TDA 2. aşama).

Üç protokol (V6.5 şablonu, foundation adaylarıyla):
  (A) STRICT zero-shot (inductive) : AdaBN YOK, CORAL YOK — yalnız tekil-görüntü
                                     fiziksel hizalama (denoise / ölçek).
  (B) TRANSDUCTIVE zero-shot       : AdaBN (VinDr etiketsiz BN) + denoise + ölçek + CORAL.
  (C) FEW-SHOT hasta kalibrasyonu   : N in {0,10,20,40,80,160}, 5-fold CV, VinDr ETİKETLİ.

Kayıt: artifacts/v6_two_stage/v6_6_vindr_e2e.json
Disk kuralı: kırpıntı/görüntü YAZILMAZ; TDA RAM'de on-the-fly.
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as v62  # noqa: E402
import v6_5_vindr_transfer_boost as v65  # noqa: E402
import v6_5d_strict_no_adabn as v65d  # noqa: E402
from detector_models_foundation import FoundationCenterNet, predict_tta  # noqa: E402
from v6_6_foundation_stage1 import adabn_foundation  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_vindr_e2e.json"
SEED = 42
NW = v65.NW
KEYS = [("fixed", 0.0), ("fixed", 1.0), ("scale", 0.0), ("scale", 1.0)]
v65.KEYS = list(KEYS)                     # spawn altında çocuk süreç de bunu görsün
CONFIGS = [
    ("base_z", "fixed", 0.0, "z"),
    ("denoise1.0_z", "fixed", 1.0, "z"),
    ("scale_z", "scale", 0.0, "z"),
    ("denoise1.0_scale_z", "scale", 1.0, "z"),
    ("denoise1.0_scale_coral", "scale", 1.0, "coral"),
]
N_FEW = (0, 10, 20, 40, 80, 160)


def load_foundation(tag: str, device):
    m = FoundationCenterNet(weights="resnet50-res512-all").to(device)
    m.load_state_dict(torch.load(CKPT / f"{tag}.pt", map_location=device))
    m.eval()
    return m


def _san(o):
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.floating, float)) and not np.isfinite(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def image_score(scores) -> float:
    if len(scores) == 0:
        return 0.0
    return float(1 - np.prod(1 - np.clip(np.sort(scores)[::-1][:5], 0, 1)))


def eval_config_froc(data, cands, feats, aligner, mode, hgb, lr, hits):
    """AUC + FROC (düzeltilmiş protokol) — hizalamayı tüm aday matrisine tek seferde uygular."""
    blocks, counts, s1s = [], [], []
    for d, f in zip(data, feats):
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([f["tda"], f["spa"]], 1))
            s1s.append(np.array([x[2] for x in c], np.float32))
    if not blocks:
        return {"auc": None, "froc": None}
    X = np.concatenate(blocks)
    X = aligner.z(X) if mode == "z" else (aligner.quantile(X) if mode == "quantile"
                                         else aligner.coral(X))
    p_tda = hgb.predict_proba(X)[:, 1]
    p_fus = lr.predict_proba(np.stack([np.concatenate(s1s), p_tda], 1))[:, 1]
    y, scores, recs, k = [], [], [], 0
    for d, n in zip(data, counts):
        y.append(d["y"])
        scores.append(image_score(p_fus[k:k + n]))
        c = cands[d["pid"]]
        lb = hits[d["pid"]]
        recs.append({"pid": d["pid"], "is_nodule": d["y"] == 1,
                     "cands": [(c[j][0], c[j][1], float(p_fus[k + j]), lb[j]) for j in range(n)]})
        k += n
    n_les = sum(len(d["gts"]) for d in data)
    fr, _ = v62.froc_fixed(recs, len(recs), n_les)
    return {"auc": round(float(roc_auc_score(y, scores)), 4) if len(set(y)) > 1 else None,
            "froc": fr}


def few_shot(data, cands, feats, aligner, mode, hgb, lr, hits):
    """N etiketli hedef görüntüyle 2. aşama tablosal modelin yeniden uyarlanması (5-fold)."""
    blocks, counts = [], []
    for d, f in zip(data, feats):
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([f["tda"], f["spa"]], 1))
    X = np.concatenate(blocks) if blocks else np.zeros((0, 1604), np.float32)
    X = aligner.z(X) if mode == "z" else (aligner.quantile(X) if mode == "quantile"
                                         else aligner.coral(X))
    Xall, kk = [], 0
    for n in counts:
        Xall.append(X[kk:kk + n])
        kk += n
    y_img = np.array([d["y"] for d in data])
    rng = np.random.RandomState(SEED)
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5

    out = {}
    for N in N_FEW:
        aucs, frocs = [], []
        for f in range(5):
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]
            if N == 0:
                p_te = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0)
                        for i in te_idx]
            else:
                pos = [i for i in tr_idx if y_img[i] == 1]
                neg = [i for i in tr_idx if y_img[i] == 0]
                r2 = np.random.RandomState(SEED + N + f)
                npos = N // 2
                pick = list(r2.choice(pos, size=min(npos, len(pos)), replace=False)) + \
                    list(r2.choice(neg, size=min(N - npos, len(neg)), replace=False))
                Xc = np.concatenate([Xall[i] for i in pick if len(Xall[i])])
                yc = np.concatenate([np.full(len(Xall[i]), y_img[i]) for i in pick if len(Xall[i])])
                if len(set(yc.tolist())) < 2:
                    p_te = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0)
                            for i in te_idx]
                else:
                    lr_cal = LogisticRegression(class_weight="balanced", max_iter=300, C=0.1,
                                                random_state=SEED).fit(Xc, yc)
                    p_te = [lr_cal.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0)
                            for i in te_idx]
            te_y = y_img[te_idx]
            s_img, recs = [], []
            for k, i in enumerate(te_idx):
                c_ = cands[data[i]["pid"]]
                lb = hits[data[i]["pid"]]
                s1_ = np.array([x[2] for x in c_], np.float32)
                pf = (lr.predict_proba(np.stack([s1_, p_te[k]], 1))[:, 1]
                      if len(c_) and len(p_te[k]) else np.zeros(0))
                s_img.append(image_score(pf))
                recs.append({"pid": data[i]["pid"], "is_nodule": y_img[i] == 1,
                             "cands": [(c_[j][0], c_[j][1], float(pf[j]), lb[j])
                                       for j in range(len(c_))]})
            n_les = sum(len(data[i]["gts"]) for i in te_idx)
            fr, _ = v62.froc_fixed(recs, len(te_idx), n_les)
            frocs.append(fr)
            if len(set(te_y)) > 1:
                aucs.append(roc_auc_score(te_y, s_img))
        a = np.array(aucs, float)
        out[str(N)] = {
            "auc_mean": round(float(np.nanmean(a)), 4),
            "auc_std": round(float(np.nanstd(a)), 4),
            "froc_mean": {k: round(float(np.mean([fr[k] for fr in frocs])), 4)
                          for k in ("0.5", "1", "2", "4")},
        }
        print(f"    N={N:3d} AUC {out[str(N)]['auc_mean']:.4f}±{out[str(N)]['auc_std']:.4f} "
              f"| FROC@1 {out[str(N)]['froc_mean']['1']:.3f} "
              f"@2 {out[str(N)]['froc_mean']['2']:.3f}", flush=True)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="M5_foundation")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.6 VinDr uçtan uca | ckpt={args.ckpt} | device={device}", flush=True)

    import joblib
    import pandas as pd
    import torchxrayvision as xrv
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    sc, hgb, lr, ranges = b["scaler"], b["hgb"], b["lr_fusion"], b["ranges"]
    fnd = load_foundation(args.ckpt, device)

    # ---------------- VinDr veri
    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(v65d.EXT / "cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    print("\n[0] VinDr hazırlık...", flush=True)
    data = v65d.build_vindr(seg, cohort, nod)
    canons = [d["canon"] for d in data]

    # ---------------- adaylar
    print("\n[1] Foundation adayları (strict, AdaBN yok)...", flush=True)
    t0 = time.time()
    cands_strict = {d["pid"]: predict_tta(fnd, d["canon"].astype(np.float32), device,
                                          shape=d["canon"].shape) for d in data}
    print(f"    {sum(len(v) for v in cands_strict.values())} aday ({time.time() - t0:.0f}s)", flush=True)

    print("[2] AdaBN + foundation adayları...", flush=True)
    fnda = adabn_foundation(load_foundation(args.ckpt, device), canons, device)
    cands_ada = {d["pid"]: predict_tta(fnda, d["canon"].astype(np.float32), device,
                                       shape=d["canon"].shape) for d in data}
    print(f"    {sum(len(v) for v in cands_ada.values())} aday", flush=True)

    # ---------------- NODE21 referans (Aligner) — foundation adaylarıyla
    print("\n[3] NODE21 referans özellikleri (foundation adayları)...", flush=True)
    n_ref = 100
    tr_pids = list(np.random.RandomState(SEED).choice(p2._nodule_pids("train"), n_ref,
                                                      replace=False)) + \
        list(np.random.RandomState(SEED + 1).choice(p2._control_pids("train"), n_ref,
                                                    replace=False))
    c_tr = {}
    for i, pid in enumerate(tr_pids):
        canon, _, _ = p2._prep(pid)
        c_tr[pid] = predict_tta(fnd, canon, device)[:20]
    ref_parts = np.array_split(np.array(tr_pids), NW)
    ref_jobs = [(list(part), {p: c_tr[p] for p in part}, ranges) for part in ref_parts]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as pool:
        ref_res = list(pool.map(v65._w_ref, ref_jobs))
    ref_tda = [x for r in ref_res for x in r[0]]
    ref_spa = [x for r in ref_res for x in r[1]]
    X_ref = np.stack([np.concatenate([t, s]) for t, s in zip(ref_tda, ref_spa)]).astype(np.float32)
    print(f"    referans aday: {len(X_ref)} ({time.time() - t0:.0f}s)", flush=True)
    aligner = v65.Aligner(X_ref)

    # ---------------- VinDr özellikleri (her iki aday kümesi)
    def extract(cands, label):
        print(f"\n[4] VinDr lokal TDA özellikleri — {label}...", flush=True)
        parts = np.array_split(np.arange(len(data)), NW)
        jobs = []
        for part in parts:
            chunk = [{"pid": data[i]["pid"], "canon": data[i]["canon"], "raw": data[i]["raw"],
                      "lung_h": data[i]["lung_h"], "cands": cands[data[i]["pid"]]} for i in part]
            jobs.append((chunk, ranges))
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=NW) as pool:
            res = list(pool.map(v65._w_extract, jobs))
        cache = {k: {} for k in KEYS}
        for pids, out in res:
            for k in KEYS:
                for pid, f in zip(pids, out[k]):
                    cache[k][pid] = f
        print(f"    bitti ({time.time() - t0:.0f}s)", flush=True)
        return cache

    feats_strict = extract(cands_strict, "strict (AdaBN yok)")
    feats_ada = extract(cands_ada, "AdaBN")

    def evalset(cands, feats, tag):
        hits = {d["pid"]: s2._labels_for(cands[d["pid"]], d["gts"]) for d in data}
        res = {}
        print(f"\n[5] Konfigürasyonlar — {tag}", flush=True)
        for name, skind, sigma, mode in CONFIGS:
            fl = [feats[(skind, sigma)][d["pid"]] for d in data]
            r = eval_config_froc(data, cands, fl, aligner, mode, hgb, lr, hits)
            res[name] = r
            print(f"    {name:26s} AUC={r['auc']}  FROC={r['froc']}", flush=True)
        return res, hits

    strict_res, hits_s = evalset(cands_strict, feats_strict, "STRICT (inductive)")
    ada_res, hits_a = evalset(cands_ada, feats_ada, "TRANSDUCTIVE (AdaBN)")

    # ---------------- few-shot
    print("\n[6] Few-shot (5-fold) — STRICT tabanı (scale_z)...", flush=True)
    fs_strict = few_shot(data, cands_strict, [feats_strict[("scale", 0.0)][d["pid"]] for d in data],
                         aligner, "z", hgb, lr, hits_s)
    print("\n[7] Few-shot (5-fold) — TRANSDUCTIVE tabanı (AdaBN + denoise+ölçek+CORAL)...", flush=True)
    fs_ada = few_shot(data, cands_ada,
                      [feats_ada[("scale", 1.0)][d["pid"]] for d in data],
                      aligner, "coral", hgb, lr, hits_a)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_san({
        "experiment": "V6.6",
        "stage1": args.ckpt,
        "n_images": len(data),
        "n_positive": int(sum(d["y"] for d in data)),
        "n_candidates_strict": int(sum(len(v) for v in cands_strict.values())),
        "n_candidates_adabn": int(sum(len(v) for v in cands_ada.values())),
        "strict_zero_shot": strict_res,
        "transductive_zero_shot": ada_res,
        "few_shot_strict_scale_z": fs_strict,
        "few_shot_transductive_coral": fs_ada,
        "protocol": {"n_few": list(N_FEW), "folds": 5,
                     "candidate_label": "görüntü etiketi (V6.5 KOL2 ile aynı)"},
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}", flush=True)


if __name__ == "__main__":
    main()
