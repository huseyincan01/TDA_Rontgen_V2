"""V6.5c — Ayristirilmis Cift-Kafa + NODE21 Oncul Capasi.

Kafa 1 (hasta kapisi): V6.5'teki goruntu-etiketli kalibrasyon (1604-boyut, LR C=0.1)
  -> P_patient = noisy-OR(top-5). Hasta AUC'yi korur.
Kafa 2 (kutu ici siralayici): yalniz POZITIF goruntulerin icinde
  gercek nodul kutusu vs ayni goruntunun en yuksek skorlu 5 sahte kutusu (1:5)
  uzerinde egitilir; NODE21 sifir-egitim skoruna Bayesci daralma ile baglanir:
      P_box = (1-alpha_N)*P_zero-shot + alpha_N*P_vindr-bbox,  alpha_N = N/(N+40)

Nihai FROC skoru: S_final(i) = P_patient^gamma * P_box(i)
  (ek olarak goruntu-ici max-normalize varyant: P_box(i)/max_j P_box(j))

Protokol: 5-fold stratified CV, N in {0,10,20,40,80,160}; Asama-1 donmus.
Ozellik uzayi: KOL1'in en iyi konfigurasyonu (despeckle sigma=1.0 + olcek-eslemeli yama),
CORAL ile NODE21 referansina hizali. Kafa-2 icin PCA-128 (NODE21'de fit, hedef etiketsiz).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import candidates_classical as cc  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as m62  # noqa: E402
import v6_4_domain_and_seal as m64  # noqa: E402
import v6_5_vindr_transfer_boost as v65  # noqa: E402
import v6_5b_fewshot_compact as v65b  # noqa: E402
from detector_models import CenterNetHead  # noqa: E402
from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter  # noqa: E402

CKPT = ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
EXT = ROOT / "data/external_vindr"
OUT = ROOT / "artifacts" / "v6_two_stage" / "v6_5c_decoupled_calibration.json"
INPUT = 512
A45 = 0.45
SEED = 42
N_FEW = (0, 10, 20, 40, 80, 160)
GAMMAS = (0.5, 1.0, 2.0)
NW = max(2, min(10, (os.cpu_count() or 4) - 2))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device)
    m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device))
    m5.eval()
    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    hgb, lr_fus, ranges = b["hgb"], b["lr_fusion"], b["ranges"]

    import torchxrayvision as xrv

    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(EXT / "cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    print("[0] VinDr hazirlik...", flush=True)
    data, t0 = [], time.time()
    for i, row in cohort.iterrows():
        pid, lab = str(row["image_id"]), int(row["patient_label"])
        img = np.array(Image.open(EXT / "images" / f"{pid}.jpg"))
        gts = []
        for _, bb in nod[nod["image_id"] == pid].iterrows():
            x0, x1 = bb["x_min"] / 3.0, bb["x_max"] / 3.0
            y0, y1 = bb["y_min"] / 3.0, bb["y_max"] / 3.0
            if x1 > x0 and y1 > y0:
                gts.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
        try:
            raw = seg.segment(img)["zone_map"] > 0
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                                       iterations=common.DILATION_PX)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            canon_u8 = np.clip(np.rint(canon), 0, 255).astype(np.uint8)
            x512 = np.array(Image.fromarray(canon_u8).resize((INPUT, INPUT), Image.Resampling.BILINEAR))
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
            print(f"    {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)

    print("[1] AdaBN + adaylar...", flush=True)
    m5a = m64.adabn(m5, [m64._norm512(d["x512"]) for d in data], device)
    cands = {}
    for d in data:
        try:
            cands[d["pid"]] = cc._nms(m64._pred_from_512(m5a, d["x512"], d["canon"].shape, device, "M5"), 20, 50)
        except Exception:
            cands[d["pid"]] = []

    print("[2] Referans + ozellikler (despeckle+olcek)...", flush=True)
    from concurrent.futures import ProcessPoolExecutor

    tr = list(np.random.RandomState(SEED).choice(p2._nodule_pids("train"), 100, replace=False)) + \
        list(np.random.RandomState(SEED + 1).choice(p2._control_pids("train"), 100, replace=False))
    c_tr = s2.stage1_candidates(tr, m5, m64.load_models(device)[1], device, top=20)
    ref_jobs = [(list(p), {q: c_tr[q]["M5_heatTTA"] for q in p}, ranges) for p in np.array_split(np.array(tr), NW)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as ex:
        ref = list(ex.map(v65._w_ref, ref_jobs))
    X_ref = np.stack([np.concatenate([t, s]) for r in ref for t, s in zip(r[0], r[1])]).astype(np.float32)
    print(f"    referans: {len(X_ref)} ({time.time() - t0:.0f}s)", flush=True)

    chunks = [([{"pid": data[i]["pid"], "canon": data[i]["canon"], "raw": data[i]["raw"],
                 "lung_h": data[i]["lung_h"], "cands": cands[data[i]["pid"]]} for i in part], ranges)
              for part in np.array_split(np.arange(len(data)), NW)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NW) as ex:
        res = list(ex.map(v65b._w, chunks))
    feats = {}
    for pids, out in res:
        feats.update(dict(zip(pids, out)))
    print(f"    ozellikler: {time.time() - t0:.0f}s", flush=True)

    aligner = v65.Aligner(X_ref)
    blocks, counts = [], []
    for d in data:
        c = cands[d["pid"]]
        counts.append(len(c))
        if len(c):
            blocks.append(np.concatenate([feats[d["pid"]]["tda"], feats[d["pid"]]["spa"]], 1))
    X = aligner.coral(np.concatenate(blocks))
    Xref_al = aligner.coral(X_ref)
    pca = PCA(n_components=128, random_state=SEED).fit(Xref_al)
    Xp = pca.transform(X).astype(np.float32)

    hits = {d["pid"]: s2._labels_for(cands[d["pid"]], d["gts"]) for d in data}
    y_img = np.array([d["y"] for d in data])
    Xall, Xpall, s1all, k = [], [], [], 0
    for d, n in zip(data, counts):
        Xall.append(X[k:k + n])
        Xpall.append(Xp[k:k + n])
        s1all.append(np.array([c[2] for c in cands[d["pid"]]], np.float32))
        k += n

    # sifir-egitim (N=0) aday skorlari: tum foldlar icin ortak
    p_zs = [hgb.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in range(len(data))]
    p_zs_fus = [lr_fus.predict_proba(np.stack([s1all[i], p_zs[i]], 1))[:, 1]
                if len(p_zs[i]) else np.zeros(0) for i in range(len(data))]

    def patient_score(p_tda, i):
        if len(p_tda) == 0:
            return 0.0
        return v65.image_score_from_cands(lr_fus.predict_proba(np.stack([s1all[i], p_tda], 1))[:, 1])

    v53 = pd.read_csv(ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv").set_index("image_id")
    v53 = v53.reindex([d["pid"] for d in data])
    g53 = np.nan_to_num((A45 * v53["p_resnet"] + (1 - A45) * v53["p_tda"]).values, nan=0.0)

    rng = np.random.RandomState(SEED)
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % 5

    rows = {N: {"auc": [], "froc": {g: [] for g in GAMMAS}, "froc_norm": {g: [] for g in GAMMAS},
                "auc_final": {g: [] for g in GAMMAS}, "comb": []} for N in N_FEW}
    for N in N_FEW:
        for f in range(5):
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]

            # ---- Kafa 1: hasta kapisi
            if N == 0:
                p1_te = [p_zs[i] for i in te_idx]
            else:
                Xc = np.concatenate([Xall[i] for i in tr_idx if len(Xall[i])])
                yc = np.concatenate([np.full(len(Xall[i]), y_img[i]) for i in tr_idx if len(Xall[i])])
                if len(set(yc.tolist())) < 2:
                    p1_te = [p_zs[i] for i in te_idx]
                else:
                    lr1 = LogisticRegression(class_weight="balanced", max_iter=400, C=0.1,
                                             random_state=SEED).fit(Xc, yc)
                    p1_te = [lr1.predict_proba(Xall[i])[:, 1] if len(Xall[i]) else np.zeros(0) for i in te_idx]
            p_patient = np.array([patient_score(p1_te[k], i) for k, i in enumerate(te_idx)])

            # ---- Kafa 2: kutu ici siralayici (1:5 within-patient) + NODE21 capasi
            pos_tr = [i for i in tr_idx if y_img[i] == 1 and any(h is not None for h in hits[data[i]["pid"]])]
            if N == 0 or not pos_tr:
                alpha = 0.0
                pbox_te = [p_zs_fus[i] for i in te_idx]
            else:
                Xn, yn = [], []
                for i in pos_tr:
                    c = cands[data[i]["pid"]]
                    lb = hits[data[i]["pid"]]
                    hit_j = [j for j, h in enumerate(lb) if h is not None]
                    neg_j = [j for j, h in enumerate(lb) if h is None]
                    # ayni goruntunun en yuksek skorlu 5 sahte kutusu (zero-shot)
                    neg_j = sorted(neg_j, key=lambda j: -p_zs_fus[i][j])[:5]
                    if not hit_j or not neg_j:
                        continue
                    for j in hit_j:
                        Xn.append(Xpall[i][j]); yn.append(1)
                    for j in neg_j:
                        Xn.append(Xpall[i][j]); yn.append(0)
                alpha = N / (N + 40.0)
                if len(set(yn)) < 2:
                    pbox_te = [p_zs_fus[i] for i in te_idx]
                else:
                    lr2 = LogisticRegression(class_weight="balanced", max_iter=500, C=0.1,
                                             random_state=SEED).fit(np.stack(Xn), np.array(yn))
                    pbox_te = []
                    for i in te_idx:
                        pv = lr2.predict_proba(Xpall[i])[:, 1] if len(Xpall[i]) else np.zeros(0)
                        pz = p_zs_fus[i]
                        pbox_te.append((1 - alpha) * pz + alpha * pv if len(pz) else pz)

            # ---- Nihai skor ve metrikler
            te_y = y_img[te_idx]
            ok = len(set(te_y)) > 1
            rows[N]["auc"].append(roc_auc_score(te_y, p_patient) if ok else np.nan)
            comb = 0.5 * rankdata(g53[te_idx]) / len(te_idx) + 0.5 * rankdata(p_patient) / len(te_idx)
            rows[N]["comb"].append(roc_auc_score(te_y, comb) if ok else np.nan)
            n_les = sum(len(data[i]["gts"]) for i in te_idx)
            for g in GAMMAS:
                recs, recs_n, img_max = [], [], []
                for kk, i in enumerate(te_idx):
                    c = cands[data[i]["pid"]]
                    lb = hits[data[i]["pid"]]
                    pb = np.clip(pbox_te[kk], 1e-6, 1.0)
                    sf = (p_patient[kk] ** g) * pb
                    mn = pb / pb.max() if pb.size else pb
                    sf_n = (p_patient[kk] ** g) * mn
                    recs.append({"pid": data[i]["pid"], "is_nodule": y_img[i] == 1,
                                 "cands": [(c[j][0], c[j][1], float(sf[j]), lb[j]) for j in range(len(c))]})
                    recs_n.append({"pid": data[i]["pid"], "is_nodule": y_img[i] == 1,
                                   "cands": [(c[j][0], c[j][1], float(sf_n[j]), lb[j]) for j in range(len(c))]})
                    img_max.append(float(sf.max()) if sf.size else 0.0)
                rows[N]["froc"][g].append(m62.froc_fixed(recs, len(te_idx), n_les)[0])
                rows[N]["froc_norm"][g].append(m62.froc_fixed(recs_n, len(te_idx), n_les)[0])
                rows[N]["auc_final"][g].append(roc_auc_score(te_y, img_max) if ok else np.nan)

    curve = {}
    for N in N_FEW:
        a = np.array(rows[N]["auc"], float)
        entry = {"patient_auc_mean": round(float(np.nanmean(a)), 4),
                 "patient_auc_std": round(float(np.nanstd(a)), 4),
                 "combined_auc_mean": round(float(np.nanmean(np.array(rows[N]["comb"], float))), 4),
                 "combined_auc_std": round(float(np.nanstd(np.array(rows[N]["comb"], float))), 4)}
        for g in GAMMAS:
            entry[f"froc_g{g}"] = {k2: round(float(np.mean([fr[k2] for fr in rows[N]["froc"][g]])), 4)
                                   for k2 in ("0.5", "1", "2", "4")}
            entry[f"froc_norm_g{g}"] = {k2: round(float(np.mean([fr[k2] for fr in rows[N]["froc_norm"][g]])), 4)
                                        for k2 in ("0.5", "1", "2", "4")}
            entry[f"auc_final_g{g}"] = round(float(np.nanmean(np.array(rows[N]["auc_final"][g], float))), 4)
        curve[str(N)] = entry
        print(f"    N={N:3d} | P_patient {entry['patient_auc_mean']:.4f}±{entry['patient_auc_std']:.4f} "
              f"| FROC@2 g1 {entry['froc_g1.0']['2']:.3f} | norm {entry['froc_norm_g1.0']['2']:.3f} "
              f"| AUC_final g1 {entry['auc_final_g1.0']:.4f}", flush=True)

    best = max(curve, key=lambda n: (curve[n]["froc_g1.0"]["2"], curve[n]["patient_auc_mean"]))
    report = {
        "experiment": "V6.5c",
        "design": "Decoupled dual-head: patient gate (image-label LR) + within-patient 1:5 box ranker (PCA-128) "
                  "with NODE21 zero-shot shrinkage alpha=N/(N+40); final S = P_patient^gamma * P_box.",
        "gamma_default": 1.0,
        "folds": 5,
        "N": list(N_FEW),
        "curve": curve,
        "verification": {
            "best_froc2_N": int(best),
            "best_froc2_at_2fp": curve[best]["froc_g1.0"]["2"],
            "patient_auc_at_best": curve[best]["patient_auc_mean"],
            "auc_ge_0.724": bool(curve[best]["patient_auc_mean"] >= 0.724),
            "froc2_ge_0.22": bool(curve[best]["froc_g1.0"]["2"] >= 0.22),
        },
        "note": "Kafa-1 sabit tutuldugu icin Hasta AUC kaybi yok; Kafa-2 yalniz pozitif goruntulerin "
                "1:5 within-patient karsilastirmasiyla egitilir. Ozellikler RAM'de yeniden cikarildi "
                "(kalici onbellek yok).",
    }
    OUT.write_text(json.dumps(v65._san(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}", flush=True)


if __name__ == "__main__":
    main()
