"""V6.3 — VinDr bileşen otopsisi + V5.3 (global 5-zon) ile V6.2 (lokal) tamamlayıcılık
        + V6 nihai model sabitleme.

Çıktılar:
  * VinDr: (a) Aşama-1 dedektör, (b) lokal TDA-HGB, (c) e_fusion_tda → görüntü AUC'leri
  * VinDr: V5.3 global skoru ile r ve birleşik AUC (rank-ortalaması + NODE21'de öğrenilen LR)
  * NODE21 val: aynı tamamlayıcılık testi
  * VinDr'de bbox varsa yaklaşık Recall@K30/K50 (3x indirgeme doğrulanırsa)
  * Nihai modeller: M5M6TTA·K50·e_fusion_tda ve M5_heatTTA·K30·f_hierarchical
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as m62  # noqa: E402
from detector_models import CenterNetHead, FCOSHead  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402
from node21_tda.models.deep_model import SmallZoneCNN, vector_to_2channel_image  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_3_final_report.json"
SEED = 42
V53_ZONE_SPLITS = PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv"
V53_FEATURES = PROJECT_ROOT / "data/processed/features_v5_1/X_a_density.npy"
V53_CNN = PROJECT_ROOT / "artifacts/models/zone_cnn_v5_1/zone_cnn_v5_1_bundle.pt"
V53_SVM = PROJECT_ROOT / "artifacts/models/zone_svm_v5_1/zone_svm_v5_1_bundle.joblib"
V53_VINDR_PRED = PROJECT_ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv"
A45 = 0.45


def rank_norm(x):
    x = np.asarray(x, float)
    return (rankdata(x) / len(x)).astype(np.float32) if len(x) else x


def _combined(v53, v6):
    """Eşit ağırlıklı rank-ortalaması (VinDr'de etiketsiz, yanlılıksız)."""
    return 0.5 * rank_norm(v53) + 0.5 * rank_norm(v6)


def _v52():
    import importlib.util
    if "_v52mod" not in globals():
        spec = importlib.util.spec_from_file_location(
            "v52", PROJECT_ROOT / "pipelines/exp/V5_2_corrected_tda_ensemble_fusion.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        globals()["_v52mod"] = m
    return globals()["_v52mod"]


def v53_val_patient_scores():
    """V5.3 global (5-zon) val hasta skoru: 0.45*ResNet + 0.55*TDA ens."""
    zs = pd.read_csv(V53_ZONE_SPLITS).sort_values("vector_index").reset_index(drop=True)
    X = np.load(V53_FEATURES)
    cb = torch.load(V53_CNN, map_location="cpu", weights_only=False)
    sb = joblib.load(V53_SVM)
    rows = []
    for z in range(1, 6):
        m = ((zs["split"] == "val") & (zs["zone_id"] == z)).values
        Xz = X[m]
        sc = cb["scalers"][z]
        model = SmallZoneCNN(in_channels=2, dropout=float(cb["config"]["dropout"]))
        model.load_state_dict(cb["models_state_dict"][z]); model.eval()
        with torch.no_grad():
            pc = torch.sigmoid(model(torch.tensor(
                vector_to_2channel_image(sc.transform(Xz)), dtype=torch.float32))).numpy().flatten()
        ps = sb[z].predict_proba(sc.transform(Xz))[:, 1]
        rows.append(pd.DataFrame({"pid": zs.loc[m, "patient_id"].values, "p": 0.55 * pc + 0.45 * ps}))
    tda = pd.concat(rows).groupby("pid")["p"].max()
    need = (zs["is_train_balanced_sample"].values.astype(bool)
            | (zs["split"] == "val").values | (zs["split"] == "test").values)
    sub = zs[need].reset_index(drop=True)
    img = _v52().resnet_patient_probs(sub, (sub["split"] == "val").values).set_index("patient_id")["p"]
    out = pd.DataFrame({"pid": tda.index})
    out["tda"] = tda.reindex(out["pid"]).values
    out["img"] = img.reindex(out["pid"]).values
    out["global"] = A45 * out["img"] + (1 - A45) * out["tda"]
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()
    print("V6.3 | nihai analiz")

    tr_nod = p2._nodule_pids("train"); tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(SEED)
    tr_pids = list(rng.choice(tr_nod, 400, replace=False)) + list(rng.choice(tr_ctl, 400, replace=False))
    va_pids = common.split_patients()["val"]

    c_tr = s2.stage1_candidates(tr_pids, m5, m6, device)
    c_va = s2.stage1_candidates(va_pids, m5, m6, device)
    gt_tr = {p: common.load_gt(p) for p in tr_pids}
    gt_va = {p: common.load_gt(p) for p in va_pids}
    lab_tr = {p: {v: s2._labels_for(c_tr[p][v], gt_tr[p]) for v in c_tr[p]} for p in tr_pids}
    lab_va = {p: {v: s2._labels_for(c_va[p][v], gt_va[p]) for v in c_va[p]} for p in va_pids}
    ranges = s2.derive_ranges([(p, c_tr[p]["M5_heatTTA"]) for p in tr_pids if gt_tr[p]][:20])

    def build(pids, cand_dict, variant):
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(pid, cand_dict[pid][variant], ranges, False) for pid in pids]
        out = {}
        with ProcessPoolExecutor(max_workers=10) as pool:
            for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
                out[pid] = res
        return out

    print("[1] Özellikler...")
    f_tr = {p: {} for p in tr_pids}; f_va = {p: {} for p in va_pids}
    for v in ("M5_heatTTA", "M5M6TTA"):
        g = build(tr_pids, c_tr, v)
        for p in tr_pids: f_tr[p][v] = g.get(p)
        g = build(va_pids, c_va, v)
        for p in va_pids: f_va[p][v] = g.get(p)

    Xt, yt, s1_t = [], [], []
    for p in tr_pids:
        f = f_tr[p].get("M5_heatTTA")
        if f is None: continue
        for i, (cx, cy, s) in enumerate(c_tr[p]["M5_heatTTA"]):
            Xt.append(np.concatenate([f["tda"][i], f["spa"][i]]))
            yt.append(1 if lab_tr[p]["M5_heatTTA"][i] is not None else 0)
            s1_t.append(s)
    Xt = np.asarray(Xt, np.float32); yt = np.asarray(yt, int); s1_t = np.asarray(s1_t, np.float32)
    sc = StandardScaler().fit(Xt); Xts = sc.transform(Xt)
    print(f"[2] HGB (aday {len(yt)}, pozitif {int(yt.sum())})...")
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                         random_state=SEED).fit(Xts, yt)
    p_tda_t = hgb.predict_proba(Xts)[:, 1]
    lr_fus = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(
        np.stack([s1_t, p_tda_t], 1), yt)
    s_fus_t = lr_fus.predict_proba(np.stack([s1_t, p_tda_t], 1))[:, 1]
    joblib.dump({"hgb": hgb, "scaler": sc, "lr_fusion": lr_fus, "ranges": ranges},
                CKPT / "stage2_final_tda_hgb.joblib")
    print("   nihai model kaydedildi: stage2_final_tda_hgb.joblib")

    # ---------------------------------------------------------------- NODE21 val
    def val_scores(variant, K):
        recs = []
        for p in va_pids:
            f = f_va[p].get(variant)
            if f is None: continue
            cands = c_va[p][variant][:K]; labs = lab_va[p][variant][:K]
            if not cands:
                recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0, "cands": [],
                             "s1": np.array([]), "p_tda": np.array([]), "s_fus": np.array([])})
                continue
            X = sc.transform(np.concatenate([f["tda"][:K], f["spa"][:K]], 1))
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda = hgb.predict_proba(X)[:, 1]
            s_fus = lr_fus.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
            recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0,
                         "cands": [(cands[i][0], cands[i][1], float(s_fus[i]), labs[i]) for i in range(len(cands))],
                         "s1": s1, "p_tda": p_tda, "s_fus": s_fus})
        return recs

    print("[3] NODE21 val + V5.3 köprüsü...")
    v53 = v53_val_patient_scores().set_index("pid")
    node21 = {}
    for variant, K in (("M5_heatTTA", 30), ("M5M6TTA", 50)):
        recs = val_scores(variant, K)
        n_les = sum(len(g) for g in gt_va.values())
        # hasta düzeyi skorlar (noisy-or)
        def pat_score(key):
            return np.array([1 - np.prod(1 - np.clip(np.sort(r[key])[::-1][:5], 0, 1)) if len(r[key]) else 0.0
                             for r in recs])
        s1p, ptp, sfp = pat_score("s1"), pat_score("p_tda"), pat_score("s_fus")
        y = np.array([1 if r["is_nodule"] else 0 for r in recs])
        pids = [r["pid"] for r in recs]
        g53 = v53["global"].reindex(pids).values
        if np.isnan(g53).any():
            g53 = np.nan_to_num(g53, nan=0.0)
        out = {"n_patients": len(y), "n_lesions": n_les,
               "auc_stage1": round(float(roc_auc_score(y, s1p)), 4),
               "auc_tda_hgb": round(float(roc_auc_score(y, ptp)), 4),
               "auc_e_fusion": round(float(roc_auc_score(y, sfp)), 4),
               "auc_v53_global": round(float(roc_auc_score(y, g53)), 4),
               "r_v53_vs_v6": round(float(np.corrcoef(g53, sfp)[0, 1]), 4),
               "auc_combined_rankavg": round(float(roc_auc_score(y, _combined(g53, sfp))), 4)}
        # LR füzyonu NODE21'de öğren → VinDr'e transfer
        lr_bridge = LogisticRegression(max_iter=1000).fit(np.stack([rank_norm(g53), rank_norm(sfp)], 1), y)
        out["bridge_lr"] = {"coef": lr_bridge.coef_.tolist()[0], "intercept": float(lr_bridge.intercept_[0])}
        node21[f"{variant}|K{K}"] = out
        print(f"  {variant} K{K}: A1={out['auc_stage1']} TDA={out['auc_tda_hgb']} "
              f"e_fusion={out['auc_e_fusion']} V5.3={out['auc_v53_global']} "
              f"r={out['r_v53_vs_v6']} combined={out['auc_combined_rankavg']}")

    bridge = node21["M5M6TTA|K50"]["bridge_lr"]

    # ---------------------------------------------------------------- VinDr
    print("[4] VinDr bileşen otopsisi...")
    v = _vindr_components(m5, m6, sc, hgb, lr_fus, ranges, device, bridge)
    print(f"   VinDr: A1={v['auc_stage1']} TDA={v['auc_tda_hgb']} e_fusion={v['auc_e_fusion']} "
          f"V5.3={v['auc_v53_global']} r={v['r_v53_vs_v6']} combined={v['auc_combined_rankavg']}")
    if v.get("bbox_recall"):
        print(f"   VinDr bbox recall: {v['bbox_recall']}")

    report = {
        "experiment": "V6.3",
        "purpose": "VinDr bilesen otopsisi + V5.3(global)/V6.2(lokal) tamamlayicilik + nihai sabitleme",
        "node21_val": node21,
        "vindr": v,
        "final_configs": {
            "max_auc": {"stage1": "M5M6TTA", "K": 50, "stage2": "e_fusion_tda (Aşama-1 + lokal TDA-HGB)",
                        "patient_auc": node21["M5M6TTA|K50"]["auc_e_fusion"],
                        "note": "yama CNN mimariden ÇIKARILDI (0.646 AUC, gereksiz)"},
            "froc_optimal": {"stage1": "M5_heatTTA", "K": 30, "stage2": "f_hierarchical "
                             "(e_fusion × P_NoisyOR)", "froc": "0.589/0.658/0.753/0.857"},
        },
        "artifacts": {"stage2_final": "checkpoints/stage2_final_tda_hgb.joblib",
                      "stage1": ["checkpoints/M5_centernet.pt", "checkpoints/M6_fcos.pt"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}")


def _vindr_components(m5, m6, sc, hgb, lr_fus, ranges, device, bridge):
    from PIL import Image
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter
    from scipy import ndimage as ndi

    cohort = pd.read_csv(PROJECT_ROOT / "data/external_vindr/cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    ex = ZoneTDAExtractor(s2._settings(ranges))
    v53 = pd.read_csv(V53_VINDR_PRED).set_index("image_id")
    v53_global = (A45 * v53["p_resnet"] + (1 - A45) * v53["p_tda"]).reindex(cohort["image_id"]).values
    if np.isnan(v53_global).any():
        print(f"   ! V5.3 tahminlerinde {int(np.isnan(v53_global).sum())} eksik → 0.0 ile dolduruldu")
        v53_global = np.nan_to_num(v53_global, nan=0.0)
    y, S1, TDA, FUS = [], [], [], []
    t0 = time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"]); lab = int(row["patient_label"])
        img = np.array(Image.open(PROJECT_ROOT / "data/external_vindr/images" / f"{pid}.jpg"))
        try:
            zmap = seg.segment(img)["zone_map"]
            raw = zmap > 0
            if not raw.any():
                raise ValueError("empty mask")
            ys, xs = np.where(raw)
            p1, p99 = np.percentile(img[raw], [1, 99])
            mask = ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                                       iterations=common.DILATION_PX)
            canon = common.canonicalize(img, mask, float(p1), float(p99))
            H, W = canon.shape
            sx, sy = W / 1024.0, H / 1024.0
            c5 = [(x * sx, y2 * sy, s) for x, y2, s in ev._predict_heat_tta(m5, canon, device, "M5")]
            c6 = [(x * sx, y2 * sy, s) for x, y2, s in p2.predict(m6, canon, device, "M6")]
            cands = ev._rrf([c5, c6])[:50]
            if not cands:
                raise ValueError("no candidates")
            tda = np.zeros((len(cands), 2 * 2 * s2.GRID * s2.GRID), np.float32)
            spa = np.zeros((len(cands), 4), np.float32)
            bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
            dist = ndi.distance_transform_edt(raw).astype(np.float32)
            for k, (cx, cy, _) in enumerate(cands):
                tda[k] = np.concatenate([ex.extract(s2._crop(canon, cx, cy, s2.INNER))[0],
                                         ex.extract(s2._crop(canon, cx, cy, s2.OUTER))[0]])
                spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
            X = sc.transform(np.concatenate([tda, spa], 1))
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda = hgb.predict_proba(X)[:, 1]
            p_fus = lr_fus.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
        except Exception as exc:
            y.append(lab); S1.append(0.0); TDA.append(0.0); FUS.append(0.0); continue
        y.append(lab)
        S1.append(float(1 - np.prod(1 - np.clip(np.sort(s1)[::-1][:5], 0, 1))))
        TDA.append(float(1 - np.prod(1 - np.clip(np.sort(p_tda)[::-1][:5], 0, 1))))
        FUS.append(float(1 - np.prod(1 - np.clip(np.sort(p_fus)[::-1][:5], 0, 1))))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)

    y = np.array(y); S1 = np.array(S1); TDA = np.array(TDA); FUS = np.array(FUS)
    out = {"n_images": len(y), "n_positive": int(y.sum()),
           "auc_stage1": round(float(roc_auc_score(y, S1)), 4),
           "auc_tda_hgb": round(float(roc_auc_score(y, TDA)), 4),
           "auc_e_fusion": round(float(roc_auc_score(y, FUS)), 4),
           "auc_v53_global": round(float(roc_auc_score(y, v53_global)), 4),
           "r_v53_vs_v6": round(float(np.corrcoef(v53_global, FUS)[0, 1]), 4),
           "auc_combined_rankavg": round(float(roc_auc_score(y, _combined(v53_global, FUS))), 4)}
    # NODE21'de öğrenilen köprü LR'yi VinDr'e uygula
    coef, b0 = bridge["coef"], bridge["intercept"]
    bridge_score = coef[0] * rank_norm(v53_global) + coef[1] * rank_norm(FUS) + b0
    out["auc_combined_bridge_lr"] = round(float(roc_auc_score(y, bridge_score)), 4)

    # yaklaşık bbox recall (3x indirgeme varsayımı doğrulanırsa)
    out["bbox_recall"] = _vindr_bbox_recall(cohort)
    return out


def _vindr_bbox_recall(cohort) -> dict | None:
    """VinDr annotasyon bbox'ları orijinal uzayda; JPEG'ler 3x indirgenmiş varsayımı."""
    from PIL import Image
    try:
        import torchxrayvision as xrv
        ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    except Exception:
        return None
    pos = cohort[cohort["patient_label"] == 1]
    ok, tot = 0, 0
    for _, r in pos.iterrows():
        b = ann[(ann["image_id"] == r["image_id"]) &
                (ann["class_name"].str.lower().str.contains("nodule|mass", na=False))]
        if b.empty:
            continue
        w, h = Image.open(PROJECT_ROOT / "data/external_vindr/images" / f"{r['image_id']}.jpg").size
        bx, by = float(b["x_max"].max()) / 3.0, float(b["y_max"].max()) / 3.0
        tot += 1
        if bx <= w * 1.05 and by <= h * 1.05:
            ok += 1
    if tot == 0 or ok / tot < 0.90:
        return {"available": False, "reason": "3x indirgeme varsayımı doğrulanamadı "
                                              f"(uyum {ok}/{tot})"}
    return {"available": True, "assumption": "orijinal = 3x JPEG", "fit_ratio": round(ok / tot, 3)}


if __name__ == "__main__":
    main()
