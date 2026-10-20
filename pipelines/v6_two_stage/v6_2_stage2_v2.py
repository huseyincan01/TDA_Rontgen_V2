"""V6.2 — FROC protokol düzeltmesi + doğrusal olmayan TDA topluluğu + hiyerarşik kalibrasyon
        + VinDr-CXR sıfır-eğitim karşılaştırması.

Düzeltmeler:
  1) (a) Post-fusion NMS, (b) duplikasyon ignore: zaten yakalanmış bir GT'ye düşen ikincil
     adaylar NE TP NE FP sayılır (LUNA16/NODE21 protokolü).
  2) 2. aşama TDA+uzamsal dalı: HGB + RBF-SVM + MLP topluluğu (rank-normalize ortalama).
     Ayrıca yama CNN'siz füzyon satırı: (Aşama-1 + güçlendirilmiş lokal TDA).
  3) Hiyerarşik kalibrasyon: S_final(i) = p_aday(i) × P_NoisyOR(hasta).

Kullanım:
    python pipelines/v6_two_stage/v6_2_stage2_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
from detector_models import CenterNetHead, FCOSHead  # noqa: E402
from node21_tda.features.extractor import ZoneTDAExtractor  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_2_froc_and_transfer.json"
SEED = 42
FP_TARGETS = (0.5, 1.0, 2.0, 4.0)
NMS_PX = 20


def rank_norm(x):
    x = np.asarray(x, float)
    return (rankdata(x) / len(x)).astype(np.float32) if len(x) else x


# ------------------------------------------------------------------ düzeltilmiş FROC
def froc_fixed(records, n_patients, n_lesions):
    """records: [{pid, is_nodule, cands:[(x,y,score,hit_idx_or_None)]}]
    Protokol: post-fusion NMS(20px) + zaten yakalanmış GT'ye düşen aday IGNORE."""
    flat = []
    for r in records:
        c = common_ish_nms(r["cands"])
        for x, y, s, hit in c:
            flat.append((s, r["pid"], hit))
    flat.sort(key=lambda t: -t[0])
    detected, fps = {}, 0
    pts = []
    n_det = 0
    for s, pid, hit in flat:
        if hit is not None:
            if hit in detected.setdefault(pid, set()):
                continue                     # IGNORE (duplikasyon): ne TP ne FP
            detected[pid].add(hit)
            n_det += 1
        else:
            fps += 1
        pts.append((fps / max(1, n_patients), n_det / max(1, n_lesions)))
    out = {}
    for t in FP_TARGETS:
        sens = max([sn for fp, sn in pts if fp <= t + 1e-9], default=0.0)
        out[f"{t:g}"] = round(sens, 4)
    # hasta AUC
    nod = [r for r in records if r["is_nodule"]]
    ctl = [r for r in records if not r["is_nodule"]]
    aucs = {}
    for agg in ("top1", "top3", "noisyor"):
        def sc(r):
            ss = sorted([s for _, _, s, _ in r["cands"]], reverse=True)
            if not ss:
                return 0.0
            if agg == "top1":
                return ss[0]
            if agg == "top3":
                return float(np.mean(ss[:3]))
            return float(1.0 - np.prod([1.0 - np.clip(s, 0, 1) for s in ss[:5]]))
        y = [1] * len(nod) + [0] * len(ctl)
        pr = [sc(r) for r in nod] + [sc(r) for r in ctl]
        aucs[agg] = round(float(roc_auc_score(y, pr)), 4) if len(set(y)) > 1 else None
    return out, aucs


def common_ish_nms(cands):
    """(x, y, score, hit) listesi üzerinde skor sıralı ızgara NMS."""
    c = sorted(cands, key=lambda t: -t[2])
    grid, out = set(), []
    for x, y, s, hit in c:
        gx, gy = int(x // NMS_PX), int(y // NMS_PX)
        if any((gx + dx, gy + dy) in grid for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        grid.add((gx, gy))
        out.append((x, y, s, hit))
    return out


def hierarchical(recs, gate_scores=None):
    """S_final(i) = s_i × P_noisyOR(hasta)."""
    out = []
    for r in recs:
        ss = np.array([s for _, _, s, _ in r["cands"]], float)
        if ss.size == 0:
            out.append(r)
            continue
        p = 1.0 - np.prod(1.0 - np.clip(ss[:5], 0, 1))
        out.append({"pid": r["pid"], "is_nodule": r["is_nodule"],
                    "cands": [(x, y, s * p, h) for x, y, s, h in r["cands"]]})
    return out


def _recompute_patient_scores(model_kind, scores, s1):
    return scores


# ------------------------------------------------------------------ ana akış
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()
    print("V6.2 | Aşama-1 dondurulmuş (V6.0)")

    tr_nod = p2._nodule_pids("train"); tr_ctl = p2._control_pids("train")
    rng = np.random.RandomState(SEED)
    tr_pids = list(rng.choice(tr_nod, 400, replace=False)) + list(rng.choice(tr_ctl, 400, replace=False))
    va_pids = common.split_patients()["val"]

    print("[1] Aşama-1 adayları...")
    c_tr = s2.stage1_candidates(tr_pids, m5, m6, device)
    c_va = s2.stage1_candidates(va_pids, m5, m6, device)
    gt_tr = {p: common.load_gt(p) for p in tr_pids}
    gt_va = {p: common.load_gt(p) for p in va_pids}
    lab_tr = {p: {v: s2._labels_for(c_tr[p][v], gt_tr[p]) for v in c_tr[p]} for p in tr_pids}
    lab_va = {p: {v: s2._labels_for(c_va[p][v], gt_va[p]) for v in c_va[p]} for p in va_pids}

    print("[2] Lokal TDA aralıkları...")
    ranges = s2.derive_ranges([(p, c_tr[p]["M5_heatTTA"]) for p in tr_pids if gt_tr[p]][:20])
    print(f"   {ranges}")

    def build(pids, cand_dict, variant):
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(pid, cand_dict[pid][variant], ranges, True) for pid in pids]
        out = {}
        with ProcessPoolExecutor(max_workers=10) as pool:
            for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
                out[pid] = res if not err else None
        return out

    print("[3] Özellikler (train)...")
    f_tr = {p: {} for p in tr_pids}
    for v in ("M5_heatTTA", "M5M6TTA"):
        got = build(tr_pids, c_tr, v)
        for p in tr_pids:
            f_tr[p][v] = got.get(p)
    print("[3] Özellikler (val)...")
    f_va = {p: {} for p in va_pids}
    for v in ("M5_heatTTA", "M5M6TTA"):
        got = build(va_pids, c_va, v)
        for p in va_pids:
            f_va[p][v] = got.get(p)

    # eğitim matrisi
    Xt_tda, Xt_spa, Xt_pat, yt, s1_t = [], [], [], [], []
    for p in tr_pids:
        f = f_tr[p].get("M5_heatTTA")
        if f is None:
            continue
        for i, (cx, cy, s) in enumerate(c_tr[p]["M5_heatTTA"]):
            Xt_tda.append(f["tda"][i]); Xt_spa.append(f["spa"][i]); Xt_pat.append(f["pat"][i])
            yt.append(1 if lab_tr[p]["M5_heatTTA"][i] is not None else 0)
            s1_t.append(s)
    Xt_tda = np.asarray(Xt_tda, np.float32); Xt_spa = np.asarray(Xt_spa, np.float32)
    Xt_pat = np.asarray(Xt_pat, np.uint8); yt = np.asarray(yt, int); s1_t = np.asarray(s1_t, np.float32)
    Xt = np.concatenate([Xt_tda, Xt_spa], 1)
    sc = StandardScaler().fit(Xt)
    Xts = sc.transform(Xt)
    print(f"[4] Modeller (aday {len(yt)}, pozitif {int(yt.sum())}, %{100 * yt.mean():.2f})...")

    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                         random_state=SEED).fit(Xts, yt)
    print(f"   HGB ({time.time() - t0:.0f}s)")
    # SVM: pozitiflerin tamamı + 8k negatif (hız)
    pos_i = np.where(yt == 1)[0]; neg_i = np.where(yt == 0)[0]
    sub = np.concatenate([pos_i, rng.choice(neg_i, size=min(8000, len(neg_i)), replace=False)])
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=SEED,
              cache_size=1000).fit(Xts[sub], yt[sub])
    print(f"   SVM ({time.time() - t0:.0f}s)")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=40, early_stopping=True,
                        random_state=SEED).fit(Xts, yt)
    print(f"   MLP ({time.time() - t0:.0f}s)")
    cnn = s2.train_patch_cnn(Xt_pat, yt, device, epochs=6)

    def ens(scores_list):
        return np.mean([rank_norm(s) for s in scores_list], axis=0)

    # füzyon LR'leri (eğitim skorlarıyla)
    p_ens_t = ens([hgb.predict_proba(Xts)[:, 1], svm.decision_function(Xts), mlp.predict_proba(Xts)[:, 1]])
    p_hgb_t = hgb.predict_proba(Xts)[:, 1]
    p_img_t = s2.predict_patch(cnn, Xt_pat, device)
    lr_all = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(
        np.stack([s1_t, p_hgb_t, p_img_t], 1), yt)
    lr_tda = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED).fit(
        np.stack([s1_t, p_hgb_t], 1), yt)
    print("   Füzyon LR'leri hazır (TDA dalı: HGB)")

    # ------------------------------------------------------------------ NODE21 val
    def eval_setting(variant, K, kind):
        recs = []
        for p in va_pids:
            f = f_va[p].get(variant)
            if f is None:
                continue
            cands = c_va[p][variant][:K]
            labs = lab_va[p][variant][:K]
            if not cands:
                recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0, "cands": []})
                continue
            tda = f["tda"][:K]; spa = f["spa"][:K]; pat = f["pat"][:K]
            X = sc.transform(np.concatenate([tda, spa], 1))
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda_hgb = hgb.predict_proba(X)[:, 1]
            p_tda_ens = ens([hgb.predict_proba(X)[:, 1], svm.decision_function(X), mlp.predict_proba(X)[:, 1]])
            p_img = s2.predict_patch(cnn, pat, device)
            s_all = lr_all.predict_proba(np.stack([s1, p_tda_hgb, p_img], 1))[:, 1]
            s_tda = lr_tda.predict_proba(np.stack([s1, p_tda_hgb], 1))[:, 1]
            sc_map = {"a_stage1": s1, "b_tda_hgb": p_tda_hgb, "b2_tda_ens": p_tda_ens,
                      "c_patch_cnn": p_img, "d_fusion_all": s_all, "e_fusion_tda": s_tda}
            sc_arr = sc_map[kind]
            recs.append({"pid": p, "is_nodule": len(gt_va[p]) > 0,
                         "cands": [(cands[i][0], cands[i][1], float(sc_arr[i]), labs[i]) for i in range(len(cands))]})
        n_les = sum(len(g) for g in gt_va.values())
        f_, a_ = froc_fixed(recs, len(recs), n_les)
        return f_, a_, recs

    print("\n[5] NODE21 val (düzeltilmiş FROC)...")
    results = {}
    for variant, K in (("M5_heatTTA", 30), ("M5M6TTA", 50)):
        for kind in ("a_stage1", "b_tda_hgb", "b2_tda_ens", "c_patch_cnn", "d_fusion_all", "e_fusion_tda"):
            f_, a_, _ = eval_setting(variant, K, kind)
            results[f"{variant}|K{K}|{kind}"] = {"froc": f_, "patient_auc": a_}
            print(f"  {variant} K{K} {kind:14s} FROC {f_} | AUC {a_}")
        # hiyerarşik (e_fusion_tda üzerine)
        f_, a_, recs = eval_setting(variant, K, "e_fusion_tda")
        fh, ah = froc_fixed(hierarchical(recs), len(recs), sum(len(g) for g in gt_va.values()))
        results[f"{variant}|K{K}|f_hierarchical"] = {"froc": fh, "patient_auc": ah}
        print(f"  {variant} K{K} {'f_hierarchical':14s} FROC {fh} | AUC {ah}")

    # ------------------------------------------------------------------ VinDr sıfır-eğitim
    print("\n[6] VinDr-CXR sıfır-eğitim (400 görüntü)...")
    vindr = _vindr_zeroshot(m5, m6, sc, hgb, svm, mlp, cnn, lr_all, lr_tda, ranges, device)

    report = {
        "experiment": "V6.2",
        "stage": "duzeltilmis FROC + dogrusal olmayan TDA toplulugu + hiyerarsik kalibrasyon + VinDr transfer",
        "protocol": {"froc": "post-fusion NMS(20px) + duplikasyon IGNORE (LUNA16 stili)",
                     "tda_ensemble": "HGB + RBF-SVM + MLP (rank-normalize ortalama)",
                     "hierarchical": "S_final = p_aday x P_NoisyOR(hasta)"},
        "node21_val": results,
        "vindr_zeroshot": vindr,
        "v5_3_reference_vindr_auc": 0.6283,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}")


def _vindr_zeroshot(m5, m6, sc, hgb, svm, mlp, cnn, lr_all, lr_tda, ranges, device):
    import pandas as pd
    from PIL import Image
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    cohort = pd.read_csv(PROJECT_ROOT / "data/external_vindr/cohort_with_diameters.csv")
    seg = PSPNet5ZoneSegmenter(device=device)
    seg_model = seg  # PSPNet
    ex = ZoneTDAExtractor(s2._settings(ranges))
    y_true, s_all_img, s_tda_img = [], [], []
    t0 = time.time()
    for i, row in cohort.iterrows():
        pid = str(row["image_id"]); y = int(row["patient_label"])
        img = np.array(Image.open(PROJECT_ROOT / "data/external_vindr/images" / f"{pid}.jpg"))
        try:
            zmap = seg.segment(img)["zone_map"]
            raw = zmap > 0
            if not raw.any():
                y_true.append(y); s_all_img.append(0.0); s_tda_img.append(0.0); continue
            ys, xs = np.where(raw)
            tissue = img[raw]
            p1, p99 = np.percentile(tissue, [1, 99])
            canon = common.canonicalize(img, common_dilate(raw), float(p1), float(p99))
            c5h = ev._predict_heat_tta(m5, canon, device, "M5")
            c6 = p2.predict(m6, canon, device, "M6")
            # M5/M6 koordinatları 1024 uzayı varsayımıyla döner → görüntünün gerçek boyutuna ölçekle
            H, W = canon.shape
            sx, sy = W / 1024.0, H / 1024.0
            c5h = [(x * sx, y * sy, s) for x, y, s in c5h][:50]
            c6 = [(x * sx, y * sy, s) for x, y, s in c6]
            cands = ev._rrf([c5h, c6])[:50]
            if not cands:
                y_true.append(y); s_all_img.append(0.0); s_tda_img.append(0.0); continue
            tda = np.zeros((len(cands), 2 * 2 * s2.GRID * s2.GRID), np.float32)
            spa = np.zeros((len(cands), 4), np.float32)
            pat = np.zeros((len(cands), s2.OUTER * s2.OUTER), np.uint8)
            bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
            from scipy import ndimage as ndi
            dist = ndi.distance_transform_edt(raw).astype(np.float32)
            for k, (cx, cy, _) in enumerate(cands):
                tda[k] = np.concatenate([ex.extract(s2._crop(canon, cx, cy, s2.INNER))[0],
                                         ex.extract(s2._crop(canon, cx, cy, s2.OUTER))[0]])
                spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
                pat[k] = np.clip(s2._crop(canon, cx, cy, s2.OUTER), 0, 255).astype(np.uint8).ravel()
            X = sc.transform(np.concatenate([tda, spa], 1))
            s1 = np.array([c[2] for c in cands], np.float32)
            p_tda = hgb.predict_proba(X)[:, 1]
            p_img = s2.predict_patch(cnn, pat, device)
            sc_all = lr_all.predict_proba(np.stack([s1, p_tda, p_img], 1))[:, 1]
            sc_tda = lr_tda.predict_proba(np.stack([s1, p_tda], 1))[:, 1]
        except Exception as exc:
            print(f"   ! {pid}: {type(exc).__name__}: {exc}")
            y_true.append(y); s_all_img.append(0.0); s_tda_img.append(0.0); continue
        y_true.append(y)
        s_all_img.append(float(1 - np.prod(1 - np.clip(np.sort(sc_all)[::-1][:5], 0, 1))))
        s_tda_img.append(float(1 - np.prod(1 - np.clip(np.sort(sc_tda)[::-1][:5], 0, 1))))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"n_images": len(y_true), "n_positive": int(np.sum(y_true))}
    out["auc_fusion_all"] = round(float(roc_auc_score(y_true, s_all_img)), 4)
    out["auc_fusion_tda"] = round(float(roc_auc_score(y_true, s_tda_img)), 4)
    out["v5_3_reference"] = 0.6283
    print(f"   VinDr AUC (fusion_all) = {out['auc_fusion_all']} | (fusion_tda) = {out['auc_fusion_tda']} "
          f"| V5.3 ref = 0.6283")
    return out


def common_dilate(raw):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(raw, structure=ndi.generate_binary_structure(2, 1),
                               iterations=common.DILATION_PX)


if __name__ == "__main__":
    main()
