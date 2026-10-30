"""V6.9j — Site kalibrasyonu: yeni mimariyle (M5_foundation + stage2_foundation_tda_hgb)
VinDr-CXR çift-katmanlı few-shot kalibrasyon.

Amaç (V6.5'in güncel mimariyle tekrarı + genişletmesi):
  Eski V6.5, few-shot kalibrasyonu M5 (ImageNet) dedektörü + lr_fusion (stacking) ile ve
  YALNIZ 2. katmanda yapmıştı. Bu deney, güncel mimaride (foundation Stage-1 + uyumlu
  2. aşama, SABİT w=0.55 füzyon) yeni bir hastane kaynağına N ∈ {0,10,20,40,80,160}
  etiketli görüntüyle (5-fold stratified CV, sızıntısız) adapte olurken 1. katmanı,
  2. katmanı ve İKİSİNİ BİRDEN kalibre etmenin etkisini ölçer.

ÜÇ KOL (N ∈ {0,10,20,40,80,160}, 5-fold CV):
  Kol 1 — Yalnız 2. katman (Lokal TDA kalibrasyonu): Stage-1 donuk; 1.600 özellikli (saf
topolojik) TDA
          sınıflandırıcısı N hedef görüntüyle (image-label) kalibre edilir.
  Kol 2 — Yalnız 1. katman (Head/Neck fine-tune): omurga donuk; yalnız boyun+CenterNet
          kafası N hedef görüntünün nodül kutularıyla (Gaussian focal, düşük LR,
          sıfır-eğitim öncül çapası) fine-tune edilir.
  Kol 3 — Tam çift-katmanlı: FT edilmiş Stage-1 skoru + kalibre edilmiş TDA birlikte.

TASARIM KARARI (kullanıcı onayı): Sabit havuz + yeniden puanlama.
  Aday KUTULARI donuk Stage-1'den SABİT kalır (TDA bir kez çıkarılır, 3 kolda ortak).
  Fine-tune edilmiş kafa, o sabit merkezlerdeki güven skorunu günceller (lokalizasyon
  değişmez, yalnız sıralama). Bu sayede "TDA'yı bir kez hesapla" koşulu birebir sağlanır.

HIZ (iki iyileştirme):
  1) Donuk omurga önbelleği: ResNet50 (self.bb) FT/re-score sırasında HİÇ değişmez.
     Bu yüzden 400 görüntünün c2..c5 özellikleri bir kez hesaplanıp float16 CPU'da
     saklanır (flip'siz + flip'li). FT ve re-score yalnız neck+head çalıştırır ⇒ ~10×.
  2) 2. katman sıfır-eğitim öncülü: küçük N'de (10/20) 1604'lü LR kırılgandır; çıktısı
     sıfır-eğitim p_tda0 ile α = N/(N+40) ağırlığıyla harmanlanır (shrinkage).

FÜZYON (SABİT, hiçbir veride yeniden öğrenilmez):
  S = sigmoid(0.55·z(s1) + 0.45·z(p_tda)); z referansı NODE21 val'den dondurulmuş
  (bundle fusion_ref).

Kullanım:
    python pipelines/v6_two_stage/v6_9j_dual_stage_fewshot.py
    python pipelines/v6_two_stage/v6_9j_dual_stage_fewshot.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_5d_strict_no_adabn as v65d  # noqa: E402
from detector_models import build_targets_heat, center_focal_loss  # noqa: E402
from detector_models_foundation import FoundationCenterNet, foundation_input, predict_tta  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import official_metric  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9j2_vindr_pure1600.json"
BUNDLE_FILE = "stage2_foundation_tda_hgb_pure1600.joblib"   # V6.9m: saf 1.600
SEED = 42
N_FEW = (0, 10, 20, 40, 80, 160)
N_FOLDS = 5
BOX_HALF = 28.0                     # 56×56 px kutu
TOP = 50
# fine-tune hiperparametreleri (kol 2)
FT_EPOCHS = 5
FT_LR = 1e-4
FT_ANCHOR = 1e-3                   # sıfır-eğitim öncül çapası (head → ilk ağırlıklar)
FT_BATCH = 4
SHRINK_C = 40.0                    # 2. katman kalibrasyon shrinkage: α = N / (N + C)


def _fuse(s1: np.ndarray, p_tda: np.ndarray, ref: dict) -> np.ndarray:
    z1 = (s1 - ref["s1_mean"]) / (ref["s1_std"] + 1e-9)
    zt = (p_tda - ref["tda_mean"]) / (ref["tda_std"] + 1e-9)
    return 1.0 / (1.0 + np.exp(-(ref["w"] * z1 + (1.0 - ref["w"]) * zt)))


def image_score(scores) -> float:
    ss = np.sort(np.asarray(scores, float))[::-1][:5]
    if len(ss) == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - np.clip(ss, 0, 1)))


# --------------------------------------------------------------------------- #
# TDA özellik çıkarımı (sabit havuz, bir kez) — VinDr orijinal-uzay kanonik
# --------------------------------------------------------------------------- #
def _extract_one(args):
    d, cands, ranges = args
    from node21_tda.features.extractor import ZoneTDAExtractor
    ex = ZoneTDAExtractor(s2._settings(ranges))
    return _extract_body(d, cands, ex)


def _extract_body(d, cands, ex):
    canon = d["canon"].astype(np.float32)
    raw = d["raw"] > 0
    ys, xs = np.where(raw)
    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if ys.size else (0, 0, 0, 0)
    dist = ndi.distance_transform_edt(raw).astype(np.float32) if ys.size else np.zeros(raw.shape, np.float32)
    n = len(cands)
    tda = np.zeros((n, 2 * 2 * s2.GRID * s2.GRID), np.float32)
    spa = np.zeros((n, 4), np.float32)
    for k, (cx, cy, _s) in enumerate(cands):
        tda[k] = np.concatenate([ex.extract(s2._crop(canon, cx, cy, s2.INNER))[0],
                                 ex.extract(s2._crop(canon, cx, cy, s2.OUTER))[0]])
        if ys.size:
            spa[k] = s2._spatial(cx, cy, raw, dist, bbox)
    return {"tda": tda, "spa": spa}


def _center_to_grid(cx, cy, H, W):
    """Orijinal-uzay merkez → 256×256 ısı haritası ızgara indeksi."""
    gx = int(round((cx * 512.0 / W - 1.0) / 2.0))
    gy = int(round((cy * 512.0 / H - 1.0) / 2.0))
    return np.clip(gx, 0, 255), np.clip(gy, 0, 255)


@torch.no_grad()
def _compute_bb_cache(model, data, device):
    """400 VinDr görüntüsünün DONUK omurga özelliklerini (c2..c5) bir kez hesapla.

    float16 CPU tensörleri olarak saklanır (bellek 2×: flip'siz + flip'li). Bu sayede
    FT ve re-score sırasında omurga YENİDEN geçilmez — yalnız neck+head çalışır (10× hız).
    """
    cache, t0 = {}, time.time()
    for i, d in enumerate(data):
        x = foundation_input(d["x512"], False).unsqueeze(0).to(device)
        feats = [c.detach().cpu().half() for c in model.backbone_features(x)]
        xf = torch.flip(x, dims=[3])
        feats_f = [c.detach().cpu().half() for c in model.backbone_features(xf)]
        cache[d["pid"]] = {"f": feats, "ff": feats_f}
        if (i + 1) % 100 == 0:
            print(f"    omurga önbelleği {i + 1}/{len(data)} ({time.time() - t0:.0f}s)",
                  flush=True)
    return cache


@torch.no_grad()
def _neck_heat(model, feats, device):
    """c2..c5 (CPU float16) → neck+head → 256×256 sigmoid ısı haritası (numpy)."""
    c2, c3, c4, c5 = [c.to(device).float() for c in feats]
    import torch.nn.functional as F
    p2, _, _, _ = model.fpn(c2, c3, c4, c5)
    f = model.fuse(p2)
    f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
    h = model.heat(f)
    return torch.sigmoid(h)[0, 0].detach().cpu().numpy()


@torch.no_grad()
def _rescore(model, data, cands, bb_cache, device):
    """Fine-tune edilmiş kafayla sabit merkezleri yeniden puanla (omurga ÖNBELLEKTEN)."""
    out, t0 = {}, time.time()
    for i, d in enumerate(data):
        H, W = d["canon"].shape
        pid = d["pid"]
        h1 = _neck_heat(model, bb_cache[pid]["f"], device)
        h2 = _neck_heat(model, bb_cache[pid]["ff"], device)[:, ::-1]
        heat = (h1 + h2) / 2.0
        c = cands[pid]
        s1 = np.zeros(len(c), np.float32)
        for k, (cx, cy, _s) in enumerate(c):
            gx, gy = _center_to_grid(cx, cy, H, W)
            s1[k] = heat[gy, gx]
        out[pid] = s1
        if (i + 1) % 400 == 0:
            print(f"      re-score {i + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _finetune_head(base_model, data, idxs, gts512, bb_cache, device):
    """Omurga DONUK ve önbellekten; yalnız neck+head N görüntüyle fine-tune (L2 çapa)."""
    m = copy.deepcopy(base_model)
    m.freeze_backbone()
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=FT_LR, weight_decay=1e-5)
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
    import torch.nn.functional as F
    rng = np.random.RandomState(SEED)
    # omurga özellikleri (flip'siz) eğitim örnekleri için — önbellekten, float32'ye taşı
    bb_dev = {}
    for i in set(idxs):
        bb_dev[i] = [c.to(device).float() for c in bb_cache[data[i]["pid"]]["f"]]
    for ep in range(FT_EPOCHS):
        order = rng.permutation(len(idxs))
        tot, nb = 0.0, 0
        for s in range(0, len(order), FT_BATCH):
            sel = [idxs[i] for i in order[s:s + FT_BATCH]]
            Ys, feats_batch = [], []
            for i in sel:
                Y, _ = build_targets_heat(gts512[i][0], gts512[i][1], device)
                Ys.append(Y)
                c2, c3, c4, c5 = bb_dev[i]
                p2, _, _, _ = m.fpn(c2, c3, c4, c5)
                f = m.fuse(p2)
                f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
                feats_batch.append(f)
            f = torch.cat(feats_batch, 0)
            Y = torch.stack(Ys).unsqueeze(1)
            h = m.heat(f)
            focal = center_focal_loss(h, Y)
            anchor = sum(((p - init[n]).pow(2)).sum() for n, p in m.named_parameters()
                         if p.requires_grad)
            loss = focal + FT_ANCHOR * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(focal)
            nb += 1
        if ep == 0 or ep == FT_EPOCHS - 1:
            print(f"      FT ep{ep + 1}/{FT_EPOCHS} focal {tot / max(1, nb):.4f}", flush=True)
    m.eval()
    return m


def _evaluate(data, te_idx, cands, s1_map, p_tda_map, ref):
    """Test-fold için: fused skor → recs → official_metric + noisy-OR AUC."""
    flat, recs, n_les = [], [], 0
    for i in te_idx:
        d = data[i]
        pid = d["pid"]
        c = cands[pid]
        s1 = s1_map[pid]
        p_tda = p_tda_map[pid]
        pf = _fuse(s1, p_tda, ref)
        n_les += len(d["gts"])
        for k, (cx, cy, _s) in enumerate(c):
            flat.append((float(pf[k]), pid, float(cx), float(cy), 0.0))
    gts = {data[i]["pid"]: [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                            for g in data[i]["gts"]] for i in te_idx}
    is_nod = {data[i]["pid"]: data[i]["y"] == 1 for i in te_idx}
    pids = [data[i]["pid"] for i in te_idx]
    om = official_metric(flat, len(te_idx), n_les, gts, is_nod, pids, box_half=BOX_HALF)
    # noisy-OR hasta AUC (V6.5/V6.6 ile karşılaştırılabilir)
    y = np.array([data[i]["y"] for i in te_idx])
    sc = np.array([image_score(_fuse(s1_map[data[i]["pid"]], p_tda_map[data[i]["pid"]], ref))
                   for i in te_idx])
    auc_nor = float(roc_auc_score(y, sc)) if len(set(y.tolist())) > 1 else np.nan
    return om, auc_nor


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    sm = args.smoke
    n_few = (0, 10) if sm else N_FEW
    n_folds = 2 if sm else N_FOLDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9j — çift-katmanlı few-shot site kalibrasyonu (VinDr)", flush=True)
    print(f"  smoke={sm} | N={list(n_few)} | folds={n_folds}", flush=True)

    import joblib
    import pandas as pd
    import torchxrayvision as xrv
    from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter

    bundle = joblib.load(CKPT / BUNDLE_FILE)
    hgb, scaler, ranges, ref = bundle["hgb"], bundle["scaler"], bundle["ranges"], bundle["fusion_ref"]
    base = load_foundation("M5_foundation", device)
    print(f"  bundle: stage2_foundation_tda_hgb (s1_yok={bundle['stage1_score_included']}, w={ref['w']})",
          flush=True)

    # ---- VinDr verisi
    ann = pd.read_csv(Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz")
    nod = ann[ann["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    cohort = pd.read_csv(v65d.EXT / "cohort_with_diameters.csv")
    if sm:
        cohort = pd.concat([cohort[cohort["patient_label"] == 1].head(12),
                            cohort[cohort["patient_label"] == 0].head(12)]).reset_index(drop=True)
    seg = PSPNet5ZoneSegmenter(device=device)
    print("\n[0] VinDr kohortu...", flush=True)
    data = v65d.build_vindr(seg, cohort, nod)
    print(f"    {len(data)} görüntü, {sum(1 for d in data if d['y'])} pozitif", flush=True)

    # ---- sabit aday havuzu + TDA özellikleri (bir kez)
    print("\n[1] Sabit aday havuzu (donuk Stage-1, AdaBN yok)...", flush=True)
    t0 = time.time()
    cands = {d["pid"]: predict_tta(base, d["canon"].astype(np.float32), device,
                                   shape=d["canon"].shape)[:TOP] for d in data}
    print(f"    {sum(len(v) for v in cands.values())} aday ({time.time() - t0:.0f}s)", flush=True)

    print("[2] TDA + uzamsal özellikler (bir kez, 3 kolda ortak)...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    jobs = [(d, cands[d["pid"]], ranges) for d in data]
    feats = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for d, res in zip(data, pool.map(_extract_one, jobs, chunksize=4)):
            feats[d["pid"]] = res
    print(f"    {len(feats)} görüntü ({time.time() - t0:.0f}s)", flush=True)

    # ölçekli matris (HGB + LR ortak uzay)
    Xs_all, s1_all, counts = [], [], []
    for d in data:
        pid = d["pid"]
        X = feats[pid]["tda"]          # V6.9m: saf 1.600 — uzamsal sütun YOK
        Xs_all.append(scaler.transform(X))
        s1_all.append(np.array([c[2] for c in cands[pid]], np.float32))
        counts.append(len(cands[pid]))
    y_img = np.array([d["y"] for d in data])

    # GT 512-uzay (fine-tune için)
    gts512 = []
    for d in data:
        H, W = d["canon"].shape
        cen, rad = [], []
        for g in d["gts"]:
            cen.append((g["cx"] * 512.0 / W, g["cy"] * 512.0 / H))
            rad.append(max(g["w"], g["h"]) / 2.0 * 512.0 / W)
        gts512.append((cen, rad))

    # ---- 5-fold stratified
    rng = np.random.RandomState(SEED)
    folds = np.zeros(len(data), int)
    for lab in (0, 1):
        idx = np.where(y_img == lab)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds

    # donuk (sıfır-eğitim) p_tda ve s1
    p_tda0 = {data[i]["pid"]: hgb.predict_proba(Xs_all[i])[:, 1] for i in range(len(data))}
    s1_0 = {data[i]["pid"]: s1_all[i] for i in range(len(data))}

    # ---- DONUK OMURGA ÖZELLİK ÖNBELLEĞİ (bir kez; FT + re-score bunu kullanır, omurga geçilmez)
    print("\n[3] Donuk omurga özellik önbelleği (c2..c5, float16, flip'siz + flip'li)...", flush=True)
    bb_cache = _compute_bb_cache(base, data, device)
    print(f"    {len(bb_cache)} görüntü önbelleklendi", flush=True)

    # sonuç tutucular: her kol, her N → listeler
    def new_acc():
        return {"auc_nor": [], "auc_top1": [], "s025": [], "s050": [], "froc2": []}
    arms = {"kol1_stage2": {str(N): new_acc() for N in n_few},
            "kol2_stage1": {str(N): new_acc() for N in n_few},
            "kol3_both": {str(N): new_acc() for N in n_few}}

    for N in n_few:
        for f in range(n_folds):
            tr_idx = [i for i in range(len(data)) if folds[i] != f]
            te_idx = [i for i in range(len(data)) if folds[i] == f]
            # ----- N etiketli görüntü seçimi (stratified)
            if N == 0:
                cal_pos, cal_neg = [], []
                lr_cal = None
                ft_model = None
            else:
                pos = [i for i in tr_idx if y_img[i] == 1]
                neg = [i for i in tr_idx if y_img[i] == 0]
                r2 = np.random.RandomState(SEED + N + f)
                npos = N // 2
                cal_pos = list(r2.choice(pos, size=min(npos, len(pos)), replace=False))
                cal_neg = list(r2.choice(neg, size=min(N - npos, len(neg)), replace=False))
                # 2. katman LR (image-label, scaled 1604)
                pick = cal_pos + cal_neg
                Xc = np.concatenate([Xs_all[i] for i in pick if counts[i]])
                yc = np.concatenate([np.full(counts[i], y_img[i], np.float32) for i in pick if counts[i]])
                if len(set(yc.tolist())) < 2:
                    lr_cal = None
                else:
                    lr_cal = LogisticRegression(class_weight="balanced", max_iter=500, C=0.1,
                                                random_state=SEED).fit(Xc, yc)
                # 1. katman head/neck FT (nodül kutuları + boş kontrol)
                ft_model = _finetune_head(base, data, pick, gts512, bb_cache, device)

            # ----- kalibrasyon çıktıları (her N, fold için)
            if N == 0:
                # sıfır-eğitim: her şey donuk
                p_tda_cal = None
                s1_ft = None
            else:
                # shrinkage: küçük N'de sıfır-eğitim p_tda0'a çek (α = N/(N+C))
                alpha = N / (N + SHRINK_C)
                # Kol 1 / Kol 3 için: 2. katman LR (görüntü etiketi, ölçekli 1604)
                p_tda_cal = {}
                for i in range(len(data)):
                    if lr_cal is not None and counts[i]:
                        pr = lr_cal.predict_proba(Xs_all[i])[:, 1]
                        p_tda_cal[data[i]["pid"]] = alpha * pr + (1.0 - alpha) * p_tda0[data[i]["pid"]]
                    else:
                        p_tda_cal[data[i]["pid"]] = p_tda0[data[i]["pid"]]
                # Kol 2 / Kol 3 için: FT edilmiş stage-1 ile yeniden puanlama (sabit merkezler)
                s1_ft = _rescore(ft_model, data, cands, bb_cache, device)
                del ft_model
                torch.cuda.empty_cache()

            # ----- üç kolun (s1, p_tda) girdi eşlemesi
            combos = [
                ("kol1_stage2", s1_0, p_tda0 if N == 0 else p_tda_cal),   # yalnız 2. katman
                ("kol2_stage1", s1_0 if N == 0 else s1_ft, p_tda0),       # yalnız 1. katman
                ("kol3_both", s1_0 if N == 0 else s1_ft,
                 p_tda0 if N == 0 else p_tda_cal),                        # ikisi birlikte
            ]
            for arm, s1_map, ptda_map in combos:
                om, auc_nor = _evaluate(data, te_idx, cands, s1_map, ptda_map, ref)
                acc = arms[arm][str(N)]
                acc["auc_nor"].append(auc_nor)
                acc["auc_top1"].append(om["auc"])
                acc["s025"].append(om["sens"]["0.25"])
                acc["s050"].append(om["sens"]["0.5"])
                acc["froc2"].append(om["sens"]["2"])
        # özet satır
        for arm in arms:
            a = np.array(arms[arm][str(N)]["auc_nor"], float)
            print(f"  N={N:3d} {arm:12s} AUC(nor) {np.nanmean(a):.4f}±{np.nanstd(a):.4f} | "
                  f"S@.25 {np.nanmean(arms[arm][str(N)]['s025']):.4f} | "
                  f"FROC@2 {np.nanmean(arms[arm][str(N)]['froc2']):.4f}", flush=True)

    # ---- rapor
    report = {"experiment": "V6.9j2-dual-stage-fewshot-pure1600", "smoke": sm,
              "architecture": "M5_foundation + stage2_foundation_tda_hgb_pure1600 "
                              "(SAF 1600, w=0.55 kilidi)",
              "design": "SABİT aday havuzu + yeniden puanlama (TDA tek çıkarım, 3 kolda ortak)",
              "n_images": len(data), "n_positive": int(y_img.sum()), "folds": n_folds,
              "N": list(n_few),
              "ft": {"epochs": FT_EPOCHS, "lr": FT_LR, "anchor": FT_ANCHOR,
                     "scope": "boyun+CenterNet kafası (omurga donuk)"},
              "stage2_shrinkage": {"C": SHRINK_C, "alpha_n": "N/(N+C)",
                                   "note": "2. katman kalibrasyon çıktısı sıfır-eğitim p_tda0 ile harmanlanır"},
              "speedup": {"backbone_cache": "c2..c5 float16 CPU, flip'siz+flip'li (bir kez)",
                          "note": "FT + re-score omurgayı YENİDEN geçmez; yalnız neck+head"},
              "calibration_label": "görüntü etiketi (kaba, V6.5 ile aynı)", "fusion": ref,
              "arms": arms,
              "note": "AUC(nor)=noisy-OR hasta AUC · AUC(top1)=max-aday · S@0.25/FROC@2 resmî IoU>0.2 · 56px kutu"}
    # nan temizle
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating, float)) and not np.isfinite(o):
            return None
        return o
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(_san(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {args.out}", flush=True)


if __name__ == "__main__":
    main()
